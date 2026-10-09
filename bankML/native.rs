// SPDX-License-Identifier: MIT OR Apache-2.0
//! The native engine behind `bankml serve --native`: chat completions from bankML's own forward pass.
//!
//! Token-identical to llama-server b11192 with one slot (`-np 1`): the same template, tokenizer, kernels, slot
//! prefix reuse, host prompt cache and default sampler chain (all of it; mirostat and a custom sampler order are
//! refused). It also owns the model lifecycle (`Registry`, `Residency`): one verified model resident at a time,
//! dropped when its keep-alive runs out, as Ollama does.
//! Details: docs/modules/native.md.

use crate::chat::{self, Message};
use crate::forward::{KvCache, Weights};
use crate::prompt_cache::PromptCache;
use crate::sampler::{Params, Sampler};
use crate::serve::{ident, FileIdent, Json};
use crate::tokenizer::Tokenizer;
use crate::Verified;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The slot file's first bytes (`Native::save_slot`).
const SLOT_MAGIC: &[u8] = b"bankML slot v1\n";

struct Slot {
    tokens: Vec<u32>,
    caches: Vec<KvCache>,
    /// llama-server's host prompt cache (`prompt_cache.rs`); `None` with `BANKML_CACHE_RAM=0`.
    saved: Option<PromptCache<Vec<KvCache>>>,
}

/// The host prompt cache's byte limit: `BANKML_CACHE_RAM` in MiB, as `--cache-ram` (0 off, -1 no limit).
///
/// Unset, it is llama-server's 8192 MiB capped at a quarter of the memory available at load.
fn cache_ram_limit() -> Option<usize> {
    const MIB: usize = 1 << 20;
    match std::env::var("BANKML_CACHE_RAM").ok().and_then(|v| v.trim().parse::<i64>().ok()) {
        Some(0) => None,
        Some(n) if n < 0 => Some(0),
        Some(n) => Some(n as usize * MIB),
        None => {
            let quarter = crate::sys::memory().map(|m| (m.available / 4) as usize).unwrap_or(usize::MAX);
            // 8192 MiB does not fit a 32-bit usize (WebAssembly): there the cap is the address space
            Some(usize::try_from(8192u64 << 20).unwrap_or(usize::MAX).min(quarter).max(MIB))
        }
    }
}

pub struct Native {
    pub w: Weights,
    pub tok: Tokenizer,
    pub defaults: Params,
    pub n_ctx: usize,
    pub model: PathBuf,
    /// Chat template, identified by the sha256 of its text (`chat::TEMPLATES`).
    pub template: chat::Template,
    eog: Vec<u32>,
    slot: Mutex<Slot>,
    /// Every token's piece and the end set, as the grammar reads them; built on first use.
    gvocab: std::sync::OnceLock<crate::grammar::Vocab>,
    /// JSON-mode grammar, parsed once.
    json_rules: std::sync::OnceLock<Arc<crate::grammar::Rules>>,
    /// DRY breakers for the last breaker list requested (building them scans the whole vocabulary).
    dry_cache: Mutex<Option<(Vec<String>, Arc<crate::sampler::DryBreakers>)>>,
}

/// How a completion ended, and its counts (llama-server's `usage` and `timings.cache_n`).
#[derive(Debug, Clone, Default)]
pub struct Done {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: &'static str,
    pub text: String,
    /// Wall time of the prompt's uncached part and of the generation, in nanoseconds.
    pub prompt_ns: u64,
    pub eval_ns: u64,
    /// Generated token ids, including the end token when one ended the answer.
    pub tokens: Vec<u32>,
    /// Under a grammar: time spent in it (checks, masks, accepts) and the number of redrawn tokens.
    pub grammar_ns: u64,
    pub resampled: usize,
    /// Time from the start of the completion to its first piece, and the CPU package energy when RAPL is readable.
    pub ttft_ns: Option<u64>,
    pub energy_j: Option<f64>,
    /// With `n_probs` > 0: each generated token's probability and the top `n_probs`, from the raw logits.
    pub probs: Vec<TokenLogprob>,
}

/// What a run that stopped because its client left returns (`complete_while`).
pub const GONE: &str = "the client went away";

/// One step of a completion, as `Native::complete_with` reports it.
///
/// - `piece`: the text this step released (may be empty).
/// - `complete`: the token left no incomplete UTF-8 behind; llama-server sends a token only then.
/// - `entry`: its logprobs, when requested and complete.
/// - `n`: tokens generated so far.
/// - `eog`: the end-of-turn token, which ends the answer.
///
/// A final step with `n` unchanged and `complete` false flushes bytes left incomplete at the end.
pub struct Step<'a> {
    pub piece: &'a str,
    pub complete: bool,
    pub entry: Option<&'a TokenLogprob>,
    pub n: usize,
    pub eog: bool,
}

/// One generated token's probability and the top tokens with their probabilities.
///
/// Top-token text excludes specials, as llama-server renders `top_logprobs`. `piece` is what this token emitted;
/// the serving layer replaces it with what was sent (`serve::logprob_entries`). `complete`: no incomplete UTF-8
/// was left after this token; llama-server keeps an entry only then.
#[derive(Debug, Clone, Default)]
pub struct TokenLogprob {
    pub id: u32,
    pub p: f32,
    pub piece: Vec<u8>,
    pub top: Vec<(u32, f32, Vec<u8>)>,
    pub emitted: bool,
    pub complete: bool,
}

impl Native {
    pub fn open(model: &Path, n_ctx: usize) -> Result<Native, String> {
        let template = chat::template_of(model)?;
        let tok = Tokenizer::from_gguf(model)?;
        let w = Weights::open(model)?;
        let defaults = Params::from_gguf(model)?;
        let eog = eog_from_gguf(model, &tok)?;
        // BANKML_CACHE_TYPE=q8_0 as llama.cpp's `--cache-type-k q8_0 --cache-type-v q8_0` (flash attention)
        let kind = crate::forward::KvType::parse(std::env::var("BANKML_CACHE_TYPE").as_deref().unwrap_or("f16"))?;
        if kind == crate::forward::KvType::Q8_0 && w.head_dim % crate::q1_0::QK8_0 != 0 {
            return Err(format!("a q8_0 cache needs a head size that is a multiple of 32; this model's is {}", w.head_dim));
        }
        let caches = w.caches_of(kind);
        Ok(Native { w, tok, defaults, n_ctx, model: model.to_path_buf(), template, eog, slot: Mutex::new(Slot { tokens: Vec::new(), caches, saved: cache_ram_limit().map(|b| PromptCache::new(b, n_ctx)) }),
                    gvocab: Default::default(), json_rules: Default::default(), dry_cache: Default::default() })
    }

    /// DRY breakers for this vocabulary (`sampler::dry_breakers`), built once per breaker list and kept.
    fn dry_breakers(&self, breakers: &[String]) -> Result<Arc<crate::sampler::DryBreakers>, String> {
        let mut cache = self.dry_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((k, b)) = cache.as_ref() {
            if k == breakers {
                return Ok(b.clone());
            }
        }
        let b = Arc::new(crate::sampler::dry_breakers(breakers, self.grammar_vocab().pieces(), |s| self.tok.encode(s, false))?);
        *cache = Some((breakers.to_vec(), b.clone()));
        Ok(b)
    }

    /// The vocabulary as the grammar and DRY read it; built once (about 10 MB) on first use.
    pub fn grammar_vocab(&self) -> &crate::grammar::Vocab {
        self.gvocab.get_or_init(|| crate::grammar::Vocab::new((0..self.tok.n_tokens() as u32).map(|i| self.tok.piece(i)).collect(), &self.eog))
    }

    /// The grammar a request's constraint asks for, in its starting state.
    ///
    /// JSON mode and schemas use the template's output-format grammar with the generation prompt already accepted,
    /// as llama-server prefills it; a user GBNF is used as given.
    pub fn grammar(&self, c: &crate::grammar::Constraint) -> Result<Option<crate::grammar::Grammar>, String> {
        use crate::grammar::{json_object_grammar, Constraint, Grammar, Rules};
        let tokenize = |b: &[u8]| self.tok.encode(&String::from_utf8_lossy(b), true);
        match c {
            Constraint::None => Ok(None),
            Constraint::JsonObject | Constraint::Schema(_) => {
                // the template's output-format grammar (JSON mode's, or the schema's via the template's chat parser):
                // its root opens with the generation prompt, which is prefilled
                let (r, prefill) = match c {
                    Constraint::Schema(s) => (Arc::new(Rules::parse(&crate::schema::chat_grammar(s, self.template)?.0, &tokenize)?), self.template.generation_prompt()),
                    _ => {
                        let (text, prefill) = json_object_grammar(self.template);
                        (self.json_rules.get_or_init(|| Arc::new(Rules::parse(text, &tokenize).expect("the JSON-mode grammar parses"))).clone(), prefill)
                    }
                };
                let v = self.grammar_vocab();
                let mut g = Grammar::new(r);
                for id in self.tok.encode(prefill, true) {
                    g.accept(v, id)?;
                }
                Ok(Some(g))
            }
            Constraint::Gbnf(text) => {
                let r = Rules::parse(text, &tokenize)?;
                self.grammar_vocab();
                Ok(Some(Grammar::new(Arc::new(r))))
            }
        }
    }

    /// The request's sampling parameters over the model's defaults; an unsupported sampler is refused.
    pub fn params(&self, req: &Json) -> Result<Params, String> {
        sampling(self.defaults.clone(), req)
    }

    /// The prompt a conversation becomes, as token ids (llama-server: the template, then tokenization with specials).
    pub fn prompt(&self, messages: &Json) -> Result<Vec<u32>, String> {
        self.prompt_fit(messages, None)
    }

    /// The prompt fitted under a `num_ctx` as Ollama v0.13.3 fits it (`fit_messages`).
    ///
    /// Tokens are counted on the prompt bankML answers (the GGUF's template). A prompt that still does not fit is
    /// refused: Ollama's runner would cut tokens from its middle (`num_keep`), which bankML does not reproduce.
    pub fn prompt_fit(&self, messages: &Json, num_ctx: Option<usize>) -> Result<Vec<u32>, String> {
        let msgs: Vec<Message> = chat::messages_from_json(messages)?;
        let enc = |m: &[Message]| -> Result<Vec<u32>, String> { Ok(self.tok.encode(&self.template.render(m)?, true)) };
        let Some(nc) = num_ctx else { return enc(&msgs) };
        let (kept, dropped) = fit_messages(&msgs, nc, |m| enc(m).map(|p| p.len()))?;
        let p = enc(&kept)?;
        if p.len() > nc {
            return Err(format!("the prompt is {} tokens, more than num_ctx {nc}{}: Ollama would cut tokens out of its middle (num_keep), \
                                which bankML does not reproduce; shorten the last message or raise num_ctx (up to this server's --ctx)",
                               p.len(), if dropped > 0 { format!(" with the {dropped} oldest messages dropped") } else { String::new() }));
        }
        Ok(p)
    }

    /// One completion; `emit` receives whole UTF-8 pieces as they come and returns false to stop.
    ///
    /// Under a grammar (`Native::grammar`) each token is drawn as llama.cpp's `common_sampler_sample` draws it.
    pub fn complete(&self, prompt: &[u32], params: Params, max_tokens: Option<usize>, grammar: Option<crate::grammar::Grammar>,
                    mut emit: impl FnMut(&str) -> bool) -> Result<Done, String> {
        self.complete_with(prompt, params, max_tokens, grammar, |s| s.eog || s.piece.is_empty() || emit(s.piece))
    }

    /// `complete`, stopping when `alive()` says the asker has gone (`complete_while`).
    pub fn complete_alive(&self, prompt: &[u32], params: Params, max_tokens: Option<usize>, grammar: Option<crate::grammar::Grammar>,
                          alive: &dyn Fn() -> bool, mut emit: impl FnMut(&str) -> bool) -> Result<Done, String> {
        self.complete_while(prompt, params, max_tokens, grammar, alive, |s| s.eog || s.piece.is_empty() || emit(s.piece))
    }

    /// `complete`, reporting every step (`Step`).
    ///
    /// Steps include whole tokens that release no text and the end-of-turn token, each with its logprobs, so a
    /// stream can send them as llama-server does.
    pub fn complete_with(&self, prompt: &[u32], params: Params, max_tokens: Option<usize>, grammar: Option<crate::grammar::Grammar>,
                         emit: impl FnMut(Step) -> bool) -> Result<Done, String> {
        self.complete_while(prompt, params, max_tokens, grammar, &|| true, emit)
    }

    /// `complete_with`, asking `alive()` between prefill micro-batches and before each generated token: when the asker
    /// has gone (its client closed the connection) the run stops there with `GONE`, and the slot keeps exactly what was
    /// computed, so a client that comes back reuses it. The bits of what is computed do not change.
    pub fn complete_while(&self, prompt: &[u32], params: Params, max_tokens: Option<usize>, mut grammar: Option<crate::grammar::Grammar>,
                          alive: &dyn Fn() -> bool, mut emit: impl FnMut(Step) -> bool) -> Result<Done, String> {
        if prompt.is_empty() {
            return Err("an empty prompt".into());
        }
        if prompt.len() >= self.n_ctx {
            return Err(format!("the prompt has {} tokens; the context is {}", prompt.len(), self.n_ctx));
        }
        let (n_probs, cache_prompt) = (params.n_probs, params.cache_prompt);
        let mut probs = Vec::new();
        let mut sampler = Sampler::new(params)?;
        if sampler.wants_dry_breakers() {
            sampler.set_dry_breakers(self.dry_breakers(sampler.dry_sequence_breakers())?);
        }
        // llama-server accepts the whole prompt into the samplers' windows before the first draw, cached or not
        for &t in prompt {
            sampler.accept(t);
        }
        let mut slot = self.slot.lock().map_err(|_| "the slot is poisoned")?;
        let t0 = Instant::now();
        let e0 = crate::sys::energy_uj();
        let mut ttft_ns = None;
        // host prompt cache: when the slot serves this prompt poorly, park its state and take back a better one
        let Slot { tokens, caches, saved } = &mut *slot;
        if let Some(pc) = saved.as_mut().filter(|_| PromptCache::<Vec<KvCache>>::wants_update(tokens, prompt)) {
            let bytes = caches.iter().map(KvCache::bytes).sum();
            pc.save(tokens, bytes, || caches.clone());
            if let Some((t, c)) = pc.take_better(tokens, prompt) {
                (*tokens, *caches) = (t, c);
            }
            pc.update();
        }
        // the slot's prefix: the longest common prefix, less one when the whole prompt is cached; none at all when the
        // request says `cache_prompt: false` (llama-server: "if we don't cache the prompt, we have to remove all
        // previous tokens", n_past = 0)
        let mut n_past = if cache_prompt { slot.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count() } else { 0 };
        if n_past == prompt.len() {
            n_past -= 1;
        }
        slot.tokens.truncate(n_past);
        for c in slot.caches.iter_mut() {
            c.truncate(n_past);
        }
        let Slot { tokens, caches, .. } = &mut *slot;
        let (computed, rn) = self.w.prefill_while(caches, &prompt[n_past..], alive)?;
        tokens.extend_from_slice(&prompt[n_past..n_past + computed]);
        let Some(mut rn) = rn else {
            crate::log(crate::LOG_INFO, &format!("bankml: the client went away; stopped after {} of {} prompt tokens (kept in the slot)",
                                                 n_past + computed, prompt.len()));
            return Err(GONE.into());
        };
        let prompt_ns = t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();
        let (mut pending, mut text, mut n, mut finish) = (Vec::<u8>::new(), String::new(), 0usize, "length");
        let (mut out, mut grammar_ns, mut resampled) = (Vec::new(), 0u64, 0usize);
        loop {
            // llama-server b11192 checks the limit only once a token is generated (`n_gen > 0`, server-context.cpp
            // "check the limits"), so max_tokens 0 still samples one token, with its logprobs when asked for
            if n > 0 && max_tokens.is_some_and(|m| n >= m) {
                break;
            }
            let logits = self.w.logits(&rn)?;
            let next = match grammar.as_mut() {
                None => sampler.sample(&logits),
                Some(g) => {
                    let tg = Instant::now();
                    let v = self.grammar_vocab();
                    let (t, again) = sampler.sample_constrained(&logits, |id| g.allows(v, id), |c| g.apply(v, c));
                    g.accept(v, t)?;
                    resampled += again as usize;
                    grammar_ns += tg.elapsed().as_nanos() as u64;
                    t
                }
            };
            out.push(next);
            sampler.accept(next);
            if n_probs > 0 {
                let (p, top) = crate::sampler::token_probs(&logits, next, n_probs);
                probs.push(TokenLogprob { id: next, p, piece: Vec::new(), complete: true, emitted: false,
                                          top: top.into_iter().map(|t| (t.id, t.p, self.tok.token_bytes(t.id))).collect() });
            }
            if self.eog.contains(&next) {
                n += 1; // llama-server counts the end-of-turn token it sampled among the completion tokens
                finish = "stop";
                // it releases no text; it is a whole token only if nothing incomplete is pending before it
                let complete = pending.is_empty();
                if let Some(e) = probs.last_mut() {
                    e.complete = complete;
                }
                emit(Step { piece: "", complete, entry: probs.last().filter(|_| complete), n, eog: true });
                break;
            }
            n += 1;
            pending.extend(self.tok.token_bytes(next));
            let incomplete_after = std::str::from_utf8(&pending).is_err();
            if let Some(e) = probs.last_mut() {
                e.complete = !incomplete_after;
            }
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            // a step for every piece of text, and for every whole token even when it releases none
            if valid > 0 || !incomplete_after {
                let piece: String = String::from_utf8(pending.drain(..valid).collect()).unwrap();
                text.push_str(&piece);
                if valid > 0 {
                    if let Some(e) = probs.last_mut() {
                        e.emitted = true;
                        e.piece = piece.as_bytes().to_vec();
                    }
                    ttft_ns.get_or_insert(t0.elapsed().as_nanos() as u64);
                }
                if !emit(Step { piece: &piece, complete: !incomplete_after, entry: probs.last().filter(|_| !incomplete_after), n, eog: false }) {
                    finish = "stop";
                    break;
                }
            }
            if max_tokens.is_some_and(|m| n >= m) || tokens.len() + 1 >= self.n_ctx {
                break;
            }
            if !alive() {
                crate::log(crate::LOG_INFO, &format!("bankml: the client went away; stopped after {n} generated tokens"));
                return Err(GONE.into());
            }
            rn = self.w.decode(caches, next)?;
            tokens.push(next);
        }
        if !pending.is_empty() {
            let piece = String::from_utf8_lossy(&pending).into_owned();
            text.push_str(&piece);
            emit(Step { piece: &piece, complete: false, entry: None, n, eog: false });
        }
        let eval_ns = t1.elapsed().as_nanos() as u64;
        let energy_j = e0.zip(crate::sys::energy_uj()).map(|(a, b)| crate::sys::joules(a, b));
        let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        crate::metrics::push(crate::metrics::Record {
            at, model: self.model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), prompt_tokens: prompt.len(),
            cached_tokens: n_past, completion_tokens: n, prompt_ms: prompt_ns as f64 / 1e6, ttft_ms: ttft_ns.map(|t| t as f64 / 1e6),
            eval_ms: eval_ns as f64 / 1e6, finish: finish.to_string(), grammar_ms: grammar_ns as f64 / 1e6, resampled, energy_j,
        });
        Ok(Done { prompt_tokens: prompt.len(), cached_tokens: n_past, completion_tokens: n, finish_reason: finish, text, prompt_ns,
                  eval_ns, tokens: out, grammar_ns, resampled, ttft_ns, energy_j, probs })
    }

    /// Saves the slot (tokens and every layer's K and V) to `path`, as llama-server's `/slots/{id}?action=save`.
    ///
    /// Returns (tokens, bytes). bankML's own format: the magic line, `model_sha256`, the context, token and layer
    /// counts, each layer's cache, the tokens, then a sha256 of all of it. The file is written under a temporary
    /// name and renamed, so a crash leaves no partial file.
    pub fn save_slot(&self, path: &Path, model_sha256: &str) -> Result<(usize, u64), String> {
        let slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(SLOT_MAGIC);
        b.extend_from_slice(format!("{model_sha256:0<64}").as_bytes()[..64].as_ref());
        for v in [self.n_ctx as u64, slot.tokens.len() as u64, slot.caches.len() as u64] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for c in &slot.caches {
            // a q8_0 cache sets the width's top bit and stores its blocks as they are
            let q8 = c.kind == crate::forward::KvType::Q8_0;
            b.extend_from_slice(&(c.width as u64 | (q8 as u64) << 63).to_le_bytes());
            if q8 {
                b.extend_from_slice(&c.kq);
                b.extend_from_slice(&c.vq);
            }
            for x in c.k.iter().chain(&c.v) {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
        for t in &slot.tokens {
            b.extend_from_slice(&t.to_le_bytes());
        }
        let mut h = crate::sha256::Sha256::default();
        h.update(&b);
        b.extend_from_slice(&h.finish());
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &b).and_then(|_| std::fs::rename(&tmp, path)).map_err(|e| format!("Unable to save slot: {e}"))?;
        Ok((slot.tokens.len(), b.len() as u64))
    }

    /// Restores a slot saved by `save_slot` for this model and context; returns (tokens, bytes).
    ///
    /// On any error the slot is emptied and the reason returned as llama-server's "Unable to restore slot: …".
    pub fn restore_slot(&self, path: &Path, model_sha256: &str) -> Result<(usize, u64), String> {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        let r = (|| -> Result<(Vec<u32>, Vec<KvCache>, u64), String> {
            let b = std::fs::read(path).map_err(|e| e.to_string())?;
            if b.len() < SLOT_MAGIC.len() + 64 + 24 + 32 || &b[..SLOT_MAGIC.len()] != SLOT_MAGIC {
                return Err("not a bankML slot file".into());
            }
            let (body, sum) = b.split_at(b.len() - 32);
            let mut h = crate::sha256::Sha256::default();
            h.update(body);
            if h.finish()[..] != sum[..] {
                return Err("the slot file is damaged (its sha256 does not match)".into());
            }
            let mut at = SLOT_MAGIC.len();
            if &body[at..at + 64] != format!("{model_sha256:0<64}").as_bytes()[..64].as_ref() {
                return Err("the slot was saved for another model".into());
            }
            at += 64;
            let u64_at = |at: &mut usize| -> Result<u64, String> {
                let v = body.get(*at..*at + 8).ok_or("truncated")?;
                *at += 8;
                Ok(u64::from_le_bytes(v.as_chunks::<8>().0[0]))
            };
            let (n_ctx, n_tok, n_layers) = (u64_at(&mut at)?, u64_at(&mut at)? as usize, u64_at(&mut at)? as usize);
            if n_ctx as usize != self.n_ctx {
                return Err(format!("the slot was saved with a {n_ctx}-token context, this engine has {}", self.n_ctx));
            }
            if n_tok > self.n_ctx {
                return Err("Restored prompt does not fit in the slot context".into());
            }
            if n_layers != slot.caches.len() {
                return Err("the slot's layers do not match this model".into());
            }
            let mut caches = Vec::with_capacity(n_layers);
            for c in slot.caches.iter() {
                let w = u64_at(&mut at)?;
                let (width, q8) = ((w & !(1 << 63)) as usize, w >> 63 == 1);
                if q8 != (c.kind == crate::forward::KvType::Q8_0) {
                    return Err(format!("the slot was saved with a {} cache, this engine keeps {}",
                                       if q8 { "q8_0" } else { "f16" }, c.kind.name()));
                }
                if width != c.width {
                    return Err("the slot's cache shape does not match this model".into());
                }
                if q8 {
                    let mut k = KvCache { head_dim: c.head_dim, ..KvCache::of(c.kind, width) };
                    let nb = n_tok * k.row_bytes();
                    k.kq = body.get(at..at + nb).ok_or("truncated")?.to_vec();
                    k.vq = body.get(at + nb..at + 2 * nb).ok_or("truncated")?.to_vec();
                    at += 2 * nb;
                    caches.push(k);
                    continue;
                }
                let n = n_tok * width;
                let mut read = |n: usize| -> Result<Vec<u16>, String> {
                    let raw = body.get(at..at + 2 * n).ok_or("truncated")?;
                    at += 2 * n;
                    Ok(raw.as_chunks::<2>().0.iter().map(|p| u16::from_le_bytes(*p)).collect())
                };
                let (k, v) = (read(n)?, read(n)?);
                caches.push(KvCache { k, v, ..KvCache::new(width) });
            }
            let raw = body.get(at..at + 4 * n_tok).ok_or("truncated")?;
            let tokens: Vec<u32> = raw.as_chunks::<4>().0.iter().map(|p| u32::from_le_bytes(*p)).collect();
            if at + 4 * n_tok != body.len() {
                return Err("the slot file has trailing data".into());
            }
            if tokens.iter().any(|&t| t as usize >= self.tok.n_tokens()) {
                return Err("Invalid tokens in slot save file".into());
            }
            Ok((tokens, caches, b.len() as u64))
        })();
        match r {
            Ok((tokens, caches, n)) => {
                let len = tokens.len();
                (slot.tokens, slot.caches) = (tokens, caches);
                Ok((len, n))
            }
            Err(e) => {
                slot.tokens.clear();
                for c in slot.caches.iter_mut() {
                    c.truncate(0);
                }
                Err(format!("Unable to restore slot: {e}"))
            }
        }
    }

    /// Empties the slot, as `/slots/{id}?action=erase`; returns the number of tokens it held.
    pub fn erase_slot(&self) -> usize {
        let n = self.slot.lock().unwrap_or_else(|e| e.into_inner()).tokens.len();
        self.reset();
        n
    }

    /// Empties the slot; the next request computes its whole prompt, as a fresh llama-server or `cache_prompt: false`.
    pub fn reset(&self) {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        slot.tokens.clear();
        for c in slot.caches.iter_mut() {
            c.truncate(0);
        }
    }

    /// `/props`, in the shape Savante reads (the context and the model's path).
    pub fn props_json(&self) -> String {
        format!("{{\"model_path\": {}, \"n_ctx\": {}, \"default_generation_settings\": {{\"n_ctx\": {}, \"params\": {{\"temperature\": {}, \"top_k\": {}, \"top_p\": {}, \"min_p\": {}}}}}, \"build_info\": \"bankML {} native\"}}",
                crate::gguf::jstr(&self.model.to_string_lossy()), self.n_ctx, self.n_ctx, self.defaults.temp, self.defaults.top_k, self.defaults.top_p, self.defaults.min_p, crate::VERSION)
    }
}

/// llama-vocab.cpp's end-of-generation set for a GGUF: by name, plus the eos / eot / eom / FIM ids its header names.
pub fn eog_from_gguf(model: &Path, tok: &Tokenizer) -> Result<Vec<u32>, String> {
    let rep = crate::gguf::guard_file(model, crate::gguf::Engine::Mainline).map_err(|e| format!("{}: {e}", model.display()))?;
    let kv = rep.header.map(|h| h.kv).unwrap_or_default();
    let ids: Vec<u32> = ["eos", "eot", "eom", "fim_pad", "fim_rep", "fim_sep"].iter()
        .filter_map(|k| match kv.get(&format!("tokenizer.ggml.{k}_token_id")) {
            Some(crate::gguf::Val::U(v)) => u32::try_from(*v).ok(),
            Some(crate::gguf::Val::I(v)) => u32::try_from(*v).ok(),
            _ => None,
        })
        .collect();
    Ok(tok.eog_ids(&ids))
}

/// The request's sampling parameters over a model's defaults, as llama-server resolves them.
///
/// Soft limits clamp; hard limits and sampler errors refuse (in `Sampler::new`). Every sampler of the default chain
/// is reproduced; mirostat and a custom `samplers` order are refused.
pub fn sampling(defaults: Params, req: &Json) -> Result<Params, String> {
    let num = |k: &str| match req.get(k) {
        Some(Json::Num(n)) => Some(*n),
        _ => None,
    };
    let mut p = defaults;
    // soft limits clamp, as llama-server's `set_limits`
    let unit = |v: f64| v.clamp(0.0, 1.0) as f32;
    if let Some(v) = num("temperature") { p.temp = v.max(0.0) as f32 }
    if let Some(v) = num("top_k") { p.top_k = v as i32 }
    if let Some(v) = num("top_p") { p.top_p = unit(v) }
    if let Some(v) = num("min_p") { p.min_p = unit(v) }
    if let Some(v) = num("min_keep") { p.min_keep = v as usize }
    if let Some(v) = num("seed") { p.seed = v as i64 as u32 }
    if let Some(v) = num("n_probs") { p.n_probs = v.max(0.0) as usize }
    if let Some(Json::Bool(b)) = req.get("cache_prompt") { p.cache_prompt = *b }
    if let Some(v) = num("repeat_last_n") { p.penalty_last_n = v as i64 as i32 }
    if let Some(v) = num("repeat_penalty") { p.penalty_repeat = v as f32 }
    if let Some(v) = num("frequency_penalty") { p.penalty_freq = v as f32 }
    if let Some(v) = num("presence_penalty") { p.penalty_present = v as f32 }
    // the rest of the default chain
    if let Some(v) = num("typical_p") { p.typical_p = v as f32 }
    if let Some(v) = num("top_n_sigma") { p.top_n_sigma = v as f32 }
    if let Some(v) = num("xtc_probability") { p.xtc_probability = unit(v) }
    if let Some(v) = num("xtc_threshold") { p.xtc_threshold = unit(v) }
    if let Some(v) = num("dynatemp_range") { p.dynatemp_range = v as f32 }
    if let Some(v) = num("dynatemp_exponent") { p.dynatemp_exponent = v as f32 }
    if let Some(v) = num("dry_multiplier") { p.dry_multiplier = v as f32 }
    if let Some(v) = num("dry_base") { if v >= 1.0 { p.dry_base = v as f32 } else { p.dry_base = 1.75 } }
    if let Some(v) = num("dry_allowed_length") { p.dry_allowed_length = v as i64 as i32 }
    if let Some(v) = num("dry_penalty_last_n") { p.dry_penalty_last_n = v as i64 as i32 }
    match req.get("dry_sequence_breakers") {
        None | Some(Json::Null) => {}
        // anything but an array of strings reads as empty, as nlohmann's json_value falls back, and is refused
        Some(Json::Arr(a)) if a.iter().all(|x| matches!(x, Json::Str(_))) => {
            p.dry_sequence_breakers = a.iter().filter_map(|x| x.as_str().map(String::from)).collect()
        }
        Some(_) => p.dry_sequence_breakers = Vec::new(),
    }
    // what changes the answer and is not reproduced is refused, never ignored
    if num("mirostat").is_some_and(|v| v != 0.0) {
        return Err("mirostat: not reproduced; bankML reproduces llama-server's default sampler chain".into());
    }
    if req.get("samplers").is_some_and(|v| *v != Json::Null) {
        return Err("samplers: a custom sampler order is not reproduced; bankML runs llama-server's default order (penalties, dry, top-n-σ, top-k, typical-p, top-p, min-p, xtc, temperature)".into());
    }
    Ok(p)
}

/// What the receipt's `engine` says for a loaded model (and whether a GPU works in it).
pub fn engine_name(eng: &Native) -> String {
    let gpu = eng.w.gpu.as_ref().and_then(|g| g.lock().ok().map(|w| format!(", GPU {} on {:.0}% of each 1-bit matrix", w.name, w.share * 100.0))).unwrap_or_default();
    format!("bankML {} native: its own forward pass, token-identical to llama.cpp b11192 on its oracle{gpu}", crate::VERSION)
}

// ---------------------------------------------------------------- the registry and residency ----------

/// What the header says about a pinned file, read once when the registry is built.
#[derive(Debug, Clone)]
pub struct Info {
    pub arch: Option<String>,
    /// Most frequent non-F32 tensor type, e.g. `Q1_0`.
    pub quant: String,
    /// `general.size_label`, else the counted weights, e.g. `1.7B`.
    pub params: String,
    /// `Ok` when bankML's forward pass plays this file; otherwise the reason.
    pub native: Result<(), String>,
    pub defaults: Option<Params>,
}

/// One pinned GGUF: a name, the file, and the FORK.json that pins it.
#[derive(Debug, Clone)]
pub struct Entry {
    /// The file's stem, lower-cased (`bonsai-1.7b-q1_0`); `:latest` and the file name are accepted as aliases.
    pub name: String,
    pub file: String,
    /// Where the file was found; `None` when it is pinned but not on this machine.
    pub path: Option<PathBuf>,
    pub fork_json: String,
    pub sha256: String,
    pub bytes: u64,
    pub info: Info,
}

/// The models `serve --native` can be asked for.
///
/// The startup model and, with `--registry`, every GGUF pinned in the forks directory. Pins are read at start; a
/// load verifies the file against them again.
#[derive(Debug, Clone)]
pub struct Registry {
    pub entries: Vec<Entry>,
    pub default: usize,
}

/// A pinned name without its weight-type suffix (`-f16`, `-q1_0`, `-q2_0_g64`, `-q8_0`, …): the base an alias may use.
pub fn base_name(name: &str) -> &str {
    for t in ["-f16", "-bf16", "-f32", "-q1_0", "-q2_0_g64", "-q2_0", "-q8_0", "-q4_k_m", "-q4_0"] {
        if let Some(b) = name.strip_suffix(t) {
            return b;
        }
    }
    name
}

/// A model's name from its file name: the stem, lower-cased.
pub fn model_name(file: &str) -> String {
    let f = file.rsplit('/').next().unwrap_or(file);
    let stem = if f.to_ascii_lowercase().ends_with(".gguf") { &f[..f.len() - 5] } else { f };
    stem.to_lowercase()
}

/// Whether bankML's forward pass plays a file, judged from its header alone.
///
/// Checks the guard, architecture, weight type, tokenizer and chat template; the error names what is missing.
pub fn header_info(path: &Path) -> Info {
    let none = |why: String| Info { arch: None, quant: String::new(), params: String::new(), native: Err(why), defaults: None };
    let r = match crate::gguf::guard_file(path, crate::gguf::Engine::Mainline) {
        Ok(r) => r,
        Err(e) => return none(format!("cannot read {}: {e}", path.display())),
    };
    let quant = r.types.iter().filter(|(n, _)| n != "F32").max_by_key(|(_, c)| *c).map(|(n, _)| n.clone()).unwrap_or_else(|| "F32".into());
    let mut params = String::new();
    if let Some(h) = &r.header {
        let n: u128 = h.tensors.iter().map(|t| t.nelem()).sum();
        params = match h.kv.get("general.size_label") {
            Some(crate::gguf::Val::S(s)) if !s.is_empty() => s.clone(),
            _ if n >= 1_000_000_000 => format!("{:.1}B", n as f64 / 1e9),
            _ => format!("{:.0}M", n as f64 / 1e6),
        };
    }
    let native = match (&r.verdict, &r.header) {
        (crate::Verdict::Refuse(w), _) => Err(format!("the guard refuses it: {}", w.join("; "))),
        (crate::Verdict::NeedMore(_), _) => Err("the guard needs more header bytes: the file is truncated".into()),
        (_, None) => Err("no GGUF header".into()),
        // the architecture, the weight type and the graph's options, from the header (forward::plan); then the
        // tokenizer and the chat template this build reproduces
        (_, Some(h)) => crate::forward::plan(h).map(|_| ()).and_then(|_| crate::tokenizer::check_gguf(h)).and_then(|_| chat::check_template(path)),
    };
    let defaults = Params::from_gguf(path).ok();
    Info { arch: r.arch, quant, params, native, defaults }
}

impl Registry {
    /// Builds the registry: the startup model first (the default), then each `.gguf` a `*.FORK.json` in `dir` pins.
    ///
    /// Pinned files are looked for beside the startup model and in `dir`. A name pinned twice keeps its first entry.
    pub fn build(model: &Path, fork_json: &str, dir: Option<&Path>) -> Registry {
        let file = model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let entry = |file: &str, path: Option<PathBuf>, fork_json: &str| {
            let sha256 = crate::sha256::pinned_sha256(fork_json, file).unwrap_or_default();
            let bytes = path.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
            let info = match &path {
                Some(p) => header_info(p),
                None => Info { arch: None, quant: String::new(), params: String::new(), native: Err("pinned, but the file is not on this machine".into()), defaults: None },
            };
            Entry { name: model_name(file), file: file.to_string(), path, fork_json: fork_json.to_string(), sha256, bytes, info }
        };
        let mut entries = vec![entry(&file, Some(model.to_path_buf()), fork_json)];
        if let Some(dir) = dir {
            let mut forks: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path())
                .filter(|p| p.to_string_lossy().ends_with(".FORK.json")).collect();
            forks.sort();
            let look: Vec<PathBuf> = model.parent().into_iter().map(Path::to_path_buf).chain([dir.to_path_buf()]).collect();
            for f in forks {
                let Ok(text) = std::fs::read_to_string(&f) else { continue };
                let Some(v) = Json::parse(&text) else { continue };
                let Some(Json::Arr(files)) = v.get("files") else { continue };
                for p in files.iter().filter_map(|x| x.get("path").and_then(Json::as_str)) {
                    let base = p.rsplit('/').next().unwrap_or(p);
                    if !base.to_ascii_lowercase().ends_with(".gguf") || entries.iter().any(|e| e.name == model_name(base)) {
                        continue;
                    }
                    let found = look.iter().map(|d| d.join(base)).find(|c| c.is_file());
                    entries.push(entry(base, found, &text));
                }
            }
        }
        Registry { entries, default: 0 }
    }

    /// Resolves a request's `model` to a pinned entry.
    ///
    /// Absent or empty is the startup model; otherwise the name, with or without `:latest`, or the file name,
    /// case-insensitively.
    pub fn resolve(&self, model: Option<&str>) -> Result<&Entry, String> {
        let m = model.map(str::trim).unwrap_or("");
        if m.is_empty() {
            return Ok(&self.entries[self.default]);
        }
        let n = model_name(m.strip_suffix(":latest").unwrap_or(m));
        // a name without its weight-type suffix matches when exactly one pin has that base (`mindx-gen39` for
        // mindx-gen39-f16, as mindX's Ollama tag names it)
        let base: Vec<&Entry> = self.entries.iter().filter(|e| base_name(&e.name) == n).collect();
        self.entries.iter().find(|e| e.name == n).or(if base.len() == 1 { Some(base[0]) } else { None }).ok_or_else(|| {
            format!("model '{m}' not found: bankml serves pinned models only ({})",
                    self.entries.iter().map(|e| e.name.clone()).collect::<Vec<_>>().join(", "))
        })
    }
}

/// How long a model stays resident after a request: Ollama's `keep_alive`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeepAlive {
    Unload,
    For(Duration),
    Forever,
}

/// A model loaded behind the gate, as a request holds it.
#[derive(Clone)]
pub struct Loaded {
    pub name: String,
    pub native: Arc<Native>,
    pub verified: Arc<Verified>,
    /// The canonical path that was hashed, and its identity at that time.
    pub model: PathBuf,
    pub ident: FileIdent,
    pub engine: String,
    pub hashed_at: u64,
    pub size: u64,
    pub sha256: String,
}

pub struct Resident {
    pub loaded: Loaded,
    /// `None`: resident until told otherwise.
    pub expires: Option<SystemTime>,
}

/// One resident model at a time (Ollama's `MAX_LOADED_MODELS=1`).
///
/// `run` is held for a whole completion, load or unload, so the engine has one user and a switch never has two
/// models in memory; `cur` is held only briefly.
pub struct Residency {
    pub reg: Registry,
    pub n_ctx: usize,
    pub engine: crate::gguf::Engine,
    pub run: Mutex<()>,
    pub cur: Mutex<Option<Resident>>,
    /// The most recent verification (what `/bankml` reports when nothing is resident).
    pub last: Mutex<Option<Loaded>>,
    /// Derived models of the registry directory (`create.rs`), layered on these pins.
    pub derived: crate::create::Store,
    /// What `/api/ps` names and the context it reports (`shown_as`).
    pub shown: Mutex<Option<(String, usize)>>,
}

/// Ollama v0.13.3's `chatPrompt` truncation (`server/prompt.go`) over any token count.
///
/// Returns the messages kept and how many were cut. `count` gives the length of a candidate conversation's prompt.
pub fn fit_messages(msgs: &[Message], num_ctx: usize, count: impl Fn(&[Message]) -> Result<usize, String>) -> Result<(Vec<Message>, usize), String> {
    let last = msgs.len().saturating_sub(1);
    let (mut n, mut system) = (last, Vec::new());
    // in reverse: the conversation from message i, with the system messages before i, while it fits (the last
    // message is always kept; `system` stays what the last candidate had, as in Ollama's loop)
    for i in (0..last).rev() {
        system = msgs[..i].iter().filter(|m| m.role == "system").cloned().collect::<Vec<_>>();
        let cand: Vec<Message> = system.iter().chain(&msgs[i..]).cloned().collect();
        if count(&cand)? > num_ctx {
            break;
        }
        n = i;
    }
    let dropped = n - system.len();
    Ok((system.into_iter().chain(msgs[n..].iter().cloned()).collect(), dropped))
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Residency {
    /// Starts with the startup model, already verified by `serve::run`, resident until told otherwise.
    pub fn start(reg: Registry, n_ctx: usize, engine: crate::gguf::Engine, verified: Verified, model: PathBuf, id: FileIdent) -> Result<Residency, String> {
        let e = &reg.entries[reg.default];
        if let Err(why) = &e.info.native {
            return Err(format!("{}: bankML's native forward pass does not play it: {why}", e.file));
        }
        let native = Native::open(&model, n_ctx)?;
        let loaded = Loaded { name: e.name.clone(), engine: engine_name(&native), native: Arc::new(native), sha256: verified.model_sha256.clone(),
                              verified: Arc::new(verified), model, ident: id, hashed_at: now_secs(), size: e.bytes };
        Ok(Residency { reg, n_ctx, engine, run: Mutex::new(()), last: Mutex::new(Some(loaded.clone())),
                       cur: Mutex::new(Some(Resident { loaded, expires: None })), derived: Default::default(), shown: Mutex::new(None) })
    }

    /// Records the model a request named (`tag`, a pin's or a derived model's) and its context for `/api/ps`.
    ///
    /// Ollama's runner keeps the name of the model whose request loaded it and reloads when `num_ctx` differs;
    /// bankML keeps the weights and records the same: when they were (re)loaded for this request (`loaded`), when
    /// nothing is recorded yet, or when the context differs from the recorded one.
    pub fn shown_as(&self, tag: &str, num_ctx: usize, loaded: bool) {
        let mut s = self.shown.lock().unwrap_or_else(|e| e.into_inner());
        if loaded || s.as_ref().is_none_or(|(_, n)| *n != num_ctx) {
            *s = Some((tag.to_string(), num_ctx));
        }
    }

    /// What `/api/ps` names for the resident weights: the recorded request's model and context (see `shown_as`).
    pub fn shown(&self) -> Option<(String, usize)> {
        self.shown.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn lock_cur(&self) -> MutexGuard<'_, Option<Resident>> {
        self.cur.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn lock_run(&self) -> MutexGuard<'_, ()> {
        self.run.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The resident model, if any, without waiting for a completion (for the tokenizer, `/props`, `/api/ps`).
    pub fn peek(&self) -> Option<(Loaded, Option<SystemTime>)> {
        self.lock_cur().as_ref().map(|r| (r.loaded.clone(), r.expires))
    }

    /// The model `e` names, loaded; the caller holds `run`.
    ///
    /// Reuses the resident model if it is `e` and its file is unchanged; otherwise drops it, then verifies `e`
    /// (guard, then the sha256 pin) and opens it. Returns the load time in nanoseconds (0 when already resident).
    /// Errors carry their HTTP status.
    pub fn acquire(&self, _run: &MutexGuard<'_, ()>, e: &Entry) -> Result<(Loaded, u64), (u16, String)> {
        {
            let mut cur = self.lock_cur();
            if let Some(r) = cur.as_ref().filter(|r| r.loaded.name == e.name) {
                if ident(&r.loaded.model).ok() == Some(r.loaded.ident) {
                    return Ok((r.loaded.clone(), 0));
                }
                *cur = None;
                return Err((503, format!("{} changed since bankml verified it: unloaded, and refused; the next request verifies it again", e.file)));
            }
        }
        // what can be refused without reading the file is refused before the resident model is touched
        let path = e.path.as_ref().ok_or((404, format!("{} is pinned but not on this machine", e.file)))?;
        if let Err(why) = &e.info.native {
            return Err((400, format!("{}: bankML's native forward pass does not play it: {why}", e.name)));
        }
        *self.lock_cur() = None; // one resident model: the old one goes before the new one is read
        let t0 = Instant::now();
        let before = ident(path).map_err(|err| (500, format!("{}: {err}", path.display())))?;
        let verified = crate::verify(path, &e.fork_json, self.engine).map_err(|why| (400, format!("refuse: {why}")))?;
        let model = path.canonicalize().map_err(|err| (500, format!("{}: {err}", path.display())))?;
        let id = ident(&model).map_err(|err| (500, format!("{}: {err}", model.display())))?;
        if id != before {
            return Err((503, format!("{} changed while it was being hashed: refused", model.display())));
        }
        crate::log(crate::LOG_INFO, &format!("bankml serve --native: {} verified (sha256 {}); loading it", e.name, verified.model_sha256));
        let native = Native::open(&model, self.n_ctx).map_err(|why| (500, why))?;
        let loaded = Loaded { name: e.name.clone(), engine: engine_name(&native), native: Arc::new(native), sha256: verified.model_sha256.clone(),
                              verified: Arc::new(verified), model, ident: id, hashed_at: now_secs(), size: e.bytes };
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Some(loaded.clone());
        *self.lock_cur() = Some(Resident { loaded: loaded.clone(), expires: None });
        Ok((loaded, t0.elapsed().as_nanos() as u64))
    }

    /// Starts the keep-alive after a request (`Unload` drops the model at once).
    pub fn touch(&self, name: &str, ka: KeepAlive) {
        let mut cur = self.lock_cur();
        if cur.as_ref().is_some_and(|r| r.loaded.name == name) {
            match ka {
                KeepAlive::Unload => {
                    *cur = None;
                    crate::log(crate::LOG_INFO, &format!("bankml serve --native: {name} unloaded (keep_alive 0)"));
                }
                KeepAlive::For(d) => cur.as_mut().unwrap().expires = Some(SystemTime::now() + d),
                KeepAlive::Forever => cur.as_mut().unwrap().expires = None,
            }
        }
    }

    /// The resident model, or the startup model loaded (for llama-server's endpoints, which name no model).
    pub fn current_or_default(&self) -> Result<Loaded, (u16, String)> {
        if let Some((l, _)) = self.peek() {
            return Ok(l);
        }
        let run = self.lock_run();
        let e = &self.reg.entries[self.reg.default];
        let (l, _) = self.acquire(&run, e)?;
        self.touch(&e.name, KeepAlive::Forever);
        Ok(l)
    }

    /// Drops the resident model once its keep-alive has run out, unless a request is using the engine.
    pub fn reap(&self) {
        let Ok(_run) = self.run.try_lock() else { return };
        let mut cur = self.lock_cur();
        if let Some(r) = cur.as_ref().filter(|r| r.expires.is_some_and(|t| t <= SystemTime::now())) {
            crate::log(crate::LOG_INFO, &format!("bankml serve --native: {} unloaded (keep_alive expired)", r.loaded.name));
            *cur = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ollama's chatPrompt truncation (v0.13.3), on a count where each message costs its content's length
    #[test]
    fn fit_messages_as_ollama() {
        let m = |r: &str, c: &str| Message::new(r, c);
        let count = |ms: &[Message]| Ok::<usize, String>(ms.iter().map(|x| x.content.len()).sum());
        let names = |ms: &[Message]| ms.iter().map(|x| x.content.clone()).collect::<Vec<_>>().join(",");
        let conv = [m("system", "SS"), m("user", "u1u1"), m("assistant", "a1a1"), m("user", "u2u2"), m("assistant", "a2"), m("user", "q")];
        // everything fits
        assert_eq!(fit_messages(&conv, 100, count).map(|(k, d)| (names(&k), d)), Ok(("SS,u1u1,a1a1,u2u2,a2,q".into(), 0)));
        // the oldest turns go, the system message stays
        assert_eq!(fit_messages(&conv, 9, count).map(|(k, d)| (names(&k), d)), Ok(("SS,u2u2,a2,q".into(), 2)));
        assert_eq!(fit_messages(&conv, 3, count).map(|(k, d)| (names(&k), d)), Ok(("SS,q".into(), 4)));
        // the last message always stays, even when it alone does not fit (the caller refuses then)
        assert_eq!(fit_messages(&conv, 1, count).map(|(k, d)| (names(&k), d)), Ok(("SS,q".into(), 4)));
        // Ollama's quirk: a system message that is itself the first one cut is dropped
        let c2 = [m("system", "SSSSSSSSSS"), m("user", "q")];
        assert_eq!(fit_messages(&c2, 5, count).map(|(k, d)| (names(&k), d)), Ok(("q".into(), 1)));
        assert_eq!(fit_messages(&[m("user", "long question")], 1, count).map(|(k, d)| (names(&k), d)), Ok(("long question".into(), 0)));
    }

    /// Savante-style conversations in one engine against llama-server b11192's chat endpoint (testing/serve_oracle.py).
    ///
    /// Every turn must match the answer text, prompt and completion counts, and cached prompt tokens.
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + its serve-*.jsonl (testing/serve_oracle.py); --release"]
    fn oracle_native_serve() {
        native_serve("Bonsai-8B-Q1_0");
    }

    /// The same conversations, then JSON mode, on the O4 models (Bonsai-1.7B, SmolLM2-135M-Instruct, mindx-gen39).
    #[test]
    #[ignore = "needs .models/{Bonsai-1.7B-Q1_0,SmolLM2-135M-Instruct-F16,mindx-gen39-F16}.gguf + their serve-*/json-*.jsonl; --release"]
    fn oracle_native_serve_o4() {
        for stem in ["Bonsai-1.7B-Q1_0", "SmolLM2-135M-Instruct-F16", "mindx-gen39-F16"] {
            native_serve(stem);
            json_replay(stem, "json");
        }
    }

    /// `cache_prompt: false` computes the whole prompt (llama-server's n_past = 0): after a warm-up of the same prompt,
    /// the answer, its tokens and its zero cache count are an empty slot's. (This is what made decode_ab.py's rounds
    /// differ: llama-server honoured the flag, bankML reused the warm-up's prefix.)
    #[test]
    #[ignore = "needs .models/SmolLM2-135M-Instruct-F16.gguf; --release"]
    fn cache_prompt_false_is_an_empty_slot() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let eng = Native::open(&dir.join("SmolLM2-135M-Instruct-F16.gguf"), 2048).unwrap();
        let req = Json::parse(r#"{"messages": [{"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": "Explain in a short paragraph why the sky is blue, and what changes at sunset."}],
            "temperature": 0, "cache_prompt": false}"#).unwrap();
        let prompt = eng.prompt(req.get("messages").unwrap()).unwrap();
        let params = eng.params(&req).unwrap();
        assert!(!params.cache_prompt);
        eng.reset();
        let empty = eng.complete(&prompt, params.clone(), Some(32), None, |_| true).unwrap();
        eng.complete(&prompt, params.clone(), Some(4), None, |_| true).unwrap(); // the warm-up fills the slot
        let off = eng.complete(&prompt, params.clone(), Some(32), None, |_| true).unwrap();
        assert_eq!((off.tokens.clone(), off.text.clone(), off.cached_tokens), (empty.tokens.clone(), empty.text.clone(), 0));
        let on = eng.complete(&prompt, Params { cache_prompt: true, ..params }, Some(32), None, |_| true).unwrap();
        assert_eq!(on.cached_tokens, prompt.len() - 1, "with the cache on, the whole prompt but one token is reused");
        eprintln!("cache_prompt: false after a warm-up = an empty slot ({} tokens, cache_n 0); true reuses {} of {}",
                  off.tokens.len(), on.cached_tokens, prompt.len());
    }

    fn native_serve(stem: &str) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let eng = Native::open(&dir.join(format!("{stem}.gguf")), 2048).unwrap();
        let turns = std::fs::read_to_string(dir.join(format!("oracle-forward/serve-{stem}.jsonl"))).unwrap();
        let num = |v: &Json, k: &str| match v.get(k) { Some(Json::Num(n)) => *n as usize, _ => usize::MAX };
        let (mut n, mut ok) = (0, 0);
        let t0 = std::time::Instant::now();
        for line in turns.lines() {
            let t = Json::parse(line).unwrap();
            let req = t.get("request").unwrap();
            let prompt = eng.prompt(req.get("messages").unwrap()).unwrap();
            let params = eng.params(req).unwrap();
            let max = req.get("max_tokens").and_then(|v| match v { Json::Num(x) => Some(*x as usize), _ => None });
            let d = eng.complete(&prompt, params, max, None, |_| true).unwrap();
            n += 1;
            let want_text = t.get("text").and_then(Json::as_str).unwrap();
            let good = d.text == want_text && d.prompt_tokens == num(&t, "prompt_tokens") && d.completion_tokens == num(&t, "completion_tokens")
                && d.cached_tokens == num(&t, "cache_n") && Some(d.finish_reason) == t.get("finish_reason").and_then(Json::as_str);
            ok += good as usize;
            if !good {
                eprintln!("  turn {n}: got prompt {} (cached {}) +{} {:?} {:?}; want prompt {} (cached {}) +{} {:?} {:?}", d.prompt_tokens, d.cached_tokens,
                          d.completion_tokens, d.finish_reason, &d.text[..d.text.len().min(60)], num(&t, "prompt_tokens"), num(&t, "cache_n"),
                          num(&t, "completion_tokens"), t.get("finish_reason").and_then(Json::as_str), &want_text[..want_text.len().min(60)]);
            }
        }
        eprintln!("native serve oracle: {stem}: {ok} of {n} conversation turns identical to llama-server b11192's /v1/chat/completions (text, token counts, prompt-cache reuse) — {:.0} s",
                  t0.elapsed().as_secs_f64());
        assert_eq!(ok, n);
    }

    /// Replays llama-server b11192's answers under JSON mode and user grammars (testing/json_oracle.py).
    ///
    /// Each from an empty cache: the same tokens (end token included), raw text, message content, finish reason and
    /// counts, and the same reported grammar and generation prompt.
    fn json_replay(stem: &str, kind: &str) {
        use crate::grammar::{json_message, json_object_grammar, Constraint};
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let eng = Native::open(&dir.join(format!("{stem}.gguf")), 2048).unwrap();
        let rec = std::fs::read_to_string(dir.join(format!("oracle-json/{kind}-{stem}.jsonl"))).unwrap();
        let num = |v: &Json, k: &str| match v.get(k) { Some(Json::Num(n)) => *n as usize, _ => usize::MAX };
        let s = |v: &Json, k: &str| v.get(k).and_then(Json::as_str).unwrap_or("").to_string();
        let (mut n, mut ok, mut toks, mut redrawn, mut gns) = (0, 0, 0, 0, 0u64);
        let t0 = std::time::Instant::now();
        for line in rec.lines() {
            let c = Json::parse(line).unwrap();
            let req = c.get("request").unwrap();
            // a schema record carries the request as sent, read exactly (a float literal stays one)
            let constraint = match c.get("request_text").and_then(Json::as_str) {
                Some(t) => crate::grammar::from_openai_text(req, t).unwrap(),
                None => crate::grammar::from_openai(req).unwrap(),
            };
            match &constraint {
                Constraint::JsonObject => {
                    let (text, prefill) = json_object_grammar(eng.template);
                    assert_eq!(s(&c, "grammar"), text, "{}: the server's grammar", s(&c, "name"));
                    assert_eq!(s(&c, "generation_prompt"), prefill);
                }
                Constraint::Gbnf(g) => assert_eq!(&s(&c, "grammar"), g),
                Constraint::Schema(sc) => {
                    let g = crate::schema::chat_grammar(sc, eng.template).unwrap().0;
                    assert_eq!(s(&c, "grammar"), g, "{}: the server's schema grammar", s(&c, "name"));
                    assert_eq!(s(&c, "generation_prompt"), eng.template.generation_prompt());
                }
                Constraint::None => panic!("an unconstrained record"),
            }
            assert_eq!(c.get("grammar_lazy"), Some(&Json::Bool(false)));
            let prompt = eng.prompt(req.get("messages").unwrap()).unwrap();
            let params = eng.params(req).unwrap();
            let max = match req.get("max_tokens") { Some(Json::Num(x)) => Some(*x as usize), _ => None };
            eng.reset();
            let d = eng.complete(&prompt, params, max, eng.grammar(&constraint).unwrap(), |_| true).unwrap();
            let want: Vec<u32> = match c.get("tokens") { Some(Json::Arr(a)) => a.iter().map(|x| match x { Json::Num(n) => *n as u32, _ => panic!() }).collect(), _ => panic!() };
            let content = if matches!(constraint, Constraint::JsonObject | Constraint::Schema(_)) { json_message(&d.text).to_string() } else { d.text.clone() };
            let good = d.tokens == want && d.text == s(&c, "raw") && content == s(&c, "content") && d.finish_reason == s(&c, "finish_reason")
                && d.prompt_tokens == num(&c, "prompt_tokens") && d.completion_tokens == num(&c, "completion_tokens");
            n += 1;
            ok += good as usize;
            toks += d.tokens.len();
            redrawn += d.resampled;
            gns += d.grammar_ns;
            if !good {
                let at = d.tokens.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(d.tokens.len().min(want.len()));
                eprintln!("  {}: first token difference at {at} of {} / {}; content {:?} want {:?}; finish {} want {}", s(&c, "name"), d.tokens.len(), want.len(),
                          &content[..content.len().min(60)], &s(&c, "content")[..s(&c, "content").len().min(60)], d.finish_reason, s(&c, "finish_reason"));
            }
        }
        eprintln!("{kind} oracle ({stem}): {ok} of {n} constrained answers token-identical to llama-server b11192 ({toks} tokens, {redrawn} redrawn \
                   under the mask; grammar {:.2} ms per token) — {:.0} s", gns as f64 / 1e6 / toks.max(1) as f64, t0.elapsed().as_secs_f64());
        assert_eq!(ok, n);
    }

    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + oracle-json/json-*.jsonl (testing/json_oracle.py --record); --release"]
    fn oracle_json_mode() {
        json_replay("Bonsai-8B-Q1_0", "json");
    }

    #[test]
    #[ignore = "needs .models/Ternary-Bonsai-8B-Q2_0_g64.gguf + oracle-json/json-*.jsonl (testing/json_oracle.py --record); --release"]
    fn oracle_json_mode_ternary() {
        json_replay("Ternary-Bonsai-8B-Q2_0_g64", "json");
    }

    /// llama-server b11192's answers under JSON schemas (testing/json_schema_oracle.py --record).
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + oracle-json/schema-*.jsonl (testing/json_schema_oracle.py --record); --release"]
    fn oracle_json_schema() {
        json_replay("Bonsai-8B-Q1_0", "schema");
    }

    #[test]
    #[ignore = "needs .models/Ternary-Bonsai-8B-Q2_0_g64.gguf + oracle-json/schema-*.jsonl (testing/json_schema_oracle.py --record); --release"]
    fn oracle_json_schema_ternary() {
        json_replay("Ternary-Bonsai-8B-Q2_0_g64", "schema");
    }

    /// Replays llama-server b11192's penalties sampler, token for token (testing/penalty_oracle.py).
    ///
    /// Each request must resolve to the parameters the server read back and give its tokens; what the server refused
    /// is refused with its message.
    fn penalty_replay(stem: &str) {
        sampler_replay("penalty", stem)
    }

    /// The same for any record kind of testing/penalty_oracle.py (`penalty`, `sampler`).
    ///
    /// Every sampler parameter the record carries must equal the resolved one, then the tokens must match.
    fn sampler_replay(kind: &str, stem: &str) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let eng = Native::open(&dir.join(format!("{stem}.gguf")), 2048).unwrap();
        let rec = std::fs::read_to_string(dir.join(format!("oracle-forward/{kind}-{stem}.jsonl"))).unwrap();
        let ids = |v: &Json, k: &str| -> Vec<u32> { match v.get(k) { Some(Json::Arr(a)) => a.iter().map(|x| match x { Json::Num(n) => *n as u32, _ => panic!() }).collect(), _ => panic!("no {k}") } };
        let num = |v: &Json, k: &str| match v.get(k) { Some(Json::Num(n)) => *n, _ => panic!("no {k}") };
        let (mut n, mut same, mut refused, mut toks, mut bad) = (0, 0, 0, 0, Vec::new());
        for line in rec.lines() {
            let c = Json::parse(line).unwrap();
            let req = c.get("request").unwrap();
            n += 1;
            let resolved = eng.params(req).and_then(|p| crate::sampler::Sampler::new(p.clone()).map(|_| p));
            if let Some(Json::Str(want)) = c.get("error").and_then(|e| e.get("message")) {
                match resolved {
                    Err(m) if m == *want => refused += 1,
                    other => bad.push(format!("case {n}: llama-server refused ({want}); bankML {:?}", other.map(|_| "accepted"))),
                }
                continue;
            }
            let p = resolved.unwrap();
            let g = c.get("params").unwrap();
            // every parameter the server reported (a record carries the keys its recorder knew)
            let floats: [(&str, f32); 15] = [("temperature", p.temp), ("top_p", p.top_p), ("min_p", p.min_p), ("repeat_penalty", p.penalty_repeat),
                ("frequency_penalty", p.penalty_freq), ("presence_penalty", p.penalty_present), ("typical_p", p.typical_p),
                ("top_n_sigma", p.top_n_sigma), ("xtc_probability", p.xtc_probability), ("xtc_threshold", p.xtc_threshold),
                ("dynatemp_range", p.dynatemp_range), ("dynatemp_exponent", p.dynatemp_exponent), ("dry_multiplier", p.dry_multiplier),
                ("dry_base", p.dry_base), ("min_keep", p.min_keep as f32)];
            for (k, v) in floats {
                if let Some(Json::Num(w)) = g.get(k) {
                    assert_eq!(v, *w as f32, "case {n}: {k} resolves to the server's");
                }
            }
            let ints: [(&str, i64); 5] = [("top_k", p.top_k as i64), ("repeat_last_n", p.penalty_last_n as i64), ("seed", p.seed as i64),
                ("dry_allowed_length", p.dry_allowed_length as i64), ("dry_penalty_last_n", p.dry_penalty_last_n as i64)];
            for (k, v) in ints {
                if let Some(Json::Num(w)) = g.get(k) {
                    assert_eq!(v, *w as i64, "case {n}: {k} resolves to the server's");
                }
            }
            if let Some(Json::Arr(b)) = g.get("dry_sequence_breakers") {
                let want: Vec<&str> = b.iter().filter_map(Json::as_str).collect();
                assert_eq!(p.dry_sequence_breakers, want, "case {n}: dry_sequence_breakers resolve to the server's");
            }
            let want = ids(&c, "ids");
            eng.reset();
            let max = num(req, "n_predict") as usize;
            let d = eng.complete(&ids(&c, "prompt_ids"), p, Some(max), None, |_| true).unwrap();
            // /completion's `tokens` hold what was generated, the end token included when one ended the answer
            toks += want.len();
            if d.tokens == want {
                same += 1;
            } else {
                let at = d.tokens.iter().zip(&want).position(|(a, b)| a != b);
                bad.push(format!("case {n} {}: first difference at {at:?} (bankML {} tokens, llama-server {})", req_text(req), d.tokens.len(), want.len()));
            }
        }
        eprintln!("{kind} oracle ({stem}): {same} of {} answers token-identical to llama-server b11192 ({toks} tokens), {refused} refusals with its message",
                  n - refused);
        for b in bad.iter().take(12) {
            eprintln!("  {b}");
        }
        assert!(bad.is_empty(), "{} differ", bad.len());
    }

    fn req_text(r: &Json) -> String {
        let Json::Obj(f) = r else { return String::new() };
        f.iter().filter(|(k, _)| !matches!(k.as_str(), "prompt" | "n_predict" | "cache_prompt" | "return_tokens"))
            .map(|(k, v)| format!("{k}={}", match v { Json::Num(x) => x.to_string(), _ => "…".into() })).collect::<Vec<_>>().join(" ")
    }

    #[test]
    #[ignore = "needs .models/{mindx-gen39-F16,Bonsai-1.7B-Q1_0}.gguf + oracle-forward/penalty-*.jsonl (testing/penalty_oracle.py); --release"]
    fn oracle_penalties() {
        for stem in std::env::var("BANKML_PENALTY_STEMS").unwrap_or("mindx-gen39-F16,Bonsai-1.7B-Q1_0".into()).split(',') {
            penalty_replay(stem);
        }
    }

    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + oracle-forward/penalty-Bonsai-8B-Q1_0.jsonl (testing/penalty_oracle.py); --release"]
    fn oracle_penalties_8b() {
        penalty_replay("Bonsai-8B-Q1_0");
    }

    /// Typical-p, top-n-σ, XTC, dynamic temperature and DRY, token for token (`--kind sampler` records).
    #[test]
    #[ignore = "needs .models/{mindx-gen39-F16,Bonsai-1.7B-Q1_0}.gguf + oracle-forward/sampler-*.jsonl (testing/penalty_oracle.py --kind sampler); --release"]
    fn oracle_samplers() {
        for stem in std::env::var("BANKML_SAMPLER_STEMS").unwrap_or("mindx-gen39-F16,Bonsai-1.7B-Q1_0".into()).split(',') {
            sampler_replay("sampler", stem);
        }
    }

    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + oracle-forward/sampler-Bonsai-8B-Q1_0.jsonl (testing/penalty_oracle.py --kind sampler); --release"]
    fn oracle_samplers_8b() {
        sampler_replay("sampler", "Bonsai-8B-Q1_0");
    }

    /// The schema oracle on the O4 models; the ChatML templates' schema grammar has no reasoning block.
    #[test]
    #[ignore = "needs .models/{Bonsai-1.7B-Q1_0,SmolLM2-135M-Instruct-F16,mindx-gen39-F16}.gguf + oracle-json/schema-*.jsonl (testing/json_schema_oracle.py --record); --release"]
    fn oracle_json_schema_o4() {
        for stem in ["Bonsai-1.7B-Q1_0", "SmolLM2-135M-Instruct-F16", "mindx-gen39-F16"] {
            json_replay(stem, "schema");
        }
    }

}

