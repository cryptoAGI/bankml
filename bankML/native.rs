// SPDX-License-Identifier: MIT OR Apache-2.0
//! The native engine behind `bankml serve --native` (0.3.0): chat completions from bankML's own forward pass,
//! token-identical to llama-server b11192 on its oracle — the same template, tokenizer, forward pass (every kernel
//! llama.cpp chooses), sampler chain and, across requests, the same prompt cache:
//!
//! - one slot, as `-np 1`; a request reuses the longest common prefix of the tokens already in the slot's KV cache
//!   and its own prompt, less one token when the whole prompt is cached (llama-server evaluates at least one), then
//!   truncates the cache there and computes the rest in micro-batches of 512 (`Weights::prefill`), as
//!   `server-context.cpp` does — so the kernels each row takes are the ones llama-server's rows take;
//! - the sampling parameters are the request's over the model's GGUF defaults, as llama-server resolves them; a
//!   sampler bankML does not reproduce (penalties, dry, typical-p, xtc, top-n-σ, dynamic temperature) is refused
//!   with a reason, never approximated;
//! - the answer streams as whole UTF-8 characters; the end-of-turn token ends it, is not part of the text, and is
//!   counted among the completion tokens as llama-server counts it.
//!
//! 0.3.1 adds the model's lifecycle (`Registry`, `Residency`): every pinned GGUF in the forks directory gets a name,
//! one model is resident at a time, a load runs the full `verify` (guard, then the sha256 pin) exactly as `serve`'s
//! start does, and an idle model is dropped (its mmap with it) when its keep-alive runs out — Ollama's lifecycle,
//! bankML's gate.

use crate::chat::{self, Message};
use crate::forward::{KvCache, Weights};
use crate::sampler::{Params, Sampler};
use crate::serve::{ident, FileIdent, Json};
use crate::tokenizer::Tokenizer;
use crate::Verified;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Slot {
    tokens: Vec<u32>,
    caches: Vec<KvCache>,
}

pub struct Native {
    pub w: Weights,
    pub tok: Tokenizer,
    pub defaults: Params,
    pub n_ctx: usize,
    pub model: PathBuf,
    /// the model's chat template (by the sha256 of its text, `chat::TEMPLATES`)
    pub template: chat::Template,
    eog: Vec<u32>,
    slot: Mutex<Slot>,
    /// every token's piece and the end set, as the grammar reads them; built on the first constrained request
    gvocab: std::sync::OnceLock<crate::grammar::Vocab>,
    /// the JSON-mode grammar, parsed once
    json_rules: std::sync::OnceLock<Arc<crate::grammar::Rules>>,
}

/// How a completion ended, and its counts (llama-server's `usage` and `timings.cache_n`).
#[derive(Debug, Clone)]
pub struct Done {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: &'static str,
    pub text: String,
    /// wall time of the prompt's computation (the uncached part) and of the generation after it, in nanoseconds
    pub prompt_ns: u64,
    pub eval_ns: u64,
    /// the generated token ids (with the end token, when one ended the answer)
    pub tokens: Vec<u32>,
    /// under a grammar: the time spent in it (the checks, the masks, the accepts) and how many tokens were redrawn
    pub grammar_ns: u64,
    pub resampled: usize,
}

impl Native {
    pub fn open(model: &Path, n_ctx: usize) -> Result<Native, String> {
        let template = chat::template_of(model)?;
        let tok = Tokenizer::from_gguf(model)?;
        let w = Weights::open(model)?;
        let defaults = Params::from_gguf(model)?;
        let eog = eog_from_gguf(model, &tok)?;
        let caches = w.caches();
        Ok(Native { w, tok, defaults, n_ctx, model: model.to_path_buf(), template, eog, slot: Mutex::new(Slot { tokens: Vec::new(), caches }),
                    gvocab: Default::default(), json_rules: Default::default() })
    }

    /// The vocabulary as the grammar reads it (built once, ~10 MB, on the first constrained request).
    pub fn grammar_vocab(&self) -> &crate::grammar::Vocab {
        self.gvocab.get_or_init(|| crate::grammar::Vocab::new((0..self.tok.n_tokens() as u32).map(|i| self.tok.piece(i)).collect(), &self.eog))
    }

    /// The grammar a request's constraint asks for, in its starting state: JSON mode's grammar with the generation
    /// prompt already taken in (llama-server prefills an output-format grammar), or the user's GBNF as it is.
    pub fn grammar(&self, c: &crate::grammar::Constraint) -> Result<Option<crate::grammar::Grammar>, String> {
        use crate::grammar::{json_object_grammar, Constraint, Grammar, Rules};
        let tokenize = |b: &[u8]| self.tok.encode(&String::from_utf8_lossy(b), true);
        match c {
            Constraint::None => Ok(None),
            Constraint::JsonObject | Constraint::Schema(_) => {
                // the template's own output-format grammar (JSON mode's, or the schema's through the template's chat
                // parser): its root opens with the template's generation prompt, which is prefilled
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

    /// 0.3.5: the prompt under a `num_ctx` (a request's option or a derived model's `PARAMETER num_ctx`), fitted as
    /// Ollama fits it (`server/prompt.go` `chatPrompt`, v0.13.3): walking back from the last message, each earlier
    /// message is kept while the conversation from it — with the system messages before it — still fits; the last
    /// message always stays, and so do the system messages before the first one kept (a system message that is itself
    /// the first one cut is dropped, as Ollama drops it). The tokens counted are the prompt bankML answers (the GGUF's
    /// template, as llama-server renders it). A prompt that still does not fit is refused: Ollama's runner would cut
    /// tokens out of its middle (`num_keep`) and shift the cache during the answer, which bankML does not reproduce.
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

    /// One completion. `emit` receives the answer as whole UTF-8 pieces as they come, and returns false to stop.
    /// With a grammar (`Native::grammar`) every token is drawn as `common_sampler_sample` draws it under one.
    pub fn complete(&self, prompt: &[u32], params: Params, max_tokens: Option<usize>, mut grammar: Option<crate::grammar::Grammar>,
                    mut emit: impl FnMut(&str) -> bool) -> Result<Done, String> {
        if prompt.is_empty() {
            return Err("an empty prompt".into());
        }
        if prompt.len() >= self.n_ctx {
            return Err(format!("the prompt has {} tokens; the context is {}", prompt.len(), self.n_ctx));
        }
        let mut sampler = Sampler::new(params)?;
        // O2: llama-server accepts the whole prompt into the penalties' window before the first draw, cached or not
        for &t in prompt {
            sampler.accept(t);
        }
        let mut slot = self.slot.lock().map_err(|_| "the slot is poisoned")?;
        let t0 = Instant::now();
        // llama-server's prompt cache: the longest common prefix, less one when the whole prompt is cached
        let mut n_past = slot.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        if n_past == prompt.len() {
            n_past -= 1;
        }
        slot.tokens.truncate(n_past);
        for c in slot.caches.iter_mut() {
            c.truncate(n_past);
        }
        let Slot { tokens, caches } = &mut *slot;
        let mut rn = self.w.prefill(caches, &prompt[n_past..], |_, _, _| {})?;
        tokens.extend_from_slice(&prompt[n_past..]);
        let prompt_ns = t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();
        let (mut pending, mut text, mut n, mut finish) = (Vec::<u8>::new(), String::new(), 0usize, "length");
        let (mut out, mut grammar_ns, mut resampled) = (Vec::new(), 0u64, 0usize);
        loop {
            if max_tokens.is_some_and(|m| n >= m) {
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
            if self.eog.contains(&next) {
                n += 1; // llama-server counts the end-of-turn token it sampled among the completion tokens
                finish = "stop";
                break;
            }
            n += 1;
            pending.extend(self.tok.token_bytes(next));
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                let piece: String = String::from_utf8(pending.drain(..valid).collect()).unwrap();
                text.push_str(&piece);
                if !emit(&piece) {
                    finish = "stop";
                    break;
                }
            }
            if max_tokens.is_some_and(|m| n >= m) || tokens.len() + 1 >= self.n_ctx {
                break;
            }
            rn = self.w.decode(caches, next)?;
            tokens.push(next);
        }
        if !pending.is_empty() {
            let piece = String::from_utf8_lossy(&pending).into_owned();
            text.push_str(&piece);
            emit(&piece);
        }
        Ok(Done { prompt_tokens: prompt.len(), cached_tokens: n_past, completion_tokens: n, finish_reason: finish, text, prompt_ns,
                  eval_ns: t1.elapsed().as_nanos() as u64, tokens: out, grammar_ns, resampled })
    }

    /// Empty the slot (the next request computes its whole prompt, as a fresh llama-server or `cache_prompt: false`).
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

/// The request's sampling parameters over a model's defaults (llama-server's resolution); a sampler bankML does not
/// reproduce (dry, typical-p, xtc, top-n-σ, dynamic temperature) is refused with a reason. O2: the repeat, frequency
/// and presence penalties and `repeat_last_n` are reproduced (`Sampler::new` refuses what llama-server refuses).
pub fn sampling(defaults: Params, req: &Json) -> Result<Params, String> {
    let num = |k: &str| match req.get(k) {
        Some(Json::Num(n)) => Some(*n),
        _ => None,
    };
    let mut p = defaults;
    if let Some(v) = num("temperature") { p.temp = v as f32 }
    if let Some(v) = num("top_k") { p.top_k = v as i32 }
    if let Some(v) = num("top_p") { p.top_p = v as f32 }
    if let Some(v) = num("min_p") { p.min_p = v as f32 }
    if let Some(v) = num("min_keep") { p.min_keep = v as usize }
    if let Some(v) = num("seed") { p.seed = v as i64 as u32 }
    if let Some(v) = num("repeat_last_n") { p.penalty_last_n = v as i64 as i32 }
    if let Some(v) = num("repeat_penalty") { p.penalty_repeat = v as f32 }
    if let Some(v) = num("frequency_penalty") { p.penalty_freq = v as f32 }
    if let Some(v) = num("presence_penalty") { p.penalty_present = v as f32 }
    for (k, neutral) in [("typical_p", 1.0), ("xtc_probability", 0.0), ("dry_multiplier", 0.0), ("dynatemp_range", 0.0)] {
        if let Some(v) = num(k) {
            if v != neutral {
                return Err(format!("{k} = {v}: bankML's native engine reproduces llama.cpp's penalties, top-k, top-p, min-p and temperature; this sampler is not reproduced yet"));
            }
        }
    }
    if let Some(v) = num("top_n_sigma") {
        if v > 0.0 {
            return Err("top_n_sigma: not reproduced yet".into());
        }
    }
    Ok(p)
}

/// What the receipt's `engine` says for a loaded model (and whether a GPU works in it).
pub fn engine_name(eng: &Native) -> String {
    let gpu = eng.w.gpu.as_ref().and_then(|g| g.lock().ok().map(|w| format!(", GPU {} on {:.0}% of each 1-bit matrix", w.name, w.share * 100.0))).unwrap_or_default();
    format!("bankML {} native: its own forward pass, token-identical to llama.cpp b11192 on its oracle{gpu}", crate::VERSION)
}

// ---------------------------------------------------------------- the registry and residency (0.3.1) ----------

/// What the header says about a pinned file, read once when the registry is built.
#[derive(Debug, Clone)]
pub struct Info {
    pub arch: Option<String>,
    /// the most frequent non-F32 tensor type, e.g. `Q1_0`
    pub quant: String,
    /// `general.size_label`, or the weights counted, e.g. `1.7B`
    pub params: String,
    /// `Ok` when bankML's forward pass plays this file; otherwise why not, and the milestone that would
    pub native: Result<(), String>,
    pub defaults: Option<Params>,
}

/// One pinned GGUF: a name, the file, and the FORK.json that pins it.
#[derive(Debug, Clone)]
pub struct Entry {
    /// the file's stem, lower-cased (`bonsai-1.7b-q1_0`); `:latest` and the file name are accepted as aliases
    pub name: String,
    pub file: String,
    /// where the file was found; `None` when it is pinned but not on this machine
    pub path: Option<PathBuf>,
    pub fork_json: String,
    pub sha256: String,
    pub bytes: u64,
    pub info: Info,
}

/// The models `serve --native` can be asked for: the one it started with, and with `--registry` every GGUF pinned in
/// the forks directory. The pins are read at start; a load verifies the file against them again.
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

/// Whether bankML's forward pass plays a file, from its header alone (the guard, the architecture, the weight type,
/// the chat template); the reason names what is missing and where it is on the road.
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
    /// The startup model first (it is the default), then, with `dir`, every `*.FORK.json` there: each `.gguf` it
    /// pins, looked for beside the startup model and in `dir`. A name pinned twice keeps its first entry.
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

    /// A request's `model` to a pinned entry: absent or empty is the startup model; otherwise the name, with or
    /// without `:latest`, or the file name, case-insensitively.
    pub fn resolve(&self, model: Option<&str>) -> Result<&Entry, String> {
        let m = model.map(str::trim).unwrap_or("");
        if m.is_empty() {
            return Ok(&self.entries[self.default]);
        }
        let n = model_name(m.strip_suffix(":latest").unwrap_or(m));
        // 0.3.4: a name without its weight-type suffix is accepted when exactly one pin has that base (`mindx-gen39`
        // for mindx-gen39-f16, as mindX's Ollama tag names it)
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
    /// the canonical path that was hashed, and its identity then
    pub model: PathBuf,
    pub ident: FileIdent,
    pub engine: String,
    pub hashed_at: u64,
    pub size: u64,
    pub sha256: String,
}

pub struct Resident {
    pub loaded: Loaded,
    /// `None`: resident until told otherwise
    pub expires: Option<SystemTime>,
}

/// One resident model at a time (Ollama's `MAX_LOADED_MODELS=1`). `run` is held for a whole completion, load or
/// unload, so the one engine has one user and a switch never has two models in memory; `cur` is held only briefly.
pub struct Residency {
    pub reg: Registry,
    pub n_ctx: usize,
    pub engine: crate::gguf::Engine,
    pub run: Mutex<()>,
    pub cur: Mutex<Option<Resident>>,
    /// the most recent verification (what `/bankml` reports when nothing is resident)
    pub last: Mutex<Option<Loaded>>,
    /// O5: the derived models of the registry directory (`create.rs`), layered on these pins
    pub derived: crate::create::Store,
    /// 0.3.5: what `/api/ps` names and the context it reports — Ollama's runner keeps the name of the model whose
    /// request loaded it, and reloads (under the new name) when a request's `num_ctx` differs; bankML keeps the
    /// weights and records the same (`shown_as`)
    pub shown: Mutex<Option<(String, usize)>>,
}

/// Ollama's `chatPrompt` truncation (`server/prompt.go`, v0.13.3), over any token count: the messages kept, and how
/// many were cut. `count` is the length of a candidate conversation's prompt.
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
    /// `serve --native`'s start: the startup model, already verified by `serve::run`, resident until told otherwise.
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

    /// Record the model a request named (`tag`, a pin's or a derived model's) and its context, as Ollama's runner
    /// would be named: when the weights were (re)loaded for it (`loaded`), when nothing is recorded yet, or when its
    /// context differs from the recorded one (Ollama reloads then).
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

    /// The model `e` names, loaded: the resident one if it is that model and its file is unchanged, else the resident
    /// one dropped and `e` verified (guard, then the sha256 pin) and opened. The caller holds `run`. Returns the load
    /// time in nanoseconds (0 when it was resident). Errors carry the HTTP status they answer with.
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

    /// After a request: the keep-alive starts now (`Unload` drops the model at once).
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

    /// Drop the resident model once its keep-alive has run out, unless a request is using the engine.
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

    /// Savante-style conversations, turn after turn in one engine, against llama-server b11192's own chat endpoint
    /// (testing/serve_oracle.py): the same answer text, the same prompt and completion counts, and the same number of
    /// prompt tokens taken from the cache, every turn.
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + its serve-*.jsonl (testing/serve_oracle.py); --release"]
    fn oracle_native_serve() {
        native_serve("Bonsai-8B-Q1_0");
    }

    /// 0.3.4: the same conversations on the models O4 opens: Bonsai-1.7B (tied embeddings), and the Llama graph in F16
    /// (SmolLM2-135M-Instruct and mindx-gen39), each against llama-server running that model; then each one's JSON
    /// mode, with its template's own grammar.
    #[test]
    #[ignore = "needs .models/{Bonsai-1.7B-Q1_0,SmolLM2-135M-Instruct-F16,mindx-gen39-F16}.gguf + their serve-*/json-*.jsonl; --release"]
    fn oracle_native_serve_o4() {
        for stem in ["Bonsai-1.7B-Q1_0", "SmolLM2-135M-Instruct-F16", "mindx-gen39-F16"] {
            native_serve(stem);
            json_replay(stem, "json");
        }
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

    /// llama-server b11192's answers under JSON mode and user grammars (testing/json_oracle.py), each from an empty
    /// cache: the same tokens (end token included), raw text, message content, finish reason and counts; and the
    /// grammar and generation prompt the server reports are the ones bankML uses.
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
            // O6b: a schema record carries the request as it was sent, read exactly (a float literal stays one)
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

    /// O6b: llama-server b11192's answers under JSON schemas (testing/json_schema_oracle.py --record): objects with
    /// required and optional fields, enums, ranges, nested `$defs`, a pattern, a top-level array and string, through
    /// `response_format: json_schema`, the top-level `json_schema` and `json_object` with a schema
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

    /// O2: llama-server b11192's penalties sampler, token for token (testing/penalty_oracle.py): repeat, frequency and
    /// presence penalties over `repeat_last_n`, greedy and seeded, the prompt in the window. Each request resolves to
    /// the parameters the server read back, its answer is the server's tokens, and what the server refused is refused
    /// with its message.
    fn penalty_replay(stem: &str) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let eng = Native::open(&dir.join(format!("{stem}.gguf")), 2048).unwrap();
        let rec = std::fs::read_to_string(dir.join(format!("oracle-forward/penalty-{stem}.jsonl"))).unwrap();
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
            let got_p = (p.temp, p.top_k, p.top_p, p.min_p, p.penalty_last_n, p.penalty_repeat, p.penalty_freq, p.penalty_present, p.seed);
            let want_p = (num(g, "temperature") as f32, num(g, "top_k") as i32, num(g, "top_p") as f32, num(g, "min_p") as f32, num(g, "repeat_last_n") as i32,
                          num(g, "repeat_penalty") as f32, num(g, "frequency_penalty") as f32, num(g, "presence_penalty") as f32, num(g, "seed") as u32);
            assert_eq!(got_p, want_p, "case {n}: the request resolves to the server's parameters");
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
        eprintln!("penalty oracle ({stem}): {same} of {} answers token-identical to llama-server b11192 ({toks} tokens), {refused} refusals with its message",
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

    /// 0.3.5: the same on the O4 models — Bonsai-1.7B (the Qwen3 template) and the two ChatML templates, whose schema
    /// grammar has no reasoning block (`schema::chat_grammar`)
    #[test]
    #[ignore = "needs .models/{Bonsai-1.7B-Q1_0,SmolLM2-135M-Instruct-F16,mindx-gen39-F16}.gguf + oracle-json/schema-*.jsonl (testing/json_schema_oracle.py --record); --release"]
    fn oracle_json_schema_o4() {
        for stem in ["Bonsai-1.7B-Q1_0", "SmolLM2-135M-Instruct-F16", "mindx-gen39-F16"] {
            json_replay(stem, "schema");
        }
    }

}

