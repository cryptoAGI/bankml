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

use crate::chat::{self, Message};
use crate::forward::{KvCache, Weights};
use crate::sampler::{Params, Sampler};
use crate::serve::Json;
use crate::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

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
    eog: Vec<u32>,
    slot: Mutex<Slot>,
}

/// How a completion ended, and its counts (llama-server's `usage` and `timings.cache_n`).
#[derive(Debug, Clone)]
pub struct Done {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: &'static str,
    pub text: String,
}

impl Native {
    pub fn open(model: &Path, n_ctx: usize) -> Result<Native, String> {
        chat::check_template(model)?;
        let tok = Tokenizer::from_gguf(model)?;
        let w = Weights::open(model)?;
        let defaults = Params::from_gguf(model)?;
        let eog = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.id(t)).collect();
        let caches = w.caches();
        Ok(Native { w, tok, defaults, n_ctx, model: model.to_path_buf(), eog, slot: Mutex::new(Slot { tokens: Vec::new(), caches }) })
    }

    /// The request's sampling parameters over the model's defaults; an unsupported sampler is refused.
    pub fn params(&self, req: &Json) -> Result<Params, String> {
        let num = |k: &str| match req.get(k) {
            Some(Json::Num(n)) => Some(*n),
            _ => None,
        };
        let mut p = self.defaults.clone();
        if let Some(v) = num("temperature") { p.temp = v as f32 }
        if let Some(v) = num("top_k") { p.top_k = v as i32 }
        if let Some(v) = num("top_p") { p.top_p = v as f32 }
        if let Some(v) = num("min_p") { p.min_p = v as f32 }
        if let Some(v) = num("min_keep") { p.min_keep = v as usize }
        if let Some(v) = num("seed") { p.seed = v as i64 as u32 }
        for (k, neutral) in [("repeat_penalty", 1.0), ("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("typical_p", 1.0),
                             ("xtc_probability", 0.0), ("dry_multiplier", 0.0), ("dynatemp_range", 0.0)] {
            if let Some(v) = num(k) {
                if v != neutral {
                    return Err(format!("{k} = {v}: bankML's native engine reproduces llama.cpp's top-k, top-p, min-p and temperature; this sampler is not reproduced yet"));
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

    /// The prompt a conversation becomes, as token ids (llama-server: the template, then tokenization with specials).
    pub fn prompt(&self, messages: &Json) -> Result<Vec<u32>, String> {
        let msgs: Vec<Message> = chat::messages_from_json(messages)?;
        Ok(self.tok.encode(&chat::render(&msgs)?, true))
    }

    /// One completion. `emit` receives the answer as whole UTF-8 pieces as they come, and returns false to stop.
    pub fn complete(&self, prompt: &[u32], params: Params, max_tokens: Option<usize>, mut emit: impl FnMut(&str) -> bool) -> Result<Done, String> {
        if prompt.is_empty() {
            return Err("an empty prompt".into());
        }
        if prompt.len() >= self.n_ctx {
            return Err(format!("the prompt has {} tokens; the context is {}", prompt.len(), self.n_ctx));
        }
        let mut sampler = Sampler::new(params)?;
        let mut slot = self.slot.lock().map_err(|_| "the slot is poisoned")?;
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
        let (mut pending, mut text, mut n, mut finish) = (Vec::<u8>::new(), String::new(), 0usize, "length");
        loop {
            if max_tokens.is_some_and(|m| n >= m) {
                break;
            }
            let next = sampler.sample(&self.w.logits(&rn)?);
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
        Ok(Done { prompt_tokens: prompt.len(), cached_tokens: n_past, completion_tokens: n, finish_reason: finish, text })
    }

    /// `/props`, in the shape Savante reads (the context and the model's path).
    pub fn props_json(&self) -> String {
        format!("{{\"model_path\": {}, \"n_ctx\": {}, \"default_generation_settings\": {{\"n_ctx\": {}, \"params\": {{\"temperature\": {}, \"top_k\": {}, \"top_p\": {}, \"min_p\": {}}}}}, \"build_info\": \"bankML {} native\"}}",
                crate::gguf::jstr(&self.model.to_string_lossy()), self.n_ctx, self.n_ctx, self.defaults.temp, self.defaults.top_k, self.defaults.top_p, self.defaults.min_p, crate::VERSION)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Savante-style conversations, turn after turn in one engine, against llama-server b11192's own chat endpoint
    /// (testing/serve_oracle.py): the same answer text, the same prompt and completion counts, and the same number of
    /// prompt tokens taken from the cache, every turn.
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + its serve-*.jsonl (testing/serve_oracle.py); --release"]
    fn oracle_native_serve() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let eng = Native::open(&dir.join("Bonsai-8B-Q1_0.gguf"), 2048).unwrap();
        let turns = std::fs::read_to_string(dir.join("oracle-forward/serve-Bonsai-8B-Q1_0.jsonl")).unwrap();
        let num = |v: &Json, k: &str| match v.get(k) { Some(Json::Num(n)) => *n as usize, _ => usize::MAX };
        let (mut n, mut ok) = (0, 0);
        let t0 = std::time::Instant::now();
        for line in turns.lines() {
            let t = Json::parse(line).unwrap();
            let req = t.get("request").unwrap();
            let prompt = eng.prompt(req.get("messages").unwrap()).unwrap();
            let params = eng.params(req).unwrap();
            let max = req.get("max_tokens").and_then(|v| match v { Json::Num(x) => Some(*x as usize), _ => None });
            let d = eng.complete(&prompt, params, max, |_| true).unwrap();
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
        eprintln!("native serve oracle: {ok} of {n} conversation turns identical to llama-server b11192's /v1/chat/completions (text, token counts, prompt-cache reuse) — {:.0} s",
                  t0.elapsed().as_secs_f64());
        assert_eq!(ok, n);
    }
}

