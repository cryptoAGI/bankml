// SPDX-License-Identifier: MIT OR Apache-2.0
//! # bankML
//!
//! A zero-dependency Rust runtime for GGUF models in `Q1_0` (1-bit), `Q2_0_g64` (ternary) and F16. Kernels, forward
//! pass, tokenizer, templates, samplers and grammars are checked against llama.cpp b11192: bit-exact where ggml
//! computes, token-identical where llama-server answers.
//!
//! Every model passes [`verify`] before it answers: the GGUF header guard ([`gguf`]), then the sha256 pin against
//! the model's `FORK.json` ([`pin`]).
//!
//! ## Modules
//!
//! | module | role |
//! |---|---|
//! | [`gguf`], [`sha256`] | GGUF header parser and guard, `Mmap`; sha256 and the pin |
//! | [`q1_0`], [`q2_0`], [`f16`](mod@f16) | ggml weight formats and their products; runtime SIMD dispatch over a scalar reference |
//! | [`par`], [`sys`] | thread pool (results independent of thread count); machine facts |
//! | [`gpu`] | Vulkan through dlopen with bankML's own SPIR-V |
//! | [`forward`] | Qwen3 and Llama graphs: weights, KV cache, attention, prefill, decode |
//! | [`tokenizer`], [`chat`], [`unicode_letters`] | llama.cpp's tokenizer and pre-tokenizers; chat templates by sha |
//! | [`sampler`], [`grammar`], [`schema`] | llama-server's sampler chain; GBNF; JSON schemas |
//! | [`native`], [`prompt_cache`], [`metrics`] | the engine, its slot and the model registry; host prompt cache; per-completion measurements |
//! | [`serve`], [`ollama`] | the OpenAI-compatible gateway; Ollama's API |
//! | [`create`], [`convert`] | derived models as verified layers; safetensors → GGUF, byte-identical to llama.cpp |
//! | [`train`] | mindXtrain's author and score stages |
//!
//! Details: docs/modules/bankml.md; one page per module in docs/modules/.

#![allow(dead_code)]

pub mod chat;
pub mod f16;
pub mod convert;
pub mod create;
pub mod forward;
pub mod gguf;
pub mod grammar;
pub mod gpu;
pub mod native;
pub mod ollama;
pub mod par;
pub mod prompt_cache;
pub mod serve;
pub mod q1_0;
pub mod q2_0;
pub mod sampler;
pub mod schema;
pub mod sha256;
pub mod sincosf;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm_simd;
pub mod sys;
pub mod metrics;
pub mod train;
pub mod tokenizer;
pub mod unicode_letters;

use std::path::Path;

/// ggml type ids bankML plays; values match b11192 `ggml.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    /// 1 bit per weight, group 128, f16 scale → 1.125 bpw.
    Q1_0 = 41,
    /// 2-bit codes → {−1, 0, +1, +2}·d (ternary models use three), group 64, f16 scale → 2.25 bpw.
    Q2_0 = 42,
}

/// Guard verdict (the verdicts of minaiml `gguf_guard.py`).
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Play,
    Refuse(Vec<String>),
    NeedMore(u64),
}

/// Design-time receipt shape, reserved for signed receipts. Served answers carry `serve.rs`'s `bankml_receipt`.
#[derive(Debug, Default)]
pub struct Receipt {
    pub model_sha256: String,
    pub guard: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub ttft_ms: u64,
    pub wall_ms: u64,
    /// THOT8 ternary commitment of the output (Keccak-256 leaf, THOTLib.sol-compatible), if requested.
    pub thot8_leaf: Option<[u8; 32]>,
}

/// Header-only GGUF check on the mainline engine; tensor data is never read.
pub fn guard(gguf: &Path) -> Verdict {
    match gguf::guard_file(gguf, gguf::Engine::Mainline) {
        Ok(r) => r.verdict,
        Err(e) => Verdict::Refuse(vec![format!("cannot read {}: {e}", gguf.display())]),
    }
}

/// Returns `Ok(sha)` only when the file's sha256 equals its `FORK.json` record.
pub fn pin(gguf: &Path, fork_json: &str) -> Result<String, String> {
    let name = gguf.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let want = sha256::pinned_sha256(fork_json, &name).ok_or(format!("{name} has no sha256 record in FORK.json: unpinned, refused"))?;
    let got = sha256::file_hex(gguf).map_err(|e| format!("cannot hash {name}: {e}"))?;
    if got == want { Ok(got) } else { Err(format!("{name} sha256 {got} != pinned {want}: refused")) }
}

/// The crate version from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Log levels of [`log`] (the C API's `BANKML_LOG_ERROR` … `BANKML_LOG_DEBUG`).
pub const LOG_ERROR: i32 = 0;
pub const LOG_WARN: i32 = 1;
pub const LOG_INFO: i32 = 2;
pub const LOG_DEBUG: i32 = 3;

/// A log sink: the level and the message.
pub type LogSink = fn(i32, &str);

static LOG_SINK: std::sync::RwLock<Option<LogSink>> = std::sync::RwLock::new(None);

/// Installs the library's log sink (the C API's `bankml_set_log`); `None` restores standard error.
pub fn set_log_sink(sink: Option<LogSink>) {
    *LOG_SINK.write().unwrap_or_else(|e| e.into_inner()) = sink;
}

/// The newest log messages (`LOG_KEEP`), with the time each was logged: what `GET /bankml/status` and serve's
/// status page show as the engine's log.
pub const LOG_KEEP: usize = 200;
static LOG_RING: std::sync::Mutex<std::collections::VecDeque<(f64, i32, String)>> = std::sync::Mutex::new(std::collections::VecDeque::new());

/// The kept log messages, oldest first: (seconds since the epoch, level, message).
pub fn log_tail() -> Vec<(f64, i32, String)> {
    LOG_RING.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect()
}

/// Sends one message at `level` to the sink, or one line to standard error, and keeps it in the log ring (the
/// newest `LOG_KEEP`).
pub fn log(level: i32, msg: &str) {
    {
        let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let mut ring = LOG_RING.lock().unwrap_or_else(|e| e.into_inner());
        if ring.len() == LOG_KEEP {
            ring.pop_front();
        }
        ring.push_back((at, level, msg.to_string()));
    }
    let sink = *LOG_SINK.read().unwrap_or_else(|e| e.into_inner());
    match sink {
        Some(f) => f(level, msg),
        None => eprintln!("{msg}"),
    }
}

/// Proof a model file passed the gate: the guard played and the pin matched.
#[derive(Debug, PartialEq, Eq)]
pub struct Verified {
    pub model_sha256: String,
    pub guard: &'static str,
    pub engine: &'static str,
    /// From the header: `general.architecture`, `general.name`, tensor types by count.
    pub arch: Option<String>,
    pub name: Option<String>,
    pub types: Vec<(String, usize)>,
}

impl Verified {
    pub fn to_json(&self) -> String {
        let opt = |s: &Option<String>| s.as_ref().map(|s| gguf::jstr(s)).unwrap_or_else(|| "null".into());
        format!(
            "{{\"verdict\": \"play\", \"guard\": \"{}\", \"engine\": \"{}\", \"model_sha256\": \"{}\", \"bankml\": \"{VERSION}\", \"arch\": {}, \"name\": {}, \"types\": {{{}}}}}",
            self.guard, self.engine, self.model_sha256, opt(&self.arch), opt(&self.name),
            self.types.iter().map(|(n, c)| format!("{}: {c}", gguf::jstr(n))).collect::<Vec<_>>().join(", ")
        )
    }
}

/// The gate: the header guard (reads the header only), then the sha256 pin (reads the whole file).
///
/// Either refusal returns `Err` with the reason.
pub fn verify(gguf: &Path, fork_json: &str, engine: gguf::Engine) -> Result<Verified, String> {
    let r = gguf::guard_file(gguf, engine).map_err(|e| format!("cannot read {}: {e}", gguf.display()))?;
    match r.verdict {
        Verdict::Play => {}
        Verdict::Refuse(why) => return Err(format!("guard refused: {}", why.join("; "))),
        Verdict::NeedMore(n) => return Err(format!("guard needs {n} header bytes: file truncated")),
    }
    let model_sha256 = pin(gguf, fork_json)?;
    Ok(Verified { model_sha256, guard: "play", engine: engine.as_str(), arch: r.arch, name: r.name, types: r.types })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_ids_match_mainline() {
        assert_eq!(GgmlType::Q1_0 as u32, 41);
        assert_eq!(GgmlType::Q2_0 as u32, 42);
    }

    #[test]
    fn verified_json_escapes_what_the_header_says() {
        // the name and arch come from the model file: a hostile header must not break /bankml's JSON
        let v = Verified {
            model_sha256: "ab".repeat(32),
            guard: "play",
            engine: "mainline",
            arch: Some("qwen3".into()),
            name: Some("a\"b\\c\nd\u{1}".into()),
            types: vec![("Q4_K".into(), 155), ("Q6_K".into(), 15)],
        };
        let j = v.to_json();
        assert!(j.contains(r#""name": "a\"b\\c\nd\u0001""#), "{j}");
        assert!(j.contains(r#""arch": "qwen3""#) && j.contains(r#""types": {"Q4_K": 155, "Q6_K": 15}"#), "{j}");
        let none = Verified { arch: None, name: None, types: vec![], ..v };
        assert!(none.to_json().contains(r#""arch": null, "name": null, "types": {}"#));
        // balanced: every quote outside an escape opens or closes a string
        let mut n = 0;
        let mut esc = false;
        for c in j.chars() {
            if esc { esc = false } else if c == '\\' { esc = true } else if c == '"' { n += 1 }
        }
        assert_eq!(n % 2, 0);
        assert!(j.starts_with('{') && j.ends_with('}'));
    }

    #[test]
    fn verify_runs_guard_then_pin() {
        let dir = std::env::temp_dir().join(format!("bankml-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // a minimal header the guard plays: no tensors, one key
        let mut g = b"GGUF".to_vec();
        g.extend(3u32.to_le_bytes());
        g.extend(0u64.to_le_bytes());
        g.extend(1u64.to_le_bytes());
        for part in [&(20u64.to_le_bytes())[..], b"general.architecture", &8u32.to_le_bytes(), &5u64.to_le_bytes(), b"qwen3"] {
            g.extend_from_slice(part);
        }
        let f = dir.join("m.gguf");
        std::fs::write(&f, &g).unwrap();
        let sha = sha256::file_hex(&f).unwrap();
        let fork = format!("{{\"files\": [{{\"path\": \"m.gguf\", \"sha256\": \"{sha}\"}}]}}");
        let v = verify(&f, &fork, gguf::Engine::Mainline).unwrap();
        assert_eq!(v.model_sha256, sha);
        assert!(v.to_json().contains(&format!("\"model_sha256\": \"{sha}\"")));
        let wrong = fork.replace(&sha[..8], "00000000");
        assert!(verify(&f, &wrong, gguf::Engine::Mainline).unwrap_err().contains("!= pinned"));
        std::fs::write(&f, b"GGUX").unwrap();
        assert!(verify(&f, &fork, gguf::Engine::Mainline).unwrap_err().starts_with("guard refused"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn bpw_from_block_layout() {
        // Q1_0: 128 weights → 16 bytes of signs + 2 bytes f16 scale
        assert_eq!((16 + 2) as f64 * 8.0 / 128.0, 1.125);
        // Q2_0 g64: 64 trits × 2 bits → 16 bytes + 2 bytes scale
        assert_eq!((16 + 2) as f64 * 8.0 / 64.0, 2.25);
    }
}
