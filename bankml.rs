//! # bankml.rs — the in-house Rust player for low-bit models
//!
//! **Status (0.0.4, 2026-09-28): P1 guard + pin (one `verify` gate) and the P2 Q1_0 and Q2_0 (ternary) kernels are native and
//! proven (see the checklist). Nothing here runs a model yet** — `answer()`'s `todo!()` is P0/P3. The
//! ternary finding: llama.cpp b11192 has **no x86 Q2_0 kernel** (scalar C, ~49 ns per 64 weights on the
//! dev box); bankml's is bit-exact with it and ~9–10× faster, and ggml's matmuls are ~90–98 % of its wall. Created 2026-09-26 on the operator's instruction: "create llama.cpp rust version
//! todo as bankml.rs", design goals **optimization, succinct, verified response**.
//!
//! Doctrine (`/core`): Python prototypes, Rust ports the *working* architecture. The working
//! architecture exists as of today: PrismML Bonsai-8B (Qwen3-8B dense, ggml `Q1_0`, 1.16 GB,
//! PYTHAI/Bonsai-8B-gguf-fork) served by **llama.cpp b11192** on the 2-core mindX node,
//! measured by `scripts/ternary_diagnostics.py` (data/monitoring/ternary_diagnostics.jsonl):
//!
//! ```text
//!   5/5 verified · 148 prompt → 130 completion tokens in 58.7 s · gen 2.73–3.36 tok/s (median 3.0)
//!   prompt eval 1.1–4.4 tok/s · first token 0.34–8.8 s · peak RSS 1,786 MB · 2 threads, CPUQuota 150 %
//! ```
//!
//! One core (mindx-bonsai.service, 1 thread, CPUQuota 100 %), the same five prompts, 2026-09-26:
//!
//! ```text
//!   5/5 verified · 148 → 138 tokens in 105.8 s · gen median 2.62 tok/s · first token 8.9–12.3 s
//!   97.3 CPU-seconds total = 0.705 CPU-s per completion token · peak RSS 1,785 MB
//! ```
//!
//! That row is the bar. bankml.rs is not done until it matches it on the same prompts, then beats it.
//! The engineer is the `bankml` subagent (`.claude/agents/bankml.md`): prefill first (the measured wall),
//! then memory traffic per token, with strategies from llama.cpp, vLLM, Ollama and KoboldCpp.
//!
//! ## The three design goals, made testable
//!
//! 1. **Optimization** — for a 1-bit weight the matmul is a signed sum: `y = s · (Σ x[w=+1] − Σ x[w=−1])`
//!    per 128-weight group. No multiplies on the weights (ggml-exactness pins the activations to q8_0,
//!    so the sums run as `maddubs`/`madd` by ±1 or {0,1}). Targets on the reference node:
//!    gen ≥ 3.0 tok/s (parity), then ≥ 4.5 tok/s; prompt eval ≥ 2× b11192 (batched GEMM, not GEMV);
//!    RSS ≤ weights + KV(ctx, kv_type) + 150 MB. Measured by ternary_diagnostics, never quoted.
//! 2. **Succinct** — one crate, no Python, no runtime downloads. Budget: ≤ 3,000 lines for
//!    GGUF + Qwen3 forward + Q1_0/Q2_0_g64 kernels + sampler + OpenAI-compatible server. Every
//!    dependency justified in `Cargo.toml` comments. A single static binary.
//! 3. **Verified response** — every answer carries its evidence, and a model that cannot be
//!    verified does not answer:
//!    - before load: the GGUF guard (port of minaiml `gguf_guard.py`) → `play | refuse | need_more`;
//!      refuse is shown with its reason, never a silent fallback (Bonsai 2 Q2_0 loads on mainline
//!      and answers in gibberish — the worst failure);
//!    - the file's sha256 must match a pinned record (`FORK.json` of the PYTHAI fork);
//!    - kernels are proven **bit-exact against ggml** on recorded tensors before they are trusted;
//!    - each response returns a `Receipt`: model sha256, guard verdict, prompt/completion tokens,
//!      timings, and an optional THOT8 ternary commitment of the output (see `thot8_cpu.py` —
//!      same codec, same Keccak leaf as `THOTLib.sol`), so a response can be checked later by
//!      anyone holding the model file.
//!
//! ## Phases (each ends in a measured row, or it is not done)
//!
//! - [ ] **P0 — wrap.** FFI to ggml/llama.cpp b11192 (`llama-cpp-2` or raw bindgen), OpenAI-compatible
//!   `/v1/chat/completions` with `usage` + `timings`, Receipt, guard in front. Parity row = the bar above.
//! - [ ] **P1 — guard + receipts native.** (guard and pin proven; receipts not yet emitted)
//!   - [x] GGUF v3 header parse + the three traps + kv_f16_bytes_per_token (`gguf.rs`). Evidence: the 9
//!     cases of `test_gguf_guard.py` as Rust tests + fail-closed extras; `testing/guard_agree.py` = **20/20**
//!     JSON-identical with the Python guard (real `Bonsai-1.7B-Q1_0.gguf` sha256 `3d7c6c90…` + all synthetic
//!     cases, both engines): play · qwen3 · F32 113 / Q1_0 197 · 114,688 kv bytes/token.
//!   - [x] sha256 (FIPS vectors; equals `sha256sum` and the HF LFS oid on the real file) + FORK.json pin
//!     (`bankml pin FILE --fork FORK.json`; parses the real PYTHAI/Bonsai-8B-gguf-fork FORK.json → `284a335a…`,
//!     refuses a mislabelled file naming both hashes).
//!   - [x] 0.0.2: `verify` = guard then pin as one gate (`bankml verify FILE --fork FORK.json --json`);
//!     on the real Ternary-Bonsai-8B + its FORK.json → play, `e17b298d…`. Guard hardened against hostile
//!     headers (nested arrays refuse instead of overflowing the stack; KV-size overflow refuses).
//!   - [ ] Receipt emitted per answer, THOT8 leaf (needs P0/P3 to have an answer to sign).
//! - [ ] **P2 — own the kernels.** (Q1_0 and Q2_0 proven on x86 AVX2; NEON, real activations open)
//!   - [x] Q1_0 exactly as ggml b11192 defines it (`q1_0.rs`, read from source): f16 `d` first, 16 sign
//!     bytes LSB-first; the matmul is `ggml_vec_dot_q1_0_q8_0` against **q8_0 activations** (AVX2
//!     quantizer: 127/amax, round-half-even). Evidence (`oracle_ggml_b11192_real_bonsai_1_7b`, against the
//!     exported symbols of the sha256-checked b11192 ubuntu-x64 release, `libggml-cpu-haswell.so`): all
//!     197 Q1_0 tensors = **1,719,904,256 weights dequantized bit-exact**; **788/788** q8_0 rows byte-exact;
//!     **788/788** vec_dot results bit-exact vs ggml AVX2, and 788/788 vs ggml generic (whose shipped
//!     build fuses to FMA — read from the disassembly). AVX2 == scalar model on 3,500 + 2,000 randomized cases.
//!   - [x] Measured (dev box Ryzen 3 3200U, Zen+, 1 thread, in-process A/B vs ggml's own kernel, same bits):
//!     decode GEMV 12288×4096 **parity** (min 10.0 vs 10.0–10.2 ns/block; `vec_dot_act`, activation scales
//!     converted once per token); prefill 1×4 tile **1.10×** ggml's per-pair vec_dot (min 9.1 vs 10.1
//!     ns/(block·col)). Not the node: a Zen3 row is the deciding measurement.
//!   - [x] Q2_0 (id 42, group 64) exactly as ggml b11192 defines it (`q2_0.rs`, read from the tag's source):
//!     f16 `d` first, 16 bytes of 2-bit codes LSB-first, code c → (c − 1)·d ∈ {−1, 0, +1, **+2**} (the
//!     Bonsai file never uses +2: 31 % / 38 % / 31 % / 0 %; the kernels take it anyway). `vec_dot_type` q8_0.
//!     **On x86, b11192 has no Q2_0 kernel**: `arch-fallback.h` renames the generic C to
//!     `ggml_vec_dot_q2_0_q8_0`, no repack, no sgemm case; the haswell symbol is scalar (64 `imul`/block,
//!     read from the disassembly), used per (row, column) for decode *and* prefill. Evidence
//!     (`oracle_ggml_b11192_real_ternary_bonsai_8b`, real Ternary-Bonsai-8B sha256 `e17b298d…`, via mmap):
//!     all 254 Q2_0 tensors = **8,188,239,872 weights dequantized bit-exact**; **762/762** q8_0 rows
//!     byte-exact; **762/762** vec_dot bit-exact vs ggml haswell (scalar model, AVX2, portable path), and
//!     762/762 of a no-FMA model vs the baseline `libggml-cpu-x64.so` (which differs from haswell in 19/762:
//!     the float order is really being tested). 3,300 randomized cases incl. code 3, q = −128, tail blocks.
//!   - [x] Measured (dev box, 1 thread, in-process A/B vs ggml's own Q2_0 kernel, real layer-0 weights,
//!     alternating, 4 runs; the laptop's boost clock moves both, so ratios are the result): decode GEMV
//!     **9.1–10.6×** (ggml 48.6–55 vs bankml 4.9–6.0 ns/block min; `output` 151669×4096: 492 → 51 ms);
//!     prefill 1×4 tile **12.3–13.1×** (49 vs 3.96 ns/(block·col)). The per-token activation layout costs
//!     11 µs (n = 4096) / 29 µs (n = 12288), ≈ 2 ms per token: outside the matmul timings, 0.5 % of them. The kernel: two blocks per 256-bit
//!     register, codes by `(v >> 2r) & 3`, `maddubs(c, q) − Σq`, the activation (not the weights) permuted
//!     once per token, ggml's float chain kept serial. One-token budget (`decode_budget_q2_0`, all 253
//!     matmuls, 2.13 GB, each tensor once in file order, llama-bench run before and after at the same clock):
//!     3 threads — llama-bench tg **2.59–2.68 s/token**, ggml's matmuls **2.54–2.64 s** (≈ 98 % of the wall),
//!     bankml **0.36–0.37 s** (7.0×; matmul-only ceiling 2.77 tok/s); 1 thread — llama-bench 4.50–5.21 s,
//!     ggml matmuls 4.22–4.42 s, bankml **0.49 s** (8.7×; ceiling 2.06 tok/s). Not the node: a Zen3 row decides.
//!   - [x] 0.0.3 threads: `par::Pool` (persistent, zero-dependency) + `mat_vec_par`/`mat_mul_par`, bits
//!     independent of thread count. One ternary token's 253 matmuls: **0.23–0.25 s at 3 threads** (ggml
//!     2.27–2.36 s), below ggml's *1-bit* 0.34 s; 1-bit at parity. New 8B Q1_0 oracle: 254 tensors bit-exact.
//!   - [x] 0.0.4: memory floor 15–17 GB/s (both kernels compute-bound: ternary ~2×, 1-bit ~5.5× above
//!     it); `mat_mul_act` 1-bit prefill 1.27–1.33× ggml; two bit-exact decode variants measured and rejected.
//!   - [ ] NEON; AVX-512 (the node has none); bit-exact on *dumped real activations* (today: real weights ×
//!     synthetic activations); a Q1_0 GEMM that beats 1.1× for prompt eval (the Q2_0 layout above runs at
//!     5.0 ns per 64 weights vs Q1_0's 7.2 — porting it to Q1_0 is the next experiment).
//! - [ ] **P3 — own the forward.** Qwen3 dense block (RMSNorm, RoPE, GQA, SwiGLU) with a quantized
//!   KV cache (q8_0 first — at 1.125 bpw the KV cache, not the weights, sets the RAM ceiling).
//!   Drop the FFI. Parity with P0 on all 5 prompts at temperature 0 (token-identical is the goal).
//! - [ ] **P4 — mindX seam.** Serve on loopback; register as an OpenAI-compatible provider in mindX
//!   (the vLLM handler shape); `inference_budget.record_inference(..., model=, caller=)` sees real
//!   usage; `/substrate/` and feedback show its row. Then the 8B migration can route to it.
//! - [ ] **P5 — handheld.** Same crate → Android (NDK) / iOS as the minaiml BROBOT engine alternative.
//!
//! Out of scope until measured need: GPU backends, PrismML fork types (PQ2_0, PTQ1_0), Bonsai 2's
//! Hadamard basis (watch ggml-org/llama.cpp#27779), training.

#![allow(dead_code)]

pub mod gguf;
pub mod par;
pub mod q1_0;
pub mod q2_0;
pub mod sha256;

use std::path::Path;

/// ggml type ids bankml.rs plays (mainline ids, read 2026-09-24; Q2_0 re-read from b11192 `ggml.h` 2026-09-26).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    /// 1 bit per weight, group 128, f16 scale → 1.125 bpw.
    Q1_0 = 41,
    /// 2-bit codes → {−1, 0, +1, +2}·d (ternary models use three), group 64, f16 scale → 2.25 bpw.
    Q2_0 = 42,
}

/// Guard verdict, as in minaiml `gguf_guard.py`.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Play,
    Refuse(Vec<String>),
    NeedMore(u64),
}

/// What every response carries (goal 3).
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

/// P1: header-only GGUF check (reads a header prefix; tensor data is never read).
pub fn guard(gguf: &Path) -> Verdict {
    match gguf::guard_file(gguf, gguf::Engine::Mainline) {
        Ok(r) => r.verdict,
        Err(e) => Verdict::Refuse(vec![format!("cannot read {}: {e}", gguf.display())]),
    }
}

/// P1: the pin. `Ok(sha)` only when the file's sha256 equals its `FORK.json` record.
pub fn pin(gguf: &Path, fork_json: &str) -> Result<String, String> {
    let name = gguf.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let want = sha256::pinned_sha256(fork_json, &name).ok_or(format!("{name} has no sha256 record in FORK.json: unpinned, refused"))?;
    let got = sha256::file_hex(gguf).map_err(|e| format!("cannot hash {name}: {e}"))?;
    if got == want { Ok(got) } else { Err(format!("{name} sha256 {got} != pinned {want}: refused")) }
}

/// The crate version (`Cargo.toml`), printed by `bankml version` and carried by every verification.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// What a model file earned before it may answer: the guard said play and the pin matched.
#[derive(Debug, PartialEq, Eq)]
pub struct Verified {
    pub model_sha256: String,
    pub guard: &'static str,
    pub engine: &'static str,
}

impl Verified {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"verdict\": \"play\", \"guard\": \"{}\", \"engine\": \"{}\", \"model_sha256\": \"{}\", \"bankml\": \"{VERSION}\"}}",
            self.guard, self.engine, self.model_sha256
        )
    }
}

/// P1 as one gate — the check P0 puts in front of every answer: the header guard first (cheap, reads
/// only the header), then the sha256 pin (reads the whole file). Either refusal stops it, reason given.
pub fn verify(gguf: &Path, fork_json: &str, engine: gguf::Engine) -> Result<Verified, String> {
    let r = gguf::guard_file(gguf, engine).map_err(|e| format!("cannot read {}: {e}", gguf.display()))?;
    match r.verdict {
        Verdict::Play => {}
        Verdict::Refuse(why) => return Err(format!("guard refused: {}", why.join("; "))),
        Verdict::NeedMore(n) => return Err(format!("guard needs {n} header bytes: file truncated")),
    }
    let model_sha256 = pin(gguf, fork_json)?;
    Ok(Verified { model_sha256, guard: "play", engine: engine.as_str() })
}

/// P0→P3: load a guarded, pinned model and answer one chat turn with its receipt.
pub fn answer(_model: &Path, _prompt: &str, _max_tokens: u32) -> Result<(String, Receipt), String> {
    todo!("P0 via ggml FFI; P3 native Qwen3 forward")
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
