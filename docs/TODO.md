# TODO

What comes next, and where each item came from. Every item ends in a measured, reproducible result, or it is recorded
as rejected with its numbers (the house rule: the same bits first, then the speed). Sources:
**R** = [research.md](research.md) (the field, 2026-09-29) · **K** = the clean-room study of KoboldCpp (docs only) ·
**V** = the vLLM code review (main @ `36768d1`, Apache-2.0) · **Rs** = the Rust 1.95 study (the `rust` skill) ·
**A** = the audits of 0.1.5 and 0.1.6.

## 0.1.8 — speed and efficiency (in progress)

- [x] **SHA-NI for the model pin** (Rs). 5.5× (0.23 s vs 1.25 s per 248 MB); the 8B model in 2.9 s; bit-exact
  against the portable rounds and `sha256sum`.
- [x] **A history window that keeps the prompt cache warm** (K: Smart Context's reserve; V: chained block hashes show
  why a one-exchange slide defeats reuse). The window moves 8 times in 60 turns instead of 48; 39 of 47 turns reuse
  the previous prompt. The chat's footer (clock, receipt) is no longer sent to the model.
- [x] **Resources sliders** (operator): CPU threads and a RAM budget that becomes the engine's context, applied by a
  verified restart; usage read by bankml's own `sys.rs` (psutil's job, no crates) at `GET /bankml/usage`.
- [x] **Licensing layers** (operator): `MIT OR Apache-2.0`; `crypto/` GPL-3.0-only; `agpl/` walled off; SPDX gate.
- [x] **Speculative decoding verdict** (R, K, V). Draft model (Bonsai-1.7B): token-identical, no gain beyond noise.
  n-gram, unpinned and interleaved: paired +0.09, +0.11, −0.06 tok/s (medians 0.61 vs 0.59): within noise, not
  adopted. On fixed resources (`testing/pinned.sh`: cores 2–3, 2 threads, 2.5 GB): +0.01, +0.36, +0.16 while the
  baseline swung 0.36–1.40 tok/s with the machine's load (load average up to 3.8). Ahead in 5 of 6 pairs, exact,
  no memory cost → **opt-in** (`--spec-ngram`, a Resources checkbox), not the default. Re-measure on an idle machine
  (TODO below: the production server).
- [x] **Fixed resources for every benchmark** (operator): `testing/pinned.sh` pins cores, caps memory in a user
  cgroup, and records the load before and after.
- [x] **A refusal no longer resets the connection** (found by the CLI suite under load): serve read and discarded
  nothing before closing, so an unread body made the kernel send RST and the client could lose the 403/415. It now
  drains the declared body first; the suite passes 10/10 in a row.
- [ ] **Measure the prefill/decode knobs** (K, V): `-t 2/3/4` with `-tb 4`; `-ub 128/256/512`; `-fa on`;
  `-ctv q8_0` (then `-ctk q8_0`) with RSS and exactness recorded. Adopt only what is faster *and* token-identical,
  or clearly label what is not exact.
- [ ] **Is batched verification cheap here?** (V) `llama-bench -p 1,2,4,8,16`: speculation can only pay if a
  k-token verify costs much less than k single steps on this CPU.

## Rejected, with reasons (keep)

- **`--cache-reuse` / KV shifting** (K recommended it; V showed why not): re-rotating cached keys and keeping KV that
  depended on dropped tokens is **not token-identical**. bankml's window policy gets most of the benefit exactly.
- **vLLM's CPU backend as an engine** (V): no GGUF on x86, fast paths need AVX-512/AMX, ~2.9× slower than llama.cpp
  on CPU (arXiv:2608.23841).
- **KoboldCpp code** (K): AGPL-3.0; ideas only, from its documentation.
- **fp8 KV cache** (V): needs AVX-512/AMX on vLLM's CPU backend; q8_0 via llama.cpp is the equivalent to measure.

## 0.1.9 — done

- [x] Third audit (0.1.7–0.1.8): 13 findings fixed, including silent loss of history (HIGH).
- [x] **bge-m3 live**: 9.2 s first call, 1.4 s after, 1.14 GB loaded; hybrid search on the real `.history`.
- [x] **The system prompt's KV saved across restarts** (K): first answer after a restart 132 s → 15.4 s, identical.
- [x] ~~madvise for hashing~~ (Rs): measured, rejected — cold 364 MB/s vs warm 419 MB/s; the disk is not the limit.
- [x] ~~PGO for `bankml serve`~~ (Rs): measured, rejected — 0.71 ms per request against 15–130 s answers.
- [ ] Target-feature 1.1 clean-up (Rs): same bits, fewer `unsafe`; no speed at stake → after 0.2.0.

## 0.2.0 — milestone

- [ ] A full docs pass (README, usage, TECHNICAL, testing/README, oracles, research, embedding, LICENSING, CHANGELOG),
  published as an artifact.
- [ ] **Offer the plain-AVX2 `Q2_0` kernel upstream** (R): llama.cpp has only scalar `Q2_0` on x86 and an open
  VNNI-only PR (#26348); the oracle is the evidence. Prepare the patch against upstream C and the bit-exact report.
- [ ] The oracle method written up for others (R: no public equivalent found).

## After 0.2.0 — P3 and beyond

- [ ] **P3: the Qwen3 forward pass, token-identical to llama.cpp at temperature 0** (TECHNICAL §VI). Design notes from
  V: kernels must be **batch-invariant** (each row reduced in the same order whatever the batch size) or prefill,
  chunked prefill and speculative verification will not be token-identical.
- [ ] **A block-hashed prefix cache in P3** (V: `hash(parent, tokens[16], extra)`, a HashMap to blocks, ref counts,
  an intrusive LRU free list, ~300 lines). Exact by construction: only the unchanged prefix is reused.
- [ ] **Prompt-lookup speculation in P3** (V: the n-gram proposer, a KMP scan over the reversed history, ~60 lines,
  no model memory) with greedy verification (accept until the first mismatch, plus the bonus token). Exact only with
  batch-invariant kernels.
- [ ] **Signed receipts** (R: EigenAI and TEE stacks show the direction): an operator key signs each receipt, in
  `crypto/` under GPL-3.0-only (LICENSING.md).
- [ ] **NEON kernels** for ARM (R: upstream `Q1_0`/`Q2_0` already have NEON; P5, handheld).
- [ ] Real activations in the oracle (TECHNICAL §VI); a Zen 3 row; AVX-512 VNNI where present.
- [ ] Optional external engines at a process boundary (bitnet.cpp, KoboldCpp) as benchmark rivals and cross-checks,
  never linked (LICENSING.md).
