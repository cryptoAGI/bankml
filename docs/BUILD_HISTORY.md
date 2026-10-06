# Build history

How bankML was built, phase by phase, with the evidence each step stood on. This record was the crate documentation of
`bankML/bankml.rs` from the first commit (0.0.1, 2026-09-26) through 0.3.5 (2026-10-03). It moved here at 0.3.6 so
the crate documentation can say what bankML *is*, while this page keeps how it *became* that. The text from
[The original design and phase ledger](#the-original-design-and-phase-ledger) on is preserved word for word, stale
status lines included: they are the record of what was true when they were written. The full per-release detail
is in [CHANGELOG.md](../CHANGELOG.md), and every release's gate record is in [testing/results/](../testing/results/).
What bankML is now, module by module, is in [modules/](modules/README.md); the design intent the phases test is the
Thesis in [TECHNICAL.md](TECHNICAL.md#thesis--professor-codephreak-and-gregory-l-magnusson), which also stands alone
in [thesis.md](thesis.md).

## Where each phase ended up (as of 2026-10-06: 0.3.6 released, 0.3.7–0.3.9 unreleased)

| phase | what it asked | where it was done |
|---|---|---|
| **P0 — wrap** | `bankml serve` in front of llama-server: the gate, the upstream bound to the verified file, a receipt on every answer | **0.0.6** (2026-09-28); the Savante UI on it from 0.0.6–0.1.0 |
| **P1 — guard + receipts** | the GGUF guard, the sha256 pin, one `verify` gate; a receipt per answer | guard and pin **0.0.1–0.0.2**; `bankml_receipt` on every answer since **0.0.6**; signed receipts and the THOT8 leaf are still open (0.8.0 in [TODO.md](TODO.md)) |
| **P2 — own the kernels** | Q1_0 and Q2_0, bit-exact against ggml, then faster | AVX2 kernels **0.0.1**, threads **0.0.3**, memory floor and experiments **0.0.4–0.0.5**, SHA-NI **0.1.8**; the GPU (Vulkan, bankML's own SPIR-V) **0.2.12–0.2.14**; F16 products **0.3.4**. Unreleased: the GPU limiter and per-shape calibration (0.3.7); ggml's `q8_0` quantizer and `vec_dot_q8_0_q8_0` for the KV cache (0.3.9). NEON and AVX-512 still open |
| **P3 — own the forward** | the Qwen3 block, token-identical to llama.cpp | eleven steps, **0.2.1–0.2.11** (tokenizer, template, layer 0 op by op, the whole model, the ternary model, the three attention kernels, sampling); milestone **0.3.0** (2026-09-29): Savante answered by bankML's own forward pass. The Llama graph followed in **0.3.4**. The `q8_0` KV cache the original ledger asked for first is unreleased (0.3.9), with llama.cpp's Hadamard rotation |
| **P4 — the mindX seam** | an OpenAI-compatible provider mindX can route to | `serve --native` **0.3.0**, Ollama's API **0.3.1**, the C API **0.3.2**, JSON mode **0.3.3**, mindX's own model **0.3.4**, JSON schemas and `bankml create` **0.3.5**, the penalties **0.3.6**; **mindX's default engine on its VPS since 2026-10-04**. Unreleased: the rest of the sampler chain (0.3.7); the context limit, slots, the host prompt cache, logprobs (0.3.8). Open: mindX's inference discovery using it as a provider (0.9.0 in [TODO.md](TODO.md)) |
| **P5 — handheld** | the same crate on Android and iOS | open (see [TODO.md](TODO.md), *Handheld and distributed intelligence*) |

Two lines of the original ledger no longer hold, and are kept below as written: "Nothing here runs a model yet" (true
until 0.2.7, when bankML's own forward pass first generated llama.cpp's tokens), and "out of scope until measured
need: GPU backends, … training" (the GPU component arrived in 0.2.12 and mindXtrain's stages in Rust in 0.2.13, each
after a measurement asked for it). The `answer()` stub the status line names was removed at 0.3.6: answers come from
`native::Native::complete`, behind `serve`, the C API and `bankml generate`.

## The phases since the ledger

The ledger below ends at 0.3.5. What followed, release by release, with the evidence each step stood on. The figures
come from [CHANGELOG.md](../CHANGELOG.md); the oracles are described in [oracles.md](oracles.md) §5f–§5h.

**0.3.6 (released 2026-10-04): the penalties (O2).** llama-server's repeat, frequency and presence penalties over
`repeat_last_n`, with the prompt in the window as llama-server fills it (`sampler.rs`). Evidence: `oracle_penalties`
and `oracle_penalties_8b`, 56 of 56 answers token-identical on mindx-gen39, Bonsai-1.7B and Bonsai-8B each, 12 of 12
refusals each; live 85 of 85 through `/v1` and `/api/chat`. Record: `testing/results/0.3.6.txt`.

**0.3.7 (unreleased): the whole sampler chain; bankML measures itself; the GPU limiter; the console.**
- Typical-p, top-n-σ, XTC, dynamic temperature and DRY, as `llama-sampler.cpp` writes each, in the default order
  ([modules/sampler.md](modules/sampler.md)). Evidence: `oracle_samplers` 76 of 76 on mindx-gen39 and on Bonsai-1.7B,
  16 of 16 refusals each (Bonsai-8B to be recorded); `oracle_std_sort` 876 of 876 orders against libstdc++.
- A record per completion (TTFT, prompt and generation speed, energy where RAPL is readable) at `GET /bankml/metrics`
  ([modules/metrics.md](modules/metrics.md)).
- `BANKML_GPU_LIMIT` and per-shape calibration ([modules/gpu.md](modules/gpu.md)). Measured on the Vega 3 with
  Bonsai-1.7B: with one global share the card cost 18 % (10.40 → 8.51–8.58 tok/s); per shape it is neutral (10.36 off,
  10.30–10.38 on), and card memory falls from 66 to about 13 MB.
- The bankML persona and console ([modules/console.md](modules/console.md)), checked by `testing/test_console.py`.

**0.3.8 (unreleased): the serving contract.** Each against llama-server b11192's answers on Bonsai-1.7B, except where
[oracles.md](oracles.md) §5g says the engine is its own reference: the context limit (8 of 8), slot save, restore and
erase (19 of 19), one slot with llama-server's host prompt cache (14 of 14,
[modules/prompt_cache.md](modules/prompt_cache.md)), and logprobs on `/v1`, streamed or not (14 of 14).

**0.3.9 (in progress): a `q8_0` KV cache and a faster grammar mask.**
- `BANKML_CACHE_TYPE=q8_0` stores K and V as `q8_0` blocks, 53 % of the f16 cache's bytes, with llama.cpp's Hadamard
  rotation around the quantized cache. Evidence: `oracle_ggml_b11192_q8_0_kv_kernels` (4,000 rows byte-exact, 4,000
  dot products bit-exact against the shipped library) and `kv_oracle_live` (6 of 6 answers against llama-server with
  `--cache-type-k/v q8_0`).
- The whole-vocabulary mask through a trie: median 2.94 ms against 38.8 ms (13×) over the oracle's 1,645 masks, every
  mask identical to llama.cpp's by both paths ([modules/grammar.md](modules/grammar.md)).
- 1-bit decode speed, the third piece, is still to be measured on an idle machine.

## The original design and phase ledger

*As it stood in `bankML/bankml.rs` at 0.3.5 (commit `e4d54c5`).*

### bankml.rs — the in-house Rust player for low-bit models

**Status (0.0.6, 2026-09-28): P0 serve + the Savante UI;: P1 guard + pin (one `verify` gate) and the P2 Q1_0 and Q2_0 (ternary) kernels are native and
proven (see the checklist). Nothing here runs a model yet** — `answer()`'s `todo!()` is P0/P3. The
ternary finding: llama.cpp b11192 has **no x86 Q2_0 kernel** (scalar C, ~49 ns per 64 weights on the
dev box); bankml's is bit-exact with it and ~9–10× faster, and ggml's matmuls are ~90–98 % of its wall. Created 2026-09-26 on the operator's instruction: "create llama.cpp rust version
todo as bankml.rs", design goals **optimization, succinct, verified response**.

Doctrine (`/core`): Python prototypes, Rust ports the *working* architecture. The working
architecture exists as of today: PrismML Bonsai-8B (Qwen3-8B dense, ggml `Q1_0`, 1.16 GB,
PYTHAI/Bonsai-8B-gguf-fork) served by **llama.cpp b11192** on the 2-core mindX node,
measured by `scripts/ternary_diagnostics.py` (data/monitoring/ternary_diagnostics.jsonl):

```text
  5/5 verified · 148 prompt → 130 completion tokens in 58.7 s · gen 2.73–3.36 tok/s (median 3.0)
  prompt eval 1.1–4.4 tok/s · first token 0.34–8.8 s · peak RSS 1,786 MB · 2 threads, CPUQuota 150 %
```

One core (mindx-bonsai.service, 1 thread, CPUQuota 100 %), the same five prompts, 2026-09-26:

```text
  5/5 verified · 148 → 138 tokens in 105.8 s · gen median 2.62 tok/s · first token 8.9–12.3 s
  97.3 CPU-seconds total = 0.705 CPU-s per completion token · peak RSS 1,785 MB
```

That row is the bar. bankml.rs is not done until it matches it on the same prompts, then beats it.
The engineer is the `bankml` subagent (`.claude/agents/bankml.md`): prefill first (the measured wall),
then memory traffic per token, with strategies from llama.cpp, vLLM, Ollama and KoboldCpp.

#### The three design goals, made testable

1. **Optimization** — for a 1-bit weight the matmul is a signed sum: `y = s · (Σ x[w=+1] − Σ x[w=−1])`
   per 128-weight group. No multiplies on the weights (ggml-exactness pins the activations to q8_0,
   so the sums run as `maddubs`/`madd` by ±1 or {0,1}). Targets on the reference node:
   gen ≥ 3.0 tok/s (parity), then ≥ 4.5 tok/s; prompt eval ≥ 2× b11192 (batched GEMM, not GEMV);
   RSS ≤ weights + KV(ctx, kv_type) + 150 MB. Measured by ternary_diagnostics, never quoted.
2. **Succinct** — one crate, no Python, no runtime downloads. Budget: ≤ 3,000 lines for
   GGUF + Qwen3 forward + Q1_0/Q2_0_g64 kernels + sampler + OpenAI-compatible server. Every
   dependency justified in `Cargo.toml` comments. A single static binary.
3. **Verified response** — every answer carries its evidence, and a model that cannot be
   verified does not answer:
   - before load: the GGUF guard (port of [minaiml](https://github.com/minaiml) `gguf_guard.py`) → `play | refuse | need_more`;
     refuse is shown with its reason, never a silent fallback (Bonsai 2 Q2_0 loads on mainline
     and answers in gibberish — the worst failure);
   - the file's sha256 must match a pinned record (`FORK.json` of the PYTHAI fork);
   - kernels are proven **bit-exact against ggml** on recorded tensors before they are trusted;
   - each response returns a `Receipt`: model sha256, guard verdict, prompt/completion tokens,
     timings, and an optional THOT8 ternary commitment of the output (see `thot8_cpu.py` —
     same codec, same Keccak leaf as `THOTLib.sol`), so a response can be checked later by
     anyone holding the model file.

#### Phases (each ends in a measured row, or it is not done)

- [x] **P0 — wrap.** (0.0.6) Not FFI: `bankml serve` (`serve.rs`), a loopback gateway to llama.cpp b11192's
  `llama-server`, OpenAI-compatible `/v1/chat/completions`; `verify` in front, the upstream bound to the
  verified path, a receipt (with the answer's sha256) on every answer. Laptop: Savante turn 316 + 28 tokens in
  125 s (prefill 2.8 tok/s — llama.cpp's). The Savante UI (`sAGI/savante.py`, view / interact) sits on it.
- [ ] **P1 — guard + receipts native.** (guard and pin proven; receipts not yet emitted)
  - [x] GGUF v3 header parse + the three traps + kv_f16_bytes_per_token (`gguf.rs`). Evidence: the 9
    cases of `test_gguf_guard.py` as Rust tests + fail-closed extras; `testing/guard_agree.py` = **20/20**
    JSON-identical with the Python guard (real `Bonsai-1.7B-Q1_0.gguf` sha256 `3d7c6c90…` + all synthetic
    cases, both engines): play · qwen3 · F32 113 / Q1_0 197 · 114,688 kv bytes/token.
  - [x] sha256 (FIPS vectors; equals `sha256sum` and the HF LFS oid on the real file) + FORK.json pin
    (`bankml pin FILE --fork FORK.json`; parses the real PYTHAI/Bonsai-8B-gguf-fork FORK.json → `284a335a…`,
    refuses a mislabelled file naming both hashes).
  - [x] 0.0.2: `verify` = guard then pin as one gate (`bankml verify FILE --fork FORK.json --json`);
    on the real Ternary-Bonsai-8B + its FORK.json → play, `e17b298d…`. Guard hardened against hostile
    headers (nested arrays refuse instead of overflowing the stack; KV-size overflow refuses).
  - [ ] Receipt emitted per answer, THOT8 leaf (needs P0/P3 to have an answer to sign).
- [ ] **P2 — own the kernels.** (Q1_0 and Q2_0 proven on x86 AVX2; NEON, real activations open)
  - [x] Q1_0 exactly as ggml b11192 defines it (`q1_0.rs`, read from source): f16 `d` first, 16 sign
    bytes LSB-first; the matmul is `ggml_vec_dot_q1_0_q8_0` against **q8_0 activations** (AVX2
    quantizer: 127/amax, round-half-even). Evidence (`oracle_ggml_b11192_real_bonsai_1_7b`, against the
    exported symbols of the sha256-checked b11192 ubuntu-x64 release, `libggml-cpu-haswell.so`): all
    197 Q1_0 tensors = **1,719,904,256 weights dequantized bit-exact**; **788/788** q8_0 rows byte-exact;
    **788/788** vec_dot results bit-exact vs ggml AVX2, and 788/788 vs ggml generic (whose shipped
    build fuses to FMA — read from the disassembly). AVX2 == scalar model on 3,500 + 2,000 randomized cases.
  - [x] Measured (dev box Ryzen 3 3200U, Zen+, 1 thread, in-process A/B vs ggml's own kernel, same bits):
    decode GEMV 12288×4096 **parity** (min 10.0 vs 10.0–10.2 ns/block; `vec_dot_act`, activation scales
    converted once per token); prefill 1×4 tile **1.10×** ggml's per-pair vec_dot (min 9.1 vs 10.1
    ns/(block·col)). Not the node: a Zen3 row is the deciding measurement.
  - [x] Q2_0 (id 42, group 64) exactly as ggml b11192 defines it (`q2_0.rs`, read from the tag's source):
    f16 `d` first, 16 bytes of 2-bit codes LSB-first, code c → (c − 1)·d ∈ {−1, 0, +1, **+2**} (the
    Bonsai file never uses +2: 31 % / 38 % / 31 % / 0 %; the kernels take it anyway). `vec_dot_type` q8_0.
    **On x86, b11192 has no Q2_0 kernel**: `arch-fallback.h` renames the generic C to
    `ggml_vec_dot_q2_0_q8_0`, no repack, no sgemm case; the haswell symbol is scalar (64 `imul`/block,
    read from the disassembly), used per (row, column) for decode *and* prefill. Evidence
    (`oracle_ggml_b11192_real_ternary_bonsai_8b`, real Ternary-Bonsai-8B sha256 `e17b298d…`, via mmap):
    all 254 Q2_0 tensors = **8,188,239,872 weights dequantized bit-exact**; **762/762** q8_0 rows
    byte-exact; **762/762** vec_dot bit-exact vs ggml haswell (scalar model, AVX2, portable path), and
    762/762 of a no-FMA model vs the baseline `libggml-cpu-x64.so` (which differs from haswell in 19/762:
    the float order is really being tested). 3,300 randomized cases incl. code 3, q = −128, tail blocks.
  - [x] Measured (dev box, 1 thread, in-process A/B vs ggml's own Q2_0 kernel, real layer-0 weights,
    alternating, 4 runs; the laptop's boost clock moves both, so ratios are the result): decode GEMV
    **9.1–10.6×** (ggml 48.6–55 vs bankml 4.9–6.0 ns/block min; `output` 151669×4096: 492 → 51 ms);
    prefill 1×4 tile **12.3–13.1×** (49 vs 3.96 ns/(block·col)). The per-token activation layout costs
    11 µs (n = 4096) / 29 µs (n = 12288), ≈ 2 ms per token: outside the matmul timings, 0.5 % of them. The kernel: two blocks per 256-bit
    register, codes by `(v >> 2r) & 3`, `maddubs(c, q) − Σq`, the activation (not the weights) permuted
    once per token, ggml's float chain kept serial. One-token budget (`decode_budget_q2_0`, all 253
    matmuls, 2.13 GB, each tensor once in file order, llama-bench run before and after at the same clock):
    3 threads — llama-bench tg **2.59–2.68 s/token**, ggml's matmuls **2.54–2.64 s** (≈ 98 % of the wall),
    bankml **0.36–0.37 s** (7.0×; matmul-only ceiling 2.77 tok/s); 1 thread — llama-bench 4.50–5.21 s,
    ggml matmuls 4.22–4.42 s, bankml **0.49 s** (8.7×; ceiling 2.06 tok/s). Not the node: a Zen3 row decides.
  - [x] 0.0.3 threads: `par::Pool` (persistent, zero-dependency) + `mat_vec_par`/`mat_mul_par`, bits
    independent of thread count. One ternary token's 253 matmuls: **0.23–0.25 s at 3 threads** (ggml
    2.27–2.36 s), below ggml's *1-bit* 0.34 s; 1-bit at parity. New 8B Q1_0 oracle: 254 tensors bit-exact.
  - [x] 0.0.4: memory floor 15–17 GB/s (both kernels compute-bound: ternary ~2×, 1-bit ~5.5× above
    it); `mat_mul_act` 1-bit prefill 1.27–1.33× ggml; two bit-exact decode variants measured and rejected.
  - [x] 0.0.5: three ternary experiments bit-exact and not reliably faster (repacked quads 0.90–1.075×,
    two-row tile 0.80×, prefetch ±3 %); no kernel change. Both kernels are at the Zen+ instruction limit:
    the next gains are P3 and a Zen3 row.
  - [ ] NEON; AVX-512 (the node has none); bit-exact on *dumped real activations* (today: real weights ×
    synthetic activations); a Q1_0 GEMM that beats 1.1× for prompt eval (the Q2_0 layout above runs at
    5.0 ns per 64 weights vs Q1_0's 7.2 — porting it to Q1_0 is the next experiment).
- [ ] **P3 — own the forward.** Qwen3 dense block (RMSNorm, RoPE, GQA, SwiGLU) with a quantized
  KV cache (q8_0 first — at 1.125 bpw the KV cache, not the weights, sets the RAM ceiling).
  Drop the FFI. Parity with P0 on all 5 prompts at temperature 0 (token-identical is the goal).
- [ ] **P4 — mindX seam.** Serve on loopback; register as an OpenAI-compatible provider in mindX
  (the vLLM handler shape); `inference_budget.record_inference(..., model=, caller=)` sees real
  usage; `/substrate/` and feedback show its row. Then the 8B migration can route to it.
  - [x] 0.3.1: Ollama's API on `serve --native` (`ollama.rs`), a registry of pinned models, one resident, verified on
    every load, `keep_alive`; native only (docs/OLLAMA.md: what mindX asks of Ollama, and the O1–O8 track).
  - [x] 0.3.2: a C API (`capi/`, `libbankml.so`/`.a`, `capi/include/bankml.h`, docs/CAPI.md) — open (the same
    verify), chat (the same answer and receipt as `serve --native`), and a printf-style `bankml_log` defined in
    Rust as a C-variadic function (Rust 1.99); the llama.h-shaped seam for embedding bankML in another program.
  - [x] 0.3.3: JSON mode (O6's first cut, `grammar.rs`) — llama.cpp's GBNF engine ported, the grammar llama-server
    builds for `response_format: json_object` with its prefill, and the sample → check → masked redraw of
    `common_sampler_sample`; `/v1` `response_format`/`grammar`, Ollama `format: "json"`, `bankml_chat`,
    `bankml generate --json`, token-identical to llama-server b11192 greedy and seeded.
  - [x] 0.3.4: O4 — tied embeddings (Bonsai-1.7B native), F16 weights (`f16.rs`: ggml's `vec_dot_f16` and
    llamafile's tinyBLAS, chosen by shape as ggml chooses them), the Llama graph (NORM RoPE, no Q/K norms), SmolLM2's
    `smollm` pre-tokenizer and two ChatML templates: SmolLM2-135M-Instruct and mindX's own `mindx-gen39` served
    natively, token-identical to llama-server b11192 (greedy, seeded, `/v1`, `/api`, C API, JSON mode).
  - [x] 0.3.5: O6b — JSON schemas (`schema.rs`: llama.cpp's `json_schema_to_grammar` and the chat parser's wrapping
    per template), answers token-identical to llama-server b11192 on all five native models, the content rule
    against llama.cpp's own parser; O5's first cut — `bankml convert` (byte-identical to b11192's converter) and
    `bankml create` (derived models as verified layers), mindX's persona layer token-identical end to end.
- [ ] **P5 — handheld.** Same crate → Android (NDK) / iOS as the minaiml BROBOT engine alternative.

Out of scope until measured need: GPU backends, PrismML fork types (PQ2_0, PTQ1_0), Bonsai 2's
Hadamard basis (watch ggml-org/llama.cpp#27779), training.

