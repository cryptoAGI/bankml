# bankml — performance

Every number in this file was measured, on the hardware named beside it, and can be reproduced with the command
given. Nothing is quoted from upstream. Where a result is a lower bound, an estimate or a single run, it says so.

- **Reference engine:** llama.cpp **b11192**, the ubuntu-x64 release checked against its sha256
  (`34cf6fa5de9da0db3932c78fe15fed2fbca17451e665dac0a4f6a3c8fc881ec7`). Kernel comparisons call that release's own
  exported symbols in-process, on the same bytes, so the A/B measures code and nothing else.
- **Models:** PrismML Bonsai, forked with provenance on Hugging Face —
  [Bonsai-8B Q1_0](https://huggingface.co/PYTHAI/Bonsai-8B-gguf-fork) (1-bit, 1.16 GB, sha256 `284a335a…bd54`) and
  [Ternary-Bonsai-8B Q2_0_g64](https://huggingface.co/PYTHAI/Ternary-Bonsai-8B-gguf-fork) (1.58-bit, 2.31 GB,
  sha256 `e17b298d…c55e`); the Q1_0 oracle uses Bonsai-1.7B-Q1_0 (sha256 `3d7c6c90…`).

## Machines

| name | CPU | threads used | RAM | notes |
|---|---|---|---|---|
| **laptop** | AMD Ryzen 3 3200U (Zen+) | 1 (kernels) · 3 (llama-server) | 5.8 GB | the development box; AVX2, no AVX-512 |
| **VPS** | AMD EPYC 7543P (Zen3), 2 vCPU | 1 (`-t 1`, CPUQuota 100 %) | 7.8 GB | mindx.pythai.net |
| **Space** | Hugging Face `zero-a10g` Space, CPU path | ~16 busy (CPU s ÷ wall s) | container | llama.cpp on CPU; the GPU is unused |

## Q2_0_g64 (ternary) kernel against llama.cpp b11192 — laptop, 1 thread

**Why ternary is slow in llama.cpp b11192:** it has **no x86 kernel for `Q2_0`**. On x86 the generic C function is
renamed to `ggml_vec_dot_q2_0_q8_0` (`ggml/src/ggml-cpu/arch-fallback.h`), and there is no repack or sgemm case for the
type; the disassembly of the shipped `libggml-cpu-haswell.so` is scalar code with 64 `imul` per block, used for decode
and prefill alike. It costs about **49 ns per 64 weights** — roughly 7× what ggml spends per weight on `Q1_0`.

`BANKML_GGML_LIB=<b11192 release dir> cargo test --release -- --ignored ab_vs_ggml_q2_0 --nocapture --test-threads=1`
— real layer-0 weights of Ternary-Bonsai-8B, the two kernels alternated in one process on the same bits.

| matmul (rows × columns) | llama.cpp b11192, ns/block min / median | bankml, ns/block min / median | speed-up (median) |
|---|---:|---:|---:|
| attn_q 4096 × 4096 | 48.98 / 49.44 | **4.98** / 5.15 | **9.60×** |
| attn_k 1024 × 4096 | 48.63 / 48.97 | **4.89** / 4.98 | **9.83×** |
| attn_output 4096 × 4096 | 48.93 / 49.92 | **4.98** / 5.13 | **9.73×** |
| ffn_gate / ffn_up 12288 × 4096 | 49.08 / 49.39 | **5.04** / 5.19 | **9.52×** |
| ffn_down 4096 × 12288 | 48.80 / 49.46 | **5.01** / 5.17 | **9.57×** |
| output 151669 × 4096 | 49.61 / 50.73 (492 ms) | **5.11** / 5.29 (51 ms) | **9.59×** |
| prefill 1024 rows × 32 columns, per row·column | 48.90 / 49.77 (per pair) | **3.96** / 3.97 (1×4 tile) | **12.54×** |

Over four runs: decode **9.1–10.6×**, prefill tile **12.3–13.1×**. Absolute ns move with the laptop's boost clock
(ggml read 34–55 ns across runs); both kernels move together, so the ratio is the result. Preparing the activation
once per token costs 11 µs (n = 4096) and 29 µs (n = 12288), about 2 ms per token — 0.5 % of the matmul time, not
included above. From L1 the kernel runs at about 4.0 ns/block against a measured memory floor of about 2.3 ns
(7.7 GB/s single-thread read), so it is still partly compute-bound.

**Exactness** (`oracle_ggml_b11192_real_ternary_bonsai_8b`, re-run independently on 2026-09-26): all **254** `Q2_0`
tensors — **8,188,239,872** weights — dequantized bit-exact; **762 / 762** q8_0 activation rows byte-exact; **762 / 762**
dot products bit-exact against ggml's haswell build (scalar model, AVX2 and portable paths), and a no-FMA model equal to
ggml's baseline `libggml-cpu-x64.so` in 762 / 762. The haswell and x64 builds themselves disagree on 19 of the 762 — the
oracle does test float order. The format's codes are {−1, 0, +1, +2}; in the real file +2 never occurs (31 % / 38 % /
31 % / 0 %).



## Q1_0 (1-bit) kernel against llama.cpp b11192 — laptop, 1 thread

`BANKML_GGML_LIB=<b11192 release dir> cargo test --release -- --ignored --nocapture --test-threads=1`

| measurement | bankml | llama.cpp b11192 | ratio |
|---|---:|---:|---:|
| decode GEMV 12288×4096, ns/block, min | **14.32** (`mat_vec`) | 14.43 | **1.01×** |
| decode GEMV, ns/block, median | 14.47 | 14.93 | 1.03× |
| decode GEMV via `vec_dot_act` (activation scales converted once per token), min / median | 14.37 / 14.61 | 14.43 / 14.93 | 1.00× / 1.02× |
| decode GEMV via per-pair `vec_dot`, min / median | 15.64 / 16.00 | 14.43 / 14.93 | 0.92× / 0.93× |
| prefill 1024×4096 × 32 columns, ns/(block·column), min / median | **12.95 / 13.31** (1×4 tile) | 14.22 / 15.43 (per-pair) | **1.10× / 1.16×** |

Within the crate: the AVX2 path is **15.2×** the portable C port of ggml's generic kernel and **20.1×** the scalar
reference model (L1-resident row, n = 4096: 16.25 vs 246.28 vs 327.02 ns/block).

**Exactness** (`oracle_ggml_b11192_real_bonsai_1_7b`, against the release's `libggml-cpu-haswell.so`):
all **197** Q1_0 tensors — **1,719,904,256** weights — dequantized bit-exact; **788 / 788** q8_0 activation rows
byte-exact; **788 / 788** dot products bit-exact against ggml's AVX2 kernel and against its generic kernel.

## End-to-end baselines: llama.cpp b11192 serving Bonsai (the bar bankml must meet)

Tokens are llama-server's own counts. "Read" is prompt evaluation; "write" is generation.

### 1-bit Bonsai-8B

| where | read tok/s | write tok/s | representative run |
|---|---:|---:|---|
| laptop, 3 threads | 2.3–3.2 | 1.5–2.4 (0.43 under memory pressure) | 310-token prompt: 137 s cold, ~7 s cached |
| VPS, 1 thread | 2.1–3.1 | 1.1–2.5 | 5 hourly runs, 272–299 tokens in 222–585 s |
| Space | cached 310-token prompt in 0.3–0.6 s | 23–26 (warm) | 198 tokens in 7.9–8.5 s |

### Ternary-Bonsai-8B vs 1-bit — the same question under the same system prompt (314 prompt tokens)

| where | model | state | first token | total | written | rate |
|---|---|---|---:|---:|---:|---|
| Space | 1-bit | warm | 0.05 s | 5.0 s | 138 | ~28 tok/s |
| Space | Ternary | warm | 0.3 s | 21.8 s | 82 | ~3.8 tok/s |
| laptop | Ternary | cold | 638.7 s | 868.6 s | 94 | read 0.49, write 0.40 tok/s |
| laptop | Ternary | warm | 2.6 s | 214.6 s | 89 | write 0.42 tok/s |

**Ternary runs about 5–7× slower than 1-bit on the same CPU** under llama.cpp b11192 (laptop ~0.4 vs ~2 tok/s; Space
~3.8 vs ~26 tok/s), although its file is only 2× larger. The next two sections show why, and what bankml's ternary kernel does about it.

## Whole-model decode budget — where one ternary token's time goes

`decode_budget_q2_0`: one token's worth of all **253** ternary matmuls (2.13 GB of weights), each tensor once in file
order as a real token touches them; llama-bench run immediately before and after each pass, clock logged.

| laptop | llama.cpp end-to-end (llama-bench) | llama.cpp matmuls only | bankml matmuls only | bankml ceiling (matmuls only) |
|---|---:|---:|---:|---:|
| 3 threads (3.2–3.3 GHz) | 0.386 / 0.373 tok/s = **2.59 / 2.68 s/token** | 2.54 / 2.64 s | **0.36 / 0.37 s** (7.0×) | **2.77 tok/s** |
| 1 thread (2.78–3.10 GHz) | 4.50 / 5.21 s/token | 4.22 / 4.42 s | **0.49 s** (8.7×) | **2.06 tok/s** |

- **The matmul is the wall:** about 98 % of llama.cpp's time per token at 3 threads (85–98 % at 1 thread). llama-bench's
  3-thread rate reproduces the 0.42 tok/s measured serving the model.
- **What it implies:** one thread of bankml's ternary matmuls (0.49 s) is about the time llama.cpp needs for a whole
  *1-bit* token on three threads (~2 tok/s) — before attention, normalisation and the rest of the forward pass, which
  bankml does not have yet (phase P3), so the ceiling is a matmul bound, not an end-to-end rate.
- bankml gains little from threads here (1.34× from 1 to 3) — the signature of a memory-bandwidth-bound kernel on a
  2-core / 4-thread part. (0.0.2 harness, before the thread pool; superseded by the 0.0.3 section below, which finds
  both kernels compute-bound.)
- Method notes: unprivileged `perf` is blocked on this machine (paranoid = 4), so the matmul share comes from same-clock
  bracketing, not a cycle profile. A second 1-thread sequence was disturbed by other load and is not used.



## Threads (0.0.3) — one token's matmuls at 1–4 threads, both formats

`decode_budget_q1_0` / `decode_budget_q2_0`: all 253 matmuls of one token, each tensor once in file order. ggml's own
kernel and bankml's run on the **same pool and row scheduler** (`par::Pool`, 16-row chunks), so the A/B compares kernel
with kernel. Laptop, two runs, min s/token; the full output is in `testing/results/0.0.3.txt`.

| threads | ternary ggml | ternary **bankml** | 1-bit ggml | 1-bit bankml |
|---:|---:|---:|---:|---:|
| 1 | 4.22–4.65 | **0.45–0.47** | 0.62–0.67 | 0.61–0.62 |
| 2 | 2.65 | **0.31** | 0.45–0.46 | 0.43–0.44 |
| 3 | 2.27–2.36 | **0.23–0.25** | 0.34–0.35 | 0.34–0.35 |
| 4 | 2.11 | **0.23** | 0.34–0.35 | 0.36 |

*These whole-token figures were measured with the model resident in memory, and reproduced on 2026-09-29 on current code: 0.231 s at three threads, 9.45× the reference (docs/PERFORMANCE.md). When other applications leave too little memory to keep the 2.31 GB file cached, the gates measure the disk instead (4.5–5.2 s).*

- **Ternary is now cheaper than 1-bit.** At 3 threads bankml's ternary matmuls (0.23–0.25 s) take less time than ggml's
  1-bit ones (0.34–0.35 s), although the ternary weights are twice the bytes. The ternary matmul-only ceiling is
  4.0–4.4 tok/s, against 0.42 for llama.cpp.
- The persistent pool replaced per-call thread spawning (12.6 against 88.8 µs per matmul; `bench_pool_overhead`). With
  it, the 3-thread ternary budget moved from 0.36 s (0.0.2 harness) to 0.23–0.25 s.
- The 1-bit kernel is at parity at every thread count (0.93–1.13×).
- Exactness: the new `oracle_ggml_b11192_real_bonsai_8b_q1_0` covers all **254** `Q1_0` tensors of Bonsai-8B
  (8,188,239,872 weights) bit-exact, 762/762 rows and dot products. Every threaded output in the budgets is compared
  bit for bit with ggml.

## The memory floor (0.0.4) — how far each kernel is from streaming speed

`bench_memory_floor`: a 768 MiB buffer read on the same pool and scheduler as the matmuls. Laptop:

| threads | read GB/s | floor, ternary token (2.13 GB) | floor, 1-bit token (1.06 GB) |
|---:|---:|---:|---:|
| 1 | 14.9–15.2 | 0.14 s | 0.07 s |
| 2–4 | 16.6–17.2 | 0.12–0.13 s | 0.06 s |

bankml's ternary matmuls (0.23–0.27 s at 3 threads) are about 2× above the floor; its 1-bit ones (0.34–0.37 s),
like ggml's, about 5.5×. Both are compute-bound. (0.0.1 quoted 7.7 GB/s single-thread from a scalar loop; this loop
vectorises, which is the fair floor.)

## 1-bit prefill over prepared activations (0.0.4)

`ab_vs_ggml` (1024 rows × 4096 × 32 columns, ns per block·column, min / median):

| kernel | ns | vs ggml per-pair (median) |
|---|---:|---:|
| ggml b11192 per-pair `vec_dot` | 10.09 / 11.12 | 1.00× |
| bankml 1×4 tile over q8 bytes (0.0.1) | 9.18 / 9.42 | 1.18× |
| **bankml `mat_mul_act`, 1×4 selection tile over `Q8Act`** | **8.25 / 8.35** | **1.33×** |

The gate run, disturbed by other load, measured 1.27×. Two decode experiments were bit-exact and not faster (see
CHANGELOG 0.0.4); 1-bit decode remains at parity with ggml.

## Ternary kernel experiments (0.0.5) — laptop, real Ternary-Bonsai-8B tensors

| experiment | bits | speed vs the file-layout kernel | kept |
|---|---|---:|---|
| `Q2Packed`: quads repacked (codes contiguous, 4 scales adjacent), same 72 bytes | exact | 1.00–1.075× first run, 0.90–1.05× in the gate: noise | no |
| two-row decode tile (activation shared) | exact | 0.80× | no |
| software prefetch 256 / 512 / 1024 bytes ahead | exact | ±3 % | no |

At 3 threads the kernel moves about 9 GB/s against the 17 GB/s floor, and scales 1.8× from 1 to 3 threads on two
physical cores. It is bound by compute per core, and at its instruction-throughput limit on Zen+. No kernel changed
in 0.0.5; the three experiments' code is in `testing/experiments/`.

## Reproduce

```sh
cargo test --release                                   # unit tests
BANKML_GGML_LIB=/path/to/llama-b11192 testing/release_gate.sh    # everything, recorded in testing/results/
BANKML_GGML_LIB=/path/to/llama-b11192 \
  cargo test --release -- --ignored --nocapture --test-threads=1   # oracles + A/B + benchmarks
python3 testing/ggml_oracle.py MODEL.gguf /path/to/llama-b11192 .models/oracle-ternary   # write an oracle
python3 testing/guard_agree.py target/release/bankml [FILE.gguf ...]                   # Rust guard == Python guard
```

The oracle and A/B tests need the model files under `.models/` (fetched from the PYTHAI forks above and checked
against their sha256) and the b11192 release directory. They are `#[ignore]`d so a plain `cargo test` stays offline.

## 0.1.8 – 0.2.0: the gateway, the pin, restarts, and what the laptop can hold

Measured on the same Ryzen 3 3200U, on fixed resources where stated (`testing/pinned.sh`: cores pinned, memory capped
in a cgroup, load recorded).

| what | before | after | how measured |
|---|---|---|---|
| sha256 pin of a 248 MB model (warm cache) | 1.25 s (portable) | **0.23 s** (SHA-NI), 5.5× | interleaved, 3 runs each (0.1.8) |
| sha256 pin of the 1.16 GB 8B model | — | **2.9 s**, equal to `sha256sum` and the published pin | 0.1.8 |
| carrier restart (hash + load), 8B | 25 s (0.1.7) | **7.0 s** | `_start_carrier`, 0.1.8 |
| first answer after an engine restart (Savante's 282-token system prompt, same question, temperature 0) | 132.3 s (prefill 118 s, 315 tokens) | **15.4 s** (restore 0.05 s, prefill 1.2 s, 1 token), identical answer | slot save / restore, 0.1.9 |
| the gateway's own cost per request | — | **0.71 ms** (`/health` 1.08 ms through serve vs 0.37 ms direct) | 200 requests, 0.1.9 |
| cold vs warm hashing (1.7B, pinned) | cold 0.68–0.72 s (364 MB/s) | warm 0.59–0.78 s (419 MB/s) | `posix_fadvise(DONTNEED)` before each cold run; no read-ahead hint adopted |
| n-gram speculation (8B, 3 prompts × 64 tokens, pinned 2 cores, 2.5 GB) | baseline 0.36–1.40 tok/s | ahead in 5 of 6 pairs, +0.01 to +0.36; one −0.06 | token-identical; not beyond this laptop's noise → opt-in |
| draft-model speculation (Bonsai-1.7B for 8B) | 0.47 tok/s | 0.39 / 0.53 | token-identical; no gain → not adopted |
| bge-m3 embedding (Ollama) | — | first call 9.2 s (load), then 1.40 s for 3 texts; 1.14 GB while loaded | 0.1.9 |
| the `Q2_0` drop-in for llama.cpp (`upstream/`) | ggml shipped scalar 0.461 s | **0.134 s** (3.4×) per 200,000 × 4096-wide rows | bit-exact 200,000/200,000, pinned (0.2.0) |

**The whole-token ternary budget on this laptop today.** The gates since 0.1.0 record 4.5–5.2 s per ternary token at
three threads for bankml, and 1.1–1.5× the reference, not the 0.23–0.25 s of the 0.0.3–0.0.6 records. The reason was
measured on 2026-09-29. With the chat engine resident and other applications holding memory, the 2.31 GB ternary file
does not stay in the page cache: 0.49 GB was resident, and after a full read 1.04 GB. Every token then re-reads most of
the weights from disk, and both runtimes wait on it. The per-matmul A/B on cached tensors still shows 9.8× (0.1.8
gate). The whole-token figure needs a machine with at least 3 GB free to be re-measured (docs/TODO.md).

**0.2.1 gate, with the chat engine stopped (2.0 GB free):** bankml 1.42 s per ternary token at one and three threads,
against llama.cpp's 5.15 s (one thread) and 2.99 s (three threads): 3.6× and 2.1×. The more of the model stays cached,
the closer the whole-token figure comes to the kernels' own ratio.

## Re-measured with the model resident (2026-09-29, after 0.2.1)

The browser and the chat engine were closed to free memory, and the ternary file was read twice. 2.25 of its 2.31 GB
stayed in the page cache. `decode_budget_q2_0` at one to four threads gave:

| threads | llama.cpp b11192 (min / median) | bankml (min / median) | ratio | matmul-only ceiling |
|---|---|---|---|---|
| 1 | 4.145 / 4.161 s | **0.435 / 0.439 s** | 9.52× | 0.24 → 2.30 tok/s |
| 2 | 2.345 / 2.500 s | **0.284 / 0.313 s** | 8.27× | 0.43 → 3.53 tok/s |
| 3 | 2.185 / 2.221 s | **0.231 / 0.232 s** | 9.45× | 0.46 → 4.32 tok/s |
| 4 | 2.179 / 2.202 s | **0.223 / 0.228 s** | 9.78× | 0.46 → 4.49 tok/s |

**This reproduces the 0.0.3–0.0.6 figure (0.23–0.25 s at three threads, about 9×) on today's code**, and it confirms
the explanation above: the kernels never slowed down. The gates in between measured a laptop that could not keep the
model in memory.


## JSON mode's cost (0.3.3) — laptop

The grammar costs time only in three places, and the gate measures each:
- **The check.** The token the chain drew is run through the grammar alone: one candidate, microseconds.
- **The mask.** When that token breaks the grammar, the grammar runs over the whole vocabulary of 151,669 tokens,
  and the chain draws again.
- **The accept.** The chosen token advances the grammar's stacks.

`oracle_grammar_masks` times the mask on every one of its 1,645 recorded steps, over 12 grammars.
`oracle_json_mode{,_ternary}` time the grammar's whole share of real constrained answers: check, mask, accept and
prefill, divided over every generated token.

| measure (gate 0.3.3, laptop) | Bonsai-8B Q1_0 | Ternary-Bonsai-8B |
|---|---:|---:|
| one whole-vocabulary mask, 151,669 tokens (`oracle_grammar_masks`: 1,645 masks over 12 grammars; the vocabulary is shared) | median **24.2 ms** · p90 47.0 · max 101.0 | the same |
| tokens redrawn under the mask, in the recorded answers | 152 of 860 (17.7 %) | 49 of 534 (9.2 %) |
| grammar time per generated token: check, mask when needed, accept, and the prefill, spread over the answer | **3.90 ms** | **3.95 ms** |
| for scale: the matmul-only decode budget at 3 threads (`decode_budget_*`, same gate) | 0.389 s | 0.279 s |

So JSON mode costs about **1 %** of a 1-bit step and 1.4 % of a ternary step on average. When a token has to be
redrawn, that step costs one mask more, about 6 % of a 1-bit step at the median. An answer the model writes as valid
JSON on its own costs almost nothing: `bankml generate --json` on the cat prompt redrew no token and spent 0.4 ms of
grammar per token.

The mask is a straight port of llama.cpp's `reject_candidates`, which walks every candidate's code points per grammar
stack. A byte trie over the pieces would share their prefixes (TODO.md). It would need the same oracle, and it is not
worth much at today's cost.

## F16 and the Llama graph (0.3.4) — laptop, against llama-server b11192

The models O4 opened: SmolLM2-135M-Instruct and mindX's own `mindx-gen39` (SmolLM2-135M + a LoRA, merged), both F16
with tied embeddings (270 MB), and Bonsai-1.7B Q1_0. Both engines at **3 threads**, greedy, the same chat messages,
alternated in pairs (`bankml generate` against `llama-server -t 3 -c 2048 -np 1 --jinja --reasoning off`, its
`/v1/chat/completions` timings, `cache_prompt: false`). The machine was not idle (load average 1–2.8): read the ranges,
not single numbers.

| model, case | bankML | llama-server b11192 | ratio |
|---|---:|---:|---:|
| SmolLM2-135M-Instruct F16, 45-token prompt, 128 tokens decoded | decode **38.0–38.9 tok/s** (3 runs) | 40.4–42.7 tok/s | ≈ 0.93× |
| the same, prompt | 0.3–0.4 s | 0.4 s (115–119 tok/s) | ≈ 1× |
| SmolLM2-135M-Instruct F16, 858-token prompt (docs/OLLAMA.md's first 2,400 characters), 64 decoded | prompt **8.6–9.6 s**, decode at ~900 cells 27.2–30.1 tok/s | prompt 7.8–8.6 s (100–110 tok/s), decode 20.2–29.9 tok/s | prompt ≈ 0.9×, decode ≈ 1× |
| Bonsai-1.7B Q1_0, 29-token prompt, 64 decoded | 8.4–8.6 tok/s (2 runs) | 10.6–10.9 tok/s | ≈ 0.8× (the 1-bit decode gap of TODO 0.4.0, O8) |

`mindx-gen39` has SmolLM2-135M's shapes, so it runs at SmolLM2's speed. Reproduce: `bankml generate
.models/SmolLM2-135M-Instruct-F16.gguf --max 128 < messages.json` with `BANKML_THREADS=3`, beside a llama-server on the
same file.

**What it took, each change the same bits** (every oracle reran after each):

| change (0.3.4) | SmolLM2 decode, 45-token prompt | 858-token prompt |
|---|---:|---:|
| first correct build (scalar f16 helpers in attention, a linear scan per tensor lookup) | 12.8 tok/s | 18.5 s |
| the reference kernel's f16 dot, scale and accumulate on F16C + FMA; tensors by hash map; norms read once | 36.1 tok/s | — |
| tinyBLAS's 4 × 3 register blocking for F16 prompts, the activations widened once per product | — | 14.4 s |
| heads and rows of attention on the pool | 37.1 tok/s | 10.8 s |
| the tiled kernel widening each 64-cell tile once per 16 rows (compiled with FMA and F16C) | — | 7.8–9.6 s |

**Rejected, with its numbers:** a bounded spin before the pool's workers sleep (to save the ~15.6 µs condvar wake per
matmul, `bench_pool_overhead`). On this 2-core SMT laptop the spinning threads took the siblings' cycles:
`pool.run` went from **15.6 µs to 162.6 µs per call**, and decode did not move (33.7–36.1 tok/s). Reverted.

The bits did not move with any of it: F16 products are checked against the shipped library at every shape class
(`oracle_ggml_b11192_f16`), and the end-to-end oracles (`oracle_llama_server_llama_f16`, the live serve and JSON
oracles) compare whole answers with llama-server's.

**The 8B models gained from the same attention work** (the 0.3.3 and 0.3.4 gates on this laptop, wall time of the same
oracle runs, token-identical in both; not a controlled benchmark — the machine's load differs between the runs):

| gate oracle (same requests, same tokens) | 0.3.3 | 0.3.4 |
|---|---:|---:|
| `oracle_greedy_llama_server_long` (Bonsai-8B Q1_0, 6 prompts of 64+ tokens) | 435 s | 261 s |
| `oracle_greedy_llama_server_deep` (Bonsai-8B Q1_0, 600 tokens past 256 cells) | 756 s | 320 s |
| `oracle_native_serve` (Bonsai-8B Q1_0, 9 turns) | 400 s | 192 s |
| `oracle_json_mode` (Bonsai-8B Q1_0, 23 answers) | 901 s | 541 s |
| `oracle_json_mode_ternary` (Ternary-Bonsai-8B, 13 answers) | 454 s | 242 s |
| `decode_budget_q1_0`, matmuls only, 3 threads (median) | 0.389 s | 0.397 s |

The matmul budget did not move, so the gain is the work around the matmuls: before 0.3.4 the reference attention
kernel converted every f16 value in software, one at a time, and every tensor lookup scanned the list.

One paired check after the gate, end to end, Bonsai-8B Q1_0, the same 29-token prompt, 32 tokens, 3 threads each
(`bench.sh`-style pairs as above; load average 2.9–3.7, so read it as a hint, not a result): bankML decode
**2.59 and 2.30 tok/s**, llama-server **2.33 and 2.48 tok/s**; prompts 8.7 / 10.5 s against 10.9 / 10.1 s. TODO 0.4.0
recorded 1.9–2.0 against 2.8 before. Whether 1-bit decode is now at parity needs the pinned, idle-machine
measurement (`testing/pinned.sh`); it is not claimed here.
