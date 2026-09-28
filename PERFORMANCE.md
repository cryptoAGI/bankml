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
  2-core / 4-thread part.
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

- **Ternary is now cheaper than 1-bit.** At 3 threads bankml's ternary matmuls (0.23–0.25 s) take less time than ggml's
  1-bit ones (0.34–0.35 s), although the ternary weights are twice the bytes. The ternary matmul-only ceiling is
  4.0–4.4 tok/s, against 0.42 for llama.cpp.
- The persistent pool replaced per-call thread spawning (12.6 against 88.8 µs per matmul; `bench_pool_overhead`). With
  it, the 3-thread ternary budget moved from 0.36 s (0.0.2 harness) to 0.23–0.25 s.
- The 1-bit kernel is at parity at every thread count (0.93–1.13×).
- Exactness: the new `oracle_ggml_b11192_real_bonsai_8b_q1_0` covers all **254** `Q1_0` tensors of Bonsai-8B
  (8,188,239,872 weights) bit-exact, 762/762 rows and dot products. Every threaded output in the budgets is compared
  bit for bit with ggml.

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
