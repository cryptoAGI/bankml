# Changelog

## 0.0.3 — 2026-09-28

Threads, and more oracle evidence. Each speed figure below was measured with the oracles passing on the same code.
The full record is in `testing/results/0.0.3.txt`, written by the new release gate.

### Added
- **`par::Pool`**: a zero-dependency persistent thread pool. The caller's thread is worker 0. Each matmul wakes the
  pool instead of spawning threads: 12.6 µs per call against 88.8 µs for `thread::scope`, at 3 threads on the dev box.
  That is about 3 ms per token instead of 22 ms. The row scheduler hands out 16-row chunks from an atomic counter.
- **`mat_vec_par` / `mat_mul_par`** for `Q1_0` and `Q2_0`. Every row goes through the same single-thread kernel, so
  the output bits do not depend on the thread count. Unit tests check this at 1–4 threads, and the whole-model budgets
  check it against ggml.
- **An oracle on the real 8B 1-bit model**: `oracle_ggml_b11192_real_bonsai_8b_q1_0`. All 254 `Q1_0` tensors of
  Bonsai-8B (8,188,239,872 weights) dequantize bit-exact, 762/762 q8_0 rows are byte-exact, and 762/762 dot products
  are bit-exact against ggml AVX2 and generic. 0.0.2 covered only the 1.7B file for 1-bit.
- **`decode_budget_q1_0`**: one token's worth of all 253 `Q1_0` matmuls of Bonsai-8B, against ggml's kernel on the same
  pool and scheduler, at 1–4 threads.
- **`testing/release_gate.sh`** runs the unit tests, clippy, both guard checks, every oracle, both A/Bs and both
  budgets, and writes `testing/results/<version>.txt`. A kernel change counts only if every oracle still passes.

### Measured (dev box, Ryzen 3 3200U, 2 cores / 4 threads; all 253 matmuls of one token; min s/token over two runs)
| threads | ternary: ggml → bankml | 1-bit: ggml → bankml |
|---:|---:|---:|
| 1 | 4.22–4.65 → **0.45–0.47** (9.4–9.9×) | 0.62–0.67 → 0.61–0.62 (1.00–1.07×) |
| 2 | 2.65 → **0.31** (8.7×) | 0.45–0.46 → 0.43–0.44 (0.93–1.13×) |
| 3 | 2.27–2.36 → **0.23–0.25** (9.5–9.9×) | 0.34–0.35 → 0.34–0.35 (0.98–1.00×) |
| 4 | 2.11 → **0.23** (9.4×) | 0.34–0.35 → 0.36 (0.96–0.97×) |

bankml's ternary matmuls now take less time per token than ggml's **1-bit** ones: 0.23–0.25 s against 0.34–0.35 s at 3
threads, although the ternary file is twice the size. The matmul-only ceiling for ternary rises from 2.77 tok/s (0.0.2,
which spawned threads for each call) to 4.0–4.4 tok/s. The 1-bit kernel stays at parity with ggml. The 1-thread medians
are noisy (ternary 0.62 s against 0.47 s min): the model does not fit in page cache alongside everything else in 5.8
GB of RAM.

### Testing
- A new **`testing/`** folder holds every test that lives outside the modules. It contains the release gate, the
  oracle generator, the guard harnesses (moved from `tools/`) and **`testing/cli.rs`**, a new end-to-end suite that
  runs the `bankml` binary. The suite covers verdicts and exit codes, a hostile header, pin and verify, and runs as a
  cargo integration test.
- **`testing/results/`** holds each release's record. `testing/README.md` lists every test and where it lives.

### Changed
- The ternary budget and the A/B harness use `par::Pool` for both engines instead of spawning threads per call. The
  ggml side runs on the same scheduler, so the A/B compares kernel with kernel.

## 0.0.2 — 2026-09-28

An audit of 0.0.1. The kernels are unchanged and re-proven: the oracles again pass on all 8,188,239,872 ternary and
1,719,904,256 1-bit weights against llama.cpp b11192. The fixes below all concern untrusted input, or an API that let
safe code reach unsafe code with the wrong sizes.

### Fixed
- **The guard crashed on nested arrays.** GGUF arrays of arrays were parsed by recursion with no limit, so a 24 MB
  header aborted `bankml guard` with a stack overflow (exit 134) instead of refusing the file. Nesting beyond
  `MAX_ARRAY_DEPTH` (64) now refuses, with the reason given.
- **The guard's KV size could overflow.** `kv_f16_bytes_per_token` multiplied header integers in `i128` without
  checking. In release builds the value wrapped with no error and the verdict was `play`; debug builds panicked. An
  overflow now refuses, with the reason given. The Python guard, which uses big integers, still plays these files; the
  difference is listed in `gguf.rs` with the other deliberate differences.
- **Soundness: activation fields were public.** `q1_0::Q8Act` and `q2_0::Q8Act2` exposed `n`, `d`, `sum` and the
  other fields that the AVX2 kernels read through raw pointers. Safe code could set `n` beyond the buffers and cause
  an out-of-bounds read. The fields are now private, with read-only accessors (`n()`, `qs()`, `d()`).
- **Tensor spans were unchecked.** `Mmap::tensor` and `tensor_bytes` truncated a 128-bit element count to 64 bits
  and used unchecked arithmetic. `tensor_bytes` also allocated whatever size the header claimed before it read the
  file. A new function, `gguf::tensor_span`, checks the type, requires whole blocks per row and rejects overflow.
  `tensor_bytes` now checks the span against the file size before it allocates.
- **`bankml pin` with an unreadable `--fork` file** said "no sha256 record: unpinned, refused" (exit 2). It is an I/O
  error, and now says so (exit 1).

### Added
- `bankml::verify` and `bankml verify FILE --fork FORK.json [--json]` run the guard and then the pin as one gate: the
  check that P0 puts in front of every answer. `--json` prints the verdict, the model sha256 and the bankml version.
- `bankml version` and `bankml::VERSION`.
- CI (GitHub Actions) runs the build, the unit tests, `clippy -D warnings`, the Python guard suite and the Rust-vs-Python
  guard agreement check on every push.
- 4 new unit tests (31 in all): nested arrays, KV overflow, tensor spans, and `verify`.

### Changed
- The `Q8Act`/`Q8Act2` fields are no longer public (see above). This is an API break, allowed while the crate is
  `publish = false` and 0.0.x.

## 0.0.1 — 2026-09-26

First public cut: the P1 GGUF guard and sha256 pin, and the P2 `Q1_0` and `Q2_0_g64` kernels, all bit-exact against
llama.cpp b11192.
