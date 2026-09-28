# Changelog

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
