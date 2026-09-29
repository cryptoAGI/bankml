# An AVX2 `Q2_0` kernel for llama.cpp (prepared, not yet submitted)

llama.cpp has no x86 kernel for `Q2_0` (ternary, group 64). At b11192, `ggml/src/ggml-cpu/arch-fallback.h` maps
`ggml_vec_dot_q2_0_q8_0` to the generic C on x86, and GCC builds it scalar. The only x86 kernel upstream,
[PR #26348](https://github.com/ggml-org/llama.cpp/pull/26348), needs AVX-VNNI, which a plain-AVX2 CPU (Zen 1/2,
Haswell–Coffee Lake) does not have ([docs/research.md](../docs/research.md)).

[`q2_0_avx2.c`](q2_0_avx2.c) is a drop-in `ggml_vec_dot_q2_0_q8_0` for AVX2 + FMA, written in ggml's style:

- **The same bits as the shipped haswell build.** The two integer sums per block are exact in any order, so they are
  vectorised. The float operations are the generic code's, in its order, with the fused multiply-adds GCC emits made
  explicit: `sumi = fma(d1₀, s₀, 0)`, `sumi = fma(d1₁, s₁, sumi)`, `sumf = fma(d0, sumi, sumf)`.
- **No lane crossings.**
  - The 8 code bytes that meet one `q8_0` block are broadcast into dwords.
  - One variable shift (`vpsrlvd` by 0, 2, 4 and 6 bits) and a mask give all 32 codes as unsigned bytes.
  - The activation is transposed 4×4 within each 128-bit lane (one `vpshufb`), so the two line up.
  - `vpmaddubsw` gives Σc·y, and Σ(c − 1)·y = Σc·y − Σy, which is exact.

## Evidence

[`test_q2_0_avx2.c`](test_q2_0_avx2.c) `dlopen`s the shipped `libggml-cpu-haswell.so` from the
`llama-b11192-bin-ubuntu-x64` release and compares against its exported `ggml_vec_dot_q2_0_q8_0`, bit for bit. It
runs random blocks with every code 0–3 (including the legal but unused +2), random scales, extreme activations, and
lengths up to 4096. On a Ryzen 3 3200U (AVX2, no VNNI), pinned to two cores (bankml's `testing/pinned.sh`):

```
bit-exact against the shipped ggml_vec_dot_q2_0_q8_0: 200000 of 200000 random cases
4096-wide row, 200,000 times: ggml (shipped, scalar) 0.461 s · this kernel 0.134 s · 3.4x (same sums)
```

The same arithmetic, in bankml's Rust kernel, is bit-exact on **all 8,188,239,872 weights and 762 of 762 recorded dot
products** of Ternary-Bonsai-8B against the same shipped library (bankml's [oracles](../docs/oracles.md)).

**Why 3.4× here and 9.5–9.8× in bankml:** a drop-in must accept ggml's `q8_0` activation as it is, so it transposes
it on every call. bankml rearranges the activation **once per token** and reuses it for every weight row, so no
shuffle remains in the inner loop. Bringing that to ggml would need a `Q2_0`-specific activation layout (a
`vec_dot_type` of its own, or a repack), which is a larger change best discussed with the maintainers first.

## Build and run

```sh
cc -O2 -mavx2 -mfma -mf16c q2_0_avx2.c test_q2_0_avx2.c -ldl -o test_q2_0_avx2
./test_q2_0_avx2 /path/to/llama-b11192        # exit 0 = every case bit-exact
```

## Where it would go upstream

Add it to `ggml/src/ggml-cpu/arch/x86/quants.c` under `#if defined(__AVX2__)`, with the scalar fallback kept for
other builds, and drop the `ggml_vec_dot_q2_0_q8_0` rename for x86 in `arch-fallback.h`. The CI's
`test-backend-ops` then compares it with the reference like every other kernel.

Licence: MIT, llama.cpp's. The authors are Professor Codephreak and Gregory L. Magnusson (cryptoAGI). Whether and when
to open the pull request is the authors' decision; nothing has been submitted.
