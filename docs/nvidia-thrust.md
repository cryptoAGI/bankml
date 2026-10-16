# NVIDIA Thrust and bankML

> **Credit: BROBOT.** This page exists because BROBOT, the companion of [DeltaV THRUST](https://deltavthrust.com), made bankML
> aware of THRUST: the name the rocket shares with NVIDIA's parallel-algorithms library. BROBOT asked the question;
> this is the answer, researched 2026-10-16, with the measurements and the limits. Estimates are marked as estimates.


*Research report, 2026-10-15. No repository was modified. Facts from NVIDIA's CCCL repository were read through the
GitHub API on this date. Local code was read from `~/bML/bankml` (bankML 0.4.5 in progress, commit `c148bd0`, the newest
local checkout; `~/cryptoAGI/bankml` is at 0.3.6) and `~/cryptoAGI/browseML` (0.8.7 in progress, `03473b5`), plus
`~/cryptoAGI/browseML-webgpu`. Anything marked **estimate** is arithmetic from public specs or one quick local
measurement, not a benchmark of a real integration. Anything marked **unverified** I could not confirm from a primary
source.*

---

## 0. The short version

- Thrust is a host-side C++ template library of generic parallel algorithms (sort, scan, reduce, compaction,
  transform). Since 2023 it lives in NVIDIA's **CCCL** repository with CUB and libcu++. The current release is
  **CCCL v3.5.0 (2026-10-07)**. Its CUDA backend needs `nvcc` (or another CUDA C++ compiler) and an NVIDIA GPU.
- **Thrust is not a matmul library.** In a low-bit LLM runtime almost all decode time goes to packed-weight
  matrix-vector products. Thrust does nothing for those. They need custom kernels (bankML already has its own CPU
  kernels and Vulkan/WGSL GPU kernels for them).
- Thrust/CUB fit the **sampling tail**: top-k, softmax over the candidates, the cumulative sum for top-p, argmax. In
  bankML today that tail costs about **0.5–1 ms per token**, measured here, against **~120–500 ms** for the token
  itself on the laptop. On the CPU it is under 1 % of the time. It becomes worth moving **only once the matmuls run on
  a GPU**, when tokens take a few milliseconds and copying 600 KB of logits to the host plus a CPU sort becomes a
  10–20 % tax (**estimate**).
- **Bit-exactness is the main constraint.** CUB's new top-k (`DeviceTopK`) *only* supports non-deterministic,
  unordered output. Thrust's float sort is a radix sort with its own tie order, not libstdc++ `partial_sort`'s. The
  GPU's `expf` is not glibc's. `nvcc` contracts mul+add into FMA by default. Each of these breaks the release gate
  unless it is designed around. The design in §3.5 gets around it: use the GPU only when the top-k set has a
  unique order, and fall back to the CPU otherwise.
- **browseML cannot use Thrust at all.** It runs in a browser (wasm + WebGPU). The equivalents are WGSL compute
  shaders: scan, reduce and radix sort, with portable single-pass scan research from 2025. For browseML's current
  speed (about 3 tok/s), none of them is on the critical path.

---

## 1. What Thrust is today

### 1.1 Identity and home

- **CCCL ("CUDA Core Compute Libraries")** merges three libraries into one repository: **Thrust** (the high-level
  parallel algorithms library), **CUB** (low-level device-, block- and warp-wide primitives) and **libcudacxx /
  libcu++** (the CUDA C++ standard library, plus `cuda::` extensions). The old `NVIDIA/thrust` repository is archived
  and points to CCCL.
  Sources: [NVIDIA/cccl README](https://github.com/NVIDIA/cccl) · [archived NVIDIA/thrust](https://github.com/NVIDIA/thrust)
  · [unification announcement](https://github.com/NVIDIA/cccl/discussions/520).
- CCCL describes Thrust as "the C++ parallel algorithms library which inspired the introduction of parallel algorithms
  to the C++ Standard Library", with backends "such as CUDA, TBB, and OpenMP" ([README](https://github.com/NVIDIA/cccl)).
  The original reference is Bell & Hoberock, "Thrust: A Productivity-Oriented Library for CUDA", *GPU Computing Gems
  Jade Edition* (2011).
- **Implementation detail that matters here:** Thrust's CUDA algorithms are thin dispatchers over CUB. For example,
  `thrust/system/cuda/detail/sort.h` includes `cub/device/device_radix_sort.cuh` and calls
  `cub::DeviceRadixSort::SortKeys/SortPairs(…Descending)` whenever keys are arithmetic and the comparator is
  `less`/`greater`. It falls back to merge sort otherwise
  ([sort.h](https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/system/cuda/detail/sort.h)). For anything
  performance-critical, calling CUB directly gives more control (temp storage, streams, determinism requirements).

### 1.2 API surface relevant to inference

| area | Thrust | CUB equivalent |
|---|---|---|
| containers | `thrust::device_vector`, `host_vector`, `universal_vector`, `device_ptr` | — (`cuda::buffer`, `cuda::device_buffer` in libcu++ 3.x) |
| element-wise | `transform`, `for_each`, `fill`, `sequence`, `gather`, `scatter` | `DeviceTransform`, `DeviceFor` |
| reductions | `reduce`, `transform_reduce`, `max_element`, `minmax_element`, `count_if`, `reduce_by_key` | `DeviceReduce` (Sum/Max/ArgMax/TransformReduce), `DeviceSegmentedReduce` |
| scans | `inclusive_scan`, `exclusive_scan`, `*_by_key` | `DeviceScan` |
| sorting | `sort`, `stable_sort`, `sort_by_key`, `stable_sort_by_key` | `DeviceRadixSort`, `DeviceMergeSort`, `DeviceSegmentedSort`, `DeviceSegmentedRadixSort` |
| selection / compaction | `copy_if`, `remove_if`, `partition`, `unique` | `DeviceSelect`, `DevicePartition` |
| top-k | none (sort, then truncate) | **`DeviceTopK`** (CCCL 3.2+), **`DeviceBatchedTopK`** (3.5) |
| fancy iterators | `counting_iterator`, `transform_iterator`, `zip_iterator`, `permutation_iterator` (several moving to `cuda::` in 3.4) | same |

Docs: [Thrust algorithm groups](https://github.com/NVIDIA/cccl/tree/main/docs/thrust/algorithms) ·
[CCCL documentation](https://nvidia.github.io/cccl).

### 1.3 Execution policies and streams

- `thrust::host`, `thrust::device` (the configured device system), `thrust::seq` (sequential, also usable inside device
  code).
- `thrust::cuda::par` and **`thrust::cuda::par.on(stream)`**: run on a given `cudaStream_t`. Defined as
  `execute_on_stream_base::on(cudaStream_t)` in
  [`thrust/system/cuda/detail/execution_policy.h`](https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/system/cuda/detail/execution_policy.h).
- **`thrust::cuda::par_nosync`** (since Thrust 1.16): a hint that lets Thrust skip non-essential synchronizations. The
  caller must synchronize before reading results ([Thrust 1.16 changelog](https://github.com/NVIDIA/thrust/blob/1.16.X/CHANGELOG.md)).
  CCCL 3.1 made `par_nosync` use **stream-ordered (async) allocations by default**
  ([v3.1.0 notes](https://github.com/NVIDIA/cccl/releases/tag/v3.1.0)).
  *Why it matters:* the default `thrust::device` path synchronizes after many algorithms and allocates temporary
  storage per call. At a decode step budget of a few milliseconds, those syncs and allocations cost real time.
- New in CCCL 3.x: CUB "environment" APIs (`cuda::std::execution::env{cuda::stream_ref{s}, cuda::execution::require(…)}`)
  carry the stream, **determinism requirements** and (3.5) **tuning policies**
  ([v3.5.0 notes](https://github.com/NVIDIA/cccl/releases/tag/v3.5.0)).
- CCCL 3.4 also adds C++-standard parallel algorithms in `cuda::std` with a `cuda::execution::gpu` policy
  ([v3.4.0 notes](https://github.com/NVIDIA/cccl/releases/tag/v3.4.0)).

### 1.4 Backends ("systems")

`cpp`, `cuda`, `omp`, `tbb` (plus internal `generic`, `sequential`). The device system is chosen at compile time
(`THRUST_DEVICE_SYSTEM`), and users can add systems
([docs/thrust/developer/systems.rst](https://github.com/NVIDIA/cccl/blob/main/docs/thrust/developer/systems.rst)).
The OMP/TBB backends let the same source run on a CPU. For bankML, though, they would replace its own measured,
bit-exact thread pool (`bankML/par.rs`) with something slower to verify. **No reason to use them.**
AMD maintains a HIP port, **rocThrust** (with `hip_rocprim::par_nosync`)
([rocThrust docs](https://rocm.docs.amd.com/projects/rocThrust/en/latest/how-to/rocThrust-hip-policies.html)). This is
relevant because bankML's GPU direction is vendor-neutral.

### 1.5 Determinism features (critical for bankML)

- **`cub::DeviceReduce`**, since CCCL 3.1, has three levels: `not_guaranteed` (single-pass, atomics), `run_to_run` (the
  traditional two-pass; bitwise identical on the same GPU) and `gpu_to_gpu` (reproducible reduction, identical across
  GPUs) ([v3.1.0 notes](https://github.com/NVIDIA/cccl/releases/tag/v3.1.0)).
- **`cub::DeviceScan`**, since 3.5, offers `determinism::run_to_run` for float sums
  ([v3.5.0](https://github.com/NVIDIA/cccl/releases/tag/v3.5.0)).
- **`cub::DeviceTopK`** (3.2+) **only** supports `determinism::not_guaranteed` with `output_ordering::unsorted`. This is
  a `static_assert` in [`device_topk.cuh`](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_topk.cuh).
  NVIDIA claims "up to 5X" over a full radix sort for small K ([v3.2.0](https://github.com/NVIDIA/cccl/releases/tag/v3.2.0)).
- **`cub::DeviceBatchedTopK`** (3.5) processes one segment per thread block, at most **8,192 float keys per segment**.
  It also requires non-deterministic, unordered, unspecified-tie-break output
  ([v3.5.0](https://github.com/NVIDIA/cccl/releases/tag/v3.5.0)). A 151,669-token vocabulary does not fit.
- **Key point:** "deterministic" in CCCL means *the same answer every run*. It does **not** mean *the same answer as
  llama.cpp's CPU loop*. bankML's oracle needs the second.

### 1.6 Version and state, 2025–2026

| release | date | relevant change |
|---|---|---|
| 2.8.0 | 2025-03-03 | last release supporting C++11/14 |
| 3.0.0 | 2025-07-16 | C++17 required; CTK 13.0 header layout |
| 3.1.0 | 2025-10-14 | deterministic-reduction levels; `par_nosync` async allocations |
| 3.2.0 | 2026-02-05 | `cub::DeviceTopK`; modern CUDA C++ runtime APIs |
| 3.4.0 | 2026-07-16 | faster `DeviceScan` on Blackwell; Thrust iterators → `cuda::`; `cuda::std` parallel algorithms |
| **3.5.0** | **2026-10-07** | public CUB tuning; reproducible float scans; `DeviceBatchedTopK`; device-resident problem sizes |

Dates are from the [GitHub releases API](https://github.com/NVIDIA/cccl/releases). Support policy: CCCL 3.x supports
**CTK 12.x and 13.x**. CCCL is backward-compatible with older CTKs but never forward-compatible. C++17 and C++20 are
supported. Architectures follow the current CTK (CTK 13 is `sm_75`+, so Turing/T4 and newer)
([README, Platform Support](https://github.com/NVIDIA/cccl#platform-support)). CCCL also ships Python packages
(`cuda-cccl`, python-1.2.1 on 2026-09-28) that can prototype the same algorithms without an `nvcc` build.

### 1.7 Licensing (more nuanced than one SPDX id)

The CCCL `LICENSE` file is a composite, and GitHub reports `NOASSERTION`:
- **Thrust: Apache-2.0**, "with some specific exceptions": portions are under the **Boost Software License 1.0**
  (iterator/tuple/system/random utilities) and a FreeBSD-derived licence (`thrust::complex`).
- **libcu++: Apache-2.0 WITH LLVM-exception.**
- **CUB: BSD-3-Clause.**

([LICENSE](https://github.com/NVIDIA/cccl/blob/main/LICENSE)). All of these are permissive and compatible with bankML's
`MIT OR Apache-2.0`, and with its rule that models and code be OSI-licensed. A shipped binary shim would need the
NOTICE/attribution text. The **CUDA runtime** (`libcudart`) and driver are not CCCL. They come under NVIDIA's CUDA EULA,
which allows redistributing listed runtime components (**unverified** for the exact clause; check the EULA's
redistributable-files attachment before shipping `cudart_static` inside a release artefact).

### 1.8 Requirements

- A CUDA C++ compiler: `nvcc`, or `nvc++`/clang-CUDA for some uses. Thrust is header-only, so code that uses it must
  be *compiled* as CUDA C++. Rust cannot call Thrust templates directly; it needs a C ABI shim built with `nvcc`.
- An NVIDIA GPU and driver at run time (`libcuda.so`). CTK 13 needs `sm_75`+.
- No browser path exists.

---

## 2. Where Thrust helps an LLM inference runtime, and where it does not

### 2.1 Where the time goes in decode (bankML's own numbers)

From `docs/PERFORMANCE.md` (laptop, Ryzen 3 3200U, 2 cores):
- Ternary-Bonsai-8B matmuls: ~0.23 s/token at 3 threads. The output layer alone (151,669 × 4096) is **51 ms**.
- Memory floor: ~17 GB/s → 0.06 s/token for 1-bit (1.06 GB of weights per token), 0.12–0.13 s for ternary (2.13 GB).
- The 1-bit 8B decodes at ~2 tok/s (0.5 s/token). Bonsai-1.7B decodes at ~8.5 tok/s (~120 ms/token).
- Grammar-constrained decoding: a whole-vocabulary mask costs a median of 24 ms (3.9 ms per token amortized).

**Local measurement for this report** (estimate quality: a standalone Rust proxy of the sampler's hot part, `rustc -O`,
same laptop, load average ~2, 151,669 random logits, three runs):

| operation over the full vocabulary | time |
|---|---|
| build candidates + top-40 heap selection (≈ `cands` + `partial_sort`) | **0.42–0.50 ms** |
| full softmax (max, `exp`, sum) | 0.74–0.87 ms |
| full sort of (logit, id) pairs (what happens if top-k is disabled) | 5.3–6.3 ms |

Conclusion: the sampling tail is **~0.1–0.8 % of a CPU decode step**. GPU sorting cannot speed bankML up on the CPU,
and copying logits to a GPU just to sort them would cost more than it saves.

### 2.2 Map of inference operations to tools

| operation | Thrust/CUB fit? | right tool |
|---|---|---|
| packed 1-bit / ternary mat-vec and mat-mat (`q1_0::mat_vec`, `q2_0::mat_vec`, prefill tiles, output layer) | **No** | custom kernels: popcount/sign-select or `dp4a`/LUT style. Prior art: llama.cpp's own `ggml-cuda` has `Q1_0`/`Q2_0` `mul_mat_vec_q` at b11192 (`mmvq.cu`, against **q8_1** activations, so bits differ from the CPU path); Microsoft BitNet's W2A8 CUDA GEMV ([gpu/README](https://raw.githubusercontent.com/microsoft/BitNet/main/gpu/README.md)); T-MAC (LUT, arXiv 2407.00088); CUTLASS/cuBLAS only after dequantizing, which defeats the format |
| q8_0 activation quantization (amax per 32, round half-even) | partial | a fused custom kernel; `cub::BlockReduce` for the amax inside it |
| RMSNorm (4096-element row, double accumulator in order) | **No** as a device-wide call (a ~5–10 µs launch for <1 µs of work); yes as `cub::BlockReduce` inside a fused kernel | custom fused kernel |
| RoPE / YaRN, SwiGLU, residual adds | trivially `thrust::transform`, but they should be fused | custom fused kernel |
| attention (f16/q8_0 KV, three ggml kernels, split-KV, Hadamard rotation) | **No** | custom (FlashAttention-style) kernel |
| embedding lookup | `thrust::gather` works, but it is one row copy | `cudaMemcpyAsync` |
| top-k over 151,669 logits | **Yes**, with caveats (§3.5) | `cub::DeviceRadixSort::SortPairsDescending`, or `DeviceTopK` if exactness were not needed |
| softmax / top-p / min-p over ≤128 candidates | too small for device-wide calls | one warp/block, or the CPU after copying k candidates back |
| greedy argmax | **Yes**: `thrust::max_element` / `cub::DeviceReduce::ArgMax` | the same |
| repetition / frequency / presence penalties, DRY | sparse scatter of ≤ `repeat_last_n` entries | tiny kernel or the CPU |
| grammar masks (GBNF trie walk) | **No** (branchy, pointer-chasing) | stays on the CPU |
| batched sampling across B sequences | **Yes**: `DeviceSegmentedSort` / segmented radix sort | CUB |
| tokenizer (BPE), chat template, sha256 pin | **No** | CPU |

Prior art that confirms the split: llama.cpp b11192 uses **CUB** (not Thrust) for `argsort` and for its `GGML_OP_TOP_K`,
the latter through `cub::DeviceTopK::MaxPairs` with `not_guaranteed`/`unsorted`
([`ggml-cuda/top-k.cu`](https://github.com/ggml-org/llama.cpp/blob/b11192/ggml/src/ggml-cuda/top-k.cu)). It also has
on-device "backend samplers" (`tests/test-backend-sampler.cpp`). FlashInfer avoids sorting entirely, using
rejection-based sampling in a single kernel
([FlashInfer docs](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_logits.html),
[NVIDIA blog](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/)).
Neither of these reproduces llama.cpp's *CPU* sampler bit for bit, which is what bankML's oracle checks.

### 2.3 Batch size 1 vs batched serving (estimates)

- **Batch 1 decode is bandwidth- or latency-bound.** Bytes per token: 1.06 GB (1-bit 8B) / 2.13 GB (ternary 8B).
  Ideal floors at nominal HBM/GDDR bandwidth (**estimate, ignores kernel efficiency**):

  | card (nominal BW) | 1-bit 8B floor | ternary 8B floor |
  |---|---|---|
  | T4 (320 GB/s) | 3.3 ms (≤ ~300 tok/s) | 6.7 ms |
  | L4 (300 GB/s) | 3.5 ms | 7.1 ms |
  | A10G (600 GB/s) | 1.8 ms | 3.6 ms |
  | A100-80GB (~2.0 TB/s) | 0.53 ms | 1.07 ms |
  | H100 SXM (3.35 TB/s) | 0.32 ms | 0.64 ms |

  At these floors, per-token overheads dominate. Qwen3-8B has 36 layers, which means ~300–450 kernel launches per token
  at a few µs each (~1–2 ms, **estimate**). CUDA Graphs or a persistent kernel are needed long before Thrust matters.
- **Where the sampling tail starts to matter:** suppose matmuls run on the GPU at ~3–5 ms/token. Then a
  device→host copy of 607 KB of f32 logits (~25–50 µs over PCIe 4, **estimate**), a sync, and a 0.5–1 ms CPU top-k plus
  softmax add up to **~10–20 %** of the step. A GPU top-k that returns only 40 (id, logit) pairs (480 bytes) removes
  most of it.
- **Batched serving (B sequences):** the weights are reused B times, so the matmul becomes compute-bound and the CPU
  sampling cost scales as B × 0.5–6 ms. Segmented GPU sort or top-k then matters directly. bankML is a deliberate
  single-slot engine today (`-np 1` contract, `native.rs`), so this is future work.

---

## 3. bankML: a realistic integration path

### 3.1 What bankML is, as it bears on this

- Zero crates. Own GGUF guard, sha256 pin, AVX2 + wasm-SIMD kernels for `Q1_0` (type 41, 128-weight blocks, 18 bytes)
  and `Q2_0_g64` (type 42). These are bit-exact against **the compiled** llama.cpp b11192 haswell binaries, including
  FMA contractions read from disassembly (`q1_0.rs`, `q2_0.rs`, `docs/oracles.md`).
- The sampler (`bankML/sampler.rs`) ports llama-server's chain: penalties → DRY → top-n-σ → **top-k by a libstdc++
  `partial_sort` port** (heap order, ties included) → typical → top-p (softmax in list order, a float running sum) →
  min-p → XTC → temperature (greedy at temp ≤ 0 = first strict max in post-sort order) → `dist` (`expf`, a **double**
  sum, mt19937 `uniform_real_distribution<double>`). `top_k` must be in 1–128 ("llama.cpp sorts larger sets another
  way").
- RMSNorm (`forward.rs::rms_norm_mul`) sums squares **in double, sequentially**, then `1/sqrtf`, then two float
  multiplies with no FMA.
- A GPU component already exists: **Vulkan through `dlopen`**, bankML's own SPIR-V assembler, `Q1_0` kernels verified
  bit-exact on-card before use (`fma_exact` because RADV does not fuse), and a calibrated row share between card and CPU
  (`bankML/gpu/`, `docs/modules/gpu.md`). The doc states: "A new backend (CUDA, ROCm, Metal, …) is a new module and one
  line in `BACKENDS`". `docs/TODO.md` (operator direction, 2026-10-05) wants AMD **and NVIDIA** support, including Hugging
  Face's rented T4/L4. It notes that a rented NVIDIA card "needs no CUDA toolkit" through Vulkan.

**Implication:** for NVIDIA, the cheapest route to GPU speed is the **existing Vulkan backend** (NVIDIA's driver ships
Vulkan), not CUDA. Thrust only enters if a CUDA backend is added, and then only for the sampling tail.

### 3.2 FFI options from Rust

| option | crates added | build needs | fit with "zero-dependency" |
|---|---|---|---|
| **A. `dlopen` an optional `libbankml_cuda.so`** built separately with `nvcc` (C ABI: `bk_cuda_open`, `bk_topk`, `bk_argmax`, …), statically linking `cudart_static` + CCCL headers | **none** (bankML already uses `dlopen` in `gpu/vulkan.rs` and `sys.rs`) | `nvcc` only to build the shim; core `cargo build` unchanged | **preserved**: the core stays crate-free and builds without CUDA; a missing `.so` simply means no CUDA device |
| B. `build.rs` invokes `nvcc` behind `--features cuda` | none (or `cc`) | CTK at build time when the feature is on | weakened: the default build is clean, feature builds need the CTK |
| C. [`cudarc`](https://github.com/chelsea0x3b/cudarc) (MIT/Apache; dynamic loading of driver/NVRTC/cuBLAS by default, CUDA 11.4–13.x) | `cudarc` | none at build time | **breaks** zero-crates. Also, NVRTC cannot compile Thrust host-side templates, so it covers custom kernels only, not Thrust |
| D. Rust-CUDA `cust` (project rebooted 2025; README still "early development") ([Phoronix](https://www.phoronix.com/news/Rust-CUDA-Project-Reboot)) | `cust` | CTK + LLVM toolchain | breaks zero-crates; immature |

**Recommendation: option A.** It mirrors the Vulkan backend's design (nothing linked at build time, run-time discovery,
on-device verification before use), keeps `cargo build` and the release gate's CPU path unchanged, and confines CCCL to
a separately licensed and separately built artefact. The C API work (`capi/`, `bankml.h`) shows the project is already
comfortable with a C ABI boundary.

### 3.3 Costs of the integration

- **Zero-dependency claim:** with option A it holds for the Rust crate, but the *shim* depends on CTK, CCCL and the
  NVIDIA driver. The README would need to say "optional CUDA module, built separately".
- **Toolchain surface:** CCCL 3.x needs C++17 and CTK 12/13, and gives no forward compatibility. The shim pins a CCCL
  version the same way bankML pins llama.cpp b11192, with its own gate record.
- **Bit-exactness hazards specific to CUDA** (each one would surface as an oracle failure):
  1. `nvcc` **contracts mul+add into FMA by default** (`--fmad=true`). Explicit `__fmaf_rn` is unaffected. Compile the
     shim with `--fmad=false` and write ggml's FMAs explicitly. This is the same lesson as RADV and WGSL
     ([NVIDIA forum summary of `--fmad`](https://forums.developer.nvidia.com/t/why-does-device-give-wrong-answer-to-simple-math/23391);
     the authoritative wording is in the NVCC options reference and the CUDA C++ Programming Guide's floating-point
     section, **not re-read here**).
  2. CUDA's `expf` is not glibc's (the Programming Guide lists a non-zero ULP bound for single-precision `expf`). Any
     GPU softmax that must match must use a software port of the same `expf` that bankML's oracle matched (libm `expf`
     in the sampler, ggml's `v_expf` in the forward pass). Never `__expf` or fast-math.
  3. Reduction order: CUB's run-to-run determinism is not llama.cpp's sequential order. For sums that must match
     (RMSNorm's double sum, `dist`'s double sum, top-p's float running sum), run them sequentially over a small list,
     or prove exactness. A double sum of float squares is order-independent only when the exponent spread keeps every
     partial sum exact. That is not guaranteed for activations.
  4. Sort tie order: radix sort is stable on (key) but its tie order is *index order*. libstdc++'s heap-based
     `partial_sort` leaves ties in a layout-dependent order. They agree only when there are no ties.
  5. llama.cpp's own CUDA kernels quantize activations to **q8_1**, not q8_0. Matching "llama.cpp with CUDA" and
     matching "llama.cpp b11192 CPU" (bankML's oracle) are different targets. The second is what the gate checks.
- **Release gate:** the gate must run unchanged on CPU-only machines. The CUDA module gets its own `--verify`-style
  on-device oracle and its own optional gate stage, run only where a card is present.

### 3.4 Which hot paths map to what

| bankML path (file) | CPU today | CUDA target | Thrust/CUB? |
|---|---|---|---|
| `q1_0::mat_vec(_par)`, `mat_mul_act(_par)` | AVX2, bit-exact | custom kernel (port of the Vulkan/WGSL kernel; `__fmaf_rn` where ggml fuses) | no |
| `q2_0::mat_vec(_par)`, `mat_mul(_par)` | AVX2 `maddubs/madd`, ggml float order | custom kernel (`dp4a`-friendly; integer sums are order-free, float ops in ggml order) | no |
| output layer 151,669 × 4096 (51 ms ternary CPU) | same kernels | same kernels; **keep the logits on the device** | no |
| `quantize_row_q8_0` | AVX2 | fused kernel; `cub::WarpReduce` max | CUB block/warp only |
| `rms_norm_mul` | sequential double | one block per row, sequential double inside one thread, or a proven-exact scheme | no |
| `Rope::apply`, `swiglu` (`v_expf`) | scalar/AVX2 | fused kernels with the same `v_expf` | no |
| attention (`attend_head*`, split-KV, q8 KV, `fwht`) | three ggml kernels | custom; it must reproduce whichever ggml kernel llama.cpp would pick | no |
| `sampler.rs` top-k / greedy | `partial_sort` port | **`cub::DeviceRadixSort::SortPairsDescending`** (or segmented, for batches) + uniqueness check | **yes** |
| top-p / min-p / XTC / temp / `dist` on ≤128 candidates | sequential | CPU after copying k pairs back (cheap, exact) | no |
| penalties / DRY | sequential | CPU computes the adjusted values; scatter into device logits | `thrust::scatter` optional |
| `grammar.rs` masks | trie walk | CPU; upload a bitmask, apply with `transform` | marginal |

### 3.5 An exactness-preserving GPU top-k (the one place Thrust/CUB pays)

**Claim:** if the top-k candidates have pairwise-distinct logits, *and* the k-th logit is strictly greater than the
(k+1)-th, then the top-k set and its descending order are unique. Every correct sort or selection gives the same list,
so it equals the `partial_sort` port's output. Proof sketch: `partial_sort` produces the k smallest under `comp` in
sorted order, and sorted order is unique when no two of those keys tie. This is an argument, not yet an oracle
result. The oracle must confirm it.

Procedure per token (logits already on the device because the matmuls run there):
1. Apply penalties/DRY adjustments as a sparse scatter. The values are computed on the CPU with the same float ops,
   unchanged.
2. `cub::DeviceRadixSort::SortPairsDescending(logits, ids)`. A full sort costs ~0.1–0.3 ms on a T4-class card for
   151,669 pairs (**estimate**, unbenchmarked). Alternatively, a cheaper two-pass "find the k-th value, then compact
   with `copy_if`" gives the candidate set.
3. Copy back the first k+1 pairs (≤ 1.5 KB) and check the uniqueness condition on the host.
4. If unique, hand the k candidates to the existing CPU chain (typical/top-p/min-p/XTC/temp/`dist`). Every float op
   then stays the CPU's. If not unique (ties among the top k+1), copy the full logits and run the CPU path. This is
   rare with real logits but must be counted. Grammar-constrained redraws keep using the CPU path.

Deterministic, exact, and the only GPU→host traffic in the common case is ~1.5 KB.

### 3.6 Expected gains (order of magnitude, estimates)

| scenario | where the sampling tail sits | gain from Thrust/CUB |
|---|---|---|
| CPU decode (today, laptop): 120–500 ms/token | 0.5–1 ms on the CPU (<1 %) | **none**; GPU sort would add copies |
| GPU matmuls via Vulkan or CUDA, batch 1, 3–7 ms/token | logits D2H + CPU top-k ≈ 0.5–1 ms (10–20 %) | **~1.1–1.2× tokens/s** from §3.5 |
| batched serving, B = 8–64 | B × CPU top-k (4–64 ms) | large, roughly proportional to B. Needs a multi-slot engine first |
| prefill | sampling happens once per request | none |

The big multiplier is moving the matmuls to the GPU (bandwidth floors in §2.3 are 100–1000× below today's CPU time).
That is custom-kernel work, which bankML is already doing in Vulkan/WGSL.

### 3.7 Keeping the release gate green

- The CPU path stays the default, compiled and gated exactly as now. The CUDA module is opt-in at run time (a
  `BANKML_GPU=…` index, or `BANKML_CUDA=off`) and absent from `cargo build`.
- Add `bankml gpu --verify` checks for the CUDA module. The checks are: kernels on layer-shaped data (the RADV lesson),
  top-k vs `sampler::partial_sort` on recorded logits *including adversarial ties*, and the fallback rate.
- Run `sample_oracle.py` / `greedy_oracle.py` / `serve_oracle.py` with the CUDA module on. The answers must be
  token-identical, as the Vulkan worker's are.
- Compile the shim with `--fmad=false -prec-div=true -prec-sqrt=true -ftz=false` and no `--use_fast_math`. The gate
  checks the flags from the build record.
- Record the CCCL version, CTK version, driver and GPU in `testing/results/<v>.txt`.

---

## 4. browseML: why Thrust cannot apply, and what does

### 4.1 What browseML actually is (from the local repository)

browseML is **bankML compiled to WebAssembly** (`wasm32-wasip1` and `-threads`, three builds: SIMD, threaded,
threaded relaxed-SIMD). It runs Bonsai-1.7B `Q1_0` (248 MB, sha256-pinned) in the visitor's browser with bankML's own
forward pass, sampler and receipts, identical to llama-server b11192 on 9 of 9 recorded conversations
(`docs/technical.md`). It has a **WebGPU** path: bankML's own WGSL `Q1_0` kernel (`bankML/gpu/wgsl.rs`), eight lanes per
row, TwoSum + round-to-odd in place of an exact FMA, and a `hide` OR-with-runtime-zero against Mesa's simplifier. The
card is verified on-card before use and gets a calibrated share of each 1-bit matrix's rows through a GPU worker and
`Atomics`. Measured on the 2-core laptop: ~7 tok/s reading, ~3 tok/s writing; writing is arithmetic-bound
(`vec_dot_act` 55 % of the profile). Sibling repositories: `browseML-webgpu` (kernels + an on-your-GPU exactness page),
`browseML-embeddings` (BM25/RRF in JS and wasm), `browseML-models`.

### 4.2 Why there is no Thrust in a browser

Thrust's device backend is CUDA C++. It compiles to PTX/SASS and runs through NVIDIA's native driver. A web page can
reach the GPU only through **WebGPU** (WGSL shaders) or WebGL, and can run native code only as WebAssembly. There is no
CUDA in the browser sandbox, and Thrust's OMP/TBB backends have no wasm target either. (In principle `thrust::cpp`
compiled to wasm is plain sequential C++, which gives nothing over Rust.)

### 4.3 The browser equivalents

| primitive | browser technique | sources |
|---|---|---|
| reduce (max, sum, argmax) | workgroup tree reduction in WGSL; **subgroups** (`subgroupAdd`/`subgroupMax`) shipped in Chrome 134, but query `subgroupMinSize/MaxSize` | [Chrome "New in WebGPU 134"](https://developer.chrome.com/blog/new-in-webgpu-134) |
| prefix scan | reduce-then-scan (portable). Single-pass decoupled look-back (Merrill & Garland 2016, CUB's scan) is **unsafe** without forward-progress guarantees, which WebGPU lacks. **Decoupled Fallback** (Smith, Levien, Owens, SPAA 2025) makes single-pass portable; near-memcpy speed on Apple M1/M3 via Dawn | [Merrill & Garland, NVR-2016-002](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf) · [Decoupled Fallback](https://escholarship.org/uc/item/0bk9z4bt) · [GPUPrefixSums](https://github.com/b0nes164/GPUPrefixSums) (CUDA, D3D12, Unity, WGPU) |
| radix sort | portable LSD radix sort. OneSweep (Adinets & Merrill, arXiv 2206.01784) assumes 32-wide subgroups and is hard to port | [Linebender GPU sorting notes](https://linebender.org/wiki/gpu/sorting/) · [wgpu_sort](https://github.com/KeKsBoTer/wgpu_sort) · [wgpu-primitives](https://github.com/samjsui/wgpu-primitives) · PlayCanvas compute radix sort (OneSweep backend NVIDIA-only) |
| top-k | for k ≤ 128: per-workgroup partial top-k, then a merge, or a k-th-value threshold + compaction | no standard WGSL library found (**unverified** gap) |
| graph-level ML | **WebNN**: W3C Candidate Recommendation Draft (2026 snapshots on w3.org/TR, latest seen dated 2026-09-10). Gives no control over float order, so it **cannot** carry an oracle-gated bit-exact engine. Browser shipping status **unverified** | [WebNN](https://www.w3.org/TR/webnn/) |
| CPU | wasm SIMD128, relaxed SIMD (only where `relaxed_madd` is fused), threads via cross-origin isolation: **already done** in browseML | `docs/technical.md` |
| reference engines | llama.cpp `ggml-webgpu` (MIT; no Q1_0 yet per browseML's catalogue), web-llm/TVM and onnxruntime-web (fast, not bit-exact) | `docs/gpu-software.md` |

**Browser-specific exactness hazards beyond CUDA's:** WGSL has **no `f64`**, so RMSNorm's double sum and `dist`'s double
sum need double-float emulation or must stay on the CPU. WGSL has no exact FMA (browseML already emulates it), no
NoContraction, and no forward-progress guarantee.

**Should browseML do GPU sampling?** Not now. At ~3 tok/s (~330 ms/token), CPU sampling in wasm is ~1 ms (**estimate**,
scaled from the native measurement). It would matter only if the output layer ran entirely on the card, so that the
logits never left the GPU. Even then the §3.5 design (sort on the GPU, read back k+1 pairs, uniqueness check, CPU tail)
carries over directly to WGSL, as a portable radix sort or a threshold + compaction. Higher value for browseML is in its
own TODO: subgroup reductions in the matvec kernel, moving more of each layer (f16 attention, RoPE, norms) to WGSL, and
measuring on discrete AMD/NVIDIA cards.

### 4.4 Where a native Thrust-accelerated backend can serve browseML clients

The plumbing exists. `bankml serve --allow-origin ORIGIN` lets a page (the HF Space) talk to a `bankml serve` on the
visitor's machine or a host, and browseML's host registry (`host.rs`) already names `bankml serve` a *server host*. A GPU
machine running `bankml serve` with GPU matmuls (Vulkan, or a future CUDA module with the §3.5 sampler) could answer
browseML clients at tens to hundreds of tok/s for the 8B models the browser cannot hold (**estimate**). Each answer
would still carry a receipt that pins the model by sha256.

Trade-off: browseML's promise is "nothing you type leaves your device". A remote host must therefore be an explicit,
labelled opt-in, as the bankML Space already labels its HF-provider mode "not bankML". The host would also need
multi-slot serving to be worth sharing, which is where batched GPU sampling (segmented sort) earns its keep.

---

## 5. Ranked recommendations

| # | recommendation | benefit | effort | main risks | minimal first experiment |
|---|---|---|---|---|---|
| 1 | **Get the matmuls onto NVIDIA through the existing Vulkan backend first** (Q2_0 and F16 kernels, batched submissions; already on `docs/TODO.md`). No Thrust. | **High**: the 100×+ headroom is here | medium–high (already planned) | NVIDIA Vulkan driver FMA/contraction behaviour; per-submit latency (already the APU bottleneck) | With the owner's go-ahead (TODO requires it), rent one T4/L4, run `bankml gpu --verify` and the 1-bit decode A/B. Record tok/s and verify bits |
| 2 | **Optional CUDA shim, dlopen'd, for the sampling tail** (§3.5: CUB radix sort of (logit, id) + uniqueness check + CPU tail), used only when logits are already on the device | low–medium at batch 1 (~1.1–1.2×, estimate); none on CPU-only | medium (shim + C ABI + on-device oracle + gate stage) | tie fallback rate; CCCL/CTK churn; "zero-dependency" wording; cudart redistribution terms | **Standalone, outside the repo:** a 100-line `.cu` that sorts recorded logits from `sample_oracle` runs with `cub::DeviceRadixSort::SortPairsDescending`, checks uniqueness, compares the top-k list with `sampler::partial_sort` output, and times it vs the CPU (0.42–0.50 ms baseline above) |
| 3 | **Measure before moving anything:** per-token breakdown (matmul / attention / norms / sampler / grammar) with `/bankml/metrics` spans | high information, low cost | low | none | One instrumented 8B decode run, pinned (`testing/pinned.sh`). It confirms §2.1's <1 % sampler share on real logits |
| 4 | **A CUDA matmul backend** (custom Q1_0/Q2_0 kernels, `__fmaf_rn`, `--fmad=false`), only if #1 shows Vulkan on NVIDIA falls short | high on NVIDIA | high | double maintenance with Vulkan; exactness vs CPU ggml (not ggml-cuda's q8_1 path) | Port `q1_0` mat-vec to CUDA in the standalone shim. Compare bits on layer-shaped data and the time vs the Vulkan kernel on the same card |
| 5 | **Batched serving** (multi-slot engine) with segmented GPU top-k | high for a shared host serving many browseML clients | high (engine change; llama-server's multi-slot contract must be reproduced to stay oracle-checked) | breaks the `-np 1` exactness contract until re-proved | Design note only. Prototype with CCCL Python (`cuda-cccl`) segmented sort on B=32 rows of recorded logits |
| 6 | **browseML: no Thrust.** Keep GPU work on matvec kernels (subgroups). Defer GPU sampling until the whole output layer is on the card | low now | low (deferral) | f64 absence, no forward progress | When it comes: a WGSL threshold-and-compact top-k on recorded logits in `browseML-webgpu`'s exactness page, with the same uniqueness rule |
| 7 | **Remote GPU host for browseML** (opt-in, labelled, receipts) | medium (8B in the browser UI) | low–medium (plumbing exists) | the privacy promise; host cost | Point the browseML app at a `bankml serve --allow-origin …` on a GPU host. Check that receipts verify in-page |

**Not recommended:** Thrust's OMP/TBB backends for bankML's CPU (they would replace a verified pool with an unverified
one), `cub::DeviceTopK`/`DeviceBatchedTopK` for the oracle-gated path (non-deterministic by design; the batched variant
caps at 8,192 keys), any `--use_fast_math`, and any GPU `expf` that is not a port of the matched one.

---

## 6. Unverified or open points

- Exact CUDA EULA redistribution terms for `cudart_static` in a shipped `.so`.
- GPU radix-sort time for 151,669 pairs on T4/L4 (the 0.1–0.3 ms figure is an estimate).
- The rate of ties in the top k+1 on real Bonsai logits (it decides the §3.5 fallback cost; measurable from
  `sample_oracle` records).
- WebNN shipping status in Chrome/Edge as of October 2026.
- Whether a maintained WGSL top-k library exists. None was found.
- bankML's real sampler cost on real logits (§2.1 used a proxy on random logits).

## 7. Sources

- NVIDIA CCCL: [repository/README](https://github.com/NVIDIA/cccl) · [LICENSE](https://github.com/NVIDIA/cccl/blob/main/LICENSE) ·
  [releases](https://github.com/NVIDIA/cccl/releases) ([3.0.0](https://github.com/NVIDIA/cccl/releases/tag/v3.0.0),
  [3.1.0](https://github.com/NVIDIA/cccl/releases/tag/v3.1.0), [3.2.0](https://github.com/NVIDIA/cccl/releases/tag/v3.2.0),
  [3.4.0](https://github.com/NVIDIA/cccl/releases/tag/v3.4.0), [3.5.0](https://github.com/NVIDIA/cccl/releases/tag/v3.5.0)) ·
  [docs](https://nvidia.github.io/cccl) · [Thrust systems](https://github.com/NVIDIA/cccl/blob/main/docs/thrust/developer/systems.rst) ·
  [CUDA execution policy](https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/system/cuda/detail/execution_policy.h) ·
  [Thrust CUDA sort → CUB](https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/system/cuda/detail/sort.h) ·
  [`device_topk.cuh`](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_topk.cuh) ·
  [Thrust 1.16 changelog (par_nosync)](https://github.com/NVIDIA/thrust/blob/1.16.X/CHANGELOG.md) · [archived NVIDIA/thrust](https://github.com/NVIDIA/thrust)
- Merrill & Garland, *Single-pass Parallel Prefix Scan with Decoupled Look-back*, NVIDIA NVR-2016-002:
  [PDF](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)
- Smith, Levien, Owens, *Decoupled Fallback: A Portable Single-Pass GPU Scan*, SPAA 2025: [eScholarship](https://escholarship.org/uc/item/0bk9z4bt)
  · [GPUPrefixSums](https://github.com/b0nes164/GPUPrefixSums)
- Adinets & Merrill, *Onesweep*, arXiv [2206.01784](https://arxiv.org/abs/2206.01784) · [Linebender GPU sorting](https://linebender.org/wiki/gpu/sorting/)
  · [wgpu_sort](https://github.com/KeKsBoTer/wgpu_sort) · [wgpu-primitives](https://github.com/samjsui/wgpu-primitives)
- llama.cpp b11192: [`ggml-cuda/top-k.cu`](https://github.com/ggml-org/llama.cpp/blob/b11192/ggml/src/ggml-cuda/top-k.cu),
  [`ggml-cuda/argsort.cu`](https://github.com/ggml-org/llama.cpp/blob/b11192/ggml/src/ggml-cuda/argsort.cu),
  [`ggml-cuda/mmvq.cu`](https://github.com/ggml-org/llama.cpp/blob/b11192/ggml/src/ggml-cuda/mmvq.cu)
- FlashInfer sampling: [API docs](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_logits.html) ·
  [NVIDIA blog](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/)
- Microsoft BitNet GPU W2A8 kernel: [gpu/README](https://raw.githubusercontent.com/microsoft/BitNet/main/gpu/README.md)
- Rust FFI: [cudarc](https://github.com/chelsea0x3b/cudarc) / [docs.rs](https://docs.rs/cudarc) · [Rust-CUDA reboot (Phoronix)](https://www.phoronix.com/news/Rust-CUDA-Project-Reboot)
- `--fmad` behaviour: [NVIDIA developer forum](https://forums.developer.nvidia.com/t/why-does-device-give-wrong-answer-to-simple-math/23391)
- rocThrust policies: [ROCm docs](https://rocm.docs.amd.com/projects/rocThrust/en/latest/how-to/rocThrust-hip-policies.html)
- WebGPU subgroups: [Chrome 134](https://developer.chrome.com/blog/new-in-webgpu-134) · WebNN: [W3C TR](https://www.w3.org/TR/webnn/)
- In-house (local): `~/bML/bankml/bankML/{sampler,forward,q1_0,q2_0,par,native}.rs`, `bankML/gpu/`, `docs/PERFORMANCE.md`,
  `docs/modules/gpu.md`, `docs/TODO.md`; `~/cryptoAGI/browseML/README.md`, `docs/technical.md`, `docs/gpu-software.md`;
  `~/cryptoAGI/browseML-webgpu/README.md`. The proxy benchmark was a standalone Rust file on random logits, not kept in this repository.

---

## Executive summary

1. Thrust is NVIDIA's CUDA C++ parallel-algorithms library (sort, scan, reduce, compaction). It now ships in CCCL with CUB and libcu++. The current release is v3.5.0 (2026-10-07), and it needs nvcc and an NVIDIA GPU. Thrust is Apache-2.0 (with Boost-licensed parts), libcu++ is Apache-2.0 with the LLVM exception, and CUB is BSD-3.
2. Thrust is not a matmul library. Packed 1-bit and ternary matmuls, which take nearly all of bankML's decode time, need custom kernels. bankML already writes those for the CPU, Vulkan and WGSL.
3. Where Thrust/CUB fit is the sampling tail: top-k over the 151,669-token vocabulary, argmax, and segmented sorts for batches.
4. On today's CPU path that tail costs about 0.5–1 ms per token, against 120–500 ms for the whole token (measured here with a proxy). That is under 1 %, so Thrust gives bankML no gain until the matmuls run on a GPU.
5. Once the matmuls are on a GPU, copying the logits back plus a CPU sort could be 10–20 % of each token (estimate). A GPU top-k that returns about 40 pairs would win back roughly 1.1–1.2× at batch 1, and more with batched serving.
6. Bit-exactness is the hard constraint. CUB's top-k is non-deterministic by design, GPU `expf` differs from glibc's, nvcc contracts mul+add into FMA by default, and sort tie order differs from libstdc++'s.
7. An exact design: sort (logit, id) on the GPU, read back k+1 pairs, use them only if the top-k order is unique, and otherwise fall back to the CPU. Everything else stays on the CPU.
8. Integration path: a separately built CUDA shim that bankML dlopens, like its Vulkan backend. That keeps the crate dependency-free and leaves the CPU release gate untouched.
9. The highest-value step for NVIDIA is to run bankML's existing Vulkan kernels on a rented T4/L4 first. That needs no CUDA toolkit.
10. browseML cannot use Thrust, because the browser has no CUDA. Its equivalents are WGSL reduce, scan and radix sort, with portable single-pass scan from SPAA 2025 and subgroups since Chrome 134. At browseML's ~3 tok/s, none of them is on the critical path.
11. A GPU host running `bankml serve` could serve browseML clients the 8B models, but only as a labelled opt-in, since browseML promises that nothing typed leaves the device.
