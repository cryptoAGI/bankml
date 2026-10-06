# `bankML/forward.rs` — the forward pass, bit-exact against llama.cpp b11192's graphs

## Summary

`forward.rs` turns token ids into logits. It reproduces, operation by operation and in the same float order, the
graph llama.cpp b11192 builds for two architectures: Qwen3 (`llm_build_qwen3`: per-head RMS norms on Q and K, RoPE in
NEOX mode, YaRN when the header asks) and Llama (`llm_build_llama`: no Q/K norms, RoPE in NORM mode on adjacent
pairs, GQA). The weights may be Q1_0 (1-bit), Q2_0_g64 (ternary) or F16, each through its own bit-exact kernel
(`q1_0.rs`, `q2_0.rs`, `f16.rs`). Tied embeddings are supported: with no `output.weight`, the logits read
`token_embd.weight`, as llama.cpp's `TENSOR_DUPLICATED` does.

The oracle is the shipped ggml computing the same graph, not the C source. The comments record why: GCC contracts
several expressions into FMAs in `libggml-cpu-haswell.so`, and the bits follow the binary. Those expressions are
written here as the `mul_add`s the disassembly shows.

Callers: the native engine (`native.rs`, which owns a `Weights` and the K/V caches of its one slot; the host prompt
cache, `prompt_cache.rs`, holds copies of those caches),
and through it `bankml serve --native` (`/v1/chat/completions`, `/api/chat`, `/api/generate`) and the C API;
`bankml generate` (`main.rs`) directly; `native.rs` also calls `plan` to check a model before it is loaded.

## Technical usage

### Deciding what a file is: `plan`

```rust
pub fn plan(h: &Header) -> Result<Plan, String>
pub struct Plan { pub arch: Arch, pub wtype: u32, pub output: &'static str }
pub enum Arch { Qwen3, Llama }
```

`plan` reads the GGUF header only, before any weight. It returns the architecture, the one weight type every matrix
must share (`TYPE_Q1_0` = 41, `TYPE_Q2_0` = 42, `TYPE_F16` = 1), and the matrix the logits come from. Anything
outside the reproduced graph is an `Err` that names the tensor or key and why (see Limitations).

### Weights and caches

```rust
pub fn Weights::open(path: &Path) -> Result<Self, String>
pub fn caches(&self) -> Vec<KvCache>                       // f16
pub fn caches_of(&self, kind: KvType) -> Vec<KvCache>      // 0.3.9: KvType::F16 or KvType::Q8_0
pub fn prefill(&self, caches: &mut [KvCache], tokens: &[u32], each: impl FnMut(usize, usize, &[f32])) -> Result<Vec<f32>, String>
pub fn prefill_with(&self, caches: &mut [KvCache], tokens: &[u32], outputs: Outputs, each: impl FnMut(usize, usize, &[f32])) -> Result<Vec<Vec<f32>>, String>
pub fn decode(&self, caches: &mut [KvCache], token: u32) -> Result<Vec<f32>, String>
pub fn logits(&self, result_norm: &[f32]) -> Result<Vec<f32>, String>
pub fn logits_rows(&self, result_norms: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String>
```

- `Weights::open` guards the file (`gguf::guard_file`, mainline engine), runs `plan`, memory-maps the file, reads
  every F32 tensor (the norms) once, and builds a name → index map of the tensors. With Q1_0 weights it also tries
  the GPU worker (`gpu::worker::Worker::open`); a card joins only after it passes the on-card oracle.
- `caches()` gives one empty `KvCache` per layer. A `KvCache` holds K and V as f16 (`set_rows` rounding), one row of
  `n_head_kv · head_dim` per position. `truncate(n)` keeps the first `n` positions, as llama.cpp's `seq_rm` does for
  its prompt cache.
- 0.3.9: `caches_of(KvType::Q8_0)` keeps them as q8_0 blocks instead (`kq`/`vq`), as llama.cpp's `--cache-type-k/v
  q8_0`. llama.cpp b11192 rotates around a quantized cache (`attn_rot_k`/`attn_rot_v`): K and Q by a Hadamard
  transform over 128 values (the head), V over 64, the attention output back by the same 64-wide transform; ggml
  computes it as a fast Walsh–Hadamard transform (`fwht`: scale by `1/sqrtf(n)`, then `u + v`, `u − v` butterflies).
  K and V are rotated and quantized with ggml's AVX2 `quantize_row_q8_0` as they are stored. Attention over a q8_0
  cache always takes ggml's reference kernel, as ggml does whenever K is not f16 (`attend_head_q8`: Q rotated and
  quantized, scores by the AVX2 `vec_dot_q8_0_q8_0`, V dequantized into an f32 accumulator). 53 % of the f16
  cache's bytes (34 bytes per 32 values against 64). The engine reads the type from `BANKML_CACHE_TYPE`
  (`KvType::parse`: `f16` or `q8_0`, anything else refused) and refuses a q8_0 cache for a head size that is not a
  multiple of 32.

  ```rust
  pub enum KvType { F16, Q8_0 }                              // parse("f16" | "q8_0"), name()
  pub fn KvCache::of(kind: KvType, width: usize) -> KvCache  // len, is_empty, bytes, row_bytes, truncate, push
  pub fn fwht(x: &mut [f32], n: usize)                       // ggml's fast Walsh–Hadamard transform, in blocks of n
  pub fn rot_k_size(head_dim: usize) -> usize                // the K/Q rotation's block; ROT_V = 64
  pub fn vec_dot_q8_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32
  pub fn attend_head_q8(q: &[f32], k: &[u8], v: &[u8], n_kv: usize, rb: usize, scale: f32, out: &mut [f32])
  ```
- `prefill` computes a prompt in micro-batches of up to `N_UBATCH` (512) tokens, each through every layer together.
  It returns the last token's `result_norm` (`output_norm` of the last layer's output). `prefill_with(…,
  Outputs::All, …)` returns every token's, which is what the model oracle's graph computes.
- `decode` runs one generated token at the next position. `logits` multiplies `result_norm` by the output matrix.
- `step` and `step_with` run one token with a given attention kernel; `step` always uses the reference kernel.
  `decode` and `prefill` choose the kernel llama.cpp would choose.

A minimal greedy loop, as `bankml generate` writes it:

```rust
let w = Weights::open(model)?;
let mut caches = w.caches();
let mut rn = w.prefill(&mut caches, &prompt, |_, _, _| {})?;
loop {
    let l = w.logits(&rn)?;
    let next = /* argmax, or Sampler::sample(&l) */;
    if ends.contains(&next) { break; }
    rn = w.decode(&mut caches, next)?;
}
```

### The three attention kernels

llama.cpp runs one of three CPU flash-attention kernels, and each gives different bits. `kernel_for` picks the one
llama.cpp would run:

```rust
pub fn kernel_for(rows: usize, cells: usize, llama_threads: usize) -> Result<Kernel, String>
pub enum Kernel { Reference, Tiled, Split { padded: usize, nth: usize } }
pub fn padded_kv(cells: usize) -> usize   // multiples of 256, at least 256
```

| kernel | when | function | what it reproduces |
|---|---|---|---|
| `Reference` | micro-batches of 2–63 rows; one-token decodes under 512 padded cells | `attend_head`, `attend_head_partial` | `flash_attn_ext_f16_one_chunk`: Q rounded to f16, online softmax, V accumulator rounded to f16 at every step |
| `Tiled` | micro-batches of 64 rows or more | `attend_head_tiled`, `attend_heads_tiled` | `flash_attn_ext_tiled`: f32 Q, 64-cell KV tiles, a vectorized softmax summed in double, f32 accumulator |
| `Split` | a one-token decode whose padded KV length reaches 512 | `attend_head_split` | the split-KV path: one chunk per llama.cpp thread, partials reduced in chunk order |

The split-KV bits depend on llama.cpp's thread count. `Weights::llama_threads` holds it.

### Other public pieces

- `rms_norm_mul(x, w, eps, out)`: `rms_norm` then `mul`, the sum of squares in double, no FMA.
- `Rope::new(n_dims, freq_base, yarn_factor, n_ctx_orig)`, `Rope::cache`, `Rope::apply`: RoPE with the parameters
  llama-context.cpp derives (YaRN: `freq_scale = 1/factor`, `ext_factor = 1`, beta_fast 32, beta_slow 1).
- `v_expf(x)`: ggml's AVX2 `ggml_v_expf`, one lane. `swiglu(gate, up, out)`: `silu(g) · u` with that `expf`, not libm's.
- `dot_f16`, `dot_f16_ref`: `ggml_vec_dot_f16` as the AVX2 build reduces it.
- `Weights::embed`, `f32_vec`, `norm`, `matrix`, `quantize`, `quantize_rows`, `mv`, `mm`, `mv_many`, `qkv`,
  `attention`, `attention_with`, `attend`, `ffn`: the layer pieces the oracle tests call one by one.

### Environment

| variable | default | effect |
|---|---|---|
| `BANKML_LLAMA_THREADS` | 3 | the `-t` of the llama.cpp being matched (Savante runs 3); decides the split-KV chunks |
| `BANKML_THREADS` | all cores | size of the thread pool (`par::Pool::from_env`) |
| `BANKML_CACHE_TYPE` | `f16` | the K/V cache type, `f16` or `q8_0` (read by `native.rs`, 0.3.9) |
| `BANKML_GPU`, `BANKML_GPU_SHARE`, `BANKML_GPU_LIMIT` | —, measured, 0.8 | the GPU worker for 1-bit matrices and its limiter ([gpu.md](gpu.md), docs/usage.md §13) |

## How it is verified

All oracle tests are `#[ignore]`d (they need the models and recorded files) and run in `testing/release_gate.sh`.

| test | oracle | last recorded result |
|---|---|---|
| `rms_norm_of_a_constant_row` | unit test | — |
| `oracle_forward_embed_norm` | shipped ggml via `testing/forward_oracle.py` | 300 of 300 rows, `inp_embd` and `attn_norm-0` |
| `oracle_forward_qkv_rope` | same | 140 of 140 rows, positions 0–27 and 7–63,214 |
| `oracle_forward_attention` | same | 112 of 112 rows through `l_out` |
| `oracle_forward_swiglu_sweep` | `ggml_swiglu_split` | 24,600 of 24,600 values |
| `oracle_forward_attention_tiled` | same | 150 of 150 rows (the reference kernel would match 1) |
| `oracle_forward_attention_split` | same | 14 of 14 decode rows, 257–1,000 cells, 3 and 4 threads |
| `oracle_forward_model`, `_ternary` | whole graph, `testing/model_oracle.py` | 1,064 of 1,064 rows each |
| `oracle_forward_model_bonsai_1_7b` | same (tied embeddings) | 840 of 840 |
| `oracle_forward_model_llama_f16` | same (SmolLM2-135M-Instruct, mindx-gen39) | 800 of 800 each |
| `oracle_greedy_llama_server` (`_ternary`, `_long`, `_deep`) | llama-server b11192, `testing/greedy_oracle.py` | 6 of 6 (164 tokens); ternary 6 of 6 (140); long 6 of 6; deep 600 tokens |
| `oracle_sample_llama_server`, `oracle_llama_server_bonsai_1_7b`, `oracle_llama_server_llama_f16` | `testing/{greedy,sample}_oracle.py` | 40 of 40 seeded continuations (1,175 tokens) on Bonsai-8B; 40 of 40 on each O4 model |
| `oracle_ggml_b11192_q8_0_kv_kernels` (0.3.9) | the shipped haswell library through `dlopen` (`BANKML_GGML_LIB`) | 4,000 rows quantized byte-exact, 4,000 dot products bit-exact, saturating `maddubs` cases included |
| `kv_oracle_live` (0.3.9, `testing/kv_oracle.py`, Bonsai-1.7B in the gate) | llama-server b11192 with `--cache-type-k q8_0 --cache-type-v q8_0` | 6 of 6 answers (greedy, seeded, a 2,244-token prompt, a 320-token answer, two turns; 567 tokens) |

Rows are compared by the sha256 of their f32 bytes, not within a tolerance. Figures are from docs/oracles.md §1d and
§5d and the 0.3.6 gate record (`testing/results/0.3.6.txt`); the two 0.3.9 rows from CHANGELOG.md (Unreleased 0.3.9).

## Advantages and efficiency

- **One answer, not a close one.** Because the float order, the FMA contractions and the kernel choice all follow
  the shipped llama.cpp, a prompt gives the same logits and the same tokens as llama-server. Speed work can be
  checked: a change counts only if every oracle above still passes.
- **Refusal up front.** `plan` decides from the header alone whether the file is in scope, so an unsupported model is
  refused with a reason before any weight is read.
- **Memory map, read once.** The weights stay in bankml's own memory map. The F32 norm weights are copied once at
  open, and tensors are found through a hash map (0.3.4; before, every lookup scanned the list).
- **Shared activations.** One quantized activation feeds Q, K and V, and one feeds gate and up (`Act`, `quantize`).
  For F16, `mv_many` runs several matrices in one pass of the pool.
- **Batched prefill.** A micro-batch goes through each layer as matrix–matrix products (`q1_0::mat_mul_act_par`,
  `q2_0::mat_mul_par`, `f16::mat_mul_par`). Each element keeps the bits of the per-pair dot. On the last layer only
  the rows llama.cpp outputs (`inp_out_ids`) go through the feed-forward block.
- **Attention on the pool.** Each (row, head) job is computed whole by one worker, so the bits do not depend on the
  thread count. Very small attention jobs run inline instead of waking the pool. The tiled kernel widens each 64-cell
  KV tile once per block of 16 rows, compiled with AVX2, FMA and F16C where the CPU has them.
- **Measured** (docs/PERFORMANCE.md, 0.3.4, laptop, 3 threads): SmolLM2-135M-Instruct F16 decodes at 38.0–38.9
  tok/s against llama-server's 40.4–42.7; the 858-token prompt went from 18.5 s in the first correct build to
  7.8–9.6 s. The same attention work cut the 8B gate runs (for example `oracle_greedy_llama_server_deep` from 756 s
  to 320 s). One loaded-machine pair on Bonsai-8B Q1_0 read 2.30–2.59 tok/s against llama-server's 2.33–2.48; parity
  is not claimed until a pinned, idle measurement.
- **Rust practice.** No crates (`Cargo.toml` has no dependencies). Errors are `Result<_, String>` with the reason.
  `unsafe` is limited to `#[target_feature]` entry points behind a CPU check and the disjoint-slice writes of the
  attention jobs, each with a `SAFETY` comment. The toolchain is pinned (`rust-toolchain.toml`, 1.99.0), and the gate
  runs `cargo clippy -D warnings`.
- **Next** (docs/TODO.md 0.4.0): 1-bit decode at least at llama-server's speed (cut per-token allocations, compute
  logits only where sampled); after the `q8_0` cache (0.3.9), a 4-bit `q4_0` K/V with the same rotation. 0.5.0:
  batched GPU submissions and the ternary GPU kernel.

## Design notes

### How the forward pass was built (P3)

The forward pass was built one verified operation at a time. Each step reproduces the float order of the ggml
operation llama.cpp b11192's graph uses (read from the tag's `ggml/src/ggml-cpu/ops.cpp`), and each was checked against
the shipped ggml computing the same graph (`testing/forward_oracle.py` → `oracle_forward_*`) before the next began.

- **Step three (0.2.3):** the token embedding (`get_rows` on a Q1_0 table → `inp_embd`) and the first RMS norm with
  its weight (`rms_norm` then `mul` → `attn_norm-0`).
- **Step four (0.2.4):** layer 0's attention inputs — the Q, K and V projections (`mul_mat` of a Q1_0 weight with the
  q8_0-quantized normed row, bankml's bit-exact kernel), the per-head RMS norms of Q and K (`attn_q_norm`,
  `attn_k_norm`), and RoPE (`rope_ext`, NEOX pairs, YaRN) with the parameters llama.cpp's context derives.
- **Step five (0.2.5):** attention — K and V kept in f16 as llama.cpp's cache keeps them, the causal flash attention
  of ggml's CPU reference path (`flash_attn_ext`, fewer than 64 query rows and fewer than 512 KV cells), then `wo` and
  the residual (`kqv_out`, `attn_out`, `ffn_inp`).
- **Step six (0.2.6):** the feed-forward block — `ffn_norm`, the gate and up projections, SwiGLU with ggml's own
  vectorized `expf` (not libm's), `ffn_down` and the residual: `l_out`, the whole of layer 0.
- **Step seven (0.2.7):** the whole model — every layer in turn with its own K/V cache, `output_norm` and the logits
  (`Weights::step`, `Weights::logits`). Each matmul runs on a thread pool; `mat_vec_par` has the same bits at any
  thread count.
- **Step eight (0.2.8):** the ternary model. The same forward pass over either weight type — Q1_0 (1-bit, 128-weight
  blocks) or Q2_0_g64 (ternary, 64-weight blocks) — through each type's bit-exact kernel (TECHNICAL.md §III.4,
  §III.6).
- **Step nine (0.2.9):** long prompts. llama.cpp computes a prompt in micro-batches of up to 512 tokens, and a
  micro-batch of 64 rows or more takes ggml's *tiled* flash attention (f32 Q, a SIMD GEMM over 64-cell KV tiles, a
  vectorized softmax summed in double, an f32 accumulator) instead of the reference path. `Weights::prefill` follows
  the same micro-batching and kernel choice.
- **Step ten (0.2.10):** long contexts. A single-token decode whose padded KV length (multiples of 256) reaches 512
  takes ggml's split-KV kernel: the padded cells cut into one chunk per llama.cpp thread, a partial reference pass per
  chunk, then a reduction — so the bits depend on llama.cpp's thread count (`Weights::llama_threads`).
- **0.2.12, batched prefill:** a micro-batch goes through each layer together. Every matmul is one matrix–matrix
  product over the micro-batch's rows (`q1_0::mat_mul_act_par`, `q2_0::mat_mul_par`: each element has the bits of the
  per-pair dot, so the result is the token-by-token result), the micro-batch's K and V enter the cache before its
  attention (as llama.cpp writes them), and each row attends over the cells up to its own position.
- **0.3.4 (O4):** tied embeddings (no `output.weight`: the logits read `token_embd`, as llama.cpp's
  `TENSOR_DUPLICATED` does), F16 weights (`f16.rs`: ggml's two F16 paths, chosen by the product's shape exactly as
  `ggml_compute_forward_mul_mat` chooses them), and the Llama graph (`llm_build_llama`: no Q/K norms, RoPE in NORM mode
  on adjacent pairs, GQA). The multi-row tiled attention (`attend_heads_tiled`) and the tensor-name hash map date from
  this release. What the forward pass runs is decided from the header alone (`plan`); everything else is refused with
  the reason.
- **0.3.9:** the q8_0 K/V cache (`KvType::Q8_0`, `caches_of`) with llama.cpp's Hadamard rotation, and its oracle
  `oracle_ggml_b11192_q8_0_kv_kernels`.

### Why only the output rows leave the last layer

On the last layer of a prompt only the rows llama.cpp outputs go through the feed-forward block (its `inp_out_ids`:
none, the last, or all). For the quantized types this changes no bits, since each element of a matrix–matrix product
has the per-pair dot's bits. For an F16 model it matters: ggml chooses between `ggml_vec_dot_f16` (one column) and
llamafile's tinyBLAS (two or more) by the product's shape, so a different number of rows would give different bits.
For the same reason the model oracle replays an F16 model as one micro-batch with every row output, which is what
the oracle's graph computes.

## Limitations

- Architectures: Qwen3 and Llama only. Each new architecture needs its own oracle set.
- Weight types: Q1_0, Q2_0_g64 and F16 only. Q8_0, BF16 and Q4_K are refused (O3). Mixed matrix types are refused;
  norms must be F32.
- Refused tensors: biases, fused QKV, rope factors, experts, and Q/K norms on Llama. On Llama, any RoPE scaling
  other than `none`, an attention scale, or experts are refused. Llama 3.x is therefore refused (docs/TODO.md).
- Partial RoPE, a value head of another width, and a head width that is not a multiple of 32 are refused.
- The K/V cache is f16 or q8_0 (both K and V the same type; llama.cpp also allows them to differ, and q4/q5 types).
  A q8_0 cache has no tiled or split-KV kernel, as in ggml, so its long prefills are slower than f16's. The
  split-KV bits are correct only when `BANKML_LLAMA_THREADS` equals the matched llama.cpp's `-t`.
- The GPU worker takes part only in 1-bit matrix–vector products; the ternary GPU kernel is next.

## See also

- [../oracles.md](../oracles.md) §1d, §5d — the forward, model, greedy and sampling oracles
- [../PERFORMANCE.md](../PERFORMANCE.md) — F16 and Llama graph speed, decode budgets
- [../TECHNICAL.md](../TECHNICAL.md) §III.7 — from a template to a token
- [../TODO.md](../TODO.md) — 0.4.0 to 0.6.0
- [../OLLAMA.md](../OLLAMA.md) — O3, O4 and what is still refused
- [../usage.md](../usage.md) §13 — `bankml generate` and the environment
- Sibling pages: [native.md](native.md), [sampler.md](sampler.md), [tokenizer.md](tokenizer.md),
  [chat.md](chat.md), [q1_0.md](q1_0.md), [q2_0.md](q2_0.md), [f16.md](f16.md), [par.md](par.md), [gguf.md](gguf.md)
