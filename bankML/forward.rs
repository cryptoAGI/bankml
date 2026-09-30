// SPDX-License-Identifier: MIT OR Apache-2.0
//! P3, the forward pass, one verified operation at a time. Each function here reproduces the float order of the ggml
//! operation llama.cpp b11192's Qwen3 graph uses (read from the tag's `ggml/src/ggml-cpu/ops.cpp`), and its oracle is
//! the SHIPPED ggml computing the same graph (`testing/forward_oracle.py` → `oracle_forward_*`).
//!
//! Step three (0.2.3): the token embedding (`get_rows` on a Q1_0 table → `inp_embd`) and the first RMS norm with its
//! weight (`rms_norm` then `mul` → `attn_norm-0`).
//!
//! Step four (0.2.4): layer 0's attention inputs — the Q, K and V projections (`mul_mat` of a Q1_0 weight with the
//! q8_0-quantized normed row, bankml's bit-exact kernel), the per-head RMS norms of Q and K (`attn_q_norm`,
//! `attn_k_norm`), and RoPE (`rope_ext`, NEOX pairs, YaRN) with the parameters llama.cpp's context derives.
//!
//! Step five (0.2.5): attention — K and V kept in f16 as llama.cpp's cache keeps them, the causal flash attention of
//! ggml's CPU reference path (`flash_attn_ext`, fewer than 64 query rows and fewer than 512 KV cells), then `wo` and
//! the residual (`kqv_out`, `attn_out`, `ffn_inp`).
//!
//! Step six (0.2.6): the feed-forward block — `ffn_norm`, the gate and up projections, SwiGLU with ggml's own
//! vectorized `expf` (not libm's), `ffn_down` and the residual: `l_out`, the whole of layer 0.
//!
//! Step seven (0.2.7): the whole model — every layer in turn with its own K/V cache, `output_norm` and the logits
//! (`Weights::step`, `Weights::logits`). Each matmul runs on a thread pool; `mat_vec_par` has the same bits at any
//! thread count.
//!
//! Step eight (0.2.8): the ternary model. The same forward pass over either weight type — Q1_0 (1-bit, 128-weight
//! blocks) or Q2_0_g64 (ternary, 64-weight blocks) — through each type's bit-exact kernel (§III.4, §III.6).
//!
//! Step nine (0.2.9): long prompts. llama.cpp computes a prompt in micro-batches of up to 512 tokens, and a
//! micro-batch of 64 rows or more takes ggml's *tiled* flash attention (f32 Q, a SIMD GEMM over 64-cell KV tiles, a
//! vectorized softmax summed in double, an f32 accumulator) instead of the reference path. `Weights::prefill` follows
//! the same micro-batching and kernel choice.
//!
//! Step ten (0.2.10): long contexts. A single-token decode whose padded KV length (multiples of 256) reaches 512
//! takes ggml's split-KV kernel: the padded cells cut into one chunk per llama.cpp thread, a partial reference pass per
//! chunk, then a reduction — so the bits depend on llama.cpp's thread count (`Weights::llama_threads`).
//!
//! 0.2.12: batched prefill. A micro-batch goes through each layer together: every matmul is one matrix–matrix
//! product over the micro-batch's rows (`q1_0::mat_mul_act_par`, `q2_0::mat_mul_par`: each element has the bits of
//! the per-pair dot, so the result is the token-by-token result), the micro-batch's K and V enter the cache before
//! its attention (as llama.cpp writes them), and each row attends over the cells up to its own position.

use crate::gguf::{guard_file, Engine, Mmap, TensorInfo, Val};
use crate::par::Pool;
use crate::q1_0::{self, f16_to_f32, f32_to_f16, Q8Act, Q1_0_BYTES, QK1_0};
use crate::q2_0::{self, Q8Act2, Q2_0_BYTES, QK2_0};
use std::path::Path;

/// `ggml_compute_forward_rms_norm_f32` followed by `mul` (fused or not, the same arithmetic): the sum of squares is
/// accumulated in double, one float product at a time, in order; `mean = sum / n` is rounded to float;
/// `scale = 1 / sqrtf(mean + eps)`; each output is `(x · scale) · w`, two float multiplies, no FMA.
pub fn rms_norm_mul(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    assert!(x.len() == w.len() && x.len() == out.len());
    let mut sum = 0.0f64;
    for &v in x {
        sum += (v * v) as f64;
    }
    let mean = (sum / x.len() as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * scale * g;
    }
}

/// `ggml_rope_yarn_corr_dims`: the first and last dimension pair of YaRN's ramp, in float, as ggml.c computes it.
fn yarn_corr_dims(n_dims: usize, n_ctx_orig: u32, freq_base: f32, beta_fast: f32, beta_slow: f32) -> [f32; 2] {
    let dim = |n_rot: f32| n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * freq_base.ln());
    [dim(beta_fast).floor().max(0.0), dim(beta_slow).ceil().min(n_dims as f32 - 1.0)]
}

/// RoPE as llama.cpp's `rope_ext` applies it to a Qwen3 head: NEOX pairs (i, i + n/2), with YaRN. The C source
/// writes plain `a*b ± c*d`; the shipped CPU backend is built with FMA contraction, and the bits follow the binary,
/// so the three contracted expressions are written here as the `mul_add`s the disassembly shows.
#[derive(Debug, Clone, Copy)]
pub struct Rope {
    pub n_dims: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    corr: [f32; 2],
    theta_scale: f32,
}

impl Rope {
    /// The parameters llama-context.cpp (b11192) passes: with YaRN scaling `factor`, `freq_scale = 1/factor`,
    /// `ext_factor = 1`, `attn_factor = get_mscale(factor, 1) / (1 + 0.1·logf(factor))`, beta_fast 32, beta_slow 1.
    pub fn new(n_dims: usize, freq_base: f32, yarn_factor: Option<f32>, n_ctx_orig: u32) -> Self {
        let (freq_scale, ext_factor, attn_factor) = match yarn_factor {
            Some(f) if f > 0.0 && f != 1.0 => {
                let freq_scale = 1.0f32 / f;
                let factor = 1.0f32 / freq_scale;
                let mscale = if factor <= 1.0 { 1.0f32 } else { 0.1f32 * 1.0 * factor.ln() + 1.0 };
                (freq_scale, 1.0f32, mscale * (1.0f32 / (1.0f32 + 0.1f32 * factor.ln())))
            }
            _ => (1.0, 0.0, 1.0),
        };
        Rope {
            n_dims, freq_base, freq_scale, ext_factor, attn_factor,
            corr: yarn_corr_dims(n_dims, n_ctx_orig, freq_base, 32.0, 1.0),
            theta_scale: freq_base.powf(-2.0f32 / n_dims as f32),
        }
    }

    /// `ggml_rope_cache_init` for position `p`: (cos, sin) per pair, theta advanced by repeated multiplication.
    pub fn cache(&self, p: i32, cache: &mut [f32]) {
        let mut theta = p as f32;
        for i0 in (0..self.n_dims).step_by(2) {
            // rope_yarn
            let theta_interp = self.freq_scale * theta;
            let (mut t, mut mscale) = (theta_interp, self.attn_factor);
            if self.ext_factor != 0.0 {
                let y = ((i0 / 2) as f32 - self.corr[0]) / 0.001f32.max(self.corr[1] - self.corr[0]);
                let ramp_mix = (1.0 - y.clamp(0.0, 1.0)) * self.ext_factor;
                // GCC contracts both into FMAs in the shipped build (objdump of libggml-cpu-haswell.so, b11192)
                t = theta_interp.mul_add(1.0 - ramp_mix, theta * ramp_mix);
                mscale *= (1.0f32 / self.freq_scale).ln().mul_add(0.1f32, 1.0f32);
            }
            cache[i0] = t.cos() * mscale;
            cache[i0 + 1] = t.sin() * mscale;
            theta *= self.theta_scale;
        }
    }

    /// `rotate_pairs` in NEOX mode over one head, in place, from a cache built by `cache`.
    pub fn apply(&self, cache: &[f32], head: &mut [f32]) {
        let half = self.n_dims / 2;
        for i0 in (0..self.n_dims).step_by(2) {
            let ic = i0 / 2;
            let (c, s) = (cache[i0], cache[i0 + 1]);
            let (x0, x1) = (head[ic], head[ic + half]);
            // as compiled: x1's products rounded, x0's fused (vmulss + vfmsub231ss / vfmadd132ss)
            head[ic] = c.mul_add(x0, -(x1 * s));
            head[ic + half] = s.mul_add(x0, x1 * c);
        }
    }
}

/// `ggml_vec_dot_f16` as the AVX2 build computes it: four 8-lane f32 accumulators fed by FMAs over 32-element
/// steps, reduced as `(a0 + a2) + (a1 + a3)`, then low half + high half, then two horizontal adds. `n % 32 == 0`.
pub fn dot_f16(x: &[u16], y: &[u16]) -> f32 {
    assert!(x.len() == y.len() && x.len().is_multiple_of(32));
    let mut acc = [[0.0f32; 8]; 4];
    for (xs, ys) in x.chunks_exact(32).zip(y.chunks_exact(32)) {
        for (j, a) in acc.iter_mut().enumerate() {
            for (l, al) in a.iter_mut().enumerate() {
                *al = f16_to_f32(xs[8 * j + l]).mul_add(f16_to_f32(ys[8 * j + l]), *al);
            }
        }
    }
    let mut r = [0.0f32; 8];
    for l in 0..8 {
        r[l] = (acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l]);
    }
    let t: [f32; 4] = std::array::from_fn(|l| r[l] + r[l + 4]);
    (t[0] + t[1]) + (t[2] + t[3])
}

/// One query head against a causal run of cached keys and values (f16, `hd` wide, `stride` apart), as ggml's
/// `flash_attn_ext_f16_one_chunk`: Q rounded to f16; each score `dot · scale`; an online softmax whose V accumulator
/// is f16 (`vec_mad_f16`: `f16(fma(v, w, acc))`, `vec_scale_f16`: `f16(acc · ms)`); finally `acc · (1/S)` in f32.
pub fn attend_head(q: &[f32], k: &[u16], v: &[u16], n_kv: usize, stride: usize, scale: f32, out: &mut [f32]) {
    let (_, sum, acc) = attend_head_partial(q, k, v, 0, n_kv, stride, scale);
    let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
    for (o, &a) in out.iter_mut().zip(&acc) {
        *o = f16_to_f32(a) * inv;
    }
}

/// The reference path over cells `from..to` without the final normalisation: (max, sum, f16 accumulator) — what
/// `flash_attn_ext_f16_one_chunk` writes as a partial for the split-KV kernel.
pub fn attend_head_partial(q: &[f32], k: &[u16], v: &[u16], from: usize, to: usize, stride: usize, scale: f32) -> (f32, f32, Vec<u16>) {
    let hd = q.len();
    let q16: Vec<u16> = q.iter().map(|&x| f32_to_f16(x)).collect();
    let mut acc = vec![0u16; hd];
    let (mut sum, mut max) = (0.0f32, f32::NEG_INFINITY);
    for ic in from..to {
        let s = dot_f16(&k[ic * stride..ic * stride + hd], &q16) * scale;
        let (mut ms, mut vs) = (1.0f32, 1.0f32);
        if s > max {
            let old = max;
            max = s;
            ms = (old - max).exp();
            for a in acc.iter_mut() {
                *a = f32_to_f16(f16_to_f32(*a) * ms);
            }
        } else {
            vs = (s - max).exp();
        }
        for (a, &vv) in acc.iter_mut().zip(&v[ic * stride..ic * stride + hd]) {
            *a = f32_to_f16(f16_to_f32(vv).mul_add(vs, f16_to_f32(*a)));
        }
        sum = sum * ms + vs;
    }
    (max, sum, acc)
}

/// One query row of a single-token decode over `padded` KV cells (`cells` of them visible), as ggml's split-KV
/// flash attention computes it with `nth` threads: the padded cells cut into `ceil(padded / nth)`-cell chunks, a
/// partial reference pass per chunk (masked cells skipped), then `ggml_flash_attn_ext_reduce_partials` in chunk
/// order — `fmaxf` of the maxima, two `expf` rescales, `fma(acc, old, chunk · new)` (the shipped build contracts it;
/// the plain form agrees only while the first chunk holds the maximum) — and `· (1/S)`.
#[allow(clippy::too_many_arguments)]
pub fn attend_head_split(q: &[f32], k: &[u16], v: &[u16], cells: usize, padded: usize, nth: usize, stride: usize, scale: f32, out: &mut [f32]) {
    let chunk = padded.div_ceil(nth);
    let (mut max, mut sum) = (f32::NEG_INFINITY, 0.0f32);
    let mut acc = vec![0.0f32; q.len()];
    for c in 0..nth {
        let start = c * chunk;
        if start >= padded {
            continue;
        }
        let end = (start + chunk).min(padded).min(cells);
        if start >= end {
            continue; // every cell of this chunk is masked: its partial sum is 0 and the reduction skips it
        }
        let (m_c, s_c, a_c) = attend_head_partial(q, k, v, start, end, stride, scale);
        if s_c == 0.0 {
            continue;
        }
        let m_new = max.max(m_c);
        let (so, sn) = ((max - m_new).exp(), (m_c - m_new).exp());
        // as compiled (vmulps + vfmadd231ps; vmulss + vfmadd132ss): the chunk's term rounded, the running one fused
        for (a, &b) in acc.iter_mut().zip(&a_c) {
            *a = a.mul_add(so, f16_to_f32(b) * sn);
        }
        sum = sum.mul_add(so, s_c * sn);
        max = m_new;
    }
    if sum != 0.0 {
        let inv = 1.0 / sum;
        for a in acc.iter_mut() {
            *a *= inv;
        }
    }
    out.copy_from_slice(&acc);
}

/// One query head against a causal run of cached keys and values, as ggml's tiled flash attention
/// (`ggml_compute_forward_flash_attn_ext_tiled`) computes one row: Q stays f32; per 64-cell KV tile the scores are
/// `simd_gemm` (one FMA chain over the head dimension), `· scale`, `+ mask`; the tile's max (`ggml_vec_max_f32`),
/// `fmaxf` with the running max, `expf` rescaling of the f32 accumulator and sum; `ggml_vec_soft_max_f32` (ggml's
/// `v_expf`, each 8-lane group reduced in f32 and summed in double, the sum added to the float `S` through double);
/// then the V GEMM (one FMA chain over the tile's cells); finally `acc · (1/S)`. A row's result does not depend on
/// which rows share its tile; cells a row cannot see weigh 0 and are left out.
pub fn attend_head_tiled(q: &[f32], k: &[u16], v: &[u16], n_kv: usize, stride: usize, scale: f32, out: &mut [f32]) {
    const T: usize = 64;
    let hd = q.len();
    let mut acc = vec![0.0f32; hd];
    let (mut sum, mut max) = (0.0f32, f32::NEG_INFINITY);
    let mut kq = [0.0f32; T];
    for ic in (0..n_kv).step_by(T) {
        let tile = T.min(n_kv - ic);
        for (tk, s) in kq.iter_mut().enumerate() {
            *s = if tk < tile {
                let kr = &k[(ic + tk) * stride..(ic + tk) * stride + hd];
                let mut a = 0.0f32;
                for (&kd, &qd) in kr.iter().zip(q) {
                    a = f16_to_f32(kd).mul_add(qd, a);
                }
                a * scale + 0.0 // the mask adds 0 to a visible cell
            } else {
                f32::NEG_INFINITY
            };
        }
        let tile_max = kq.iter().fold(f32::NEG_INFINITY, |m, &x| if m > x { m } else { x });
        if tile_max == f32::NEG_INFINITY {
            continue;
        }
        let new_max = max.max(tile_max);
        if new_max > max {
            let ms = (max - new_max).exp();
            for a in acc.iter_mut() {
                *a *= ms;
            }
            sum *= ms;
        }
        max = new_max;
        let mut tsum = 0.0f64;
        for g in kq.chunks_exact_mut(8) {
            for x in g.iter_mut() {
                *x = v_expf(*x - new_max);
            }
            let h: [f32; 4] = std::array::from_fn(|l| g[l + 4] + g[l]);
            tsum += ((h[0] + h[2]) + (h[1] + h[3])) as f64;
        }
        sum = (sum as f64 + tsum) as f32;
        for (tk, &p) in kq.iter().enumerate().take(tile) {
            let vr = &v[(ic + tk) * stride..(ic + tk) * stride + hd];
            for (a, &vd) in acc.iter_mut().zip(vr) {
                *a = f16_to_f32(vd).mul_add(p, *a);
            }
        }
    }
    let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
    for (o, &a) in out.iter_mut().zip(&acc) {
        *o = a * inv;
    }
}

/// Which of ggml's CPU flash-attention kernels llama.cpp would run for a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// `flash_attn_ext_f16_one_chunk`: micro-batches of 2–63 rows, and single-token decodes over fewer than 512
    /// padded KV cells
    Reference,
    /// `flash_attn_ext_tiled`: micro-batches of 64 rows or more
    Tiled,
    /// the split-KV path: a single-token step over `padded` ≥ 512 KV cells, one chunk per llama.cpp thread
    Split { padded: usize, nth: usize },
}

/// llama.cpp's micro-batch (`--ubatch-size`, default 512).
pub const N_UBATCH: usize = 512;

/// The KV length llama.cpp attends over with `cells` in use: padded to a multiple of 256, at least 256
/// (`llama_kv_cache::get_n_kv`).
pub fn padded_kv(cells: usize) -> usize {
    cells.div_ceil(256).max(1) * 256
}

/// The kernel llama.cpp (running `llama_threads` threads) uses for a micro-batch of `rows` whose last row makes
/// `cells` cells in use.
pub fn kernel_for(rows: usize, cells: usize, llama_threads: usize) -> Result<Kernel, String> {
    Ok(match rows {
        64.. => Kernel::Tiled,
        1 if padded_kv(cells) >= 512 => Kernel::Split { padded: padded_kv(cells), nth: llama_threads.max(1) },
        _ => Kernel::Reference,
    })
}

/// ggml's AVX2 `ggml_v_expf` (vec.h, "adapted from arm limited optimized routine"), one lane: a range reduction by
/// `2^n`, a degree-5 polynomial in FMAs, and the scaled path for `|n| > 126`. Each lane depends only on itself, so
/// the scalar form gives the vector's bits.
pub fn v_expf(x: f32) -> f32 {
    let c = |b: u32| f32::from_bits(b);
    let r = c(0x4b40_0000); // 0x1.8p23
    let z = x.mul_add(c(0x3fb8_aa3b), r);
    let n = z - r;
    let b = (-n).mul_add(c(0x35bf_be8e), (-n).mul_add(c(0x3f31_7200), x));
    let e = z.to_bits() << 23;
    let k = f32::from_bits(e.wrapping_add(1.0f32.to_bits()));
    let big = n.abs() > 126.0;
    let u = b * b;
    let j = c(0x3c07_2010).mul_add(b, c(0x3d2b_9f17)).mul_add(u, c(0x3e2a_af33).mul_add(b, c(0x3eff_fedb))).mul_add(u, c(0x3f7f_fff6) * b);
    if !big {
        return j.mul_add(k, k);
    }
    let g = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(g));
    if n.abs() > 192.0 { s1 * s1 } else { s2.mul_add(j, s2) * s1 }
}

/// `ggml_vec_swiglu_f32` on AVX2: `silu(g) · u` with `silu(x) = x / (1 + v_expf(0 − x))`.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = g / (1.0 + v_expf(0.0 - g)) * u;
    }
}

/// One layer's K/V cache, f16, one row of `n_head_kv · head_dim` per position (llama.cpp's `cache_k_l*`/`cache_v_l*`
/// without the transposed-V layout, which flash attention does not use).
pub struct KvCache {
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub width: usize,
}

impl KvCache {
    pub fn new(width: usize) -> Self {
        KvCache { k: Vec::new(), v: Vec::new(), width }
    }
    pub fn len(&self) -> usize {
        self.k.len() / self.width
    }
    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }
    /// Store a position's K and V as the cache stores them (`set_rows` f32 → f16, round to nearest even).
    pub fn push(&mut self, k: &[f32], v: &[f32]) {
        self.k.extend(k.iter().map(|&x| f32_to_f16(x)));
        self.v.extend(v.iter().map(|&x| f32_to_f16(x)));
    }
}

/// GGUF tensor types bankml's forward pass runs.
pub const TYPE_Q1_0: u32 = 41;
pub const TYPE_Q2_0: u32 = 42;

/// An activation row quantized to q8_0 and prepared for the model's weight type (shared by every matrix that reads
/// the same row: Q, K and V; gate and up).
pub enum Act {
    Q1(Q8Act),
    Q2(Q8Act2),
}

/// Activation rows prepared for the model's weight type: one per token of a micro-batch.
pub enum Acts {
    Q1(Vec<Q8Act>),
    Q2(Vec<Q8Act2>),
}

/// A weight matrix: its bytes, rows and type.
pub struct Mat<'a> {
    name: &'a str,
    bytes: &'a [u8],
    pub rows: usize,
    ty: u32,
}

fn row_bytes(ty: u32, n: usize) -> usize {
    if ty == TYPE_Q1_0 { n / QK1_0 * Q1_0_BYTES } else { n / QK2_0 * Q2_0_BYTES }
}

/// The model's tensors by name, read through bankml's own memory map.
pub struct Weights {
    mm: Mmap,
    tensors: Vec<TensorInfo>,
    data_start: u64,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub rms_eps: f32,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope: Rope,
    pub n_layer: usize,
    /// the matrices' type: `TYPE_Q1_0` or `TYPE_Q2_0`
    pub wtype: u32,
    /// the thread count of the llama.cpp being matched (`-t`; Savante runs 3): its split-KV decode kernel cuts the KV
    /// cells into one chunk per thread, so the bits depend on it. `BANKML_LLAMA_THREADS`, default 3.
    pub llama_threads: usize,
    pool: Pool,
    /// a verified GPU taking a share of every 1-bit matrix–vector product's rows (`gpu::worker`), if one was found
    pub gpu: Option<std::sync::Mutex<crate::gpu::worker::Worker>>,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self, String> {
        let rep = guard_file(path, Engine::Mainline).map_err(|e| format!("{}: {e}", path.display()))?;
        let h = rep.header.ok_or("no GGUF header")?;
        let arch = match h.kv.get("general.architecture") {
            Some(Val::S(a)) => a.clone(),
            _ => return Err("no architecture".into()),
        };
        let num = |k: &str| match h.kv.get(&format!("{arch}.{k}")) {
            Some(Val::U(v)) => Some(*v as f64),
            Some(Val::I(v)) => Some(*v as f64),
            Some(Val::F(v)) => Some(*v),
            _ => None,
        };
        let emb = h.tensors.iter().find(|t| t.name == "token_embd.weight").ok_or("no token_embd.weight")?;
        let (n_embd, n_vocab) = (emb.dims[0] as usize, emb.dims[1] as usize);
        let rms_eps = num("attention.layer_norm_rms_epsilon").ok_or("no rms epsilon")? as f32;
        let n_head = num("attention.head_count").ok_or("no head count")? as usize;
        let n_head_kv = num("attention.head_count_kv").unwrap_or(n_head as f64) as usize;
        let head_dim = num("attention.key_length").map(|v| v as usize).unwrap_or(n_embd / n_head);
        let yarn = match h.kv.get(&format!("{arch}.rope.scaling.type")) {
            Some(Val::S(t)) if t == "yarn" => num("rope.scaling.factor").map(|f| f as f32),
            _ => None,
        };
        let n_ctx_orig = num("rope.scaling.original_context_length").or_else(|| num("context_length")).ok_or("no context length")? as u32;
        let rope = Rope::new(head_dim, num("rope.freq_base").unwrap_or(10000.0) as f32, yarn, n_ctx_orig);
        let n_layer = num("block_count").ok_or("no block count")? as usize;
        let wtype = emb.ty;
        if wtype != TYPE_Q1_0 && wtype != TYPE_Q2_0 {
            return Err(format!("token_embd is type {wtype}; the forward pass runs Q1_0 and Q2_0_g64 models"));
        }
        let mm = Mmap::open(path).map_err(|e| e.to_string())?;
        let pool = Pool::from_env();
        // a card joins only for the 1-bit kernel (the ternary kernel is next), only after it passes the on-card
        // oracle, and only if the calibration gives it a share worth sending; any failure leaves the CPU path alone
        let gpu = if wtype == TYPE_Q1_0 {
            match crate::gpu::worker::Worker::open(&pool) {
                Ok(Some(w)) => {
                    eprintln!("bankml: GPU {} verified; it computes {:.0}% of each 1-bit matrix's rows", w.name, w.share * 100.0);
                    Some(std::sync::Mutex::new(w))
                }
                Ok(None) => None,
                Err(e) => {
                    eprintln!("bankml: GPU not used: {e}");
                    None
                }
            }
        } else {
            None
        };
        Ok(Weights { mm, tensors: h.tensors, data_start: h.data_start, n_embd, n_vocab, rms_eps, n_head, n_head_kv, head_dim, rope, n_layer, wtype,
                     llama_threads: std::env::var("BANKML_LLAMA_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(3),
                     pool, gpu })
    }

    fn tensor(&self, name: &str) -> Result<(&TensorInfo, &[u8]), String> {
        let t = self.tensors.iter().find(|t| t.name == name).ok_or_else(|| format!("no tensor {name}"))?;
        let start = (self.data_start + t.offset) as usize;
        Ok((t, self.mm.bytes().get(start..).ok_or("tensor outside the file")?))
    }

    /// An f32 vector tensor (norm weights).
    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>, String> {
        let (t, b) = self.tensor(name)?;
        if t.ty != 0 {
            return Err(format!("{name} is type {}, not F32", t.ty));
        }
        let n = t.dims.iter().product::<u64>() as usize;
        let b = b.get(..n * 4).ok_or("tensor truncated")?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// A weight matrix of the model's type, checked against the input width.
    pub fn matrix(&self, name: &str, n_in: usize) -> Result<Mat<'_>, String> {
        let (t, b) = self.tensor(name)?;
        if t.ty != self.wtype || t.dims[0] as usize != n_in {
            return Err(format!("{name}: type {} dims {:?}; expected type {} with {n_in} columns", t.ty, t.dims, self.wtype));
        }
        let rows = t.dims[1] as usize;
        Ok(Mat { name: &t.name, bytes: b.get(..rows * row_bytes(t.ty, n_in)).ok_or("tensor truncated")?, rows, ty: t.ty })
    }

    /// `x` quantized to q8_0 as ggml quantizes a matmul's activation, prepared for the model's kernel.
    pub fn quantize(&self, x: &[f32]) -> Act {
        if self.wtype == TYPE_Q1_0 { Act::Q1(Q8Act::quantize(x)) } else { Act::Q2(Q8Act2::quantize(x)) }
    }

    /// Every row of `xs` quantized to q8_0, prepared for the model's kernel.
    pub fn quantize_rows(&self, xs: &[Vec<f32>]) -> Acts {
        if self.wtype == TYPE_Q1_0 {
            Acts::Q1(xs.iter().map(|x| Q8Act::quantize(x)).collect())
        } else {
            Acts::Q2(xs.iter().map(|x| Q8Act2::quantize(x)).collect())
        }
    }

    /// `m` times every prepared row, one matrix–matrix product on the pool; returns one output row per input row.
    /// Each element has the bits of the per-pair dot, so the rows equal `mv` on each row.
    pub fn mm(&self, m: &Mat, a: &Acts) -> Result<Vec<Vec<f32>>, String> {
        let n = match a { Acts::Q1(v) => v.len(), Acts::Q2(v) => v.len() };
        let mut out = vec![0.0f32; m.rows * n];
        match (m.ty, a) {
            (TYPE_Q1_0, Acts::Q1(cols)) => q1_0::mat_mul_act_par(&self.pool, m.bytes, m.rows, cols, &mut out),
            (TYPE_Q2_0, Acts::Q2(cols)) => q2_0::mat_mul_par(&self.pool, m.bytes, m.rows, cols, &mut out),
            _ => return Err("activations prepared for another weight type".into()),
        }
        Ok(out.chunks_exact(m.rows).map(<[f32]>::to_vec).collect())
    }

    /// `out = m · a` on the pool (each type's `mat_vec_par`: the bits of ggml's per-row `vec_dot`).
    pub fn mv(&self, m: &Mat, a: &Act, out: &mut [f32]) -> Result<(), String> {
        if out.len() != m.rows {
            return Err(format!("{} rows, buffer {}", m.rows, out.len()));
        }
        match (m.ty, a) {
            (TYPE_Q1_0, Act::Q1(a)) => {
                // a verified card, if there is one, takes the first rows while the CPU pool computes the rest
                let mut worker = self.gpu.as_ref().and_then(|w| w.lock().ok());
                let g = match worker.as_mut() {
                    Some(w) => w.begin(m.name, m.bytes, m.rows, a)?,
                    None => 0,
                };
                let rb = a.n() / QK1_0 * Q1_0_BYTES;
                q1_0::mat_vec_par(&self.pool, &m.bytes[g * rb..], m.rows - g, a, &mut out[g..]);
                if let Some(w) = worker.as_mut() {
                    w.finish(&mut out[..g])?;
                }
            }
            (TYPE_Q2_0, Act::Q2(a)) => q2_0::mat_vec_par(&self.pool, m.bytes, m.rows, a, out),
            _ => return Err("activation prepared for another weight type".into()),
        }
        Ok(())
    }

    /// Layer `il`'s attention inputs for one token at position `pos`, from its normed row `xn` (`attn_norm-il`):
    /// Q (`n_head · head_dim`), K and V (`n_head_kv · head_dim`) — projected, Q and K normed per head and roped,
    /// as `Qcur`, `Kcur`, `Vcur` in llama.cpp's qwen3 graph.
    pub fn qkv(&self, il: usize, xn: &[f32], pos: i32, q: &mut [f32], k: &mut [f32], v: &mut [f32]) -> Result<(), String> {
        let a = self.quantize(xn);
        let hd = self.head_dim;
        let mut cache = vec![0.0f32; hd];
        self.rope.cache(pos, &mut cache);
        let mut tmp = vec![0.0f32; hd];
        for (w, norm, out) in [("attn_q", Some("attn_q_norm"), &mut *q), ("attn_k", Some("attn_k_norm"), &mut *k), ("attn_v", None, &mut *v)] {
            let m = self.matrix(&format!("blk.{il}.{w}.weight"), self.n_embd)?;
            self.mv(&m, &a, out).map_err(|e| format!("{w}: {e}"))?;
            if let Some(norm) = norm {
                let g = self.f32_vec(&format!("blk.{il}.{norm}.weight"))?;
                for h in out.chunks_exact_mut(hd) {
                    tmp.copy_from_slice(h);
                    rms_norm_mul(&tmp, &g, self.rms_eps, h);
                    self.rope.apply(&cache, h);
                }
            }
        }
        Ok(())
    }

    /// Layer `il`'s attention for the newest position in `cache` (causal: every cached position is visible), then
    /// `wo`: writes `kqv_out` (`n_head · head_dim`) and returns nothing else; `attn_out` goes to `out` (`n_embd`).
    pub fn attention(&self, il: usize, q: &[f32], cache: &KvCache, kqv: &mut [f32], out: &mut [f32]) -> Result<(), String> {
        self.attention_with(il, q, cache, Kernel::Reference, kqv, out)
    }

    /// `attention` with the kernel llama.cpp would choose for this row.
    pub fn attention_with(&self, il: usize, q: &[f32], cache: &KvCache, kernel: Kernel, kqv: &mut [f32], out: &mut [f32]) -> Result<(), String> {
        self.attend(q, cache, cache.len(), kernel, kqv);
        let m = self.matrix(&format!("blk.{il}.attn_output.weight"), kqv.len())?;
        self.mv(&m, &self.quantize(kqv), out).map_err(|e| format!("attn_output: {e}"))?;
        Ok(())
    }

    /// Every head of one query row over the first `visible` cached cells (causal: the row's own position + 1).
    pub fn attend(&self, q: &[f32], cache: &KvCache, visible: usize, kernel: Kernel, kqv: &mut [f32]) {
        let hd = self.head_dim;
        let group = self.n_head / self.n_head_kv;
        let scale = 1.0f32 / (hd as f32).sqrt();
        for (h, (qh, oh)) in q.chunks_exact(hd).zip(kqv.chunks_exact_mut(hd)).enumerate() {
            let off = (h / group) * hd;
            match kernel {
                Kernel::Reference => attend_head(qh, &cache.k[off..], &cache.v[off..], visible, cache.width, scale, oh),
                Kernel::Tiled => attend_head_tiled(qh, &cache.k[off..], &cache.v[off..], visible, cache.width, scale, oh),
                Kernel::Split { padded, nth } => {
                    attend_head_split(qh, &cache.k[off..], &cache.v[off..], visible, padded, nth, cache.width, scale, oh)
                }
            }
        }
    }

    /// Layer `il`'s feed-forward block on `ffn_inp` (the residual stream after attention), written back in place as
    /// `l_out`: `ffn_norm`, gate and up (one q8_0 activation), SwiGLU, `ffn_down`, the residual.
    pub fn ffn(&self, il: usize, x: &mut [f32]) -> Result<(), String> {
        let g = self.f32_vec(&format!("blk.{il}.ffn_norm.weight"))?;
        let mut xn = vec![0.0f32; x.len()];
        rms_norm_mul(x, &g, self.rms_eps, &mut xn);
        let a = self.quantize(&xn);
        let wg = self.matrix(&format!("blk.{il}.ffn_gate.weight"), self.n_embd)?;
        let wu = self.matrix(&format!("blk.{il}.ffn_up.weight"), self.n_embd)?;
        let n_ff = wg.rows;
        if wu.rows != n_ff {
            return Err(format!("ffn_gate has {n_ff} rows, ffn_up {}", wu.rows));
        }
        let (mut gate, mut up, mut h) = (vec![0.0f32; n_ff], vec![0.0f32; n_ff], vec![0.0f32; n_ff]);
        self.mv(&wg, &a, &mut gate)?;
        self.mv(&wu, &a, &mut up)?;
        swiglu(&gate, &up, &mut h);
        let wd = self.matrix(&format!("blk.{il}.ffn_down.weight"), n_ff)?;
        let mut down = vec![0.0f32; x.len()];
        self.mv(&wd, &self.quantize(&h), &mut down).map_err(|e| format!("ffn_down: {e}"))?;
        for (xi, d) in x.iter_mut().zip(&down) {
            *xi += d; // ggml adds ffn_out + ffn_inp; IEEE addition commutes, so the bits are the same
        }
        Ok(())
    }

    /// Empty K/V caches, one per layer.
    pub fn caches(&self) -> Vec<KvCache> {
        (0..self.n_layer).map(|_| KvCache::new(self.n_head_kv * self.head_dim)).collect()
    }

    /// One token through every layer at the next position (the caches' length), appending its K and V to each
    /// layer's cache. `each_layer(il, l_out)` sees every layer's output; returns `result_norm` (`output_norm` of the
    /// last layer's output), which `logits` turns into scores.
    pub fn step(&self, caches: &mut [KvCache], token: u32, each_layer: impl FnMut(usize, &[f32])) -> Result<Vec<f32>, String> {
        self.step_with(caches, token, Kernel::Reference, each_layer)
    }

    /// A prompt, as llama.cpp computes it: micro-batches of up to `N_UBATCH` tokens, each row's attention by the
    /// kernel llama.cpp chooses for its micro-batch (`kernel_for`). Returns the last token's `result_norm`.
    pub fn prefill(&self, caches: &mut [KvCache], tokens: &[u32], mut each: impl FnMut(usize, usize, &[f32])) -> Result<Vec<f32>, String> {
        let mut rn = Vec::new();
        for ub in tokens.chunks(N_UBATCH) {
            let kernel = kernel_for(ub.len(), caches[0].len() + ub.len(), self.llama_threads)?;
            rn = self.ubatch(caches, ub, kernel, &mut each)?;
        }
        Ok(rn)
    }

    /// One micro-batch through every layer together (see the module notes: the same bits as token by token).
    /// Returns the last row's `result_norm`.
    fn ubatch(&self, caches: &mut [KvCache], toks: &[u32], kernel: Kernel, each: &mut impl FnMut(usize, usize, &[f32])) -> Result<Vec<f32>, String> {
        let (n, p0, hd) = (toks.len(), caches[0].len(), self.head_dim);
        let mut xs = vec![vec![0.0f32; self.n_embd]; n];
        for (x, &t) in xs.iter_mut().zip(toks) {
            self.embed(t, x)?;
        }
        let ropes: Vec<Vec<f32>> = (0..n).map(|i| {
            let mut c = vec![0.0f32; hd];
            self.rope.cache((p0 + i) as i32, &mut c);
            c
        }).collect();
        let norm_rows = |xs: &[Vec<f32>], g: &[f32]| -> Vec<Vec<f32>> {
            xs.iter().map(|x| {
                let mut y = vec![0.0f32; x.len()];
                rms_norm_mul(x, g, self.rms_eps, &mut y);
                y
            }).collect()
        };
        let mut tmp = vec![0.0f32; hd];
        for (il, cache) in caches.iter_mut().enumerate() {
            let a = self.quantize_rows(&norm_rows(&xs, &self.f32_vec(&format!("blk.{il}.attn_norm.weight"))?));
            let mut q = self.mm(&self.matrix(&format!("blk.{il}.attn_q.weight"), self.n_embd)?, &a)?;
            let mut k = self.mm(&self.matrix(&format!("blk.{il}.attn_k.weight"), self.n_embd)?, &a)?;
            let v = self.mm(&self.matrix(&format!("blk.{il}.attn_v.weight"), self.n_embd)?, &a)?;
            let (gq, gk) = (self.f32_vec(&format!("blk.{il}.attn_q_norm.weight"))?, self.f32_vec(&format!("blk.{il}.attn_k_norm.weight"))?);
            for (rows, g) in [(&mut q, &gq), (&mut k, &gk)] {
                for (row, rc) in rows.iter_mut().zip(&ropes) {
                    for h in row.chunks_exact_mut(hd) {
                        tmp.copy_from_slice(h);
                        rms_norm_mul(&tmp, g, self.rms_eps, h);
                        self.rope.apply(rc, h);
                    }
                }
            }
            for (kr, vr) in k.iter().zip(&v) {
                cache.push(kr, vr); // the whole micro-batch's K and V first, as llama.cpp writes them; causality is `visible`
            }
            let mut kqv = vec![vec![0.0f32; self.n_head * hd]; n];
            for (i, (qr, o)) in q.iter().zip(kqv.iter_mut()).enumerate() {
                self.attend(qr, cache, p0 + i + 1, kernel, o);
            }
            let att = self.mm(&self.matrix(&format!("blk.{il}.attn_output.weight"), self.n_head * hd)?, &self.quantize_rows(&kqv))?;
            for (x, a) in xs.iter_mut().zip(&att) {
                for (xi, ai) in x.iter_mut().zip(a) {
                    *xi += ai;
                }
            }
            // the feed-forward block
            let a = self.quantize_rows(&norm_rows(&xs, &self.f32_vec(&format!("blk.{il}.ffn_norm.weight"))?));
            let gate = self.mm(&self.matrix(&format!("blk.{il}.ffn_gate.weight"), self.n_embd)?, &a)?;
            let up = self.mm(&self.matrix(&format!("blk.{il}.ffn_up.weight"), self.n_embd)?, &a)?;
            let h: Vec<Vec<f32>> = gate.iter().zip(&up).map(|(g, u)| {
                let mut o = vec![0.0f32; g.len()];
                swiglu(g, u, &mut o);
                o
            }).collect();
            let n_ff = h[0].len();
            let down = self.mm(&self.matrix(&format!("blk.{il}.ffn_down.weight"), n_ff)?, &self.quantize_rows(&h))?;
            for (i, (x, d)) in xs.iter_mut().zip(&down).enumerate() {
                for (xi, di) in x.iter_mut().zip(d) {
                    *xi += di;
                }
                each(p0 + i, il, x);
            }
        }
        let mut rn = vec![0.0f32; self.n_embd];
        rms_norm_mul(&xs[n - 1], &self.f32_vec("output_norm.weight")?, self.rms_eps, &mut rn);
        Ok(rn)
    }

    /// One generated token (a micro-batch of one), with the kernel llama.cpp would use at this length.
    pub fn decode(&self, caches: &mut [KvCache], token: u32) -> Result<Vec<f32>, String> {
        let kernel = kernel_for(1, caches[0].len() + 1, self.llama_threads)?;
        self.step_with(caches, token, kernel, |_, _| {})
    }

    /// `step` with the attention kernel given.
    pub fn step_with(&self, caches: &mut [KvCache], token: u32, kernel: Kernel, mut each_layer: impl FnMut(usize, &[f32])) -> Result<Vec<f32>, String> {
        if caches.len() != self.n_layer {
            return Err(format!("{} caches for {} layers", caches.len(), self.n_layer));
        }
        let pos = caches[0].len() as i32;
        let (qd, kd) = (self.n_head * self.head_dim, self.n_head_kv * self.head_dim);
        let mut x = vec![0.0f32; self.n_embd];
        self.embed(token, &mut x)?;
        let (mut xn, mut q, mut k, mut v) = (vec![0.0f32; self.n_embd], vec![0.0f32; qd], vec![0.0f32; kd], vec![0.0f32; kd]);
        let (mut kqv, mut att) = (vec![0.0f32; qd], vec![0.0f32; self.n_embd]);
        for (il, cache) in caches.iter_mut().enumerate() {
            rms_norm_mul(&x, &self.f32_vec(&format!("blk.{il}.attn_norm.weight"))?, self.rms_eps, &mut xn);
            self.qkv(il, &xn, pos, &mut q, &mut k, &mut v)?;
            cache.push(&k, &v);
            self.attention_with(il, &q, cache, kernel, &mut kqv, &mut att)?;
            for (xi, a) in x.iter_mut().zip(&att) {
                *xi += a; // ggml: attn_out + inpSA; addition commutes
            }
            self.ffn(il, &mut x)?;
            each_layer(il, &x);
        }
        let mut rn = vec![0.0f32; self.n_embd];
        rms_norm_mul(&x, &self.f32_vec("output_norm.weight")?, self.rms_eps, &mut rn);
        Ok(rn)
    }

    /// The logits: `output.weight` (Q1_0, one row per vocabulary entry) times the q8_0-quantized `result_norm`.
    pub fn logits(&self, result_norm: &[f32]) -> Result<Vec<f32>, String> {
        let m = self.matrix("output.weight", self.n_embd)?;
        let mut out = vec![0.0f32; m.rows];
        self.mv(&m, &self.quantize(result_norm), &mut out)?;
        Ok(out)
    }

    /// `get_rows(token_embd, [id])`: the id's row of the table, dequantized (bit-exact with ggml's
    /// `dequantize_row_q1_0` / `dequantize_row_q2_0`, §III.4, §III.6).
    pub fn embed(&self, id: u32, out: &mut [f32]) -> Result<(), String> {
        let (t, b) = self.tensor("token_embd.weight")?;
        let ty = t.ty;
        if id as usize >= self.n_vocab || out.len() != self.n_embd {
            return Err(format!("token {id} outside the vocabulary of {}", self.n_vocab));
        }
        let rb = row_bytes(ty, self.n_embd);
        let row = b.get(id as usize * rb..(id as usize + 1) * rb).ok_or("tensor truncated")?;
        if ty == TYPE_Q1_0 { q1_0::dequantize_row(row, out) } else { q2_0::dequantize_row(row, out) }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_of_a_constant_row() {
        // x = 2 everywhere: mean 4, scale 1/sqrt(4 + eps) ≈ 0.5, out ≈ w
        let (x, w) = (vec![2.0f32; 8], vec![3.0f32; 8]);
        let mut o = vec![0.0f32; 8];
        rms_norm_mul(&x, &w, 0.0, &mut o);
        assert!(o.iter().all(|&v| v == 3.0));
    }

    /// Every token's inp_embd and attn_norm-0 row, bit for bit, against the shipped ggml (testing/forward_oracle.py).
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + .models/oracle-forward (testing/forward_oracle.py); --release"]
    fn oracle_forward_embed_norm() {
        use crate::sha256::{hex, Sha256};
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let w = Weights::open(&dir.join("Bonsai-8B-Q1_0.gguf")).unwrap();
        let g = w.f32_vec("blk.0.attn_norm.weight").unwrap();
        let sha = |v: &[f32]| {
            let mut h = Sha256::default();
            v.iter().for_each(|x| h.update(&x.to_le_bytes()));
            hex(&h.finish())
        };
        let tsv = std::fs::read_to_string(dir.join("oracle-forward/embed_norm.tsv")).unwrap();
        let (mut n, mut e_ok, mut n_ok) = (0, 0, 0);
        let (mut x, mut y) = (vec![0.0f32; w.n_embd], vec![0.0f32; w.n_embd]);
        for line in tsv.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            w.embed(f[0].parse().unwrap(), &mut x).unwrap();
            rms_norm_mul(&x, &g, w.rms_eps, &mut y);
            n += 1;
            e_ok += (sha(&x) == f[1]) as usize;
            n_ok += (sha(&y) == f[2]) as usize;
        }
        eprintln!("forward oracle: inp_embd {e_ok} of {n} rows and attn_norm-0 {n_ok} of {n} rows bit-exact against the shipped ggml b11192");
        assert_eq!((e_ok, n_ok), (n, n));
    }

    /// Layer 0's Q, K and V for a chat prompt — projections, head norms, RoPE — bit for bit against the shipped ggml.
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + .models/oracle-forward (testing/forward_oracle.py); --release"]
    fn oracle_forward_qkv_rope() {
        use crate::sha256::{hex, Sha256};
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let w = Weights::open(&dir.join("Bonsai-8B-Q1_0.gguf")).unwrap();
        let tsv = std::fs::read_to_string(dir.join("oracle-forward/qkv.tsv")).unwrap();
        let head = tsv.lines().next().unwrap();
        let bits = head.split("bits ").nth(1).unwrap().split(' ').next().unwrap();
        assert_eq!(format!("{:08x}", w.rope.attn_factor.to_bits()), bits, "attn_factor differs from llama.cpp's");
        let toks: Vec<u32> = head.split("tokens ").nth(1).unwrap().split(' ').map(|t| t.parse().unwrap()).collect();
        let sha = |v: &[f32]| {
            let mut h = Sha256::default();
            v.iter().for_each(|x| h.update(&x.to_le_bytes()));
            hex(&h.finish())
        };
        let g = w.f32_vec("blk.0.attn_norm.weight").unwrap();
        let (qd, kd) = (w.n_head * w.head_dim, w.n_head_kv * w.head_dim);
        let mut got = std::collections::HashMap::new();
        let (mut x, mut xn) = (vec![0.0f32; w.n_embd], vec![0.0f32; w.n_embd]);
        let (mut q, mut k, mut v) = (vec![0.0f32; qd], vec![0.0f32; kd], vec![0.0f32; kd]);
        let far: Vec<i32> = head.split("far ").nth(1).unwrap().split(" · ").next().unwrap().split(' ').map(|t| t.parse().unwrap()).collect();
        for (p, &t) in toks.iter().enumerate() {
            w.embed(t, &mut x).unwrap();
            rms_norm_mul(&x, &g, w.rms_eps, &mut xn);
            w.qkv(0, &xn, p as i32, &mut q, &mut k, &mut v).unwrap();
            got.insert(("Q_rope", p), sha(&q));
            got.insert(("K_rope", p), sha(&k));
            got.insert(("V", p), sha(&v));
            w.qkv(0, &xn, far[p], &mut q, &mut k, &mut v).unwrap();
            got.insert(("Q_rope_far", p), sha(&q));
            got.insert(("K_rope_far", p), sha(&k));
        }
        let (mut n, mut ok) = (0, 0);
        for line in tsv.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            let key = match f[0] { "Q_rope" => "Q_rope", "K_rope" => "K_rope", "V" => "V", "Q_rope_far" => "Q_rope_far", "K_rope_far" => "K_rope_far", _ => continue };
            n += 1;
            let good = got[&(key, f[1].parse::<usize>().unwrap())] == f[2];
            ok += good as usize;
            if !good && n - ok <= 4 {
                eprintln!("  {key} position {} differs (ggml first values {})", f[1], f[3]);
            }
        }
        eprintln!("forward oracle: layer-0 Qcur/Kcur/Vcur {ok} of {n} rows ({} tokens at positions 0–{} and {}–{}, YaRN attn_factor {}) bit-exact against the shipped ggml b11192",
                  toks.len(), toks.len() - 1, far[0], far[far.len() - 1], w.rope.attn_factor);
        assert_eq!(ok, n);
    }

    /// Layer 0 whole: attention over a 28-token chat prompt (f16 cache, causal), `wo`, the residual, and the
    /// feed-forward block to `l_out`, bit for bit.
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + .models/oracle-forward (testing/forward_oracle.py); --release"]
    fn oracle_forward_attention() {
        use crate::sha256::{hex, Sha256};
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let w = Weights::open(&dir.join("Bonsai-8B-Q1_0.gguf")).unwrap();
        let tsv = std::fs::read_to_string(dir.join("oracle-forward/qkv.tsv")).unwrap();
        let head = tsv.lines().next().unwrap();
        let bits = head.split("kq_scale bits ").nth(1).unwrap().split(' ').next().unwrap();
        assert_eq!(format!("{:08x}", (1.0f32 / (w.head_dim as f32).sqrt()).to_bits()), bits, "kq_scale differs from llama.cpp's");
        let toks: Vec<u32> = head.split("tokens ").nth(1).unwrap().split(' ').map(|t| t.parse().unwrap()).collect();
        let sha = |v: &[f32]| {
            let mut h = Sha256::default();
            v.iter().for_each(|x| h.update(&x.to_le_bytes()));
            hex(&h.finish())
        };
        let g = w.f32_vec("blk.0.attn_norm.weight").unwrap();
        let (qd, kd) = (w.n_head * w.head_dim, w.n_head_kv * w.head_dim);
        let mut cache = KvCache::new(kd);
        let mut got = std::collections::HashMap::new();
        let (mut x, mut xn) = (vec![0.0f32; w.n_embd], vec![0.0f32; w.n_embd]);
        let (mut q, mut k, mut v) = (vec![0.0f32; qd], vec![0.0f32; kd], vec![0.0f32; kd]);
        let (mut kqv, mut att) = (vec![0.0f32; qd], vec![0.0f32; w.n_embd]);
        for (p, &t) in toks.iter().enumerate() {
            w.embed(t, &mut x).unwrap();
            rms_norm_mul(&x, &g, w.rms_eps, &mut xn);
            w.qkv(0, &xn, p as i32, &mut q, &mut k, &mut v).unwrap();
            cache.push(&k, &v);
            w.attention(0, &q, &cache, &mut kqv, &mut att).unwrap();
            let res: Vec<f32> = att.iter().zip(&x).map(|(a, b)| a + b).collect();
            got.insert(("kqv_out", p), sha(&kqv));
            got.insert(("attn_out", p), sha(&att));
            got.insert(("ffn_inp", p), sha(&res));
            let mut l = res.clone();
            w.ffn(0, &mut l).unwrap();
            got.insert(("l_out", p), sha(&l));
        }
        let (mut n, mut ok) = (0, 0);
        for line in tsv.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            let key = match f[0] { "kqv_out" => "kqv_out", "attn_out" => "attn_out", "ffn_inp" => "ffn_inp", "l_out" => "l_out", _ => continue };
            n += 1;
            let good = got[&(key, f[1].parse::<usize>().unwrap())] == f[2];
            ok += good as usize;
            if !good && n - ok <= 6 {
                eprintln!("  {key} position {} differs (ggml first values {})", f[1], f[3]);
            }
        }
        eprintln!("forward oracle: layer-0 kqv_out/attn_out/ffn_inp/l_out {ok} of {n} rows ({} tokens, causal, f16 K/V) bit-exact against the shipped ggml b11192", toks.len());
        assert_eq!(ok, n);
    }

    /// SwiGLU (`silu(x) · 1`) on 24,600 values over ±120 and the edges, bit for bit against the shipped
    /// `ggml_swiglu_split`: covers ggml's large-|x| expf path, which no prompt reaches.
    #[test]
    #[ignore = "needs .models/oracle-forward/swiglu.tsv (testing/forward_oracle.py)"]
    fn oracle_forward_swiglu_sweep() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let tsv = std::fs::read_to_string(dir.join("oracle-forward/swiglu.tsv")).unwrap();
        let (mut n, mut bad) = (0, Vec::new());
        for line in tsv.lines() {
            let (x, y) = line.split_once('\t').unwrap();
            let x = f32::from_bits(u32::from_str_radix(x, 16).unwrap());
            let want = u32::from_str_radix(y, 16).unwrap();
            let mut o = [0.0f32];
            swiglu(&[x], &[1.0], &mut o);
            n += 1;
            if o[0].to_bits() != want {
                bad.push(format!("x {x:e}: got {:08x} want {want:08x}", o[0].to_bits()));
            }
        }
        eprintln!("forward oracle: swiglu {} of {n} values bit-exact against the shipped ggml b11192 (±120 and the edges)", n - bad.len());
        bad.iter().take(6).for_each(|b| eprintln!("  {b}"));
        assert!(bad.is_empty());
    }

    /// The whole model on a 28-token chat prompt: every layer's `l_out` (36 × 28 rows), `result_norm` and the
    /// logits (151,669 per token), bit for bit against the shipped ggml (testing/model_oracle.py).
    fn model_oracle(stem: &str) {
        use crate::sha256::{hex, Sha256};
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let w = Weights::open(&dir.join(format!("{stem}.gguf"))).unwrap();
        let tsv = std::fs::read_to_string(dir.join(format!("oracle-forward/model-{stem}.tsv"))).unwrap();
        let toks: Vec<u32> = tsv.lines().next().unwrap().split("tokens ").nth(1).unwrap().split(' ').map(|t| t.parse().unwrap()).collect();
        let sha = |v: &[f32]| {
            let mut h = Sha256::default();
            v.iter().for_each(|x| h.update(&x.to_le_bytes()));
            hex(&h.finish())
        };
        let mut want = std::collections::HashMap::new();
        for line in tsv.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            want.insert((f[0].to_string(), if f[0] == "l_out" { f[1].to_string() } else { String::new() }, f[2].parse::<usize>().unwrap()),
                        (f[3].to_string(), f[1].to_string()));
        }
        let mut caches = w.caches();
        let (mut n, mut ok, mut argmax_ok) = (0, 0, 0);
        let mut first_bad: Option<String> = None;
        let t0 = std::time::Instant::now();
        for (p, &t) in toks.iter().enumerate() {
            let mut layer_rows = Vec::new();
            let rn = w.step(&mut caches, t, |il, l| layer_rows.push((il, sha(l)))).unwrap();
            let logits = w.logits(&rn).unwrap();
            let mut check = |kind: &str, key: String, got: String| {
                n += 1;
                let good = want[&(kind.to_string(), key.clone(), p)].0 == got;
                ok += good as usize;
                if !good && first_bad.is_none() {
                    first_bad = Some(format!("{kind} {key} at position {p}"));
                }
            };
            for (il, s) in layer_rows {
                check("l_out", il.to_string(), s);
            }
            check("result_norm", String::new(), sha(&rn));
            check("logits", String::new(), sha(&logits));
            let am = logits.iter().enumerate().fold(0, |b, (i, &v)| if v > logits[b] { i } else { b });
            argmax_ok += (want[&("logits".to_string(), String::new(), p)].1 == am.to_string()) as usize;
        }
        eprintln!("forward oracle: {stem}: whole model {ok} of {n} rows ({} layers' l_out, result_norm, logits × {} tokens; greedy token {argmax_ok} of {}) bit-exact against the shipped ggml b11192 — {:.1} s, {} threads",
                  w.n_layer, toks.len(), toks.len(), t0.elapsed().as_secs_f64(), w.pool.threads());
        if let Some(b) = first_bad {
            eprintln!("  first difference: {b}");
        }
        assert_eq!(ok, n);
    }

    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + its model-*.tsv (testing/model_oracle.py); --release"]
    fn oracle_forward_model() {
        model_oracle("Bonsai-8B-Q1_0");
    }

    /// The ternary model, the same way.
    #[test]
    #[ignore = "needs .models/Ternary-Bonsai-8B-Q2_0_g64.gguf + its model-*.tsv (testing/model_oracle.py); --release"]
    fn oracle_forward_model_ternary() {
        model_oracle("Ternary-Bonsai-8B-Q2_0_g64");
    }

    /// Greedy generation against llama-server b11192 itself (testing/greedy_oracle.py): the same chat prompts, the
    /// same continuations token for token, end-of-turn tokens included.
    fn greedy_oracle(kind: &str, stem: &str) {
        use crate::serve::Json;
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let w = Weights::open(&dir.join(format!("{stem}.gguf"))).unwrap();
        let cases = std::fs::read_to_string(dir.join(format!("oracle-forward/greedy-{kind}{stem}.jsonl"))).unwrap();
        let ids = |v: &Json, k: &str| -> Vec<u32> {
            match v.get(k) {
                Some(Json::Arr(a)) => a.iter().map(|x| match x { Json::Num(n) => *n as u32, _ => panic!("{k}: not a number") }).collect(),
                _ => panic!("no {k}"),
            }
        };
        let argmax = |l: &[f32]| l.iter().enumerate().fold(0, |b, (i, &v)| if v > l[b] { i } else { b }) as u32;
        let (mut n_cases, mut n_same, mut n_tok) = (0, 0, 0);
        let t0 = std::time::Instant::now();
        for line in cases.lines() {
            let v = Json::parse(line).unwrap();
            let (prompt, want) = (ids(&v, "prompt_ids"), ids(&v, "ids"));
            let mut caches = w.caches();
            let mut rn = w.prefill(&mut caches, &prompt, |_, _, _| {}).unwrap();
            // exactly as many tokens as the server generated: its list includes the end-of-turn token when it
            // produced one, and where it stopped otherwise is its stopping policy, not a token choice
            let mut got = Vec::new();
            while got.len() < want.len() {
                let next = argmax(&w.logits(&rn).unwrap());
                got.push(next);
                if got.len() < want.len() {
                    rn = w.decode(&mut caches, next).unwrap();
                }
            }
            n_cases += 1;
            n_tok += want.len();
            let same = got == want;
            n_same += same as usize;
            if !same {
                let at = got.iter().zip(&want).position(|(a, b)| a != b);
                eprintln!("  case {n_cases}: first difference at generated token {at:?} (got {:?}, want {:?})", &got[..got.len().min(12)], &want[..want.len().min(12)]);
            }
        }
        eprintln!("greedy oracle: {kind}{stem}: {n_same} of {n_cases} chat prompts generate llama-server b11192's tokens exactly ({n_tok} tokens, ends of turn included) — {:.0} s, {} threads",
                  t0.elapsed().as_secs_f64(), w.pool.threads());
        assert_eq!(n_same, n_cases);
    }

    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + its greedy-*.jsonl (testing/greedy_oracle.py); --release"]
    fn oracle_greedy_llama_server() {
        greedy_oracle("", "Bonsai-8B-Q1_0");
    }

    /// The ternary model against llama-server running the ternary model.
    #[test]
    #[ignore = "needs .models/Ternary-Bonsai-8B-Q2_0_g64.gguf + its greedy-*.jsonl (testing/greedy_oracle.py); --release"]
    fn oracle_greedy_llama_server_ternary() {
        greedy_oracle("", "Ternary-Bonsai-8B-Q2_0_g64");
    }

    /// ggml's tiled flash attention: layer 0's `kqv_out` for every row of a 150-row micro-batch, bit for bit.
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + .models/oracle-forward/tiled.tsv (testing/forward_oracle.py); --release"]
    fn oracle_forward_attention_tiled() {
        use crate::sha256::{hex, Sha256};
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let w = Weights::open(&dir.join("Bonsai-8B-Q1_0.gguf")).unwrap();
        let tsv = std::fs::read_to_string(dir.join("oracle-forward/tiled.tsv")).unwrap();
        let toks: Vec<u32> = tsv.lines().next().unwrap().split("tokens ").nth(1).unwrap().split(' ').map(|t| t.parse().unwrap()).collect();
        assert_eq!(kernel_for(toks.len(), toks.len(), 3).unwrap(), Kernel::Tiled);
        let sha = |v: &[f32]| {
            let mut h = Sha256::default();
            v.iter().for_each(|x| h.update(&x.to_le_bytes()));
            hex(&h.finish())
        };
        let g = w.f32_vec("blk.0.attn_norm.weight").unwrap();
        let (qd, kd) = (w.n_head * w.head_dim, w.n_head_kv * w.head_dim);
        let mut cache = KvCache::new(kd);
        let (mut x, mut xn) = (vec![0.0f32; w.n_embd], vec![0.0f32; w.n_embd]);
        let (mut q, mut k, mut v) = (vec![0.0f32; qd], vec![0.0f32; kd], vec![0.0f32; kd]);
        // the whole micro-batch's K and V are in the cache before attention, as in llama.cpp; causality is the mask
        let mut qs = Vec::new();
        for (p, &t) in toks.iter().enumerate() {
            w.embed(t, &mut x).unwrap();
            rms_norm_mul(&x, &g, w.rms_eps, &mut xn);
            w.qkv(0, &xn, p as i32, &mut q, &mut k, &mut v).unwrap();
            cache.push(&k, &v);
            qs.push(q.clone());
        }
        let (hd, group, scale) = (w.head_dim, w.n_head / w.n_head_kv, 1.0f32 / (w.head_dim as f32).sqrt());
        let (mut n, mut ok, mut ref_same) = (0, 0, 0);
        let mut kqv = vec![0.0f32; qd];
        for (line, (p, q)) in tsv.lines().skip(1).zip(qs.iter().enumerate()) {
            for (h, (qh, oh)) in q.chunks_exact(hd).zip(kqv.chunks_exact_mut(hd)).enumerate() {
                let off = (h / group) * hd;
                attend_head_tiled(qh, &cache.k[off..], &cache.v[off..], p + 1, cache.width, scale, oh);
            }
            let want = line.split('\t').nth(2).unwrap();
            n += 1;
            ok += (sha(&kqv) == want) as usize;
            // the reference path on the same row, to show the two kernels really differ
            for (h, (qh, oh)) in q.chunks_exact(hd).zip(kqv.chunks_exact_mut(hd)).enumerate() {
                let off = (h / group) * hd;
                attend_head(qh, &cache.k[off..], &cache.v[off..], p + 1, cache.width, scale, oh);
            }
            ref_same += (sha(&kqv) == want) as usize;
        }
        eprintln!("forward oracle: tiled flash attention {ok} of {n} rows bit-exact against the shipped ggml b11192 (the reference kernel would match {ref_same})");
        assert_eq!(ok, n);
    }

    /// Prompts of 64 tokens or more, which llama.cpp computes with the tiled kernel, against llama-server itself.
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + its greedy-long-*.jsonl (testing/greedy_oracle.py --long); --release"]
    fn oracle_greedy_llama_server_long() {
        greedy_oracle("long-", "Bonsai-8B-Q1_0");
    }

    /// ggml's split-KV flash attention: one decode row over 257–1,000 cells at 3 and 4 threads, bit for bit.
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + .models/oracle-forward/{tiled,splitkv}.tsv (testing/forward_oracle.py); --release"]
    fn oracle_forward_attention_split() {
        use crate::sha256::{hex, Sha256};
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let w = Weights::open(&dir.join("Bonsai-8B-Q1_0.gguf")).unwrap();
        let tsv = std::fs::read_to_string(dir.join("oracle-forward/splitkv.tsv")).unwrap();
        let toks: Vec<u32> = tsv.lines().next().unwrap().split("tokens ").nth(1).unwrap().split(' ').map(|t| t.parse().unwrap()).collect();
        let sha = |v: &[f32]| {
            let mut h = Sha256::default();
            v.iter().for_each(|x| h.update(&x.to_le_bytes()));
            hex(&h.finish())
        };
        let g = w.f32_vec("blk.0.attn_norm.weight").unwrap();
        let (qd, kd) = (w.n_head * w.head_dim, w.n_head_kv * w.head_dim);
        let mut prompt_cache = KvCache::new(kd);
        let (mut x, mut xn) = (vec![0.0f32; w.n_embd], vec![0.0f32; w.n_embd]);
        let (mut q, mut k, mut v) = (vec![0.0f32; qd], vec![0.0f32; kd], vec![0.0f32; kd]);
        for (p, &t) in toks.iter().enumerate() {
            w.embed(t, &mut x).unwrap();
            rms_norm_mul(&x, &g, w.rms_eps, &mut xn);
            w.qkv(0, &xn, p as i32, &mut q, &mut k, &mut v).unwrap();
            prompt_cache.push(&k, &v);
        }
        let (hd, group, scale) = (w.head_dim, w.n_head / w.n_head_kv, 1.0f32 / (w.head_dim as f32).sqrt());
        let (mut n, mut ok) = (0, 0);
        let mut kqv = vec![0.0f32; qd];
        for line in tsv.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            let (cells, nth): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
            // the oracle's cells cycle the prompt's K/V rows
            let mut cache = KvCache::new(kd);
            for c in 0..cells {
                let r = c % toks.len();
                cache.k.extend_from_slice(&prompt_cache.k[r * kd..(r + 1) * kd]);
                cache.v.extend_from_slice(&prompt_cache.v[r * kd..(r + 1) * kd]);
            }
            assert_eq!(kernel_for(1, cells, nth).unwrap(), Kernel::Split { padded: padded_kv(cells), nth });
            for (h, (qh, oh)) in q.chunks_exact(hd).zip(kqv.chunks_exact_mut(hd)).enumerate() {
                let off = (h / group) * hd;
                attend_head_split(qh, &cache.k[off..], &cache.v[off..], cells, padded_kv(cells), nth, cache.width, scale, oh);
            }
            n += 1;
            let good = sha(&kqv) == f[3];
            ok += good as usize;
            if !good {
                eprintln!("  {cells} cells, {nth} threads: differs (ggml first values {})", f[4]);
            }
        }
        eprintln!("forward oracle: split-KV flash attention {ok} of {n} decode rows (257–1,000 cells, 3 and 4 threads) bit-exact against the shipped ggml b11192");
        assert_eq!(ok, n);
    }

    /// Continuations past 256 cells — the prompt by the tiled kernel, the decode by the reference kernel and then,
    /// from 257 cells, by the split-KV kernel at llama-server's 3 threads — against llama-server itself.
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + its greedy-deep-*.jsonl (testing/greedy_oracle.py --deep); --release"]
    fn oracle_greedy_llama_server_deep() {
        greedy_oracle("deep-", "Bonsai-8B-Q1_0");
    }

    /// Sampling with a fixed seed against llama-server b11192 itself (testing/sample_oracle.py): the forward pass,
    /// then the sampler chain with the parameters the server reported, token for token.
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + its sample-*.jsonl (testing/sample_oracle.py); --release"]
    fn oracle_sample_llama_server() {
        use crate::sampler::{Params, Sampler};
        use crate::serve::Json;
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let w = Weights::open(&dir.join("Bonsai-8B-Q1_0.gguf")).unwrap();
        let cases = std::fs::read_to_string(dir.join("oracle-forward/sample-Bonsai-8B-Q1_0.jsonl")).unwrap();
        let num = |v: &Json, k: &str| match v.get(k) { Some(Json::Num(n)) => *n, other => panic!("{k}: {other:?}") };
        let ids = |v: &Json, k: &str| -> Vec<u32> {
            match v.get(k) { Some(Json::Arr(a)) => a.iter().map(|x| match x { Json::Num(n) => *n as u32, _ => panic!() }).collect(), _ => panic!("{k}") }
        };
        let (mut n_cases, mut n_same, mut n_tok) = (0, 0, 0);
        let mut prefilled: Option<(Vec<u32>, Vec<KvCache>, Vec<f32>)> = None;
        for line in cases.lines() {
            let v = Json::parse(line).unwrap();
            let (prompt, want, pr) = (ids(&v, "prompt_ids"), ids(&v, "ids"), v.get("params").unwrap());
            // the chain this test reproduces; anything else in the record is neutral or the case is refused
            assert_eq!((num(pr, "typical_p"), num(pr, "xtc_probability"), num(pr, "dry_multiplier"), num(pr, "repeat_penalty"),
                        num(pr, "presence_penalty"), num(pr, "frequency_penalty"), num(pr, "dynatemp_range")),
                       (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0));
            let mut smp = Sampler::new(Params { temp: num(pr, "temperature") as f32, top_k: num(pr, "top_k") as i32,
                                                top_p: num(pr, "top_p") as f32, min_p: num(pr, "min_p") as f32,
                                                min_keep: num(pr, "min_keep") as usize, seed: num(pr, "seed") as u32 }).unwrap();
            // the prompt's caches, computed once per prompt and cloned per case
            if prefilled.as_ref().map(|p| p.0 != prompt).unwrap_or(true) {
                let mut caches = w.caches();
                let rn = w.prefill(&mut caches, &prompt, |_, _, _| {}).unwrap();
                prefilled = Some((prompt.clone(), caches, rn));
            }
            let (_, c0, rn0) = prefilled.as_ref().unwrap();
            let mut caches: Vec<KvCache> = c0.iter().map(|c| KvCache { k: c.k.clone(), v: c.v.clone(), width: c.width }).collect();
            let mut rn = rn0.clone();
            let mut got = Vec::new();
            while got.len() < want.len() {
                let next = smp.sample(&w.logits(&rn).unwrap());
                got.push(next);
                if got.len() < want.len() {
                    rn = w.decode(&mut caches, next).unwrap();
                }
            }
            n_cases += 1;
            n_tok += want.len();
            let same = got == want;
            n_same += same as usize;
            if !same {
                let at = got.iter().zip(&want).position(|(a, b)| a != b);
                eprintln!("  case {n_cases} (temp {}, top_k {}, top_p {}, min_p {}, seed {}): first difference at token {at:?}",
                          num(pr, "temperature"), num(pr, "top_k"), num(pr, "top_p"), num(pr, "min_p"), num(pr, "seed"));
            }
        }
        eprintln!("sample oracle: {n_same} of {n_cases} seeded continuations identical to llama-server b11192's ({n_tok} tokens; top-k, top-p, min-p, temperature, the mt19937 draw)");
        assert_eq!(n_same, n_cases);
    }
}
