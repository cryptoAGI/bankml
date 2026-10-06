// SPDX-License-Identifier: MIT OR Apache-2.0
//! The forward pass: token ids to logits, bit-exact against llama.cpp b11192.
//!
//! Reproduces, operation by operation and in the same float order, the CPU graphs of `llm_build_qwen3` and
//! `llm_build_llama` (`ggml/src/ggml-cpu/ops.cpp`), over Q1_0, Q2_0_g64 or F16 weights, with llama.cpp's
//! micro-batching, attention-kernel choice and f16 or q8_0 K/V cache. The oracle is the shipped ggml, not the C
//! source: where GCC contracts an expression into an FMA, the code writes the same `mul_add`.
//! `plan` decides from the header alone whether a file is in scope; anything else is refused with the reason.
//! Details: docs/modules/forward.md.

use crate::gguf::{guard_file, Engine, Header, Mmap, TensorInfo, Val};
use crate::par::Pool;
use crate::q1_0::{self, f16_to_f32, f32_to_f16, Q8Act, Q1_0_BYTES, QK1_0};
use crate::q2_0::{self, Q8Act2, Q2_0_BYTES, QK2_0};
use std::path::Path;

/// The architectures the forward pass plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// `llm_build_qwen3`: per-head RMS norms on Q and K, RoPE in NEOX mode (YaRN when the header asks for it)
    Qwen3,
    /// `llm_build_llama`: no Q/K norms, RoPE in NORM mode (adjacent pairs), no biases, no rope factors
    Llama,
}

/// What bankML's forward pass needs to know about a file before a weight is read, or why it does not play it.
#[derive(Debug, Clone)]
pub struct Plan {
    pub arch: Arch,
    /// every matrix's type: `TYPE_Q1_0`, `TYPE_Q2_0` or `TYPE_F16`
    pub wtype: u32,
    /// the matrix the logits come from: `output.weight`, or `token_embd.weight` when the embeddings are tied
    pub output: &'static str,
}

/// From the header alone: the architecture, the weight type and the graph's options this forward pass reproduces.
/// Anything outside them is an `Err` naming the tensor or key and the reason (docs/OLLAMA.md).
pub fn plan(h: &Header) -> Result<Plan, String> {
    let arch_s = match h.kv.get("general.architecture") {
        Some(Val::S(a)) => a.clone(),
        _ => return Err("no general.architecture".into()),
    };
    let arch = match arch_s.as_str() {
        "qwen3" => Arch::Qwen3,
        "llama" => Arch::Llama,
        a => return Err(format!("architecture {a}: the forward pass plays Qwen3 and Llama (each new architecture needs its own oracle set)")),
    };
    let num = |k: &str| match h.kv.get(&format!("{arch_s}.{k}")) {
        Some(Val::U(v)) => Some(*v as f64),
        Some(Val::I(v)) => Some(*v as f64),
        Some(Val::F(v)) => Some(*v),
        _ => None,
    };
    let emb = h.tensors.iter().find(|t| t.name == "token_embd.weight").ok_or("no token_embd.weight")?;
    let wtype = emb.ty;
    if ![TYPE_Q1_0, TYPE_Q2_0, TYPE_F16].contains(&wtype) {
        return Err(format!("weights are {}: the forward pass runs Q1_0, Q2_0_g64 and F16 (Q8_0, BF16 and Q4_K are O3)", crate::gguf::type_name(wtype)));
    }
    let has = |n: &str| h.tensors.iter().any(|t| t.name == n);
    let qk_norm = has("blk.0.attn_q_norm.weight");
    for t in &h.tensors {
        let kind = t.name.strip_prefix("blk.").and_then(|r| r.split_once('.')).map(|(_, k)| k).unwrap_or(&t.name);
        let known = match kind {
            "token_embd.weight" | "output.weight" | "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "attn_output.weight"
            | "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight" => Some(true),
            "output_norm.weight" | "attn_norm.weight" | "ffn_norm.weight" => Some(false),
            "attn_q_norm.weight" | "attn_k_norm.weight" if arch == Arch::Qwen3 => Some(false),
            _ => None,
        };
        match known {
            None => {
                return Err(format!("tensor {}: not part of the {arch_s} graph this forward pass reproduces (biases, fused QKV, rope factors, \
                                    experts and Q/K norms on Llama are not proven)", t.name))
            }
            Some(true) if t.ty != wtype => {
                return Err(format!("{} is {} while token_embd is {}: mixed weight types are not proven", t.name, crate::gguf::type_name(t.ty),
                                   crate::gguf::type_name(wtype)))
            }
            Some(false) if t.ty != 0 => return Err(format!("{} is {}, not F32", t.name, crate::gguf::type_name(t.ty))),
            _ => {}
        }
    }
    if arch == Arch::Qwen3 && !qk_norm {
        return Err("a Qwen3 file without attn_q_norm".into());
    }
    if arch == Arch::Llama {
        if let Some(Val::S(t)) = h.kv.get(&format!("{arch_s}.rope.scaling.type")) {
            if t != "none" {
                return Err(format!("RoPE scaling {t} on Llama: not proven (the oracle covers plain RoPE)"));
            }
        }
        if num("attention.scale").is_some() || num("expert_count").is_some_and(|e| e > 0.0) {
            return Err("a Llama variant (attention scale or experts): not proven".into());
        }
    }
    let n_embd = emb.dims[0] as f64;
    let n_head = num("attention.head_count").ok_or("no head count")?;
    let head_dim = num("attention.key_length").unwrap_or(n_embd / n_head);
    if num("attention.value_length").is_some_and(|v| v != head_dim) || num("rope.dimension_count").is_some_and(|r| r != head_dim) {
        return Err("partial RoPE or a value head of another width: not proven".into());
    }
    if !(head_dim as usize).is_multiple_of(32) {
        return Err(format!("head width {head_dim}: the attention kernels reproduced here need a multiple of 32"));
    }
    Ok(Plan { arch, wtype, output: if has("output.weight") { "output.weight" } else { "token_embd.weight" } })
}

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

/// RoPE as llama.cpp's `rope_ext` applies it: NEOX pairs (i, i + n/2) for Qwen3, with YaRN; NORM pairs (i, i + 1)
/// for Llama (`rotate_pairs` with `scale` 1, the same arithmetic). The C source
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
    /// NEOX mode (Qwen3) or NORM mode (Llama)
    pub neox: bool,
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
            neox: true,
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

    /// `rotate_pairs` over one head, in place, from a cache built by `cache`: NEOX mode pairs (i/2, i/2 + n/2), NORM
    /// mode pairs (i, i + 1).
    pub fn apply(&self, cache: &[f32], head: &mut [f32]) {
        let (half, neox) = (self.n_dims / 2, self.neox);
        for i0 in (0..self.n_dims).step_by(2) {
            let (ic, off) = if neox { (i0 / 2, half) } else { (i0, 1) };
            let (c, s) = (cache[i0], cache[i0 + 1]);
            let (x0, x1) = (head[ic], head[ic + off]);
            // as compiled: x1's products rounded, x0's fused (vmulss + vfmsub231ss / vfmadd132ss)
            head[ic] = c.mul_add(x0, -(x1 * s));
            head[ic + off] = s.mul_add(x0, x1 * c);
        }
    }
}

/// `ggml_vec_dot_f16` as the AVX2 build computes it: four 8-lane f32 accumulators fed by FMAs over 32-element
/// steps, reduced as `(a0 + a2) + (a1 + a3)`, then low half + high half, then two horizontal adds. `n % 32 == 0`.
pub fn dot_f16(x: &[u16], y: &[u16]) -> f32 {
    assert!(x.len() == y.len() && x.len().is_multiple_of(32));
    crate::f16::vec_dot_u16(x, y)
}

/// `dot_f16` written out (the definition the oracle tests pinned; `f16::vec_dot` has its bits on every path).
pub fn dot_f16_ref(x: &[u16], y: &[u16]) -> f32 {
    assert!(x.len() == y.len() && x.len().is_multiple_of(32));
    let mut acc = [[0.0f32; 8]; 4];
    for (xs, ys) in x.as_chunks::<32>().0.iter().zip(y.as_chunks::<32>().0.iter()) {
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
    let q16 = crate::f16::to_f16(q);
    let mut acc = vec![0u16; hd];
    let (mut sum, mut max) = (0.0f32, f32::NEG_INFINITY);
    for ic in from..to {
        let s = dot_f16(&k[ic * stride..ic * stride + hd], &q16) * scale;
        let (mut ms, mut vs) = (1.0f32, 1.0f32);
        if s > max {
            let old = max;
            max = s;
            ms = (old - max).exp();
            crate::f16::scale(&mut acc, ms); // vec_scale_f16: f16(acc · ms)
        } else {
            vs = (s - max).exp();
        }
        crate::f16::mad(&mut acc, &v[ic * stride..ic * stride + hd], vs); // vec_mad_f16: f16(fma(v, vs, acc))
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
    attend_heads_tiled(&[q], &[n_kv], k, v, stride, scale, &mut [out]);
}

/// `attend_head_tiled` for several rows of one head at once: each 64-cell tile's keys and values are widened
/// (and the keys transposed) once for the block, then every row that sees the tile takes its step with its own state,
/// in the per-row order — so each row has the per-row bits. Compiled with FMA and F16C where the CPU has them.
pub fn attend_heads_tiled(qs: &[&[f32]], n_kv: &[usize], k: &[u16], v: &[u16], stride: usize, scale: f32, outs: &mut [&mut [f32]]) {
    #[cfg(target_arch = "x86_64")]
    if crate::q1_0::has_avx2() {
        // SAFETY: AVX2 + FMA + F16C checked
        return unsafe { attend_heads_tiled_avx2(qs, n_kv, k, v, stride, scale, outs) };
    }
    tiled_body(qs, n_kv, k, v, stride, scale, outs)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn attend_heads_tiled_avx2(qs: &[&[f32]], n_kv: &[usize], k: &[u16], v: &[u16], stride: usize, scale: f32, outs: &mut [&mut [f32]]) {
    tiled_body(qs, n_kv, k, v, stride, scale, outs)
}

#[inline(always)]
fn tiled_body(qs: &[&[f32]], n_kv: &[usize], k: &[u16], v: &[u16], stride: usize, scale: f32, outs: &mut [&mut [f32]]) {
    const T: usize = 64;
    let hd = qs[0].len();
    let rows = qs.len();
    // per row: the f32 accumulator, the running sum and max (ggml's tiled kernel state)
    let mut acc = vec![0.0f32; rows * hd];
    let mut st = vec![(0.0f32, f32::NEG_INFINITY); rows];
    // the tile's keys widened and transposed (kt[d·T + cell]) and its values widened (vt[cell·hd + d])
    let (mut kt, mut vt) = (vec![0.0f32; hd * T], vec![0.0f32; T * hd]);
    let mut row = vec![0.0f32; hd];
    let most = n_kv.iter().copied().max().unwrap_or(0);
    for ic in (0..most).step_by(T) {
        let width = T.min(most - ic);
        for tk in 0..width {
            crate::f16::widen(&k[(ic + tk) * stride..(ic + tk) * stride + hd], &mut row);
            for (d, &x) in row.iter().enumerate() {
                kt[d * T + tk] = x;
            }
            crate::f16::widen(&v[(ic + tk) * stride..(ic + tk) * stride + hd], &mut vt[tk * hd..(tk + 1) * hd]);
        }
        for r in 0..rows {
            if ic >= n_kv[r] {
                continue; // this row sees none of the tile: its loop has already ended
            }
            let tile = T.min(n_kv[r] - ic);
            let mut chains = [0.0f32; T];
            for (d, &qd) in qs[r].iter().enumerate() {
                for (c, &kd) in chains.iter_mut().zip(&kt[d * T..d * T + tile]) {
                    *c = kd.mul_add(qd, *c); // each cell's own chain over the head dimension, in order
                }
            }
            let mut kq = [0.0f32; T];
            for (tk, s) in kq.iter_mut().enumerate() {
                *s = if tk < tile {
                    chains[tk] * scale + 0.0 // the mask adds 0 to a visible cell
                } else {
                    f32::NEG_INFINITY
                };
            }
            let tile_max = kq.iter().fold(f32::NEG_INFINITY, |m, &x| if m > x { m } else { x });
            if tile_max == f32::NEG_INFINITY {
                continue;
            }
            let (sum, max) = &mut st[r];
            let a = &mut acc[r * hd..(r + 1) * hd];
            let new_max = max.max(tile_max);
            if new_max > *max {
                let ms = (*max - new_max).exp();
                for x in a.iter_mut() {
                    *x *= ms;
                }
                *sum *= ms;
            }
            *max = new_max;
            let mut tsum = 0.0f64;
            for g in kq.as_chunks_mut::<8>().0.iter_mut() {
                for x in g.iter_mut() {
                    *x = v_expf(*x - new_max);
                }
                let h: [f32; 4] = std::array::from_fn(|l| g[l + 4] + g[l]);
                tsum += ((h[0] + h[2]) + (h[1] + h[3])) as f64;
            }
            *sum = (*sum as f64 + tsum) as f32;
            for (tk, &p) in kq.iter().enumerate().take(tile) {
                for (x, &vd) in a.iter_mut().zip(&vt[tk * hd..(tk + 1) * hd]) {
                    *x = vd.mul_add(p, *x);
                }
            }
        }
    }
    for (r, o) in outs.iter_mut().enumerate() {
        let sum = st[r].0;
        let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
        for (x, &a) in o.iter_mut().zip(&acc[r * hd..(r + 1) * hd]) {
            *x = a * inv;
        }
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
#[inline(always)]
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
    #[cfg(target_arch = "x86_64")]
    if crate::q1_0::has_avx2() {
        // SAFETY: AVX2 + FMA checked
        return unsafe { swiglu_avx2(gate, up, out) };
    }
    swiglu_body(gate, up, out)
}

/// `swiglu` compiled with FMA (the polynomial's `mul_add`s become instructions, not libm calls): the same bits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_avx2(gate: &[f32], up: &[f32], out: &mut [f32]) {
    swiglu_body(gate, up, out)
}

#[inline(always)]
fn swiglu_body(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = g / (1.0 + v_expf(0.0 - g)) * u;
    }
}

/// The type of the KV cache's cells: llama.cpp's `--cache-type-k/-v` (both the same here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvType {
    /// f16, llama.cpp's default
    #[default]
    F16,
    /// q8_0 (blocks of 32: an f16 scale and 32 signed bytes), 34 bytes per 32 values against f16's 64
    Q8_0,
}

impl KvType {
    pub fn parse(s: &str) -> Result<KvType, String> {
        match s {
            "f16" => Ok(KvType::F16),
            "q8_0" => Ok(KvType::Q8_0),
            _ => Err(format!("cache type {s:?}: bankML keeps f16 or q8_0")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
        }
    }
}

/// One layer's K/V cache, one row of `n_head_kv · head_dim` per position (llama.cpp's `cache_k_l*`/`cache_v_l*`
/// without the transposed-V layout, which flash attention does not use): f16 in `k`/`v`, or q8_0 blocks in
/// `kq`/`vq`.
#[derive(Clone)]
pub struct KvCache {
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub width: usize,
    pub kind: KvType,
    /// one head's width: the K rotation's block (q8_0 only)
    pub head_dim: usize,
    pub kq: Vec<u8>,
    pub vq: Vec<u8>,
}

impl KvCache {
    pub fn new(width: usize) -> Self {
        Self::of(KvType::F16, width)
    }
    pub fn of(kind: KvType, width: usize) -> Self {
        KvCache { k: Vec::new(), v: Vec::new(), width, kind, head_dim: 128, kq: Vec::new(), vq: Vec::new() }
    }
    /// A q8_0 row's bytes.
    pub fn row_bytes(&self) -> usize {
        self.width / q1_0::QK8_0 * q1_0::Q8_0_BYTES
    }
    pub fn len(&self) -> usize {
        match self.kind {
            KvType::F16 => self.k.len() / self.width,
            KvType::Q8_0 => self.kq.len() / self.row_bytes(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The memory the cells hold, in bytes.
    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 2 + self.kq.len() + self.vq.len()
    }
    /// Keep the first `n` positions (llama.cpp's `seq_rm` of the tail, as its prompt cache does).
    pub fn truncate(&mut self, n: usize) {
        let rb = self.row_bytes();
        self.k.truncate(n * self.width);
        self.v.truncate(n * self.width);
        self.kq.truncate(n * rb);
        self.vq.truncate(n * rb);
    }

    /// Store a position's K and V as the cache stores them: `set_rows` f32 → f16 (round to nearest even), or f32 →
    /// q8_0 with the CPU's `from_float` (ggml's AVX2 `quantize_row_q8_0`).
    pub fn push(&mut self, k: &[f32], v: &[f32]) {
        match self.kind {
            KvType::F16 => {
                self.k.extend(k.iter().map(|&x| f32_to_f16(x)));
                self.v.extend(v.iter().map(|&x| f32_to_f16(x)));
            }
            KvType::Q8_0 => {
                // llama.cpp rotates K and V before they reach a quantized cache (the attention rotates Q to match)
                let rb = self.row_bytes();
                let (mut k, mut v) = (k.to_vec(), v.to_vec());
                if rotates(self.head_dim) {
                    fwht(&mut k, rot_k_size(self.head_dim));
                    fwht(&mut v, ROT_V);
                }
                for (src, dst) in [(&k[..], &mut self.kq), (&v[..], &mut self.vq)] {
                    let at = dst.len();
                    dst.resize(at + rb, 0);
                    q1_0::quantize_row_q8_0(src, &mut dst[at..]);
                }
            }
        }
    }
}

/// The Hadamard rotation llama.cpp b11192 applies around a quantized KV cache (`attn_rot_k`/`attn_rot_v`,
/// `llama_mul_mat_hadamard`, computed by ggml's CPU as `ggml_compute_forward_fwht`): each `n`-wide block scaled by
/// `1/sqrtf(n)`, then the butterflies `(u + v, u − v)` with the span doubling from 1 (Sylvester order; ggml's SIMD
/// passes write `u − v` as `fma(v, −1, u)`, the same single rounding). The rotation is its own inverse.
pub fn fwht(x: &mut [f32], n: usize) {
    let scale = 1.0f32 / (n as f32).sqrt();
    for row in x.chunks_exact_mut(n) {
        for v in row.iter_mut() {
            *v *= scale;
        }
        let mut len = 1;
        while len < n {
            for i in (0..n).step_by(2 * len) {
                for j in i..i + len {
                    let (u, v) = (row[j], row[j + len]);
                    row[j] = u + v;
                    row[j + len] = u - v;
                }
            }
            len <<= 1;
        }
    }
}

/// llama.cpp's rotation sizes: K (and Q) by the largest power of two from 128 that divides the head size (`nrot`
/// doubled from 64 while it divides), V by 64.
pub fn rot_k_size(head_dim: usize) -> usize {
    let mut n = 64;
    while head_dim.is_multiple_of(n * 2) {
        n *= 2;
    }
    n
}
pub const ROT_V: usize = 64;

/// Whether llama.cpp rotates a quantized cache for this head size (`attn_rot_k`/`attn_rot_v`: a multiple of 64;
/// below that the blocks would cross heads, and llama.cpp leaves the cache unrotated).
pub fn rotates(head_dim: usize) -> bool {
    head_dim.is_multiple_of(64)
}

/// ggml's AVX2 `ggml_vec_dot_q8_0_q8_0`, float op for float op: per block the combined scale `dx · dy`; eight lanes,
/// lane l the exact integer sum of elements 4l..4l+3 (`maddubs` of |x| and y signed by x, saturating to i16 per pair,
/// then `madd`); `acc = fma(d, lane, acc)`; then `hsum_float_8`.
pub fn vec_dot_q8_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use q1_0::{Q8_0_BYTES, QK8_0};
    let mut acc = [0.0f32; 8];
    for ib in 0..n / QK8_0 {
        let (xb, yb) = (&x[ib * Q8_0_BYTES..(ib + 1) * Q8_0_BYTES], &y[ib * Q8_0_BYTES..(ib + 1) * Q8_0_BYTES]);
        let d = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]])) * f16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
        for (l, a) in acc.iter_mut().enumerate() {
            let mut s = 0i32;
            for p in 0..2 {
                let mut pair = 0i32;
                for e in [4 * l + 2 * p, 4 * l + 2 * p + 1] {
                    let (xi, yi) = (xb[2 + e] as i8, yb[2 + e] as i8);
                    let ax = xi.unsigned_abs() as i32; // _mm256_sign_epi8(x, x) read unsigned: |−128| = 128
                    let sy = if xi < 0 { yi.wrapping_neg() } else if xi == 0 { 0 } else { yi } as i32; // _mm256_sign_epi8(y, x)
                    pair += ax * sy;
                }
                s += pair.clamp(i16::MIN as i32, i16::MAX as i32); // maddubs saturates each pair to i16
            }
            *a = d.mul_add(s as f32, *a);
        }
    }
    ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]))
}

/// One query head against a causal run of q8_0 cached keys and values (`rb` bytes a row, the head's blocks at the
/// start of each slice), as ggml's `flash_attn_ext_f16_one_chunk` computes it for a q8_0 cache: Q quantized to q8_0
/// (`from_float`); each score `vec_dot_q8_0_q8_0 · scale`; an f32 accumulator (`vec_scale_f32`: `acc · ms`; V
/// dequantized as `qs · d`, then `vec_mad_f32`: `fma(v, vs, acc)`); finally `acc · (1/S)`.
pub fn attend_head_q8(q: &[f32], k: &[u8], v: &[u8], n_kv: usize, rb: usize, scale: f32, out: &mut [f32]) {
    let hd = q.len();
    let hb = hd / q1_0::QK8_0 * q1_0::Q8_0_BYTES;
    let mut qq = vec![0u8; hb];
    q1_0::quantize_row_q8_0(q, &mut qq);
    let (mut acc, mut vrow) = (vec![0.0f32; hd], vec![0.0f32; hd]);
    let (mut sum, mut max) = (0.0f32, f32::NEG_INFINITY);
    for ic in 0..n_kv {
        let s = vec_dot_q8_0_q8_0(hd, &k[ic * rb..ic * rb + hb], &qq) * scale;
        let (mut ms, mut vs) = (1.0f32, 1.0f32);
        if s > max {
            let old = max;
            max = s;
            ms = (old - max).exp();
            for a in acc.iter_mut() {
                *a *= ms;
            }
        } else {
            vs = (s - max).exp();
        }
        for (blk, vb) in vrow.as_chunks_mut::<{ q1_0::QK8_0 }>().0.iter_mut().zip(v[ic * rb..ic * rb + hb].as_chunks::<{ q1_0::Q8_0_BYTES }>().0) {
            let d = f16_to_f32(u16::from_le_bytes([vb[0], vb[1]]));
            for (y, &b) in blk.iter_mut().zip(&vb[2..]) {
                *y = (b as i8) as f32 * d;
            }
        }
        for (a, &x) in acc.iter_mut().zip(&vrow) {
            *a = x.mul_add(vs, *a);
        }
        sum = sum * ms + vs;
    }
    let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
    for (o, &a) in out.iter_mut().zip(&acc) {
        *o = a * inv;
    }
}

/// GGUF tensor types bankml's forward pass runs.
pub const TYPE_F16: u32 = 1;
pub const TYPE_Q1_0: u32 = 41;
pub const TYPE_Q2_0: u32 = 42;

/// An activation row quantized to q8_0 and prepared for the model's weight type (shared by every matrix that reads
/// the same row: Q, K and V; gate and up).
pub enum Act {
    Q1(Q8Act),
    Q2(Q8Act2),
    /// an F16 model's activation, rounded to f16 (`ggml_cpu_fp32_to_fp16`)
    F16(Vec<u16>),
}

/// Activation rows prepared for the model's weight type: one per token of a micro-batch.
pub enum Acts {
    Q1(Vec<Q8Act>),
    Q2(Vec<Q8Act2>),
    F16(Vec<Vec<u16>>),
}

/// A weight matrix: its bytes, rows and type.
pub struct Mat<'a> {
    name: &'a str,
    bytes: &'a [u8],
    pub rows: usize,
    ty: u32,
}

fn row_bytes(ty: u32, n: usize) -> usize {
    match ty {
        TYPE_Q1_0 => n / QK1_0 * Q1_0_BYTES,
        TYPE_F16 => n * crate::f16::F16_BYTES,
        _ => n / QK2_0 * Q2_0_BYTES,
    }
}

/// Which rows of a prompt's last micro-batch leave the last layer: llama-server asks for the last token's logits
/// only (`inp_out_ids`); the model oracle asks for every row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outputs {
    Last,
    All,
}

/// The model's tensors by name, read through bankml's own memory map.
pub struct Weights {
    mm: Mmap,
    tensors: Vec<TensorInfo>,
    /// tensor name → index into `tensors`
    index: std::collections::HashMap<String, usize>,
    /// every F32 tensor (the norms), read once at open
    vecs: std::collections::HashMap<String, Vec<f32>>,
    data_start: u64,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub rms_eps: f32,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope: Rope,
    pub n_layer: usize,
    /// the matrices' type: `TYPE_Q1_0`, `TYPE_Q2_0` or `TYPE_F16`
    pub wtype: u32,
    pub arch: Arch,
    /// the matrix the logits come from (`token_embd.weight` when the embeddings are tied)
    pub output: &'static str,
    /// the thread count of the llama.cpp being matched (`-t`); the split-KV decode kernel cuts the KV cells into one
    /// chunk per thread, so the bits depend on it. `BANKML_LLAMA_THREADS`, default 3.
    pub llama_threads: usize,
    pool: Pool,
    /// a verified GPU taking a share of every 1-bit matrix–vector product's rows (`gpu::worker`), if one was found
    pub gpu: Option<std::sync::Mutex<crate::gpu::worker::Worker>>,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self, String> {
        let rep = guard_file(path, Engine::Mainline).map_err(|e| format!("{}: {e}", path.display()))?;
        let h = rep.header.ok_or("no GGUF header")?;
        let plan = plan(&h).map_err(|e| format!("{}: {e}", path.display()))?;
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
        let mut rope = Rope::new(head_dim, num("rope.freq_base").unwrap_or(10000.0) as f32, yarn, n_ctx_orig);
        rope.neox = plan.arch == Arch::Qwen3;
        let n_layer = num("block_count").ok_or("no block count")? as usize;
        let wtype = plan.wtype;
        let mm = Mmap::open(path).map_err(|e| e.to_string())?;
        let pool = Pool::from_env();
        // a card joins only for the 1-bit kernel (the ternary kernel is next), only after it passes the on-card
        // oracle, and only if the calibration gives it a share worth sending; any failure leaves the CPU path alone
        let gpu = if wtype == TYPE_Q1_0 {
            match crate::gpu::worker::Worker::open(&pool) {
                Ok(Some(w)) => {
                    crate::log(crate::LOG_INFO, &format!("bankml: GPU {} verified; it computes {:.0}% of each 1-bit matrix's rows", w.name, w.share * 100.0));
                    Some(std::sync::Mutex::new(w))
                }
                Ok(None) => None,
                Err(e) => {
                    crate::log(crate::LOG_WARN, &format!("bankml: GPU not used: {e}"));
                    None
                }
            }
        } else {
            None
        };
        let index = h.tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        let mut vecs = std::collections::HashMap::new();
        for t in h.tensors.iter().filter(|t| t.ty == 0) {
            let n = t.dims.iter().product::<u64>() as usize;
            let b = mm.bytes().get((h.data_start + t.offset) as usize..).and_then(|b| b.get(..n * 4)).ok_or_else(|| format!("{} truncated", t.name))?;
            vecs.insert(t.name.clone(), b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect::<Vec<f32>>());
        }
        Ok(Weights { mm, index, vecs, tensors: h.tensors, data_start: h.data_start, n_embd, n_vocab, rms_eps, n_head, n_head_kv, head_dim, rope, n_layer, wtype,
                     arch: plan.arch, output: plan.output,
                     llama_threads: std::env::var("BANKML_LLAMA_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(3),
                     pool, gpu })
    }

    fn tensor(&self, name: &str) -> Result<(&TensorInfo, &[u8]), String> {
        let t = self.index.get(name).map(|&i| &self.tensors[i]).ok_or_else(|| format!("no tensor {name}"))?;
        let start = (self.data_start + t.offset) as usize;
        Ok((t, self.mm.bytes().get(start..).ok_or("tensor outside the file")?))
    }

    /// An f32 vector tensor (norm weights), copied.
    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>, String> {
        self.norm(name).map(<[f32]>::to_vec)
    }

    /// An f32 vector tensor (norm weights), as read at open.
    pub fn norm(&self, name: &str) -> Result<&[f32], String> {
        self.vecs.get(name).map(Vec::as_slice).ok_or_else(|| format!("no F32 tensor {name}"))
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
        match self.wtype {
            TYPE_Q1_0 => Act::Q1(Q8Act::quantize(x)),
            TYPE_F16 => Act::F16(crate::f16::to_f16(x)),
            _ => Act::Q2(Q8Act2::quantize(x)),
        }
    }

    /// Every row of `xs` quantized to q8_0, prepared for the model's kernel.
    pub fn quantize_rows(&self, xs: &[Vec<f32>]) -> Acts {
        match self.wtype {
            TYPE_Q1_0 => Acts::Q1(xs.iter().map(|x| Q8Act::quantize(x)).collect()),
            TYPE_F16 => Acts::F16(xs.iter().map(|x| crate::f16::to_f16(x)).collect()),
            _ => Acts::Q2(xs.iter().map(|x| Q8Act2::quantize(x)).collect()),
        }
    }

    /// `m` times every prepared row, one matrix–matrix product on the pool; returns one output row per input row.
    /// Each element has the bits of the per-pair dot, so the rows equal `mv` on each row.
    pub fn mm(&self, m: &Mat, a: &Acts) -> Result<Vec<Vec<f32>>, String> {
        let n = match a { Acts::Q1(v) => v.len(), Acts::Q2(v) => v.len(), Acts::F16(v) => v.len() };
        let mut out = vec![0.0f32; m.rows * n];
        match (m.ty, a) {
            (TYPE_Q1_0, Acts::Q1(cols)) => q1_0::mat_mul_act_par(&self.pool, m.bytes, m.rows, cols, &mut out),
            (TYPE_Q2_0, Acts::Q2(cols)) => q2_0::mat_mul_par(&self.pool, m.bytes, m.rows, cols, &mut out),
            // one column takes ggml_vec_dot_f16, two or more llamafile's tinyBLAS (f16.rs): mat_mul_par chooses as ggml does
            (TYPE_F16, Acts::F16(cols)) => crate::f16::mat_mul_par(&self.pool, m.bytes, m.rows, cols, &mut out),
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
            (TYPE_F16, Act::F16(a)) => crate::f16::mat_vec_par(&self.pool, m.bytes, m.rows, a, out),
            _ => return Err("activation prepared for another weight type".into()),
        }
        Ok(())
    }

    /// Several matrices by the same prepared row: one pass of the pool for F16 (`f16::mat_vec_multi_par`), each
    /// matrix in turn otherwise. The same bits as `mv` on each.
    pub fn mv_many(&self, ms: &[&Mat], a: &Act, outs: &mut [&mut [f32]]) -> Result<(), String> {
        if let Act::F16(a) = a {
            if ms.iter().zip(outs.iter()).all(|(m, o)| m.ty == TYPE_F16 && o.len() == m.rows) {
                let ws: Vec<(&[u8], usize)> = ms.iter().map(|m| (m.bytes, m.rows)).collect();
                crate::f16::mat_vec_multi_par(&self.pool, &ws, a, outs);
                return Ok(());
            }
        }
        for (m, o) in ms.iter().zip(outs.iter_mut()) {
            self.mv(m, a, o).map_err(|e| format!("{}: {e}", m.name))?;
        }
        Ok(())
    }

    /// Layer `il`'s attention inputs for one token at position `pos`, from its normed row `xn` (`attn_norm-il`):
    /// Q (`n_head · head_dim`), K and V (`n_head_kv · head_dim`) — projected, Q and K normed per head (Qwen3 only)
    /// and roped, as `Qcur`, `Kcur`, `Vcur` in llama.cpp's graph.
    pub fn qkv(&self, il: usize, xn: &[f32], pos: i32, q: &mut [f32], k: &mut [f32], v: &mut [f32]) -> Result<(), String> {
        let a = self.quantize(xn);
        let hd = self.head_dim;
        let mut cache = vec![0.0f32; hd];
        self.rope.cache(pos, &mut cache);
        let mut tmp = vec![0.0f32; hd];
        let qk_norm = self.arch == Arch::Qwen3;
        let (wq, wk, wv) = (self.matrix(&format!("blk.{il}.attn_q.weight"), self.n_embd)?, self.matrix(&format!("blk.{il}.attn_k.weight"), self.n_embd)?,
                            self.matrix(&format!("blk.{il}.attn_v.weight"), self.n_embd)?);
        self.mv_many(&[&wq, &wk, &wv], &a, &mut [&mut *q, &mut *k, &mut *v])?;
        for (norm, out) in [(Some("attn_q_norm"), &mut *q), (Some("attn_k_norm"), &mut *k), (None, &mut *v)] {
            if let Some(norm) = norm {
                let g = if qk_norm { Some(self.norm(&format!("blk.{il}.{norm}.weight"))?) } else { None };
                for h in out.chunks_exact_mut(hd) {
                    if let Some(g) = g {
                        tmp.copy_from_slice(h);
                        rms_norm_mul(&tmp, g, self.rms_eps, h);
                    }
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
        kqv.copy_from_slice(&self.attend_rows(&[q.to_vec()], cache, visible - 1, kernel)[0]);
    }

    /// Every head of every query row, row `i` seeing the cells up to position `p0 + i` (causal), on the pool: a head is
    /// computed whole by one worker, so the bits do not depend on the thread count. Returns the rows' `kqv`.
    fn attend_rows(&self, qs: &[Vec<f32>], cache: &KvCache, p0: usize, kernel: Kernel) -> Vec<Vec<f32>> {
        let (hd, nh) = (self.head_dim, self.n_head);
        let group = nh / self.n_head_kv;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; qs.len() * nh * hd];
        let one = |j: usize, oh: &mut [f32]| {
            let (i, h) = (j / nh, j % nh);
            let (qh, off, visible) = (&qs[i][h * hd..(h + 1) * hd], (h / group) * hd, p0 + i + 1);
            if cache.kind == KvType::Q8_0 {
                // ggml takes the reference kernel whenever K is not f16/f32 (no tiled, no split-KV)
                let (rb, ob) = (cache.row_bytes(), off / q1_0::QK8_0 * q1_0::Q8_0_BYTES);
                let mut qr = qh.to_vec();
                if rotates(hd) {
                    fwht(&mut qr, rot_k_size(hd));
                }
                attend_head_q8(&qr, &cache.kq[ob..], &cache.vq[ob..], visible, rb, scale, oh);
                if rotates(hd) {
                    fwht(oh, ROT_V);
                }
                return;
            }
            match kernel {
                Kernel::Reference => attend_head(qh, &cache.k[off..], &cache.v[off..], visible, cache.width, scale, oh),
                Kernel::Tiled => attend_head_tiled(qh, &cache.k[off..], &cache.v[off..], visible, cache.width, scale, oh),
                Kernel::Split { padded, nth } => attend_head_split(qh, &cache.k[off..], &cache.v[off..], visible, padded, nth, cache.width, scale, oh),
            }
        };
        if kernel == Kernel::Tiled && cache.kind == KvType::F16 {
            // ggml's tiled kernel: a job is one head over a block of 16 rows, so each tile is widened once per block
            const B: usize = 16;
            let blocks = qs.len().div_ceil(B);
            let (next, base) = (std::sync::atomic::AtomicUsize::new(0), out.as_mut_ptr() as usize);
            self.pool.run(&|_| loop {
                let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if j >= blocks * nh {
                    break;
                }
                let (blk, h) = (j / nh, j % nh);
                let rs = blk * B..(blk * B + B).min(qs.len());
                let off = (h / group) * hd;
                let q: Vec<&[f32]> = rs.clone().map(|i| &qs[i][h * hd..(h + 1) * hd]).collect();
                let vis: Vec<usize> = rs.clone().map(|i| p0 + i + 1).collect();
                // SAFETY: job (blk, h) alone writes head h of rows rs, disjoint slices inside the buffer
                let mut o: Vec<&mut [f32]> = rs.map(|i| unsafe { std::slice::from_raw_parts_mut((base as *mut f32).add((i * nh + h) * hd), hd) }).collect();
                attend_heads_tiled(&q, &vis, &cache.k[off..], &cache.v[off..], cache.width, scale, &mut o);
            });
            return out.chunks_exact(nh * hd).map(<[f32]>::to_vec).collect();
        }
        let jobs = qs.len() * nh;
        // a few short heads are cheaper here than waking the pool
        if self.pool.threads() == 1 || jobs * (p0 + qs.len()) < 2048 {
            for (j, oh) in out.chunks_exact_mut(hd).enumerate() {
                one(j, oh);
            }
        } else {
            let (next, base) = (std::sync::atomic::AtomicUsize::new(0), out.as_mut_ptr() as usize);
            self.pool.run(&|_| loop {
                let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if j >= jobs {
                    break;
                }
                // SAFETY: job j alone writes out[j·hd .. (j + 1)·hd], inside the buffer
                one(j, unsafe { std::slice::from_raw_parts_mut((base as *mut f32).add(j * hd), hd) });
            });
        }
        out.chunks_exact(nh * hd).map(<[f32]>::to_vec).collect()
    }

    /// Layer `il`'s feed-forward block on `ffn_inp` (the residual stream after attention), written back in place as
    /// `l_out`: `ffn_norm`, gate and up (one q8_0 activation), SwiGLU, `ffn_down`, the residual.
    pub fn ffn(&self, il: usize, x: &mut [f32]) -> Result<(), String> {
        let g = self.norm(&format!("blk.{il}.ffn_norm.weight"))?;
        let mut xn = vec![0.0f32; x.len()];
        rms_norm_mul(x, g, self.rms_eps, &mut xn);
        let a = self.quantize(&xn);
        let wg = self.matrix(&format!("blk.{il}.ffn_gate.weight"), self.n_embd)?;
        let wu = self.matrix(&format!("blk.{il}.ffn_up.weight"), self.n_embd)?;
        let n_ff = wg.rows;
        if wu.rows != n_ff {
            return Err(format!("ffn_gate has {n_ff} rows, ffn_up {}", wu.rows));
        }
        let (mut gate, mut up, mut h) = (vec![0.0f32; n_ff], vec![0.0f32; n_ff], vec![0.0f32; n_ff]);
        self.mv_many(&[&wg, &wu], &a, &mut [&mut gate, &mut up])?;
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
        self.caches_of(KvType::F16)
    }

    /// Empty caches of `kind` (q8_0 needs `head_dim` a multiple of 32, as ggml's blocks do).
    pub fn caches_of(&self, kind: KvType) -> Vec<KvCache> {
        (0..self.n_layer).map(|_| KvCache { head_dim: self.head_dim, ..KvCache::of(kind, self.n_head_kv * self.head_dim) }).collect()
    }

    /// One token through every layer at the next position (the caches' length), appending its K and V to each
    /// layer's cache. `each_layer(il, l_out)` sees every layer's output; returns `result_norm` (`output_norm` of the
    /// last layer's output), which `logits` turns into scores. Always uses the reference attention kernel.
    pub fn step(&self, caches: &mut [KvCache], token: u32, each_layer: impl FnMut(usize, &[f32])) -> Result<Vec<f32>, String> {
        self.step_with(caches, token, Kernel::Reference, each_layer)
    }

    /// A prompt, as llama.cpp computes it: micro-batches of up to `N_UBATCH` tokens, each row's attention by the
    /// kernel llama.cpp chooses for its micro-batch (`kernel_for`). Returns the last token's `result_norm`.
    pub fn prefill(&self, caches: &mut [KvCache], tokens: &[u32], each: impl FnMut(usize, usize, &[f32])) -> Result<Vec<f32>, String> {
        Ok(self.prefill_with(caches, tokens, Outputs::Last, each)?.pop().unwrap_or_default())
    }

    /// `prefill` with the rows asked for: `Outputs::Last` gives the last token's `result_norm` (llama-server's
    /// request: only the last micro-batch's last row leaves the last layer); `Outputs::All` gives every token's.
    pub fn prefill_with(&self, caches: &mut [KvCache], tokens: &[u32], outputs: Outputs, mut each: impl FnMut(usize, usize, &[f32]))
                        -> Result<Vec<Vec<f32>>, String> {
        let mut rn = Vec::new();
        let n_ub = tokens.len().div_ceil(N_UBATCH);
        for (i, ub) in tokens.chunks(N_UBATCH).enumerate() {
            let kernel = kernel_for(ub.len(), caches[0].len() + ub.len(), self.llama_threads)?;
            let out = match outputs {
                Outputs::All => Some(Outputs::All),
                Outputs::Last if i + 1 == n_ub => Some(Outputs::Last),
                Outputs::Last => None,
            };
            rn.extend(self.ubatch(caches, ub, kernel, out, &mut each)?);
        }
        Ok(rn)
    }

    /// One micro-batch through every layer together (the same bits as token by token for the quantized types,
    /// ggml's batched bits for F16). On the last layer only the output rows go on, as llama.cpp's
    /// `inp_out_ids` selects them (none, the last, or all); returns their `result_norm`s.
    fn ubatch(&self, caches: &mut [KvCache], toks: &[u32], kernel: Kernel, outputs: Option<Outputs>, each: &mut impl FnMut(usize, usize, &[f32]))
              -> Result<Vec<Vec<f32>>, String> {
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
        let n_layer = caches.len();
        // the rows that leave the last layer, by position in the micro-batch
        let mut rows: Vec<usize> = (0..n).collect();
        for (il, cache) in caches.iter_mut().enumerate() {
            let a = self.quantize_rows(&norm_rows(&xs, self.norm(&format!("blk.{il}.attn_norm.weight"))?));
            let mut q = self.mm(&self.matrix(&format!("blk.{il}.attn_q.weight"), self.n_embd)?, &a)?;
            let mut k = self.mm(&self.matrix(&format!("blk.{il}.attn_k.weight"), self.n_embd)?, &a)?;
            let v = self.mm(&self.matrix(&format!("blk.{il}.attn_v.weight"), self.n_embd)?, &a)?;
            let norms = match self.arch {
                Arch::Qwen3 => Some((self.norm(&format!("blk.{il}.attn_q_norm.weight"))?, self.norm(&format!("blk.{il}.attn_k_norm.weight"))?)),
                Arch::Llama => None,
            };
            for (which, rws) in [(0, &mut q), (1, &mut k)] {
                for (row, rc) in rws.iter_mut().zip(&ropes) {
                    for h in row.chunks_exact_mut(hd) {
                        if let Some((gq, gk)) = &norms {
                            tmp.copy_from_slice(h);
                            rms_norm_mul(&tmp, if which == 0 { gq } else { gk }, self.rms_eps, h);
                        }
                        self.rope.apply(rc, h);
                    }
                }
            }
            for (kr, vr) in k.iter().zip(&v) {
                cache.push(kr, vr); // the whole micro-batch's K and V first, as llama.cpp writes them; causality is `visible`
            }
            let kqv = self.attend_rows(&q, cache, p0, kernel);
            let att = self.mm(&self.matrix(&format!("blk.{il}.attn_output.weight"), self.n_head * hd)?, &self.quantize_rows(&kqv))?;
            if il + 1 == n_layer {
                // llama.cpp's get_rows(inp_out_ids) after attention: the rest of the last layer sees only the output rows
                rows = match outputs {
                    None => Vec::new(),
                    Some(Outputs::Last) => vec![n - 1],
                    Some(Outputs::All) => (0..n).collect(),
                };
                if rows.is_empty() {
                    return Ok(Vec::new());
                }
                xs = rows.iter().map(|&i| xs[i].clone()).collect();
            }
            for (x, &i) in xs.iter_mut().zip(&rows) {
                for (xi, ai) in x.iter_mut().zip(&att[i]) {
                    *xi += ai;
                }
            }
            // the feed-forward block
            let a = self.quantize_rows(&norm_rows(&xs, self.norm(&format!("blk.{il}.ffn_norm.weight"))?));
            let gate = self.mm(&self.matrix(&format!("blk.{il}.ffn_gate.weight"), self.n_embd)?, &a)?;
            let up = self.mm(&self.matrix(&format!("blk.{il}.ffn_up.weight"), self.n_embd)?, &a)?;
            let h: Vec<Vec<f32>> = gate.iter().zip(&up).map(|(g, u)| {
                let mut o = vec![0.0f32; g.len()];
                swiglu(g, u, &mut o);
                o
            }).collect();
            let n_ff = h[0].len();
            let down = self.mm(&self.matrix(&format!("blk.{il}.ffn_down.weight"), n_ff)?, &self.quantize_rows(&h))?;
            for ((x, d), &i) in xs.iter_mut().zip(&down).zip(&rows) {
                for (xi, di) in x.iter_mut().zip(d) {
                    *xi += di;
                }
                each(p0 + i, il, x);
            }
        }
        Ok(norm_rows(&xs, self.norm("output_norm.weight")?))
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
            rms_norm_mul(&x, self.norm(&format!("blk.{il}.attn_norm.weight"))?, self.rms_eps, &mut xn);
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
        rms_norm_mul(&x, self.norm("output_norm.weight")?, self.rms_eps, &mut rn);
        Ok(rn)
    }

    /// The logits: the output matrix (`output.weight`, or `token_embd.weight` when the embeddings are tied; one row
    /// per vocabulary entry) times `result_norm`, prepared for the weight type.
    pub fn logits(&self, result_norm: &[f32]) -> Result<Vec<f32>, String> {
        let m = self.matrix(self.output, self.n_embd)?;
        let mut out = vec![0.0f32; m.rows];
        self.mv(&m, &self.quantize(result_norm), &mut out)?;
        Ok(out)
    }

    /// The logits of several rows as one product (what a graph asking for every row's logits computes).
    pub fn logits_rows(&self, result_norms: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        self.mm(&self.matrix(self.output, self.n_embd)?, &self.quantize_rows(result_norms))
    }

    /// `get_rows(token_embd, [id])`: the id's row of the table, dequantized (bit-exact with ggml's
    /// `dequantize_row_q1_0` / `dequantize_row_q2_0`, §III.4, §III.6, and its f16 → f32 widening).
    pub fn embed(&self, id: u32, out: &mut [f32]) -> Result<(), String> {
        let (t, b) = self.tensor("token_embd.weight")?;
        let ty = t.ty;
        if id as usize >= self.n_vocab || out.len() != self.n_embd {
            return Err(format!("token {id} outside the vocabulary of {}", self.n_vocab));
        }
        let rb = row_bytes(ty, self.n_embd);
        let row = b.get(id as usize * rb..(id as usize + 1) * rb).ok_or("tensor truncated")?;
        match ty {
            TYPE_Q1_0 => q1_0::dequantize_row(row, out),
            TYPE_F16 => crate::f16::dequantize_row(row, out),
            _ => q2_0::dequantize_row(row, out),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The q8_0 KV cache's two kernels against the shipped ggml b11192 haswell library (dlopen, no crate):
    /// `quantize_row_q8_0` byte for byte (the cache store and Q), and `ggml_vec_dot_q8_0_q8_0` bit for bit (the scores)
    /// — random activations with outliers, and blocks full of −128 and 127 where `maddubs` saturates.
    #[test]
    #[ignore = "needs BANKML_GGML_LIB (llama.cpp b11192 ubuntu-x64 release dir)"]
    fn oracle_ggml_b11192_q8_0_kv_kernels() {
        use std::ffi::{c_char, c_int, c_void, CString};
        extern "C" {
            fn dlopen(f: *const c_char, flag: c_int) -> *mut c_void;
            fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void;
        }
        type Dot = unsafe extern "C" fn(i32, *mut f32, usize, *const u8, usize, *const u8, usize, i32);
        type Quant = unsafe extern "C" fn(*const f32, *mut u8, i64);
        let dir = std::env::var("BANKML_GGML_LIB").expect("BANKML_GGML_LIB");
        let (base, cpu) = (CString::new(format!("{dir}/libggml-base.so")).unwrap(), CString::new(format!("{dir}/libggml-cpu-haswell.so")).unwrap());
        let (dot, quant) = unsafe {
            assert!(!dlopen(base.as_ptr(), 0x102).is_null());
            let h = dlopen(cpu.as_ptr(), 0x102);
            assert!(!h.is_null());
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(dlsym(h, c"ggml_cpu_init".as_ptr()))();
            (std::mem::transmute::<*mut c_void, Dot>(dlsym(h, c"ggml_vec_dot_q8_0_q8_0".as_ptr())),
             std::mem::transmute::<*mut c_void, Quant>(dlsym(h, c"quantize_row_q8_0".as_ptr())))
        };
        let mut r = crate::q1_0::tests::Rng(0x9e37_79b9_7f4a_7c15);
        let (mut rows, mut dots) = (0, 0);
        for case in 0..4000 {
            let n = [32, 64, 128, 256][case % 4];
            let nb = n / q1_0::QK8_0 * q1_0::Q8_0_BYTES;
            let x: Vec<f32> = (0..n).map(|_| r.act()).collect();
            let y: Vec<f32> = (0..n).map(|_| r.act()).collect();
            let (mut qx, mut qy, mut gx) = (vec![0u8; nb], vec![0u8; nb], vec![0u8; nb]);
            q1_0::quantize_row_q8_0(&x, &mut qx);
            q1_0::quantize_row_q8_0(&y, &mut qy);
            unsafe { quant(x.as_ptr(), gx.as_mut_ptr(), n as i64) };
            assert_eq!(qx, gx, "quantize_row_q8_0, case {case}");
            rows += 1;
            if case % 3 == 0 {
                // saturation: whole blocks of −128 against −128 and 127 (|−128|·−128 pairs overflow i16)
                for (i, b) in qx.iter_mut().enumerate() {
                    if i % q1_0::Q8_0_BYTES >= 2 && r.next().is_multiple_of(2) {
                        *b = 0x80;
                    }
                }
                for (i, b) in qy.iter_mut().enumerate() {
                    if i % q1_0::Q8_0_BYTES >= 2 && r.next().is_multiple_of(3) {
                        *b = if r.next().is_multiple_of(2) { 0x80 } else { 0x7f };
                    }
                }
            }
            let mut want = 0.0f32;
            unsafe { dot(n as i32, &mut want, 0, qx.as_ptr(), 0, qy.as_ptr(), 0, 1) };
            let got = vec_dot_q8_0_q8_0(n, &qx, &qy);
            assert_eq!(got.to_bits(), want.to_bits(), "vec_dot_q8_0_q8_0, case {case}: {got} vs {want}");
            dots += 1;
        }
        eprintln!("q8_0 KV kernels: {rows} rows quantized byte-exact, {dots} dot products bit-exact against ggml b11192 haswell");
    }

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
        // the oracle's graph computes the whole prompt at once and asks every row's logits: for the quantized types
        // that is token by token's bits (the per-pair dot), for F16 it is ggml's batched product (f16.rs), so an F16
        // model is replayed as one micro-batch with every row output
        let batched = (w.wtype == TYPE_F16).then(|| {
            let mut rows: Vec<Vec<(usize, String)>> = vec![Vec::new(); toks.len()];
            let rns = w.prefill_with(&mut caches, &toks, Outputs::All, |p, il, l| rows[p].push((il, sha(l)))).unwrap();
            let logits = w.logits_rows(&rns).unwrap();
            (rows, rns, logits)
        });
        for (p, &t) in toks.iter().enumerate() {
            let (layer_rows, rn, logits) = match &batched {
                Some((rows, rns, lg)) => (rows[p].clone(), rns[p].clone(), lg[p].clone()),
                None => {
                    let mut layer_rows = Vec::new();
                    let rn = w.step(&mut caches, t, |il, l| layer_rows.push((il, sha(l)))).unwrap();
                    let logits = w.logits(&rn).unwrap();
                    (layer_rows, rn, logits)
                }
            };
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

    /// Bonsai-1.7B, whose logits come from its tied token table (no output.weight).
    #[test]
    #[ignore = "needs .models/Bonsai-1.7B-Q1_0.gguf + its model-*.tsv (testing/model_oracle.py); --release"]
    fn oracle_forward_model_bonsai_1_7b() {
        model_oracle("Bonsai-1.7B-Q1_0");
    }

    /// The Llama graph in F16 with tied embeddings — SmolLM2-135M-Instruct and mindX's gen 39.
    #[test]
    #[ignore = "needs .models/{SmolLM2-135M-Instruct,mindx-gen39}-F16.gguf + their model-*.tsv (testing/model_oracle.py); --release"]
    fn oracle_forward_model_llama_f16() {
        model_oracle("SmolLM2-135M-Instruct-F16");
        model_oracle("mindx-gen39-F16");
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
        sample_oracle("Bonsai-8B-Q1_0");
    }

    /// Bonsai-1.7B (tied embeddings) against llama-server running it: greedy and seeded.
    #[test]
    #[ignore = "needs .models/Bonsai-1.7B-Q1_0.gguf + its greedy-*/sample-*.jsonl (testing/{greedy,sample}_oracle.py); --release"]
    fn oracle_llama_server_bonsai_1_7b() {
        greedy_oracle("", "Bonsai-1.7B-Q1_0");
        sample_oracle("Bonsai-1.7B-Q1_0");
    }

    /// The Llama graph in F16 against llama-server running SmolLM2-135M-Instruct and mindx-gen39: short prompts
    /// (ggml's reference attention), long ones (the tiled kernel, and tinyBLAS over the micro-batch), continuations
    /// past 256 cells (the split-KV kernel), and seeded sampling.
    #[test]
    #[ignore = "needs .models/{SmolLM2-135M-Instruct,mindx-gen39}-F16.gguf + their greedy-*/sample-*.jsonl (testing/{greedy,sample}_oracle.py); --release"]
    fn oracle_llama_server_llama_f16() {
        for stem in ["SmolLM2-135M-Instruct-F16", "mindx-gen39-F16"] {
            for kind in ["", "long-", "deep-"] {
                greedy_oracle(kind, stem);
            }
            sample_oracle(stem);
        }
    }

    fn sample_oracle(stem: &str) {
        use crate::sampler::{Params, Sampler};
        use crate::serve::Json;
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let w = Weights::open(&dir.join(format!("{stem}.gguf"))).unwrap();
        let cases = std::fs::read_to_string(dir.join(format!("oracle-forward/sample-{stem}.jsonl"))).unwrap();
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
                                                min_keep: num(pr, "min_keep") as usize, seed: num(pr, "seed") as u32,
                                                // records from before O2 carry no repeat_last_n; their penalties are
                                                // neutral (asserted above), so llama.cpp's default window changes nothing
                                                penalty_last_n: match pr.get("repeat_last_n") { Some(Json::Num(n)) => *n as i32, _ => 64 },
                                                ..Params::default() }).unwrap();
            // the prompt's caches, computed once per prompt and cloned per case
            if prefilled.as_ref().map(|p| p.0 != prompt).unwrap_or(true) {
                let mut caches = w.caches();
                let rn = w.prefill(&mut caches, &prompt, |_, _, _| {}).unwrap();
                prefilled = Some((prompt.clone(), caches, rn));
            }
            let (_, c0, rn0) = prefilled.as_ref().unwrap();
            let mut caches: Vec<KvCache> = c0.to_vec();
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
        eprintln!("sample oracle: {stem}: {n_same} of {n_cases} seeded continuations identical to llama-server b11192's ({n_tok} tokens; top-k, top-p, min-p, temperature, the mt19937 draw)");
        assert_eq!(n_same, n_cases);
    }
}
