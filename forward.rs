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

use crate::gguf::{guard_file, Engine, Mmap, TensorInfo, Val};
use crate::q1_0::{dequantize_row, mat_vec, Q8Act, Q1_0_BYTES, QK1_0};
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
        let mm = Mmap::open(path).map_err(|e| e.to_string())?;
        Ok(Weights { mm, tensors: h.tensors, data_start: h.data_start, n_embd, n_vocab, rms_eps, n_head, n_head_kv, head_dim, rope })
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

    /// A Q1_0 weight matrix's bytes and its row count, checked against the input width.
    pub fn q1_matrix(&self, name: &str, n_in: usize) -> Result<(&[u8], usize), String> {
        let (t, b) = self.tensor(name)?;
        if t.ty != 41 || t.dims[0] as usize != n_in {
            return Err(format!("{name}: type {} dims {:?}; expected Q1_0 with {n_in} columns", t.ty, t.dims));
        }
        let rows = t.dims[1] as usize;
        Ok((b.get(..rows * (n_in / QK1_0 * Q1_0_BYTES)).ok_or("tensor truncated")?, rows))
    }

    /// Layer `il`'s attention inputs for one token at position `pos`, from its normed row `xn` (`attn_norm-il`):
    /// Q (`n_head · head_dim`), K and V (`n_head_kv · head_dim`) — projected, Q and K normed per head and roped,
    /// as `Qcur`, `Kcur`, `Vcur` in llama.cpp's qwen3 graph.
    pub fn qkv(&self, il: usize, xn: &[f32], pos: i32, q: &mut [f32], k: &mut [f32], v: &mut [f32]) -> Result<(), String> {
        let a = Q8Act::quantize(xn);
        let hd = self.head_dim;
        let mut cache = vec![0.0f32; hd];
        self.rope.cache(pos, &mut cache);
        let mut tmp = vec![0.0f32; hd];
        for (w, norm, out) in [("attn_q", Some("attn_q_norm"), &mut *q), ("attn_k", Some("attn_k_norm"), &mut *k), ("attn_v", None, &mut *v)] {
            let (m, rows) = self.q1_matrix(&format!("blk.{il}.{w}.weight"), self.n_embd)?;
            if out.len() != rows {
                return Err(format!("{w}: {rows} rows, buffer {}", out.len()));
            }
            mat_vec(m, rows, &a, out);
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

    /// `get_rows(token_embd, [id])`: the id's row of the Q1_0 table, dequantized (bit-exact with
    /// `dequantize_row_q1_0`, §III.4).
    pub fn embed(&self, id: u32, out: &mut [f32]) -> Result<(), String> {
        let (t, b) = self.tensor("token_embd.weight")?;
        if t.ty != 41 {
            return Err(format!("token_embd is type {}; only Q1_0 is implemented", t.ty));
        }
        if id as usize >= self.n_vocab || out.len() != self.n_embd {
            return Err(format!("token {id} outside the vocabulary of {}", self.n_vocab));
        }
        let rb = self.n_embd / QK1_0 * Q1_0_BYTES;
        let row = b.get(id as usize * rb..(id as usize + 1) * rb).ok_or("tensor truncated")?;
        dequantize_row(row, out);
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
}
