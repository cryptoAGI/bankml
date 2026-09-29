// SPDX-License-Identifier: MIT OR Apache-2.0
//! P3, the forward pass, one verified operation at a time. Each function here reproduces the float order of the ggml
//! operation llama.cpp b11192's Qwen3 graph uses (read from the tag's `ggml/src/ggml-cpu/ops.cpp`), and its oracle is
//! the SHIPPED ggml computing the same graph (`testing/forward_oracle.py` → `oracle_forward_*`).
//!
//! Step three (0.2.3): the token embedding (`get_rows` on a Q1_0 table → `inp_embd`) and the first RMS norm with its
//! weight (`rms_norm` then `mul` → `attn_norm-0`).

use crate::gguf::{guard_file, Engine, Mmap, TensorInfo, Val};
use crate::q1_0::{dequantize_row, Q1_0_BYTES, QK1_0};
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

/// The model's tensors by name, read through bankml's own memory map.
pub struct Weights {
    mm: Mmap,
    tensors: Vec<TensorInfo>,
    data_start: u64,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub rms_eps: f32,
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
        let mm = Mmap::open(path).map_err(|e| e.to_string())?;
        Ok(Weights { mm, tensors: h.tensors, data_start: h.data_start, n_embd, n_vocab, rms_eps })
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
}
