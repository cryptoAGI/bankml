// SPDX-License-Identifier: MIT OR Apache-2.0
//! The GPU as a worker beside the CPU's threads. For each 1-bit matrix–vector product the card takes the first
//! `share` of the rows (asynchronously) while the CPU pool computes the rest; the forward pass then waits for the
//! card and copies its rows in. Every row the card computes carries the CPU kernel's bits (the card passed
//! `kernels::verify_q1_0` before it was given any work), so the forward pass — and every token oracle — is unchanged.
//!
//! The share comes from a calibration when the card is opened: the same 4096×4096 product timed on the card and on
//! the CPU pool, `share = card rate / (card rate + CPU rate)`. `BANKML_GPU_SHARE` overrides it; `BANKML_GPU=off`
//! turns the worker off. Only the card's share of each matrix is copied to it (repacked, exactly), on first use.

use super::compute::{Buffer, Gpu, Pipeline};
use super::kernels::{pack_q1_0, q1_0_mat_vec8, verify_q1_0, LOCAL_SIZE, Q1_0_BINDINGS};
use crate::par::Pool;
use crate::q1_0::{mat_vec_par, Q8Act, Q1_0_BYTES, QK1_0};
use std::collections::HashMap;

struct GMat {
    wd: Buffer,
    wb: Buffer,
    out: Buffer,
    g: usize,
}

pub struct Worker {
    gpu: Gpu,
    pipe: Pipeline,
    pub name: String,
    /// the fraction of each matrix's rows the card computes
    pub share: f64,
    mats: HashMap<String, GMat>,
    act_d: Buffer,
    act_q: Buffer,
    cap_n: usize,
    pending: Option<(String, usize)>,
}

impl Worker {
    /// The first selected card, verified and calibrated against `pool`; `None` when there is no usable card, the
    /// component is off, or the calibration gives the card nothing worth sending (the reason is returned too).
    pub fn open(pool: &Pool) -> Result<Option<Worker>, String> {
        let sel = super::selected(&super::discover().0);
        let Some(d) = sel.first() else { return Ok(None) };
        let gpu = Gpu::open(d.index)?;
        verify_q1_0(&gpu).map_err(|e| format!("{}: not verified ({e}); bankml will not use it", d.name))?;
        let pipe = gpu.pipeline(&q1_0_mat_vec8(), Q1_0_BINDINGS, 8)?;
        let cap_n = 16384;
        let (act_d, act_q) = (gpu.buffer(cap_n / 32 * 4)?, gpu.buffer(cap_n)?);
        let mut w = Worker { gpu, pipe, name: d.name.clone(), share: 0.0, mats: HashMap::new(), act_d, act_q, cap_n, pending: None };
        w.share = match std::env::var("BANKML_GPU_SHARE").ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(s) => s.clamp(0.0, 0.95),
            None => w.calibrate(pool)?,
        };
        Ok((w.share >= 0.02).then_some(w))
    }

    /// Time a 4096×4096 product on the card and on the pool; the card's share of the combined rate.
    fn calibrate(&mut self, pool: &Pool) -> Result<f64, String> {
        let (rows, n) = (4096usize, 4096usize);
        let nb = n / QK1_0;
        let mut s = 0x9e37_79b9u32;
        let w: Vec<u8> = (0..rows * nb * Q1_0_BYTES).map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 8) as u8 & 0x7f
        }).collect();
        let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01).collect();
        let a = Q8Act::quantize(&x);
        let mut out = vec![0.0f32; rows];
        let best = |f: &mut dyn FnMut()| (0..3).map(|_| { let t = std::time::Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::MAX, f64::min);
        let t_cpu = best(&mut || mat_vec_par(pool, &w, rows, &a, &mut out));
        self.mats.remove("calibrate");
        let t_gpu = best(&mut || {
            self.pending = None;
            if self.begin_share("calibrate", &w, rows, &a, 1.0).is_ok() {
                let _ = self.finish(&mut out);
            }
        });
        if let Some(m) = self.mats.remove("calibrate") {
            for b in [m.wd, m.wb, m.out] {
                self.gpu.free(b);
            }
        }
        let (rg, rc) = (1.0 / t_gpu, 1.0 / t_cpu);
        Ok(rg / (rg + rc))
    }

    /// Start the card on the first `share` of `rows` rows of the Q1_0 matrix `name` (bytes `w`) times `a`; returns
    /// the rows it took (0 when none), which `finish` then fills.
    pub fn begin(&mut self, name: &str, w: &[u8], rows: usize, a: &Q8Act) -> Result<usize, String> {
        self.begin_share(name, w, rows, a, self.share)
    }

    fn begin_share(&mut self, name: &str, w: &[u8], rows: usize, a: &Q8Act, share: f64) -> Result<usize, String> {
        let n = a.n();
        let g = ((rows as f64 * share) as usize).min(rows);
        if g == 0 || n > self.cap_n {
            return Ok(0);
        }
        if !self.mats.contains_key(name) {
            let (wd, wb) = pack_q1_0(w, g, n);
            let m = GMat { wd: self.gpu.upload(&wd)?, wb: self.gpu.upload(&wb)?, out: self.gpu.buffer(g * 4)?, g };
            self.mats.insert(name.to_string(), m);
        }
        let m = &self.mats[name];
        let g = m.g;
        let (d, q) = super::kernels::pack_act(a);
        self.gpu.write(&self.act_d, &d);
        self.gpu.write(&self.act_q, &q);
        self.gpu.submit(&self.pipe, &[&m.wd, &m.wb, &self.act_d, &self.act_q, &m.out], &[g as u32, (n / QK1_0) as u32],
                        (g as u32 * 8).div_ceil(LOCAL_SIZE))?;
        self.pending = Some((name.to_string(), g));
        Ok(g)
    }

    /// Wait for the card and write its rows into the start of `out`.
    pub fn finish(&mut self, out: &mut [f32]) -> Result<(), String> {
        let Some((name, g)) = self.pending.take() else { return Ok(()) };
        self.gpu.wait()?;
        out[..g].copy_from_slice(&self.gpu.read_f32(&self.mats[&name].out, g));
        Ok(())
    }
}
