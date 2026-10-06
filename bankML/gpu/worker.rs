// SPDX-License-Identifier: MIT OR Apache-2.0
//! The GPU worker: a verified card computes the first `share` of each Q1_0 matrix–vector product's rows while the
//! CPU pool computes the rest.
//!
//! The card passes `kernels::verify_q1_0` before it gets any work, so every row it returns carries the CPU kernel's
//! bits and the forward pass is unchanged. `share` is calibrated at open (`BANKML_GPU_SHARE` overrides it);
//! `BANKML_GPU_LIMIT` caps the card's memory and duty cycle; each matrix shape keeps the card only if timing in the
//! real pipeline shows it faster. A product the card does not take runs whole on the CPU, with the same bits.
//! Details: docs/modules/gpu.md.

use super::compute::{Buffer, Gpu, Pipeline};
use super::kernels::{pack_q1_0, q1_0_mat_vec8, verify_q1_0, LOCAL_SIZE, Q1_0_BINDINGS};
use crate::par::Pool;
use crate::q1_0::{mat_vec_par, Q8Act, Q1_0_BYTES, QK1_0};
use std::collections::HashMap;
use std::time::Instant;

/// Timed products per candidate and shape, after one warm-up each (the warm-up includes the upload).
const TRIALS: usize = 6;

/// One shape's tuning: timings per candidate (0 = the CPU alone, 1 = the card's share) and the decision once made.
#[derive(Debug, Clone, Default)]
struct Tune {
    times: [Vec<f64>; 2],
    seen: [usize; 2],
    next: usize,
    chosen: Option<bool>,
}

/// `BANKML_GPU_LIMIT`: the fraction of the card's memory and time bankML may use (clamped 0.05–1, default 0.8).
pub fn limit() -> f64 {
    std::env::var("BANKML_GPU_LIMIT").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.8).clamp(0.05, 1.0)
}

/// The limiter's state, for `/bankml/usage` and the console.
#[derive(Debug, Clone, Default)]
pub struct Status {
    pub card: String,
    pub limit: f64,
    pub share: f64,
    pub allocated: u64,
    pub heap: u64,
    pub integrated: bool,
    /// The card's busy fraction over recent products (smoothed).
    pub busy: f64,
    pub on_card: u64,
    pub on_cpu_resting: u64,
    pub on_cpu_memory: u64,
    /// Per-shape calibration: shapes decided for the card, for the CPU, and still being measured.
    pub shapes_card: u64,
    pub shapes_cpu: u64,
    pub shapes_tuning: u64,
}

static STATUS: std::sync::Mutex<Option<Status>> = std::sync::Mutex::new(None);

/// The limiter's last state (`None` when no card works in this process).
pub fn status() -> Option<Status> {
    STATUS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// [`status`] as JSON (`null` when no card works), for `/bankml/usage`'s `gpu_limiter`.
pub fn status_json() -> String {
    match status() {
        None => "null".into(),
        Some(s) => format!("{{\"card\": {}, \"limit\": {:.2}, \"share\": {:.3}, \"allocated_bytes\": {}, \"heap_bytes\": {}, \"integrated\": {}, \
                            \"busy\": {:.3}, \"products_on_card\": {}, \"products_on_cpu_resting\": {}, \"products_on_cpu_memory\": {}, \
                            \"shapes_on_card\": {}, \"shapes_on_cpu\": {}, \"shapes_tuning\": {}}}",
                           crate::gguf::jstr(&s.card), s.limit, s.share, s.allocated, s.heap, s.integrated, s.busy, s.on_card, s.on_cpu_resting, s.on_cpu_memory,
                           s.shapes_card, s.shapes_cpu, s.shapes_tuning),
    }
}

struct GMat {
    wd: Buffer,
    wb: Buffer,
    out: Buffer,
    g: usize,
    shape: (usize, usize),
}

/// The card's side of each Q1_0 product: [`Worker::begin`] starts it and [`Worker::finish`] collects it.
pub struct Worker {
    gpu: Gpu,
    pipe: Pipeline,
    pub name: String,
    /// The fraction of each matrix's rows the card computes.
    pub share: f64,
    mats: HashMap<String, GMat>,
    act_d: Buffer,
    act_q: Buffer,
    cap_n: usize,
    pending: Option<(String, usize)>,
    /// The limiter: L, whether the heap is system RAM, when the card may work again, when the pending dispatch began.
    limit: f64,
    integrated: bool,
    rest_until: Option<Instant>,
    started: Option<Instant>,
    st: Status,
    /// Per-shape calibration, and the product being timed: shape, candidate (1 = the card's share), start.
    tunes: HashMap<(usize, usize), Tune>,
    trial: Option<((usize, usize), usize, Instant)>,
}

impl Worker {
    /// Open the first selected card, verify it, and calibrate its share against `pool`.
    ///
    /// `Ok(None)` when there is no usable card, the component is off, or the share is below 0.02; `Err` with the
    /// reason when the card fails to open, verify, or set up its pipeline and buffers.
    pub fn open(pool: &Pool) -> Result<Option<Worker>, String> {
        let sel = super::selected(&super::discover().0);
        let Some(d) = sel.first() else { return Ok(None) };
        let gpu = Gpu::open(d.index)?;
        verify_q1_0(&gpu).map_err(|e| format!("{}: not verified ({e}); bankml will not use it", d.name))?;
        let pipe = gpu.pipeline(&q1_0_mat_vec8(), Q1_0_BINDINGS, 8)?;
        let cap_n = 16384;
        let (act_d, act_q) = (gpu.buffer(cap_n / 32 * 4)?, gpu.buffer(cap_n)?);
        let integrated = matches!(d.kind, super::Kind::Integrated);
        let mut w = Worker { gpu, pipe, name: d.name.clone(), share: 0.0, mats: HashMap::new(), act_d, act_q, cap_n, pending: None,
                             limit: limit(), integrated, rest_until: None, started: None, st: Status::default(),
                             tunes: HashMap::new(), trial: None };
        w.share = match std::env::var("BANKML_GPU_SHARE").ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(s) => s.clamp(0.0, 0.95),
            None => w.calibrate(pool)?,
        };
        Ok((w.share >= 0.02).then_some(w))
    }

    /// Time a 4096×4096 product on the card and on the pool (best of 3); the card's share of the combined rate.
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
            if self.begin_share("calibrate", &w, rows, &a, 1.0, false).is_ok() {
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

    /// Start the card on the first `share` of `rows` rows of the Q1_0 matrix `name` (bytes `w`) times `a`.
    ///
    /// Returns the rows the card took (0 when none). The caller computes the rest, then calls [`Worker::finish`]
    /// even when 0 was returned, since `finish` also records the shape's timing.
    pub fn begin(&mut self, name: &str, w: &[u8], rows: usize, a: &Q8Act) -> Result<usize, String> {
        let shape = (rows, a.n());
        let t = self.tunes.entry(shape).or_default();
        let cand = match t.chosen {
            Some(card) => {
                self.trial = None;
                card as usize
            }
            None => {
                let c = t.next % 2;
                t.next += 1;
                self.trial = Some((shape, c, Instant::now()));
                c
            }
        };
        if cand == 0 {
            return Ok(0);
        }
        let g = self.begin_share(name, w, rows, a, self.share, true)?;
        if g == 0 {
            self.trial = None; // the card rested or the matrix did not fit: not a measurement of the card's share
        }
        Ok(g)
    }

    /// Record a timed product and, once each candidate has `TRIALS` timings after its warm-up, decide the shape.
    fn record(&mut self, shape: (usize, usize), cand: usize, secs: f64) {
        let Some(t) = self.tunes.get_mut(&shape) else { return };
        t.seen[cand] += 1;
        if t.seen[cand] > 1 {
            t.times[cand].push(secs);
        }
        if t.chosen.is_none() && t.times.iter().all(|v| v.len() >= TRIALS) {
            let median = |v: &[f64]| {
                let mut v = v.to_vec();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let card = median(&t.times[1]) < median(&t.times[0]);
            t.chosen = Some(card);
            if !card {
                // The CPU alone is faster for this shape: its matrices leave the card.
                let names: Vec<String> = self.mats.iter().filter(|(_, m)| m.shape == shape).map(|(k, _)| k.clone()).collect();
                for n in names {
                    if let Some(m) = self.mats.remove(&n) {
                        for b in [m.wd, m.wb, m.out] {
                            self.gpu.free(b);
                        }
                    }
                }
            }
        }
        let (mut card, mut cpu, mut tuning) = (0, 0, 0);
        for t in self.tunes.values() {
            match t.chosen {
                Some(true) => card += 1,
                Some(false) => cpu += 1,
                None => tuning += 1,
            }
        }
        (self.st.shapes_card, self.st.shapes_cpu, self.st.shapes_tuning) = (card, cpu, tuning);
    }

    /// Whether `bytes` more stay within the limit of the heap and, on an integrated card, of available RAM.
    fn fits(&self, bytes: u64) -> bool {
        let held = self.gpu.allocated() + bytes;
        if held as f64 > self.limit * self.gpu.heap_bytes as f64 {
            return false;
        }
        if self.integrated {
            let avail = crate::sys::memory().map(|m| m.available).unwrap_or(0);
            return held as f64 <= self.limit * (self.gpu.allocated() + avail) as f64;
        }
        true
    }

    fn publish(&mut self) {
        self.st.card = self.name.clone();
        self.st.limit = self.limit;
        self.st.share = self.share;
        self.st.allocated = self.gpu.allocated();
        self.st.heap = self.gpu.heap_bytes;
        self.st.integrated = self.integrated;
        *STATUS.lock().unwrap_or_else(|e| e.into_inner()) = Some(self.st.clone());
    }

    /// Upload `name`'s rows on first use, write the activation, and submit; returns the rows taken.
    ///
    /// Returns 0 for activations wider than `cap_n` and, when `limited`, while the card rests or when the matrix
    /// does not fit the memory limit.
    fn begin_share(&mut self, name: &str, w: &[u8], rows: usize, a: &Q8Act, share: f64, limited: bool) -> Result<usize, String> {
        let n = a.n();
        let g = ((rows as f64 * share) as usize).min(rows);
        if g == 0 || n > self.cap_n {
            return Ok(0);
        }
        if limited && self.rest_until.is_some_and(|t| Instant::now() < t) {
            self.st.on_cpu_resting += 1;
            self.publish();
            return Ok(0);
        }
        if !self.mats.contains_key(name) {
            let need = (g * (n / QK1_0) * Q1_0_BYTES + g * 4) as u64;
            if limited && !self.fits(need) {
                self.st.on_cpu_memory += 1;
                self.publish();
                return Ok(0);
            }
            let (wd, wb) = pack_q1_0(w, g, n);
            let m = GMat { wd: self.gpu.upload(&wd)?, wb: self.gpu.upload(&wb)?, out: self.gpu.buffer(g * 4)?, g, shape: (rows, n) };
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
        self.started = Some(Instant::now());
        if limited {
            self.st.on_card += 1;
        }
        Ok(g)
    }

    /// Wait for the card, write its rows into the start of `out`, and record a pending timing.
    pub fn finish(&mut self, out: &mut [f32]) -> Result<(), String> {
        let r = self.finish_card(out);
        if let Some((shape, cand, t0)) = self.trial.take() {
            self.record(shape, cand, t0.elapsed().as_secs_f64());
            self.publish();
        }
        r
    }

    fn finish_card(&mut self, out: &mut [f32]) -> Result<(), String> {
        let Some((name, g)) = self.pending.take() else { return Ok(()) };
        self.gpu.wait()?;
        out[..g].copy_from_slice(&self.gpu.read_f32(&self.mats[&name].out, g));
        // Duty cycle: rest d·(1−L)/L after a dispatch of d; `busy` is the smoothed busy fraction.
        if let Some(t) = self.started.take() {
            let d = t.elapsed();
            let rest = d.mul_f64((1.0 - self.limit) / self.limit);
            self.rest_until = (self.limit < 1.0).then(|| Instant::now() + rest);
            let frac = d.as_secs_f64() / (d + rest).as_secs_f64().max(1e-9);
            self.st.busy = if self.st.busy == 0.0 { frac } else { 0.9 * self.st.busy + 0.1 * frac };
        }
        self.publish();
        Ok(())
    }
}
