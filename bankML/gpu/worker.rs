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
/// Marks a shape's activation width as a many-token product's (prompt reading) in the calibration: each width is
/// at most `cap_n` (16384), so the two kinds never share a key, and they are decided separately.
const MANY: usize = 1 << 30;

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
    /// Prompt reading, where the card's kernel takes many activations in one dispatch ([`Gpu::COLUMNS`]): its
    /// pipeline (rows, blocks, columns), and buffers for the activations and the result that grow to the largest
    /// micro-batch seen; the pending product's rows and columns.
    pipe_many: Option<Pipeline>,
    many: Option<(Buffer, Buffer, Buffer)>,
    pending_many: Option<(usize, usize)>,
    /// Matrices sharing one activation (Q/K/V, gate/up) submitted together, one wait for all ([`Gpu::MANY_IN_FLIGHT`]):
    /// each one's name and rows taken.
    pending_group: Vec<(String, usize)>,
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
        let pipe_many = if Gpu::COLUMNS { Some(gpu.pipeline(&super::kernels::q1_0_mat_mul8(), Q1_0_BINDINGS, 12)?) } else { None };
        let mut w = Worker { gpu, pipe, name: d.name.clone(), share: 0.0, mats: HashMap::new(), act_d, act_q, cap_n, pending: None,
                             pipe_many, many: None, pending_many: None, pending_group: Vec::new(),
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
        // a card run that fails gives no share (before, its quick failure timed as a very fast card)
        let mut ok = true;
        let t_gpu = best(&mut || {
            self.pending = None;
            match self.begin_share("calibrate", &w, rows, &a, 1.0, false) {
                Ok(g) if g > 0 => ok &= self.finish(&mut out).is_ok(),
                _ => ok = false,
            }
        });
        if let Some(m) = self.mats.remove("calibrate") {
            for b in [m.wd, m.wb, m.out] {
                self.gpu.free(b);
            }
        }
        if !ok {
            return Ok(0.0);
        }
        let (rg, rc) = (1.0 / t_gpu, 1.0 / t_cpu);
        Ok(rg / (rg + rc))
    }

    /// Start the card on the first `share` of `rows` rows of the Q1_0 matrix `name` (bytes `w`) times `a`.
    ///
    /// Returns the rows the card took (0 when none). The caller computes the rest, then calls [`Worker::finish`]
    /// even when 0 was returned, since `finish` also records the shape's timing.
    pub fn begin(&mut self, name: &str, w: &[u8], rows: usize, a: &Q8Act) -> Result<usize, String> {
        if self.candidate((rows, a.n())) == 0 {
            return Ok(0);
        }
        let g = self.begin_share(name, w, rows, a, self.share, true)?;
        if g == 0 {
            self.trial = None; // the card rested or the matrix did not fit: not a measurement of the card's share
        }
        Ok(g)
    }

    /// The candidate for this product of `shape`: its decision, or (while it is measured) the CPU and the card in
    /// turn, timed from here to `finish`.
    fn candidate(&mut self, shape: (usize, usize)) -> usize {
        let t = self.tunes.entry(shape).or_default();
        match t.chosen {
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
        }
    }

    /// Prompt reading: start the card on the first `share` of `rows` rows of `name` times every activation in `cols`
    /// (one micro-batch's tokens), in one dispatch. Returns the rows the card took (0 when none: no kernel with a
    /// token dimension here, too wide, resting, over the memory limit, or the CPU measured faster for this shape).
    /// The caller computes the other rows of every column, then calls [`Worker::finish_many`] even when 0 was
    /// returned. Each row the card returns has the bits of the CPU kernel's, as in [`Worker::begin`].
    pub fn begin_many(&mut self, name: &str, w: &[u8], rows: usize, cols: &[Q8Act]) -> Result<usize, String> {
        let Some(a0) = cols.first() else { return Ok(0) };
        let n = a0.n();
        if self.pipe_many.is_none() || cols.len() < Gpu::MIN_MANY || n > self.cap_n || (rows as f64 * self.share) as usize == 0 || cols.iter().any(|a| a.n() != n) {
            return Ok(0);
        }
        if self.candidate((rows, n + MANY)) == 0 {
            return Ok(0);
        }
        let Some(g) = self.upload(name, w, rows, n, self.share, true)? else {
            self.trial = None;
            return Ok(0);
        };
        let (mut d, mut q) = (Vec::with_capacity(cols.len() * n / 32), Vec::with_capacity(cols.len() * n / 4));
        for a in cols {
            let (ad, aq) = super::kernels::pack_act(a);
            d.extend_from_slice(&ad);
            q.extend_from_slice(&aq);
        }
        let out_bytes = g * cols.len() * 4;
        let grow = self.many.as_ref().is_none_or(|(bd, bq, bo)| bd.bytes < d.len() * 4 || bq.bytes < q.len() * 4 || bo.bytes < out_bytes);
        if grow {
            if let Some((bd, bq, bo)) = self.many.take() {
                for b in [bd, bq, bo] {
                    self.gpu.free(b);
                }
            }
            // room for the largest micro-batch so far, and for the widest matrix's share of it
            let (cd, cq, co) = (d.len() * 4, q.len() * 4, out_bytes.max(self.mats.values().map(|m| m.g).max().unwrap_or(g) * cols.len() * 4));
            self.many = Some((self.gpu.buffer(cd)?, self.gpu.buffer(cq)?, self.gpu.buffer(co)?));
        }
        let (bd, bq, bo) = self.many.as_ref().expect("made above");
        self.gpu.write(bd, &d);
        self.gpu.write(bq, &q);
        let m = &self.mats[name];
        self.gpu.submit(self.pipe_many.as_ref().expect("checked"), &[&m.wd, &m.wb, bd, bq, bo],
                        &[g as u32, (n / QK1_0) as u32, cols.len() as u32], (g as u32 * 8).div_ceil(LOCAL_SIZE))?;
        self.pending_many = Some((g, cols.len()));
        self.started = Some(Instant::now());
        self.st.on_card += 1;
        Ok(g)
    }

    /// Wait for [`Worker::begin_many`]'s product and write the card's rows of every column into `out`
    /// (`out[col · rows + row]`, the CPU's layout), and record a pending timing.
    pub fn finish_many(&mut self, rows: usize, out: &mut [f32]) -> Result<(), String> {
        let r = (|| {
            let Some((g, cols)) = self.pending_many.take() else { return Ok(()) };
            self.gpu.wait()?;
            let got = self.gpu.read_f32(&self.many.as_ref().expect("submitted").2, g * cols);
            for (c, part) in got.chunks_exact(g).enumerate() {
                out[c * rows..c * rows + g].copy_from_slice(part);
            }
            self.rest();
            Ok(())
        })();
        if let Some((shape, cand, t0)) = self.trial.take() {
            self.record(shape, cand, t0.elapsed().as_secs_f64());
        }
        self.publish();
        r
    }

    /// Several matrices by one activation (Q/K/V, gate/up): the card's share of each submitted at once, one wait for
    /// all in [`Worker::finish_group`]. `None` (do them one by one) where submits cannot overlap, while any of their
    /// shapes is still being measured, or while the card rests; otherwise the rows the card took of each (0 for a
    /// shape decided for the CPU, or a matrix over the memory limit).
    pub fn begin_group(&mut self, mats: &[(&str, &[u8], usize)], a: &Q8Act) -> Result<Option<Vec<usize>>, String> {
        let n = a.n();
        if !Gpu::MANY_IN_FLIGHT || mats.len() > Gpu::IN_FLIGHT || n > self.cap_n || self.rest_until.is_some_and(|t| Instant::now() < t) {
            return Ok(None);
        }
        let mut card = Vec::with_capacity(mats.len());
        for &(_, _, rows) in mats {
            match self.tunes.get(&(rows, n)).and_then(|t| t.chosen) {
                Some(c) => card.push(c),
                None => return Ok(None),
            }
        }
        self.trial = None;
        let mut gs = vec![0usize; mats.len()];
        let mut wrote = false;
        for (i, &(name, w, rows)) in mats.iter().enumerate() {
            if !card[i] {
                continue;
            }
            let Some(g) = self.upload(name, w, rows, n, self.share, true)? else { continue };
            if !wrote {
                let (d, q) = super::kernels::pack_act(a);
                self.gpu.write(&self.act_d, &d);
                self.gpu.write(&self.act_q, &q);
                wrote = true;
            }
            let m = &self.mats[name];
            self.gpu.submit(&self.pipe, &[&m.wd, &m.wb, &self.act_d, &self.act_q, &m.out], &[g as u32, (n / QK1_0) as u32],
                            (g as u32 * 8).div_ceil(LOCAL_SIZE))?;
            self.pending_group.push((name.to_string(), g));
            self.st.on_card += 1;
            gs[i] = g;
        }
        if wrote {
            self.started = Some(Instant::now());
        }
        Ok(Some(gs))
    }

    /// Wait once for [`Worker::begin_group`]'s products and write each card's rows into the start of its output, in
    /// the order the matrices were given (`outs[i]` for the i-th; those the card did not take are left alone).
    pub fn finish_group(&mut self, outs: &mut [&mut [f32]], taken: &[usize]) -> Result<(), String> {
        if self.pending_group.is_empty() {
            return Ok(());
        }
        let r = self.gpu.wait();
        let pending = std::mem::take(&mut self.pending_group);
        r?;
        let mut it = pending.into_iter();
        for (o, &g) in outs.iter_mut().zip(taken) {
            if g > 0 {
                let (name, g2) = it.next().ok_or("a group product went missing")?;
                debug_assert_eq!(g, g2);
                o[..g].copy_from_slice(&self.gpu.read_f32(&self.mats[&name].out, g));
            }
        }
        self.rest();
        self.publish();
        Ok(())
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
        // where the RAM available is unknown (a browser: no /proc), the heap's limit is the only one
        if let (true, Some(m)) = (self.integrated, crate::sys::memory()) {
            return held as f64 <= self.limit * (self.gpu.allocated() + m.available) as f64;
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
        let Some(g) = self.upload(name, w, rows, n, share, limited)? else { return Ok(0) };
        let m = &self.mats[name];
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

    /// `name`'s first `share` of rows on the card (uploaded on first use); the rows it holds, or `None` when, `limited`,
    /// the card rests or the matrix does not fit the memory limit (counted in the status).
    fn upload(&mut self, name: &str, w: &[u8], rows: usize, n: usize, share: f64, limited: bool) -> Result<Option<usize>, String> {
        if limited && self.rest_until.is_some_and(|t| Instant::now() < t) {
            self.st.on_cpu_resting += 1;
            self.publish();
            return Ok(None);
        }
        if !self.mats.contains_key(name) {
            let g = ((rows as f64 * share) as usize).min(rows);
            let need = (g * (n / QK1_0) * Q1_0_BYTES + g * 4) as u64;
            if limited && !self.fits(need) {
                self.st.on_cpu_memory += 1;
                self.publish();
                return Ok(None);
            }
            let (wd, wb) = pack_q1_0(w, g, n);
            let m = GMat { wd: self.gpu.upload(&wd)?, wb: self.gpu.upload(&wb)?, out: self.gpu.buffer(g * 4)?, g, shape: (rows, n) };
            self.mats.insert(name.to_string(), m);
        }
        Ok(Some(self.mats[name].g))
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
        self.rest();
        self.publish();
        Ok(())
    }

    /// Duty cycle: rest d·(1−L)/L after a dispatch of d; `busy` is the smoothed busy fraction.
    fn rest(&mut self) {
        if let Some(t) = self.started.take() {
            let d = t.elapsed();
            let rest = d.mul_f64((1.0 - self.limit) / self.limit);
            self.rest_until = (self.limit < 1.0).then(|| Instant::now() + rest);
            let frac = d.as_secs_f64() / (d + rest).as_secs_f64().max(1e-9);
            self.st.busy = if self.st.busy == 0.0 { frac } else { 0.9 * self.st.busy + 0.1 * frac };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The calibration's spread: the worker opened eight times on this card, each share printed (a measurement, not a
    /// bound: the machine's own load moves it); each share is a real one, in (0, 0.95].
    #[test]
    #[ignore = "needs a Vulkan GPU"]
    fn the_calibrated_share_is_repeatable() {
        let pool = Pool::new(2);
        let mut shares = Vec::new();
        for _ in 0..8 {
            let Some(w) = Worker::open(&pool).unwrap() else {
                eprintln!("calibration: no usable GPU on this machine; skipped");
                return;
            };
            shares.push(w.share);
        }
        let (lo, hi) = shares.iter().fold((f64::MAX, f64::MIN), |(l, h), &s| (l.min(s), h.max(s)));
        eprintln!("calibration: {} · spread {:.3}", shares.iter().map(|s| format!("{:.3}", s)).collect::<Vec<_>>().join(" "), hi - lo);
        assert!(shares.iter().all(|&s| s > 0.0 && s <= 0.95), "{shares:?}");
    }
}
