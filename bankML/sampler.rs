// SPDX-License-Identifier: MIT OR Apache-2.0
//! P3, step eleven — sampling: llama.cpp b11192's sampler chain as llama-server builds it for a chat request, token
//! for token with the same seed. The chain (`common/sampling.cpp`) is penalties → dry → top-n-σ → top-k → typical-p
//! → top-p → min-p → xtc → temperature → dist; with the Bonsai GGUF's own defaults (top-k 20, top-p 0.85, min-p 0,
//! temperature 0.5) and neutral penalties, dry, top-n-σ, typical-p and xtc, the active part is top-k, top-p, min-p,
//! temperature and the draw. Each is written from `src/llama-sampler.cpp` in its float order:
//!
//! - top-k: `std::partial_sort` of the whole vocabulary by logit (for k ≤ 128), which is libstdc++'s heap select and
//!   heap sort — ported exactly, because logits can tie (a 1-bit model's often do) and the order of equal logits is
//!   what that algorithm leaves;
//! - top-p: a float softmax over the kept tokens (`expf`, a float sum, a division), a float running sum cut where it
//!   reaches p;
//! - min-p: a cut at `max + logf(p)` on the sorted logits;
//! - temperature: `logit / temp` (at temp ≤ 0: every logit but the first maximum set to −∞);
//! - dist: `expf(logit − max)` per token summed in double, one `uniform_real_distribution<double>` draw from the
//!   request's `std::mt19937` (libstdc++'s `generate_canonical`: two 32-bit outputs), a double running sum.
//!
//! - penalties (O2): `llama_sampler_penalties`, first in the chain — over the last `penalty_last_n` tokens of the slot
//!   (llama-server accepts every prompt token into that window before the first draw, then each token drawn), a token
//!   seen `count` times has its logit divided by the repeat penalty when positive and multiplied by it otherwise, then
//!   `count · freq + present` taken off; applied again on a grammar's redraw, as the chain runs again there.
//!
//! Anything outside that — dry, typical-p, xtc, top-n-σ, dynamic temperature, top-k 0 or above 128 — is refused rather
//! than approximated.

/// `std::mt19937`, seeded as `std::mt19937(seed)`.
pub struct Mt19937 {
    mt: [u32; 624],
    i: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, i: 624 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                self.mt[k] = self.mt[(k + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `std::uniform_real_distribution<double>(0, 1)(rng)` in libstdc++: `generate_canonical<double, 53>` takes two
    /// outputs, `(a + b·2³²) / 2⁶⁴`, kept below 1.
    pub fn uniform(&mut self) -> f64 {
        let a = self.next_u32() as f64;
        let b = self.next_u32() as f64;
        let r = (a + b * 4_294_967_296.0) / 18_446_744_073_709_551_616.0;
        if r >= 1.0 { 1.0f64.next_down() } else { r }
    }
}

fn cands(logits: &[f32]) -> Vec<Cand> {
    logits.iter().enumerate().map(|(i, &l)| Cand { id: i as u32, logit: l, p: 0.0 }).collect()
}

/// `llama_token_data`.
#[derive(Clone, Copy, Debug)]
pub struct Cand {
    pub id: u32,
    pub logit: f32,
    pub p: f32,
}

// libstdc++'s heap algorithms (bits/stl_heap.h, stl_algo.h) with llama.cpp's comparator `a.logit > b.logit`
fn comp(a: &Cand, b: &Cand) -> bool {
    a.logit > b.logit
}

fn push_heap(v: &mut [Cand], mut hole: usize, top: usize, value: Cand) {
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && comp(&v[parent], &value) {
        v[hole] = v[parent];
        hole = parent;
        parent = hole.wrapping_sub(1) / 2;
    }
    v[hole] = value;
}

fn adjust_heap(v: &mut [Cand], mut hole: usize, len: usize, value: Cand) {
    let top = hole;
    let mut second = hole;
    while second < len.saturating_sub(1) / 2 {
        second = 2 * (second + 1);
        if comp(&v[second], &v[second - 1]) {
            second -= 1;
        }
        v[hole] = v[second];
        hole = second;
    }
    if len & 1 == 0 && len >= 2 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[hole] = v[second - 1];
        hole = second - 1;
    }
    push_heap(v, hole, top, value);
}

fn make_heap(v: &mut [Cand], len: usize) {
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[parent];
        adjust_heap(v, parent, len, value);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// `std::__pop_heap(first, last, result)`: the heap is `v[..last]`, the popped element goes to `v[result]`.
fn pop_heap(v: &mut [Cand], last: usize, result: usize) {
    let value = v[result];
    v[result] = v[0];
    adjust_heap(v, 0, last, value);
}

/// `std::partial_sort(v, v + middle, v + v.len(), comp)` as libstdc++ does it: heap select, then heap sort.
pub fn partial_sort(v: &mut [Cand], middle: usize) {
    make_heap(v, middle);
    for i in middle..v.len() {
        if comp(&v[i], &v[0]) {
            pop_heap(v, middle, i);
        }
    }
    let mut last = middle;
    while last > 1 {
        last -= 1;
        pop_heap(v, last, last);
    }
}

/// The request's sampling parameters, as llama-server resolves them (the request's values over the GGUF's
/// `general.sampling.*` defaults over llama.cpp's own).
#[derive(Clone, Debug)]
pub struct Params {
    pub temp: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub min_keep: usize,
    pub seed: u32,
    /// O2: `repeat_last_n` (0 turns the penalties off), `repeat_penalty`, `frequency_penalty`, `presence_penalty`
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
}

impl Params {
    /// llama-server's resolution without a request: the GGUF's `general.sampling.*` over llama.cpp's own defaults
    /// (temperature 0.8, top-k 40, top-p 0.95, min-p 0.05), a random seed.
    pub fn from_gguf(path: &std::path::Path) -> Result<Self, String> {
        let rep = crate::gguf::guard_file(path, crate::gguf::Engine::Mainline).map_err(|e| format!("{}: {e}", path.display()))?;
        let h = rep.header.ok_or("no GGUF header")?;
        let num = |k: &str, d: f64| match h.kv.get(&format!("general.sampling.{k}")) {
            Some(crate::gguf::Val::F(v)) => *v,
            Some(crate::gguf::Val::U(v)) => *v as f64,
            Some(crate::gguf::Val::I(v)) => *v as f64,
            _ => d,
        };
        // llama.cpp's defaults (common.h): penalty_last_n 64, the penalties neutral; the GGUF may set the first two
        // (`common_init_sampler_from_model`), as it may the others
        Ok(Params { temp: num("temp", 0.8) as f32, top_k: num("top_k", 40.0) as i32, top_p: num("top_p", 0.95) as f32,
                    min_p: num("min_p", 0.05) as f32, min_keep: 0, seed: DEFAULT_SEED,
                    penalty_last_n: num("penalty_last_n", 64.0) as i32, penalty_repeat: num("penalty_repeat", 1.0) as f32,
                    penalty_freq: 0.0, penalty_present: 0.0 })
    }
}

/// `LLAMA_DEFAULT_SEED`: llama.cpp then seeds from the system's random device.
pub const DEFAULT_SEED: u32 = 0xFFFF_FFFF;

pub struct Sampler {
    p: Params,
    rng: Mt19937,
    /// O2: the penalties' window (`prev`, a ring of `penalty_last_n`) and each token's count in it (`token_count`)
    prev: std::collections::VecDeque<u32>,
    counts: std::collections::HashMap<u32, i32>,
}

impl Sampler {
    pub fn new(p: Params) -> Result<Self, String> {
        if !(1..=128).contains(&p.top_k) {
            return Err(format!("top_k {}: only 1–128 is reproduced (llama.cpp sorts larger sets another way)", p.top_k));
        }
        // llama-server's own refusals, word for word (server-schema.cpp's limit on the field, then common_sampler_init)
        if p.penalty_last_n < 0 {
            return Err(format!("Field 'repeat_last_n': Value must be between 0 <= value <= 2147483647, but got {}", p.penalty_last_n));
        }
        if !p.penalty_repeat.is_finite() || p.penalty_repeat <= 0.0 || !(1.0 / p.penalty_repeat).is_finite() {
            return Err("Failed to initialize samplers: penalty_repeat must be finite and greater than 0".into());
        }
        if !p.penalty_freq.is_finite() {
            return Err("Failed to initialize samplers: penalty_freq must be finite".into());
        }
        if !p.penalty_present.is_finite() {
            return Err("Failed to initialize samplers: penalty_present must be finite".into());
        }
        let seed = if p.seed == DEFAULT_SEED {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
            (t ^ (t >> 32)) as u32 ^ std::process::id()
        } else {
            p.seed
        };
        Ok(Sampler { rng: Mt19937::new(seed), p, prev: Default::default(), counts: Default::default() })
    }

    /// One token into the penalties' window (`llama_sampler_penalties_accept`): llama-server accepts every prompt token
    /// before the first draw, then each token drawn. The count rises first; when the window is full its oldest token
    /// leaves (its count falls, and goes when it reaches 0).
    pub fn accept(&mut self, t: u32) {
        if self.p.penalty_last_n == 0 {
            return;
        }
        *self.counts.entry(t).or_insert(0) += 1;
        if self.prev.len() >= self.p.penalty_last_n as usize {
            if let Some(old) = self.prev.pop_front() {
                if let Some(c) = self.counts.get_mut(&old) {
                    *c -= 1;
                    if *c == 0 {
                        self.counts.remove(&old);
                    }
                }
            }
        }
        self.prev.push_back(t);
    }

    /// `llama_sampler_penalties::is_disabled`
    fn penalties_off(&self) -> bool {
        self.p.penalty_last_n == 0 || (self.p.penalty_repeat == 1.0 && self.p.penalty_freq == 0.0 && self.p.penalty_present == 0.0)
    }

    /// `llama_sampler_penalties_apply`, over the whole candidate list in vocabulary order.
    fn penalties(&self, c: &mut [Cand]) {
        if self.penalties_off() {
            return;
        }
        for x in c.iter_mut() {
            let Some(&count) = self.counts.get(&x.id) else { continue };
            if x.logit <= 0.0 {
                x.logit *= self.p.penalty_repeat;
            } else {
                x.logit /= self.p.penalty_repeat;
            }
            x.logit -= count as f32 * self.p.penalty_freq + (count > 0) as i32 as f32 * self.p.penalty_present;
        }
    }

    /// One token from the logits, through the chain; the RNG advances once per token, as llama.cpp's does.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        self.chain(cands(logits))
    }

    /// One token under a grammar, as `common_sampler_sample` (`grammar_first = false`) draws it: the chain on the
    /// raw logits; if `allows` the token, it stands (one draw). Otherwise the logits are taken afresh, `mask` sets
    /// every token the grammar rejects to −∞, and the chain runs again — a second draw from the same generator.
    /// Returns the token and whether it had to be drawn again under the mask.
    pub fn sample_constrained(&mut self, logits: &[f32], allows: impl Fn(u32) -> bool, mask: impl Fn(&mut [Cand])) -> (u32, bool) {
        let id = self.chain(cands(logits));
        if allows(id) {
            return (id, false);
        }
        let mut c = cands(logits);
        mask(&mut c);
        (self.chain(c), true)
    }

    /// The chain on a candidate list in vocabulary order: penalties, top-k, top-p, min-p, temperature, dist.
    fn chain(&mut self, mut c: Vec<Cand>) -> u32 {
        self.penalties(&mut c);
        // top-k (k ≤ 128: std::partial_sort in place; the set is sorted from here on)
        let k = (self.p.top_k as usize).min(c.len());
        partial_sort(&mut c, k);
        c.truncate(k);
        // top-p
        if self.p.top_p < 1.0 {
            let max_l = c[0].logit;
            let mut cum = 0.0f32;
            for x in c.iter_mut() {
                x.p = (x.logit - max_l).exp();
                cum += x.p;
            }
            for x in c.iter_mut() {
                x.p /= cum;
            }
            let mut run = 0.0f32;
            let mut last = c.len();
            for (i, x) in c.iter().enumerate() {
                run += x.p;
                if run >= self.p.top_p && i + 1 >= self.p.min_keep {
                    last = i + 1;
                    break;
                }
            }
            c.truncate(last);
        }
        // min-p (sorted: the second implementation)
        if self.p.min_p > 0.0 {
            let min_logit = c[0].logit + self.p.min_p.ln();
            let mut i = 1;
            while i < c.len() {
                if c[i].logit < min_logit && i >= self.p.min_keep {
                    break;
                }
                i += 1;
            }
            c.truncate(i);
        }
        // temperature
        if self.p.temp <= 0.0 {
            let (mut max_i, mut max_l) = (0, c[0].logit);
            for i in 1..c.len() {
                if c[i].logit > max_l {
                    c[max_i].logit = f32::NEG_INFINITY;
                    max_i = i;
                    max_l = c[i].logit;
                } else {
                    c[i].logit = f32::NEG_INFINITY;
                }
            }
        } else {
            for x in c.iter_mut() {
                x.logit /= self.p.temp;
            }
        }
        // dist
        if c.len() == 1 {
            self.rng.uniform();
            return c[0].id;
        }
        let max_l = c[0].logit;
        let mut sum = 0.0f64;
        for x in c.iter_mut() {
            x.p = (x.logit - max_l).exp();
            sum += x.p as f64;
        }
        let target = sum * self.rng.uniform();
        let mut run = 0.0f64;
        for x in &c {
            run += x.p as f64;
            if run >= target {
                return x.id;
            }
        }
        c[c.len() - 1].id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_reference_value() {
        // the C++ standard's check: the 10000th output of a default-seeded (5489) mt19937 is 4123659995
        let mut r = Mt19937::new(5489);
        let mut x = 0;
        for _ in 0..10_000 {
            x = r.next_u32();
        }
        assert_eq!(x, 4_123_659_995);
    }

    #[test]
    fn partial_sort_orders_the_top() {
        let v: Vec<f32> = (0..500).map(|i| ((i * 7919) % 101) as f32 * 0.5 - 20.0).collect();
        let mut c: Vec<Cand> = v.iter().enumerate().map(|(i, &l)| Cand { id: i as u32, logit: l, p: 0.0 }).collect();
        partial_sort(&mut c, 40);
        let mut s = v.clone();
        s.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(c[..40].iter().zip(&s[..40]).all(|(a, &b)| a.logit == b));
    }
}
