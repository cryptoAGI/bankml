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
    /// `std::uniform_real_distribution<float>(0, 1)(rng)` in libstdc++: `generate_canonical<float, 24>` takes one
    /// output, `x / 2³²` in float, kept below 1 (XTC's draw).
    pub fn uniform_f32(&mut self) -> f32 {
        let r = self.next_u32() as f32 / 4_294_967_296.0f32;
        if r >= 1.0 { 1.0f32.next_down() } else { r }
    }

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

// libstdc++'s heap and sort algorithms (bits/stl_heap.h, stl_algo.h), generic over the comparator; llama.cpp's
// comparator for candidates is `a.logit > b.logit`
fn comp(a: &Cand, b: &Cand) -> bool {
    a.logit > b.logit
}

fn push_heap<T: Copy>(v: &mut [T], mut hole: usize, top: usize, value: T, less: &impl Fn(&T, &T) -> bool) {
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && less(&v[parent], &value) {
        v[hole] = v[parent];
        hole = parent;
        parent = hole.wrapping_sub(1) / 2;
    }
    v[hole] = value;
}

fn adjust_heap<T: Copy>(v: &mut [T], mut hole: usize, len: usize, value: T, less: &impl Fn(&T, &T) -> bool) {
    let top = hole;
    let mut second = hole;
    while second < len.saturating_sub(1) / 2 {
        second = 2 * (second + 1);
        if less(&v[second], &v[second - 1]) {
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
    push_heap(v, hole, top, value, less);
}

fn make_heap<T: Copy>(v: &mut [T], len: usize, less: &impl Fn(&T, &T) -> bool) {
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[parent];
        adjust_heap(v, parent, len, value, less);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// `std::__pop_heap(first, last, result)`: the heap is `v[..last]`, the popped element goes to `v[result]`.
fn pop_heap<T: Copy>(v: &mut [T], last: usize, result: usize, less: &impl Fn(&T, &T) -> bool) {
    let value = v[result];
    v[result] = v[0];
    adjust_heap(v, 0, last, value, less);
}

/// `std::partial_sort(v, v + middle, v + v.len(), less)`: heap select, then heap sort.
fn partial_sort_by<T: Copy>(v: &mut [T], middle: usize, less: &impl Fn(&T, &T) -> bool) {
    make_heap(v, middle, less);
    for i in middle..v.len() {
        if less(&v[i], &v[0]) {
            pop_heap(v, middle, i, less);
        }
    }
    let mut last = middle;
    while last > 1 {
        last -= 1;
        pop_heap(v, last, last, less);
    }
}

/// `std::partial_sort` of candidates by logit, descending, as llama.cpp's top-k and sorted softmax call it.
pub fn partial_sort(v: &mut [Cand], middle: usize) {
    partial_sort_by(v, middle, &comp);
}

/// `std::sort` as libstdc++ does it (introsort, depth 2·⌊log₂ n⌋, threshold 16, then insertion sort): not stable,
/// so equal keys end where this algorithm leaves them — llama.cpp's typical-p sorts its scores this way.
pub fn sort_by<T: Copy>(v: &mut [T], less: &impl Fn(&T, &T) -> bool) {
    const THRESHOLD: usize = 16;
    fn move_median_to_first<T: Copy>(v: &mut [T], result: usize, a: usize, b: usize, c: usize, less: &impl Fn(&T, &T) -> bool) {
        let pick = if less(&v[a], &v[b]) {
            if less(&v[b], &v[c]) { b } else if less(&v[a], &v[c]) { c } else { a }
        } else if less(&v[a], &v[c]) {
            a
        } else if less(&v[b], &v[c]) {
            c
        } else {
            b
        };
        v.swap(result, pick);
    }
    fn unguarded_partition<T: Copy>(v: &mut [T], mut first: usize, mut last: usize, pivot: usize, less: &impl Fn(&T, &T) -> bool) -> usize {
        loop {
            while less(&v[first], &v[pivot]) {
                first += 1;
            }
            last -= 1;
            while less(&v[pivot], &v[last]) {
                last -= 1;
            }
            if first >= last {
                return first;
            }
            v.swap(first, last);
            first += 1;
        }
    }
    fn introsort_loop<T: Copy>(v: &mut [T], first: usize, mut last: usize, mut depth: usize, less: &impl Fn(&T, &T) -> bool) {
        while last - first > THRESHOLD {
            if depth == 0 {
                partial_sort_by(&mut v[first..last], last - first, less);
                return;
            }
            depth -= 1;
            let mid = first + (last - first) / 2;
            move_median_to_first(v, first, first + 1, mid, last - 1, less);
            let cut = unguarded_partition(v, first + 1, last, first, less);
            introsort_loop(v, cut, last, depth, less);
            last = cut;
        }
    }
    fn unguarded_linear_insert<T: Copy>(v: &mut [T], mut last: usize, less: &impl Fn(&T, &T) -> bool) {
        let val = v[last];
        let mut next = last - 1;
        while less(&val, &v[next]) {
            v[last] = v[next];
            last = next;
            next -= 1;
        }
        v[last] = val;
    }
    fn insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, less: &impl Fn(&T, &T) -> bool) {
        if first == last {
            return;
        }
        for i in first + 1..last {
            if less(&v[i], &v[first]) {
                let val = v[i];
                v.copy_within(first..i, first + 1);
                v[first] = val;
            } else {
                unguarded_linear_insert(v, i, less);
            }
        }
    }
    let n = v.len();
    if n < 2 {
        return;
    }
    introsort_loop(v, 0, n, 2 * (usize::BITS - 1 - n.leading_zeros()) as usize, less);
    if n > THRESHOLD {
        insertion_sort(v, 0, THRESHOLD, less);
        for i in THRESHOLD..n {
            unguarded_linear_insert(v, i, less);
        }
    } else {
        insertion_sort(v, 0, n, less);
    }
}

/// The request's sampling parameters, as llama-server resolves them (the request's values over the GGUF's
/// `general.sampling.*` defaults over llama.cpp's own, `common.h`).
#[derive(Clone, Debug)]
pub struct Params {
    pub temp: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub min_keep: usize,
    pub seed: u32,
    /// `repeat_last_n` (0 turns the penalties off), `repeat_penalty`, `frequency_penalty`, `presence_penalty` (0.3.6)
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    /// `typical_p` (1 = off), `top_n_sigma` (≤ 0 = off), `xtc_probability` (0 = off), `xtc_threshold` (> 0.5 = off),
    /// `dynatemp_range` (0 = off), `dynatemp_exponent` (0.3.7)
    pub typical_p: f32,
    pub top_n_sigma: f32,
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    /// DRY (0.3.7): `dry_multiplier` (0 = off), `dry_base`, `dry_allowed_length`, `dry_penalty_last_n` (0 = off) and the
    /// sequence breakers, whose token sequences the engine supplies (`Sampler::set_dry_breakers`)
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    pub dry_sequence_breakers: Vec<String>,
}

impl Default for Params {
    /// llama.cpp's own defaults (`common_params_sampling`, common.h), a random seed.
    fn default() -> Self {
        Params { temp: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, min_keep: 0, seed: DEFAULT_SEED,
                 penalty_last_n: 64, penalty_repeat: 1.0, penalty_freq: 0.0, penalty_present: 0.0,
                 typical_p: 1.0, top_n_sigma: -1.0, xtc_probability: 0.0, xtc_threshold: 0.1, dynatemp_range: 0.0, dynatemp_exponent: 1.0,
                 dry_multiplier: 0.0, dry_base: 1.75, dry_allowed_length: 2, dry_penalty_last_n: 64,
                 dry_sequence_breakers: ["\n", ":", "\"", "*"].map(String::from).to_vec() }
    }
}

impl Params {
    /// llama-server's resolution without a request: the GGUF's `general.sampling.*` (the keys
    /// `common_init_sampler_from_model` reads) over llama.cpp's defaults, a random seed.
    pub fn from_gguf(path: &std::path::Path) -> Result<Self, String> {
        let rep = crate::gguf::guard_file(path, crate::gguf::Engine::Mainline).map_err(|e| format!("{}: {e}", path.display()))?;
        let h = rep.header.ok_or("no GGUF header")?;
        let num = |k: &str, d: f64| match h.kv.get(&format!("general.sampling.{k}")) {
            Some(crate::gguf::Val::F(v)) => *v,
            Some(crate::gguf::Val::U(v)) => *v as f64,
            Some(crate::gguf::Val::I(v)) => *v as f64,
            _ => d,
        };
        let d = Params::default();
        Ok(Params { temp: num("temp", d.temp as f64) as f32, top_k: num("top_k", d.top_k as f64) as i32,
                    top_p: num("top_p", d.top_p as f64) as f32, min_p: num("min_p", d.min_p as f64) as f32,
                    penalty_last_n: num("penalty_last_n", d.penalty_last_n as f64) as i32,
                    penalty_repeat: num("penalty_repeat", d.penalty_repeat as f64) as f32,
                    xtc_probability: num("xtc_probability", d.xtc_probability as f64) as f32,
                    xtc_threshold: num("xtc_threshold", d.xtc_threshold as f64) as f32, ..d })
    }

    fn dry_on(&self) -> bool {
        self.dry_multiplier != 0.0 && self.dry_base >= 1.0 && self.dry_penalty_last_n != 0
    }
}

/// `LLAMA_DEFAULT_SEED`: llama.cpp then seeds from the system's random device.
pub const DEFAULT_SEED: u32 = 0xFFFF_FFFF;

/// DRY's processed sequence breakers (`get_overlapping_token_sequences`): for each head token, the tails of the
/// breaker sequences it begins (an empty tail: the token holds a whole breaker).
pub type DryBreakers = std::collections::HashMap<u32, Vec<Vec<u32>>>;

/// DRY's breakers for a vocabulary, as `llama_sampler_init_dry` builds them: each breaker cut to 40 bytes; a token
/// whose piece contains it is a single-token breaker; a token whose piece ends with the breaker's start is a head, its
/// tail the rest of the breaker tokenized (`encode`, no specials), cut to 20 tokens. `pieces` are the tokens'
/// `token_to_piece(special = true)` bytes.
pub fn dry_breakers(breakers: &[String], pieces: &[Vec<u8>], encode: impl Fn(&str) -> Vec<u32>) -> Result<DryBreakers, String> {
    const MAX_CHAR_LEN: usize = 40;
    const MAX_SEQ_LEN: usize = 20;
    let mut out = DryBreakers::new();
    for b in breakers.iter().filter(|b| !b.is_empty()) {
        let s = &b.as_bytes()[..b.len().min(MAX_CHAR_LEN)];
        for (id, word) in pieces.iter().enumerate() {
            let id = id as u32;
            if word.windows(s.len()).any(|w| w == s) {
                out.entry(id).or_default().push(Vec::new());
                continue;
            }
            let mut from = 0;
            while let Some(off) = word[from..].iter().position(|&c| c == s[0]) {
                let pos = from + off;
                let mut i = 1;
                let mut matched = true;
                while i < s.len() && pos + i < word.len() {
                    if word[pos + i] != s[i] {
                        matched = false;
                        break;
                    }
                    i += 1;
                }
                if matched {
                    let rest = std::str::from_utf8(&s[i..]).map_err(|_| format!("dry_sequence_breakers: {b:?} splits a character where a token ends; bankML tokenizes only whole characters"))?;
                    let mut t = encode(rest);
                    t.truncate(MAX_SEQ_LEN);
                    let tails = out.entry(id).or_default();
                    if !tails.contains(&t) {
                        tails.push(t);
                    }
                }
                from = pos + 1;
            }
        }
    }
    Ok(out)
}

pub struct Sampler {
    p: Params,
    rng: Mt19937,
    /// XTC's own generator (`llama_sampler_xtc`), seeded as the draw's is
    xtc_rng: Mt19937,
    /// the penalties' window (`prev`, a ring of `penalty_last_n`) and each token's count in it (`token_count`)
    prev: std::collections::VecDeque<u32>,
    counts: std::collections::HashMap<u32, i32>,
    /// DRY's window (a ring of `dry_penalty_last_n`) and its breakers
    dry_last: std::collections::VecDeque<u32>,
    dry_breakers: std::sync::Arc<DryBreakers>,
}

/// `get_rng_seed`: the seed, or one from the clock and process when it is `LLAMA_DEFAULT_SEED`.
fn rng_seed(seed: u32) -> u32 {
    if seed == DEFAULT_SEED {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        (t ^ (t >> 32)) as u32 ^ std::process::id()
    } else {
        seed
    }
}

impl Sampler {
    pub fn new(p: Params) -> Result<Self, String> {
        if !(1..=128).contains(&p.top_k) {
            return Err(format!("top_k {}: only 1–128 is reproduced (llama.cpp sorts larger sets another way)", p.top_k));
        }
        // llama-server's own refusals, word for word (server-schema.cpp's hard limits, then common_sampler_init)
        for (name, v) in [("repeat_last_n", p.penalty_last_n), ("dry_allowed_length", p.dry_allowed_length), ("dry_penalty_last_n", p.dry_penalty_last_n)] {
            if v < 0 {
                return Err(format!("Field '{name}': Value must be between 0 <= value <= 2147483647, but got {v}"));
            }
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
        if p.dry_sequence_breakers.is_empty() {
            return Err("Field 'dry_sequence_breakers': Error: dry_sequence_breakers must be a non-empty array of strings".into());
        }
        let dry_cap = if p.dry_on() { p.dry_penalty_last_n as usize } else { 0 };
        Ok(Sampler { rng: Mt19937::new(rng_seed(p.seed)), xtc_rng: Mt19937::new(rng_seed(p.seed)), p, prev: Default::default(),
                     counts: Default::default(), dry_last: std::collections::VecDeque::with_capacity(dry_cap.min(1 << 16)),
                     dry_breakers: Default::default() })
    }

    /// Whether DRY is on, and so needs its breakers (`set_dry_breakers`).
    pub fn wants_dry_breakers(&self) -> bool {
        self.p.dry_on()
    }

    /// DRY's processed breakers for the model's vocabulary (`dry_breakers`), shared with the engine's cache.
    pub fn set_dry_breakers(&mut self, b: std::sync::Arc<DryBreakers>) {
        self.dry_breakers = b;
    }

    /// The DRY sequence breakers the request asked for.
    pub fn dry_sequence_breakers(&self) -> &[String] {
        &self.p.dry_sequence_breakers
    }

    /// One token into the samplers' windows (`llama_sampler_penalties_accept`, `llama_sampler_dry_accept`):
    /// llama-server accepts every prompt token before the first draw, then each token drawn.
    pub fn accept(&mut self, t: u32) {
        if self.p.penalty_last_n != 0 {
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
        if self.p.dry_on() {
            if self.dry_last.len() >= self.p.dry_penalty_last_n as usize {
                self.dry_last.pop_front();
            }
            self.dry_last.push_back(t);
        }
    }

    /// `llama_sampler_penalties_apply`, over the whole candidate list in vocabulary order.
    fn penalties(&self, c: &mut [Cand]) {
        if self.p.penalty_last_n == 0 || (self.p.penalty_repeat == 1.0 && self.p.penalty_freq == 0.0 && self.p.penalty_present == 0.0) {
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

    /// `llama_sampler_dry_apply` (KoboldCpp's DRY): a token that would extend a repeat of at least
    /// `dry_allowed_length` tokens loses `multiplier · base^(length − allowed)`; breakers cap the repeat length.
    fn dry(&self, c: &mut [Cand]) {
        if !self.p.dry_on() {
            return;
        }
        let n = self.dry_last.len().min(self.p.dry_penalty_last_n as usize) as i32;
        let allowed = self.p.dry_allowed_length;
        if n <= allowed {
            return;
        }
        // rat(i): the i-th token from the end
        let rat = |i: i32| self.dry_last[self.dry_last.len() - 1 - i as usize];
        // step 1: the nearest restart sequence limits the repeat length
        let mut rep_limit = n;
        for i in 0..n {
            let Some(tails) = self.dry_breakers.get(&rat(i)) else { continue };
            let mut longest = -1;
            for t in tails {
                let len = t.len() as i32;
                if len > longest && len <= i && (0..len).all(|o| t[o as usize] == rat(i - o - 1)) {
                    longest = len;
                }
            }
            if longest >= 0 {
                rep_limit = i - longest;
                break;
            }
        }
        if rep_limit < allowed {
            return;
        }
        // step 2: the reverse Z-algorithm — repeat[last − k] is how far the suffix repeats ending k tokens back
        let mut repeat = vec![0i32; n as usize];
        let last = n - 1;
        let (mut lt, mut rt) = (0i32, 0i32);
        for k in 1..n {
            if k > rt {
                let mut m = 0;
                while m + k < n && rat(m) == rat(m + k) {
                    m += 1;
                }
                repeat[(last - k) as usize] = m.min(rep_limit);
                if m > 0 {
                    lt = k;
                    rt = k + m - 1;
                }
            } else {
                let p = k - lt;
                let right = rt - k + 1;
                if repeat[(last - p) as usize] < right {
                    repeat[(last - k) as usize] = repeat[(last - p) as usize].min(rep_limit);
                } else {
                    let mut i = rt + 1;
                    while i < n && rat(i) == rat(i - k) {
                        i += 1;
                    }
                    repeat[(last - k) as usize] = (i - k).min(rep_limit);
                    lt = k;
                    rt = i - 1;
                }
            }
        }
        // step 3: the longest repeat each next token would extend
        let mut max_repeat = std::collections::HashMap::<u32, i32>::new();
        for i in 0..n - 1 {
            let len = repeat[i as usize];
            if len >= allowed {
                let t = rat(n - 2 - i);
                let e = max_repeat.entry(t).or_insert(len);
                if *e < len {
                    *e = len;
                }
            }
        }
        // step 4: the penalty, its exponent clamped so `pow` stays finite; a single-token breaker is never penalised
        const FLOAT_MAX_LOG: f32 = 88.722_84;
        let max_exponent = if self.p.dry_base > 1.000_001 { (FLOAT_MAX_LOG / self.p.dry_base.ln()) as i32 } else { 0 };
        for x in c.iter_mut() {
            let Some(&len) = max_repeat.get(&x.id) else { continue };
            if self.dry_breakers.get(&x.id).is_some_and(|ts| ts.iter().any(Vec::is_empty)) {
                continue;
            }
            let mut e = len - allowed;
            if max_exponent > 0 && e > max_exponent {
                e = max_exponent;
            }
            // std::pow(float, int) is libm's pow in double (not repeated multiplication), then narrowed
            let penalty = (self.p.dry_multiplier as f64 * (self.p.dry_base as f64).powf(e as f64)) as f32;
            x.logit -= penalty;
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

    /// The chain on a candidate list in vocabulary order: penalties, DRY, top-n-σ, top-k, typical-p, top-p, min-p,
    /// XTC, temperature (dynamic when `dynatemp_range` > 0), dist — each as `src/llama-sampler.cpp` writes it,
    /// tracking `sorted` as llama.cpp's array does.
    fn chain(&mut self, mut c: Vec<Cand>) -> u32 {
        self.penalties(&mut c);
        self.dry(&mut c);
        top_n_sigma(&mut c, self.p.top_n_sigma);
        // top-k (k ≤ 128: std::partial_sort in place; the set is sorted from here on)
        let k = (self.p.top_k as usize).min(c.len());
        partial_sort(&mut c, k);
        c.truncate(k);
        let mut sorted = true;
        if self.p.typical_p < 1.0 {
            typical(&mut c, self.p.typical_p, self.p.min_keep);
            sorted = false;
        }
        if self.p.top_p < 1.0 {
            softmax(&mut c, sorted);
            if !sorted {
                let n = c.len();
                partial_sort(&mut c, n);
                sorted = true;
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
        if self.p.min_p > 0.0 && !c.is_empty() {
            let mut applied = false;
            if !sorted {
                // the unsorted implementation, kept when enough tokens pass
                let max = c.iter().fold(-f32::MAX, |m, x| if m < x.logit { x.logit } else { m });
                let min_logit = max + self.p.min_p.ln();
                let kept: Vec<Cand> = c.iter().copied().filter(|x| x.logit >= min_logit).collect();
                if !kept.is_empty() && kept.len() >= self.p.min_keep {
                    c = kept;
                    applied = true;
                }
            }
            if !applied {
                if !sorted {
                    let n = c.len();
                    partial_sort(&mut c, n);
                    sorted = true;
                }
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
        }
        // XTC: its own draw decides; then the most probable tokens above the threshold go, but the last of them stays
        if self.p.xtc_probability > 0.0 && self.p.xtc_threshold <= 0.5 && c.len() >= 2 {
            let chance = self.xtc_rng.uniform_f32();
            if chance <= self.p.xtc_probability {
                if !sorted {
                    let n = c.len();
                    partial_sort(&mut c, n);
                    sorted = true;
                }
                softmax(&mut c, true);
                let mut pos_last = 0;
                for (i, x) in c.iter().enumerate() {
                    if x.p >= self.p.xtc_threshold {
                        pos_last = i;
                    } else {
                        break;
                    }
                }
                if c.len() - pos_last >= self.p.min_keep && pos_last > 0 {
                    c.drain(..pos_last);
                }
            }
        }
        // temperature, dynamic when a range is set (`llama_sampler_temp_ext`)
        if self.p.dynatemp_range > 0.0 && c.len() > 1 {
            let min_t = (self.p.temp - self.p.dynatemp_range).max(0.0);
            let max_t = self.p.temp + self.p.dynatemp_range;
            let max_entropy = -(1.0f32 / c.len() as f32).ln();
            if !sorted {
                let n = c.len();
                partial_sort(&mut c, n);
                sorted = true;
            }
            softmax(&mut c, true);
            let mut entropy = 0.0f32;
            for x in &c {
                if x.p > 0.0 {
                    entropy -= x.p * x.p.ln();
                }
            }
            let t = min_t + (max_t - min_t) * (entropy / max_entropy).powf(self.p.dynatemp_exponent);
            temperature(&mut c, t);
        } else if self.p.dynatemp_range <= 0.0 {
            temperature(&mut c, self.p.temp);
        }
        dist(&mut self.rng, &c, sorted)
    }
}

/// `llama_sampler_softmax_impl(cur_p, do_sort = false)` on a list in its current order: the maximum (the first
/// logit when sorted), then `expf` and a float sum in list order, then the division.
fn softmax(c: &mut [Cand], sorted: bool) {
    if c.is_empty() {
        return;
    }
    let mut max_l = c[0].logit;
    if !sorted {
        for x in &c[1..] {
            max_l = max_l.max(x.logit);
        }
    }
    let mut cum = 0.0f32;
    for x in c.iter_mut() {
        x.p = (x.logit - max_l).exp();
        cum += x.p;
    }
    for x in c.iter_mut() {
        x.p /= cum;
    }
}

/// `llama_sampler_top_n_sigma_apply`: logits below `max − n·σ` (over the finite ones) become −∞. The squares are
/// taken in double and added back into a float, as C++'s `pow(float, 2)` does.
fn top_n_sigma(c: &mut [Cand], n: f32) {
    if n <= 0.0 || c.len() <= 1 {
        return;
    }
    let mut max = c[0].logit;
    let (mut sum, mut count) = (0.0f32, 0usize);
    for x in c.iter() {
        if x.logit != f32::NEG_INFINITY {
            max = max.max(x.logit);
            sum += x.logit;
            count += 1;
        }
    }
    let mean = if count > 0 { sum / count as f32 } else { 0.0 };
    let mut acc = 0.0f32;
    for x in c.iter() {
        if x.logit != f32::NEG_INFINITY {
            let d = (x.logit - mean) as f64;
            acc = (acc as f64 + d * d) as f32;
        }
    }
    let std = if count > 0 { (acc / count as f32).sqrt() } else { 0.0 };
    let cut = max - n * std;
    for x in c.iter_mut() {
        if x.logit < cut {
            x.logit = f32::NEG_INFINITY;
        }
    }
}

/// `llama_sampler_typical_apply` on a sorted list: the entropy of the softmax, each token's distance from it,
/// the indices sorted by that distance with libstdc++'s `std::sort`, kept while the running probability is at most
/// `p` (and `min_keep`). The result is in distance order (unsorted by logit).
fn typical(c: &mut Vec<Cand>, p: f32, min_keep: usize) {
    softmax(c, true);
    let mut entropy = 0.0f32;
    for x in c.iter() {
        entropy += -x.p * x.p.ln();
    }
    let shifted: Vec<f32> = c.iter().map(|x| (-x.p.ln() - entropy).abs()).collect();
    let mut idx: Vec<usize> = (0..c.len()).collect();
    sort_by(&mut idx, &|a: &usize, b: &usize| shifted[*a] < shifted[*b]);
    let mut cum = 0.0f32;
    let mut last = idx.len();
    for (i, &j) in idx.iter().enumerate() {
        cum += c[j].p;
        if cum > p && (min_keep == 0 || i + 1 >= min_keep) {
            last = i + 1;
            break;
        }
    }
    *c = idx[..last].iter().map(|&j| c[j]).collect();
}

/// `llama_sampler_temp_impl`: `logit / temp`; at temp ≤ 0 every logit but the first maximum becomes −∞.
fn temperature(c: &mut [Cand], temp: f32) {
    if c.is_empty() {
        return;
    }
    if temp <= 0.0 {
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
            x.logit /= temp;
        }
    }
}

/// `llama_sampler_dist_apply`: `expf(logit − max)` per token summed in double, one `uniform_real_distribution<double>`
/// draw, a double running sum; one candidate still costs a draw.
fn dist(rng: &mut Mt19937, c: &[Cand], sorted: bool) -> u32 {
    if c.len() == 1 {
        rng.uniform();
        return c[0].id;
    }
    let mut max_l = c[0].logit;
    if !sorted {
        for x in &c[1..] {
            max_l = max_l.max(x.logit);
        }
    }
    let ps: Vec<f32> = c.iter().map(|x| (x.logit - max_l).exp()).collect();
    let sum: f64 = ps.iter().map(|&p| p as f64).sum();
    let target = sum * rng.uniform();
    let mut run = 0.0f64;
    for (x, &p) in c.iter().zip(&ps) {
        run += p as f64;
        if run >= target {
            return x.id;
        }
    }
    c[c.len() - 1].id
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

    /// 0.3.7: libstdc++'s own `std::sort` (testing/sort_oracle.cpp) on 876 key arrays, sizes 0–1000, heavy ties,
    /// sorted, reversed and equal keys: `sort_by` leaves every one in the same order (typical-p depends on it).
    #[test]
    #[ignore = "needs .models/oracle-sort/cases.txt (g++ -O2 testing/sort_oracle.cpp, then run it)"]
    fn oracle_std_sort() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/oracle-sort/cases.txt");
        let text = std::fs::read_to_string(path).unwrap();
        let (mut n_cases, mut bad) = (0, 0);
        for line in text.lines() {
            let f: Vec<&str> = line.split(' ').collect();
            let n: usize = f[0].parse().unwrap();
            let key: Vec<f32> = f[1..1 + n].iter().map(|h| f32::from_bits(u32::from_str_radix(h, 16).unwrap())).collect();
            let want: Vec<usize> = f[1 + n..].iter().map(|x| x.parse().unwrap()).collect();
            let mut idx: Vec<usize> = (0..n).collect();
            sort_by(&mut idx, &|a: &usize, b: &usize| key[*a] < key[*b]);
            n_cases += 1;
            if idx != want {
                bad += 1;
                eprintln!("  n = {n}: first difference at {:?}", idx.iter().zip(&want).position(|(a, b)| a != b));
            }
        }
        eprintln!("std::sort oracle: {} of {n_cases} orders identical to libstdc++'s", n_cases - bad);
        assert_eq!(bad, 0);
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
