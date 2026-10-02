// SPDX-License-Identifier: MIT OR Apache-2.0
//! P2 — the ternary kernel: `Q2_0` (ggml type 42, group 64), as mainline ggml defines it at **b11192**
//! (read from the tag's source 2026-09-26):
//!
//! - `ggml-common.h`: `block_q2_0 { ggml_half d; uint8_t qs[16]; }` — 18 bytes, scale first, 2.25 bpw.
//! - `ggml-quants.c: dequantize_row_q2_0`: weight j is bits `2·(j%4)` of byte `j/4` (LSB first);
//!   code c ∈ {0,1,2,3} → `(c − 1)·d`, i.e. {−1, 0, +1, **+2**}. The Bonsai files are ternary, the format
//!   is not: code 3 is legal and every kernel here takes it.
//! - `ggml-cpu.c` traits: `vec_dot = ggml_vec_dot_q2_0_q8_0`, `vec_dot_type = Q8_0`, one Q2_0 block
//!   meets two q8_0 blocks (the same AVX2 q8_0 quantizer as Q1_0, `q1_0::quantize_row_q8_0`).
//! - **x86 has no Q2_0 kernel.** `arch-fallback.h` renames `ggml_vec_dot_q2_0_q8_0_generic` to
//!   `ggml_vec_dot_q2_0_q8_0` on x86; `repack.cpp` has no Q2_0 case and llamafile's sgemm none either.
//!   The shipped haswell binary's symbol is that scalar C, unvectorised: 64 `imul`s per block, then
//!   `vfmadd231ss` for `sumi += d1·s` (twice) and `sumf += d0·sumi` (read from the disassembly). Decode
//!   *and* prefill run through it one (row, column) pair at a time. That is where the 5× goes.
//!
//! Bit-exactness: the two integer sums per block are exact in any order, so the AVX2 path may reorder
//! the activation (once per token, `Q8Act2`) and use `maddubs`/`madd`; the float ops are ggml's, in
//! ggml's order: `sumi = fma(d1₁, s₁, fma(d1₀, s₀, 0))`, `sumf = fma(d0, sumi, sumf)` block after block.

use crate::q1_0::{f16_to_f32, quantize_row_q8_0, Q8_0_BYTES, QK8_0};

pub const QK2_0: usize = 64;
pub const Q2_0_BYTES: usize = 18;

/// Dequantize n Q2_0 blocks into `out` (len n·64). Bit-exact with `dequantize_row_q2_0`.
pub fn dequantize_row(blocks: &[u8], out: &mut [f32]) {
    assert_eq!(blocks.len() % Q2_0_BYTES, 0);
    assert_eq!(out.len(), blocks.len() / Q2_0_BYTES * QK2_0);
    for (b, o) in blocks.as_chunks::<Q2_0_BYTES>().0.iter().zip(out.as_chunks_mut::<QK2_0>().0.iter_mut()) {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for (j, y) in o.iter_mut().enumerate() {
            *y = (((b[2 + j / 4] >> (2 * (j % 4))) & 3) as i32 - 1) as f32 * d;
        }
    }
}

fn check(n: usize, x: &[u8], y: &[u8]) {
    assert_eq!(n % QK2_0, 0);
    assert!(x.len() >= n / QK2_0 * Q2_0_BYTES && y.len() >= n / QK8_0 * Q8_0_BYTES);
}

/// Scalar model of the shipped `ggml_vec_dot_q2_0_q8_0` (generic C, FMA-contracted by GCC).
pub fn vec_dot_ref(n: usize, x: &[u8], y: &[u8]) -> f32 {
    check(n, x, y);
    let mut sumf = 0.0f32;
    for i in 0..n / QK2_0 {
        let xb = &x[i * Q2_0_BYTES..];
        let d0 = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let mut sumi = 0.0f32;
        for k in 0..2 {
            let yb = &y[(i * 2 + k) * Q8_0_BYTES..];
            let d1 = f16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
            let mut s = 0i32;
            for e in 0..32 {
                s += (((xb[2 + k * 8 + e / 4] >> (2 * (e % 4))) & 3) as i32 - 1) * (yb[2 + e] as i8 as i32);
            }
            sumi = d1.mul_add(s as f32, sumi);
        }
        sumf = d0.mul_add(sumi, sumf);
    }
    sumf
}

/// The same C as built by ggml's baseline `libggml-cpu-x64.so` (SSE only, no FMA): `sumi += d1·s` as
/// mul then add, `sumf += sumi·d0` likewise. Not the node's kernel; kept so both shipped variants are proven.
pub fn vec_dot_ref_nofma(n: usize, x: &[u8], y: &[u8]) -> f32 {
    check(n, x, y);
    let mut sumf = 0.0f32;
    for i in 0..n / QK2_0 {
        let xb = &x[i * Q2_0_BYTES..];
        let mut sumi = 0.0f32;
        for k in 0..2 {
            let yb = &y[(i * 2 + k) * Q8_0_BYTES..];
            let s: i32 = (0..32).map(|e| (((xb[2 + k * 8 + e / 4] >> (2 * (e % 4))) & 3) as i32 - 1) * (yb[2 + e] as i8 as i32)).sum();
            sumi += s as f32 * f16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
        }
        sumf += sumi * f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
    }
    sumf
}

/// A q8_0 activation prepared once per token for Q2_0 rows, laid out for the AVX2 kernel so that no
/// shuffle ever touches the weights: blocks go in quads (A,B,C,D); a 256-bit register holds the 16 code
/// bytes of A (low half) and B (high half), and `(v >> 2r) & 3` is code r of every byte — so the
/// activation stores, per pair and per r, the 16 quants of elements 4b+r of A then of B. `d` (f32, exact)
/// and `sum` (Σq per q8 block, for Σ(c−1)·q = Σc·q − Σq with c ∈ {0..3} unsigned) are stored in the
/// lane order `hadd` leaves: [A₀ A₁ C₀ C₁ | B₀ B₁ D₀ D₁]. Tail blocks (nb % 4) keep a plain r-major layout.
/// Private fields, built only by `from_q8_0`/`quantize`: `qs()` and the AVX2 kernels trust `n` (0.0.1
/// made `n`, `d` and `sum` public, so safe code could send the kernels past their buffers).
pub struct Q8Act2 {
    n: usize,
    qs: Vec<Line>,
    d: Vec<f32>,
    sum: Vec<i32>,
}

/// 64-byte-aligned storage, so no 32-byte activation load straddles a cache line (a `Vec<i8>` is only
/// 16-aligned). Not measured to matter on the dev box, whose run-to-run swings are the boost clock
/// (both kernels move together); kept because it cannot cost.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([i8; 64]);

/// Where element e (0..64) of block i sits in `Q8Act2::qs`.
fn qpos(nb: usize, i: usize, e: usize) -> usize {
    let (r, b) = (e % 4, e / 4);
    if i < nb / 4 * 4 {
        let (g, p, h) = (i / 4, (i % 4) / 2, i % 2);
        g * 256 + p * 128 + r * 32 + h * 16 + b
    } else {
        i * 64 + r * 16 + b
    }
}

/// Where q8 block k (0,1) of Q2_0 block i sits in `Q8Act2::{d,sum}`.
fn dpos(nb: usize, i: usize, k: usize) -> usize {
    if i < nb / 4 * 4 {
        (i / 4) * 8 + [0, 4, 2, 6][i % 4] + k
    } else {
        2 * i + k
    }
}

impl Q8Act2 {
    pub fn from_q8_0(n: usize, q8: &[u8]) -> Self {
        assert_eq!(n % QK2_0, 0);
        let nb = n / QK2_0;
        let (mut qs, mut d, mut sum) = (vec![Line([0; 64]); nb], vec![0f32; 2 * nb], vec![0i32; 2 * nb]);
        for i in 0..nb {
            for k in 0..2 {
                let yb = &q8[(2 * i + k) * Q8_0_BYTES..(2 * i + k + 1) * Q8_0_BYTES];
                d[dpos(nb, i, k)] = f16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
                sum[dpos(nb, i, k)] = yb[2..].iter().map(|&v| v as i8 as i32).sum();
                for e in 0..32 {
                    let j = qpos(nb, i, 32 * k + e);
                    qs[j / 64].0[j % 64] = yb[2 + e] as i8;
                }
            }
        }
        Q8Act2 { n, qs, d, sum }
    }
    /// Elements (a multiple of 64).
    pub fn n(&self) -> usize {
        self.n
    }
    /// The prepared quants (n of them, 64-byte aligned).
    pub fn qs(&self) -> &[i8] {
        // SAFETY: `qs` holds n / 64 Lines of 64 i8 each, and `n` is private and fixed at construction
        unsafe { std::slice::from_raw_parts(self.qs.as_ptr() as *const i8, self.n) }
    }
    /// Quantize f32 activations exactly as ggml's AVX2 `quantize_row_q8_0`.
    pub fn quantize(x: &[f32]) -> Self {
        let mut q = vec![0u8; x.len() / QK8_0 * Q8_0_BYTES];
        quantize_row_q8_0(x, &mut q);
        Self::from_q8_0(x.len(), &q)
    }
}

/// Q2_0 row · prepared activation; bit-identical to `vec_dot_ref` on the q8_0 bytes it came from.
pub fn vec_dot_act(x: &[u8], a: &Q8Act2) -> f32 {
    assert!(x.len() >= a.n / QK2_0 * Q2_0_BYTES);
    #[cfg(target_arch = "x86_64")]
    if crate::q1_0::has_avx2() {
        return unsafe { vec_dot_act_avx2(x, a) };
    }
    vec_dot_act_scalar(x, a)
}

/// Same float ops as ggml; integer sums from the prepared activation (the portable path).
pub fn vec_dot_act_scalar(x: &[u8], a: &Q8Act2) -> f32 {
    let nb = a.n / QK2_0;
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let xb = &x[i * Q2_0_BYTES..];
        let mut s = [0i32; 2];
        for e in 0..64 {
            s[e / 32] += ((xb[2 + e / 4] >> (2 * (e % 4))) & 3) as i32 * a.qs()[qpos(nb, i, e)] as i32;
        }
        let (j0, j1) = (dpos(nb, i, 0), dpos(nb, i, 1));
        let sumi = a.d[j1].mul_add((s[1] - a.sum[j1]) as f32, a.d[j0].mul_add((s[0] - a.sum[j0]) as f32, 0.0));
        sumf = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]])).mul_add(sumi, sumf);
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// The four code vectors of blocks A (low half) and B (high half): `(v >> 2r) & 3` — a 16-bit
    /// shift is fine, the `& 3` drops whatever crossed in from the neighbouring byte.
    #[inline(always)]
    pub unsafe fn codes2(xa: *const u8) -> [__m256i; 4] {
        let v = _mm256_loadu2_m128i(xa.add(Q2_0_BYTES + 2) as *const __m128i, xa.add(2) as *const __m128i);
        let m3 = _mm256_set1_epi8(3);
        [
            _mm256_and_si256(v, m3),
            _mm256_and_si256(_mm256_srli_epi16(v, 2), m3),
            _mm256_and_si256(_mm256_srli_epi16(v, 4), m3),
            _mm256_and_si256(_mm256_srli_epi16(v, 6), m3),
        ]
    }

    /// Σc·q for a block pair against its 128 prepared bytes → i32 [A₀ A₀ A₁ A₁ | B₀ B₀ B₁ B₁] partials.
    /// i16 lanes stay within ±4·768: no saturation (c ≤ 3, q ≥ −128).
    #[inline(always)]
    pub unsafe fn dot2(c: &[__m256i; 4], q: *const i8) -> __m256i {
        let m = |r: usize| _mm256_maddubs_epi16(c[r], _mm256_loadu_si256(q.add(32 * r) as *const __m256i));
        let s16 = _mm256_add_epi16(_mm256_add_epi16(m(0), m(1)), _mm256_add_epi16(m(2), m(3)));
        _mm256_madd_epi16(s16, _mm256_set1_epi16(1))
    }

    /// ggml's float ops for a quad: `hadd` → [A₀ A₁ C₀ C₁ | B₀ B₁ D₀ D₁]; sumi in odd lanes; then the
    /// serial `sumf = fma(d0, sumi, sumf)` in block order A, B, C, D.
    #[inline(always)]
    pub unsafe fn finish4(ab: __m256i, cd: __m256i, a: &Q8Act2, g: usize, xb: *const u8, mut sumf: __m128) -> __m128 {
        let s = _mm256_sub_epi32(_mm256_hadd_epi32(ab, cd), _mm256_loadu_si256(a.sum.as_ptr().add(8 * g) as *const __m256i));
        let (s, d1) = (_mm256_cvtepi32_ps(s), _mm256_loadu_ps(a.d.as_ptr().add(8 * g)));
        let u = _mm256_fmadd_ps(d1, s, _mm256_moveldup_ps(_mm256_fmadd_ps(d1, s, _mm256_setzero_ps())));
        let (lo, hi) = (_mm256_castps256_ps128(u), _mm256_extractf128_ps(u, 1));
        let v = [_mm_movehdup_ps(lo), _mm_movehdup_ps(hi), _mm_shuffle_ps(lo, lo, 3), _mm_shuffle_ps(hi, hi, 3)];
        for (b, vb) in v.into_iter().enumerate() {
            let d0 = _mm_cvtph_ps(_mm_cvtsi32_si128((xb.add(b * Q2_0_BYTES) as *const u16).read_unaligned() as i32));
            sumf = _mm_fmadd_ss(d0, vb, sumf);
        }
        sumf
    }

    /// One tail block (nb % 4), 128-bit.
    #[inline(always)]
    pub unsafe fn block1(xb: *const u8, a: &Q8Act2, i: usize, sumf: __m128) -> __m128 {
        let v = _mm_loadu_si128(xb.add(2) as *const __m128i);
        let (m3, q) = (_mm_set1_epi8(3), a.qs().as_ptr().add(i * 64));
        let m = |c: __m128i, r: usize| _mm_maddubs_epi16(_mm_and_si128(c, m3), _mm_loadu_si128(q.add(16 * r) as *const __m128i));
        let s16 = _mm_add_epi16(_mm_add_epi16(m(v, 0), m(_mm_srli_epi16(v, 2), 1)), _mm_add_epi16(m(_mm_srli_epi16(v, 4), 2), m(_mm_srli_epi16(v, 6), 3)));
        let h = _mm_madd_epi16(s16, _mm_set1_epi16(1));
        let h = _mm_hadd_epi32(h, h);
        let (s0, s1) = ((_mm_cvtsi128_si32(h) - a.sum[2 * i]) as f32, (_mm_extract_epi32(h, 1) - a.sum[2 * i + 1]) as f32);
        let sumi = a.d[2 * i + 1].mul_add(s1, a.d[2 * i].mul_add(s0, 0.0));
        let d0 = _mm_cvtph_ps(_mm_cvtsi32_si128((xb as *const u16).read_unaligned() as i32));
        _mm_fmadd_ss(d0, _mm_set_ss(sumi), sumf)
    }
}

/// # Safety
/// AVX2+FMA+F16C present (`q1_0::has_avx2`); `x` holds `a.n / 64` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_act_avx2(x: &[u8], a: &Q8Act2) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.n / QK2_0;
    let (xp, qp) = (x.as_ptr(), a.qs().as_ptr());
    let mut sumf = _mm_setzero_ps();
    for g in 0..nb / 4 {
        let xb = xp.add(4 * g * Q2_0_BYTES);
        let ab = avx2::dot2(&avx2::codes2(xb), qp.add(g * 256));
        let cd = avx2::dot2(&avx2::codes2(xb.add(2 * Q2_0_BYTES)), qp.add(g * 256 + 128));
        sumf = avx2::finish4(ab, cd, a, g, xb, sumf);
    }
    for i in nb / 4 * 4..nb {
        sumf = avx2::block1(xp.add(i * Q2_0_BYTES), a, i, sumf);
    }
    _mm_cvtss_f32(sumf)
}

/// out[r] = row r of `w` · `a` for `rows` rows (the decode matvec); bits of per-row `vec_dot_ref`.
pub fn mat_vec(w: &[u8], rows: usize, a: &Q8Act2, out: &mut [f32]) {
    let rb = a.n / QK2_0 * Q2_0_BYTES;
    assert!(w.len() >= rows * rb && out.len() >= rows);
    for (r, o) in out[..rows].iter_mut().enumerate() {
        *o = vec_dot_act(&w[r * rb..], a);
    }
}

/// Prefill tile: one Q2_0 row against four prepared columns; the code expansion is done once per
/// weight block and reused by the four columns. Every output has the bits of its own `vec_dot_ref`.
///
/// # Safety
/// As `vec_dot_act_avx2`, for every column (all four with the same `n`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_1x4_avx2(x: &[u8], a: [&Q8Act2; 4]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let nb = a[0].n / QK2_0;
    let xp = x.as_ptr();
    let mut sumf = [_mm_setzero_ps(); 4];
    for g in 0..nb / 4 {
        let xb = xp.add(4 * g * Q2_0_BYTES);
        let (c0, c1) = (avx2::codes2(xb), avx2::codes2(xb.add(2 * Q2_0_BYTES)));
        for col in 0..4 {
            let qp = a[col].qs().as_ptr().add(g * 256);
            sumf[col] = avx2::finish4(avx2::dot2(&c0, qp), avx2::dot2(&c1, qp.add(128)), a[col], g, xb, sumf[col]);
        }
    }
    for i in nb / 4 * 4..nb {
        for col in 0..4 {
            sumf[col] = avx2::block1(xp.add(i * Q2_0_BYTES), a[col], i, sumf[col]);
        }
    }
    sumf.map(|s| _mm_cvtss_f32(s))
}

/// out[c·rows + r] = row r of `w` · column c — the Q2_0 matmul for a batch of columns (prefill).
/// 1×4 tiles where the CPU has AVX2; every element has the bits of ggml's per-pair vec_dot.
pub fn mat_mul(w: &[u8], rows: usize, cols: &[Q8Act2], out: &mut [f32]) {
    if mat_mul_check(w, rows, cols, out) {
        // SAFETY: checked above; one caller owns all of `out`
        unsafe { mat_mul_rows(w, rows, 0..rows, cols, out.as_mut_ptr()) }
    }
}

fn mat_mul_check(w: &[u8], rows: usize, cols: &[Q8Act2], out: &[f32]) -> bool {
    let Some(n) = cols.first().map(|c| c.n) else { return false };
    assert!(cols.iter().all(|c| c.n == n) && w.len() >= rows * (n / QK2_0 * Q2_0_BYTES) && out.len() >= rows * cols.len());
    true
}

/// Rows `rs` of the product, written to `out[c·rows + r]`.
///
/// # Safety
/// `mat_mul_check` passed for these arguments, and no other thread writes rows `rs` of `out`.
unsafe fn mat_mul_rows(w: &[u8], rows: usize, rs: std::ops::Range<usize>, cols: &[Q8Act2], out: *mut f32) {
    let rb = cols[0].n / QK2_0 * Q2_0_BYTES;
    let mut c0 = 0;
    #[cfg(target_arch = "x86_64")]
    if crate::q1_0::has_avx2() {
        while c0 + 4 <= cols.len() {
            let t = [&cols[c0], &cols[c0 + 1], &cols[c0 + 2], &cols[c0 + 3]];
            for r in rs.clone() {
                let v = vec_dot_1x4_avx2(&w[r * rb..], t);
                for (c, v) in v.into_iter().enumerate() {
                    *out.add((c0 + c) * rows + r) = v;
                }
            }
            c0 += 4;
        }
    }
    for (c, a) in cols.iter().enumerate().skip(c0) {
        for r in rs.clone() {
            *out.add(c * rows + r) = vec_dot_act(&w[r * rb..], a);
        }
    }
}

/// `mat_vec` on every thread of `pool`; the same bits at any thread count.
pub fn mat_vec_par(pool: &crate::par::Pool, w: &[u8], rows: usize, a: &Q8Act2, out: &mut [f32]) {
    let rb = a.n / QK2_0 * Q2_0_BYTES;
    assert!(w.len() >= rows * rb && out.len() >= rows);
    pool.rows(rows, out, &|r0, o| mat_vec(&w[r0 * rb..], o.len(), a, o));
}

/// `mat_mul` on every thread of `pool` (row chunks, all columns each); the same bits at any thread count.
pub fn mat_mul_par(pool: &crate::par::Pool, w: &[u8], rows: usize, cols: &[Q8Act2], out: &mut [f32]) {
    if !mat_mul_check(w, rows, cols, out) {
        return;
    }
    let (next, base) = (std::sync::atomic::AtomicUsize::new(0), out.as_mut_ptr() as usize);
    let step = crate::par::CHUNK_ROWS;
    pool.run(&|_| loop {
        let r0 = next.fetch_add(step, std::sync::atomic::Ordering::Relaxed);
        if r0 >= rows {
            break;
        }
        // SAFETY: checked; each row range is claimed by exactly one worker
        unsafe { mat_mul_rows(w, rows, r0..(r0 + step).min(rows), cols, base as *mut f32) };
    });
}

// ---------------------------------------------------------------- tests -------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q1_0::f32_to_f16;
    use crate::q1_0::tests::Rng;

    fn random_q2(r: &mut Rng, nb: usize) -> Vec<u8> {
        let mut x = vec![0u8; nb * Q2_0_BYTES];
        for b in x.as_chunks_mut::<Q2_0_BYTES>().0.iter_mut() {
            b[..2].copy_from_slice(&f32_to_f16(r.f().abs() * 0.05 + 1e-4).to_le_bytes());
            b[2..].iter_mut().for_each(|v| *v = r.next() as u8); // all four codes, +2 included
        }
        x
    }

    fn random_q8(r: &mut Rng, nb: usize, full_range: bool) -> Vec<u8> {
        let mut y = vec![0u8; nb * 2 * Q8_0_BYTES];
        for b in y.as_chunks_mut::<Q8_0_BYTES>().0.iter_mut() {
            b[..2].copy_from_slice(&f32_to_f16(r.f().abs() * 0.1 + 1e-5).to_le_bytes());
            b[2..].iter_mut().for_each(|v| *v = if full_range { r.next() as u8 } else { ((r.next() % 255) as i32 - 127) as i8 as u8 });
        }
        y
    }

    #[test]
    fn dequantize_layout_scale_first_2bit_lsb_first() {
        let mut b = [0x55u8; 18]; // every code 01 → 0
        b[..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        b[2] = 0b11_10_01_00; // weights 0..3: codes 0,1,2,3 → −1, 0, +1, +2
        b[17] = 0b10_01_01_01; // weight 63: code 2 → +1
        let mut o = [0f32; 64];
        dequantize_row(&b, &mut o);
        assert_eq!((o[0], o[1], o[2], o[3], o[4], o[62], o[63]), (-0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.5));
    }

    #[test]
    fn ref_matches_dequantized_math() {
        let mut r = Rng(17);
        for nb in [1usize, 3, 64] {
            let (x, y) = (random_q2(&mut r, nb), random_q8(&mut r, nb, false));
            let mut w = vec![0f32; nb * 64];
            dequantize_row(&x, &mut w);
            let exact: f64 = (0..nb * 64)
                .map(|j| {
                    let yb = &y[(j / 32) * Q8_0_BYTES..];
                    w[j] as f64 * f16_to_f32(u16::from_le_bytes([yb[0], yb[1]])) as f64 * (yb[2 + j % 32] as i8) as f64
                })
                .sum();
            let got = vec_dot_ref(nb * 64, &x, &y) as f64;
            assert!((got - exact).abs() < 1e-5 * exact.abs().max(1.0), "nb={nb} exact={exact} ref={got}");
        }
    }

    /// The fast paths (AVX2 and the portable reordered scalar) against the ggml model, bit for bit,
    /// with block counts that exercise the 4-block body and the tail, and q = −128.
    #[test]
    fn act_paths_bit_exact_with_ref_randomized() {
        let mut r = Rng(0x7e27_a7e2);
        for round in 0..3000 {
            let nb = 1 + (r.next() % 70) as usize;
            let (x, y) = (random_q2(&mut r, nb), random_q8(&mut r, nb, round % 2 == 0));
            let a = Q8Act2::from_q8_0(nb * 64, &y);
            let want = vec_dot_ref(nb * 64, &x, &y).to_bits();
            assert_eq!(vec_dot_act_scalar(&x, &a).to_bits(), want, "scalar round {round} nb {nb}");
            assert_eq!(vec_dot_act(&x, &a).to_bits(), want, "fast round {round} nb {nb}");
        }
        for _ in 0..300 {
            let nb = 1 + (r.next() % 40) as usize;
            let x = random_q2(&mut r, nb);
            let act: Vec<f32> = (0..nb * 64).map(|_| r.act()).collect();
            let mut y = vec![0u8; nb * 2 * Q8_0_BYTES];
            quantize_row_q8_0(&act, &mut y);
            assert_eq!(vec_dot_act(&x, &Q8Act2::quantize(&act)).to_bits(), vec_dot_ref(nb * 64, &x, &y).to_bits());
        }
    }

    /// Against ggml itself: `testing/ggml_oracle.py` on the real Ternary-Bonsai-8B file records what the
    /// b11192 exported symbols return; this re-derives every value in Rust (weights via mmap).
    #[test]
    #[ignore = "needs .models/Ternary-Bonsai-8B-Q2_0_g64.gguf + .models/oracle-ternary (testing/ggml_oracle.py); --release"]
    fn oracle_ggml_b11192_real_ternary_bonsai_8b() {
        use crate::gguf::{guard_file, Engine, Mmap};
        use crate::sha256::{hex, Sha256};
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let model = dir.join("Ternary-Bonsai-8B-Q2_0_g64.gguf");
        let rep = guard_file(&model, Engine::Mainline).unwrap();
        let h = rep.header.as_ref().unwrap();
        let mm = Mmap::open(&model).unwrap();
        let tensor = |name: &str| mm.tensor(h, h.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{name}"))).unwrap();

        let tsv = std::fs::read_to_string(dir.join("oracle-ternary/dequant.tsv")).unwrap();
        let (mut tensors, mut elems) = (0, 0u64);
        let (mut buf, mut bytes) = (vec![0f32; 1 << 20], vec![0u8; 4 << 20]);
        for line in tsv.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            let mut sh = Sha256::default();
            for c in tensor(f[0]).chunks(Q2_0_BYTES * (1 << 14)) {
                let n = c.len() / Q2_0_BYTES * QK2_0;
                dequantize_row(c, &mut buf[..n]);
                bytes.as_chunks_mut::<4>().0.iter_mut().zip(&buf[..n]).for_each(|(b, v)| b.copy_from_slice(&v.to_le_bytes()));
                sh.update(&bytes[..4 * n]);
            }
            assert_eq!(hex(&sh.finish()), f[2], "dequantize {}", f[0]);
            tensors += 1;
            elems += f[1].parse::<u64>().unwrap();
        }

        let vb = std::fs::read(dir.join("oracle-ternary/vecdot.bin")).unwrap();
        let u32at = |o: usize| u32::from_le_bytes(vb[o..o + 4].try_into().unwrap()) as usize;
        let f32at = |o: usize| f32::from_le_bytes(vb[o..o + 4].try_into().unwrap());
        let (mut o, mut cases, mut nofma_same) = (0usize, 0, 0);
        while o < vb.len() {
            let nl = u32at(o);
            let name = std::str::from_utf8(&vb[o + 4..o + 4 + nl]).unwrap().to_string();
            let (row, n) = (u32at(o + 4 + nl), u32at(o + 8 + nl));
            o += 12 + nl;
            let x: Vec<f32> = (0..n).map(|i| f32at(o + 4 * i)).collect();
            o += 4 * n;
            let q8_ggml = &vb[o..o + n / 32 * Q8_0_BYTES];
            o += n / 32 * Q8_0_BYTES;
            let (s_avx, s_x64) = (f32at(o), f32at(o + 4));
            o += 8;
            let rb = n / QK2_0 * Q2_0_BYTES;
            let w = &tensor(&name)[row * rb..(row + 1) * rb];
            let mut q8 = vec![0u8; n / 32 * Q8_0_BYTES];
            quantize_row_q8_0(&x, &mut q8);
            assert!(q8 == q8_ggml, "quantize_row_q8_0 differs: {name} row {row}");
            let (r, v) = (vec_dot_ref(n, w, &q8), vec_dot_act(w, &Q8Act2::quantize(&x)));
            assert_eq!((r.to_bits(), v.to_bits()), (s_avx.to_bits(), s_avx.to_bits()), "{name} row {row}: ggml {s_avx:e} ref {r:e} fast {v:e}");
            assert_eq!(vec_dot_act_scalar(w, &Q8Act2::from_q8_0(n, &q8)).to_bits(), s_avx.to_bits(), "scalar {name} row {row}");
            nofma_same += (vec_dot_ref_nofma(n, w, &q8).to_bits() == s_x64.to_bits()) as usize;
            cases += 1;
        }
        eprintln!("oracle: {tensors} Q2_0 tensors / {elems} weights dequantized bit-exact; {cases} q8_0 rows byte-exact; \
                   {cases} vec_dot_ref, vec_dot_act (AVX2) and scalar bit-exact vs ggml haswell; no-FMA model == ggml x64 in {nofma_same}/{cases}");
        assert_eq!((tensors, cases, nofma_same), (254, 762, 762));
    }

    type VecDot = unsafe extern "C" fn(i32, *mut f32, usize, *const u8, usize, *const u8, usize, i32);

    /// ggml b11192's own `ggml_vec_dot_q2_0_q8_0` from the shipped haswell library (dlopen, no crate).
    fn ggml_q2_0_vec_dot() -> VecDot {
        use std::ffi::{c_char, c_int, c_void, CString};
        extern "C" {
            fn dlopen(f: *const c_char, flag: c_int) -> *mut c_void;
            fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void;
        }
        let dir = std::env::var("BANKML_GGML_LIB").expect("BANKML_GGML_LIB (llama.cpp b11192 ubuntu-x64 release dir)");
        let (base, cpu) = (CString::new(format!("{dir}/libggml-base.so")).unwrap(), CString::new(format!("{dir}/libggml-cpu-haswell.so")).unwrap());
        unsafe {
            assert!(!dlopen(base.as_ptr(), 0x102).is_null(), "dlopen base"); // RTLD_NOW|RTLD_GLOBAL
            let h = dlopen(cpu.as_ptr(), 0x102);
            assert!(!h.is_null(), "dlopen cpu");
            let init = dlsym(h, c"ggml_cpu_init".as_ptr()); // fills the f16→f32 table the kernel reads
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(init)();
            let f = dlsym(h, c"ggml_vec_dot_q2_0_q8_0".as_ptr());
            assert!(!f.is_null());
            std::mem::transmute::<*mut c_void, VecDot>(f)
        }
    }

    fn stat(v: &mut [f64]) -> (f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0], v[v.len() / 2])
    }

    /// All rows of a Q2_0 matrix against one q8_0 column, through ggml's kernel (`g = Some`) or
    /// bankml's `mat_vec_par`, both on the same pool and row scheduler (`par::Pool::rows`, 16-row
    /// chunks from an atomic counter, as ggml's mul_mat hands them out), so the A/B is kernel against kernel.
    fn gemv(g: Option<VecDot>, w: &[u8], rows: usize, q8: &[u8], a: &Q8Act2, out: &mut [f32], pool: &crate::par::Pool) {
        let n = a.n;
        let rb = n / QK2_0 * Q2_0_BYTES;
        match g {
            Some(f) => pool.rows(rows, out, &|r0, o| {
                o.iter_mut().enumerate().for_each(|(i, s)| unsafe { f(n as i32, s, 0, w[(r0 + i) * rb..].as_ptr(), 0, q8.as_ptr(), 0, 1) })
            }),
            None => mat_vec_par(pool, w, rows, a, out),
        }
    }

    /// Activation-like f32 → (ggml's q8_0 bytes, bankml's prepared activation).
    fn column(r: &mut Rng, n: usize) -> (Vec<u8>, Q8Act2) {
        let x: Vec<f32> = (0..n).map(|_| r.act()).collect();
        let mut q8 = vec![0u8; n / 32 * Q8_0_BYTES];
        quantize_row_q8_0(&x, &mut q8);
        let a = Q8Act2::from_q8_0(n, &q8);
        (q8, a)
    }

    /// In-process A/B against ggml b11192's own Q2_0 kernel on the **real** Ternary-Bonsai-8B weights
    /// (layer 0, every projection shape Qwen3-8B has, plus `output`), 1 thread, alternating on the same
    /// bits, min/median. Decode GEMV, then prefill (32 columns, ggml = per-pair vec_dot as its mul_mat does).
    /// `BANKML_GGML_LIB=… cargo test --release -- --ignored ab_vs_ggml_q2_0 --nocapture --test-threads=1`
    #[test]
    #[ignore = "needs BANKML_GGML_LIB + .models/Ternary-Bonsai-8B-Q2_0_g64.gguf"]
    fn ab_vs_ggml_q2_0() {
        use crate::gguf::{guard_file, Engine, Mmap};
        use std::time::Instant;
        let ggml = ggml_q2_0_vec_dot();
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/Ternary-Bonsai-8B-Q2_0_g64.gguf");
        let rep = guard_file(&model, Engine::Mainline).unwrap();
        let h = rep.header.as_ref().unwrap();
        let mm = Mmap::open(&model).unwrap();
        let mut r = Rng(20260926);
        eprintln!("decode GEMV, 1 thread, ns/block (64 weights) min/median, ms/matmul = median:");
        for name in ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "output.weight"] {
            let t = h.tensors.iter().find(|t| t.name == name).unwrap();
            let (n, rows) = (t.dims[0] as usize, t.dims[1] as usize);
            let w = mm.tensor(h, t).unwrap();
            let (q8, a) = column(&mut r, n);
            let (mut og, mut ob) = (vec![0f32; rows], vec![0f32; rows]);
            let blocks = (rows * n / QK2_0) as f64;
            let reps = if rows > 100_000 { 5 } else { 15 };
            let (mut tg, mut tb) = (vec![], vec![]);
            let one = crate::par::Pool::new(1);
            gemv(Some(ggml), w, rows, &q8, &a, &mut og, &one); // page in
            for _ in 0..reps {
                let t0 = Instant::now();
                gemv(Some(ggml), w, rows, &q8, &a, &mut og, &one);
                tg.push(t0.elapsed().as_nanos() as f64 / blocks);
                let t0 = Instant::now();
                gemv(None, w, rows, &q8, &a, &mut ob, &one);
                tb.push(t0.elapsed().as_nanos() as f64 / blocks);
            }
            assert!(og.iter().zip(&ob).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: bits");
            let (g, b) = (stat(&mut tg), stat(&mut tb));
            eprintln!("  {name:26} {rows:6}x{n:<5} ggml {:6.2}/{:6.2} ({:7.1} ms) | bankml {:5.2}/{:5.2} ({:6.1} ms) | {:5.2}x",
                g.0, g.1, g.1 * blocks / 1e6, b.0, b.1, b.1 * blocks / 1e6, g.1 / b.1);
        }
        // per-token activation cost bankml adds (ggml pays quantize_row_q8_0 once per matmul too)
        for n in [4096usize, 12288] {
            let x: Vec<f32> = (0..n).map(|_| r.act()).collect();
            let mut q8 = vec![0u8; n / 32 * Q8_0_BYTES];
            let mut ts = [vec![], vec![]];
            for _ in 0..200 {
                let t0 = Instant::now();
                quantize_row_q8_0(&x, &mut q8);
                ts[0].push(t0.elapsed().as_nanos() as f64 / 1e3);
                let t0 = Instant::now();
                std::hint::black_box(Q8Act2::from_q8_0(n, &q8));
                ts[1].push(t0.elapsed().as_nanos() as f64 / 1e3);
            }
            eprintln!("activation n={n}: q8_0 quantize {:.1} µs + bankml layout {:.1} µs (median)", stat(&mut ts[0]).1, stat(&mut ts[1]).1);
        }
        // prefill: 1024 rows of ffn_gate x 32 columns
        let t = h.tensors.iter().find(|t| t.name == "blk.0.ffn_gate.weight").unwrap();
        let (n, rows, cols) = (t.dims[0] as usize, 1024usize, 32usize);
        let w = mm.tensor(h, t).unwrap();
        let rb = n / QK2_0 * Q2_0_BYTES;
        let cs: Vec<(Vec<u8>, Q8Act2)> = (0..cols).map(|_| column(&mut r, n)).collect();
        let acts: Vec<Q8Act2> = cs.iter().map(|c| Q8Act2::from_q8_0(n, &c.0)).collect();
        let (mut og, mut ob) = (vec![0f32; rows * cols], vec![0f32; rows * cols]);
        let pb = (rows * cols * n / QK2_0) as f64;
        let (mut tg, mut tb, mut tv) = (vec![], vec![], vec![]);
        for _ in 0..9 {
            let t0 = Instant::now();
            for c in 0..cols {
                for i in 0..rows {
                    unsafe { ggml(n as i32, &mut og[c * rows + i], 0, w[i * rb..].as_ptr(), 0, cs[c].0.as_ptr(), 0, 1) };
                }
            }
            tg.push(t0.elapsed().as_nanos() as f64 / pb);
            let t0 = Instant::now();
            mat_mul(w, rows, &acts, &mut ob);
            tb.push(t0.elapsed().as_nanos() as f64 / pb);
            let t0 = Instant::now();
            for c in 0..cols {
                mat_vec(w, rows, &acts[c], &mut ob[c * rows..]);
            }
            tv.push(t0.elapsed().as_nanos() as f64 / pb);
        }
        assert!(og.iter().zip(&ob).all(|(x, y)| x.to_bits() == y.to_bits()), "prefill bits");
        let (g, b, v) = (stat(&mut tg), stat(&mut tb), stat(&mut tv));
        eprintln!("prefill 1024x{n} x {cols} cols, ns/(block·col) min/median: ggml per-pair {:.2}/{:.2} | bankml mat_vec per column {:.2}/{:.2} ({:.2}x) | bankml 1x4 tile {:.2}/{:.2} ({:.2}x)",
            g.0, g.1, v.0, v.1, g.1 / v.1, b.0, b.1, g.1 / b.1);
    }

    /// One token's worth of every Q2_0 matmul in Ternary-Bonsai-8B (36 × q,k,v,o,gate,up,down + output),
    /// as a token touches them: each tensor once, in file order (nothing stays cached between tensors).
    /// Per thread count (`BANKML_THREADS`, default "1,3"; the laptop llama-server ran `-t 3`): one
    /// paging pass, then 3 passes of ggml and 3 of bankml, alternating; min and median pass.
    /// `BANKML_GGML_LIB=… cargo test --release -- --ignored decode_budget_q2_0 --nocapture --test-threads=1`
    #[test]
    #[ignore = "needs BANKML_GGML_LIB + .models/Ternary-Bonsai-8B-Q2_0_g64.gguf; ~1 min"]
    fn decode_budget_q2_0() {
        use crate::gguf::{guard_file, Engine, Mmap};
        use std::time::Instant;
        let ggml = ggml_q2_0_vec_dot();
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/Ternary-Bonsai-8B-Q2_0_g64.gguf");
        let rep = guard_file(&model, Engine::Mainline).unwrap();
        let h = rep.header.as_ref().unwrap();
        let mm = Mmap::open(&model).unwrap();
        let mut r = Rng(36);
        let mats: Vec<_> = h.tensors.iter().filter(|t| t.ty == 42 && t.name != "token_embd.weight").collect();
        let mut acts = std::collections::HashMap::new();
        for t in &mats {
            acts.entry(t.dims[0] as usize).or_insert_with(|| column(&mut r, t.dims[0] as usize));
        }
        let weights: u64 = mats.iter().map(|t| t.dims[0] * t.dims[1]).sum();
        eprintln!("decode budget: {} Q2_0 matmuls, {weights} weights ({:.2} GB of blocks) per token", mats.len(), weights as f64 / 64.0 * 18.0 / 1e9);
        let threads = std::env::var("BANKML_THREADS").unwrap_or("1,3".into());
        for th in threads.split(',').map(|v| v.parse::<usize>().unwrap()) {
            let pool = crate::par::Pool::new(th);
            let mut kinds: std::collections::BTreeMap<String, [f64; 2]> = Default::default();
            let pass = |g: bool, kinds: &mut std::collections::BTreeMap<String, [f64; 2]>| {
                let t0 = Instant::now();
                for t in &mats {
                    let (n, rows) = (t.dims[0] as usize, t.dims[1] as usize);
                    let (q8, a) = &acts[&n];
                    let mut o = vec![0f32; rows];
                    let t1 = Instant::now();
                    gemv(if g { Some(ggml) } else { None }, mm.tensor(h, t).unwrap(), rows, q8, a, &mut o, &pool);
                    let kind = if t.name.starts_with("output") { "output" } else { t.name.rsplit('.').nth(1).unwrap() };
                    kinds.entry(kind.into()).or_default()[!g as usize] += t1.elapsed().as_secs_f64();
                }
                t0.elapsed().as_secs_f64()
            };
            pass(true, &mut Default::default()); // page in
            let (mut tg, mut tb) = (vec![], vec![]);
            for _ in 0..3 {
                tg.push(pass(true, &mut kinds));
                tb.push(pass(false, &mut kinds));
            }
            let (g, b) = (stat(&mut tg), stat(&mut tb));
            eprintln!("  {th} thread(s): ggml b11192 {:.3}/{:.3} s/token (min/median) | bankml {:.3}/{:.3} s/token | {:.2}x | matmul-only ceiling ggml {:.2}, bankml {:.2} tok/s",
                g.0, g.1, b.0, b.1, g.0 / b.0, 1.0 / g.0, 1.0 / b.0);
            for (k, v) in &kinds {
                eprintln!("      {k:12} ggml {:6.3} s  bankml {:6.3} s  (mean of 3 passes)", v[0] / 3.0, v[1] / 3.0);
            }
        }
        // bits: every tensor's first 64 rows, both kernels, bankml threaded (the timing passes do not compare)
        let (one, par) = (crate::par::Pool::new(1), crate::par::Pool::new(3));
        for t in &mats {
            let (n, rows) = (t.dims[0] as usize, 64usize);
            let (q8, a) = &acts[&n];
            let (mut og, mut ob) = (vec![0f32; rows], vec![0f32; rows]);
            gemv(Some(ggml), mm.tensor(h, t).unwrap(), rows, q8, a, &mut og, &one);
            gemv(None, mm.tensor(h, t).unwrap(), rows, q8, a, &mut ob, &par);
            assert!(og.iter().zip(&ob).all(|(x, y)| x.to_bits() == y.to_bits()), "{}: bits", t.name);
        }
    }

    #[test]
    fn par_bit_exact_with_single_thread() {
        let mut r = Rng(777);
        let (rows, nb, cols) = (203usize, 9usize, 6usize);
        let w = random_q2(&mut r, rows * nb);
        let acts: Vec<Q8Act2> = (0..cols).map(|_| Q8Act2::from_q8_0(nb * 64, &random_q8(&mut r, nb, true))).collect();
        let (mut mv, mut mm) = (vec![0f32; rows], vec![0f32; rows * cols]);
        mat_vec(&w, rows, &acts[0], &mut mv);
        mat_mul(&w, rows, &acts, &mut mm);
        for th in [1, 2, 3, 4] {
            let pool = crate::par::Pool::new(th);
            let (mut pv, mut pm) = (vec![0f32; rows], vec![0f32; rows * cols]);
            mat_vec_par(&pool, &w, rows, &acts[0], &mut pv);
            mat_mul_par(&pool, &w, rows, &acts, &mut pm);
            assert!(pv.iter().zip(&mv).all(|(a, b)| a.to_bits() == b.to_bits()), "mat_vec_par {th}");
            assert!(pm.iter().zip(&mm).all(|(a, b)| a.to_bits() == b.to_bits()), "mat_mul_par {th}");
        }
    }

    #[test]
    fn mat_vec_and_mat_mul_bit_exact_with_ref() {
        let mut r = Rng(4242);
        for (rows, nb, cols) in [(1usize, 1usize, 1usize), (3, 5, 4), (7, 64, 9), (5, 7, 13)] {
            let w = random_q2(&mut r, rows * nb);
            let ys: Vec<Vec<u8>> = (0..cols).map(|_| random_q8(&mut r, nb, true)).collect();
            let acts: Vec<Q8Act2> = ys.iter().map(|y| Q8Act2::from_q8_0(nb * 64, y)).collect();
            let (mut mv, mut mm) = (vec![0f32; rows], vec![0f32; rows * cols]);
            mat_vec(&w, rows, &acts[0], &mut mv);
            mat_mul(&w, rows, &acts, &mut mm);
            for c in 0..cols {
                for i in 0..rows {
                    let want = vec_dot_ref(nb * 64, &w[i * nb * Q2_0_BYTES..], &ys[c]).to_bits();
                    assert_eq!(mm[c * rows + i].to_bits(), want, "mat_mul rows {rows} nb {nb} col {c} row {i}");
                    if c == 0 {
                        assert_eq!(mv[i].to_bits(), want, "mat_vec rows {rows} nb {nb} row {i}");
                    }
                }
            }
        }
    }
}
