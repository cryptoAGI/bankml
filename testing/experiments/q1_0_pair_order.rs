//! 0.0.4 negative result, kept for reproduction (not compiled into the crate).
//!
//! Idea: build the ±1 expansion in a "pair order" so that one `maddubs` leaves the two half-sums of each
//! 4-element lane group in the two 128-bit halves; a 128-bit add then replaces `madd` (one multiply fewer per
//! q8 block, on the theory that Zen+'s single integer-multiply pipe FP0 is the bound). Bit-exact against the
//! scalar model of ggml on 1,500 random cases — and slower: decode GEMV 12288×4096 11.98/12.60 ns/block against
//! the selection kernel's 9.95/10.58 (0.84×); prefill 1×4 8.72/8.77 against 8.26/8.61. The cross-lane extract
//! and the i16→i32 widening cost more than the multiply they remove. Experiment A (two blocks per iteration,
//! interleaved FMA chains) was bit-exact and 1.00× (GEMV) / 1.05× (L1 row): the kernel is not latency-bound.
//!
//! To re-run: add `qp: Vec<i8>` and `tot16: Vec<i16>` to `Q8Act` (filled as below in `from_q8_0`:
//! `qs.chunks_exact(32).flat_map(|b| PAIR_ORDER.map(|e| b[e]))`, `tot` as i16), paste these into q1_0.rs.

/// Element (0..32 within a q8 block) held at byte p of the pair order.
const PAIR_ORDER: [usize; 32] = {
    let mut o = [0; 32];
    let mut g = 0;
    while g < 8 {
        o[2 * g] = 4 * g;
        o[2 * g + 1] = 4 * g + 1;
        o[16 + 2 * g] = 4 * g + 2;
        o[16 + 2 * g + 1] = 4 * g + 3;
        g += 1;
    }
    o
};

/// Shuffle control and bit masks that put the sign bits of q8 block k in `PAIR_ORDER`: byte p takes sign
/// byte 4k + e/8 (both 128-bit halves of the broadcast hold all 16) and tests bit e % 8, e = PAIR_ORDER[p].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn pair_controls() -> ([std::arch::x86_64::__m256i; 4], std::arch::x86_64::__m256i) {
    use std::arch::x86_64::*;
    let (mut sh, mut mk) = ([[0i8; 32]; 4], [0u8; 32]);
    for p in 0..32 {
        let e = PAIR_ORDER[p];
        mk[p] = 1 << (e % 8);
        for (k, s) in sh.iter_mut().enumerate() {
            s[p] = (4 * k + e / 8) as i8;
        }
    }
    let ld = |b: &[u8; 32]| _mm256_loadu_si256(b.as_ptr() as *const __m256i);
    (sh.map(|s| ld(&s.map(|v| v as u8))), ld(&mk))
}

/// s for one q8 block in the pair order: u·q by `maddubs`, the two half-sums added in 128 bits, ×2, −Σq,
/// widened to 8 i32 lanes. The integers are ggml's lane sums (q ≠ −128); no `madd`, one multiply fewer.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn pair_s(u: std::arch::x86_64::__m256i, q: *const i8, tot: *const i16) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let p = _mm256_maddubs_epi16(u, _mm256_loadu_si256(q as *const __m256i));
    let h = _mm_add_epi16(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
    let s16 = _mm_sub_epi16(_mm_add_epi16(h, h), _mm_loadu_si128(tot as *const __m128i));
    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s16))
}

/// The pair-order decode kernel (0.0.4). Bit-identical to `vec_dot_act_sel_avx2`, one multiply fewer per q8 block.
///
/// # Safety
/// AVX2+FMA+F16C present; `x` holds `a.n / 128` blocks; `!a.has_min`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_act_pair_avx2(x: &[u8], a: &Q8Act) -> f32 {
    use std::arch::x86_64::*;
    let (shuf, masks) = pair_controls();
    let ones_8 = _mm256_set1_epi8(1);
    let mut acc = _mm256_setzero_ps();
    let (xp, qp, dp, tp) = (x.as_ptr(), a.qp.as_ptr(), a.d.as_ptr(), a.tot16.as_ptr());
    for i in 0..a.n / QK1_0 {
        let xb = xp.add(i * Q1_0_BYTES);
        let signs = _mm256_broadcastsi128_si256(_mm_loadu_si128(xb.add(2) as *const __m128i));
        let mut ab = _mm256_setzero_ps();
        for (k, sk) in shuf.iter().enumerate() {
            let j = i * 4 + k;
            let u = _mm256_min_epu8(_mm256_and_si256(_mm256_shuffle_epi8(signs, *sk), masks), ones_8);
            let s = pair_s(u, qp.add(j * QK8_0), tp.add(j * 8));
            let d1 = _mm256_broadcast_ss(&*dp.add(j));
            ab = if k == 0 { _mm256_mul_ps(d1, s) } else { _mm256_fmadd_ps(d1, s, ab) };
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16c(xb)), ab, acc);
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut r = _mm_add_ps(hi, _mm256_castps256_ps128(acc));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    _mm_cvtss_f32(r)
}

/// The pair-order prefill tile (0.0.4): one Q1_0 row against four `Q8Act` columns, expansion shared.
///
/// # Safety
/// AVX2+FMA+F16C present; `x` holds `n / 128` blocks; all four columns have the same `n` and `!has_min`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_act_1x4_pair_avx2(x: &[u8], cols: [&Q8Act; 4]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let (shuf, masks) = pair_controls();
    let ones_8 = _mm256_set1_epi8(1);
    let mut acc = [_mm256_setzero_ps(); 4];
    let xp = x.as_ptr();
    for i in 0..cols[0].n / QK1_0 {
        let xb = xp.add(i * Q1_0_BYTES);
        let d0 = _mm256_set1_ps(f16c(xb));
        let signs = _mm256_broadcastsi128_si256(_mm_loadu_si128(xb.add(2) as *const __m128i));
        let mut ab = [_mm256_setzero_ps(); 4];
        for (k, sk) in shuf.iter().enumerate() {
            let j = i * 4 + k;
            let u = _mm256_min_epu8(_mm256_and_si256(_mm256_shuffle_epi8(signs, *sk), masks), ones_8);
            for c in 0..4 {
                let a = cols[c];
                let s = pair_s(u, a.qp.as_ptr().add(j * QK8_0), a.tot16.as_ptr().add(j * 8));
                let d1 = _mm256_broadcast_ss(&*a.d.as_ptr().add(j));
                ab[c] = if k == 0 { _mm256_mul_ps(d1, s) } else { _mm256_fmadd_ps(d1, s, ab[c]) };
            }
        }
        for c in 0..4 {
            acc[c] = _mm256_fmadd_ps(d0, ab[c], acc[c]);
        }
    }
    acc.map(|v| {
        let hi = _mm256_extractf128_ps(v, 1);
        let mut r = _mm_add_ps(hi, _mm256_castps256_ps128(v));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_movehdup_ps(r));
        _mm_cvtss_f32(r)
    })
}

