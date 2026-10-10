// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exact SIMD for browseML (WebAssembly): the operations attention and the f16 dot products need, bit for bit as the
//! reference computes them.
//!
//! - [`fmadd`]: four lanes of `a·b + c` rounded once — `f32x4.relaxed_madd` in the relaxed build (opened only where
//!   it is fused: `q1_0::relaxed_madd_is_fused`), otherwise through f64 (the product of two f32 is exact there; one
//!   rounding of an exact sum is the fused result; the rare inexact halfway, subnormal or non-finite lanes are
//!   computed again by the scalar `mul_add`).
//! - [`widen4`]: four f16 to f32 exactly, as `q1_0::f16_to_f32`: the bits shifted into place and scaled by 2¹¹² (exact
//!   for normal and subnormal values), infinities and NaNs set apart.
use std::arch::wasm32::*;

/// Four lanes of `a.mul_add(b, c)`, bit for bit.
#[inline(always)]
pub fn fmadd(a: v128, b: v128, c: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    {
        f32x4_relaxed_madd(a, b, c)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        let pair = |a2: v128, b2: v128, c2: v128| {
            let (p, c) = (f64x2_mul(f64x2_promote_low_f32x4(a2), f64x2_promote_low_f32x4(b2)), f64x2_promote_low_f32x4(c2));
            let s = f64x2_add(p, c);
            let halfway = i64x2_eq(v128_and(s, i64x2_splat(0x1fff_ffff)), i64x2_splat(0x1000_0000));
            let abs = f64x2_abs(s);
            let odd = v128_not(v128_and(f64x2_ge(abs, f64x2_splat(f64::from_bits(0x3820_0000_0000_0000))), f64x2_lt(abs, f64x2_splat(f64::from_bits(0x47e0_0000_0000_0000)))));
            let suspect = v128_or(halfway, odd);
            // only a suspect lane needs the TwoSum error (an exact sum is rounded once whatever it is)
            let fix = v128_any_true(suspect) && {
                let bb = f64x2_sub(s, p);
                let err = f64x2_add(f64x2_sub(p, f64x2_sub(s, bb)), f64x2_sub(c, bb));
                v128_any_true(v128_and(suspect, f64x2_ne(err, f64x2_splat(0.0))))
            };
            (f32x4_demote_f64x2_zero(s), fix)
        };
        let hi = |v: v128| i32x4_shuffle::<2, 3, 0, 1>(v, v);
        let (lo, f1) = pair(a, b, c);
        let (h, f2) = pair(hi(a), hi(b), hi(c));
        if f1 || f2 {
            let l = |v: v128| [f32x4_extract_lane::<0>(v), f32x4_extract_lane::<1>(v), f32x4_extract_lane::<2>(v), f32x4_extract_lane::<3>(v)];
            let (a, b, c) = (l(a), l(b), l(c));
            return f32x4(a[0].mul_add(b[0], c[0]), a[1].mul_add(b[1], c[1]), a[2].mul_add(b[2], c[2]), a[3].mul_add(b[3], c[3]));
        }
        i32x4_shuffle::<0, 1, 4, 5>(lo, h)
    }
}

/// Four f16 (at `p`, little-endian) widened to f32 exactly.
///
/// # Safety
/// `p` points to 8 readable bytes.
#[inline(always)]
pub unsafe fn widen4(p: *const u16) -> v128 {
    let h = u32x4_extend_low_u16x8(unsafe { v128_load64_zero(p as *const u64) });
    let sign = i32x4_shl(v128_and(h, u32x4_splat(0x8000)), 16);
    let em = v128_and(h, u32x4_splat(0x7fff));
    let mag = f32x4_mul(i32x4_shl(em, 13), f32x4_splat(f32::from_bits(0x7780_0000))); // · 2¹¹²
    let m = v128_and(h, u32x4_splat(0x3ff));
    let nan = v128_and(v128_not(u32x4_eq(m, u32x4_splat(0))), u32x4_splat(0x0040_0000)); // quiet bit when m ≠ 0
    let special = v128_or(v128_or(u32x4_splat(0x7f80_0000), i32x4_shl(m, 13)), nan);
    v128_or(v128_bitselect(special, mag, u32x4_ge(em, u32x4_splat(0x7c00))), sign)
}

/// `chains[c] = k[c].mul_add(q, chains[c])` for every c: the tiled kernel's score chains, four cells at a time.
#[inline(always)]
pub fn fma_chain(chains: &mut [f32], k: &[f32], q: f32) {
    let n = chains.len().min(k.len());
    let (qv, mut i) = (f32x4_splat(q), 0);
    while i + 4 <= n {
        // SAFETY: i + 4 ≤ n ≤ both lengths.
        unsafe {
            let c = chains.as_mut_ptr().add(i) as *mut v128;
            v128_store(c, fmadd(v128_load(k.as_ptr().add(i) as *const v128), qv, v128_load(c)));
        }
        i += 4;
    }
    for j in i..n {
        chains[j] = k[j].mul_add(q, chains[j]);
    }
}

/// `x[d] = v[d].mul_add(p, x[d])` for every d: the tiled kernel's V update, four at a time.
#[inline(always)]
pub fn axpy(x: &mut [f32], v: &[f32], p: f32) {
    fma_chain(x, v, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q1_0::f16_to_f32;
    use crate::q1_0::tests::Rng;

    fn lanes(v: v128) -> [u32; 4] {
        [u32x4_extract_lane::<0>(v), u32x4_extract_lane::<1>(v), u32x4_extract_lane::<2>(v), u32x4_extract_lane::<3>(v)]
    }

    /// `widen4` equals `f16_to_f32` on all 65,536 f16 values (signs, subnormals, infinities and NaNs included).
    #[test]
    fn widen4_is_f16_to_f32_on_every_value() {
        for h in (0..=u16::MAX).step_by(4) {
            let src = [h, h + 1, h + 2, h + 3];
            // SAFETY: src holds four f16.
            let got = lanes(unsafe { widen4(src.as_ptr()) });
            for (g, s) in got.iter().zip(src) {
                assert_eq!(*g, f16_to_f32(s).to_bits(), "f16 {s:#06x}");
            }
        }
    }

    /// `fmadd` equals `mul_add` lane by lane: random operands over the whole exponent range, products whose sum lands
    /// on an f32 halfway point (the double-rounding case), subnormal and overflowing results, zeros, infinities, NaNs.
    #[test]
    fn fmadd_is_mul_add() {
        let mut rng = Rng(0xf3a);
        let mut cases: Vec<[f32; 3]> = Vec::new();
        let any = |r: &mut Rng| f32::from_bits(r.next() as u32);
        let near = |r: &mut Rng, e: i32| r.f() * 2f32.powi(e);
        for _ in 0..200_000 {
            cases.push([any(&mut rng), any(&mut rng), any(&mut rng)]);
            let e = (rng.next() % 60) as i32 - 30;
            cases.push([near(&mut rng, e), near(&mut rng, e), near(&mut rng, 2 * e)]);
            // halfway: a·b exact with a low half that, added to c, sits on an f32 midpoint, plus a tiny remainder
            let (a, b) = (1.0 + (rng.next() % (1 << 23)) as f32 * 2f32.powi(-23), 1.0 + (rng.next() % (1 << 12)) as f32 * 2f32.powi(-23));
            cases.push([a, b, -(a * b)]);
            cases.push([a, b, 2f32.powi(-24) - a * b]);
            // the double-rounding trap: c with an odd significand, a·b = ½ ulp(c)·(1 − 2⁻⁴⁶) exactly; the f64 sum rounds onto
            // the f32 midpoint, and ties-to-even then rounds away from the true (below-midpoint) result
            let (p, sg) = ((rng.next() % 200) as i32 - 100, if rng.next() & 1 == 0 { 1.0f32 } else { -1.0 });
            let c = sg * (1.0 + ((rng.next() % (1 << 22)) * 2 + 1) as f32 * 2f32.powi(-23)) * 2f32.powi(p);
            let (a, b) = ((1.0 + 2f32.powi(-23)) * 2f32.powi(p - 12), sg * (1.0 - 2f32.powi(-23)) * 2f32.powi(-12));
            cases.push([a, b, c]);
            let tiny = (rng.next() % 40) as i32 + 110;
            cases.push([near(&mut rng, -tiny / 2), near(&mut rng, -tiny / 2), near(&mut rng, -tiny)]);
        }
        let sp = [0.0, -0.0, 1.0, -1.0, f32::MIN_POSITIVE, f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-45, 3.0e38];
        for a in sp {
            for b in sp {
                for c in sp {
                    cases.push([a, b, c]);
                }
            }
        }
        for q in cases.chunks(4) {
            let g = |i: usize, j: usize| q.get(i).map_or(0.0, |c| c[j]);
            let v = |j: usize| f32x4(g(0, j), g(1, j), g(2, j), g(3, j));
            let got = lanes(fmadd(v(0), v(1), v(2)));
            for (i, c) in q.iter().enumerate() {
                let want = c[0].mul_add(c[1], c[2]);
                let ok = got[i] == want.to_bits() || (want.is_nan() && f32::from_bits(got[i]).is_nan());
                assert!(ok, "{:?}: {:#010x} vs {:#010x}", c, got[i], want.to_bits());
            }
        }
    }

    /// `f16::vec_dot` (this module's lanes inside) equals `f16::vec_dot_ref` on random rows of many lengths, tails included.
    #[test]
    fn f16_vec_dot_is_the_reference() {
        let mut rng = Rng(0xd07);
        for &k in &[8usize, 31, 32, 37, 64, 96, 576, 1000, 1536] {
            for _ in 0..20 {
                let x: Vec<u8> = (0..k).flat_map(|_| crate::q1_0::f32_to_f16(rng.act() * 0.2).to_le_bytes()).collect();
                let y = crate::f16::to_f16(&(0..k).map(|_| rng.act()).collect::<Vec<_>>());
                assert_eq!(crate::f16::vec_dot(&x, &y).to_bits(), crate::f16::vec_dot_ref(&x, &y).to_bits(), "k {k}");
            }
        }
    }

    /// `fma_chain` (and so `axpy`) equals the scalar chain on every length, tails included.
    #[test]
    fn fma_chain_is_the_scalar_chain() {
        let mut rng = Rng(0xc4a);
        for n in [1usize, 3, 4, 7, 64, 65, 128] {
            let k: Vec<f32> = (0..n).map(|_| rng.act()).collect();
            let mut a: Vec<f32> = (0..n).map(|_| rng.act()).collect();
            let mut b = a.clone();
            for _ in 0..50 {
                let q = rng.act();
                fma_chain(&mut a, &k, q);
                for (c, &kd) in b.iter_mut().zip(&k) {
                    *c = kd.mul_add(q, *c);
                }
            }
            assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "n {n}");
        }
    }
}
