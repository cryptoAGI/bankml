// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 1-bit `Q1_0` kernel (ggml type 41), the AVX2 q8_0 activation quantizer and exact f16 conversion,
//! bit-exact with ggml b11192's haswell build (`libggml-cpu-haswell.so`, loaded on AVX2 CPUs without
//! AVX-512).
//!
//! - Block (`ggml-common.h`): `block_q1_0 { ggml_half d; uint8_t qs[16]; }`, 18 bytes, scale first.
//! - Decode (`dequantize_row_q1_0`): weight j is bit `j % 8` of byte `j / 8`, LSB first; 1 → `+d`,
//!   0 → `−d`.
//! - Dot (`ggml_vec_dot_q1_0_q8_0`, `vec_dot_type = Q8_0`): activations are quantized to q8_0 first, so
//!   one Q1_0 block meets four q8_0 blocks. On x86 every Q1_0 matmul goes through this vec_dot (repack
//!   is NEON-only; llamafile's sgemm has no Q1_0 case).
//! - `arch/x86/quants.c`: AVX2 `quantize_row_q8_0` (id = 127/amax, round half to even) and AVX2
//!   `ggml_vec_dot_q1_0_q8_0` (8 f32 lanes; mul, then FMA, per q8 block; FMA per Q1_0 block;
//!   `hsum_float_8`).
//!
//! `vec_dot_ref` is a scalar model of that float order; the AVX2 kernels equal it bit for bit, and both
//! equal ggml's exported symbol on real tensors (`testing/ggml_oracle.py`). `vec_dot_generic` ports
//! ggml's generic C, a different float order that is not expected to match the AVX2 result.
//!
//! Details: docs/modules/q1_0.md.

pub const QK1_0: usize = 128;
pub const QK8_0: usize = 32;
pub const Q1_0_BYTES: usize = 18;
pub const Q8_0_BYTES: usize = 34;

// ---------------------------------------------------------------- f16 (IEEE binary16) -----------

/// f16 → f32, exact (every f16 is representable as an f32).
pub fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h as u32 & 0x8000) << 16, (h >> 10) as u32 & 0x1f, h as u32 & 0x3ff);
    let bits = match (e, m) {
        (0, 0) => s,
        (0, _) => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3ff) << 13)
        }
        (31, 0) => s | 0x7f80_0000,
        (31, _) => s | 0x7fc0_0000 | (m << 13),
        _ => s | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 → f16, round to nearest even, as F16C `vcvtps2ph imm=0` (ggml's `GGML_CPU_FP32_TO_FP16` on x86).
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32;
    let m = x & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 | (m >> 13) as u16 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00;
    }
    let (mant, shift) = if e16 <= 0 {
        if e16 < -10 {
            return sign; // rounds to ±0
        }
        (m | 0x80_0000, (14 - e16) as u32) // subnormal
    } else {
        (m, 13)
    };
    let half = 1u32 << (shift - 1);
    let rest = mant & ((1 << shift) - 1);
    let mut r = mant >> shift;
    if rest > half || (rest == half && r & 1 == 1) {
        r += 1;
    }
    if e16 <= 0 {
        sign | r as u16 // a carry into bit 10 lands exactly on the smallest normal
    } else {
        sign | ((((e16 as u32) << 10) + r) as u16) // a mantissa carry bumps the exponent, may reach inf
    }
}

// ---------------------------------------------------------------- blocks ------------------------

/// Dequantize n Q1_0 blocks into `out` (len n·128). Bit-exact with `dequantize_row_q1_0`.
pub fn dequantize_row(blocks: &[u8], out: &mut [f32]) {
    assert_eq!(blocks.len() % Q1_0_BYTES, 0);
    assert_eq!(out.len(), blocks.len() / Q1_0_BYTES * QK1_0);
    for (b, o) in blocks.as_chunks::<Q1_0_BYTES>().0.iter().zip(out.as_chunks_mut::<QK1_0>().0.iter_mut()) {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for (j, y) in o.iter_mut().enumerate() {
            *y = if (b[2 + j / 8] >> (j % 8)) & 1 == 1 { d } else { -d };
        }
    }
}

/// Quantize activations to q8_0 exactly as ggml's AVX2 `quantize_row_q8_0`.
///
/// id = 127/amax, round half to even, saturating packs. Not `quantize_row_q8_0_ref` (1/d, round half
/// away from zero).
pub fn quantize_row_q8_0(x: &[f32], out: &mut [u8]) {
    assert_eq!(x.len() % QK8_0, 0);
    assert_eq!(out.len(), x.len() / QK8_0 * Q8_0_BYTES);
    for (xb, yb) in x.as_chunks::<QK8_0>().0.iter().zip(out.as_chunks_mut::<Q8_0_BYTES>().0.iter_mut()) {
        let amax = xb.iter().fold(0.0f32, |m, v| if v.abs() > m { v.abs() } else { m });
        yb[..2].copy_from_slice(&f32_to_f16(amax / 127.0).to_le_bytes());
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        for (q, &v) in yb[2..].iter_mut().zip(xb) {
            let r = (v * id).round_ties_even();
            // cvtps_epi32 then packs_epi32/packs_epi16: NaN → i32::MIN → −128; saturate to i8
            *q = if r.is_nan() { -128i8 as u8 } else { r.clamp(-128.0, 127.0) as i8 as u8 };
        }
    }
}

// ---------------------------------------------------------------- dot products ------------------

fn check(n: usize, x: &[u8], y: &[u8]) {
    assert_eq!(n % QK1_0, 0);
    assert!(x.len() >= n / QK1_0 * Q1_0_BYTES && y.len() >= n / QK8_0 * Q8_0_BYTES);
}

/// Port of `ggml_vec_dot_q1_0_q8_0_generic` (scalar C, its own float order).
///
/// Both accumulations are `mul_add`: the shipped b11192 haswell binary compiles them to `vfmadd231ss`
/// (GCC's default FP contraction). The instructions, not the C text, are the reference.
pub fn vec_dot_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    check(n, x, y);
    let mut sumf = 0.0f32;
    for i in 0..n / QK1_0 {
        let xb = &x[i * Q1_0_BYTES..];
        let d0 = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let mut sumi = 0.0f32;
        for k in 0..4 {
            let yb = &y[(i * 4 + k) * Q8_0_BYTES..];
            let d1 = f16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
            let mut s = 0i32;
            for e in 0..32 {
                let q = yb[2 + e] as i8 as i32;
                s += if (xb[2 + k * 4 + e / 8] >> (e % 8)) & 1 == 1 { q } else { -q };
            }
            sumi = d1.mul_add(s as f32, sumi);
        }
        sumf = d0.mul_add(sumi, sumf);
    }
    sumf
}

/// Scalar model of ggml's AVX2 `ggml_vec_dot_q1_0_q8_0`, float op for float op.
///
/// Lane l of a q8 block holds the exact integer sum of elements 4l..4l+3, signs applied as
/// `(q ^ m) − m` on i8 (−128 wraps, as `_mm256_sub_epi8` does). The first q8 block is a multiply, the
/// other three FMA into it; the Q1_0 scale FMAs into the running accumulator; then `hsum_float_8`.
pub fn vec_dot_ref(n: usize, x: &[u8], y: &[u8]) -> f32 {
    check(n, x, y);
    let mut acc = [0.0f32; 8];
    for i in 0..n / QK1_0 {
        let xb = &x[i * Q1_0_BYTES..];
        let d0 = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let mut ab = [0.0f32; 8];
        for k in 0..4 {
            let yb = &y[(i * 4 + k) * Q8_0_BYTES..];
            let d1 = f16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
            for (l, a) in ab.iter_mut().enumerate() {
                let mut s = 0i32;
                for e in 4 * l..4 * l + 4 {
                    let q = yb[2 + e] as i8;
                    let bit = (xb[2 + k * 4 + e / 8] >> (e % 8)) & 1 == 1;
                    s += if bit { q } else { q.wrapping_neg() } as i32;
                }
                *a = if k == 0 { d1 * s as f32 } else { d1.mul_add(s as f32, *a) };
            }
        }
        for l in 0..8 {
            acc[l] = d0.mul_add(ab[l], acc[l]);
        }
    }
    hsum8(acc)
}

/// `hsum_float_8`: ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)).
fn hsum8(a: [f32; 8]) -> f32 {
    let r = [a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]];
    (r[0] + r[2]) + (r[1] + r[3])
}

/// Fastest path this CPU has; bit-identical to `vec_dot_ref` on every path.
pub fn vec_dot(n: usize, x: &[u8], y: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        check(n, x, y);
        return unsafe { vec_dot_avx2(n, x, y) };
    }
    vec_dot_ref(n, x, y)
}

#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")
}

/// One f16 scale → f32 with F16C; exact and equal to `f16_to_f32` (`f16_conversions_match_f16c_hardware`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn f16c(p: *const u8) -> f32 {
    use std::arch::x86_64::*;
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((p as *const u16).read_unaligned() as i32)))
}

/// The AVX2 kernel: the same instruction sequence as ggml b11192's `ggml_vec_dot_q1_0_q8_0`.
///
/// Sign expansion: broadcast 4 sign bytes, `pshufb` each to 8 lanes, AND with the bit mask, compare to
/// zero → a −1 mask where the weight is −1; `(q ^ m) − m` negates there. `maddubs(1, ·)` and
/// `madd(·, 1)` are the horizontal adds; there are no multiplies on the weights.
///
/// # Safety
/// Caller checks AVX2+FMA+F16C (`has_avx2`) and the slice lengths (`check`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_avx2(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let ones_8 = _mm256_set1_epi8(1);
    let ones_16 = _mm256_set1_epi16(1);
    let byte_shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    let bit_masks = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
    );
    let zero = _mm256_setzero_si256();
    let mut acc = _mm256_setzero_ps();
    let (xp, yp) = (x.as_ptr(), y.as_ptr());
    for i in 0..n / QK1_0 {
        let xb = xp.add(i * Q1_0_BYTES);
        let d0 = f16c(xb);
        let mut acc_block = _mm256_setzero_ps();
        for k in 0..4 {
            let yb = yp.add((i * 4 + k) * Q8_0_BYTES);
            let d1 = f16c(yb);
            let qy = _mm256_loadu_si256(yb.add(2) as *const __m256i);
            let bits = (xb.add(2 + 4 * k) as *const i32).read_unaligned();
            let sm = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(bits), byte_shuf), bit_masks), zero);
            let sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            let s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block = if k == 0 {
                _mm256_mul_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(s32))
            } else {
                _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(s32), acc_block)
            };
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d0), acc_block, acc);
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut r = _mm_add_ps(hi, _mm256_castps256_ps128(acc));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    _mm_cvtss_f32(r)
}

/// A q8_0 activation prepared once per token: quants contiguous (no 34-byte stride), scales already
/// f32 (exact), and per-lane sums for the selection kernel.
///
/// Fields are private: the AVX2 kernels read them through raw pointers sized by `n`, so only
/// `from_q8_0`/`quantize` may build them.
pub struct Q8Act {
    n: usize,
    qs: Vec<i8>,
    d: Vec<f32>,
    /// Σq over each 4-element lane (8 per q8 block): the activation half of Σ±q = 2·Σ₊q − Σq.
    tot: Vec<i32>,
    /// Some q == −128: ggml's i8 negation wraps there and the Σ₊ identity does not; use the wrap kernel.
    has_min: bool,
}

impl Q8Act {
    /// From ggml-layout q8_0 blocks (what `quantize_row_q8_0` writes).
    pub fn from_q8_0(n: usize, q8: &[u8]) -> Self {
        assert_eq!(n % QK1_0, 0);
        let nb = n / QK8_0;
        let (mut qs, mut d) = (Vec::with_capacity(n), Vec::with_capacity(nb));
        for b in q8[..nb * Q8_0_BYTES].as_chunks::<Q8_0_BYTES>().0.iter() {
            d.push(f16_to_f32(u16::from_le_bytes([b[0], b[1]])));
            qs.extend(b[2..].iter().map(|&v| v as i8));
        }
        let tot = qs.as_chunks::<4>().0.iter().map(|l| l.iter().map(|&v| v as i32).sum()).collect();
        let has_min = qs.contains(&i8::MIN);
        Q8Act { n, qs, d, tot, has_min }
    }
    /// Quantize f32 activations exactly as ggml's AVX2 `quantize_row_q8_0`.
    pub fn quantize(x: &[f32]) -> Self {
        let mut q = vec![0u8; x.len() / QK8_0 * Q8_0_BYTES];
        quantize_row_q8_0(x, &mut q);
        Self::from_q8_0(x.len(), &q)
    }
    /// Elements (a multiple of 128).
    pub fn n(&self) -> usize {
        self.n
    }
    /// The quants, contiguous.
    pub fn qs(&self) -> &[i8] {
        &self.qs
    }
    /// One f32 scale per q8 block.
    pub fn d(&self) -> &[f32] {
        &self.d
    }
}

/// Q1_0 row · prepared activation; bit-identical to `vec_dot` on the q8_0 bytes it came from.
pub fn vec_dot_act(x: &[u8], a: &Q8Act) -> f32 {
    assert!(x.len() >= a.n / QK1_0 * Q1_0_BYTES);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { if a.has_min { vec_dot_act_avx2(x, a) } else { vec_dot_act_sel_avx2(x, a) } };
    }
    let mut q8 = vec![0u8; a.n / QK8_0 * Q8_0_BYTES];
    for (b, (q, d)) in q8.as_chunks_mut::<Q8_0_BYTES>().0.iter_mut().zip(a.qs.as_chunks::<QK8_0>().0.iter().zip(&a.d)) {
        b[..2].copy_from_slice(&f32_to_f16(*d).to_le_bytes());
        b[2..].iter_mut().zip(q).for_each(|(o, v)| *o = *v as u8);
    }
    vec_dot_ref(a.n, x, &q8)
}

/// The wrap kernel: `vec_dot_avx2` over a prepared activation; exact for any q, including −128.
///
/// # Safety
/// AVX2+FMA+F16C present; `x` holds `a.n / 128` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_act_avx2(x: &[u8], a: &Q8Act) -> f32 {
    use std::arch::x86_64::*;
    let ones_8 = _mm256_set1_epi8(1);
    let ones_16 = _mm256_set1_epi16(1);
    let byte_shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    let bit_masks = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
    );
    let zero = _mm256_setzero_si256();
    let mut acc = _mm256_setzero_ps();
    let (xp, qp, dp) = (x.as_ptr(), a.qs.as_ptr(), a.d.as_ptr());
    for i in 0..a.n / QK1_0 {
        let xb = xp.add(i * Q1_0_BYTES);
        let d0 = f16c(xb);
        let mut acc_block = _mm256_setzero_ps();
        for k in 0..4 {
            let qy = _mm256_loadu_si256(qp.add((i * 4 + k) * QK8_0) as *const __m256i);
            let bits = (xb.add(2 + 4 * k) as *const i32).read_unaligned();
            let sm = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(bits), byte_shuf), bit_masks), zero);
            let sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            let s = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16));
            let d1 = _mm256_broadcast_ss(&*dp.add(i * 4 + k));
            acc_block = if k == 0 { _mm256_mul_ps(d1, s) } else { _mm256_fmadd_ps(d1, s, acc_block) };
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d0), acc_block, acc);
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut r = _mm_add_ps(hi, _mm256_castps256_ps128(acc));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    _mm_cvtss_f32(r)
}

/// The selection kernel: per lane s = 2·Σ_{bit=1} q − Σq, with Σq from the activation.
///
/// Signs of all four q8 blocks come from one 128-bit broadcast; `min(and(shuffle, bit), 1)` gives
/// u ∈ {0,1}; `maddubs(u, q)` then `madd(·, 2)` is 2·Σ₊q. The integers equal ggml's for q ≠ −128, and
/// the float ops (cvt, mul/FMA per q8 block, FMA per Q1_0 block, hsum) are ggml's, in ggml's order.
///
/// # Safety
/// AVX2+FMA+F16C present; `x` holds `a.n / 128` blocks; `!a.has_min`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_act_sel_avx2(x: &[u8], a: &Q8Act) -> f32 {
    use std::arch::x86_64::*;
    let ones_8 = _mm256_set1_epi8(1);
    let twos_16 = _mm256_set1_epi16(2);
    // shuffle k picks sign byte 4k+j/8 of the broadcast 16 bytes for element j (both 128-bit halves hold all 16)
    let sh = |k: i8| _mm256_setr_epi8(
        4 * k, 4 * k, 4 * k, 4 * k, 4 * k, 4 * k, 4 * k, 4 * k, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1,
        4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3,
    );
    let shuf = [sh(0), sh(1), sh(2), sh(3)];
    let bit_masks = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
    );
    let mut acc = _mm256_setzero_ps();
    let (xp, qp, dp, tp) = (x.as_ptr(), a.qs.as_ptr(), a.d.as_ptr(), a.tot.as_ptr());
    for i in 0..a.n / QK1_0 {
        let xb = xp.add(i * Q1_0_BYTES);
        let d0 = f16c(xb);
        let signs = _mm256_broadcastsi128_si256(_mm_loadu_si128(xb.add(2) as *const __m128i));
        let mut acc_block = _mm256_setzero_ps();
        for (k, shuf_k) in shuf.iter().enumerate() {
            let j = i * 4 + k;
            let qy = _mm256_loadu_si256(qp.add(j * QK8_0) as *const __m256i);
            let u = _mm256_min_epu8(_mm256_and_si256(_mm256_shuffle_epi8(signs, *shuf_k), bit_masks), ones_8);
            let pos2 = _mm256_madd_epi16(_mm256_maddubs_epi16(u, qy), twos_16);
            let s = _mm256_cvtepi32_ps(_mm256_sub_epi32(pos2, _mm256_loadu_si256(tp.add(j * 8) as *const __m256i)));
            let d1 = _mm256_broadcast_ss(&*dp.add(j));
            acc_block = if k == 0 { _mm256_mul_ps(d1, s) } else { _mm256_fmadd_ps(d1, s, acc_block) };
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d0), acc_block, acc);
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut r = _mm_add_ps(hi, _mm256_castps256_ps128(acc));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    _mm_cvtss_f32(r)
}

/// Prefill tile: one Q1_0 row against four `Q8Act` columns with the selection kernel.
///
/// The sign expansion is done once per q8 block and shared by the four columns; each column keeps
/// ggml's float order, so output c has the bits of `vec_dot_act(x, cols[c])`. The wrap case
/// (`has_min`) goes through `vec_dot_act`.
///
/// # Safety
/// AVX2+FMA+F16C present; `x` holds `n / 128` blocks; all four columns have the same `n` and `!has_min`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_act_1x4_sel_avx2(x: &[u8], cols: [&Q8Act; 4]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let ones_8 = _mm256_set1_epi8(1);
    let twos_16 = _mm256_set1_epi16(2);
    let sh = |k: i8| _mm256_setr_epi8(
        4 * k, 4 * k, 4 * k, 4 * k, 4 * k, 4 * k, 4 * k, 4 * k, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1,
        4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3,
    );
    let shuf = [sh(0), sh(1), sh(2), sh(3)];
    let bit_masks = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
    );
    let mut acc = [_mm256_setzero_ps(); 4];
    let xp = x.as_ptr();
    for i in 0..cols[0].n / QK1_0 {
        let xb = xp.add(i * Q1_0_BYTES);
        let d0 = _mm256_set1_ps(f16c(xb));
        let signs = _mm256_broadcastsi128_si256(_mm_loadu_si128(xb.add(2) as *const __m128i));
        let mut ab = [_mm256_setzero_ps(); 4];
        for (k, shuf_k) in shuf.iter().enumerate() {
            let j = i * 4 + k;
            let u = _mm256_min_epu8(_mm256_and_si256(_mm256_shuffle_epi8(signs, *shuf_k), bit_masks), ones_8);
            for c in 0..4 {
                let a = cols[c];
                let pos2 = _mm256_madd_epi16(_mm256_maddubs_epi16(u, _mm256_loadu_si256(a.qs.as_ptr().add(j * QK8_0) as *const __m256i)), twos_16);
                let s = _mm256_cvtepi32_ps(_mm256_sub_epi32(pos2, _mm256_loadu_si256(a.tot.as_ptr().add(j * 8) as *const __m256i)));
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

/// `out[r]` = row r of `w` · `a` for `rows` rows (the decode matvec); bits of per-row `vec_dot_act`.
pub fn mat_vec(w: &[u8], rows: usize, a: &Q8Act, out: &mut [f32]) {
    // One row at a time: 2- and 4-row register tiles measured slower (docs/modules/q1_0.md).
    let rb = a.n / QK1_0 * Q1_0_BYTES;
    assert!(w.len() >= rows * rb && out.len() >= rows);
    for (r, o) in out[..rows].iter_mut().enumerate() {
        *o = vec_dot_act(&w[r * rb..], a);
    }
}

/// Prefill tile: one Q1_0 row against four q8_0 columns.
///
/// The sign expansion (broadcast, shuffle, AND, compare) is done once per weight block and reused by
/// all four columns; each column keeps ggml's per-lane float order, so `out[c]` has the bits of
/// `vec_dot(n, x, ys[c])`. ggml b11192 on x86 has no Q1_0 GEMM; its prefill is one vec_dot per pair.
///
/// # Safety
/// As `vec_dot_avx2`, for every column.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_1x4_avx2(n: usize, x: &[u8], ys: [&[u8]; 4]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let ones_8 = _mm256_set1_epi8(1);
    let ones_16 = _mm256_set1_epi16(1);
    let byte_shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    let bit_masks = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
    );
    let zero = _mm256_setzero_si256();
    let mut acc = [_mm256_setzero_ps(); 4];
    let xp = x.as_ptr();
    let yp = [ys[0].as_ptr(), ys[1].as_ptr(), ys[2].as_ptr(), ys[3].as_ptr()];
    for i in 0..n / QK1_0 {
        let xb = xp.add(i * Q1_0_BYTES);
        let d0 = _mm256_set1_ps(f16c(xb));
        let mut ab = [_mm256_setzero_ps(); 4];
        for k in 0..4 {
            let bits = (xb.add(2 + 4 * k) as *const i32).read_unaligned();
            let sm = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(bits), byte_shuf), bit_masks), zero);
            for c in 0..4 {
                let yb = yp[c].add((i * 4 + k) * Q8_0_BYTES);
                let qy = _mm256_loadu_si256(yb.add(2) as *const __m256i);
                let sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
                let s = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16));
                let d1 = _mm256_set1_ps(f16c(yb));
                ab[c] = if k == 0 { _mm256_mul_ps(d1, s) } else { _mm256_fmadd_ps(d1, s, ab[c]) };
            }
        }
        for c in 0..4 {
            acc[c] = _mm256_fmadd_ps(d0, ab[c], acc[c]);
        }
    }
    let mut out = [0f32; 4];
    for c in 0..4 {
        let hi = _mm256_extractf128_ps(acc[c], 1);
        let mut r = _mm_add_ps(hi, _mm256_castps256_ps128(acc[c]));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_movehdup_ps(r));
        out[c] = _mm_cvtss_f32(r);
    }
    out
}

/// out[c·rows + r] = row r of `w` · column c of `ys`: the Q1_0 matmul for a batch of columns (prefill).
///
/// 1×4 tiles where the CPU has AVX2, `vec_dot` for the rest; every element has the bits of ggml's
/// per-pair vec_dot.
pub fn mat_mul(w: &[u8], rows: usize, n: usize, ys: &[&[u8]], out: &mut [f32]) {
    mat_mul_check(w, rows, n, ys, out);
    // SAFETY: checked above; one caller owns all of `out`
    unsafe { mat_mul_rows(w, rows, n, 0..rows, ys, out.as_mut_ptr()) }
}

fn mat_mul_check(w: &[u8], rows: usize, n: usize, ys: &[&[u8]], out: &[f32]) {
    assert!(w.len() >= rows * (n / QK1_0 * Q1_0_BYTES) && out.len() >= rows * ys.len());
    for y in ys {
        check(n, w, y);
    }
}

/// Rows `rs` of the product, written to `out[c·rows + r]`.
///
/// # Safety
/// `mat_mul_check` passed for these arguments, and no other thread writes rows `rs` of `out`.
unsafe fn mat_mul_rows(w: &[u8], rows: usize, n: usize, rs: std::ops::Range<usize>, ys: &[&[u8]], out: *mut f32) {
    let rb = n / QK1_0 * Q1_0_BYTES;
    let mut c0 = 0;
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        while c0 + 4 <= ys.len() {
            let t = [ys[c0], ys[c0 + 1], ys[c0 + 2], ys[c0 + 3]];
            for r in rs.clone() {
                let v = vec_dot_1x4_avx2(n, &w[r * rb..], t);
                for (c, v) in v.into_iter().enumerate() {
                    *out.add((c0 + c) * rows + r) = v;
                }
            }
            c0 += 4;
        }
    }
    for (c, y) in ys.iter().enumerate().skip(c0) {
        for r in rs.clone() {
            *out.add(c * rows + r) = vec_dot(n, &w[r * rb..], y);
        }
    }
}

/// `mat_vec` on every thread of `pool`; the same bits at any thread count.
pub fn mat_vec_par(pool: &crate::par::Pool, w: &[u8], rows: usize, a: &Q8Act, out: &mut [f32]) {
    let rb = a.n / QK1_0 * Q1_0_BYTES;
    assert!(w.len() >= rows * rb && out.len() >= rows);
    pool.rows(rows, out, &|r0, o| mat_vec(&w[r0 * rb..], o.len(), a, o));
}

/// out[c·rows + r] = row r of `w` · prepared column c: the prefill matmul over `Q8Act` columns.
///
/// 1×4 selection tiles where the CPU has AVX2 and no column holds q = −128; `vec_dot_act` otherwise.
/// Every element has the bits of ggml's per-pair vec_dot.
pub fn mat_mul_act(w: &[u8], rows: usize, cols: &[Q8Act], out: &mut [f32]) {
    if mat_mul_act_check(w, rows, cols, out) {
        // SAFETY: checked above; one caller owns all of `out`
        unsafe { mat_mul_act_rows(w, rows, 0..rows, cols, out.as_mut_ptr()) }
    }
}

/// `mat_mul_act` on every thread of `pool`; the same bits at any thread count.
pub fn mat_mul_act_par(pool: &crate::par::Pool, w: &[u8], rows: usize, cols: &[Q8Act], out: &mut [f32]) {
    if !mat_mul_act_check(w, rows, cols, out) {
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
        unsafe { mat_mul_act_rows(w, rows, r0..(r0 + step).min(rows), cols, base as *mut f32) };
    });
}

fn mat_mul_act_check(w: &[u8], rows: usize, cols: &[Q8Act], out: &[f32]) -> bool {
    let Some(n) = cols.first().map(|c| c.n) else { return false };
    assert!(cols.iter().all(|c| c.n == n) && w.len() >= rows * (n / QK1_0 * Q1_0_BYTES) && out.len() >= rows * cols.len());
    true
}

/// # Safety
/// `mat_mul_act_check` passed, and no other thread writes rows `rs` of `out`.
unsafe fn mat_mul_act_rows(w: &[u8], rows: usize, rs: std::ops::Range<usize>, cols: &[Q8Act], out: *mut f32) {
    let rb = cols[0].n / QK1_0 * Q1_0_BYTES;
    let mut c0 = 0;
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        while c0 + 4 <= cols.len() {
            let t = [&cols[c0], &cols[c0 + 1], &cols[c0 + 2], &cols[c0 + 3]];
            if t.iter().any(|a| a.has_min) {
                break; // the wrap case: per-pair from here on
            }
            for r in rs.clone() {
                let v = vec_dot_act_1x4_sel_avx2(&w[r * rb..], t);
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

/// `mat_mul` on every thread of `pool` (row chunks, all columns each); the same bits at any thread count.
pub fn mat_mul_par(pool: &crate::par::Pool, w: &[u8], rows: usize, n: usize, ys: &[&[u8]], out: &mut [f32]) {
    mat_mul_check(w, rows, n, ys, out);
    let (next, base) = (std::sync::atomic::AtomicUsize::new(0), out.as_mut_ptr() as usize);
    let step = crate::par::CHUNK_ROWS;
    pool.run(&|_| loop {
        let r0 = next.fetch_add(step, std::sync::atomic::Ordering::Relaxed);
        if r0 >= rows {
            break;
        }
        // SAFETY: checked; each row range is claimed by exactly one worker
        unsafe { mat_mul_rows(w, rows, n, r0..(r0 + step).min(rows), ys, base as *mut f32) };
    });
}

// ---------------------------------------------------------------- tests -------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// xorshift64* — deterministic, dependency-free test randomness.
    pub struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        pub fn f(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
        /// activation-like: mostly small, a few large outliers (as in real hidden states)
        pub fn act(&mut self) -> f32 {
            let v = self.f() * self.f() * 4.0;
            if self.next().is_multiple_of(97) { v * 60.0 } else { v }
        }
    }

    fn random_q1(r: &mut Rng, nb: usize) -> Vec<u8> {
        let mut x = vec![0u8; nb * Q1_0_BYTES];
        for b in x.as_chunks_mut::<Q1_0_BYTES>().0.iter_mut() {
            b[..2].copy_from_slice(&f32_to_f16(r.f().abs() * 0.05 + 1e-4).to_le_bytes());
            for v in &mut b[2..] {
                *v = r.next() as u8;
            }
        }
        x
    }

    fn random_q8(r: &mut Rng, nb: usize, full_range: bool) -> Vec<u8> {
        let mut y = vec![0u8; nb * 4 * Q8_0_BYTES];
        for b in y.as_chunks_mut::<Q8_0_BYTES>().0.iter_mut() {
            b[..2].copy_from_slice(&f32_to_f16(r.f().abs() * 0.1 + 1e-5).to_le_bytes());
            for v in &mut b[2..] {
                // quantizer output is −127..=127; full_range also feeds −128 (the wrap case)
                *v = if full_range { r.next() as u8 } else { ((r.next() % 255) as i32 - 127) as i8 as u8 };
            }
        }
        y
    }

    #[test]
    fn f16_roundtrip_all_65536() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f32_to_f16(f) & 0x7c00 == 0x7c00 && f32_to_f16(f) & 0x3ff != 0);
            } else {
                assert_eq!(f32_to_f16(f), h, "h={h:#06x} f={f:e}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f16_conversions_match_f16c_hardware() {
        use std::arch::x86_64::*;
        if !is_x86_feature_detected!("f16c") {
            eprintln!("no F16C on this CPU: hardware cross-check skipped");
            return;
        }
        #[target_feature(enable = "f16c")]
        unsafe fn hw(f: f32) -> (u16, f32) {
            let h = _mm_extract_epi16(_mm_cvtps_ph(_mm_set1_ps(f), 0), 0) as u16;
            (h, _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(h as i16))))
        }
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let mut specials: Vec<u32> = vec![0, 0x8000_0000, 0x3380_0000, 0x3300_0000, 0x3300_0001, 0x477f_f000, 0x477f_efff, 0x7f80_0000, 0x3880_0000, 0x387f_e000];
        specials.extend((0..2_000_000).map(|_| r.next() as u32));
        // dense sweep of the f16 range, where rounding decisions happen
        specials.extend((0x3000_0000u32..0x4780_0000).step_by(97));
        for bits in specials {
            let f = f32::from_bits(bits);
            if f.is_nan() {
                continue;
            }
            let (h, back) = unsafe { hw(f) };
            assert_eq!(f32_to_f16(f), h, "f32 {bits:#010x}");
            assert_eq!(f16_to_f32(h).to_bits(), back.to_bits(), "f16 {h:#06x}");
        }
    }

    #[test]
    fn dequantize_layout_scale_first_lsb_first() {
        let mut b = [0u8; 18];
        b[..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        b[2] = 0b0000_0101; // weights 0 and 2 positive
        b[17] = 0b1000_0000; // weight 127 positive
        let mut o = [0f32; 128];
        dequantize_row(&b, &mut o);
        assert_eq!((o[0], o[1], o[2], o[3], o[126], o[127]), (0.5, -0.5, 0.5, -0.5, -0.5, 0.5));
    }

    #[test]
    fn q8_0_matches_avx2_semantics() {
        // half-way cases round to even (AVX2), not away from zero (_ref): amax 127 → id 1.0
        let mut x = [0f32; 32];
        x[0] = 127.0;
        x[1] = 2.5;
        x[2] = -3.5;
        x[3] = 0.5;
        let mut q = [0u8; 34];
        quantize_row_q8_0(&x, &mut q);
        assert_eq!(f16_to_f32(u16::from_le_bytes([q[0], q[1]])), 1.0);
        assert_eq!([q[2] as i8, q[3] as i8, q[4] as i8, q[5] as i8], [127, 2, -4, 0]);
        let mut z = [0u8; 34];
        quantize_row_q8_0(&[0.0; 32], &mut z);
        assert_eq!(z, [0u8; 34]);
    }

    /// The generic port is the mathematical definition: check it against f64 over dequantized data.
    #[test]
    fn generic_matches_dequantized_math() {
        let mut r = Rng(7);
        for nb in [1usize, 3, 16] {
            let (x, y) = (random_q1(&mut r, nb), random_q8(&mut r, nb, false));
            let mut w = vec![0f32; nb * 128];
            dequantize_row(&x, &mut w);
            let exact: f64 = (0..nb * 128)
                .map(|j| {
                    let yb = &y[(j / 32) * Q8_0_BYTES..];
                    w[j] as f64 * f16_to_f32(u16::from_le_bytes([yb[0], yb[1]])) as f64 * (yb[2 + j % 32] as i8) as f64
                })
                .sum();
            let (g, rf) = (vec_dot_generic(nb * 128, &x, &y) as f64, vec_dot_ref(nb * 128, &x, &y) as f64);
            let tol = 1e-5 * exact.abs().max(1.0);
            assert!((g - exact).abs() < tol && (rf - exact).abs() < tol, "nb={nb} exact={exact} generic={g} ref={rf}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_bit_exact_with_ref_randomized() {
        if !has_avx2() {
            eprintln!("no AVX2+FMA: AVX2 path not exercised on this CPU");
            return;
        }
        let mut r = Rng(0xba4c_4d4c_0000_0001);
        let mut cases = 0;
        for round in 0..3000 {
            let nb = 1 + (r.next() % 48) as usize;
            let (x, y) = (random_q1(&mut r, nb), random_q8(&mut r, nb, round % 3 == 0));
            let (a, b) = (vec_dot_ref(nb * 128, &x, &y), unsafe { vec_dot_avx2(nb * 128, &x, &y) });
            assert_eq!(a.to_bits(), b.to_bits(), "round {round} nb {nb}: ref {a:e} avx2 {b:e}");
            cases += 1;
        }
        // through the quantizer, with activation-like inputs
        for _ in 0..500 {
            let nb = 1 + (r.next() % 32) as usize;
            let x = random_q1(&mut r, nb);
            let a: Vec<f32> = (0..nb * 128).map(|_| r.act()).collect();
            let mut y = vec![0u8; nb * 4 * Q8_0_BYTES];
            quantize_row_q8_0(&a, &mut y);
            assert_eq!(vec_dot_ref(nb * 128, &x, &y).to_bits(), vec_dot(nb * 128, &x, &y).to_bits());
            cases += 1;
        }
        assert_eq!(cases, 3500);
    }

    /// Against ggml itself: `testing/ggml_oracle.py` records what b11192's exported symbols return on
    /// the real Bonsai-1.7B file; this re-derives every value in Rust and demands equal bits.
    #[test]
    #[ignore = "needs .models/Bonsai-1.7B-Q1_0.gguf + .models/oracle (testing/ggml_oracle.py); run with --release"]
    fn oracle_ggml_b11192_real_bonsai_1_7b() {
        assert_eq!(oracle_q1_0("Bonsai-1.7B-Q1_0.gguf", "oracle"), (197, 788, 788));
    }

    /// The same oracle on the 8B 1-bit model the node serves (254 tensors, 8.19 B weights).
    #[test]
    #[ignore = "needs .models/Bonsai-8B-Q1_0.gguf + .models/oracle-8b-q1 (testing/ggml_oracle.py); run with --release"]
    fn oracle_ggml_b11192_real_bonsai_8b_q1_0() {
        let (tensors, cases, generic_same) = oracle_q1_0("Bonsai-8B-Q1_0.gguf", "oracle-8b-q1");
        assert_eq!((tensors, generic_same), (254, cases));
    }

    /// Every dequantized tensor, q8_0 row and dot product the oracle recorded, re-derived and compared
    /// bit for bit; returns (tensors, cases, cases where the generic port equals ggml generic).
    fn oracle_q1_0(file: &str, oracle: &str) -> (usize, usize, usize) {
        use crate::gguf::{guard_file, Engine, Mmap};
        use crate::sha256::{hex, Sha256};
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let model = dir.join(file);
        let rep = guard_file(&model, Engine::Mainline).unwrap();
        let h = rep.header.as_ref().unwrap();
        let mm = Mmap::open(&model).unwrap();
        let find = |name: &str| h.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{name}"));
        let tensor_bytes = |_: &std::path::Path, _: &crate::gguf::Header, t: &crate::gguf::TensorInfo| mm.tensor(h, t).map(|b| b.to_vec()).ok_or(());

        // 1. dequantize: sha256 of the f32 output, every Q1_0 tensor
        let tsv = std::fs::read_to_string(dir.join(oracle).join("dequant.tsv")).unwrap();
        let (mut tensors, mut elems) = (0, 0u64);
        for line in tsv.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            let t = find(f[0]);
            let raw = tensor_bytes(&model, h, t).unwrap();
            let mut sh = Sha256::default();
            let mut buf = vec![0f32; 1 << 20];
            let mut bytes = vec![0u8; 4 << 20];
            for c in raw.chunks(Q1_0_BYTES * (1 << 13)) {
                let n = c.len() / Q1_0_BYTES * QK1_0;
                dequantize_row(c, &mut buf[..n]);
                for (b, v) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(&buf[..n]) {
                    b.copy_from_slice(&v.to_le_bytes());
                }
                sh.update(&bytes[..4 * n]);
            }
            assert_eq!(hex(&sh.finish()), f[2], "dequantize {}", f[0]);
            tensors += 1;
            elems += f[1].parse::<u64>().unwrap();
        }

        // 2. q8_0 activations + vec_dot: bytes and bits
        let vb = std::fs::read(dir.join(oracle).join("vecdot.bin")).unwrap();
        let (mut o, mut cases, mut generic_same) = (0usize, 0, 0);
        let u32at = |o: usize| u32::from_le_bytes(vb[o..o + 4].try_into().unwrap()) as usize;
        let mut cache: Option<(String, Vec<u8>)> = None;
        while o < vb.len() {
            let nl = u32at(o);
            let name = std::str::from_utf8(&vb[o + 4..o + 4 + nl]).unwrap().to_string();
            o += 4 + nl;
            let (row, n) = (u32at(o), u32at(o + 4));
            o += 8;
            let x: Vec<f32> = vb[o..o + 4 * n].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            o += 4 * n;
            let q8_ggml = &vb[o..o + n / 32 * Q8_0_BYTES];
            o += n / 32 * Q8_0_BYTES;
            let (s_avx, s_gen) = (f32::from_le_bytes(vb[o..o + 4].try_into().unwrap()), f32::from_le_bytes(vb[o + 4..o + 8].try_into().unwrap()));
            o += 8;
            if cache.as_ref().map(|c| c.0 != name).unwrap_or(true) {
                cache = Some((name.clone(), tensor_bytes(&model, h, find(&name)).unwrap()));
            }
            let rb = n / QK1_0 * Q1_0_BYTES;
            let w = &cache.as_ref().unwrap().1[row * rb..(row + 1) * rb];
            let mut q8 = vec![0u8; n / 32 * Q8_0_BYTES];
            quantize_row_q8_0(&x, &mut q8);
            assert!(q8 == q8_ggml, "quantize_row_q8_0 differs: {name} row {row}");
            let (r, v) = (vec_dot_ref(n, w, &q8), vec_dot(n, w, &q8));
            assert_eq!((r.to_bits(), v.to_bits()), (s_avx.to_bits(), s_avx.to_bits()), "{name} row {row}: ggml {s_avx:e} ref {r:e} fast {v:e}");
            let act = Q8Act::quantize(&x);
            assert_eq!(vec_dot_act(w, &act).to_bits(), s_avx.to_bits(), "vec_dot_act {name} row {row}");
            generic_same += (vec_dot_generic(n, w, &q8).to_bits() == s_gen.to_bits()) as usize;
            cases += 1;
        }
        eprintln!("oracle: {tensors} tensors / {elems} weights dequantized bit-exact; {cases} q8_0 rows byte-exact; \
                   {cases} vec_dot and vec_dot_act bit-exact vs ggml AVX2; generic port == ggml generic in {generic_same}/{cases}");
        (tensors, cases, generic_same)
    }

    /// ns per Q1_0 block (128 weights), one thread. Two regimes: L1-resident (one 32-block row,
    /// compute-bound) and a Bonsai-8B-shaped GEMV (ffn_gate: 12,288 rows × 4,096 = 6.75 MiB of
    /// weights, larger than L2/L3 slices — memory-bound, the regime decode lives in).
    /// `cargo test --release -- --ignored bench_q1_0 --nocapture`
    #[test]
    #[ignore = "benchmark; run with --release --nocapture"]
    fn bench_q1_0() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut r = Rng(42);
        let time = |label: &str, blocks_per_call: usize, calls: usize, f: &dyn Fn(usize) -> f32| {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                let mut acc = 0f32;
                for c in 0..calls {
                    acc += f(c);
                }
                black_box(acc);
                best = best.min(t.elapsed().as_nanos() as f64 / (calls * blocks_per_call) as f64);
            }
            eprintln!("{label:44} {best:7.2} ns/block");
            best
        };
        let n = 4096;
        let (x, y) = (random_q1(&mut r, n / 128), random_q8(&mut r, n / 128, false));
        let g = time("L1 row n=4096   generic (ggml C port)", 32, 20_000, &|_| vec_dot_generic(n, black_box(&x), black_box(&y)));
        let s = time("L1 row n=4096   ref (scalar AVX2 model)", 32, 20_000, &|_| vec_dot_ref(n, black_box(&x), black_box(&y)));
        let a = time("L1 row n=4096   AVX2", 32, 200_000, &|_| vec_dot(n, black_box(&x), black_box(&y)));
        eprintln!("  AVX2 speedup: {:.1}x over generic, {:.1}x over ref", g / a, s / a);
        let rows = 12288;
        let w = random_q1(&mut r, rows * n / 128);
        let rb = n / 128 * Q1_0_BYTES;
        let m = time("GEMV 12288x4096 (6.75 MiB) generic", 32, rows, &|i| vec_dot_generic(n, &w[i * rb..], &y));
        let v = time("GEMV 12288x4096 (6.75 MiB) AVX2", 32, rows, &|i| vec_dot(n, &w[i * rb..], &y));
        eprintln!("  GEMV AVX2 speedup {:.1}x; effective weight bandwidth {:.2} GB/s", m / v, 18.0 / v);
    }

    #[test]
    fn act_path_bit_exact_with_vec_dot() {
        let mut r = Rng(1234);
        for round in 0..2000 {
            let nb = 1 + (r.next() % 40) as usize;
            let (x, y) = (random_q1(&mut r, nb), random_q8(&mut r, nb, round % 2 == 0));
            let a = Q8Act::from_q8_0(nb * 128, &y);
            assert_eq!(vec_dot_act(&x, &a).to_bits(), vec_dot_ref(nb * 128, &x, &y).to_bits(), "round {round}");
        }
    }

    #[test]
    fn mat_vec_bit_exact_with_vec_dot() {
        let mut r = Rng(4321);
        for (rows, nb) in [(1usize, 1usize), (4, 3), (7, 32), (13, 5), (64, 48)] {
            let w = random_q1(&mut r, rows * nb);
            let y = random_q8(&mut r, nb, false);
            let a = Q8Act::from_q8_0(nb * 128, &y);
            let mut out = vec![0f32; rows];
            mat_vec(&w, rows, &a, &mut out);
            for i in 0..rows {
                let want = vec_dot_ref(nb * 128, &w[i * nb * Q1_0_BYTES..], &y);
                assert_eq!(out[i].to_bits(), want.to_bits(), "rows {rows} nb {nb} row {i}");
            }
        }
    }

    #[test]
    fn mat_mul_bit_exact_with_per_pair_vec_dot() {
        let mut r = Rng(99);
        for (rows, nb, cols) in [(1usize, 1usize, 1usize), (3, 2, 4), (5, 7, 9), (8, 32, 13)] {
            let w = random_q1(&mut r, rows * nb);
            let ys: Vec<Vec<u8>> = (0..cols).map(|_| random_q8(&mut r, nb, true)).collect();
            let yr: Vec<&[u8]> = ys.iter().map(|v| v.as_slice()).collect();
            let mut out = vec![0f32; rows * cols];
            mat_mul(&w, rows, nb * 128, &yr, &mut out);
            for c in 0..cols {
                for i in 0..rows {
                    let want = vec_dot_ref(nb * 128, &w[i * nb * Q1_0_BYTES..], ys[c].as_slice());
                    assert_eq!(out[c * rows + i].to_bits(), want.to_bits(), "rows {rows} nb {nb} col {c} row {i}");
                }
            }
        }
    }

    /// The `Q8Act` prefill tile against the scalar model of ggml, bit for bit.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn act_tile_bit_exact() {
        if !has_avx2() {
            return;
        }
        let mut r = Rng(4040);
        for round in 0..1500 {
            let nb = 1 + (r.next() % 41) as usize;
            let x = random_q1(&mut r, nb);
            let ys: Vec<Vec<u8>> = (0..4).map(|_| random_q8(&mut r, nb, false)).collect();
            let acts: Vec<Q8Act> = ys.iter().map(|y| Q8Act::from_q8_0(nb * 128, y)).collect();
            let want: Vec<u32> = ys.iter().map(|y| vec_dot_ref(nb * 128, &x, y).to_bits()).collect();
            let t = unsafe { vec_dot_act_1x4_sel_avx2(&x, [&acts[0], &acts[1], &acts[2], &acts[3]]) };
            assert_eq!(t.map(f32::to_bits).to_vec(), want, "1x4 sel round {round} nb {nb}");
        }
    }

    /// The `Q8Act` prefill tile timed against the q8-byte tile it replaces (1024×4096 × 32 columns).
    /// `cargo test --release -- --ignored bench_q1_0_prefill_act --nocapture --test-threads=1`
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "benchmark; run with --release --nocapture"]
    fn bench_q1_0_prefill_act() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut r = Rng(4041);
        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[0], v[v.len() / 2])
        };
        let n = 4096;
        let rb = n / 128 * Q1_0_BYTES;
        let w = random_q1(&mut r, 1024 * n / 128);
        // prefill
        let (pr, cols) = (1024usize, 32usize);
        let ys: Vec<Vec<u8>> = (0..cols).map(|_| random_q8(&mut r, n / 128, false)).collect();
        let yr: Vec<&[u8]> = ys.iter().map(|v| v.as_slice()).collect();
        let acts: Vec<Q8Act> = ys.iter().map(|y| Q8Act::from_q8_0(n, y)).collect();
        let pb = (pr * cols * n / 128) as f64;
        let (mut old, mut new) = (vec![], vec![]);
        let mut o = vec![0f32; pr * cols];
        for _ in 0..9 {
            let t = Instant::now();
            mat_mul(&w, pr, n, &yr, &mut o);
            old.push(t.elapsed().as_nanos() as f64 / pb);
            let t = Instant::now();
            for c0 in (0..cols).step_by(4) {
                for i in 0..pr {
                    black_box(unsafe { vec_dot_act_1x4_sel_avx2(&w[i * rb..], [&acts[c0], &acts[c0 + 1], &acts[c0 + 2], &acts[c0 + 3]]) });
                }
            }
            new.push(t.elapsed().as_nanos() as f64 / pb);
        }
        let (p1, p2) = (med(old), med(new));
        eprintln!("prefill 1024x4096 x 32 ns/(block·col) min/median: 1x4 (q8 bytes) {:.2}/{:.2} | 1x4 sel (Q8Act) {:.2}/{:.2} ({:.3}x)", p1.0, p1.1, p2.0, p2.1, p1.1 / p2.1);
    }

    #[test]
    fn par_bit_exact_with_single_thread() {
        let mut r = Rng(778);
        let (rows, nb, cols) = (203usize, 5usize, 7usize);
        let w = random_q1(&mut r, rows * nb);
        let ys: Vec<Vec<u8>> = (0..cols).map(|_| random_q8(&mut r, nb, true)).collect();
        let yr: Vec<&[u8]> = ys.iter().map(|v| v.as_slice()).collect();
        let a = Q8Act::from_q8_0(nb * 128, &ys[0]);
        let (mut mv, mut mm) = (vec![0f32; rows], vec![0f32; rows * cols]);
        mat_vec(&w, rows, &a, &mut mv);
        mat_mul(&w, rows, nb * 128, &yr, &mut mm);
        for th in [1, 2, 3, 4] {
            let pool = crate::par::Pool::new(th);
            let (mut pv, mut pm) = (vec![0f32; rows], vec![0f32; rows * cols]);
            mat_vec_par(&pool, &w, rows, &a, &mut pv);
            mat_mul_par(&pool, &w, rows, nb * 128, &yr, &mut pm);
            let acts: Vec<Q8Act> = ys.iter().map(|y| Q8Act::from_q8_0(nb * 128, y)).collect();
            let mut pa = vec![0f32; rows * cols];
            mat_mul_act_par(&pool, &w, rows, &acts, &mut pa);
            assert!(pa.iter().zip(&mm).all(|(a, b)| a.to_bits() == b.to_bits()), "mat_mul_act_par {th}");
            assert!(pv.iter().zip(&mv).all(|(a, b)| a.to_bits() == b.to_bits()), "mat_vec_par {th}");
            assert!(pm.iter().zip(&mm).all(|(a, b)| a.to_bits() == b.to_bits()), "mat_mul_par {th}");
        }
    }

    /// Prefill: ns per (block × column) — per-pair vec_dot (what ggml does) vs the 1×4 tile.
    #[test]
    #[ignore = "benchmark; run with --release --nocapture"]
    fn bench_q1_0_prefill() {
        use std::time::Instant;
        let mut r = Rng(5);
        let (rows, n, cols) = (1024usize, 4096usize, 32usize);
        let w = random_q1(&mut r, rows * n / 128);
        let ys: Vec<Vec<u8>> = (0..cols).map(|_| random_q8(&mut r, n / 128, false)).collect();
        let yr: Vec<&[u8]> = ys.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![0f32; rows * cols];
        let pairs_blocks = (rows * cols * n / 128) as f64;
        let mut best = [f64::MAX; 2];
        for _ in 0..5 {
            let t = Instant::now();
            for c in 0..cols {
                for i in 0..rows {
                    out[c * rows + i] = vec_dot(n, &w[i * n / 128 * Q1_0_BYTES..], yr[c]);
                }
            }
            best[0] = best[0].min(t.elapsed().as_nanos() as f64 / pairs_blocks);
            std::hint::black_box(&out);
            let t = Instant::now();
            mat_mul(&w, rows, n, &yr, &mut out);
            best[1] = best[1].min(t.elapsed().as_nanos() as f64 / pairs_blocks);
            std::hint::black_box(&out);
        }
        eprintln!("prefill 1024x4096 x 32 cols: per-pair vec_dot {:.2} ns/block·col, 1x4 tile {:.2} ns/block·col ({:.2}x)", best[0], best[1], best[0] / best[1]);
    }

    /// One token's worth of every Q1_0 matmul in Bonsai-8B (36 × q,k,v,o,gate,up,down + output), each
    /// tensor once in file order, ggml's AVX2 kernel vs bankml's `mat_vec_par`, both on the same pool and
    /// row scheduler, per thread count (`BANKML_THREADS`, default "1,2,3,4"); then every tensor's first 64
    /// rows compared bit for bit, bankml threaded.
    /// `BANKML_GGML_LIB=… cargo test --release -- --ignored decode_budget_q1_0 --nocapture --test-threads=1`
    #[test]
    #[ignore = "needs BANKML_GGML_LIB + .models/Bonsai-8B-Q1_0.gguf; ~1 min"]
    fn decode_budget_q1_0() {
        use crate::gguf::{guard_file, Engine, Mmap};
        use crate::par::Pool;
        use std::time::Instant;
        let ggml = ggml_q1_0_vec_dot();
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/Bonsai-8B-Q1_0.gguf");
        let rep = guard_file(&model, Engine::Mainline).unwrap();
        let h = rep.header.as_ref().unwrap();
        let mm = Mmap::open(&model).unwrap();
        let mut r = Rng(81);
        let mats: Vec<_> = h.tensors.iter().filter(|t| t.ty == 41 && t.name != "token_embd.weight").collect();
        let mut acts = std::collections::HashMap::new();
        for t in &mats {
            let n = t.dims[0] as usize;
            acts.entry(n).or_insert_with(|| {
                let x: Vec<f32> = (0..n).map(|_| r.act()).collect();
                let mut q8 = vec![0u8; n / QK8_0 * Q8_0_BYTES];
                quantize_row_q8_0(&x, &mut q8);
                let a = Q8Act::from_q8_0(n, &q8);
                (q8, a)
            });
        }
        let weights: u64 = mats.iter().map(|t| t.dims[0] * t.dims[1]).sum();
        eprintln!("decode budget: {} Q1_0 matmuls, {weights} weights ({:.2} GB of blocks) per token", mats.len(), weights as f64 / 128.0 * 18.0 / 1e9);
        let gemv = |g: bool, t: &crate::gguf::TensorInfo, o: &mut [f32], rows: usize, pool: &Pool| {
            let (n, w) = (t.dims[0] as usize, mm.tensor(h, t).unwrap());
            let (q8, a) = &acts[&n];
            let rb = n / QK1_0 * Q1_0_BYTES;
            if g {
                pool.rows(rows, o, &|r0, o| o.iter_mut().enumerate().for_each(|(i, s)| unsafe { ggml(n as i32, s, 0, w[(r0 + i) * rb..].as_ptr() as _, 0, q8.as_ptr() as _, 0, 1) }));
            } else {
                mat_vec_par(pool, w, rows, a, o);
            }
        };
        let threads = std::env::var("BANKML_THREADS").unwrap_or("1,2,3,4".into());
        for th in threads.split(',').map(|v| v.parse::<usize>().unwrap()) {
            let pool = Pool::new(th);
            let pass = |g: bool| {
                let t0 = Instant::now();
                for t in &mats {
                    let rows = t.dims[1] as usize;
                    gemv(g, t, &mut vec![0f32; rows], rows, &pool);
                }
                t0.elapsed().as_secs_f64()
            };
            pass(true); // page in
            let (mut tg, mut tb) = (vec![], vec![]);
            for _ in 0..3 {
                tg.push(pass(true));
                tb.push(pass(false));
            }
            tg.sort_by(|a, b| a.partial_cmp(b).unwrap());
            tb.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!("  {th} thread(s): ggml b11192 {:.3}/{:.3} s/token (min/median) | bankml {:.3}/{:.3} | {:.2}x | matmul-only ceiling ggml {:.2}, bankml {:.2} tok/s",
                tg[0], tg[1], tb[0], tb[1], tg[1] / tb[1], 1.0 / tg[0], 1.0 / tb[0]);
        }
        let (one, par) = (Pool::new(1), Pool::new(3));
        for t in &mats {
            let (mut og, mut ob) = (vec![0f32; 64], vec![0f32; 64]);
            gemv(true, t, &mut og, 64, &one);
            gemv(false, t, &mut ob, 64, &par);
            assert!(og.iter().zip(&ob).all(|(x, y)| x.to_bits() == y.to_bits()), "{}: bits", t.name);
        }
    }

    type VecDot = unsafe extern "C" fn(i32, *mut f32, usize, *const std::ffi::c_void, usize, *const std::ffi::c_void, usize, i32);

    /// ggml b11192's own `ggml_vec_dot_q1_0_q8_0` from the shipped haswell library (dlopen, no crate).
    pub(crate) fn ggml_q1_0_vec_dot() -> VecDot {
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
            let f = dlsym(h, c"ggml_vec_dot_q1_0_q8_0".as_ptr());
            assert!(!f.is_null());
            std::mem::transmute::<*mut c_void, VecDot>(f)
        }
    }

    /// In-process A/B against ggml b11192's own exported AVX2 kernel (dlopen, no crate): the two
    /// alternate on identical data, 25 rounds, so CPU frequency drift hits both alike. Reports min
    /// and median ns/block. `BANKML_GGML_LIB=<dir with libggml-cpu-haswell.so> cargo test --release
    /// -- --ignored ab_vs_ggml --nocapture --test-threads=1`
    #[test]
    #[ignore = "needs BANKML_GGML_LIB (llama.cpp b11192 ubuntu-x64 release dir)"]
    fn ab_vs_ggml() {
        use std::ffi::{c_char, c_int, c_void, CString};
        use std::time::Instant;
        extern "C" {
            fn dlopen(f: *const c_char, flag: c_int) -> *mut c_void;
            fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void;
        }
        type VecDot = unsafe extern "C" fn(c_int, *mut f32, usize, *const c_void, usize, *const c_void, usize, c_int);
        let dir = std::env::var("BANKML_GGML_LIB").expect("BANKML_GGML_LIB");
        let (base, cpu) = (CString::new(format!("{dir}/libggml-base.so")).unwrap(), CString::new(format!("{dir}/libggml-cpu-haswell.so")).unwrap());
        let ggml: VecDot = unsafe {
            assert!(!dlopen(base.as_ptr(), 0x102).is_null(), "dlopen base"); // RTLD_NOW|RTLD_GLOBAL
            let h = dlopen(cpu.as_ptr(), 0x102);
            assert!(!h.is_null(), "dlopen cpu");
            // ggml's CPU fp16→fp32 goes through a table that ggml_cpu_init fills; uninitialised it reads 0
            let init = dlsym(h, c"ggml_cpu_init".as_ptr());
            assert!(!init.is_null());
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(init)();
            let f = dlsym(h, c"ggml_vec_dot_q1_0_q8_0".as_ptr());
            assert!(!f.is_null());
            std::mem::transmute::<*mut c_void, VecDot>(f)
        };
        let mut r = Rng(11);
        let (rows, n) = (12288usize, 4096usize);
        let rb = n / 128 * Q1_0_BYTES;
        let w = random_q1(&mut r, rows * n / 128);
        let y = random_q8(&mut r, n / 128, false);
        let blocks = (rows * n / 128) as f64;
        let act = Q8Act::from_q8_0(n, &y);
        let (mut tb, mut tg, mut ta, mut tm) = (vec![], vec![], vec![], vec![]);
        let mut chk = (0f32, 0f32, 0f32);
        let mut mv = vec![0f32; rows];
        for _ in 0..25 {
            let t = Instant::now();
            mat_vec(&w, rows, &act, &mut mv);
            tm.push(t.elapsed().as_nanos() as f64 / blocks);
            let t = Instant::now();
            for i in 0..rows {
                chk.2 += vec_dot_act(&w[i * rb..], &act);
            }
            ta.push(t.elapsed().as_nanos() as f64 / blocks);
            let t = Instant::now();
            for i in 0..rows {
                chk.0 += vec_dot(n, &w[i * rb..], &y);
            }
            tb.push(t.elapsed().as_nanos() as f64 / blocks);
            let t = Instant::now();
            for i in 0..rows {
                let mut s = 0f32;
                unsafe { ggml(n as c_int, &mut s, 0, w[i * rb..].as_ptr() as _, 0, y.as_ptr() as _, 0, 1) };
                chk.1 += s;
            }
            tg.push(t.elapsed().as_nanos() as f64 / blocks);
        }
        assert_eq!((chk.0.to_bits(), chk.2.to_bits()), (chk.1.to_bits(), chk.1.to_bits()), "same data, same bits");
        let stat = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[0], v[v.len() / 2])
        };
        for i in 0..rows {
            let mut s = 0f32;
            unsafe { ggml(n as c_int, &mut s, 0, w[i * rb..].as_ptr() as _, 0, y.as_ptr() as _, 0, 1) };
            assert_eq!(mv[i].to_bits(), s.to_bits(), "mat_vec row {i}");
        }
        let m = stat(&mut tm);
        let (b, g, a) = (stat(&mut tb), stat(&mut tg), stat(&mut ta));
        eprintln!("GEMV 12288x4096, ns/block min/median: bankml mat_vec {:.2}/{:.2} ({:.3}x ggml median)", m.0, m.1, stat(&mut tg.clone()).1 / m.1);
        eprintln!("GEMV 12288x4096, ns/block min/median: ggml b11192 {:.2}/{:.2} | bankml vec_dot {:.2}/{:.2} ({:.3}x) | bankml vec_dot_act {:.2}/{:.2} ({:.3}x)",
            g.0, g.1, b.0, b.1, g.1 / b.1, a.0, a.1, g.1 / a.1);

        // prefill: 1024 rows x 32 columns — ggml's per-pair vec_dot vs bankml's 1x4 tile (mat_mul)
        let (pr, cols) = (1024usize, 32usize);
        let ys: Vec<Vec<u8>> = (0..cols).map(|_| random_q8(&mut r, n / 128, false)).collect();
        let yr: Vec<&[u8]> = ys.iter().map(|v| v.as_slice()).collect();
        let (mut og, mut ob, mut oa) = (vec![0f32; pr * cols], vec![0f32; pr * cols], vec![0f32; pr * cols]);
        let pacts: Vec<Q8Act> = ys.iter().map(|y| Q8Act::from_q8_0(n, y)).collect();
        let pb = (pr * cols * n / 128) as f64;
        let (mut pg, mut pbk, mut pact) = (vec![], vec![], vec![]);
        for _ in 0..15 {
            let t = Instant::now();
            for c in 0..cols {
                for i in 0..pr {
                    unsafe { ggml(n as c_int, &mut og[c * pr + i], 0, w[i * rb..].as_ptr() as _, 0, yr[c].as_ptr() as _, 0, 1) };
                }
            }
            pg.push(t.elapsed().as_nanos() as f64 / pb);
            let t = Instant::now();
            mat_mul(&w, pr, n, &yr, &mut ob);
            pbk.push(t.elapsed().as_nanos() as f64 / pb);
            let t = Instant::now();
            mat_mul_act(&w, pr, &pacts, &mut oa);
            pact.push(t.elapsed().as_nanos() as f64 / pb);
        }
        assert!(og.iter().zip(&ob).all(|(a, b)| a.to_bits() == b.to_bits()), "prefill bits");
        assert!(og.iter().zip(&oa).all(|(a, b)| a.to_bits() == b.to_bits()), "prefill (Q8Act tile) bits");
        let (pg, pbk, pact) = (stat(&mut pg), stat(&mut pbk), stat(&mut pact));
        eprintln!("prefill 1024x4096 x 32 cols, ns/(block·col) min/median: ggml per-pair {:.2}/{:.2} | bankml 1x4 tile {:.2}/{:.2} ({:.3}x) | bankml Q8Act tile {:.2}/{:.2} ({:.3}x)",
            pg.0, pg.1, pbk.0, pbk.1, pg.1 / pbk.1, pact.0, pact.1, pg.1 / pact.1);
    }
}
