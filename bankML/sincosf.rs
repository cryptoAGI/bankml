// SPDX-License-Identifier: MIT OR Apache-2.0
//! `sinf` and `cosf` as glibc computes them on x86-64 with FMA — the reference's own rope.
//!
//! llama.cpp b11192 builds its rope cache with the C library's `cosf`/`sinf`; on the reference machine that is
//! glibc 2.35, whose x86-64 build picks its FMA variant at run time. Native bankML calls the same functions and so
//! gets the same bits. WebAssembly has no glibc: Rust's own libm differs from glibc in the last bit for about 1 % of
//! the rope's angles, and a long conversation then parts from the reference (browseML's oracle, turn 2.2). So
//! browseML uses this port instead.
//!
//! The algorithm is ARM's optimized-routines `sinf`/`cosf` (math/sinf.c, cosf.c, sincosf.h, sincosf_data.c;
//! Copyright (c) 2018-2022 Arm Limited; MIT OR Apache-2.0 WITH LLVM-exception), which glibc adopted unchanged:
//! the same tables, the same range reduction (`TOINT_INTRINSICS` 0 on x86-64), the same polynomials. Where GCC
//! contracts a multiply and its add into an FMA in glibc's `-mfma` build, this writes `mul_add`.
//! `sincosf_matches_glibc_on_every_float` checks the port against the C library on all 2^32 inputs.

const PI63: f64 = f64::from_bits(0x3c19_21fb_5444_2d18); // 0x1.921FB54442D18p-62

struct SinCos {
    sign: [f64; 4],
    hpi_inv: f64,
    hpi: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    s1: f64,
    s2: f64,
    s3: f64,
}

/// `__sincosf_table`: the second entry computes −cos(x), giving the negation for free.
const TABLE: [SinCos; 2] = [
    SinCos {
        sign: [1.0, -1.0, -1.0, 1.0],
        hpi_inv: f64::from_bits(0x4164_5f30_6dc9_c883), // 0x1.45F306DC9C883p+23
        hpi: f64::from_bits(0x3ff9_21fb_5444_2d18),     // 0x1.921FB54442D18p0
        c0: 1.0,
        c1: f64::from_bits(0xbfdf_ffff_fd0c_621c), // -0x1.ffffffd0c621cp-2
        c2: f64::from_bits(0x3fa5_5553_e106_8f19), // 0x1.55553e1068f19p-5
        c3: f64::from_bits(0xbf56_c087_e89a_359d), // -0x1.6c087e89a359dp-10
        c4: f64::from_bits(0x3ef9_9343_027b_f8c3), // 0x1.99343027bf8c3p-16
        s1: f64::from_bits(0xbfc5_5554_5995_a603), // -0x1.555545995a603p-3
        s2: f64::from_bits(0x3f81_1076_0523_0bc4), // 0x1.1107605230bc4p-7
        s3: f64::from_bits(0xbf29_94eb_3774_cf24), // -0x1.994eb3774cf24p-13
    },
    SinCos {
        sign: [1.0, -1.0, -1.0, 1.0],
        hpi_inv: f64::from_bits(0x4164_5f30_6dc9_c883),
        hpi: f64::from_bits(0x3ff9_21fb_5444_2d18),
        c0: -1.0,
        c1: f64::from_bits(0x3fdf_ffff_fd0c_621c),
        c2: f64::from_bits(0xbfa5_5553_e106_8f19),
        c3: f64::from_bits(0x3f56_c087_e89a_359d),
        c4: f64::from_bits(0xbef9_9343_027b_f8c3),
        s1: f64::from_bits(0xbfc5_5554_5995_a603),
        s2: f64::from_bits(0x3f81_1076_0523_0bc4),
        s3: f64::from_bits(0xbf29_94eb_3774_cf24),
    },
];

/// `__inv_pio4`: 4/π, 192 bits, 8 new bits per entry.
const INV_PIO4: [u32; 24] = [
    0xa2, 0xa2f9, 0xa2f983, 0xa2f9836e, 0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27,
    0x29fc2757, 0xfc2757d1, 0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62, 0xc0db6295,
    0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041,
];

#[inline(always)]
fn abstop12(x: f32) -> u32 {
    (x.to_bits() >> 20) & 0x7ff
}

/// `sinf_poly`: sin(x) for even `n`, cos(x) for odd, x in [−π/4, π/4].
#[inline(always)]
fn poly(x: f64, x2: f64, p: &SinCos, n: i32) -> f32 {
    if n & 1 == 0 {
        let x3 = x * x2;
        let s1 = x2.mul_add(p.s3, p.s2);
        let x7 = x3 * x2;
        let s = x3.mul_add(p.s1, x);
        x7.mul_add(s1, s) as f32
    } else {
        let x4 = x2 * x2;
        let c2 = x2.mul_add(p.c4, p.c3);
        let c1 = x2.mul_add(p.c1, p.c0);
        let x6 = x4 * x2;
        let c = x4.mul_add(p.c2, c1);
        x6.mul_add(c2, c) as f32
    }
}

/// `reduce_fast`: |x| < 120, one multiply-subtract; the quadrant in bits 24..31 of the scaled product.
#[inline(always)]
fn reduce_fast(x: f64, p: &SinCos) -> (f64, i32) {
    let r = x * p.hpi_inv;
    let n = ((r as i32).wrapping_add(0x80_0000)) >> 24;
    ((-(n as f64)).mul_add(p.hpi, x), n)
}

/// `reduce_large`: the exact 2.62-bit fixed-point modulo from a 32×96 → 128-bit multiply.
#[inline(always)]
fn reduce_large(xi: u32) -> (f64, i32) {
    let arr = &INV_PIO4[((xi >> 26) & 15) as usize..];
    let shift = (xi >> 23) & 7;
    let xi = ((xi & 0xff_ffff) | 0x80_0000) << shift;
    let mut res0 = xi.wrapping_mul(arr[0]) as u64;
    let res1 = xi as u64 * arr[4] as u64;
    let res2 = xi as u64 * arr[8] as u64;
    res0 = (res2 >> 32) | (res0 << 32);
    res0 = res0.wrapping_add(res1);
    let n = (res0.wrapping_add(1u64 << 61)) >> 62;
    res0 = res0.wrapping_sub(n << 62);
    ((res0 as i64) as f64 * PI63, n as i32)
}

fn sin_or_cos(y: f32, cos: bool) -> f32 {
    let x = y as f64;
    let p = &TABLE[0];
    let odd = cos as i32;
    if abstop12(y) < abstop12(f32::from_bits(0x3f49_0fdb)) {
        // |y| < π/4
        if abstop12(y) < abstop12(f32::from_bits(0x3980_0000)) {
            // |y| < 2^-12
            return if cos { 1.0 } else { y };
        }
        poly(x, x * x, p, odd)
    } else if abstop12(y) < abstop12(120.0) {
        let (x, n) = reduce_fast(x, p);
        let s = p.sign[(n & 3) as usize];
        let p = if n & 2 != 0 { &TABLE[1] } else { p };
        poly(x * s, x * x, p, n ^ odd)
    } else if abstop12(y) < abstop12(f32::INFINITY) {
        let xi = y.to_bits();
        let sign = (xi >> 31) as i32;
        let (x, n) = reduce_large(xi);
        let s = p.sign[((n + sign) & 3) as usize];
        let p = if (n + sign) & 2 != 0 { &TABLE[1] } else { p };
        poly(x * s, x * x, p, n ^ odd)
    } else {
        f32::NAN // __math_invalidf: (y − y) / (y − y)
    }
}

/// glibc's `sinf` (x86-64, FMA).
pub fn sinf(y: f32) -> f32 {
    sin_or_cos(y, false)
}

/// glibc's `cosf` (x86-64, FMA).
pub fn cosf(y: f32) -> f32 {
    sin_or_cos(y, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All 2^32 inputs against the C library this machine runs (the reference's glibc, its FMA variant on an FMA
    /// CPU). Ignored by default: a few minutes on one core.
    #[test]
    #[ignore]
    fn sincosf_matches_glibc_on_every_float() {
        extern "C" {
            fn sinf(x: f32) -> f32;
            fn cosf(x: f32) -> f32;
        }
        let (mut bad_s, mut bad_c) = (0u64, 0u64);
        for b in 0..=u32::MAX {
            let y = f32::from_bits(b);
            let (gs, gc) = unsafe { (sinf(y), cosf(y)) };
            let (ms, mc) = (super::sinf(y), super::cosf(y));
            if gs.to_bits() != ms.to_bits() && !(gs.is_nan() && ms.is_nan()) {
                if bad_s < 5 {
                    eprintln!("sinf({y:e} = {b:#010x}): glibc {gs:e} port {ms:e}");
                }
                bad_s += 1;
            }
            if gc.to_bits() != mc.to_bits() && !(gc.is_nan() && mc.is_nan()) {
                if bad_c < 5 {
                    eprintln!("cosf({y:e} = {b:#010x}): glibc {gc:e} port {mc:e}");
                }
                bad_c += 1;
            }
        }
        assert_eq!((bad_s, bad_c), (0, 0), "inputs where the port differs from glibc (sinf, cosf)");
    }
}
