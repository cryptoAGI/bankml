// SPDX-License-Identifier: MIT OR Apache-2.0
//! F16 weights (O3, 0.3.4), as the shipped ggml-cpu b11192 (`libggml-cpu-haswell.so`) multiplies them. Read from
//! the tag's `ggml-cpu.c` (`ggml_compute_forward_mul_mat`), `vec.cpp` (`ggml_vec_dot_f16`) and
//! `llamafile/sgemm.cpp` (`llamafile_sgemm`, `tinyBLAS`):
//!
//! - the activation is rounded to f16 first (F16's `vec_dot_type` is F16; `from_float` is `ggml_cpu_fp32_to_fp16`,
//!   F16C round-to-nearest-even): `to_f16`;
//! - **one column** (a decode step, the logits, a one-token micro-batch): `llamafile_sgemm` refuses `n < 2`, so
//!   every element is `ggml_vec_dot_f16`: four 8-lane f32 accumulators fed by FMAs over 32-element steps, reduced
//!   `(a0 + a2) + (a1 + a3)`, low half + high half, two horizontal adds, then any tail (`n % 32`) added in double:
//!   `vec_dot`;
//! - **two columns or more** (a prompt's micro-batch): the activations are converted, then `llamafile_sgemm` runs
//!   `tinyBLAS<8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, float>` whenever the weight has a multiple of 4 rows and
//!   the row a multiple of 8 (otherwise it returns false and ggml falls back to `vec_dot`): each element is one
//!   8-lane FMA chain over the row, `hsum`'d as `((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))`: `sgemm_dot`.
//!   The tiling and the thread split decide only who computes an element, never its bits.
//!
//! Each kernel has a scalar model (the definition) and an AVX2 + FMA + F16C path with the same bits; the oracle is the
//! shipped library computing `mul_mat` graphs on the real weights (`testing/f16_oracle.py` → `oracle_ggml_b11192_f16`).

use crate::q1_0::{f16_to_f32, f32_to_f16};

/// Bytes per F16 weight.
pub const F16_BYTES: usize = 2;

/// `ggml_cpu_fp32_to_fp16`: every value rounded to f16, to nearest even (F16C `vcvtps2ph imm=0`).
pub fn to_f16(x: &[f32]) -> Vec<u16> {
    x.iter().map(|&v| f32_to_f16(v)).collect()
}

/// An F16 row's bytes, widened to f32 (exact): `get_rows` on an F16 table.
pub fn dequantize_row(row: &[u8], out: &mut [f32]) {
    for (o, b) in out.iter_mut().zip(row.as_chunks::<2>().0) {
        *o = f16_to_f32(u16::from_le_bytes(*b));
    }
}

#[inline(always)]
fn w(x: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([x[2 * i], x[2 * i + 1]]))
}

/// `ggml_vec_dot_f16` (the definition): `x` an F16 weight row (little-endian bytes), `y` the f16 activation.
pub fn vec_dot_ref(x: &[u8], y: &[u16]) -> f32 {
    let n = y.len();
    assert!(x.len() >= n * F16_BYTES);
    let np = n & !31;
    let mut acc = [[0.0f32; 8]; 4];
    for i in (0..np).step_by(32) {
        for (j, a) in acc.iter_mut().enumerate() {
            for (l, al) in a.iter_mut().enumerate() {
                let k = i + 8 * j + l;
                *al = w(x, k).mul_add(f16_to_f32(y[k]), *al);
            }
        }
    }
    let r: [f32; 8] = std::array::from_fn(|l| (acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l]));
    let t: [f32; 4] = std::array::from_fn(|l| r[l] + r[l + 4]);
    let mut sum = ((t[0] + t[1]) + (t[2] + t[3])) as f64;
    for (k, &yk) in y.iter().enumerate().skip(np) {
        sum += (w(x, k) * f16_to_f32(yk)) as f64;
    }
    sum as f32
}

/// One element of `tinyBLAS<8, __m256, …>` (the definition): one 8-lane FMA chain over the row, then `hsum`.
/// `y.len() % 8 == 0`.
pub fn sgemm_dot_ref(x: &[u8], y: &[u16]) -> f32 {
    let n = y.len();
    assert!(n.is_multiple_of(8) && x.len() >= n * F16_BYTES);
    let mut acc = [0.0f32; 8];
    for l in (0..n).step_by(8) {
        for (j, a) in acc.iter_mut().enumerate() {
            *a = w(x, l + j).mul_add(f16_to_f32(y[l + j]), *a);
        }
    }
    let h: [f32; 4] = std::array::from_fn(|i| acc[i + 4] + acc[i]);
    (h[0] + h[2]) + (h[1] + h[3])
}

/// `vec_dot_ref`'s bits on the fastest path this CPU has.
pub fn vec_dot(x: &[u8], y: &[u16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::q1_0::has_avx2() {
        assert!(x.len() >= y.len() * F16_BYTES);
        // SAFETY: AVX2 + FMA + F16C checked; lengths checked
        return unsafe { vec_dot_avx2(x, y) };
    }
    vec_dot_ref(x, y)
}

/// # Safety
/// AVX2, FMA and F16C are present and `x` holds at least `y.len()` f16 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn vec_dot_avx2(x: &[u8], y: &[u16]) -> f32 {
    use std::arch::x86_64::*;
    let n = y.len();
    let np = n & !31;
    let (xp, yp) = (x.as_ptr(), y.as_ptr() as *const u8);
    macro_rules! ld {
        ($p:expr, $i:expr) => {
            _mm256_cvtph_ps(_mm_loadu_si128($p.add(2 * $i) as *const __m128i))
        };
    }
    let mut s = [_mm256_setzero_ps(); 4];
    for i in (0..np).step_by(32) {
        for (j, sj) in s.iter_mut().enumerate() {
            *sj = _mm256_fmadd_ps(ld!(xp, i + 8 * j), ld!(yp, i + 8 * j), *sj);
        }
    }
    let r = _mm256_add_ps(_mm256_add_ps(s[0], s[2]), _mm256_add_ps(s[1], s[3]));
    let t0 = _mm_add_ps(_mm256_castps256_ps128(r), _mm256_extractf128_ps(r, 1));
    let t1 = _mm_hadd_ps(t0, t0);
    let mut sum = _mm_cvtss_f32(_mm_hadd_ps(t1, t1)) as f64;
    for (k, &yk) in y.iter().enumerate().skip(np) {
        sum += (w(x, k) * f16_to_f32(yk)) as f64;
    }
    sum as f32
}

/// `sgemm_dot_ref`'s bits for a block of four weight rows by `RN` columns (tinyBLAS's own register blocking: twelve
/// accumulators for 4 × 3). The columns are already widened to f32 (exact), so each step loads four weight vectors
/// and `RN` activation vectors; every element is still its own 8-lane FMA chain over the row, then `hsum`.
///
/// # Safety
/// AVX2, FMA and F16C are present; each row holds `k` f16 weights and each column `k` values, `k % 8 == 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn sgemm_block_avx2<const RN: usize>(x: [&[u8]; 4], cols: [&[f32]; RN], out: &mut [[f32; RN]; 4]) {
    use std::arch::x86_64::*;
    let k = cols[0].len();
    let mut c = [[_mm256_setzero_ps(); RN]; 4];
    let mut bv = [_mm256_setzero_ps(); RN];
    for l in (0..k).step_by(8) {
        for j in 0..RN {
            bv[j] = _mm256_loadu_ps(cols[j].as_ptr().add(l));
        }
        for i in 0..4 {
            let a = _mm256_cvtph_ps(_mm_loadu_si128(x[i].as_ptr().add(2 * l) as *const __m128i));
            for j in 0..RN {
                c[i][j] = _mm256_fmadd_ps(a, bv[j], c[i][j]);
            }
        }
    }
    for i in 0..4 {
        for j in 0..RN {
            let v = c[i][j];
            let h = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
            let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            out[i][j] = _mm_cvtss_f32(_mm_add_ss(h, _mm_movehdup_ps(h)));
        }
    }
}

/// `vec_dot` with the weight row as f16 values (a K cache row): the same bits.
pub fn vec_dot_u16(x: &[u16], y: &[u16]) -> f32 {
    vec_dot(as_bytes(x), y)
}

/// An f16 slice's bytes (little-endian, as GGUF and the KV cache hold them).
fn as_bytes(x: &[u16]) -> &[u8] {
    #[cfg(not(target_endian = "little"))]
    compile_error!("bankML reads f16 values as little-endian bytes");
    // SAFETY: u8 has no alignment and every byte of an initialised u16 is initialised
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x)) }
}

/// f16 values widened to f32 (exact; F16C where the CPU has it).
pub fn widen(src: &[u16], dst: &mut [f32]) {
    let mut i0 = 0;
    #[cfg(target_arch = "x86_64")]
    if crate::q1_0::has_avx2() {
        i0 = src.len().min(dst.len()) & !7;
        // SAFETY: F16C checked; i0 ≤ both lengths
        unsafe { widen_f16c(&src[..i0], &mut dst[..i0]) };
    }
    for (d, &s) in dst[i0..].iter_mut().zip(&src[i0..]) {
        *d = f16_to_f32(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn widen_f16c(src: &[u16], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    for (d, s) in dst.as_chunks_mut::<8>().0.iter_mut().zip(src.as_chunks::<8>().0) {
        _mm256_storeu_ps(d.as_mut_ptr(), _mm256_cvtph_ps(_mm_loadu_si128(s.as_ptr() as *const __m128i)));
    }
}

/// `ggml_vec_mad_f16` (flash attention's V accumulator): `acc[i] = f16(fma(v[i], w, acc[i]))`, each element alone.
pub fn mad(acc: &mut [u16], v: &[u16], w: f32) {
    assert!(v.len() >= acc.len());
    let mut i0 = 0;
    #[cfg(target_arch = "x86_64")]
    if crate::q1_0::has_avx2() {
        i0 = acc.len() & !7;
        // SAFETY: AVX2 + F16C checked; i0 ≤ both lengths
        unsafe { mad_avx2(&mut acc[..i0], &v[..i0], w) };
    }
    for (a, &vv) in acc[i0..].iter_mut().zip(&v[i0..]) {
        *a = f32_to_f16(f16_to_f32(vv).mul_add(w, f16_to_f32(*a)));
    }
}

/// `ggml_vec_scale_f16`: `acc[i] = f16(acc[i] · s)`.
pub fn scale(acc: &mut [u16], s: f32) {
    let mut i0 = 0;
    #[cfg(target_arch = "x86_64")]
    if crate::q1_0::has_avx2() {
        i0 = acc.len() & !7;
        // SAFETY: AVX2 + F16C checked
        unsafe { scale_avx2(&mut acc[..i0], s) };
    }
    for a in acc[i0..].iter_mut() {
        *a = f32_to_f16(f16_to_f32(*a) * s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn mad_avx2(acc: &mut [u16], v: &[u16], w: f32) {
    use std::arch::x86_64::*;
    let wv = _mm256_set1_ps(w);
    for (a, vv) in acc.as_chunks_mut::<8>().0.iter_mut().zip(v.as_chunks::<8>().0) {
        let ap = a.as_mut_ptr() as *mut __m128i;
        let r = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(vv.as_ptr() as *const __m128i)), wv, _mm256_cvtph_ps(_mm_loadu_si128(ap)));
        _mm_storeu_si128(ap, _mm256_cvtps_ph::<0>(r));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn scale_avx2(acc: &mut [u16], s: f32) {
    use std::arch::x86_64::*;
    let sv = _mm256_set1_ps(s);
    for a in acc.as_chunks_mut::<8>().0.iter_mut() {
        let ap = a.as_mut_ptr() as *mut __m128i;
        _mm_storeu_si128(ap, _mm256_cvtps_ph::<0>(_mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(ap)), sv)));
    }
}

/// Whether ggml's `llamafile_sgemm` takes an F16 product of `rows` × `k` weights by `n` columns (else `vec_dot`).
pub fn takes_sgemm(rows: usize, k: usize, n: usize) -> bool {
    n >= 2 && rows.is_multiple_of(4) && k.is_multiple_of(8)
}

/// `out[r] = row r of w · a` (one column: `vec_dot`), rows in chunks on the pool.
pub fn mat_vec_par(pool: &crate::par::Pool, wb: &[u8], rows: usize, a: &[u16], out: &mut [f32]) {
    let rb = a.len() * F16_BYTES;
    assert!(wb.len() >= rows * rb && out.len() >= rows);
    pool.rows(rows, out, &|r0, o| {
        for (i, v) in o.iter_mut().enumerate() {
            *v = vec_dot(&wb[(r0 + i) * rb..(r0 + i + 1) * rb], a);
        }
    });
}

/// Several matrices by one column in a single pass of the pool (Q, K and V; gate and up): each output row is still
/// one `vec_dot`, so the bits are those of `mat_vec_par` on each matrix.
pub fn mat_vec_multi_par(pool: &crate::par::Pool, ws: &[(&[u8], usize)], a: &[u16], outs: &mut [&mut [f32]]) {
    let rb = a.len() * F16_BYTES;
    assert!(ws.len() == outs.len() && ws.iter().zip(outs.iter()).all(|((w, r), o)| w.len() >= r * rb && o.len() >= *r));
    let total: usize = ws.iter().map(|w| w.1).sum();
    let bases: Vec<usize> = outs.iter_mut().map(|o| o.as_mut_ptr() as usize).collect();
    let (next, step) = (std::sync::atomic::AtomicUsize::new(0), crate::par::CHUNK_ROWS);
    pool.run(&|_| loop {
        let g0 = next.fetch_add(step, std::sync::atomic::Ordering::Relaxed);
        if g0 >= total {
            break;
        }
        for g in g0..(g0 + step).min(total) {
            let (mut m, mut r) = (0, g);
            while r >= ws[m].1 {
                r -= ws[m].1;
                m += 1;
            }
            // SAFETY: global row g is claimed by this worker alone; r < rows of matrix m, inside its output
            unsafe { *(bases[m] as *mut f32).add(r) = vec_dot(&ws[m].0[r * rb..(r + 1) * rb], a) };
        }
    });
}

/// `out[c·rows + r] = row r of w · column c`, as ggml computes the product for these shapes: `sgemm_dot` when
/// `takes_sgemm`, else `vec_dot` per element. Rows in chunks on the pool; the same bits at any thread count.
pub fn mat_mul_par(pool: &crate::par::Pool, wb: &[u8], rows: usize, cols: &[Vec<u16>], out: &mut [f32]) {
    let Some(k) = cols.first().map(Vec::len) else { return };
    let rb = k * F16_BYTES;
    assert!(cols.iter().all(|c| c.len() == k) && wb.len() >= rows * rb && out.len() >= rows * cols.len());
    let sgemm = takes_sgemm(rows, k, cols.len());
    #[cfg(target_arch = "x86_64")]
    let fast = sgemm && crate::q1_0::has_avx2();
    #[cfg(not(target_arch = "x86_64"))]
    let fast = false;
    // the tiles read the activations as f32 (f16 → f32 is exact): widened once here, not once per weight row
    let wide: Vec<Vec<f32>> = if fast {
        cols.iter().map(|c| {
            let mut w = vec![0.0f32; k];
            widen(c, &mut w);
            w
        }).collect()
    } else {
        Vec::new()
    };
    let (next, base) = (std::sync::atomic::AtomicUsize::new(0), out.as_mut_ptr() as usize);
    let step = crate::par::CHUNK_ROWS; // a multiple of 4: with rows % 4 == 0 every chunk is whole 4-row blocks
    let n = cols.len();
    pool.run(&|_| loop {
        let r0 = next.fetch_add(step, std::sync::atomic::Ordering::Relaxed);
        if r0 >= rows {
            break;
        }
        let o = base as *mut f32;
        let r1 = (r0 + step).min(rows);
        #[cfg(target_arch = "x86_64")]
        if fast {
            for r in (r0..r1).step_by(4) {
                let x: [&[u8]; 4] = [&wb[r * rb..], &wb[(r + 1) * rb..], &wb[(r + 2) * rb..], &wb[(r + 3) * rb..]];
                let mut c0 = 0;
                while c0 < n {
                    let w = (n - c0).min(3);
                    // SAFETY: AVX2 + FMA + F16C checked; four whole rows (rows % 4 == 0) and w columns of k values (k % 8 == 0)
                    unsafe {
                        let put = |i: usize, j: usize, v: f32| *o.add((c0 + j) * rows + r + i) = v;
                        match w {
                            3 => {
                                let mut t = [[0.0f32; 3]; 4];
                                sgemm_block_avx2(x, [&wide[c0], &wide[c0 + 1], &wide[c0 + 2]], &mut t);
                                t.iter().enumerate().for_each(|(i, tr)| tr.iter().enumerate().for_each(|(j, &v)| put(i, j, v)));
                            }
                            2 => {
                                let mut t = [[0.0f32; 2]; 4];
                                sgemm_block_avx2(x, [&wide[c0], &wide[c0 + 1]], &mut t);
                                t.iter().enumerate().for_each(|(i, tr)| tr.iter().enumerate().for_each(|(j, &v)| put(i, j, v)));
                            }
                            _ => {
                                let mut t = [[0.0f32; 1]; 4];
                                sgemm_block_avx2(x, [&wide[c0]], &mut t);
                                t.iter().enumerate().for_each(|(i, tr)| put(i, 0, tr[0]));
                            }
                        }
                    }
                    c0 += w;
                }
            }
            continue;
        }
        for r in r0..r1 {
            let x = &wb[r * rb..(r + 1) * rb];
            for (c, col) in cols.iter().enumerate() {
                let v = if sgemm { sgemm_dot_ref(x, col) } else { vec_dot(x, col) };
                // SAFETY: row r is claimed by this worker alone; c < n
                unsafe { *o.add(c * rows + r) = v };
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q1_0::tests::Rng;

    fn row(rng: &mut Rng, n: usize) -> Vec<u8> {
        (0..n).flat_map(|_| f32_to_f16(rng.act() * 0.2).to_le_bytes()).collect()
    }

    /// The AVX2 paths equal the scalar models on random rows of many lengths (tails included), every bit.
    #[test]
    fn avx2_equals_the_scalar_models() {
        let mut rng = Rng(0xf16);
        let pool = crate::par::Pool::new(3);
        for &k in &[8usize, 32, 40, 64, 72, 576, 1536, 1000, 37] {
            let rows = 13;
            let wb: Vec<u8> = (0..rows).flat_map(|_| row(&mut rng, k)).collect();
            let cols: Vec<Vec<u16>> = (0..6).map(|_| to_f16(&(0..k).map(|_| rng.act()).collect::<Vec<_>>())).collect();
            for r in 0..rows {
                let x = &wb[r * k * 2..(r + 1) * k * 2];
                assert_eq!(vec_dot(x, &cols[0]).to_bits(), vec_dot_ref(x, &cols[0]).to_bits(), "vec_dot k {k}");
            }
            for n in 1..=6 {
                for rows in [12usize, 13] {
                    let mut out = vec![0.0f32; rows * n];
                    mat_mul_par(&pool, &wb, rows, &cols[..n], &mut out);
                    for c in 0..n {
                        for r in 0..rows {
                            let x = &wb[r * k * 2..(r + 1) * k * 2];
                            let want = if takes_sgemm(rows, k, n) { sgemm_dot_ref(x, &cols[c]) } else { vec_dot_ref(x, &cols[c]) };
                            assert_eq!(out[c * rows + r].to_bits(), want.to_bits(), "k {k} rows {rows} n {n}");
                        }
                    }
                }
            }
        }
    }

    /// The attention helpers' AVX2 paths equal their scalar definitions, every element (odd tails included).
    #[test]
    fn mad_and_scale_match_the_scalar_definitions() {
        let mut rng = Rng(0xa77);
        for n in [8usize, 64, 128, 13] {
            let v = to_f16(&(0..n).map(|_| rng.act()).collect::<Vec<_>>());
            let mut a = to_f16(&(0..n).map(|_| rng.act()).collect::<Vec<_>>());
            let mut b = a.clone();
            for w in [0.0f32, 1.0, 0.3, -2.5e-3, 7.0e4] {
                mad(&mut a, &v, w);
                for (x, &vv) in b.iter_mut().zip(&v) {
                    *x = f32_to_f16(f16_to_f32(vv).mul_add(w, f16_to_f32(*x)));
                }
                assert_eq!(a, b);
                scale(&mut a, w * 0.5);
                for x in b.iter_mut() {
                    *x = f32_to_f16(f16_to_f32(*x) * (w * 0.5));
                }
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn the_two_reductions_differ() {
        // the same row and column through ggml's two paths: the bits can differ, which is why the shape matters
        let mut rng = Rng(7);
        let (mut differ, k) = (0, 576);
        for _ in 0..200 {
            let x = row(&mut rng, k);
            let y = to_f16(&(0..k).map(|_| rng.act()).collect::<Vec<_>>());
            differ += (vec_dot_ref(&x, &y).to_bits() != sgemm_dot_ref(&x, &y).to_bits()) as usize;
        }
        assert!(differ > 0);
    }

    /// The shipped ggml b11192's `mul_mat` on F16 weights (testing/f16_oracle.py): every tensor widened to f32 with
    /// ggml's own conversion, and products of real weight matrices (and synthetic shapes for the tails and the
    /// fallback) by 1–28 columns, every element bit for bit.
    #[test]
    #[ignore = "needs .models/SmolLM2-135M-Instruct-F16.gguf + .models/oracle-f16 (testing/f16_oracle.py); --release"]
    fn oracle_ggml_b11192_f16() {
        use crate::sha256::{hex, Sha256};
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let path = dir.join("SmolLM2-135M-Instruct-F16.gguf");
        let rep = crate::gguf::guard_file(&path, crate::gguf::Engine::Mainline).unwrap();
        let h = rep.header.unwrap();
        let mm = crate::gguf::Mmap::open(&path).unwrap();
        let data = |name: &str| {
            let t = h.tensors.iter().find(|t| t.name == name).unwrap();
            let n = t.dims.iter().product::<u64>() as usize;
            (&mm.bytes()[(h.data_start + t.offset) as usize..][..n * 2], t.dims[0] as usize, t.dims[1] as usize)
        };
        let (mut nt, mut nt_ok) = (0, 0);
        for line in std::fs::read_to_string(dir.join("oracle-f16/dequant.tsv")).unwrap().lines() {
            let f: Vec<&str> = line.split('\t').collect();
            let (b, k, r) = data(f[0]);
            let mut out = vec![0.0f32; k * r];
            dequantize_row(b, &mut out);
            let mut s = Sha256::default();
            out.iter().for_each(|v| s.update(&v.to_le_bytes()));
            nt += 1;
            nt_ok += (hex(&s.finish()) == f[1]) as usize;
        }
        let bin = std::fs::read(dir.join("oracle-f16/matmul.bin")).unwrap();
        let mut p = 0;
        let u32_ = |b: &[u8], p: &mut usize| {
            let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
            *p += 4;
            v as usize
        };
        let pool = crate::par::Pool::new(3);
        let (mut cases, mut elems, mut ok, mut sg, mut vd) = (0, 0, 0, 0, 0);
        while p < bin.len() {
            let nl = u32_(&bin, &mut p);
            let name = String::from_utf8(bin[p..p + nl].to_vec()).unwrap();
            p += nl;
            let (m, k, n) = (u32_(&bin, &mut p), u32_(&bin, &mut p), u32_(&bin, &mut p));
            let synth;
            let wb: &[u8] = if name == "synthetic" {
                synth = bin[p..p + m * k * 2].to_vec();
                p += m * k * 2;
                &synth
            } else {
                let (b, kk, _) = data(&name);
                assert_eq!(kk, k);
                &b[..m * k * 2]
            };
            let rd = |p: &mut usize, cnt: usize| -> Vec<f32> {
                let v = bin[*p..*p + cnt * 4].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                *p += cnt * 4;
                v
            };
            let x = rd(&mut p, n * k);
            let want = rd(&mut p, n * m);
            let cols: Vec<Vec<u16>> = x.chunks_exact(k).map(to_f16).collect();
            let mut out = vec![0.0f32; m * n];
            if n == 1 {
                mat_vec_par(&pool, wb, m, &cols[0], &mut out);
            } else {
                mat_mul_par(&pool, wb, m, &cols, &mut out);
            }
            let good = out.iter().zip(&want).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            cases += 1;
            elems += want.len();
            ok += good;
            if takes_sgemm(m, k, n) { sg += 1 } else { vd += 1 }
            if good != want.len() {
                eprintln!("  {name} {m}×{k} by {n}: {good} of {} elements equal", want.len());
            }
        }
        eprintln!("f16 oracle: {nt_ok} of {nt} F16 tensors widened bit-exact; {ok} of {elems} mul_mat elements bit-exact against the shipped ggml b11192 \
                   ({cases} products: {sg} through llamafile_sgemm's tinyBLAS, {vd} through ggml_vec_dot_f16)");
        assert_eq!((nt_ok, ok), (nt, elems));
    }
}


