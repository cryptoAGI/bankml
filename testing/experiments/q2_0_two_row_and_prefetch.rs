// SPDX-License-Identifier: MIT OR Apache-2.0
//! 0.0.5 negative results, kept for reproduction (not compiled into the crate). Paste into q2_0.rs (kernels
//! above `mat_vec`, tests into `mod tests`) to re-run. Both were bit-exact against the scalar model of ggml.
//!
//! - Two-row decode tile (the activation quad loaded once for two rows): 0.795× — ternary decode GEMV
//!   12288×4096, 1 thread, 3.54/3.66 ns/block (mat_vec) against 4.54/4.61 (tile). Register pressure costs more
//!   than the shared loads save.
//! - Software prefetch of the weight row, 256/512/1024 bytes ahead: within ±3 % at 1 and 3 threads on the real
//!   ffn_gate and output tensors; the hardware prefetcher already covers a sequential stream.
//!
//! What 0.0.5 did keep: `Q2Packed` (repacked quads, 1.00–1.075×). Laptop: Ryzen 3 3200U (Zen+).

/// Two Q2_0 rows against one prepared activation (0.0.5 experiment): the activation quads are loaded once
/// and used by both rows; each row keeps its own serial `sumf` in ggml's block order, so each output has the
/// bits of its own `vec_dot_act`.
///
/// # Safety
/// AVX2+FMA+F16C present; `x0` and `x1` each hold `a.n / 64` blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_act_2x1_avx2(x0: &[u8], x1: &[u8], a: &Q8Act2) -> [f32; 2] {
    use std::arch::x86_64::*;
    let nb = a.n / QK2_0;
    let (p0, p1, qp) = (x0.as_ptr(), x1.as_ptr(), a.qs().as_ptr());
    let (mut s0, mut s1) = (_mm_setzero_ps(), _mm_setzero_ps());
    let ones = _mm256_set1_epi16(1);
    for g in 0..nb / 4 {
        let q = qp.add(g * 256);
        let ql: [__m256i; 8] = std::array::from_fn(|r| _mm256_loadu_si256(q.add(32 * r) as *const __m256i));
        let dot = |c: [__m256i; 4], o: usize| {
            let m = |r: usize| _mm256_maddubs_epi16(c[r], ql[o + r]);
            _mm256_madd_epi16(_mm256_add_epi16(_mm256_add_epi16(m(0), m(1)), _mm256_add_epi16(m(2), m(3))), ones)
        };
        let (xa, xb) = (p0.add(4 * g * Q2_0_BYTES), p1.add(4 * g * Q2_0_BYTES));
        let ab0 = dot(avx2::codes2(xa), 0);
        let cd0 = dot(avx2::codes2(xa.add(2 * Q2_0_BYTES)), 4);
        let ab1 = dot(avx2::codes2(xb), 0);
        let cd1 = dot(avx2::codes2(xb.add(2 * Q2_0_BYTES)), 4);
        s0 = avx2::finish4(ab0, cd0, a, g, xa, s0);
        s1 = avx2::finish4(ab1, cd1, a, g, xb, s1);
    }
    for i in nb / 4 * 4..nb {
        s0 = avx2::block1(p0.add(i * Q2_0_BYTES), a, i, s0);
        s1 = avx2::block1(p1.add(i * Q2_0_BYTES), a, i, s1);
    }
    [_mm_cvtss_f32(s0), _mm_cvtss_f32(s1)]
}

/// `vec_dot_act_avx2` with a software prefetch `PF` bytes ahead in the weight row (0.0.5 experiment 2).
/// The arithmetic is unchanged; only the prefetch is added.
///
/// # Safety
/// As `vec_dot_act_avx2`. (A prefetch past the end of the slice is a hint and cannot fault.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_act_pf_avx2<const PF: usize>(x: &[u8], a: &Q8Act2) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.n / QK2_0;
    let (xp, qp) = (x.as_ptr(), a.qs().as_ptr());
    let mut sumf = _mm_setzero_ps();
    for g in 0..nb / 4 {
        let xb = xp.add(4 * g * Q2_0_BYTES);
        _mm_prefetch::<_MM_HINT_T0>(xb.wrapping_add(PF) as *const i8);
        let ab = avx2::dot2(&avx2::codes2(xb), qp.add(g * 256));
        let cd = avx2::dot2(&avx2::codes2(xb.add(2 * Q2_0_BYTES)), qp.add(g * 256 + 128));
        sumf = avx2::finish4(ab, cd, a, g, xb, sumf);
    }
    for i in nb / 4 * 4..nb {
        sumf = avx2::block1(xp.add(i * Q2_0_BYTES), a, i, sumf);
    }
    _mm_cvtss_f32(sumf)
}


// ---- tests

    /// 0.0.5 experiment: the 2-row decode tile, bits first, then speed against `mat_vec` on 12288×4096.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn two_row_tile_bit_exact() {
        if !crate::q1_0::has_avx2() {
            return;
        }
        let mut r = Rng(5050);
        for round in 0..1500 {
            let nb = 1 + (r.next() % 70) as usize;
            let (x0, x1) = (random_q2(&mut r, nb), random_q2(&mut r, nb));
            let y = random_q8(&mut r, nb, round % 2 == 0);
            let a = Q8Act2::from_q8_0(nb * 64, &y);
            let v = unsafe { vec_dot_act_2x1_avx2(&x0, &x1, &a) };
            assert_eq!(v.map(f32::to_bits), [vec_dot_ref(nb * 64, &x0, &y).to_bits(), vec_dot_ref(nb * 64, &x1, &y).to_bits()], "round {round} nb {nb}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn prefetch_variant_bit_exact() {
        if !crate::q1_0::has_avx2() {
            return;
        }
        let mut r = Rng(6060);
        for _ in 0..500 {
            let nb = 1 + (r.next() % 70) as usize;
            let x = random_q2(&mut r, nb);
            let y = random_q8(&mut r, nb, true);
            let a = Q8Act2::from_q8_0(nb * 64, &y);
            let want = vec_dot_ref(nb * 64, &x, &y).to_bits();
            assert_eq!(unsafe { vec_dot_act_pf_avx2::<512>(&x, &a) }.to_bits(), want);
        }
    }

    /// Prefetch distances against `mat_vec_par` on the real ffn_gate (12288×4096) and `output` (151669×4096),
    /// at 1 and 3 threads, alternating. `--ignored bench_q2_0_prefetch --nocapture --test-threads=1`
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "needs .models/Ternary-Bonsai-8B-Q2_0_g64.gguf; benchmark"]
    fn bench_q2_0_prefetch() {
        use crate::gguf::{guard_file, Engine, Mmap};
        use crate::par::Pool;
        use std::time::Instant;
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/Ternary-Bonsai-8B-Q2_0_g64.gguf");
        let rep = guard_file(&model, Engine::Mainline).unwrap();
        let h = rep.header.as_ref().unwrap();
        let mm = Mmap::open(&model).unwrap();
        let mut r = Rng(6061);
        for name in ["blk.0.ffn_gate.weight", "output.weight"] {
            let t = h.tensors.iter().find(|t| t.name == name).unwrap();
            let (n, rows) = (t.dims[0] as usize, t.dims[1] as usize);
            let w = mm.tensor(h, t).unwrap();
            let (_, a) = column(&mut r, n);
            let rb = n / QK2_0 * Q2_0_BYTES;
            let blocks = (rows * n / QK2_0) as f64;
            for th in [1, 3] {
                let pool = Pool::new(th);
                let mut out = vec![0f32; rows];
                let run = |pf: usize, out: &mut [f32]| {
                    pool.rows(rows, out, &|r0, o| {
                        for (i, v) in o.iter_mut().enumerate() {
                            let x = &w[(r0 + i) * rb..];
                            *v = unsafe {
                                match pf {
                                    256 => vec_dot_act_pf_avx2::<256>(x, &a),
                                    512 => vec_dot_act_pf_avx2::<512>(x, &a),
                                    1024 => vec_dot_act_pf_avx2::<1024>(x, &a),
                                    _ => vec_dot_act_avx2(x, &a),
                                }
                            };
                        }
                    })
                };
                run(0, &mut out);
                let mut ts: Vec<Vec<f64>> = vec![vec![]; 4];
                for _ in 0..9 {
                    for (k, pf) in [0usize, 256, 512, 1024].into_iter().enumerate() {
                        let t0 = Instant::now();
                        run(pf, &mut out);
                        ts[k].push(t0.elapsed().as_nanos() as f64 / blocks);
                    }
                }
                let m: Vec<(f64, f64)> = ts.iter_mut().map(|v| stat(v)).collect();
                eprintln!("{name:22} {th} thread(s) ns/block min/median: none {:.2}/{:.2} | pf256 {:.2}/{:.2} | pf512 {:.2}/{:.2} | pf1024 {:.2}/{:.2}",
                    m[0].0, m[0].1, m[1].0, m[1].1, m[2].0, m[2].1, m[3].0, m[3].1);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "benchmark; run with --release --nocapture"]
    fn bench_q2_0_two_row() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut r = Rng(5051);
        let (rows, n) = (12288usize, 4096usize);
        let rb = n / QK2_0 * Q2_0_BYTES;
        let w = random_q2(&mut r, rows * n / QK2_0);
        let a = Q8Act2::from_q8_0(n, &random_q8(&mut r, n / QK2_0, false));
        let blocks = (rows * n / QK2_0) as f64;
        let mut out = vec![0f32; rows];
        let (mut t1, mut t2) = (vec![], vec![]);
        for _ in 0..15 {
            let t = Instant::now();
            mat_vec(&w, rows, &a, &mut out);
            t1.push(t.elapsed().as_nanos() as f64 / blocks);
            black_box(&out);
            let t = Instant::now();
            for i in (0..rows).step_by(2) {
                let v = unsafe { vec_dot_act_2x1_avx2(&w[i * rb..], &w[(i + 1) * rb..], &a) };
                out[i] = v[0];
                out[i + 1] = v[1];
            }
            t2.push(t.elapsed().as_nanos() as f64 / blocks);
            black_box(&out);
        }
        let (a1, a2) = (stat(&mut t1), stat(&mut t2));
        eprintln!("ternary decode GEMV 12288x4096 ns/block min/median: mat_vec {:.2}/{:.2} | 2-row tile {:.2}/{:.2} ({:.3}x)", a1.0, a1.1, a2.0, a2.1, a1.1 / a2.1);
    }

