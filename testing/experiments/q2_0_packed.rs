//! 0.0.5 — measured and not adopted: `Q2Packed`, the ternary weights repacked once (per quad: 64 contiguous
//! code bytes, then the four f16 scales; the same 72 bytes, arithmetic unchanged). Bit-exact against the scalar
//! model of ggml and against the file-layout kernel on the real Ternary-Bonsai-8B tensors — and not reliably
//! faster: 1.00–1.075× in the first run, 0.90–1.05× in the release gate (`testing/results/0.0.5.txt`), on the
//! laptop (Ryzen 3 3200U). Within noise, so by the gate's rule it does not ship. Paste into q2_0.rs (code above
//! `mat_vec`, tests into `mod tests`) to re-run on another core.

/// A Q2_0 matrix repacked once for the AVX2 kernel (0.0.5): each quad of blocks (A, B, C, D) becomes
/// `codes A | codes B | codes C | codes D | d_A d_B d_C d_D` — the 64 code bytes contiguous, so A|B and C|D are
/// one 256-bit load each, and the four f16 scales adjacent, so one 64-bit load and one `vcvtph2ps` give all four.
/// Same 72 bytes per quad as the file (18 per block), only moved; tail blocks (nb % 4) keep the file layout.
/// No arithmetic changes, so the bits are the file's (checked against `vec_dot_ref` on the unpacked rows).
pub struct Q2Packed {
    pub rows: usize,
    pub n: usize,
    data: Vec<u8>,
}

impl Q2Packed {
    /// Repack `rows` rows of `n` elements from the GGUF layout.
    pub fn pack(raw: &[u8], rows: usize, n: usize) -> Self {
        assert_eq!(n % QK2_0, 0);
        let nb = n / QK2_0;
        let rb = nb * Q2_0_BYTES;
        assert!(raw.len() >= rows * rb);
        let mut data = vec![0u8; rows * rb];
        for (src, dst) in raw[..rows * rb].chunks_exact(rb).zip(data.chunks_exact_mut(rb)) {
            for g in 0..nb / 4 {
                let (s, d) = (&src[g * 72..g * 72 + 72], &mut dst[g * 72..g * 72 + 72]);
                for b in 0..4 {
                    d[16 * b..16 * b + 16].copy_from_slice(&s[b * 18 + 2..b * 18 + 18]);
                    d[64 + 2 * b..66 + 2 * b].copy_from_slice(&s[b * 18..b * 18 + 2]);
                }
            }
            let t = nb / 4 * 72;
            dst[t..].copy_from_slice(&src[t..]);
        }
        Q2Packed { rows, n, data }
    }

    fn row(&self, r: usize) -> &[u8] {
        let rb = self.n / QK2_0 * Q2_0_BYTES;
        &self.data[r * rb..(r + 1) * rb]
    }
}

/// One packed row · prepared activation; the float ops of `vec_dot_act_avx2`, in the same order.
///
/// # Safety
/// AVX2+FMA+F16C present; `x` is a row of a `Q2Packed` with `a.n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_packed_avx2(x: &[u8], a: &Q8Act2) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.n / QK2_0;
    let (xp, qp) = (x.as_ptr(), a.qs().as_ptr());
    let m3 = _mm256_set1_epi8(3);
    let codes = |v: __m256i| {
        [
            _mm256_and_si256(v, m3),
            _mm256_and_si256(_mm256_srli_epi16(v, 2), m3),
            _mm256_and_si256(_mm256_srli_epi16(v, 4), m3),
            _mm256_and_si256(_mm256_srli_epi16(v, 6), m3),
        ]
    };
    let mut sumf = _mm_setzero_ps();
    for g in 0..nb / 4 {
        let xq = xp.add(g * 72);
        let ab = avx2::dot2(&codes(_mm256_loadu_si256(xq as *const __m256i)), qp.add(g * 256));
        let cd = avx2::dot2(&codes(_mm256_loadu_si256(xq.add(32) as *const __m256i)), qp.add(g * 256 + 128));
        let s = _mm256_sub_epi32(_mm256_hadd_epi32(ab, cd), _mm256_loadu_si256(a.sum.as_ptr().add(8 * g) as *const __m256i));
        let (s, d1) = (_mm256_cvtepi32_ps(s), _mm256_loadu_ps(a.d.as_ptr().add(8 * g)));
        let u = _mm256_fmadd_ps(d1, s, _mm256_moveldup_ps(_mm256_fmadd_ps(d1, s, _mm256_setzero_ps())));
        let (lo, hi) = (_mm256_castps256_ps128(u), _mm256_extractf128_ps(u, 1));
        let d0 = _mm_cvtph_ps(_mm_loadl_epi64(xq.add(64) as *const __m128i)); // [dA dB dC dD]
        sumf = _mm_fmadd_ss(d0, _mm_movehdup_ps(lo), sumf);
        sumf = _mm_fmadd_ss(_mm_movehdup_ps(d0), _mm_movehdup_ps(hi), sumf);
        sumf = _mm_fmadd_ss(_mm_movehl_ps(d0, d0), _mm_shuffle_ps(lo, lo, 3), sumf);
        sumf = _mm_fmadd_ss(_mm_shuffle_ps(d0, d0, 3), _mm_shuffle_ps(hi, hi, 3), sumf);
    }
    for i in nb / 4 * 4..nb {
        sumf = avx2::block1(xp.add(i * Q2_0_BYTES), a, i, sumf);
    }
    _mm_cvtss_f32(sumf)
}

/// out[r] = packed row r · `a`; the bits of `vec_dot_ref` on the unpacked row.
pub fn mat_vec_packed(w: &Q2Packed, a: &Q8Act2, out: &mut [f32]) {
    assert!(w.n == a.n && out.len() >= w.rows);
    #[cfg(target_arch = "x86_64")]
    if crate::q1_0::has_avx2() {
        for (r, o) in out[..w.rows].iter_mut().enumerate() {
            *o = unsafe { vec_dot_packed_avx2(w.row(r), a) };
        }
        return;
    }
    let rb = w.n / QK2_0 * Q2_0_BYTES;
    let mut raw = vec![0u8; rb];
    for (r, o) in out[..w.rows].iter_mut().enumerate() {
        unpack_row(w.row(r), &mut raw);
        *o = vec_dot_act_scalar(&raw, a);
    }
}

/// `mat_vec_packed` on every thread of `pool`; the same bits at any thread count.
pub fn mat_vec_packed_par(pool: &crate::par::Pool, w: &Q2Packed, a: &Q8Act2, out: &mut [f32]) {
    assert!(w.n == a.n && out.len() >= w.rows);
    pool.rows(w.rows, out, &|r0, o| {
        #[cfg(target_arch = "x86_64")]
        if crate::q1_0::has_avx2() {
            for (i, v) in o.iter_mut().enumerate() {
                *v = unsafe { vec_dot_packed_avx2(w.row(r0 + i), a) };
            }
            return;
        }
        let mut raw = vec![0u8; w.n / QK2_0 * Q2_0_BYTES];
        for (i, v) in o.iter_mut().enumerate() {
            unpack_row(w.row(r0 + i), &mut raw);
            *v = vec_dot_act_scalar(&raw, a);
        }
    });
}

/// The file layout of one packed row (the portable path, and tests).
pub fn unpack_row(x: &[u8], out: &mut [u8]) {
    let nb = out.len() / Q2_0_BYTES;
    for g in 0..nb / 4 {
        let (s, d) = (&x[g * 72..g * 72 + 72], &mut out[g * 72..g * 72 + 72]);
        for b in 0..4 {
            d[b * 18..b * 18 + 2].copy_from_slice(&s[64 + 2 * b..66 + 2 * b]);
            d[b * 18 + 2..b * 18 + 18].copy_from_slice(&s[16 * b..16 * b + 16]);
        }
    }
    let t = nb / 4 * 72;
    let len = out.len();
    out[t..].copy_from_slice(&x[t..len]);
}


// ---- tests

    #[test]
    fn packed_bit_exact_with_ref() {
        let mut r = Rng(7070);
        for round in 0..400 {
            let (rows, nb) = (1 + (r.next() % 9) as usize, 1 + (r.next() % 70) as usize);
            let w = random_q2(&mut r, rows * nb);
            let y = random_q8(&mut r, nb, round % 2 == 0);
            let a = Q8Act2::from_q8_0(nb * 64, &y);
            let p = Q2Packed::pack(&w, rows, nb * 64);
            let mut raw = vec![0u8; nb * Q2_0_BYTES];
            let (mut o1, mut o3) = (vec![0f32; rows], vec![0f32; rows]);
            mat_vec_packed(&p, &a, &mut o1);
            mat_vec_packed_par(&crate::par::Pool::new(3), &p, &a, &mut o3);
            for i in 0..rows {
                unpack_row(p.row(i), &mut raw);
                assert_eq!(raw, w[i * nb * Q2_0_BYTES..(i + 1) * nb * Q2_0_BYTES], "unpack round {round}");
                let want = vec_dot_ref(nb * 64, &w[i * nb * Q2_0_BYTES..], &y).to_bits();
                assert_eq!((o1[i].to_bits(), o3[i].to_bits()), (want, want), "round {round} rows {rows} nb {nb} row {i}");
            }
        }
    }

    /// Packed vs file layout on the real ffn_gate and output tensors, 1 and 3 threads, alternating, bits compared.
    #[test]
    #[ignore = "needs .models/Ternary-Bonsai-8B-Q2_0_g64.gguf; benchmark"]
    fn bench_q2_0_packed() {
        use crate::gguf::{guard_file, Engine, Mmap};
        use crate::par::Pool;
        use std::time::Instant;
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/Ternary-Bonsai-8B-Q2_0_g64.gguf");
        let rep = guard_file(&model, Engine::Mainline).unwrap();
        let h = rep.header.as_ref().unwrap();
        let mm = Mmap::open(&model).unwrap();
        let mut r = Rng(7071);
        for name in ["blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "output.weight"] {
            let t = h.tensors.iter().find(|t| t.name == name).unwrap();
            let (n, rows) = (t.dims[0] as usize, t.dims[1] as usize);
            let w = mm.tensor(h, t).unwrap();
            let t0 = Instant::now();
            let p = Q2Packed::pack(w, rows, n);
            let pack_ms = t0.elapsed().as_secs_f64() * 1e3;
            let (_, a) = column(&mut r, n);
            let blocks = (rows * n / QK2_0) as f64;
            for th in [1, 3] {
                let pool = Pool::new(th);
                let (mut o1, mut o2) = (vec![0f32; rows], vec![0f32; rows]);
                let (mut t1, mut t2) = (vec![], vec![]);
                for _ in 0..9 {
                    let t0 = Instant::now();
                    mat_vec_par(&pool, w, rows, &a, &mut o1);
                    t1.push(t0.elapsed().as_nanos() as f64 / blocks);
                    let t0 = Instant::now();
                    mat_vec_packed_par(&pool, &p, &a, &mut o2);
                    t2.push(t0.elapsed().as_nanos() as f64 / blocks);
                }
                assert!(o1.iter().zip(&o2).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: bits");
                let (a1, a2) = (stat(&mut t1), stat(&mut t2));
                eprintln!("{name:22} {rows:6}x{n:<5} {th} thread(s) ns/block min/median: file layout {:.2}/{:.2} | packed {:.2}/{:.2} ({:.3}x) · repack {pack_ms:.0} ms, bits equal",
                    a1.0, a1.1, a2.0, a2.1, a1.1 / a2.1);
            }
        }
    }

