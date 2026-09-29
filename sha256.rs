// SPDX-License-Identifier: MIT OR Apache-2.0
//! P1 — SHA-256 (FIPS 180-4), dependency-free, for the model pin: a GGUF plays only if its sha256
//! equals the record in the PYTHAI fork's `FORK.json` (`files[].sha256` for its path).

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    nbuf: usize,
    len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Sha256 {
            h: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19],
            buf: [0; 64],
            nbuf: 0,
            len: 0,
        }
    }
}

impl Sha256 {
    /// Compress whole 64-byte blocks: the CPU's SHA extensions when it has them (x86 SHA-NI: Zen, Ice Lake and
    /// later), else the portable rounds. Both give the same digest; the tests check them against each other.
    fn blocks(&mut self, data: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        {
            if shani::available() {
                // SAFETY: the CPU reports sha, sse4.1 and ssse3; data is whole 64-byte blocks
                unsafe { shani::compress(&mut self.h, data) };
                return;
            }
        }
        for c in data.chunks_exact(64) {
            self.block(c);
        }
    }

    fn block(&mut self, b: &[u8]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b_, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let t1 = h
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ (!e & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22)).wrapping_add((a & b_) ^ (a & c) ^ (b_ & c));
            (h, g, f, e, d, c, b_, a) = (g, f, e, d.wrapping_add(t1), c, b_, a, t1.wrapping_add(t2));
        }
        for (s, v) in self.h.iter_mut().zip([a, b_, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len += data.len() as u64;
        if self.nbuf > 0 {
            let n = (64 - self.nbuf).min(data.len());
            self.buf[self.nbuf..self.nbuf + n].copy_from_slice(&data[..n]);
            self.nbuf += n;
            data = &data[n..];
            if self.nbuf == 64 {
                let b = self.buf;
                self.blocks(&b);
                self.nbuf = 0;
            }
        }
        let whole = data.len() / 64 * 64;
        self.blocks(&data[..whole]);
        let r = &data[whole..];
        self.buf[..r.len()].copy_from_slice(r);
        self.nbuf += r.len();
    }

    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.nbuf != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        let mut o = [0u8; 32];
        for (i, v) in self.h.iter().enumerate() {
            o[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
        }
        o
    }
}

#[cfg(target_arch = "x86_64")]
mod shani {
    //! SHA-256 with the x86 SHA extensions (`sha256rnds2`, `sha256msg1`, `sha256msg2`): two rounds per instruction,
    //! the message schedule in hardware. The layout is Intel's reference (the state held as ABEF / CDGH).
    use std::arch::x86_64::*;
    use std::sync::atomic::{AtomicU8, Ordering};

    static HAVE: AtomicU8 = AtomicU8::new(0); // 0 unknown, 1 no, 2 yes

    pub fn available() -> bool {
        match HAVE.load(Ordering::Relaxed) {
            2 => true,
            1 => false,
            _ => {
                let ok = std::env::var_os("BANKML_NO_SHANI").is_none()
                    && is_x86_feature_detected!("sha")
                    && is_x86_feature_detected!("sse4.1")
                    && is_x86_feature_detected!("ssse3");
                HAVE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
                ok
            }
        }
    }

    #[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
    pub unsafe fn compress(h: &mut [u32; 8], data: &[u8]) {
        let mask = _mm_set_epi64x(0x0c0d_0e0f_0809_0a0b, 0x0405_0607_0001_0203);
        let mut tmp = _mm_loadu_si128(h.as_ptr() as *const __m128i); // A B C D
        let mut st1 = _mm_loadu_si128(h.as_ptr().add(4) as *const __m128i); // E F G H
        tmp = _mm_shuffle_epi32(tmp, 0xB1); // C D A B
        st1 = _mm_shuffle_epi32(st1, 0x1B); // H G F E
        let mut st0 = _mm_alignr_epi8(tmp, st1, 8); // A B E F
        st1 = _mm_blend_epi16(st1, tmp, 0xF0); // C D G H
        let k = super::K.as_ptr() as *const __m128i;
        for b in data.chunks_exact(64) {
            let (s0, s1) = (st0, st1);
            let p = b.as_ptr() as *const __m128i;
            let mut m = [
                _mm_shuffle_epi8(_mm_loadu_si128(p), mask),
                _mm_shuffle_epi8(_mm_loadu_si128(p.add(1)), mask),
                _mm_shuffle_epi8(_mm_loadu_si128(p.add(2)), mask),
                _mm_shuffle_epi8(_mm_loadu_si128(p.add(3)), mask),
            ];
            for i in 0..16 {
                let w = _mm_add_epi32(m[i % 4], _mm_loadu_si128(k.add(i)));
                st1 = _mm_sha256rnds2_epu32(st1, st0, w);
                st0 = _mm_sha256rnds2_epu32(st0, st1, _mm_shuffle_epi32(w, 0x0E));
                if i < 12 {
                    // schedule the words four groups ahead: W[i+4] from W[i..i+4]
                    let t = _mm_sha256msg1_epu32(m[i % 4], m[(i + 1) % 4]);
                    let t = _mm_add_epi32(t, _mm_alignr_epi8(m[(i + 3) % 4], m[(i + 2) % 4], 4));
                    m[i % 4] = _mm_sha256msg2_epu32(t, m[(i + 3) % 4]);
                }
            }
            st0 = _mm_add_epi32(st0, s0);
            st1 = _mm_add_epi32(st1, s1);
        }
        tmp = _mm_shuffle_epi32(st0, 0x1B); // F E B A
        st1 = _mm_shuffle_epi32(st1, 0xB1); // D C H G
        st0 = _mm_blend_epi16(tmp, st1, 0xF0); // D C B A
        st1 = _mm_alignr_epi8(st1, tmp, 8); // H G F E
        _mm_storeu_si128(h.as_mut_ptr() as *mut __m128i, st0);
        _mm_storeu_si128(h.as_mut_ptr().add(4) as *mut __m128i, st1);
    }
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn file_hex(p: &std::path::Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::default();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            return Ok(hex(&h.finish()));
        }
        h.update(&buf[..n]);
    }
}

/// The pinned sha256 for `name` in a FORK.json (`"files": [{"path", "bytes", "sha256"}, …]`).
/// A narrow scan, not a JSON parser: the object whose `"path"` equals `name` must carry a
/// 64-hex `"sha256"` before the object closes, else `None` (refuse — unpinned).
pub fn pinned_sha256(fork_json: &str, name: &str) -> Option<String> {
    let quoted = crate::gguf::jstr(name);
    let value = |rest: &str| -> Option<(usize, String)> {
        // rest starts after a key: optional ws, ':', optional ws, a JSON string (no escapes needed here)
        let t = rest.trim_start().strip_prefix(':')?.trim_start();
        let v = t.strip_prefix('"')?;
        let e = v.find('"')?;
        Some((rest.len() - v.len() + e + 1, v[..e].to_string()))
    };
    let mut from = 0;
    while let Some(k) = fork_json[from..].find("\"path\"") {
        let at = from + k + 6;
        from = at;
        let Some((_, p)) = value(&fork_json[at..]) else { continue };
        if format!("\"{p}\"") != quoted {
            continue;
        }
        let start = fork_json[..at].rfind('{')?;
        let obj = &fork_json[start..at + fork_json[at..].find('}')?];
        let k = obj.find("\"sha256\"")?;
        let (_, h) = value(&obj[k + 8..])?;
        return (h.len() == 64 && h.bytes().all(|c| c.is_ascii_hexdigit())).then(|| h.to_ascii_lowercase());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn portable(data: &[u8]) -> [u8; 32] {
        let mut h = Sha256 { len: data.len() as u64, ..Default::default() };
        let whole = data.len() / 64 * 64;
        for c in data[..whole].chunks_exact(64) {
            h.block(c);
        }
        // finish without the dispatching update(): pad by hand through block()
        let mut tail = data[whole..].to_vec();
        tail.push(0x80);
        while tail.len() % 64 != 56 {
            tail.push(0);
        }
        tail.extend((data.len() as u64 * 8).to_be_bytes());
        for c in tail.chunks_exact(64) {
            h.block(c);
        }
        let mut o = [0u8; 32];
        for (i, v) in h.h.iter().enumerate() {
            o[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
        }
        o
    }

    #[test]
    fn hardware_path_equals_portable_on_every_length() {
        // the SHA-NI path (when the CPU has it) against the portable rounds, lengths 0..=1000 and split updates
        let mut x = 0x9e3779b97f4a7c15u64;
        let data: Vec<u8> = (0..70_000).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect();
        for n in (0..=1000).chain([4095, 4096, 4097, 65_536, 69_999]) {
            let mut h = Sha256::default();
            h.update(&data[..n]);
            assert_eq!(h.finish(), portable(&data[..n]), "length {n}");
        }
        let mut h = Sha256::default();
        for piece in data.chunks(777) {
            h.update(piece);
        }
        assert_eq!(h.finish(), portable(&data), "split updates");
        #[cfg(target_arch = "x86_64")]
        eprintln!("sha256: hardware path {}", if shani::available() { "SHA-NI (tested above)" } else { "absent; portable only" });
    }

    fn h(b: &[u8]) -> String {
        let mut s = Sha256::default();
        s.update(b);
        hex(&s.finish())
    }

    #[test]
    fn fips_vectors() {
        assert_eq!(h(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(h(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            h(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let mut s = Sha256::default();
        for _ in 0..1000 {
            s.update(&[b'a'; 1000]); // streaming across block boundaries: 10^6 × 'a'
        }
        assert_eq!(hex(&s.finish()), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn fork_json_pin() {
        let j = r#"{"files": [
 {"path": "LICENSE", "bytes": 10174, "sha256": "aa"},
 {"path": "Bonsai-8B-Q1_0.gguf", "bytes": 1, "sha256": "284A335AA3FB2CED3B1B01FCB40B08AA783E3B70832767F0DD2E3FDFA134BD54"}]}"#;
        assert_eq!(pinned_sha256(j, "Bonsai-8B-Q1_0.gguf").as_deref(), Some("284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54"));
        assert_eq!(pinned_sha256(j, "LICENSE"), None); // not 64 hex → unpinned
        assert_eq!(pinned_sha256(j, "other.gguf"), None);
        // sha256 before path, no spaces, the indent=1 shape fork_bonsai.py writes
        let k = "{\"files\":[{\"sha256\":\"284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54\",\"path\":\"a.gguf\"}]}";
        assert_eq!(pinned_sha256(k, "a.gguf").as_deref(), Some("284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54"));
        assert_eq!(pinned_sha256(k, "a.ggu"), None);
    }
}
