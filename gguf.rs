//! P1 — the GGUF guard: a header-only GGUF v3 parse and the three traps, ported from minaiml
//! `minaiml-bonsai/tools/gguf_guard.py` (same verdicts, same reasons, same JSON keys). Tensor data is
//! never read by `judge`; `tensor_bytes` exists for kernels/tests that need a tensor's raw blocks.
//!
//! Deliberate divergences from the Python (each one fails *closed*, never open):
//! - a malformed header (bad magic, unknown value type, alignment 0, arrays nested deeper than
//!   `MAX_ARRAY_DEPTH`) is `Refuse(reason)`; Python raises (for deep nesting, `RecursionError`).
//! - a KV size per token that overflows i128 is `Refuse(reason)`; Python's big integers report it and play.
//! - `guard(path)` re-reads with a larger prefix when the header outgrows 32 MiB, so `NeedMore` only
//!   survives when the file itself is truncated; the pure `judge(bytes, ..)` behaves exactly like Python.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Read;
use std::path::Path;

pub const MAINLINE_COUNT: u32 = 43; // GGML_TYPE_COUNT, ggml.h @ b11192 (read 2026-09-25)
pub const FORK_ONLY: [u32; 2] = [142, 143]; // PQ2_0, PTQ1_0 (PrismML fork)
const DEFAULT_PREFIX: u64 = 32 << 20;
/// Arrays of arrays deeper than this refuse. Each level costs 12 header bytes and one stack frame, so an
/// unbounded parse lets a 24 MB header overflow the stack (0.0.1 aborted there instead of refusing).
pub const MAX_ARRAY_DEPTH: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Mainline,
    Prism,
}

impl Engine {
    pub fn as_str(self) -> &'static str {
        match self {
            Engine::Mainline => "mainline",
            Engine::Prism => "prism",
        }
    }
}

/// A GGUF metadata value. Arrays are skipped, as in the Python (`"<array n>"`).
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    U(u64),
    I(i64),
    F(f64),
    B(bool),
    S(String),
    Arr(u64),
}

impl Val {
    /// Python `isinstance(x, int)` (bool included, as in Python).
    fn int(&self) -> Option<i128> {
        match *self {
            Val::U(v) => Some(v as i128),
            Val::I(v) => Some(v as i128),
            Val::B(v) => Some(v as i128),
            _ => None,
        }
    }
    /// Python `str(x)`.
    fn py_str(&self) -> String {
        match self {
            Val::U(v) => v.to_string(),
            Val::I(v) => v.to_string(),
            Val::F(v) => format!("{v:?}"),
            Val::B(v) => (if *v { "True" } else { "False" }).into(),
            Val::S(s) => s.clone(),
            Val::Arr(n) => format!("<array {n}>"),
        }
    }
    fn truthy(&self) -> bool {
        match self {
            Val::U(v) => *v != 0,
            Val::I(v) => *v != 0,
            Val::F(v) => *v != 0.0,
            Val::B(v) => *v,
            Val::S(s) => !s.is_empty(),
            Val::Arr(_) => true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: u32,
    /// relative to the data section
    pub offset: u64,
}

impl TensorInfo {
    pub fn nelem(&self) -> u128 {
        self.dims.iter().fold(1u128, |a, &d| a.saturating_mul(d as u128))
    }
}

#[derive(Debug)]
pub struct Header {
    pub version: u32,
    pub kv: HashMap<String, Val>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub data_start: u64,
}

enum Err {
    NeedMore(u64),
    Bad(String),
}

struct Rd<'a> {
    b: &'a [u8],
    o: u64,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: u64) -> Result<&'a [u8], Err> {
        let end = self.o.checked_add(n).ok_or(Err::NeedMore(u64::MAX))?;
        if end > self.b.len() as u64 {
            return Err(Err::NeedMore(end));
        }
        let s = &self.b[self.o as usize..end as usize];
        self.o = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, Err> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Err> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String, Err> {
        let n = self.u64()?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn val(&mut self, t: u32, depth: u32) -> Result<Val, Err> {
        let size = |t: u32| match t {
            0 | 1 | 7 => Some(1u64), // u8 i8 bool
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        Ok(match t {
            0 => Val::U(self.take(1)?[0] as u64),
            1 => Val::I(self.take(1)?[0] as i8 as i64),
            2 => Val::U(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Val::I(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Val::U(self.u32()? as u64),
            5 => Val::I(self.u32()? as i32 as i64),
            6 => Val::F(f32::from_bits(self.u32()?) as f64),
            7 => Val::B(self.take(1)?[0] != 0),
            8 => Val::S(self.str()?),
            9 => {
                let (et, n) = (self.u32()?, self.u64()?);
                if et == 9 && depth >= MAX_ARRAY_DEPTH {
                    return Err(Err::Bad(format!("gguf arrays nested deeper than {MAX_ARRAY_DEPTH}")));
                }
                if let Some(sz) = size(et) {
                    self.take(n.saturating_mul(sz))?; // bulk scalar arrays are skipped, not decoded
                } else {
                    for _ in 0..n {
                        self.val(et, depth + 1)?;
                    }
                }
                Val::Arr(n)
            }
            10 => Val::U(self.u64()?),
            11 => Val::I(self.u64()? as i64),
            12 => Val::F(f64::from_bits(self.u64()?)),
            _ => return Err(Err::Bad(format!("unknown gguf value type {t}"))),
        })
    }
}

fn parse(b: &[u8]) -> Result<Header, Err> {
    let mut r = Rd { b, o: 0 };
    if r.take(4)? != b"GGUF" {
        return Err(Err::Bad("not a GGUF file".into()));
    }
    let (version, nt, nkv) = (r.u32()?, r.u64()?, r.u64()?);
    let mut kv = HashMap::new();
    for _ in 0..nkv {
        let k = r.str()?;
        let t = r.u32()?;
        kv.insert(k, r.val(t, 0)?); // duplicate keys: last wins, as a Python dict
    }
    let mut tensors = Vec::new();
    for _ in 0..nt {
        let name = r.str()?;
        let nd = r.u32()?;
        let mut dims = Vec::new();
        for _ in 0..nd {
            dims.push(r.u64()?);
        }
        tensors.push(TensorInfo { name, dims, ty: r.u32()?, offset: r.u64()? });
    }
    let alignment = match kv.get("general.alignment") {
        None => 32,
        Some(Val::F(f)) if *f >= 1.0 => *f as u64,
        Some(v) => match v.int() {
            Some(a) if a >= 1 && a <= u64::MAX as i128 => a as u64,
            _ => return Err(Err::Bad(format!("general.alignment {} is not a positive integer", v.py_str()))),
        },
    };
    let data_start = r.o.div_ceil(alignment).saturating_mul(alignment);
    Ok(Header { version, kv, tensors, alignment, data_start })
}

/// The guard's answer: `verdict` plus everything the Python reports.
#[derive(Debug)]
pub struct Report {
    pub verdict: crate::Verdict,
    pub engine: Engine,
    pub gguf_version: u32,
    pub arch: Option<String>,
    pub name: Option<String>,
    pub tensors: usize,
    /// (type name, count) sorted by type id, as the Python dict
    pub types: Vec<(String, usize)>,
    pub kv_f16_bytes_per_token: Option<i128>,
    pub header: Option<Header>,
}

pub fn type_name(t: u32) -> String {
    let n = match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0",
        10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 30 => "BF16",
        34 => "TQ1_0", 35 => "TQ2_0", 41 => "Q1_0", 42 => "Q2_0", 142 => "PQ2_0", 143 => "PTQ1_0",
        _ => return t.to_string(),
    };
    n.into()
}

/// `bonsai[\s_-]*2|prism-fork-required`, case-insensitive.
fn bonsai2(label: &str) -> bool {
    let l = label.to_lowercase();
    if l.contains("prism-fork-required") {
        return true;
    }
    l.match_indices("bonsai").any(|(i, _)| {
        l[i + 6..].chars().find(|c| !(c.is_whitespace() || *c == '_' || *c == '-')) == Some('2')
    })
}

/// Pure judgement over a header prefix, as `gguf_guard.judge(b, engine, file_size, filename)`.
pub fn judge(b: &[u8], engine: Engine, file_size: Option<u64>, filename: &str) -> Report {
    let mut rep = Report {
        verdict: crate::Verdict::Play,
        engine,
        gguf_version: 0,
        arch: None,
        name: None,
        tensors: 0,
        types: vec![],
        kv_f16_bytes_per_token: None,
        header: None,
    };
    let h = match parse(b) {
        Ok(h) => h,
        Err(Err::NeedMore(n)) => {
            rep.verdict = crate::Verdict::NeedMore(n);
            return rep;
        }
        Err(Err::Bad(why)) => {
            rep.verdict = crate::Verdict::Refuse(vec![why]);
            return rep;
        }
    };
    let mut ids: Vec<u32> = h.tensors.iter().map(|t| t.ty).collect();
    ids.sort_unstable();
    ids.dedup();
    let arch_v = h.kv.get("general.architecture");
    rep.gguf_version = h.version;
    rep.arch = arch_v.map(Val::py_str);
    rep.name = h.kv.get("general.name").map(Val::py_str);
    rep.tensors = h.tensors.len();
    rep.types = ids.iter().map(|&t| (type_name(t), h.tensors.iter().filter(|x| x.ty == t).count())).collect();
    let mut why = Vec::new();

    // KV bytes/token at f16 (R2 RAM plan); None when the header uses per-layer arrays.
    let a = arch_v.map(Val::py_str).unwrap_or_else(|| "None".into());
    let g = |k: &str| h.kv.get(&format!("{a}.{k}"));
    let (l, hk) = (g("block_count").and_then(Val::int), g("attention.head_count_kv").and_then(Val::int));
    let (mut k, mut v) = (g("attention.key_length"), g("attention.value_length").and_then(Val::int));
    let fallback;
    if k.is_none() {
        if let (Some(e), Some(nh)) = (g("embedding_length").and_then(Val::int), g("attention.head_count").and_then(Val::int)) {
            if nh != 0 {
                fallback = Val::I((e.div_euclid(nh)) as i64);
                k = Some(&fallback);
                v = fallback.int();
            }
        }
    }
    if let (Some(l), Some(hk), Some(k)) = (l, hk, k.and_then(Val::int)) {
        match k.checked_add(v.unwrap_or(k)).and_then(|kv| l.checked_mul(hk)?.checked_mul(kv)?.checked_mul(2)) {
            Some(b) => rep.kv_f16_bytes_per_token = Some(b),
            None => why.push(format!("KV cache size per token overflows ({a}.block_count × head_count_kv × key/value length): malformed header")),
        }
    }

    let unknown: Vec<u32> = ids.iter().copied().filter(|t| *t >= MAINLINE_COUNT && !FORK_ONLY.contains(t)).collect();
    if !unknown.is_empty() {
        why.push(format!("type ids {unknown:?} unknown to every build this guard knows"));
    }
    if engine == Engine::Mainline && ids.iter().any(|t| FORK_ONLY.contains(t)) {
        why.push("PQ2_0/PTQ1_0 are PrismML-fork types; mainline and llama.rn refuse them".into());
    }

    // legacy Q2_0: bytes between offsets vs the group-64 expectation (64 trits -> 18 B)
    let mut ord: Vec<&TensorInfo> = h.tensors.iter().collect();
    ord.sort_by_key(|t| t.offset);
    let mut short = 0;
    for (i, t) in ord.iter().enumerate() {
        let end: Option<i128> = match ord.get(i + 1) {
            Some(n) => Some(n.offset as i128),
            None => file_size.map(|s| s as i128 - h.data_start as i128),
        };
        let Some(end) = end else { continue };
        if t.ty != 42 {
            continue;
        }
        let want = (t.nelem() / 64).saturating_mul(18) as i128;
        if end - (t.offset as i128) < want {
            short += 1;
        }
    }
    if short > 0 {
        why.push(format!(
            "{short} Q2_0 tensors smaller than group-64 layout: legacy group-128 Prism file, use *-Q2_0_g64.gguf or *-PQ2_0.gguf"
        ));
    }

    let label: Vec<String> = [h.kv.get("general.name").cloned(), h.kv.get("general.basename").cloned(), Some(Val::S(filename.into()))]
        .into_iter()
        .flatten()
        .filter(Val::truthy)
        .map(|v| v.py_str())
        .collect();
    if engine == Engine::Mainline && bonsai2(&label.join(" ")) {
        why.push(
            "Bonsai 2 weights are in a rotated basis and need the fork's Hadamard transform; mainline loads the Q2_0 band silently and outputs gibberish".into(),
        );
    }
    rep.verdict = if why.is_empty() { crate::Verdict::Play } else { crate::Verdict::Refuse(why) };
    rep.header = Some(h);
    rep
}

/// File-level guard: reads a header prefix (32 MiB, grown on demand up to the file size).
pub fn guard_file(path: &Path, engine: Engine) -> std::io::Result<Report> {
    let size = std::fs::metadata(path)?.len();
    let fname = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut want = DEFAULT_PREFIX.min(size);
    loop {
        let mut buf = Vec::with_capacity(want as usize);
        std::fs::File::open(path)?.take(want).read_to_end(&mut buf)?;
        let rep = judge(&buf, engine, Some(size.max(buf.len() as u64)), &fname);
        match rep.verdict {
            crate::Verdict::NeedMore(n) if n <= size && n > want => want = n.max(want * 2).min(size),
            _ => return Ok(rep),
        }
    }
}

/// (elements per block, bytes per block) for the types bankml reads; None = not playable here.
pub fn block_layout(ty: u32) -> Option<(u64, u64)> {
    match ty {
        0 => Some((1, 4)),
        1 | 30 => Some((1, 2)),
        8 => Some((32, 34)),
        41 => Some((128, 18)),
        42 => Some((64, 18)),
        _ => None,
    }
}

/// (absolute start, byte length) of a tensor's blocks, or why it cannot be read. Checked end to end:
/// the type must be one bankml reads, rows (`dims[0]`) must be whole blocks, and no size may overflow.
pub fn tensor_span(h: &Header, t: &TensorInfo) -> Result<(u64, u64), String> {
    let (per, bb) = block_layout(t.ty).ok_or_else(|| format!("{}: type {} not readable", t.name, type_name(t.ty)))?;
    if t.dims.first().is_some_and(|&d0| d0 % per != 0) {
        return Err(format!("{}: row length {} is not a whole number of {per}-element blocks", t.name, t.dims[0]));
    }
    let len = u64::try_from(t.nelem()).ok().and_then(|n| (n / per).checked_mul(bb));
    let start = h.data_start.checked_add(t.offset);
    match (start, len) {
        (Some(s), Some(l)) if s.checked_add(l).is_some() => Ok((s, l)),
        _ => Err(format!("{}: size or offset overflows", t.name)),
    }
}

/// Raw bytes of one tensor (for kernels and oracle tests; the player will mmap instead). The span is
/// checked against the file before anything is allocated, so a lying header cannot ask for a huge buffer.
pub fn tensor_bytes(path: &Path, h: &Header, t: &TensorInfo) -> std::io::Result<Vec<u8>> {
    use std::io::{Seek, SeekFrom};
    let (start, n) = tensor_span(h, t).map_err(std::io::Error::other)?;
    let mut f = std::fs::File::open(path)?;
    if start + n > f.metadata()?.len() {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, format!("{}: runs past the end of the file", t.name)));
    }
    f.seek(SeekFrom::Start(start))?;
    let mut v = vec![0u8; usize::try_from(n).map_err(std::io::Error::other)?];
    f.read_exact(&mut v)?;
    Ok(v)
}

/// The GGUF, mapped read-only (weights are never copied; pages load on first touch). Unix only.
/// No crate: `mmap`/`munmap` are libc, which std already links.
pub struct Mmap {
    ptr: *const u8,
    len: usize,
}

// SAFETY: a PROT_READ, MAP_PRIVATE mapping never changes under the process; sharing reads is sound.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

#[cfg(unix)]
extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

#[cfg(unix)]
impl Mmap {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        use std::os::fd::AsRawFd;
        let f = std::fs::File::open(path)?;
        let len = f.metadata()?.len() as usize;
        // PROT_READ = 1, MAP_PRIVATE = 2; the mapping outlives the fd
        let p = unsafe { mmap(std::ptr::null_mut(), len.max(1), 1, 2, f.as_raw_fd(), 0) };
        if p as isize == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Mmap { ptr: p, len })
    }
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    /// One tensor's blocks, bounds-checked against the file (`tensor_span`, then the mapping's length).
    pub fn tensor(&self, h: &Header, t: &TensorInfo) -> Option<&[u8]> {
        let (start, n) = tensor_span(h, t).ok()?;
        self.bytes().get(usize::try_from(start).ok()?..usize::try_from(start + n).ok()?)
    }
}

#[cfg(unix)]
impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe { munmap(self.ptr as *mut u8, self.len.max(1)) };
    }
}

impl Report {
    pub fn verdict_str(&self) -> &'static str {
        match self.verdict {
            crate::Verdict::Play => "play",
            crate::Verdict::Refuse(_) => "refuse",
            crate::Verdict::NeedMore(_) => "need_more",
        }
    }

    /// The same JSON object `gguf_guard.py --json` prints (key set and values; whitespace differs).
    pub fn to_json(&self) -> String {
        let mut o = String::from("{");
        if let crate::Verdict::NeedMore(n) = self.verdict {
            let _ = write!(o, "\"verdict\": \"need_more\", \"bytes_needed\": {n}}}");
            return o;
        }
        let opt = |s: &Option<String>| s.as_ref().map(|s| jstr(s)).unwrap_or_else(|| "null".into());
        let _ = write!(o, "\"gguf_version\": {}, \"arch\": {}, \"name\": {}, \"tensors\": {}, \"types\": {{",
            self.gguf_version, opt(&self.arch), opt(&self.name), self.tensors);
        let types: Vec<String> = self.types.iter().map(|(n, c)| format!("{}: {c}", jstr(n))).collect();
        let reasons: Vec<String> = match &self.verdict {
            crate::Verdict::Refuse(r) => r.iter().map(|s| jstr(s)).collect(),
            _ => vec![],
        };
        let _ = write!(o, "{}}}, \"engine\": \"{}\", \"reasons\": [{}]", types.join(", "), self.engine.as_str(), reasons.join(", "));
        if let Some(k) = self.kv_f16_bytes_per_token {
            let _ = write!(o, ", \"kv_f16_bytes_per_token\": {k}");
        }
        let _ = write!(o, ", \"verdict\": \"{}\"}}", self.verdict_str());
        o
    }
}

pub fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Verdict;

    fn s(x: &str) -> Vec<u8> {
        let mut v = (x.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(x.as_bytes());
        v
    }

    /// The synthetic GGUF of test_gguf_guard.py: tensors = [(name, n_elements, type_id)];
    /// pad = (elems_per_block, bytes_per_block) actually written for type 42 / >= 142.
    fn gguf(name: &str, tensors: &[(&str, u64, u32)], pad: (u64, u64)) -> Vec<u8> {
        let mut h = b"GGUF".to_vec();
        h.extend(3u32.to_le_bytes());
        h.extend((tensors.len() as u64).to_le_bytes());
        h.extend(3u64.to_le_bytes());
        for (k, v) in [("general.architecture", Some("qwen3")), ("general.name", Some(name)), ("general.alignment", None)] {
            h.extend(s(k));
            match v {
                None => {
                    h.extend(4u32.to_le_bytes());
                    h.extend(32u32.to_le_bytes());
                }
                Some(v) => {
                    h.extend(8u32.to_le_bytes());
                    h.extend(s(v));
                }
            }
        }
        let mut off = 0u64;
        for &(tn, n, ty) in tensors {
            let (per, bb) = if ty == 42 || ty >= 142 { pad } else { (1, 2) };
            let sz = n / per * bb;
            h.extend(s(tn));
            h.extend(1u32.to_le_bytes());
            h.extend(n.to_le_bytes());
            h.extend(ty.to_le_bytes());
            h.extend(off.to_le_bytes());
            off += sz.div_ceil(32) * 32;
        }
        h.resize(h.len().div_ceil(32) * 32, 0);
        h.resize(h.len() + off as usize, 0);
        h
    }

    const N: u64 = 4096 * 4;

    fn case(name: &str, ts: &[(&str, u64, u32)], blk: (u64, u64), eng: Engine) -> Report {
        let b = gguf(name, ts, blk);
        judge(&b, eng, Some(b.len() as u64), name)
    }

    #[test]
    fn t1_q1_0_plays_on_mainline() {
        let r = case("Bonsai-1.7B", &[("a", N, 41), ("b", N, 1)], (128, 18), Engine::Mainline);
        assert_eq!(r.verdict, Verdict::Play);
        assert_eq!(r.types, vec![("F16".to_string(), 1), ("Q1_0".to_string(), 1)]);
    }

    #[test]
    fn t2_q2_0_g64_plays_on_mainline() {
        let r = case("Ternary-Bonsai-1.7B", &[("a", N, 42), ("b", N, 42)], (64, 18), Engine::Mainline);
        assert_eq!(r.verdict, Verdict::Play);
    }

    #[test]
    fn t3_legacy_g128_refused_by_size() {
        let r = case("Ternary-Bonsai-1.7B", &[("a", N, 42), ("b", N, 42)], (128, 34), Engine::Mainline);
        match r.verdict {
            Verdict::Refuse(w) => assert!(w[0].starts_with("2 Q2_0 tensors smaller than group-64"), "{w:?}"),
            v => panic!("{v:?}"),
        }
    }

    #[test]
    fn t4_pq2_0_refused_on_mainline() {
        let r = case("Ternary-Bonsai-1.7B", &[("a", N, 142)], (128, 34), Engine::Mainline);
        assert!(matches!(r.verdict, Verdict::Refuse(ref w) if w[0].contains("PrismML-fork types")), "{:?}", r.verdict);
    }

    #[test]
    fn t5_pq2_0_plays_on_prism() {
        assert_eq!(case("Ternary-Bonsai-1.7B", &[("a", N, 142)], (128, 34), Engine::Prism).verdict, Verdict::Play);
    }

    #[test]
    fn t6_bonsai2_refused_on_mainline_by_name() {
        let r = case("Ternary Bonsai 2 27B", &[("a", N, 42)], (64, 18), Engine::Mainline);
        assert!(matches!(r.verdict, Verdict::Refuse(ref w) if w[0].contains("rotated basis")), "{:?}", r.verdict);
    }

    #[test]
    fn t7_bonsai2_plays_on_prism() {
        assert_eq!(case("Ternary Bonsai 2 27B", &[("a", N, 42)], (64, 18), Engine::Prism).verdict, Verdict::Play);
    }

    #[test]
    fn t8_kv_bytes_per_token_qwen3_8b_shape() {
        let b = gguf("Bonsai-8B", &[("a", N, 41)], (128, 18));
        let mut extra = s("general.architecture");
        extra.extend(8u32.to_le_bytes());
        extra.extend(s("qwen3"));
        for (k, v) in [("qwen3.block_count", 36u32), ("qwen3.attention.head_count_kv", 8), ("qwen3.attention.key_length", 128)] {
            extra.extend(s(k));
            extra.extend(4u32.to_le_bytes());
            extra.extend(v.to_le_bytes());
        }
        let nkv = u64::from_le_bytes(b[16..24].try_into().unwrap());
        let mut raw = b[..16].to_vec();
        raw.extend((nkv + 4).to_le_bytes());
        raw.extend(extra);
        raw.extend(&b[24..]);
        let r = judge(&raw, Engine::Mainline, None, "Bonsai-8B");
        assert_eq!(r.kv_f16_bytes_per_token, Some(147456));
        assert_eq!(r.verdict, Verdict::Play);
    }

    #[test]
    fn t9_truncated_prefix_needs_more() {
        let b = gguf("x", &[("a", N, 42)], (64, 18));
        assert!(matches!(judge(&b[..40], Engine::Mainline, None, "").verdict, Verdict::NeedMore(n) if n > 40));
    }

    // beyond the Python suite: fail-closed cases
    #[test]
    fn bad_magic_and_unknown_types_refuse() {
        assert!(matches!(judge(b"GGUX\x03\0\0\0", Engine::Mainline, None, "").verdict, Verdict::Refuse(_)));
        let r = case("x", &[("a", N, 77)], (64, 18), Engine::Prism);
        assert!(matches!(r.verdict, Verdict::Refuse(ref w) if w[0].contains("[77]")));
        assert!(bonsai2("Bonsai_-  2") && bonsai2("x PRISM-FORK-REQUIRED") && !bonsai2("Bonsai-1.7B") && !bonsai2("Bonsai-8B-2x"));
    }

    fn kv_entry(k: &str, t: u32, v: &[u8]) -> Vec<u8> {
        let mut e = s(k);
        e.extend(t.to_le_bytes());
        e.extend_from_slice(v);
        e
    }

    fn raw(entries: &[Vec<u8>]) -> Vec<u8> {
        let mut h = b"GGUF".to_vec();
        h.extend(3u32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        h.extend((entries.len() as u64).to_le_bytes());
        entries.iter().for_each(|e| h.extend(e));
        h.resize(h.len().div_ceil(32) * 32 + 32, 0);
        h
    }

    /// 0.0.1 recursed once per nesting level and aborted on a stack overflow; now a bounded refusal.
    #[test]
    fn deeply_nested_arrays_refuse_not_crash() {
        let nest = |depth: usize| {
            let mut v = Vec::new();
            for _ in 0..depth {
                v.extend(9u32.to_le_bytes());
                v.extend(1u64.to_le_bytes());
            }
            v.extend(4u32.to_le_bytes());
            v.extend(0u64.to_le_bytes());
            raw(&[kv_entry("a", 9, &v)])
        };
        assert_eq!(judge(&nest(3), Engine::Mainline, None, "").verdict, Verdict::Play);
        let deep = nest(2_000_000);
        assert!(matches!(judge(&deep, Engine::Mainline, None, "").verdict, Verdict::Refuse(ref w) if w[0].contains("nested deeper")));
    }

    /// 0.0.1 wrapped i128 silently in release (and panicked in debug) and played.
    #[test]
    fn kv_size_overflow_refuses() {
        let mut e = vec![kv_entry("general.architecture", 8, &s("qwen3"))];
        for k in ["qwen3.block_count", "qwen3.attention.head_count_kv", "qwen3.attention.key_length"] {
            e.push(kv_entry(k, 10, &u64::MAX.to_le_bytes()));
        }
        let r = judge(&raw(&e), Engine::Mainline, None, "");
        assert_eq!(r.kv_f16_bytes_per_token, None);
        assert!(matches!(r.verdict, Verdict::Refuse(ref w) if w[0].contains("overflows")), "{:?}", r.verdict);
    }

    #[test]
    fn tensor_span_is_checked() {
        let h = Header { version: 3, kv: HashMap::new(), tensors: vec![], alignment: 32, data_start: 64 };
        let t = |dims: Vec<u64>, ty: u32, offset: u64| TensorInfo { name: "t".into(), dims, ty, offset };
        assert_eq!(tensor_span(&h, &t(vec![128, 2], 41, 0)), Ok((64, 36)));
        assert_eq!(tensor_span(&h, &t(vec![64, 3], 42, 32)), Ok((96, 54)));
        assert!(tensor_span(&h, &t(vec![100, 2], 41, 0)).unwrap_err().contains("whole number"));
        // 0.0.1 truncated this element count to 64 bits and returned a short slice
        assert!(tensor_span(&h, &t(vec![1 << 62, 1 << 40], 42, 0)).unwrap_err().contains("overflows"));
        assert!(tensor_span(&h, &t(vec![64], 42, u64::MAX)).unwrap_err().contains("overflows"));
        assert!(tensor_span(&h, &t(vec![64], 12, 0)).unwrap_err().contains("not readable"));
    }

    /// Real file (Apache-2.0, prism-ml/Bonsai-1.7B-gguf, 248,302,272 B). The expected values are
    /// `python3 minaiml-bonsai/tools/gguf_guard.py <file> --json` on 2026-09-25; `testing/guard_agree.py`
    /// re-checks the full JSON against the Python at any time.
    #[test]
    #[ignore = "needs bankml/.models/Bonsai-1.7B-Q1_0.gguf"]
    fn real_bonsai_1_7b_q1_0() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/Bonsai-1.7B-Q1_0.gguf");
        let r = guard_file(&p, Engine::Mainline).expect("model file");
        assert_eq!(r.verdict, Verdict::Play);
        assert_eq!((r.gguf_version, r.arch.as_deref(), r.name.as_deref(), r.tensors), (3, Some("qwen3"), Some("Bonsai-1.7B"), 310));
        assert_eq!(r.types, vec![("F32".into(), 113), ("Q1_0".into(), 197)]);
        assert_eq!(r.kv_f16_bytes_per_token, Some(114688));
    }
}
