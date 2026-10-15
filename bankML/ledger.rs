// SPDX-License-Identifier: MIT OR Apache-2.0
//! Receipts banked: token usage and the prompt/response receipts as a ledger, exact to 18 decimal places, ready for a
//! blockchain by default.
//!
//! - [`Amount`]: an amount in units of 10⁻¹⁸ (as an ERC-20 balance with 18 decimals), held in a `u128` (it fits a
//!   `uint256`), parsed and printed exactly — always 18 decimals — with checked arithmetic. Never a float.
//! - [`Entry`]: one receipt banked — its source (bankML here, or an external provider), the model, the request and
//!   response sha256, the prompt / completion / total tokens, the rate per token and the cost — and its hash:
//!   keccak256 (the EVM's) of its canonical bytes, which include the previous entry's hash, so the ledger's head commits
//!   to every entry before it (a hash chain).
//! - [`Ledger`]: append, verify (a changed entry breaks the chain at that entry), totals per source, JSON lines in and
//!   out — the same format on the server (`bankml serve`) and in the browser (browseML), with the same test vectors.
//! - [`Anchor`]: the blockchain as an extensible module. None by default (the ledger stands alone); THOT (the head as a
//!   commitment in an agent's bundle, `sAGI/thot.py`) and iNFT (an unsigned anchoring transaction the owner signs,
//!   `sAGI/chain.py`) connect through it. bankML never signs or sends a transaction itself.

use std::collections::BTreeMap;
use std::fmt;

/// 10¹⁸: one whole unit (one token, one LUV) in base units.
pub const SCALE: u128 = 1_000_000_000_000_000_000;

/// An exact amount with 18 decimals (base units of 10⁻¹⁸).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Amount(pub u128);

impl Amount {
    pub const ZERO: Amount = Amount(0);
    /// `n` whole units (e.g. `n` tokens).
    pub fn units(n: u64) -> Amount {
        Amount(n as u128 * SCALE)
    }
    /// Parse "123", "123.4", "0.000000000000000001" exactly; more than 18 decimals, a sign, or an overflow is refused.
    pub fn parse(s: &str) -> Result<Amount, String> {
        let s = s.trim();
        let (whole, frac) = s.split_once('.').unwrap_or((s, ""));
        if whole.is_empty() || !whole.bytes().all(|b| b.is_ascii_digit()) || !frac.bytes().all(|b| b.is_ascii_digit()) || (s.contains('.') && frac.is_empty()) {
            return Err(format!("not an amount: {s:?}"));
        }
        if frac.len() > 18 {
            return Err(format!("more than 18 decimals: {s:?}"));
        }
        let w: u128 = whole.parse().map_err(|_| format!("too large: {s:?}"))?;
        let f: u128 = if frac.is_empty() { 0 } else { format!("{frac:0<18}").parse().map_err(|_| format!("not an amount: {s:?}"))? };
        w.checked_mul(SCALE).and_then(|x| x.checked_add(f)).map(Amount).ok_or(format!("too large: {s:?}"))
    }
    pub fn checked_add(self, o: Amount) -> Option<Amount> {
        self.0.checked_add(o.0).map(Amount)
    }
    pub fn checked_sub(self, o: Amount) -> Option<Amount> {
        self.0.checked_sub(o.0).map(Amount)
    }
    /// `tokens` × this rate (a rate per token): exact, or `None` on overflow.
    pub fn times(self, tokens: u64) -> Option<Amount> {
        self.0.checked_mul(tokens as u128).map(Amount)
    }
}

impl fmt::Display for Amount {
    /// Always 18 decimals: "1.500000000000000000".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{:018}", self.0 / SCALE, self.0 % SCALE)
    }
}

// ── keccak256 (the EVM's: Keccak-256 with the original 0x01 padding, not SHA3-256's 0x06) ──────────────────────────

const RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];
const ROT: [u32; 25] = [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14];

fn keccak_f(a: &mut [u64; 25]) {
    for rc in RC {
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = a[x + 5 * y].rotate_left(ROT[x + 5 * y]);
            }
        }
        for x in 0..5 {
            for y in 0..5 {
                a[x + 5 * y] = b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
            }
        }
        a[0] ^= rc;
    }
}

/// keccak256(data), as Solidity's `keccak256` and `sAGI/agents.py`'s.
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    const RATE: usize = 136;
    let mut st = [0u64; 25];
    let absorb = |st: &mut [u64; 25], blk: &[u8; RATE]| {
        for (i, w) in blk.as_chunks::<8>().0.iter().enumerate() {
            st[i] ^= u64::from_le_bytes(*w);
        }
        keccak_f(st);
    };
    let (blocks, rest) = data.as_chunks::<RATE>();
    for b in blocks {
        absorb(&mut st, b);
    }
    let mut block = [0u8; RATE];
    block[..rest.len()].copy_from_slice(rest);
    block[rest.len()] ^= 0x01;
    block[RATE - 1] ^= 0x80;
    absorb(&mut st, &block);
    let mut out = [0u8; 32];
    for (i, w) in st[..4].iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
    }
    out
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ── the ledger ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// The hash before the first entry.
pub const GENESIS: [u8; 32] = [0u8; 32];

/// One receipt banked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// 0, 1, 2 … in order.
    pub seq: u64,
    /// When, in milliseconds since the Unix epoch (as the receipt's).
    pub at_ms: u64,
    /// Who answered: `bankml` (this engine, here) or `external:<provider>` (inference elsewhere, its counts as reported).
    pub source: String,
    /// The model: its sha256 where bankML verified it, else the provider's model name.
    pub model: String,
    pub request_sha256: String,
    pub response_sha256: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// The price per token, 18 decimals (zero where nothing is charged).
    pub rate: Amount,
    /// The previous entry's hash (`GENESIS` for the first).
    pub prev: [u8; 32],
}

/// A banked entry: the entry and its hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Banked {
    pub entry: Entry,
    pub hash: [u8; 32],
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

impl Entry {
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
    /// The total tokens as an amount (18 decimals): what the ledger counts.
    pub fn tokens(&self) -> Amount {
        Amount::units(self.total_tokens())
    }
    /// rate × total tokens, exact.
    pub fn cost(&self) -> Result<Amount, String> {
        self.rate.times(self.total_tokens()).ok_or_else(|| "cost overflows".into())
    }
    /// The canonical bytes the hash is taken over: JSON with this key order, no spaces, amounts as 18-decimal strings,
    /// hashes as lowercase hex. The browser (browseML) produces the same bytes.
    pub fn canonical(&self) -> Result<String, String> {
        Ok(format!(
            "{{\"v\":1,\"seq\":{},\"at_ms\":{},\"source\":{},\"model\":{},\"request_sha256\":{},\"response_sha256\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"tokens\":\"{}\",\"rate\":\"{}\",\"cost\":\"{}\",\"prev\":\"{}\"}}",
            self.seq, self.at_ms, json_str(&self.source), json_str(&self.model), json_str(&self.request_sha256), json_str(&self.response_sha256),
            self.prompt_tokens, self.completion_tokens, self.tokens(), self.rate, self.cost()?, hex(&self.prev)
        ))
    }
    pub fn hash(&self) -> Result<[u8; 32], String> {
        Ok(keccak256(self.canonical()?.as_bytes()))
    }
}

impl Banked {
    /// One JSON line: the canonical entry with its hash added last.
    pub fn line(&self) -> Result<String, String> {
        let c = self.entry.canonical()?;
        Ok(format!("{},\"hash\":\"{}\"}}", &c[..c.len() - 1], hex(&self.hash)))
    }
}

/// Totals for one source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Totals {
    pub entries: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub tokens: Amount,
    pub cost: Amount,
}

/// The ledger: entries in order, each chained to the one before.
#[derive(Debug, Clone, Default)]
pub struct Ledger {
    pub entries: Vec<Banked>,
}

impl Ledger {
    /// The head: the last entry's hash (`GENESIS` when empty). Anchoring it commits to the whole ledger.
    pub fn head(&self) -> [u8; 32] {
        self.entries.last().map(|b| b.hash).unwrap_or(GENESIS)
    }
    /// Bank a receipt: its `seq` and `prev` are set here; returns the banked entry.
    pub fn bank(&mut self, mut e: Entry) -> Result<&Banked, String> {
        e.seq = self.entries.len() as u64;
        e.prev = self.head();
        let hash = e.hash()?;
        self.entries.push(Banked { entry: e, hash });
        Ok(self.entries.last().expect("pushed"))
    }
    /// Every entry's hash recomputed and its link checked: `Err` names the first entry that does not hold.
    pub fn verify(&self) -> Result<(), String> {
        let mut prev = GENESIS;
        for (i, b) in self.entries.iter().enumerate() {
            if b.entry.seq != i as u64 {
                return Err(format!("entry {i}: seq {} out of order", b.entry.seq));
            }
            if b.entry.prev != prev {
                return Err(format!("entry {i}: its prev is not entry {}'s hash: the chain is broken here", i as i64 - 1));
            }
            if b.entry.hash()? != b.hash {
                return Err(format!("entry {i}: its contents do not match its hash: changed after banking"));
            }
            prev = b.hash;
        }
        Ok(())
    }
    /// Totals per source, and over all (key `*`), exact.
    pub fn totals(&self) -> Result<BTreeMap<String, Totals>, String> {
        let mut t: BTreeMap<String, Totals> = BTreeMap::new();
        for b in &self.entries {
            let e = &b.entry;
            for key in [e.source.as_str(), "*"] {
                let x = t.entry(key.to_string()).or_default();
                x.entries += 1;
                x.prompt_tokens += e.prompt_tokens;
                x.completion_tokens += e.completion_tokens;
                x.tokens = x.tokens.checked_add(e.tokens()).ok_or("token total overflows")?;
                x.cost = x.cost.checked_add(e.cost()?).ok_or("cost total overflows")?;
            }
        }
        Ok(t)
    }
    /// The ledger as JSON lines.
    pub fn to_jsonl(&self) -> Result<String, String> {
        let mut o = String::new();
        for b in &self.entries {
            o.push_str(&b.line()?);
            o.push('\n');
        }
        Ok(o)
    }
    /// Read JSON lines back (each entry's fields and hash as written; `verify` checks them).
    pub fn from_jsonl(text: &str) -> Result<Ledger, String> {
        let mut l = Ledger::default();
        for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let v = crate::serve::Json::parse(line).ok_or(format!("line {}: not JSON", i + 1))?;
            let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string).ok_or(format!("line {}: no {k}", i + 1));
            let n = |k: &str| v.get(k).and_then(|x| x.as_u64()).ok_or(format!("line {}: no {k}", i + 1));
            let h = |k: &str| -> Result<[u8; 32], String> {
                let t = s(k)?;
                if t.len() != 64 {
                    return Err(format!("line {}: {k} is not 32 bytes of hex", i + 1));
                }
                let mut o = [0u8; 32];
                for (j, b) in o.iter_mut().enumerate() {
                    *b = u8::from_str_radix(&t[2 * j..2 * j + 2], 16).map_err(|_| format!("line {}: {k} is not hex", i + 1))?;
                }
                Ok(o)
            };
            let entry = Entry {
                seq: n("seq")?, at_ms: n("at_ms")?, source: s("source")?, model: s("model")?, request_sha256: s("request_sha256")?,
                response_sha256: s("response_sha256")?, prompt_tokens: n("prompt_tokens")?, completion_tokens: n("completion_tokens")?,
                rate: Amount::parse(&s("rate")?)?, prev: h("prev")?,
            };
            l.entries.push(Banked { entry, hash: h("hash")? });
        }
        Ok(l)
    }
}

/// The ledger file `bankml serve` banks to (`BANKML_LEDGER`, or `--ledger FILE`, which sets it): its next seq and head,
/// read and verified once; `None` while unset (no file is written unless asked for).
static SERVED: std::sync::Mutex<Option<(String, u64, [u8; 32])>> = std::sync::Mutex::new(None);

/// Bank one served receipt to the ledger file, if one is configured (`BANKML_LEDGER`; the rate per token from
/// `BANKML_RATE`, 18 decimals, default 0). The file is verified in full the first time; a broken chain is never appended
/// to (the error says where it breaks). Each entry is appended as one line.
pub fn bank_served(source: &str, model: &str, request_sha256: &str, response_sha256: &str, prompt: u64, completion: u64) -> Result<(), String> {
    let Ok(path) = std::env::var("BANKML_LEDGER") else { return Ok(()) };
    if path.is_empty() {
        return Ok(());
    }
    let rate = match std::env::var("BANKML_RATE") { Ok(r) => Amount::parse(&r)?, Err(_) => Amount::ZERO };
    let mut g = SERVED.lock().unwrap_or_else(|e| e.into_inner());
    if g.as_ref().is_none_or(|(p, _, _)| *p != path) {
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let l = Ledger::from_jsonl(&text)?;
        l.verify().map_err(|e| format!("{path}: {e}; not appending to a broken ledger"))?;
        *g = Some((path.clone(), l.entries.len() as u64, l.head()));
    }
    let (_, seq, head) = g.as_mut().expect("set above");
    let at_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let entry = Entry { seq: *seq, at_ms, source: source.into(), model: model.into(), request_sha256: request_sha256.into(),
                        response_sha256: response_sha256.into(), prompt_tokens: prompt, completion_tokens: completion, rate, prev: *head };
    let b = Banked { hash: entry.hash()?, entry };
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).map_err(|e| format!("{path}: {e}"))?;
    writeln!(f, "{}", b.line()?).map_err(|e| format!("{path}: {e}"))?;
    *seq += 1;
    *head = b.hash;
    Ok(())
}

/// The blockchain, as a module: something that anchors a ledger's head (THOT: a commitment in an agent's bundle; iNFT:
/// an unsigned transaction for the owner to sign). bankML calls it; it never signs or sends on its own.
pub trait Anchor {
    fn name(&self) -> &str;
    /// Prepare the anchoring of `head` over `entries` entries; returns what was prepared (a commitment, an unsigned
    /// transaction), for the owner to act on.
    fn anchor(&self, head: [u8; 32], entries: u64) -> Result<String, String>;
}

/// The default: no chain. The ledger stands alone and verifies itself.
pub struct NoAnchor;
impl Anchor for NoAnchor {
    fn name(&self) -> &str {
        "none"
    }
    fn anchor(&self, head: [u8; 32], entries: u64) -> Result<String, String> {
        Ok(format!("no chain connected: head 0x{} over {entries} entries, kept in the ledger", hex(&head)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keccak256_is_the_evms() {
        assert_eq!(hex(&keccak256(b"")), "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
        assert_eq!(hex(&keccak256(b"abc")), "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
        // across a block boundary (136 bytes) and exactly one block
        let long = vec![b'a'; 200];
        assert_eq!(hex(&keccak256(&long)), "96ea54061def936c4be90b518992fdc6f12f535068a256229aca54267b4d084d"); // sAGI/agents.py's
        assert_ne!(keccak256(&[0u8; 135]), keccak256(&[0u8; 136]));
        // Solidity: keccak256("transfer(address,uint256)")[:4] = a9059cbb
        assert_eq!(&hex(&keccak256(b"transfer(address,uint256)"))[..8], "a9059cbb");
    }

    #[test]
    fn amounts_are_exact_to_18_decimals() {
        assert_eq!(Amount::parse("1.5").unwrap().to_string(), "1.500000000000000000");
        assert_eq!(Amount::parse("0.000000000000000001").unwrap(), Amount(1));
        assert_eq!(Amount::parse("123").unwrap(), Amount::units(123));
        assert!(Amount::parse("0.0000000000000000001").is_err()); // 19 decimals
        assert!(Amount::parse("-1").is_err() && Amount::parse("1.").is_err() && Amount::parse(".5").is_err() && Amount::parse("1e3").is_err());
        assert!(Amount::parse("340282366920938463464").is_err()); // past u128 / 10¹⁸
        // 0.1 + 0.2 is 0.3 here (it is not in floating point)
        let s = Amount::parse("0.1").unwrap().checked_add(Amount::parse("0.2").unwrap()).unwrap();
        assert_eq!(s, Amount::parse("0.3").unwrap());
        // a rate times tokens, exact; overflow refused
        assert_eq!(Amount::parse("0.000001").unwrap().times(1_234_567).unwrap().to_string(), "1.234567000000000000");
        assert!(Amount(u128::MAX).times(2).is_none());
        for s in ["0.000000000000000000", "7.000000000000000001", "18446744073709551615.123456789012345678"] {
            assert_eq!(Amount::parse(s).unwrap().to_string(), s);
        }
    }

    fn entry(source: &str, p: u64, c: u64, rate: &str) -> Entry {
        Entry { seq: 0, at_ms: 1_760_400_000_000, source: source.into(), model: "3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3".into(),
                request_sha256: "aa".repeat(32), response_sha256: "bb".repeat(32), prompt_tokens: p, completion_tokens: c, rate: Amount::parse(rate).unwrap(), prev: GENESIS }
    }

    #[test]
    fn the_ledger_chains_verifies_and_totals_exactly() {
        let mut l = Ledger::default();
        l.bank(entry("bankml", 53, 64, "0")).unwrap();
        l.bank(entry("external:openai", 1000, 250, "0.000002")).unwrap();
        l.bank(entry("bankml", 85, 64, "0")).unwrap();
        assert!(l.verify().is_ok());
        assert_eq!(l.entries[1].entry.prev, l.entries[0].hash);
        let t = l.totals().unwrap();
        assert_eq!(t["*"].entries, 3);
        assert_eq!(t["*"].tokens.to_string(), "1516.000000000000000000");
        assert_eq!(t["external:openai"].cost.to_string(), "0.002500000000000000");
        assert_eq!(t["bankml"].cost, Amount::ZERO);
        // round trip through JSON lines, then tamper: the chain says where
        let text = l.to_jsonl().unwrap();
        let back = Ledger::from_jsonl(&text).unwrap();
        assert!(back.verify().is_ok());
        assert_eq!(back.head(), l.head());
        let bad = Ledger::from_jsonl(&text.replacen("\"completion_tokens\":250", "\"completion_tokens\":25", 1)).unwrap();
        assert!(bad.verify().unwrap_err().starts_with("entry 1:"));
        // the default anchor connects no chain
        assert!(NoAnchor.anchor(l.head(), 3).unwrap().starts_with("no chain connected"));
    }

    /// The vector the browser must reproduce byte for byte (browseML's ledger test reads it).
    #[test]
    fn the_shared_vector() {
        let mut l = Ledger::default();
        l.bank(entry("bankml", 53, 64, "0")).unwrap();
        l.bank(entry("external:openai", 1000, 250, "0.000002")).unwrap();
        let text = l.to_jsonl().unwrap();
        eprintln!("ledger vector:\n{text}head {}", hex(&l.head()));
        assert!(text.starts_with("{\"v\":1,\"seq\":0,\"at_ms\":1760400000000,\"source\":\"bankml\""));
        // pinned: the Rust keccak256 agreed with sAGI/agents.py's on entry 0 (2026-10-14); the browser must give these
        assert_eq!(hex(&l.entries[0].hash), "783acbe6f272321c0653a29fb1e310d4c251651c217405bb9783a9bf296d4f30");
        assert_eq!(hex(&l.head()), "7b6663da1634f8da5559fec8b8675c5c798d135d49261b6edfeb44cb4cc40a65");
    }
}
