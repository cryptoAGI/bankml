// SPDX-License-Identifier: MIT OR Apache-2.0
//! P3, step one — the tokenizer: GGUF `gpt2`-model byte-level BPE with the `qwen2` pre-tokenizer (Qwen3, Bonsai),
//! with no crates, **token-identical to llama.cpp b11192** (the oracle: `testing/tokenizer_oracle.py` records the
//! engine's own `/tokenize`; `oracle_tokenizer` re-derives every case).
//!
//! What llama.cpp does, and this reproduces:
//! 1. **Special tokens first.** Tokens whose type is CONTROL (3) or USER_DEFINED (4) are cut out of the text, the
//!    longest first; without `parse_special` only USER_DEFINED ones are (llama-vocab's `tokenizer_st_partition`).
//! 2. **Pre-tokenize** each remaining span with Qwen2's pattern, whose alternatives are tried in order at each
//!    position (ECMAScript semantics, backtracking results written out below):
//!    `'s|'t|'re|'ve|'m|'ll|'d` (either case) · `[^\r\n\p{L}\p{N}]?\p{L}+` · `\p{N}` · ` ?[^\s\p{L}\p{N}]+[\r\n]*` ·
//!    `\s*[\r\n]+` · `\s+(?!\S)` · `\s+`.
//! 3. **Byte-level BPE** on each piece: its UTF-8 bytes become GPT-2's printable byte characters, each a token; the
//!    adjacent pair with the lowest merge rank is merged (the leftmost on ties) until none is left.

use crate::gguf::Mmap;
use crate::unicode_letters::is_letter;
use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    pub tokens: Vec<String>,
    types: Vec<i32>,
    ids: HashMap<String, u32>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_id: [u32; 256],
    /// (text, id, type), longest first
    specials: Vec<(String, u32, i32)>,
}

const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;

/// GPT-2's map from each byte to a printable character (`bytes_to_unicode`).
fn byte_chars() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    out
}

// ---------------------------------------------------------------- reading the vocabulary from GGUF ----------

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.o.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("GGUF header truncated")?;
        let s = &self.b[self.o..end];
        self.o = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn len(&mut self) -> Result<usize, String> {
        let n = self.u64()?;
        usize::try_from(n).ok().filter(|&n| n <= self.b.len()).ok_or_else(|| format!("GGUF length {n} exceeds the file"))
    }
    fn str(&mut self) -> Result<String, String> {
        let n = self.len()?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn skip(&mut self, t: u32, depth: u32) -> Result<(), String> {
        match t {
            0 | 1 | 7 => self.take(1).map(|_| ()),
            2 | 3 => self.take(2).map(|_| ()),
            4..=6 => self.take(4).map(|_| ()),
            10..=12 => self.take(8).map(|_| ()),
            8 => self.str().map(|_| ()),
            9 if depth < 4 => {
                let (et, n) = (self.u32()?, self.len()?);
                for _ in 0..n {
                    self.skip(et, depth + 1)?;
                }
                Ok(())
            }
            _ => Err(format!("unsupported GGUF value type {t}")),
        }
    }
}

impl Tokenizer {
    /// The vocabulary of a GGUF file (`tokenizer.ggml.{model,pre,tokens,token_type,merges}`); refused unless it is
    /// the `gpt2` model with the `qwen2` pre-tokenizer, the one this module reproduces.
    pub fn from_gguf(path: &Path) -> Result<Self, String> {
        let mm = Mmap::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut r = Rd { b: mm.bytes(), o: 0 };
        if r.take(4)? != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        let (_version, _nt, nkv) = (r.u32()?, r.u64()?, r.u64()?);
        let (mut model, mut pre) = (String::new(), String::new());
        let (mut tokens, mut types, mut merges) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..nkv {
            let k = r.str()?;
            let t = r.u32()?;
            match (k.as_str(), t) {
                ("tokenizer.ggml.model", 8) => model = r.str()?,
                ("tokenizer.ggml.pre", 8) => pre = r.str()?,
                ("tokenizer.ggml.tokens" | "tokenizer.ggml.merges", 9) => {
                    let (et, n) = (r.u32()?, r.len()?);
                    if et != 8 {
                        return Err(format!("{k}: expected strings, found type {et}"));
                    }
                    let v = if k.ends_with("tokens") { &mut tokens } else { &mut merges };
                    for _ in 0..n {
                        v.push(r.str()?);
                    }
                }
                ("tokenizer.ggml.token_type", 9) => {
                    let (et, n) = (r.u32()?, r.len()?);
                    if et != 5 {
                        return Err(format!("{k}: expected i32, found type {et}"));
                    }
                    for _ in 0..n {
                        types.push(r.u32()? as i32);
                    }
                }
                _ => r.skip(t, 0)?,
            }
        }
        if model != "gpt2" || pre != "qwen2" {
            return Err(format!("tokenizer {model:?}/{pre:?}: only gpt2/qwen2 (Qwen3, Bonsai) is implemented"));
        }
        Self::new(tokens, types, &merges)
    }

    pub fn new(tokens: Vec<String>, types: Vec<i32>, merges: &[String]) -> Result<Self, String> {
        if types.len() != tokens.len() {
            return Err("token_type and tokens differ in length".into());
        }
        let ids: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let mut byte_id = [0u32; 256];
        for (b, c) in byte_chars().iter().enumerate() {
            byte_id[b] = *ids.get(&c.to_string()).ok_or_else(|| format!("no token for byte {b:#04x}"))?;
        }
        let mut mm = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let (a, b) = m.split_once(' ').ok_or_else(|| format!("merge {rank} has no space: {m:?}"))?;
            if let (Some(&ia), Some(&ib), Some(&im)) = (ids.get(a), ids.get(b), ids.get(&format!("{a}{b}"))) {
                mm.entry((ia, ib)).or_insert((rank as u32, im));
            }
        }
        let mut specials: Vec<(String, u32, i32)> = tokens
            .iter()
            .zip(&types)
            .enumerate()
            .filter(|(_, (_, &ty))| ty == CONTROL || ty == USER_DEFINED)
            .map(|(i, (t, &ty))| (t.clone(), i as u32, ty))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        Ok(Tokenizer { tokens, types, ids, merges: mm, byte_id, specials })
    }

    /// Token ids of `text`, as llama.cpp's `llama_tokenize(…, add_special = false, parse_special)`.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        // 1. cut out special tokens, longest first, from the spans that are still raw text
        let mut parts: Vec<Result<u32, &str>> = vec![Err(text)];
        for (s, id, ty) in &self.specials {
            if s.is_empty() || (!parse_special && *ty != USER_DEFINED) {
                continue;
            }
            let mut next = Vec::with_capacity(parts.len());
            for p in parts {
                match p {
                    Err(raw) if raw.contains(s.as_str()) => {
                        let mut rest = raw;
                        while let Some(i) = rest.find(s.as_str()) {
                            if i > 0 {
                                next.push(Err(&rest[..i]));
                            }
                            next.push(Ok(*id));
                            rest = &rest[i + s.len()..];
                        }
                        if !rest.is_empty() {
                            next.push(Err(rest));
                        }
                    }
                    other => next.push(other),
                }
            }
            parts = next;
        }
        // 2 and 3. pre-tokenize the raw spans; BPE each piece
        let mut out = Vec::new();
        for p in parts {
            match p {
                Ok(id) => out.push(id),
                Err(raw) => {
                    for piece in pretokenize(raw) {
                        self.bpe(piece, &mut out);
                    }
                }
            }
        }
        out
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let mut syms: Vec<u32> = piece.bytes().map(|b| self.byte_id[b as usize]).collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None; // (rank, position, merged id)
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&(rank, id)) = self.merges.get(&(syms[i], syms[i + 1])) {
                    if best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, i, id));
                    }
                }
            }
            match best {
                Some((_, i, id)) => {
                    syms[i] = id;
                    syms.remove(i + 1);
                }
                None => break,
            }
        }
        out.extend(syms);
    }

    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_type(&self, id: u32) -> Option<i32> {
        self.types.get(id as usize).copied()
    }

    pub fn id(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }
}

// ---------------------------------------------------------------- Qwen2's pre-tokenizer, written out ----------

fn is_num(c: char) -> bool {
    c.is_numeric() // Nd, Nl, No: \p{N}
}

/// Split `text` into pieces as Qwen2's pattern does, alternative by alternative, first match wins.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| if i < n { Some(cs[i].1) } else { None };
    let off = |i: usize| if i < n { cs[i].0 } else { text.len() };
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        let end = (|| {
            // 1. contractions: 's 't 're 've 'm 'll 'd, either case
            if c == '\'' {
                let l = |k: usize| at(i + k).map(|c| c.to_ascii_lowercase());
                match (l(1), l(2)) {
                    (Some('s' | 't' | 'm' | 'd'), _) => return i + 2,
                    (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => return i + 3,
                    _ => {}
                }
            }
            // 2. [^\r\n\p{L}\p{N}]?\p{L}+
            let lead = c != '\r' && c != '\n' && !is_letter(c) && !is_num(c);
            let s = if lead && at(i + 1).is_some_and(is_letter) { i + 1 } else { i };
            if at(s).is_some_and(is_letter) {
                let mut j = s;
                while at(j).is_some_and(is_letter) {
                    j += 1;
                }
                return j;
            }
            // 3. \p{N}
            if is_num(c) {
                return i + 1;
            }
            // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
            let other = |c: char| !c.is_whitespace() && !is_letter(c) && !is_num(c);
            let s = if c == ' ' && at(i + 1).is_some_and(other) { i + 1 } else { i };
            if at(s).is_some_and(other) {
                let mut j = s;
                while at(j).is_some_and(other) {
                    j += 1;
                }
                while matches!(at(j), Some('\r' | '\n')) {
                    j += 1;
                }
                return j;
            }
            // the whitespace run starting here
            let mut j = i;
            while at(j).is_some_and(char::is_whitespace) {
                j += 1;
            }
            // 5. \s*[\r\n]+ : up to and including the run's last newline
            if let Some(k) = (i..j).rev().find(|&k| matches!(cs[k].1, '\r' | '\n')) {
                return k + 1;
            }
            // 6. \s+(?!\S) : the whole run at the end of the text, else all but its last character
            if j == n {
                return j;
            }
            if j - i >= 2 {
                return j - 1;
            }
            // 7. \s+
            j.max(i + 1)
        })();
        out.push(&text[off(i)..off(end)]);
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenizer_shapes() {
        assert_eq!(pretokenize("Hello world"), ["Hello", " world"]);
        assert_eq!(pretokenize("it's DON'T"), ["it", "'s", " DON", "'T"]);
        assert_eq!(pretokenize("a  b"), ["a", " ", " b"]);
        assert_eq!(pretokenize("x\n\n  y"), ["x", "\n\n", " ", " y"]); // the space before a word joins the word
        assert_eq!(pretokenize("123"), ["1", "2", "3"]);
        assert_eq!(pretokenize("end  "), ["end", "  "]);
        assert_eq!(pretokenize("!!! ok"), ["!!!", " ok"]);
        assert_eq!(pretokenize(""), Vec::<&str>::new());
    }

    #[test]
    fn byte_chars_are_gpt2s() {
        let b = byte_chars();
        assert_eq!((b[b'!' as usize], b[b' ' as usize], b[b'\n' as usize], b[0]), ('!', 'Ġ', 'Ċ', 'Ā'));
    }

    /// Every case llama.cpp b11192's own /tokenize produced (testing/tokenizer_oracle.py), token for token.
    #[test]
    #[ignore = "needs .models/Bonsai-1.7B-Q1_0.gguf + .models/oracle-tokenizer (testing/tokenizer_oracle.py); --release"]
    fn oracle_tokenizer() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let t = Tokenizer::from_gguf(&dir.join("Bonsai-1.7B-Q1_0.gguf")).unwrap();
        let cases = std::fs::read_to_string(dir.join("oracle-tokenizer/cases.jsonl")).unwrap();
        let (mut n, mut bad) = (0, Vec::new());
        for line in cases.lines() {
            let v = crate::serve::Json::parse(line).unwrap();
            let text = v.get("text").and_then(crate::serve::Json::as_str).unwrap().to_string();
            let sp = v.get("parse_special").and_then(crate::serve::Json::as_bool).unwrap();
            let want: Vec<u32> = match v.get("ids") {
                Some(crate::serve::Json::Arr(a)) => a.iter().map(|x| x.as_u64().unwrap() as u32).collect(),
                _ => panic!("ids"),
            };
            let got = t.encode(&text, sp);
            n += 1;
            if got != want {
                bad.push(format!("parse_special={sp} {:?}…: got {} tokens, llama.cpp {}", text.chars().take(40).collect::<String>(), got.len(), want.len()));
            }
        }
        eprintln!("tokenizer oracle: {} of {n} cases token-identical to llama.cpp b11192", n - bad.len());
        for b in bad.iter().take(10) {
            eprintln!("  differs: {b}");
        }
        assert!(bad.is_empty());
    }
}
