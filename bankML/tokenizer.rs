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
//!
//! 0.3.4 (O4): the `smollm` pre-tokenizer (SmolLM2, mindX's `mindx-genN`), read from llama-vocab.cpp and unicode.cpp
//! b11192: two passes, as llama.cpp runs them. First `\p{N}` cuts every digit out on its own (the `std::regex` pass
//! over the collapsed text: ASCII digits, and non-ASCII codepoints whose only category is Number and that are not
//! whitespace); then GPT-2's pattern, `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`, runs
//! inside each piece (`unicode_regex_split_custom_gpt2`: the end of a piece is the end of its text, so a space before
//! a digit stays alone). Its oracle is llama-server's `/tokenize` on the SmolLM2 vocabulary.

use crate::gguf::Mmap;
use crate::unicode_letters::is_letter;
use std::collections::HashMap;
use std::path::Path;

/// The pre-tokenizers this module reproduces (`tokenizer.ggml.pre`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pre {
    /// `qwen2`: Qwen3, Bonsai
    Qwen2,
    /// `smollm`: SmolLM2 and mindX's lineage built on it
    Smollm,
}

/// Whether a header's vocabulary is one this module reproduces (byte-level BPE with a known pre-tokenizer).
pub fn check_gguf(h: &crate::gguf::Header) -> Result<(), String> {
    let s = |k: &str| match h.kv.get(k) {
        Some(crate::gguf::Val::S(v)) => v.clone(),
        _ => String::new(),
    };
    pre_of(&s("tokenizer.ggml.model"), &s("tokenizer.ggml.pre")).map(|_| ())
}

fn pre_of(model: &str, pre: &str) -> Result<Pre, String> {
    match (model, pre) {
        ("gpt2", "qwen2") => Ok(Pre::Qwen2),
        ("gpt2", "smollm") => Ok(Pre::Smollm),
        _ => Err(format!("tokenizer {model:?}/{pre:?}: the tokenizers reproduced are gpt2/qwen2 (Qwen3, Bonsai) and gpt2/smollm (SmolLM2)")),
    }
}

pub struct Tokenizer {
    pub tokens: Vec<String>,
    pub pre: Pre,
    types: Vec<i32>,
    ids: HashMap<String, u32>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// each byte's token, or `NO_TOKEN` when the vocabulary has none (SmolLM2 lacks 21 of them)
    byte_id: [u32; 256],
    /// (text, id, type), longest first
    specials: Vec<(String, u32, i32)>,
}

const CONTROL: i32 = 3;

/// llama-vocab.cpp b11192's end-of-generation names: each one in the vocabulary ends generation, and is made a
/// CONTROL token if the file says otherwise ("control-looking token … its type will be overridden").
const EOG_NAMES: [&str; 22] = ["<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>", "<end_of_turn>",
    "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>", "<end_of_utterance>", "<eos>",
    "<turn|>", "<|tool_response>", "<｜end▁of▁sentence｜>", "[e~["];

/// llama-vocab.cpp b11192's fill-in-the-middle names, by role (prefix, suffix, middle, pad, repo, file separator): the
/// one found becomes a CONTROL token; pad, repo and separator also end generation.
const FIM_NAMES: [&[&str]; 6] = [
    &["<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<｜fim▁begin｜>", "<PRE>", "▁<PRE>", "<|code_prefix|>", "<|prefix|>"],
    &["<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<｜fim▁hole｜>", "<SUF>", "▁<SUF>", "<|code_suffix|>", "<|suffix|>"],
    &["<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<｜fim▁end｜>", "<MID>", "▁<MID>", "<|code_middle|>", "<|middle|>"],
    &["<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"],
    &["<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>"],
    &["<|file_sep|>"],
];

/// The token types llama.cpp actually uses: llama-vocab.cpp b11192 overrides the file's by text after loading. Every
/// end-of-generation name and the fill-in-the-middle token of each role become CONTROL (so `</s>` in the Qwen3
/// vocabulary, NORMAL in the file, is parsed as a special token and renders as nothing); gpt-oss's four channel
/// markers become USER_DEFINED. A role with two candidates is found in hash-map order there, so such a vocabulary is
/// refused rather than guessed.
fn llama_vocab_types(tokens: &[String], types: &mut [i32]) -> Result<(), String> {
    let ids: HashMap<&str, usize> = tokens.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();
    for names in FIM_NAMES {
        let found: Vec<usize> = names.iter().filter_map(|n| ids.get(n).copied()).collect();
        if found.len() > 1 && found.iter().any(|&i| types[i] != CONTROL) {
            return Err(format!("the vocabulary has {} candidates for one fill-in-the-middle role ({}); llama.cpp picks one in hash order: not reproduced",
                               found.len(), found.iter().map(|&i| tokens[i].as_str()).collect::<Vec<_>>().join(", ")));
        }
        for i in found {
            types[i] = CONTROL;
        }
    }
    for n in EOG_NAMES {
        if let Some(&i) = ids.get(n) {
            types[i] = CONTROL;
        }
    }
    for n in ["<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"] {
        if let Some(&i) = ids.get(n) {
            types[i] = USER_DEFINED;
        }
    }
    Ok(())
}
/// a byte the vocabulary has no token for
const NO_TOKEN: u32 = u32::MAX;
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
                (k, _) if k.starts_with("tokenizer.ggml.fim_") => {
                    return Err(format!("{k} in the header: llama.cpp then skips its detection by text for that role; not reproduced"))
                }
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
        let pre = pre_of(&model, &pre)?;
        let mut t = Self::new(tokens, types, &merges)?;
        t.pre = pre;
        Ok(t)
    }

    pub fn new(tokens: Vec<String>, mut types: Vec<i32>, merges: &[String]) -> Result<Self, String> {
        if types.len() != tokens.len() {
            return Err("token_type and tokens differ in length".into());
        }
        llama_vocab_types(&tokens, &mut types)?;
        let ids: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let mut byte_id = [NO_TOKEN; 256];
        for (b, c) in byte_chars().iter().enumerate() {
            byte_id[b] = ids.get(&c.to_string()).copied().unwrap_or(NO_TOKEN);
        }
        let mut mm = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let (a, b) = m.split_once(' ').ok_or_else(|| format!("merge {rank} has no space: {m:?}"))?;
            match (ids.get(a), ids.get(b), ids.get(&format!("{a}{b}"))) {
                (Some(&ia), Some(&ib), Some(&im)) => {
                    mm.entry((ia, ib)).or_insert((rank as u32, im));
                }
                // llama.cpp merges by text, so a merge naming a string outside the vocabulary still applies there;
                // merging by id here would differ, so such a vocabulary is refused (none of the pinned ones has one)
                _ => return Err(format!("merge {rank} ({m:?}) names a string outside the vocabulary: not reproduced")),
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
        Ok(Tokenizer { tokens, pre: Pre::Qwen2, types, ids, merges: mm, byte_id, specials })
    }

    /// Token ids of `text`, as llama.cpp's `llama_tokenize(…, add_special = false, parse_special)`.
    /// A token's bytes, as llama.cpp's detokenizer gives them without special tokens: a normal token's GPT-2
    /// characters mapped back to bytes, a user-defined token's text as written, a control token nothing.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else { return Vec::new() };
        match self.types.get(id as usize) {
            Some(3) => Vec::new(),
            Some(4) => t.as_bytes().to_vec(),
            _ => {
                let chars = byte_chars();
                t.chars().map(|c| chars.iter().position(|&b| b == c).map(|b| b as u8).unwrap_or(b'?')).collect()
            }
        }
    }

    /// A token's piece as llama.cpp's `token_to_piece(…, special = true)` gives it, the bytes its grammar matches:
    /// control, user-defined and unknown tokens as written; a normal token's GPT-2 characters mapped back to bytes
    /// (`llama_decode_text`, with its `[UNK_BYTE_0x…]` for a character outside the map); anything else nothing.
    pub fn piece(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else { return Vec::new() };
        match self.types.get(id as usize) {
            Some(2..=4) => t.as_bytes().to_vec(),
            Some(1) => {
                let chars = byte_chars();
                let mut out = Vec::with_capacity(t.len());
                for c in t.chars() {
                    match chars.iter().position(|&b| b == c) {
                        Some(b) => out.push(b as u8),
                        None => {
                            let mut u = [0u8; 4];
                            out.extend(b"[UNK_BYTE_0x");
                            for b in c.encode_utf8(&mut u).bytes() {
                                out.extend(format!("{b:02x}").bytes());
                            }
                            out.extend(t.bytes());
                            out.push(b']');
                        }
                    }
                }
                out
            }
            _ => Vec::new(),
        }
    }

    /// The tokens that end generation, as llama-vocab.cpp b11192 collects them: the FIM pad / repo / file-separator
    /// tokens, every token named in its end-of-turn list, and the GGUF's own eos / eot / eom ids.
    pub fn eog_ids(&self, kv_ids: &[u32]) -> Vec<u32> {
        let mut out: Vec<u32> = FIM_NAMES[3..].iter().filter_map(|names| names.iter().find_map(|n| self.id(n))).collect();
        out.extend(EOG_NAMES.iter().filter_map(|n| self.id(n)));
        out.extend(kv_ids.iter().copied().filter(|&i| (i as usize) < self.tokens.len()));
        let text = |i: &u32| self.tokens[*i as usize].as_str();
        // b11192's two exceptions: <|end|> is not an end when <|return|> and <|call|> (or <|calls|>, <|flush|>) are;
        // </s> is not when <|tool_response> is
        let has = |out: &[u32], s: &str| out.iter().any(|i| text(i) == s);
        let call = has(&out, "<|call|>") || has(&out, "<|calls|>");
        if call && has(&out, "<|end|>") && (has(&out, "<|return|>") || has(&out, "<|flush|>")) {
            out.retain(|i| text(i) != "<|end|>");
        }
        if has(&out, "<|tool_response>") && has(&out, "</s>") {
            out.retain(|i| text(i) != "</s>");
        }
        out.sort_unstable();
        out.dedup();
        out
    }

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
                    let pieces = match self.pre {
                        Pre::Qwen2 => pretokenize(raw),
                        Pre::Smollm => pretokenize_smollm(raw),
                    };
                    for piece in pieces {
                        self.bpe(piece, &mut out);
                    }
                }
            }
        }
        out
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        // a byte without a token never merges (no merge names it: checked at load) and is dropped at the end, as
        // llama-vocab's fallback drops it: none of its UTF-8 bytes is a token on its own
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
        out.extend(syms.into_iter().filter(|&t| t != NO_TOKEN));
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

// ---------------------------------------------------------------- SmolLM2's pre-tokenizer, written out ----------

/// `\p{N}` as llama.cpp's first pass sees it on the collapsed text: an ASCII digit, or a non-ASCII codepoint that is
/// a Number and not whitespace.
fn is_digit_piece(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_digit() } else { !c.is_whitespace() && is_num(c) }
}

/// Split `text` as llama.cpp's `smollm` pre-tokenizer does: digits cut out one by one, then GPT-2's pattern inside
/// each remaining piece.
pub fn pretokenize_smollm(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if is_digit_piece(c) {
            if start < i {
                gpt2_split(&text[start..i], &mut out);
            }
            out.push(&text[i..i + c.len_utf8()]);
            start = i + c.len_utf8();
        }
    }
    if start < text.len() {
        gpt2_split(&text[start..], &mut out);
    }
    out
}

/// `unicode_regex_split_custom_gpt2` over one piece (the piece's end is the text's end for every lookahead).
fn gpt2_split<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| if i < n { Some(cs[i].1) } else { None };
    let off = |i: usize| if i < n { cs[i].0 } else { text.len() };
    let letter = |i: usize| at(i).is_some_and(is_letter);
    let number = |i: usize| at(i).is_some_and(is_num);
    let space = |i: usize| at(i).is_some_and(char::is_whitespace);
    // [^\s\p{L}\p{N}]: every codepoint in range has flags, so only the end of the piece fails it
    let other = |i: usize| at(i).is_some_and(|c| !c.is_whitespace() && !is_letter(c) && !is_num(c));
    let (mut pos, mut prev) = (0usize, 0usize);
    let mut emit = |end: usize, prev: &mut usize| {
        if end > *prev {
            out.push(&text[off(*prev)..off(end)]);
        }
        *prev = end;
    };
    while pos < n {
        let c = cs[pos].1;
        if c == '\'' && pos + 1 < n {
            match (at(pos + 1), at(pos + 2)) {
                (Some('s' | 't' | 'm' | 'd'), _) => {
                    pos += 2;
                    emit(pos, &mut prev);
                    continue;
                }
                (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => {
                    pos += 3;
                    emit(pos, &mut prev);
                    continue;
                }
                _ => {}
            }
        }
        let p2 = if c == ' ' { pos + 1 } else { pos };
        if letter(p2) || number(p2) || other(p2) {
            let class = |i: usize| if letter(p2) { letter(i) } else if number(p2) { number(i) } else { other(i) };
            pos = p2;
            while class(pos) {
                pos += 1;
            }
            emit(pos, &mut prev);
            continue;
        }
        let mut ws = 0;
        while space(pos + ws) {
            ws += 1;
        }
        if ws > 1 && pos + ws < n {
            pos += ws - 1; // \s+(?!\S): leave the last whitespace for the next piece
        } else if ws > 0 {
            pos += ws;
        } else {
            pos += 1;
        }
        emit(pos, &mut prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smollm_pretokenizer_shapes() {
        assert_eq!(pretokenize_smollm("Hello world"), ["Hello", " world"]);
        assert_eq!(pretokenize_smollm("it's DON'T"), ["it", "'s", " DON", "'", "T"]); // GPT-2's contractions are lower-case only
        assert_eq!(pretokenize_smollm("a 12b"), ["a", " ", "1", "2", "b"]); // a digit never takes the space
        assert_eq!(pretokenize_smollm("x\n\n  y"), ["x", "\n\n ", " y"]);
        assert_eq!(pretokenize_smollm("end  "), ["end", "  "]);
        assert_eq!(pretokenize_smollm("!!! ok"), ["!!!", " ok"]);
        assert_eq!(pretokenize_smollm("<|im_start|>"), ["<|", "im", "_", "start", "|>"]);
    }

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
        tokenizer_oracle("Bonsai-1.7B-Q1_0.gguf", "oracle-tokenizer");
    }

    /// 0.3.4: SmolLM2's vocabulary and the `smollm` pre-tokenizer, against llama-server running SmolLM2.
    #[test]
    #[ignore = "needs .models/SmolLM2-135M-Instruct-F16.gguf + .models/oracle-tokenizer-smollm (testing/tokenizer_oracle.py); --release"]
    fn oracle_tokenizer_smollm() {
        tokenizer_oracle("SmolLM2-135M-Instruct-F16.gguf", "oracle-tokenizer-smollm");
    }

    fn tokenizer_oracle(gguf: &str, sub: &str) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let t = Tokenizer::from_gguf(&dir.join(gguf)).unwrap();
        let cases = std::fs::read_to_string(dir.join(sub).join("cases.jsonl")).unwrap();
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
        eprintln!("tokenizer oracle ({gguf}, {:?}): {} of {n} cases token-identical to llama.cpp b11192", t.pre, n - bad.len());
        for b in bad.iter().take(10) {
            eprintln!("  differs: {b}");
        }
        assert!(bad.is_empty());
    }
}
