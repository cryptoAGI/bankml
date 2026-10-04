// SPDX-License-Identifier: MIT OR Apache-2.0
//! `bankml convert` (O5): a Hugging Face safetensors directory → a GGUF, for the **Llama architecture as SmolLM2-135M
//! and mindX's `mindx-genN` use it**, written to be **byte-identical** to llama.cpp b11192's
//! `convert_hf_to_gguf.py DIR --outtype f16` (read from the tag's `conversion/{base,llama}.py` and `gguf-py`):
//!
//! - **metadata, in llama.cpp's order**: `general.architecture`; `general.type`; the sampling defaults of
//!   `generation_config.json` (`general.sampling.*`); the name heuristics of `gguf-py/metadata.py` over `_name_or_path`
//!   and then the directory's name (`general.name`, `organization`, `finetune`, `basename`, `version`, `size_label`, the
//!   size label otherwise from the weights counted); `TextModel.set_gguf_parameters` with the defaults
//!   `AutoConfig(LlamaConfig).to_dict()` supplies (`head_dim`, `rope_theta` 10000, `rms_norm_eps` 1e-6, kv heads);
//!   `file_type` 1; Llama's `vocab_size` and `rope.dimension_count`; `quantization_version` 2; then the tokenizer;
//! - **tensors**: every safetensors part in name order (the reader sorts by name), HF names mapped to GGUF names, the
//!   Q and K rows permuted from HF's rotate-half layout back to GGML's pairs (`LlamaModel.permute`), 1-D tensors and
//!   `*_norm.weight` in F32, every other matrix in F16 by round-to-nearest-even (numpy's `astype(float16)`), BF16 read
//!   exactly as F32 first; tied embeddings write no `output.weight`;
//! - **tokenizer**: the `gpt2` (byte-level BPE) path — tokens in id order with their types, merges, the special token
//!   ids (`tokenizer_config.json`, then `config.json`), `add_*_token`, the chat template (`tokenizer_config.json` or
//!   `chat_template.jinja`), and Llama's `add_bos_token = false` for 49,152-token vocabularies;
//! - **layout**: GGUF v3, little-endian, alignment 32, tensor data in the order the tensors were added.
//!
//! What would make llama.cpp write something this does not reproduce is **refused, with the reason** — never
//! approximated: another architecture, a SentencePiece / Llama-HF vocabulary, rope scaling, biases, experts, a model
//! card (`README.md`: llama.cpp would add its licence, tags and base models), and any tokenizer whose pre-tokenizer
//! llama.cpp would not name `smollm`. llama.cpp names it by hashing the token ids its Python tokenizer gives one fixed
//! text (`chkhsh`); bankML has no Python, so it recognises a tokenizer by the sha256 of its vocabulary and merges and
//! the shape of its pre-tokenizer, and the table below records the `chkhsh` each entry was measured to give
//! (`testing/convert_oracle.py --chkhsh`).
//!
//! **Why it exists:** mindXtrain (github.com/Professor-Codephreak/mindXtrain, continued at
//! huggingface.co/PYTHAI/mindXtrain; Apache-2.0) merges each generation's LoRA into safetensors
//! (`ollama_push/merged`) and serves it with `serve --to ollama`; mindX's `promote.py` then layers the persona. This
//! converter and `bankml create` (`create.rs`) take that merged directory instead: one pinned GGUF, the persona a
//! verified layer. llama.cpp's converter names the model from the directory (`general.name`), so the bytes — and the
//! pin — depend on the directory's name, exactly as they do for llama.cpp.

use crate::gguf::jstr;
use crate::serve::Json;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Tokenizers bankML recognises: (the `tokenizer.ggml.pre` llama.cpp writes, sha256 of the vocabulary and merges as
/// `vocab_merges_sha256` computes it, the `chkhsh` llama.cpp b11192's rule gave on it, where it was measured).
pub const KNOWN_TOKENIZERS: [(&str, &str, &str, &str); 1] = [(
    "smollm",
    "60e5ec7477255109c571457df28dbbc3b15a8a3d36d35a5ae7c7c5824479dd2f",
    "855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249",
    "HuggingFaceTB/SmolLM2-135M-Instruct and PYTHAI/mindXascension weights/gen39 (transformers 5.8.0, tokenizers 0.22.2)",
)];

/// Options of `bankml convert` (llama.cpp's flags of the same meaning).
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// `--model-name`: `general.name` instead of the one the heuristics find
    pub model_name: Option<String>,
    /// `--ignore-model-card`: convert although a README.md is present (the result then differs from llama.cpp's)
    pub ignore_model_card: bool,
}

/// What a conversion wrote.
#[derive(Debug, Clone)]
pub struct Report {
    pub out: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub tensors: usize,
    pub kv: usize,
    pub params: u64,
    pub name: String,
    /// sha256 of every input file read, by name (the provenance a FORK.json records)
    pub inputs: Vec<(String, u64, String)>,
}

// ---------------------------------------------------------------- GGUF metadata values -------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Kv {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
    I32s(Vec<i32>),
}

/// llama.cpp's `kv_data`: a dict, so a key set twice keeps its first position and takes the new value.
#[derive(Debug, Default)]
pub struct KvList(pub Vec<(String, Kv)>);

impl KvList {
    fn set(&mut self, k: &str, v: Kv) {
        match self.0.iter_mut().find(|(n, _)| n == k) {
            Some(e) => e.1 = v,
            None => self.0.push((k.to_string(), v)),
        }
    }
    /// `add_string`: an empty string is not written
    fn str(&mut self, k: &str, v: &str) {
        if !v.is_empty() {
            self.set(k, Kv::Str(v.to_string()));
        }
    }
    pub fn get(&self, k: &str) -> Option<&Kv> {
        self.0.iter().find(|(n, _)| n == k).map(|(_, v)| v)
    }
}

fn put_str(o: &mut Vec<u8>, s: &str) {
    o.extend((s.len() as u64).to_le_bytes());
    o.extend(s.as_bytes());
}

fn kv_bytes(kv: &KvList) -> Vec<u8> {
    let mut o = Vec::new();
    for (k, v) in &kv.0 {
        put_str(&mut o, k);
        match v {
            Kv::U32(x) => { o.extend(4u32.to_le_bytes()); o.extend(x.to_le_bytes()) }
            Kv::I32(x) => { o.extend(5u32.to_le_bytes()); o.extend(x.to_le_bytes()) }
            Kv::F32(x) => { o.extend(6u32.to_le_bytes()); o.extend(x.to_le_bytes()) }
            Kv::Bool(x) => { o.extend(7u32.to_le_bytes()); o.push(*x as u8) }
            Kv::Str(s) => { o.extend(8u32.to_le_bytes()); put_str(&mut o, s) }
            Kv::Strs(a) => {
                o.extend(9u32.to_le_bytes());
                o.extend(8u32.to_le_bytes());
                o.extend((a.len() as u64).to_le_bytes());
                a.iter().for_each(|s| put_str(&mut o, s));
            }
            Kv::I32s(a) => {
                o.extend(9u32.to_le_bytes());
                o.extend(5u32.to_le_bytes());
                o.extend((a.len() as u64).to_le_bytes());
                a.iter().for_each(|x| o.extend(x.to_le_bytes()));
            }
        }
    }
    o
}

// ---------------------------------------------------------------- gguf-py/metadata.py, the name heuristics ------

fn is_cased(c: char) -> bool {
    c.is_uppercase() || c.is_lowercase()
}

/// Python `str.islower()`: at least one cased character, none upper-case.
fn py_islower(w: &str) -> bool {
    w.chars().any(is_cased) && !w.chars().any(|c| c.is_uppercase())
}

/// Python `str.title()`: a cased character is upper-cased after an uncased one, lower-cased after a cased one.
fn py_title(w: &str) -> String {
    let mut prev = false;
    let mut o = String::new();
    for c in w.chars() {
        if is_cased(c) {
            if prev { o.extend(c.to_lowercase()) } else { o.extend(c.to_uppercase()) }
            prev = true;
        } else {
            o.push(c);
            prev = false;
        }
    }
    o
}

fn digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `Metadata.id_to_title`: words title-cased unless an acronym or a version number (`v1.2`, or starting with a digit).
pub fn id_to_title(s: &str) -> String {
    // `^(v\d+(?:\.\d+)*|\d.*)$`
    let versionish = |w: &str| w.starts_with(|c: char| c.is_ascii_digit()) || w.strip_prefix('v').is_some_and(|r| r.split('.').all(digits));
    s.trim().replace('-', " ").split_whitespace().map(|w| if py_islower(w) && !versionish(w) { py_title(w) } else { w.to_string() }).collect::<Vec<_>>().join(" ")
}

/// `(v|iter)?\d+([.]\d+)*`, whole, case-insensitive
fn re_version(p: &str) -> bool {
    let l = p.to_ascii_lowercase();
    let r = l.strip_prefix("iter").or_else(|| l.strip_prefix('v')).unwrap_or(&l);
    // the prefix is optional: "v" alone is not a version, and digits must follow
    let ok = |r: &str| !r.is_empty() && r.split('.').all(digits);
    ok(r) || ok(&l)
}

/// `i?q\d(_\w)*|b?fp?(16|32)`, whole, case-insensitive
fn re_type(p: &str) -> bool {
    let l = p.to_ascii_lowercase();
    let q = {
        let r = l.strip_prefix('i').unwrap_or(&l);
        r.strip_prefix('q').is_some_and(|r| {
            let b = r.as_bytes();
            !b.is_empty() && b[0].is_ascii_digit() && {
                let rest = &b[1..];
                rest.len() % 2 == 0 && rest.chunks(2).all(|c| c[0] == b'_' && (c[1].is_ascii_alphanumeric() || c[1] == b'_'))
            }
        })
    };
    let f = {
        let r = l.strip_prefix('b').unwrap_or(&l);
        let r2 = r.strip_prefix('f').map(|x| x.strip_prefix('p').unwrap_or(x));
        // b?f p? (16|32): the f is required
        r2.is_some_and(|x| x == "16" || x == "32")
    };
    q || f
}

/// `(([A]|\d+[x])?\d+([._]\d+)?[KMBT][\d]?|small|mini|medium|large|x?xl)`, whole, case-insensitive
fn re_size(p: &str) -> bool {
    let l = p.to_ascii_lowercase();
    if ["small", "mini", "medium", "large", "xl", "xxl"].contains(&l.as_str()) {
        return true;
    }
    let core = |r: &str| -> bool {
        // \d+([._]\d+)?[kmbt]\d?
        let b = r.as_bytes();
        let mut i = 0;
        while i < b.len() && b[i].is_ascii_digit() { i += 1 }
        if i == 0 { return false }
        if i < b.len() && (b[i] == b'.' || b[i] == b'_') {
            let s = i + 1;
            let mut j = s;
            while j < b.len() && b[j].is_ascii_digit() { j += 1 }
            if j > s { i = j }
        }
        if i >= b.len() || !b"kmbt".contains(&b[i]) { return false }
        i += 1;
        i == b.len() || (i + 1 == b.len() && b[i].is_ascii_digit())
    };
    if core(&l) || l.strip_prefix('a').is_some_and(core) {
        return true;
    }
    // \d+x prefix (any split of the leading digits)
    let b = l.as_bytes();
    (1..b.len()).any(|k| b[..k].iter().all(u8::is_ascii_digit) && b[k] == b'x' && core(&l[k + 1..]))
}

/// The six components `get_model_id_components` returns: full name, organization, basename, finetune, version, size label.
type IdParts = (Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>);

/// `Metadata.get_model_id_components`, ported line for line.
pub fn model_id_components(model_id: &str, total_params: i64) -> IdParts {
    if model_id.contains(' ') {
        return (Some(model_id.to_string()), None, None, None, None, None);
    }
    let (mut org, full) = match model_id.split_once('/') {
        Some((o, n)) => (Some(o.to_string()), n.to_string()),
        None => (None, model_id.to_string()),
    };
    if org.as_deref().is_some_and(|o| o.starts_with('.')) {
        org = None;
    }
    let mut parts: Vec<String> = full.split('-').filter(|p| !p.is_empty()).map(str::to_string).collect();
    let mut types: Vec<Vec<&str>> = vec![Vec::new(); parts.len()];
    let add = |t: &mut Vec<&'static str>, k: &'static str| if !t.contains(&k) { t.push(k) };
    for i in 0..parts.len() {
        let part = parts[i].clone();
        if re_version(&part) {
            add(&mut types[i], "version");
        } else if re_type(&part) {
            add(&mut types[i], "type");
            parts[i] = part.to_uppercase();
        } else if i > 0 && re_size(&part) {
            let mut p: Vec<char> = part.replace('_', ".").chars().collect();
            let n = p.len();
            if p[n - 1].is_ascii_digit() && n >= 2 {
                // bloom-7b1 → 7.1b
                let (a, b) = (p[n - 2], p[n - 1]);
                p.truncate(n - 2);
                p.extend(['.', b, a]);
            }
            let n = p.len();
            if n > 1 && p[n - 2].is_ascii_digit() && "kmbt".contains(p[n - 1]) {
                p[n - 1] = p[n - 1].to_ascii_uppercase();
            }
            if total_params != 0 {
                let n = p.len();
                let num: String = p[..n - 1].iter().collect();
                if let Ok(v) = num.parse::<f64>() {
                    let at = " KMBT".find(p[n - 1]).map(|k| k as i32).unwrap_or(-1);
                    let label = v * 1000f64.powi(at);
                    let tp = total_params as f64;
                    let neg = total_params < 0 && label < (total_params.abs() / 8) as f64;
                    let pos = total_params > 0 && (label - tp).abs() > (7 * total_params / 8) as f64;
                    if neg || pos {
                        add(&mut types[i], "finetune");
                        p[n - 1] = p[n - 1].to_ascii_lowercase();
                    }
                }
            }
            if types[i].is_empty() {
                add(&mut types[i], "size_label");
            }
            parts[i] = p.into_iter().collect();
        } else if i > 0 && ["chat", "instruct", "vision", "lora"].contains(&part.to_ascii_lowercase().as_str()) {
            if total_params < 0 && part.eq_ignore_ascii_case("lora") {
                add(&mut types[i], "type");
            } else {
                add(&mut types[i], "finetune");
            }
        }
    }
    // word-based size labels go when a number-based one is present
    if parts.iter().zip(&types).any(|(n, t)| t.contains(&"size_label") && n.chars().any(|c| c.is_ascii_digit())) {
        for (n, t) in parts.iter().zip(types.iter_mut()) {
            if t.contains(&"size_label") && n.chars().all(char::is_alphabetic) {
                t.retain(|k| *k != "size_label");
            }
        }
    }
    let mut at_start = true;
    for (part, t) in parts.iter().zip(types.iter_mut()) {
        if at_start && ((t.is_empty() && part.starts_with(char::is_alphabetic)) || t.contains(&"version")) {
            add(t, "basename");
        } else {
            at_start = false;
            if t.is_empty() {
                add(t, "finetune");
            }
        }
    }
    for t in types.iter_mut().rev() {
        if t.contains(&"basename") && t.len() > 1 {
            t.retain(|k| *k != "basename");
        } else {
            break;
        }
    }
    let join = |k: &str, skip_basename: bool| {
        let v: Vec<&str> = parts.iter().zip(&types).filter(|(_, t)| t.contains(&k) && !(skip_basename && t.contains(&"basename"))).map(|(n, _)| n.as_str()).collect();
        (!v.is_empty()).then(|| v.join("-"))
    };
    let mut basename = join("basename", false);
    let mut seen: Vec<&str> = Vec::new();
    for (n, t) in parts.iter().zip(&types) {
        if t.contains(&"size_label") && !seen.contains(&n.as_str()) {
            seen.push(n);
        }
    }
    let size = (!seen.is_empty()).then(|| seen.join("-"));
    let finetune = join("finetune", false);
    let version = join("version", true);
    if size.is_none() && finetune.is_none() && version.is_none() {
        basename = None;
    }
    (Some(full), org, basename, finetune, version, size)
}

/// `model_weight_count_rounded_notation(n, min_digits=2)`
pub fn size_label(n: u64) -> String {
    let n = n as f64;
    let (v, s) = if n > 1e12 { (n * 1e-12, "T") } else if n > 1e9 { (n * 1e-9, "B") } else if n > 1e6 { (n * 1e-6, "M") } else { (n * 1e-3, "K") };
    // Python round() is half-to-even; str(int) has no leading zeros except "0", which lstrip('0') empties
    let r = v.round_ties_even() as u64;
    let len = if r == 0 { 0 } else { r.to_string().len() };
    let fix = 2usize.saturating_sub(len);
    format!("{v:.fix$}{s}")
}

#[derive(Default, Debug)]
struct Meta {
    name: Option<String>,
    org: Option<String>,
    basename: Option<String>,
    finetune: Option<String>,
    version: Option<String>,
    size_label: Option<String>,
}

impl Meta {
    fn absorb(&mut self, id: &str, total: i64) {
        let (full, org, base, ft, ver, size) = model_id_components(id, total);
        if self.name.is_none() { self.name = full.map(|f| id_to_title(&f)) }
        if self.org.is_none() { self.org = org.map(|o| id_to_title(&o)) }
        if self.basename.is_none() { self.basename = base }
        if self.finetune.is_none() { self.finetune = ft }
        if self.version.is_none() { self.version = ver }
        if self.size_label.is_none() { self.size_label = size }
    }
}

// ---------------------------------------------------------------- safetensors ---------------------------------

struct StTensor {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    part: usize,
    start: usize,
    len: usize,
}

fn read_json(p: &Path) -> Result<Json, String> {
    let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Json::parse(&t).ok_or(format!("{}: not JSON", p.display()))
}

fn num(v: &Json, k: &str) -> Option<f64> {
    match v.get(k) {
        Some(Json::Num(n)) => Some(*n),
        _ => None,
    }
}

fn int(v: &Json, k: &str) -> Option<i64> {
    num(v, k).filter(|n| n.fract() == 0.0).map(|n| n as i64)
}

/// The safetensors header of one part: tensors sorted by name (gguf-py's `SafetensorsLocal`).
fn st_header(bytes: &[u8], part: usize, file: &str) -> Result<Vec<StTensor>, String> {
    let n = u64::from_le_bytes(bytes.get(..8).ok_or(format!("{file}: too short"))?.try_into().unwrap()) as usize;
    let hdr = bytes.get(8..8usize.checked_add(n).ok_or("bad header length")?).ok_or(format!("{file}: header runs past the file"))?;
    let j = Json::parse(std::str::from_utf8(hdr).map_err(|_| format!("{file}: header is not UTF-8"))?).ok_or(format!("{file}: header is not JSON"))?;
    let Json::Obj(fields) = j else { return Err(format!("{file}: header is not an object")) };
    let mut out = Vec::new();
    for (name, m) in fields {
        if name == "__metadata__" {
            continue;
        }
        let dtype = m.get("dtype").and_then(Json::as_str).ok_or(format!("{file}: {name} has no dtype"))?.to_string();
        let shape: Vec<u64> = match m.get("shape") {
            Some(Json::Arr(a)) => a.iter().map(|x| x.as_u64().ok_or("bad shape")).collect::<Result<_, _>>()?,
            _ => return Err(format!("{file}: {name} has no shape")),
        };
        let (a, b) = match m.get("data_offsets") {
            Some(Json::Arr(o)) if o.len() == 2 => (o[0].as_u64().ok_or("bad offset")? as usize, o[1].as_u64().ok_or("bad offset")? as usize),
            _ => return Err(format!("{file}: {name} has no data_offsets")),
        };
        let start = 8 + n + a;
        if b < a || start + (b - a) > bytes.len() {
            return Err(format!("{file}: {name} runs past the file"));
        }
        out.push(StTensor { name, dtype, shape, part, start, len: b - a });
    }
    out.sort_by(|x, y| x.name.cmp(&y.name));
    Ok(out)
}

// ---------------------------------------------------------------- the Llama tensor map -------------------------

/// HF Llama names to GGUF names (gguf-py's `TensorNameMap` for `MODEL_ARCH.LLAMA`, the names SmolLM2 uses).
fn map_name(name: &str) -> Option<String> {
    let w = name.strip_suffix(".weight")?;
    let fixed = match w {
        "model.embed_tokens" => Some("token_embd"),
        "model.norm" => Some("output_norm"),
        "lm_head" => Some("output"),
        _ => None,
    };
    if let Some(f) = fixed {
        return Some(format!("{f}.weight"));
    }
    let rest = w.strip_prefix("model.layers.")?;
    let (bid, t) = rest.split_once('.')?;
    let bid: u32 = bid.parse().ok()?;
    let g = match t {
        "input_layernorm" => "attn_norm",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "post_attention_layernorm" => "ffn_norm",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        _ => return None,
    };
    Some(format!("blk.{bid}.{g}.weight"))
}

/// One source tensor as f32 values (BF16 exactly, F16 exactly, F32 as is), little-endian.
fn to_f32(dtype: &str, b: &[u8]) -> Result<Vec<f32>, String> {
    Ok(match dtype {
        "BF16" => b.as_chunks::<2>().0.iter().map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16)).collect(),
        "F16" => b.as_chunks::<2>().0.iter().map(|c| crate::q1_0::f16_to_f32(u16::from_le_bytes(*c))).collect(),
        "F32" => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        d => return Err(format!("dtype {d}: bankml convert reads BF16, F16 and F32 safetensors")),
    })
}

/// `LlamaModel.permute`: rows (n_head, 2, dim/2) → (n_head, dim/2, 2), the HF rotate-half layout back to GGML's pairs.
fn permute_rows<T: Copy>(v: &[T], rows: usize, cols: usize, n_head: usize) -> Vec<T> {
    let dim = rows / n_head;
    let half = dim / 2;
    let mut o = Vec::with_capacity(v.len());
    for h in 0..n_head {
        for j in 0..half {
            for s in 0..2 {
                let r = h * dim + s * half + j;
                o.extend_from_slice(&v[r * cols..(r + 1) * cols]);
            }
        }
    }
    o
}

// ---------------------------------------------------------------- the tokenizer (gpt2 path) ---------------------

/// sha256 over every token in id order (each followed by `\n`), a blank line, then every merge as `a b\n` — the
/// fingerprint `KNOWN_TOKENIZERS` is keyed on.
pub fn vocab_merges_sha256(tokens: &[String], merges: &[String]) -> String {
    let mut h = crate::sha256::Sha256::default();
    for t in tokens {
        h.update(t.as_bytes());
        h.update(b"\n");
    }
    h.update(b"\n");
    for m in merges {
        h.update(m.as_bytes());
        h.update(b"\n");
    }
    crate::sha256::hex(&h.finish())
}

fn pre_tokenizer_is_smollm(pt: Option<&Json>) -> bool {
    let bytelevel = |p: &Json| {
        p.get("type").and_then(Json::as_str) == Some("ByteLevel")
            && p.get("add_prefix_space").and_then(Json::as_bool) == Some(false)
            && p.get("use_regex").and_then(Json::as_bool) != Some(false)
    };
    match pt {
        Some(p) if bytelevel(p) => true,
        Some(p) if p.get("type").and_then(Json::as_str) == Some("Sequence") => match p.get("pretokenizers") {
            Some(Json::Arr(a)) if a.len() == 2 => {
                a[0].get("type").and_then(Json::as_str) == Some("Digits") && a[0].get("individual_digits").and_then(Json::as_bool) == Some(true) && bytelevel(&a[1])
            }
            _ => false,
        },
        _ => false,
    }
}

const SPECIAL_TYPES: [&str; 7] = ["bos", "eos", "unk", "sep", "pad", "cls", "mask"];

fn special_key(t: &str) -> Option<&'static str> {
    Some(match t {
        "bos" => "tokenizer.ggml.bos_token_id",
        "eos" => "tokenizer.ggml.eos_token_id",
        "unk" => "tokenizer.ggml.unknown_token_id",
        "sep" => "tokenizer.ggml.seperator_token_id",
        "pad" => "tokenizer.ggml.padding_token_id",
        "mask" => "tokenizer.ggml.mask_token_id",
        _ => return None, // cls: gguf-py has no handler; llama.cpp skips it with a warning
    })
}

/// `_set_vocab_gpt2` (+ `SpecialVocab(load_merges=True)`) and Llama's `set_vocab` around it.
fn vocab(dir: &Path, hp: &Json, kv: &mut KvList, inputs: &mut Vec<PathBuf>) -> Result<(), String> {
    let tcj = dir.join("tokenizer_config.json");
    let tc = if tcj.is_file() { inputs.push(tcj.clone()); Some(read_json(&tcj)?) } else { None };
    if let Some(v) = tc.as_ref().and_then(|t| t.get("add_prefix_space")).and_then(Json::as_bool) {
        kv.set("tokenizer.ggml.add_space_prefix", Kv::Bool(v));
    }
    if tc.as_ref().and_then(|t| t.get("tokenizer_class")).and_then(Json::as_str) == Some("HybridDNATokenizer") {
        return Err("tokenizer_class HybridDNATokenizer: not ported".into());
    }
    if dir.join("tokenizer.model").is_file() {
        return Err("tokenizer.model: a SentencePiece vocabulary (llama.cpp's `llama` tokenizer path) is not ported; bankml convert writes byte-level BPE (gpt2) vocabularies".into());
    }
    let tjp = dir.join("tokenizer.json");
    let tj = read_json(&tjp).map_err(|e| format!("{e} (bankml convert needs tokenizer.json)"))?;
    inputs.push(tjp);
    let model = tj.get("model").ok_or("tokenizer.json has no model")?;
    if model.get("type").and_then(Json::as_str) == Some("BPE") && model.get("byte_fallback").and_then(Json::as_bool) == Some(true)
        && tj.get("decoder").and_then(|d| d.get("type")).and_then(Json::as_str) == Some("Sequence") {
        return Err("a Llama-HF BPE tokenizer (byte fallback): llama.cpp writes it as its `llama` tokenizer; not ported".into());
    }
    if model.get("type").and_then(Json::as_str) != Some("BPE") {
        return Err("tokenizer.json model is not BPE: not ported".into());
    }
    // the vocabulary: model.vocab with the added tokens over it (AutoTokenizer's get_vocab())
    let Some(Json::Obj(v)) = model.get("vocab") else { return Err("tokenizer.json has no model.vocab".into()) };
    let added: Vec<(u64, String, bool, bool)> = match tj.get("added_tokens") {
        Some(Json::Arr(a)) => a.iter().map(|t| {
            Ok((t.get("id").and_then(Json::as_u64).ok_or("added token without id")?,
                t.get("content").and_then(Json::as_str).ok_or("added token without content")?.to_string(),
                t.get("special").and_then(Json::as_bool).unwrap_or(false),
                t.get("normalized").and_then(Json::as_bool).unwrap_or(true)))
        }).collect::<Result<_, String>>()?,
        _ => Vec::new(),
    };
    let n_vocab = int(hp, "vocab_size").unwrap_or(32000) as usize; // LlamaConfig's default, as AutoConfig supplies it
    let mut rev: Vec<Option<String>> = vec![None; n_vocab];
    for (tok, id) in v {
        let id = id.as_u64().ok_or("a vocab id is not an integer")? as usize;
        *rev.get_mut(id).ok_or(format!("token id {id} is not below vocab_size {n_vocab}"))? = Some(tok.clone());
    }
    for (id, c, _, _) in &added {
        *rev.get_mut(*id as usize).ok_or(format!("added token id {id} is not below vocab_size {n_vocab}"))? = Some(c.clone());
    }
    let looks_special = |t: &str| ["<pad>", "<mask>", "<2mass>", "[@BOS@]"].contains(&t) || (t.starts_with("<|") && t.ends_with("|>"))
        || (t.starts_with("<｜") && t.ends_with("｜>")) || (t.starts_with("<unused") && t.ends_with('>'));
    let (mut tokens, mut types) = (Vec::with_capacity(n_vocab), Vec::with_capacity(n_vocab));
    for (i, t) in rev.into_iter().enumerate() {
        match t {
            None => {
                tokens.push(format!("[PAD{i}]"));
                types.push(5);
            }
            Some(t) => {
                if let Some((_, _, special, _)) = added.iter().find(|a| a.1 == t) {
                    if !special && !looks_special(&t) {
                        return Err(format!("added token {t:?} is not special: llama.cpp re-encodes it through the Python tokenizer; not ported"));
                    }
                    types.push(3);
                } else {
                    types.push(1);
                }
                tokens.push(t);
            }
        }
    }
    // merges: tokenizer.json, else merges.txt
    let mut merges: Vec<String> = match model.get("merges") {
        Some(Json::Arr(a)) if !a.is_empty() => a.iter().map(|m| match m {
            Json::Str(s) => Ok(s.clone()),
            Json::Arr(p) if p.len() == 2 => {
                let enc = |x: &Json| x.as_str().map(|s| s.chars().map(|c| if c == ' ' { '\u{120}' } else { c }).collect::<String>());
                Ok(format!("{} {}", enc(&p[0]).ok_or("bad merge")?, enc(&p[1]).ok_or("bad merge")?))
            }
            _ => Err("unknown tokenizer merges format".to_string()),
        }).collect::<Result<_, _>>()?,
        _ => Vec::new(),
    };
    if merges.is_empty() {
        if let Ok(t) = std::fs::read_to_string(dir.join("merges.txt")) {
            inputs.push(dir.join("merges.txt"));
            let mut lines = t.lines().peekable();
            if lines.peek().is_some_and(|l| l.trim().starts_with('#')) {
                lines.next();
            }
            merges = lines.map(str::trim).filter(|l| !l.is_empty()).filter_map(|l| {
                let p: Vec<&str> = l.split_whitespace().collect();
                (p.len() == 2).then(|| format!("{} {}", p[0], p[1]))
            }).collect();
        }
    }
    // the pre-tokenizer llama.cpp would name, recognised (see KNOWN_TOKENIZERS)
    let fp = vocab_merges_sha256(&tokens, &merges);
    let pre = KNOWN_TOKENIZERS.iter().find(|k| k.1 == fp).map(|k| k.0)
        .filter(|p| *p == "smollm" && pre_tokenizer_is_smollm(tj.get("pre_tokenizer")))
        .ok_or(format!("tokenizer not recognised (vocabulary+merges sha256 {fp}): llama.cpp names a BPE pre-tokenizer by the token ids its Python \
                        tokenizer gives one fixed text; bankml recognises only tokenizers whose name was measured that way \
                        (testing/convert_oracle.py --chkhsh DIR prints it) — known: {}", KNOWN_TOKENIZERS.map(|k| k.0).join(", ")))?;
    if let Some(n) = tj.get("normalizer").filter(|n| **n != Json::Null) {
        return Err(format!("tokenizer normalizer {}: not ported (a gpt2 vocabulary has none)", n.get("type").and_then(Json::as_str).unwrap_or("?")));
    }
    kv.set("tokenizer.ggml.model", Kv::Str("gpt2".into()));
    kv.str("tokenizer.ggml.pre", pre);
    kv.set("tokenizer.ggml.tokens", Kv::Strs(tokens));
    kv.set("tokenizer.ggml.token_type", Kv::I32s(types));
    // SpecialVocab
    let mut add_special: Vec<(&str, bool)> = Vec::new();
    let set_add = |a: &mut Vec<(&str, bool)>, k: &'static str, v: bool, default: bool| match a.iter_mut().find(|x| x.0 == k) {
        Some(e) if !default => e.1 = v,
        Some(_) => {}
        None => a.push((k, v)),
    };
    // `if not tokenizer_config: return` — an empty object counts as none
    let mut tc = tc.filter(|t| *t != Json::Obj(Vec::new()));
    if let Some(pp) = tj.get("post_processor").filter(|p| **p != Json::Null) {
        let list: Vec<&Json> = match pp.get("processors") { Some(Json::Arr(a)) => a.iter().collect(), _ => vec![pp] };
        for p in list {
            match p.get("type").and_then(Json::as_str) {
                Some("ByteLevel") => {
                    set_add(&mut add_special, "bos", false, true);
                    set_add(&mut add_special, "eos", false, true);
                }
                Some("TemplateProcessing") => {
                    // only the shape that adds nothing is ported: single = [A], pair = [A, B]
                    let single = match p.get("single") { Some(Json::Arr(a)) => a.len(), _ => 0 };
                    let pair_ab = match p.get("pair") {
                        Some(Json::Arr(a)) => a.iter().all(|x| x.get("Sequence").is_some()) && a.len() <= 2,
                        None => true,
                        _ => false,
                    };
                    if single > 1 || !pair_ab {
                        return Err("post_processor TemplateProcessing with special tokens: not ported".into());
                    }
                }
                t => return Err(format!("post_processor {}: not ported", t.unwrap_or("?"))),
            }
        }
    }
    let mut ids: Vec<(&str, u32)> = Vec::new();
    let set_id = |ids: &mut Vec<(&str, u32)>, t: &'static str, id: Option<i64>| {
        if let Some(id) = id {
            if id < 0 {
                return Err(format!("invalid value for special token type {t}: {id}"));
            }
            if !ids.iter().any(|x| x.0 == t) {
                ids.push((t, id as u32));
            }
        }
        Ok(())
    };
    let mut template = None;
    if let Some(tc) = tc.as_mut() {
        let has = |tc: &Json, k: &str| tc.get(k).is_some_and(|v| !matches!(v, Json::Null | Json::Bool(false)) && *v != Json::Str(String::new()));
        if !has(tc, "bos_token") && has(tc, "cls_token") {
            return Err("cls_token standing in for bos_token: not ported".into());
        }
        if !has(tc, "eos_token") && has(tc, "sep_token") {
            return Err("sep_token standing in for eos_token: not ported".into());
        }
        let jinja = dir.join("chat_template.jinja");
        let alt = if jinja.is_file() {
            if dir.join("additional_chat_templates").is_dir() {
                return Err("additional_chat_templates: named templates are not ported".into());
            }
            inputs.push(jinja.clone());
            // Python's text mode: universal newlines
            Some(Json::Str(std::fs::read_to_string(&jinja).map_err(|e| e.to_string())?.replace("\r\n", "\n").replace('\r', "\n")))
        } else if dir.join("chat_template.json").is_file() {
            read_json(&dir.join("chat_template.json"))?.get("chat_template").cloned()
        } else {
            None
        };
        template = match tc.get("chat_template").cloned().or(alt) {
            Some(Json::Str(s)) => Some(s),
            None | Some(Json::Null) => None,
            Some(Json::Arr(_)) => return Err("a list of named chat templates: not ported".into()),
            Some(_) => None,
        };
        for t in SPECIAL_TYPES {
            if let Some(b) = tc.get(&format!("add_{t}_token")).and_then(Json::as_bool) {
                set_add(&mut add_special, t, b, true);
            }
            let content = match tc.get(&format!("{t}_token")) {
                Some(Json::Str(s)) => Some(s.clone()),
                Some(o @ Json::Obj(_)) => o.get("content").and_then(Json::as_str).map(str::to_string),
                _ => None,
            };
            if let Some(c) = content {
                set_id(&mut ids, t, added.iter().find(|a| a.1 == c).map(|a| a.0 as i64))?;
            }
        }
    }
    let cfg = read_json(&dir.join("config.json"))?;
    for t in SPECIAL_TYPES {
        let k = format!("{t}_token_id");
        let v = int(&cfg, &k).or_else(|| cfg.get("text_config").and_then(|c| int(c, &k)));
        set_id(&mut ids, t, v)?;
    }
    if !merges.is_empty() {
        kv.set("tokenizer.ggml.merges", Kv::Strs(merges));
    }
    for (t, id) in &ids {
        if let Some(k) = special_key(t) {
            kv.set(k, Kv::U32(*id));
        }
    }
    for (t, b) in &add_special {
        match *t {
            "bos" | "eos" | "sep" => kv.set(&format!("tokenizer.ggml.add_{t}_token"), Kv::Bool(*b)),
            _ => {} // gguf-py has no add_{unk,pad,cls,mask}_token handler
        }
    }
    if let Some(t) = template {
        kv.set("tokenizer.chat_template", Kv::Str(t));
    }
    match int(hp, "vocab_size").unwrap_or(32000) {
        32016 => return Err("vocab_size 32016 (CodeLlama's FIM tokens): not ported".into()),
        49152 => kv.set("tokenizer.ggml.add_bos_token", Kv::Bool(false)),
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------- the conversion -------------------------------

/// One output tensor, planned from the headers (its data is read and converted while it is written).
struct Plan {
    gguf: String,
    src: usize,
    /// GGML type: 0 F32, 1 F16
    ty: u32,
    /// numpy shape (rows first)
    shape: Vec<u64>,
    permute_heads: Option<usize>,
}

/// Converts `dir` (config.json, model*.safetensors, tokenizer files) to `out`, byte-identical to llama.cpp b11192's
/// `convert_hf_to_gguf.py DIR --outtype f16` for the Llama models in scope, or refuses with the reason.
pub fn convert(dir: &Path, out: &Path, opt: &Options) -> Result<Report, String> {
    let cfg_path = dir.join("config.json");
    let cfg = read_json(&cfg_path)?;
    let arch = match cfg.get("architectures") {
        Some(Json::Arr(a)) => a.first().and_then(Json::as_str).unwrap_or("").to_string(),
        _ => String::new(),
    };
    if arch != "LlamaForCausalLM" {
        return Err(format!("architecture {arch:?}: bankml convert writes the Llama architecture (LlamaForCausalLM: SmolLM2, mindx-genN) only; \
                            use llama.cpp's convert_hf_to_gguf.py for the rest"));
    }
    for k in ["text_config", "quantization_config", "num_local_experts", "draft_vocab_size"] {
        if cfg.get(k).is_some_and(|v| *v != Json::Null) {
            return Err(format!("config.json {k}: not ported (it changes what llama.cpp writes)"));
        }
    }
    if cfg.get("is_causal") == Some(&Json::Bool(false)) {
        return Err("config.json is_causal false: not ported".into());
    }
    let rope = cfg.get("rope_parameters").filter(|r| **r != Json::Null).or(cfg.get("rope_scaling").filter(|r| **r != Json::Null));
    let rope_type = rope.and_then(|r| r.get("rope_type").or(r.get("type"))).and_then(Json::as_str);
    if !matches!(rope_type, None | Some("default")) {
        return Err(format!("rope type {:?}: rope scaling is not ported", rope_type.unwrap_or("")));
    }
    if cfg.get("attention_bias") == Some(&Json::Bool(true)) || cfg.get("mlp_bias") == Some(&Json::Bool(true)) {
        return Err("attention or MLP biases: not ported (SmolLM2 has none)".into());
    }
    if dir.join("README.md").is_file() && !opt.ignore_model_card {
        return Err("README.md: llama.cpp reads the model card into the GGUF (licence, tags, languages, base models, datasets), which is not ported; \
                    move it aside, or pass --ignore-model-card knowing the result will not be byte-identical to llama.cpp's".into());
    }
    let mut inputs: Vec<PathBuf> = vec![cfg_path.clone()];
    // LlamaConfig's defaults, as AutoConfig(...).to_dict() supplies them
    let hidden = int(&cfg, "hidden_size").unwrap_or(4096);
    let n_head = int(&cfg, "num_attention_heads").unwrap_or(32);
    let n_kv = int(&cfg, "num_key_value_heads").unwrap_or(n_head);
    let head_dim = int(&cfg, "head_dim").unwrap_or(hidden / n_head);
    let n_layer = int(&cfg, "num_hidden_layers").unwrap_or(32);
    let n_ff = int(&cfg, "intermediate_size").unwrap_or(11008);
    let n_ctx = int(&cfg, "max_position_embeddings").unwrap_or(2048);
    let rope_theta = rope.and_then(|r| num(r, "rope_theta")).or(num(&cfg, "rope_theta")).unwrap_or(10000.0);
    let eps = num(&cfg, "rms_norm_eps").unwrap_or(1e-6);

    // the safetensors parts
    let mut parts: Vec<String> = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned()).filter(|f| f.starts_with("model") && f.ends_with(".safetensors")).collect();
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let j = read_json(&index)?;
        let Some(Json::Obj(map)) = j.get("weight_map") else { return Err("model.safetensors.index.json has no weight_map".into()) };
        let mut p: Vec<String> = map.iter().filter_map(|(_, v)| v.as_str().map(str::to_string)).collect();
        p.sort();
        p.dedup();
        parts = p;
        inputs.push(index);
    }
    parts.sort();
    if parts.is_empty() {
        return Err(format!("{}: no model*.safetensors", dir.display()));
    }
    let maps: Vec<crate::gguf::Mmap> = parts.iter().map(|p| crate::gguf::Mmap::open(&dir.join(p)).map_err(|e| format!("{p}: {e}"))).collect::<Result<_, _>>()?;
    let mut st = Vec::new();
    for (i, p) in parts.iter().enumerate() {
        st.extend(st_header(maps[i].bytes(), i, p)?);
        inputs.push(dir.join(p));
    }
    // plan every output tensor in llama.cpp's order
    let mut plan = Vec::new();
    for (i, t) in st.iter().enumerate() {
        if t.name.ends_with(".attention.masked_bias") || t.name.ends_with(".attention.bias") || t.name.ends_with(".rotary_emb.inv_freq") {
            continue;
        }
        let g = map_name(&t.name).ok_or(format!("can not map tensor {:?} (bankml convert knows the Llama names SmolLM2 uses)", t.name))?;
        if !["BF16", "F16", "F32"].contains(&t.dtype.as_str()) {
            return Err(format!("{}: dtype {} (bankml convert reads BF16, F16 and F32)", t.name, t.dtype));
        }
        let ty = if t.shape.len() <= 1 || g.ends_with("_norm.weight") { 0 } else { 1 };
        let permute_heads = if t.name.ends_with("q_proj.weight") {
            Some(n_head as usize)
        } else if t.name.ends_with("k_proj.weight") {
            Some(n_kv as usize)
        } else {
            None
        };
        if let Some(h) = permute_heads {
            if t.shape.len() != 2 || !(t.shape[0] as usize).is_multiple_of(2 * h) {
                return Err(format!("{}: shape {:?} cannot be permuted into {h} heads", t.name, t.shape));
            }
        }
        if plan.iter().any(|p: &Plan| p.gguf == g) {
            return Err(format!("duplicated tensor name {g:?}"));
        }
        plan.push(Plan { gguf: g, src: i, ty, shape: t.shape.clone(), permute_heads });
    }
    let total: u64 = plan.iter().map(|p| p.shape.iter().product::<u64>()).sum();

    // metadata, in llama.cpp's order
    let mut kv = KvList::default();
    kv.str("general.architecture", "llama");
    kv.str("general.type", "model");
    let gen = dir.join("generation_config.json");
    if gen.is_file() {
        inputs.push(gen.clone());
        if let Ok(g) = read_json(&gen) {
            if let Some(s) = g.get("sequence").and_then(Json::as_str) {
                kv.str("general.sampling.sequence", s);
            }
            let ints = |k: &str| -> Result<Option<i32>, String> {
                match g.get(k) {
                    None | Some(Json::Null) => Ok(None),
                    Some(Json::Num(n)) if n.fract() == 0.0 => Ok(Some(*n as i32)),
                    Some(_) => Err(format!("generation_config.json {k}: not an integer")),
                }
            };
            let floats = |k: &str| match g.get(k) { Some(Json::Num(n)) => Some(*n as f32), _ => None };
            if let Some(v) = ints("top_k")? { kv.set("general.sampling.top_k", Kv::I32(v)) }
            if let Some(v) = floats("top_p") { kv.set("general.sampling.top_p", Kv::F32(v)) }
            if let Some(v) = floats("min_p") { kv.set("general.sampling.min_p", Kv::F32(v)) }
            if let Some(v) = floats("xtc_probability") { kv.set("general.sampling.xtc_probability", Kv::F32(v)) }
            if let Some(v) = floats("xtc_threshold") { kv.set("general.sampling.xtc_threshold", Kv::F32(v)) }
            if let Some(v) = floats("temperature") { kv.set("general.sampling.temp", Kv::F32(v)) }
            if let Some(v) = ints("penalty_last_n")? { kv.set("general.sampling.penalty_last_n", Kv::I32(v)) }
            if let Some(v) = floats("penalty_repeat").or(floats("repetition_penalty")) { kv.set("general.sampling.penalty_repeat", Kv::F32(v)) }
            if let Some(v) = ints("mirostat")? { kv.set("general.sampling.mirostat", Kv::I32(v)) }
            if let Some(v) = floats("mirostat_tau") { kv.set("general.sampling.mirostat_tau", Kv::F32(v)) }
            if let Some(v) = floats("mirostat_eta") { kv.set("general.sampling.mirostat_eta", Kv::F32(v)) }
        }
    }
    let mut meta = Meta::default();
    if let Some(n) = cfg.get("_name_or_path").and_then(Json::as_str).filter(|n| n.matches('/').count() <= 1) {
        meta.absorb(n, total as i64);
    }
    let dir_name = dir.file_name().map(|n| n.to_string_lossy().into_owned())
        .or_else(|| dir.canonicalize().ok().and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))).unwrap_or_default();
    meta.absorb(&dir_name, total as i64);
    if meta.name.is_none() {
        meta.name = Some(dir_name.clone());
    }
    if let Some(n) = &opt.model_name {
        meta.name = Some(n.clone());
    }
    if meta.size_label.is_none() && total > 0 {
        meta.size_label = Some(size_label(total));
    }
    let name = meta.name.clone().unwrap_or_default();
    kv.str("general.name", &name);
    if let Some(v) = &meta.version { kv.str("general.version", v) }
    if let Some(v) = &meta.org { kv.str("general.organization", v) }
    if let Some(v) = &meta.finetune { kv.str("general.finetune", v) }
    if let Some(v) = &meta.basename { kv.str("general.basename", v) }
    if let Some(v) = &meta.size_label { kv.str("general.size_label", v) }
    kv.set("llama.block_count", Kv::U32(n_layer as u32));
    kv.set("llama.context_length", Kv::U32(n_ctx as u32));
    kv.set("llama.embedding_length", Kv::U32(hidden as u32));
    kv.set("llama.feed_forward_length", Kv::U32(n_ff as u32));
    kv.set("llama.attention.head_count", Kv::U32(n_head as u32));
    kv.set("llama.attention.head_count_kv", Kv::U32(n_kv as u32));
    kv.set("llama.rope.freq_base", Kv::F32(rope_theta as f32));
    kv.set("llama.attention.layer_norm_rms_epsilon", Kv::F32(eps as f32));
    kv.set("llama.attention.key_length", Kv::U32(head_dim as u32));
    kv.set("llama.attention.value_length", Kv::U32(head_dim as u32));
    kv.set("general.file_type", Kv::U32(1));
    kv.set("llama.vocab_size", Kv::U32(int(&cfg, "vocab_size").unwrap_or(32000) as u32));
    kv.set("llama.rope.dimension_count", Kv::U32(head_dim as u32));
    kv.set("general.quantization_version", Kv::U32(2));
    vocab(dir, &cfg, &mut kv, &mut inputs)?;

    // write: header, metadata, tensor infos, then each tensor's data, every piece aligned to 32
    const ALIGN: u64 = 32;
    let pad = |n: u64| n.div_ceil(ALIGN) * ALIGN;
    let mut head = b"GGUF".to_vec();
    head.extend(3u32.to_le_bytes());
    head.extend((plan.len() as u64).to_le_bytes());
    head.extend((kv.0.len() as u64).to_le_bytes());
    head.extend(kv_bytes(&kv));
    let mut off = 0u64;
    for p in &plan {
        put_str(&mut head, &p.gguf);
        head.extend((p.shape.len() as u32).to_le_bytes());
        p.shape.iter().rev().for_each(|d| head.extend(d.to_le_bytes()));
        head.extend(p.ty.to_le_bytes());
        head.extend(off.to_le_bytes());
        off += pad(p.shape.iter().product::<u64>() * if p.ty == 0 { 4 } else { 2 });
    }
    let tmp = out.with_extension("gguf.part");
    let f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let mut w = Hashed { w: std::io::BufWriter::with_capacity(1 << 20, f), h: crate::sha256::Sha256::default(), n: 0 };
    let io = |e: std::io::Error| format!("{}: {e}", tmp.display());
    w.write_all(&head).map_err(io)?;
    w.zeros(pad(w.n) - w.n).map_err(io)?;
    for p in &plan {
        let t = &st[p.src];
        let raw = &maps[t.part].bytes()[t.start..t.start + t.len];
        let n: u64 = p.shape.iter().product();
        if raw.len() as u64 != n * match t.dtype.as_str() { "F32" => 4, _ => 2 } {
            return Err(format!("{}: {} bytes for shape {:?}", t.name, raw.len(), t.shape));
        }
        let mut v = to_f32(&t.dtype, raw)?;
        if let Some(h) = p.permute_heads {
            v = permute_rows(&v, p.shape[0] as usize, p.shape[1] as usize, h);
        }
        let bytes: Vec<u8> = if p.ty == 0 {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        } else {
            v.iter().flat_map(|x| crate::q1_0::f32_to_f16(*x).to_le_bytes()).collect()
        };
        w.write_all(&bytes).map_err(io)?;
        w.zeros(pad(bytes.len() as u64) - bytes.len() as u64).map_err(io)?;
    }
    w.w.flush().map_err(io)?;
    let (bytes, sha) = (w.n, crate::sha256::hex(&w.h.finish()));
    std::fs::rename(&tmp, out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut ins = Vec::new();
    for p in inputs {
        let n = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let len = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        ins.push((n, len, crate::sha256::file_hex(&p).map_err(|e| format!("{}: {e}", p.display()))?));
    }
    Ok(Report { out: out.to_path_buf(), sha256: sha, bytes, tensors: plan.len(), kv: kv.0.len(), params: total, name, inputs: ins })
}

/// A writer that hashes and counts what it writes.
struct Hashed<W: Write> {
    w: W,
    h: crate::sha256::Sha256,
    n: u64,
}

impl<W: Write> Hashed<W> {
    fn write_all(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.h.update(b);
        self.n += b.len() as u64;
        self.w.write_all(b)
    }
    fn zeros(&mut self, n: u64) -> std::io::Result<()> {
        self.write_all(&vec![0u8; n as usize])
    }
}

/// The FORK.json a conversion is pinned by: the output's sha256 and size, and every input file's, with the rule it
/// was made by. `source` names where the directory came from (a Hub repo and revision, or a local path).
pub fn fork_json(r: &Report, source: &str) -> String {
    let file = r.out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let ins: Vec<String> = r.inputs.iter().map(|(n, b, s)| format!("   {{\"path\": {}, \"bytes\": {b}, \"sha256\": \"{s}\"}}", jstr(n))).collect();
    format!("{{\n \"kind\": \"bankml convert (safetensors → GGUF F16, made here, not downloaded)\",\n \"source\": {},\n \"converter\": \"bankml {} convert --outtype f16: byte-identical to llama.cpp b11192 convert_hf_to_gguf.py --outtype f16 on its oracle\",\n \"converted_at_utc\": \"{}\",\n \"inputs\": [\n{}\n ],\n \"files\": [\n  {{\n   \"path\": {},\n   \"bytes\": {},\n   \"sha256\": \"{}\"\n  }}\n ]\n}}\n",
            jstr(source), crate::VERSION, crate::ollama::rfc3339(std::time::SystemTime::now()).split('.').next().unwrap_or("").to_string() + "Z", ins.join(",\n"), jstr(&file), r.bytes, r.sha256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_heuristics_as_gguf_py() {
        // the two directories of the oracle, and the cases gguf-py's own tests exercise
        let c = |id: &str, n: i64| model_id_components(id, n);
        let s = |x: &str| Some(x.to_string());
        assert_eq!(c("smollm2-135m-instruct", 134_515_008), (s("smollm2-135m-instruct"), None, s("smollm2"), s("instruct"), None, s("135M")));
        assert_eq!(c("mindx-gen39", 134_515_008), (s("mindx-gen39"), None, None, None, None, None));
        assert_eq!(c("Mixtral-8x7B-Instruct-v0.1", 0), (s("Mixtral-8x7B-Instruct-v0.1"), None, s("Mixtral"), s("Instruct"), s("v0.1"), s("8x7B")));
        assert_eq!(c("meta-llama/Meta-Llama-3-8B", 0), (s("Meta-Llama-3-8B"), s("meta-llama"), s("Meta-Llama-3"), None, None, s("8B")));
        assert_eq!(c("bigscience/bloom-7b1-petals", 0), (s("bloom-7b1-petals"), s("bigscience"), s("bloom"), s("petals"), None, s("7.1B")));
        assert_eq!(c("NousResearch/Meta-Llama-3-8B", 0).2, s("Meta-Llama-3"));
        assert_eq!(c("Qwen/Qwen2.5-0.5B-Instruct", 0), (s("Qwen2.5-0.5B-Instruct"), s("Qwen"), s("Qwen2.5"), s("Instruct"), None, s("0.5B")));
        assert_eq!(c("a b c", 0).0, s("a b c"));
        assert_eq!(c("./my-model", 0).1, None);
        // a size far from the weights counted is a context length: a finetune, lower-cased
        assert_eq!(c("model-32k-chat", 100_000_000), (s("model-32k-chat"), None, s("model"), s("32k-chat"), None, None));
        assert_eq!(id_to_title("smollm2-135m-instruct"), "Smollm2 135m Instruct");
        assert_eq!(id_to_title("mindx-gen39"), "Mindx Gen39");
        assert_eq!(id_to_title("Qwen2.5-0.5B-Instruct"), "Qwen2.5 0.5B Instruct");
        assert_eq!(id_to_title("llama-v2-chat"), "Llama v2 Chat");
        assert_eq!(size_label(134_515_008), "135M");
        assert_eq!(size_label(1_720_000_000), "1.7B");
        assert_eq!(size_label(8_190_000_000), "8.2B");
        assert_eq!(size_label(500_000), "500K");
        assert_eq!(size_label(1_500_000), "1.5M");
    }

    #[test]
    fn permute_matches_numpy_reshape_swapaxes() {
        // 2 heads × 4 rows: in rows [h][s][j] → out rows [h][j][s]
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(permute_rows(&v, 8, 1, 2), vec![0, 2, 1, 3, 4, 6, 5, 7]);
        let w: Vec<u32> = (0..16).collect();
        assert_eq!(permute_rows(&w, 8, 2, 1), vec![0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]);
    }

    #[test]
    fn kv_set_keeps_the_first_position() {
        let mut kv = KvList::default();
        kv.set("a", Kv::Bool(true));
        kv.set("b", Kv::U32(1));
        kv.set("a", Kv::Bool(false));
        kv.str("c", "");
        assert_eq!(kv.0, vec![("a".to_string(), Kv::Bool(false)), ("b".to_string(), Kv::U32(1))]);
        let b = kv_bytes(&kv);
        assert_eq!(&b[..9], &[1, 0, 0, 0, 0, 0, 0, 0, b'a']);
        assert_eq!(&b[9..14], &[7, 0, 0, 0, 0]);
    }

    #[test]
    fn regexes_as_gguf_py() {
        for v in ["v1", "1", "v0.1", "iter3", "ITER3", "2.5.1"] { assert!(re_version(v), "{v}") }
        for v in ["v", "gen39", "v1.", "x1"] { assert!(!re_version(v), "{v}") }
        for t in ["q4", "Q4_K_M", "iq2_x", "f16", "bf16", "bfp16", "fp32", "F32"] { assert!(re_type(t), "{t}") }
        for t in ["q", "fp", "b16", "q4_", "iq2_xs"] { assert!(!re_type(t), "{t}") }
        for z in ["135m", "8B", "8x7B", "7b1", "0.5B", "1_5B", "A3B", "small", "XL", "xxl"] { assert!(re_size(z), "{z}") }
        for z in ["gen39", "instruct", "8", "B", "x7B", "8xB"] { assert!(!re_size(z), "{z}") }
    }

    /// The name heuristics against gguf-py's own, on a corpus of Hub ids (`testing/convert_oracle.py --names OUT.jsonl`
    /// writes one `{"id", "total", "parts", "title", "size"}` per line from llama.cpp b11192's gguf-py).
    #[test]
    #[ignore = "needs testing/convert_oracle.py --names output in BANKML_NAMES_ORACLE"]
    fn oracle_name_heuristics() {
        let t = std::fs::read_to_string(std::env::var("BANKML_NAMES_ORACLE").expect("BANKML_NAMES_ORACLE")).unwrap();
        let (mut n, mut bad) = (0, 0);
        for line in t.lines() {
            let v = Json::parse(line).unwrap();
            let id = v.get("id").and_then(Json::as_str).unwrap();
            let total = num(&v, "total").unwrap() as i64;
            let got = model_id_components(id, total);
            let o = |x: &Option<String>| x.as_ref().map(|s| Json::Str(s.clone())).unwrap_or(Json::Null);
            let want = v.get("parts").unwrap().clone();
            let mine = Json::Arr(vec![o(&got.0), o(&got.1), o(&got.2), o(&got.3), o(&got.4), o(&got.5)]);
            let title_ok = v.get("title").and_then(Json::as_str) == Some(id_to_title(id).as_str());
            let size_ok = v.get("size").and_then(Json::as_str) == Some(size_label(total.unsigned_abs().max(1)).as_str()) || total == 0;
            n += 1;
            if mine != want || !title_ok || !size_ok {
                bad += 1;
                println!("{id} {total}: bankml {mine:?} {:?} / gguf-py {want:?} {:?}", id_to_title(id), v.get("title"));
            }
        }
        println!("{n} ids, {bad} different");
        assert_eq!(bad, 0);
    }

    /// The oracle: bankml convert's GGUF against llama.cpp b11192's `convert_hf_to_gguf.py --outtype f16`, byte for byte
    /// (testing/convert_oracle.py records it). `BANKML_CONVERT_DIR` = the safetensors directory (named as the oracle's
    /// was, since llama.cpp names the model from it), `BANKML_CONVERT_ORACLE` = llama.cpp's GGUF.
    #[test]
    #[ignore = "needs a safetensors directory and llama.cpp's conversion of it (testing/convert_oracle.py); --release"]
    fn oracle_convert_b11192() {
        let dir = std::env::var("BANKML_CONVERT_DIR").expect("BANKML_CONVERT_DIR");
        let oracle = std::env::var("BANKML_CONVERT_ORACLE").expect("BANKML_CONVERT_ORACLE");
        let out = std::env::var("BANKML_CONVERT_OUT").unwrap_or_else(|_| format!("{}/bankml-convert-oracle.gguf", std::env::temp_dir().display()));
        let r = convert(Path::new(&dir), Path::new(&out), &Options::default()).unwrap();
        let want = crate::sha256::file_hex(Path::new(&oracle)).unwrap();
        println!("bankml convert {dir}: {} tensors, {} keys, {} bytes, sha256 {}; llama.cpp b11192: {want}", r.tensors, r.kv, r.bytes, r.sha256);
        if std::env::var("BANKML_CONVERT_KEEP").is_err() {
            let _ = std::fs::remove_file(&out);
        }
        assert_eq!(r.sha256, want, "not byte-identical to llama.cpp's GGUF");
    }
}
