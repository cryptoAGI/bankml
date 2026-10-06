// SPDX-License-Identifier: MIT OR Apache-2.0
//! O6, first cut (0.3.3) — grammar-constrained sampling: JSON mode, token-identical to llama-server b11192.
//!
//! Ported from llama.cpp b11192 (MIT, © the ggml authors, github.com/ggml-org/llama.cpp @ 171e8846b):
//! `src/llama-grammar.cpp` (the GBNF parser, the pushdown stacks, `advance_stack`, `reject_candidates`, the partial
//! UTF-8 decoder, `accept_token`, `apply`) and the grammar's place in `common/sampling.cpp`'s `common_sampler_sample`.
//! The port keeps llama.cpp's semantics exactly; only its data layout differs (a stack element is an index into one
//! flat element array instead of a pointer, which compares the same way). llama.cpp's notice, kept as its licence
//! asks (LICENSING.md):
//!
//! > MIT License — Copyright (c) 2023-2026 The ggml authors. Permission is hereby granted, free of charge, to any
//! > person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the
//! > Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
//! > distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
//! > furnished to do so, subject to the following conditions: The above copyright notice and this permission notice
//! > shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
//! > WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//! > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
//! > COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//! > OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//!
//! **Where the grammar sits** (`common_sampler_sample`, `grammar_first = false`, as llama-server calls it): the
//! unconstrained chain runs first (top-k … temperature → dist, one RNG draw); if the token it picked passes the
//! grammar it is taken. Only if it does not, the logits are reset, every token the grammar rejects is set to −∞, and
//! the whole chain runs again — a **second** draw from the same `mt19937`. `Sampler::sample_constrained` does exactly
//! that, so a seeded run consumes the generator as llama-server's does.
//!
//! **What llama-server asks for** with `response_format: {"type": "json_object"}` on the pinned Qwen3 template with
//! thinking off is not `grammars/json.gbnf`: the chat layer turns the schema `{"type": "object"}` into a PEG parser and
//! that into GBNF (`common/chat-auto-parser-generator.cpp`), whose root starts with the generation prompt, and the
//! sampler then *accepts the generation prompt's tokens* into the grammar before the first draw
//! (`common_grammar_needs_prefill`). `JSON_OBJECT_GRAMMAR` is that text, byte for byte as b11192 reports it in
//! `generation_settings.grammar` (the oracle checks it on every recorded request); `JSON_OBJECT_PREFILL` is the
//! generation prompt. A user's own `grammar` is a USER grammar: no prefill.
//!
//! O6b: every other JSON schema is converted as llama-server converts it (`schema.rs`: `json_schema_to_grammar` and
//! the chat parser's wrapping, byte-identical over llama.cpp's own test cases and a mindX-shaped corpus) and answered
//! through the same prefill, fence and content rule as JSON mode. A schema b11192 refuses is refused with its reason.

use crate::sampler::Cand;
use crate::schema::Value;
use crate::serve::Json;

/// The rules every JSON-mode grammar of llama-server b11192 shares (the schema `{"type": "object"}` turned into GBNF).
macro_rules! json_object_rules {
    () => {
        concat!(
            "array ::= \"[\" space ( value (\",\" space value)* )? space \"]\"\n",
            "boolean ::= (\"true\" | \"false\")\n",
            "char ::= [^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})\n",
            "decimal-part ::= [0-9]{1,16}\n",
            "integral-part ::= [0] | [1-9] [0-9]{0,15}\n",
            "json-array ::= \"[\" space (\"]\" | json-value (space \",\" space json-value)* space \"]\")\n",
            "json-bool ::= \"true\" | \"false\"\n",
            "json-null ::= \"null\"\n",
            "json-number ::= \"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ((\"e\" | \"E\") [+-]? [0-9]+)?\n",
            "json-object ::= \"{\" space (\"}\" | json-string space \":\" space json-value (space \",\" space json-string space \":\" space json-value)* space \"}\")\n",
            "json-string ::= \"\\\"\" ( [^\"\\\\] | \"\\\\\" ( [\"\\\\/ bfnrt] | \"u\" [0-9a-fA-F]{4} ) )* \"\\\"\"\n",
            "json-value ::= json-object | json-array | json-string | json-number | json-bool | json-null\n",
            "null ::= \"null\"\n",
            "number ::= (\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?\n",
            "object ::= \"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"\n",
            "response-format ::= response-format-schema\n",
            "response-format-schema ::= object\n",
        )
    };
}

/// The grammar llama-server b11192 builds for `response_format: {"type": "json_object"}` (and for the schemas `{}` and
/// `{"type": "object"}`) on the pinned Qwen3 template with `--reasoning off`.
pub const JSON_OBJECT_GRAMMAR: &str = concat!(
    json_object_rules!(),
    "root ::= \"<|im_start|>assistant\\n\" space (\"<think>\" \"\\n\"? until-13 \"\\n\"? \"</think>\" \"\\n\"? \"\\n\"?)? space (\"```json\" space response-format space \"```\" | space response-format space)\n",
    "space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n",
    "string ::= \"\\\"\" char* \"\\\"\"\n",
    "until-13 ::= | [<] until-13-01 | [^<] until-13\n",
    "until-13-01 ::= | [<] until-13-01 | [/] until-13-02 | [^/<] until-13\n",
    "until-13-02 ::= | [<] until-13-01 | [t] until-13-03 | [^<t] until-13\n",
    "until-13-03 ::= | [<] until-13-01 | [h] until-13-04 | [^<h] until-13\n",
    "until-13-04 ::= | [<] until-13-01 | [i] until-13-05 | [^<i] until-13\n",
    "until-13-05 ::= | [<] until-13-01 | [n] until-13-06 | [^<n] until-13\n",
    "until-13-06 ::= | [<] until-13-01 | [k] until-13-07 | [^<k] until-13\n",
    "until-13-07 ::= | [<] until-13-01 | [^<>] until-13\n",
    "value ::= object | array | string | number | boolean | null\n",
);

/// The same request on a ChatML template without reasoning (SmolLM2-Instruct's, mindx-genN's; 0.3.4): the root
/// opens with that template's generation prompt and has no `<think>` block.
pub const JSON_OBJECT_GRAMMAR_CHATML: &str = concat!(
    json_object_rules!(),
    "root ::= \"<|im_start|>assistant\\n\" space space (\"```json\" space response-format space \"```\" | space response-format space)\n",
    "space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n",
    "string ::= \"\\\"\" char* \"\\\"\"\n",
    "value ::= object | array | string | number | boolean | null\n",
);

/// The JSON-mode grammar and its prefill (the generation prompt the sampler accepts before the first draw) for a
/// model's template.
pub fn json_object_grammar(t: crate::chat::Template) -> (&'static str, &'static str) {
    match t {
        crate::chat::Template::Qwen3 => (JSON_OBJECT_GRAMMAR, JSON_OBJECT_PREFILL),
        crate::chat::Template::SmolLm2 | crate::chat::Template::ChatMl => (JSON_OBJECT_GRAMMAR_CHATML, t.generation_prompt()),
    }
}

/// The template's generation prompt (thinking off): the tokens the JSON grammar accepts before the first draw.
pub const JSON_OBJECT_PREFILL: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

// ---------------------------------------------------------------- the request: which constraint ----------

/// What a request asks the sampler to obey.
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    None,
    /// `response_format` `json_object` (or the schemas `{}` / `{"type": "object"}`), Ollama's `format: "json"`
    JsonObject,
    /// O6b: any other JSON schema — the grammar llama-server builds for it on the model's template
    /// (`schema::chat_grammar`; converted once at parse so a refusal is a 400), prefilled as JSON mode is
    Schema(Value),
    /// llama-server's raw `grammar` field: GBNF, root `root`, no prefill
    Gbnf(String),
}

/// The constraint for a response-format schema: llama-server's chat-path grammar (JSON mode's own when the schema is
/// "any object"), or llama.cpp's refusal.
fn schema_constraint(schema: &Value) -> Result<Constraint, String> {
    // the refusals and warnings do not depend on the template (llama.cpp b11192's own output on each, the oracle)
    let (g, warnings) = crate::schema::chat_grammar(schema, crate::chat::Template::Qwen3)?;
    for w in warnings {
        eprintln!("bankml: JSON schema conversion was incomplete (as llama.cpp b11192's): {w}");
    }
    Ok(if g == JSON_OBJECT_GRAMMAR { Constraint::JsonObject } else { Constraint::Schema(schema.clone()) })
}

/// The constraint of an OpenAI / llama-server chat request, resolved as `oaicompat_chat_params_parse` does: the
/// top-level `json_schema` and `grammar` fields, then `response_format` (`text`, `json_object` with an optional
/// `schema`, `json_schema` with `json_schema.schema`); an empty schema means any object. The numbers of the schema
/// are read as bankML's JSON reads them; `from_openai_text` reads them from the request's own text, exactly.
pub fn from_openai(req: &Json) -> Result<Constraint, String> {
    openai(&Value::from_json(req))
}

/// `from_openai` over the request's text (as llama-server parses it: `2.0` stays a float in an `enum`).
pub fn from_openai_text(req: &Json, body: &str) -> Result<Constraint, String> {
    openai(&Value::parse(body).unwrap_or_else(|| Value::from_json(req)))
}

fn openai(req: &Value) -> Result<Constraint, String> {
    // json_value(body, key, default): a missing or null field is the default
    let field = |v: &Value, k: &str| v.get(k).filter(|x| !x.is_null()).cloned();
    let mut schema = field(req, "json_schema").unwrap_or(Value::Null);
    let grammar = match req.get("grammar") {
        None | Some(Value::Null) => String::new(),
        Some(Value::Str(s)) => s.clone(),
        Some(_) => return Err("grammar must be a string (GBNF)".into()),
    };
    if !schema.is_null() && !grammar.is_empty() {
        return Err("Cannot use both json_schema and grammar".into());
    }
    if req.get("response_format").is_some() {
        let rf = field(req, "response_format").unwrap_or(Value::Obj(vec![]));
        let ty = rf.get("type").and_then(Value::as_str).unwrap_or("");
        match ty {
            "json_object" => {
                if rf.get("schema").is_some() || schema.is_empty() {
                    schema = field(&rf, "schema").unwrap_or(Value::Obj(vec![]));
                }
            }
            "json_schema" => {
                let w = field(&rf, "json_schema").unwrap_or(Value::Obj(vec![]));
                schema = field(&w, "schema").unwrap_or(Value::Obj(vec![]));
            }
            "" | "text" => {}
            t => return Err(format!("response_format type must be one of \"text\" or \"json_object\", but got: {t}")),
        }
    }
    if schema == Value::Obj(vec![]) {
        schema = Value::Obj(vec![("type".into(), Value::Str("object".into()))]);
    }
    match &schema {
        // the chat path builds the grammar for a non-empty object schema, and llama-server uses it over any other
        Value::Obj(_) if !grammar.is_empty() => Err("response_format and grammar together: send one (llama-server keeps only the format's grammar)".into()),
        Value::Obj(_) => schema_constraint(&schema),
        // otherwise the top-level `json_schema` field goes to the sampler's own reading, which fails on this path:
        // a non-object is not a schema, and null becomes {"type": "object"} converted bare, which the generation
        // prompt cannot be fed to (llama-server answers with an error in both cases)
        _ if req.get("json_schema").is_some() && grammar.is_empty() => Err(match req.get("json_schema") {
            Some(Value::Null) => "json_schema: null: llama-server b11192 converts it to a bare {\"type\": \"object\"} grammar that rejects the \
                                  chat template's generation prompt and fails the request; send {} or {\"type\": \"object\"}".into(),
            _ => "\"json_schema\": JSON schema conversion failed:\nJSON schema error at #: schema must be an object".into(),
        }),
        // a response_format schema that is not an object: llama-server builds no grammar and answers unconstrained
        _ if grammar.is_empty() => Ok(Constraint::None),
        _ => Ok(Constraint::Gbnf(grammar)),
    }
}

/// Ollama's `format`: `"json"` is JSON mode (mapped onto llama-server's `json_object`); a schema object is mapped
/// onto `response_format: {"type": "json_schema", …}` — the grammar llama-server would build for it.
pub fn from_ollama(format: Option<&Json>) -> Result<Constraint, String> {
    ollama(format.map(Value::from_json).as_ref())
}

/// `from_ollama` reading `format` from the request's own text (exact numbers).
pub fn from_ollama_text(req: &Json, body: &str) -> Result<Constraint, String> {
    match Value::parse(body) {
        Some(v) => ollama(v.get("format")),
        None => from_ollama(req.get("format")),
    }
}

fn ollama(format: Option<&Value>) -> Result<Constraint, String> {
    match format {
        None | Some(Value::Null) => Ok(Constraint::None),
        Some(Value::Str(s)) if s.is_empty() => Ok(Constraint::None),
        Some(Value::Str(s)) if s == "json" => Ok(Constraint::JsonObject),
        Some(Value::Str(s)) => Err(format!("format {s:?}: only \"json\" or a JSON schema object")),
        Some(Value::Obj(o)) if o.is_empty() => Ok(Constraint::JsonObject),
        Some(s @ Value::Obj(_)) => schema_constraint(s).map_err(|e| format!("format: {e}")),
        Some(_) => Err("format: only \"json\" or a JSON schema object".into()),
    }
}

/// The answer's `content` in JSON mode, as llama-server's chat parser gives it: the response-format rule is
/// `space ("```json" space VALUE space "```" | space VALUE space)`, and the content is VALUE alone — the object once it
/// is complete, everything after its first byte while it is still open (a length-limited answer). It only grows as
/// the raw text grows, so a stream can send the difference. O6b: a schema's value need not be an object — a string
/// ends at its closing quote, a number or literal at the first space (or the fence).
pub fn json_content(raw: &str) -> &str {
    let ws = |c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c');
    let mut s = raw.trim_start_matches(ws);
    if let Some(r) = s.strip_prefix("```json") {
        s = r.trim_start_matches(ws);
    } else if "```json".starts_with(s) {
        return "";
    }
    match s.bytes().next() {
        Some(b'{' | b'[' | b'"') | None => {}
        Some(_) => return &s[..s.find(|c: char| c.is_ascii_whitespace() || c == '`').unwrap_or(s.len())],
    }
    // 0.3.5: an escape the cut left unfinished (`\` alone, or `\u` with fewer than four hex digits) is not content:
    // llama.cpp's JSON parser ends the string before it (the content oracle)
    let (mut depth, mut in_str, mut esc, mut hex_left, mut esc_at) = (0usize, false, false, 0u8, 0usize);
    for (i, b) in s.bytes().enumerate() {
        if in_str {
            match b {
                _ if hex_left > 0 => hex_left -= 1,
                b'u' if esc => {
                    esc = false;
                    hex_left = 4;
                }
                _ if esc => esc = false,
                b'\\' => {
                    esc = true;
                    esc_at = i;
                }
                b'"' => {
                    in_str = false;
                    if depth == 0 {
                        return &s[..=i];
                    }
                }
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return &s[..=i];
                }
            }
            _ => {}
        }
    }
    if in_str && (esc || hex_left > 0) {
        return &s[..esc_at];
    }
    s
}

/// 0.3.5: the `content` llama-server b11192 answers a whole (non-streamed) constrained request with: the chat parser's
/// content (`json_content`), or — when that parse gives an empty message, e.g. an answer cut inside or right after
/// the opening fence — the raw text (`server_task_result_cmpl_final::to_json_oaicompat_chat`). A stream never sends
/// that fallback (the server's final parse adds no difference), so `ContentStream` keeps `json_content`.
pub fn json_message(raw: &str) -> &str {
    match json_content(raw) {
        "" => raw,
        c => c,
    }
}

/// JSON mode's content as a stream: raw pieces in, the growth of `json_content` out (the whole piece otherwise).
pub struct ContentStream {
    json: bool,
    raw: String,
    sent: usize,
}

impl ContentStream {
    pub fn new(c: &Constraint) -> ContentStream {
        ContentStream { json: matches!(c, Constraint::JsonObject | Constraint::Schema(_)), raw: String::new(), sent: 0 }
    }
    pub fn push(&mut self, piece: &str) -> String {
        if !self.json {
            return piece.to_string();
        }
        self.raw.push_str(piece);
        let c = json_content(&self.raw);
        let d = c.get(self.sent..).unwrap_or("").to_string();
        self.sent = self.sent.max(c.len());
        d
    }
    /// The answer's content from its whole raw text: what the stream sent (`streamed`), or the whole answer's message
    /// (`json_message`: the raw text when the parse is empty, as llama-server answers a non-streamed request).
    pub fn content(&self, raw: &str, streamed: bool) -> String {
        match (self.json, streamed) {
            (false, _) => raw.to_string(),
            (true, true) => json_content(raw).to_string(),
            (true, false) => json_message(raw).to_string(),
        }
    }
}

// ---------------------------------------------------------------- GBNF: the parser ----------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum T {
    End,
    Alt,
    RuleRef,
    Char,
    CharNot,
    CharRngUpper,
    CharAlt,
    CharAny,
    Token,
    TokenNot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct El {
    t: T,
    v: u32,
}

const fn el(t: T, v: u32) -> El {
    El { t, v }
}

const MAX_REPETITION_THRESHOLD: u64 = 2000;

fn is_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

fn is_word(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'-' || is_digit(c)
}

/// llama-grammar.cpp's parser over a NUL-terminated byte string (`s` ends with 0; reading stops there, as C does).
struct Parser<'a> {
    s: Vec<u8>,
    symbol_ids: std::collections::HashMap<Vec<u8>, u32>,
    rules: Vec<Vec<El>>,
    tokenize: &'a dyn Fn(&[u8]) -> Vec<u32>,
}

type R<T> = Result<T, String>;

impl Parser<'_> {
    fn at(&self, i: usize) -> u8 {
        self.s.get(i).copied().unwrap_or(0)
    }
    fn rest(&self, i: usize) -> String {
        let e = self.s[i.min(self.s.len())..].iter().position(|&b| b == 0).map(|p| i + p).unwrap_or(self.s.len());
        String::from_utf8_lossy(&self.s[i.min(e)..e]).chars().take(40).collect()
    }

    fn space(&self, mut p: usize, newline_ok: bool) -> usize {
        loop {
            let c = self.at(p);
            if c == b' ' || c == b'\t' || c == b'#' || (newline_ok && (c == b'\r' || c == b'\n')) {
                if c == b'#' {
                    while self.at(p) != 0 && self.at(p) != b'\r' && self.at(p) != b'\n' {
                        p += 1;
                    }
                } else {
                    p += 1;
                }
            } else {
                return p;
            }
        }
    }

    fn name(&self, src: usize) -> R<usize> {
        let mut p = src;
        while is_word(self.at(p)) {
            p += 1;
        }
        if p == src {
            return Err(format!("expecting name at {}", self.rest(src)));
        }
        Ok(p)
    }

    fn int(&self, src: usize) -> R<usize> {
        let mut p = src;
        while is_digit(self.at(p)) {
            p += 1;
        }
        if p == src {
            return Err(format!("expecting integer at {}", self.rest(src)));
        }
        Ok(p)
    }

    fn hex(&self, src: usize, size: usize) -> R<(u32, usize)> {
        let (mut p, end, mut v) = (src, src + size, 0u32);
        while p < end && self.at(p) != 0 {
            v = v.wrapping_shl(4);
            let c = self.at(p);
            match c {
                b'a'..=b'f' => v = v.wrapping_add((c - b'a' + 10) as u32),
                b'A'..=b'F' => v = v.wrapping_add((c - b'A' + 10) as u32),
                b'0'..=b'9' => v = v.wrapping_add((c - b'0') as u32),
                _ => break,
            }
            p += 1;
        }
        if p != end {
            return Err(format!("expecting {size} hex chars at {}", self.rest(src)));
        }
        Ok((v, p))
    }

    /// `decode_utf8(const char *)`: a lead byte's length, continuation bytes taken as they come, stopping at NUL.
    fn utf8(&self, src: usize) -> (u32, usize) {
        const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
        let first = self.at(src);
        let len = LOOKUP[(first >> 4) as usize];
        let mut v = (first as u32) & ((1u32 << (8 - len)) - 1);
        let (end, mut p) = (src + len, src + 1);
        while p < end && self.at(p) != 0 {
            v = (v << 6) + (self.at(p) as u32 & 0x3f);
            p += 1;
        }
        (v, p)
    }

    fn char(&self, src: usize) -> R<(u32, usize)> {
        if self.at(src) == b'\\' {
            return match self.at(src + 1) {
                b'x' => self.hex(src + 2, 2),
                b'u' => self.hex(src + 2, 4),
                b'U' => self.hex(src + 2, 8),
                b't' => Ok((b'\t' as u32, src + 2)),
                b'r' => Ok((b'\r' as u32, src + 2)),
                b'n' => Ok((b'\n' as u32, src + 2)),
                c @ (b'\\' | b'"' | b'[' | b']' | b'-') => Ok((c as u32, src + 2)),
                _ => Err(format!("unknown escape at {}", self.rest(src))),
            };
        }
        if self.at(src) != 0 {
            return Ok(self.utf8(src));
        }
        Err("unexpected end of input".into())
    }

    fn token(&self, src: usize) -> R<(u32, usize)> {
        let mut p = src;
        if self.at(p) != b'<' {
            return Err(format!("expecting '<' at {}", self.rest(p)));
        }
        p += 1;
        if self.at(p) == b'[' {
            p += 1;
            let e = self.int(p)?;
            let id: u64 = std::str::from_utf8(&self.s[p..e]).ok().and_then(|t| t.parse().ok()).ok_or("parsed token id is too big")?;
            if id > u32::MAX as u64 {
                return Err(format!("parsed token id is too big at {}", self.rest(p)));
            }
            p = e;
            if self.at(p) != b']' {
                return Err(format!("expecting ']' at {}", self.rest(p)));
            }
            p += 1;
            if self.at(p) != b'>' {
                return Err(format!("expecting '>' at {}", self.rest(p)));
            }
            return Ok((id as u32, p + 1));
        }
        while self.at(p) != 0 && self.at(p) != b'>' {
            p += 1;
        }
        if self.at(p) != b'>' {
            return Err(format!("expecting '>' at {}", self.rest(p)));
        }
        p += 1;
        let ids = (self.tokenize)(&self.s[src..p]);
        if ids.len() != 1 {
            return Err(format!("invalid token '{}'", String::from_utf8_lossy(&self.s[src..p])));
        }
        Ok((ids[0], p))
    }

    fn symbol_id(&mut self, name: &[u8]) -> u32 {
        let next = self.symbol_ids.len() as u32;
        *self.symbol_ids.entry(name.to_vec()).or_insert(next)
    }

    fn generate_symbol_id(&mut self, base: &[u8]) -> u32 {
        let next = self.symbol_ids.len() as u32;
        let mut k = base.to_vec();
        k.push(b'_');
        k.extend(next.to_string().bytes());
        self.symbol_ids.insert(k, next);
        next
    }

    fn add_rule(&mut self, id: u32, rule: Vec<El>) {
        if self.rules.len() <= id as usize {
            self.rules.resize(id as usize + 1, Vec::new());
        }
        self.rules[id as usize] = rule;
    }

    fn alternates(&mut self, src: usize, name: &[u8], id: u32, nested: bool) -> R<usize> {
        let mut rule = Vec::new();
        let mut p = self.sequence(src, name, &mut rule, nested)?;
        while self.at(p) == b'|' {
            rule.push(el(T::Alt, 0));
            p = self.space(p + 1, true);
            p = self.sequence(p, name, &mut rule, nested)?;
        }
        rule.push(el(T::End, 0));
        self.add_rule(id, rule);
        Ok(p)
    }

    fn sequence(&mut self, src: usize, name: &[u8], rule: &mut Vec<El>, nested: bool) -> R<usize> {
        let mut last_sym_start = rule.len();
        let mut p = src;
        let mut n_prev_rules: u64 = 1;
        while self.at(p) != 0 {
            let c = self.at(p);
            if c == b'"' {
                p += 1;
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while self.at(p) != b'"' {
                    if self.at(p) == 0 {
                        return Err("unexpected end of input".into());
                    }
                    let (ch, q) = self.char(p)?;
                    p = q;
                    rule.push(el(T::Char, ch));
                }
                p = self.space(p + 1, nested);
            } else if c == b'[' {
                p += 1;
                let mut start = T::Char;
                if self.at(p) == b'^' {
                    p += 1;
                    start = T::CharNot;
                }
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while self.at(p) != b']' {
                    if self.at(p) == 0 {
                        return Err("unexpected end of input".into());
                    }
                    let (ch, q) = self.char(p)?;
                    p = q;
                    let t = if last_sym_start < rule.len() { T::CharAlt } else { start };
                    rule.push(el(t, ch));
                    if self.at(p) == b'-' && self.at(p + 1) != b']' {
                        if self.at(p + 1) == 0 {
                            return Err("unexpected end of input".into());
                        }
                        let (e, q) = self.char(p + 1)?;
                        p = q;
                        rule.push(el(T::CharRngUpper, e));
                    }
                }
                p = self.space(p + 1, nested);
            } else if c == b'<' || c == b'!' {
                let mut t = T::Token;
                if c == b'!' {
                    t = T::TokenNot;
                    p += 1;
                }
                let (id, e) = self.token(p)?;
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(el(t, id));
                p = self.space(e, nested);
            } else if is_word(c) {
                let e = self.name(p)?;
                let nm = self.s[p..e].to_vec();
                let id = self.symbol_id(&nm);
                p = self.space(e, nested);
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(el(T::RuleRef, id));
            } else if c == b'(' {
                p = self.space(p + 1, true);
                let before = self.symbol_ids.len() as u32;
                let sub = self.generate_symbol_id(name);
                p = self.alternates(p, name, sub, true)?;
                n_prev_rules = 1u32.max((self.symbol_ids.len() as u32).wrapping_sub(before)) as u64;
                last_sym_start = rule.len();
                rule.push(el(T::RuleRef, sub));
                if self.at(p) != b')' {
                    return Err(format!("expecting ')' at {}", self.rest(p)));
                }
                p = self.space(p + 1, nested);
            } else if c == b'.' {
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(el(T::CharAny, 0));
                p = self.space(p + 1, nested);
            } else if c == b'*' || c == b'+' || c == b'?' {
                p = self.space(p + 1, nested);
                let (mn, mx) = match c {
                    b'*' => (0, u64::MAX),
                    b'+' => (1, u64::MAX),
                    _ => (0, 1),
                };
                self.repetitions(rule, last_sym_start, &mut n_prev_rules, name, mn, mx, p)?;
            } else if c == b'{' {
                p = self.space(p + 1, nested);
                if !is_digit(self.at(p)) {
                    return Err(format!("expecting an int at {}", self.rest(p)));
                }
                let e = self.int(p)?;
                let mn: u64 = std::str::from_utf8(&self.s[p..e]).ok().and_then(|t| t.parse().ok()).ok_or("repetition count out of range")?;
                p = self.space(e, nested);
                let mut mx = u64::MAX;
                if self.at(p) == b'}' {
                    mx = mn;
                    p = self.space(p + 1, nested);
                } else if self.at(p) == b',' {
                    p = self.space(p + 1, nested);
                    if is_digit(self.at(p)) {
                        let e = self.int(p)?;
                        mx = std::str::from_utf8(&self.s[p..e]).ok().and_then(|t| t.parse().ok()).ok_or("repetition count out of range")?;
                        p = self.space(e, nested);
                    }
                    if self.at(p) != b'}' {
                        return Err(format!("expecting '}}' at {}", self.rest(p)));
                    }
                    p = self.space(p + 1, nested);
                } else {
                    return Err(format!("expecting ',' at {}", self.rest(p)));
                }
                if mn > MAX_REPETITION_THRESHOLD {
                    return Err("number of repetitions exceeds sane defaults, please reduce the number of repetitions".into());
                }
                if mx != u64::MAX && mx > MAX_REPETITION_THRESHOLD {
                    mx = u64::MAX;
                }
                self.repetitions(rule, last_sym_start, &mut n_prev_rules, name, mn, mx, p)?;
            } else {
                break;
            }
        }
        Ok(p)
    }

    /// S{m,n} → S…S (m times) S'(n−m), S'(x) ::= S S'(x−1) |, …; S{m,} → S…S S', S' ::= S S' |.
    #[allow(clippy::too_many_arguments)]
    fn repetitions(&mut self, rule: &mut Vec<El>, last_sym_start: usize, n_prev_rules: &mut u64, name: &[u8], mn: u64, mx: u64, p: usize) -> R<()> {
        let no_max = mx == u64::MAX;
        if last_sym_start == rule.len() {
            return Err(format!("expecting preceding item to */+/?/{{ at {}", self.rest(p)));
        }
        let prev: Vec<El> = rule[last_sym_start..].to_vec();
        let total = if !no_max && mx > 0 { mx } else if mn > 0 { mn } else { 1 };
        if n_prev_rules.wrapping_mul(total) > MAX_REPETITION_THRESHOLD {
            return Err("number of rules that are going to be repeated multiplied by the new repetition exceeds sane defaults, please reduce the number of repetitions or rule complexity".into());
        }
        if !no_max && mx < mn {
            // llama.cpp computes max − min in unsigned arithmetic here and loops (practically) forever; refused instead
            return Err(format!("repetition {{{mn},{mx}}}: the maximum is below the minimum"));
        }
        if mn == 0 {
            rule.truncate(last_sym_start);
        } else {
            for _ in 1..mn {
                rule.extend_from_slice(&prev);
            }
        }
        let mut last_rec = 0u32;
        let n_opt = if no_max { 1 } else { mx - mn };
        let mut rec = prev.clone();
        for i in 0..n_opt {
            rec.truncate(prev.len());
            let id = self.generate_symbol_id(name);
            if i > 0 || no_max {
                rec.push(el(T::RuleRef, if no_max { id } else { last_rec }));
            }
            rec.push(el(T::Alt, 0));
            rec.push(el(T::End, 0));
            self.add_rule(id, rec.clone());
            last_rec = id;
        }
        if n_opt > 0 {
            rule.push(el(T::RuleRef, last_rec));
        }
        *n_prev_rules = n_prev_rules.wrapping_mul(total);
        Ok(())
    }

    fn rule(&mut self, src: usize) -> R<usize> {
        let e = self.name(src)?;
        let mut p = self.space(e, false);
        let name = self.s[src..e].to_vec();
        let id = self.symbol_id(&name);
        if !(self.at(p) == b':' && self.at(p + 1) == b':' && self.at(p + 2) == b'=') {
            return Err(format!("expecting ::= at {}", self.rest(p)));
        }
        p = self.space(p + 3, true);
        p = self.alternates(p, &name, id, false)?;
        if self.at(p) == b'\r' {
            p += if self.at(p + 1) == b'\n' { 2 } else { 1 };
        } else if self.at(p) == b'\n' {
            p += 1;
        } else if self.at(p) != 0 {
            return Err(format!("expecting newline or end at {}", self.rest(p)));
        }
        Ok(self.space(p, true))
    }
}

fn is_eos(e: El) -> bool {
    matches!(e.t, T::End | T::Alt)
}

/// A parsed, checked grammar: every rule's elements in one array (`start[rule]` is where it begins).
#[derive(Debug)]
pub struct Rules {
    els: Vec<El>,
    start: Vec<u32>,
    root: u32,
}

impl Rules {
    /// `llama_grammar_init_impl(vocab, text, "root")`: parse, check every reference, refuse left recursion. `tokenize`
    /// is the vocabulary's tokenizer with special tokens parsed (for `<token>` terminals).
    pub fn parse(text: &str, tokenize: &dyn Fn(&[u8]) -> Vec<u32>) -> Result<Rules, String> {
        let mut s = text.as_bytes().to_vec();
        s.push(0);
        let mut ps = Parser { s, symbol_ids: Default::default(), rules: Vec::new(), tokenize };
        let mut p = ps.space(0, true);
        while ps.at(p) != 0 {
            p = ps.rule(p).map_err(|e| format!("error parsing grammar: {e}"))?;
        }
        if ps.rules.is_empty() {
            return Err("failed to parse grammar: no rules".into());
        }
        for r in &ps.rules {
            if r.is_empty() {
                return Err("error parsing grammar: Undefined rule".into());
            }
            for e in r {
                if e.t == T::RuleRef && (e.v as usize >= ps.rules.len() || ps.rules[e.v as usize].is_empty()) {
                    let name = ps.symbol_ids.iter().find(|(_, &v)| v == e.v).map(|(k, _)| String::from_utf8_lossy(k).into_owned()).unwrap_or_default();
                    return Err(format!("error parsing grammar: Undefined rule identifier '{name}'"));
                }
            }
        }
        let root = *ps.symbol_ids.get(&b"root"[..]).ok_or("grammar does not contain a 'root' symbol")?;
        let rules = ps.rules;
        let n = rules.len();
        let (mut visited, mut in_progress, mut may_be_empty) = (vec![false; n], vec![false; n], vec![false; n]);
        for i in 0..n {
            if !visited[i] && left_recursion(&rules, i, &mut visited, &mut in_progress, &mut may_be_empty) {
                return Err(format!("unsupported grammar, left recursion detected for nonterminal at index {i}"));
            }
        }
        let (mut els, mut start) = (Vec::new(), Vec::new());
        for r in &rules {
            start.push(els.len() as u32);
            els.extend_from_slice(r);
        }
        Ok(Rules { els, start, root })
    }
}

fn left_recursion(rules: &[Vec<El>], i: usize, visited: &mut [bool], in_progress: &mut [bool], may_be_empty: &mut [bool]) -> bool {
    if in_progress[i] {
        return true;
    }
    in_progress[i] = true;
    let rule = &rules[i];
    let mut at_start = true;
    for e in rule {
        if is_eos(*e) {
            if at_start {
                may_be_empty[i] = true;
                break;
            }
            at_start = true;
        } else {
            at_start = false;
        }
    }
    let mut recurse = true;
    for e in rule {
        if e.t == T::RuleRef && recurse {
            if left_recursion(rules, e.v as usize, visited, in_progress, may_be_empty) {
                return true;
            }
            if !may_be_empty[e.v as usize] {
                recurse = false;
            }
        } else {
            recurse = is_eos(*e);
        }
    }
    in_progress[i] = false;
    visited[i] = true;
    false
}

// ---------------------------------------------------------------- the vocabulary as the grammar sees it ----------

/// A partial UTF-8 sequence: the bits so far and how many continuation bytes are still due (−1: invalid).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Partial {
    value: u32,
    n_remain: i32,
}

/// llama-grammar.cpp's `decode_utf8(const std::string &, partial)`: code points (with a terminating 0) and the
/// sequence left open at the end. Reading stops at a NUL byte, as it does on a C string.
fn decode(src: &[u8], start: Partial) -> (Vec<u32>, Partial) {
    const LOOKUP: [i32; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4];
    let at = |i: usize| src.get(i).copied().unwrap_or(0);
    let mut cps = Vec::with_capacity(src.len() + 1);
    let (mut value, mut n_remain, mut p) = (start.value, start.n_remain, 0usize);
    while at(p) != 0 && n_remain > 0 {
        let b = at(p);
        if b >> 6 != 2 {
            cps.push(0);
            return (cps, Partial { value: 0, n_remain: -1 });
        }
        value = (value << 6) + (b as u32 & 0x3f);
        p += 1;
        n_remain -= 1;
    }
    if start.n_remain > 0 && n_remain == 0 {
        cps.push(value);
    }
    while at(p) != 0 {
        let first = at(p);
        n_remain = LOOKUP[(first >> 4) as usize] - 1;
        if n_remain < 0 {
            cps.clear();
            cps.push(0);
            return (cps, Partial { value: 0, n_remain });
        }
        let mask = (1u32 << (7 - n_remain)) - 1;
        value = first as u32 & mask;
        p += 1;
        while at(p) != 0 && n_remain > 0 {
            value = (value << 6) + (at(p) as u32 & 0x3f);
            p += 1;
            n_remain -= 1;
        }
        if n_remain == 0 {
            cps.push(value);
        }
    }
    cps.push(0);
    (cps, Partial { value, n_remain })
}

/// Every token's piece as llama.cpp's grammar reads it (`token_to_piece`, special tokens rendered), decoded once,
/// and the tokens that end generation (llama-vocab.cpp's EOG set).
pub struct Vocab {
    pieces: Vec<Vec<u8>>,
    decoded: Vec<(Vec<u32>, Partial)>,
    eog: Vec<bool>,
    /// 0.3.9: the candidates' code points as a trie, built on the first whole-vocabulary mask
    trie: std::sync::OnceLock<Trie>,
}

/// 0.3.9: the vocabulary's grammar candidates (not end-of-generation, a piece that does not start with NUL) in a trie
/// over their decoded code points, so a mask walks each stack over a shared prefix once instead of once per token.
/// Tokens are sorted by their code points, so each node's subtree is one range of `order` and the tokens that end at
/// the node come first in it.
struct Trie {
    nodes: Vec<TNode>,
    /// each node's children, sorted by code point: (code point, child)
    edges: Vec<(u32, u32)>,
    order: Vec<u32>,
}

#[derive(Clone, Copy, Default)]
struct TNode {
    edges: (u32, u32),
    ends: (u32, u32),
    sub: (u32, u32),
}

impl Trie {
    fn new(v: &Vocab) -> Trie {
        let key = |id: u32| -> &[u32] {
            let c = &v.decoded[id as usize].0;
            &c[..c.iter().position(|&x| x == 0).unwrap_or(c.len())]
        };
        let mut order: Vec<u32> = (0..v.len() as u32).filter(|&id| !v.is_eog(id) && v.piece(id).first().is_some_and(|&b| b != 0)).collect();
        order.sort_by(|&a, &b| key(a).cmp(key(b)));
        let mut t = Trie { nodes: Vec::new(), edges: Vec::new(), order };
        t.node(&key, 0, t.order.len(), 0);
        t
    }

    fn node<'v>(&mut self, key: &dyn Fn(u32) -> &'v [u32], lo: usize, hi: usize, depth: usize) -> u32 {
        let me = self.nodes.len();
        self.nodes.push(TNode::default());
        let mut i = lo;
        while i < hi && key(self.order[i]).len() == depth {
            i += 1;
        }
        let ends = (lo as u32, i as u32);
        let mut kids = Vec::new();
        while i < hi {
            let cp = key(self.order[i])[depth];
            let mut j = i;
            while j < hi && key(self.order[j])[depth] == cp {
                j += 1;
            }
            kids.push((cp, self.node(key, i, j, depth + 1)));
            i = j;
        }
        let e0 = self.edges.len() as u32;
        self.edges.extend(kids);
        self.nodes[me] = TNode { edges: (e0, self.edges.len() as u32), ends, sub: (lo as u32, hi as u32) };
        me as u32
    }
}

impl Vocab {
    /// Every token's `token_to_piece(special = true)` bytes, by id (DRY's breakers read them too).
    pub fn pieces(&self) -> &[Vec<u8>] {
        &self.pieces
    }

    pub fn new(pieces: Vec<Vec<u8>>, eog_ids: &[u32]) -> Vocab {
        let decoded = pieces.iter().map(|p| decode(p, Partial::default())).collect();
        let mut eog = vec![false; pieces.len()];
        for &i in eog_ids {
            if let Some(e) = eog.get_mut(i as usize) {
                *e = true;
            }
        }
        Vocab { pieces, decoded, eog, trie: Default::default() }
    }
    pub fn piece(&self, id: u32) -> &[u8] {
        self.pieces.get(id as usize).map(Vec::as_slice).unwrap_or(&[])
    }
    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.get(id as usize).copied().unwrap_or(false)
    }
    pub fn len(&self) -> usize {
        self.pieces.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }
}

// ---------------------------------------------------------------- the pushdown automaton ----------

type Stack = Vec<u32>;

/// From this many candidates a mask walks the vocabulary's trie rather than each candidate.
const TRIE_MIN: usize = 1024;

/// A grammar's state: the rules, the set of stacks, and a UTF-8 sequence a token may have left open.
pub struct Grammar {
    r: std::sync::Arc<Rules>,
    stacks: Vec<Stack>,
    partial: Partial,
}

#[derive(Clone, Copy)]
struct GCand<'a> {
    index: usize,
    cps: &'a [u32],
    at: usize,
    partial: Partial,
    id: u32,
}

impl Grammar {
    /// The initial stacks: every alternate of `root`, advanced to its first terminals.
    pub fn new(r: std::sync::Arc<Rules>) -> Grammar {
        let mut g = Grammar { r, stacks: Vec::new(), partial: Partial::default() };
        let mut p = g.r.start[g.r.root as usize];
        let mut stacks = Vec::new();
        loop {
            let mut st = Vec::new();
            if !is_eos(g.r.els[p as usize]) {
                st.push(p);
            }
            g.advance(st, &mut stacks);
            while !is_eos(g.r.els[p as usize]) {
                p += 1;
            }
            if g.r.els[p as usize].t == T::Alt {
                p += 1;
            } else {
                break;
            }
        }
        g.stacks = stacks;
        g
    }

    fn e(&self, p: u32) -> El {
        self.r.els[p as usize]
    }

    /// `llama_grammar_advance_stack`: a stack expanded into the stacks that end at a terminal (or are empty).
    fn advance(&self, stack: Stack, out: &mut Vec<Stack>) {
        let mut todo = vec![stack];
        let mut seen: std::collections::HashSet<Stack> = Default::default();
        while let Some(cur) = todo.pop() {
            if seen.contains(&cur) {
                continue;
            }
            seen.insert(cur.clone());
            let Some(&pos) = cur.last() else {
                if !out.contains(&cur) {
                    out.push(cur);
                }
                continue;
            };
            match self.e(pos).t {
                T::RuleRef => {
                    let mut sub = self.r.start[self.e(pos).v as usize];
                    loop {
                        let mut next: Stack = cur[..cur.len() - 1].to_vec();
                        if !is_eos(self.e(pos + 1)) {
                            next.push(pos + 1);
                        }
                        if !is_eos(self.e(sub)) {
                            next.push(sub);
                        }
                        todo.push(next);
                        while !is_eos(self.e(sub)) {
                            sub += 1;
                        }
                        if self.e(sub).t == T::Alt {
                            sub += 1;
                        } else {
                            break;
                        }
                    }
                }
                T::Char | T::CharNot | T::CharAny | T::Token | T::TokenNot => {
                    if !out.contains(&cur) {
                        out.push(cur);
                    }
                }
                _ => unreachable!("a stack left on the end of an alternate or inside a character range"),
            }
        }
    }

    /// `llama_grammar_match_char`: whether `chr` satisfies the range at `pos`, and the element after the range.
    fn match_char(&self, mut pos: u32, chr: u32) -> (bool, u32) {
        let positive = matches!(self.e(pos).t, T::Char | T::CharAny);
        let mut found = false;
        loop {
            if self.e(pos + 1).t == T::CharRngUpper {
                found = found || (self.e(pos).v <= chr && chr <= self.e(pos + 1).v);
                pos += 2;
            } else if self.e(pos).t == T::CharAny {
                found = true;
                pos += 1;
            } else {
                found = found || self.e(pos).v == chr;
                pos += 1;
            }
            if self.e(pos).t != T::CharAlt {
                break;
            }
        }
        (found == positive, pos)
    }

    /// `llama_grammar_match_partial_char`: whether some completion of an open UTF-8 sequence could satisfy `pos`.
    fn match_partial_char(&self, mut pos: u32, pu: Partial) -> bool {
        let positive = matches!(self.e(pos).t, T::Char | T::CharAny);
        let (value, n_remain) = (pu.value, pu.n_remain);
        if n_remain < 0 || (n_remain == 1 && value < 2) {
            return false;
        }
        let mut low = value.wrapping_shl((n_remain * 6) as u32);
        let high = low | ((1u32 << (n_remain * 6)) - 1);
        if low == 0 {
            if n_remain == 2 {
                low = 1 << 11;
            } else if n_remain == 3 {
                low = 1 << 16;
            }
        }
        loop {
            if self.e(pos + 1).t == T::CharRngUpper {
                if self.e(pos).v <= high && low <= self.e(pos + 1).v {
                    return positive;
                }
                pos += 2;
            } else if self.e(pos).t == T::CharAny {
                return true;
            } else {
                if low <= self.e(pos).v && self.e(pos).v <= high {
                    return positive;
                }
                pos += 1;
            }
            if self.e(pos).t != T::CharAlt {
                break;
            }
        }
        !positive
    }

    fn match_token(&self, pos: u32, id: u32) -> bool {
        match self.e(pos).t {
            T::Token => self.e(pos).v == id,
            T::TokenNot => self.e(pos).v != id,
            _ => false,
        }
    }

    fn reject<'a>(&self, stacks: &[Stack], cands: Vec<GCand<'a>>) -> Vec<GCand<'a>> {
        assert!(!stacks.is_empty());
        if cands.is_empty() {
            return cands;
        }
        let mut r = self.reject_for_stack(&stacks[0], &cands);
        for s in &stacks[1..] {
            r = self.reject_for_stack(s, &r);
        }
        r
    }

    /// `llama_grammar_reject_candidates_for_stack`.
    fn reject_for_stack<'a>(&self, stack: &[u32], cands: &[GCand<'a>]) -> Vec<GCand<'a>> {
        let mut rejects = Vec::with_capacity(cands.len());
        let Some(&sp) = stack.last() else {
            rejects.extend(cands.iter().filter(|c| c.cps[c.at] != 0 || c.partial.n_remain != 0));
            return rejects;
        };
        if matches!(self.e(sp).t, T::Token | T::TokenNot) {
            for c in cands {
                if c.cps[c.at] == 0 {
                    if c.partial.n_remain != 0 {
                        rejects.push(*c);
                    }
                } else if !self.match_token(sp, c.id) {
                    rejects.push(*c);
                }
            }
            return rejects;
        }
        let mut next = Vec::with_capacity(cands.len());
        for c in cands {
            if c.cps[c.at] == 0 {
                if c.partial.n_remain != 0 && !self.match_partial_char(sp, c.partial) {
                    rejects.push(*c);
                }
            } else if self.match_char(sp, c.cps[c.at]).0 {
                next.push(GCand { at: c.at + 1, ..*c });
            } else {
                rejects.push(*c);
            }
        }
        let after = self.match_char(sp, 0).1;
        let mut stack_after: Stack = stack[..stack.len() - 1].to_vec();
        if !is_eos(self.e(after)) {
            stack_after.push(after);
        }
        let mut next_stacks = Vec::new();
        self.advance(stack_after, &mut next_stacks);
        for c in self.reject(&next_stacks, next) {
            rejects.push(GCand { at: c.at - 1, ..c });
        }
        rejects
    }

    /// `llama_grammar_apply_impl`: every candidate the grammar rejects gets a logit of −∞. An end-of-generation token
    /// passes only when a stack is empty (the grammar is complete); an empty piece never does.
    pub fn apply(&self, v: &Vocab, cands: &mut [Cand]) {
        // 0.3.9: a long candidate list with no UTF-8 left open walks the trie; the same mask (oracle_grammar_masks)
        if self.partial.n_remain == 0 && cands.len() >= TRIE_MIN {
            self.apply_trie(v, cands)
        } else {
            self.apply_each(v, cands)
        }
    }

    /// `apply` through the vocabulary's trie (no UTF-8 left open by the last token).
    fn apply_trie(&self, v: &Vocab, cands: &mut [Cand]) {
        debug_assert_eq!(self.partial.n_remain, 0);
        let allow_eog = self.stacks.iter().any(Vec::is_empty);
        let t = v.trie.get_or_init(|| Trie::new(v));
        let mut ok = vec![false; v.len()];
        let mut after = Default::default();
        for s in &self.stacks {
            self.walk(t, v, s, 0, &mut ok, &mut after);
        }
        for c in cands.iter_mut() {
            let pass = if v.is_eog(c.id) { allow_eog } else { ok.get(c.id as usize).copied().unwrap_or(false) };
            if !pass {
                c.logit = f32::NEG_INFINITY;
            }
        }
    }

    /// The tokens under trie node `node` that `stack` accepts, marked in `ok` — `reject_for_stack` over a whole
    /// subtree at once: a token ending here is judged by its open UTF-8, a deeper one by the next code point.
    /// `after` memoizes, per stack, the stacks that follow its terminal (`advance` of the rest): they depend on the
    /// stack alone, not on the node, so each is computed once per mask rather than once per node. Keyed by the
    /// stack's address and length: every stack walked lives in `self.stacks` or in an `Rc` the memo keeps until the
    /// mask is done, so an address is never reused for another stack meanwhile.
    fn walk(&self, t: &Trie, v: &Vocab, stack: &[u32], node: u32, ok: &mut [bool], after: &mut std::collections::HashMap<(usize, usize), std::rc::Rc<Vec<Stack>>>) {
        let n = t.nodes[node as usize];
        let ends = &t.order[n.ends.0 as usize..n.ends.1 as usize];
        let partial = |id: u32| v.decoded[id as usize].1;
        let Some(&sp) = stack.last() else {
            for &id in ends {
                if partial(id).n_remain == 0 {
                    ok[id as usize] = true;
                }
            }
            return;
        };
        if matches!(self.e(sp).t, T::Token | T::TokenNot) {
            for &id in ends {
                if partial(id).n_remain == 0 {
                    ok[id as usize] = true;
                }
            }
            for &id in &t.order[n.ends.1 as usize..n.sub.1 as usize] {
                if self.match_token(sp, id) {
                    ok[id as usize] = true;
                }
            }
            return;
        }
        for &id in ends {
            let pu = partial(id);
            if pu.n_remain == 0 || self.match_partial_char(sp, pu) {
                ok[id as usize] = true;
            }
        }
        let mut next: Option<std::rc::Rc<Vec<Stack>>> = None;
        for &(cp, child) in &t.edges[n.edges.0 as usize..n.edges.1 as usize] {
            if !self.match_char(sp, cp).0 {
                continue;
            }
            let stacks = next.get_or_insert_with(|| {
                after.entry((stack.as_ptr() as usize, stack.len())).or_insert_with(|| {
                    let end = self.match_char(sp, 0).1;
                    let mut st: Stack = stack[..stack.len() - 1].to_vec();
                    if !is_eos(self.e(end)) {
                        st.push(end);
                    }
                    let mut out = Vec::new();
                    self.advance(st, &mut out);
                    std::rc::Rc::new(out)
                }).clone()
            }).clone();
            for s in stacks.iter() {
                self.walk(t, v, s, child, ok, after);
            }
        }
    }

    /// `apply`, one candidate at a time, as llama.cpp writes it (short lists, and UTF-8 left open by the last token).
    fn apply_each(&self, v: &Vocab, cands: &mut [Cand]) {
        let allow_eog = self.stacks.iter().any(Vec::is_empty);
        let fresh: Vec<(usize, (Vec<u32>, Partial))> = if self.partial.n_remain > 0 {
            cands.iter().enumerate().filter(|(_, c)| !v.is_eog(c.id) && v.piece(c.id).first().is_some_and(|&b| b != 0))
                .map(|(i, c)| (i, decode(v.piece(c.id), self.partial))).collect()
        } else {
            Vec::new()
        };
        let mut g: Vec<GCand> = Vec::with_capacity(cands.len());
        let mut k = 0;
        for (i, c) in cands.iter_mut().enumerate() {
            if v.is_eog(c.id) {
                if !allow_eog {
                    c.logit = f32::NEG_INFINITY;
                }
            } else if v.piece(c.id).first().is_none_or(|&b| b == 0) {
                c.logit = f32::NEG_INFINITY;
            } else if self.partial.n_remain > 0 {
                let (_, (cps, pu)) = &fresh[k];
                k += 1;
                g.push(GCand { index: i, cps, at: 0, partial: *pu, id: c.id });
            } else {
                let (cps, pu) = &v.decoded[c.id as usize];
                g.push(GCand { index: i, cps, at: 0, partial: *pu, id: c.id });
            }
        }
        for r in self.reject(&self.stacks, g) {
            cands[r.index].logit = f32::NEG_INFINITY;
        }
    }

    /// The single-candidate check of `common_sampler_sample`: would `apply` leave `id` above −∞?
    pub fn allows(&self, v: &Vocab, id: u32) -> bool {
        let mut c = [Cand { id, logit: 1.0, p: 0.0 }];
        self.apply(v, &mut c);
        c[0].logit != f32::NEG_INFINITY
    }

    fn accept_chr(&self, stack: &[u32], chr: u32, out: &mut Vec<Stack>) {
        let Some(&pos) = stack.last() else { return };
        if matches!(self.e(pos).t, T::Token | T::TokenNot) {
            return;
        }
        let (ok, next) = self.match_char(pos, chr);
        if ok {
            let mut ns: Stack = stack[..stack.len() - 1].to_vec();
            if !is_eos(self.e(next)) {
                ns.push(next);
            }
            self.advance(ns, out);
        }
    }

    /// `llama_grammar_accept_impl`: the token taken into the grammar. An end-of-generation token is taken only by a
    /// complete grammar; a token that leaves no stack is an error (llama.cpp throws; bankML never samples one).
    pub fn accept(&mut self, v: &Vocab, id: u32) -> Result<(), String> {
        if v.is_eog(id) {
            return if self.stacks.iter().any(Vec::is_empty) { Ok(()) } else { Err(format!("end-of-generation token {id} before the grammar is complete")) };
        }
        let piece = v.piece(id);
        let (cps, pu) = decode(piece, self.partial);
        let mut new: Vec<Stack> = Vec::with_capacity(self.stacks.len());
        for st in &self.stacks {
            let Some(&pos) = st.last() else { continue };
            if matches!(self.e(pos).t, T::Token | T::TokenNot) {
                if self.match_token(pos, id) {
                    let mut ns: Stack = st[..st.len() - 1].to_vec();
                    if !is_eos(self.e(pos + 1)) {
                        ns.push(pos + 1);
                    }
                    self.advance(ns, &mut new);
                }
            } else {
                let mut cur = vec![st.clone()];
                for &cp in &cps[..cps.len() - 1] {
                    let mut next = Vec::new();
                    for s in &cur {
                        self.accept_chr(s, cp, &mut next);
                    }
                    cur = next;
                    if cur.is_empty() {
                        break;
                    }
                }
                for s in cur {
                    if !new.contains(&s) {
                        new.push(s);
                    }
                }
            }
        }
        self.stacks = new;
        self.partial = pu;
        if self.stacks.is_empty() {
            return Err(format!("Unexpected empty grammar stack after accepting piece: {} ({id})", String::from_utf8_lossy(piece)));
        }
        Ok(())
    }

    /// Whether a stack is empty (the grammar could end here).
    pub fn complete(&self) -> bool {
        self.stacks.iter().any(Vec::is_empty)
    }
}

/// The FNV-1a 64 of a mask's allowed ids (little-endian u32 each), as `testing/grammar_oracle.cpp` prints it.
pub fn mask_hash(allowed: impl Iterator<Item = u32>) -> (usize, u64) {
    let (mut h, mut n) = (1_469_598_103_934_665_603u64, 0usize);
    for id in allowed {
        n += 1;
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1_099_511_628_211);
        }
    }
    (n, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// 0.3.5: llama.cpp b11192's own chat parser on every prefix of every recorded constrained answer, and edge cases
    /// (testing/content_oracle.py): bankML's content is the parser's, and the answered message is the server's.
    #[test]
    #[ignore = "needs .models/oracle-content/content-*.jsonl (testing/content_oracle.py)"]
    fn oracle_json_content() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/oracle-content");
        for t in ["qwen3", "smollm2", "chatml"] {
            let recs = std::fs::read_to_string(dir.join(format!("content-{t}.jsonl"))).unwrap();
            let (mut n, mut same, mut refused, mut bad) = (0, 0, 0, Vec::new());
            for line in recs.lines() {
                let r = Json::parse(line).unwrap();
                let raw = r.get("raw").and_then(Json::as_str).unwrap();
                n += 1;
                match r.get("content").and_then(Json::as_str) {
                    Some(want) => {
                        let server = if want.is_empty() { raw } else { want };
                        if json_content(raw) == want && json_message(raw) == server {
                            same += 1;
                        } else {
                            bad.push(format!("{:?} under {}: bankML {:?} / {:?}, llama.cpp {want:?} / {server:?}", raw,
                                             r.get("schema").and_then(Json::as_str).unwrap_or(""), json_content(raw), json_message(raw)));
                        }
                    }
                    None => {
                        // the parser refuses only a reasoning block in the answer, which no JSON-mode or schema grammar
                        // admits after its prefill (the answers bankML gives never contain one)
                        assert!(raw.starts_with("<think>"), "llama.cpp refuses {raw:?}");
                        refused += 1;
                    }
                }
            }
            eprintln!("json content oracle ({t}): {same} of {} raw texts give llama.cpp b11192's content and llama-server's message \
                       ({refused} reasoning-block texts refused by its parser, unreachable under the grammar)", n - refused);
            for b in bad.iter().take(12) {
                eprintln!("  {b}");
            }
            assert!(bad.is_empty(), "{} differ", bad.len());
        }
    }

    fn no_tok(_: &[u8]) -> Vec<u32> {
        Vec::new()
    }

    /// A toy vocabulary: one token per printable ASCII byte, a few multi-byte pieces, and an end token.
    fn toy() -> (Vocab, Vec<String>) {
        let mut names: Vec<String> = (0x20u8..0x7f).map(|b| (b as char).to_string()).collect();
        for s in ["{\"", "\":", "true", "é", "\u{4e2d}", "ab"] {
            names.push(s.to_string());
        }
        let mut pieces: Vec<Vec<u8>> = names.iter().map(|s| s.as_bytes().to_vec()).collect();
        pieces.push(vec![0xc3]); // the first half of é
        pieces.push(vec![0xa9]); // its second half
        pieces.push(b"<end>".to_vec());
        names.extend(["<c3>".into(), "<a9>".into(), "<end>".into()]);
        let eog = vec![pieces.len() as u32 - 1];
        (Vocab::new(pieces, &eog), names)
    }

    /// The allowed tokens; with no UTF-8 open, the trie's mask must equal the one-by-one mask (0.3.9).
    fn allowed(g: &Grammar, v: &Vocab) -> Vec<u32> {
        let all = || (0..v.len() as u32).map(|id| Cand { id, logit: 0.0, p: 0.0 }).collect::<Vec<Cand>>();
        let pass = |c: &[Cand]| c.iter().filter(|c| c.logit != f32::NEG_INFINITY).map(|c| c.id).collect::<Vec<u32>>();
        let mut c = all();
        g.apply_each(v, &mut c);
        if g.partial.n_remain == 0 {
            let mut t = all();
            g.apply_trie(v, &mut t);
            assert_eq!(pass(&t), pass(&c), "the trie's mask");
        }
        pass(&c)
    }

    fn id(names: &[String], s: &str) -> u32 {
        names.iter().position(|n| n == s).unwrap() as u32
    }

    #[test]
    fn the_server_grammar_parses() {
        let r = Rules::parse(JSON_OBJECT_GRAMMAR, &no_tok).unwrap();
        assert!(r.els.len() > 200);
    }

    #[test]
    fn parse_errors_say_where() {
        for (g, want) in [("root ::= foo", "Undefined rule"), ("root ::= \"a", "unexpected end"), ("root ::= [a", "unexpected end"),
                          ("x ::= \"a\"", "'root'"), ("root ::= root \"a\"", "left recursion"), ("root ::= \"\\q\"", "unknown escape"),
                          ("root ::= \"a\"{3,1}", "below the minimum"), ("root ::= *", "expecting preceding item")] {
            let e = Rules::parse(g, &no_tok).unwrap_err();
            assert!(e.contains(want), "{g}: {e}");
        }
    }

    #[test]
    fn a_small_grammar_masks_and_accepts() {
        let (v, n) = toy();
        let r = Arc::new(Rules::parse("root ::= \"{\\\"\" [a-c]{1,2} \"\\\":\" (\"true\" | [0-9]+) \"}\"", &no_tok).unwrap());
        let mut g = Grammar::new(r);
        // the first byte: '{' alone or the two-byte piece {"
        let a = allowed(&g, &v);
        assert_eq!(a, vec![id(&n, "{"), id(&n, "{\"")]);
        g.accept(&v, id(&n, "{\"")).unwrap();
        assert!(allowed(&g, &v).contains(&id(&n, "ab")));
        g.accept(&v, id(&n, "ab")).unwrap();
        // two letters taken: the range is spent, so only the closing quote (alone or as \":)
        assert_eq!(allowed(&g, &v), vec![id(&n, "\""), id(&n, "\":")]);
        g.accept(&v, id(&n, "\":")).unwrap();
        g.accept(&v, id(&n, "7")).unwrap();
        assert!(!g.complete());
        assert!(!g.allows(&v, v.len() as u32 - 1)); // the end token: not before the grammar is complete
        g.accept(&v, id(&n, "}")).unwrap();
        assert!(g.complete());
        assert_eq!(allowed(&g, &v), vec![v.len() as u32 - 1]);
        assert!(g.accept(&v, id(&n, "x")).is_err());
    }

    #[test]
    fn partial_utf8_across_tokens() {
        let (v, n) = toy();
        let mut g = Grammar::new(Arc::new(Rules::parse("root ::= [^a-z]+", &no_tok).unwrap()));
        let (c3, a9) = (id(&n, "<c3>"), id(&n, "<a9>"));
        // a lead byte alone may complete to a character the range allows; a lone continuation byte never may
        assert!(g.allows(&v, c3));
        assert!(!g.allows(&v, a9));
        g.accept(&v, c3).unwrap();
        assert_eq!(g.partial.n_remain, 1);
        // mid-character: only a continuation completes it
        assert!(g.allows(&v, a9));
        assert!(!g.allows(&v, id(&n, "A")));
        g.accept(&v, a9).unwrap();
        assert!(g.complete());
        // a range that excludes é: the lead byte that can only make é-like characters (U+00C0–U+00FF) is refused
        let g2 = Grammar::new(Arc::new(Rules::parse("root ::= [\\x00-\\x7F]+", &no_tok).unwrap()));
        assert!(!g2.allows(&v, c3));
        assert!(!g2.allows(&v, id(&n, "é")));
        assert!(g2.allows(&v, id(&n, "A")));
    }

    #[test]
    fn decode_matches_the_c_reading() {
        assert_eq!(decode("aé".as_bytes(), Partial::default()), (vec![0x61, 0xe9, 0], Partial { value: 0xe9, n_remain: 0 }));
        assert_eq!(decode(&[0xe4, 0xb8], Partial::default()), (vec![0], Partial { value: 0x138, n_remain: 1 }));
        assert_eq!(decode(&[0xad], Partial { value: 0x138, n_remain: 1 }), (vec![0x4e2d, 0], Partial { value: 0x4e2d, n_remain: 0 }));
        assert_eq!(decode(&[0x80, 0x41], Partial::default()).1.n_remain, -1);
        assert_eq!(decode(&[0x41, 0, 0x42], Partial::default()).0, vec![0x41, 0]);
    }

    #[test]
    fn requests_resolve_as_llama_server_does() {
        let c = |j: &str| from_openai(&Json::parse(j).unwrap());
        assert_eq!(c(r#"{"messages": []}"#), Ok(Constraint::None));
        assert_eq!(c(r#"{"response_format": {"type": "text"}}"#), Ok(Constraint::None));
        assert_eq!(c(r#"{"response_format": {"type": "json_object"}}"#), Ok(Constraint::JsonObject));
        assert_eq!(c(r#"{"response_format": {"type": "json_object", "schema": {"type": "object"}}}"#), Ok(Constraint::JsonObject));
        assert_eq!(c(r#"{"response_format": {"type": "json_schema", "json_schema": {"schema": {}}}}"#), Ok(Constraint::JsonObject));
        assert_eq!(c(r#"{"json_schema": {}}"#), Ok(Constraint::JsonObject));
        assert_eq!(c(r#"{"grammar": "root ::= \"a\""}"#), Ok(Constraint::Gbnf("root ::= \"a\"".into())));
        // O6b: any schema, converted as llama-server converts it; what b11192 refuses or ignores, the same way
        let g = |c: &Constraint| match c {
            Constraint::Schema(s) => crate::schema::chat_grammar(s, crate::chat::Template::Qwen3).unwrap().0,
            _ => String::new(),
        };
        let arr = c(r#"{"response_format": {"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": {"type": "array"}}}}"#).unwrap();
        assert!(g(&arr).contains("response-format ::= array\n"), "{arr:?}");
        assert_eq!(c(r#"{"json_schema": {"type": "array"}}"#), Ok(arr.clone()));
        assert_eq!(c(r#"{"response_format": {"type": "json_object", "schema": {"type": "array"}}}"#), Ok(arr.clone()));
        assert_eq!(c(r#"{"response_format": {"type": "json_object"}, "json_schema": {"type": "array"}}"#), Ok(arr));
        assert!(c(r#"{"json_schema": {"type": "frob"}}"#).unwrap_err().ends_with("JSON schema error at #: unrecognized type frob"));
        assert!(c(r#"{"json_schema": "x"}"#).unwrap_err().contains("schema must be an object"));
        assert!(c(r#"{"json_schema": null}"#).unwrap_err().contains("generation prompt"));
        assert_eq!(c(r#"{"json_schema": null, "grammar": "root ::= \"a\""}"#), Ok(Constraint::Gbnf("root ::= \"a\"".into())));
        assert_eq!(c(r#"{"response_format": {"type": "json_schema", "json_schema": {"schema": [1]}}}"#), Ok(Constraint::None));
        assert_eq!(c(r#"{"response_format": null}"#), Ok(Constraint::None));
        // a float literal stays a float when the request's own text is read
        let body = r#"{"json_schema": {"enum": [2.0, 3]}}"#;
        let exact = from_openai_text(&Json::parse(body).unwrap(), body).unwrap();
        assert!(g(&exact).contains("(\"2.0\" | \"3\")"), "{exact:?}");
        assert!(c(r#"{"response_format": {"type": "xml"}}"#).unwrap_err().contains("but got: xml"));
        assert!(c(r#"{"json_schema": {}, "grammar": "root ::= \"a\""}"#).unwrap_err().contains("both"));
        assert!(c(r#"{"response_format": {"type": "json_object"}, "grammar": "root ::= \"a\""}"#).is_err());
        let o = |j: &str| from_ollama(Json::parse(j).as_ref());
        assert_eq!(o("\"json\""), Ok(Constraint::JsonObject));
        assert_eq!(o(r#"{"type": "object"}"#), Ok(Constraint::JsonObject));
        assert_eq!(o("null"), Ok(Constraint::None));
        assert!(matches!(o(r#"{"type": "object", "properties": {"a": {"type": "string"}}}"#), Ok(Constraint::Schema(_))));
        assert_eq!(o("{}"), Ok(Constraint::JsonObject));
        assert!(o(r#"{"type": "frob"}"#).unwrap_err().starts_with("format: "));
        assert!(o("[1]").is_err());
        assert!(o("\"yaml\"").is_err());
    }

    #[test]
    fn json_content_is_the_value() {
        assert_eq!(json_content("```json\n{\n  \"a\": 1\n}\n```"), "{\n  \"a\": 1\n}");
        assert_eq!(json_content("  {\"a\": \"}\\\"\"} \n"), "{\"a\": \"}\\\"\"}");
        assert_eq!(json_content("```json\n{\n  \"a\": [1, {\"b\": 2"), "{\n  \"a\": [1, {\"b\": 2");
        assert_eq!(json_content("{\n "), "{\n ");
        assert_eq!(json_content("``"), "");
        assert_eq!(json_content("\n"), "");
        // O6b: values that are not objects
        assert_eq!(json_content(" \"yes\" "), "\"yes\"");
        assert_eq!(json_content("```json\n\"a\\\"b\"\n```"), "\"a\\\"b\"");
        assert_eq!(json_content("\"ope"), "\"ope");
        assert_eq!(json_content("-12.5e3 \n"), "-12.5e3");
        assert_eq!(json_content("```json\ntrue\n```"), "true");
        assert_eq!(json_content("[1, 2] "), "[1, 2]");
        // 0.3.5: a cut inside an escape (llama.cpp's parser ends the string before it)
        assert_eq!(json_content("{\"k\": \"v\\"), "{\"k\": \"v");
        assert_eq!(json_content("\"a\\u00e"), "\"a");
        assert_eq!(json_content("\"a\\u00e9"), "\"a\\u00e9");
        assert_eq!(json_content("{\"k\": \"v\\\\\\"), "{\"k\": \"v\\\\");
        assert_eq!(json_content("\"\\ud83d\\ude"), "\"\\ud83d");
        assert_eq!((json_message("```json\n\n"), json_message("```json\n1.5")), ("```json\n\n", "1.5"));
        // growth only: every prefix's content is a prefix of the next one's
        let raw = "```json\n{\"k\": [\"x\", {\"y\": null}], \"z\": \"}\"}\n```";
        let mut prev = "";
        for i in 0..=raw.len() {
            let c = json_content(&raw[..i]);
            assert!(c.starts_with(prev), "{i}: {c:?} after {prev:?}");
            prev = c;
        }
    }

    /// llama.cpp b11192's grammar sampler on the pinned vocabulary (testing/grammar_oracle.py): every token's piece and
    /// end flag, then every recorded mask over the whole vocabulary and every rejection, for 12 grammars × inputs ×
    /// two tokenizations. Also times the whole-vocabulary mask (the cost of a resampled step).
    #[test]
    #[ignore = "needs .models/oracle-grammar (testing/grammar_oracle.py) and .models/Bonsai-8B-Q1_0.gguf; --release"]
    fn oracle_grammar_masks() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let model = dir.join("Bonsai-8B-Q1_0.gguf");
        let tok = crate::tokenizer::Tokenizer::from_gguf(&model).unwrap();
        let eog = crate::native::eog_from_gguf(&model, &tok).unwrap();
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        let rec = std::fs::read_to_string(dir.join("oracle-grammar/pieces-Bonsai-8B-Q1_0.txt")).unwrap();
        let mut bad = 0;
        for (i, line) in rec.lines().enumerate() {
            let want = format!("{i} {} {}", hex(&tok.piece(i as u32)), eog.contains(&(i as u32)) as u8);
            if line != want {
                bad += 1;
                if bad < 5 {
                    eprintln!("  piece {i}: got {want:?}, llama.cpp {line:?}");
                }
            }
        }
        assert_eq!(rec.lines().count(), tok.n_tokens());
        eprintln!("grammar oracle: {} of {} token pieces and end flags identical to llama.cpp b11192 (end set {eog:?})", tok.n_tokens() - bad, tok.n_tokens());
        assert_eq!(bad, 0);
        let v = Vocab::new((0..tok.n_tokens() as u32).map(|i| tok.piece(i)).collect(), &eog);
        let gs = Json::parse(&std::fs::read_to_string(dir.join("oracle-grammar/grammars.json")).unwrap()).unwrap();
        assert_eq!(gs.get("chat-json-object").and_then(Json::as_str), Some(JSON_OBJECT_GRAMMAR), "the server's JSON grammar is the constant");
        let tokenize = |b: &[u8]| tok.encode(&String::from_utf8_lossy(b), true);
        let cases = std::fs::read_to_string(dir.join("oracle-grammar/masks-Bonsai-8B-Q1_0.jsonl")).unwrap();
        let ids = |x: &Json| match x { Json::Arr(a) => a.iter().map(|n| match n { Json::Num(n) => *n as u64, _ => panic!() }).collect::<Vec<u64>>(), _ => panic!() };
        let (mut runs, mut masks, mut rejections, mut fails) = (0, 0, 0, 0);
        let (mut times, mut trie_times) = (Vec::new(), Vec::new());
        let mut rules: std::collections::HashMap<String, Arc<Rules>> = Default::default();
        for line in cases.lines() {
            let c = Json::parse(line).unwrap();
            let name = c.get("grammar").and_then(Json::as_str).unwrap().to_string();
            let r = rules.entry(name.clone()).or_insert_with(|| Arc::new(Rules::parse(gs.get(&name).and_then(Json::as_str).unwrap(), &tokenize).unwrap())).clone();
            let mut g = Grammar::new(r);
            for id in ids(c.get("prefill").unwrap()) {
                g.accept(&v, id as u32).unwrap();
            }
            let toks = ids(c.get("tokens").unwrap());
            let want: Vec<String> = match c.get("masks").unwrap() {
                Json::Arr(a) => a.iter().map(|m| match m { Json::Arr(x) => match (&x[0], &x[1]) { (Json::Num(n), Json::Str(h)) => format!("{} {h}", *n as u64), _ => panic!() }, _ => panic!() }).collect(),
                _ => panic!(),
            };
            let want_rej = match c.get("rejected_at") { Some(Json::Num(n)) => Some(*n as usize), _ => None };
            let (mut got_rej, mut ok) = (None, true);
            for step in 0..=toks.len() {
                // 0.3.9: both paths, each against llama.cpp's mask: one candidate at a time, and the trie's walk
                let all = || (0..v.len() as u32).map(|id| Cand { id, logit: 0.0, p: 0.0 }).collect::<Vec<Cand>>();
                let t0 = std::time::Instant::now();
                let mut each = all();
                g.apply_each(&v, &mut each);
                times.push(t0.elapsed().as_secs_f64() * 1e3);
                let t1 = std::time::Instant::now();
                let mut cs = all();
                g.apply(&v, &mut cs);
                trie_times.push(t1.elapsed().as_secs_f64() * 1e3);
                let mask = |c: &[Cand]| mask_hash(c.iter().filter(|c| c.logit != f32::NEG_INFINITY).map(|c| c.id));
                if mask(&each) != mask(&cs) {
                    ok = false;
                    eprintln!("  {name} step {step}: the trie's mask differs from the one-by-one mask");
                    break;
                }
                let (n, h) = mask(&cs);
                masks += 1;
                if want.get(step) != Some(&format!("{n} {h:016x}")) {
                    ok = false;
                    eprintln!("  {name} {:?} ({}) step {step}: {n} allowed, llama.cpp {:?}", c.get("input").and_then(Json::as_str).unwrap(),
                              c.get("how").and_then(Json::as_str).unwrap(), want.get(step));
                    break;
                }
                if step == toks.len() {
                    break;
                }
                let id = toks[step] as u32;
                let refused = if v.is_eog(id) { !g.allows(&v, id) } else { false };
                if refused || g.accept(&v, id).is_err() {
                    got_rej = Some(step);
                    break;
                }
            }
            if got_rej != want_rej && ok {
                ok = false;
                eprintln!("  {name} {:?}: rejected at {got_rej:?}, llama.cpp {want_rej:?}", c.get("input").and_then(Json::as_str).unwrap());
            }
            runs += 1;
            rejections += got_rej.is_some() as usize;
            fails += !ok as usize;
        }
        let q = |t: &mut Vec<f64>| {
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (t[t.len() / 2], t[t.len() * 9 / 10], t[t.len() - 1])
        };
        let (a, b) = (q(&mut times), q(&mut trie_times));
        eprintln!("grammar oracle: {} of {runs} runs identical to llama.cpp b11192 ({masks} whole-vocabulary masks, {rejections} rejections, \
                   each by both paths); one mask over {} tokens: one by one median {:.1} ms, p90 {:.1}, max {:.1}; \
                   the trie median {:.2} ms, p90 {:.2}, max {:.2} (the first builds it)", runs - fails, v.len(), a.0, a.1, a.2, b.0, b.1, b.2);
        assert_eq!(fails, 0);
    }

}
