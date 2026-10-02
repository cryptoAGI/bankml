// SPDX-License-Identifier: MIT OR Apache-2.0
//! O6b (JSON schemas) — llama.cpp b11192's `json_schema_to_grammar`, and the wrapping llama-server puts around it on
//! the jinja chat path, ported so that bankML builds, for any schema, the GBNF text llama-server would, byte for byte.
//!
//! Ported from llama.cpp b11192 (MIT, © the ggml authors, github.com/ggml-org/llama.cpp @ 171e8846b):
//! `common/json-schema.cpp` (the schema tree: which keywords decide a node's kind, `$ref` resolution, the errors),
//! `common/json-schema-to-grammar.cpp` (`common_chat_schema_converter`: rule naming and de-duplication, objects with
//! required / optional / additional properties, `_not_strings`, arrays and tuples, integer ranges, the regex → GBNF
//! pattern translation, string formats, the primitive rules), `common/trie.cpp` (the trie `_not_strings` walks),
//! the parts of nlohmann's `ordered_json` the grammar text depends on (object key order and duplicate keys, the
//! integer / float distinction, `dump()`), and from `common/chat-auto-parser-generator.cpp` + `common/peg-parser.cpp`
//! the GBNF the PEG chat parser contributes on the pinned Qwen3 template with `--reasoning off` (the `json-*` rules,
//! the `until-13` rules for the reasoning block, `response-format` and `root`). llama.cpp's notice, kept as its
//! licence asks (LICENSING.md):
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
//! **The oracle** is llama.cpp itself: `testing/schema_oracle.cpp` calls `json_schema_to_grammar` and
//! `common_chat_templates_apply` inside the b11192 release's `libllama-common.so` (no model is loaded) over a corpus
//! (llama.cpp's own `tests/test-json-schema-to-grammar.cpp` cases, Pydantic-shaped schemas like mindX's, edge cases);
//! `oracle_schema_grammars` requires every grammar and every refusal to be the same. The unit tests below carry
//! llama.cpp's test cases with their expected grammars (`testing/json_schema_cases.json`).
//!
//! **Known limit, measured not assumed:** a float in an `enum` / `const` is printed as nlohmann prints it (Grisu2,
//! shortest round-trip, `1e+20` style exponents); Rust's shortest round-trip digits are used, which agree with Grisu2
//! except on rare doubles where Grisu2 is not shortest.

use std::collections::{BTreeMap, HashMap, HashSet};

// ---------------------------------------------------------------- JSON as nlohmann::ordered_json sees it ----------

/// A JSON value with what the grammar text depends on: integers apart from floats (`1` vs `1.0`), unsigned apart
/// from signed, and object keys in first-seen order with the last duplicate's value (`ordered_json`'s parse).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

fn obj_insert(o: &mut Vec<(String, Value)>, k: String, v: Value) {
    match o.iter_mut().find(|(n, _)| *n == k) {
        Some(e) => e.1 = v,
        None => o.push((k, v)),
    }
}

impl Value {
    /// The exact reading (nlohmann's): `None` if the text is not JSON.
    pub fn parse(s: &str) -> Option<Value> {
        let mut p = VP { b: s.as_bytes(), i: 0, depth: 0 };
        let v = p.value()?;
        p.ws();
        (p.i == p.b.len()).then_some(v)
    }
    /// From bankML's request JSON, whose numbers are f64: an integral number within i64 is taken as an integer.
    /// (A schema written `2.0` reads as `2` this way; the servers re-read the body with `parse` to keep it a float.)
    pub fn from_json(j: &crate::serve::Json) -> Value {
        use crate::serve::Json;
        match j {
            Json::Null => Value::Null,
            Json::Bool(b) => Value::Bool(*b),
            Json::Num(n) if n.fract() == 0.0 && n.abs() < 9.2e18 => {
                if *n < 0.0 { Value::Int(*n as i64) } else { Value::UInt(*n as u64) }
            }
            Json::Num(n) => Value::Float(*n),
            Json::Str(s) => Value::Str(s.clone()),
            Json::Arr(a) => Value::Arr(a.iter().map(Value::from_json).collect()),
            Json::Obj(o) => {
                let mut v = Vec::new();
                for (k, x) in o {
                    obj_insert(&mut v, k.clone(), Value::from_json(x));
                }
                Value::Obj(v)
            }
        }
    }
    pub fn get(&self, k: &str) -> Option<&Value> {
        match self {
            Value::Obj(o) => o.iter().find(|(n, _)| n == k).map(|(_, v)| v),
            _ => None,
        }
    }
    fn contains(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
    /// nlohmann's `empty()`: null and empty containers are empty, every other value is not
    pub fn is_empty(&self) -> bool {
        match self {
            Value::Null => true,
            Value::Arr(a) => a.is_empty(),
            Value::Obj(o) => o.is_empty(),
            _ => false,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    fn is_int(&self) -> bool {
        matches!(self, Value::Int(_) | Value::UInt(_))
    }
    /// `get<int64_t>()` on an integer (an unsigned one is cast, as C++ does)
    fn as_i64(&self) -> i64 {
        match self {
            Value::Int(i) => *i,
            Value::UInt(u) => *u as i64,
            Value::Float(f) => *f as i64,
            _ => 0,
        }
    }
    /// `dump()`: compact, keys in order, non-ASCII as it is, floats as nlohmann prints them
    pub fn dump(&self) -> String {
        let mut s = String::new();
        self.dump_to(&mut s);
        s
    }
    fn dump_to(&self, s: &mut String) {
        match self {
            Value::Null => s.push_str("null"),
            Value::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
            Value::Int(i) => s.push_str(&i.to_string()),
            Value::UInt(u) => s.push_str(&u.to_string()),
            Value::Float(f) => s.push_str(&dump_float(*f)),
            Value::Str(t) => dump_str(t, s),
            Value::Arr(a) => {
                s.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    v.dump_to(s);
                }
                s.push(']');
            }
            Value::Obj(o) => {
                s.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    dump_str(k, s);
                    s.push(':');
                    v.dump_to(s);
                }
                s.push('}');
            }
        }
    }
}

fn dump_str(t: &str, s: &mut String) {
    s.push('"');
    for c in t.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\u{8}' => s.push_str("\\b"),
            '\u{c}' => s.push_str("\\f"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => s.push_str(&format!("\\u{:04x}", c as u32)),
            c => s.push(c),
        }
    }
    s.push('"');
}

/// nlohmann's `dtoa_impl::format_buffer` (min_exp −4, max_exp 15) over the shortest round-trip digits.
fn dump_float(f: f64) -> String {
    if !f.is_finite() {
        return "null".into();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let e = format!("{:e}", f.abs());
    let (m, x) = e.split_once('e').unwrap();
    let digits: String = m.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = x.parse::<i32>().unwrap() + 1;
    let mut s = String::from(if f < 0.0 { "-" } else { "" });
    if k <= n && n <= 15 {
        s += &digits;
        s += &"0".repeat((n - k) as usize);
        s += ".0";
    } else if 0 < n && n <= 15 {
        s += &digits[..n as usize];
        s.push('.');
        s += &digits[n as usize..];
    } else if -4 < n && n <= 0 {
        s += "0.";
        s += &"0".repeat((-n) as usize);
        s += &digits;
    } else {
        s += &digits[..1];
        if k > 1 {
            s.push('.');
            s += &digits[1..];
        }
        let ex = n - 1;
        s.push('e');
        s.push(if ex < 0 { '-' } else { '+' });
        let a = ex.unsigned_abs();
        if a < 10 {
            s.push('0');
        }
        s += &a.to_string();
    }
    s
}

struct VP<'a> {
    b: &'a [u8],
    i: usize,
    depth: u32,
}

impl VP<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn eat(&mut self, lit: &str) -> bool {
        let ok = self.b[self.i..].starts_with(lit.as_bytes());
        if ok {
            self.i += lit.len();
        }
        ok
    }
    fn value(&mut self) -> Option<Value> {
        self.ws();
        self.depth += 1;
        if self.depth > 512 {
            return None;
        }
        let v = match *self.b.get(self.i)? {
            b'n' => self.eat("null").then_some(Value::Null),
            b't' => self.eat("true").then_some(Value::Bool(true)),
            b'f' => self.eat("false").then_some(Value::Bool(false)),
            b'"' => self.string().map(Value::Str),
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.eat("]") {
                    Some(Value::Arr(v))
                } else {
                    loop {
                        v.push(self.value()?);
                        self.ws();
                        if self.eat("]") {
                            break Some(Value::Arr(v));
                        }
                        if !self.eat(",") {
                            break None;
                        }
                    }
                }
            }
            b'{' => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.eat("}") {
                    Some(Value::Obj(v))
                } else {
                    loop {
                        self.ws();
                        let k = self.string()?;
                        self.ws();
                        if !self.eat(":") {
                            break None;
                        }
                        let x = self.value()?;
                        obj_insert(&mut v, k, x);
                        self.ws();
                        if self.eat("}") {
                            break Some(Value::Obj(v));
                        }
                        if !self.eat(",") {
                            break None;
                        }
                    }
                }
            }
            _ => self.number(),
        };
        self.depth -= 1;
        v
    }
    /// nlohmann's lexer: `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`; an integer is i64 when negative and u64
    /// otherwise, and a float when it does not fit
    fn number(&mut self) -> Option<Value> {
        let s = self.i;
        let d = |p: &VP, i: usize| p.b.get(i).is_some_and(u8::is_ascii_digit);
        let mut i = self.i;
        if self.b.get(i) == Some(&b'-') {
            i += 1;
        }
        match self.b.get(i)? {
            b'0' => i += 1,
            b'1'..=b'9' => {
                while d(self, i) {
                    i += 1;
                }
            }
            _ => return None,
        }
        let mut float = false;
        if self.b.get(i) == Some(&b'.') {
            float = true;
            i += 1;
            if !d(self, i) {
                return None;
            }
            while d(self, i) {
                i += 1;
            }
        }
        if matches!(self.b.get(i), Some(b'e' | b'E')) {
            float = true;
            i += 1;
            if matches!(self.b.get(i), Some(b'+' | b'-')) {
                i += 1;
            }
            if !d(self, i) {
                return None;
            }
            while d(self, i) {
                i += 1;
            }
        }
        self.i = i;
        let t = std::str::from_utf8(&self.b[s..i]).ok()?;
        if !float {
            if t.starts_with('-') {
                if let Ok(v) = t.parse::<i64>() {
                    return Some(Value::Int(v));
                }
            } else if let Ok(v) = t.parse::<u64>() {
                return Some(Value::UInt(v));
            }
        }
        t.parse::<f64>().ok().map(Value::Float)
    }
    fn hex4(&mut self) -> Option<u32> {
        let h = std::str::from_utf8(self.b.get(self.i..self.i + 4)?).ok()?;
        self.i += 4;
        u32::from_str_radix(h, 16).ok()
    }
    fn string(&mut self) -> Option<String> {
        if !self.eat("\"") {
            return None;
        }
        let mut out = Vec::new();
        loop {
            let c = *self.b.get(self.i)?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).ok(),
                0..=0x1f => return None,
                b'\\' => {
                    let e = *self.b.get(self.i)?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xd800..0xdc00).contains(&hi) {
                                if !self.eat("\\u") {
                                    return None;
                                }
                                let lo = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&lo) {
                                    return None;
                                }
                                0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
                            } else {
                                hi
                            };
                            char::from_u32(cp)?
                        }
                        _ => return None,
                    };
                    let mut buf = [0; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                _ => out.push(c),
            }
        }
    }
}

// ---------------------------------------------------------------- the schema tree (json-schema.cpp) ----------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Fmt {
    None,
    Uuid,
    Date,
    Time,
    DateTime,
}

#[derive(Debug)]
enum Node {
    Any,
    Ref(String),
    AnyOf(Vec<Node>),
    AllOf(Vec<Node>),
    Const(Value),
    Enum(Vec<Value>),
    Null,
    Boolean,
    Number,
    Integer { min: i64, max: i64 },
    Str { pattern: String, format: Fmt, min_len: i32, max_len: i32 },
    Array { items: Box<Node>, min: i32, max: i32 },
    Tuple(Vec<Node>),
    Object { props: Vec<(String, Node, bool)>, additional: Option<Box<Node>> },
}

struct Doc {
    root: Node,
    refs: BTreeMap<String, Node>,
}

type SR<T> = Result<T, String>;

fn fail<T>(path: &str, msg: &str) -> SR<T> {
    Err(format!("JSON schema error at {path}: {msg}"))
}

/// `std::stoull` / `std::stoi` as the C++ reads a number out of a string: leading space, a sign, digits, and the
/// rest ignored; `None` where it throws (no digits, out of range)
fn sto(s: &str, lo: i128, hi: i128, unsigned: bool) -> Option<i128> {
    let b = s.trim_start_matches([' ', '\t', '\n', '\x0b', '\x0c', '\r']).as_bytes();
    let (neg, b) = match b.first() {
        Some(b'-') => (true, &b[1..]),
        Some(b'+') => (false, &b[1..]),
        _ => (false, b),
    };
    let n = b.iter().take_while(|c| c.is_ascii_digit()).count();
    if n == 0 {
        return None;
    }
    let mut v: i128 = 0;
    for c in &b[..n] {
        v = v.checked_mul(10)?.checked_add((c - b'0') as i128)?;
        if v > u64::MAX as i128 + 1 {
            return None;
        }
    }
    if unsigned {
        // strtoull negates in the unsigned type: "-1" is 2^64 − 1
        if v > u64::MAX as i128 {
            return None;
        }
        return Some(if neg { ((v as u64).wrapping_neg()) as i128 } else { v });
    }
    let v = if neg { -v } else { v };
    (lo..=hi).contains(&v).then_some(v)
}

struct Builder<'a> {
    root: &'a Value,
    refs: BTreeMap<String, Option<Node>>,
}

impl<'a> Builder<'a> {
    fn get_count(s: &Value, key: &str, path: &str, def: i32) -> SR<i32> {
        let Some(v) = s.get(key) else { return Ok(def) };
        let n = v.as_i64() as i32;
        if !v.is_int() || n < 0 {
            return fail(path, &format!("{key} must be a non-negative integer"));
        }
        Ok(n)
    }
    fn get_bound(s: &Value, key: &str, path: &str, round_up: bool) -> SR<i64> {
        let v = s.get(key).unwrap();
        match v {
            Value::Int(_) | Value::UInt(_) => Ok(v.as_i64()),
            Value::Float(d) => {
                let r = if round_up { d.ceil() } else { d.floor() };
                // (int64_t) of a double out of range is x86's "integer indefinite", INT64_MIN
                Ok(if r.is_nan() || r >= 9_223_372_036_854_775_808.0 || r < -9_223_372_036_854_775_808.0 { i64::MIN } else { r as i64 })
            }
            _ => fail(path, &format!("{key} must be a number")),
        }
    }
    fn get_format(s: &Value, path: &str) -> SR<Fmt> {
        let Some(v) = s.get("format") else { return Ok(Fmt::None) };
        let Some(f) = v.as_str() else { return fail(path, "format must be a string") };
        Ok(match f {
            "date" => Fmt::Date,
            "time" => Fmt::Time,
            "date-time" => Fmt::DateTime,
            "uuid" => Fmt::Uuid,
            f if f.len() == 5 && f.starts_with("uuid") && (b'1'..=b'5').contains(&f.as_bytes()[4]) => Fmt::Uuid,
            _ => Fmt::None,
        })
    }
    fn resolve_ref(&self, r: &str, path: &str) -> SR<&'a Value> {
        let mut t: &'a Value = self.root;
        let toks: Vec<&str> = r[1..].split('/').collect();
        for sel in toks.iter().skip(1) {
            match t {
                Value::Obj(_) if t.contains(sel) => t = t.get(sel).unwrap(),
                Value::Arr(a) => {
                    let idx = sto(sel, 0, u64::MAX as i128, true).map(|v| v as u64).unwrap_or(a.len() as u64);
                    if idx >= a.len() as u64 {
                        return fail(path, &format!("cannot resolve $ref {r}, {sel} is out of range"));
                    }
                    t = &a[idx as usize];
                }
                _ => return fail(path, &format!("cannot resolve $ref {r}, {sel} not found")),
            }
        }
        Ok(t)
    }
    fn build_ref(&mut self, v: &Value, path: &str) -> SR<Node> {
        let Some(r) = v.as_str() else { return fail(path, "$ref must be a string") };
        if !r.starts_with("#/") {
            return fail(path, &format!("unsupported $ref {r}, only references into the same document are supported"));
        }
        if !self.refs.contains_key(r) {
            self.refs.insert(r.to_string(), None);
            let target = self.resolve_ref(r, path)?;
            let n = self.build_node(target, r)?;
            self.refs.insert(r.to_string(), Some(n));
        }
        Ok(Node::Ref(r.to_string()))
    }
    fn alternatives(&mut self, alts: &Value, path: &str, all: bool) -> SR<Node> {
        let Value::Arr(a) = alts else { return fail(path, "must be an array of schemas") };
        if a.is_empty() {
            return fail(path, "must not be empty");
        }
        let mut c = Vec::new();
        for (i, alt) in a.iter().enumerate() {
            c.push(self.build_node(alt, &format!("{path}/{i}"))?);
        }
        Ok(if all { Node::AllOf(c) } else { Node::AnyOf(c) })
    }
    fn build_object(&mut self, s: &Value, path: &str) -> SR<Node> {
        let mut required = HashSet::new();
        if let Some(Value::Arr(r)) = s.get("required") {
            for n in r {
                if let Value::Str(n) = n {
                    required.insert(n.clone());
                }
            }
        }
        let mut props = Vec::new();
        if let Some(p) = s.get("properties") {
            let Value::Obj(p) = p else { return fail(path, "properties must be an object") };
            for (name, prop) in p {
                let n = self.build_node(prop, &format!("{path}/properties/{name}"))?;
                props.push((name.clone(), n, required.contains(name)));
            }
        }
        let additional = match s.get("additionalProperties") {
            Some(Value::Bool(true)) => Some(Box::new(Node::Any)),
            Some(Value::Bool(false)) => None,
            Some(a @ Value::Obj(_)) => Some(Box::new(self.build_node(a, &format!("{path}/additionalProperties"))?)),
            Some(_) => return fail(path, "additionalProperties must be a boolean or a schema"),
            None if !s.contains("properties") => Some(Box::new(Node::Any)),
            None => None,
        };
        Ok(Node::Object { props, additional })
    }
    fn build_array(&mut self, s: &Value, path: &str) -> SR<Node> {
        let items = if s.contains("items") || s.contains("prefixItems") {
            let key = if s.contains("items") { "items" } else { "prefixItems" };
            let it = s.get(key).unwrap();
            if let Value::Arr(a) = it {
                let mut t = Vec::new();
                for (i, item) in a.iter().enumerate() {
                    t.push(self.build_node(item, &format!("{path}/{key}/{i}"))?);
                }
                return Ok(Node::Tuple(t));
            }
            self.build_node(it, &format!("{path}/{key}"))?
        } else {
            Node::Any
        };
        let min = Self::get_count(s, "minItems", path, 0)?;
        let max = Self::get_count(s, "maxItems", path, -1)?;
        Ok(Node::Array { items: Box::new(items), min, max })
    }
    fn build_string(s: &Value, path: &str) -> SR<Node> {
        let mut pattern = String::new();
        if let Some(p) = s.get("pattern") {
            let Some(p) = p.as_str() else { return fail(path, "pattern must be a string") };
            pattern = p.to_string();
        }
        let format = Self::get_format(s, path)?;
        let min_len = Self::get_count(s, "minLength", path, 0)?;
        let max_len = Self::get_count(s, "maxLength", path, -1)?;
        Ok(Node::Str { pattern, format, min_len, max_len })
    }
    fn build_integer(s: &Value, path: &str) -> SR<Node> {
        let (mut min, mut max) = (i64::MIN, i64::MAX);
        if s.contains("minimum") {
            min = Self::get_bound(s, "minimum", path, true)?;
        } else if s.contains("exclusiveMinimum") {
            min = Self::get_bound(s, "exclusiveMinimum", path, false)?.wrapping_add(1);
        }
        if s.contains("maximum") {
            max = Self::get_bound(s, "maximum", path, false)?;
        } else if s.contains("exclusiveMaximum") {
            max = Self::get_bound(s, "exclusiveMaximum", path, true)?.wrapping_sub(1);
        }
        Ok(Node::Integer { min, max })
    }
    fn build_node(&mut self, s: &Value, path: &str) -> SR<Node> {
        if !matches!(s, Value::Obj(_)) {
            return fail(path, "schema must be an object");
        }
        if let Some(r) = s.get("$ref") {
            return self.build_ref(r, path);
        }
        if s.contains("oneOf") || s.contains("anyOf") {
            let key = if s.contains("oneOf") { "oneOf" } else { "anyOf" };
            return self.alternatives(s.get(key).unwrap(), &format!("{path}/{key}"), false);
        }
        let ty = s.get("type").cloned().unwrap_or(Value::Null);
        if let Value::Arr(ts) = &ty {
            if ts.is_empty() {
                return fail(path, "type must not be empty");
            }
            let mut c = Vec::new();
            for (i, t) in ts.iter().enumerate() {
                let mut alt = s.clone();
                if let Value::Obj(o) = &mut alt {
                    obj_insert(o, "type".into(), t.clone());
                }
                c.push(self.build_node(&alt, &format!("{path}/type/{i}"))?);
            }
            return Ok(Node::AnyOf(c));
        }
        if let Some(c) = s.get("const") {
            return Ok(Node::Const(c.clone()));
        }
        if let Some(e) = s.get("enum") {
            return match e {
                Value::Arr(a) if !a.is_empty() => Ok(Node::Enum(a.clone())),
                _ => fail(path, "enum must be a non-empty array"),
            };
        }
        if !matches!(ty, Value::Null | Value::Str(_)) {
            return fail(path, "type must be a string or an array of strings");
        }
        let tn = ty.as_str().unwrap_or("");
        let has_properties = s.contains("properties") || s.get("additionalProperties").is_some_and(|a| *a != Value::Bool(true));
        match tn {
            "" => {
                if has_properties {
                    return self.build_object(s, path);
                }
                if let Some(a) = s.get("allOf") {
                    return self.alternatives(a, &format!("{path}/allOf"), true);
                }
                if s.contains("items") || s.contains("prefixItems") {
                    return self.build_array(s, path);
                }
                if s.contains("pattern") || s.contains("minLength") || s.contains("maxLength") || Self::get_format(s, path)? != Fmt::None {
                    return Self::build_string(s, path);
                }
                Ok(Node::Any)
            }
            "object" => match s.get("allOf") {
                Some(a) if !has_properties => self.alternatives(a, &format!("{path}/allOf"), true),
                _ => self.build_object(s, path),
            },
            "string" => match s.get("allOf") {
                Some(a) => self.alternatives(a, &format!("{path}/allOf"), true),
                None => Self::build_string(s, path),
            },
            "array" => self.build_array(s, path),
            "integer" => Self::build_integer(s, path),
            "number" => Ok(Node::Number),
            "boolean" => Ok(Node::Boolean),
            "null" => Ok(Node::Null),
            t => fail(path, &format!("unrecognized type {t}")),
        }
    }
}

fn schema_doc(schema: &Value) -> SR<Doc> {
    let mut b = Builder { root: schema, refs: BTreeMap::new() };
    let root = b.build_node(schema, "#")?;
    Ok(Doc { root, refs: b.refs.into_iter().map(|(k, v)| (k, v.expect("every $ref is built"))).collect() })
}

// ---------------------------------------------------------------- the converter (json-schema-to-grammar.cpp) ----------

const SPACE_RULE: &str = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

/// (name, content, deps): PRIMITIVE_RULES, then STRING_FORMAT_RULES
const PRIMITIVES: &[(&str, &str, &[&str])] = &[
    ("boolean", "(\"true\" | \"false\")", &[]),
    ("decimal-part", "[0-9]{1,16}", &[]),
    ("integral-part", "[0] | [1-9] [0-9]{0,15}", &[]),
    ("number", "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", &["integral-part", "decimal-part"]),
    ("integer", "(\"-\"? integral-part)", &["integral-part"]),
    ("value", "object | array | string | number | boolean | null", &["object", "array", "string", "number", "boolean", "null"]),
    ("object", "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", &["string", "value"]),
    ("array", "\"[\" space ( value (\",\" space value)* )? space \"]\"", &["value"]),
    ("uuid", "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"", &[]),
    ("char", "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", &[]),
    ("string", "\"\\\"\" char* \"\\\"\"", &["char"]),
    ("null", "\"null\"", &[]),
    ("date", "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", &[]),
    ("time", "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", &[]),
    ("date-time", "date \"T\" time", &["date", "time"]),
    ("date-string", "\"\\\"\" date \"\\\"\"", &["date"]),
    ("time-string", "\"\\\"\" time \"\\\"\"", &["time"]),
    ("date-time-string", "\"\\\"\" date-time \"\\\"\"", &["date-time"]),
];

fn prim(name: &str) -> (&'static str, &'static [&'static str]) {
    let p = PRIMITIVES.iter().find(|p| p.0 == name).expect("a known primitive");
    (p.1, p.2)
}

fn is_reserved(name: &str) -> bool {
    name == "root" || PRIMITIVES.iter().any(|p| p.0 == name)
}

/// `regex_replace(s, "[^a-zA-Z0-9-]+", "-")`, over bytes as std::regex reads them
fn esc_name(s: &str) -> String {
    let mut o = String::new();
    let mut run = false;
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b == b'-' {
            o.push(b as char);
            run = false;
        } else if !run {
            o.push('-');
            run = true;
        }
    }
    o
}

/// `format_literal`: `\r \n " \` escaped, in quotes (llama.cpp's `gbnf_format_literal`)
pub fn format_literal(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '\r' => o.push_str("\\r"),
            '\n' => o.push_str("\\n"),
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn build_repetition(item: &str, min: i32, max: i32, sep: &str) -> String {
    let has_max = max != i32::MAX;
    if max == 0 {
        return String::new();
    }
    if min == 0 && max == 1 {
        return format!("{item}?");
    }
    if sep.is_empty() {
        if min == 1 && !has_max {
            return format!("{item}+");
        }
        if min == 0 && !has_max {
            return format!("{item}*");
        }
        return format!("{item}{{{min},{}}}", if has_max { max.to_string() } else { String::new() });
    }
    let r = format!("{item} {}", build_repetition(&format!("({sep} {item})"), if min == 0 { 0 } else { min - 1 }, if has_max { max - 1 } else { max }, ""));
    if min == 0 { format!("({r})?") } else { r }
}

fn digit_range(o: &mut String, from: u8, to: u8) {
    o.push('[');
    o.push(from as char);
    if from != to {
        o.push('-');
        o.push(to as char);
    }
    o.push(']');
}

fn more_digits(o: &mut String, min: i32, max: i32) {
    o.push_str("[0-9]");
    if min == max && min == 1 {
        return;
    }
    o.push('{');
    o.push_str(&min.to_string());
    if max != min {
        o.push(',');
        if max != i32::MAX {
            o.push_str(&max.to_string());
        }
    }
    o.push('}');
}

fn uniform_range(o: &mut String, from: &[u8], to: &[u8]) {
    let mut i = 0;
    while i < from.len() && i < to.len() && from[i] == to[i] {
        i += 1;
    }
    if i > 0 {
        o.push('"');
        o.push_str(std::str::from_utf8(&from[..i]).unwrap());
        o.push('"');
    }
    if i < from.len() && i < to.len() {
        if i > 0 {
            o.push(' ');
        }
        let sub_len = from.len() - i - 1;
        if sub_len > 0 {
            let (from_sub, to_sub) = (&from[i + 1..], &to[i + 1..]);
            let (zeros, nines) = ("0".repeat(sub_len), "9".repeat(sub_len));
            let mut to_reached = false;
            o.push('(');
            if from_sub == zeros.as_bytes() {
                digit_range(o, from[i], to[i] - 1);
                o.push(' ');
                more_digits(o, sub_len as i32, sub_len as i32);
            } else {
                o.push('[');
                o.push(from[i] as char);
                o.push_str("] (");
                uniform_range(o, from_sub, nines.as_bytes());
                o.push(')');
                if from[i] < to[i] - 1 {
                    o.push_str(" | ");
                    if to_sub == nines.as_bytes() {
                        digit_range(o, from[i] + 1, to[i]);
                        to_reached = true;
                    } else {
                        digit_range(o, from[i] + 1, to[i] - 1);
                    }
                    o.push(' ');
                    more_digits(o, sub_len as i32, sub_len as i32);
                }
            }
            if !to_reached {
                o.push_str(" | ");
                digit_range(o, to[i], to[i]);
                o.push(' ');
                uniform_range(o, zeros.as_bytes(), to_sub);
            }
            o.push(')');
        } else {
            o.push('[');
            o.push(from[i] as char);
            o.push('-');
            o.push(to[i] as char);
            o.push(']');
        }
    }
}

fn build_min_max_int(min: i64, max: i64, o: &mut String, decimals_left: i32, top_level: bool) {
    let (has_min, has_max) = (min != i64::MIN, max != i64::MAX);
    if has_min && has_max {
        if min < 0 && max < 0 {
            o.push_str("\"-\" (");
            build_min_max_int(max.wrapping_neg(), min.wrapping_neg(), o, decimals_left, true);
            o.push(')');
            return;
        }
        let mut min = min;
        if min < 0 {
            o.push_str("\"-\" (");
            build_min_max_int(0, min.wrapping_neg(), o, decimals_left, true);
            o.push_str(") | ");
            min = 0;
        }
        let mut min_s = min.to_string();
        let max_s = max.to_string();
        for digits in min_s.len()..max_s.len() {
            uniform_range(o, min_s.as_bytes(), "9".repeat(digits).as_bytes());
            min_s = format!("1{}", "0".repeat(digits));
            o.push_str(" | ");
        }
        uniform_range(o, min_s.as_bytes(), max_s.as_bytes());
        return;
    }
    let less_decimals = (decimals_left - 1).max(1);
    if has_min {
        if min < 0 {
            o.push_str("\"-\" (");
            build_min_max_int(i64::MIN, min.wrapping_neg(), o, decimals_left, false);
            o.push_str(") | [0] | [1-9] ");
            more_digits(o, 0, decimals_left - 1);
        } else if min == 0 {
            if top_level {
                o.push_str("[0] | [1-9] ");
                more_digits(o, 0, less_decimals);
            } else {
                more_digits(o, 1, decimals_left);
            }
        } else if min <= 9 {
            let c = b'0' + min as u8;
            let start = if top_level { b'1' } else { b'0' };
            if c > start {
                digit_range(o, start, c - 1);
                o.push(' ');
                more_digits(o, 1, less_decimals);
                o.push_str(" | ");
            }
            digit_range(o, c, b'9');
            o.push(' ');
            more_digits(o, 0, less_decimals);
        } else {
            let min_s = min.to_string();
            let len = min_s.len() as i32;
            let c = min_s.as_bytes()[0];
            if c > b'1' {
                digit_range(o, if top_level { b'1' } else { b'0' }, c - 1);
                o.push(' ');
                more_digits(o, len, less_decimals);
                o.push_str(" | ");
            }
            digit_range(o, c, c);
            o.push_str(" (");
            build_min_max_int(min_s[1..].parse().unwrap(), i64::MAX, o, less_decimals, false);
            o.push(')');
            if c < b'9' {
                o.push_str(" | ");
                digit_range(o, c + 1, b'9');
                o.push(' ');
                more_digits(o, len - 1, less_decimals);
            }
        }
        return;
    }
    if has_max {
        if max >= 0 {
            if top_level {
                o.push_str("\"-\" [1-9] ");
                more_digits(o, 0, less_decimals);
                o.push_str(" | ");
            }
            build_min_max_int(0, max, o, decimals_left, true);
        } else {
            o.push_str("\"-\" (");
            build_min_max_int(max.wrapping_neg(), i64::MAX, o, decimals_left, false);
            o.push(')');
        }
    }
}

/// `gbnf_escape_length`: the length of an escape GBNF reads the same way as the regex (`\x..`, `\u....`,
/// `\U........`, `\t \r \n \\ \" \[ \] \-`), 0 for any other
fn gbnf_escape_length(p: &[u8], pos: usize) -> usize {
    if pos + 1 >= p.len() || p[pos] != b'\\' {
        return 0;
    }
    let n_hex = match p[pos + 1] {
        b'x' => 2,
        b'u' => 4,
        b'U' => 8,
        b't' | b'r' | b'n' | b'\\' | b'"' | b'[' | b']' | b'-' => return 2,
        _ => return 0,
    };
    if pos + 2 + n_hex > p.len() || !p[pos + 2..pos + 2 + n_hex].iter().all(u8::is_ascii_hexdigit) {
        return 0;
    }
    2 + n_hex
}

enum PatErr {
    Unsupported(String),
    Invalid(String),
    /// the translation splits a multi-byte character (a non-ASCII character before a quantifier): llama.cpp writes
    /// the grammar byte by byte and it is not UTF-8; bankML refuses it rather than reproduce a grammar that misreads
    Split,
}

const NON_LITERAL: &[u8] = b"|.()[]{}*+?^$";
const ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS: &[u8] = b"^$.[]()|{}*+?";
const MAX_PATTERN_DEPTH: i32 = 100;

struct Pat<'a> {
    p: &'a [u8],
    i: usize,
    depth: i32,
    name: String,
    sub_ids: HashMap<Vec<u8>, String>,
}

fn lossy(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// A piece of a translated pattern: its text, and whether it is a literal (to be quoted, and merged with its
/// literal neighbours).
type Lit = (Vec<u8>, bool);

fn to_rule(l: &Lit) -> Vec<u8> {
    if l.1 { [&b"\""[..], &l.0, b"\""].concat() } else { l.0.clone() }
}

fn utf8(b: Vec<u8>) -> Result<String, PatErr> {
    String::from_utf8(b).map_err(|_| PatErr::Split)
}

/// The converter: rules by name (a `std::map`, so the grammar is written in byte order of the names).
struct Conv {
    rules: BTreeMap<String, String>,
    resolving: HashSet<String>,
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl Conv {
    fn new() -> Conv {
        let mut rules = BTreeMap::new();
        rules.insert("space".to_string(), SPACE_RULE.to_string());
        Conv { rules, resolving: HashSet::new(), errors: Vec::new(), warnings: Vec::new() }
    }

    fn add_rule(&mut self, name: &str, rule: &str) -> String {
        let esc = esc_name(name);
        if self.rules.get(&esc).is_none_or(|r| r == rule) {
            self.rules.insert(esc.clone(), rule.to_string());
            return esc;
        }
        let mut i = 0;
        while self.rules.get(&format!("{esc}{i}")).is_some_and(|r| r != rule) {
            i += 1;
        }
        let key = format!("{esc}{i}");
        self.rules.insert(key.clone(), rule.to_string());
        key
    }

    fn add_primitive(&mut self, name: &str, prim_name: &str) -> String {
        let (content, deps) = prim(prim_name);
        let n = self.add_rule(name, content);
        for dep in deps {
            if !self.rules.contains_key(*dep) {
                self.add_primitive(dep, dep);
            }
        }
        n
    }

    fn visit_primitive(&mut self, rule_name: &str, ty: &str) -> String {
        self.add_primitive(if rule_name == "root" { "root" } else { ty }, ty)
    }

    fn union(&mut self, doc: &Doc, name: &str, alts: &[Node]) -> String {
        let mut rules = Vec::new();
        for (i, a) in alts.iter().enumerate() {
            rules.push(self.visit(doc, a, &format!("{name}{}{i}", if name.is_empty() { "alternative-" } else { "-" })));
        }
        rules.join(" | ")
    }

    fn visit_pattern(&mut self, pattern: &str, name: &str) -> String {
        let snapshot = self.rules.clone();
        match self.pattern_to_rule(pattern, name) {
            Ok(r) => r,
            Err(PatErr::Unsupported(e)) => {
                self.rules = snapshot;
                self.warnings.push(format!("pattern {pattern} is not supported ({e}), accepting any string"));
                let s = self.add_primitive("string", "string");
                self.add_rule(name, &s)
            }
            Err(PatErr::Invalid(e)) => {
                self.rules = snapshot;
                self.errors.push(format!("Invalid pattern {pattern}: {e}"));
                String::new()
            }
            Err(PatErr::Split) => {
                self.rules = snapshot;
                self.errors.push(format!("pattern {pattern}: llama.cpp b11192 translates it into a grammar that is not UTF-8 (a \
                    multi-byte character before a quantifier is split byte by byte); bankML refuses it — put the character in a \
                    group, e.g. (é)+"));
                String::new()
            }
        }
    }

    fn pattern_to_rule(&mut self, pattern: &str, name: &str) -> Result<String, PatErr> {
        let b = pattern.as_bytes();
        if b.len() < 2 || b[0] != b'^' || b[b.len() - 1] != b'$' {
            return Err(PatErr::Unsupported("not anchored with '^' and '$'".into()));
        }
        let mut st = Pat { p: &b[1..b.len() - 1], i: 0, depth: 0, name: name.to_string(), sub_ids: HashMap::new() };
        let rule = to_rule(&self.transform(&mut st)?);
        if st.depth != 0 {
            return Err(PatErr::Invalid("unbalanced parentheses".into()));
        }
        let rule = utf8(rule)?;
        Ok(self.add_rule(name, &format!("\"\\\"\" ({rule}) \"\\\"\"")))
    }

    fn transform(&mut self, st: &mut Pat) -> Result<Lit, PatErr> {
        let len = st.p.len();
        let mut seq: Vec<Lit> = Vec::new();
        let join = |seq: &[Lit]| -> Lit {
            let mut ret: Vec<Lit> = Vec::new();
            let mut lit = Vec::new();
            for item in seq {
                if item.1 {
                    lit.extend_from_slice(&item.0);
                } else {
                    if !lit.is_empty() {
                        ret.push((std::mem::take(&mut lit), true));
                    }
                    ret.push(item.clone());
                }
            }
            if !lit.is_empty() {
                ret.push((lit, true));
            }
            (ret.iter().map(to_rule).collect::<Vec<_>>().join(&b' '), false)
        };
        let nonlit = |c: u8| NON_LITERAL.contains(&c);
        while st.i < len {
            let c = st.p[st.i];
            if c == b'.' {
                seq.push((self.add_rule("dot", "[^\\x0A\\x0D]").into_bytes(), false));
                st.i += 1;
            } else if c == b'(' {
                st.i += 1;
                if st.i < len && st.p[st.i] == b'?' {
                    if st.i + 1 < len && st.p[st.i + 1] == b':' {
                        st.i += 2;
                    } else {
                        return Err(PatErr::Unsupported("unsupported group syntax".into()));
                    }
                }
                st.depth += 1;
                if st.depth > MAX_PATTERN_DEPTH {
                    return Err(PatErr::Unsupported("pattern nesting too deep".into()));
                }
                let inner = self.transform(st)?;
                seq.push(([&b"("[..], &to_rule(&inner), b")"].concat(), false));
            } else if c == b')' {
                st.i += 1;
                if st.depth == 0 {
                    return Err(PatErr::Invalid("unbalanced parentheses".into()));
                }
                st.depth -= 1;
                return Ok(join(&seq));
            } else if c == b'^' || c == b'$' {
                return Err(PatErr::Unsupported("anchor inside the pattern".into()));
            } else if c == b'[' {
                let mut sq = vec![b'['];
                st.i += 1;
                while st.i < len && st.p[st.i] != b']' {
                    if st.p[st.i] == b'\\' {
                        let el = gbnf_escape_length(st.p, st.i);
                        if el == 0 {
                            return Err(PatErr::Unsupported(format!("unsupported escape in character class: {}", lossy(&st.p[st.i..(st.i + 2).min(len)]))));
                        }
                        sq.extend_from_slice(&st.p[st.i..st.i + el]);
                        st.i += el;
                    } else {
                        sq.push(st.p[st.i]);
                        st.i += 1;
                    }
                }
                if st.i >= len {
                    return Err(PatErr::Invalid("unterminated character class".into()));
                }
                sq.push(b']');
                st.i += 1;
                seq.push((sq, false));
            } else if c == b'|' {
                seq.push((b"|".to_vec(), false));
                st.i += 1;
            } else if c == b'*' || c == b'+' || c == b'?' {
                let Some(last) = seq.last_mut() else { return Err(PatErr::Invalid("nothing to repeat".into())) };
                let mut r = to_rule(last);
                r.push(c);
                *last = (r, false);
                st.i += 1;
            } else if c == b'{' {
                let mut cb = vec![b'{'];
                st.i += 1;
                while st.i < len && st.p[st.i] != b'}' {
                    cb.push(st.p[st.i]);
                    st.i += 1;
                }
                if st.i >= len {
                    return Err(PatErr::Unsupported("unterminated curly brackets".into()));
                }
                cb.push(b'}');
                st.i += 1;
                let inner = lossy(&cb[1..cb.len() - 1]);
                let nums: Vec<&str> = inner.split(',').collect();
                if nums.len() != 1 && nums.len() != 2 {
                    return Err(PatErr::Unsupported("wrong number of values in curly brackets".into()));
                }
                let stoi = |s: &str| sto(s, i32::MIN as i128, i32::MAX as i128, false).map(|v| v as i32).ok_or_else(|| PatErr::Unsupported("invalid number in curly brackets".into()));
                let (mut mn, mut mx) = (0, i32::MAX);
                if nums.len() == 1 {
                    mn = stoi(nums[0])?;
                    mx = mn;
                } else {
                    if !nums[0].is_empty() {
                        mn = stoi(nums[0])?;
                    }
                    if !nums[1].is_empty() {
                        mx = stoi(nums[1])?;
                    }
                }
                if seq.is_empty() {
                    return Err(PatErr::Invalid("nothing to repeat".into()));
                }
                let (mut sub, is_lit) = seq.pop().unwrap();
                if !is_lit {
                    let n = st.sub_ids.len();
                    sub = match st.sub_ids.get(&sub) {
                        Some(id) if !id.is_empty() => id.clone().into_bytes(),
                        _ => {
                            let id = self.add_rule(&format!("{}-{}", st.name, n + 1), &utf8(sub.clone())?);
                            st.sub_ids.insert(sub.clone(), id.clone());
                            id.into_bytes()
                        }
                    };
                }
                let item = utf8(if is_lit { [&b"\""[..], &sub, b"\""].concat() } else { sub })?;
                seq.push((build_repetition(&item, mn, mx, "").into_bytes(), false));
            } else {
                let mut lit: Vec<u8> = Vec::new();
                while st.i < len {
                    let p = st.p;
                    if p[st.i] == b'\\' {
                        if st.i == len - 1 {
                            return Err(PatErr::Invalid("trailing backslash".into()));
                        }
                        let next = p[st.i + 1];
                        if ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.contains(&next) {
                            st.i += 1;
                            lit.push(p[st.i]);
                            st.i += 1;
                        } else {
                            let el = gbnf_escape_length(p, st.i);
                            if el == 0 {
                                return Err(PatErr::Unsupported(format!("unsupported escape: {}", lossy(&p[st.i..st.i + 2]))));
                            }
                            lit.extend_from_slice(&p[st.i..st.i + el]);
                            st.i += el;
                        }
                    } else if p[st.i] == b'"' {
                        lit.extend_from_slice(b"\\\"");
                        st.i += 1;
                    } else if !nonlit(p[st.i]) && (st.i == len - 1 || lit.is_empty() || p[st.i + 1] == b'.' || !nonlit(p[st.i + 1])) {
                        lit.push(p[st.i]);
                        st.i += 1;
                    } else {
                        break;
                    }
                }
                if lit.is_empty() {
                    return Err(PatErr::Unsupported(format!("unsupported character: {}", c as char)));
                }
                seq.push((lit, true));
            }
        }
        Ok(join(&seq))
    }

    /// A JSON string that is none of `strings` (a trie over their code points)
    fn not_strings(&mut self, strings: &[String]) -> String {
        struct TNode {
            children: BTreeMap<u32, usize>,
            pattern: bool,
        }
        let mut nodes = vec![TNode { children: BTreeMap::new(), pattern: false }];
        for w in strings {
            let mut cur = 0;
            for ch in w.chars() {
                cur = match nodes[cur].children.get(&(ch as u32)) {
                    Some(&c) => c,
                    None => {
                        nodes.push(TNode { children: BTreeMap::new(), pattern: false });
                        let c = nodes.len() - 1;
                        nodes[cur].children.insert(ch as u32, c);
                        c
                    }
                };
            }
            nodes[cur].pattern = true;
        }
        let char_rule = self.add_primitive("char", "char");
        fn walk(nodes: &[TNode], idx: usize, char_rule: &str, out: &mut String) {
            let node = &nodes[idx];
            let mut rejects = String::new();
            for (i, (cpt, child)) in node.children.iter().enumerate() {
                let c = char::from_u32(*cpt).unwrap();
                rejects.push(c);
                if i > 0 {
                    out.push_str(" | ");
                }
                out.push('[');
                out.push(c);
                out.push(']');
                if !nodes[*child].children.is_empty() {
                    out.push_str(" (");
                    walk(nodes, *child, char_rule, out);
                    out.push(')');
                } else {
                    out.push_str(&format!(" {char_rule}+"));
                }
            }
            if !node.children.is_empty() {
                out.push_str(&format!(" | [^\"{rejects}] {char_rule}*"));
            }
        }
        let mut out = String::from("[\"] ( ");
        walk(&nodes, 0, &char_rule, &mut out);
        out.push_str(" )");
        if !nodes[0].pattern {
            out.push('?');
        }
        out.push_str(" [\"]");
        out
    }

    fn resolve_ref(&mut self, doc: &Doc, r: &str) -> String {
        let frag = match r.find('#') {
            Some(i) => &r[i + 1..],
            None => r,
        };
        let mut ref_name = format!("ref{}", esc_name(frag));
        if !self.rules.contains_key(&ref_name) && !self.resolving.contains(r) {
            let Some(target) = doc.refs.get(r) else {
                self.errors.push(format!("Unresolved $ref {r}"));
                return String::new();
            };
            self.resolving.insert(r.to_string());
            ref_name = self.visit(doc, target, &ref_name);
            self.resolving.remove(r);
        }
        ref_name
    }

    fn object_rule(&mut self, doc: &Doc, properties: &[(String, &Node)], required: &HashSet<String>, name: &str, additional: Option<&Node>) -> String {
        let dash = if name.is_empty() { "" } else { "-" };
        let (mut req, mut opt) = (Vec::<String>::new(), Vec::<String>::new());
        let mut kv: HashMap<String, String> = HashMap::new();
        let mut names = Vec::new();
        for (pn, ps) in properties {
            let pr = self.visit(doc, ps, &format!("{name}{dash}{pn}"));
            let r = self.add_rule(&format!("{name}{dash}{pn}-kv"), &format!("{} space \":\" space {pr}", format_literal(&Value::Str(pn.clone()).dump())));
            kv.insert(pn.clone(), r);
            if required.contains(pn) {
                req.push(pn.clone());
            } else {
                opt.push(pn.clone());
            }
            names.push(pn.clone());
        }
        if let Some(a) = additional {
            let sub = format!("{name}{dash}additional");
            let value_rule = if matches!(a, Node::Any) { self.add_primitive("value", "value") } else { self.visit(doc, a, &format!("{sub}-value")) };
            let key_rule = if names.is_empty() {
                self.add_primitive("string", "string")
            } else {
                let ns = self.not_strings(&names);
                self.add_rule(&format!("{sub}-k"), &ns)
            };
            let k = self.add_rule(&format!("{sub}-kv"), &format!("{key_rule} \":\" space {value_rule}"));
            kv.insert("*".into(), k);
            opt.push("*".into());
        }
        if req.is_empty() && opt.is_empty() {
            return "\"{\" space \"}\"".into();
        }
        let mut rule = String::from("\"{\" space ");
        for (i, r) in req.iter().enumerate() {
            if i > 0 {
                rule.push_str(" \",\" space ");
            }
            rule.push_str(&kv[r]);
        }
        if !opt.is_empty() {
            rule.push_str(" (");
            if !req.is_empty() {
                rule.push_str(" \",\" space ( ");
            }
            for i in 0..opt.len() {
                if i > 0 {
                    rule.push_str(" | ");
                }
                let r = self.recursive_refs(&opt[i..], false, &kv, name);
                rule.push_str(&r);
            }
            if !req.is_empty() {
                rule.push_str(" )");
            }
            rule.push_str(" )?");
        }
        rule.push_str(" space \"}\"");
        rule
    }

    fn recursive_refs(&mut self, ks: &[String], first_is_optional: bool, kv: &HashMap<String, String>, name: &str) -> String {
        let Some(k) = ks.first() else { return String::new() };
        let kvn = &kv[k];
        let comma = format!("( \",\" space {kvn} )");
        let mut res = if first_is_optional {
            format!("{comma}{}", if k == "*" { "*" } else { "?" })
        } else if k == "*" {
            format!("{kvn} {comma}*")
        } else {
            kvn.clone()
        };
        if ks.len() > 1 {
            let rest = self.recursive_refs(&ks[1..], true, kv, name);
            let n = self.add_rule(&format!("{name}{}{k}-rest", if name.is_empty() { "" } else { "-" }), &rest);
            res.push(' ');
            res.push_str(&n);
        }
        res
    }

    fn all_of(&mut self, doc: &Doc, children: &[Node], name: &str, rule_name: &str) -> String {
        let mut required = HashSet::new();
        let mut props: Vec<(String, &Node)> = Vec::new();
        let mut enums: BTreeMap<String, usize> = BTreeMap::new();
        fn add<'d>(doc: &'d Doc, c: &'d Node, req: bool, required: &mut HashSet<String>, props: &mut Vec<(String, &'d Node)>, enums: &mut BTreeMap<String, usize>) {
            match c {
                Node::Ref(r) => {
                    if let Some(t) = doc.refs.get(r) {
                        add(doc, t, req, required, props, enums);
                    }
                }
                Node::Object { props: ps, .. } => {
                    for (n, s, _) in ps {
                        props.push((n.clone(), s));
                        if req {
                            required.insert(n.clone());
                        }
                    }
                }
                Node::Enum(vs) => {
                    for v in vs {
                        *enums.entry(format_literal(&v.dump())).or_default() += 1;
                    }
                }
                _ => {}
            }
        }
        for c in children {
            if let Node::AnyOf(alts) = c {
                for a in alts {
                    add(doc, a, false, &mut required, &mut props, &mut enums);
                }
            } else {
                add(doc, c, true, &mut required, &mut props, &mut enums);
            }
        }
        if !enums.is_empty() {
            let inter: Vec<&str> = enums.iter().filter(|(_, n)| **n == children.len()).map(|(k, _)| k.as_str()).collect();
            if !inter.is_empty() {
                let r = format!("({})", inter.join(" | "));
                return self.add_rule(rule_name, &r);
            }
        }
        let r = self.object_rule(doc, &props, &required, name, None);
        self.add_rule(rule_name, &r)
    }

    fn visit(&mut self, doc: &Doc, s: &Node, name: &str) -> String {
        let rule_name = if is_reserved(name) { format!("{name}-") } else if name.is_empty() { "root".into() } else { name.to_string() };
        let sub_name = format!("{name}{}", if name.is_empty() { "" } else { "-" });
        match s {
            Node::Ref(r) => {
                let t = self.resolve_ref(doc, r);
                self.add_rule(&rule_name, &t)
            }
            Node::AnyOf(c) => {
                let u = self.union(doc, name, c);
                self.add_rule(&rule_name, &u)
            }
            Node::AllOf(c) => self.all_of(doc, c, name, &rule_name),
            Node::Const(v) => self.add_rule(&rule_name, &format_literal(&v.dump())),
            Node::Enum(vs) => {
                let r = format!("({})", vs.iter().map(|v| format_literal(&v.dump())).collect::<Vec<_>>().join(" | "));
                self.add_rule(&rule_name, &r)
            }
            Node::Object { props, additional } => {
                if props.is_empty() && matches!(additional.as_deref(), Some(Node::Any)) {
                    let o = self.add_primitive("object", "object");
                    return self.add_rule(&rule_name, &o);
                }
                let ps: Vec<(String, &Node)> = props.iter().map(|(n, s, _)| (n.clone(), s)).collect();
                let req: HashSet<String> = props.iter().filter(|p| p.2).map(|p| p.0.clone()).collect();
                let r = self.object_rule(doc, &ps, &req, name, additional.as_deref());
                self.add_rule(&rule_name, &r)
            }
            Node::Tuple(items) => {
                let mut r = String::from("\"[\" space ");
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        r.push_str(" \",\" space ");
                    }
                    r.push_str(&self.visit(doc, it, &format!("{sub_name}tuple-{i}")));
                }
                r.push_str(" space \"]\"");
                self.add_rule(&rule_name, &r)
            }
            Node::Array { items, min, max } => {
                if matches!(**items, Node::Any) && *min == 0 && *max < 0 {
                    return self.visit_primitive(&rule_name, "array");
                }
                let item = self.visit(doc, items, &format!("{sub_name}item"));
                let max = if *max < 0 { i32::MAX } else { *max };
                let r = format!("\"[\" space {} space \"]\"", build_repetition(&item, *min, max, "\",\" space"));
                self.add_rule(&rule_name, &r)
            }
            Node::Str { pattern, format, min_len, max_len } => {
                if !pattern.is_empty() {
                    return self.visit_pattern(pattern, &rule_name);
                }
                if *format == Fmt::Uuid {
                    return self.visit_primitive(&rule_name, "uuid");
                }
                if *format != Fmt::None {
                    let p = match format {
                        Fmt::Date => "date-string",
                        Fmt::Time => "time-string",
                        _ => "date-time-string",
                    };
                    let n = self.add_primitive(p, p);
                    return self.add_rule(&rule_name, &n);
                }
                if *min_len > 0 || *max_len >= 0 {
                    let c = self.add_primitive("char", "char");
                    let mx = if *max_len < 0 { i32::MAX } else { *max_len };
                    return self.add_rule(&rule_name, &format!("\"\\\"\" {} \"\\\"\"", build_repetition(&c, *min_len, mx, "")));
                }
                self.visit_primitive(&rule_name, "string")
            }
            Node::Integer { min, max } => {
                if *min == i64::MIN && *max == i64::MAX {
                    return self.visit_primitive(&rule_name, "integer");
                }
                let mut o = String::from("(");
                build_min_max_int(*min, *max, &mut o, 16, true);
                o.push(')');
                self.add_rule(&rule_name, &o)
            }
            Node::Number => self.visit_primitive(&rule_name, "number"),
            Node::Boolean => self.visit_primitive(&rule_name, "boolean"),
            Node::Null => self.visit_primitive(&rule_name, "null"),
            Node::Any => {
                let v = self.add_primitive("value", "value");
                self.add_rule(&rule_name, &v)
            }
        }
    }

    fn finish(self) -> Result<String, String> {
        if !self.errors.is_empty() {
            return Err(format!("JSON schema conversion failed:\n{}", self.errors.join("\n")));
        }
        Ok(self.rules.iter().map(|(k, v)| format!("{k} ::= {v}\n")).collect())
    }
}

/// llama.cpp's `json_schema_to_grammar(schema)` (`force_gbnf`): the grammar whose root is the schema. A schema it
/// cannot read, or a pattern no regex would accept, is an error with llama.cpp's message; a pattern it reads but
/// cannot translate becomes "any string" (llama.cpp warns, and so does this, in the second value).
pub fn json_schema_to_grammar(schema: &Value) -> Result<(String, Vec<String>), String> {
    let doc = schema_doc(schema).map_err(|e| format!("JSON schema conversion failed:\n{e}"))?;
    let mut c = Conv::new();
    c.visit(&doc, &doc.root, "");
    let w = std::mem::take(&mut c.warnings);
    Ok((c.finish()?, w))
}

// ---------------------------------------------------------------- the chat wrapping (jinja path, Qwen3) ----------

/// The rules the PEG chat parser's `json()` adds (reachable through the schema parser's child, whatever the schema).
const PEG_JSON_RULES: &[(&str, &str)] = &[
    ("json-array", "\"[\" space (\"]\" | json-value (space \",\" space json-value)* space \"]\")"),
    ("json-bool", "\"true\" | \"false\""),
    ("json-null", "\"null\""),
    ("json-number", "\"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ((\"e\" | \"E\") [+-]? [0-9]+)?"),
    ("json-object", "\"{\" space (\"}\" | json-string space \":\" space json-value (space \",\" space json-string space \":\" space json-value)* space \"}\")"),
    ("json-string", "\"\\\"\" ( [^\"\\\\] | \"\\\\\" ( [\"\\\\/ bfnrt] | \"u\" [0-9a-fA-F]{4} ) )* \"\\\"\""),
    ("json-value", "json-object | json-array | json-string | json-number | json-bool | json-null"),
];

/// The reasoning block's `until("</think>")` (parser id 13 on the pinned template) and the root: the generation
/// prompt, an optional `<think>…</think>`, then the value, bare or in a ```` ```json ```` fence.
const PEG_UNTIL_RULES: &[(&str, &str)] = &[
    ("until-13", "| [<] until-13-01 | [^<] until-13"),
    ("until-13-01", "| [<] until-13-01 | [/] until-13-02 | [^/<] until-13"),
    ("until-13-02", "| [<] until-13-01 | [t] until-13-03 | [^<t] until-13"),
    ("until-13-03", "| [<] until-13-01 | [h] until-13-04 | [^<h] until-13"),
    ("until-13-04", "| [<] until-13-01 | [i] until-13-05 | [^<i] until-13"),
    ("until-13-05", "| [<] until-13-01 | [n] until-13-06 | [^<n] until-13"),
    ("until-13-06", "| [<] until-13-01 | [k] until-13-07 | [^<k] until-13"),
    ("until-13-07", "| [<] until-13-01 | [^<>] until-13"),
];
const PEG_ROOT_QWEN3: &str = "\"<|im_start|>assistant\\n\" space (\"<think>\" \"\\n\"? until-13 \"\\n\"? \"</think>\" \"\\n\"? \"\\n\"?)? space (\"```json\" space response-format space \"```\" | space response-format space)";

/// 0.3.5: on a ChatML template without reasoning (SmolLM2-Instruct's, mindx-genN's) the parser has no reasoning
/// block: no `until-13` rules, and the root is the generation prompt then the value (llama.cpp b11192's own output on
/// both templates, `oracle_schema_grammars`).
const PEG_ROOT_CHATML: &str = "\"<|im_start|>assistant\\n\" space space (\"```json\" space response-format space \"```\" | space response-format space)";

/// The grammar llama-server b11192 builds for a response-format schema on the jinja path (thinking off) for the
/// model's template: `common_peg_arena::build_grammar` through one converter, so the schema's rules sit beside the
/// parser's. The schema must be a non-empty object (llama-server builds no grammar for anything else).
pub fn chat_grammar(schema: &Value, t: crate::chat::Template) -> Result<(String, Vec<String>), String> {
    // common_chat_templates_apply_jinja wraps whatever the parser generation throws (chat.cpp:1364)
    const WRAP: &str = "Unable to generate parser for this template. Automatic parser generation failed: ";
    chat_grammar_inner(schema, t).map_err(|e| if e.contains("not UTF-8") { e } else { format!("{WRAP}{e}") })
}

fn chat_grammar_inner(schema: &Value, t: crate::chat::Template) -> Result<(String, Vec<String>), String> {
    // `common_chat_schema_from_json` throws out of `p.schema(…)` before any grammar is built
    let doc = schema_doc(schema)?;
    let mut c = Conv::new();
    for (n, r) in PEG_JSON_RULES {
        c.add_rule(n, r);
    }
    let r = c.visit(&doc, &doc.root, "response-format-schema");
    c.add_rule("response-format", &r);
    if t == crate::chat::Template::Qwen3 {
        for (n, r) in PEG_UNTIL_RULES {
            c.add_rule(n, r);
        }
        c.add_rule("root", PEG_ROOT_QWEN3);
    } else {
        c.add_rule("root", PEG_ROOT_CHATML);
    }
    let w = std::mem::take(&mut c.warnings);
    Ok((c.finish()?, w))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Value {
        Value::parse(s).unwrap()
    }

    fn chat_grammar_q(s: &Value) -> Result<(String, Vec<String>), String> {
        chat_grammar(s, crate::chat::Template::Qwen3)
    }

    #[test]
    fn json_reads_and_prints_as_nlohmann() {
        assert_eq!(v(r#"{"b": 1, "a": [1.0, -2, 18446744073709551615, 1e20, 1.5e-7, 0.001, 123456789012345680000.0], "b": "x\u0001\"é"}"#).dump(),
                   r#"{"b":"x\u0001\"é","a":[1.0,-2,18446744073709551615,1e+20,1.5e-07,0.001,1.2345678901234568e+20]}"#);
        assert_eq!(v("[1e15, 1e14, 100, 0.0, -0.0, 3.14]").dump(), "[1e+15,100000000000000.0,100,0.0,-0.0,3.14]");
        assert!(Value::parse("[01]").is_none() && Value::parse("[1.]").is_none() && Value::parse("\"\\ud800\"").is_none());
    }

    #[test]
    fn the_object_schema_is_the_json_mode_grammar() {
        // the constant 0.3.3 checked against the server's own report, now built by the converter
        for s in [r#"{"type": "object"}"#, r#"{"type": "object", "additionalProperties": true}"#] {
            assert_eq!(chat_grammar_q(&v(s)).unwrap().0, crate::grammar::JSON_OBJECT_GRAMMAR);
            for t in [crate::chat::Template::SmolLm2, crate::chat::Template::ChatMl] {
                assert_eq!(chat_grammar(&v(s), t).unwrap().0, crate::grammar::JSON_OBJECT_GRAMMAR_CHATML);
            }
        }
    }

    /// llama.cpp's own `tests/test-json-schema-to-grammar.cpp` cases (b11192), with their expected grammars
    #[test]
    fn llama_cpp_test_cases() {
        let trim = |s: &str| s.trim().lines().map(str::trim_start).collect::<Vec<_>>().join("\n");
        let file = crate::serve::Json::parse(include_str!("../testing/json_schema_cases.json")).unwrap();
        let Some(crate::serve::Json::Arr(cases)) = file.get("cases") else { panic!() };
        let mut n = 0;
        for c in cases {
            let name = c.get("name").and_then(|x| x.as_str()).unwrap();
            let ok = c.get("success").and_then(|x| x.as_bool()).unwrap();
            let schema = Value::parse(c.get("schema").and_then(|x| x.as_str()).unwrap()).unwrap();
            let got = json_schema_to_grammar(&schema);
            if ok {
                let want = c.get("grammar").and_then(|x| x.as_str()).unwrap();
                let g = got.unwrap_or_else(|e| panic!("{name}: {e}")).0;
                assert_eq!(trim(&g), trim(want), "{name}");
                crate::grammar::Rules::parse(&g, &|_: &[u8]| vec![]).unwrap_or_else(|e| panic!("{name}: the grammar parses: {e}"));
            } else {
                assert!(got.is_err(), "{name}: llama.cpp refuses it");
            }
            n += 1;
        }
        assert!(n >= 80, "{n} cases");
    }

    /// llama.cpp b11192's own converter and chat wrapping, in libllama-common.so (testing/schema_oracle.py): for every
    /// schema of the corpus, the same grammar text byte for byte, or the same refusal; every grammar parses.
    #[test]
    #[ignore = "needs .models/oracle-schema (testing/schema_oracle.py)"]
    fn oracle_schema_grammars() {
        use crate::chat::Template;
        // 0.3.5: each reproduced template, from the model that carries it (schema_oracle.py records each)
        let mut files = 0;
        for (file, t) in [("schemas.jsonl", Template::Qwen3), ("schemas-SmolLM2-135M-Instruct-F16.jsonl", Template::SmolLm2),
                          ("schemas-mindx-gen39-F16.jsonl", Template::ChatMl)] {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/oracle-schema").join(file);
            if t == Template::Qwen3 || p.exists() {
                schema_oracle_file(&p, t);
                files += 1;
            }
        }
        eprintln!("schema oracle: {files} templates");
    }

    fn schema_oracle_file(path: &std::path::Path, t: crate::chat::Template) {
        use crate::serve::Json;
        let recs = std::fs::read_to_string(path).unwrap();
        let (mut n, mut same, mut msg_same, mut errs, mut split, mut parsed) = ([0usize; 2], [0usize; 2], [0usize; 2], [0usize; 2], [0usize; 2], 0);
        let no_tok = |_: &[u8]| vec![];
        for line in recs.lines() {
            let r = Json::parse(line).unwrap();
            let s = |k: &str| r.get(k).and_then(Json::as_str).map(str::to_string);
            let name = s("name").unwrap();
            let schema = Value::parse(&s("schema").unwrap()).unwrap_or_else(|| panic!("{name}: the schema reads"));
            // the chat path's input, as oaicompat_chat_params_parse makes it: an empty object is any object
            let chat_in = if schema == Value::Obj(vec![]) { Value::parse(r#"{"type":"object"}"#).unwrap() } else { schema.clone() };
            let chat_applies = matches!(&chat_in, Value::Obj(o) if !o.is_empty());
            for (i, (got, key)) in [(json_schema_to_grammar(&schema), "raw"), (if chat_applies { chat_grammar(&chat_in, t) } else { Ok((String::new(), vec![])) }, "chat")]
                .into_iter().enumerate() {
                n[i] += 1;
                let ok = match (&got, s(key), s(&format!("{key}_err")), s(&format!("{key}_not_utf8"))) {
                    (Ok((g, _)), Some(want), _, _) => {
                        if !g.is_empty() {
                            crate::grammar::Rules::parse(g, &no_tok).unwrap_or_else(|e| panic!("{name} ({key}): the grammar parses: {e}"));
                            parsed += 1;
                        }
                        *g == want
                    }
                    (Err(e), None, Some(want), _) => {
                        errs[i] += 1;
                        msg_same[i] += (*e == want) as usize;
                        if *e != want {
                            eprintln!("  {name} ({key}): refused by both; message differs: bankML {e:?}, llama.cpp {want:?}");
                        }
                        true
                    }
                    (Err(e), None, None, Some(_)) => {
                        split[i] += 1;
                        e.contains("not UTF-8")
                    }
                    _ => false,
                };
                if ok {
                    same[i] += 1;
                } else {
                    eprintln!("  {name} ({key}): bankML {got:?}\n  llama.cpp {:?} {:?}", s(key), s(&format!("{key}_err")));
                }
            }
            if chat_applies && r.get("chat").is_some() {
                assert_eq!(s("prompt").as_deref(), Some(t.generation_prompt()), "{name}: the generation prompt");
            }
        }
        eprintln!("schema oracle ({t:?}): json_schema_to_grammar {} of {} as llama.cpp b11192 ({} refused by both, {} with the same message; {} \
                   grammar llama.cpp writes as non-UTF-8, refused by bankML); the chat grammar {} of {} ({} refused by both, {} with the same \
                   message; {} non-UTF-8 refused); the rest byte-identical, and all {parsed} grammars parse",
                  same[0], n[0], errs[0], msg_same[0], split[0], same[1], n[1], errs[1], msg_same[1], split[1]);
        assert_eq!(same, n);
    }

    #[test]
    fn refusals_say_where() {
        let e = chat_grammar_q(&v(r#"{"type": "object", "properties": {"a": {"type": "frob"}}}"#)).unwrap_err();
        assert_eq!(e, "Unable to generate parser for this template. Automatic parser generation failed: JSON schema error at #/properties/a: unrecognized type frob");
        let e = chat_grammar_q(&v(r#"{"type": "string", "pattern": "^é+$"}"#)).unwrap_err();
        assert!(e.contains("not UTF-8"), "{e}");
        assert!(chat_grammar_q(&v(r#"{"type": "string", "pattern": "^(é)+$"}"#)).is_ok());
        let e = json_schema_to_grammar(&v(r#"{"type": "string", "pattern": "^(a$"}"#)).unwrap_err();
        assert_eq!(e, "JSON schema conversion failed:\nInvalid pattern ^(a$: unbalanced parentheses");
        let (g, w) = json_schema_to_grammar(&v(r#"{"type": "string", "pattern": "^\\d+$"}"#)).unwrap();
        assert!(g.contains("root ::= string") && w[0].contains("not supported"), "{g} {w:?}");
    }
}
