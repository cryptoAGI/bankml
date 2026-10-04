// SPDX-License-Identifier: MIT OR Apache-2.0
//! `bankml create` (O5 of docs/OLLAMA.md): Ollama's `ollama create` over bankML's gate. A **derived model** is a
//! manifest that layers configuration on a **pinned base** — it never copies weights:
//!
//! ```text
//!   <forks>/<name>.MODEL.json   {kind, bankml, name, created_at, base: {name, file, sha256, fork, path},
//!                                system, parameters, stop, template_sha256, license, messages, requires, digest}
//! ```
//!
//! `digest` is the sha256 of the manifest's *content* (the base's sha256 and the layer, not the name or the time), so
//! the same Modelfile over the same base gives the same digest (Ollama's content addressing) and a hand-edited
//! manifest is refused when it is read. Loading a derived model verifies the **base** exactly as a pinned model is
//! verified (the guard, then the sha256 pin of its FORK.json), then applies the layer.
//!
//! **The Modelfile subset** (`ollama/parser/parser.go`'s grammar: case-insensitive instructions, `#` comments at a
//! line's start, a value runs to the end of the line, `"…"` may span lines, `"""…"""` too, no escapes; the vendored
//! reference is mindX `docs/ollama/setup/modelfile.md`):
//!
//! - `FROM` a registry name (pinned or derived), a pinned GGUF path, or a safetensors directory — converted by
//!   `convert.rs` (byte-identical to llama.cpp b11192's `convert_hf_to_gguf.py --outtype f16`) and pinned by a FORK.json
//!   that records every input's sha256;
//! - `SYSTEM`; `PARAMETER` temperature, top_k, top_p, min_p, seed, num_ctx, num_predict and stop (the parameters
//!   bankML reproduces) — any other is refused with the reason; `TEMPLATE` only when it is the base's own pinned
//!   template (bankML renders that template, byte-identical to llama.cpp; a different one would change every prompt);
//!   `ADAPTER` refused (LoRA merging is a later O-phase: merge first, then `FROM` the merged directory); `LICENSE`,
//!   `MESSAGE` and `REQUIRES` recorded (and `MESSAGE`s are applied, as Ollama applies them).
//!
//! **The layer, applied as Ollama applies it** (`server/routes.go`): `/api/chat` (and `/v1/chat/completions`) — the
//! model's `MESSAGE`s go before the request's messages, and its `SYSTEM` goes first **unless the request's first
//! message is a system message**; `/api/generate` — the request's `system` if it has one, else the model's, then the
//! `MESSAGE`s, then the prompt (`raw` prompts get no system, as in Ollama); the parameters are defaults the request's
//! `options` override key by key (`stop` as a whole list).

use crate::gguf::jstr;
use crate::native::{Entry, Registry, Residency};
use crate::serve::{respond, Json};
use std::io::Write;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------- the Modelfile parser (Ollama's) --------------

/// One instruction: `name` is `model` (FROM), `system`, `template`, `adapter`, `license`, `message`, `requires`, or a
/// parameter's name; `args` its value (`role: content` for a message), as Ollama's parser gives them.
#[derive(Debug, Clone, PartialEq)]
pub struct Command {
    pub name: String,
    pub args: String,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum St {
    Nil,
    Name,
    Value,
    Parameter,
    Message,
    Comment,
}

/// Go's `strconv.IsPrint`, closely enough: no control characters, no space other than U+0020, no format characters.
fn is_print(c: char) -> bool {
    !c.is_control() && !(c.is_whitespace() && c != ' ') && !matches!(c, '\u{ad}' | '\u{200b}'..='\u{200f}' | '\u{2028}'..='\u{202e}' | '\u{2060}'..='\u{2064}' | '\u{feff}')
}

/// `unquote` of parser.go: `"""x"""` and `"x"` lose their quotes; an unterminated quote is not yet a value.
fn unquote(s: &str) -> Option<String> {
    if let Some(r) = s.strip_prefix("\"\"\"") {
        return (s.len() >= 6 && s.ends_with("\"\"\"")).then(|| r[..r.len() - 3].to_string());
    }
    if let Some(r) = s.strip_prefix('"') {
        return (s.len() >= 2 && s.ends_with('"')).then(|| r[..r.len() - 1].to_string());
    }
    Some(s.to_string())
}

const COMMANDS: [&str; 8] = ["from", "license", "template", "system", "adapter", "parameter", "message", "requires"];

/// Ollama's Modelfile parser (`parser.ParseFile`), state for state.
pub fn parse_modelfile(text: &str) -> Result<Vec<Command>, String> {
    let bad_cmd = || "command must be one of \"from\", \"license\", \"template\", \"system\", \"adapter\", \"parameter\", \"message\" or \"requires\"".to_string();
    let (mut cmds, mut cur, mut b, mut name, mut role) = (Vec::new(), St::Nil, String::new(), String::new(), String::new());
    let mut line = 1;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let at = |line: usize, m: String| format!("Modelfile line {line}: {m}");
    for r in text.chars() {
        let (sp, nl) = (r == ' ' || r == '\t', r == '\r' || r == '\n');
        if nl {
            line += 1;
        }
        let (mut next, rr): (St, Option<char>) = match cur {
            St::Nil if r == '#' => (St::Comment, None),
            St::Nil if sp || nl => (St::Nil, None),
            St::Nil => (St::Name, Some(r)),
            St::Name if r.is_ascii_alphabetic() => (St::Name, Some(r)),
            St::Name if sp => (St::Value, None),
            St::Name => return Err(at(line, bad_cmd())),
            St::Value if nl || sp => (St::Nil, Some(r)),
            St::Value => (St::Value, Some(r)),
            St::Parameter if r.is_ascii_alphanumeric() || r == '_' => (St::Parameter, Some(r)),
            St::Parameter if sp => (St::Value, None),
            St::Parameter => return Err(at(line, "a PARAMETER needs a name and a value".into())),
            St::Message if r.is_ascii_alphabetic() => (St::Message, Some(r)),
            St::Message if sp => (St::Value, None),
            St::Message => return Err(at(line, "a MESSAGE needs a role and a message".into())),
            St::Comment if nl => (St::Nil, None),
            St::Comment => (St::Comment, None),
        };
        if next != cur {
            match cur {
                St::Name => {
                    let s = b.to_ascii_lowercase();
                    if !COMMANDS.contains(&s.as_str()) {
                        return Err(at(line, bad_cmd()));
                    }
                    match s.as_str() {
                        "from" => name = "model".into(),
                        "parameter" => next = St::Parameter,
                        "message" => {
                            next = St::Message;
                            name = s;
                        }
                        _ => name = s,
                    }
                }
                St::Parameter => {
                    if b.is_empty() {
                        return Err(at(line, "a PARAMETER needs a name".into()));
                    }
                    name = b.clone();
                }
                St::Message => {
                    if !["system", "user", "assistant"].contains(&b.as_str()) {
                        return Err(at(line, "message role must be one of \"system\", \"user\", or \"assistant\"".into()));
                    }
                    role = b.clone();
                }
                St::Value => {
                    let v = unquote(b.trim());
                    if v.is_none() || sp {
                        b.push(r); // not yet a whole value (an open quote), or a space inside it
                        continue;
                    }
                    let mut s = v.unwrap();
                    if !role.is_empty() {
                        s = format!("{role}: {s}");
                        role.clear();
                    }
                    cmds.push(Command { name: name.clone(), args: s });
                }
                St::Nil | St::Comment => {}
            }
            b.clear();
            cur = next;
        }
        if let Some(c) = rr.filter(|c| is_print(*c)) {
            b.push(c);
        }
    }
    match cur {
        St::Nil | St::Comment => {}
        St::Value => {
            let mut s = unquote(b.trim()).ok_or_else(|| at(line, format!("unexpected EOF: an unterminated quote in {b:?}")))?;
            if !role.is_empty() {
                s = format!("{role}: {s}");
            }
            cmds.push(Command { name, args: s });
        }
        _ => return Err(at(line, "unexpected EOF".into())),
    }
    Ok(cmds)
}

/// Ollama's `quote` (how `ollama show --modelfile` writes a value): `"""…"""` when it has a newline or an edge space and
/// a quote, `"…"` when it has a newline or an edge space, bare otherwise.
pub fn quote(s: &str) -> String {
    if s.contains('\n') || s.starts_with(' ') || s.ends_with(' ') {
        if s.contains('"') { format!("\"\"\"{s}\"\"\"") } else { format!("\"{s}\"") }
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------- the layer ------------------------------------

/// A parameter value bankML reproduces.
#[derive(Debug, Clone, PartialEq)]
pub enum P {
    Int(i64),
    Float(f64),
}

impl P {
    fn json(&self) -> String {
        match self {
            P::Int(i) => i.to_string(),
            P::Float(f) => format!("{}", *f as f32), // stored at 32 bits, written in the shortest form that reads back
        }
    }
    fn num(&self) -> f64 {
        match self {
            P::Int(i) => *i as f64,
            P::Float(f) => *f,
        }
    }
}

/// A JSON number: the shortest form that reads back (`0.7`, `2048`).
fn fmt_f(f: f64) -> String {
    format!("{f}")
}

/// The parameters bankML reproduces, in the order a manifest keeps them, and whether each is an integer.
/// O2 (0.3.6): the penalties join them.
pub const PARAMS: [(&str, bool); 11] = [("temperature", false), ("top_k", true), ("top_p", false), ("min_p", false), ("seed", true), ("num_ctx", true), ("num_predict", true),
                                        ("repeat_penalty", false), ("repeat_last_n", true), ("presence_penalty", false), ("frequency_penalty", false)];

/// Why a parameter bankML does not reproduce is refused as a model default.
fn param_refusal(k: &str) -> String {
    match k {
        "penalize_newline" | "typical_p" | "tfs_z" | "mirostat" | "mirostat_eta" | "mirostat_tau" | "min_keep" =>
            format!("PARAMETER {k}: bankML reproduces llama.cpp's penalties, temperature, top-k, top-p, min-p and seed; this sampler is not reproduced, so it cannot become a default of the model"),
        "num_thread" | "num_gpu" | "main_gpu" | "num_batch" | "use_mmap" | "use_mlock" | "low_vram" | "numa" | "f16_kv" | "vocab_only" | "logits_all" | "num_keep" | "num_gqa" | "rope_frequency_base" | "rope_frequency_scale" =>
            format!("PARAMETER {k}: a resource option; it does not change an answer bankML gives, so it is not part of a model (give it to bankml serve)"),
        _ => format!("PARAMETER {k}: not a parameter bankML reproduces (it reproduces {} and stop)", PARAMS.map(|p| p.0).join(", ")),
    }
}

/// What a Modelfile (or a structured `/api/create` request) asks for, checked.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Spec {
    pub from: String,
    pub system: Option<String>,
    pub template: Option<String>,
    pub params: Vec<(String, P)>,
    pub stop: Option<Vec<String>>,
    pub license: Vec<String>,
    pub messages: Option<Vec<(String, String)>>,
    pub requires: Option<String>,
}

fn set_param(params: &mut Vec<(String, P)>, stop: &mut Option<Vec<String>>, k: &str, v: &str) -> Result<(), String> {
    if k == "stop" {
        stop.get_or_insert_with(Vec::new).push(v.to_string());
        return Ok(());
    }
    let Some(&(_, is_int)) = PARAMS.iter().find(|p| p.0 == k) else { return Err(param_refusal(k)) };
    let p = if is_int {
        P::Int(v.trim().parse().map_err(|_| format!("PARAMETER {k} {v:?}: not an integer"))?)
    } else {
        let f: f64 = v.trim().parse().map_err(|_| format!("PARAMETER {k} {v:?}: not a number"))?;
        if !f.is_finite() {
            return Err(format!("PARAMETER {k} {v:?}: not a finite number"));
        }
        P::Float(f as f32 as f64) // Ollama parses floats at 32 bits
    };
    match params.iter_mut().find(|(n, _)| n == k) {
        Some(e) => e.1 = p, // the last one wins, as in Ollama
        None => params.push((k.to_string(), p)),
    }
    Ok(())
}

impl Spec {
    /// From a Modelfile: one FROM, the last SYSTEM and TEMPLATE, every LICENSE and MESSAGE, parameters with the last
    /// value winning and `stop` accumulating — Ollama's rules; ADAPTER refused.
    pub fn from_modelfile(text: &str) -> Result<Spec, String> {
        let mut s = Spec::default();
        let mut from = Vec::new();
        for c in parse_modelfile(text)? {
            match c.name.as_str() {
                "model" => from.push(c.args),
                "system" => s.system = Some(c.args),
                "template" => s.template = Some(c.args),
                "license" => s.license.push(c.args),
                "requires" => s.requires = Some(c.args),
                "adapter" => return Err(format!("ADAPTER {}: bankML does not merge LoRA adapters yet (a later O-phase); merge it into the base weights \
                                                 (mindXtrain serve writes ollama_push/merged), then FROM the merged safetensors directory", c.args)),
                "message" => {
                    let (r, m) = c.args.split_once(": ").unwrap_or((&c.args, ""));
                    s.messages.get_or_insert_with(Vec::new).push((r.to_string(), m.to_string()));
                }
                k => set_param(&mut s.params, &mut s.stop, k, &c.args)?,
            }
        }
        match from.len() {
            0 => Err("no FROM line for the model was specified".into()),
            1 => {
                s.from = from.pop().unwrap();
                Ok(s)
            }
            _ => Err("more than one FROM: a bankML model has one base".into()),
        }
    }

    /// From `/api/create`'s structured form: `from`, `system`, `template`, `license` (a string or strings),
    /// `parameters` (an object), `messages`; `files`, `adapters` and `quantize` refused with the reason.
    pub fn from_request(req: &Json) -> Result<Spec, String> {
        let nonempty = |k: &str| match req.get(k) {
            None | Some(Json::Null) => false,
            Some(Json::Str(s)) => !s.is_empty(),
            Some(Json::Obj(o)) => !o.is_empty(),
            Some(Json::Arr(a)) => !a.is_empty(),
            Some(_) => true,
        };
        if nonempty("files") {
            return Err("files: bankML does not take uploaded blobs; FROM a pinned GGUF (its FORK.json in the registry) or a safetensors directory on this machine".into());
        }
        if nonempty("adapters") {
            return Err("adapters: bankML does not merge LoRA adapters yet (a later O-phase); merge it, then create FROM the merged safetensors directory".into());
        }
        if nonempty("quantize") {
            return Err("quantize: bankML serves the weights it pinned, as pinned; quantizing makes a new file, which must be pinned on its own".into());
        }
        let mut s = Spec { from: req.get("from").and_then(Json::as_str).unwrap_or("").to_string(), ..Spec::default() };
        if s.from.is_empty() {
            return Err("from: name the base (a registry name, a pinned GGUF path, or a safetensors directory)".into());
        }
        s.system = req.get("system").and_then(Json::as_str).map(str::to_string);
        s.template = req.get("template").and_then(Json::as_str).filter(|t| !t.is_empty()).map(str::to_string);
        s.license = match req.get("license") {
            Some(Json::Str(l)) if !l.is_empty() => vec![l.clone()],
            Some(Json::Arr(a)) => a.iter().filter_map(Json::as_str).map(str::to_string).collect(),
            _ => Vec::new(),
        };
        if let Some(Json::Arr(a)) = req.get("messages") {
            s.messages = Some(a.iter().map(|m| {
                let role = m.get("role").and_then(Json::as_str).unwrap_or("");
                if !["system", "user", "assistant"].contains(&role) {
                    return Err("message role must be one of \"system\", \"user\", or \"assistant\"".to_string());
                }
                Ok((role.to_string(), m.get("content").and_then(Json::as_str).unwrap_or("").to_string()))
            }).collect::<Result<_, _>>()?);
        }
        match req.get("parameters") {
            None | Some(Json::Null) => {}
            Some(Json::Obj(o)) => {
                for (k, v) in o {
                    match (k.as_str(), v) {
                        ("stop", Json::Arr(a)) => {
                            for x in a {
                                set_param(&mut s.params, &mut s.stop, "stop", x.as_str().ok_or("parameters.stop must be strings")?)?;
                            }
                        }
                        ("stop", Json::Str(x)) => set_param(&mut s.params, &mut s.stop, "stop", x)?,
                        (k, Json::Num(n)) => set_param(&mut s.params, &mut s.stop, k, &fmt_f(*n))?,
                        (k, _) => return Err(format!("parameters.{k}: must be a number")),
                    }
                }
            }
            Some(_) => return Err("parameters must be an object".into()),
        }
        Ok(s)
    }
}

/// The pinned base a derived model layers on.
#[derive(Debug, Clone, PartialEq)]
pub struct Base {
    pub name: String,
    pub file: String,
    pub sha256: String,
    /// the FORK.json that pins it (a file name in the registry directory)
    pub fork: String,
    /// where the file was verified when the model was created
    pub path: String,
}

/// A derived model: its name, the base, the layer, and the digest of its content.
#[derive(Debug, Clone, PartialEq)]
pub struct Derived {
    pub name: String,
    pub created_at: String,
    pub base: Base,
    pub system: Option<String>,
    pub params: Vec<(String, P)>,
    pub stop: Vec<String>,
    pub template_sha256: Option<String>,
    pub license: Vec<String>,
    pub messages: Vec<(String, String)>,
    pub requires: Option<String>,
    pub digest: String,
}

fn opt_str(s: &Option<String>) -> String {
    s.as_deref().map(jstr).unwrap_or("null".into())
}

fn strs(v: &[String]) -> String {
    format!("[{}]", v.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(", "))
}

impl Derived {
    /// The content the digest is taken over: the base's identity and the layer (not the name, not the time).
    pub fn content(&self) -> String {
        let params = self.params.iter().map(|(k, v)| format!("{}: {}", jstr(k), v.json())).collect::<Vec<_>>().join(", ");
        let msgs = self.messages.iter().map(|(r, c)| format!("{{\"role\": {}, \"content\": {}}}", jstr(r), jstr(c))).collect::<Vec<_>>().join(", ");
        format!("{{\"base_sha256\": \"{}\", \"system\": {}, \"parameters\": {{{params}}}, \"stop\": {}, \"template_sha256\": {}, \"license\": {}, \"messages\": [{msgs}], \"requires\": {}}}",
                self.base.sha256, opt_str(&self.system), strs(&self.stop), opt_str(&self.template_sha256), strs(&self.license), opt_str(&self.requires))
    }

    pub fn compute_digest(&self) -> String {
        let mut h = crate::sha256::Sha256::default();
        h.update(self.content().as_bytes());
        crate::sha256::hex(&h.finish())
    }

    /// The manifest file's text.
    pub fn to_json(&self) -> String {
        let b = &self.base;
        format!("{{\n \"kind\": \"bankml derived model (a layer over a pinned base; no weights)\",\n \"bankml\": \"{}\",\n \"name\": {},\n \"created_at\": \"{}\",\n \"base\": {{\"name\": {}, \"file\": {}, \"sha256\": \"{}\", \"fork\": {}, \"path\": {}}},\n \"layer\": {},\n \"digest\": \"sha256:{}\"\n}}\n",
                crate::VERSION, jstr(&self.name), self.created_at, jstr(&b.name), jstr(&b.file), b.sha256, jstr(&b.fork), jstr(&b.path), self.content(), self.digest)
    }

    /// A manifest read back; refused if its digest is not its content's.
    pub fn from_json(text: &str) -> Result<Derived, String> {
        let v = Json::parse(text).ok_or("not JSON")?;
        let s = |o: &Json, k: &str| o.get(k).and_then(Json::as_str).map(str::to_string);
        let b = v.get("base").ok_or("no base")?;
        let l = v.get("layer").ok_or("no layer")?;
        let list = |k: &str| -> Vec<String> { match l.get(k) { Some(Json::Arr(a)) => a.iter().filter_map(Json::as_str).map(str::to_string).collect(), _ => Vec::new() } };
        let params = match l.get("parameters") {
            Some(Json::Obj(o)) => o.iter().map(|(k, v)| {
                let is_int = PARAMS.iter().find(|p| p.0 == k).ok_or(format!("parameter {k} is not one bankML reproduces"))?.1;
                let n = match v { Json::Num(n) => *n, _ => return Err(format!("parameter {k} is not a number")) };
                Ok((k.clone(), if is_int { P::Int(n as i64) } else { P::Float(n as f32 as f64) }))
            }).collect::<Result<_, String>>()?,
            _ => Vec::new(),
        };
        let messages = match l.get("messages") {
            Some(Json::Arr(a)) => a.iter().map(|m| (s(m, "role").unwrap_or_default(), s(m, "content").unwrap_or_default())).collect(),
            _ => Vec::new(),
        };
        let d = Derived {
            name: s(&v, "name").ok_or("no name")?,
            created_at: s(&v, "created_at").unwrap_or_default(),
            base: Base { name: s(b, "name").unwrap_or_default(), file: s(b, "file").ok_or("no base file")?, sha256: s(b, "sha256").ok_or("no base sha256")?,
                         fork: s(b, "fork").unwrap_or_default(), path: s(b, "path").unwrap_or_default() },
            system: s(l, "system"),
            params,
            stop: list("stop"),
            template_sha256: s(l, "template_sha256"),
            license: list("license"),
            messages,
            requires: s(l, "requires"),
            digest: s(&v, "digest").unwrap_or_default().trim_start_matches("sha256:").to_string(),
        };
        let want = d.compute_digest();
        if d.digest != want || s(l, "base_sha256").as_deref() != Some(d.base.sha256.as_str()) {
            return Err(format!("its digest {} is not its content's ({want}): edited by hand, refused — create it again", d.digest));
        }
        Ok(d)
    }

    pub fn tag(&self) -> String {
        format!("{}:latest", self.name)
    }

    /// The Modelfile `ollama show --modelfile` would print for it (Ollama's `Command.String` quoting), FROM the base's name.
    pub fn modelfile(&self) -> String {
        let mut o = format!("# bankml {}: derived model {} (digest sha256:{}), a layer over the pinned base {} (sha256 {}),\n# verified before every load\n# To build a new Modelfile based on this, replace FROM with:\n# FROM {}\n\nFROM {}\n",
                            crate::VERSION, self.name, self.digest, self.base.name, self.base.sha256, self.tag(), self.base.name);
        if let Some(s) = &self.system {
            o += &format!("SYSTEM {}\n", quote(s));
        }
        for (k, v) in &self.params {
            o += &format!("PARAMETER {k} {}\n", v.json());
        }
        for s in &self.stop {
            o += &format!("PARAMETER stop {}\n", quote(s));
        }
        for l in &self.license {
            o += &format!("LICENSE {}\n", quote(l));
        }
        for (r, m) in &self.messages {
            o += &format!("MESSAGE {r} {}\n", quote(m));
        }
        if let Some(r) = &self.requires {
            o += &format!("REQUIRES {}\n", quote(r));
        }
        o
    }

    /// `/api/show`'s `parameters` text (Ollama's: one `key value` per line, the stops quoted).
    pub fn parameters_text(&self) -> String {
        let mut v: Vec<String> = self.params.iter().map(|(k, p)| format!("{k:<30} {}", p.json())).collect();
        v.extend(self.stop.iter().map(|s| format!("{:<30} {}", "stop", jstr(s))));
        v.join("\n")
    }

    /// Ollama's chat rule: the model's MESSAGEs before the request's; its SYSTEM first unless the request's first
    /// message is a system message.
    fn chat_messages(&self, req: &[Json]) -> Vec<Json> {
        let msg = |r: &str, c: &str| Json::Obj(vec![("role".into(), Json::Str(r.into())), ("content".into(), Json::Str(c.into()))]);
        let mut out: Vec<Json> = self.messages.iter().map(|(r, c)| msg(r, c)).collect();
        out.extend(req.iter().cloned());
        let first_is_system = req.first().and_then(|m| m.get("role")).and_then(Json::as_str) == Some("system");
        if let Some(s) = self.system.as_deref().filter(|s| !s.is_empty()) {
            if !first_is_system {
                out.insert(0, msg("system", s));
            }
        }
        out
    }

    /// An Ollama `/api/chat` or `/api/generate` request with the layer applied: `options` over the model's
    /// parameters, the messages by Ollama's rule; for a (non-raw) generate the conversation it becomes goes in
    /// `bankml_messages` (the system, the MESSAGEs, the prompt as the user turn).
    pub fn apply_ollama(&self, req: &Json, chat: bool) -> Json {
        let Json::Obj(mut f) = req.clone() else { return req.clone() };
        let mut opts: Vec<(String, Json)> = match req.get("options") { Some(Json::Obj(o)) => o.clone(), _ => Vec::new() };
        for (k, v) in self.params.iter().rev() {
            if !opts.iter().any(|(n, _)| n == k) {
                opts.insert(0, (k.clone(), Json::Num(v.num())));
            }
        }
        if !self.stop.is_empty() && !opts.iter().any(|(n, _)| n == "stop") {
            opts.push(("stop".into(), Json::Arr(self.stop.iter().map(|s| Json::Str(s.clone())).collect())));
        }
        f.retain(|(k, _)| k != "options" && k != "bankml_messages");
        f.push(("options".into(), Json::Obj(opts)));
        if chat {
            if let Some(Json::Arr(m)) = req.get("messages").filter(|m| matches!(m, Json::Arr(a) if !a.is_empty())) {
                let m = self.chat_messages(m);
                f.retain(|(k, _)| k != "messages");
                f.push(("messages".into(), Json::Arr(m)));
            }
        } else if !req.get("raw").and_then(Json::as_bool).unwrap_or(false) {
            let msg = |r: &str, c: &str| Json::Obj(vec![("role".into(), Json::Str(r.into())), ("content".into(), Json::Str(c.into()))]);
            let mut m = Vec::new();
            match req.get("system").and_then(Json::as_str).filter(|s| !s.is_empty()) {
                Some(s) => m.push(msg("system", s)),
                None => {
                    if let Some(s) = self.system.as_deref().filter(|s| !s.is_empty()) {
                        m.push(msg("system", s));
                    }
                }
            }
            m.extend(self.messages.iter().map(|(r, c)| msg(r, c)));
            m.push(msg("user", req.get("prompt").and_then(Json::as_str).unwrap_or("")));
            f.push(("bankml_messages".into(), Json::Arr(m)));
        }
        Json::Obj(f)
    }

    /// An OpenAI `/v1/chat/completions` request with the layer applied: the messages by Ollama's chat rule (its
    /// OpenAI endpoint goes through the same handler), the parameters as defaults of the top-level fields.
    pub fn apply_openai(&self, req: &Json) -> Json {
        let Json::Obj(mut f) = req.clone() else { return req.clone() };
        let has = |f: &[(String, Json)], k: &str| f.iter().any(|(n, v)| n == k && *v != Json::Null);
        for (k, v) in &self.params {
            match k.as_str() {
                "num_ctx" => {}
                "num_predict" => {
                    if v.num() >= 0.0 && !has(&f, "max_tokens") && !has(&f, "n_predict") {
                        f.push(("max_tokens".into(), Json::Num(v.num())));
                    }
                }
                k if !has(&f, k) => f.push((k.to_string(), Json::Num(v.num()))),
                _ => {}
            }
        }
        if !self.stop.is_empty() && !has(&f, "stop") {
            f.push(("stop".into(), Json::Arr(self.stop.iter().map(|s| Json::Str(s.clone())).collect())));
        }
        if let Some(Json::Arr(m)) = req.get("messages") {
            let m = self.chat_messages(m);
            f.retain(|(k, _)| k != "messages");
            f.push(("messages".into(), Json::Arr(m)));
        }
        Json::Obj(f)
    }

    /// The layer's `num_ctx`, if any (checked against what the server holds).
    pub fn num_ctx(&self) -> Option<i64> {
        self.params.iter().find(|(k, _)| k == "num_ctx").map(|(_, v)| v.num() as i64)
    }
}

/// JSON text of a value (the same escaping as `jstr`).
pub fn to_json(v: &Json) -> String {
    match v {
        Json::Null => "null".into(),
        Json::Bool(b) => b.to_string(),
        Json::Num(n) => fmt_f(*n),
        Json::Str(s) => jstr(s),
        Json::Arr(a) => format!("[{}]", a.iter().map(to_json).collect::<Vec<_>>().join(", ")),
        Json::Obj(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}: {}", jstr(k), to_json(v))).collect::<Vec<_>>().join(", ")),
    }
}

/// A model name as bankML keeps it: lower-case, `:latest` dropped; letters, digits, `.`, `_`, `-`.
pub fn check_name(n: &str) -> Result<String, String> {
    let n = n.trim();
    let n = n.strip_suffix(":latest").unwrap_or(n);
    if n.contains(':') {
        return Err(format!("model name {n:?}: bankML names have no tag but latest"));
    }
    let l = n.to_lowercase();
    if l.is_empty() || l.len() > 128 || !l.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) || l.starts_with(['.', '-']) || l.ends_with(".gguf") {
        return Err(format!("model name {n:?}: use letters, digits, '.', '_' and '-' (at most 128, not ending in .gguf)"));
    }
    Ok(l)
}

// ---------------------------------------------------------------- the store ------------------------------------

/// The derived models of a registry directory, with the entry each one loads through (its base's).
#[derive(Default)]
pub struct Store {
    dir: RwLock<Option<PathBuf>>,
    models: RwLock<Vec<(Arc<Derived>, Entry)>>,
}

fn manifest_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.MODEL.json"))
}

/// Every GGUF pinned in `dir` (each `*.FORK.json`'s `.gguf` files, looked for in `dir` and the `look` directories).
pub fn pinned_entries(dir: &Path, look: &[PathBuf]) -> Vec<Entry> {
    let mut forks: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(".FORK.json")).collect();
    forks.sort();
    let mut out: Vec<Entry> = Vec::new();
    for f in forks {
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        let Some(Json::Arr(files)) = Json::parse(&text).and_then(|v| v.get("files").cloned()) else { continue };
        for p in files.iter().filter_map(|x| x.get("path").and_then(Json::as_str)) {
            let base = p.rsplit('/').next().unwrap_or(p);
            if !base.to_ascii_lowercase().ends_with(".gguf") || out.iter().any(|e| e.name == crate::native::model_name(base)) {
                continue;
            }
            let found = std::iter::once(dir.to_path_buf()).chain(look.iter().cloned()).map(|d| d.join(base)).find(|c| c.is_file());
            out.push(entry_for(base, found, &text));
        }
    }
    out
}

fn entry_for(file: &str, path: Option<PathBuf>, fork_json: &str) -> Entry {
    let sha256 = crate::sha256::pinned_sha256(fork_json, file).unwrap_or_default();
    let bytes = path.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
    let info = match &path {
        Some(p) => crate::native::header_info(p),
        None => crate::native::Info { arch: None, quant: String::new(), params: String::new(), native: Err("pinned, but the file is not on this machine".into()), defaults: None },
    };
    Entry { name: crate::native::model_name(file), file: file.to_string(), path, fork_json: fork_json.to_string(), sha256, bytes, info }
}

/// The FORK.json in `dir` that pins `file`, and its text.
fn fork_for(dir: &Path, file: &str) -> Option<(String, String)> {
    let mut forks: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(".FORK.json")).collect();
    forks.sort();
    forks.into_iter().find_map(|f| {
        let t = std::fs::read_to_string(&f).ok()?;
        crate::sha256::pinned_sha256(&t, file)?;
        Some((f.file_name()?.to_string_lossy().into_owned(), t))
    })
}

impl Store {
    /// Reads every `*.MODEL.json` of `dir`; a manifest that does not verify is skipped, and said why.
    pub fn open(&self, dir: Option<&Path>, reg: &Registry) -> Vec<String> {
        *self.dir.write().unwrap_or_else(|e| e.into_inner()) = dir.map(Path::to_path_buf);
        let mut errs = Vec::new();
        let mut models = Vec::new();
        if let Some(dir) = dir {
            let mut files: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(".MODEL.json")).collect();
            files.sort();
            for f in files {
                match std::fs::read_to_string(&f).map_err(|e| e.to_string()).and_then(|t| Derived::from_json(&t)).and_then(|d| Self::bind(dir, reg, d)) {
                    Ok(x) => models.push(x),
                    Err(e) => errs.push(format!("{}: {e}", f.display())),
                }
            }
        }
        *self.models.write().unwrap_or_else(|e| e.into_inner()) = models;
        errs
    }

    /// The entry a derived model loads through: its base as the registry has it, else as the manifest recorded it.
    fn bind(dir: &Path, reg: &Registry, d: Derived) -> Result<(Arc<Derived>, Entry), String> {
        let e = match reg.entries.iter().find(|e| e.sha256 == d.base.sha256 && e.path.is_some()) {
            Some(e) => e.clone(),
            None => {
                let fork = std::fs::read_to_string(dir.join(&d.base.fork)).map_err(|e| format!("its base's pin {}: {e}", d.base.fork))?;
                let p = PathBuf::from(&d.base.path);
                entry_for(&d.base.file, p.is_file().then_some(p), &fork)
            }
        };
        if e.sha256 != d.base.sha256 {
            return Err(format!("its base {} was {} when it was created; the pin now says {:?} — create it again", d.base.file, d.base.sha256, e.sha256));
        }
        Ok((Arc::new(d), e))
    }

    pub fn dir(&self) -> Option<PathBuf> {
        self.dir.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn list(&self) -> Vec<(Arc<Derived>, Entry)> {
        self.models.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn find(&self, name: &str) -> Option<(Arc<Derived>, Entry)> {
        let n = check_name(name).ok()?;
        self.models.read().unwrap_or_else(|e| e.into_inner()).iter().find(|(d, _)| d.name == n).cloned()
    }

    /// Writes the manifest (atomically) and lists it.
    pub fn insert(&self, d: Derived, e: Entry) -> Result<(), String> {
        let dir = self.dir().ok_or("no registry directory")?;
        let p = manifest_path(&dir, &d.name);
        let tmp = p.with_extension("json.part");
        std::fs::write(&tmp, d.to_json()).and_then(|_| std::fs::rename(&tmp, &p)).map_err(|err| format!("{}: {err}", p.display()))?;
        let mut m = self.models.write().unwrap_or_else(|e| e.into_inner());
        m.retain(|(x, _)| x.name != d.name);
        m.push((Arc::new(d), e));
        Ok(())
    }

    pub fn remove(&self, name: &str) -> Result<(), String> {
        let dir = self.dir().ok_or("no registry directory")?;
        let n = check_name(name)?;
        let p = manifest_path(&dir, &n);
        std::fs::remove_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        self.models.write().unwrap_or_else(|e| e.into_inner()).retain(|(d, _)| d.name != n);
        Ok(())
    }
}

/// A request's `model` to the entry it loads and the layer over it: a pinned model (no layer) or a derived one.
pub fn resolve(rs: &Residency, model: Option<&str>) -> Result<(Entry, Option<Arc<Derived>>), String> {
    // 0.3.5: a pin's exact name first, then a derived model's, then the registry's suffix-less alias — so the derived
    // `mindx-gen39` promote.py layers in place (FROM mindx-gen39, the alias of mindx-gen39-f16) answers as itself
    if let Some(m) = model.map(str::trim).filter(|m| !m.is_empty()) {
        let n = crate::native::model_name(m.strip_suffix(":latest").unwrap_or(m));
        if !rs.reg.entries.iter().any(|e| e.name == n) {
            if let Some((d, e)) = rs.derived.find(m) {
                return Ok((e, Some(d)));
            }
        }
    }
    match rs.reg.resolve(model) {
        Ok(e) => Ok((e.clone(), None)),
        Err(why) => match model.and_then(|m| rs.derived.find(m)) {
            Some((d, e)) => Ok((e, Some(d))),
            None => {
                let derived: Vec<String> = rs.derived.list().iter().map(|(d, _)| d.name.clone()).collect();
                Err(if derived.is_empty() { why } else { format!("{why}; derived: {}", derived.join(", ")) })
            }
        },
    }
}

// ---------------------------------------------------------------- create ---------------------------------------

/// What a create did, step by step (Ollama's status lines).
pub struct Created {
    pub derived: Derived,
    pub entry: Entry,
    pub status: Vec<String>,
}

/// Where the base is: a registry name, a pinned GGUF path, or a safetensors directory (converted and pinned here).
fn base_for(spec_from: &str, name: &str, dir: &Path, reg_entries: &[Entry], store: &Store, status: &mut Vec<String>) -> Result<(Base, Entry, Option<Arc<Derived>>), String> {
    let p = Path::new(spec_from);
    let is_path = spec_from.contains('/') || spec_from.starts_with('.') || spec_from.to_ascii_lowercase().ends_with(".gguf");
    let (path, fork_name, fork_text, parent) = if is_path && p.is_dir() {
        // a safetensors directory: converted to GGUF F16 beside the pins, and pinned by what was written
        let file = format!("{name}-F16.gguf");
        let out = dir.join(&file);
        if out.exists() || dir.join(format!("{file}.FORK.json")).exists() {
            // 0.3.5: an existing pin (its FORK.json, wherever the file is) is never overwritten either
            return Err(format!("FROM {spec_from}: {file} is already pinned in {} (a conversion is pinned once); FROM it by name, {}",
                               dir.display(), crate::native::model_name(&file)));
        }
        status.push(format!("converting {} to GGUF F16 (llama.cpp b11192's convert_hf_to_gguf.py --outtype f16, byte for byte)", p.display()));
        let r = crate::convert::convert(p, &out, &crate::convert::Options::default())?;
        let src = p.canonicalize().map(|c| c.display().to_string()).unwrap_or(spec_from.to_string());
        let fork_name = format!("{file}.FORK.json");
        let text = crate::convert::fork_json(&r, &src);
        std::fs::write(dir.join(&fork_name), &text).map_err(|e| format!("{fork_name}: {e}"))?;
        status.push(format!("pinned {file} sha256:{} in {fork_name}", r.sha256));
        (out, fork_name, text, None)
    } else if is_path {
        let path = p.canonicalize().map_err(|e| format!("FROM {spec_from}: {e}"))?;
        let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let (fork_name, text) = fork_for(dir, &file).ok_or(format!("FROM {spec_from}: {file} is not pinned in {} — bankML layers only on pinned files \
                                                                   (import it with its published sha256, or convert it here: FROM its safetensors directory)", dir.display()))?;
        (path, fork_name, text, None)
    } else if let Some(e) = reg_entries.iter().find(|e| e.name == crate::native::model_name(spec_from.strip_suffix(":latest").unwrap_or(spec_from))) {
        let path = e.path.clone().ok_or(format!("FROM {spec_from}: {} is pinned but not on this machine", e.file))?;
        let (fork_name, text) = fork_for(dir, &e.file).unwrap_or((String::new(), e.fork_json.clone()));
        (path, fork_name, text, None)
    } else if let Some((d, e)) = store.find(spec_from) {
        let path = e.path.clone().ok_or(format!("FROM {spec_from}: its base {} is not on this machine", e.file))?;
        (path, d.base.fork.clone(), e.fork_json.clone(), Some(d))
    } else if let Some(e) = {
        // 0.3.5: the registry's suffix-less alias (`mindx-gen39` for the one pin mindx-gen39-f16), as requests resolve it
        let n = crate::native::model_name(spec_from.strip_suffix(":latest").unwrap_or(spec_from));
        let hits: Vec<&Entry> = reg_entries.iter().filter(|e| crate::native::base_name(&e.name) == n).collect();
        if hits.len() == 1 { Some(hits[0]) } else { None }
    } {
        let path = e.path.clone().ok_or(format!("FROM {spec_from}: {} is pinned but not on this machine", e.file))?;
        let (fork_name, text) = fork_for(dir, &e.file).unwrap_or((String::new(), e.fork_json.clone()));
        (path, fork_name, text, None)
    } else {
        return Err(format!("FROM {spec_from}: no such model — a registry name, a pinned GGUF path or a safetensors directory"));
    };
    if fork_name.is_empty() {
        return Err(format!("FROM {spec_from}: its FORK.json is not in the registry directory {}; a derived model records its base's pin there", dir.display()));
    }
    let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    status.push(format!("verifying {file} (the guard, then its sha256 against {fork_name})"));
    let v = crate::verify(&path, &fork_text, crate::gguf::Engine::Mainline).map_err(|e| format!("FROM {spec_from}: refuse: {e}"))?;
    status.push(format!("using existing layer sha256:{}", v.model_sha256));
    let entry = entry_for(&file, Some(path.clone()), &fork_text);
    let base = Base { name: crate::native::model_name(&file), file, sha256: v.model_sha256, fork: fork_name, path: path.display().to_string() };
    Ok((base, entry, parent))
}

/// `bankml create` and `POST /api/create`: the spec checked, the base verified, the layer merged over its parent's
/// (a FROM naming a derived model inherits its layer, Ollama's rule), the manifest written.
pub fn create(name: &str, spec: &Spec, dir: &Path, reg_entries: &[Entry], store: &Store) -> Result<Created, String> {
    let name = check_name(name)?;
    if reg_entries.iter().any(|e| e.name == name) {
        return Err(format!("{name} is a pinned model's name (its file's); a derived model needs its own — e.g. {name}-persona"));
    }
    let mut status = vec!["parsing modelfile".to_string()];
    let (base, entry, parent) = base_for(&spec.from, &name, dir, reg_entries, store, &mut status)?;
    // the template: only the base's own
    let base_template = entry.path.as_ref().and_then(|p| crate::gguf::guard_file(p, crate::gguf::Engine::Mainline).ok())
        .and_then(|r| r.header).and_then(|h| match h.kv.get("tokenizer.chat_template") { Some(crate::gguf::Val::S(t)) => Some(t.clone()), _ => None });
    let mut template_sha256 = parent.as_ref().and_then(|p| p.template_sha256.clone());
    if let Some(t) = &spec.template {
        if base_template.as_deref() != Some(t.as_str()) {
            return Err("TEMPLATE: bankML renders the base's own chat template (byte-identical to llama.cpp's rendering of the pinned GGUF); \
                        a different template — Ollama's are Go templates — would change every prompt and is not reproduced. Leave TEMPLATE out".into());
        }
        let mut h = crate::sha256::Sha256::default();
        h.update(t.as_bytes());
        template_sha256 = Some(crate::sha256::hex(&h.finish()));
    }
    // the layer: the parent's, then this spec's over it
    let mut params = parent.as_ref().map(|p| p.params.clone()).unwrap_or_default();
    for (k, v) in &spec.params {
        match params.iter_mut().find(|(n, _)| n == k) {
            Some(e) => e.1 = v.clone(),
            None => params.push((k.clone(), v.clone())),
        }
    }
    params.sort_by_key(|(k, _)| PARAMS.iter().position(|p| p.0 == k));
    let mut license = parent.as_ref().map(|p| p.license.clone()).unwrap_or_default();
    for l in &spec.license {
        if !license.contains(l) {
            license.push(l.clone());
        }
    }
    let mut d = Derived {
        name: name.clone(),
        created_at: crate::ollama::rfc3339(std::time::SystemTime::now()),
        base,
        system: spec.system.clone().or_else(|| parent.as_ref().and_then(|p| p.system.clone())),
        params,
        stop: spec.stop.clone().or_else(|| parent.as_ref().map(|p| p.stop.clone())).unwrap_or_default(),
        template_sha256,
        license,
        messages: spec.messages.clone().or_else(|| parent.as_ref().map(|p| p.messages.clone())).unwrap_or_default(),
        requires: spec.requires.clone().or_else(|| parent.as_ref().and_then(|p| p.requires.clone())),
        digest: String::new(),
    };
    for (k, v) in &d.params {
        let bad = match k.as_str() {
            "num_ctx" => v.num() < 1.0,
            "num_predict" => v.num() < -2.0,
            "top_k" | "seed" => false,
            // llama.cpp takes any finite frequency or presence penalty, negative ones too; a repeat penalty must be > 0
            "frequency_penalty" | "presence_penalty" => !v.num().is_finite(),
            "repeat_penalty" => v.num() <= 0.0,
            _ => v.num() < 0.0,
        };
        if bad {
            return Err(format!("PARAMETER {k} {}: out of range", v.json()));
        }
    }
    if d.system.is_some() {
        status.push("creating system layer".into());
    }
    if !d.params.is_empty() || !d.stop.is_empty() {
        status.push("creating parameters layer".into());
    }
    if !d.license.is_empty() {
        status.push("creating license layer".into());
    }
    if !d.messages.is_empty() {
        status.push("creating messages layer".into());
    }
    d.digest = d.compute_digest();
    status.push(format!("writing manifest {name}.MODEL.json (digest sha256:{})", d.digest));
    store.insert(d.clone(), entry.clone())?;
    status.push("success".into());
    Ok(Created { derived: d, entry, status })
}

// ---------------------------------------------------------------- HTTP -----------------------------------------

fn err(c: &mut TcpStream, code: u16, msg: &str) -> std::io::Result<()> {
    respond(c, code, "application/json", format!("{{\"error\": {}}}", jstr(msg)).as_bytes())
}

/// `POST /api/create`: `{model, modelfile}` (a Modelfile's text) or Ollama's structured form
/// `{model, from, system, template, license, parameters, messages}`; streamed status lines (NDJSON) unless `stream: false`.
pub fn http_create(c: &mut TcpStream, rs: &Residency, body: &[u8]) -> std::io::Result<()> {
    let Some(req) = Json::parse(&String::from_utf8_lossy(body)) else { return err(c, 400, "body is not JSON") };
    let Some(name) = req.get("model").or(req.get("name")).and_then(Json::as_str) else { return err(c, 400, "model: name the model to create") };
    let Some(dir) = rs.derived.dir() else {
        return err(c, 400, "bankml serve was started without --registry: a derived model is written beside the pins of the registry directory");
    };
    let spec = match req.get("modelfile").and_then(Json::as_str).filter(|m| !m.trim().is_empty()) {
        Some(m) => Spec::from_modelfile(m),
        None => Spec::from_request(&req),
    };
    let spec = match spec {
        Ok(s) => s,
        Err(m) => return err(c, 400, &m),
    };
    // a create rewrites what a request may resolve: it waits for the engine like a load does
    let run = rs.lock_run();
    let r = create(name, &spec, &dir, &rs.reg.entries, &rs.derived);
    drop(run);
    let created = match r {
        Ok(x) => x,
        Err(m) => return err(c, 400, &m),
    };
    if let Some(n) = created.derived.num_ctx().filter(|n| *n as usize > rs.n_ctx) {
        crate::log(crate::LOG_WARN, &format!("bankml create: {} asks num_ctx {n}; this server holds {} — requests will be refused until serve runs with --ctx {n}", created.derived.name, rs.n_ctx));
    }
    if req.get("stream").and_then(Json::as_bool).unwrap_or(true) {
        let lines: String = created.status.iter().map(|s| format!("{{\"status\": {}}}\n", jstr(s))).collect();
        respond(c, 200, "application/x-ndjson", lines.as_bytes())
    } else {
        respond(c, 200, "application/json", b"{\"status\": \"success\"}")
    }
}

/// `DELETE /api/delete`: a derived model only; a pinned model is refused with the reason.
pub fn http_delete(c: &mut TcpStream, rs: &Residency, body: &[u8]) -> std::io::Result<()> {
    let Some(req) = Json::parse(&String::from_utf8_lossy(body)) else { return err(c, 400, "body is not JSON") };
    let Some(name) = req.get("model").or(req.get("name")).and_then(Json::as_str) else { return err(c, 400, "model: name the model to delete") };
    if name.trim().is_empty() {
        return err(c, 400, "model: name the model to delete");
    }
    if let Ok(e) = rs.reg.resolve(Some(name)) {
        return err(c, 400, &format!("{} is a pinned model (its FORK.json pins {}): bankml does not delete pins over HTTP; remove the file and its FORK.json by hand", e.name, e.file));
    }
    if rs.derived.find(name).is_none() {
        return err(c, 404, &format!("model '{name}' not found"));
    }
    let run = rs.lock_run();
    let r = rs.derived.remove(name);
    drop(run);
    match r {
        Ok(()) => respond(c, 200, "application/json", b""),
        Err(m) => err(c, 500, &m),
    }
}

/// `POST /api/copy`: a derived model under a second name (the same content and digest); a pinned model's copy is a
/// derived model with an empty layer over it.
pub fn http_copy(c: &mut TcpStream, rs: &Residency, body: &[u8]) -> std::io::Result<()> {
    let Some(req) = Json::parse(&String::from_utf8_lossy(body)) else { return err(c, 400, "body is not JSON") };
    let (Some(src), Some(dst)) = (req.get("source").and_then(Json::as_str), req.get("destination").and_then(Json::as_str)) else {
        return err(c, 400, "source and destination: name both");
    };
    let Some(dir) = rs.derived.dir() else { return err(c, 400, "bankml serve was started without --registry: nowhere to write the copy") };
    let spec = Spec { from: src.to_string(), ..Spec::default() };
    let run = rs.lock_run();
    let r = create(dst, &spec, &dir, &rs.reg.entries, &rs.derived);
    drop(run);
    match r {
        Ok(_) => respond(c, 200, "application/json", b""),
        Err(m) => err(c, 400, &m),
    }
}

/// `/api/show` of a derived model: the base's own answer (`base_show`), with the derived model's Modelfile,
/// parameters, system, licence, messages and `details.parent_model`.
pub fn show_json(d: &Derived, base_show: &str) -> String {
    let Some(Json::Obj(mut f)) = Json::parse(base_show) else { return base_show.to_string() };
    let set = |f: &mut Vec<(String, Json)>, k: &str, v: Json| match f.iter_mut().find(|(n, _)| n == k) {
        Some(e) => e.1 = v,
        None => f.push((k.to_string(), v)),
    };
    set(&mut f, "modelfile", Json::Str(d.modelfile()));
    set(&mut f, "parameters", Json::Str(d.parameters_text()));
    if let Some(s) = &d.system {
        set(&mut f, "system", Json::Str(s.clone()));
    }
    if !d.license.is_empty() {
        set(&mut f, "license", Json::Str(d.license.join("\n")));
    }
    if !d.messages.is_empty() {
        set(&mut f, "messages", Json::Arr(d.messages.iter().map(|(r, c)| Json::Obj(vec![("role".into(), Json::Str(r.clone())), ("content".into(), Json::Str(c.clone()))])).collect()));
    }
    set(&mut f, "modified_at", Json::Str(d.created_at.clone()));
    if let Some((_, Json::Obj(det))) = f.iter_mut().find(|(n, _)| n == "details") {
        match det.iter_mut().find(|(n, _)| n == "parent_model") {
            Some(e) => e.1 = Json::Str(format!("{}:latest", d.base.name)),
            None => det.insert(0, ("parent_model".into(), Json::Str(format!("{}:latest", d.base.name)))),
        }
    }
    let b = Json::Obj(vec![("derived".into(), Json::Bool(true)), ("digest".into(), Json::Str(format!("sha256:{}", d.digest))),
                           ("base_sha256".into(), Json::Str(d.base.sha256.clone())), ("base_fork".into(), Json::Str(d.base.fork.clone()))]);
    if let Some((_, Json::Obj(bm))) = f.iter_mut().find(|(n, _)| n == "bankml") {
        bm.push(("layer".into(), b));
    } else {
        f.push(("bankml".into(), b));
    }
    to_json(&Json::Obj(f))
}

/// The derived models' `/api/tags` objects (`details` from the base's, `parent_model` set).
pub fn tag_objects(rs: &Residency, details: impl Fn(&Entry) -> String, native: impl Fn(&Entry) -> String) -> Vec<String> {
    rs.derived.list().iter().map(|(d, e)| {
        let det = details(e).replacen("\"parent_model\": \"\"", &format!("\"parent_model\": {}", jstr(&format!("{}:latest", d.base.name))), 1);
        format!("{{\"name\": {t}, \"model\": {t}, \"modified_at\": {}, \"size\": {}, \"digest\": \"sha256:{}\", \"details\": {det}, \"bankml\": {}}}",
                jstr(&d.created_at), e.bytes, d.digest, native(e), t = jstr(&d.tag()))
    }).collect()
}

// ---------------------------------------------------------------- the CLI --------------------------------------

/// `bankml create NAME -f Modelfile [--registry DIR] [--models DIR]`: Ollama's status lines on stdout.
pub fn cli(name: &str, modelfile: &str, dir: &Path, models: &[PathBuf]) -> Result<Derived, String> {
    let text = std::fs::read_to_string(modelfile).map_err(|e| format!("{modelfile}: {e}"))?;
    let spec = Spec::from_modelfile(&text)?;
    // a relative FROM path is relative to the Modelfile, as in Ollama
    let spec = if spec.from.starts_with("./") || spec.from.starts_with("../") {
        let base = Path::new(modelfile).parent().unwrap_or(Path::new("."));
        Spec { from: base.join(&spec.from).display().to_string(), ..spec }
    } else {
        spec
    };
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let reg = Registry { entries: pinned_entries(dir, models), default: 0 };
    let store = Store::default();
    for e in store.open(Some(dir), &reg) {
        eprintln!("bankml create: skipping {e}");
    }
    let c = create(name, &spec, dir, &reg.entries, &store)?;
    let mut out = std::io::stdout();
    for s in &c.status {
        let _ = writeln!(out, "{s}");
    }
    Ok(c.derived)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmds(t: &str) -> Vec<(String, String)> {
        parse_modelfile(t).unwrap().into_iter().map(|c| (c.name, c.args)).collect()
    }
    fn s(a: &str, b: &str) -> (String, String) {
        (a.to_string(), b.to_string())
    }

    #[test]
    fn modelfile_parses_as_ollama() {
        // the Modelfiles mindX writes today (PYTHAI/mindXascension weights/gen39, promote.py's persona layer)
        assert_eq!(cmds("FROM /home/mindx/x/ollama_push/merged\n"), vec![s("model", "/home/mindx/x/ollama_push/merged")]);
        assert_eq!(cmds("FROM qwen3:0.6b\nADAPTER /a/checkpoint\nSYSTEM \"\"\"You are mindX generation 39, fine-tuned on your own consolidated dream wisdom (knowledge->wisdom->weights).\"\"\"\n"),
                   vec![s("model", "qwen3:0.6b"), s("adapter", "/a/checkpoint"), s("system", "You are mindX generation 39, fine-tuned on your own consolidated dream wisdom (knowledge->wisdom->weights).")]);
        assert_eq!(cmds("FROM mindx-gen39\nSYSTEM \"\"\"line one\nline \"two\"\n\"\"\"\nPARAMETER stop <|im_end|>\nPARAMETER num_ctx 2048\n"),
                   vec![s("model", "mindx-gen39"), s("system", "line one\nline \"two\"\n"), s("stop", "<|im_end|>"), s("num_ctx", "2048")]);
        // case-insensitive instructions, comments, blank lines, CRLF, a BOM, tabs
        assert_eq!(cmds("\u{feff}# a comment\n\nfrom  base\r\nsystem\tbe brief  \r\nPaRaMeTeR temperature 0.2\n"),
                   vec![s("model", "base"), s("system", "be brief"), s("temperature", "0.2")]);
        // a single-quoted value may span lines; quotes inside an unquoted value stay
        assert_eq!(cmds("SYSTEM \"a\nb\"\nSYSTEM say \"hi\" now\n"), vec![s("system", "a\nb"), s("system", "say \"hi\" now")]);
        // a # after a value is part of it (comments only open a line)
        assert_eq!(cmds("FROM x # not a comment"), vec![s("model", "x # not a comment")]);
        // messages carry their role
        assert_eq!(cmds("MESSAGE user What is your purpose?\nMESSAGE assistant \"\"\"I am mindX.\"\"\""),
                   vec![s("message", "user: What is your purpose?"), s("message", "assistant: I am mindX.")]);
        // errors: an unknown instruction, a bad role, an unterminated quote, a name with no value
        assert!(parse_modelfile("FORM x").unwrap_err().contains("command must be one of"));
        assert!(parse_modelfile("MESSAGE robot hi").unwrap_err().contains("message role"));
        assert!(parse_modelfile("SYSTEM \"\"\"never closed\n").unwrap_err().contains("unexpected EOF"));
        assert!(parse_modelfile("FROM\nx").is_err());
        assert!(parse_modelfile("PARAMETER\n").is_err());
        // an empty SYSTEM is a value
        assert_eq!(cmds("SYSTEM \"\"\n"), vec![s("system", "")]);
    }

    #[test]
    fn quote_round_trips_through_the_parser() {
        for v in ["plain", "two words", " edge", "line\nbreak", "said \"hi\"\nthen", "<|im_end|>", "tab\there", "multi\n\nparagraph \"q\" end"] {
            let t = format!("SYSTEM {}\n", quote(v));
            assert_eq!(cmds(&t), vec![s("system", v)], "{t:?}");
        }
        assert_eq!(quote("<|im_end|>"), "<|im_end|>");
        assert_eq!(quote("a\nb"), "\"a\nb\"");
        assert_eq!(quote("a\n\"b\""), "\"\"\"a\n\"b\"\"\"\"");
    }

    #[test]
    fn spec_keeps_what_is_reproduced_and_refuses_the_rest() {
        let sp = Spec::from_modelfile("FROM b\nSYSTEM one\nSYSTEM two\nPARAMETER temperature 0.7\nPARAMETER stop <|im_end|>\nPARAMETER stop \"END\"\nPARAMETER temperature 0.5\nPARAMETER num_ctx 2048\nLICENSE MIT\nMESSAGE user hi\n").unwrap();
        assert_eq!(sp.from, "b");
        assert_eq!(sp.system.as_deref(), Some("two"));
        assert_eq!(sp.params, vec![("temperature".into(), P::Float(0.5)), ("num_ctx".into(), P::Int(2048))]);
        assert_eq!(sp.stop, Some(vec!["<|im_end|>".into(), "END".into()]));
        assert_eq!(sp.license, vec!["MIT".to_string()]);
        assert_eq!(sp.messages, Some(vec![("user".into(), "hi".into())]));
        for (mf, why) in [("FROM b\nPARAMETER typical_p 0.9", "not reproduced"), ("FROM b\nPARAMETER num_thread 4", "resource option"),
                          ("FROM b\nPARAMETER warp 9", "not a parameter"), ("FROM b\nPARAMETER num_ctx 2048.0", "not an integer"),
                          ("FROM b\nPARAMETER temperature hot", "not a number"), ("FROM b\nADAPTER /x", "LoRA"), ("SYSTEM x", "no FROM"),
                          ("FROM a\nFROM b", "more than one FROM")] {
            let e = Spec::from_modelfile(mf).unwrap_err();
            assert!(e.contains(why), "{mf:?}: {e}");
        }
        let r = Spec::from_request(&Json::parse(r#"{"model": "m", "from": "b", "system": "s", "parameters": {"temperature": 0.7, "num_ctx": 4096, "stop": ["a", "b"]}, "license": ["L1", "L2"]}"#).unwrap()).unwrap();
        assert_eq!(r.params, vec![("temperature".into(), P::Float(0.7f32 as f64)), ("num_ctx".into(), P::Int(4096))]);
        assert_eq!((r.stop, r.license), (Some(vec!["a".into(), "b".into()]), vec!["L1".to_string(), "L2".to_string()]));
        for (j, why) in [(r#"{"from": "b", "quantize": "q8_0"}"#, "quantize"), (r#"{"from": "b", "adapters": {"a": "sha256:x"}}"#, "adapters"),
                         (r#"{"from": "b", "files": {"m.gguf": "sha256:x"}}"#, "files"), (r#"{"system": "x"}"#, "from")] {
            assert!(Spec::from_request(&Json::parse(j).unwrap()).unwrap_err().contains(why), "{j}");
        }
        assert_eq!(check_name("Mindx-Gen39:latest").unwrap(), "mindx-gen39");
        for bad in ["a:b", "", "x.gguf", "a/b", "-x", "sp ace"] {
            assert!(check_name(bad).is_err(), "{bad}");
        }
    }

    fn derived() -> Derived {
        let mut d = Derived {
            name: "mindx-gen39".into(),
            created_at: "2026-10-02T00:00:00.000000000Z".into(),
            base: Base { name: "mindx-gen39-f16".into(), file: "mindx-gen39-F16.gguf".into(), sha256: "6b".repeat(32), fork: "mindx-gen39-F16.gguf.FORK.json".into(), path: "/m/mindx-gen39-F16.gguf".into() },
            system: Some("You are mindX.\nBe \"precise\".".into()),
            params: vec![("temperature".into(), P::Float(0.7f32 as f64)), ("num_ctx".into(), P::Int(2048))],
            stop: vec!["<|im_end|>".into()],
            template_sha256: None,
            license: vec!["Apache-2.0".into()],
            messages: vec![("user".into(), "who are you?".into()), ("assistant".into(), "mindX.".into())],
            requires: None,
            digest: String::new(),
        };
        d.digest = d.compute_digest();
        d
    }

    #[test]
    fn manifest_digest_is_the_content() {
        let d = derived();
        let back = Derived::from_json(&d.to_json()).unwrap();
        assert_eq!(back, d);
        // the name and the time are not content: a copy has the same digest
        let copy = Derived { name: "other".into(), created_at: "x".into(), ..d.clone() };
        assert_eq!(copy.compute_digest(), d.digest);
        // any change to the layer or the base changes it
        for e in [Derived { system: Some("x".into()), ..d.clone() }, Derived { stop: vec![], ..d.clone() },
                  Derived { base: Base { sha256: "00".repeat(32), ..d.base.clone() }, ..d.clone() }, Derived { params: vec![], ..d.clone() }] {
            assert_ne!(e.compute_digest(), d.digest);
        }
        // a hand-edited manifest is refused
        let edited = d.to_json().replace("Be \\\"precise\\\".", "Be loose.");
        assert!(Derived::from_json(&edited).unwrap_err().contains("edited by hand"));
        // the reconstructed Modelfile creates the same layer (FROM the base by name)
        let sp = Spec::from_modelfile(&d.modelfile()).unwrap();
        assert_eq!((sp.from.as_str(), sp.system.as_ref(), sp.stop.as_ref(), &sp.license), ("mindx-gen39-f16", d.system.as_ref(), Some(&d.stop), &d.license));
        assert_eq!(sp.params, d.params);
        assert_eq!(sp.messages.unwrap(), d.messages);
        assert!(d.parameters_text().contains("stop                           \"<|im_end|>\""));
    }

    #[test]
    fn the_layer_applies_as_ollama_applies_it() {
        let d = derived();
        let roles = |v: &Json, k: &str| -> Vec<String> {
            match v.get(k) { Some(Json::Arr(a)) => a.iter().map(|m| format!("{}:{}", m.get("role").and_then(Json::as_str).unwrap(), m.get("content").and_then(Json::as_str).unwrap())).collect(), _ => vec![] }
        };
        // chat: the system first, then the MESSAGEs, then the request
        let r = d.apply_ollama(&Json::parse(r#"{"model": "mindx-gen39", "messages": [{"role": "user", "content": "hi"}]}"#).unwrap(), true);
        assert_eq!(roles(&r, "messages"), vec!["system:You are mindX.\nBe \"precise\".", "user:who are you?", "assistant:mindX.", "user:hi"]);
        // a request that brings its own system message keeps it, and the model's is not added (the MESSAGEs still are)
        let r = d.apply_ollama(&Json::parse(r#"{"messages": [{"role": "system", "content": "mine"}, {"role": "user", "content": "hi"}]}"#).unwrap(), true);
        assert_eq!(roles(&r, "messages"), vec!["user:who are you?", "assistant:mindX.", "system:mine", "user:hi"]);
        // options: the model's parameters are defaults; the request's win key by key; stop as a whole
        let r = d.apply_ollama(&Json::parse(r#"{"messages": [{"role": "user", "content": "hi"}], "options": {"temperature": 0, "stop": ["X"]}}"#).unwrap(), true);
        let o = r.get("options").unwrap();
        assert_eq!(o.get("temperature"), Some(&Json::Num(0.0)));
        assert_eq!(o.get("num_ctx"), Some(&Json::Num(2048.0)));
        assert_eq!(o.get("stop"), Some(&Json::Arr(vec![Json::Str("X".into())])));
        let r = d.apply_ollama(&Json::parse(r#"{"messages": [{"role": "user", "content": "hi"}]}"#).unwrap(), true);
        assert_eq!(r.get("options").and_then(|o| o.get("stop")), Some(&Json::Arr(vec![Json::Str("<|im_end|>".into())])));
        // generate: the request's system over the model's, the MESSAGEs, then the prompt; raw gets nothing
        let r = d.apply_ollama(&Json::parse(r#"{"prompt": "hi"}"#).unwrap(), false);
        assert_eq!(roles(&r, "bankml_messages"), vec!["system:You are mindX.\nBe \"precise\".", "user:who are you?", "assistant:mindX.", "user:hi"]);
        let r = d.apply_ollama(&Json::parse(r#"{"prompt": "hi", "system": "mine"}"#).unwrap(), false);
        assert_eq!(roles(&r, "bankml_messages")[0], "system:mine");
        assert!(d.apply_ollama(&Json::parse(r#"{"prompt": "hi", "raw": true}"#).unwrap(), false).get("bankml_messages").is_none());
        // OpenAI: the same message rule; parameters fill absent top-level fields
        let r = d.apply_openai(&Json::parse(r#"{"messages": [{"role": "user", "content": "hi"}], "temperature": 1.0}"#).unwrap());
        assert_eq!(roles(&r, "messages").len(), 4);
        assert_eq!(r.get("temperature"), Some(&Json::Num(1.0)));
        assert_eq!(r.get("stop"), Some(&Json::Arr(vec![Json::Str("<|im_end|>".into())])));
        assert!(r.get("num_ctx").is_none());
        let none = Derived { system: None, messages: vec![], ..d };
        let r = none.apply_ollama(&Json::parse(r#"{"messages": [{"role": "user", "content": "hi"}]}"#).unwrap(), true);
        assert_eq!(roles(&r, "messages"), vec!["user:hi"]);
    }

    /// A tiny GGUF the guard plays (Llama header, a chat template, no tensors) and its FORK.json, in `dir`.
    pub(crate) fn pinned_fixture(dir: &Path, file: &str, template: &str) -> String {
        let mut g = b"GGUF".to_vec();
        g.extend(3u32.to_le_bytes());
        g.extend(0u64.to_le_bytes());
        g.extend(2u64.to_le_bytes());
        for (k, v) in [("general.architecture", "llama"), ("tokenizer.chat_template", template)] {
            g.extend((k.len() as u64).to_le_bytes());
            g.extend(k.as_bytes());
            g.extend(8u32.to_le_bytes());
            g.extend((v.len() as u64).to_le_bytes());
            g.extend(v.as_bytes());
        }
        std::fs::write(dir.join(file), &g).unwrap();
        let sha = crate::sha256::file_hex(&dir.join(file)).unwrap();
        std::fs::write(dir.join(format!("{file}.FORK.json")), format!("{{\"files\": [{{\"path\": \"{file}\", \"bytes\": {}, \"sha256\": \"{sha}\"}}]}}", g.len())).unwrap();
        sha
    }

    #[test]
    fn create_layers_over_a_pinned_base() {
        let dir = std::env::temp_dir().join(format!("bankml-create-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let sha = pinned_fixture(&dir, "base-F16.gguf", "{{ chatml }}");
        let reg = Registry { entries: pinned_entries(&dir, &[]), default: 0 };
        assert_eq!(reg.entries.len(), 1);
        let store = Store::default();
        assert!(store.open(Some(&dir), &reg).is_empty());
        // FROM a name, then FROM the derived one (its layer inherited), then the template rule
        let c = create("persona", &Spec::from_modelfile("FROM base-f16\nSYSTEM \"\"\"You are mindX.\"\"\"\nPARAMETER stop <|im_end|>\nPARAMETER num_ctx 2048").unwrap(), &dir, &reg.entries, &store).unwrap();
        assert_eq!(c.derived.base.sha256, sha);
        assert_eq!(c.status.last().map(String::as_str), Some("success"));
        assert!(c.status.iter().any(|s| s == &format!("using existing layer sha256:{sha}")));
        let c2 = create("persona", &Spec::from_modelfile("FROM persona\nPARAMETER temperature 0.2").unwrap(), &dir, &reg.entries, &store).unwrap();
        assert_eq!(c2.derived.system.as_deref(), Some("You are mindX."));
        assert_eq!(c2.derived.params, vec![("temperature".into(), P::Float(0.2f32 as f64)), ("num_ctx".into(), P::Int(2048))]);
        assert_eq!(c2.derived.stop, vec!["<|im_end|>".to_string()]);
        // FROM the pinned file's path
        let c3 = create("by-path", &Spec { from: dir.join("base-F16.gguf").display().to_string(), ..Spec::default() }, &dir, &reg.entries, &store).unwrap();
        assert_eq!(c3.derived.base.name, "base-f16");
        let ok = create("tmpl", &Spec { from: "base-f16".into(), template: Some("{{ chatml }}".into()), ..Spec::default() }, &dir, &reg.entries, &store).unwrap();
        assert!(ok.derived.template_sha256.is_some());
        let e = create("tmpl", &Spec { from: "base-f16".into(), template: Some("{{ .Prompt }}".into()), ..Spec::default() }, &dir, &reg.entries, &store).err().unwrap();
        assert!(e.starts_with("TEMPLATE"), "{e}");
        // refusals: a pinned name, an unpinned file, an unknown base
        assert!(create("base-f16", &Spec { from: "base-f16".into(), ..Spec::default() }, &dir, &reg.entries, &store).err().unwrap().contains("pinned model's name"));
        std::fs::write(dir.join("loose.gguf"), b"GGUF").unwrap();
        assert!(create("x", &Spec { from: dir.join("loose.gguf").display().to_string(), ..Spec::default() }, &dir, &reg.entries, &store).err().unwrap().contains("not pinned"));
        assert!(create("x", &Spec { from: "nothing".into(), ..Spec::default() }, &dir, &reg.entries, &store).err().unwrap().contains("no such model"));
        // a fresh store reads the manifests back; a tampered one is skipped with the reason
        let again = Store::default();
        assert!(again.open(Some(&dir), &reg).is_empty());
        assert_eq!(again.find("persona").unwrap().0.digest, c2.derived.digest);
        let p = dir.join("by-path.MODEL.json");
        std::fs::write(&p, std::fs::read_to_string(&p).unwrap().replace("\"license\": []", "\"license\": [\"x\"]")).unwrap();
        let errs = Store::default().open(Some(&dir), &reg);
        assert!(errs.len() == 1 && errs[0].contains("edited by hand"), "{errs:?}");
        // a base whose pin changed is refused
        let fork = dir.join("base-F16.gguf.FORK.json");
        std::fs::write(&fork, std::fs::read_to_string(&fork).unwrap().replace(&sha, &"0".repeat(64))).unwrap();
        let reg2 = Registry { entries: pinned_entries(&dir, &[]), default: 0 };
        let errs = Store::default().open(Some(&dir), &reg2);
        assert!(errs.iter().any(|e| e.contains("create it again")), "{errs:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// One request through the Ollama router, in process, over a loopback socket pair: (status, body).
    fn call(rs: &Residency, method: &str, path: &str, body: &str) -> (u16, String) {
        use std::io::Read;
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (mut server, _) = l.accept().unwrap();
        crate::ollama::route(&mut server, rs, crate::ollama::KEEP_ALIVE_DEFAULT, method, path, body.as_bytes()).unwrap();
        drop(server);
        let mut s = String::new();
        client.read_to_string(&mut s).unwrap();
        let code = s.split(' ').nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
        (code, s.split_once("\r\n\r\n").map(|x| x.1.to_string()).unwrap_or_default())
    }

    /// create → tags → show → copy → delete through the HTTP router, no model loaded (the base is a header-only GGUF).
    #[test]
    fn http_create_show_copy_delete_round_trip() {
        use std::sync::Mutex;
        let dir = std::env::temp_dir().join(format!("bankml-http-create-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let sha = pinned_fixture(&dir, "base-F16.gguf", "{{ chatml }}");
        let reg = Registry { entries: pinned_entries(&dir, &[]), default: 0 };
        let rs = Residency { reg, n_ctx: 4096, engine: crate::gguf::Engine::Mainline, run: Mutex::new(()), cur: Mutex::new(None), last: Mutex::new(None), derived: Store::default(), shown: Mutex::new(None) };
        // without a registry directory there is nowhere to write
        let (code, body) = call(&rs, "POST", "/api/create", r#"{"model": "persona", "modelfile": "FROM base-f16"}"#);
        assert!(code == 400 && body.contains("--registry"), "{code} {body}");
        assert!(rs.derived.open(Some(&dir), &rs.reg).is_empty());
        // a Modelfile, streamed status lines
        let (code, body) = call(&rs, "POST", "/api/create", r#"{"model": "persona", "modelfile": "FROM base-f16\nSYSTEM \"\"\"You are mindX.\"\"\"\nPARAMETER stop <|im_end|>\nPARAMETER temperature 0.7\n"}"#);
        assert_eq!(code, 200, "{body}");
        let lines: Vec<Json> = body.lines().map(|l| Json::parse(l).unwrap()).collect();
        assert_eq!(lines.last().and_then(|l| l.get("status")).and_then(Json::as_str), Some("success"));
        assert!(lines.iter().any(|l| l.get("status").and_then(Json::as_str) == Some(&format!("using existing layer sha256:{sha}"))));
        // listed, with its parent
        let (code, body) = call(&rs, "GET", "/api/tags", "");
        assert_eq!(code, 200);
        let tags = Json::parse(&body).unwrap();
        let Some(Json::Arr(models)) = tags.get("models") else { panic!("{body}") };
        let p = models.iter().find(|m| m.get("name").and_then(Json::as_str) == Some("persona:latest")).expect("persona listed");
        assert_eq!(p.get("details").and_then(|d| d.get("parent_model")).and_then(Json::as_str), Some("base-f16:latest"));
        assert!(p.get("digest").and_then(Json::as_str).unwrap().starts_with("sha256:"));
        // shown: the Modelfile reconstructed from the manifest
        let (code, body) = call(&rs, "POST", "/api/show", r#"{"model": "persona:latest"}"#);
        assert_eq!(code, 200, "{body}");
        let show = Json::parse(&body).unwrap();
        let mf = show.get("modelfile").and_then(Json::as_str).unwrap();
        assert!(mf.contains("\nFROM base-f16\nSYSTEM You are mindX.\nPARAMETER temperature 0.7\nPARAMETER stop <|im_end|>\n"), "{mf}");
        assert_eq!(show.get("system").and_then(Json::as_str), Some("You are mindX."));
        assert_eq!(show.get("details").and_then(|d| d.get("parent_model")).and_then(Json::as_str), Some("base-f16:latest"));
        // the structured form, FROM the derived model (its layer inherited), not streamed
        let (code, body) = call(&rs, "POST", "/api/create", r#"{"model": "p2", "from": "persona", "parameters": {"num_ctx": 2048}, "stream": false}"#);
        assert_eq!((code, body.as_str()), (200, r#"{"status": "success"}"#));
        let p2 = rs.derived.find("p2").unwrap().0;
        assert_eq!((p2.system.as_deref(), p2.stop.clone()), (Some("You are mindX."), vec!["<|im_end|>".to_string()]));
        // copy: the same content, the same digest
        let (code, _) = call(&rs, "POST", "/api/copy", r#"{"source": "p2", "destination": "p3"}"#);
        assert_eq!(code, 200);
        assert_eq!(rs.derived.find("p3").unwrap().0.digest, p2.digest);
        // the refusals, with their reasons
        for (body, why) in [(r#"{"model": "x", "modelfile": "FROM base-f16\nADAPTER /lora"}"#, "LoRA"),
                            (r#"{"model": "x", "modelfile": "FROM base-f16\nPARAMETER typical_p 0.9"}"#, "not reproduced"),
                            (r#"{"model": "x", "modelfile": "FROM base-f16\nTEMPLATE {{ .Prompt }}"}"#, "TEMPLATE"),
                            (r#"{"model": "base-f16", "from": "base-f16"}"#, "pinned model's name"),
                            (r#"{"model": "x", "from": "base-f16", "quantize": "q4_K_M"}"#, "quantize")] {
            let (code, b) = call(&rs, "POST", "/api/create", body);
            assert!(code == 400 && b.contains(why), "{body}: {code} {b}");
        }
        // delete: derived only
        let (code, body) = call(&rs, "DELETE", "/api/delete", r#"{"model": "base-f16"}"#);
        assert!(code == 400 && body.contains("pinned model"), "{body}");
        assert_eq!(call(&rs, "DELETE", "/api/delete", r#"{"model": "p3"}"#).0, 200);
        assert_eq!(call(&rs, "DELETE", "/api/delete", r#"{"model": "p3"}"#).0, 404);
        assert!(!dir.join("p3.MODEL.json").exists());
        // a chat to a derived model resolves through its base (here a header-only file the engine does not play)
        let (code, body) = call(&rs, "POST", "/api/chat", r#"{"model": "persona", "messages": [{"role": "user", "content": "hi"}], "stream": false}"#);
        assert!(code == 400 && body.contains("does not play it"), "{code} {body}");
        assert_eq!(call(&rs, "POST", "/api/chat", r#"{"model": "nothing", "messages": [{"role": "user", "content": "hi"}]}"#).0, 404);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// 0.3.5 (O5 end to end): mindx-gen39 created with promote.py's persona layer (testing/persona_oracle.py: FROM the
    /// merged safetensors directory, and promote.py's own Modelfile FROM mindx-gen39 in place), asked the user's turns
    /// alone, must give llama-server b11192's tokens for the same GGUF given the persona as the system message.
    #[test]
    #[ignore = "needs .models/oracle-persona (testing/persona_oracle.py --record); --release"]
    fn oracle_persona_layer() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/oracle-persona");
        let eng = crate::native::Native::open(&dir.join("registry-dir/mindx-gen39-F16.gguf"), 2048).unwrap();
        let recs: Vec<Json> = std::fs::read_to_string(dir.join("persona-mindx-gen39.jsonl")).unwrap().lines().map(|l| Json::parse(l).unwrap()).collect();
        let num = |v: &Json, k: &str| match v.get(k) { Some(Json::Num(n)) => *n as usize, _ => usize::MAX };
        for variant in ["dir", "alias"] {
            let d = Derived::from_json(&std::fs::read_to_string(dir.join(format!("registry-{variant}/mindx-gen39.MODEL.json"))).unwrap()).unwrap();
            let (mut n, mut ok) = (0, 0);
            for c in &recs {
                let Some(Json::Obj(fields)) = c.get("request") else { panic!() };
                // the user's turns only: the persona and the stop come from the layer
                let mut req: Vec<(String, Json)> = fields.iter().filter(|(k, _)| k != "messages" && k != "stop").cloned().collect();
                req.push(("messages".into(), c.get("user_messages").unwrap().clone()));
                let req = d.apply_openai(&Json::Obj(req));
                let Some(Json::Arr(sent)) = c.get("request").and_then(|r| r.get("messages")) else { panic!() };
                let Some(Json::Arr(layered)) = req.get("messages") else { panic!() };
                assert_eq!(layered, sent, "{}: the layer gives the conversation llama-server was sent", c.get("name").and_then(Json::as_str).unwrap());
                assert_eq!(req.get("stop"), c.get("request").and_then(|r| r.get("stop")));
                let prompt = eng.prompt_fit(req.get("messages").unwrap(), d.num_ctx().map(|x| x as usize)).unwrap();
                let params = eng.params(&req).unwrap();
                let max = match req.get("max_tokens") { Some(Json::Num(x)) => Some(*x as usize), _ => None };
                eng.reset();
                let mut stop = crate::ollama::StopFilter::new(crate::ollama::stops(req.get("stop")).unwrap());
                let done = eng.complete(&prompt, params, max, None, |p| stop.push(p).1).unwrap();
                stop.finish();
                let want: Vec<u32> = match c.get("tokens") { Some(Json::Arr(a)) => a.iter().map(|x| match x { Json::Num(n) => *n as u32, _ => panic!() }).collect(), _ => panic!() };
                let good = done.tokens == want && stop.text() == c.get("text").and_then(Json::as_str).unwrap() && done.finish_reason == c.get("finish_reason").and_then(Json::as_str).unwrap()
                    && done.prompt_tokens == num(c, "prompt_tokens") && done.completion_tokens == num(c, "completion_tokens");
                n += 1;
                ok += good as usize;
                if !good {
                    eprintln!("  {variant} {}: {} tokens vs {} (prompt {} vs {}), {:?}", c.get("name").and_then(Json::as_str).unwrap(), done.tokens.len(), want.len(),
                              done.prompt_tokens, num(c, "prompt_tokens"), &stop.text()[..stop.text().len().min(60)]);
                }
            }
            eprintln!("persona oracle ({variant}: digest {}…): {ok} of {n} answers of the created mindx-gen39 to the user's turns alone token-identical to \
                       llama-server b11192 given promote.py's persona as the system message", &d.digest[..16]);
            assert_eq!(ok, n);
        }
    }
}
