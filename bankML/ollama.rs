// SPDX-License-Identifier: MIT OR Apache-2.0
//! Ollama's API on `bankml serve --native` (0.3.1, phase O1 of docs/OLLAMA.md): the shape mindX and other Ollama
//! clients speak, over bankML's own engine and its gate. **Native only**: nothing here proxies to llama-server or to
//! Ollama; what the verified forward pass cannot do is refused with HTTP 400 `{"error": …}` that says what it does.
//!
//! - `GET /api/version`, `GET /api/tags` (every pinned model, with `"bankml": {"native", "reason"}` so a pin bankML
//!   cannot play is listed honestly), `GET /api/ps` (the resident model and when it expires), `POST /api/show`;
//! - `POST /api/chat` and `POST /api/generate`, streamed as NDJSON (Ollama's default) or not; the final object carries
//!   Ollama's counts and durations (nanoseconds) and the `bankml_receipt`;
//! - `keep_alive` as Ollama reads it (`"5m"`, `"1h"`, seconds, `0` unloads now, a negative value keeps it for good),
//!   and an empty prompt with a `keep_alive` only loads or unloads (`done_reason` `load` / `unload`);
//! - options: temperature, top_k, top_p, min_p, seed, num_predict, stop and (0.3.6) repeat_penalty, repeat_last_n,
//!   presence_penalty, frequency_penalty are honoured; `num_ctx` up to the served context fits the conversation as
//!   Ollama does (0.3.5); resource options (threads, batch, GPU, mmap) do not change an answer bankML gives and are
//!   ignored; anything else is refused.
//!
//! 0.3.3: `format: "json"` (and the schemas `{}`, `{"type": "object"}`) is JSON mode, answered by the grammar llama-server
//! uses for `response_format: json_object` (`grammar.rs`); 0.3.5: any other schema gets llama-server's grammar for it
//! (`schema.rs`). `format` with `raw` is refused.
//!
//! Refused: `tools` (O6), `images` (out of scope), `suffix`, `template`,
//! `context`, and `think: true` (the template is rendered with thinking off, as llama-server `--reasoning off`).
//! `/api/embed` (O7), `/api/pull` and `/api/push` answer why not.
//!
//! O5 (`create.rs`): `/api/create` (a Modelfile, or Ollama's structured form), `/api/delete` (derived models only) and
//! `/api/copy` write and remove **derived models** — a layer (system, parameters, stop, messages) over a pinned base;
//! `/api/tags` lists them with `details.parent_model`, `/api/show` reconstructs their Modelfile, and chat/generate
//! apply their layer as Ollama does.

use crate::gguf::jstr;
use crate::native::{Entry, KeepAlive, Loaded, Residency};
use crate::serve::{respond, Json, Tally};
use std::io::Write;
use std::net::TcpStream;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Ollama's default keep-alive: five minutes after the last request.
pub const KEEP_ALIVE_DEFAULT: KeepAlive = KeepAlive::For(Duration::from_secs(300));

/// A Go duration (`"5m"`, `"1h30m"`, `"2.5s"`, `"-1m"`) or a bare number of seconds, in seconds.
fn duration_secs(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Ok(n) = s.parse::<f64>() {
        return n.is_finite().then_some(n);
    }
    let (neg, mut rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    if rest.is_empty() {
        return None;
    }
    let mut total = 0.0;
    while !rest.is_empty() {
        let n = rest.find(|ch: char| !(ch.is_ascii_digit() || ch == '.')).unwrap_or(rest.len());
        let v: f64 = rest[..n].parse().ok()?;
        rest = &rest[n..];
        let u = rest.find(|ch: char| ch.is_ascii_digit() || ch == '.').unwrap_or(rest.len());
        total += v * match &rest[..u] {
            "ns" => 1e-9,
            "us" | "µs" => 1e-6,
            "ms" => 1e-3,
            "s" => 1.0,
            "m" => 60.0,
            "h" => 3600.0,
            _ => return None,
        };
        rest = &rest[u..];
    }
    Some(if neg { -total } else { total })
}

/// `keep_alive` as Ollama reads it: absent or null is `default`; 0 unloads now; negative keeps the model for good.
pub fn keep_alive(v: Option<&Json>, default: KeepAlive) -> Result<KeepAlive, String> {
    let secs = match v {
        None | Some(Json::Null) => return Ok(default),
        Some(Json::Str(s)) if s.trim().is_empty() => return Ok(default),
        Some(Json::Num(n)) => *n,
        Some(Json::Str(s)) => duration_secs(s).ok_or(format!("keep_alive {s:?}: not a duration (\"5m\", \"1h\", seconds, 0 or -1)"))?,
        Some(_) => return Err("keep_alive must be a duration string or a number of seconds".into()),
    };
    Ok(if secs < 0.0 {
        KeepAlive::Forever
    } else if secs == 0.0 {
        KeepAlive::Unload
    } else {
        KeepAlive::For(Duration::from_secs_f64(secs))
    })
}

/// `stop`: a string or an array of strings.
pub fn stops(v: Option<&Json>) -> Result<Vec<String>, String> {
    match v {
        None | Some(Json::Null) => Ok(Vec::new()),
        Some(Json::Str(s)) => Ok(if s.is_empty() { Vec::new() } else { vec![s.clone()] }),
        Some(Json::Arr(a)) => a.iter().map(|x| x.as_str().map(str::to_string).ok_or("stop must be a string or an array of strings".to_string()))
            .filter(|r| r.as_ref().map_or(true, |s| !s.is_empty())).collect(),
        Some(_) => Err("stop must be a string or an array of strings".into()),
    }
}

/// Stop strings over a stream of pieces: the text up to the first stop is passed on, the stop and what follows are
/// not, and a piece that might be the start of a stop is held back until it is known not to be.
pub struct StopFilter {
    stops: Vec<String>,
    held: String,
    out: String,
    hit: bool,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> Self {
        StopFilter { stops, held: String::new(), out: String::new(), hit: false }
    }
    /// The text to pass on now, and whether generation should go on.
    pub fn push(&mut self, piece: &str) -> (String, bool) {
        if self.hit {
            return (String::new(), false);
        }
        self.held.push_str(piece);
        if let Some(at) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            let pass: String = self.held[..at].to_string();
            self.held.clear();
            self.hit = true;
            self.out.push_str(&pass);
            return (pass, false);
        }
        // keep back the longest tail of `held` that begins some stop string
        let keep = self.stops.iter().flat_map(|s| (1..s.len()).rev().filter(|&k| s.is_char_boundary(k)).find(|&k| self.held.ends_with(&s[..k]))).max().unwrap_or(0);
        let cut = self.held.len() - keep;
        let pass: String = self.held.drain(..cut).collect();
        self.out.push_str(&pass);
        (pass, true)
    }
    /// The end of the answer: anything held back that turned out not to be a stop.
    pub fn finish(&mut self) -> String {
        let rest = std::mem::take(&mut self.held);
        if !self.hit {
            self.out.push_str(&rest);
        }
        if self.hit { String::new() } else { rest }
    }
    /// Everything passed on so far: the answer's text.
    pub fn text(&self) -> &str {
        &self.out
    }
}

/// Ollama's `options`, mapped onto the engine: the sampling keys as a flat object for `Native::params` (which refuses
/// what it does not reproduce), the token limit, and the stop strings.
#[derive(Debug)]
pub struct Opts {
    pub flat: Json,
    pub max: Option<usize>,
    pub stops: Vec<String>,
    /// 0.3.5: `num_ctx`, when the request (or a derived model's layer) sets it: the prompt is fitted to it as Ollama
    /// fits it (`Native::prompt_fit`)
    pub num_ctx: Option<usize>,
}

/// Resource options: they change how the work is scheduled, not the answer bankML gives (its kernels' bits do not
/// depend on threads or batch size), so they are accepted and ignored.
const IGNORED: [&str; 12] = ["num_thread", "num_gpu", "main_gpu", "num_batch", "use_mmap", "use_mlock", "low_vram", "numa", "f16_kv",
                             "vocab_only", "logits_all", "num_keep"];
/// O2: `repeat_last_n` moved here from IGNORED — with the penalties reproduced, it changes the answer.
const SAMPLING: [&str; 11] = ["temperature", "top_k", "top_p", "min_p", "seed", "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty", "typical_p", "min_keep"];

pub fn options(o: Option<&Json>, n_ctx: usize) -> Result<Opts, String> {
    let mut flat = Vec::new();
    let (mut max, mut st, mut nctx) = (None, Vec::new(), None);
    let fields: &[(String, Json)] = match o {
        None | Some(Json::Null) => &[],
        Some(Json::Obj(f)) => f,
        Some(_) => return Err("options must be an object".into()),
    };
    let num = |k: &str, v: &Json| match v {
        Json::Num(n) => Ok(*n),
        _ => Err(format!("options.{k} must be a number")),
    };
    for (k, v) in fields {
        match k.as_str() {
            k if SAMPLING.contains(&k) => flat.push((k.to_string(), Json::Num(num(k, v)?))),
            "num_predict" => {
                let n = num(k, v)?;
                max = (n >= 0.0).then_some(n as usize); // -1: until the turn ends; -2: until the context is full
            }
            "num_ctx" => {
                let n = num(k, v)?;
                if n > n_ctx as f64 {
                    return Err(format!("num_ctx {n} is larger than the context this server holds ({n_ctx}); restart bankml serve with --ctx {n} if it fits in memory"));
                }
                if n >= 1.0 {
                    nctx = Some(n as usize);
                }
            }
            "stop" => st = stops(Some(v))?,
            "mirostat" if num(k, v)? != 0.0 => return Err("mirostat: not reproduced; bankML reproduces llama.cpp's top-k, top-p, min-p and temperature".into()),
            "tfs_z" if num(k, v)? != 1.0 => return Err("tfs_z: tail-free sampling is not reproduced (llama.cpp removed it)".into()),
            "mirostat" | "mirostat_eta" | "mirostat_tau" | "tfs_z" | "penalize_newline" => {}
            k if IGNORED.contains(&k) => {}
            k => return Err(format!("options.{k}: not an option bankML knows; it refuses rather than ignore what might change the answer")),
        }
    }
    Ok(Opts { flat: Json::Obj(flat), max, stops: st, num_ctx: nctx })
}

/// What the request asks for that the native forward pass does not do, refused before any model is loaded.
pub fn refusals(req: &Json) -> Result<(), String> {
    let present = |k: &str| match req.get(k) {
        None | Some(Json::Null) | Some(Json::Bool(false)) => false,
        Some(Json::Str(s)) => !s.is_empty(),
        Some(Json::Arr(a)) => !a.is_empty(),
        Some(Json::Obj(o)) => !o.is_empty(),
        Some(_) => true,
    };
    // 0.3.3: `format: "json"` is JSON mode (O6); O6b: a schema object is converted as llama-server converts it, and
    // what b11192 refuses in it is refused here (`grammar::from_ollama`)
    let constraint = crate::grammar::from_ollama(req.get("format"))?;
    if constraint != crate::grammar::Constraint::None && req.get("raw").and_then(Json::as_bool).unwrap_or(false) {
        return Err("format with raw: JSON mode follows llama-server's chat path (the template's generation prompt is part of its grammar); a raw prompt has none — drop raw, or ask for JSON in the prompt".into());
    }
    if present("tools") {
        return Err("tools: tool calling needs the template's tool rendering and a grammar (O6 in docs/OLLAMA.md); not reproduced yet".into());
    }
    if present("images") {
        return Err("images: bankML's models are text-only; vision is out of scope".into());
    }
    if present("suffix") {
        return Err("suffix: fill-in-the-middle is not reproduced; send the whole prompt".into());
    }
    if present("template") {
        return Err("template: bankML renders the pinned model's own chat template (byte-identical to llama.cpp); a replacement template is not taken".into());
    }
    if present("context") {
        return Err("context: send the conversation as /api/chat messages instead; the prompt cache reuses the shared prefix".into());
    }
    match req.get("think") {
        None | Some(Json::Null) | Some(Json::Bool(false)) => {}
        Some(_) => return Err("think: the pinned template is rendered with thinking off (an empty <think> block, as llama-server --reasoning off); thinking is not reproduced yet".into()),
    }
    Ok(())
}

/// Ollama messages to the chat module's shape: `thinking` is its `reasoning_content`; images and tool calls refused.
pub fn chat_messages(v: &Json) -> Result<Json, String> {
    let Json::Arr(a) = v else { return Err("messages must be an array".into()) };
    let mut out = Vec::new();
    for m in a {
        if matches!(m.get("images"), Some(Json::Arr(x)) if !x.is_empty()) {
            return Err("images: bankML's models are text-only; vision is out of scope".into());
        }
        if matches!(m.get("tool_calls"), Some(Json::Arr(x)) if !x.is_empty()) {
            return Err("tool_calls: tool calling is not reproduced yet (O6 in docs/OLLAMA.md)".into());
        }
        let mut f = vec![("role".to_string(), Json::Str(m.get("role").and_then(Json::as_str).ok_or("a message without a role")?.to_string()))];
        match m.get("content") {
            Some(Json::Str(s)) => f.push(("content".into(), Json::Str(s.clone()))),
            None | Some(Json::Null) => {}
            _ => return Err("only text content is in scope".into()),
        }
        if let Some(t) = m.get("thinking").or(m.get("reasoning_content")).and_then(Json::as_str) {
            f.push(("reasoning_content".into(), Json::Str(t.to_string())));
        }
        out.push(Json::Obj(f));
    }
    Ok(Json::Arr(out))
}

/// RFC 3339 in UTC with nanoseconds, as Ollama writes `created_at`.
pub fn rfc3339(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let (secs, ns) = (d.as_secs() as i64, d.subsec_nanos());
    let (days, sod) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil from days (H. Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + (month <= 2) as i64;
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{ns:09}Z", sod / 3600, sod % 3600 / 60, sod % 60)
}

/// One NDJSON line of a streamed answer (`chat`: a `message`; else a `response`).
pub fn piece_line(model: &str, chat: bool, piece: &str) -> String {
    let body = if chat { format!("\"message\": {{\"role\": \"assistant\", \"content\": {}}}", jstr(piece)) } else { format!("\"response\": {}", jstr(piece)) };
    format!("{{\"model\": {}, \"created_at\": \"{}\", {body}, \"done\": false}}\n", jstr(model), rfc3339(SystemTime::now()))
}

/// The last object: the whole text when not streamed (empty when streamed), Ollama's counts and durations, the receipt.
#[allow(clippy::too_many_arguments)]
pub fn final_line(model: &str, chat: bool, text: &str, reason: &str, total_ns: u64, load_ns: u64, d: Option<&crate::native::Done>, receipt: Option<&str>) -> String {
    let body = if chat { format!("\"message\": {{\"role\": \"assistant\", \"content\": {}}}", jstr(text)) } else { format!("\"response\": {}", jstr(text)) };
    let counts = d.map(|d| format!(", \"prompt_eval_count\": {}, \"prompt_eval_duration\": {}, \"eval_count\": {}, \"eval_duration\": {}, \"bankml_cache_n\": {}",
                                    d.prompt_tokens, d.prompt_ns, d.completion_tokens, d.eval_ns, d.cached_tokens)).unwrap_or_default();
    let r = receipt.map(|r| format!(", \"bankml_receipt\": {r}")).unwrap_or_default();
    format!("{{\"model\": {}, \"created_at\": \"{}\", {body}, \"done\": true, \"done_reason\": {}, \"total_duration\": {total_ns}, \"load_duration\": {load_ns}{counts}{r}}}\n",
            jstr(model), rfc3339(SystemTime::now()), jstr(reason))
}

fn err(c: &mut TcpStream, code: u16, msg: &str) -> std::io::Result<()> {
    respond(c, code, "application/json", format!("{{\"error\": {}}}", jstr(msg)).as_bytes())
}

pub(crate) fn tag(e: &Entry) -> String {
    format!("{}:latest", e.name)
}

pub(crate) fn details(e: &Entry) -> String {
    let a = e.info.arch.as_deref().unwrap_or("");
    format!("{{\"parent_model\": \"\", \"format\": \"gguf\", \"family\": {}, \"families\": [{}], \"parameter_size\": {}, \"quantization_level\": {}}}",
            jstr(a), jstr(a), jstr(&e.info.params), jstr(&e.info.quant))
}

pub(crate) fn native_json(e: &Entry) -> String {
    let reason = e.info.native.as_ref().err().map(|r| jstr(r)).unwrap_or("null".into());
    format!("{{\"native\": {}, \"reason\": {reason}, \"file\": {}, \"found\": {}}}", e.info.native.is_ok(), jstr(&e.file), e.path.is_some())
}

fn tags(rs: &Residency) -> String {
    let mut models: Vec<String> = rs.reg.entries.iter().map(|e| {
        let modified = e.path.as_ref().and_then(|p| std::fs::metadata(p).ok()).and_then(|m| m.modified().ok()).unwrap_or(UNIX_EPOCH);
        format!("{{\"name\": {t}, \"model\": {t}, \"modified_at\": \"{}\", \"size\": {}, \"digest\": {}, \"details\": {}, \"bankml\": {}}}",
                rfc3339(modified), e.bytes, jstr(&e.sha256), details(e), native_json(e), t = jstr(&tag(e)))
    }).collect();
    models.extend(crate::create::tag_objects(rs, details, native_json));
    format!("{{\"models\": [{}]}}", models.join(", "))
}

fn ps(rs: &Residency) -> String {
    let Some((l, expires)) = rs.peek() else { return "{\"models\": []}".into() };
    let Some(e) = rs.reg.entries.iter().find(|e| e.name == l.name) else { return "{\"models\": []}".into() };
    // resident for good: Ollama writes now + the largest duration (≈ 292 years)
    let exp = expires.unwrap_or_else(|| SystemTime::now() + Duration::from_secs(292 * 365 * 86400));
    // 0.3.5: named as Ollama names its runner — the model whose request loaded it (a derived model's own name, its
    // manifest digest and parent), with that request's context
    let (name, n_ctx) = rs.shown().unwrap_or_else(|| (tag(e), rs.n_ctx));
    let derived = rs.derived.find(name.trim_end_matches(":latest")).filter(|(_, de)| de.name == e.name);
    let (digest, det) = match &derived {
        Some((d, _)) => (jstr(&format!("sha256:{}", d.digest)),
                         details(e).replacen("\"parent_model\": \"\"", &format!("\"parent_model\": {}", jstr(&format!("{}:latest", d.base.name))), 1)),
        None => (jstr(&l.sha256), details(e)),
    };
    // a record left by other weights (they were unloaded since) names nothing here
    let (name, n_ctx) = if derived.is_some() || name == tag(e) { (name, n_ctx) } else { (tag(e), rs.n_ctx) };
    format!("{{\"models\": [{{\"name\": {t}, \"model\": {t}, \"size\": {}, \"digest\": {digest}, \"details\": {det}, \"expires_at\": \"{}\", \"size_vram\": 0, \"context_length\": {n_ctx}}}]}}",
            l.size, rfc3339(exp), t = jstr(&name))
}

fn show(rs: &Residency, e: &Entry) -> Result<String, String> {
    let path = e.path.as_ref().ok_or(format!("{} is pinned but not on this machine", e.file))?;
    let r = crate::gguf::guard_file(path, crate::gguf::Engine::Mainline).map_err(|err| format!("{}: {err}", path.display()))?;
    let h = r.header.ok_or("no GGUF header")?;
    let mut keys: Vec<&String> = h.kv.keys().filter(|k| *k != "tokenizer.chat_template").collect();
    keys.sort();
    let info: Vec<String> = keys.iter().filter_map(|k| {
        let v = match &h.kv[*k] {
            crate::gguf::Val::U(v) => v.to_string(),
            crate::gguf::Val::I(v) => v.to_string(),
            crate::gguf::Val::F(v) if v.is_finite() => format!("{v}"),
            crate::gguf::Val::B(v) => v.to_string(),
            crate::gguf::Val::S(s) => jstr(s),
            _ => return None,
        };
        Some(format!("{}: {v}", jstr(k)))
    }).collect();
    let template = match h.kv.get("tokenizer.chat_template") {
        Some(crate::gguf::Val::S(t)) => t.clone(),
        _ => String::new(),
    };
    let params = e.info.defaults.as_ref().map(|p| format!("temperature {}\ntop_k {}\ntop_p {}\nmin_p {}\nstop \"<|im_end|>\"", p.temp, p.top_k, p.top_p, p.min_p)).unwrap_or_default();
    let modelfile = format!("# bankml {}: a pinned model (sha256 {}), verified before every load; not editable\nFROM {}\n{}\n",
                            crate::VERSION, e.sha256, path.display(), params.lines().map(|l| format!("PARAMETER {l}")).collect::<Vec<_>>().join("\n"));
    let caps = if e.info.native.is_ok() { "[\"completion\"]" } else { "[]" };
    Ok(format!("{{\"modelfile\": {}, \"parameters\": {}, \"template\": {}, \"details\": {}, \"model_info\": {{{}}}, \"capabilities\": {caps}, \"modified_at\": \"{}\", \"bankml\": {{\"sha256\": {}, \"native\": {}, \"n_ctx\": {}}}}}",
               jstr(&modelfile), jstr(&params), jstr(&template), details(e), info.join(", "),
               rfc3339(std::fs::metadata(path).and_then(|m| m.modified()).unwrap_or(UNIX_EPOCH)), jstr(&e.sha256), native_json(e), rs.n_ctx))
}

/// The `/api/*` routes (and `/`). `default_ka` is the keep-alive a request that names none gets.
pub fn route(c: &mut TcpStream, rs: &Residency, default_ka: KeepAlive, method: &str, path: &str, body: &[u8]) -> std::io::Result<()> {
    let req = || Json::parse(&String::from_utf8_lossy(body));
    match (method, path) {
        ("GET" | "HEAD", "/") => respond(c, 200, "text/plain", b"bankml is running (Ollama's API at /api, native only)"),
        ("GET", "/api/version") => respond(c, 200, "application/json", format!("{{\"version\": \"{}\", \"bankml\": true}}", crate::VERSION).as_bytes()),
        ("GET", "/api/tags") => respond(c, 200, "application/json", tags(rs).as_bytes()),
        ("GET", "/api/ps") => respond(c, 200, "application/json", ps(rs).as_bytes()),
        ("POST", "/api/show") => {
            let Some(v) = req() else { return err(c, 400, "body is not JSON") };
            let name = v.get("model").or(v.get("name")).and_then(Json::as_str);
            let r = crate::create::resolve(rs, name).and_then(|(e, d)| show(rs, &e).map(|j| match d {
                Some(d) => crate::create::show_json(&d, &j),
                None => j,
            }));
            match r {
                Ok(j) => respond(c, 200, "application/json", j.as_bytes()),
                Err(e) => err(c, 404, &e),
            }
        }
        ("POST", "/api/chat") => generate(c, rs, default_ka, body, true),
        ("POST", "/api/generate") => generate(c, rs, default_ka, body, false),
        ("POST", "/api/embed" | "/api/embeddings") => err(c, 400, "embeddings need an encoder graph (bge-m3 is XLM-R: LayerNorm, bidirectional attention, CLS pooling), which bankML does not have yet (O7 in docs/OLLAMA.md); keep Ollama for embeddings"),
        ("POST", "/api/create") => crate::create::http_create(c, rs, body),
        ("POST", "/api/pull") => err(c, 400, "bankml does not pull: models are imported sha256-pinned with open licences only (sAGI/models.py import), then served by name"),
        ("DELETE", "/api/delete") => crate::create::http_delete(c, rs, body),
        ("POST", "/api/copy") => crate::create::http_copy(c, rs, body),
        ("POST", "/api/push") => err(c, 400, "not offered: bankml publishes nothing; a model is shared by its pinned file and FORK.json"),
        _ => err(c, 404, "bankml serve --native (Ollama API): GET /api/version /api/tags /api/ps, POST /api/show /api/chat /api/generate /api/create /api/copy, DELETE /api/delete"),
    }
}

/// `/api/chat` (`chat`) and `/api/generate`.
fn generate(c: &mut TcpStream, rs: &Residency, default_ka: KeepAlive, body: &[u8], chat: bool) -> std::io::Result<()> {
    let Some(req) = Json::parse(&String::from_utf8_lossy(body)) else { return err(c, 400, "body is not JSON") };
    let (e, layer) = match crate::create::resolve(rs, req.get("model").and_then(Json::as_str)) {
        Ok(x) => x,
        Err(m) => return err(c, 404, &m),
    };
    let e = &e;
    // O5: a derived model's layer — its SYSTEM, MESSAGEs and parameters, as Ollama applies them
    let req = match &layer {
        Some(d) => d.apply_ollama(&req, chat),
        None => req,
    };
    let ka = match keep_alive(req.get("keep_alive"), default_ka) {
        Ok(k) => k,
        Err(m) => return err(c, 400, &m),
    };
    let stream = req.get("stream").and_then(Json::as_bool).unwrap_or(true);
    let model = layer.as_ref().map(|d| d.tag()).unwrap_or_else(|| tag(e));
    let t = Tally::new(body);
    // an empty prompt (no messages) only loads or unloads, as Ollama does
    let empty = if chat { !matches!(req.get("messages"), Some(Json::Arr(a)) if !a.is_empty()) } else { req.get("prompt").and_then(Json::as_str).unwrap_or("").is_empty() };
    if empty {
        let run = rs.lock_run();
        let (reason, load_ns) = if ka == KeepAlive::Unload {
            rs.touch(&e.name, KeepAlive::Unload);
            ("unload", 0)
        } else {
            match rs.acquire(&run, e) {
                Ok((_, ns)) => {
                    rs.touch(&e.name, ka);
                    let nctx = options(req.get("options"), rs.n_ctx).ok().and_then(|o| o.num_ctx).unwrap_or(rs.n_ctx);
                    rs.shown_as(&model, nctx, ns > 0);
                    ("load", ns)
                }
                Err((code, m)) => return err(c, code, &m),
            }
        };
        let line = final_line(&model, chat, "", reason, t.t0.elapsed().as_nanos() as u64, load_ns, None, None);
        return respond(c, 200, if stream { "application/x-ndjson" } else { "application/json" }, line.as_bytes());
    }
    if let Err(m) = refusals(&req) {
        return err(c, 400, &m);
    }
    // O6b: the schema's numbers read from the request's own text, as llama-server would read them
    let constraint = match crate::grammar::from_ollama_text(&req, &String::from_utf8_lossy(body)) {
        Ok(k) => k,
        Err(m) => return err(c, 400, &m),
    };
    let o = match options(req.get("options"), rs.n_ctx) {
        Ok(o) => o,
        Err(m) => return err(c, 400, &m),
    };
    // the sampler refusals before a model is loaded for nothing
    if let Some(d) = &e.info.defaults {
        if let Err(m) = crate::native::sampling(d.clone(), &o.flat).and_then(crate::sampler::Sampler::new) {
            return err(c, 400, &m);
        }
    }
    let msgs = if chat || (layer.is_some() && req.get("bankml_messages").is_some()) {
        match chat_messages(req.get(if chat { "messages" } else { "bankml_messages" }).unwrap()) {
            Ok(m) => Some(m),
            Err(m) => return err(c, 400, &m),
        }
    } else {
        None
    };
    let run = rs.lock_run();
    let (l, load_ns) = match rs.acquire(&run, e) {
        Ok(x) => x,
        Err((code, m)) => return err(c, code, &m),
    };
    rs.shown_as(&model, o.num_ctx.unwrap_or(rs.n_ctx), load_ns > 0);
    let r = answer(c, &l, &req, msgs.as_ref(), o, constraint, chat, stream, load_ns, &model, t);
    rs.touch(&e.name, ka);
    r
}

#[allow(clippy::too_many_arguments)]
fn answer(c: &mut TcpStream, l: &Loaded, req: &Json, msgs: Option<&Json>, o: Opts, constraint: crate::grammar::Constraint, chat: bool, stream: bool, load_ns: u64, model: &str, mut t: Tally) -> std::io::Result<()> {
    let eng = &l.native;
    // 0.3.5: under a num_ctx the conversation is fitted as Ollama's chatPrompt fits it (generate goes the same way)
    let prompt = match msgs {
        Some(m) => eng.prompt_fit(m, o.num_ctx),
        None => {
            let p = req.get("prompt").and_then(Json::as_str).unwrap_or("");
            if req.get("raw").and_then(Json::as_bool).unwrap_or(false) {
                let toks = eng.tok.encode(p, true);
                match o.num_ctx.filter(|n| toks.len() > *n) {
                    Some(n) => Err(format!("the raw prompt is {} tokens, more than num_ctx {n}: Ollama would cut tokens out of its middle (num_keep), \
                                            which bankML does not reproduce; shorten it or raise num_ctx (up to this server's --ctx)", toks.len())),
                    None => Ok(toks),
                }
            } else {
                let msg = |r: &str, c: &str| Json::Obj(vec![("role".into(), Json::Str(r.into())), ("content".into(), Json::Str(c.into()))]);
                let mut m = Vec::new();
                if let Some(s) = req.get("system").and_then(Json::as_str).filter(|s| !s.is_empty()) {
                    m.push(msg("system", s));
                }
                m.push(msg("user", p));
                eng.prompt_fit(&Json::Arr(m), o.num_ctx)
            }
        }
    };
    let prompt = match prompt {
        Ok(p) => p,
        Err(m) => return err(c, 400, &m),
    };
    let params = match eng.params(&o.flat) {
        Ok(p) => p,
        Err(m) => return err(c, 400, &m),
    };
    let mut stop = StopFilter::new(o.stops);
    let grammar = match eng.grammar(&constraint) {
        Ok(g) => g,
        Err(m) => return err(c, 400, &m),
    };
    let mut content = crate::grammar::ContentStream::new(&constraint);
    if stream {
        write!(c, "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
        let send = |c: &mut TcpStream, piece: &str| piece.is_empty() || c.write_all(piece_line(model, chat, piece).as_bytes()).and_then(|_| c.flush()).is_ok();
        let done = eng.complete(&prompt, params, o.max, grammar, |piece| {
            if t.ttft.is_none() {
                t.ttft = Some(t.t0.elapsed());
            }
            let (out, go) = stop.push(piece);
            send(c, &content.push(&out)) && go
        });
        let rest = stop.finish();
        send(c, &content.push(&rest));
        let d = match done {
            Ok(d) => d,
            Err(m) => return writeln!(c, "{{\"error\": {}}}", jstr(&m)),
        };
        t.text = content.content(stop.text(), true);
        t.prompt = d.prompt_tokens as u64;
        t.completion = d.completion_tokens as u64;
        let line = final_line(model, chat, "", d.finish_reason, t.t0.elapsed().as_nanos() as u64, load_ns, Some(&d), Some(&t.receipt(&l.engine, &l.verified)));
        c.write_all(line.as_bytes())?;
        return c.flush();
    }
    let d = match eng.complete(&prompt, params, o.max, grammar, |piece| {
        if t.ttft.is_none() {
            t.ttft = Some(t.t0.elapsed());
        }
        stop.push(piece).1
    }) {
        Ok(d) => d,
        Err(m) => return err(c, 500, &m),
    };
    stop.finish();
    t.text = content.content(stop.text(), false);
    t.prompt = d.prompt_tokens as u64;
    t.completion = d.completion_tokens as u64;
    let line = final_line(model, chat, &t.text, d.finish_reason, t.t0.elapsed().as_nanos() as u64, load_ns, Some(&d), Some(&t.receipt(&l.engine, &l.verified)));
    respond(c, 200, "application/json", line.trim_end().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{Info, Registry};

    #[test]
    fn keep_alive_as_ollama_reads_it() {
        let ka = |j: &str| keep_alive(Json::parse(j).as_ref(), KEEP_ALIVE_DEFAULT);
        assert_eq!(ka("\"5m\""), Ok(KeepAlive::For(Duration::from_secs(300))));
        assert_eq!(ka("\"1h\""), Ok(KeepAlive::For(Duration::from_secs(3600))));
        assert_eq!(ka("\"1h30m\""), Ok(KeepAlive::For(Duration::from_secs(5400))));
        assert_eq!(ka("\"2.5s\""), Ok(KeepAlive::For(Duration::from_millis(2500))));
        assert_eq!(ka("90"), Ok(KeepAlive::For(Duration::from_secs(90))));
        assert_eq!(ka("\"90\""), Ok(KeepAlive::For(Duration::from_secs(90))));
        assert_eq!(ka("0"), Ok(KeepAlive::Unload));
        assert_eq!(ka("\"0\""), Ok(KeepAlive::Unload));
        assert_eq!(ka("\"0s\""), Ok(KeepAlive::Unload));
        assert_eq!(ka("-1"), Ok(KeepAlive::Forever));
        assert_eq!(ka("\"-1\""), Ok(KeepAlive::Forever));
        assert_eq!(ka("\"-1m\""), Ok(KeepAlive::Forever));
        assert_eq!(keep_alive(None, KEEP_ALIVE_DEFAULT), Ok(KEEP_ALIVE_DEFAULT));
        assert_eq!(ka("null"), Ok(KEEP_ALIVE_DEFAULT));
        assert_eq!(ka("\"\""), Ok(KEEP_ALIVE_DEFAULT));
        for bad in ["\"5 minutes\"", "\"m\"", "\"5x\"", "true", "[1]"] {
            assert!(ka(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn options_map_onto_the_engine() {
        let o = options(Json::parse(r#"{"temperature": 0.3, "top_k": 20, "top_p": 0.9, "min_p": 0.1, "seed": 42, "num_predict": 64, "stop": ["\n\n", "END"], "num_ctx": 2048, "num_thread": 3, "repeat_last_n": 64}"#).as_ref(), 4096).unwrap();
        assert_eq!(o.max, Some(64));
        assert_eq!(o.stops, vec!["\n\n".to_string(), "END".to_string()]);
        let base = crate::sampler::Params::default();
        let p = crate::native::sampling(base.clone(), &o.flat).unwrap();
        assert_eq!((p.temp, p.top_k, p.top_p, p.min_p, p.seed), (0.3, 20, 0.9, 0.1, 42));
        // num_predict -1 (until the turn ends) and no options at all
        assert_eq!(options(Json::parse(r#"{"num_predict": -1}"#).as_ref(), 4096).unwrap().max, None);
        assert_eq!(options(None, 4096).unwrap().max, None);
        // O2: the penalties map onto the engine (the coach's repeat_penalty 1.3 included); 0.3.7: typical-p too
        let pen = options(Json::parse(r#"{"repeat_penalty": 1.3, "repeat_last_n": 32, "presence_penalty": 0.5, "frequency_penalty": -0.25}"#).as_ref(), 4096).unwrap();
        let p = crate::native::sampling(base.clone(), &pen.flat).unwrap();
        assert_eq!((p.penalty_repeat, p.penalty_last_n, p.penalty_present, p.penalty_freq), (1.3, 32, 0.5, -0.25));
        let typ = options(Json::parse(r#"{"typical_p": 0.9}"#).as_ref(), 4096).unwrap();
        assert_eq!(crate::native::sampling(base, &typ.flat).unwrap().typical_p, 0.9);
        // num_ctx above the served context, unknown options, non-numbers, mirostat: refused
        assert!(options(Json::parse(r#"{"num_ctx": 8192}"#).as_ref(), 4096).unwrap_err().contains("8192"));
        assert!(options(Json::parse(r#"{"warp": 1}"#).as_ref(), 4096).unwrap_err().contains("options.warp"));
        assert!(options(Json::parse(r#"{"temperature": "hot"}"#).as_ref(), 4096).is_err());
        assert!(options(Json::parse(r#"{"mirostat": 2}"#).as_ref(), 4096).is_err());
        assert!(options(Json::parse(r#"{"mirostat": 0, "tfs_z": 1.0}"#).as_ref(), 4096).is_ok());
    }

    #[test]
    fn refusals_say_why() {
        let r = |j: &str| refusals(&Json::parse(j).unwrap());
        // 0.3.3: JSON mode is answered, and format with raw is refused; O6b: a schema is answered, a bad one refused
        assert!(r(r#"{"format": "json"}"#).is_ok());
        assert!(r(r#"{"format": {"type": "object"}}"#).is_ok());
        assert!(r(r#"{"format": {"type": "object", "required": ["a"]}}"#).is_ok());
        assert!(r(r#"{"format": {"type": "object", "properties": {"a": {"enum": []}}}}"#).unwrap_err().starts_with("format:"));
        assert!(r(r#"{"format": "json", "raw": true}"#).unwrap_err().contains("raw"));
        assert!(r(r#"{"tools": [{"type": "function"}]}"#).unwrap_err().starts_with("tools:"));
        assert!(r(r#"{"images": ["aGk="]}"#).unwrap_err().starts_with("images:"));
        assert!(r(r#"{"suffix": "}"}"#).unwrap_err().starts_with("suffix:"));
        assert!(r(r#"{"think": true}"#).unwrap_err().starts_with("think:"));
        assert!(r(r#"{"think": "high"}"#).is_err());
        // what does not change the answer passes: empty values, think false, no format
        assert!(r(r#"{"format": "", "tools": [], "images": null, "think": false, "suffix": ""}"#).is_ok());
        assert!(chat_messages(&Json::parse(r#"[{"role": "user", "content": "hi", "images": ["x"]}]"#).unwrap()).unwrap_err().starts_with("images:"));
        assert!(chat_messages(&Json::parse(r#"[{"role": "assistant", "content": "", "tool_calls": [{}]}]"#).unwrap()).is_err());
        let m = chat_messages(&Json::parse(r#"[{"role": "assistant", "content": "4", "thinking": "2+2"}, {"role": "user", "content": "ok"}]"#).unwrap()).unwrap();
        let msgs = crate::chat::messages_from_json(&m).unwrap();
        assert_eq!(msgs[0].reasoning.as_deref(), Some("2+2"));
    }

    #[test]
    fn ndjson_framing() {
        let p = piece_line("bonsai-1.7b-q1_0:latest", true, "line one\n\"two\"");
        assert!(p.ends_with('\n') && p.matches('\n').count() == 1, "one object per line: {p:?}");
        let v = Json::parse(p.trim_end()).unwrap();
        assert_eq!(v.get("message").and_then(|m| m.get("content")).and_then(Json::as_str), Some("line one\n\"two\""));
        assert_eq!(v.get("done"), Some(&Json::Bool(false)));
        let g = Json::parse(piece_line("m", false, "x").trim_end()).unwrap();
        assert_eq!(g.get("response").and_then(Json::as_str), Some("x"));
        let d = crate::native::Done { prompt_tokens: 12, cached_tokens: 3, completion_tokens: 5, finish_reason: "stop", text: "hi".into(), prompt_ns: 7, eval_ns: 9, tokens: vec![], grammar_ns: 0, resampled: 0, ..Default::default() };
        let f = final_line("m", true, "", "stop", 100, 4, Some(&d), Some("{\"signed\": false}"));
        assert!(f.ends_with('\n') && f.matches('\n').count() == 1);
        let v = Json::parse(f.trim_end()).unwrap();
        for (k, want) in [("prompt_eval_count", 12), ("eval_count", 5), ("prompt_eval_duration", 7), ("eval_duration", 9), ("total_duration", 100), ("load_duration", 4)] {
            assert_eq!(v.get(k).and_then(Json::as_u64), Some(want), "{k}");
        }
        assert_eq!(v.get("done_reason").and_then(Json::as_str), Some("stop"));
        assert!(v.get("bankml_receipt").is_some());
        let u = Json::parse(final_line("m", false, "", "unload", 1, 0, None, None).trim_end()).unwrap();
        assert_eq!((u.get("done_reason").and_then(Json::as_str), u.get("response").and_then(Json::as_str)), (Some("unload"), Some("")));
        assert!(u.get("eval_count").is_none());
    }

    #[test]
    fn stop_strings_end_the_answer() {
        let run = |stops: &[&str], pieces: &[&str]| {
            let mut f = StopFilter::new(stops.iter().map(|s| s.to_string()).collect());
            let (mut out, mut went) = (String::new(), 0);
            for p in pieces {
                let (o, go) = f.push(p);
                out += &o;
                went += 1;
                if !go {
                    break;
                }
            }
            out += &f.finish();
            assert_eq!(out, f.text());
            (out, went)
        };
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), 2));
        assert_eq!(run(&["END"], &["one E", "N", "D two"]), ("one ".into(), 3));
        assert_eq!(run(&["END"], &["one E", "NOUGH"]), ("one ENOUGH".into(), 2));
        assert_eq!(run(&["\n\n", "x"], &["ab\n", "\ncd"]), ("ab".into(), 2));
        assert_eq!(run(&["é!"], &["café", "!", "more"]), ("caf".into(), 2));
        // a held-back prefix that never completes is released at the end
        assert_eq!(run(&["END"], &["the E"]), ("the E".into(), 1));
        assert_eq!(stops(Json::parse(r#""s""#).as_ref()), Ok(vec!["s".to_string()]));
        assert!(stops(Json::parse("[1]").as_ref()).is_err());
    }

    #[test]
    fn names_resolve_to_pins() {
        let info = || Info { arch: Some("qwen3".into()), quant: "Q1_0".into(), params: "1.7B".into(), native: Ok(()), defaults: None };
        let e = |file: &str| Entry { name: crate::native::model_name(file), file: file.into(), path: None, fork_json: String::new(), sha256: String::new(), bytes: 0, info: info() };
        let reg = Registry { entries: vec![e("Ternary-Bonsai-8B-Q2_0_g64.gguf"), e("Bonsai-1.7B-Q1_0.gguf")], default: 0 };
        assert_eq!(reg.entries[0].name, "ternary-bonsai-8b-q2_0_g64");
        for (ask, want) in [(None, "ternary-bonsai-8b-q2_0_g64"), (Some(""), "ternary-bonsai-8b-q2_0_g64"), (Some("bonsai-1.7b-q1_0"), "bonsai-1.7b-q1_0"),
                            (Some("bonsai-1.7b-q1_0:latest"), "bonsai-1.7b-q1_0"), (Some("Bonsai-1.7B-Q1_0.gguf"), "bonsai-1.7b-q1_0"),
                            (Some("TERNARY-BONSAI-8B-Q2_0_G64:latest"), "ternary-bonsai-8b-q2_0_g64")] {
            assert_eq!(reg.resolve(ask).map(|e| e.name.as_str()), Ok(want), "{ask:?}");
        }
        let miss = reg.resolve(Some("llama3:8b")).unwrap_err();
        assert!(miss.contains("not found") && miss.contains("bonsai-1.7b-q1_0"), "{miss}");
        assert!(reg.resolve(Some("bonsai-1.7b-q1_0:q4")).is_err());
    }

    #[test]
    fn timestamps_are_rfc3339() {
        assert_eq!(rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(rfc3339(UNIX_EPOCH + Duration::new(1_727_740_800, 5)), "2024-10-01T00:00:00.000000005Z");
        assert_eq!(rfc3339(UNIX_EPOCH + Duration::from_secs(951_782_400)), "2000-02-29T00:00:00.000000000Z");
    }
}
