//! P0 — `bankml serve`: answers now, through the reference, behind the gate. A loopback HTTP gateway (std only)
//! in front of llama.cpp b11192's `llama-server`:
//!
//! - it does not start until the model file passes `verify` (guard, then the sha256 pin to FORK.json);
//! - the upstream must be serving **that file**: either bankml launches it (`--spawn LLAMA_SERVER`, the model
//!   argument is the verified path) or an already-running server's `/props` must name the same canonical path;
//! - every `/v1/chat/completions` answer (streamed or not) carries a receipt — bankml version, engine, model
//!   sha256, guard verdict, tokens, time to first token, wall time, and the sha256 of the answer text exactly as
//!   the model produced it — so a client can check the text it shows is the text the verified model wrote.
//!   Non-streamed: a top-level `bankml_receipt` object. Streamed: one extra `data: {"bankml_receipt": …}` event
//!   before `data: [DONE]` (OpenAI clients ignore it).
//!
//! What it is not: bankml's own forward pass (P3). The tokens come from ggml's kernels in llama-server; bankml
//! vouches for the file, the path and the transcript, not for the arithmetic. The file is hashed at start, so a
//! file replaced afterwards is not detected until the next start (`GET /bankml` shows when it was hashed).

use crate::{gguf, Verified};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct Config {
    pub model: PathBuf,
    pub fork_json: String,
    pub engine: gguf::Engine,
    pub listen: String,
    /// host:port of llama-server
    pub upstream: String,
    /// launch llama-server from this binary instead of trusting a running one
    pub spawn: Option<PathBuf>,
    pub threads: usize,
    pub ctx: usize,
}

struct State {
    verified: Verified,
    model: PathBuf,
    upstream: String,
    engine: String,
    hashed_at: u64,
}

/// Verify, bind the upstream, then serve until killed.
pub fn run(cfg: Config) -> Result<(), String> {
    let verified = crate::verify(&cfg.model, &cfg.fork_json, cfg.engine)?;
    let model = cfg.model.canonicalize().map_err(|e| format!("{}: {e}", cfg.model.display()))?;
    let upstream = cfg.upstream.trim_start_matches("http://").trim_end_matches('/').to_string();
    let _child = match &cfg.spawn {
        Some(bin) => {
            let port = upstream.rsplit(':').next().unwrap_or("18092").to_string();
            let host = upstream.rsplit_once(':').map(|h| h.0).unwrap_or("127.0.0.1").to_string();
            let c = std::process::Command::new(bin)
                .args(["-m", &model.to_string_lossy(), "--host", &host, "--port", &port, "-t", &cfg.threads.to_string()])
                .args(["-c", &cfg.ctx.to_string(), "-np", "1", "--jinja", "--reasoning", "off", "--no-webui"])
                .stdout(std::process::Stdio::null())
                .spawn()
                .map_err(|e| format!("cannot launch {}: {e}", bin.display()))?;
            eprintln!("bankml serve: launched {} (pid {}) on {upstream}; waiting for /health", bin.display(), c.id());
            Some(ChildGuard(c))
        }
        None => None,
    };
    wait_healthy(&upstream, if cfg.spawn.is_some() { 900 } else { 10 })?;
    let served = upstream_model_path(&upstream)?;
    let served_c = Path::new(&served).canonicalize().map_err(|e| format!("upstream serves {served}, which is not readable here: {e}"))?;
    if served_c != model {
        return Err(format!("upstream {upstream} serves {}, not the verified {}: refused", served_c.display(), model.display()));
    }
    let hashed_at = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let engine = "llama.cpp b11192 llama-server (loopback), behind bankml P0".to_string();
    let st = Arc::new(State { verified, model, upstream, engine, hashed_at });
    let l = TcpListener::bind(&cfg.listen).map_err(|e| format!("cannot listen on {}: {e}", cfg.listen))?;
    eprintln!("bankml serve {}: {} verified (sha256 {}), upstream {} serves it; listening on http://{}",
        crate::VERSION, st.model.display(), st.verified.model_sha256, st.upstream, cfg.listen);
    for c in l.incoming().flatten() {
        let st = st.clone();
        std::thread::spawn(move || {
            let _ = handle(c, &st);
        });
    }
    Ok(())
}

struct ChildGuard(std::process::Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}

fn wait_healthy(up: &str, secs: u64) -> Result<(), String> {
    let t = Instant::now();
    loop {
        if let Ok((200, _, _)) = request(up, "GET", "/health", b"") {
            return Ok(());
        }
        if t.elapsed() > Duration::from_secs(secs) {
            return Err(format!("upstream {up} not healthy after {secs} s"));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn upstream_model_path(up: &str) -> Result<String, String> {
    let (code, _, body) = request(up, "GET", "/props", b"").map_err(|e| format!("upstream {up}: {e}"))?;
    let v = Json::parse(&String::from_utf8_lossy(&body)).ok_or(format!("upstream {up}/props: not JSON (HTTP {code})"))?;
    v.get("model_path").and_then(Json::as_str).map(str::to_string).ok_or(format!("upstream {up}/props names no model_path"))
}

// ---------------------------------------------------------------- HTTP (just enough) -----------

type Headers = Vec<(String, String)>;

/// One request to the upstream, whole response: (status, headers lowercased, body de-chunked).
fn request(up: &str, method: &str, path: &str, body: &[u8]) -> std::io::Result<(u16, Headers, Vec<u8>)> {
    let mut s = TcpStream::connect(up)?;
    s.set_read_timeout(Some(Duration::from_secs(3600)))?;
    write!(s, "{method} {path} HTTP/1.1\r\nHost: {up}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())?;
    s.write_all(body)?;
    let mut r = BufReader::new(s);
    let (code, h) = read_head(&mut r)?;
    let mut out = Vec::new();
    let mut b = Body::new(&h);
    while let Some(chunk) = b.next(&mut r)? {
        out.extend(chunk);
    }
    Ok((code, h, out))
}

fn read_head(r: &mut impl BufRead) -> std::io::Result<(u16, Headers)> {
    let mut line = String::new();
    r.read_line(&mut line)?;
    let code = line.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let mut h = Vec::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            h.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    Ok((code, h))
}

fn header<'a>(h: &'a [(String, String)], k: &str) -> Option<&'a str> {
    h.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

/// A response body: chunked, sized, or until close.
enum Body {
    Chunked,
    Len(usize),
    Close,
    Done,
}

impl Body {
    fn new(h: &[(String, String)]) -> Self {
        if header(h, "transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked")) {
            Body::Chunked
        } else if let Some(n) = header(h, "content-length").and_then(|v| v.parse().ok()) {
            Body::Len(n)
        } else {
            Body::Close
        }
    }
    fn next(&mut self, r: &mut impl BufRead) -> std::io::Result<Option<Vec<u8>>> {
        match *self {
            Body::Done => Ok(None),
            Body::Len(n) => {
                let mut v = vec![0; n];
                r.read_exact(&mut v)?;
                *self = Body::Done;
                Ok(Some(v))
            }
            Body::Close => {
                let mut v = vec![0; 8192];
                let n = r.read(&mut v)?;
                v.truncate(n);
                Ok(if n == 0 { None } else { Some(v) })
            }
            Body::Chunked => {
                let mut line = String::new();
                r.read_line(&mut line)?;
                let n = usize::from_str_radix(line.trim().split(';').next().unwrap_or(""), 16).unwrap_or(0);
                if n == 0 {
                    *self = Body::Done;
                    return Ok(None);
                }
                let mut v = vec![0; n + 2];
                r.read_exact(&mut v)?;
                v.truncate(n);
                Ok(Some(v))
            }
        }
    }
}

fn respond(c: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Bad Gateway",
    };
    write!(c, "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())?;
    c.write_all(body)
}

fn handle(mut c: TcpStream, st: &State) -> std::io::Result<()> {
    let mut r = BufReader::new(c.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut it = line.split_whitespace();
    let (method, path) = (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string());
    let (_, h) = read_head_rest(&mut r)?;
    let n: usize = header(&h, "content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    if n > 16 << 20 {
        return respond(&mut c, 400, "text/plain", b"request too large");
    }
    let mut body = vec![0; n];
    r.read_exact(&mut body)?;
    match (method.as_str(), path.as_str()) {
        ("GET", "/bankml") => {
            let j = format!(
                "{{\"verified\": {}, \"model\": {}, \"upstream\": {}, \"engine\": {}, \"hashed_at\": {}}}",
                st.verified.to_json(), crate::gguf::jstr(&st.model.to_string_lossy()), crate::gguf::jstr(&st.upstream),
                crate::gguf::jstr(&st.engine), st.hashed_at
            );
            respond(&mut c, 200, "application/json", j.as_bytes())
        }
        ("GET", "/health" | "/v1/models" | "/props") => match request(&st.upstream, "GET", &path, b"") {
            Ok((code, _, b)) => respond(&mut c, code, "application/json", &b),
            Err(e) => respond(&mut c, 502, "text/plain", format!("upstream: {e}").as_bytes()),
        },
        ("POST", "/v1/chat/completions") => chat(&mut c, st, &body),
        _ => respond(&mut c, 404, "text/plain", b"bankml serve: GET /bankml /health /v1/models, POST /v1/chat/completions"),
    }
}

/// Headers after a request line already consumed.
fn read_head_rest(r: &mut impl BufRead) -> std::io::Result<(u16, Headers)> {
    let mut h = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            h.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    Ok((0, h))
}

struct Tally {
    t0: Instant,
    ttft: Option<Duration>,
    text: String,
    prompt: u64,
    completion: u64,
}

impl Tally {
    fn receipt(&self, st: &State) -> String {
        let h = crate::sha256::Sha256::default();
        let mut h = h;
        h.update(self.text.as_bytes());
        format!(
            "{{\"bankml\": \"{}\", \"engine\": {}, \"model_sha256\": \"{}\", \"guard\": \"{}\", \"prompt_tokens\": {}, \"completion_tokens\": {}, \"ttft_ms\": {}, \"wall_ms\": {}, \"response_sha256\": \"{}\"}}",
            crate::VERSION, crate::gguf::jstr(&st.engine), st.verified.model_sha256, st.verified.guard, self.prompt, self.completion,
            self.ttft.map(|d| d.as_millis().to_string()).unwrap_or("null".into()), self.t0.elapsed().as_millis(), crate::sha256::hex(&h.finish())
        )
    }
    fn usage(&mut self, v: &Json) {
        if let Some(u) = v.get("usage") {
            self.prompt = u.get("prompt_tokens").and_then(Json::as_u64).unwrap_or(self.prompt);
            self.completion = u.get("completion_tokens").and_then(Json::as_u64).unwrap_or(self.completion);
        }
    }
}

fn chat(c: &mut TcpStream, st: &State, body: &[u8]) -> std::io::Result<()> {
    let Some(req) = Json::parse(&String::from_utf8_lossy(body)) else {
        return respond(c, 400, "text/plain", b"body is not JSON");
    };
    let stream = req.get("stream").and_then(Json::as_bool).unwrap_or(false);
    let mut t = Tally { t0: Instant::now(), ttft: None, text: String::new(), prompt: 0, completion: 0 };
    let mut up = match TcpStream::connect(&st.upstream) {
        Ok(s) => s,
        Err(e) => return respond(c, 502, "text/plain", format!("upstream {}: {e}", st.upstream).as_bytes()),
    };
    up.set_read_timeout(Some(Duration::from_secs(3600)))?;
    write!(up, "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", st.upstream, body.len())?;
    up.write_all(body)?;
    let mut r = BufReader::new(up);
    let (code, h) = read_head(&mut r)?;
    let mut b = Body::new(&h);
    if !stream || code != 200 {
        let mut out = Vec::new();
        while let Some(ch) = b.next(&mut r)? {
            out.extend(ch);
        }
        if code != 200 {
            return respond(c, code, "application/json", &out);
        }
        let s = String::from_utf8_lossy(&out).into_owned();
        let Some(v) = Json::parse(&s) else { return respond(c, 502, "text/plain", b"upstream answer is not JSON") };
        t.text = v.get("choices").and_then(|a| a.idx(0)).and_then(|c| c.get("message")).and_then(|m| m.get("content")).and_then(Json::as_str).unwrap_or("").to_string();
        t.usage(&v);
        let end = s.rfind('}').unwrap_or(s.len());
        let merged = format!("{}, \"bankml_receipt\": {}}}", &s[..end], t.receipt(st));
        return respond(c, 200, "application/json", merged.as_bytes());
    }
    write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    let mut pending = Vec::new();
    while let Some(ch) = b.next(&mut r)? {
        pending.extend(ch);
        while let Some(i) = pending.iter().position(|&x| x == b'\n') {
            let line: Vec<u8> = pending.drain(..=i).collect();
            let s = String::from_utf8_lossy(&line);
            let s = s.trim_end();
            if s == "data: [DONE]" {
                write!(c, "data: {{\"bankml_receipt\": {}}}\n\n", t.receipt(st))?;
            } else if let Some(v) = s.strip_prefix("data:").and_then(|j| Json::parse(j.trim())) {
                if let Some(piece) = v.get("choices").and_then(|a| a.idx(0)).and_then(|c| c.get("delta")).and_then(|d| d.get("content")).and_then(Json::as_str) {
                    if !piece.is_empty() && t.ttft.is_none() {
                        t.ttft = Some(t.t0.elapsed());
                    }
                    t.text.push_str(piece);
                }
                t.usage(&v);
            }
            c.write_all(&line)?;
            c.flush()?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- JSON (just enough) -----------

/// A JSON value; numbers kept as f64 (token counts are small integers).
#[derive(Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(s: &str) -> Option<Json> {
        let mut p = P { b: s.as_bytes(), i: 0, depth: 0 };
        let v = p.value()?;
        p.ws();
        (p.i == p.b.len()).then_some(v)
    }
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn idx(&self, i: usize) -> Option<&Json> {
        match self {
            Json::Arr(v) => v.get(i),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
            _ => None,
        }
    }
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
    depth: u32,
}

impl P<'_> {
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
    fn value(&mut self) -> Option<Json> {
        self.ws();
        self.depth += 1;
        if self.depth > 128 {
            return None;
        }
        let v = match *self.b.get(self.i)? {
            b'n' => self.eat("null").then_some(Json::Null),
            b't' => self.eat("true").then_some(Json::Bool(true)),
            b'f' => self.eat("false").then_some(Json::Bool(false)),
            b'"' => self.string().map(Json::Str),
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.eat("]") {
                    Some(Json::Arr(v))
                } else {
                    loop {
                        v.push(self.value()?);
                        self.ws();
                        if self.eat("]") {
                            break Some(Json::Arr(v));
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
                    Some(Json::Obj(v))
                } else {
                    loop {
                        self.ws();
                        let k = self.string()?;
                        self.ws();
                        if !self.eat(":") {
                            break None;
                        }
                        v.push((k, self.value()?));
                        self.ws();
                        if self.eat("}") {
                            break Some(Json::Obj(v));
                        }
                        if !self.eat(",") {
                            break None;
                        }
                    }
                }
            }
            _ => {
                let s = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                std::str::from_utf8(&self.b[s..self.i]).ok()?.parse().ok().map(Json::Num)
            }
        };
        self.depth -= 1;
        v
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
                            let cp = if (0xd800..0xdc00).contains(&hi) && self.eat("\\u") {
                                let lo = self.hex4()?;
                                0x10000 + ((hi - 0xd800) << 10) + (lo.wrapping_sub(0xdc00) & 0x3ff)
                            } else {
                                hi
                            };
                            char::from_u32(cp).unwrap_or('\u{fffd}')
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_parses_what_llama_server_sends() {
        let v = Json::parse(r#"{"choices":[{"delta":{"content":"Savé \"q\"\n😀"}}],"usage":{"prompt_tokens":12,"completion_tokens":3},"x":[1,-2.5e1,true,null]}"#).unwrap();
        let piece = v.get("choices").and_then(|a| a.idx(0)).and_then(|c| c.get("delta")).and_then(|d| d.get("content")).and_then(Json::as_str);
        assert_eq!(piece, Some("Savé \"q\"\n😀"));
        assert_eq!(v.get("usage").and_then(|u| u.get("completion_tokens")).and_then(Json::as_u64), Some(3));
        assert_eq!(v.get("x").and_then(|a| a.idx(1)), Some(&Json::Num(-25.0)));
        assert!(Json::parse("{\"a\":1,}").is_none() && Json::parse("[1] x").is_none() && Json::parse(&"[".repeat(1000)).is_none());
    }

    #[test]
    fn chunked_and_sized_bodies() {
        let raw = b"5\r\nhello\r\n6;x=y\r\n world\r\n0\r\n\r\n";
        let mut r = BufReader::new(&raw[..]);
        let mut b = Body::Chunked;
        let mut out = Vec::new();
        while let Some(c) = b.next(&mut r).unwrap() {
            out.extend(c);
        }
        assert_eq!(out, b"hello world");
        let h = vec![("content-length".to_string(), "3".to_string())];
        assert!(matches!(Body::new(&h), Body::Len(3)));
    }
}
