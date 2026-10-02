// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end tests of the `bankml` binary: what an operator types, what comes back, which exit code.
//! Offline — synthetic GGUFs and a mock llama-server; the real-model oracles are in the modules (see
//! testing/README.md). `cargo test --release --test cli`

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bankml(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bankml")).args(args).output().expect("run bankml")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

/// A scratch directory for this test process. Directories left by earlier runs (their process is gone) are removed
/// first: the fixtures are synthetic GGUFs of several MB each, and 234 runs had left 1.2 GB behind.
fn dir(tag: &str) -> PathBuf {
    static SWEPT: std::sync::Once = std::sync::Once::new();
    SWEPT.call_once(|| {
        for e in std::fs::read_dir(std::env::temp_dir()).into_iter().flatten().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let pid = name.strip_prefix("bankml-cli-").and_then(|r| r.rsplit('-').next()).and_then(|p| p.parse::<u32>().ok());
            if let Some(pid) = pid {
                if pid != std::process::id() && !std::path::Path::new(&format!("/proc/{pid}")).exists() {
                    let _ = std::fs::remove_dir_all(e.path());
                }
            }
        }
    });
    let d = std::env::temp_dir().join(format!("bankml-cli-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn s(x: &str) -> Vec<u8> {
    let mut v = (x.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(x.as_bytes());
    v
}

/// A GGUF v3 with the given string keys and tensors (name, elements, type, bytes per block, elems per block).
fn gguf(keys: &[(&str, &str)], tensors: &[(&str, u64, u32, u64, u64)]) -> Vec<u8> {
    let mut h = b"GGUF".to_vec();
    h.extend(3u32.to_le_bytes());
    h.extend((tensors.len() as u64).to_le_bytes());
    h.extend((keys.len() as u64).to_le_bytes());
    for (k, v) in keys {
        h.extend(s(k));
        h.extend(8u32.to_le_bytes());
        h.extend(s(v));
    }
    let mut off = 0u64;
    for &(n, ne, ty, bb, per) in tensors {
        h.extend(s(n));
        h.extend(1u32.to_le_bytes());
        h.extend(ne.to_le_bytes());
        h.extend(ty.to_le_bytes());
        h.extend(off.to_le_bytes());
        off += (ne / per * bb).div_ceil(32) * 32;
    }
    h.resize(h.len().div_ceil(32) * 32 + off as usize, 0);
    h
}

fn write(d: &Path, name: &str, b: &[u8]) -> String {
    let p = d.join(name);
    std::fs::write(&p, b).unwrap();
    p.to_string_lossy().into_owned()
}

fn sha(p: &str) -> String {
    out(&bankml(&["sha256", p])).split_whitespace().next().unwrap().to_string()
}

#[test]
fn version_and_usage() {
    let v = bankml(&["version"]);
    assert_eq!((code(&v), out(&v).trim().to_string()), (0, format!("bankml {}", env!("CARGO_PKG_VERSION"))));
    assert_eq!(code(&bankml(&[])), 1);
    assert_eq!(code(&bankml(&["nonsense"])), 1);
}

#[test]
fn guard_verdicts_and_exit_codes() {
    let d = dir("guard");
    let q2 = [("general.architecture", "qwen3"), ("general.name", "Ternary-Bonsai-1.7B")];
    let good = write(&d, "good.gguf", &gguf(&q2, &[("a", 4096, 42, 18, 64)]));
    let legacy = write(&d, "legacy.gguf", &gguf(&q2, &[("a", 4096, 42, 34, 128), ("b", 4096, 42, 34, 128)]));
    let b2 = write(&d, "b2.gguf", &gguf(&[("general.name", "Ternary Bonsai 2 27B")], &[("a", 4096, 42, 18, 64)]));
    let trunc = write(&d, "trunc.gguf", &gguf(&q2, &[("a", 4096, 42, 18, 64)])[..40]);
    let g = bankml(&["guard", &good, "--json"]);
    assert_eq!(code(&g), 0);
    assert!(out(&g).contains("\"verdict\": \"play\"") && out(&g).contains("\"Q2_0\": 1"), "{}", out(&g));
    let l = bankml(&["guard", &legacy]);
    assert_eq!(code(&l), 2);
    assert!(out(&l).contains("legacy group-128"), "{}", out(&l));
    assert_eq!(code(&bankml(&["guard", &b2])), 2);
    assert_eq!(code(&bankml(&["guard", &b2, "--engine", "prism"])), 0);
    assert_eq!(code(&bankml(&["guard", &trunc])), 3);
    assert_eq!(code(&bankml(&["guard", &d.join("missing.gguf").to_string_lossy()])), 1);
}

/// 0.0.1 aborted with a stack overflow on this file (exit 134); it must refuse.
#[test]
fn hostile_headers_refuse_not_crash() {
    let d = dir("hostile");
    let mut h = b"GGUF".to_vec();
    h.extend(3u32.to_le_bytes());
    h.extend(0u64.to_le_bytes());
    h.extend(1u64.to_le_bytes());
    h.extend(s("a"));
    h.extend(9u32.to_le_bytes());
    for _ in 0..2_000_000 {
        h.extend(9u32.to_le_bytes());
        h.extend(1u64.to_le_bytes());
    }
    h.extend(4u32.to_le_bytes());
    h.extend(0u64.to_le_bytes());
    let nested = write(&d, "nested.gguf", &h);
    let o = bankml(&["guard", &nested]);
    assert_eq!(code(&o), 2, "{}", out(&o));
    assert!(out(&o).contains("nested deeper"));
}

#[test]
fn pin_and_verify() {
    let d = dir("verify");
    let m = write(&d, "m.gguf", &gguf(&[("general.architecture", "qwen3")], &[("a", 4096, 41, 18, 128)]));
    let h = sha(&m);
    let fork = write(&d, "FORK.json", format!("{{\"files\": [{{\"path\": \"m.gguf\", \"sha256\": \"{h}\"}}]}}").as_bytes());
    let wrong = write(&d, "WRONG.json", format!("{{\"files\": [{{\"path\": \"m.gguf\", \"sha256\": \"{}\"}}]}}", "0".repeat(64)).as_bytes());
    let p = bankml(&["pin", &m, "--fork", &fork]);
    assert_eq!((code(&p), out(&p).trim().to_string()), (0, format!("pinned {h}")));
    assert_eq!(code(&bankml(&["pin", &m, "--fork", &wrong])), 2);
    assert_eq!(code(&bankml(&["pin", &m, "--fork", &d.join("none.json").to_string_lossy()])), 1); // io, not "unpinned"
    let v = bankml(&["verify", &m, "--fork", &fork, "--json"]);
    assert_eq!(code(&v), 0);
    assert!(out(&v).contains(&format!("\"model_sha256\": \"{h}\"")) && out(&v).contains("\"verdict\": \"play\""));
    let w = bankml(&["verify", &m, "--fork", &wrong, "--json"]);
    assert_eq!(code(&w), 2);
    assert!(out(&w).contains("\"verdict\": \"refuse\"") && out(&w).contains("!= pinned"));
}

// ---------------------------------------------------------------- serve, against a mock llama-server

/// A mock llama-server: /health, /props naming `model`, and a chat endpoint that streams (chunked) or not.
fn mock_upstream(model: String) -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        for c in l.incoming().flatten() {
            let model = model.clone();
            std::thread::spawn(move || mock_one(c, &model));
        }
    });
    addr
}

fn mock_one(mut c: TcpStream, model: &str) {
    let mut r = BufReader::new(c.try_clone().unwrap());
    let mut line = String::new();
    r.read_line(&mut line).unwrap();
    let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
    let mut n = 0;
    loop {
        let mut h = String::new();
        r.read_line(&mut h).unwrap();
        if h == "\r\n" || h.is_empty() {
            break;
        }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
            n = v.trim().parse().unwrap();
        }
    }
    let mut body = vec![0; n];
    r.read_exact(&mut body).unwrap();
    let json = |b: &str| format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len());
    let reply = match path.as_str() {
        "/health" => json("{\"status\":\"ok\"}"),
        "/props" => json(&format!("{{\"model_path\":{:?}}}", model)),
        _ if String::from_utf8_lossy(&body).contains("\"stream\": true") || String::from_utf8_lossy(&body).contains("\"stream\":true") => {
            let events = [
                "data: {\"choices\":[{\"delta\":{\"content\":\"Savante \"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"knows.\"}}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2}}\n\n",
                "data: [DONE]\n\n",
            ];
            let mut s = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_string();
            for e in events {
                s += &format!("{:x}\r\n{e}\r\n", e.len());
            }
            s + "0\r\n\r\n"
        }
        _ => json("{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Savante knows.\"}}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2}}"),
    };
    let _ = c.write_all(reply.as_bytes());
}

fn http(addr: &str, method: &str, path: &str, body: &str) -> String {
    http_h(addr, method, path, body, "Host: 127.0.0.1\r\nContent-Type: application/json\r\n")
}

fn http_h(addr: &str, method: &str, path: &str, body: &str, headers: &str) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(std::time::Duration::from_secs(20))).unwrap();
    write!(s, "{method} {path} HTTP/1.1\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut o = String::new();
    s.read_to_string(&mut o).unwrap();
    o
}

fn free_port() -> String {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string()
}

/// The serve under test, killed when the test ends — pass or panic (a leaked child keeps the test's pipes open).
struct Serve(std::process::Child);
impl Drop for Serve {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn start_serve(m: &str, fork: &str, upstream: &str) -> (Serve, String) {
    let listen = free_port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_bankml"))
        .args(["serve", m, "--fork", fork, "--upstream", upstream, "--listen", &listen])
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    for _ in 0..100 {
        if TcpStream::connect(&listen).is_ok() {
            return (Serve(child), listen);
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
    panic!("serve did not start");
}

fn sha_hex(b: &[u8]) -> String {
    let mut h = bankml::sha256::Sha256::default();
    h.update(b);
    bankml::sha256::hex(&h.finish())
}

#[test]
fn serve_gates_and_signs_answers() {
    let d = dir("serve");
    let m = write(&d, "m.gguf", &gguf(&[("general.architecture", "qwen3")], &[("a", 4096, 41, 18, 128)]));
    let h = sha(&m);
    let fork = write(&d, "FORK.json", format!("{{\"files\": [{{\"path\": \"m.gguf\", \"sha256\": \"{h}\"}}]}}").as_bytes());
    let canon = Path::new(&m).canonicalize().unwrap().to_string_lossy().into_owned();
    let up = mock_upstream(canon);
    let (child, addr) = start_serve(&m, &fork, &up);

    let st = http(&addr, "GET", "/bankml", "");
    assert!(st.contains(&h) && st.contains("\"verdict\": \"play\""), "{st}");

    let one = http(&addr, "POST", "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    assert!(one.contains("\"bankml_receipt\"") && one.contains(&format!("\"model_sha256\": \"{h}\"")), "{one}");
    assert!(one.contains(&format!("\"response_sha256\": \"{}\"", sha_hex(b"Savante knows."))), "{one}");
    assert!(one.contains("\"completion_tokens\": 2"));

    let st = http(&addr, "POST", "/v1/chat/completions", "{\"stream\": true, \"messages\":[]}");
    let receipt = st.lines().find(|l| l.contains("bankml_receipt")).unwrap_or_else(|| panic!("{st}"));
    assert!(receipt.contains(&format!("\"response_sha256\": \"{}\"", sha_hex(b"Savante knows."))), "{receipt}");
    assert!(st.find("bankml_receipt").unwrap() < st.find("[DONE]").unwrap(), "receipt before DONE");
    assert!(st.contains("\"content\":\"Savante \""), "the stream itself passes through untouched");
    let body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    assert!(one.contains(&format!("\"request_sha256\": \"{}\"", sha_hex(body.as_bytes()))), "the receipt names the request: {one}");
    // a web page cannot drive it: a foreign Host (DNS rebinding) and a form-style POST are refused
    let r = http_h(&addr, "POST", "/v1/chat/completions", body, "Host: evil.example\r\nContent-Type: application/json\r\n");
    assert!(r.starts_with("HTTP/1.1 403"), "{r}");
    let r = http_h(&addr, "POST", "/v1/chat/completions", body, "Host: 127.0.0.1\r\nContent-Type: text/plain\r\n");
    assert!(r.starts_with("HTTP/1.1 415"), "{r}");
    // the model file changes after verification: no answer, no receipt
    std::fs::write(&m, b"replaced weights").unwrap();
    let r = http(&addr, "POST", "/v1/chat/completions", body);
    assert!(r.starts_with("HTTP/1.1 503") && !r.contains("bankml_receipt"), "{r}");
    drop(child);
}

#[test]
fn serve_refuses_an_unverified_model_or_a_different_upstream_file() {
    let d = dir("serve-refuse");
    let m = write(&d, "m.gguf", &gguf(&[("general.architecture", "qwen3")], &[("a", 4096, 41, 18, 128)]));
    let other = write(&d, "other.gguf", b"GGUF");
    let h = sha(&m);
    let fork = write(&d, "FORK.json", format!("{{\"files\": [{{\"path\": \"m.gguf\", \"sha256\": \"{h}\"}}]}}").as_bytes());
    let wrong = write(&d, "WRONG.json", format!("{{\"files\": [{{\"path\": \"m.gguf\", \"sha256\": \"{}\"}}]}}", "1".repeat(64)).as_bytes());
    let up = mock_upstream(Path::new(&other).canonicalize().unwrap().to_string_lossy().into_owned());
    let run = |f: &str| {
        let o = Command::new(env!("CARGO_BIN_EXE_bankml")).args(["serve", &m, "--fork", f, "--upstream", &up, "--listen", &free_port()]).output().unwrap();
        (code(&o), String::from_utf8_lossy(&o.stderr).into_owned() + &out(&o))
    };
    let (c, msg) = run(&wrong);
    assert_eq!(c, 2, "{msg}");
    assert!(msg.contains("!= pinned"), "{msg}");
    let (c, msg) = run(&fork);
    assert_eq!(c, 2, "{msg}");
    assert!(msg.contains("not the verified"), "{msg}");
}

/// O5: `bankml create` over a pinned (header-only) GGUF, by name and by path; the manifest; the refusals.
#[test]
fn create_writes_a_layer_over_a_pin() {
    let d = dir("create");
    let reg = d.join("forks");
    std::fs::create_dir_all(&reg).unwrap();
    let base = write(&reg, "base-F16.gguf", &gguf(&[("general.architecture", "llama"), ("tokenizer.chat_template", "{{ chatml }}")], &[]));
    std::fs::write(reg.join("base-F16.gguf.FORK.json"), format!("{{\"files\": [{{\"path\": \"base-F16.gguf\", \"sha256\": \"{}\"}}]}}", sha(&base))).unwrap();
    let mf = d.join("Modelfile");
    std::fs::write(&mf, "FROM base-f16\nSYSTEM \"\"\"You are mindX.\"\"\"\nPARAMETER stop <|im_end|>\nPARAMETER num_ctx 2048\n").unwrap();
    let r = reg.to_str().unwrap();
    let o = bankml(&["create", "persona", "-f", mf.to_str().unwrap(), "--registry", r]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    assert!(out(&o).lines().last() == Some("success") && out(&o).contains(&format!("using existing layer sha256:{}", sha(&base))), "{}", out(&o));
    let m = std::fs::read_to_string(reg.join("persona.MODEL.json")).unwrap();
    assert!(m.contains("\"system\": \"You are mindX.\"") && m.contains("\"parameters\": {\"num_ctx\": 2048}") && m.contains("\"stop\": [\"<|im_end|>\"]"), "{m}");
    // by path, relative to the Modelfile
    std::fs::write(&mf, "FROM ./forks/base-F16.gguf\n").unwrap();
    assert_eq!(code(&bankml(&["create", "by-path", "-f", mf.to_str().unwrap(), "--registry", r])), 0);
    // refusals: exit 2 with the reason
    for (text, why) in [("FROM base-f16\nADAPTER /x\n", "LoRA"), ("FROM base-f16\nPARAMETER mirostat 2\n", "not reproduced"), ("FROM nothing\n", "no such model"),
                        ("SYSTEM x\n", "no FROM")] {
        std::fs::write(&mf, text).unwrap();
        let o = bankml(&["create", "x", "-f", mf.to_str().unwrap(), "--registry", r]);
        let e = String::from_utf8_lossy(&o.stderr).into_owned();
        assert!(code(&o) == 2 && e.contains(why), "{text:?}: {e}");
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// O5: `bankml convert` refuses what it would not write as llama.cpp writes it (the byte-identity oracle on real
/// weights is `convert::tests::oracle_convert_b11192`).
#[test]
fn convert_refuses_with_the_reason() {
    let d = dir("convert");
    let m = d.join("tiny-llama");
    std::fs::create_dir_all(&m).unwrap();
    let o = |args: &[&str]| {
        let r = bankml(args);
        (code(&r), String::from_utf8_lossy(&r.stderr).into_owned())
    };
    let out_gguf = d.join("out.gguf");
    let (ms, og) = (m.to_str().unwrap(), out_gguf.to_str().unwrap());
    std::fs::write(m.join("config.json"), r#"{"architectures": ["Qwen3ForCausalLM"]}"#).unwrap();
    let (c, e) = o(&["convert", ms, "-o", og]);
    assert!(c == 2 && e.contains("Llama architecture"), "{e}");
    // a tiny Llama: one BF16 tensor per role, and a tokenizer no one has measured
    std::fs::write(m.join("config.json"), r#"{"architectures": ["LlamaForCausalLM"], "hidden_size": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
        "num_hidden_layers": 1, "intermediate_size": 8, "vocab_size": 3, "tie_word_embeddings": true}"#).unwrap();
    let ts = [("model.embed_tokens.weight", vec![3u64, 4]), ("model.layers.0.input_layernorm.weight", vec![4]), ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
              ("model.layers.0.self_attn.k_proj.weight", vec![2, 4]), ("model.norm.weight", vec![4])];
    let (mut hdr, mut data, mut off) = (Vec::new(), Vec::new(), 0u64);
    for (n, sh) in &ts {
        let len = sh.iter().product::<u64>() * 2;
        hdr.push(format!("\"{n}\": {{\"dtype\": \"BF16\", \"shape\": {sh:?}, \"data_offsets\": [{off}, {}]}}", off + len));
        data.extend(std::iter::repeat_n(0x3fu8, len as usize));
        off += len;
    }
    let h = format!("{{{}}}", hdr.join(", "));
    let mut st = (h.len() as u64).to_le_bytes().to_vec();
    st.extend(h.as_bytes());
    st.extend(data);
    std::fs::write(m.join("model.safetensors"), st).unwrap();
    std::fs::write(m.join("tokenizer.json"), r#"{"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}, "added_tokens": [], "pre_tokenizer": null}"#).unwrap();
    let (c, e) = o(&["convert", ms, "-o", og]);
    assert!(c == 2 && e.contains("tokenizer not recognised"), "{e}");
    assert!(!out_gguf.exists());
    std::fs::write(m.join("README.md"), "---\nlicense: mit\n---\n").unwrap();
    let (c, e) = o(&["convert", ms, "-o", og]);
    assert!(c == 2 && e.contains("README.md"), "{e}");
    let (c, e) = o(&["convert", ms, "-o", og, "--outtype", "q8_0"]);
    assert!(c == 2 && e.contains("f16 only"), "{e}");
    let _ = std::fs::remove_dir_all(&d);
}
