//! End-to-end tests of the `bankml` binary: what an operator types, what comes back, which exit code.
//! Offline — synthetic GGUFs and a mock llama-server; the real-model oracles are in the modules (see
//! testing/README.md). `cargo test --release --test cli`

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

fn dir(tag: &str) -> PathBuf {
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
