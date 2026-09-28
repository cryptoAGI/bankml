//! `bankml` — the binary. Today: the guard, the pin, and `verify` (both, as one gate) — P1.
//! Later: `serve` (P0/P3/P4).
//! Exit codes follow gguf_guard.py: 0 play · 2 refuse · 3 need_more · 1 usage/io.

use bankml::{gguf, sha256, Verdict};
use std::path::Path;

const USAGE: &str = "usage: bankml guard FILE [--engine mainline|prism] [--json]
       bankml sha256 FILE
       bankml pin FILE --fork FORK.json
       bankml verify FILE --fork FORK.json [--engine mainline|prism] [--json]
       bankml serve FILE --fork FORK.json [--upstream HOST:PORT | --spawn LLAMA_SERVER] [--listen HOST:PORT] [--threads N] [--ctx N]
       bankml version";

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| a.iter().any(|x| x == f);
    let opt = |f: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1)).cloned();
    let engine = if opt("--engine").as_deref() == Some("prism") { gguf::Engine::Prism } else { gguf::Engine::Mainline };
    // an unreadable FORK.json is an io error (1), not "unpinned" (2): 0.0.1 conflated the two
    let fork = || -> Result<String, i32> {
        let p = opt("--fork").ok_or_else(|| {
            eprintln!("{USAGE}");
            1
        })?;
        std::fs::read_to_string(&p).map_err(|e| {
            eprintln!("bankml: cannot read {p}: {e}");
            1
        })
    };
    let code = match (a.first().map(String::as_str), a.get(1)) {
        (Some("version" | "--version" | "-V"), _) => {
            println!("bankml {}", bankml::VERSION);
            0
        }
        (Some("guard"), Some(file)) => match gguf::guard_file(Path::new(file), engine) {
            Err(e) => {
                eprintln!("bankml guard: cannot read {file}: {e}");
                1
            }
            Ok(r) => {
                if flag("--json") {
                    println!("{}", r.to_json());
                } else {
                    let t: Vec<String> = r.types.iter().map(|(n, c)| format!("{n}:{c}")).collect();
                    println!("{} {}", r.verdict_str(), t.join(" "));
                    if let Verdict::Refuse(w) = &r.verdict {
                        w.iter().for_each(|w| println!("  - {w}"));
                    }
                }
                match r.verdict {
                    Verdict::Play => 0,
                    Verdict::Refuse(_) => 2,
                    Verdict::NeedMore(_) => 3,
                }
            }
        },
        (Some("sha256"), Some(file)) => match sha256::file_hex(Path::new(file)) {
            Ok(h) => {
                println!("{h}  {file}");
                0
            }
            Err(e) => {
                eprintln!("bankml sha256: {e}");
                1
            }
        },
        (Some("pin"), Some(file)) => match fork() {
            Err(c) => c,
            Ok(fork) => match bankml::pin(Path::new(file), &fork) {
                Ok(h) => {
                    println!("pinned {h}");
                    0
                }
                Err(e) => {
                    println!("refuse: {e}");
                    2
                }
            },
        },
        (Some("verify"), Some(file)) => match fork() {
            Err(c) => c,
            Ok(fork) => match bankml::verify(Path::new(file), &fork, engine) {
                Ok(v) => {
                    println!("{}", if flag("--json") { v.to_json() } else { format!("play pinned {}", v.model_sha256) });
                    0
                }
                Err(e) => {
                    if flag("--json") {
                        println!("{{\"verdict\": \"refuse\", \"reason\": {}, \"bankml\": \"{}\"}}", gguf::jstr(&e), bankml::VERSION);
                    } else {
                        println!("refuse: {e}");
                    }
                    match e.split(' ').next() {
                        Some("cannot") => 1,
                        _ if e.starts_with("guard needs") => 3,
                        _ => 2,
                    }
                }
            },
        },
        (Some("serve"), Some(file)) => match fork() {
            Err(c) => c,
            Ok(fork_json) => {
                let cfg = bankml::serve::Config {
                    model: file.into(),
                    fork_json,
                    engine,
                    listen: opt("--listen").unwrap_or("127.0.0.1:18093".into()),
                    upstream: opt("--upstream").unwrap_or("127.0.0.1:18092".into()),
                    spawn: opt("--spawn").map(Into::into),
                    threads: opt("--threads").and_then(|v| v.parse().ok()).unwrap_or(3),
                    ctx: opt("--ctx").and_then(|v| v.parse().ok()).unwrap_or(4096),
                };
                match bankml::serve::run(cfg) {
                    Ok(()) => 0,
                    Err(e) => {
                        eprintln!("bankml serve: refuse: {e}");
                        2
                    }
                }
            }
        },
        _ => {
            eprintln!("{USAGE}");
            1
        }
    };
    std::process::exit(code);
}
