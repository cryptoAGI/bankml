//! `bankml` — the binary. Today: the guard and the pin (P1). Later: `serve` (P0/P3/P4).
//! Exit codes follow gguf_guard.py: 0 play · 2 refuse · 3 need_more · 1 usage/io.

use bankml::{gguf, sha256, Verdict};
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| a.iter().any(|x| x == f);
    let opt = |f: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1)).cloned();
    let code = match (a.first().map(String::as_str), a.get(1)) {
        (Some("guard"), Some(file)) => {
            let engine = if opt("--engine").as_deref() == Some("prism") { gguf::Engine::Prism } else { gguf::Engine::Mainline };
            match gguf::guard_file(Path::new(file), engine) {
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
            }
        }
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
        (Some("pin"), Some(file)) => {
            let fork = opt("--fork").and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
            match bankml::pin(Path::new(file), &fork) {
                Ok(h) => {
                    println!("pinned {h}");
                    0
                }
                Err(e) => {
                    println!("refuse: {e}");
                    2
                }
            }
        }
        _ => {
            eprintln!("usage: bankml guard FILE [--engine mainline|prism] [--json]\n       bankml sha256 FILE\n       bankml pin FILE --fork FORK.json");
            1
        }
    };
    std::process::exit(code);
}
