// SPDX-License-Identifier: MIT OR Apache-2.0
//! `bankml` — the binary. Today: the guard, the pin, and `verify` (both, as one gate) — P1.
//! Later: `serve` (P0/P3/P4).
//! Exit codes follow gguf_guard.py: 0 play · 2 refuse · 3 need_more · 1 usage/io.

use bankml::{gguf, sha256, Verdict};
use std::path::Path;

const USAGE: &str = "usage: bankml usage [PID …]
       bankml guard FILE [--engine mainline|prism] [--json]
       bankml sha256 FILE
       bankml pin FILE --fork FORK.json
       bankml verify FILE --fork FORK.json [--engine mainline|prism] [--json]
       bankml serve FILE --fork FORK.json [--upstream HOST:PORT | --spawn LLAMA_SERVER] [--listen HOST:PORT] [--threads N] [--ctx N] [--spec-ngram] [--slot-dir DIR]
       bankml serve FILE --fork FORK.json --native [--listen HOST:PORT] [--upstream HOST:PORT] [--ctx N] [--registry [DIR]] [--keep-alive DUR]
                                                              (answers from bankML's own forward pass; also serves the engine address;
                                                              OpenAI /v1 and Ollama /api; --registry: every model pinned in DIR,
                                                              default ~/.local/share/bankml/forks, by name, one resident at a time)
       bankml tokenize MODEL.gguf [--no-special] < text      (token ids, as llama.cpp's /tokenize)
       bankml chat-template MODEL.gguf < messages.json        (the prompt, as llama.cpp's /apply-template)
       bankml generate MODEL.gguf [--max N] [--json] [--sample [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S]] < messages.json|text
                                                              (bankml's own forward pass: greedy, or llama-server's sampler chain;
                                                              --json: JSON mode, llama-server's response_format json_object)
       bankml create NAME -f Modelfile [--registry DIR] [--models DIR]
                                                              (O5: a derived model — FROM a pinned model, a pinned GGUF or a safetensors
                                                              directory, + SYSTEM, PARAMETER, stop, MESSAGE, LICENSE — written as
                                                              DIR/NAME.MODEL.json over the base's pin; no weights copied)
       bankml convert SAFETENSORS_DIR -o OUT.gguf [--outtype f16] [--model-name NAME] [--fork FORK.json --source SRC] [--ignore-model-card]
                                                              (Llama → GGUF F16, byte-identical to llama.cpp b11192's convert_hf_to_gguf.py
                                                              --outtype f16; --fork writes the FORK.json that pins the result)
       bankml gpu [--remote | --verify]                        (every video card found, and which bankml will use;
                                                              --remote adds the GPUs Hugging Face rents, listed only;
                                                              --verify runs the bit-exact kernel oracle on each card)
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
        (Some("chat-template"), Some(file)) => {
            // P3 step two: the prompt a conversation becomes, byte-identical to llama.cpp's /apply-template
            let mut text = String::new();
            let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut text);
            let r = bankml::chat::template_of(Path::new(file)).and_then(|t| {
                let v = bankml::serve::Json::parse(&text).ok_or("stdin is not JSON")?;
                let msgs = v.get("messages").unwrap_or(&v);
                t.render(&bankml::chat::messages_from_json(msgs)?)
            });
            match r {
                Ok(p) => {
                    print!("{p}");
                    0
                }
                Err(e) => {
                    eprintln!("bankml chat-template: {e}");
                    2
                }
            }
        }
        (Some("generate"), Some(file)) => {
            // P3: bankml's own forward pass, greedy — token-identical to llama-server b11192 on its oracle (prompts
            // under 64 tokens, contexts under 512 cells); stdin is OpenAI-style messages, or plain text for one user turn
            let mut text = String::new();
            let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut text);
            let max: usize = opt("--max").and_then(|v| v.parse().ok()).unwrap_or(256);
            let sample = if flag("--sample") {
                match bankml::sampler::Params::from_gguf(Path::new(file)) {
                    Ok(mut p) => {
                        let f = |k: &str| opt(k).and_then(|v| v.parse::<f64>().ok());
                        if let Some(v) = f("--temp") { p.temp = v as f32 }
                        if let Some(v) = f("--top-k") { p.top_k = v as i32 }
                        if let Some(v) = f("--top-p") { p.top_p = v as f32 }
                        if let Some(v) = f("--min-p") { p.min_p = v as f32 }
                        if let Some(v) = f("--seed") { p.seed = v as u32 }
                        Some(p)
                    }
                    Err(e) => {
                        eprintln!("bankml generate: {e}");
                        std::process::exit(1);
                    }
                }
            } else {
                None
            };
            match generate(Path::new(file), &text, max, sample, flag("--json")) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("bankml generate: {e}");
                    2
                }
            }
        }
        (Some("create"), Some(name)) => {
            // O5: `ollama create`, natively: a layer over a pinned base, verified as the base is verified
            let Some(mf) = opt("-f").or(opt("--file")) else {
                eprintln!("bankml create: -f Modelfile is required");
                std::process::exit(1);
            };
            let dir = registry_dir(&a);
            let models: Vec<std::path::PathBuf> = opt("--models").map(Into::into).into_iter().collect();
            match bankml::create::cli(name, &mf, &dir, &models) {
                Ok(d) => {
                    eprintln!("bankml create: {} (digest sha256:{}) over {} (sha256 {})", d.name, d.digest, d.base.file, d.base.sha256);
                    0
                }
                Err(e) => {
                    eprintln!("bankml create: refuse: {e}");
                    2
                }
            }
        }
        (Some("convert"), Some(dir)) => {
            // O5: safetensors → GGUF F16, byte-identical to llama.cpp b11192's convert_hf_to_gguf.py --outtype f16
            let Some(out) = opt("-o").or(opt("--outfile")) else {
                eprintln!("bankml convert: -o OUT.gguf is required");
                std::process::exit(1);
            };
            if opt("--outtype").is_some_and(|t| t != "f16") {
                eprintln!("bankml convert: refuse: --outtype f16 only (the one byte-identical to llama.cpp that bankml has proven)");
                std::process::exit(2);
            }
            let o = bankml::convert::Options { model_name: opt("--model-name"), ignore_model_card: flag("--ignore-model-card") };
            match bankml::convert::convert(Path::new(dir), Path::new(&out), &o) {
                Ok(r) => {
                    println!("{}  {}", r.sha256, r.out.display());
                    eprintln!("bankml convert: {} — {} tensors, {} metadata keys, {} weights, {} bytes", r.name, r.tensors, r.kv, r.params, r.bytes);
                    if let Some(f) = opt("--fork") {
                        let src = opt("--source").unwrap_or_else(|| Path::new(dir).canonicalize().map(|p| p.display().to_string()).unwrap_or(dir.clone()));
                        if let Err(e) = std::fs::write(&f, bankml::convert::fork_json(&r, &src)) {
                            eprintln!("bankml convert: cannot write {f}: {e}");
                            std::process::exit(1);
                        }
                        eprintln!("bankml convert: pinned in {f}");
                    }
                    0
                }
                Err(e) => {
                    eprintln!("bankml convert: refuse: {e}");
                    2
                }
            }
        }
        (Some("gpu"), _) => {
            // the video-card component: every GPU found (Vulkan, merged with the kernel's sysfs view) and the selection
            if flag("--verify") {
                // the on-card oracle: each selected card runs bankml's kernels and must give the CPU kernels' bits
                let sel = bankml::gpu::selected(&bankml::gpu::discover().0);
                if sel.is_empty() {
                    println!("bankml gpu --verify: no usable GPU found; bankml runs on the CPU");
                }
                let mut bad = 0;
                for d in &sel {
                    match bankml::gpu::compute::Gpu::open(d.index).and_then(|g| bankml::gpu::kernels::verify_q1_0(&g)) {
                        Ok(lines) => lines.iter().for_each(|l| println!("{}: {l}", d.name)),
                        Err(e) => {
                            bad += 1;
                            println!("{}: NOT VERIFIED — {e}; bankml will not use this card", d.name);
                        }
                    }
                }
                if bad > 0 { 2 } else { 0 }
            } else {
                println!("{}", bankml::gpu::report_json(flag("--remote")));
                0
            }
        }
        (Some("tokenize"), Some(file)) => {
            // bankml's tokenizer (P3, step one): token-identical to llama.cpp b11192 on its oracle; text from stdin
            match bankml::tokenizer::Tokenizer::from_gguf(Path::new(file)) {
                Err(e) => {
                    eprintln!("bankml tokenize: {e}");
                    1
                }
                Ok(t) => {
                    let mut text = String::new();
                    let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut text);
                    let ids = t.encode(&text, !flag("--no-special"));
                    println!("{{\"tokens\": [{}]}}", ids.iter().map(u32::to_string).collect::<Vec<_>>().join(","));
                    0
                }
            }
        }
        (Some("usage"), _) => {
            // bankml's psutil: memory, cores, and rss + CPU % of the given pids (default: bankml itself), over 0.5 s
            let pids: Vec<u32> = a[1..].iter().filter_map(|x| x.parse().ok()).collect();
            if a.len() > 1 && pids.len() != a.len() - 1 {
                eprintln!("bankml usage: not a process id: {}", a[1..].iter().filter(|x| x.parse::<u32>().is_err()).cloned().collect::<Vec<_>>().join(" "));
                std::process::exit(1);
            }
            let named: Vec<(String, u32)> = if pids.is_empty() { vec![("bankml".into(), std::process::id())] } else { pids.iter().map(|p| (format!("pid {p}"), *p)).collect() };
            let refs: Vec<(&str, u32)> = named.iter().map(|(n, p)| (n.as_str(), *p)).collect();
            println!("{}", bankml::sys::usage_json(&refs, std::time::Duration::from_millis(500)));
            0
        }
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
                    spec_ngram: flag("--spec-ngram"),
                    slot_dir: opt("--slot-dir").map(Into::into),
                    native: flag("--native"),
                    // `--registry` alone means the importer's forks directory ($BANKML_FORKS, as sAGI/models.py)
                    registry: a.iter().any(|x| x == "--registry").then(|| registry_dir(&a)),
                    keep_alive: opt("--keep-alive"),
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

/// `--registry DIR`, or the importer's forks directory ($BANKML_FORKS, else ~/.local/share/bankml/forks, as sAGI/models.py).
fn registry_dir(a: &[String]) -> std::path::PathBuf {
    match a.iter().position(|x| x == "--registry").and_then(|i| a.get(i + 1)) {
        Some(d) if !d.starts_with("--") => d.into(),
        _ => std::env::var("BANKML_FORKS").map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local/share/bankml/forks")),
    }
}

/// `bankml generate`: render, tokenize, run the prompt through bankml's forward pass, then greedy tokens to stdout as
/// they come, until the turn ends or `max` tokens.
fn generate(model: &Path, input: &str, max: usize, sample: Option<bankml::sampler::Params>, json: bool) -> Result<(), String> {
    use bankml::{chat, forward::Weights, serve::Json, tokenizer::Tokenizer};
    use std::io::Write;
    let template = chat::template_of(model)?;
    let msgs = match Json::parse(input.trim()) {
        Some(v) if matches!(v, Json::Arr(_)) || v.get("messages").is_some() => chat::messages_from_json(v.get("messages").unwrap_or(&v))?,
        _ => vec![chat::Message::new("user", input.trim_end_matches('\n'))],
    };
    if json {
        // 0.3.3: JSON mode through the native engine — the grammar, its prefill and the redraw exactly as llama-server
        // answers `response_format: {"type": "json_object"}` (greedy is its temperature 0); the content is printed
        use bankml::grammar::{Constraint, ContentStream};
        let eng = bankml::native::Native::open(model, 4096)?;
        let params = sample.unwrap_or_else(|| bankml::sampler::Params { temp: 0.0, ..eng.defaults.clone() });
        let prompt = eng.tok.encode(&template.render(&msgs)?, true);
        let mut cs = ContentStream::new(&Constraint::JsonObject);
        let mut out = std::io::stdout();
        let d = eng.complete(&prompt, params, Some(max), eng.grammar(&Constraint::JsonObject)?, |piece| {
            let _ = out.write_all(cs.push(piece).as_bytes());
            let _ = out.flush();
            true
        })?;
        println!();
        eprintln!("bankml generate --json: {} prompt tokens, {} generated ({}), {} redrawn under the grammar, {:.1} ms of grammar per token",
                  d.prompt_tokens, d.completion_tokens, d.finish_reason, d.resampled, d.grammar_ns as f64 / 1e6 / d.tokens.len().max(1) as f64);
        return Ok(());
    }
    let tok = Tokenizer::from_gguf(model)?;
    let prompt = tok.encode(&template.render(&msgs)?, true);
    let w = Weights::open(model)?;
    let ends = bankml::native::eog_from_gguf(model, &tok)?;
    let mut sampler = sample.map(bankml::sampler::Sampler::new).transpose()?;
    let t0 = std::time::Instant::now();
    let mut caches = w.caches();
    let mut rn = w.prefill(&mut caches, &prompt, |_, _, _| {})?;
    let t_prompt = t0.elapsed().as_secs_f64();
    let (mut out, mut n) = (std::io::stdout(), 0);
    while n < max {
        let l = w.logits(&rn)?;
        let next = match sampler.as_mut() {
            Some(s) => s.sample(&l),
            None => l.iter().enumerate().fold(0, |b, (i, &v)| if v > l[b] { i } else { b }) as u32,
        };
        if ends.contains(&next) {
            break;
        }
        let _ = out.write_all(&tok.token_bytes(next));
        let _ = out.flush();
        n += 1;
        rn = match w.decode(&mut caches, next) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("\nbankml generate: stopping: {e}");
                break;
            }
        };
    }
    let dt = t0.elapsed().as_secs_f64() - t_prompt;
    println!();
    eprintln!("bankml generate: {} prompt tokens in {t_prompt:.1} s, {n} generated in {dt:.1} s ({:.2} tok/s)", prompt.len(), n as f64 / dt.max(1e-9));
    Ok(())
}

