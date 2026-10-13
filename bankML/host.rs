// SPDX-License-Identifier: MIT OR Apache-2.0
//! Where bankML runs: its hosts, and which one is running it now.
//!
//! A host is a program that loads the engine and hands it a model, threads, a GPU and a place for its files. Each
//! host is one entry in [`HOSTS`]: what it is built for, how it gives the engine those four things, and where its
//! code lives. The engine's own code is the same in every host. Only the edges differ: the GPU backend
//! ([`crate::gpu::BACKENDS`]), the thread pool's spawn, and whether `/proc` exists. To add a host, add an entry here
//! and, if it brings a new GPU API, a backend in `gpu/`.
//!
//! **Server and client stay distinct.** The server hosts (`native`, `capi`) run on the operator's machine and
//! answer others (`bankml serve`). The client hosts (`browseml`, `wasi`) run on the visitor's own machine and answer
//! only the visitor; nothing they compute leaves it. Every report names the side.
//!
//! **browseML** (github.com/browseML/browseML) is the browser host. bankML's engine runs there as WebAssembly,
//! on the visitor's CPU and (verified) GPU: the same bits as here, the model pinned by sha256, and a receipt on
//! every answer. The page names itself with `BANKML_HOST=browseml` (and `BANKML_HOST_VERSION`), so the engine's
//! reports say where they came from. Details: docs/modules/host.md.

/// One host of bankML's engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Host {
    /// Its name, as `BANKML_HOST` gives it.
    pub name: &'static str,
    /// "server" (the operator's machine: `bankml serve` answers others over HTTP) or "client" (the visitor's own
    /// machine: browseML answers the visitor in their browser, and nothing leaves it). The engine is the same; who
    /// owns the machine, who asks, and where the answer goes differ, and reports say which.
    pub side: &'static str,
    /// The Rust target(s) it is built for.
    pub target: &'static str,
    /// How the model reaches the engine.
    pub model: &'static str,
    /// How the engine gets its threads.
    pub threads: &'static str,
    /// The GPU backend it offers (`gpu::BACKENDS`), or how it does without one.
    pub gpu: &'static str,
    /// What it knows of the machine's memory.
    pub memory: &'static str,
    /// Where it lives.
    pub url: &'static str,
}

/// Every host bankML knows. Add a host here.
pub const HOSTS: &[Host] = &[
    Host {
        name: "native",
        side: "server",
        target: "x86_64 / aarch64 Linux (the bankml binary: serve, generate, the CLI)",
        model: "a GGUF file on disk, pinned by FORK.json",
        threads: "std::thread, the cores the OS reports (BANKML_THREADS)",
        gpu: "vulkan (dlopen'd, bankML's own SPIR-V)",
        memory: "/proc/meminfo",
        url: "https://github.com/cryptoAGI/bankml",
    },
    Host {
        name: "capi",
        side: "server",
        target: "libbankml.so / libbankml.a (bankml-capi)",
        model: "a path the application gives, verified at open",
        threads: "as native",
        gpu: "as native",
        memory: "as native",
        url: "https://github.com/cryptoAGI/bankml/blob/main/docs/CAPI.md",
    },
    Host {
        name: "browseml",
        side: "client",
        target: "wasm32-wasip1 (one thread) and wasm32-wasip1-threads (shared memory, the pool on Web Workers)",
        model: "downloaded from its pinned sources into the browser's cache (or read from a folder the visitor grants), sha256 at every open",
        threads: "wasi thread-spawn on Web Workers, where the page is cross-origin isolated; the free cores measured by the page",
        gpu: "webgpu (the page's GPU worker serves the bankml_gpu_* imports; threads needed)",
        memory: "no /proc: the engine's own WebAssembly memory, and what the page reports",
        url: "https://github.com/browseML/browseML",
    },
    Host {
        name: "wasi",
        side: "client",
        target: "wasm32-wasip1 under any WASI runtime (Node's WASI: the engine's kernel tests)",
        model: "a preopened file",
        threads: "one",
        gpu: "none: the bankml_gpu_* imports answer \"no GPU\"",
        memory: "the engine's own WebAssembly memory",
        url: "https://github.com/browseML/browseML/blob/main/tools/build.sh",
    },
];

/// The host by name.
pub fn by_name(name: &str) -> Option<&'static Host> {
    HOSTS.iter().find(|h| h.name.eq_ignore_ascii_case(name.trim()))
}

/// The host running this engine: the one `BANKML_HOST` names, else the build's own (native, or WASI on WebAssembly).
pub fn current() -> &'static Host {
    std::env::var("BANKML_HOST").ok().and_then(|n| by_name(&n)).unwrap_or_else(|| by_name(if cfg!(target_family = "wasm") { "wasi" } else { "native" }).expect("listed"))
}

/// The host's own version, when it gives one (`BANKML_HOST_VERSION`, e.g. browseML's).
pub fn version() -> Option<String> {
    std::env::var("BANKML_HOST_VERSION").ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty() && v.len() <= 64)
}

/// Whether the engine runs on the client side (in the visitor's browser), not on a server.
pub fn client_side() -> bool {
    current().side == "client"
}

/// Whether the engine runs in browseML (the browser host).
pub fn in_browser() -> bool {
    current().name == "browseml"
}

/// One host as JSON.
pub fn host_json(h: &Host) -> String {
    use crate::gguf::jstr;
    format!("{{\"name\": {}, \"side\": {}, \"target\": {}, \"model\": {}, \"threads\": {}, \"gpu\": {}, \"memory\": {}, \"url\": {}}}",
            jstr(h.name), jstr(h.side), jstr(h.target), jstr(h.model), jstr(h.threads), jstr(h.gpu), jstr(h.memory), jstr(h.url))
}

/// The running host and every known one, as JSON (`bankml hosts`).
pub fn report_json() -> String {
    let c = current();
    format!("{{\"current\": {}, \"version\": {}, \"engine\": {}, \"hosts\": [{}]}}",
            crate::gguf::jstr(c.name), version().map(|v| crate::gguf::jstr(&v)).unwrap_or_else(|| "null".into()),
            crate::gguf::jstr(env!("CARGO_PKG_VERSION")), HOSTS.iter().map(host_json).collect::<Vec<_>>().join(", "))
}

/// The running host in one line, for logs and SELF: "browseml 0.6.7" or "native".
pub fn line() -> String {
    match version() {
        Some(v) => format!("{} {v}", current().name),
        None => current().name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_host_is_named_once_and_found_by_name() {
        for h in HOSTS {
            assert_eq!(HOSTS.iter().filter(|o| o.name == h.name).count(), 1, "{} listed twice", h.name);
            assert_eq!(by_name(&h.name.to_uppercase()), Some(h));
            assert!(h.url.starts_with("https://"), "{}: url", h.name);
            assert!(h.side == "server" || h.side == "client", "{}: side", h.name);
        }
        assert!(by_name("browseml").is_some_and(|h| h.gpu.starts_with("webgpu") && h.side == "client"));
        assert!(by_name("native").is_some_and(|h| h.side == "server"));
        assert_eq!(by_name("nowhere"), None);
    }

    #[test]
    fn the_report_is_json_with_every_host() {
        let r = report_json();
        let v = crate::serve::Json::parse(&r).expect("valid JSON");
        assert!(matches!(v.get("hosts"), Some(crate::serve::Json::Arr(a)) if a.len() == HOSTS.len()));
        assert!(v.get("current").and_then(|c| c.as_str()).is_some_and(|c| by_name(c).is_some()));
    }
}
