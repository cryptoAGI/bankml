# `host.rs` — where bankML runs

## Summary

bankML's engine is one body of code with several **hosts**: programs that load it and give it a model, threads, a
GPU and a place for its files. `host.rs` lists them, says which one is running the engine now, and names the
**side** each is on, so the bankML server and client-side interaction stay distinct:

| host | side | built for | model | threads | GPU |
|---|---|---|---|---|---|
| `native` | server | x86_64 / aarch64 Linux: `bankml serve`, `generate`, the CLI | a GGUF on disk, pinned by FORK.json | `std::thread` | Vulkan (`dlopen`, bankML's SPIR-V) |
| `capi` | server | `libbankml` (bankml-capi) | a path the application gives | as native | as native |
| `browseml` | client | `wasm32-wasip1` and `wasm32-wasip1-threads` | downloaded from its pinned sources into the browser (or read from a granted folder), sha256 at every open | Web Workers through `wasi thread-spawn` | WebGPU (the page's GPU worker serves `bankml_gpu_*`) |
| `wasi` | client | `wasm32-wasip1` under any WASI runtime (the kernel tests) | a preopened file | one | none |

A **server** host runs on the operator's machine and answers others over HTTP, with receipts. A **client** host runs
on the visitor's own machine and answers only the visitor: nothing it computes leaves that machine. The engine, its
oracles and its receipts are the same on both sides; who owns the machine, who asks, and where the answer goes are
not, and every report names the side.

[browseML](https://github.com/browseML/browseML) is the client host: bankML in the visitor's browser, on their CPU
and (verified) GPU. Its engine names itself (`BANKML_HOST=browseml`, `BANKML_HOST_VERSION`) at open, and its open
report carries `host` and `side`.

## Technical usage

```rust
pub struct Host { pub name, pub side, pub target, pub model, pub threads, pub gpu, pub memory, pub url: &'static str }
pub const HOSTS: &[Host];                       // add a host here
pub fn by_name(name: &str) -> Option<&'static Host>;
pub fn current() -> &'static Host;              // BANKML_HOST, else the build's own (native, or wasi on WebAssembly)
pub fn version() -> Option<String>;             // BANKML_HOST_VERSION
pub fn client_side() -> bool;  pub fn in_browser() -> bool;
pub fn report_json() -> String;                 // `bankml hosts`
pub fn line() -> String;                        // "browseml 0.6.7"
```

CLI: `bankml hosts` prints the report. browseML's engine exports `browseml_hosts()` (the same report) and
`browseml_gpu_status()` (the GPU limiter's JSON).

**Adding a host** (a phone app over the C API, another WASI runtime, a desktop shell): one entry in `HOSTS`; if it
brings a new GPU API, a backend in `gpu/` listed in `gpu::BACKENDS` (as `webgpu` was for browseML); if it has no
`/proc`, the GPU limiter already falls back to the heap's limit alone.

## How it is verified

`host::tests`: every host named once, found by name in any case, a side of `server` or `client`, an `https` URL;
browseML is a client with WebGPU, native a server; the report is valid JSON with every host. browseML's gate runs
the engine as the browser does (`testing/bench.mjs`, `testing/oracle.mjs`) and its answers are the record's.

## What browseML brought back to bankML

- **`gpu/compute_webgpu.rs`**: `compute`'s API on WebGPU through host imports. The worker, its on-card oracle and its
  calibration run unchanged on it.
- **`gpu/wgsl.rs`**: the exact WGSL kernel, the lessons that made it exact on a real AMD card, and the vendor
  workgroup sizes ([gpu.md](gpu.md#webgpu-browseml)).
- **The worker's many-token product** (`begin_many` / `finish_many`): prompt reading on a card whose kernel has a
  token dimension (WebGPU's).
- **The limiter without `/proc`**: on an integrated card whose RAM is unknown, the heap's limit alone (before, a
  browser's card was never given a matrix).

## Limitations

The registry describes; it does not load anything. A host's version is what the host says (`BANKML_HOST_VERSION`,
64 characters at most).

## See also

[gpu.md](gpu.md) · [capi.md](capi.md) · [serve.md](serve.md) · [browseML](https://github.com/browseML/browseML)
