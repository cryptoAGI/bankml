# `bankML/bankml.rs` — the crate root: the verification gate, shared types, logging

## Summary

`bankml.rs` is the library root (`[lib] path = "bankML/bankml.rs"` in `Cargo.toml`). It declares the modules and
holds the one check that stands in front of every answer: `verify`, the header guard followed by the sha256 pin.
A model that does not pass it does not answer, and the refusal says why.

It also defines the types the rest of the crate shares: the ggml type ids bankml plays (`GgmlType`), the guard's
`Verdict`, the `Verified` record a passing model earns, the design-time `Receipt` shape, the crate `VERSION`, and
the library's log sink.

Callers of `verify`: `bankml verify` and `bankml serve` (CLI), the native engine when it loads a registry model
(`native.rs`), `bankml create` for its `FROM` model, and the C API's open (`capi/src/lib.rs`). `pin` is also
reachable on its own as `bankml pin`.

## Technical usage

### Modules declared

`chat`, `f16`, `convert`, `create`, `forward`, `gguf`, `grammar`, `gpu`, `native`, `ollama`, `par`, `serve`,
`q1_0`, `q2_0`, `sampler`, `schema`, `sha256`, `sys`, `train`, `tokenizer`, `unicode_letters`.
The crate has `#![allow(dead_code)]` at its root.

### The gate

```rust
pub fn guard(gguf: &Path) -> Verdict
pub fn pin(gguf: &Path, fork_json: &str) -> Result<String, String>
pub fn verify(gguf: &Path, fork_json: &str, engine: gguf::Engine) -> Result<Verified, String>
```

- `guard` runs `gguf::guard_file` on the mainline engine; an unreadable file becomes `Refuse`.
- `pin` looks up the file's name in the `FORK.json` text and hashes the file. `Ok(sha)` only when they are equal.
  Errors: `"<name> has no sha256 record in FORK.json: unpinned, refused"`, `"cannot hash <name>: …"`,
  `"<name> sha256 <got> != pinned <want>: refused"`.
- `verify` runs the guard first (reads only the header), then the pin (reads the whole file). Its errors begin
  `cannot read`, `guard refused: <reasons joined by "; ">`, or `guard needs N header bytes: file truncated`, or are
  the pin's error.

### Types

```rust
pub enum GgmlType { F32 = 0, F16 = 1, Q1_0 = 41, Q2_0 = 42 }
pub enum Verdict { Play, Refuse(Vec<String>), NeedMore(u64) }
pub struct Verified {
    pub model_sha256: String, pub guard: &'static str, pub engine: &'static str,
    pub arch: Option<String>, pub name: Option<String>, pub types: Vec<(String, usize)>,
}
pub struct Receipt {
    pub model_sha256: String, pub guard: String, pub prompt_tokens: u32, pub completion_tokens: u32,
    pub ttft_ms: u64, pub wall_ms: u64, pub thot8_leaf: Option<[u8; 32]>,
}
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
```

- `GgmlType`: Q1_0 is 1 bit per weight, group 128, f16 scale (1.125 bpw). Q2_0 is 2-bit codes mapping to
  {−1, 0, +1, +2}·d, group 64, f16 scale (2.25 bpw).
- `Verified::to_json()` prints `verdict`, `guard`, `engine`, `model_sha256`, `bankml` (the version), `arch`,
  `name` and `types`. Header strings are escaped with `gguf::jstr`. `bankml serve`'s `GET /bankml` includes it.
- `Receipt` is the receipt shape fixed at design time (model sha256, guard verdict, token counts, timings, an
  optional THOT8 leaf). Answers actually served carry `serve.rs`'s `bankml_receipt` object.
- `VERSION` is printed by `bankml version` (currently `0.3.6`).

### Logging

```rust
pub const LOG_ERROR: i32 = 0;  pub const LOG_WARN: i32 = 1;  pub const LOG_INFO: i32 = 2;  pub const LOG_DEBUG: i32 = 3;
pub type LogSink = fn(i32, &str);
pub fn set_log_sink(sink: Option<LogSink>)
pub fn log(level: i32, msg: &str)
```

Messages go to standard error, one line each, unless an embedder installs a sink. The C API's `bankml_set_log`
does; `None` restores standard error.

### CLI

```sh
bankml verify FILE --fork FORK.json [--engine mainline|prism] [--json]
# exit 0 play · 1 cannot read · 2 refused · 3 truncated header
bankml pin FILE --fork FORK.json                      # "pinned <sha>" or "refuse: <reason>"
```

With `--json`, a refusal prints `{"verdict": "refuse", "reason": …, "bankml": "<version>"}`.

## How it is verified

- `type_ids_match_mainline`: Q1_0 = 41, Q2_0 = 42.
- `verified_json_escapes_what_the_header_says`: a hostile `general.name` (quotes, backslash, newline, control
  character) cannot break the JSON; `null` arch and name and empty types are printed correctly.
- `verify_runs_guard_then_pin`: a minimal header plays and pins; a wrong pin gives `!= pinned`; a bad magic gives
  `guard refused`.
- `bpw_from_block_layout`: 1.125 and 2.25 bits per weight.
- `testing/cli.rs`: `pin_and_verify`, `serve_gates_and_signs_answers`,
  `serve_refuses_an_unverified_model_or_a_different_upstream_file`.
- The guard and hash halves have their own oracles: [gguf.md](gguf.md), [sha256.md](sha256.md).

## Advantages and efficiency

- **One gate for every entry point.** The CLI, the gateway, the native engine, `bankml create` and the C API all
  call the same `verify`, so a model is checked the same way however it is reached.
- **Cheap check first.** The guard reads a header prefix; only a model that passes it is hashed in full.
- **Errors as values.** The gate returns `Result<Verified, String>` with a human reason, which the CLI maps to exit
  codes and `serve`/`native` map to HTTP refusals. Nothing in the gate panics on a bad file.
- **What was verified travels with it.** `Verified` records the sha256, the engine, and what the header said
  (architecture, name, tensor types by count), and its JSON carries the bankml version.
- **Zero dependencies.** `Cargo.toml`'s `[dependencies]` is empty; the workspace adds only the C API package, which
  depends on this crate alone. The toolchain is pinned to Rust 1.99.0 in `rust-toolchain.toml` (edition 2021,
  `rust-version = "1.99"`), and the release profile uses `lto = true`, `codegen-units = 1`, `panic = "abort"`.
  The release gate runs `cargo clippy … -D warnings` over the workspace.
- **A log sink, not a logging crate.** A `RwLock<Option<fn>>` lets an embedding program receive messages through one
  callback.

## Limitations

- `guard` always uses the mainline engine; use `gguf::guard_file` or `verify` to choose the Prism engine.
- `GgmlType` lists only F32, F16, Q1_0 and Q2_0. The guard names and plays other types; the native matmul kernels
  are Q1_0, Q2_0 and F16 ([q1_0.md](q1_0.md), [q2_0.md](q2_0.md), [f16.md](f16.md)).
- `Receipt` is not what answers carry; see `serve.rs`'s `bankml_receipt` and [../usage.md](../usage.md) §9.
- The pin needs a `FORK.json` record for the file's name. A model without one is refused as unpinned.

## See also

- [gguf.md](gguf.md), [sha256.md](sha256.md), [par.md](par.md), [sys.md](sys.md)
- [../usage.md](../usage.md) §6 and §9, [../CAPI.md](../CAPI.md), [../OLLAMA.md](../OLLAMA.md)
- [../TECHNICAL.md](../TECHNICAL.md) §III.1, [../oracles.md](../oracles.md), [../TODO.md](../TODO.md)
