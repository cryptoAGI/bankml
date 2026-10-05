# `bankML/gguf.rs` — the GGUF header parser, the guard, and the read-only model map

## Summary

`gguf.rs` reads the header of a GGUF v3 file and decides whether bankml may play it. The decision is the
**guard**: `play`, `refuse` with reasons, or `need_more` header bytes. It is a port of minaiml's Python
`gguf_guard.py` (same verdicts, same reasons, same JSON keys). The guard never reads tensor data.

The module also holds what the rest of bankml needs to reach tensor data safely: type names, block layouts,
checked tensor spans, and `Mmap`, a read-only memory map of the file with no crate behind it.

Callers: `bankml guard` (CLI), `bankml::verify` (the guard is its first half, see [bankml.md](bankml.md)), and
through `verify` every path that loads a model: `bankml serve`, `serve --native`, the C API, `bankml create`.
`Mmap` is used by the native forward pass (`forward.rs`), the tokenizer, `bankml convert`, and the kernel oracles.
`jstr` is the crate's JSON string escaper, used across many modules.

## Technical usage

### The guard

```rust
pub fn judge(b: &[u8], engine: Engine, file_size: Option<u64>, filename: &str) -> Report
pub fn guard_file(path: &Path, engine: Engine) -> std::io::Result<Report>
```

- `judge` is the pure function over a header prefix, as `gguf_guard.judge(b, engine, file_size, filename)`.
- `guard_file` reads a 32 MiB prefix (or the whole file if smaller) and re-reads with a larger prefix when the
  header outgrows it. `NeedMore` therefore survives only when the file itself is truncated.
- `Engine` is `Mainline` or `Prism` (`as_str()` gives `"mainline"` / `"prism"`). The CLI sets it with
  `--engine mainline|prism`; the default is mainline.

`Report` carries `verdict: crate::Verdict`, `engine`, `gguf_version`, `arch`, `name`, `tensors` (count),
`types` (type name and count, sorted by type id), `kv_f16_bytes_per_token`, and the parsed `header`.
`Report::to_json()` prints the same object as `gguf_guard.py --json` (keys and values; whitespace differs).
`Report::verdict_str()` gives `"play"`, `"refuse"` or `"need_more"`.

### What the guard refuses

| check | reason text begins |
|---|---|
| bad magic, unknown value type, alignment that is not a positive integer | `not a GGUF file`, `unknown gguf value type`, `general.alignment … is not a positive integer` |
| arrays nested deeper than `MAX_ARRAY_DEPTH` (64) | `gguf arrays nested deeper than 64` |
| KV bytes per token overflowing `i128` | `KV cache size per token overflows` |
| a type id ≥ `MAINLINE_COUNT` (43) that is not a known fork type | `type ids [..] unknown to every build this guard knows` |
| `PQ2_0` / `PTQ1_0` (ids 142, 143) on mainline | `PQ2_0/PTQ1_0 are PrismML-fork types` |
| `Q2_0` tensors smaller than the group-64 layout (legacy group-128 Prism files) | `N Q2_0 tensors smaller than group-64 layout` |
| a Bonsai 2 label (`bonsai[\s_-]*2` or `prism-fork-required` in name, basename or file name) on mainline | `Bonsai 2 weights are in a rotated basis` |

`kv_f16_bytes_per_token` is `block_count × head_count_kv × (key_length + value_length) × 2`, with
`embedding_length / head_count` as the fallback head size. It is `None` when the header uses per-layer arrays.
The Savante UI's RAM budget reads it (`sAGI/models.py`).

### Header and tensors

```rust
pub struct Header { pub version: u32, pub kv: HashMap<String, Val>, pub tensors: Vec<TensorInfo>,
                    pub alignment: u64, pub data_start: u64 }
pub fn type_name(t: u32) -> String
pub fn block_layout(ty: u32) -> Option<(u64, u64)>
pub fn tensor_span(h: &Header, t: &TensorInfo) -> Result<(u64, u64), String>
pub fn tensor_bytes(path: &Path, h: &Header, t: &TensorInfo) -> std::io::Result<Vec<u8>>
pub fn jstr(s: &str) -> String
```

- `Val` is a metadata value (`U`, `I`, `F`, `B`, `S`, `Arr(n)`). Arrays are skipped, not decoded, as in the Python.
- `type_name` names the standard ggml types at b11192 plus the fork types; an unnamed id prints as its number.
- `block_layout` gives (elements, bytes) per block for the types bankml reads: F32, F16, BF16, Q8_0, Q1_0
  (128, 18) and Q2_0 (64, 18). Anything else is `None`.
- `tensor_span` checks that the type is readable, that `dims[0]` is a whole number of blocks, and that no size
  or offset overflows. `tensor_bytes` also checks the span against the file size before it allocates.

### `Mmap` (Unix only)

```rust
impl Mmap {
    pub fn open(path: &Path) -> std::io::Result<Self>
    pub fn bytes(&self) -> &[u8]
    pub fn tensor(&self, h: &Header, t: &TensorInfo) -> Option<&[u8]>
}
```

`PROT_READ`, `MAP_PRIVATE`; `mmap`/`munmap` are declared by hand from libc. `tensor` is bounds-checked by
`tensor_span` and then by the mapping's length.

```sh
bankml guard .models/Bonsai-8B-Q1_0.gguf --json      # exit 0 play, 2 refuse, 3 need_more, 1 unreadable
```

## How it is verified

- Unit tests in `gguf.rs`: the nine cases of `testing/test_gguf_guard.py` (`t1_q1_0_plays_on_mainline` …
  `t10_standard_k_quant_model_plays_on_mainline`), plus fail-closed cases: `bad_magic_and_unknown_types_refuse`,
  `deeply_nested_arrays_refuse_not_crash` (2,000,000 levels), `kv_size_overflow_refuses`, `tensor_span_is_checked`.
- `real_bonsai_1_7b_q1_0` (`#[ignore]`, needs the model): play, qwen3, 310 tensors, F32 113 / Q1_0 197,
  114,688 KV bytes per token, the values the Python guard printed on 2026-09-25.
- `testing/guard_agree.py` runs the Rust and Python guards on every synthetic case and every real model present and
  requires identical JSON. Last recorded result in [oracles.md](../oracles.md) §3: 28/28 agree.
- `testing/cli.rs`: `guard_verdicts_and_exit_codes`, `hostile_headers_refuse_not_crash`.
- The release gate runs `testing/test_gguf_guard.py` and `guard_agree.py`.

## Advantages and efficiency

- **Cheap before expensive.** The guard reads only a header prefix, so a wrong or hostile file is refused before
  the whole-file sha256 pin runs. A refusal always carries its reason; there is no silent fallback. The Bonsai 2
  check exists because mainline llama.cpp loads those files and answers in gibberish.
- **Fails closed where the Python fails open or crashes.** The deliberate divergences listed at the top of the file
  (malformed header, deep nesting, KV overflow) all refuse. Both 0.0.1 crashes (stack overflow on nested arrays,
  silent `i128` wrap) became refusals in 0.0.2.
- **Zero copy for weights.** `Mmap` maps the file read-only; pages load on first touch and weights are never copied.
  The kernels read blocks straight from the mapping.
- **Bulk arrays skipped, not decoded.** Arrays of fixed-size scalars in the metadata are stepped over by length,
  without allocation; only string and nested arrays are walked element by element.
- **Hostile-input practice.** All header arithmetic is checked (`checked_add`, `checked_mul`, `saturating_mul`,
  `u128` element counts). `tensor_bytes` checks the span against the file before allocating, so a lying header cannot
  ask for a huge buffer. Errors are `Result<_, String>` with a reason, not panics.
- **`unsafe` kept small.** It is limited to the `mmap`/`munmap` FFI and the slice view of the mapping. `Send`/`Sync`
  are justified in a comment: a read-only private mapping never changes under the process.
- **No dependencies.** The parser and the map use only `std` and libc symbols std already links.

## Limitations

- Only the header is judged. Whether the weights are the right ones is the pin's job (`sha256.rs`).
- `Mmap` is Unix only.
- `block_layout` covers F32, F16, BF16, Q8_0, Q1_0 and Q2_0. Other types play under the guard (the pin then
  decides) but `tensor_span` refuses to read them.
- PrismML fork types (`PQ2_0`, `PTQ1_0`) and Bonsai 2's rotated basis are refused on mainline. bankml has no
  kernels for them; they are out of scope until there is a measured need.
- `kv_f16_bytes_per_token` is `None` for headers with per-layer head counts.

## See also

- [bankml.md](bankml.md) (`verify`, `Verdict`), [sha256.md](sha256.md) (the pin)
- [q1_0.md](q1_0.md), [q2_0.md](q2_0.md), [f16.md](f16.md) (the kernels that read through `Mmap`)
- [../oracles.md](../oracles.md) §3, [../usage.md](../usage.md), [../TECHNICAL.md](../TECHNICAL.md) §III.1
