# `bankML/metrics.rs` — bankML's own measurements of its answers

## Summary

`metrics.rs` (0.3.7) keeps one record per completion, taken inside the engine (`Native::complete_with`, which
`Native::complete` wraps since 0.3.8, so streamed, non-streamed and logprobs answers are all counted), in a ring of the
last 256, and serves them at `GET /bankml/metrics`. The definitions are the conventional ones, so the numbers compare with
llama.cpp's: time to first token (TTFT), prompt processing in tokens per second (llama-bench's `pp`), generation in tokens
per second (`tg`), and — when the CPU package's RAPL counter is readable — the energy a completion took and joules per
generated token. Nothing is estimated: a field that was not measured is `null`. The console's charts and the bankML
persona's SELF block read from here.

## Technical usage

```rust
pub const KEEP: usize = 256;
pub struct Record { pub at: f64, pub model: String, pub prompt_tokens: usize, pub cached_tokens: usize,
                    pub completion_tokens: usize, pub prompt_ms: f64, pub ttft_ms: Option<f64>, pub eval_ms: f64,
                    pub finish: String, pub grammar_ms: f64, pub resampled: usize, pub energy_j: Option<f64> }
impl Record { pub fn prompt_tps(&self) -> Option<f64>; pub fn eval_tps(&self) -> Option<f64>;
              pub fn joules_per_token(&self) -> Option<f64>; pub fn json(&self) -> String }
pub fn push(r: Record); pub fn records() -> Vec<Record>; pub fn json() -> String
```

- `model` is the GGUF's file stem; `at` the end of the completion in Unix seconds.
- `prompt_tps` counts only the prompt tokens actually computed, over `prompt_ms`. `cached_tokens` is the prefix found
  in the slot, or brought back from the host prompt cache ([prompt_cache.md](prompt_cache.md)); it is the `cache_n`
  of the answer's `timings`.
- `energy_j` is the difference of two `sys::energy_uj()` readings around the completion (the counter's wrap handled); it
  includes everything the package drew in that time, an integrated GPU too.
- `Record::json` adds the derived `prompt_tps`, `eval_tps` and `joules_per_token` to the stored fields.
- `GET /bankml/metrics` → `{"source", "kept", "keep", "prompt_tokens", "completion_tokens", "energy_j",
  "joules_per_token", "records": […]}`. The token totals cover every kept record; `energy_j` and `joules_per_token`
  only the records whose energy was measured (`null` when none was).

```sh
curl -s 127.0.0.1:18093/bankml/metrics | python3 -m json.tool | head
```

## How it is verified

Unit tests: `rates_and_energy_are_measured_or_null` (the rates, J/token, and `null` when not measured) and
`the_ring_keeps_the_last_ones`. Live (0.3.7, Bonsai-1.7B): a non-streamed answer's receipt carries `ttft_ms` (it was
`null` before), and the record matches the answer's `timings`.

## Advantages and efficiency

- **Measured where the work is done.** The engine times its own prompt computation, first piece and generation, so the
  numbers do not include HTTP or the client.
- **Bounded and cheap.** A fixed ring of 256 records behind one mutex, one RAPL read before and after a completion.
- **Honest by construction.** `Option` fields render `null`; no field is ever derived from a guess.

## Limitations

- Energy needs `./install.sh power` (the counter is root-only by default) and covers the whole CPU package, not bankML
  alone; a discrete GPU's power is not included (no driver-neutral sensor).
- The ring is in memory: a restart empties it (the console keeps its own log of exchanges).

## See also

[sys.md](sys.md) · [native.md](native.md) · [serve.md](serve.md) · [prompt_cache.md](prompt_cache.md) · [console.md](console.md) · [../usage.md](../usage.md) §13
