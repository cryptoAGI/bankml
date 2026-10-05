# `bankML/native.rs` — the engine behind `bankml serve --native`

## Summary

`native.rs` answers chat completions from bankML's own forward pass. It is built to be token-identical to
llama-server b11192 on its oracle: the same template, tokenizer, forward pass, sampler chain and, across requests,
the same prompt cache. It holds one conversation slot, as llama-server's `-np 1`.

Since 0.3.1 it also owns the model's lifecycle. `Registry` names every GGUF pinned in the forks directory.
`Residency` keeps one model resident at a time, runs the full `verify` (the guard, then the sha256 pin) on every
load, and drops an idle model, with its memory map, when its keep-alive runs out.

`serve.rs` routes `/v1/chat/completions` and llama-server's endpoints here; `ollama.rs` routes `/api/*` here; the
C API's `bankml_chat` uses the same engine. All of them share one `Residency`, so they share one engine and one slot.

## Technical usage

### `Native`: one loaded model, one slot

| item | what it does |
|---|---|
| `Native::open(model: &Path, n_ctx: usize) -> Result<Native, String>` | reads the template, tokenizer, weights, the GGUF's sampling defaults and the end-of-generation set |
| `prompt(&self, messages: &Json) -> Result<Vec<u32>, String>` | the template, then tokenization with special tokens |
| `prompt_fit(&self, messages, num_ctx: Option<usize>)` | the same, fitted under a `num_ctx` as Ollama fits it (below) |
| `params(&self, req: &Json) -> Result<Params, String>` | `sampling(self.defaults.clone(), req)` |
| `grammar(&self, c: &Constraint)` | JSON mode's or a schema's grammar with the template's generation prompt prefilled, or a user GBNF as given |
| `complete(&self, prompt, params, max_tokens: Option<usize>, grammar, emit: impl FnMut(&str) -> bool) -> Result<Done, String>` | one completion; `emit` gets whole UTF-8 pieces and returns false to stop |
| `reset(&self)` | empties the slot, as a fresh llama-server or `cache_prompt: false` |
| `props_json(&self)` | `/props` in the shape Savante reads |

`Done` carries `prompt_tokens`, `cached_tokens`, `completion_tokens`, `finish_reason` (`stop` or `length`), `text`,
`prompt_ns`, `eval_ns`, the generated `tokens` (the end token included when one ended the answer), and, under a
grammar, `grammar_ns` and `resampled`.

### The prompt cache rule

`complete` reproduces llama-server's slot reuse (`server-context.cpp`):

- the longest common prefix of the tokens already in the slot and the new prompt is kept;
- when the whole prompt is cached, one token less is kept (llama-server evaluates at least one);
- the KV cache is truncated there, and the rest is computed in micro-batches of 512 (`Weights::prefill`).

Before the first draw, every prompt token is accepted into the sampler, cached or not. That is the penalties'
window as llama-server fills it (O2). The end-of-turn token ends the answer, is not part of the text, and is
counted among the completion tokens, as llama-server counts it. A prompt as long as the context or longer is
refused (`the prompt has N tokens; the context is M`).

### `sampling()`: the request over the model's defaults

`pub fn sampling(defaults: Params, req: &Json) -> Result<Params, String>` reads these numeric fields of the request:
`temperature`, `top_k`, `top_p`, `min_p`, `min_keep`, `seed`, and since 0.3.6 `repeat_last_n`, `repeat_penalty`,
`frequency_penalty`, `presence_penalty`. A sampler bankML does not reproduce is refused when set to a non-neutral
value: `typical_p` (≠ 1), `xtc_probability`, `dry_multiplier`, `dynatemp_range` (≠ 0), and `top_n_sigma` (> 0).
`Sampler::new` (in `sampler.rs`) then refuses what llama-server refuses, with its message: a negative
`repeat_last_n`, or a repeat penalty of 0 or below.

### `prompt_fit` and `fit_messages`: `num_ctx` as Ollama applies it

`pub fn fit_messages(msgs, num_ctx, count) -> Result<(Vec<Message>, usize), String>` is Ollama v0.13.3's `chatPrompt`
truncation (`server/prompt.go`), step for step:

- walking back from the last message, each earlier one is kept while the conversation from it, with the system
  messages before it, still fits;
- the last message always stays, and so do the system messages before the first one kept;
- Ollama's quirk is kept: a system message that is itself the first one cut is dropped.

The tokens counted are the prompt bankML answers. If the fitted prompt still exceeds `num_ctx`, `prompt_fit` refuses:
Ollama's runner would cut tokens out of the middle (`num_keep`), which bankML does not reproduce.

### `Registry`: names for pinned files

- `Registry::build(model, fork_json, dir)`: the startup model first (the default), then, with `dir`, every `.gguf`
  named in every `*.FORK.json` there, looked for beside the startup model and in `dir`. A name pinned twice keeps its
  first entry.
- `model_name(file)`: the file's stem, lower-cased (`bonsai-1.7b-q1_0`).
- `resolve(model)`: empty means the startup model; otherwise the name, with or without `:latest`, or the file name,
  case-insensitively. Since 0.3.4, the name without its weight-type suffix (`base_name`: `-f16`, `-q1_0`,
  `-q2_0_g64`, …) is accepted when exactly one pin has that base (`mindx-gen39`).
- `header_info(path)`: whether the forward pass plays a file, from its header alone (guard, architecture, weight
  type, tokenizer, chat template), and if not, why. `/api/tags` reports it.

### `Residency`: one resident model

| method | behaviour |
|---|---|
| `start(...)` | the startup model, already verified by `serve::run`, resident until told otherwise; refused if the forward pass does not play it |
| `acquire(&run, e)` | the resident model if it is `e` and its file identity is unchanged; otherwise the resident one is dropped first, then `e` is verified and opened. Errors carry their HTTP status (404 not on this machine, 400 not playable or refused by `verify`, 503 file changed) |
| `touch(name, KeepAlive)` | after a request: `Unload` drops it now, `For(d)` sets the expiry, `Forever` clears it |
| `reap()` | drops the resident model once it has expired, unless a request holds the engine (`serve.rs` calls it every second) |
| `current_or_default()` | the resident model, or the startup model loaded (for endpoints that name no model) |
| `shown_as(tag, num_ctx, loaded)` / `shown()` | what `/api/ps` names: the model whose request loaded the weights, and its context (0.3.5) |

`run: Mutex<()>` is held for a whole completion, load or unload, so the engine has one user and a switch never has
two models in memory. `cur` is held only briefly.

## How it is verified

- `fit_messages_as_ollama` (unit test): Ollama's truncation on a synthetic count, including the system-message quirk.
- `oracle_native_serve` (`#[ignore]`, in the gate): three Savante-style conversations recorded from llama-server
  b11192 by `testing/serve_oracle.py`, replayed in one engine; text, prompt and completion counts and cache reuse
  must match on every turn. docs/oracles.md records 9 of 9 turns; turn 2 reuses 49 of turn 1's 53 prompt tokens.
- `oracle_native_serve_o4`: the same on Bonsai-1.7B, SmolLM2-135M-Instruct-F16 and mindx-gen39-F16, plus JSON mode.
- `oracle_json_mode`, `oracle_json_mode_ternary`, `oracle_json_schema`, `oracle_json_schema_ternary`,
  `oracle_json_schema_o4`: constrained answers against `testing/json_oracle.py` and `testing/json_schema_oracle.py`
  records, token for token, with the server's reported grammar and generation prompt.
- `oracle_penalties` and `oracle_penalties_8b` (0.3.6): `testing/penalty_oracle.py` records. Each request must
  resolve to the parameters llama-server read back, give its tokens, and be refused with its message where it
  refused. CHANGELOG 0.3.6: mindx-gen39 56 / 56, Bonsai-1.7B 56 / 56, Bonsai-8B 56 / 56; 12 / 12 refusals each.
- Live, in the gate: `serve_oracle_ollama_shape` and `penalty_oracle_live` drive a running `serve --native`.

## Advantages and efficiency

- **A verified engine, not a trusted one.** Every load runs the guard and the sha256 pin before a byte of weights
  is used. llama-server and Ollama load whatever file they are given. The pin uses the CPU's SHA extensions when
  present: docs/PERFORMANCE.md measures 0.23 s for a 248 MB model (5.5× the portable hash) and 2.9 s for the
  1.16 GB 8B model.
- **llama-server's prompt cache, reproduced.** A follow-up turn computes only the tokens after the shared prefix,
  as the server does, so a conversation does not pay for its history each turn.
- **One resident model, freed on time.** As Ollama's `MAX_LOADED_MODELS=1`: the old model goes before the new one is
  read, and the reaper releases the memory map when the keep-alive expires. The file identity (device, inode, size,
  modification time) is checked before each reuse instead of rehashing.
- **Streaming by whole characters.** Pieces are emitted as soon as they form valid UTF-8; the grammar vocabulary
  (about 10 MB) and the JSON-mode rules are built once, on first use (`OnceLock`).
- **Rust practice visible here.** No external crates (Cargo.toml has an empty `[dependencies]`); errors are
  `Result<_, String>` that carry a reason, or `(u16, String)` with the HTTP status; poisoned mutexes are recovered
  with `unwrap_or_else(|e| e.into_inner())`; the toolchain is pinned to 1.99.0 in `rust-toolchain.toml`.
- **Next** (docs/TODO.md, 0.4.0): slot save and restore in `--native`; more than one slot with llama-server's
  queueing, then continuous batching across slots (O8); behaviour at the context limit as llama-server's; Ollama's
  ContextShift past `num_ctx` (O2).

## Limitations

- One slot. Requests are served one at a time; the slot holds one conversation's prefix.
- One resident model. A request for another model unloads the current one.
- Not reproduced, so refused with a reason: typical-p, xtc, dry, dynamic temperature, top-n-σ.
- A prompt that does not fit `num_ctx` after Ollama's message truncation is refused; Ollama would cut tokens out of
  its middle. Tokens past `num_ctx` during an answer are not shifted out (ContextShift); the answer runs within the
  served `--ctx` (docs/OLLAMA.md, "Still open").
- The module's header comment still lists penalties among the refused samplers; since 0.3.6 `sampling()` honours
  them, as its own comment says.

## See also

- [../usage.md §6a, Ollama's API](../usage.md#6a-ollamas-api) · [../OLLAMA.md](../OLLAMA.md) ·
  [../oracles.md](../oracles.md) · [../TODO.md](../TODO.md) · [../CAPI.md](../CAPI.md)
- [serve.md](serve.md) · [ollama.md](ollama.md) · [create.md](create.md) · [sampler.md](sampler.md) ·
  [grammar.md](grammar.md) · [schema.md](schema.md) · [forward.md](forward.md) · [gguf.md](gguf.md)
