# `bankML/native.rs` — the engine behind `bankml serve --native`

## Summary

`native.rs` answers chat completions from bankML's own forward pass. It was added in 0.3.0. It is built to be
token-identical to llama-server b11192 on its oracle: the same template, tokenizer, forward pass (every kernel
llama.cpp chooses), sampler chain and, across requests, the same prompt cache. It holds one conversation slot, as
llama-server's `-np 1`.

0.3.8 (unreleased) adds what llama-server does at the slot's edges: the context limit, saving and restoring the slot
to a file, llama-server's host prompt cache for conversations that take turns, and each token's logprobs, reported
step by step for streaming. 0.3.9 (unreleased, in progress) adds an optional q8_0 KV cache
(`BANKML_CACHE_TYPE=q8_0`).

Since 0.3.1 it also owns the model's lifecycle. `Registry` names every GGUF pinned in the forks directory.
`Residency` keeps one model resident at a time, runs the full `verify` (the guard, then the sha256 pin) on every
load, and drops an idle model, with its memory map, when its keep-alive runs out.

`serve.rs` routes `/v1/chat/completions` and llama-server's endpoints here; `ollama.rs` routes `/api/*` here; the
C API's `bankml_chat` uses the same engine. All of them share one `Residency`, so they share one engine and one slot.

## Technical usage

### `Native`: one loaded model, one slot

| item | what it does |
|---|---|
| `Native::open(model: &Path, n_ctx: usize) -> Result<Native, String>` | reads the template, tokenizer, weights, the GGUF's sampling defaults and the end-of-generation set; allocates the KV cache in the type `BANKML_CACHE_TYPE` names (below) |
| `prompt(&self, messages: &Json) -> Result<Vec<u32>, String>` | the template, then tokenization with special tokens |
| `prompt_fit(&self, messages, num_ctx: Option<usize>)` | the same, fitted under a `num_ctx` (a request's option or a derived model's `PARAMETER num_ctx`) as Ollama fits it (below; 0.3.5) |
| `params(&self, req: &Json) -> Result<Params, String>` | `sampling(self.defaults.clone(), req)` |
| `grammar(&self, c: &Constraint)` | JSON mode's or a schema's grammar with the template's generation prompt prefilled, or a user GBNF as given |
| `grammar_vocab(&self)` | every token's piece and the end set, as the grammar and DRY read them; built once, on first use |
| `complete(&self, prompt, params, max_tokens: Option<usize>, grammar, emit: impl FnMut(&str) -> bool) -> Result<Done, String>` | one completion; `emit` gets whole UTF-8 pieces and returns false to stop; under a grammar each token is drawn as llama.cpp's `common_sampler_sample` draws it |
| `complete_with(&self, prompt, params, max_tokens, grammar, emit: impl FnMut(Step) -> bool) -> Result<Done, String>` | the same, reporting every step (0.3.8); `complete` is built on it |
| `save_slot(&self, path, model_sha256) -> Result<(usize, u64), String>` | writes the slot to a file; returns (tokens, bytes) (0.3.8) |
| `restore_slot(&self, path, model_sha256) -> Result<(usize, u64), String>` | reads one back; on any error empties the slot and refuses with `Unable to restore slot: …` (0.3.8) |
| `erase_slot(&self) -> usize` | empties the slot; returns the tokens it held (0.3.8) |
| `reset(&self)` | empties the slot, as a fresh llama-server or `cache_prompt: false` |
| `props_json(&self)` | `/props` in the shape Savante reads |

`Done` carries `prompt_tokens`, `cached_tokens`, `completion_tokens`, `finish_reason` (`stop` or `length`), `text`,
`prompt_ns`, `eval_ns`, the generated `tokens` (the end token included when one ended the answer), under a grammar
`grammar_ns` and `resampled`, since 0.3.7 `ttft_ns` and `energy_j` (the CPU package's energy when RAPL is readable,
else `None`), and since 0.3.8 `probs`: one `TokenLogprob` per generated token when `Params::n_probs` > 0.

`Step` (0.3.8) is one step of `complete_with`:

| field | meaning |
|---|---|
| `piece: &str` | the text this step released; may be empty |
| `complete: bool` | the token left no incomplete UTF-8 behind (llama-server sends a token only then) |
| `entry: Option<&TokenLogprob>` | its logprobs, when asked for and complete |
| `n: usize` | tokens generated so far |
| `eog: bool` | the end-of-turn token, which ends the answer and releases no text |

A last step with `n` unchanged and `complete` false flushes bytes left incomplete at the end. `TokenLogprob` holds
the token's `id`, its probability `p`, its `piece`, the `top` tokens as (id, p, bytes), and whether it was `emitted`
and `complete`. The probabilities come from `sampler::token_probs` on the raw logits. `serve.rs` turns the steps into
llama-server's stream (`NativeChat::run_steps`, see [serve.md](serve.md)).

```rust
let d = eng.complete_with(&prompt, params, Some(64), None, |s| {
    if s.complete && !s.piece.is_empty() { print!("{}", s.piece); }
    true // false stops the answer
})?;
```

### The prompt cache rule

`complete` reproduces llama-server's slot reuse (`server-context.cpp`):

- first, llama-server's host prompt cache ([prompt_cache.md](prompt_cache.md), 0.3.8): when the slot serves the
  prompt poorly, its state goes to RAM and a cached state that serves it better comes back;
- the longest common prefix of the tokens already in the slot and the new prompt is kept;
- when the whole prompt is cached, one token less is kept (llama-server evaluates at least one);
- the KV cache is truncated there, and the rest is computed in micro-batches of 512 (`Weights::prefill`).

Starting prefill where llama-server starts it means each row takes the same kernels as llama-server's rows, which is
why the reuse rule is part of token-identity.

Before the first draw, every prompt token is accepted into the sampler, cached or not. That is how llama-server
fills the penalties' window (O2) and DRY's. The end-of-turn token ends the answer, is not part of the text, and is
counted among the completion tokens, as llama-server counts it.

The context limit (0.3.8), as llama-server with context shift off: a prompt as long as the context or longer is
refused (`the prompt has N tokens; the context is M`; `serve.rs` refuses it earlier, with llama-server's 400 body),
and a generation that fills the context stops there with `finish_reason` `length`.

### `sampling()`: the request over the model's defaults

`pub fn sampling(defaults: Params, req: &Json) -> Result<Params, String>` resolves the request as llama-server does.
It reads these numeric fields:

- `temperature`, `top_k`, `top_p`, `min_p`, `min_keep`, `seed`;
- since 0.3.6, the penalties: `repeat_last_n`, `repeat_penalty`, `frequency_penalty`, `presence_penalty`;
- since 0.3.7, the rest of llama-server's default chain: `typical_p`, `top_n_sigma`, `xtc_probability`,
  `xtc_threshold`, `dynatemp_range`, `dynatemp_exponent`, `dry_multiplier`, `dry_base`, `dry_allowed_length`,
  `dry_penalty_last_n`, and the array `dry_sequence_breakers`;
- since 0.3.8, `n_probs` (`serve.rs` sets it from OpenAI's `logprobs` and `top_logprobs`).

llama-server's soft limits clamp: `top_p`, `min_p` and the two XTC fields to [0, 1], `temperature` to ≥ 0, and a
`dry_base` below 1 falls back to 1.75. Refused, because llama-server would act on them and bankML does not reproduce
them: `mirostat` (non-zero) and a custom `samplers` order. `Sampler::new` (in `sampler.rs`) then refuses what
llama-server refuses, with its message: a negative `repeat_last_n`, `dry_allowed_length` or `dry_penalty_last_n`, a
repeat penalty of 0 or below, an empty `dry_sequence_breakers`; and top-k outside 1–128, which bankML does not
reproduce. DRY's breakers are built from the vocabulary's pieces once per breaker list and kept on the engine.

### Slot files and the cache type (0.3.8, 0.3.9)

`save_slot` writes bankML's own format: a magic line (`bankML slot v1`), the model's sha256, the context, token and
layer counts, each layer's cache, the tokens, then a sha256 of all of it. It is written to a temporary name and
renamed, so a crash leaves no half file. `restore_slot` refuses a file for another model, another context or layer
shape, a damaged file (the sha256 does not match), trailing data, or tokens outside the vocabulary. The slot is then
emptied, never half-filled, and the next answer computes its whole prompt. `serve.rs` exposes these as
`POST /slots/0?action=save|restore|erase` (see [serve.md](serve.md)).

`BANKML_CACHE_TYPE` picks the KV cache's type when the model opens: `f16` (the default, llama.cpp's) or `q8_0`, as
llama.cpp's `--cache-type-k q8_0 --cache-type-v q8_0`. A q8_0 cache takes 53 % of the f16 cache's bytes (CHANGELOG
0.3.9) and needs a head size that is a multiple of 32; anything else is refused. The attention over it, with
llama.cpp's Hadamard rotation, is in `forward.rs` ([forward.md](forward.md)). A slot file records its cache type, and
one saved with the other type is refused with the reason. The host prompt cache counts each saved state's real
bytes, so a q8_0 cache keeps more states within `BANKML_CACHE_RAM`.

### `prompt_fit` and `fit_messages`: `num_ctx` as Ollama applies it

`pub fn fit_messages(msgs, num_ctx, count) -> Result<(Vec<Message>, usize), String>` is Ollama v0.13.3's `chatPrompt`
truncation (`server/prompt.go`), step for step:

- walking back from the last message, each earlier one is kept while the conversation from it, with the system
  messages before it, still fits;
- the last message always stays, and so do the system messages before the first one kept;
- Ollama's quirk is kept: a system message that is itself the first one cut is dropped.

The tokens counted are the prompt bankML answers (the GGUF's template, as llama-server renders it). If the fitted
prompt still exceeds `num_ctx`, `prompt_fit` refuses: Ollama's runner would cut tokens out of the middle (`num_keep`)
and shift the cache during the answer, which bankML does not reproduce.

### `Registry`: names for pinned files

- `Registry::build(model, fork_json, dir)`: the startup model first (the default), then, with `dir`, every `.gguf`
  named in every `*.FORK.json` there, looked for beside the startup model and in `dir`. A name pinned twice keeps its
  first entry.
- `model_name(file)`: the file's stem, lower-cased (`bonsai-1.7b-q1_0`).
- `resolve(model)`: empty means the startup model; otherwise the name, with or without `:latest`, or the file name,
  case-insensitively. Since 0.3.4, the name without its weight-type suffix (`base_name`: `-f16`, `-q1_0`,
  `-q2_0_g64`, …) is accepted when exactly one pin has that base (`mindx-gen39`).
- `header_info(path)`: whether the forward pass plays a file, from its header alone (guard, architecture, weight
  type and graph options via `forward::plan`, tokenizer, chat template), and if not, why: the reason names what is
  missing and where it is on the roadmap. `/api/tags` reports it.

### `Residency`: one resident model

| method | behaviour |
|---|---|
| `start(...)` | the startup model, already verified by `serve::run`, resident until told otherwise; refused if the forward pass does not play it |
| `acquire(&run, e)` | the resident model if it is `e` and its file identity is unchanged; otherwise the resident one is dropped first, then `e` is verified and opened. Errors carry their HTTP status (404 not on this machine, 400 not playable or refused by `verify`, 503 file changed) |
| `touch(name, KeepAlive)` | after a request: `Unload` drops it now, `For(d)` sets the expiry, `Forever` clears it |
| `reap()` | drops the resident model once it has expired, unless a request holds the engine (`serve.rs` calls it every second) |
| `current_or_default()` | the resident model, or the startup model loaded (for endpoints that name no model) |
| `shown_as(tag, num_ctx, loaded)` / `shown()` | what `/api/ps` names: the model whose request loaded the weights, and its context (0.3.5). Ollama's runner keeps the name of the model whose request loaded it and reloads, under the new name, when a request's `num_ctx` differs; bankML keeps the weights and records the same: when they were (re)loaded for the request, when nothing is recorded yet, or when the context differs |

`run: Mutex<()>` is held for a whole completion, load or unload, so the engine has one user and a switch never has
two models in memory. `cur` is held only briefly. `derived` holds the derived models of the registry directory
(O5, [create.md](create.md)), layered on the pins; `last` is the most recent verification, which `/bankml` reports
when nothing is resident.

## How it is verified

- `fit_messages_as_ollama` (unit test): Ollama's truncation on a synthetic count, including the system-message quirk.
- `oracle_native_serve` (`#[ignore]`, in the gate): three Savante-style conversations recorded from llama-server
  b11192 by `testing/serve_oracle.py`, replayed in one engine; text, prompt and completion counts and cache reuse
  must match on every turn. docs/oracles.md records 9 of 9 turns; turn 2 reuses 49 of turn 1's 53 prompt tokens.
- `oracle_native_serve_o4` (0.3.4): the same on the models O4 opens, Bonsai-1.7B (tied embeddings) and the Llama
  graph in F16 (SmolLM2-135M-Instruct and mindx-gen39), each against llama-server running that model, then each
  one's JSON mode with its template's own grammar.
- `oracle_json_mode`, `oracle_json_mode_ternary`, `oracle_json_schema`, `oracle_json_schema_ternary`,
  `oracle_json_schema_o4`: constrained answers against `testing/json_oracle.py` and `testing/json_schema_oracle.py`
  records, each from an empty cache: the same tokens (end token included), raw text, message content, finish reason
  and counts, with the server's reported grammar and generation prompt. The schema records (O6b) cover objects with
  required and optional fields, enums, ranges, nested `$defs`, a pattern, and a top-level array and string, through
  `response_format: json_schema`, the top-level `json_schema` and `json_object` with a schema; each record carries
  the request as sent, read exactly (a float literal stays one). `oracle_json_schema_o4` (0.3.5) runs them on
  Bonsai-1.7B (the Qwen3 template) and the two ChatML templates, whose schema grammar has no reasoning block
  (`schema::chat_grammar`).
- `oracle_penalties` and `oracle_penalties_8b` (0.3.6, O2): `testing/penalty_oracle.py` records of repeat, frequency
  and presence penalties over `repeat_last_n`, greedy and seeded, with the prompt in the window. Each request must
  resolve to the parameters llama-server read back, give its tokens, and be refused with its message where it
  refused. CHANGELOG 0.3.6: mindx-gen39 56 / 56, Bonsai-1.7B 56 / 56, Bonsai-8B 56 / 56; 12 / 12 refusals each.
- `oracle_samplers` (0.3.7): `testing/penalty_oracle.py --kind sampler` records, 23 variants × 4 prompts (typical-p,
  top-n-σ, XTC, dynamic temperature, DRY, alone and together). CHANGELOG 0.3.7: mindx-gen39 76 / 76, Bonsai-1.7B
  76 / 76; 16 / 16 refusals each. `oracle_samplers_8b` replays a Bonsai-8B record in the gate;
  CHANGELOG 0.3.7 does not yet give its count.
- Live, in the gate, each driving a running `serve --native`: `serve_oracle_ollama_shape`, `penalty_oracle_live`,
  `sampler_oracle_live`, and from 0.3.8 `context_oracle_live` (8 / 8), `slot_oracle_live` (19 / 19: the answer after
  a restore, in the same server and after a restart, equals an empty slot's), `session_oracle_live` (14 / 14, the host
  prompt cache), `logprobs_oracle_live` (14 / 14, five streamed); from 0.3.9 `kv_oracle_live` (6 / 6 with
  `BANKML_CACHE_TYPE=q8_0` against llama-server `--cache-type-k/v q8_0`). Counts from CHANGELOG 0.3.8 and 0.3.9.

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
- **A warm start that survives a restart.** A slot restore replaces the prompt's prefill with one read checked by
  sha256; Savante and the console pass `--slot-dir` for this (CHANGELOG 0.3.8).
- **Half the KV memory, on request.** `BANKML_CACHE_TYPE=q8_0` stores the cache in 53 % of the f16 bytes, with
  answers token-identical to llama-server configured the same way (CHANGELOG 0.3.9).
- **Logprobs cost nothing when off.** `token_probs` runs only when `n_probs` > 0: one partial sort of the vocabulary
  per generated token.
- **Next** (docs/TODO.md 0.4.0, docs/OLLAMA.md): more than one slot with continuous batching (O8); a 4-bit KV cache
  (`q4_0`); Ollama's ContextShift past `num_ctx` (O2).

## Limitations

- One slot, as llama-server `-np 1`. Requests are served one at a time; other conversations' states wait in the
  host prompt cache (`testing/session_oracle.py`).
- One resident model. A request for another model unloads the current one.
- Not reproduced, so refused with a reason: mirostat, a custom sampler order, top-k 0 or above 128.
- A prompt that does not fit `num_ctx` after Ollama's message truncation is refused; Ollama would cut tokens out of
  its middle. Tokens past `num_ctx` during an answer are not shifted out (ContextShift); the answer runs within the
  served `--ctx` (docs/OLLAMA.md, "Still open").
- The KV cache is f16 or q8_0 only; another `BANKML_CACHE_TYPE` is refused when the model opens. The type is read
  from the environment, not per request, so a slot file saved under one type does not restore under the other.
- Slot files are bankML's own format, not llama-server's; a file from one does not load in the other.

## See also

- [../usage.md §6a, Ollama's API](../usage.md#6a-ollamas-api) · [../OLLAMA.md](../OLLAMA.md) ·
  [../oracles.md](../oracles.md) · [../TODO.md](../TODO.md) · [../CAPI.md](../CAPI.md)
- [serve.md](serve.md) · [ollama.md](ollama.md) · [create.md](create.md) · [sampler.md](sampler.md) ·
  [prompt_cache.md](prompt_cache.md) · [grammar.md](grammar.md) · [schema.md](schema.md) · [forward.md](forward.md) ·
  [gguf.md](gguf.md) · [metrics.md](metrics.md)
