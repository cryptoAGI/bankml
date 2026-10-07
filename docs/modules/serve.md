# `bankML/serve.rs` — `bankml serve`, the verifying loopback gateway

## Summary

`serve.rs` is the HTTP gateway behind `bankml serve`. It does not start until the model file passes `verify` (the
guard, then the sha256 pin in FORK.json). Every `/v1/chat/completions` answer carries a receipt: the bankml version,
the engine, the model's sha256, the guard verdict, token counts, timings, and the sha256 of the answer text and of
the request it answered.

It runs in one of two modes. In front of llama.cpp b11192's `llama-server` (P0, since 0.0.6), it vouches for the
file, the path and the transcript, not for the arithmetic. With `--native` (0.3.0), the answers come from bankML's own forward pass
(`native.rs`), and the gateway also speaks llama-server's endpoints on the engine address and Ollama's API
(`ollama.rs`).

The module also holds the shared pieces other modules use: `NativeChat` (one native chat request, shared with the C
API's `bankml_chat`), `Tally` (what a receipt is made from), `FileIdent`, and `Json`, a small JSON reader.

## Technical usage

### Modes

```sh
bankml serve FILE --fork FORK.json --spawn /path/to/llama-server --threads 3 --ctx 2048   # launch the engine on the verified path
bankml serve FILE --fork FORK.json --upstream 127.0.0.1:18092                          # check a running one by its /props
bankml serve FILE --fork FORK.json --native --registry --ctx 2048                      # bankML's own forward pass
```

- `--spawn`: refuses if something already answers on the upstream address. It launches llama-server with
  `-m <verified path> --host --port -t -c -np 1 --jinja --reasoning off --no-webui`, plus `--spec-type ngram-simple`
  with `--spec-ngram` (opt-in since 0.1.8; exact at temperature 0) and `--slot-save-path` with `--slot-dir` (0.1.9). It waits up to 900 s for `/health` and notices at
  once if the engine exits.
- `--upstream`: waits up to 10 s for `/health`, then requires `/props` `model_path` to be the same canonical path.
- In both, before every answer the file's identity (device, inode, size, modification time) and the engine's model
  path are checked again; a change answers 503 and no receipt.
- `--native`: also binds the `--upstream` address and answers llama-server's endpoints there, so a client written
  for llama-server (Savante) reaches it unchanged. Since 0.3.1 it also speaks Ollama's API, and with `--registry`
  every model pinned in the forks directory may be asked for by name. A reaper thread drops an expired model every
  second.

Defaults (from `main.rs`): `--listen 127.0.0.1:18093`, `--upstream 127.0.0.1:18092`, `--threads 3`, `--ctx 4096`;
`--keep-alive` defaults to Ollama's `5m`.

### Endpoints

| method, path | P0 (llama-server) | `--native` |
|---|---|---|
| `GET /bankml` | what was verified, when, the upstream | the same, plus `resident` and `models` (the last verification when none is resident) |
| `GET /bankml/usage` | memory and CPU of `bankml serve` and the spawned engine; since 0.3.7 also `package_watts`, each GPU's busy %, VRAM and GTT, and the GPU limiter's state (`sys.rs`, sampled over 250 ms, at most once a second however many clients poll; the sample is shared) | the same |
| `GET /bankml/metrics` (0.3.7) | the native engine's records (empty in P0) | one record per completion, a ring of 256 with totals: TTFT, prompt and generation tokens per second, grammar time, energy ([metrics.md](metrics.md)) |
| `GET /health` | proxied | `{"status": "ok"}` |
| `GET /props`, `GET /v1/models` | proxied | from the resident model (or the startup model, loaded) |
| `POST /tokenize`, `POST /apply-template` | — | as llama-server's |
| `POST /slots/0?action=save\|restore\|erase` (engine address) | llama-server's own: the engine address is llama-server itself, and `--spawn` passes `--slot-dir` as `--slot-save-path` | KV cache to and from `--slot-dir` (0.3.8) |
| `POST /v1/chat/completions` | proxied, receipt added | `native_chat`, receipt added |
| `/` and `/api/*` | — | `ollama::route` (see [ollama.md](ollama.md)) |

```sh
curl -s 127.0.0.1:18093/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello. /no_think"}],"max_tokens":32}'
```

### The receipt

Non-streamed: a top-level `bankml_receipt` object. Streamed: one extra `data: {"bankml_receipt": …}` event before
`data: [DONE]` (OpenAI clients ignore it). With it a client can check that the text it shows is the text this
gateway received from the engine serving the verified file, for this request. Fields, from `Tally::receipt`:

`bankml`, `engine`, `model_sha256`, `guard`, `prompt_tokens`, `completion_tokens`, `ttft_ms`, `wall_ms`,
`response_sha256` (of the answer text exactly as produced), `request_sha256` (of the request body), `signed: false`.

Receipts are not signed. They prove integrity between a client and its own `bankml serve`, not to a third party.
In `--native` mode `ttft_ms` is measured for non-streamed answers too (0.3.7; before, it was `null`).

### Loopback rules

| condition | answer |
|---|---|
| `Host` not `127.0.0.1`, `localhost` or `[::1]` (any port) | 403 |
| `Transfer-Encoding` on a request | 400, "send Content-Length" |
| body over 16 MiB | 413 |
| `POST` without `Content-Type: application/json` | 415 (`curl -d` alone sends a form type) |
| more than 32 open connections | 503, answered off the accept thread after draining what the client sent for at most 200 ms, so a slow client cannot hold the accept loop |
| request line or header line over 16 KiB, or more than 100 header lines | 400 |

A refusal reads and discards the body it was sent before closing, so the client does not lose the reason to a
connection reset. Connections get a 30 s read timeout and a 120 s write timeout. An upstream body or chunk over
64 MiB is refused before allocation. SIGTERM, SIGINT and SIGHUP stop a spawned engine too.

### `--native` `/v1/chat/completions`

`native_chat` resolves `model` through the registry and the derived models (`create::resolve`; absent means the
startup model), applies a derived model's layer (`apply_openai`), and refuses a malformed `stop` or a layer
`num_ctx` above `--ctx` before any load. It then acquires the model and keeps it resident afterwards: the OpenAI API
names no keep-alive.

`NativeChat::parse_ctx(eng, req, text, num_ctx)` builds the request: stop strings; the constraint
(`response_format`, `json_schema`, `grammar`, read from the request's own text so a schema's numbers are exact); a
grammar check with llama.cpp's parser (an unparsable grammar is a 400 with the parser's reason, not a failed run); the prompt (fitted under `num_ctx`); the sampler parameters; and `max_tokens`
or `n_predict`. Since 0.3.6 it also builds the sampler once, so a negative `repeat_last_n` or a repeat penalty of 0
or below is a 400 with llama-server's message (the live penalty oracle had found it answered 500). `NativeChat` is
shared with the C API's `bankml_chat` since 0.3.2, so both answer a request the same way; the constraint fields
arrived in 0.3.3 and `num_ctx` fitting under a derived model in 0.3.5. `NativeChat::run` passes the answer through the stop filter and the
JSON content stream. The non-streamed response (`completion_json`) carries OpenAI's object, llama-server's `timings`
(`timings_json`: `cache_n`, `prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`,
`predicted_per_second`, measured in the engine since 0.3.7), `logprobs` when asked for, and the receipt. A streamed
answer's last chunk carries `usage` and `timings.cache_n`. In `timings`, a rate is `null` when its count or time
is 0. The model stays resident after a `/v1` answer (the OpenAI API has no keep-alive; 0.3.0 likewise kept its one
model for good).

| item | what it does |
|---|---|
| `NativeChat::parse(eng, req, text)` / `parse_ctx(…, num_ctx)` | the request, parsed and checked before any token is computed |
| `NativeChat::run(self, eng, t: &mut Tally, stream, emit: impl FnMut(&str) -> bool)` | the answer as text pieces |
| `NativeChat::run_steps(self, eng, t, stream, emit: impl FnMut(Chunk) -> bool)` | the answer as `Chunk`s (0.3.8); `run` is built on it |
| `Chunk { text, entry: Option<TokenLogprob>, first }` | one streamed piece: its text, the logprobs of the token that released it, and whether it is the first token's (the role delta goes with it) |
| `completion_json`, `timings_json` | the non-streamed answer and its `timings` |
| `exceed_context_json(n_prompt, n_ctx)` | llama-server's 400 body for a prompt that does not fit (0.3.8) |
| `logprob_entries`, `logprobs_json`, `utf8_complete_len` | llama-server's logprobs rules and JSON (0.3.8, below) |
| `slot_filename_ok(name)` | llama.cpp's `fs_validate_filename` (0.3.8, below) |

`--native` reads `BANKML_CACHE_RAM` (the host prompt cache's size, [prompt_cache.md](prompt_cache.md)) and
`BANKML_CACHE_TYPE` (`f16` or `q8_0`, the KV cache's type, [native.md](native.md)) when a model opens.

### At the edges: the context limit, slots, logprobs (0.3.8)

- **Context limit.** `parse_ctx` refuses a prompt of `n_ctx` tokens or more with `exceed_context_json(n_prompt,
  n_ctx)`, llama-server's `exceed_context_size_error` body; an `Err` that starts with `{` is sent as that JSON with
  status 400. A generation stops when prompt and completion fill the context, `finish_reason` `"length"`.
- **Slots.** `slots()` answers `POST /slots/{id}?action=…` on the engine address. `slot_filename_ok` is llama.cpp's
  `fs_validate_filename` without subdirectories: no path separators or reserved characters (`: * ? " < > | / \`),
  no control characters (C0, DEL, C1), no U+FFFD, BOM, U+FF0E, U+2215 or U+2216, no leading or trailing space, no
  trailing dot, at most 255 bytes. `error_json` writes llama-server's
  error object. Without `--slot-dir` the answer is llama-server's 501; any id but 0 is "Invalid slot ID"; a failed restore is
  a 400 and a failed save a 500, as llama-server answers. The KV
  file itself is `Native::save_slot` / `restore_slot` (see [native.md](native.md)).
- **Logprobs.** `logprobs` + `top_logprobs` set `Params::n_probs`; the engine returns per token its probability and the
  top `n` (`sampler::token_probs`) with whether its UTF-8 was complete. `logprob_entries` applies llama-server's rules
  — an entry only after complete UTF-8, carrying the text the stop filter released since the previous entry; a stop
  word's own tokens (`tokenize(stop, add_special = false)`) dropped — and `logprobs_json` writes them, a top token's
  text cut by `utf8_complete_len` (`validate_utf8`) and otherwise U+FFFD-replaced, as nlohmann's `dump` writes it.
  Streamed, `NativeChat::run_steps` turns the engine's steps (`Native::complete_with`) into `Chunk`s: text only at
  whole tokens, each with that token's entry, the first behind the role delta (sent only when logprobs are asked
  for, so plain streams are unchanged); the end-of-turn token releases what a partial stop match held.

### `Json`

`pub enum Json { Null, Bool, Num(f64), Str, Arr, Obj(Vec<(String, Json)>) }` with `parse`, `get` (the last key wins),
`idx`, `as_str`, `as_bool`, `as_u64`. Numbers are kept as `f64` (token counts are small integers). Nesting deeper
than 128 is refused. A high surrogate escape joins a following low one; an unpaired one becomes U+FFFD, as the UI
does, and an escape after it that is not a low surrogate is read on its own.

## How it is verified

- Unit tests: `json_parses_what_llama_server_sends`, `chunked_and_sized_bodies`, `heads_are_bounded`,
  `only_loopback_hosts`, `unpaired_surrogates_become_replacement_characters`, `file_identity_changes_are_seen`.
- `testing/cli.rs` against a mock llama-server: `serve_gates_and_signs_answers` (the `response_sha256` equals the
  sha256 of the text the mock sent, streamed and not; `request_sha256` the body's) and
  `serve_refuses_an_unverified_model_or_a_different_upstream_file`; `hostile_headers_refuse_not_crash`.
- `--native` is covered by the live gate stages: `serve_oracle_ollama_shape` (`testing/serve_oracle.py --bankml`),
  `json_oracle_live`, `json_schema_oracle_live`, `o4_live`, `persona_oracle_live`, `penalty_oracle_live` (85 / 85
  through `/v1` and `/api/chat` on mindx-gen39, CHANGELOG 0.3.6), `sampler_oracle_live` (0.3.7).
- `capi_chat_oracle`: `bankml_chat` equals `serve --native` turn by turn, receipt hashes included.
- 0.3.8: `context_oracle_live` (8 / 8 against llama-server at `-c 256`), `slot_oracle_live` (19 / 19: answers after a
  restore equal an empty slot's, across a restart; llama-server's refusals), `logprobs_oracle_live` (14 / 14, five
  streamed; every logprob the same float), `session_oracle_live` (14 / 14: interleaved conversations through one
  slot and the host prompt cache, turn by turn; simultaneous requests queued); the unit test
  `logprob_texts_as_llama_server_writes_them`. Counts from CHANGELOG 0.3.8.
- 0.4.0: `kv_oracle_live` (6 / 6: `BANKML_CACHE_TYPE=q8_0` against llama-server `--cache-type-k q8_0 --cache-type-v
  q8_0`, text, counts and `cache_n`; CHANGELOG 0.4.0).

## Advantages and efficiency

- **What it adds over a bare llama-server.** The file is guarded and pinned before the engine sees it, the engine
  must be serving that file, and every answer is receipted against the text and the request. llama-server checks
  none of this.
- **Cheap per request.** The file is hashed once at start; each answer re-checks only its identity. docs/PERFORMANCE.md
  measures the gateway's own cost at 0.71 ms per request (`/health` 1.08 ms through serve vs 0.37 ms direct), and the
  sha256 pin of the 1.16 GB 8B model at 2.9 s with SHA-NI. A PGO build was measured and rejected for that reason
  (docs/TODO.md).
- **Streaming end to end.** Upstream events are forwarded line by line as they arrive; native pieces are written and
  flushed as they form.
- **Bounded by construction.** Fixed limits on connections, head size, body size and upstream bodies; no panic paths
  on request input (`Json::parse` returns `Option`, errors are `Result`); std-only HTTP and no external crates.
- **Cheap edges.** A prompt past the context is refused before any computation; a slot restore replaces the
  prompt's whole prefill with one sequential read checked by sha256; logprobs cost one partial sort of the vocabulary
  per token, and nothing when not asked for.
- **Next** (docs/TODO.md): an independent review of `serve` (loopback rules, limits, receipts) before 1.0; more than
  one slot with continuous batching in `--native` (O8).

## Limitations

- Loopback only, by design. View mode for the LAN is a separate Python page (usage.md §7).
- Receipts are unsigned.
- In P0 mode the tokens come from ggml's kernels in llama-server; bankML vouches for the file, the path and the
  transcript, not the arithmetic.
- `--native` serves one slot and one resident model (see [native.md](native.md)); `/slots/{id}` accepts only id 0.
- `/bankml/usage` samples for 250 ms under a lock, so a poller arriving during a sample waits for it.
- `/completion` (llama-server's own endpoint, and its `n_probs`) is not served natively; `/v1` and `/api` are.
- A plain stream (no logprobs) has no role delta, and leftover incomplete bytes at the very end are sent replaced
  by U+FFFD, where llama-server drops them.

## See also

- [../usage.md §6](../usage.md#6-by-hand-build-verify-serve) · [§9 Receipts](../usage.md#9-receipts-and-how-to-check-an-answer) ·
  [§13 Reference](../usage.md#13-reference-commands-ports-environment)
- [../oracles.md §5, the gateway](../oracles.md#5-the-gateway-receipts-against-the-text) · [../OLLAMA.md](../OLLAMA.md) ·
  [../CAPI.md](../CAPI.md) · [../TODO.md](../TODO.md)
- [native.md](native.md) · [ollama.md](ollama.md) · [create.md](create.md) · [grammar.md](grammar.md) ·
  [schema.md](schema.md) · [main.md](main.md)
