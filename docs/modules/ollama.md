# `bankML/ollama.rs` — Ollama's API on `bankml serve --native`

## Summary

`ollama.rs` gives `bankml serve --native` the API mindX and other Ollama clients speak (phase O1, 0.3.1), over
bankML's own engine and its gate. It is **native only**: nothing here proxies to llama-server or to Ollama. What the
verified forward pass cannot do is refused with HTTP 400 and `{"error": "…"}` saying why.

It maps Ollama's requests onto the engine in `native.rs` (one slot, one resident model, every load verified),
streams NDJSON as Ollama does, and puts the same `bankml_receipt` on the final object as `/v1/chat/completions`
carries. Derived models (`/api/create`, `/api/copy`, `/api/delete`) are handled in `create.rs`; this module routes
to them and applies their layer.

## Technical usage

### Endpoints (`route`)

| method, path | answer |
|---|---|
| `GET`/`HEAD /` | `bankml is running (Ollama's API at /api, native only)` |
| `GET /api/version` | `{"version": "<bankml version>", "bankml": true}` |
| `GET /api/tags` | every pin: `name`, `model` (`<name>:latest`), `modified_at`, `size`, `digest` (the pinned sha256), `details`, and `"bankml": {"native", "reason", "file", "found"}`; derived models with `details.parent_model` |
| `GET /api/ps` | the resident model, `expires_at` (for a model kept for good, now + about 292 years, Ollama's largest duration), `size_vram: 0`, `context_length`; named after the model whose request loaded it, as Ollama names its runner (a derived model's name, manifest digest and parent, with that request's context, since 0.3.5); a record left by weights since unloaded is not used |
| `POST /api/show` | `modelfile`, `parameters`, `template` (the GGUF's), `details`, `model_info` (header keys), `capabilities`, and `"bankml": {sha256, native, n_ctx}` |
| `POST /api/chat`, `POST /api/generate` | an answer, streamed (default) or not |
| `POST /api/create`, `POST /api/copy`, `DELETE /api/delete` | derived models ([create.md](create.md)) |
| `POST /api/embed`, `/api/embeddings`, `/api/pull`, `/api/push` | 400 with the reason (below) |

```sh
B=127.0.0.1:18093; J='Content-Type: application/json'
curl -s $B/api/chat -H "$J" -d '{"model": "bonsai-8b-q1_0", "messages": [{"role": "user", "content": "Name a prime."}],
    "stream": false, "options": {"temperature": 0.3, "seed": 7, "num_predict": 32}}'
curl -s $B/api/generate -H "$J" -d '{"model": "mindx-gen39", "prompt": "Hello", "stream": false,
    "options": {"temperature": 0, "repeat_penalty": 1.3}}'
curl -s $B/api/generate -H "$J" -d '{"model": "bonsai-8b-q1_0", "keep_alive": 0}'   # unload now
```

### Answers

- **Streaming** is on by default: `application/x-ndjson`, one object per line (`message` for chat, `response` for
  generate, `done: false`), then a final object with `done: true`, `done_reason` (`stop` or `length`),
  `total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`
  (nanoseconds), `bankml_cache_n` (prompt tokens reused from the cache) and `bankml_receipt`.
- `eval_count` includes the end-of-turn token, as llama-server counts it.
- **An empty prompt only loads or unloads**: `/api/generate` without `prompt`, `/api/chat` without `messages`. The
  answer's `done_reason` is `load` or `unload`.
- **`raw: true`** tokenizes `prompt` as given, with special tokens and no template; `system` is then ignored.
- **`format`**: `"json"`, `{}` and `{"type": "object"}` are JSON mode (llama-server's `json_object` grammar, since
  0.3.3); since 0.3.5 any other schema object gets the grammar llama-server b11192 builds for it
  (`grammar::from_ollama_text`, `schema.rs`), its numbers read from the request's raw text as llama-server reads them,
  and what b11192 refuses in a schema is refused.
- Messages: `thinking` (or `reasoning_content`) is carried as the turn's reasoning; only text content is in scope.

### `keep_alive`

`keep_alive(v, default)` reads it as Ollama does: a Go duration (`"5m"`, `"1h30m"`, `"2.5s"`, `"-1m"`) or a number of
seconds. `0` unloads after the answer; a negative value keeps the model for good; absent, `null` or `""` takes the
default. `KEEP_ALIVE_DEFAULT` is 5 minutes; `serve --keep-alive DUR` changes it. Anything else answers 400.

### Options (`options(o, n_ctx) -> Result<Opts, String>`)

| option | handling |
|---|---|
| `SAMPLING`: `temperature`, `top_k`, `top_p`, `min_p`, `seed`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `typical_p`, `min_keep` | passed to `native::sampling`, which honours or refuses each |
| `num_predict` | the token limit; a negative value means no limit (Ollama's `-1`, until the turn ends, and `-2`, until the context is full) |
| `num_ctx` | refused above the served `--ctx`; under it, the conversation is fitted as Ollama's `chatPrompt` fits it (`Native::prompt_fit`, since 0.3.5; `/api/generate` the same way) |
| `stop` | a string or an array of strings |
| `IGNORED`: `num_thread`, `num_gpu`, `main_gpu`, `num_batch`, `use_mmap`, `use_mlock`, `low_vram`, `numa`, `f16_kv`, `vocab_only`, `logits_all`, `num_keep` | accepted and ignored: they change scheduling, not the answer (the kernels' bits do not depend on threads or batch size) |
| `mirostat` (non-zero), `tfs_z` (≠ 1) | refused (llama.cpp removed tail-free sampling); at their neutral values, and `mirostat_eta`, `mirostat_tau`, `penalize_newline`, accepted |
| anything else | refused: "it refuses rather than ignore what might change the answer" |

0.3.6 (phase O2): `repeat_last_n` moved from `IGNORED` to `SAMPLING`. With the penalties reproduced it changes the answer. The
sampler is built once before any model is loaded, so a negative `repeat_last_n` or a repeat penalty of 0 or below is
refused up front with llama-server's message.

0.3.7 (unreleased): `typical_p` and `min_keep` are honoured, as llama-server applies them. DRY, XTC, top-n-σ and dynamic temperature
are not Ollama options, so an `options` key for them is refused as unknown; they are reached through
`/v1/chat/completions` and the C API ([sampler.md](sampler.md)).

### Stop strings

`StopFilter` passes on the text up to the first stop; the stop and what follows are dropped. A tail that might begin
a stop is held back until it is known not to be one, and released at the end if it never completes. `/v1` uses the
same filter. `StopFilter::matched` records the stop string that ended the answer, so `/v1` can drop that stop's own
tokens from the logprobs as llama-server does (0.3.8).

## How it is verified

- Unit tests: `keep_alive_as_ollama_reads_it`, `options_map_onto_the_engine` (including the penalties and, since
  0.3.7, `typical_p` reaching the engine), `refusals_say_why`, `ndjson_framing`, `stop_strings_end_the_answer`, `names_resolve_to_pins`,
  `timestamps_are_rfc3339`.
- `serve_oracle_ollama_shape` (gate, `testing/serve_oracle.py --bankml`): three conversations through `/api/chat`
  (the first streamed), `/v1/chat/completions`, and `/v1` again after unload and reload. Every turn equals the others
  and llama-server b11192's record: text, counts and cache reuse.
- `json_oracle_live`, `json_schema_oracle_live`: `/api/chat` with `format: "json"` and `format: <schema>`.
- `persona_oracle_live`: a derived model through `/api/chat`, `/api/generate` and `/v1`, and `/api/ps`.
- `penalty_oracle_live`: `options` penalties on mindx-gen39, refusals included (85 / 85 with `/v1`, CHANGELOG 0.3.6).
- `session_oracle_live` (0.3.8): interleaved conversations through the one slot and the host prompt cache; `/api/chat`
  and `/v1` share that slot, so the same rules hold on both.

## Advantages and efficiency

- **Ollama's shape, bankML's guarantees.** A client written for Ollama gets answers token-identical to llama-server
  b11192 on the oracles, a sha256-pinned model verified on every load, and a receipt on every answer. `/api/tags`
  lists pins bankML cannot play with the reason, instead of failing later.
- **Refuse, never approximate.** An unknown option is a 400, not silently dropped, so an answer is never quietly
  different from what was asked.
- **Efficient lifecycle.** Refusals (options, sampler, format) are checked before a model is loaded for nothing. One
  model is resident; `keep_alive` releases it and its memory map. `/api/chat` and `/v1` share one slot, so the prompt
  cache carries across both APIs.
- **Rust practice.** std-only HTTP and JSON; RFC 3339 timestamps (UTC, nanoseconds, as Ollama writes `created_at`)
  computed in-crate with H. Hinnant's civil-from-days algorithm; errors as `Result<_, String>`.
- **Next** (docs/OLLAMA.md, docs/TODO.md): tool calls (O6), embeddings with an encoder graph (O7), more than one slot
  and continuous batching (O2, O8), ContextShift past `num_ctx` (O2).

## Limitations

Refused, with the reason in the source:

- `tools` and `tool_calls`: "tool calling needs the template's tool rendering and a grammar (O6 …)".
- `images`: "bankML's models are text-only; vision is out of scope".
- `suffix` (fill-in-the-middle), `template` (a replacement template), `context` (send messages instead).
- `think: true`: the template is rendered with thinking off, as llama-server `--reasoning off`; `think: false` passes.
- `format` with `raw: true`: the grammar begins with the template's generation prompt, which a raw prompt lacks.
- `/api/embed`: "embeddings need an encoder graph (bge-m3 is XLM-R …) … (O7)"; `/api/pull`: models are imported
  sha256-pinned with open licences only; `/api/push`: bankml publishes nothing.
- One resident model and one slot (see [native.md](native.md)).
- DRY, XTC, top-n-σ and dynamic temperature are not Ollama options, and logprobs (0.3.8) are wired only on `/v1`.
  For any of them, use `/v1/chat/completions`.

## See also

- [../usage.md §6a, Ollama's API](../usage.md#6a-ollamas-api) · [../OLLAMA.md](../OLLAMA.md) ·
  [../oracles.md](../oracles.md) · [../TODO.md](../TODO.md)
- [native.md](native.md) · [serve.md](serve.md) · [create.md](create.md) · [sampler.md](sampler.md) ·
  [grammar.md](grammar.md) · [schema.md](schema.md)
