# `bankML/create.rs` — `bankml create`: derived models over pinned bases

## Summary

`create.rs` is Ollama's `ollama create` over bankML's gate (O5, 0.3.5). A **derived model** is a manifest that
layers configuration on a **pinned base**: a system prompt, parameters, stop strings, example messages, a licence.
It never copies weights. Loading one verifies the base exactly as a pinned model is verified (the guard, then the
sha256 pin of its FORK.json), then applies the layer.

It exists for mindX's flow: mindXtrain merges each generation into safetensors, and mindX's `promote.py` layers a
persona with `ollama create`. `bankml create` takes the same Modelfile. `FROM` a safetensors directory converts it
with `convert.rs` and pins the result, so the persona becomes a verified layer over a pinned GGUF.

The module holds Ollama's Modelfile parser, the manifest format, the derived-model store, the layer's application to
Ollama and OpenAI requests, the HTTP handlers for `/api/create`, `/api/delete` and `/api/copy`, and the CLI.

## Technical usage

### The manifest

```text
<registry>/<name>.MODEL.json   {kind, bankml, name, created_at, base: {name, file, sha256, fork, path},
                                layer: {base_sha256, system, parameters, stop, template_sha256, license, messages, requires},
                                digest: "sha256:…"}
```

`digest` is the sha256 of the layer's content (the base's sha256 and the layer), not of the name or the time. The
same Modelfile over the same base gives the same digest, and a copy keeps it. A manifest whose digest is not its
content's is refused when read: "edited by hand, refused — create it again". A manifest whose base pin has changed
is refused too. It is written atomically (a `.json.part` file, then a rename).

### The Modelfile (`parse_modelfile`, `Spec::from_modelfile`)

The parser is Ollama's `parser/parser.go`, state for state: case-insensitive instructions; `#` opens a comment only at
a line's start; a value runs to the end of the line; `"…"` and `"""…"""` may span lines, with no escapes. `quote()`
writes values back as Ollama's `ollama show --modelfile` does, and they round-trip.

| instruction | handling |
|---|---|
| `FROM` | exactly one: a registry name (pinned or derived), its suffix-less alias, a pinned GGUF path, or a safetensors directory (converted to `<name>-F16.gguf` and pinned with every input's sha256) |
| `SYSTEM`, `TEMPLATE` | the last wins; `TEMPLATE` only when it is the base's own chat template |
| `PARAMETER` | one of `PARAMS` (below) or `stop`; the last value wins, `stop` accumulates |
| `MESSAGE role content` | role `system`, `user` or `assistant`; recorded and applied |
| `LICENSE`, `REQUIRES` | recorded |
| `ADAPTER` | refused |

`FROM` a derived model inherits its layer: new values win, `stop` is replaced as a list, licences add up. A relative
`FROM ./x` on the CLI is relative to the Modelfile.

### `PARAMS`

```rust
pub const PARAMS: [(&str, bool); 11] = [("temperature", false), ("top_k", true), ("top_p", false), ("min_p", false),
    ("seed", true), ("num_ctx", true), ("num_predict", true), ("repeat_penalty", false), ("repeat_last_n", true),
    ("presence_penalty", false), ("frequency_penalty", false)];
```

The four penalties joined in 0.3.6 (O2). Floats are parsed at 32 bits, as Ollama parses them. Range checks at create:
`num_ctx` ≥ 1; `num_predict` ≥ −2; `repeat_penalty` > 0; `frequency_penalty` and `presence_penalty` any finite value
(negative too, as llama.cpp takes them); `top_k` and `seed` unchecked; the rest ≥ 0. A `PARAMETER` outside `PARAMS`
is refused with one of three reasons: a sampler not reproduced (`typical_p`, `tfs_z`, `mirostat*`,
`penalize_newline`, `min_keep`); a resource option that belongs to `bankml serve`; or not a parameter bankML knows.

### Applying the layer (Ollama's `server/routes.go`)

- `apply_ollama(&self, req, chat)`: the layer's parameters become defaults under the request's `options`, overridden
  key by key (`stop` as a whole list). For chat, the model's `MESSAGE`s go before the request's, and its `SYSTEM` goes
  first unless the request's first message is a system message. For a non-raw generate, the conversation goes in
  `bankml_messages`: the request's `system` if any, else the model's, then the `MESSAGE`s, then the prompt. A `raw`
  prompt gets neither.
- `apply_openai(&self, req)`: the same message rule; the parameters fill absent top-level fields (`num_predict` as
  `max_tokens`); `num_ctx` is not a field but fits the conversation (`num_ctx()`).

### Resolution

`resolve(rs, model)` tries a pin's exact name first, then a derived model's name, then the registry's suffix-less
alias. So promote.py's re-create in place (`FROM mindx-gen39` as `mindx-gen39`) gives the layer, and
`mindx-gen39-f16` the pin. A derived model loads through its base's entry, so it and its base share one residency.

### HTTP (needs `serve --native --registry`)

```sh
curl -s $B/api/create -H "$J" -d '{"model": "mindx-persona", "from": "mindx-gen39", "system": "You are mindX.", "parameters": {"temperature": 0.7}}'
curl -s $B/api/create -H "$J" -d '{"model": "p", "modelfile": "FROM mindx-gen39\nPARAMETER repeat_penalty 1.3"}'
curl -s $B/api/copy   -H "$J" -d '{"source": "mindx-persona", "destination": "mindx-persona-b"}'
curl -s -X DELETE $B/api/delete -H "$J" -d '{"model": "mindx-persona-b"}'
```

- `/api/create`: `{model, modelfile}` or Ollama's structured `{model, from, system, template, license, parameters,
  messages}`; status lines streamed as NDJSON unless `stream: false`. Without `--registry`: 400, nowhere to write.
- `/api/delete`: derived models only; a pin is refused ("bankml does not delete pins over HTTP").
- `/api/copy`: the same content and digest under a new name; a pin's copy is an empty layer over it.
- `/api/show` of a derived model (`show_json`): the reconstructed Modelfile, `parameters`, `system`, `license`,
  `messages`, `details.parent_model`, and `bankml.layer` with the digest and the base's sha256 and pin.

### CLI

`bankml create NAME -f Modelfile [--registry DIR] [--models DIR]` prints Ollama's status lines and exits 0, or
`bankml create: refuse: …` and 2. The registry defaults to `$BANKML_FORKS`, else `~/.local/share/bankml/forks`.

## How it is verified

- Unit tests: `modelfile_parses_as_ollama` (mindX's own Modelfiles, comments, CRLF, BOM, quoting, errors),
  `quote_round_trips_through_the_parser`, `spec_keeps_what_is_reproduced_and_refuses_the_rest`,
  `manifest_digest_is_the_content`, `the_layer_applies_as_ollama_applies_it`, `create_layers_over_a_pinned_base`,
  `http_create_show_copy_delete_round_trip`.
- `testing/cli.rs`: `create_writes_a_layer_over_a_pin`.
- `oracle_persona_layer` (`#[ignore]`, gate; `testing/persona_oracle.py`): promote.py's persona Modelfile created two
  ways, `FROM` the merged safetensors directory and `FROM mindx-gen39` in place. Asked the user's turns alone, each
  must give llama-server b11192's tokens for the same GGUF given the persona as the system message. CHANGELOG 0.3.5:
  27 / 27 each. Live: `persona_oracle_live` through `/api/chat`, `/api/generate`, `/v1` and `/api/ps`.

## Advantages and efficiency

- **No weights copied.** A layer is a small JSON file over a pinned GGUF; its digest is content-addressed, as
  Ollama's content addressing does. Every load of a derived model verifies its base (the guard, then the pin).
- **mindX's Modelfiles unchanged.** The parser is Ollama's state machine, so promote.py's files create the same layer,
  and `/api/show` gives back a Modelfile that recreates it.
- **Efficient.** Tamper detection is one sha256 over a short text. A derived model and its base share one load. A
  conversion never overwrites an existing pin.
- **Rust practice.** No external crates; atomic manifest writes; `RwLock` poisoning recovered; errors carry the
  reason as `Result<_, String>`.
- **Next** (docs/OLLAMA.md, docs/TODO.md): `ADAPTER` (LoRA merging) is a later O-phase; `promote.py --to bankml` in
  mindX and a bankml backend for mindXtrain (O5, proposed).

## Limitations

- `ADAPTER`: "bankML does not merge LoRA adapters yet (a later O-phase); merge it into the base weights … then FROM
  the merged safetensors directory".
- `TEMPLATE` other than the base's own: Ollama's are Go templates; bankML renders the GGUF's Jinja byte-identically.
- `/api/create` `files`, `adapters`, `quantize`: refused, each with its reason.
- More than one `FROM`; a name a pinned file already has; a `FROM` file not pinned in the registry directory.
- Model names: letters, digits, `.`, `_`, `-`, at most 128, no tag but `latest`, not ending in `.gguf`.
- A layer `num_ctx` above the server's `--ctx` is accepted at create (with a warning) and refused per request.
- The module's header comment lists the `PARAMETER`s without the penalties; `PARAMS` includes them since 0.3.6.

## See also

- [../usage.md, derived models and conversion](../usage.md#derived-models-and-conversion-bankml-create-bankml-convert-o5-035) ·
  [../OLLAMA.md, what O5's first cut built](../OLLAMA.md#what-o5s-first-cut-built-035) · [../oracles.md](../oracles.md) ·
  [../TODO.md](../TODO.md)
- [convert.md](convert.md) · [ollama.md](ollama.md) · [native.md](native.md) · [serve.md](serve.md) · [main.md](main.md)
