# bankML as mindX's Ollama, and as its replacement for llama.cpp

*Written for 0.3.1 (2026-10-01), updated for 0.3.5 (2026-10-02). Phase O1 shipped in 0.3.1, O6's first cut (JSON
mode) in 0.3.3, O4 (the Llama graph, tied embeddings) with O3's F16 kernels in 0.3.4, and O6b (JSON schemas) with O5's
first cut (`bankml convert`, `bankml create`) in 0.3.5; every later phase is a plan, folded into the milestones of
[TODO.md](TODO.md). Nothing here counts until its oracle passes in a gate record.*

## The question, and the short answer

Can bankML become a complete, optimized Ollama for mindX, and replace llama.cpp there? **As a direction, yes.
Today, no.** bankML is narrow on purpose:

- **What it runs.** Qwen3 in `Q1_0` and `Q2_0_g64`, tied embeddings or not (Bonsai-8B, Ternary-Bonsai-8B,
  Bonsai-1.7B), and since 0.3.4 the Llama architecture in F16 (SmolLM2-135M-Instruct, and mindX's own `mindx-gen39`),
  with AVX2 (plus a verified Vulkan GPU share for 1-bit). One conversation slot.
- **What it proves.** It is token-identical to llama.cpp b11192 on every oracle in the gate, and about 8× faster on
  the ternary model; on the F16 Llama models it is within about 10 % of llama-server's speed (PERFORMANCE.md).
- **What mindX asks of Ollama.** Of the models mindX serves through Ollama today:
  - `mindx-genN`: SmolLM2, Llama architecture, F16 — **native since 0.3.4** for every generation converted and
    pinned, and since 0.3.5 made by `bankml create` from mindXtrain's merged output with promote.py's persona layer
    (gen 39 answers token-identically to llama-server given the same persona);
  - `qwen3`: `Q4_K`, and the `qwen3:0.6b` pin in `Q8_0` — refused (O3);
  - `bge-m3`: an XLM-R encoder — refused (O7).

**The decision (the operator's): native only.** bankML answers only what its own verified forward pass can do. It
never proxies to llama-server or to Ollama to fill a gap; it refuses with HTTP 400 and a reason. Ollama keeps
running beside it until native coverage catches up. The field agrees with the narrow scope. Ollama left llama.cpp in
2025 for its own ggml-based engine, met regressions llama.cpp had already fixed, and went back to depending on
llama.cpp in **v0.30.0 (May 2026)** (the field survey below). Replacing llama.cpp across the board is a treadmill.
bankML wins where it owns the format, proves every bit, and grows one oracle-backed architecture at a time.

## The gap matrix: what mindX calls, and where bankML stands

The evidence is mindX's own code (file:line in the mindX repository, read 2026-10-01).

| mindX asks for | where (mindX) | bankML | phase |
|---|---|---|---|
| `POST /api/generate`, non-streamed; reads `prompt_eval_count` and `eval_count` | `llm/ollama_handler.py:312` | **done (0.3.1)**: Ollama's counts and durations (ns), plus `bankml_receipt` | O1 |
| `format: "json"` on generate | `llm/ollama_handler.py:121`, `:265` | **done (0.3.3)**: llama-server b11192's `json_object` grammar, its prefill and its redraw; token-identical, greedy and seeded. **JSON schemas done (0.3.5)**: `format: <schema object>` gets the grammar llama-server b11192 builds on the model's template (173 of 173 schemas on each of the three templates against libllama-common), and the answers are token-identical (Bonsai-8B 28 / 28, ternary 11 / 11, Bonsai-1.7B, SmolLM2 and mindx-gen39 56 / 56 each, greedy and seeded, length cuts included) | O6 |
| `/api/chat` and `/api/generate` with `keep_alive: "5m"`, `options.num_predict`, `options.temperature` | `api/ollama/ollama_url.py:219-243` | **done (0.3.1)** | O1 |
| passthrough fields `format`, `system`, `template`, `raw`, `suffix`, `images`, `think` | `api/ollama/ollama_url.py:250-256` | `system`, `raw`, `format: "json"` and `format: <schema>` (0.3.5) **done**; `think: false` accepted; `format` with `raw`, `template`, `suffix`, `images` and `think: true` **refused** with the reason | O1 · O6 |
| `/api/tags` and `/api/ps` (the lineage models, which are resident, `expires_at`) | `agents/storage/hf_client.py:3728-3753` | **done (0.3.1)**: every pinned model, `digest` = the pinned sha256, plus `"bankml": {"native", "reason"}` | O1 |
| load and unload by an empty `/api/generate` with `keep_alive` (`done_reason` `load` / `unload`) | `agents/storage/hf_client.py:3760-3777` | **done (0.3.1)**; each load runs the full guard + sha256 pin | O1 |
| `ollama_predict`: `/api/chat` with `temperature 0`, **`repeat_penalty 1.3`**, `num_ctx 2048` | `agents/storage/hf_client.py:2423-2431` | `num_ctx` **done** (0.3.5: a conversation longer than it is cut as Ollama cuts it, oldest messages first); `repeat_penalty` **done (O2, unreleased)**: llama-server b11192's penalties sampler, token-identical on its oracle | O2 |
| prune old generations with `DELETE /api/delete` | `agents/storage/hf_client.py:945`, `:974` | **derived models deleted (0.3.5)**; a pinned file is **refused by design** (not deleted over HTTP) | O5 |
| `/api/embed` with `bge-m3` (XLM-R encoder, 1024 dimensions), `truncate: true` | `agents/memory_pgvector.py:937-958` | **refused** (400): no encoder graph yet | O7 |
| `ollama create` from a Modelfile (`FROM` + `ADAPTER`), then a persona `SYSTEM` layer (`ollama show --modelfile`, re-create) | `mindx/godel/mindxtrain/promote.py:52-85`, `:181-323` | **done (0.3.5)**: `bankml create` and `/api/create` — `FROM` the merged safetensors directory (converted byte-identical to llama.cpp b11192), a pinned GGUF, a registry name or its suffix-less alias, `SYSTEM`, `PARAMETER`, `stop`, `MESSAGE`; `/api/show` gives the Modelfile back. promote.py's own persona Modelfile, re-created in place (`FROM mindx-gen39`), and the same layer `FROM` the merged directory both answer token-identically to llama-server given the persona (27 / 27 each). `ADAPTER` **refused** (merge first) | O5 |
| `ollama show <tag>` to check a base's architecture | `mindx/godel/mindxtrain/promote.py:88-119`; `/api/show` at `llm/ollama_handler.py:479` | **done (0.3.1)**: `details`, `model_info` from the GGUF header, `parameters` from its sampling defaults | O1 |
| the models themselves: SmolLM2-135M `mindx-genN` (Llama architecture, F16 safetensors merged to GGUF) | `mindx/godel/mindxtrain/promote.py` | **done (0.3.4)** for a pinned conversion: `mindx-gen39` (asked for as Ollama's tag `mindx-gen39`) token-identical to llama-server b11192, greedy, seeded, `/v1`, `/api`, C API, JSON mode. Since 0.3.5 a new generation is converted and pinned by `bankml create … FROM <merged dir>` | O3 + O4 + O5 |
| the boardroom's `num_ctx 8192` | `daio/governance/boardroom.py:988-996` | accepted when `serve --ctx` is at least 8192; the f16 KV cache is what makes it costly | O8 (quantized KV) |

**Also refused, each with a reason:**
- `tools` and message `tool_calls` (O6, after JSON schemas);
- a JSON schema that llama.cpp b11192 itself refuses (an unknown type, an empty `enum`, a `$ref` outside the
  document, a pattern no regex reads, …), with its message; and the one kind it converts into a grammar that is
  not UTF-8 (a non-ASCII character right before a quantifier in a `pattern`), which bankML refuses rather than
  reproduce (O6b);
- `images` (vision is out of scope);
- `/api/pull`: import through the pinned importer instead;
- `/api/push`;
- `mirostat`, `tfs_z`, and any option bankML does not know.

**Accepted and ignored**, because they change scheduling, not the answer bankML gives:
`num_thread`, `num_gpu`, `main_gpu`, `num_batch`, `use_mmap`, `use_mlock`, `low_vram`, `numa`, `f16_kv`,
`vocab_only`, `logits_all`, `num_keep`. `num_keep` matters only with a context shift, which bankML does not do.
(`repeat_last_n` left this list with O2: it sets the penalties' window, and is honoured.)

## The track: O1–O8, folded into the milestones

Each phase is ordered by what mindX needs, and sits inside an existing milestone of [TODO.md](TODO.md), never beside
it.

| phase | content | milestone | what it retires in mindX |
|---|---|---|---|
| **O1** | Ollama's native API, a model registry and residency (**0.3.1, done**) | 0.3.x | — |
| **O2 (in progress)** | The sampler chain, including repeat, presence and frequency penalties (`last_n`, **first cut done, unreleased**: token-identical to llama-server b11192 greedy and seeded, the prompt in the window), each with a seeded oracle; behaviour at the context limit; more than one slot. **Top item since 0.3.5**: mindXtrain's imprint gate needs `repetition_penalty 1.3` (and `no_repeat_ngram_size 3`), and `mindx-gen39` degenerates without one | 0.4.0 | the coach's `repeat_penalty: 1.3`; mindXtrain's imprint gate on bankML |
| O3 | `Q8_0`, **F16 (0.3.4, done)** and BF16 weight kernels, then `Q4_K`, each bit-exact against ggml | 0.3.4 → 0.6.0 | opens the standard-quant Qwen3 family |
| **O4** | The Llama architecture, tied embeddings (which also opens Bonsai-1.7B), SmolLM2's tokenizer and templates (**0.3.4, done**) | 0.3.4 | **`mindx-genN` served natively** |
| **O5** | `bankml create`: a Modelfile subset (`FROM` a pinned GGUF or a safetensors directory, `SYSTEM`, `PARAMETER`, `stop`, `MESSAGE`) recorded beside the FORK.json pins, and a Rust safetensors → GGUF converter for the merged SmolLM2, byte-identical to llama.cpp b11192's (**0.3.5, first cut done**: mindX's persona layer token-identical end to end); then `promote.py --to bankml` in mindX, and a bankml backend for [mindXtrain](https://huggingface.co/PYTHAI/mindXtrain) ([proposed](https://huggingface.co/PYTHAI/mindXtrain/discussions/1)) | 0.3.5 → 0.7.0 (beside mindXtrain in Rust) | `ollama create` |
| **O6** | A grammar engine: **JSON mode and GBNF (0.3.3, done)**; **JSON schemas (`json_schema_to_grammar`, 0.3.5, done)**; then tool calls | 0.3.3 → 0.6.0 | **`format: "json"` and `format: <schema>`, used across mindX** |
| O7 | The encoder graph (XLM-R: LayerNorm, bidirectional attention, CLS pooling), `/api/embed` and `/v1/embeddings`, with a bge-m3 oracle against llama.cpp's embedding output | 0.6.0 | step 2 of the embedding cascade |
| O8 | Optimisation: continuous batching across slots, AVX-512 and VNNI, NEON, a quantized KV cache, and 1-bit decode at least at llama-server's speed | 0.4.0 / 0.5.0 | — |

### What O1 built (0.3.1)

- **`bankML/ollama.rs`**: the Ollama shape over the existing engine:
  - `GET /api/version`, `/api/tags`, `/api/ps`; `POST /api/show`, `/api/chat`, `/api/generate`;
  - NDJSON streaming, Ollama's default;
  - option mapping and the refusals above;
  - stop strings, which `/v1/chat/completions` now honours too.
- **The registry and residency** (`native.rs`):
  - **Registry.** `bankml serve --native --registry [DIR]` names every GGUF pinned in the forks directory: the file's stem, lower-cased (`bonsai-8b-q1_0`, `ternary-bonsai-8b-q2_0_g64`). `:latest` and the file name are accepted as aliases.
  - **One resident model.** As Ollama's `MAX_LOADED_MODELS=1`. A switch drops the old model before the new one is read.
  - **Every load verified.** Each load runs the same `verify` as the start (the guard, then the sha256 pin).
  - **keep_alive.** `keep_alive` and `--keep-alive` (default `5m`) drop an idle model and its memory map.
  - **One engine for both APIs.** `/v1/chat/completions` goes through the same resolver, so both APIs share one engine and one slot.
- **Honest listing.** `/api/tags` lists a pin bankML cannot play with the reason:
  - `qwen3-0.6b-q8_0`: "weights are Q8_0 … O3";
  - `bonsai-1.7b-q1_0` said "tied embeddings (no output.weight) … O4" until 0.3.4, which plays it.
  - A startup model of that kind is now refused at start. In 0.3.0 it started and then failed every request.
- **The oracle** (`testing/serve_oracle.py --bankml`, in the gate). It runs the 0.3.0 conversations through three paths:
  - `/api/chat`, the first conversation streamed;
  - `/v1/chat/completions`;
  - `/v1/chat/completions` again, after unload and reload.

  All three must equal each other and llama-server b11192's record, turn by turn.

### What O6's first cut built (0.3.3)

- **What llama-server actually does with `response_format: {"type": "json_object"}`.** It is not `grammars/json.gbnf`.
  On the jinja chat path (Savante's flags) the server turns the schema `{"type": "object"}` into a PEG parser, and
  that parser into GBNF. The GBNF's root begins with the template's generation prompt
  (`<|im_start|>assistant\n` and the empty `<think>` block). It allows an optional ```` ```json ```` fence, then the
  object, with `space ::= | " " | "\n"{1,2} [ \t]{0,20}` around it. The sampler then *accepts the generation
  prompt's seven tokens* into the grammar before the first draw. The answer's `content` is the JSON value alone,
  without the fence. bankML carries that grammar text as a constant, and the oracle checks it on every recorded
  request against the `generation_settings.grammar` the server reports.
- **`bankML/grammar.rs`: llama.cpp's grammar engine, ported** (MIT, attributed; zero crates). It holds:
  - the GBNF parser (repetitions, character classes, token terminals, comments);
  - the left-recursion check;
  - the pushdown stacks and `advance_stack`;
  - `reject_candidates` over code points, with partial UTF-8 carried across tokens;
  - accept and apply, the end-of-generation rule, and llama-vocab's end set. That set is 6 tokens on Qwen3, not 2:
    an answer under the grammar did end on `<|file_sep|>`.
- **Where it sits** (`sampler.rs`, as `common_sampler_sample`). The chain draws first. If the token passes the grammar,
  it stands. If it does not, the logits are masked and the chain draws again from the same `mt19937`, so a seeded
  answer consumes the generator exactly as llama-server's does.
- **Surfaces.**
  - `/v1/chat/completions`: `response_format` (`json_object`, or `json_schema` whose schema is `{}` or
    `{"type": "object"}`), the top-level `json_schema`, and `grammar` (any GBNF; a user grammar, no prefill).
  - `/api/chat` and `/api/generate` with `format: "json"`.
  - `bankml_chat` (C API).
  - `bankml generate --json`.

  Everything else is refused with the reason.

### What O4 built (0.3.4)

- **The models, pinned.** No F16 GGUF is published for SmolLM2-135M-Instruct or for any `mindx-genN`, so each is
  converted from its pinned safetensors by llama.cpp b11192's own `convert_hf_to_gguf.py --outtype f16`, unmodified.
  The conversion is reproducible (twice, the same sha256), every tensor was checked against the safetensors rounded
  to f16, and the FORK.json records the source's LFS sha256, the revision and the tools (`sAGI/models.py`
  `CONVERTED` / `pin_converted`):
  - `SmolLM2-135M-Instruct-F16.gguf` (`e9aba089…`) from HuggingFaceTB/SmolLM2-135M-Instruct@`12fd25f7`, Apache-2.0;
  - `mindx-gen39-F16.gguf` (`6b64c748…`) from the dataset PYTHAI/mindXascension@`4bd31b9d`,
    `weights/gen39/ollama_push/merged/`, Apache-2.0 (the card; base SmolLM2-135M). Generation 39 is mindXtrain39,
    the last generation the imprint gate accepted.
- **What llama.cpp does, and bankML now does** (CHANGELOG 0.3.4 has the detail): the Llama graph (NORM RoPE, no Q/K
  norms); tied embeddings (the logits from `token_embd`); F16 products by the two paths ggml takes, chosen by the
  product's shape; only the output rows through the last layer; SmolLM2's `smollm` pre-tokenizer; SmolLM2-Instruct's
  template and the plain ChatML of `mindx-genN`; JSON mode with the grammar llama-server builds for a ChatML template.
- **Surfaces.** The new models are `native: true` in `/api/tags`; `/api/chat`, `/api/generate`, `/v1/chat/completions`,
  `bankml_chat` and `bankml generate` answer them; a name without its type suffix (`mindx-gen39`) resolves to the one
  pin with that base.
- **Ollama's persona layer is not in the GGUF.** mindX's `promote.py` puts each generation's persona in its
  Modelfile's `SYSTEM`; bankML serves the GGUF, so that persona is the caller's system message until `bankml create`
  (O5) records a Modelfile subset in FORK.json.
- **Still refused, with the reason:** Q8_0 (the Qwen3-0.6B pin: its own template differs from the Bonsai one on 4 of
  317 oracle conversations), BF16, Q4_K, other architectures, and any Llama variant outside what was proven (biases,
  fused QKV, rope factors or scaling, experts).
### What O6b built (0.3.5)

- **`bankML/schema.rs`: `json_schema_to_grammar`, ported** (MIT, attributed; zero crates): the schema reader
  (`json-schema.cpp`: which keywords decide a node, `$ref` into the same document, the errors), the converter
  (rule naming and de-duplication, required / optional / additional properties, `_not_strings`, tuples, integer
  ranges, the regex → GBNF translation, formats, the primitives), and nlohmann's JSON as the grammar text depends on
  it (key order, duplicate keys, `1` vs `1.0`, its float printing).
- **The chat path's wrapping, per template.** With thinking off, the schema's rules share one converter with the PEG
  parser's: the seven `json-*` rules, `response-format` and `root`; on the Qwen3 template also the `until-13`
  reasoning rules. On the two ChatML templates (SmolLM2-Instruct's, mindx-genN's) the parser has no reasoning block:
  no `until-13` rules, and the root is the generation prompt then the value — read from llama.cpp's own output on each
  template, not assumed. `{"type": "object"}` gives exactly each template's JSON-mode grammar (a unit test).
- **The oracle is llama.cpp's own code**: `testing/schema_oracle.cpp` calls `json_schema_to_grammar` and
  `common_chat_templates_apply` inside the b11192 release's `libllama-common.so` (no model, no server). Over 173
  schemas — llama.cpp's 81 test cases, Pydantic-shaped schemas like mindX's, the model oracle's nine, edge cases —
  bankML's text is byte-identical on both paths and on each of the three templates (148 grammars each), and every
  refusal carries llama.cpp's message (24 bare, 20 on the chat path).
- **Surfaces.** `response_format` `json_schema` (OpenAI's `{"name", "strict", "schema"}`), `json_object` with a
  `schema`, the top-level `json_schema`, Ollama's `format: <schema>`, and `bankml_chat`: all through JSON mode's
  prefill, redraw, fence and content rule (a top-level string, number or literal is its own content). The schema is
  read from the request's own text, so a float literal stays one. What b11192 ignores (a non-object schema inside
  `response_format`) is ignored; what it fails on (a non-object or `null` top-level `json_schema`) is refused.
- **Answers, against llama-server b11192** (`testing/json_schema_oracle.py`, `oracle_json_schema*`): objects with
  required and optional fields, enums, ranges, nested `$defs`, a pattern, a date format, a top-level array, enum,
  string, integer and number, through all three request shapes, greedy and seeded, and answers cut by `max_tokens`
  inside an object, a top-level string and a top-level number — token-identical: Bonsai-8B **28 / 28**, the ternary
  **11 / 11**, Bonsai-1.7B, SmolLM2-135M-Instruct and mindx-gen39 **56 / 56** each; live through `/v1` (one
  streamed) and Ollama's `/api/chat` with `format: <schema>`.
- **The content rule, from llama.cpp's own parser** (`testing/content_oracle.py`, `oracle_json_content`):
  `common_chat_parse` on every prefix of every recorded constrained answer and on edge cases — **30,063 of 30,063**
  texts on each template. It found two things the answer oracles alone had not: an answer cut inside an escape
  (`\` or `\u` with fewer than four hex digits) ends its string before the escape; and when the parse is empty (an
  answer cut inside or right after the opening fence) llama-server answers with the raw text
  (`to_json_oaicompat_chat`), which a non-streamed bankML answer now does too (`grammar::json_message`).

### What O5's first cut built (0.3.5)

O5 is what mindX needs from [mindXtrain](https://huggingface.co/PYTHAI/mindXtrain) (archived source:
[github.com/Professor-Codephreak/mindXtrain](https://github.com/Professor-Codephreak/mindXtrain)): mindXtrain merges each
generation into safetensors and serves it with `serve --to ollama`, and mindX's `promote.py` layers the persona with
`ollama create`. `bankml convert` and `bankml create` replace that path: the merged directory becomes a pinned GGUF and
the persona a verified layer. mindXtrain now has a bankml backend module, proposed in [PYTHAI/mindXtrain discussion #1](https://huggingface.co/PYTHAI/mindXtrain/discussions/1) (a backend,
`serve --to bankml`, `imprint-bankml`); it lives in mindXtrain, not in this release.

**A finding from it, and what it reorders.** Served greedily without a repetition penalty, `mindx-gen39` degenerates
into repetition (`,,,,` / `?||`). mindXtrain's canonical imprint gate decodes with `repetition_penalty 1.3` and
`no_repeat_ngram_size 3`, and bankML refuses both today (no oracle yet). So **O2 — the penalty samplers, each
oracle-exact against llama.cpp — is now the top item** for that gate to run on bankML. (Note that
`no_repeat_ngram_size` is a transformers decoding rule, not a llama.cpp sampler: its oracle is transformers'
`generate`, as for the 0.7.0 probe stage.)

- **`bankml convert DIR -o OUT.gguf` (`convert.rs`): llama.cpp b11192's `convert_hf_to_gguf.py --outtype f16`, byte
  for byte, for the Llama architecture as SmolLM2 uses it.** Read from the tag's `conversion/{base,llama}.py` and
  `gguf-py`, and written in its order of operations:
  - the metadata in llama.cpp's order, with the defaults `AutoConfig(LlamaConfig).to_dict()` adds (`head_dim`, hence
    `key_length`/`value_length`; `rope_theta` 10000; `rms_norm_eps` 1e-6);
  - the name heuristics of `gguf-py/metadata.py` (`general.name`, `basename`, `finetune`, `size_label` from the
    directory's name; the label from the weights counted otherwise); 168 of 168 Hub-style ids agree with gguf-py;
  - the tensors in name order (the safetensors reader sorts), Q/K rows permuted back from HF's rotate-half layout,
    norms in F32, matrices in F16 by round-to-nearest-even, BF16 read exactly; tied embeddings write no output matrix;
  - the gpt2 tokenizer path: tokens and types, merges, the special ids from `tokenizer_config.json` then
    `config.json`, the chat template (`chat_template.jinja` too), and Llama's `add_bos_token = false` at 49,152 tokens.

  **Result (2026-10-02):**

  | input | bankml convert | llama.cpp b11192 |
  |---|---|---|
  | HuggingFaceTB/SmolLM2-135M-Instruct @ `12fd25f7` (`model.safetensors` `5af571cb…`) | `e9aba089704487f72efa3c6cbb6d4c748d4c515428247f97679063137e45a222` | the same |
  | PYTHAI/mindXascension `weights/gen39/ollama_push/merged` @ `4bd31b9d` (`19b62829…`) | `6b64c748d96ad26fd72402299bd27b2ae82f489bd0469498dd18eb6054058266` | the same |

  The oracle files are O4's conversions with b11192's own script (`.models/*-F16.gguf`). The pins, with every input's
  sha256, are `testing/pins/*.FORK.json`. A conversion takes 2.6 s and maps the safetensors read-only. **0.3.5, run
  directly:** llama.cpp b11192's `convert_hf_to_gguf.py` (torch 2.14.1+cpu, transformers 5.18.0) and `bankml convert`
  on the same freshly downloaded directories wrote the same bytes — gen39 `6b64c748…` (the pin), SmolLM2 `ec30a679…`
  (its directory named `SmolLM2-135M-Instruct`; the pin's was named so that `general.name` reads `Smollm2 135m
  Instruct`). **The name is part of the bytes:** both converters take `general.name` from the directory's name, so the
  same weights in a directory named `merged` give another sha256 — for llama.cpp and bankML alike.
  - **What is refused, with the reason, rather than approximated:** other architectures; SentencePiece and Llama-HF
    vocabularies; rope scaling; biases; experts; a `README.md` model card (llama.cpp would write its licence, tags and
    base models; `--ignore-model-card` converts anyway and says the result will differ); and any tokenizer bankML has
    not seen named. llama.cpp names a BPE pre-tokenizer by hashing the token ids its *Python* tokenizer gives one fixed
    text (`chkhsh`). bankML has no Python, so it keys a table on the sha256 of the vocabulary and merges, plus the
    pre-tokenizer's shape, and records the `chkhsh` each entry gave (`testing/convert_oracle.py --chkhsh`).
  - **A finding:** gen39's `tokenizer.json` was re-saved by transformers 5.8. It **lost SmolLM2's `Digits`
    pre-tokenizer** (a bare `ByteLevel`), yet llama.cpp's hash still names it `smollm`. The vocabulary has no
    multi-digit tokens, so the two configurations tokenize alike on the hash text. Both are accepted, and that is why.
- **Derived models (`create.rs`).** A derived model is `<forks>/<name>.MODEL.json`: a layer over a pinned base, and
  never a copy of weights. It holds:
  - the base's name, file, sha256, FORK.json and path;
  - the layer: `system`, `parameters`, `stop`, `template_sha256`, `license`, `messages`, `requires`;
  - `digest`, the sha256 of that content. The name and time are not in it, so a copy has the same digest. A
    hand-edited manifest is refused when it is read.

  A load verifies the **base** exactly as a pinned model is verified (the guard, then its sha256 pin). A manifest whose
  base pin has changed is refused. Residency is keyed by the base, so a derived model and its base share one load.
- **The Modelfile subset.** The parser is Ollama's (`parser/parser.go`) state for state, with its quoting:
  - a value runs to the end of the line;
  - `"…"` and `"""…"""` may span lines, with no escapes;
  - `#` opens a comment only at a line's start;
  - instructions are case-insensitive.

  Show writes values back with Ollama's `quote`, and they round-trip.
  - **Taken:** `FROM` (a registry name, a pinned GGUF path, or a safetensors directory, which is converted, then pinned
    with its inputs' sha256s); `SYSTEM`; `PARAMETER` `temperature`, `top_k`, `top_p`, `min_p`, `seed`, `num_ctx`,
    `num_predict`, `stop`; `TEMPLATE` only when it is the base's own; `LICENSE`, `MESSAGE` and `REQUIRES` recorded.
    `FROM` a derived model inherits its layer (Ollama's rule: new values win, `stop` as a list, licences add up).
  - **Refused:** `ADAPTER` (LoRA merging is a later O-phase: merge, then `FROM` the merged directory); a `TEMPLATE`
    other than the pinned one (Ollama's are Go templates, and bankML renders the GGUF's own Jinja byte-identically);
    any other `PARAMETER` (typical-p and mirostat are not reproduced; resource options are not part of a model; the
    penalties `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty` are taken since O2); more
    than one `FROM`; a name a pinned file already has.
- **The layer is applied as Ollama applies it** (`server/routes.go`):
  - **chat** (`/api/chat` and `/v1/chat/completions`): the model's `MESSAGE`s go before the request's, and its `SYSTEM`
    goes first unless the request's first message is a system message;
  - **generate:** the request's `system` if it has one, else the model's, then the `MESSAGE`s, then the prompt; `raw`
    prompts get neither;
  - the parameters are defaults the request's options override key by key (`stop` as a whole list).
- **Endpoints and CLI.**
  - `bankml create NAME -f Modelfile [--registry DIR]`;
  - `POST /api/create`: `{model, modelfile}`, or Ollama's structured `{model, from, system, template, license,
    parameters, messages}`, with status lines streamed as NDJSON. `files`, `adapters` and `quantize` are refused;
  - `DELETE /api/delete`: derived models only; a pin is refused with the reason;
  - `POST /api/copy`: the same digest under a new name;
  - `/api/tags` lists derived models with `details.parent_model`;
  - `/api/show` returns the reconstructed Modelfile, `system`, `parameters`, `license` and `messages`.
- **End to end (0.3.5, `testing/persona_oracle.py`, `oracle_persona_layer`).** promote.py's
  `persona_modelfile_layer` Modelfile — `SYSTEM` (mindX's persona, 1,014 characters), `PARAMETER stop <|im_end|>`,
  `PARAMETER num_ctx 2048` — created two ways: `FROM` the merged safetensors directory (converted to the pin
  `6b64c748…`), and promote.py's file unchanged, `FROM mindx-gen39` re-created in place over the pinned file. Asked the
  user's turns alone, each answers as llama-server b11192 answers the same GGUF given the persona as the system
  message: **27 / 27** answers token-identical for each (9 turns over 6 conversations, greedy and two seeds), and live
  through `/api/chat`, `/api/generate` and `/v1`.
- **Closed in 0.3.5:**
  - **`num_ctx` behaves as Ollama's** (read from Ollama v0.13.3's `server/prompt.go` and `runner/llamarunner`): under
    a `num_ctx` (the request's option, or the layer's on `/api` and `/v1`) a conversation longer than it loses its
    oldest messages first — the last message always stays, and the system messages before the first one kept stay
    (`native::fit_messages`, Ollama's loop step for step, with its quirk that a system message which is itself the
    first one cut is dropped). The tokens counted are the prompt bankML answers. What still does not fit is refused
    with the reason: Ollama's runner would cut tokens out of the middle of the prompt (`num_keep`) and shift its cache
    during the answer, which bankML does not do. A `num_ctx` above `serve --ctx` is refused, as before.
  - **`/api/ps` names the derived model**, with its digest, `parent_model` and the request's context: Ollama names a
    runner after the model whose request loaded it and reloads under the new name when the context differs; bankML
    keeps the weights and records the same name (`Residency::shown_as`).
  - **Names:** a derived model's own name wins over the registry's suffix-less alias (`mindx-gen39` the layer, not
    the pin `mindx-gen39-f16`), and `FROM` resolves that alias — so promote.py's re-create in place works. A
    conversion never overwrites an existing pin.
- **Still open, with what decides it:**
  - Tokens past `num_ctx` during an answer: Ollama shifts its cache (ContextShift) and goes on; bankML answers
    within its served context (`--ctx`), so the two agree up to `num_ctx` tokens and not after. Deciding it needs
    ContextShift with an oracle (O2).
  - Ollama's own prompt is rendered from its Go template, bankML's (and llama-server's) from the GGUF's Jinja; on the
    plain ChatML of `mindx-genN` they agree in shape, but the oracle is llama-server, not Ollama.

## "Replacement for llama.cpp" means bankML's own 1.0.0

The criteria are the six in [TODO.md](TODO.md#the-road-from-030-to-100), unchanged:
- llama.cpp is needed only as the **oracle in the gate**, never at run time;
- every supported architecture × format passes its full oracle set;
- bankML is **at parity or better** on every supported format, on named reference machines;
- the interfaces are stable under semver;
- receipts are signed;
- Savante and mindX run on bankML by default.

An Ollama-shaped API does not make bankML a replacement. Coverage proven by oracles does.

## Out of scope, stated honestly

- **Ollama Cloud models** (gpt-oss cloud and the others). They stay with mindX's `tools/cloud/ollama_cloud_tool.py`.
- **Vision**: `images` is refused.
- **Models without an OSI licence** (Gemma, Llama weights). The importer already refuses them; the Llama
  *architecture* is in scope (O4) for open weights such as SmolLM2.
- **GPU parity beyond Vulkan**: no CUDA, ROCm or Metal backends.
- **Proxying**: bankML never forwards a request it cannot answer natively.

## Field survey (2026-10-01)

The survey of Rust serving and low-bit kernels is in the mindX repository, at `docs/BANKML_RUST_SERVING_RESEARCH.md`.
Its figures are what each project reports; none has been reproduced on our hardware.

Two findings matter most:
- **Crane**, in Rust on Candle, natively loads Prism's `PTQ1_0` and `PQ2_0` GGUF. That makes it the direct competitor in bankML's own format.
- **Ollama v0.30.0** went back to llama.cpp after its own engine regressed. That supports the native-only, narrow scope.

Its ranked adoption list, mapped onto this track:

| rank | adopt | phase |
|---|---|---|
| 1 | Bench Crane's CPU ternary path against bankML on the same Ternary-Bonsai-8B file, pinned (`testing/pinned.sh`) | now (O8 baseline) |
| 2 | A native JSON-mode grammar mask, with llama.cpp's mask as the oracle, token for token (**0.3.3, done**: the server's own grammar, not `json.gbnf`) | O6 |
| 3 | Continuous batching across slots (the boardroom's parallel soldiers) | O8, after O2's multi-slot |
| 4 | A `q8_0` KV cache (llama.cpp's `--cache-type-k/v q8_0`, so the oracle exists), then a Hadamard-rotated 4-bit KV | O3 → O8 |
| 5 | A lookup-table (TL-style) ternary matrix–vector product, and tunable tiling | O8 |
| 6 | Prompt-lookup (n-gram) speculation: no model memory, exact under greedy verification | O8, adopted only on a measured gain |
