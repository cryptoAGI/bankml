# Using bankml and Savante

How to install bankml, start it, and use it on your own computer. Everything runs on this machine: the model, the
gate that verifies it, and Savante's page. New to Savante's page and her voice? Read **[playback.md](playback.md)**.
Every flag, environment variable and service setting is in **[install.md](install.md)**; each source module, its
use and its limits, in **[modules/](modules/README.md)**.

This guide covers the latest release, 0.3.6. Sections marked 0.3.7, 0.3.8 or 0.3.9 describe the next release, which
is not yet tagged; the details are in the *Unreleased* sections of [CHANGELOG.md](../CHANGELOG.md).

- [1. Install](#1-install)
- [2. What you need](#2-what-you-need)
- [3. Start, stop, check](#3-start-stop-check)
- [4. Talk to Savante](#4-talk-to-savante)
- [5. Models](#5-models)
- [6. By hand: build, verify, serve](#6-by-hand-build-verify-serve)
- [6a. Ollama's API](#6a-ollamas-api)
- [6b. The C API](#6b-the-c-api)
- [6c. The bankML console: bankML as itself (0.3.7)](#6c-the-bankml-console-bankml-as-itself-037)
- [7. Let others watch (view mode, on the LAN)](#7-let-others-watch-view-mode-on-the-lan)
- [8. The files Savante keeps: `.history`, `.memory`, `.prompt`](#8-the-files-savante-keeps-history-memory-prompt)
- [8a. Proof of data without the data](#8a-proof-of-data-without-the-data)
- [8b. Custom agents from the Savante template](#8b-custom-agents-from-the-savante-template)
- [8c. THOT bundles: the dataset an iNFT points to](#8c-thot-bundles-the-dataset-an-inft-points-to)
- [8d. PostgreSQL: publish an agent, load one back](#8d-postgresql-publish-an-agent-load-one-back)
- [8e. iNFT: mint an agent, load one from a token](#8e-inft-mint-an-agent-load-one-from-a-token)
- [9. Receipts, and how to check an answer](#9-receipts-and-how-to-check-an-answer)
- [10. Savante's canon and the iNFT ledger](#10-savantes-canon-and-the-inft-ledger)
- [11. Testing and the release gate](#11-testing-and-the-release-gate)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Reference: commands, ports, environment](#13-reference-commands-ports-environment)

---

## 1. Install

One command, from a fresh machine to Savante answering on your own CPU:

```sh
git clone https://github.com/cryptoAGI/bankml && cd bankml
chmod +x install.sh
./install.sh
```

`chmod +x install.sh` makes the installer executable. A `git clone` keeps the bit already, so the line is harmless
there; a copy that arrived another way (a zip download, a USB stick, some file systems) may have lost it, and then
`./install.sh` answers *permission denied*. `bash install.sh` works either way.

or, without cloning first (the installer clones bankml to `~/bankml`, or to `$BANKML_DIR`):

```sh
curl -fsSL https://raw.githubusercontent.com/cryptoAGI/bankml/main/install.sh | bash
```

It runs these steps in order, and stops at the first one that fails, saying why:

| step | what it does |
|---|---|
| `check` | Rust 1.99+ (pinned in `rust-toolchain.toml`), Python 3.10+, git, curl, tar, ss, sha256sum; reports AVX2, RAM and free disk |
| `build` | `cargo build --release` and `cargo test --release` |
| `engine` | downloads llama.cpp b11192 (17 MB) and refuses it unless its sha256 is the published one |
| `python` | uses your Python if it has Gradio and numpy; otherwise makes a venv with Gradio 3 |
| `canon` | clones Savante's canon to `~/cryptoAGI/savante` (beside jaimla and luvai) if it is not there; an existing canon is left as it is, and a machine that still has the older `~/savante` keeps using it |
| `model` | imports Bonsai-8B (1.16 GB) if absent, verifies it, and starts `bankml serve` with llama-server behind it |
| `start` | starts Savante on `http://127.0.0.1:7873` |

Run any steps on their own: `./install.sh build`, `./install.sh model start`. Options: `--skip-tests`, `--no-start`,
`--view` (also start view mode, §7), `--voice` (also install Savante's voice; see [playback.md](playback.md)).
Nothing needs sudo. What the installer chose is kept in `~/.local/share/bankml/install.env`, and logs in
`~/.local/share/bankml/logs/`.

## 2. What you need

| what | where it comes from |
|---|---|
| Linux on x86_64 (AVX2 for the fast kernels; without it bankml is correct, just slower) | — |
| Rust 1.99 (pinned in `rust-toolchain.toml`; rustup fetches it) | [rustup](https://rustup.rs) |
| Python 3.10+ | your system |
| about 2 GB of free RAM and 3 GB of free disk | the 1-bit 8B model maps 1.16 GB |

The installer fetches the rest: llama.cpp b11192 (`llama-b11192-bin-ubuntu-x64.tar.gz`, sha256
`34cf6fa5de9da0db3932c78fe15fed2fbca17451e665dac0a4f6a3c8fc881ec7`), Gradio 3 if needed, the model from
[PYTHAI/Bonsai-8B-gguf-fork](https://huggingface.co/PYTHAI/Bonsai-8B-gguf-fork), and Savante's canon (§10). On
another platform, build llama.cpp b11192 yourself and set `BANKML_LLAMA_SERVER` to its `llama-server`.

The ternary model (2.31 GB) also works and gives better answers, but llama.cpp runs it about 5× slower on a CPU (see
[PERFORMANCE.md](PERFORMANCE.md)).

## 3. Start, stop, check

```sh
./install.sh start           # bankml serve (if not up) is started by `model`; this starts Savante's page
./install.sh start --view    # also the read-only page for your network
./install.sh status          # what is running, and which model bankml verified
./install.sh stop            # stops the page(s), bankml serve and llama-server
```

What runs where:

```
 you (browser) ──► Savante's page (127.0.0.1:7873) ──► bankml serve (127.0.0.1:18093) ──► llama-server (127.0.0.1:18092)
                                                        verifies the model file            runs the verified file
 anyone on the LAN ──► view mode (0.0.0.0:7874, read-only)
```

- **bankml serve** is the gate. It refuses to start unless the model passes the guard, its sha256 equals the pin in
  the model's `FORK.json`, and llama-server serves that same file. Every answer carries a receipt (§9).
- **With the native engine** (`--native`; Savante's `auto` engine setting chooses it for the ternary files), there is
  no llama-server: `bankml serve` computes the answer itself and also listens on 18092 with llama-server's endpoints.
- **Savante's page** is reachable only from this computer.
- **View mode** is a read-only page for others on your network (§7).

## 4. Talk to Savante

Open **http://127.0.0.1:7873**, type a question, press **Send**.

- The first answer takes about **2 minutes** on a laptop CPU, while the model reads Savante's instructions. Later
  answers start sooner.
- Every answer is a **draft · not a finding**. The chat shows the answer alone; its clock, receipt (§9) and prompt
  provenance are on the **Admin** tab under *the last answer*, and in `.history` and the Responses tab.
- To ask for a review, say so: *"review: is Bonsai-8B ready to serve mindX?"*. She then answers as FINDINGS /
  VERDICT / RATIONALE / CONDITIONS / RISKS WATCHED. The 1-bit carrier was graded REJECT for review duty in the sAGI
  carrier test, so treat its verdicts as drafts.
- **Stop** cancels an answer; **New session** starts a fresh conversation.

The Interaction tab is only the conversation: the chat, the question field, the clock and the portrait (press it for
the card; **← back to Interaction** or Esc closes it, and every other tab has a **← Interaction** button). The
**Admin** tab holds the settings: the system prompt (`.prompt`, §8), `.memory`, max tokens, temperature, the carrier,
and the engine's CPU threads and RAM budget (**Apply** restarts the engine with them, and restores your previous
settings if it will not start; the engine choice and n-gram speculation are under *advanced*). The other tabs hold
your history, notes, metrics, integrity checks and the verifier. [playback.md](playback.md) walks through the
page, her card and her voice.

## 5. Models

The **Models** tab and `sAGI/models.py` import a model, verify it, and switch to it:

```sh
python3 sAGI/models.py list                 # what is here, pinned or not, and which one is running
python3 sAGI/models.py catalog              # the curated catalogue, and whether each fits this machine
python3 sAGI/models.py import qwen3-1.7b    # a catalogue id, a Hugging Face URL, or ollama:NAME:TAG
python3 sAGI/models.py use Qwen3-1.7B-Q8_0.gguf   # switch (rolls back if the new model fails)
```

An import is kept only if its sha256 equals the one its publisher lists, `bankml guard` passes, and a `FORK.json`
pin is written; `bankml serve` checks that pin before every start. Only open-source models are accepted (Gemma and
Llama are refused), and an import that would not fit this machine's RAM or disk is refused with the numbers.

**Converted models (0.3.4).** SmolLM2-135M-Instruct and mindX's `mindx-gen39` have no published F16 GGUF. They are
made from their pinned safetensors by llama.cpp b11192's own converter, and `sAGI/models.py` pins the result against
the recorded conversion (`CONVERTED` in the file says how each was made):

```sh
python3 llama.cpp/convert_hf_to_gguf.py SmolLM2-135M-Instruct/ --outtype f16 --outfile .models/SmolLM2-135M-Instruct-F16.gguf
#   (b11192, commit 171e8846; the safetensors at HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f7; torch is needed here only)
python3 -c "import sys; sys.path.insert(0, 'sAGI'); import models; print(models.adopt('SmolLM2-135M-Instruct-F16.gguf'))"
#   → hashes the file against the recorded e9aba089…, re-reads the source's LFS sha256 at the revision, writes FORK.json
```

What bankML's own forward pass plays (`bankml serve --native`, `bankml generate`, the C API): Qwen3 in Q1_0 and
Q2_0_g64 (Bonsai-8B, Ternary-Bonsai-8B, Bonsai-1.7B) and the Llama architecture in F16 (SmolLM2-135M-Instruct,
`mindx-gen39`). Anything else is refused with the reason; `/api/tags` lists it with that reason.

## 6. By hand: build, verify, serve

What the installer does, step by step:

```sh
cargo build --release && cargo test --release
target/release/bankml guard  .models/Bonsai-8B-Q1_0.gguf                   # play | refuse (reason) | need_more
target/release/bankml verify .models/Bonsai-8B-Q1_0.gguf --fork FORK.json  # guard, then the sha256 pin
target/release/bankml serve  .models/Bonsai-8B-Q1_0.gguf --fork FORK.json \
    --spawn /path/to/llama-b11192/llama-server --threads 3 --ctx 2048
python3 sAGI/savante.py --mode interact
```

`verify` exits 0 on play, 2 on refuse, 1 on an I/O error, 3 on a truncated file. `serve` can also sit in front of a
llama-server you started (`--upstream 127.0.0.1:18092`); it then checks through `/props` that the server runs the
verified file. It listens on `127.0.0.1:18093` and speaks the OpenAI API:

```sh
curl -s 127.0.0.1:18093/bankml     # what was verified, when, and what is upstream
curl -s 127.0.0.1:18093/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello. /no_think"}],"max_tokens":32}'
```

### JSON mode and grammars (0.3.3)

`/v1/chat/completions` (and the C API's `bankml_chat`) takes what llama-server takes, resolved the way it resolves it:

```sh
B=127.0.0.1:18093; J='Content-Type: application/json'
curl -s $B/v1/chat/completions -H "$J" -d '{"messages": [{"role": "user", "content": "Describe a cat."}],
    "response_format": {"type": "json_object"}, "temperature": 0, "max_tokens": 64}'
curl -s $B/v1/chat/completions -H "$J" -d '{"messages": [{"role": "user", "content": "Is the sun a star?"}],
    "grammar": "root ::= (\"yes\" | \"no\") \".\""}'
echo 'Describe a cat.' | target/release/bankml generate .models/Bonsai-8B-Q1_0.gguf --json --max 64
```

- **`response_format: {"type": "json_object"}`**, and the schemas `{}` and `{"type": "object"}` (in `response_format`
  or the top-level `json_schema`), give **llama-server b11192's grammar for this template**, byte for byte. Its root
  begins with the generation prompt, which the sampler takes in before the first token, as the server does. Under
  that grammar the model may open with a fenced ```` ```json ```` block, and `content` is the JSON value alone. That
  is the object once it closes, or what there is of it when `max_tokens` cuts the answer. A stream sends the
  content's growth.
- **`grammar`**: any GBNF that llama.cpp parses, with root `root`. It is taken as a user grammar, with no prefill, and
  `content` is the text as generated. A grammar llama.cpp would not parse is refused with its parser's reason.
- **Every token is drawn as llama-server draws it under a grammar.** The usual sampler chain runs first. If its token
  breaks the grammar, the logits are masked and the chain runs again, which takes a second draw from the seeded
  generator. Greedy and seeded answers are token-identical to the server (the oracle is in [oracles.md](oracles.md)).
- **Any other JSON schema** (0.3.5): in `response_format` (`json_schema`, or `json_object` with a `schema`), the
  top-level `json_schema`, or Ollama's `format: {…}`, it is converted into the grammar llama-server b11192 builds for
  it on the model's own template (`bankML/schema.rs`; byte-identical on 173 of 173 schemas on each template against
  llama.cpp's own code), and answered as JSON mode is: prefilled, fenced or bare, `content` the value alone (a
  top-level string, number or literal included). The answers are token-identical to llama-server's on every native
  model, greedy and seeded. A schema llama.cpp refuses is refused with its message.
- **An answer cut by `max_tokens`** gets llama-server's content: the value as far as it got (an unfinished escape
  dropped), or, when nothing of the value came yet (`"```json\n"`), the raw text, as the server answers a whole
  request. A stream sends only the value's growth.
- **Refused, with the reason:** a schema llama.cpp b11192 refuses; a `pattern` llama.cpp would turn into a grammar
  that is not UTF-8; a non-object or `null` top-level `json_schema` (llama-server fails those requests);
  `response_format` together with `grammar`; `json_schema` together with `grammar`; a `response_format` type other
  than `text`, `json_object` or `json_schema`.

### The context limit, slots and logprobs (0.3.8)

In the next release (not yet tagged); each is checked against llama-server b11192's own answers
([CHANGELOG.md](../CHANGELOG.md), [modules/serve.md](modules/serve.md), [modules/prompt_cache.md](modules/prompt_cache.md)).

- **At the context limit** the native engine answers as llama-server does with context shift off: a generation that
  reaches `--ctx` stops with `finish_reason: "length"`; a prompt that does not fit is refused at once with HTTP 400
  and llama-server's `exceed_context_size_error` body, which names both counts — so a client can trim and retry.
- **Slots.** Start `bankml serve --native` with `--slot-dir DIR` (created at start). Save the conversation's KV cache
  by name, restore it later — in another process too — and skip recomputing the prompt. Without `--slot-dir` the
  slot actions answer 501, as llama-server's do:

  ```sh
  curl -s 127.0.0.1:18092/slots/0?action=save    -H 'Content-Type: application/json' -d '{"filename":"warm.bin"}'
  curl -s 127.0.0.1:18092/slots/0?action=restore -H 'Content-Type: application/json' -d '{"filename":"warm.bin"}'
  curl -s 127.0.0.1:18092/slots/0?action=erase   -H 'Content-Type: application/json' -d '{}'
  ```

  The file is checked when it is read (the model's sha256 and a sha256 trailer): a file from another model or a
  damaged one is refused and the slot emptied (never half-filled), so the next answer simply recomputes. Filenames
  are plain names inside the directory; a path is refused.
- **Conversations that take turns.** The native engine has one slot and serves requests one at a time, as
  llama-server `-np 1`. Like llama-server, it keeps the states of other conversations in RAM (`BANKML_CACHE_RAM`, MiB;
  0 off, -1 no limit; unset, 8192 MiB but at most a quarter of the memory available at load), so a conversation that comes back after
  another does not recompute its whole history.
- **Your bankML from a web page (0.3.9).** `./install.sh start --space` lets the bankML Space's page
  ([PYTHAI/bankml](https://huggingface.co/spaces/PYTHAI/bankml), *Your own bankML*) talk to your own `bankml serve`
  from your browser: your CPU, your verified model, a receipt on every answer, nothing sent anywhere else. It starts
  serve with `--allow-origin https://pythai-bankml.static.hf.space`, so only that one page gets CORS headers;
  `./install.sh start --no-space` takes it back.
- **A smaller conversation memory (0.3.9, in progress).** `BANKML_CACHE_TYPE=q8_0` keeps the KV cache in q8_0 instead of f16,
  about half the memory, so a long context fits on a small machine. It is llama.cpp's `--cache-type-k q8_0
  --cache-type-v q8_0` exactly, with the Hadamard rotation llama.cpp applies around a quantized cache, and gives the
  same tokens as llama-server so configured (`testing/kv_oracle.py`). A slot saved with one cache type is refused by
  an engine with the other.
- **Logprobs.** `"logprobs": true, "top_logprobs": 5` on `/v1/chat/completions` returns each token's log-probability
  and the five most likely alternatives (`top_logprobs` defaults to 20), the same floats llama-server reports.
  Streamed answers carry each token's entry in the chunk its text makes. `top_logprobs` without `logprobs` is refused
  with llama-server's message. llama-server's own `/completion` endpoint (and its `n_probs`) is not served; use `/v1`.

## 6a. Ollama's API

Since 0.3.1 `bankml serve --native` also speaks Ollama's API, over the same engine, gate and receipts. **Native
only:** bankML answers what its own verified forward pass can do and refuses the rest with HTTP 400
`{"error": "…"}` that says why; it never hands a request to llama-server or to Ollama. What mindX asks of Ollama,
and what is still missing, is in [OLLAMA.md](OLLAMA.md).

```sh
target/release/bankml serve .models/Bonsai-8B-Q1_0.gguf --fork ~/.local/share/bankml/forks/Bonsai-8B-Q1_0.gguf.FORK.json \
    --native --registry --ctx 2048            # --registry [DIR]: every model pinned there, by name; --keep-alive 5m
B=127.0.0.1:18093; J='Content-Type: application/json'
curl -s $B/api/version
curl -s $B/api/tags                          # every pin: name, size, digest = its sha256, details, "bankml": {"native", "reason"}
curl -s $B/api/ps                            # the resident model and its expires_at
curl -s $B/api/show -H "$J" -d '{"model": "bonsai-8b-q1_0"}'
curl -s $B/api/chat -H "$J" -d '{"model": "bonsai-8b-q1_0", "messages": [{"role": "user", "content": "Name a prime."}],
    "stream": false, "options": {"temperature": 0.3, "seed": 7, "num_predict": 32}}'
curl -s $B/api/generate -H "$J" -d '{"model": "bonsai-8b-q1_0", "system": "Answer in one word.", "prompt": "Capital of France?", "stream": false}'
curl -s $B/api/generate -H "$J" -d '{"model": "bonsai-8b-q1_0", "keep_alive": 0}'          # unload now (done_reason "unload")
curl -s $B/api/generate -H "$J" -d '{"model": "ternary-bonsai-8b-q2_0_g64", "keep_alive": "30m"}'  # verify and load (done_reason "load")
```

- **Names.** A model's name is its file's stem, lower-cased: `bonsai-8b-q1_0`, `ternary-bonsai-8b-q2_0_g64`,
  `smollm2-135m-instruct-f16`, `mindx-gen39-f16`.
  - `:latest` and the file name are accepted as aliases.
  - Since 0.3.4 so is the name without its weight-type suffix, when exactly one pin has that base: `mindx-gen39`
    (Ollama's tag for mindX's generation 39), `smollm2-135m-instruct`, `bonsai-1.7b`.
  - A request without a model uses the one `serve` started with.
  - An unknown name answers 404.
- **Where models come from.** Without `--registry`, only the startup model is served. With it, every GGUF pinned in
  `DIR` (default `$BANKML_FORKS`, else `~/.local/share/bankml/forks`) can be asked for. The file is looked for
  beside the startup model and in `DIR`.
- **`/api/tags` lists every pin, honestly.** A pin bankML cannot play is listed with `"native": false` and the reason.
  Asking for one answers 400 with that reason. Two examples:
  - `qwen3-0.6b-q8_0`: Q8_0 weights, O3;
  - `qwen3-0.6b` (Ollama's blob): Q4_K weights, O3.
- **One resident model.** A request for another model drops the resident one, then verifies the new one before it
  loads it: the guard, then the sha256 pin, as at the start.
- **`keep_alive`** is read as Ollama reads it:
  - `"5m"`, `"1h"`, `"1h30m"`, or a number of seconds;
  - `0` unloads after the answer;
  - a negative value keeps the model for good.

  `/api/*` requests that name none get `--keep-alive` (default `5m`). An idle model is dropped when its time runs out,
  and its memory map with it. The OpenAI and llama-server endpoints name no keep-alive. They keep the model resident,
  as 0.3.0 did, and load the startup model again if nothing is resident.
- **An empty prompt only loads or unloads.** That is a `/api/generate` without a `prompt`, or a `/api/chat` without
  `messages`. The answer's `done_reason` is `load` or `unload`.
- **Streaming is on by default**, as in Ollama: NDJSON (`application/x-ndjson`), one object per line, then a final
  `done: true` object. That object carries:
  - `done_reason` (`stop` or `length`);
  - `total_duration`, `load_duration`, `prompt_eval_duration` and `eval_duration`, in nanoseconds;
  - `prompt_eval_count`: the whole prompt;
  - `eval_count`: generated tokens, the end-of-turn token included, as llama-server counts it;
  - `bankml_cache_n`: prompt tokens reused from the cache;
  - `bankml_receipt`, as on `/v1/chat/completions`.
- **Options.**
  - **Honoured:** `temperature`, `top_k`, `top_p`, `min_p`, `seed`, `num_predict` and `stop` (also on `/v1`).
    `top_k` is reproduced from 1 to 128; 0 or a larger value is refused, because llama.cpp sorts larger sets another
    way.
  - **`num_ctx`:** accepted up to the served `--ctx`, refused above it. Under it (0.3.5) a conversation that does not
    fit loses its oldest messages first, as Ollama's do (the last message and the system messages stay); a prompt that
    still does not fit is refused with its size.
  - **Penalties (O2):** `repeat_penalty`, `repeat_last_n`, `presence_penalty` and `frequency_penalty` are honoured as
    llama-server b11192 applies them, token for token on its oracle: the window holds the last `repeat_last_n` tokens
    (default 64) **including the prompt's**, a repeated token's logit is divided by the repeat penalty (multiplied when
    not positive), then `count × frequency + presence` is taken off. `repeat_last_n` below 0 and a repeat penalty of 0
    or less are refused with llama-server's message. Also on `/v1` and as a Modelfile `PARAMETER`.
  - **The rest of the chain (0.3.7, next release; [modules/sampler.md](modules/sampler.md)):** `typical_p` on `/api/*` and `/v1`; on `/v1` (not Ollama options) also
    `top_n_sigma`, `xtc_probability`, `xtc_threshold`, `dynatemp_range`, `dynatemp_exponent`, `dry_multiplier`,
    `dry_base`, `dry_allowed_length`, `dry_penalty_last_n` and `dry_sequence_breakers` — each as llama-server b11192
    applies it, token for token on its oracle, in its default order. Clamped as it clamps (`top_p`, `min_p`, the XTC
    fields to [0, 1]); refused as it refuses (a negative DRY window or allowed length, empty breakers). `mirostat` and a
    custom `samplers` order are refused: they are not reproduced.
  - **Ignored**, because they do not change the answer bankML gives: `num_thread`, `num_batch`, `num_gpu`, `use_mmap`
    and the other resource options.
  - **Refused:** any other option.
- **`format: "json"` is JSON mode (0.3.3).** It is answered by the same grammar llama-server b11192 uses for
  `response_format: {"type": "json_object"}` (below), so `/api/chat` with `format: "json"` gives llama-server's tokens.
  `format: {}` and `format: {"type": "object"}` are the same request. Since 0.3.5 any other schema object is
  answered with the grammar llama-server builds for it (below). `format` with `raw: true` is refused (the grammar
  begins with the template's generation prompt, which a raw prompt does not have).
- **Refused, with the reason:**
  - a `format` schema llama.cpp b11192 refuses (with its message);
  - `tools` and `tool_calls` (O6);
  - `images`;
  - `suffix`, `template`, `context`;
  - `think: true` (the template is rendered with thinking off; `think: false` is accepted);
  - `/api/embed` (O7), `/api/pull`, `/api/push`, and `DELETE /api/delete` of a pinned file (derived models: below).
- **`raw: true`** tokenizes `prompt` as given, with special tokens and no template. `system` is ignored then, as in
  Ollama.
- **The loopback rules still apply.** POST bodies need `Content-Type: application/json` (`curl -d` alone sends a
  form type and gets 415), and `Host` must be a loopback name.

### Derived models and conversion: `bankml create`, `bankml convert` (O5, 0.3.5)

`ollama create`, natively. A derived model is a layer (system prompt, parameters, stop strings, example messages,
licence) over a **pinned** base, written as `<registry>/<name>.MODEL.json`. Weights are never copied. Loading it
verifies the base exactly as a pinned model is verified. The details are in
[OLLAMA.md](OLLAMA.md#what-o5s-first-cut-built-035). mindX's persona layer, made this way, answers token-identically
to llama-server given the same persona (`testing/persona_oracle.py`).

```sh
# mindX's flow, without Ollama: the merged safetensors directory → a pinned GGUF → the persona layer
cat > Modelfile <<'MF'
FROM /home/mindx/mindX/data/godel/ascend/gen39/out/runs/<recipe>/ollama_push/merged
MF
bankml create mindx-gen39 -f Modelfile            # converts (byte-identical to llama.cpp) → mindx-gen39-F16.gguf + its FORK.json
cat > Modelfile.persona <<'MF'
FROM mindx-gen39
SYSTEM """You are mindX generation 39, …"""
PARAMETER stop <|im_end|>
PARAMETER num_ctx 2048
MF
bankml create mindx-gen39 -f Modelfile.persona    # re-created in place: the layer over the same pinned base
# the same over HTTP (bankml serve … --native --registry)
curl -s $B/api/create -H "$J" -d '{"model": "mindx-persona", "from": "mindx-gen39", "system": "You are mindX.", "parameters": {"temperature": 0.7}}'
curl -s $B/api/show   -H "$J" -d '{"model": "mindx-persona"}'     # the Modelfile back, details.parent_model
curl -s $B/api/copy   -H "$J" -d '{"source": "mindx-persona", "destination": "mindx-persona-b"}'
curl -s -X DELETE $B/api/delete -d '{"model": "mindx-persona-b"}'
# conversion alone, with the FORK.json that pins it
bankml convert DIR -o mindx-gen39-F16.gguf --fork mindx-gen39-F16.gguf.FORK.json --source "PYTHAI/mindXascension@4bd31b9d weights/gen39"
```

- **Modelfile.** Ollama's grammar: `"""…"""` blocks, `#` comments, case-insensitive instructions.
  - **Taken:** `FROM`, `SYSTEM`, `PARAMETER` (`temperature`, `top_k`, `top_p`, `min_p`, `seed`, `num_ctx`,
    `num_predict`, `stop`), `MESSAGE`, `LICENSE`, `REQUIRES`.
  - **`TEMPLATE`:** only when it is the base's own.
  - **Refused, with the reason:** `ADAPTER` (merge the LoRA first), other parameters, `quantize`, uploaded `files`.
  - A relative `FROM ./x` is relative to the Modelfile.
- **Where.** `--registry DIR` (default `$BANKML_FORKS`, else `~/.local/share/bankml/forks`).
  - `FROM` a name looks there, and in `--models DIR`.
  - `FROM` a GGUF needs its FORK.json there.
  - `serve --native --registry` lists the derived models and answers `/api/create`. Without `--registry` it refuses,
    because it has nowhere to write.
- **A derived model's name** wins over the suffix-less alias of a pin (`mindx-gen39` is the layer, `mindx-gen39-f16`
  the pin), and `/api/ps` names it, with its digest and parent, as Ollama names the model that loaded its runner.
  A layer's `PARAMETER num_ctx` fits the conversation on `/api` and `/v1` as the request's option does.
- **`bankml convert`.** The Llama architecture (SmolLM2, `mindx-genN`) to GGUF F16, byte-identical to llama.cpp b11192's
  `convert_hf_to_gguf.py --outtype f16`: `e9aba089…` for SmolLM2-135M-Instruct, `6b64c748…` for gen39.
  - llama.cpp names the model from the **directory's name**. A directory called `merged` gives `general.name`
    `Merged`, as llama.cpp's would. `--model-name` sets it, as llama.cpp's flag does.
  - A `README.md` is refused, because llama.cpp would copy the card's metadata in.
  - A tokenizer that llama.cpp has not been seen to name is refused (`testing/convert_oracle.py --chkhsh DIR`
    measures one).

## 6b. The C API

Since 0.3.2 bankML is also a C library, for a program that embeds it rather than calling a server. The full
reference, with the ownership and threading rules and the log formatter, is **[CAPI.md](CAPI.md)**.

```sh
cargo build --release -p bankml-capi       # target/release/libbankml.so, libbankml.a
cc app.c -I capi/include -L target/release -lbankml -Wl,-rpath,$PWD/target/release -o app
```

```c
#include <bankml.h>
static int piece(const char *p, size_t n, void *u) { fwrite(p, 1, n, stdout); return 1; }
int main(void) {
  char *err = NULL, *result = NULL;
  bankml_t *h = bankml_open(".models/Bonsai-8B-Q1_0.gguf", "Bonsai-8B-Q1_0.gguf.FORK.json", 2048, &err);
  if (!h) { fprintf(stderr, "%s\n", err); bankml_free(err); return 2; }   /* refused: the reason, as serve gives it */
  bankml_chat(h, "{\"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}], \"max_tokens\": 32}", piece, NULL, &result);
  puts(result);                         /* the /v1/chat/completions object, with usage and bankml_receipt */
  bankml_free(result);
  bankml_close(h);
}
```

`bankml_open` runs the same verification as `bankml serve`, and `bankml_chat` gives the answer and receipt that
`serve --native` gives (the gate's C API oracle compares them, turn by turn). `bankml_set_log` and the printf-style
`bankml_log` send the library's messages to your own callback.

## 6c. The bankML console: bankML as itself (0.3.7)

In the next release (not yet tagged). The design is in [modules/console.md](modules/console.md) and
[modules/metrics.md](modules/metrics.md).

```sh
python3 sAGI/console.py          # http://127.0.0.1:7875 — talks to bankml serve at BANKML_SERVE_LISTEN (127.0.0.1:18093)
```

`--port` moves it; any `--host` other than loopback is refused, because the console can restart the engine.
`./install.sh start` starts it beside Savante (:7875), `./install.sh stop` stops it, and Savante's header has a
**bankML ↗** link to it; its own bar has the **Savante | bankML** switch back.

bankML's own interface, deliberately unlike Savante's — a quiet page, light or dark with the system — with four tabs:

- **Ask** (the landing): one question, the streamed answer, and under it the receipt — `✓ receipt` when the sha256 of
  the text you received is the receipt's (checked in the browser), the model, the tokens, the time to first token, the
  generation speed; while the prompt is read, the answer shows the seconds elapsed. bankML answers as
  itself (`sAGI/personas/bankml.persona`): each question carries a SELF block measured at that moment from
  `/bankml/usage` and `/bankml/metrics`, so "how many tokens have you used, how fast, how much power" is answered from
  measurement, and "not measured" where it was not.
- **Admin**: three sliders — CPU threads, the RAM budget (weights plus KV cache, which sets the context) and the GPU
  limit (0 = off) — applied as one verified restart of the engine, rolled back if it fails; and D3 charts of what bankML
  measures: tokens per second, TTFT, CPU, memory, GPU busy against its limit, power.
- **Receipts**: every answer's receipt, newest first — each re-checked in the browser and opening to the answer and
  the receipt's JSON; refusals included. Below, the commitments an iNFT publication of the session carries (ERC-721
  shape): the engine, the model and its sha256, the persona and its doctrine root, the token totals, and an RFC 6962
  Merkle root over the exchanges with their CID. The exchanges stay on this machine; the root lets a holder check any
  one of them. Download it as JSON.
- **Logs**: the engine's own log.

Power appears only after `./install.sh power` (opt-in, the one step that uses sudo; see
[install.md §2](install.md#the-steps)); until then it reads "not measured".

## 7. Let others watch (view mode, on the LAN)

```sh
./install.sh start --view          # or: python3 sAGI/view.py --host 0.0.0.0 --port 7874
```

Others open `http://<this computer's LAN address>:7874` (`ip -4 addr` shows it). The page is read-only: the live
testing log, this machine's load, the verified model, the release records, CI, Savante's office and ledger, and her
recordings (see [playback.md](playback.md)). It never shows the content of `.history` or `.memory`, only their
commitments (§8a). It is the Python standard library, not Gradio, because the Gradio installed for the chat has
path-traversal bugs and must stay on this computer.

## 8. The files Savante keeps: `.history`, `.memory`, `.prompt`

Nothing is ever written into the canon (`~/cryptoAGI/savante`). What the UI writes lives in `BANKML_UI_STATE` (default
`~/.local/share/bankml/savante/`):

- **`savante.history`**: one JSON object per exchange (JSONL):

  ```json
  {"ts": 1790633549.123, "sent_at": "2026-09-28T15:12:29.123-0700", "first_token_s": 113.02,
   "response_s": 125.11, "answered_at": "2026-09-28T15:14:34.233-0700", "session": "e2e-test",
   "user": "In one sentence: …", "assistant": "Savante knows …", "assistant_raw": "Savante knows …", "shown": "… (with the footer)",
   "prompt": "persona · system_prompt (canon, ledgered)", "prompt_provenance": "persona.system_prompt · persona sha256 d96556b11989… (ledgered)",
   "receipt": {"bankml": "0.0.7", "model_sha256": "284a335a…", "prompt_tokens": 316, "completion_tokens": 28,
               "ttft_ms": 113000, "wall_ms": 125100, "response_sha256": "80010408…"}}
  ```

  `sent_at` is the press of Send; `first_token_s` and `response_s` are measured from it. The receipt's
  `ttft_ms`/`wall_ms` are measured by bankml serve from when it forwarded the request. `assistant_raw` (0.1.7+) is
  the text exactly as the model wrote it, which is what the receipt hashes; `assistant` is the same text with any
  `<think>` block and surrounding whitespace removed. Each record is appended as one locked, fsynced write; a line
  damaged by a crash is skipped (and counted), never allowed to hide the rest. The history is plain text: back it
  up, grep it, or delete it.
- **`savante.memory`**: one JSON object per note: `{"ts", "at", "text", "sha256", "source": {"kind": "typed" |
  "response", "session", "sent_at", "response_sha256"}}`.
- **`Savante.prompt`**: the Space template's prompt, cached the first time it is chosen.

**The history the model sees.**
- **Without a context limit:** the last 12–17 exchanges, each cut to 4,000 characters. The window starts at 12
  exchanges and moves in steps of six, so between moves each prompt is the previous prompt plus one exchange, and
  the engine's prompt cache reuses all of it.
- **Within the engine's context.** The window must also fit the context the engine actually runs with (`n_ctx` from
  `/props`), counted by its own tokenizer.
  - The start stays where it was last turn while that still fits, so the cache is reused.
  - When it no longer fits, the start jumps so that the newest half of what fits is kept, leaving room for the next
    turns. When only two or three exchanges fit, all of them are kept.
  - The answer's trail (Admin → *the last answer*) then says so: "history: 4 of 13 exchanges fit the engine's
    2048-token context — raise the RAM budget on the Admin tab for more".
  - Exchanges older than the 12–17 window are dropped without a note, as they always were.
- **If the system prompt and the question alone do not fit,** the answer is refused with the numbers.
- On a 2048-token context the persona prompt plus `.memory` leaves room for only a few exchanges; 4096 tokens (about
  0.3 GB more RAM for the 8B model) gives the window room to stay put for several turns. **After an engine
restart**, the first question restores the saved KV of the system prompt (a 51 MB file per model, context and system
prompt, in `~/.local/share/bankml/savante/slots/`), so it skips the system prompt's prefill: on this laptop 132 s
became 15 s, with the same answer.
(A window that slid by one exchange per turn, as before 0.1.8, changed the text right after the system prompt and
forced the whole history to be re-read every turn; over 60 turns the stable window moves 8 times instead of 48.) The
chat's footer (clock, receipt) is never sent to the model.

## 8a. Proof of data without the data

`.history` and `.memory` stay on this computer. What can leave it is a **commitment**:

| part | what |
|---|---|
| leaf | sha256(0x00 ‖ one JSONL line, exactly as written, without its newline) |
| Merkle root | RFC 6962 / RFC 9162: node = sha256(0x01 ‖ left ‖ right), the tree split at the largest power of two below n (an odd node is promoted, never paired with itself). Scheme `rfc6962-sha256`; 0.1.6 and earlier used an unprefixed tree, so their roots differ |
| file sha256 / CIDv1 | of the whole file; CIDv1 raw (0x55), sha2-256, base32 lower: the construction Savante's iNFT ledger uses |

**🔏 proof** (Responses tab) produces, for one exchange, `{"commitment": …, "proof": {"scheme", "record", "records",
"leaf", "path": [hash, …], "merkle_root"}, "verifies": true}`. Give someone the exchange's line and the proof: they
verify it as RFC 9162 §2.1.3.2 describes (the index and the tree size decide at each step which side the sibling is
on) and compare the result with the root **and record count** you published; the pair is the commitment. That proves the exchange is in your history, unchanged,
and reveals nothing else. The same works for any data kept in a browser's local storage: keep the data, publish the
hash or CID. Metrics and the view page show the commitments; the view page never serves a line.

To check a proof by hand, in Python:

```python
import hashlib, json
line = b'...the exact JSONL line...'
p = json.load(open("proof.json"))["proof"]
h = hashlib.sha256(line).hexdigest()
for s in p["path"]:
    a, b = (h, s["hash"]) if s["side"] == "right" else (s["hash"], h)
    h = hashlib.sha256(bytes.fromhex(a) + bytes.fromhex(b)).hexdigest()
print(h == p["merkle_root"])
```

## 8b. Custom agents from the Savante template

**Agents** tab. Savante's canon is the template and is never written. **Derive a new agent** creates, in
`~/.local/share/bankml/agents/<name>/` (`BANKML_AGENTS`):

| file | what |
|---|---|
| `<name>.persona` | the persona (mindX `.persona v1` shape): a new name and identity, Savante's BDI/skills/safety as a starting point; the `token` bindings start empty (Savante's are never copied) |
| `<name>.prompt` | the system prompt, as text: the `.prompt` the chat uses |
| `<name>.agentcard.json` | EIP-721 metadata ∪ ERC-8004 registration-v1; status `not_yet_minted`; `derived_from` names Savante's persona sha256 and doctrine root |
| `<name>.commitments.json` | the ledger: sha256 + CIDv1 of each file, and the **doctrine root** (keccak256 over the same 15 JSON pointers as Savante's ledger) |
| `<name>.history`, `<name>.memory` | the agent's own conversations and notes |

**Use this agent** switches the chat, `.history`, `.memory`, Responses and Metrics to it. **Edit** (custom agents
only) opens the `.persona` as JSON and the `.prompt` as text. **Save and re-ledger** refuses a persona that breaks
the binder's preflight (non-ASCII keys, floats, integers beyond ±2⁵³), lacks a doctrine clause, or **changes one**:
the 15 doctrine clauses (name, source, format, system prompt, mantra, oath, beliefs, skills, safety, embodiment, tool
allowlist …) are fixed when the agent is derived, and their root is recorded then and carried forward. To change them,
derive a new agent. Everything else (the `.prompt` file, voice examples, description, the aivatar) stays editable.
Otherwise it regenerates the card and ledger. An agent whose files do not match its ledger refuses to speak.

The keccak256 is written in pure Python. It reproduces Savante's published doctrine root
(`0x92fe83eb…ae137d0`) from her persona, and it equals pycryptodome's keccak for every input length from 0 to 400
bytes (tested).

## 8c. THOT bundles: the dataset an iNFT points to

**Agents → THOT bundle** builds `<name>.thot.json` to `sagi.thot_manifest/1` (`~/sagi/engine/THOT_MANIFEST.md`):

- **Facets.** Core facets `persona` (the ternary head, leaf 0) and `prompt`, plus declared custom facets
  `x-bankml.agentcard`, `x-bankml.history` and `x-bankml.memory`. The history and memory are committed **by digest
  only**: the manifest holds no conversation text and can be published.
- **Bundle root.** `bundle_root` = keccak256 over `label ‖ 0x1f ‖ sha256_hex ‖ 0x1e`, core facets in registry order,
  then custom facets by name.
- **Merkle root.** A 64-leaf keccak256 tree, padded with `keccak256(b"")`, with unsorted pairs and no prefixes.
- **Identity.** `thot:<sha256>`, a CIDv1, `thot-<cid>`, and `contentRoot` (keccak256), all over the canonical
  manifest without its identity block.
- **Lineage.** Genesis is generation 1. Every facet change (a new chat, a new note, an edited prompt) makes the next
  build generation n+1, with `parent` = the previous manifest's CID. A re-bind with no change keeps the generation.
  `relations` records `derived_from` Savante's bundle.
- **Rung.** Each agent directory is its own git repository, so the rung's locator
  (`localhost/<user>/<name>@<commit>`) names real bytes. History and memory are git-ignored and listed as
  `locator_lacks`.

The builder is checked against the spec's own test vectors: Savante @1fcca89, Jaimla @8b57ccf and LuvAI @0c1eef7 give
the same bundle roots, Merkle roots and identity CIDs. Savante's current generation-9 manifest verifies with no findings.

## 8d. PostgreSQL: publish an agent, load one back

**Agents → PostgreSQL**. The connection is `BANKML_PG_DSN` (default `dbname=bankml`, the local socket, your role).
One-time setup, which needs sudo:

```sh
sudo -u postgres psql -c "CREATE ROLE $USER LOGIN CREATEDB" -c "CREATE DATABASE bankml OWNER $USER"
sudo -u postgres psql -d bankml -c "CREATE EXTENSION IF NOT EXISTS vector"
```

- **Publish the agent in use.** This rebuilds and verifies its THOT bundle, then upserts `bankml_agents`: persona
  (the exact bytes, plus a jsonb copy for queries), prompt, card, ledger, manifest, THOT CID, contentRoot and
  generation.
  - **Off by default:** `.history` and `.memory` lines go in (`bankml_exchanges`, `bankml_memory`, exact line text,
    sequence and sha256) only when *include .history and .memory lines* is ticked. Otherwise the database holds only
    their commitments, inside the manifest.
- **Load.** Rebuilds a published agent as a local one. It is **refused** unless the stored manifest's identity
  recomputes and every restored file (persona, prompt, and history and memory when included) re-hashes to the
  manifest's digest. A tampered row cannot be loaded.
- **Vectors.** `bankml_exchanges.embedding` is `vector(1024)` (bge-m3's width, as mindX uses), indexed with DiskANN
  when pgvectorscale is installed and HNSW (pgvector) otherwise. It is filled with bge-m3 vectors when private lines are
  published and bge-m3 can run (see [embedding.md](embedding.md)); otherwise it stays empty.
- **Safety.** Values travel to psql as COPY data into a temporary table; no value is ever part of SQL text.
  `testing/test_connectors.py` runs every step against a throwaway cluster (initdb in a temp dir, pgvector, its own
  socket), including a tampered prompt, a tampered history line and an SQL-injection string, all handled.

## 8e. iNFT: mint an agent, load one from a token

**Agents → iNFT**, for the agent in use. Savante herself is not minted from here: her verdict on minting is DEFER.

The contract is the house ERC-7857 `iNFT_7857` (DeltaVerse iNFT4), and an open (unsealed) agent is minted with:

```
mintOpenAgent(address to, bytes32 contentRoot, string storageURI, bytes32 metadataRoot,
              uint256 dimensions, uint8 parallelUnits, string tokenURI)        — MINTER_ROLE only
```

| argument | from the agent |
|---|---|
| `contentRoot` | its THOT manifest's `identity.contentRoot` (keccak256 of the canonical manifest). The contract accepts a content root once, ever |
| `metadataRoot` | keccak256 of the agent card's canonical bytes |
| `storageURI`, `tokenURI` | `local://thot/<cid>` and `local://card/<cid>` until the bundle and card are stored somewhere (the rung stays `referenced`) |
| `dimensions`, `parallelUnits` | one of 8 … 1048576 (768 by default), and 1 |

- **Plan and simulate** builds the calldata and runs it as an `eth_call` from the minter. You get the token id it
  would receive, or the decoded revert (`AccessControlUnauthorizedAccount`, `ContentRootAlreadyMinted`,
  `InvalidDimension`, …).
- **Unsigned transaction** gives the transaction (chain id, to, data, gas estimate) and the same call as a
  `cast send … --account <your keystore>` line. **You sign it**, in your own wallet: bankml never signs on a
  public chain.
- **Mint on the local devnet** sends only when the chain id is 31337 (anvil). On any other chain it refuses.
- **Load the agent from this token** reads `getPayload`, `tokenURI`, `ownerOf` and `openMint`, finds the agent whose
  THOT generation has that `contentRoot` (every manifest is archived by CID in `<agent>/thot/`), verifies its
  bundle, and walks the THOT parent links from the current generation back to the minted one. The genesis stays
  attached to the token forever; later generations are proven to descend from it.
- **Start a local devnet** runs anvil (127.0.0.1:8545, chain 31337), deploys `iNFT_7857` from its compiled artifact
  (`DeltaVerse/deploy/iNFT4/out`) with account 0 as admin and minter, and fills in the fields. The whole path then
  runs on this computer.

Where it stands: `iNFT_7857` is **not deployed on any public chain**, its audit (`iNFT4/audit.json`) is not cleared,
and a real mint needs a wallet holding MINTER_ROLE. `testing/test_chain.py` runs the whole path on a throwaway
anvil: deploy, simulate (and the refusal without the role), unsigned transaction, mint, read-back, the double-mint
refusal, load, two more generations and load again, and the refusal to send off a devnet.

## 9. Receipts, and how to check an answer

Every answer from `bankml serve` carries:

| field | meaning |
|---|---|
| `bankml` | the version that served it |
| `engine` | what computed it: `llama.cpp b11192 llama-server (loopback), behind bankml P0`, or with `--native` `bankML <version> native: its own forward pass, …` (and the GPU when one took part) |
| `model_sha256`, `guard` | the file that was verified and pinned, and the guard's verdict |
| `prompt_tokens`, `completion_tokens` | the engine's own counts |
| `ttft_ms`, `wall_ms` | time to first token and total, at the gateway (`ttft_ms` is `null` on a non-streamed answer in 0.3.6; the next release sets it there too) |
| `response_sha256` | sha256 of the answer text exactly as the model produced it (`choices[0]`, UTF-8; an unpaired `\u` surrogate counts as U+FFFD) |
| `request_sha256` | (0.1.7+) sha256 of the request body bankml forwarded: which prompt this answer is to |
| `signed` | always `false`: a receipt is not a signature (below) |

The UI recomputes the answer's sha256 and shows ✓ when it matches. A mismatch shows `(≠ received!)`. To check an
answer yourself, hash `assistant_raw` from `.history` (older records: `assistant`) and compare it with
`receipt.response_sha256`.

**What a receipt proves, and what it does not.** It binds an answer to the request it answered and to the sha256 of
the model file bankml verified before serving, and it catches any change to the text afterwards. It is **not
signed**: anyone can write a receipt, so it shows integrity between you and your own bankml serve, not to a third
party who does not trust your machine. bankml serve also re-checks, before each answer, that the model file is the
one it verified (device, inode, size and modification time) and that the engine still serves that file; if either
changed, it refuses to answer rather than issue a receipt for weights it did not verify.

## 10. Savante's canon and the iNFT ledger

Savante is defined by her canon, `~/cryptoAGI/savante` (github.com/cryptoAGI/savante), bound for an iNFT by the ledger
`savante.commitments.json`. The ledger commits by sha256 (and CIDv1) to:
- the persona;
- the charter;
- the sagi skill;
- six `sAGI.*` facets;
- the agent card;
- the image;
- the thot bundle;
- a keccak256 **doctrine root** over 15 clauses an owner may not edit.

At start the UI re-hashes every one of those files and shows the result in **Integrity**. If the persona does not
verify, the UI refuses to speak as Savante. The **Verifier** tab runs the full offline check. Nothing here mints: the
card's status is `not_yet_minted`, and the decision to mint belongs to the owner's signature. Point the UI at another
checkout with `SAVANTE_CANON=/path`.

## 11. Testing and the release gate

```sh
cargo test --release                                             # offline
BANKML_GGML_LIB=/path/to/llama-b11192 testing/release_gate.sh    # everything → testing/results/<version>.txt
testing/live.sh "title" <command…>                               # one step, shown live in view mode
```

The gate runs, in order:
1. the build, the unit tests and the end-to-end CLI tests;
2. clippy (`-D warnings`);
3. the licence headers (`testing/spdx_check.py`);
4. the Python suites: the guard, the UI data layer, the PostgreSQL connector (a throwaway cluster), the iNFT path (a
   throwaway anvil devnet), the model importer (with a real carrier on spare ports);
5. the Rust and Python guards agreeing on every case; the C API's printf oracle (`bankml_log` against libc
   `snprintf`, from C);
6. every oracle: bankml's kernels must equal llama.cpp's compiled kernels, bit for bit, on every weight of the real
   models;
7. the A/B speed tests, the prefill tile, the memory floor and the whole-token budgets;
8. the conversation oracles: `serve --native` through `/api/chat` and `/v1`, and the C API's `bankml_chat` against
   `serve --native` (ternary) and llama-server's record (1-bit);
9. the live oracles that start `serve --native` and compare it with llama-server's answers: JSON mode and schemas, the
   persona layer, the penalties, and in the next release the samplers, the context limit, slots, sessions, logprobs
   and the q8_0 KV cache.

A speed counts only if every oracle passed on the same code. See [testing/README.md](../testing/README.md), and
[oracles.md](oracles.md) for what each oracle compares.

## 12. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| chat says *Cannot reach bankml serve* | serve not running | `./install.sh status`, then `./install.sh model` (§3) |
| the installer stops at a step | that step's prerequisite is missing | it says which; fix it and run `./install.sh` again (steps already done are quick) |
| serve prints `refuse: … != pinned` | the file is not the one the fork pinned | re-download; `bankml sha256 FILE` |
| serve prints `upstream … serves X, not the verified Y` | the llama-server on that port runs another file | restart it with the verified file, or use `--spawn` |
| serve prints `upstream … not healthy` | llama-server not up (a cold load can take a minute) | wait; check its log |
| the first answer takes minutes | CPU prefill of the ~300-token system prompt (≈3 tok/s on a laptop) | expected; later turns reuse the cache |
| *refused: savante.persona does not verify* | the canon was edited or is incomplete | `git -C ~/cryptoAGI/savante status`; run the Verifier |
| view mode says *swap full* | the machine is under memory pressure | close other work before measuring |
| view from another computer does not load | firewall, or bound to 127.0.0.1 | `--host 0.0.0.0`; allow TCP 7874 |

## 13. Reference: commands, ports, environment

| command | what |
|---|---|
| `./install.sh [step …] [--skip-tests] [--no-start] [--view] [--voice]` | install and start (§1); steps: check build engine python canon model voice start stop status, and (0.3.7, opt-in, sudo) `power [--remove]` ([install.md §2](install.md#the-steps)) |
| `bankml guard FILE [--engine mainline\|prism] [--json]` | header check: play / refuse / need_more |
| `bankml sha256 FILE` | the file's sha256 |
| `bankml pin FILE --fork FORK.json` | sha256 against the fork's record |
| `bankml verify FILE --fork FORK.json [--engine mainline\|prism] [--json]` | guard, then pin |
| `bankml serve FILE --fork FORK.json [--upstream H:P \| --spawn BIN] [--listen H:P] [--threads N] [--ctx N] [--spec-ngram] [--slot-dir DIR]` | the gate in front of llama-server (n-gram speculation opt-in; `--slot-dir` becomes llama-server's `--slot-save-path`) |
| `bankml chat-template MODEL.gguf < messages.json` | the prompt a conversation becomes, as llama.cpp's `/apply-template` (P3; byte-identical on its oracle; tools and assistant prefills refused) |
| `BANKML_GPU=off` · `BANKML_GPU_SHARE=0.3` | the GPU worker (0.2.14): a verified card takes a calibrated share of every 1-bit matrix's rows; `off` disables it, a number overrides the share |
| `BANKML_GPU_LIMIT=0.8` | the GPU limiter (0.3.7, next release): bankML's GPU buffers stay within this share of the card's heap (of the RAM they could use, on an integrated card) and the card rests so it is busy at most this share of the time; what does not fit or arrives while it rests runs on the CPU, the same bits. Each matrix shape also decides, measured in the pipeline, whether the card's share pays (on the Vega 3 most shapes choose the CPU) |
| `bankml serve FILE --fork FORK.json --native [--listen H:P] [--upstream H:P] [--ctx N] [--registry [DIR]] [--keep-alive DUR] [--slot-dir DIR]` | answers from bankML's own forward pass (0.3.0); since 0.3.1 also Ollama's API (§6a) and, with `--registry`, any pinned model by name, one resident at a time; token-identical to llama-server on its oracle; also serves llama-server's endpoints on the engine address, so Savante reaches it unchanged. `--slot-dir` (0.3.8, next release): llama-server's slot save/restore (§6). Savante's Admin tab **engine** setting (under *advanced*) chooses it: `auto` (bankML for the ternary files), `native`, `llama.cpp` |
| `bankml create NAME -f Modelfile [--registry DIR] [--models DIR]` | O5 (0.3.5): a derived model, a layer (SYSTEM, PARAMETER, stop, MESSAGE, LICENSE) over a pinned base, `FROM` a name, a pinned GGUF or a safetensors directory (converted); written as `DIR/NAME.MODEL.json`, no weights copied (§6a) |
| `bankml convert DIR -o OUT.gguf [--model-name N] [--fork F --source S] [--ignore-model-card]` | O5 (0.3.5): Llama safetensors → GGUF F16, byte-identical to llama.cpp b11192's `convert_hf_to_gguf.py --outtype f16`; `--fork` writes the pin with every input's sha256 |
| `bankml gpu --verify` | runs the bit-exact kernel oracle on every usable card (0.2.13); a card that fails is named and never used |
| `bankml gpu [--remote]` | every video card found (Vulkan, merged with `/sys/class/drm`) and which bankml will use; `--remote` adds the GPUs Hugging Face rents (22 NVIDIA flavors with card counts and prices; listed, never started); `BANKML_GPU=off` turns the component off, `BANKML_GPU=0,2` picks cards (0.2.12; the GPU kernels are the next steps) |
| `bankml generate MODEL.gguf [--max N] [--sample [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S]] < messages.json` (or plain text) | bankml's own forward pass, greedy, or with `--sample` llama-server's sampler chain (the model's defaults unless given; same seed, same tokens as llama-server, 0.2.11), streamed (P3, 0.2.7): token-identical to llama-server b11192 on its oracle for the 1-bit and ternary models, prompts of any length and contexts of any length (all three of ggml's CPU attention kernels, since 0.2.10); `BANKML_LLAMA_THREADS` (default 3) must equal the `-t` of the llama.cpp being matched, because its long-context decode kernel chunks by thread; `BANKML_THREADS` sets the threads. Ternary: 2.3–2.4 tokens/s against llama-server's 0.30; 1-bit: 1.8 against 2.8 |
| `bankml tokenize MODEL.gguf [--no-special] < text` | token ids, as llama.cpp's `/tokenize` (P3's tokenizer; token-identical on its oracle) |
| `bankml usage [PID …]` | memory, cores, and each process's resident memory and CPU % (bankml's psutil, from `/proc`); `bankml serve` answers the same at `GET /bankml/usage`, with (0.3.7, next release) package watts when RAPL is readable (`./install.sh power`), each GPU's busy %, VRAM and GTT, and the GPU limiter's state |
| `GET /bankml/metrics` | (0.3.7, next release) bankML's own measurements of its last 256 answers: TTFT, prompt and generation tokens per second, grammar time, energy and joules per token where measured; totals |
| `python3 sAGI/savante.py --mode interact [--port 7873]` | talk to Savante (loopback) |
| `python3 sAGI/view.py [--host 0.0.0.0] [--port 7874]` | the read-only page for the LAN |
| `python3 sAGI/console.py [--port 7875]` | (0.3.7, next release) the bankML console, loopback only (§6c) |
| `python3 sAGI/models.py list \| catalog \| search Q \| import ID\|URL\|ollama:NAME:TAG \| use FILE \| first-run` | the model importer and carrier switch |
| `python3 sAGI/speak.py [--prune]` | render every voice clip, write both exports (`--prune`: drop unused clips) |

| port | service |
|---|---|
| 18092 | llama-server b11192 (upstream, loopback) |
| 18093 | bankml serve (loopback) |
| 7873 | interact mode (loopback) |
| 7874 | view mode (LAN) |
| 7875 | the bankML console (loopback; 0.3.7, next release) |

The variables below are the ones a user meets most. The complete list, with where each is read, is in
[install.md §6](install.md#6-environment-variables-the-complete-list).

| variable | default | meaning |
|---|---|---|
| `SAVANTE_CANON` | `~/cryptoAGI/savante` | Savante's canon (read-only) |
| `BANKML_SERVE` | `http://127.0.0.1:18093` | where the UI finds bankml serve |
| `BANKML_UI_STATE` | `~/.local/share/bankml/savante` | `.history` and caches |
| `BANKML_REPO` | the checkout | where view mode reads `testing/` |
| `BANKML_AGENTS` | `~/.local/share/bankml/agents` | custom agents (one git repository each) |
| `BANKML_PG_DSN` | `dbname=bankml` | PostgreSQL for publishing and loading agents |
| `RAGE_PATH` | `~/mindX/mindx/godel/mindxtrain/hf/space_ui` | where the ragebar finds mindX's `rage.py` (built-in BM25 otherwise) |
| `BANKML_GGML_LIB` | — | llama.cpp b11192 release dir, for the oracles |
| `BANKML_THREADS` | pool: all cores; budgets: `1,2,3,4` (Q1_0), `1,3` (Q2_0) | the thread pool's size, or a comma list of thread counts for the decode budgets |
| `BANKML_NO_SHANI` | unset | set to hash with the portable SHA-256 instead of the CPU's SHA extensions |
| `BANKML_OLLAMA_MODELS` | `/usr/share/ollama/.ollama/models` | the local Ollama store the importer adopts from |
| `BANKML_PIPER`, `BANKML_PRONUNCIATION` | `~/.local/share/bankml/piper`, the house table | Savante's voice engine and pronunciation table |
| `BANKML_ESPEAK`, `BANKML_ESPEAK_VOICES` | the DeltaVerse vendor dirs | the eSpeak fallback voice |
| `BANKML_OLLAMA`, `BANKML_EMBED_MODEL`, `BANKML_EMBED_KEEP_ALIVE`, `BANKML_EMBED_NEED_GB` | see [embedding.md](embedding.md) | the embedding model |
| `BANKML_MODELS` | `.models` in the checkout | where imported models go |
| `BANKML_FORKS` | `~/.local/share/bankml/forks` | FORK.json pins, one per imported model |
| `BANKML_LLAMA_SERVER` | `~/sAGI/bonsai/llama-b11192/llama-server` | the engine the carrier spawns |
| `BANKML_BIN` | `target/release/bankml` | the bankml binary the importer and switch call |
| `BANKML_FIRST_RUN` | `1` | `0` stops interact from starting the Bonsai-8B carrier by itself |
| `BANKML_CTX`, `BANKML_THREADS_SERVE` | `2048`, `3` | the engine's context and threads when the importer starts the carrier |
| `BANKML_SERVE_LISTEN`, `BANKML_UPSTREAM` | `127.0.0.1:18093`, `127.0.0.1:18092` | the ports the switch manages |
| `BANKML_VOICE_DIR`, `BANKML_EXPORT_DIR` | `sAGI/voice/cache`, `sAGI/voice/export` | voice clips and the two exports |
| `BANKML_VOICE_ASYNC` | `1` | `0` stops the UI rendering missing clips in the background |
| `BANKML_CACHE_RAM` | 8192 MiB, at most a quarter of the memory available at load | (0.3.8, next release) the native engine's host prompt cache in MiB, as llama-server's `--cache-ram` (0 off, -1 no limit) |
| `BANKML_CACHE_TYPE` | `f16` | (0.3.9, in progress) `q8_0` keeps the native KV cache as llama.cpp's `--cache-type-k/v q8_0` (§6) |
