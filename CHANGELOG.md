# Changelog

## Unreleased (0.4.1) — a request whose client has gone stops

### The console's SELF: who is speaking, then one line of measurements
- An A/B by the ultimate-bankml-ui session (bankML 0.4.0, Bonsai-8B, "who are you?", three fresh conversations each,
  every receipt ✓): with SELF's first line "the model I am running: Bonsai-8B …", bankML said "I am bankML" 1 of 3
  times and "I am … Bonsai-8B" 2 of 3; with "I am bankML 0.4.0. The model file I run is Bonsai-8B (sha256 …): it is
  what I compute with, not who I am", 3 of 3. The console's SELF now opens with that line.
- The nine measurement lines are one line, each value still with its unit and "not measured" where it was not: the
  block a question carries is two lines instead of ten, a smaller prefix to recompute whichever turn it rides on
  (the same session measured why that matters for the prompt cache).

### GPU objects are released on drop (`gpu_objects_are_released_on_drop`; the first 0.5.0 item)
- Found by the 2026-10-06 audit: `Gpu` destroyed nothing on drop (device, buffers, pipelines), so dropping and reopening
  a worker leaked its card memory; and `unsafe impl Sync for Buffer` let two `Gpu`s on two threads write one buffer.
- Every object is now released on drop. The Vulkan instance, the device and its entry points live in one shared
  `Device`, held by the `Gpu` and by every `Buffer` and `Pipeline` made on it. The device, and then the instance, go
  with the last of them, so nothing is used after its device is gone. A `Buffer` is no longer `Sync`, and `write` and
  `submit` refuse a buffer or pipeline of another device.
- The new test found what the audit had not: each `Gpu::open` also made a Vulkan instance and never destroyed it.
  - Measured with the driver's own counters (sysfs VRAM + GTT), over 30 rounds of open, five 16 MB buffers, a
    pipeline, a run and drop.
  - Before, card memory grew 65 MB, about 2.2 MB a round; after, 221.4 → 223.4 MB, within the desktop's own noise.
- The same bits: the GPU kernel oracle and `bankml gpu --verify` are bit-exact on every row, and the 1.7B forward
  oracle is 840 / 840 with the card computing 35 % of each 1-bit matrix.

### Abort a native run when its client goes away (`testing/disconnect_oracle.py`, 6 / 6)
- Found by the ultimate-bankml-ui session under contention: `serve` noticed a closed connection only when a write
  failed, so an abandoned request's whole prefill (and, unstreamed, its whole answer) ran for nobody, minutes per
  request on the laptop, with the one slot held.
- `Native::complete_while` asks `alive()` between prefill micro-batches (`Weights::prefill_while`: the same 512-token
  micro-batches, so the same bits) and before each generated token. `serve::client_alive` is a non-blocking peek at the
  client's socket: end of file or a reset means gone, anything else (bytes waiting, nothing yet, any doubt) means
  there. `/v1/chat/completions` (streamed or not) and Ollama's `/api/chat` and `/api/generate` all use it.
- A stopped run logs where it stopped (`bankml: the client went away; stopped after N of M prompt tokens (kept in the
  slot)`), records no answer, and leaves the slot holding exactly what it computed: the next request reuses it
  (`cache_n` > 0) and its answer is still an empty slot's.
- Checked live: an abandoned 2,294-token prompt stopped after its first micro-batch; the reuse and its answer; an
  abandoned 400-token stream stopped early; `/api/generate` abandoned the same way. No answer changed:
  `oracle_native_serve_o4`, `cache_prompt_false_is_an_empty_slot`, the session (14 / 14), logprobs (14 / 14) and
  Ollama-shape (9 / 9) oracles all pass. `disconnect_oracle_live` is in the release gate.

## 0.4.0 — 2026-10-07 — milestone: native serving complete

**Everything Savante and mindX ask of llama-server, answered by bankML's own engine, identical to llama-server
b11192.** 0.3.3–0.3.8 brought JSON mode and schemas, the penalties and the whole default sampler chain,
self-measurement, the context limit, slots, the prompt cache and logprobs; 0.4.0 closes the list: a q8_0 KV cache
with llama.cpp's Hadamard rotation, the grammar mask 13× faster, `cache_prompt: false` as llama-server honours it,
fewer pool wakes per token, and the engine's own status, diagnostics and a console whose landing is the ultimate
input field. **1-bit decode is at least at llama-server's speed**: the pinned 8B A/B, three rounds, bankML at or
above llama-server in every one, median 2.02 against 0.77 tokens/s under the same load, every answer token-identical
([PERFORMANCE.md](docs/PERFORMANCE.md#1-bit-decode-against-llama-server-040)) — so the importer's `auto` engine now
chooses bankML's own forward pass for the 1-bit files as well as the ternary ones (`AUTO_NATIVE_Q1`). Record:
`testing/results/0.4.0.txt`, one run of the full gate, every stage passed (2026-10-07, the laptop; the decode budgets
at three threads: 1-bit 1.02× ggml b11192's own kernel, ternary 8.9×).

### The q8_0 KV cache (`BANKML_CACHE_TYPE=q8_0`; `testing/kv_oracle.py`, 6 / 6)
- K and V stored as q8_0 blocks (`KvType::Q8_0`), 53 % of the f16 cache's bytes. Attention over it is ggml's
  reference kernel for a non-f16 K (`attend_head_q8`): Q quantized with the AVX2 `quantize_row_q8_0`, scores by
  `vec_dot_q8_0_q8_0`, V dequantized into an f32 accumulator (`vec_mad_f32`, `vec_scale_f32`).
- **llama.cpp b11192 rotates around a quantized cache**, which the first replay found (2 / 6, the answers drifting
  after dozens of tokens): Q and K through a Hadamard transform over the head (128), V over 64, and the attention
  output back (`attn_rot_k`/`attn_rot_v`, `llama_mul_mat_hadamard`, which ggml's CPU computes as a fast
  Walsh–Hadamard transform). bankML's `fwht` does the same arithmetic: scale by `1/sqrtf(n)`, then `u + v`, `u − v`.
- `oracle_ggml_b11192_q8_0_kv_kernels`: against the shipped haswell library (dlopen), 4,000 rows quantized
  byte-exact and 4,000 dot products bit-exact, saturating `maddubs` cases included.
- `kv_oracle_live`: llama-server b11192 with `--cache-type-k q8_0 --cache-type-v q8_0` and bankML with
  `BANKML_CACHE_TYPE=q8_0` give the same answers, 6 of 6 (greedy, seeded, a 2,244-token prompt, a 320-token answer,
  a two-turn conversation; text, counts and `cache_n`; 567 tokens).
- A slot file records its cache type; one saved with the other type is refused with the reason.

### The grammar mask, through a trie (`oracle_grammar_masks`, both paths)
- A whole-vocabulary mask walks a trie of the vocabulary's code points, built once: each grammar stack meets a
  shared prefix once, and the stacks after a terminal are computed once per stack and mask. Median **2.94 ms
  against 38.8 ms** (13×), p90 49.3 against 80.6 ms, over the oracle's 1,645 masks. The oracle now computes every
  mask both ways: **196 / 196 runs, 1,645 / 1,645 masks** identical to llama.cpp b11192 by each.

### `cache_prompt: false`, as llama-server (found by `testing/decode_ab.py`)
- The first pinned 1-bit A/B refused its first round: bankML's and llama-server's answers parted after about 40
  tokens. The harness sends a warm-up, then the measured request with `cache_prompt: false`; llama-server b11192
  then computes the whole prompt (`n_past = 0`, server-context.cpp), while bankML ignored the flag and reused the
  warm-up's prefix, a different split of the prefill and so different bits. bankML now reads `cache_prompt` (default
  true) and, when it is false, starts from position 0 as llama-server does (`Params::cache_prompt`,
  `Native::complete_with`). Test: `cache_prompt_false_is_an_empty_slot` (after a warm-up, the answer, tokens and
  `cache_n` 0 are an empty slot's; with the flag on, all but one prompt token are reused), in the release gate.

### A Thesis tab in the console
- The authors' thesis and the sixteen contributions measured against it, read from docs/TECHNICAL.md's
  [Thesis](docs/TECHNICAL.md#thesis--professor-codephreak-and-gregory-l-magnusson) section each time the tab opens
  (`GET /api/thesis`), so the page never drifts from the report; drawn as DOM nodes by a small Markdown reader (bold,
  italic, code, links, headings, lists; never `innerHTML`), relative links resolved to the repository, in a window
  with the same depth, set for reading.
- `sAGI/console/thesis.css`: the thesis as a scroll of accordions. Each principle and each of the sixteen
  contributions is a section titled by its bold lead; the window scrolls on its own with a progress line along its
  top, and a section opens as it comes into view (one the reader closes stays closed); expand all, collapse all. On a
  narrow screen it scrolls with the page, and the page is what is watched. Reduced motion is honoured.
- Fixed on the way: the input field's stylesheet defined generic theme tokens (`--accent`, `--muted`, …) on `:root`,
  which overrode the console's own since the landing arrived (accent bars and fills went transparent); the console
  build now renames them `--uif-k-*`.

### Diagnostics: every component of sAGI, in windows with depth, on any screen
- The Diagnostics tab first asks every part of sAGI, in parallel and within a time limit, each in its own words
  (`diagnostics.components`): bankml serve, the console, Savante (`:7873`), view (`:7874`), the model importer
  (pins, the binary, llama-server, the engine setting), each persona (preflight, doctrine root), the agents, THOT
  manifests (each bound agent's verified), the chain artifact for iNFT mints, PostgreSQL (`connectors.status`),
  embeddings (`embed.status`), the voice (`speak.available`) and the spans. Ready, idle, to check or failing, with
  what each said and how long it took; nothing is started or written. A public console shows each component's
  level, not its paths or agent names.
- Each section is a window: a title bar, a hairline border and two shadows (near and far), in light and dark; the
  components are cards with a status edge. The Engine tab's cards share the depth.
- Mobile: the tab bar scrolls sideways under the title, windows run edge to edge, cards stack in one column, trace
  rows wrap; the ultimate input field keeps a separate layout on a narrow screen (full width at the bottom, without
  side modules), so a desktop's saved positions never put it off a phone's screen.

### Fewer wakes of the pool per token (1-bit and ternary decode)
- Measured first: under the same pool, bankML's 1-bit kernel is 1.58× ggml b11192's own `vec_dot` over the 8B
  model's 253 matrices (`decode_budget_q1_0`), so the decode gap to llama-server is not the arithmetic. A token's
  time inside bankML is about 91 % matrix–vector products and under 2 % everything else; the rest is how the work
  is handed out: each product woke the pool once, seven times per layer.
- `Pool::rows_multi`: several outputs' rows in one wake, in chunks that never cross a matrix, each row by the same
  single-thread kernel. `Weights::mv_many` uses it for Q1_0 and Q2_0 on the CPU, so Q/K/V and gate/up are one wake
  each (seven per layer become four). The bits are the same: `oracle_forward_model_bonsai_1_7b` 840 / 840 and
  `oracle_forward_model_ternary` 1,064 / 1,064 with the GPU off, `oracle_native_serve_o4` 9 / 9 and 23 / 23 per model.

### The console's landing is the ultimate input field
- The Ask tab now opens on [ultimate-input-field](https://github.com/Professor-Codephreak/ultimate-input-field) (MIT):
  one elegant field (chat, or **T** for terminal commands) whose answers stream into output fields you can move,
  spawn, pop out to another monitor and call home, each one its own conversation with bankML through the console's
  `POST /api/ask`, every answer ending with its receipt, checked again in the browser. The build lives in
  `sAGI/console/uif/` (`uif.js`, `uif.css`; React bundled, so the console's CSP still loads nothing from elsewhere);
  its source is ultimate-bankml-ui's `console-mount` branch (`src/console/`). Without the build, the plain question box
  comes back.

### The engine's own status page, and an Engine tab (`GET /bankml/status`)
- `GET /bankml/status` on both of serve's addresses: the verified model, serve's settings (listen, upstream, native,
  threads, KV cache type, GPU, allowed origins, uptime), the CPU (`/proc/cpuinfo`: model, each logical CPU's clock),
  memory and swap, the disk where the model lives (`statvfs`: total and available; the model's size; this process's
  bytes read and written, `/proc/self/io`), every GPU and bankML's limiter (`/bankml/usage`), the answers measured
  (`/bankml/metrics`), the checks drawn from those (verified, memory, swap, disk, GPU, the log's last warning) and the
  engine's log — the newest 200 messages, now kept in a ring by `bankml::log` (serve's own startup lines go through it).
- A browser at serve's root (`http://127.0.0.1:18093/` or the engine address `:18094/`) gets that status as a page,
  refreshed every 3 s; every other client still gets Ollama's `bankml is running…` line, which Ollama clients check.
- The bankML console's **Engine** tab draws the same status with the same renderer (`bankML/status.js`,
  `status.css`: one source for both pages), through `GET /api/engine`.
- The first reading found the laptop's swap full (2.15 of 2.15 GB): the `swap` check now says when the machine is
  paging, because speeds measured then are not comparable.

### scientific.diagnostic — every number to 18 decimals (`testing/scientific_diagnostic.py`)
- One greedy request with `logprobs` and `top_logprobs: 5`, sent to a fresh llama-server b11192 and a fresh
  `bankml serve --native` (both computing the whole prompt). For every token and each of its top five: the 32-bit
  float's bits, its exact value to 18 decimals (rounded half-even from the float's exact binary value) and as an
  integer count of 10⁻¹⁸ (the 18-decimal fixed point of an ERC-20 amount), and the exact difference between the
  engines. Sums are exact; the perplexity is computed with 60 significant digits; timings are measured to the
  nanosecond and the record says the digits past the ninth are zero by construction.
- First record, Bonsai-1.7B, 32 tokens: **identical** — 32 of 32 tokens bit-equal with their top fives, max |Δ|
  `0.000000000000000000`, log-likelihood `-8.221305353127718263` on both engines. Written to the console's state as
  `scientific.diagnostic` and shown in the Diagnostics tab.

### A restart never trades a running engine for a missing binary
- The console's *Apply* (and the model switch) stopped the engine first and only then found there was no `bankml`
  to start: a raw `[Errno 2] No such file or directory` and, at worst, no engine. `models._need_bankml()` now refuses
  first — nothing is read, saved or stopped — and says how to fix it (build, `./install.sh`, or `BANKML_BIN`).
  Tested in `testing/test_models.py` for `apply_resources` and `switch`.

### Diagnostics in the console (`sAGI/diagnostics.py`)
- A **Diagnostics** tab beside Logs: measured checks (serve reachable and its latency, the model verified, metrics,
  memory, CPU, the engine log's last error, failed spans) and the trace of each recent answer, span by span with
  durations and timed events (`ask` › `self_block`, `engine.stream` with *headers*, *first piece* and *receipt*,
  `metrics`, `log.write`, `receipt.verify`). The first live trace showed where a slow answer's time goes: the console
  saw the first piece at 196,241 ms, bankML's own metrics said TTFT 196,240 ms, and the 524-token persona prompt
  read at 2.67 tokens/s was nearly all of it.
- The span model is LlamaIndex's instrumentation (run-llama/llama_index, MIT, `llama-index-instrumentation` at
  ec837e5): SimpleSpan, BaseSpanHandler's open / completed / dropped spans, SimpleSpanHandler's tree building and its
  repair of a missing parent. Ported to the standard library (no pydantic, no treelib); a contextvar holds the current
  span, spans carry timed events, and only the newest 400 are kept. Spans hold counts and statuses, never a question.
  `GET /api/diagnostics`; tested in `testing/test_console.py`.

### Your own bankML from a web page (`--allow-origin`)
- `bankml serve … --allow-origin ORIGIN` lets one named web page call the gateway from a browser: its CORS preflight
  is answered (with Chrome's private-network grant, as a public page reaching a loopback address requires) and its
  answers, streamed or not, carry `Access-Control-Allow-Origin`; any other origin gets none and a 403 preflight. The
  loopback `Host` and JSON-POST rules are unchanged. `./install.sh start --space` turns it on for the Space (remembered
  in `install.env`, applied to the carrier `sAGI/models.py` starts; `--no-space` turns it off). It is what lets the bankML Space's page
  ([PYTHAI/bankml](https://huggingface.co/spaces/PYTHAI/bankml)) talk to a visitor's own bankML, free, on their own
  CPU — the way Savante's page reaches a local engine. Checked live: the preflight, plain and streamed answers with
  the receipt, and another origin refused.
- Several pages: `--allow-origin` takes a comma-separated list, each origin compared exactly (`origin_allowed`; a
  suffix or an empty entry never matches). `--space` now allows both of bankML's Spaces: the classic page and
  [PYTHAI/ultimate-bankml-ui](https://huggingface.co/spaces/PYTHAI/ultimate-bankml-ui), bankML behind the ultimate input field.

### The code audit and the documentation (2026-10-06)
- **Comments, professional and short.** Every source file's comments were cut to the contract, the invariants, the
  safety reasoning and the exact upstream behaviour that keeps the bits; the history and rationale moved to its
  `docs/modules/<name>.md` page (a new **Design notes** section where needed). Each header ends with
  `Details: docs/modules/<name>.md`. A checker held every non-comment line identical; clippy and rustdoc are clean.
- **Fixed by the audit:** `bankml create` accepts `PARAMETER typical_p` and `min_keep` (refused as "not reproduced"
  although 0.3.7 reproduces them); the q8_0 cache rotates only for a head size that is a multiple of 64, as
  llama.cpp's `attn_rot_k/v` (128 unchanged, so 6 / 6 stands); `bankml_log`'s `%f` honours the 65,536
  width/precision limit like every other conversion (`%.100000000f` allocated); the 404 bodies list every route;
  stale refusal texts (the `mirostat` message, the C API's header and docs, the train `probe` stage) and ten comments
  that misstated the code (slot file layout, the `/bankml/usage` interval, doc blocks attached to the wrong item…).
- **Found, recorded for 0.5.0:** the GPU worker destroys nothing on drop (a reload leaks card memory), and
  `Sync for Buffer` is wider than it should be (TODO 0.5.0).
- **Docs:** a full pass over every page against the code (the oracles table now lists every gate stage; TECHNICAL,
  BUILD_HISTORY, OLLAMA, CAPI, usage and install brought to 0.3.9); `docs/thesis.md`, the bankML thesis; 48 Markdown
  files, every link and anchor resolving.

## 0.3.8 — 2026-10-06 — the serving contract: the context limit, slots, the prompt cache, logprobs

**`bankml serve --native` now behaves as llama-server b11192 at the edges a client meets in production: a prompt
that does not fit, a slot saved and restored, conversations that take turns, and the probabilities behind each
token.** Each is checked against
llama-server's own answers, as everything before it. Record: `testing/results/0.3.8.txt`, one run of the full
gate, every stage passed (2026-10-06, the laptop).

### The context limit (`testing/context_oracle.py`, 8 / 8)
- With context shift off (llama-server's default), a generation that fills the context stops there with
  `finish_reason` `"length"`, and a prompt that does not fit is refused before any work with llama-server's 400 body:
  `{"error": {"code": 400, "message": "request (N tokens) exceeds the available context size (M tokens), try increasing
  it", "type": "exceed_context_size_error", "n_prompt_tokens": N, "n_ctx": M}}`. Eight requests from a few tokens to
  past the context, at `--ctx 256`: the same text, counts, finish and refusal bodies.

### Slots: save, restore, erase (`testing/slot_oracle.py`, 19 / 19)
- `POST /slots/0?action=save|restore|erase` on the engine address, as llama-server's slot API, with `--slot-dir DIR`
  (created at start). Replies carry `n_saved` / `n_written` / `timings.save_ms`, `n_restored` / `n_read` /
  `timings.restore_ms`, `n_erased`; refusals are llama-server's (501 without a slot directory, "Invalid slot ID",
  "Invalid action", "Invalid filename" by its `fs_validate_filename` rules, "Unable to restore slot: …").
- The file: a magic line, the model's sha256, the context and token counts, each layer's K and V (f16) and the tokens,
  then a sha256 of all of it; written to a temporary name and renamed. A file from another model, or damaged by one
  bit, is refused and the slot emptied — never half-filled; the next answer recomputes its prompt.
- The oracle is the engine itself: the answer after a restore — in the same server and after a restart — equals the
  answer from an empty slot, and the restore skips the prompt (`cache_n` > 0). `n_saved` counts as llama-server
  counts: the prompt and every generated token but the last.
- Savante and the console pass `--slot-dir` to the native engine, so a warm start survives a restart.

### One slot, and llama-server's host prompt cache (`testing/session_oracle.py`, 14 / 14)
- The single-slot contract, stated and checked: requests are served one at a time, as llama-server `-np 1` (Savante's
  flags). Four requests sent at once are all answered, none refused or mixed, each with the text it gets alone.
- `bankML/prompt_cache.rs`: llama-server's `--cache-ram` cache (on by default there). When a request shares little
  with the slot, the slot's tokens and KV rows go to RAM, and a cached state that keeps more of the prompt comes back.
  Three conversations taking turns (A1 B1 C1 A2 …) now match llama-server turn by turn: text, counts and `cache_n`.
  Before, each turn found only the system prompt in the slot, so prefill started elsewhere and the answers differed.
- `BANKML_CACHE_RAM` (MiB, as `--cache-ram`: 0 off, -1 no limit); unset, 8192 MiB but at most a quarter of the memory
  available at load.

### Docs and tools
- `docs/why-bankml.md`: the plain-language version (advantages, uses, how Rust helps, the road ahead), dated
  2026-10-06; published as a web page over the DeltaVerse substrate at
  [deltaverse.pythai.net/bankml](https://deltaverse.pythai.net/bankml).
- `tools/makecards.py`: a page's share cards (1200×630, 1200×1200, a 180 px touch icon) with the DeltaVerse $.
  `tools/seo.py`: audits a page or URL as a crawler and a link preview read it — the meta set, a canonical that must
  not redirect, JSON-LD, and each card's real size against its declared one (standard library only). The bankML page:
  47 / 47.

### Logprobs on `/v1/chat/completions`, streamed or not (`testing/logprobs_oracle.py`, 14 / 14)
- `logprobs: true` with `top_logprobs` (default 20) → `choices[0].logprobs.content`: per token its `id`, `token`,
  `bytes`, `logprob`, and the top tokens with theirs. Every logprob is the same 32-bit float as llama-server's:
  `get_token_probabilities` ported — a partial sort of the whole vocabulary (libstdc++'s, from 0.3.7), the softmax
  in that order, `logf`.
- llama-server's entry rules, reproduced: a token gets an entry only once the text has no incomplete UTF-8 after it,
  and carries the text sent since the previous entry; a stop word drops its own tokens' entries; control tokens read
  `""`; a top token holding part of a character is cut at the incomplete end (`validate_utf8`), other invalid bytes
  read U+FFFD, `bytes` raw. `top_logprobs` without `logprobs` is refused with llama-server's message.
- Streamed, as llama-server's partial responses: the stream opens with the role delta, and each whole token's entry
  rides on the chunk its text makes (a token that sends no text sends no entry: a held stop-word prefix, the
  end-of-turn token with nothing held back). Text now waits for a whole token in every stream, as llama-server sends
  nothing while UTF-8 is incomplete (`" 🍕"` is one chunk, not `" "` then `"🍕"`). Five streamed cases, every chunk's
  delta and entry identical. The engine reports each step (`Native::complete_with`, `Step`); `NativeChat::run_steps`
  makes the chunks.
- Not served: `/completion` (llama-server's own endpoint, so also its `n_probs`); clients use `/v1`.

## 0.3.7 — 2026-10-06 — the whole sampler chain; bankML measures itself; the GPU limiter; the bankML console

**Every sampler in llama-server b11192's default chain is now reproduced, token for token: after 0.3.6's penalties,
typical-p, top-n-σ, XTC, dynamic temperature and DRY.** A request that sets any of them gets llama-server's answer
instead of a refusal; what llama-server clamps is clamped, what it refuses is refused with its message. **bankML now
measures itself** — time to first token on every answer, prompt and generation speed, CPU, memory, the GPU and, where
the operator allows it, power and joules per token — **limits its GPU** to a share of the card's memory and time and
lets each matrix shape decide whether the card pays, and has **a console** in which it answers as itself, from those
measurements. Record: `testing/results/0.3.7.txt`, one
run of the full gate, every stage passed (2026-10-06, the laptop, the GPU limiter at its default).

### Measurement: what bankML knows of itself
- `bankML/metrics.rs`: one record per completion, taken inside the engine — TTFT, prompt and generation tokens per second
  (llama-bench's `pp` and `tg`), grammar time, and the CPU package's energy and joules per generated token when RAPL is
  readable; `null`, never estimated. A ring of 256 at `GET /bankml/metrics`, with totals.
- The receipt's `ttft_ms` is set for non-streamed answers too (it was `null`); `/v1` answers carry llama-server's
  `timings` fields (`prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`,
  `predicted_per_second`, `cache_n`).
- `GET /bankml/usage` adds `package_watts` (RAPL over the sampling interval, its wrap handled), each GPU's busy %, VRAM and
  GTT from sysfs, and the GPU limiter's state.
- `./install.sh power` (opt-in, the only step that uses sudo): a udev rule lets a `rapl` group that the user joins read
  the energy counter, which the kernel has kept root-only since PLATYPUS (CVE-2020-8694); it says so and asks first;
  `--remove` undoes it.

### The GPU: a limit, and per-shape calibration
- `BANKML_GPU_LIMIT` (default 0.8): bankML's GPU buffers stay within that share of the heap they come from — on an
  integrated card, of the RAM they could use — and after a dispatch of `d` the card rests `d·(1−L)/L`; a matrix that does
  not fit, or a product arriving while the card rests, runs whole on the CPU. Never a wait, the same bits.
- Per-shape calibration: each matrix shape's first products alternate between the card's share and the CPU alone, timed
  in the real pipeline; the faster median is kept and a shape decided for the CPU frees its buffers. Measured on the Vega
  3 (Bonsai-1.7B, interleaved A/B, 4 rounds, the same answer every time): with one global share the card cost 18 %
  (10.40 → 8.51–8.58 tok/s); per shape, 10.36 off against 10.30–10.38 on — neutral, as an integrated card that shares
  decode's memory bandwidth can be; card memory 66 → ~13 MB.

### bankML as itself: the persona and the console
- `sAGI/personas/bankml.persona`: bankML speaking as itself, projected from the README, the Thesis and the oracles
  (doctrine root `0x4d5ef2d9…`; adopts and verifies). Its knowledge of its own use comes only from a measured SELF block,
  rendered as sentences with units (a bare key was read as seconds in testing).
- `sAGI/console.py` (port 7875, loopback only): Interaction (the streamed answer and its receipt checked against the
  text), Admin (CPU threads, RAM budget and GPU limit sliders — one verified restart with rollback — and D3 charts of
  what bankML measured), Logging (every exchange with its timings and receipt), Infotags (ERC-721 metadata for an iNFT
  publication: the model, the persona and its doctrine root, an RFC 6962 root over the exchanges, CIDs). Standard library
  only, a strict CSP, same-origin JSON POSTs; D3 v7.9.0 vendored with its ISC licence. `testing/test_console.py`.

### Measured
- `oracle_samplers` (`testing/penalty_oracle.py --kind sampler`, llama-server b11192 from an empty cache; 23 variants
  × 4 prompts: each sampler alone, typical-p before top-p and min-p's unsorted path, XTC with a clamped probability and
  a disabling threshold, dynamic temperature at 0, DRY with defaults, custom breakers and with the repeat penalty, all
  five at once): mindx-gen39 **76 / 76** answers token-identical (3,576 tokens), Bonsai-1.7B **76 / 76** (2,587
  tokens), Bonsai-8B **76 / 76** (2,361 tokens, `oracle_samplers_8b`); **16 / 16** refusals on each, with
  llama-server's message.
- `oracle_std_sort` (`testing/sort_oracle.cpp`, libstdc++'s own `std::sort`): **876 / 876** orders identical — sizes 0
  to 1,000, heavy ties, sorted, reversed and equal keys — the order typical-p's unstable sort leaves equal scores in.

### Added
- `sampler.rs`: each sampler as `src/llama-sampler.cpp` writes it, in the default order (penalties, DRY, top-n-σ,
  top-k, typical-p, top-p, min-p, XTC, temperature, dist), with llama.cpp's `sorted` state tracked through the chain
  (typical-p leaves the set unsorted, so top-p sorts after its softmax and min-p takes its unsorted path); XTC's own
  `mt19937` and its float draw (`generate_canonical<float, 24>`: one 32-bit output); top-n-σ's squares in double
  (C++'s `pow(float, 2)`); dynamic temperature's float entropy; DRY's restart sequences, reverse Z-algorithm and libm
  `pow`; libstdc++'s `std::sort` (introsort) and generic heaps; `Params::default` = `common.h`.
- `native.rs`: the request fields `typical_p`, `top_n_sigma`, `xtc_probability`, `xtc_threshold`, `dynatemp_range`,
  `dynatemp_exponent`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`,
  `dry_sequence_breakers`; llama-server's soft limits clamp (`top_p`, `min_p`, the XTC fields to [0, 1], temperature
  to ≥ 0), a `dry_base` below 1 falls back to 1.75; DRY's breakers are built from the vocabulary's pieces once per
  breaker list and cached on the engine.
- On Ollama's API `typical_p` is honoured; DRY, XTC, top-n-σ and dynamic temperature are not Ollama options, so they
  come through `/v1` (and the C API).

### Changed
- **Refused instead of ignored:** `mirostat` on `/v1` (the Ollama path already refused it) and a custom `samplers`
  order — llama-server would act on either, so answering without them would not be its answer.

## 0.3.6 — 2026-10-04 — the penalties: llama-server's repeat, frequency and presence penalties (O2)

**`repeat_penalty`, `repeat_last_n`, `presence_penalty` and `frequency_penalty` are no longer refused: bankML applies
them as llama.cpp b11192's `llama_sampler_penalties` does, and its answers are token-identical to llama-server's.** The
coach's `ollama_predict` (`temperature 0`, `repeat_penalty 1.3`) and mindXtrain's imprint gate (`repetition_penalty 1.3`)
can now run on bankML. Greedy at 1.3, `mindx-gen39` answers instead of degenerating into `,,,,`. Record:
`testing/results/0.3.6.txt`: one run to `json_schema_oracle_live`, where the Vega 3 worker's memory (taken from the
laptop's RAM) stalled it; the three stages left ran next from the same tree and build with `BANKML_GPU=off`, marked in
the record. 0.3.7 adds a GPU limiter for this (`BANKML_GPU_LIMIT`).

### Measured
- `oracle_penalties` (`testing/penalty_oracle.py`, llama-server b11192 from an empty cache, greedy and seeded; 17
  variants × 4 prompts made to repeat): mindx-gen39 **56 / 56** answers token-identical (2,478 tokens), Bonsai-1.7B
  **56 / 56** (1,895 tokens); each **12 / 12** refusals with llama-server's message (`repeat_last_n` −1, a repeat
  penalty of 0 or below); Bonsai-8B **56 / 56** (1,568 tokens, `oracle_penalties_8b`), 12 / 12 refusals.

### Added
- `sampler.rs`: the penalties, first in the chain and again on a grammar's redraw; the window as llama-server fills it
  — every prompt token, cached or not, before the first draw (`server-context.cpp`), then each token drawn
  (`Sampler::accept`); the logit divided by the repeat penalty when positive, multiplied when not, then
  `count × frequency + presence` taken off; disabled as llama.cpp disables it (`last_n` 0, or all neutral).
  `general.sampling.penalty_last_n` and `penalty_repeat` read from the GGUF as llama.cpp reads them (none of the five
  native models sets them).
- The surfaces: `/v1` and `/api/*` (`repeat_last_n` moves from the ignored resource options to the honoured ones), a
  Modelfile's `PARAMETER`, the C API (through the same engine). `bankml generate --sample` applies them too, but has
  no flags for them: only a GGUF's own `general.sampling.penalty_*` defaults reach it (none of the five models sets one).
- `testing/penalty_oracle.py` (record, and `--bankml` live); `oracle_penalties`, `oracle_penalties_8b` and
  `penalty_oracle_live` in the gate. Live on mindx-gen39: **85 / 85** through `/v1` (top-level fields) and `/api/chat`
  (`options`), refusals included.

### Fixed (found by the live oracle)
- **A sampler refusal on `/v1` answered 500.** A negative `repeat_last_n` or a repeat penalty of 0 or below reached
  the engine before it was checked; the Ollama path already refused it up front. `/v1` now refuses it as llama-server
  does: a 400 with its message.

## 0.3.5 — 2026-10-03 — JSON schemas, and `bankml create`: mindX's persona layer, natively (O6b, O5)

**Any JSON schema now gets the grammar llama-server b11192 builds for it, on the model's own template, and the answers
are token-identical to llama-server's. And mindX's own model can be made the way mindX makes it for Ollama:
`bankml create` takes mindXtrain's merged safetensors and promote.py's persona Modelfile, and the created
`mindx-gen39` answers the user's turns exactly as llama-server answers the same GGUF given that persona.** Phases O6b
and O5 (first cut) of docs/OLLAMA.md; built on two branches and merged here. Record: `testing/results/0.3.5.txt`. The
gate run was stopped by the development session's memory guard, not by a failure, as `json_schema_oracle_live` began
(the Vega 3 worker's memory comes out of the laptop's RAM); the three stages left ran next from the same tree and build
with `BANKML_GPU=off`, and the record marks where. The first run also found the ternary schema record had never been
taken; it was recorded from llama-server b11192 and passes, 11 of 11.

### Measured
- **Schema grammars** (`oracle_schema_grammars`, llama.cpp's own `json_schema_to_grammar` and
  `common_chat_templates_apply` in the release's `libllama-common.so`, no model): 173 schemas (llama.cpp's 81 test
  cases, Pydantic-shaped schemas like mindX's, edge cases) on each of the three templates — 148 grammars byte-identical
  bare and on the chat path, every refusal with llama.cpp's message (24 bare, 20 chat).
- **Answers under schemas** (`oracle_json_schema`, `_ternary`, `_o4`; llama-server b11192 greedy and seeded, each
  from an empty cache): Bonsai-8B **28 / 28**, Ternary-Bonsai-8B **11 / 11**, Bonsai-1.7B, SmolLM2-135M-Instruct and
  mindx-gen39 **56 / 56** each — objects, enums, ranges, `$defs`, a pattern, a date format, top-level arrays, enums,
  strings, integers and numbers, and answers cut by `max_tokens` inside an object, a top-level string and a number.
  Live: `/v1` (one streamed) and Ollama's `/api/chat` with `format: <schema>`.
- **The content rule** (`oracle_json_content`, llama.cpp's `common_chat_parse` itself): every prefix of every
  recorded constrained answer and edge cases, **30,063 of 30,063** texts on each template.
- **mindX's persona, end to end** (`oracle_persona_layer`, `testing/persona_oracle.py`): promote.py's
  `persona_modelfile_layer` Modelfile (mindX's persona as `SYSTEM`, `stop <|im_end|>`, `num_ctx 2048`), created
  `FROM` the merged safetensors directory and, unchanged, `FROM mindx-gen39` in place: **27 / 27** answers each
  token-identical to llama-server given the persona as the system message; live through `/api/chat`,
  `/api/generate` and `/v1`, and `/api/ps` names `mindx-gen39:latest`.
- **The converter, run against llama.cpp's directly** (b11192's `convert_hf_to_gguf.py`, torch 2.14.1+cpu): the same
  bytes as `bankml convert` on freshly downloaded mindx-gen39 (`6b64c748…`, the pin) and SmolLM2-135M-Instruct
  (`ec30a679…`). Until now the identity rested on O4's conversions.
- **The name heuristics** (`gguf-py/metadata.py`) agree with gguf-py on 168 of 168 Hub-style ids (`oracle_name_heuristics`).

### Added
- **`bankML/schema.rs`**: llama.cpp b11192's `common/json-schema.cpp` and `json-schema-to-grammar.cpp` (MIT,
  attributed; zero crates), nlohmann's JSON as the grammar text depends on it, and the chat parser's wrapping — the
  Qwen3 template's (with the `until-13` reasoning rules) and the ChatML templates' (none), as llama.cpp writes each.
  Wired: `response_format` `json_schema` and `json_object` + `schema`, the top-level `json_schema`, Ollama's
  `format: <schema>`, `bankml_chat`. The schema is read from the request's own text (`1.0` stays a float). Refused as
  b11192 refuses, with its message; and one deliberate divergence: a `pattern` with a non-ASCII character before a
  quantifier, which llama.cpp turns into a grammar that is not UTF-8, is refused.
- **`bankML/convert.rs`, `bankml convert`**: safetensors → GGUF F16 for the Llama architecture as SmolLM2 uses it,
  byte-identical to `convert_hf_to_gguf.py --outtype f16`; what would make llama.cpp write something else is refused
  with the reason.
- **`bankML/create.rs`, `bankml create`, `POST /api/create`, `DELETE /api/delete` (derived models only),
  `POST /api/copy`**: Ollama's Modelfile parser state for state; a derived model is `<registry>/<name>.MODEL.json`, a
  layer (`SYSTEM`, `PARAMETER`, `stop`, `MESSAGE`, licence) over a pinned base, verified through the base on every
  load; the layer applied as Ollama applies it; `/api/show` gives the Modelfile back. `FROM` a safetensors directory
  converts and pins it.
- **Oracles**: `testing/schema_oracle.{cpp,py}` (now all three templates), `testing/json_schema_oracle.py` (now every
  native model, and length cuts), `testing/content_oracle.{cpp,py}`, `testing/persona_oracle.py`,
  `testing/convert_oracle.py`; the gate runs them.
- mindXtrain ([huggingface.co/PYTHAI/mindXtrain](https://huggingface.co/PYTHAI/mindXtrain), archived source
  [github.com/Professor-Codephreak/mindXtrain](https://github.com/Professor-Codephreak/mindXtrain)) is linked from the
  README, OLLAMA.md and the `train/` and `convert.rs` headers: `bankml convert` and `create` replace its
  `serve --to ollama` path for mindX.

### Fixed (found by the new oracles)
- **A schema on a ChatML template got the Qwen3 grammar.** The O6b branch wrapped every schema as the Qwen3 template
  does; on SmolLM2's and mindx-genN's templates llama.cpp writes no reasoning rules and another root. bankML now
  builds the grammar on the model's template (the schema oracle checks all three).
- **The content of an answer cut inside an escape**: llama.cpp's parser ends the string before an unfinished `\` or
  `\uXX`; bankML kept it (12 of 29,973 texts differed before the fix).
- **A non-streamed answer whose parse is empty** (cut inside or right after the opening fence): llama-server answers
  with the raw text; bankML answered `""` (`pi` seeded on Bonsai-1.7B: `"```json\n\n"`). `grammar::json_message`.

### Changed (O5's two open gaps, closed)
- **`num_ctx` as Ollama applies it** (read from Ollama v0.13.3's `server/prompt.go`): under a request's or a layer's
  `num_ctx`, a conversation that does not fit loses its oldest messages first (`native::fit_messages`, Ollama's loop
  step for step); what still does not fit is refused with the reason, since Ollama would cut tokens out of the middle
  of the prompt. Tokens past `num_ctx` during an answer still differ: Ollama shifts its cache, bankML does not.
- **`/api/ps` names the derived model** (its digest, `parent_model`, the request's context), as Ollama names a runner
  after the model that loaded it.
- A derived model's name wins over the registry's suffix-less alias, and `FROM` resolves that alias, so promote.py's
  re-create in place (`FROM mindx-gen39` as `mindx-gen39`) works; a conversion never overwrites an existing pin.

### Found
- **The converter's bytes depend on the directory's name**, for llama.cpp and bankML alike: `general.name` comes from
  it. gen39 converted from a directory named `merged` is `bb41f62d…`, from `mindx-gen39` the pin `6b64c748…`.
- gen39's `tokenizer.json` lost SmolLM2's `Digits` pre-tokenizer when transformers 5.8 re-saved it; llama.cpp's
  `chkhsh` still names it `smollm` (the vocabulary has no multi-digit tokens), and bankML accepts both shapes.

### Not in this release
- `ADAPTER` (LoRA) in a Modelfile: merge first, then `FROM` the merged directory. Tool calls (O6).
- **Next: O2, the penalty samplers.** mindXtrain's bankml backend ([proposed](https://huggingface.co/PYTHAI/mindXtrain/discussions/1)) found that `mindx-gen39`
  degenerates into repetition (`,,,,`, `?||`) without a repetition penalty; its imprint gate decodes with
  `repetition_penalty 1.3` and `no_repeat_ngram_size 3`, which bankML refuses until each has an oracle.

## 0.3.4 — 2026-10-02 — mindX's own model, natively: the Llama graph, F16, tied embeddings (O4)

**mindX serves its own trained lineage, `mindx-genN`, through Ollama. Generation 39 (mindXtrain39, the last accepted
generation: SmolLM2-135M + its LoRA, merged, F16) is now answered by bankML's own forward pass, token-identical to
llama-server b11192 — greedy, seeded, on `/v1`, on Ollama's `/api`, through the C API and in JSON mode. So are
SmolLM2-135M-Instruct and Bonsai-1.7B, which was refused until now only because its embeddings are tied.** Phase O4 of
docs/OLLAMA.md, with the F16 part of O3. Record: `testing/results/0.3.4.txt`.

### What llama.cpp does, read from its source (b11192, `171e8846b`)
- **F16 weights take two paths, chosen by the product's shape** (`ggml_compute_forward_mul_mat`). The activation is
  rounded to f16 first (F16's `vec_dot_type`). With one column (a decode step, the logits, a one-token micro-batch)
  `llamafile_sgemm` refuses (`n < 2`) and every element is `ggml_vec_dot_f16`: four 8-lane FMA accumulators over
  32-element steps, then a fixed reduction, any tail summed in double. With two or more columns `llamafile_sgemm`
  runs `tinyBLAS<8, __m256>`: each element one 8-lane FMA chain over the row and its own `hsum` — **other bits** —
  whenever the weight has a multiple of 4 rows and the row a multiple of 8. The two differ, so a prompt's
  micro-batch and a decode step must each take their own.
- **The last layer of a prompt only computes the rows that are output** (`inp_out_ids`): llama-server asks for the
  last token's logits, so the last layer's feed-forward block is a one-column product there. For the quantized types
  that changed nothing; for F16 it decides the bits.
- **Tied embeddings**: with no `output.weight`, the loader duplicates `token_embd` (`TENSOR_DUPLICATED`) and the
  logits are that table times the final row.
- **The Llama graph** (`llm_build_llama`): no Q/K norms, RoPE in NORM mode (`rotate_pairs` on adjacent pairs, the same
  contracted arithmetic as NEOX), no YaRN (`ext_factor` 0, `attn_factor` 1), GQA from the header.
- **SmolLM2's pre-tokenizer** (`smollm`) is two passes: every digit cut out alone (a `std::regex` over the collapsed
  text), then GPT-2's pattern inside each piece, where the end of a piece is the end of the text. Its vocabulary has
  no token for 21 bytes; llama-vocab drops them.
- **llama-vocab overrides the file's token types by text**: every end-of-generation name and the fill-in-the-middle
  token of each role become CONTROL. In the Qwen3 vocabulary `</s>` (token 128247) is NORMAL in the file and CONTROL
  in llama.cpp, so the text `</s>` is cut out as one special token.

### Added
- **`bankML/f16.rs`**: ggml's two F16 products, each a scalar definition and an AVX2 + FMA + F16C path with the same
  bits (tinyBLAS's own 4 × 3 register blocking for prompts); `mat_mul_par` chooses the path as ggml does.
- **`forward::plan`**: from the header alone, the architecture (`qwen3`, `llama`), the weight type (Q1_0, Q2_0_g64,
  F16), the logits matrix (tied or not), and a refusal with the reason for everything else: other architectures,
  other types, mixed types, biases, fused QKV, rope factors, experts, Q/K norms on Llama, RoPE scaling on Llama,
  partial RoPE, a head width that is not a multiple of 32.
- **The Llama graph** in `forward.rs` (NORM RoPE, no Q/K norms) and tied embeddings for both architectures.
- **The `smollm` pre-tokenizer** in `tokenizer.rs`; a byte without a token is dropped as llama.cpp drops it.
- **Chat templates per model** (`chat::TEMPLATES`, each pinned by the sha256 of its text): Bonsai / Qwen3,
  SmolLM2-Instruct's (ChatML with a default system message), and the plain ChatML `mindx-genN` carries. Ollama's
  persona `SYSTEM` for `mindx-genN` lives in its Modelfile, not the GGUF: here it is the caller's system message
  (`bankml create` is O5).
- **JSON mode on ChatML templates**: llama-server builds another grammar there (its root opens with
  `<|im_start|>assistant\n` and has no `<think>` block); `grammar::json_object_grammar` picks the template's, checked
  against the server's own report on every recorded request.
- **The models, pinned** (`sAGI/models.py` `CONVERTED`, `pin_converted`): no F16 GGUF is published for either, so each
  is converted from its pinned safetensors by llama.cpp b11192's own `convert_hf_to_gguf.py --outtype f16`
  (unmodified). The conversion is reproducible (twice, the same sha256), and every tensor was checked against the
  safetensors rounded to f16. The FORK.json records the source's LFS sha256, the revision and the tools.
  - `SmolLM2-135M-Instruct-F16.gguf` `e9aba089…5a222` from HuggingFaceTB/SmolLM2-135M-Instruct@`12fd25f7`
    (`model.safetensors` `5af571cb…`), Apache-2.0.
  - `mindx-gen39-F16.gguf` `6b64c748…58266` from the dataset PYTHAI/mindXascension@`4bd31b9d`
    (`weights/gen39/ollama_push/merged/model.safetensors` `19b62829…`), Apache-2.0 (the dataset card; the base is
    SmolLM2-135M).
- **Registry aliases**: a name without its weight-type suffix resolves when one pin has that base, so Ollama's tag
  `mindx-gen39` reaches `mindx-gen39-f16`.
- **Oracles, in the gate:**
  - `oracle_ggml_b11192_f16` (`testing/f16_oracle.py`): the shipped library's `mul_mat` on SmolLM2's real F16 matrices
    and on synthetic shapes for the tails and the fallbacks — 211 of 211 tensors widened bit-exact, **552,268 of
    552,268 elements** bit-exact (61 products through tinyBLAS, 26 through `ggml_vec_dot_f16`).
  - `oracle_forward_model_bonsai_1_7b`, `oracle_forward_model_llama_f16`: the whole model (every layer, `result_norm`,
    logits) bit-exact — Bonsai-1.7B **840 of 840** rows, SmolLM2-135M-Instruct and mindx-gen39 **800 of 800** each.
  - `oracle_tokenizer_smollm` **4,346 of 4,346**; `oracle_chat_template_chatml` **317 of 317** on each template.
  - `oracle_llama_server_llama_f16`, `oracle_llama_server_bonsai_1_7b`: llama-server's tokens — greedy on short, long
    (the tiled kernel, tinyBLAS over the micro-batch) and deep prompts (the split-KV kernel), and seeded sampling:
    SmolLM2 6 / 6 / 3 and **40 of 40**; mindx-gen39 6 / 6 / 3 and **40 of 40**; Bonsai-1.7B 6 and **40 of 40**.
  - `oracle_native_serve_o4`: the Savante conversations (**9 of 9** turns each) and JSON mode (**23 of 23** answers
    each) on the three models.
  - Live: `serve_oracle.py --bankml` and `json_oracle.py --bankml` on each (mindx-gen39 asked for as `mindx-gen39`),
    and the C API's `bankml_chat` against each model's llama-server record.

### Fixed
- **The tokenizer took `</s>` in the Qwen3 vocabulary as plain text.** llama.cpp makes it a CONTROL token by its name,
  so with special tokens parsed it is one token there and three here. The old oracle corpus never contained it; the
  corpus grew with the docs and the oracle caught it (2 of 4,346 cases). bankML now applies llama-vocab's type
  overrides; a vocabulary where they are ambiguous, or whose header sets FIM ids, is refused.
- A vocabulary whose merges name a string outside it is refused: llama.cpp merges by text, bankML by id.

### Changed (faster, the same bits)
- **Attention** (every model): the reference kernel's f16 dot, scale and accumulate run on F16C + FMA; the tiled
  kernel is compiled with FMA and F16C and widens each 64-cell tile once per block of 16 rows; heads (and rows) run on
  the pool. Each head is still computed whole by one thread in llama.cpp's order.
- Tensor lookups by name are a hash map, and the norm vectors are read once at load.
- **The 8B models gained most**: the same gate oracles ran in about half the wall time (Bonsai-8B conversations 400 →
  192 s, deep greedy 756 → 320 s, JSON mode 901 → 541 s; ternary JSON 454 → 242 s), with the matmul budget unchanged
  — the old reference attention converted every f16 value in software. Gate wall times, not a controlled benchmark.
- **Speed, SmolLM2-135M-Instruct F16, laptop (Ryzen 3 3200U), 3 threads each, paired:** decode 38.0–38.9 tok/s vs
  llama-server's 40.4–42.7 (about 0.93×); an 858-token prompt in 8.6–9.6 s vs 7.8–8.6 s, then decode at ~900 cells
  27.2–30.1 vs 20.2–29.9 tok/s. Bonsai-1.7B decode 8.4–8.6 vs 10.6–10.9 tok/s (the known 1-bit decode gap, O8).
  docs/PERFORMANCE.md has the table and a rejected experiment.

### Not in this release (refused, with the reason)
- **Q8_0** (Qwen3-0.6B, mindX's `qwen3:0.6b` pin): from the source the kernels look cheap (ggml's AVX2
  `vec_dot_q8_0_q8_0` and tinyBLAS's Q8_0 path appear to share their per-block arithmetic; not built, not measured),
  but that file's template is Qwen's own, which renders 4 of 317 oracle conversations differently from the Bonsai
  template (reasoning rules). It stays refused ("weights are Q8_0 … O3") until its template has an oracle.
- BF16, Q4_K and other architectures: refused by `forward::plan`, as before.

## 0.3.3 — 2026-10-02 — JSON mode, token-identical to llama-server

**mindX's most common request that bankML refused, `format: "json"`, is now answered natively. The answer has the
same tokens llama-server b11192 gives under the same request, greedy and seeded.** O6's first cut (docs/OLLAMA.md).
Record: `testing/results/0.3.3.txt`.

### What llama-server does, read from its source (b11192, `171e8846b`)
- **The grammar is not `grammars/json.gbnf`.** On the jinja chat path (Savante's flags), `response_format: {"type":
  "json_object"}` becomes the schema `{"type": "object"}`. The chat layer turns the schema into a PEG parser
  (`common/chat-auto-parser-generator.cpp`), and that parser into GBNF. The GBNF's root begins with the template's
  generation prompt, then allows an optional ```` ```json ```` fence around the object.
- **The grammar is prefilled.** It is typed as a tool-call grammar, so `common_sampler_init` *accepts the generation
  prompt's 7 tokens* into it before the first draw (`common_grammar_needs_prefill`).
- **The sampler draws, then checks, then redraws.** `common_sampler_sample` (`grammar_first = false`) runs the whole
  chain on the raw logits. If the token it picks passes the grammar, it stands. If not, the logits are reset, every
  rejected token goes to −∞, and the chain runs again: **a second draw from the same `mt19937`**.
- **The end set is wider.** An end-of-generation token passes only when the grammar is complete, and llama-vocab's end
  set on Qwen3 is 6 tokens: `<|endoftext|>`, `<|im_end|>`, `<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>` and token
  128247 `</s>`. One recorded answer ends on `<|file_sep|>`.
- **The content is the value alone.** The answer's `content` is the JSON value, without the fence or the space around it.

### Added
- **`bankML/grammar.rs`: llama.cpp's grammar engine, ported** (MIT, attributed with its notice; no crates). It holds:
  - the GBNF parser (literals, classes, `.`, token terminals `<…>` `!<…>` `<[id]>`, groups, `* + ? {m,n}`, comments);
  - the reference and left-recursion checks;
  - the pushdown stacks and `advance_stack`;
  - `reject_candidates` over code points, with partial UTF-8 carried across tokens;
  - accept and apply, with the end rule.

  It also carries the server's JSON-mode grammar (a constant the oracle checks against the server's own report) and
  its prefill, the request resolution of `oaicompat_chat_params_parse`, and the content rule.
- **`Sampler::sample_constrained`**: the draw, the check and the masked redraw, in the server's order.
- **Surfaces.** All of them use one engine path (`Native::grammar`, `Native::complete`):
  - `/v1/chat/completions`: `response_format` (`json_object`, `json_schema` with schema `{}` or
    `{"type": "object"}`), the top-level `json_schema`, and `grammar` (any GBNF; a user grammar, not prefilled);
  - Ollama's `/api/chat` and `/api/generate` with `format: "json"`;
  - the C API's `bankml_chat`;
  - `bankml generate --json`.

  A stream sends the content's growth.
- **Oracles, in the gate:**
  - `oracle_grammar_masks` (`testing/grammar_oracle.cpp` + `.py`) drives libllama b11192's public grammar sampler on
    the pinned vocabulary. All 151,669 token pieces and the end set are identical. Then 196 of 196 runs over 12
    grammars are identical: llama-server's JSON grammar with its prefill, every grammar in llama.cpp's `grammars/`,
    and three written for token terminals, edges and UTF-8. Each input goes in tokenized and byte by byte. That is
    **1,645 of 1,645 whole-vocabulary masks and 116 of 116 rejection points**.
  - `oracle_json_mode` and `oracle_json_mode_ternary` (`testing/json_oracle.py --record`) replay llama-server's
    answers under `response_format: json_object`. The prompts invite JSON or tempt prose and code; one is a nested
    object and one is unicode. They run greedy and seeded at 0.7, 1.0 and 1.3, plus two user grammars, each from an
    empty cache. The results: Bonsai-8B Q1_0 **23 of 23** (860 tokens, 152 of them redrawn under the mask), and
    Ternary-Bonsai-8B **13 of 13** (534 tokens, 49 redrawn). The same token ids, end token included, and the same raw text, content, finish
    and counts.
  - `json_oracle_live` (`testing/json_oracle.py --bankml`) runs a live `serve --native` through `/v1` (streamed
    once), `grammar`, and `/api/chat` with `format: "json"`: **8 of 8**.

### Changed
- **The end of an answer** is llama-vocab's whole end set (6 tokens), not `<|im_end|>` and `<|endoftext|>` alone,
  with or without a grammar. Before this, an answer that sampled `<|file_sep|>` would have run on where llama-server
  stops.
- **`response_format` on `serve --native`'s `/v1` was silently ignored before** (an unconstrained answer; the proxy mode always passed it to llama-server). It is now honoured, or
  refused with the reason.
- `serve::Json` derives `Clone`. `native::Done` carries the generated token ids, the time spent in the grammar and
  the number of redraws.

### Still refused, with the reason
- A JSON schema other than `{}` or `{"type": "object"}`, in `response_format`, `json_schema` or Ollama's `format`:
  llama.cpp's `json_schema_to_grammar` and its chat-parser wrapping are not ported (O6, next).
- `format` with `raw: true`, because JSON mode's grammar starts with the template's generation prompt;
  `response_format` or `json_schema` together with `grammar`; tools.

### Measured
- **Grammar cost** (PERFORMANCE.md, "JSON mode's cost"):
  - one whole-vocabulary mask over 151,669 tokens costs a median of **24.2 ms** (p90 47.0, max 101.0) across 1,645
    masks;
  - spread over real answers, the grammar costs **3.90 ms per token** on the 1-bit model (152 of 860 tokens
    redrawn) and **3.95 ms** on the ternary model (49 of 534). That is about 1 % of a decode step.
- **The rest of the gate, against 0.3.2.** No kernel changed, and every bit-exact oracle passed again: tokenizer
  4,258/4,258, template 317/317, whole model 1,064/1,064 for both models, greedy 6/6, 6/6, 6/6 long and 3/3 deep,
  sampling 40/40, native serve 9/9, the Ollama shape 9/9, the C API 9/9 + 9/9, and printf 47/47.
  - The decode budgets read faster than in 0.3.2's gate because the laptop was quieter, not because of the code.
    1-bit: bankML 0.641 s/token at 1 thread (0.3.2: 0.677) and 0.389 s at 3 (0.3.2: 0.787, under load). Ternary:
    0.436 s at 1 thread (0.477) and 0.279 s at 3 (0.394), **9.62×** and 8.36× ahead of ggml b11192.

## 0.3.2 — 2026-10-01 — a C API, on Rust 1.99

**bankML is now also a C library, `libbankml`, with one hand-written header. A program that embeds it gets the same
verification as `bankml serve` and the same answers and receipts as `serve --native`. Its printf-style log is a
C-variadic function defined in Rust, which Rust 1.99 stabilized.** The toolchain moves to Rust 1.99. The gate's
bit-exact oracles are what show that the compiler upgrade changed no bits. Record: `testing/results/0.3.2.txt`.

### Added
- **`capi/` (package `bankml-capi`): `libbankml.so` and `libbankml.a`, declared in `capi/include/bankml.h`, documented in
  [docs/CAPI.md](docs/CAPI.md).** It is a second workspace member, and its only dependency is the bankml crate, by
  path. The workspace still has no external crate, and the licence is `MIT OR Apache-2.0`.
  - `bankml_open(model, fork_json, n_ctx, &err)` runs `serve`'s verification: the guard, the sha256 pin, the
    file-identity check across the hash, and the native check. A refusal returns NULL with `serve`'s reason, for
    example Bonsai-1.7B's tied embeddings, or a file its FORK.json does not pin.
  - `bankml_chat(h, request_json, cb, user, &result_json)` takes the OpenAI / llama-server chat request and refuses
    what `serve --native` refuses. It streams whole UTF-8 pieces to `cb` (return 0 to stop), and returns the
    `/v1/chat/completions` object with `usage`, `timings.cache_n` and the same `bankml_receipt`. The request also
    hits the slot's prompt cache, as `-np 1` does.
  - `bankml_close`, `bankml_free` and `bankml_version`.
  - The rules: error codes, never an unwind into C, NULL-safe; one slot per handle, so calls serialize; returned
    strings are freed with `bankml_free`.
- **`bankml_set_log` and `bankml_log(level, fmt, ...)`, defined in Rust with `mut args: ...`.**
  - The formatter (`capi/src/printf.rs`) reads each argument with `VaList::next_arg::<T>()` at the C type its
    conversion names.
  - It supports `%d %i %u %x %X` (with `hh h l ll z`), `%f %F %lf`, `%c`, `%s`, `%p` and `%%`, with the flags
    `- + space # 0` and a width and a precision (numbers or `*`).
  - Anything else is written as `%<unsupported:SPEC>` and never guessed. After an unknown conversion no argument is
    read (`%<skipped:SPEC>`), and `%n` never writes.
  - The library's own messages (a model verified, the GPU) go to the same sink, through a small hook in the core:
    `bankml::log` and `bankml::set_log_sink`. It writes to stderr, as before, unless an embedder installs a sink.
- **The C API's oracles, in the gate** (`testing/capi/`, compiled with the system `cc`):
  - **printf oracle:** `bankml_log` against libc `snprintf`, byte for byte, 47 of 47 formats identical, 17 of 17 unsupported cases marked, `%n` never written;
  - **capi chat oracle:** `bankml_chat` is identical to `serve --native` on 9 of 9 Ternary-Bonsai-8B turns (text, streamed pieces, counts, cache reuse, finish, and the receipt's response, request and model sha256), and to llama-server b11192's record on 9 of 9 Bonsai-8B Q1_0 turns. Bonsai-1.7B and an unpinned file are refused with `serve`'s reasons, and the library's own log reaches the installed sink;
  - the formatter's Rust unit tests: a typed-queue model, and the variadic function called from Rust against
    `snprintf`.

### Changed
- **Rust 1.99.0, pinned** in `rust-toolchain.toml`, with `rust-version = "1.99"` in both manifests. rustup fetches it;
  the machine's default toolchain is untouched. `install.sh` checks for 1.99.
  - 1.99's clippy adds one lint, `chunks_exact_to_as_chunks`, at 29 sites: the kernels' block loops, the quantizer,
    the dequantizers, SHA-256 and the GPU packers. Each was rewritten to `as_chunks::<N>()`, which visits the same
    elements in the same order.
  - **Rust 1.99 passed every bit-exact oracle in the gate**: tokenizer 4,258 of 4,258; chat template 317 of 317; the forward-pass
    steps (300/300, 140/140, 112/112, 150/150, 14/14 rows, SwiGLU 24,600/24,600 values); the whole model 1,064 of
    1,064 rows for both the 1-bit and the ternary model; greedy 6/6, 6/6, 6/6 long and 3/3 deep; seeded sampling 40 of
    40; native serve 9 of 9; the Ollama shape 9 of 9; the GPU kernels; and 8,188,239,872 weights with 762 of 762 dot
    products bit-exact for each 8B model (788 of 788 for the 1.7B).
- **`serve --native`'s chat path is factored, not changed.**
  - `serve::NativeChat` (parse and run), `serve::completion_json` and a public `serve::Tally` are what both
    `/v1/chat/completions` and `bankml_chat` call, so the receipt is made by one piece of code.
  - The gate's `serve_oracle --bankml` passed on the refactored path.
- `testing/spdx_check.py` now covers C sources and headers, with `upstream/` held to llama.cpp's `MIT`.

### Measured (the 0.3.2 gate against 0.3.1's, same laptop, Ryzen 3 3200U)
Every oracle passed on 1.99, so the bits did not move. The speed rows did move, and the cause is *not yet known*:
this run shared the 4-thread laptop with other work. The load average was 2.2–3.0 at its end, another session's
`cargo build` was running right after it, and swap was full. Under that load the multi-threaded rows moved for
**ggml and bankml alike**, so the per-pass ratio is the number to read.

| row | 0.3.1 (Rust 1.95) | 0.3.2 (Rust 1.99) | reading |
|---|---|---|---|
| Q1_0 decode budget, 3 threads, bankml median | 0.415 s/token, **0.98×** ggml | 0.787 s/token (min 0.522), **0.73×** ggml (0.571, min 0.473) | beyond noise in absolute terms, but ggml's own median moved 0.406 → 0.571; contention |
| Q1_0 decode budget, 1 thread | 0.682 s, 1.02× | 0.677 s, 1.05× | unchanged |
| Q2_0 decode budget, 3 threads | 0.274 s/token, **9.23×** | 0.394 s/token (min 0.327), **7.99×** | ggml 2.612 → 3.603 s as well; contention |
| Q2_0 decode budget, 1 thread | 0.491 s, 9.63× | 0.477 s, 10.25× | unchanged or slightly better |
| Q2_0 prefill 1×4 tile | **13.0×** | **13.25×** | unchanged |
| Q1_0 GEMV `mat_vec` / prefill Q8Act tile | 1.008× / 1.233× | 1.104× / 1.308× | unchanged or slightly better |

An interleaved A/B after the gate, with the same tests built by 1.95 and by 1.99 and run alternately (two rounds),
ran at a load average of 4–5. ggml's own 1-thread token ranged from 0.73 s to 1.88 s across those runs. That is too
noisy to separate the compilers, and it is recorded here as inconclusive. **The deciding experiment** is the same
alternating A/B under `testing/pinned.sh` (pinned cores, a RAM cap) on a quiet machine. Until it runs, 0.3.2 claims
no speed change in either direction.

## 0.3.1 — 2026-10-01 — Ollama's API, natively

**`bankml serve --native` speaks Ollama's API over bankML's own forward pass, and can be asked for any pinned model
by name. It is native only: what the verified forward pass cannot do is refused with a reason, never proxied to
llama-server or Ollama.** The roadmap is [docs/OLLAMA.md](docs/OLLAMA.md): what mindX asks of Ollama, with
file:line evidence, against what bankML does, and the O1–O8 track folded into docs/TODO.md. Record:
`testing/results/0.3.1.txt`.

### Added
- **`bankML/ollama.rs`, Ollama's API (phase O1).**
  - **Listing:** `GET /api/version`. `GET /api/tags` lists every pinned model: the digest is its pinned sha256,
    the details come from the GGUF header, and `"bankml": {"native", "reason"}` says whether bankML plays it.
  - **State and metadata:** `GET /api/ps` (the resident model and its `expires_at`); `POST /api/show` (a
    read-only modelfile, the sampling defaults, the template, `model_info` from the header).
  - **Answers:** `POST /api/chat` and `POST /api/generate`: `system`, and `raw: true` for direct tokenization.
  - **Streaming:** NDJSON, Ollama's default. The final object carries `done_reason`, the durations in
    nanoseconds, `prompt_eval_count`, `eval_count`, `bankml_cache_n` and `bankml_receipt`.
  - **Options:** `temperature`, `top_k`, `top_p`, `min_p`, `seed`, `num_predict` and `stop` are honoured.
    `num_ctx` is accepted up to the served context. The penalties go through the engine's own refusal. Resource
    options are ignored, because they do not change the answer. Anything else is refused.
  - **Refused (400, with the reason):** `format` (O6), `tools` and `tool_calls` (O6), `images`, `suffix`,
    `template`, `context`, `think: true`. Also `/api/embed` (O7), `/api/create` (O5), `/api/pull`,
    `DELETE /api/delete`, `/api/copy` and `/api/push`.
- **A model registry, and residency** (`native.rs`: `Registry`, `Residency`).
  - `--registry [DIR]` names every GGUF pinned in the forks directory (default `$BANKML_FORKS`, else
    `~/.local/share/bankml/forks`). A name is the file's stem, lower-cased; `:latest` and the file name are
    accepted as aliases.
  - **One resident model.** A switch drops the old model before the new one is read, then runs the full `verify`
    (the guard, then the sha256 pin) exactly as the start does.
  - **`keep_alive`** as Ollama reads it (`"5m"`, `"1h"`, seconds, `0`, negative), with `--keep-alive` for the
    `/api` default (`5m`). A reaper thread drops an idle model and its memory map when its time runs out.
  - **Load or unload only:** an empty prompt with a `keep_alive` answers `done_reason` `load` or `unload`, as
    mindX's coach uses it.
  - **The file-identity check now lives with the resident model.** A changed file is unloaded and refused; the
    next request verifies it again, with no restart.
- **`Done.prompt_ns` and `Done.eval_ns`.** The prefill and the generation are timed inside `complete()`.
- **The Ollama-shape oracle** (`testing/serve_oracle.py --bankml`, in the release gate). The 0.3.0 conversations
  go three ways, each from an empty slot:
  - through `/api/chat` (the first conversation streamed as NDJSON);
  - through `/v1/chat/completions`;
  - through `/v1/chat/completions` again, after unload and a reload that verifies the file again.

  All of them must equal each other and llama-server b11192's record, turn by turn.

### Changed
- **`/v1/chat/completions` now honours `stop`.** 0.3.0 ignored it silently. It also resolves `model` through the
  registry: absent means the startup model, and an unknown name now answers 404 where 0.3.0 ignored the field.
  The OpenAI and llama-server endpoints keep the model resident, as 0.3.0 did.
- **A startup model the forward pass cannot play is refused at start, with the reason.** 0.3.0 started and then
  failed every request: Bonsai-1.7B has tied embeddings (no `output.weight`), which is phase O4.
- `GET /bankml` (native) adds `resident` and `models`. Its `verified` is the most recent verification.
- **The way back to the Interaction tab.**
  - Savante's card, and every derived agent's card, now has a bar pinned to its top: **← back to Interaction**. It
    stays visible however far the card is scrolled, and Esc closes the card. Before, the only close control was a ✕
    that scrolled away with the card's content.
  - Every other tab shows a **← Interaction** pill in the bottom-left corner. It goes straight back to the chat, with
    the cursor in the question field.
  - *use this agent* on the Agents tab now returns to the Interaction tab with that agent in use.
- **The Interaction tab is only the conversation.**
  - The chat, the question field, the clock and the portrait. The settings (`.prompt`, `.memory`, max tokens,
    temperature), the carrier, and the CPU and RAM controls moved to a new **Admin** tab; the engine choice and
    n-gram speculation are under its *advanced*.
  - An answer is shown alone. Its clock, receipt and prompt provenance are kept, but not in the chat: they are on
    the Admin tab under *the last answer*, in `.history` (a new `trail` field) and on the Responses tab. Earlier
    turns reload as their answers alone.

## 0.3.0 — 2026-09-29 — milestone: Savante answered by bankML's own forward pass

**`bankml serve --native` answers Savante from bankML's own forward pass. Whole conversations are identical to
llama-server's turn by turn, and about 8× faster on the ternary model.** Record: `testing/results/0.3.0.txt`.

### Added
- **`bankML/native.rs`, the native engine.**
  - One conversation slot that reuses the KV cache exactly as llama-server's prompt cache does: it keeps the
    longest common prefix, less one token when the whole prompt is cached, truncates there, and computes the rest
    in micro-batches of 512, so every row takes the kernel llama.cpp's row takes. `KvCache::truncate` was added
    for this.
  - Sampling uses the request's parameters over the model's GGUF defaults. Samplers it does not reproduce are
    refused with a reason.
  - It streams whole UTF-8 characters and counts the end-of-turn token among the completion tokens, as
    llama-server does.
- **`bankml serve --native`.**
  - The same gateway as before: guard, sha256 pin, loopback rules, and the file's identity checked before every
    answer. It also listens on the engine address (`--upstream`) and answers llama-server's endpoints there
    (`/health`, `/props`, `/tokenize`, `/apply-template`, `/v1/chat/completions`, `/v1/models`), so a client
    written for llama-server, Savante included, reaches it unchanged.
  - Streamed and non-streamed OpenAI-style answers, each with `usage`, `timings.cache_n` and the receipt. The
    receipt's `engine` names bankML's own forward pass and the GPU when one works in it.
- **Savante's engine setting** (Models → Resources: `auto`, `native` or `llama.cpp`; `models.py native_for`).
  `auto` uses bankML for the ternary files, where it is about 8× llama-server, and llama-server for the rest, where
  llama-server is still faster. The importer starts the carrier accordingly.
- **The milestone oracle** (`testing/serve_oracle.py` → `oracle_native_serve`, in the gate). It runs three
  Savante-style conversations of three turns each, sampled at Savante's temperature 0.3 with fixed seeds, through
  llama-server b11192's own `/v1/chat/completions` in one fresh server process, so its prompt cache carries each
  conversation. bankML's engine gives **the same answer text, prompt and completion counts and prompt-cache reuse
  on 9 of 9 turns**, with the GPU working. Along the way the oracle caught that llama-server counts the
  end-of-turn token among completion tokens.

### Checked end to end
- `python3 sAGI/models.py use Ternary-Bonsai-8B-Q2_0_g64.gguf` starts the native carrier on its own. The model is
  verified, and it answers with llama-server's greedy answer.
- Savante's own client code, unchanged, streams from it, reads its receipt, counts tokens through its `/tokenize`
  and reads its context from `/props`.

### Limits
- The Qwen3 1-bit and ternary models; one conversation slot; no slot save or restore yet (Savante then prefills
  as usual); top-k, top-p, min-p and temperature.
- On the 1-bit model llama-server is still faster: 2.8 tokens/s against 1.9–2.0.

## 0.2.14 — 2026-09-29

**The GPU works inside the forward pass without changing a bit, and the FMA a driver would not fuse.** Record:
`testing/results/0.2.14.txt`.

### Found
- **The Vega 3's driver does not fuse `Fma`.** On real layer weights the 0.2.13 kernels differed from the CPU by
  about one ulp on 70 % of rows, even though they had passed on random data. The card agreed on every row with the
  CPU recipe whose outer `fma(d0, ab, acc)` is an unfused multiply-then-add; the inner `fma(d1, s, ab)` never
  showed, because `d1·s` is exact in f32. Decorating the `Fma` `NoContraction` did not change it. The random
  verification data was too kind: its products were mostly exact.

### Fixed
- **An exact FMA from plain arithmetic** (`spirv.rs` `fma_exact`), correct on every GPU whether its driver fuses
  or not:
  - `d0` comes from f16 and has 11 significant bits, so splitting the other operand into two 12-bit halves makes
    both partial products exact;
  - Knuth's TwoSum keeps every rounding error;
  - Boldo and Melquiond's `RN(th + RO(tl + ul))` (rounding to odd, 2008) gives the FMA's single rounding.

  Both kernels are now bit-exact on real weights: 0 of 4,096 and 0 of 12,288 rows differ.
- **The on-card oracle** (`kernels::verify_q1_0`, `bankml gpu --verify`) adds a layer-shaped regime: weight
  scales and activation magnitudes that vary per block, and no −128. With the driver's `Fma` put back, it now
  refuses the card: "row 1 differs … bankml will not use this card".

### Added
- **`gpu/worker.rs`: the GPU as a worker beside the CPU's threads.** When a card is selected and passes the
  on-card oracle, each 1-bit matrix–vector product gives the card the first share of its rows (asynchronously)
  while the CPU pool computes the rest, then copies the card's rows in.
  - The share is calibrated when the card is opened: the card rate divided by the combined rate, 26–35 % on the
    Vega 3. `BANKML_GPU_SHARE` overrides it and `BANKML_GPU=off` disables the worker.
  - Only the card's share of each matrix is copied to it (repacked exactly), on first use.
  - `Gpu::submit` and `Gpu::wait` split a dispatch from its fence.
- **Every token oracle now runs with the GPU working, and passes.** The whole 1-bit model is 1,064 of 1,064 rows
  bit-exact with the Vega 3 computing 26 % of every matrix's rows, and the greedy and sampling checks go through
  the same path.

### Speed, honestly
- Decode on this laptop is **unchanged within noise**: 1.93–2.00 tokens/s with the card against 1.96–1.97 without.
  The card is worth about one CPU core here, and each of the 253 matrix calls per token pays one submit and wait
  on it, which cancels the gain. Batching the submissions that share an input (Q, K, V; gate and up) into one is
  the next step. A discrete card gains directly.

## 0.2.13 — 2026-09-29

**The first GPU kernels, bit-exact on the card; mindXtrain begins in Rust; bankML branding.** Record:
`testing/results/0.2.13.txt`.

### Added
- **GPU kernels, and bankml's own SPIR-V.**
  - `gpu/spirv.rs` is a small SPIR-V assembler, so bankml needs no shader compiler at build time or run time. Every
    float multiply and add it emits is decorated `NoContraction`, and every fused step is an explicit `Fma`, so a
    driver cannot change the float order.
  - `gpu/kernels.rs` has two Q1_0 matrix–vector kernels that follow the CPU kernel's recipe step by step:
    `q1_0_mat_vec` uses one invocation per row, and `q1_0_mat_vec8` uses eight per row, one per accumulation lane,
    with the lanes summed in the CPU's order through workgroup memory. Weights are repacked once, exactly
    (scales f16→f32, bits as words).
  - `gpu/compute.rs` is the Vulkan compute runtime (device, buffers, pipelines, dispatch), again through the
    run-time loader and structs declared in-crate.
  - **On the Radeon Vega 3, both kernels are bit-exact against the CPU kernel**, which matches ggml, at every shape
    tried: 64×128 to 12288×4096, with −128 quants included.
  - **`bankml gpu --verify`** runs that check on every card found. A card that fails is named and never used; this
    is how bankml puts a card to work only after proving it gives the same bits.
  - Speed, honestly: on this integrated GPU the eight-lane kernel matches one CPU thread (1.7–2.0 ms per
    4096×4096). It becomes an extra worker next to the CPU's threads when rows are shared (next step). On a
    discrete card the same kernel has far more hardware.
- **mindXtrain in Rust (`bankML/train/`).** A stage registry for the proof loop (author → imprint → probe → score
  → classroom → boardroom), built one verified stage at a time against mindXtrain's own Python
  (`testing/train_oracle.py`, run with mindXtrain's interpreter):
  - `train/script.rs`, the author stage: persona + exchanges → chat-JSONL script. **84 of 84 scripts
    byte-identical** to `scripts.py`, over every persona in mindX and cryptoAGI plus edge cases.
  - `train/imprint.rs`, the score stage (lexical path): the recall gate's voice score, shift and verdict. **3,000
    of 3,000 reports identical** to `score_imprint`.
  - Next come the probe on bankml's forward pass (it needs the Llama architecture, since mindXtrain imprints
    SmolLM2, and adapter loading), the verdicts, and LoRA training on the CPU.
- **The Savante UI is branded bankML** ("bankML · Savante — verified low-bit inference on this computer") and
  carries the DeltaVerse **$** as its tab icon (deltaverse.pythai.net/favicon.ico, MIT; served by both the interact
  and view pages).
- **Install docs** now show `chmod +x install.sh` before `./install.sh`, for copies that lost the executable bit.

## 0.2.12 — 2026-09-29

**Batched prefill, and the video-card component.** Record: `testing/results/0.2.12.txt`.

### Added
- **Batched prefill** (`bankML/forward.rs`). A micro-batch now goes through each layer together:
  - every matmul is one matrix–matrix product over the micro-batch's rows (`Weights::mm`, `Weights::quantize_rows`,
    through `q1_0::mat_mul_act_par` and `q2_0::mat_mul_par`);
  - the micro-batch's K and V enter the cache before its attention, as llama.cpp writes them;
  - each row attends over the cells up to its own position (`Weights::attend`).

  Every element of those products has the bits of the per-pair dot, so the result is the token-by-token result.
  Every oracle that goes through `prefill` checks this: greedy, long, deep and sampling.
- **`bankML/gpu/`: the video-card component.**
  - `gpu/mod.rs` is the registry. Each backend is a module with one discovery function, listed in `BACKENDS`; a
    new backend (CUDA, ROCm, Metal) is a new module and one line.
  - Devices are described by merging the backend's view with the kernel's (`/sys/class/drm`: driver, VRAM, GTT,
    PCI address, NUMA node).
  - `selected()` uses every real GPU found, discrete cards first and the largest first. Software renderers (Mesa
    llvmpipe) are refused. `BANKML_GPU=off` turns the component off, and `BANKML_GPU=0,2` picks cards.
  - `gpu/vulkan.rs` opens `libvulkan.so.1` at run time: no crate and no link-time dependency, the entry points from
    `vkGetInstanceProcAddr`, the few C structs declared from the Vulkan headers. It covers AMD, NVIDIA, Intel, Arm
    and Qualcomm GPUs through one API.
  - `gpu/hf.rs`, the first remote backend, lists the GPUs Hugging Face rents through Jobs
    (`huggingface.co/api/jobs/hardware`, public). That is 22 NVIDIA flavors, T4 to 8× H200, provisioned on AWS,
    Azure and GCP, each with its card count, memory and price per hour. They are read through the system `curl`
    (bankml has no TLS of its own), **listed and never selected or provisioned**: a rented card is used by running
    bankml on it as a Hugging Face Job, where the Vulkan backend finds it. NVIDIA agreed on 2026-09-02 to acquire
    Hugging Face; closing is expected in the first half of 2027. The dated addendum is bankML's
    [`docs/huggingface.md`](docs/huggingface.md).
  - **`bankml gpu [--remote]`** prints what was found and what will be used. Here it finds the Radeon Vega 3 (RADV, Vulkan
    1.3, 5 compute queues, CPU-mappable memory, `amdgpu` at 0000:04:00.0) and refuses llvmpipe.
  - **No kernels run on the GPU yet.** They come next, and each must reproduce the CPU kernels' bits on the card
    before it is used, so a GPU changes the speed and never the tokens. Several cards will share every matrix by
    rows, which keeps each element one card's exact dot product.

### Also since 0.2.11 (separate commits)
- **Savante's canon moved to `~/cryptoAGI/savante`**, beside jaimla and luvai (991ab1a). The installer and UI
  default to it, and a machine with only the older `~/savante` keeps using that.
- **Open-source coder models in the catalogue** (4e9f4de): Qwen2.5-Coder 1.5B and 7B, and Qwen3.8-27B (the newest
  Qwen). StarCoder and WizardCoder are excluded because their licences (OpenRAIL-M, Llama 2) are not OSI.
- **Two new agents**, `codephreak` (Professor Codephreak, aware of github.com/Professor-Codephreak and its orgmap)
  and `simplecoder`. They were derived from Savante's template with bankml's own ledger, live in
  `~/cryptoAGI/<name>` as private local repositories, and appear in the Agents tab.

## 0.2.11 — 2026-09-29

**P3, step eleven: sampling. bankml draws llama.cpp's tokens with the same seed.** Record:
`testing/results/0.2.11.txt`.

### Added
- **`bankML/sampler.rs`: llama-server's sampler chain.** The chain is penalties → dry → top-n-σ → top-k → typical-p →
  top-p → min-p → xtc → temperature → dist. With the Bonsai GGUF's defaults and neutral penalties, dry, typical-p,
  xtc and top-n-σ, the active part is written from b11192's `llama-sampler.cpp` in its float order:
  - **top-k** is `std::partial_sort` over the vocabulary, **ported from libstdc++'s heap select and heap sort**.
    Logits can tie, and the order the algorithm leaves equal logits in decides which token a draw lands on.
  - **top-p** is a float softmax with a float running cut.
  - **min-p** cuts at `max + logf(p)`.
  - **temperature** is `logit / temp`; at 0 or below, every logit but the first maximum is set to −∞.
  - **dist** computes `expf(logit − max)` summed in double, takes one `uniform_real_distribution<double>` draw from
    `std::mt19937(seed)` (libstdc++'s `generate_canonical`: two 32-bit outputs), and walks a double running sum.
    The RNG advances once per token, including when only one candidate remains.
  - Anything outside that is refused, not approximated: top-k 0 or above 128 (llama.cpp sorts those another way),
    and non-neutral penalties, dry, typical-p, xtc, top-n-σ or dynamic temperature.
  - `Params::from_gguf` resolves the defaults as llama-server does: the GGUF's `general.sampling.*` over llama.cpp's
    own.
- **`bankml generate --sample [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S]`.** Without `--sample` it
  stays greedy.
- **The sampling oracle** (`testing/sample_oracle.py` → `oracle_sample_llama_server`, in the gate). It records 4
  prompts × 10 settings with fixed seeds: the defaults, temperatures 0, 0.3, 1.0 and 1.5, top-k 5, 40 and 128,
  top-p 0.95 and 1.0, and min-p 0.05 and 0.2. Each case keeps the parameters the server reports it ran. **40 of 40
  continuations identical, 1,175 tokens**, on the first run.

## 0.2.10 — 2026-09-29

**P3, step ten: long contexts. All three of ggml's CPU attention kernels are now reproduced.** Record:
`testing/results/0.2.10.txt`.

### Added
- **`attend_head_split`: ggml's split-KV flash attention.** llama.cpp runs it for a single-token decode once the
  padded KV length reaches 512, that is, beyond 256 cells in use.
  - The padded cells are cut into `ceil(padded / nth)`-cell chunks, one per llama.cpp thread.
  - Each chunk runs a partial reference pass (`attend_head_partial`: max, sum and the f16 accumulator, without
    normalizing), skipping masked cells.
  - `ggml_flash_attn_ext_reduce_partials` combines the chunks in order: `fmaxf` of the maxima, two `expf` rescales,
    `fma(acc, old, chunk · new)` for the accumulator and likewise for the sum, then `· (1/S)`.
- **The thread count is part of the result.** `Weights::llama_threads` (`BANKML_LLAMA_THREADS`, default 3, the `-t`
  Savante runs) sets the chunking. `kernel_for(rows, cells, llama_threads)` now returns `Kernel::Split { padded, nth }`
  where 0.2.9 refused, so `bankml generate` runs to any context length.
- **Oracles, in the gate.**
  - `oracle_forward_attention_split` records one decode row over 257, 300, 511, 512, 513, 700 and 1,000 cells, at 3
    and at 4 threads, in the shipped ggml: **14 of 14 bit-exact**. Every hash differs between 3 and 4 threads, so the
    dependence on the thread count is real.
  - `oracle_greedy_llama_server_deep` uses `greedy_oracle.py --deep`: 3 prompts of 98–100 tokens asking for long
    answers, 200 generated tokens each, running to about 300 cells. **bankml generates llama-server's 600 tokens
    exactly.** Each case passes through all three kernels: tiled for the prompt, reference up to 256 cells,
    split-KV after that.

### Found
- **A reduction that the first test cases could not see.** In the shipped binary the reduction is
  `vmulps` + `vfmadd231ps`: the chunk's term is rounded and the running term fused. Written plainly, it passed every
  3-thread case, because there the first chunk held the maximum, so the old scale was exactly 1 and the two forms
  agree. It failed at 4 threads with 512 padded cells, where a later chunk held the maximum. The 4-thread cases were
  added to show the thread-count dependence, and they also exposed this.

## 0.2.9 — 2026-09-29

**P3, step nine: long prompts.** Record: `testing/results/0.2.9.txt`. Built on the new layout: the Rust runtime is
in `bankML/` and Savante's UI in `sAGI/` (ef65eb4).

### Added
- **`attend_head_tiled`: ggml's tiled flash attention** (`flash_attn_ext_tiled`), which llama.cpp runs for any
  micro-batch of 64 rows or more. It is a different algorithm from the reference path:
  - Q stays f32, and K and V are widened from f16;
  - per 64-cell KV tile, the scores are a SIMD GEMM (one FMA chain over the head dimension), then `· scale + mask`;
  - the tile's max joins the running max through `fmaxf`, and an `expf` rescales the f32 accumulator and the sum;
  - `ggml_vec_soft_max_f32` uses ggml's `v_expf`, reduces each 8-lane group in f32, sums the groups in double, and
    adds that to the float sum through double;
  - the V GEMM is one FMA chain over the tile's cells;
  - finally `acc · (1/S)`.

  A row's result does not depend on which rows share its tile, so bankml computes it row by row.
- **llama.cpp's kernel choice.**
  - `kernel_for(rows, cells)` returns the tiled kernel for micro-batches of 64 rows or more and the reference kernel
    otherwise. A single-token step whose padded KV length reaches 512 (`padded_kv`, multiples of 256, at least 256)
    is llama.cpp's split-KV kernel; bankml refuses it with a reason rather than compute something different.
  - `Weights::prefill` splits a prompt into micro-batches of `N_UBATCH` = 512, as llama-server does, with each
    micro-batch's kernel.
  - `Weights::decode` runs one generated token with the decode rule.
- **Oracles, in the gate.**
  - `oracle_forward_attention_tiled` checks layer 0's `kqv_out` for a 150-row micro-batch in the shipped ggml:
    **150 of 150 rows bit-exact**. The reference kernel would match 1 of them, so the choice matters.
  - `oracle_greedy_llama_server_long` uses `greedy_oracle.py --long` with a longer system prompt, so the prompts
    are 111–116 tokens. **bankml generates llama-server's tokens on 6 of 6.**

### Fixed
- **`bankml generate`'s range check.** It warned at 512 cells; it now stops, with the reason, exactly where
  llama.cpp would switch to the split-KV kernel (beyond 256 cells), and has no warning for long prompts because they
  are now reproduced.

### Limits
- Contexts up to 256 cells. The split-KV decode kernel is next.
- The long-prompt end-to-end check ran on the 1-bit model. Llama-server's ternary prefill runs at 0.35 tokens/s, so
  a ternary long-prompt recording takes most of an hour. The tiled kernel is model-independent and is proven at the
  kernel level.
- Prefill still runs one token at a time. The long-prompt oracle takes 8 minutes. Batched prefill is the next speed
  step.

## 0.2.8 — 2026-09-29

**P3, step eight: the ternary model. bankml's own forward pass is token-identical to llama.cpp and about 8× faster
end to end.** Record: `testing/results/0.2.8.txt`.

### Added
- **`forward.rs` runs Q1_0 and Q2_0_g64 models.**
  - `Weights::matrix`, `Weights::quantize` and `Weights::mv` replace the Q1_0-only path. The q8_0 activation is
    prepared once for the model's type (`Act::Q1` or `Act::Q2`) and shared by the matrices that read it. Each type
    goes through its own proven kernel (`q1_0::mat_vec_par`, `q2_0::mat_vec_par`).
  - The embedding dequantizes either type.
  - `bankml generate` takes either model.
- **Ternary oracles, in the gate.**
  - `oracle_forward_model_ternary`: the whole ternary graph in the shipped ggml (`testing/model_oracle.py`, now
    type-general). **1,064 of 1,064 rows bit-exact**, including all 151,669 logits for each of 28 tokens.
  - `oracle_greedy_llama_server_ternary`: llama-server b11192 running the ternary model, with Savante's flags.
    **6 of 6 chat prompts token-identical (140 tokens).**
  - The oracle files are now named per model (`model-<stem>.tsv`, `greedy-<stem>.jsonl`).

### Speed (this laptop, 3 threads, same prompt, sequential runs)
| | llama-server b11192 | bankml 0.2.8 |
|---|---|---|
| ternary, decode | 0.30 tokens/s | **2.32–2.41 tokens/s** |
| ternary, prompt | 0.35 tokens/s | **2.8 tokens/s** (one token at a time) |
| 1-bit, decode | 2.80 tokens/s | 1.80 tokens/s |

The ternary lead is the kernel's (§III.6). On the 1-bit model bankml is still behind, because prefill runs one token
at a time and the per-token overheads are not tuned yet.

### Fixed
- **The stated context limit was wrong: 256 cells, not 512.** llama.cpp pads the KV length it attends over to a
  multiple of 256, with a minimum of 256 (`llama_kv_cache::get_n_kv`). A single-token decode switches to the
  split-KV kernel when that padded length reaches 512, which happens once more than 256 cells are in use. 0.2.7 said
  "under 512 cells". No oracle result changes, since the largest case uses 80 cells, but the limit in the docs does.
  `bankml generate`'s warning threshold is corrected in 0.2.9.
- **The greedy oracle's stopping rule.** The server's token list already includes the end-of-turn token when it
  produces one. The 0.2.7 test then also required bankml's *next* token to end the turn, which is a check with no
  meaning. It passed on the 1-bit model by coincidence and failed on two ternary cases where the server had stopped
  at 16 tokens for its own reasons. The test now generates exactly as many tokens as the server did and compares
  them all. `greedy_oracle.py` now also records the server's `stop_type`.

## 0.2.7 — 2026-09-29

**P3, step seven: the whole model. bankml's own forward pass generates llama.cpp's tokens.** Record:
`testing/results/0.2.7.txt`.

### Added
- **`forward.rs`.**
  - `Weights::step` runs one token through all 36 layers at the next position. Each layer appends to its own f16
    K/V cache (`Weights::caches`), and the step returns `result_norm`.
  - `Weights::logits` is `output.weight` (Q1_0, 151,669 rows) times the q8_0-quantized `result_norm`.
  - Every matmul runs on the thread pool (`mat_vec_par`, same bits at any thread count; `BANKML_THREADS`).
- **`bankml generate MODEL.gguf [--max N] < messages.json | text`.** It renders the conversation (0.2.2), tokenizes
  it (0.2.1), runs the prompt through the forward pass and streams greedy tokens until the turn ends. It warns when a
  prompt or context leaves the range verified against llama.cpp (see Limits).
  - `Tokenizer::token_bytes` gives a token's bytes as llama.cpp's detokenizer does without special tokens.
- **The whole-model oracle** (`testing/model_oracle.py` → `oracle_forward_model`, in the gate). The shipped ggml
  computes every layer of the Qwen3 graph, `output_norm` and the logits for the 28-token prompt. It works one layer
  per context and carries the residual stream between contexts, so memory stays near one layer's weights. **1,064 of
  1,064 rows bit-exact**: 36 layers' `l_out`, `result_norm` and all 151,669 logits, for each of 28 tokens. The
  greedy token matches at 28 of 28 positions. Time: 14 s on 3 threads.
- **The end-to-end oracle** (`testing/greedy_oracle.py` → `oracle_greedy_llama_server`, in the gate). It records
  llama-server b11192's own greedy continuations (top-k 1, prompt cache off) for 6 chat prompts rendered by
  `/apply-template` and tokenized by `/tokenize`. **bankml generates the same tokens on 6 of 6 prompts** (164
  tokens). Where the server stopped before its limit, bankml's next token also ends the turn.

### Limits (stated, and warned about by `bankml generate`)
- Verified range: prompts under 64 tokens, contexts under 512 cells. Outside it, llama.cpp switches to its tiled
  (prefill) and split-KV (decode) attention kernels, which are the next steps.
- The 1-bit model only. The ternary model's forward pass, where bankml's kernel is 9× llama.cpp's, is next.
- **Speed:** measured on this laptop, the 1-bit model decodes at 1.80 tokens/s against llama-server's 2.80. The
  prefill runs one token at a time, against llama-server's batched 3.27 tokens/s. The forward pass is correct first;
  batched prefill and the ternary path are where its speed will come from.

## 0.2.6 — 2026-09-29

**P3, step six: the feed-forward block. With it, all of layer 0 is bit-exact against llama.cpp's.** Record:
`testing/results/0.2.6.txt`.

### Added
- **`forward.rs`.**
  - `Weights::ffn` runs the feed-forward block: `ffn_norm`, then the gate and up projections (one shared q8_0
    activation, the proven Q1_0 kernel), SwiGLU, `ffn_down`, and the residual. The result is llama.cpp's `l_out`,
    the layer's output.
  - `v_expf` is ggml's AVX2 `ggml_v_expf`, lane by lane, and **not libm's `expf`**. It has a range reduction by
    2ⁿ, a degree-5 polynomial evaluated in FMAs, and the scaled path for |n| > 126, with the constants carried as
    exact bit patterns.
  - `swiglu` is `silu(g) · u`, with `silu(x) = x / (1 + v_expf(0 − x))`, as `ggml_vec_swiglu_f32` computes it.
- **The oracle extended.**
  - `oracle_forward_attention` now runs through `l_out`: **112 of 112 rows bit-exact** (`kqv_out`, `attn_out`,
    `ffn_inp`, `l_out`, 28 tokens).
  - New `oracle_forward_swiglu_sweep`, in the gate: the shipped `ggml_swiglu_split` on 24,600 values across ±120
    and the edges. That covers the large-|x| path of ggml's `expf`, which no prompt reaches. **24,600 of 24,600
    bit-exact.**

### Found
- **A whole-layer check can hide a wrong function.** With libm's `expf` in place of ggml's, `l_out` still matched
  27 of 28 rows, because the q8_0 quantization before `ffn_down` absorbs most of the difference. The direct sweep
  matched only 19,245 of 24,600. Every function gets its own oracle for this reason.

## 0.2.5 — 2026-09-29

**P3, step five: layer 0's attention, the output projection and the residual, bit-exact against llama.cpp's.**
Record: `testing/results/0.2.5.txt`.

### Added
- **`forward.rs`.**
  - `KvCache` holds one layer's K and V in f16, one row per position, rounded as llama.cpp's cache stores them.
  - `attend_head` and `Weights::attention` compute llama-server's default attention, ggml's CPU flash attention
    (`flash_attn_ext`), in its reference path:
    - Q rounded to f16;
    - each score is `ggml_vec_dot_f16` as the AVX2 build computes it: four 8-lane FMA accumulators over 32-element
      steps, then a fixed reduction tree;
    - the score is multiplied by `1/√128`;
    - an online softmax whose V accumulator is **f16**, with every step rounded back: `f16(fma(v, w, acc))` and
      `f16(acc · ms)`;
    - finally `acc · (1/S)`.

    Grouped-query attention maps 4 query heads to each KV head. Then `wo` runs through the proven Q1_0 kernel, and
    the residual adds the layer's input.
- **The oracle extended** (`oracle_forward_attention`, in the gate). The shipped ggml's `cpy` to f16, a causal f16
  mask, `flash_attn_ext` (with F32 accumulation set, as llama-graph sets it), `mul_mat` and `add` run on the 28-token
  prompt. **84 of 84 rows bit-exact** (`kqv_out`, `attn_out`, `ffn_inp`). The check discriminates: with the softmax
  sum contracted into an FMA, only 59 of 84 match. The binary computes `S·ms + vs` unfused.

### Scope
- ggml switches attention to other kernels in two cases, which are the next attention steps:
  - a batch of 64 or more query rows (a long prompt) takes the **tiled** kernel;
  - a single-token decode over 512 or more KV cells takes the **split-KV** kernel, whose partial sums depend on the
    thread count.

  Cells masked out are skipped, so padding the cache does not change a result.

## 0.2.4 — 2026-09-29

**P3, step four: layer 0's attention inputs (Q, K and V projected, the head norms, YaRN RoPE), bit-exact against
llama.cpp's.** Record: `testing/results/0.2.4.txt`.

### Added
- **`forward.rs`.**
  - `Weights::qkv` gives llama.cpp's `Qcur`, `Kcur` and `Vcur` for one token:
    - the normed row quantized to q8_0 and multiplied by the Q1_0 `attn_q`, `attn_k` and `attn_v` weights, through
      the kernel already proven bit-exact;
    - Q and K normed per 128-wide head by `attn_q_norm` and `attn_k_norm`;
    - RoPE applied to each.
  - `Rope` carries the parameters llama.cpp's context derives for Bonsai's YaRN: `freq_scale = 1/4`,
    `ext_factor = 1`, `attn_factor = get_mscale(4, 1) / (1 + 0.1·logf 4)` (exactly 1.0 in float, checked against
    the bits the oracle records), beta 32 and 1, `n_ctx_orig` 16,384, and correction dims [20, 37]. The cos/sin
    cache is built as ggml builds it: theta advances by repeated multiplication by `powf(base, −2/n)`, and the
    pairs are NEOX (i, i + 64).
- **The oracle extended** (`testing/forward_oracle.py` → `oracle_forward_qkv_rope`, in the gate). The shipped ggml's
  `mul_mat`, `rms_norm`, `mul` and `rope_ext` run on a 28-token chat prompt at positions 0–27, and again at positions
  7 to 63,214, where theta is a long product. **140 of 140 rows bit-exact.**

### Found
- **The bits follow the binary, not the source.** Written as the C source reads (`x0·cos − x1·sin`), RoPE matched only
  30 of 84 rows. The projections and head norms were exact (28 of 28 each), which isolated the difference to RoPE.
  The disassembly of the shipped `libggml-cpu-haswell.so` shows GCC contracting three expressions into FMAs:
  - the rotation: `fma(x0, cos, −(x1·sin))` and `fma(x0, sin, x1·cos)`, with x1's products rounded and x0's fused;
  - the YaRN mix: `fma(θ_interp, 1 − r, θ_extrap·r)`;
  - the magnitude: `fma(logf(1/fs), 0.1, 1)`.

  bankml writes the same three as `mul_add`s. The RMS norm has nothing to contract, which is why 0.2.3 matched as
  written.

## 0.2.3 — 2026-09-29

**P3, step three: the first operations of bankml's own forward pass, bit-exact against llama.cpp's.** Record:
`testing/results/0.2.3.txt`.

### Added
- **`forward.rs`.**
  - `Weights` reads the model's tensors through bankml's own memory map.
  - `embed` is llama.cpp's `get_rows` on the Q1_0 token table (`inp_embd`), through the dequantizer already proven
    bit-exact.
  - `rms_norm_mul` is `rms_norm` then `mul` by the layer's norm weight (`attn_norm-0`), in the float order read from
    b11192's `ggml-cpu/ops.cpp`: the sum of squares accumulated in double, one float product at a time; the mean
    rounded to float; `1 / sqrtf(mean + eps)`; each output `(x · scale) · w`, with no FMA.
- **The forward-pass oracle** (`testing/forward_oracle.py` → `oracle_forward_embed_norm`, in the gate). The release has
  no tool that prints intermediate values, so the oracle drives the **shipped** `libggml` through ctypes with the same
  ops llama.cpp's Qwen3 graph uses, computed by the release's own CPU backend, on 300 real token ids. **300 of 300 rows
  bit-exact for both `inp_embd` and `attn_norm-0`.**

## 0.2.2 — 2026-09-29

**P3, step two: the chat template, byte-identical to llama.cpp. The ternary headline re-measured and confirmed.**
Record: `testing/results/0.2.2.txt`.

### Added
- **`chat.rs`** renders a conversation into the prompt exactly as llama.cpp b11192 does with the Bonsai / Qwen3
  template in the GGUF, without a Jinja engine. `check_template` accepts only that template (sha256 `30a75d10e60b57e2…`
  of its text; the 1.7B and 8B files carry the same one). Its rules are written out with Python's `split`/`strip`
  semantics:
  - the first system message at the top;
  - the last real user query found by scanning back;
  - `<think>` blocks kept only after that query;
  - tool results grouped into one user turn;
  - the thinking-off generation prompt.

  Tool definitions, tool calls and assistant prefills are refused, not guessed.
- **The chat-template oracle** (`testing/template_oracle.py` → `oracle_chat_template`, in the gate): llama-server's
  own `/apply-template` on 317 conversations. **317 of 317 byte-identical.** The oracle caught one server behaviour
  the template does not show: an empty `reasoning_content` is dropped before templating.
- `bankml chat-template MODEL.gguf < messages.json`.

### Measured
- **The whole-token ternary budget, with the model resident.** The browser and the chat engine were closed, and 2.25
  of 2.31 GB stayed cached. bankml then took **0.231 s per ternary token at three threads against llama.cpp's
  2.185 s (9.45×)**, and 0.223 s (9.78×) at four. That reproduces the 0.0.3–0.0.6 figure on today's code and confirms
  that the slower gates measured the disk, not the kernel. README, TECHNICAL and PERFORMANCE now state it as measured
  (docs/PERFORMANCE.md has the table). The 0.2.2 gate itself ran with 3.3 GB free and recorded the same: **0.233 s
  against 2.204 s (9.46×)** by the fastest runs, 0.287 s against 2.23 s (7.8×) by medians.

### Documentation
- **The README's results checked against the record.** Every figure now comes from the 0.2.2 gate record and says
  which statistic it is. The corrections:
  - ternary per matrix 9.4–10.0×, was "9.5–9.8×" from 0.0.1;
  - ternary prefill 12.9×, was 12.5×;
  - 1-bit prefill 1.23×, was "1.2–1.3×";
  - llama.cpp's 1-bit token 0.35 s, was 0.34 s;
  - the ceiling 4.3 vs 0.45 tok/s, was "4 vs 0.4";
  - the badge.
- **The speed cards are now drawn from a gate record** by `tools/cards.py`, not by hand. The 0.0.1 cards had drifted
  (7.06× from a pre-pool harness; 14.47 ns per 1-bit block from an older build). Each card names its record, says
  whether the model was resident, and gives the min and median where they differ.
- **Savante's public places are linked from the README**: the Hugging Face Space, the sAGI skill, her loop dataset,
  the sAGI engine and her canon. The Documentation table links straight to **the Thesis** in TECHNICAL.md.

## 0.2.1 — 2026-09-29

**P3, step one: bankml's own tokenizer, token-identical to llama.cpp.** P3, a forward pass of bankml's own, has to
produce the same tokens as llama.cpp; the tokenizer is where that starts. Record: `testing/results/0.2.1.txt`.

### Added
- **`tokenizer.rs`** (no crates) reads the vocabulary straight from the GGUF: 151,669 tokens, 151,387 merges and
  their types, through a bounded reader of its own. The guard's header parser skips arrays by design. It then
  reproduces llama.cpp's gpt2/qwen2 tokenization:
  - special tokens cut out first, longest first. USER_DEFINED ones such as `<think>` always split the text; CONTROL
    ones such as `<|im_start|>` only when special tokens are parsed;
  - Qwen2's pre-tokenizer pattern, written out alternative by alternative, including the backtracking outcomes of
    `\s*[\r\n]+` and `\s+(?!\S)`;
  - GPT-2 byte-level BPE by merge rank, leftmost on ties.

  The pattern's `\p{L}` is Unicode general category L, generated into `unicode_letters.rs` (Rust's
  `is_alphabetic` is a different property).
- **The tokenizer oracle** (`testing/tokenizer_oracle.py` → `oracle_tokenizer`, in the gate). It records the running
  engine's `/tokenize` on every document, Savante's canon, hand-picked edge cases and a seeded 2,000-string fuzz set
  over 17 Unicode ranges, with special tokens parsed and not. **4,258 of 4,258 cases token-identical.**
- **`bankml tokenize MODEL.gguf [--no-special]`** (text on stdin, ids out).

- **The whole-token ternary budget with more memory free.** For this gate the chat engine was stopped (2.0 GB free
  instead of about 1.4 GB). bankml's ternary token then took **1.42 s against llama.cpp's 3.0–5.2 s (2.1–3.6×)**,
  against about 4.6 s (1.1×) in the disk-bound gates before. That is further evidence for the page-cache explanation
  in docs/PERFORMANCE.md. It is still short of the 0.23 s measured with the model fully resident: 2.31 GB does not
  quite fit in 2 GB.

### Fixed
- The CLI tests left their synthetic GGUF fixtures in `/tmp`: 234 runs had accumulated 1.2 GB on a disk with 4.2 GB
  free. Each run now removes the directories of earlier runs whose process is gone.

### Measured
- 243 KB of docs (72,403 tokens): bankml loads the vocabulary in 0.17 s and encodes in about 0.04 s. llama-server's
  `/tokenize` answers the same text in 0.175 s, vocabulary already loaded, HTTP and JSON included. The ids are
  identical. That is a measured comparison, not a claim about llama.cpp's tokenizer alone.

## 0.2.0 — 2026-09-29 — milestone

**Verified, documented, and offered back.** 0.2.0 closes the 0.1.x run of audits and speed work:
- every document checked line by line against the code and the gate records;
- a fourth audit fixed;
- bankml's ternary kernel prepared as a contribution to llama.cpp, bit-exact against its shipped library.

Record: `testing/results/0.2.0.txt`.

### Added
- **`upstream/`: an AVX2 `Q2_0` kernel for llama.cpp.** At b11192 x86 has only scalar `Q2_0`, and the one open x86
  PR needs AVX-VNNI. `q2_0_avx2.c` is a drop-in `ggml_vec_dot_q2_0_q8_0` in ggml's C:
  - the generic code's float order, with its FMAs explicit;
  - codes expanded by one variable dword shift;
  - the activation transposed 4×4 within each lane, with no lane crossings.

  `test_q2_0_avx2.c` compares it with the shipped `libggml-cpu-haswell.so`: **bit-exact on 200,000 of 200,000 random
  cases** (every code 0–3, extreme activations), and **3.4×** the shipped scalar path per 4096-wide row, pinned to two
  cores. The README explains why bankml's own kernel is faster (9.5–9.8×): it lays the activation out once per token,
  where a drop-in must do it per call. MIT, llama.cpp's licence. Opening the pull request is the authors' decision;
  nothing has been submitted.

### Fixed (the fourth audit, of 0.1.9)
- **The history window at a tight context.** It moved almost every turn, so the prompt cache did not help exactly
  when it mattered. Now:
  - the start is kept per session while it still fits;
  - when it stops fitting, the newest half of what fits is kept, so the next turns reuse the prompt (when only two or
    three fit, all are kept: history is worth more then);
  - with room, the window moves 8 times in 60 turns.
- **The "history trimmed" note** now appears only when the context forced a trim, not for the ordinary 12–17 window.
- **A prompt that cannot fit is refused with the numbers,** instead of the engine's error.
- **Two tabs no longer share one window record**: each request gets its own.
- **Slot saves run in the background**, after the answer and its `.history` line, and only when no other answer is
  being written: they no longer save another tab's KV or delay a footer. A failed save is not retried every turn, and
  the `slot` field records what a turn did, not the engine's state.
- **Token counts**: history outside the window is not tokenized; the cache evicts its oldest half instead of
  clearing. The fallback's docstring no longer claims to be an overestimate for every script.
- **The engine's context** is cached per engine start (a restart with a new context is seen at once) and falls back
  to the saved setting.
- `bankml serve`: the over-capacity 503 is answered on its own short thread with one overall deadline, so a slow
  client can no longer hold the accept loop.
- The embedding indexer cannot stay disabled if its thread fails to start. `testing/pinned.sh` no longer needs
  python3.

### Documentation
- A line-by-line accuracy review of every document against the code and the gate records gave **23 corrections**:
  - the usage guide's commands, routes, environment table, window policy, Apply behaviour and gate steps;
  - dead anchors;
  - guard-agreement counts (28 of 28);
  - future work that is done;
  - LICENSING.md's tense for modules that do not exist yet;
  - the testing README's tables;
  - a new PERFORMANCE section for 0.1.8–0.2.0.
- **The whole-token ternary figures are now dated.** The 0.23–0.25 s per token of the 0.0.3–0.0.6 records was
  measured with the model resident in memory. The gates since 0.1.0 measure 4.5–5.2 s, because this laptop can no
  longer keep the 2.31 GB file in its page cache (1.04 GB stayed resident after a full read, with other applications
  holding memory), so both runtimes wait on the disk. The per-matmul A/B on cached tensors still shows 9.8×. A
  re-measure on a machine with ≥ 3 GB free is in `docs/TODO.md`.

## 0.1.9 — 2026-09-29

The TODO's 0.1.9 items, and the third audit fixed. The first answer after an engine restart is **8.6× faster**, the
chat can no longer silently lose its history, and bge-m3 runs live. Record: `testing/results/0.1.9.txt`.

### Faster
- **The system prompt's KV survives a restart** (KoboldCpp's and llama.cpp's slot saving). The carrier starts with a
  slot directory (`bankml serve --slot-dir`, passed on as `--slot-save-path`). After the first answer of an engine's
  life, the chat saves the slot: 51 MB for Savante's 282-token prompt. On the first question to a new engine it
  restores it. Measured live, same question at temperature 0:
  - **first answer 132 s → 15.4 s**;
  - prefill 118 s → 1.2 s (315 prompt tokens → 1);
  - restore 0.05 s;
  - **identical answer**, because the same tokens give the same KV.

  A model switch or a Resources change no longer costs a two-minute first answer. The newest three slot files are
  kept, and each is keyed by model sha256, context and system prompt.

### Fixed (the third audit, of 0.1.7–0.1.8)
- **HIGH: the history window could send no history at all.** With the persona prompt plus `.memory`, 0.1.8's budget
  (at ~3 characters per token) left room for about one exchange. The window then stepped past even that, and with
  the long `sAGI.prompt` nothing was ever sent. Now:
  - the budget is counted in **the engine's own tokens** (llama-server `/tokenize`, cached per text), against the
    context the engine **actually runs with** (`n_ctx` from `/props`, not the saved setting);
  - the window keeps as many recent exchanges as fit, moving in steps of at most half of what fits, so at most a
    couple are left out and the prompt cache stays warm between moves. At 2048 tokens it now sends 4–6 exchanges,
    where 0.1.8 sent 0 or 1;
  - when history is trimmed, the answer says so: "history: k of n exchanges fit the engine's N-token context — raise
    the RAM budget in Resources for more". Nothing is silent.
  - It was hot-patched into the running UI before this release.
- An answer's own `<sub>` (a formula) is no longer cut when the chat footer is stripped.
- Resources:
  - the context is re-planned per model from the saved RAM budget, so a larger model no longer inherits a context
    planned for a smaller one;
  - a failed Apply restores **your previous** settings (not the defaults), and reports both errors.
- `connectors.load` checks every private blob before writing anything: a refusal no longer leaves an agent
  half-overwritten.
- Embeddings:
  - one writer per cache file (a lock, and one `write` per line);
  - the background indexer is marked running before its thread starts, so two quick searches start one indexer;
  - query vectors are keyed by the model's digest.
- `bankml serve`:
  - `/bankml/usage` samples at most once a second, so polling cannot tie up connection slots;
  - the over-capacity 503 is not lost to a reset;
  - `sysconf` is declared with C types.
- `bankml usage abc` is an error, not a report on bankml itself.
- `testing/pinned.sh` takes its CPUs from the shell's allowed set, and claims a memory cap only after reading it back
  from the cgroup.
- `::1` is no longer offered for interact mode: the Host check cannot parse a bracketed address.

### Measured, and not done (docs/TODO.md)
- **madvise / fadvise for hashing.** From a cold page cache the 1.7B model hashes in 0.68–0.72 s against 0.59–0.78 s
  warm (364 vs 419 MB/s). The disk keeps up with SHA-NI; there is nothing for a read-ahead hint to win.
- **PGO for `bankml serve`.** The gateway adds 0.71 ms per request against 15–130 s answers (0.005 %).

### Live
- **bge-m3** ran for real: 9.2 s for the first call (load), 1.4 s for three texts after, 1.14 GB while loaded. It
  indexed `.history`, and the ragebar reports mindX's `rage.py` BM25 fused with bge-m3 (`docs/embedding.md`).

## 0.1.8 — 2026-09-29

Speed and efficiency, measured on fixed resources, and the house in order: the licence layers, the docs in `docs/`
behind a README that routes to them, and `docs/TODO.md` from four studies (the field, KoboldCpp, vLLM, Rust).
Record: `testing/results/0.1.8.txt`.

### Faster
- **SHA-256 with the CPU's SHA extensions (SHA-NI)**, detected at run time, with the portable rounds kept as the
  model. It is **5.5× faster**: 0.23 s against 1.25 s for 248 MB, and 2.9 s for the 1.16 GB 8B model, which hashes to
  its published pin and to coreutils `sha256sum`'s value. Every `bankml serve` start and every model switch hashes the
  whole model, so every one of them is faster: restarting the 8B carrier took 7.0 s, against 25 s on 0.1.7.
  `BANKML_NO_SHANI=1` selects the portable path.
- **A chat window that keeps the engine's prompt cache warm.** The history the model sees now moves in steps of six
  exchanges instead of sliding by one each turn. A one-exchange slide changed the text right after the system prompt,
  so the whole history was re-read every turn: vLLM's chained block hashes show why no prefix cache can help then,
  and KoboldCpp's "Smart Context" reserve is the same remedy.
  - Over 60 turns the window moves 8 times instead of 48, and 39 of 47 turns reuse the previous prompt.
  - The window also shrinks, in whole steps, to fit the engine's context.
  - The chat's footer (the clock and the receipt) is no longer sent back to the model; it was wasting tokens.
- Measured and **not adopted**, because neither is a gain beyond the noise, though both were token-identical:
  - speculative decoding with a draft model (Bonsai-1.7B for Bonsai-8B);
  - n-gram speculation, measured twice, the second time on fixed resources (`testing/pinned.sh`: cores 2–3, 2
    threads, 2.5 GB). It was ahead in 5 of 6 paired runs (+0.01 to +0.36 tok/s, one −0.06), but the baseline itself
    ranged from 0.36 to 1.40 tok/s as other programs loaded the same cores. Being exact and costing no memory, it is
    offered as an **opt-in**: `bankml serve --spec-ngram`, and a checkbox in Resources, off by default.
  - KV shifting (`--cache-reuse`) is rejected outright: it is not token-identical.

  Details and sources are in `docs/TODO.md`.

### Added
- **Resources: CPU and RAM sliders** in the chat's side column.
  - The threads slider and a RAM budget, which bankml turns into the largest context that fits (weights + engine
    overhead + KV cache at the architecture's bytes per token, as `bankml guard` reports it).
  - **Apply** restarts the engine through the verified switch, is refused mid-answer, and is remembered.
  - **Usage now** reads what the engine uses.
- **`sys.rs`, bankml's psutil, with no crates.** It reads memory, cores, and each process's resident memory and CPU
  time from `/proc`, which is what psutil and Rust's `sysinfo` read on Linux. `GET /bankml/usage` on serve (itself
  and the engine it launched), and `bankml usage [PID …]`.
- **`testing/pinned.sh`**: benchmarks on a fixed amount of processor and memory, with cores pinned (`taskset`), a hard
  RAM cap and no swap (a user cgroup), and the load recorded before and after.
- **Licensing layers** (`LICENSING.md`). Own code is **`MIT OR Apache-2.0`** (`LICENSE-MIT`, `LICENSE-APACHE`): open
  source, do what you want, rights preserved. Key handling will be `GPL-3.0-only` in an opt-in module, so no
  black-box modification ships. AGPL-derived code only walled off. Every source file carries an SPDX header, checked
  in the gate (`testing/spdx_check.py`).
- **Docs in `docs/`** (`usage.md`, `TECHNICAL.md`, `PERFORMANCE.md`, `oracles.md`, `research.md`, `embedding.md`,
  `TODO.md`). The README opens with a Documentation section that sends each reader to the right one.

### Fixed
- **A refusal could reset the connection.** When `bankml serve` refused a request (a foreign `Host`, a non-JSON POST)
  it closed without reading the body. Closing with unread data makes the kernel send RST, so the client could lose
  the refusal. It now reads and discards the declared body first. The CLI suite passes 10 times in a row, where it
  had failed under load.
- The CLI tests kill the `serve` they start even when an assertion fails (a leaked child held the test's pipes open).

## 0.1.7 — 2026-09-29

A second audit, of the parts the first did not cover (the gateway, the chat path, commitments, agents, PostgreSQL,
the chain), with every finding fixed and tested. Savante gains the embedding model mindX uses (bge-m3), and
`research.md` places bankml among its peers. Record: `testing/results/0.1.7.txt`.

### Fixed — `bankml serve` (Rust)
- **A loopback caller could crash it or pile up threads.** Its limits are now:
  - at most 32 connections;
  - a 30 s read and 120 s write timeout;
  - request-head lines of at most 16 KiB and at most 100 headers;
  - upstream bodies and chunks of at most 64 MiB;
  - chunked requests refused.
- **A receipt could name a model the engine was not running.**
  - `--spawn` now refuses when the port already answers, so it can no longer bind to a leftover engine.
  - A spawned llama-server that dies is noticed at once, not after the 900 s health timeout. That timeout was what
    made a failed model switch take 15 minutes.
  - SIGTERM, SIGINT and SIGHUP stop the spawned engine too.
  - Before every answer, serve re-checks the file's identity (device, inode, size, mtime) and the engine's
    `/props` path. If either changed it refuses (503) rather than receipt weights it did not verify.
- **Receipts name the request.** A new `request_sha256` field hashes the request body, and a `"signed": false`
  field says what the receipt is. The docs now say plainly that receipts prove integrity between a client and its
  own gateway, not to a third party.
- **Web pages cannot drive it.** It requires a loopback `Host`, which blocks DNS rebinding, and a JSON
  `Content-Type` on POST, which blocks the simple cross-origin form POST.
- **Correctness.**
  - A final SSE line without its newline is forwarded and counted.
  - Unpaired `\u` surrogates decode to U+FFFD, the same way the UI hashes them, and an escape after an unpaired
    high surrogate is no longer swallowed.
  - HTTP reason phrases are right (403, 413, 415, 500, 503, 504), and a garbled upstream status becomes 502.

### Fixed — the chat, `.history` and commitments (Python)
- **One torn line no longer blanks `.history` or `.memory`.** Damaged lines are skipped and counted. Each record is
  one locked, fsynced write, and it starts on a new line if the file ends mid-line. Rewrites of `.memory` are
  atomic.
- **Model switches can no longer be blocked forever.** Busy tracking is a per-request in-flight table with stale
  expiry, instead of one global that a closed tab or an exception in `system_prompt()` could leave set, and one
  tab's `finally` could clear for another. A timer left running by a lost request expires.
- **Receipt checks no longer fail on answers with a `<think>` block or surrounding whitespace.** The raw text is
  stored as `assistant_raw` and checked; the docs had said this was already so. The text is `choices[0]` only, as
  the receipt hashes it.
- **An exchange cannot land in the wrong agent's history.** The `.history` path is fixed when an answer starts.
- **The Merkle tree is now RFC 6962.**
  - Leaves are sha256(0x00 ‖ line) and nodes are sha256(0x01 ‖ l ‖ r), so an interior node can no longer pass as a
    record.
  - The tree is split at the largest power of two, so an odd node is promoted, and 3 records and 4 records with the
    last repeated no longer share a root.
  - Proofs carry the index and the tree size and are verified as RFC 9162 §2.1.3.2 prescribes.
  - The roots match the Certificate Transparency reference vectors.
  - Scheme `rfc6962-sha256`. Roots differ from 0.1.6's.
- **Interact mode is loopback only, enforced.** `--mode interact --host 0.0.0.0` is refused, and a non-loopback
  `Host` header gets a 400.

### Fixed — agents, PostgreSQL, the chain
- **The 15 doctrine clauses are now fixed at derivation, as the docs said.** `save()` refuses a persona edit that
  changes them and names the clauses. `verify()` uses bankml's pointer list, not the one inside the ledger it is
  checking, and also checks that the root is unchanged since derivation.
- **Publishing to PostgreSQL is one transaction.** A failure part-way changes nothing. `load()` can be anchored to an
  outside contentRoot (a token's), and says whether it checked against one or only for self-consistency.
- **`chain.py`.**
  - A devnet send goes only to a node on this machine, since a remote node can report chain 31337.
  - Addresses must be 20 bytes; a short one would have shifted every argument the owner signs.
  - The `cast` line quotes every argument.

### Fixed — view mode
- A 20 s socket timeout, so a slow LAN client cannot hold a thread.
- `/api/state` is computed at most every 2 s, however many viewers poll.

### Added
- **Embedding with bge-m3, the model mindX uses** (1024-d, MIT; served by the local Ollama). See `embedding.md`.
  - The `.history` ragebar ranks by words (BM25) and meaning (bge-m3) together, by reciprocal rank fusion.
  - PostgreSQL publishing fills `bankml_exchanges.embedding` in the same transaction.
  - Vectors are cached beside `.history`, keyed by the sha256 of the embedded text and tied to the model's digest.
  - It is optional and guarded: without the model or the memory to load it, search is BM25 alone.
  - On this laptop, with other applications holding memory, the guard refused to load bge-m3 (1.0 GB free, 1.3 GB
    needed), which is what it is for.
  - An embedding model is never offered as the chat carrier.
- **`oracles.md`**: every oracle bankml is checked against, what each checks and how, and what it found in this
  gate. The oracles are the compiled llama.cpp kernels, the scalar models, the Python guard, FIPS 180-4 and RFC 6962
  vectors, published roots and CIDs, the THOT spec, and the iNFT contract. It also states the rule: the same bits
  first, then the speed.
- **`research.md`**: Rust inference engines, 1-bit and ternary inference, and verifiable inference as of
  2026-09-29, with links and 26 papers. Its verdict:
  - bankml's bit-exact oracle against the compiled llama.cpp, and its plain-AVX2 `Q2_0` kernel, have no public
    equivalent;
  - others are ahead on a native forward pass, GPUs, breadth of formats, ARM and attestation.

### Tests
- Rust:
  - bounded heads and bodies;
  - loopback hosts only;
  - unpaired surrogates;
  - file-identity changes;
  - in the CLI test against a mock engine: 403 for a foreign Host, 415 for a non-JSON POST, 503 without a receipt
    once the model file changes, and `request_sha256`.
- `test_ui.py`:
  - RFC 6962 reference roots and second-preimage cases;
  - a torn line;
  - raw-text receipts;
  - stale in-flight expiry;
  - fixed doctrine;
  - the offline embedding path (a fake Ollama).
- `test_connectors.py`: an atomic publish, and an anchored load.
- `test_chain.py`: a remote devnet refused, a short address refused, and a quoted cast line.

## 0.1.6 — 2026-09-29

An audit of 0.1.5, and every finding fixed with a test that would have caught it. Record:
`testing/results/0.1.6.txt`.

### Fixed — the carrier switch (ui/models.py)
- **A slow start was declared failed, leaving two carriers running.** bankml serve hashes the whole model before it
  binds a port, so a 5 GB file on a cold cache took longer than the 20 s "no ports, so it died" heuristic. The switch
  then rolled back while the new serve kept going, and both fought over 18092/18093. The switch now waits on the
  process itself: it stops only if the process exits or the deadline (30 min) passes, and then kills the process
  group (serve and the llama-server it spawned).
- **Success now means the chosen model.** A carrier that answers counts only if its verified sha256 is the chosen
  file's. An old carrier still holding the port is reported, not mistaken for success.
- **Ports are matched on the configured host** (127.0.0.1, 0.0.0.0, [::1] …) and only for this user's
  `bankml` / `llama-server`. Stopping never crashes on a process that has already gone. If the ports cannot be freed,
  the switch says so.
- **Rollback also restores a previous carrier outside `.models`** (for example `~/sAGI/bonsai/…`), found by its
  verified sha256 and its pin.
- **Memory is checked before any switch and any adoption,** not only before a download.

### Fixed — downloads
- **A complete `.part`** (the process died between the last byte and the rename) is verified and kept. Previously
  every retry asked for `bytes=N-` and got 416.
- **A 206 whose `Content-Range` starts elsewhere** is not appended; the download starts over.
- **A source that sends more than the published size** is cut off and discarded, not written until the disk fills.
- **Ollama adoption is really offline.** It reads the local manifest and licence layer, so it is pinned to what was
  pulled even if the registry's tag has moved since.
- **Input.** `.`/`..` segments, slashes in a revision, and odd Ollama tags are refused. Hugging Face files in
  sub-folders keep their folder in the local name, so two `model.gguf` files no longer collide.
- **Gated repositories** are refused with a reason, not a raw 401.
- **If an answer is streaming when an import finishes,** the import stands and the switch waits, instead of the whole
  job failing.
- **Cancel download** in the Models tab. The partial file is kept, and importing again resumes it.

### Fixed — the UI and view mode
- **The Models tab no longer holds a queue worker** for a whole download. Handlers return at once, and a light
  2-second poll follows the job, so chat and the timers keep their workers.
- **One damaged file no longer breaks `/api/state` for the whole LAN.** A damaged export record reads as "not
  current". The voice manifest and export records are written atomically. Any unexpected error returns a 500
  instead of a dropped connection.
- **Exports stream** in 1 MiB chunks instead of being read into memory whole.
- **`/api/state` is cached** until the voice manifest, an export or TECHNICAL.md changes (at most 30 s). The voice
  manifest is re-read only when it changes, and the pronunciation table is compiled once, not per sentence.
- **Export robustness.** Exports are built beside their destination, so the final rename is atomic even where `/tmp`
  is tmpfs. The concat list quotes paths safely. A clip missing from the manifest is measured with ffprobe, so the
  chapter marks don't drift.

### Fixed — pronunciation
- bankml's respellings now run **before** underscores become spaces, so `Q1_0`, `Q2_0_g64`, `Q4_K_M` and `q1_0.rs`
  are read as formats and files ("Q one zero", "Q four K M", "bank M L dot R S").
- Maths symbols are read **only in maths**:
  - "Professor / OVERLORD" and "DAIO · savante_sagi" keep their separators;
  - "8 / 128" still becomes "8 over 128".
- Section references read as "section three point four"; a parenthesis that only points somewhere is dropped.
- Names are matched case-insensitively ("Bankml").
- The 18 sentences whose spoken form changed were re-rendered, and both exports rebuilt.

### Tests
- `test_models.py` gains 9 checks (28 in all with the carrier):
  - input refusal (`..` segments, revisions with slashes, odd tags);
  - complete `.part`;
  - misplaced 206;
  - over-send;
  - one job at a time;
  - memory on adoption;
  - exact listener matching;
  - a different model answering is not success;
  - a slow start is waited for, then killed.
- `test_ui.py` gains pronunciation regressions and a damaged export record.
- Rust: `Verified::to_json` stays valid JSON with a hostile model name (quote, backslash, newline, control
  character).

## 0.1.5 — 2026-09-28

Models come in without friction, and only open-source, sha256-pinned ones: Bonsai-8B on first run, then the
catalogue, any Hugging Face GGUF, or Ollama. The metrics become readable charts. The voice controls get a dock. Savante
reads bankml's thesis and the binary / ternary argument, and her audio is exported and committed. Record:
`testing/results/0.1.5.txt`.

### Added — models
- **The importer** (`ui/models.py`, the **Models** tab, `python3 ui/models.py`). Its sources:
  - **First run.** If no carrier answers when interact starts, Bonsai-8B is imported (only if absent), pinned and
    verified, and `bankml serve` is started in the background with progress shown.
  - **Catalogue.** Eleven open-source models: Bonsai-8B (default), Ternary-Bonsai-8B, Bonsai-4B and 1.7B, Qwen3
    0.6B / 1.7B / 4B / 8B, SmolLM2-1.7B, SmolLM3-3B and Granite-3.3-2B. Each is pinned to its repository's revision,
    size and sha256, read 2026-09-28, and an import refuses if the repository no longer agrees.
  - **Hugging Face.** Any repository or file URL. The licence is read from the repository, and the pin is its LFS
    sha256 at the resolved revision.
  - **Ollama.** Search ollama.com; import from the registry, where the model layer's digest is the GGUF's sha256 and
    the licence layer is read; or adopt a model the local Ollama holds, by link with no download.
- **The trust chain.** Every file is hashed as it streams, and kept only if it equals the published sha256; an
  interrupted download resumes. `bankml guard` must then say play, and a FORK.json pin is written, so `bankml serve`
  verifies it before every start.
- **Open source only.** A licence outside the OSI list is refused before any byte is fetched: Gemma and Llama on
  Hugging Face and on Ollama, and anything unrecognised.
- **The machine's limits.** An import that would leave less than 1.5 GB free on disk, or whose weights exceed about
  70 % of RAM, is refused with the numbers.
- **Carrier switch.** Stop, then start `bankml serve --spawn llama-server` on the chosen model. It is refused while an
  answer is being written. If the new model does not come up, the previous one is restored; it is found by its
  verified sha256, because serve reports canonical paths.
- **Verified on this laptop:**
  - Qwen3-0.6B Q8_0 was downloaded from Hugging Face, and qwen3:0.6b (a Q4_K_M) was adopted from the local Ollama;
    both are pinned, and both answered through bankml with receipts;
  - Bonsai-1.7B, already here, was pinned against the published sha256;
  - a switch, a refusal while busy, and a rollback from a bad pin all ran on spare ports.
- `testing/test_models.py`: 19 checks, in the gate. The source is a loopback server with a synthetic GGUF, and a
  real carrier runs on spare ports.

### Changed — bankml (Rust)
- **Standard models, named and reported.** `gguf::type_name` names all of ggml's standard types (Q8_1, Q8_K, the
  IQ family, I8–I64, F64, MXFP4), and a new test confirms a K-quant model plays on mainline. The guard already played
  such files; nothing about them was allow-listed or refused.
- **`Verified` carries what was verified:** `arch`, `name` and the tensor types. `/bankml` reports them, so the UI
  shows "qwen3 · Q4_K×155, Q6_K×15" and sends `/no_think` only to families that honour it (Qwen3, SmolLM3, Bonsai).

### Changed — metrics
- **Readable charts** replace the old bar strip, whose bars became "giant blue blobs" when there were few exchanges:
  one bar was stretched across a third of the width.
  - Summary tiles: exchanges and sessions, median and p90 first-token and response times, writing speed, and answers
    matching their receipts.
  - Response time per exchange: stacked bars (waiting, then writing), at most 18 px wide, on a round-numbered axis
    with a median line. One slow outlier is clipped with a ▲ so the rest stay readable.
  - Throughput per exchange: prefill and writing tok/s as lines.
  - The charts keep their proportions at any width, and dark mode is transparent.
- **Commitments** read one per line: the .history record count, Merkle root, CID, and .memory. An empty root reads
  "— (empty)", not "None".

### Added
- **The VOICE dock.** The DreamKnobs no longer sit inside each section. They appear only when a PLAY is pressed, in
  one dock beside the card: a vertical stack in the side column by default. ⤒, ⤓ and ⇥ dock it to the top of the
  card, the bottom, or back to the side (top and bottom lay the knobs in a row). It can also be dragged by its grip
  onto a drop zone and resized from its corner (40–96 px). Place and size are remembered per browser. Verified in
  headless Firefox:
  - hidden before play, and a vertical stack to the right of the card after it, with no overlap;
  - above the card when docked top and below it when docked bottom, horizontal both times;
  - back at the side, with the choice remembered.
- **SCIEN·TIFIC in the introduction.** A new chapter, marked as bankml's own note rather than her canon
  (`ui/voice/readings/scientific.md`), follows "Why I exist". It takes 2²⁵⁶ − 1 (SCIEN·TIFIC's whole supply, the
  largest value one EVM word holds, about 1.16 × 10⁷⁷; scientific.pythai.net) first as the finest resolution a
  single word allows, the maximum measurement of accuracy, and then as a measure of size. The atom count is stated as
  measured: estimates for the observable universe run from 10⁷⁸ to 10⁸², so by a strict count 2²⁵⁶ falls short, but
  it is of the same order of scale.
- **The reading.** Savante reads bankml's thesis and the binary / ternary sections of `TECHNICAL.md` (§II.1, §III.2,
  §III.5, §III.8) verbatim, as its own section of the card with **▶ PLAY THE READING**. It is also in the view
  mode's Listen panel.
- **TECHNICAL.md §III.8, "Can a binary computer perform a ternary operation?"** Yes, exactly. A trit is stored in bits
  (log₂ 3 ≈ 1.585 bits of information; `Q2_0_g64` spends 2 bits plus the scale). The ternary product is a signed
  sum, which bankml computes as Σ c·x − Σ x with the offset code c = w + 1 (`vpmaddubsw`, then one subtraction of an
  activation sum shared by every row). The section also covers Setun and why packing, not native ternary gates, is
  the next ternary speed-up on a bandwidth-bound CPU.
- **Export.** `⤓ Savante.opus` holds the voice examples and the whole introduction; `⤓ Savante-reading.opus` holds
  the reading. Each is one Ogg Opus file (40 kbit/s) with a chapter mark per chapter. It is made only when every
  sentence is rendered, carries a signature of the exact clips it was built from, and is rebuilt when one changes.
  Both are offered in the card, and in view mode at the fixed routes `/export/Savante.opus` and
  `/export/Savante-reading.opus`, which serve only a named export that is complete and current. `python3
  ui/speak.py` renders everything and writes both files; `--prune` drops clips no current text uses.
- **The audio is in the repository.** `ui/voice/cache/savante/` (every clip plus its manifest) and `ui/voice/export/`
  are committed, so a fresh clone plays and downloads without rendering. The voice is built from public-domain Cori
  (see 0.1.3).

### Changed
- **Speech.** List markers are removed only at the start of a line, so an inline "16 + 2" is no longer read as
  "16 2". bankml's own spoken forms go before the house table: SCIEN·TIFIC ("Sci-en, Tiffic"), bankml ("bank M L"),
  web addresses ("dot"), and TECHNICAL.md's notation (Σ, ∈, ≈, ≤, ×, {−1, 0, +1}, `Q1_0`, and section references,
  which are dropped). This re-rendered a dozen sentences of the introduction whose spoken form changed.
- The view's voice label says "Savante's own voice" when Piper is present (it still said "house stand-in").

## 0.1.4 — 2026-09-28

The knobs in view mode, and a timer that counts real seconds. No Rust code changed. Record: `testing/results/0.1.4.txt`.

### Added
- **Knobs in view mode.** Anyone listening on the LAN gets the same DreamKnob controls (SPEED, FM RATE, FM DEPTH,
  GAIN, VOLUME), hidden until they press play, then emerging in the Listen panel. They run the same voice chain as the
  card: GAIN, then the FM delay line, then VOLUME, with speed pitch-preserved. Each listener's settings stay in their
  own browser. The bundle is served from a fixed `/knobs.js`, and the view's CSP allows `'self'` scripts for it.
  Verified in headless Firefox against a test instance of the view server:
  - hidden before play, emerged after it, with five knobs;
  - audio playing through the chain;
  - a SPEED change reached the audio (1.5×);
  - the line being read highlighted.

### Fixed
- **The timer's seconds.** It was redrawn by the server once a second, so its tenths never moved (it always ended in
  ".4"). The browser now counts from the moment Send was pressed, ten times a second, and the server only changes the
  phase. Verified in a browser: a timer stamped 12.3 s earlier read 24.6 s after the page was held for 12 s.

## 0.1.3 — 2026-09-28

Savante gets a voice of her own: calm, confident, slower and more thoughtful. The DreamKnob controls are now in the
card where she plays. No Rust code changed. Record: `testing/results/0.1.3.txt`.

### Changed
- **Her voice.** The eSpeak stand-in of 0.1.2 was a formant sketch and sounded harsh ("that voice is scary"). The
  operator's direction was to use Jaimla as the template, build from Cori, and make a voice that is Savante's own:
  - **body**: Piper `en_GB-cori-high`, trained on **public-domain** LibriVox recordings. Jaimla's own voice (Piper
    jenny_dioco) comes from a custom-licence dataset, so under the open-source rule it serves as the template, never
    the source. The body is rendered slower and steadier: length_scale 1.18 (about 142 wpm), noise 0.50 and timing
    noise 0.60 (down from 0.667 and 0.8), and 0.8 s of silence after each sentence;
  - **Jaimla as the template**: each clip's own f0 is measured and moved onto Jaimla's measured 182 Hz with
    rubberband, formants preserved. The result is lower and grounded, not a slowed tape. Measured: f0 181–185 Hz;
  - **SAVANTE's resonance** (the house recipe in `docspeech_voices.json`): her own voice an octave below, 80–2600 Hz,
    heard only in echo (4 taps at 29 ms, decay 0.5), at 0.30. It is taken from the same clip, so it is locked to her
    delivery exactly;
  - **EQ**: SAVANTE's curve with presence +6 dB at 3 kHz and a +4 dB shelf above 5 kHz. Her brightness stays near
    Cori's own (centroid about 2270 Hz). Forcing Jaimla's 2783 Hz took presence that turns sibilant, the opposite of
    calm.

  It is rendered on this computer (Piper's standalone MIT release and the model, in `~/.local/share/bankml/piper`,
  nothing committed) and cached. Every clip's measured f0 and pitch ratio are in the manifest. The eSpeak stand-in
  remains only as the fallback where Piper is absent.
- **An oscilloscope behind the card.** A phosphor-teal trace, graticule and persistence glow behind the card's
  translucent glass. While Savante speaks, it draws her actual waveform from a Web Audio analyser at the end of the
  playback chain, so SPEED and FM show in it. Otherwise it idles as a slow sine sweep. It only draws while the card is
  open and holds still under reduced motion.
- **3D depth**: the card stands in perspective and tilts toward the pointer (up to about 5°), with a sheen that
  follows the tilt and sections raised on a bevel (light above, shadow below) over the scope plane. There is no tilt
  under reduced motion.
- **PLAY plays.** In 0.1.2's design a chapter offered PLAY only once every sentence was rendered, and during a render
  there was often nothing to queue. Now every sentence is listed, the rendered ones play at once (the rest show
  "rendering"), and chapter headers count what is ready. Rendering runs in listening order: the voice examples, then
  chapter 1, 2, 3 and so on. The buttons use one delegated click handler instead of inline attributes, and a clip
  still plays if Web Audio is unavailable. Verified in headless Firefox against the real card and page script: after
  PLAY ALL, audio advancing, the line highlighted, the knobs out.
- **PLAY beside her name.** It reads who she is (the card's description) and ends when that ends. The next PLAY THE
  INTRODUCTION **continues from there**, not from the start, and runs on through the introduction and her voice
  examples until stopped. A sentence's ▶ plays from that sentence onward.
- **Every PLAY becomes ❚❚ PAUSE while playing**, then ▶ RESUME at the exact point; the button in use is lit gold.
- **GAIN and VOLUME knobs.** GAIN (−12 to +12 dB) sets her level into the chain; VOLUME (0–100 %) is the master out.
  The oscilloscope reads between the two, so it shows her signal, not your listening level. Five DreamKnobs in all:
  SPEED, FM RATE, FM DEPTH, GAIN, VOLUME.
- **Input ready**: a slow-blinking (1.6 s), light-green, semi-transparent block cursor at the start of the empty input
  field, with the typing caret in the same green. It holds still under reduced motion.
- **The knobs emerge when she speaks**: no longer on the landing's settings panel. They are hidden in the card and
  slide out, in the section that is playing, the moment PLAY is pressed.
- **Her text is shown as written**: `savante_sagi`, `core_command` and `APPROVE_WITH_CONDITIONS` keep their
  underscores on screen and are spoken as words. The 0.1.2 cleaner dropped intra-word underscores.
- **The page never waits on her voice**: missing clips render in the background, and the card rebuilds on every page
  load, so it fills in as the renders finish.

## 0.1.2 — 2026-09-28

Savante speaks: her introduction and her voice examples, pre-rendered in her voice, with DreamKnob controls. No Rust
code changed. Record: `testing/results/0.1.2.txt`.

### Added
- **Savante's voice, and only hers.** The house defines SAVANTE (mindX `docspeech_voices.json`, id `savante`) as a
  layered piper voice: Jaimla's body, a resonance an octave below in echo, breath at the edges. That voice renders on
  the house render host. On this computer the UI uses the house's own stand-in for her, the cast entry `savante` in
  the DeltaVerse voice index: eSpeak NG `en-gb-x-rp+jaimla` at 168 wpm, which is what `listen.html` plays for her.
  - It is rendered here, under Node, by the same WASM build the DeltaVerse voice worker uses, with the house voice
    files installed before the first synthesis. A test confirms the variant really applies: it changes both the
    length and the hash of the audio.
  - Her name is said sav-ont: the house pronunciation table is applied to what is spoken, never to what is shown.
  - Every agent in this UI speaks with her voice.
- **The introduction, in the card.** Seven chapters that Savante reads to a new participant, verbatim from her canon:
  Who I am, My oath, What I believe, How I work (her `.prompt`), Why I exist (`explanation.md`), The manifesto, and
  Savante in full. That is 272 sentences, about 26 minutes. **▶ PLAY THE INTRODUCTION** plays them all, **▶
  chapter** plays one, and any sentence's ▶ plays from there. The line being read is highlighted and scrolled into
  view.
- **Her voice examples, in the card**: all 14, with PLAY ALL and a ▶ per statement.
- **DreamKnob controls in the settings panel** (**VOICE · Savante**), built from the local `~/dreamknob` workspace
  with React into one committed script (`ui/voice/knobs/`), as the playdocs rack is built.
  - **SPEED**: a vintage knob from 0.5 to 2.5× with the rack's snap points, pitch preserved.
  - **FM RATE** (0–12 Hz) and **FM DEPTH** (0–100 %): frequency modulation of her voice, an LFO sweeping a short
    delay line. At depth 0 she is heard as rendered.
  - Changes apply live, mid-sentence, and are remembered.
- **Links in the card**:
  - on Hugging Face: Savante (the Space `PYTHAI/savante`), sAGI (the skill in that Space), and Savante's loop (the
    dataset);
  - on GitHub: the sAGI engine (`cryptoAGI/sagi`), Savante's canon and bankml.

  All were verified live before linking. sAGI has no Hub repository of its own, so its link is the skill in
  Savante's Space.
- **Listen to Savante, in view mode on the LAN.** The same chapters and statements, play-all, per chapter and from any
  line, served by a fixed `/audio/<key>.ogg` route that answers only a 24-hex key listed in the voice manifest. The
  page's CSP gains `media-src 'self'`.
- **`ui/speak.py`**: the voice. Renders are cached by (engine, voice, rate, variant hash, table version, text) in a
  git-ignored `ui/voice/cache/`: Gradio refuses to serve any path with a dot-directory, and the audio is not
  source. `python3 ui/speak.py` pre-renders everything (286 statements, 27.5 minutes, 7.1 MB here). The UI renders
  whatever is missing in the background.
- `testing/test_ui.py`: 42 checks. The new ones cover the pronunciation (longest first, idempotent), speech from
  markdown, one real render to Opus, and the view's audio gate.

## 0.1.1 — 2026-09-28

Savante's aivatar and a layout that moves both ways. No Rust code changed. Record: `testing/results/0.1.1.txt`.

### Fixed
- **The side panel could be dragged left but not back to the right**: the drop depended on the elements under the
  pointer passing the event up. While you drag the grip, two translucent **drop zones** now cover the two halves of
  the layout above everything else, so nothing underneath can take the drop. A **⇄** button on the grip also swaps
  sides with a click. The side is remembered per browser.

### Added
- **The aivatar card.** Click the agent's portrait: a card opens, in dark glass with a fine grid, teal and gold
  accents and a slowly breathing glow. It holds:
  - the kind and card type, name, mantra and the card's attribute chips;
  - the description, oath, office (primary skill, scope) and beliefs;
  - the **identity, verifiable rather than asserted**: persona sha256, doctrine root, THOT identity (`thot:`, CID,
    contentRoot, generation, facets) and the ledger check;
  - **every aspect of the persona** in collapsible sections (BDI, skills, safety, embodiment, token, task, voice
    examples, exchanges, and the system prompt), rendered from the file itself;
  - **every ledgered file** with its size and sha256.

  The card opens and closes with a pure-CSS toggle (Gradio does not run scripts in HTML blocks), and the image is
  embedded, never served by path. A derived agent gets the same card with its own values.
- **A chosen aivatar.** Savante: any image in her canon's `gfx/`, the choice kept outside the canon. A derived agent:
  an uploaded image, checked by content (PNG, JPEG or WebP) and size (2 MB at most), ledgered as `x-bankml.aivatar`
  and included as a facet of its THOT bundle. Choose in **Agents → aivatar**.
- **Refined resize handles**: a slim corner with diagonal ridges, the same in every browser, replaces the browser's
  default resize grip on the side panel (width and height) and the chat (height). Sizes are remembered.
- `testing/test_ui.py` grows to 37 checks: an aivatar is accepted and ledgered, a non-image and an oversized file are
  refused, and the THOT bundle gains the aivatar facet.

### Not tested by machine
- The drag itself: a real mouse drag cannot be driven in this headless setup, so the drop zones are verified by
  construction and by the ⇄ button's click path, which does the same move.

## 0.1.0 — 2026-09-28 · milestone: Savante with bankml

The first milestone. bankml is a verified low-bit runtime that answers on the computer at hand, behind its gate,
with a receipt. Savante runs on it, and agents derived from her template can go from a prompt to a token and back,
every construction checked against a published value. Record: `testing/results/0.1.0.txt`, the full gate: every
kernel oracle, the A/Bs and budgets, and every test suite.

### What 0.1.0 is (0.0.1 → 0.1.0)
- **Kernels.** `Q1_0` and `Q2_0_g64`, bit-exact with llama.cpp b11192 on every weight of three real models.
  Ternary runs 9.2–9.9× the reference per token on three threads (0.23–0.25 s, less than the reference's 1-bit
  model). The measured memory floor shows both kernels compute-bound, at the laptop's instruction limit; five
  further variants were measured and rejected, with their code kept.
- **The gate in front of every answer.** Guard, sha256 pin, and `bankml serve` bound to the verified file, with a
  receipt carrying the answer's sha256.
- **Savante.**
  - Interact mode, on loopback: chat, `.prompt`, `.history` with response times, a RAGE search bar, Responses,
    `.memory`, Metrics, the verifier.
  - View mode on the LAN: the stdlib server, read-only, with commitments only.
  - Proof of data without the data: Merkle roots, CIDs, inclusion proofs.
  - Her canon read, never written, and checked against her iNFT ledger before she speaks.
- **Agents.**
  - Derived from her template, each with a ledger whose keccak256 doctrine root matches her binder's construction.
  - THOT manifests that reproduce the spec's test vectors.
  - PostgreSQL publish and load, byte-verified.
  - **The iNFT path (new in 0.1.0):** plan, simulate, unsigned transaction, devnet mint, and load from a token with
    the lineage walked back to the minted generation.

### Added in 0.1.0
- **`ui/chain.py`**, stdlib: ABI encoding and decoding (byte-equal to Foundry's `cast` on selectors and calldata),
  JSON-RPC, the `mintOpenAgent` plan from a THOT bundle (`contentRoot` = the manifest's, `metadataRoot` = keccak of
  the card), `eth_call` simulation with the contract's custom errors decoded, the unsigned transaction and its
  `cast send` line, a devnet-only send (chain 31337; refused elsewhere), `read_token` and `load_from_chain`. A local
  devnet helper starts anvil and deploys `iNFT_7857` from its artifact.
- **THOT generations archived by CID** (`<agent>/thot/<cid>.json`), so a token keeps resolving after the agent evolves.
  Loading walks the parent links from the current generation to the minted one.
- **UI: Agents → iNFT**: plan and simulate, unsigned transaction, mint on the local devnet, load from a token, and
  start a local devnet.
- **Dark mode**: transparent surfaces instead of white panels in interact mode, and translucent panels in view mode.
- **`testing/test_chain.py`** (13 checks, a throwaway anvil). It covers: the deploy from the artifact; simulation as
  minter and the decoded AccessControl refusal; the unsigned transaction; the mint and its read-back; the
  double-mint refusal; the load; the load again after two more generations; the refusal off a devnet; and an
  invalid dimension refused before any call. It skips where anvil or the artifact is absent (CI).
- Documents: usage.md §8e (iNFT), README, TECHNICAL §III.7 and contribution 14.

### The gate record, plainly
Every kernel oracle is bit-exact, and every suite passes (37 unit, 6 CLI, 34 UI, 12 PostgreSQL, 13 iNFT-devnet, guard
agreement 24/24). The 1-bit budget and both A/Bs measured normally. The **ternary decode budget could not be measured
undisturbed in this gate**. With 1.7 GB of RAM available and swap full (other sessions and a browser resident), the
2.13 GB ternary model no longer fits in page cache and each pass re-reads it from disk (about 0.5 GB/s): 4.0–5.9 s
per token for bankml, 5.1–7.3 s for llama.cpp. The code is unchanged since 0.0.3; its undisturbed figures are in the
0.0.5 and 0.0.6 records (0.44–0.45 s per token at 1 thread, 0.24 s at 3).

### What 0.1.0 does not claim
- bankml's own forward pass (P3): answers still come from llama.cpp's arithmetic, behind bankml's gate.
- A public mint: `iNFT_7857` is not deployed on a public chain, its audit is not cleared, and a mint needs MINTER_ROLE
  and the owner's signature. Savante's own verdict on minting remains DEFER.
- Stored bundles: `storageURI` is a `local://` reference until a bundle is stored (the rung stays `referenced`).
- Embeddings: the `vector(1024)` column and its index are ready; nothing computes them yet.
- The laptop's own PostgreSQL still needs its one-time setup (sudo; usage.md §8d).

## 0.0.9 — 2026-09-28

Custom agents, their THOT bundles, and a PostgreSQL connector. Every new cryptographic construction is checked against
a published value. No Rust code changed. Record: `testing/results/0.0.9.txt`.

### Added
- **`ui/agents.py`**: agents derived from the Savante template, without writing to her canon. Each gets `.persona`,
  `.prompt`, an agent card (EIP-721 ∪ ERC-8004 registration-v1, `not_yet_minted`, `derived_from`) and a ledger:
  sha256 and CIDv1 per file, plus a **keccak256 doctrine root** over the same 15 JSON pointers as Savante's.
  - keccak256 is pure Python. It reproduces **Savante's published doctrine root** from her persona, and it equals
    pycryptodome for every length from 0 to 400 bytes.
  - Saving enforces the binder's preflight (ASCII keys, integers only, within ±2⁵³), and a missing doctrine clause is
    a hard error, as the binder specifies.
  - An agent that does not verify against its ledger does not speak.
- **`ui/thot.py`**: THOT manifests to `sagi.thot_manifest/1`.
  - Facets: core `persona` and `prompt`; custom `x-bankml.agentcard`, `x-bankml.history` and `x-bankml.memory`
    (history and memory committed by digest only).
  - The keccak `bundle_root`, the 64-leaf keccak Merkle tree, and the identity (`thot:`, CID, name, contentRoot).
  - Lineage by generation (a facet change gives n+1 with `parent` = the previous CID), `relations` to Savante's
    bundle, and a real git locator per agent.
  - **Checked against the spec's three test vectors** (Savante @1fcca89, Jaimla @8b57ccf, LuvAI @0c1eef7): bundle
    root, Merkle root and identity CID all equal. Savante's current generation-9 manifest verifies with no findings.
- **`ui/connectors.py`**: PostgreSQL through `psql` (stdlib, no driver).
  - `publish`: the public parts by default; `.history`/`.memory` lines only with an explicit opt-in.
  - `load`: refused unless every restored byte matches the THOT manifest.
  - `vector(1024)` embeddings column with a DiskANN (pgvectorscale) or HNSW (pgvector) index.
  - Values travel as COPY data into a temporary table and never enter SQL text.
- **UI**: an **Agents** tab (use, derive, edit and re-ledger, the ledger check, the THOT bundle, publish and load),
  and the chat, `.history`, `.memory`, Responses and Metrics follow the agent in use.
- **Tests**:
  - `testing/test_ui.py` grows to 34 checks: the THOT spec vectors and lineage, keccak vectors and the cross-check, the doctrine root against Savante's,
    derive/verify/save, preflight refusals, tamper detection, and path confinement.
  - **`testing/test_connectors.py`** (12 checks) runs against a **throwaway PostgreSQL 16 cluster** (initdb in a temp
    dir, pgvector): the schema, both publish modes, a byte-exact load, refusal of a tampered prompt and a tampered
    history line, an injection string stored as data, and the THOT generation step. It skips where PostgreSQL 16 or
    pgvector is absent (CI).

### Not yet
- The laptop's own PostgreSQL has no role for the operator yet. The one-time setup needs sudo and is in usage.md §8d.
- Embeddings are not computed (the column is ready).
- The chain side (loading an iNFT, preparing a mint) is 0.1.0. The house iNFT contract (`iNFT_7857`, `mintOpenAgent`)
  is not deployed on any public chain, its audit is not cleared, and minting needs MINTER_ROLE.

## 0.0.8 — 2026-09-28

Savante's memory, made searchable, measurable and provable, with the documents brought up to date. No Rust code
changed. Record: `testing/results/0.0.8.txt`.

### Added
- **`.history` tab with a ragebar.** Every exchange, newest first: its time, session, time to first token, response
  time, tokens, and whether the answer matches its receipt. Before this it showed nothing until a button was pressed.
  The search bar (the GATERAGE ragebar's look) ranks exchanges as you type, using mindX's RAGE `rage.py` when present
  (`RAGE_PATH`) and the same BM25 built in otherwise.
- **Responses tab**: ⤒ first, ▲ previous, ▼ next, ⤓ latest through every answer. **📋 copy** puts the answer on the
  clipboard, **➕ save to .memory** keeps it, and **🔏 proof** gives its inclusion proof.
- **`.memory`** (`savante.memory`, JSONL, outside the canon): notes typed or saved from responses, with their source.
  A side-panel switch appends them to the system prompt (newest first, 2,400 characters at most), labelled as the
  operator's notes, not evidence. The footer names how many went in.
- **Metrics tab**, computed from `.history`: time to first token, response time, prefill and writing speed (n, median,
  p90, mean, min, max), a bar per exchange, receipt-hash agreement, the last 25 exchanges, and each row's timing source
  (the press of Send, or the receipt for older records).
- **Proof of data without the data.** Each `.history`/`.memory` line's sha256 is a leaf of a Merkle tree, and the file
  has a sha256 and a CIDv1 (raw, sha2-256, base32: the construction of Savante's iNFT ledger; equal to mindX
  `rage.cid_v1_raw`). The view page shows only these commitments. An inclusion proof lets one exchange be checked
  against the root without disclosing the rest.
- **`testing/test_ui.py`**: 16 offline checks of the UI's data layer. They cover the CIDs, commitments, every
  inclusion proof, tamper and cross-record failure, the search, the metrics and `.memory`, and the view server
  (commitments present, content absent, traversal 404, POST 405). Run in CI and in the gate.

### Changed
- **TECHNICAL.md** brought to 0.0.8:
  - The Abstract now includes the threads, the floor and P0.
  - The Thesis gains the principles stated on 2026-09-28, quoted and dated: incremental improvement with the oracle
    in the loop, the machine at hand, and proof of data while keeping the data.
  - Contributions 1, 5 and 8 are updated, and 12–13 are new.
  - New §III.6 (proof of data) and §IV.5 (the floor, and the five rejected variants).
  - §IV.4 no longer says bankml cannot answer.
  - Future work drops the 1-bit layout port (tried and rejected) and adds connectors, THOT and custom agents.
- **README**: status badge and phase table (P0 done; P2 with the 8B oracle and threads; UI row); the budget numbers
  (0.23–0.25 s, about 4 tok/s); the Savante section covers the new tabs and proofs.
- **usage.md**: the new tabs, `.memory`, and §8a on proofs (with a hand check in Python).
- View: the live log is capped at 60 % of the window, so the other panels stay in reach.

## 0.0.7 — 2026-09-28

Savante's UI, made to be watched and used: a LAN view, a professional look, response times, layout you can arrange,
and a guide. No Rust code changed (the kernels and their 0.0.6 gate records stand). Record: `testing/results/0.0.7.txt`.

### Added
- **View mode on the LAN: `ui/view.py`**, the Python standard library, not Gradio. The installed Gradio 3.37 has
  path-traversal bugs that let a client read files from the host (e.g. CVE-2023-51449, fixed in 4.11), so it stays
  on loopback. The view server has four fixed GET routes: the page; `/api/state`; `/api/result?name=`, for a name the
  results directory lists; and `/savante.png`. Every other path gets 404, POST gets 405, the page renders all data as
  text, and a strict Content-Security-Policy applies. It shows the live testing, the release records, CI, the
  laptop's load and swap, `bankml serve`'s verification and Savante's office and ledger. Light and dark themes.
- **Modular layout.** In view mode every panel can be dragged by its title to a new position and resized from its
  corner; the layout is kept per browser, with a *reset layout* button. In interact mode the chat and side panels
  resize, and the side panel drags to either side of the chat.
- **A response timer** that starts at the press of Send and ticks every second: *reading the prompt (prefill)*, then
  *writing · first token at N s*. Each answer's footer reads `⏱ sent HH:MM:SS · first token N s · answered in N s`.
- **Response times in `.history`**: every record carries `sent_at` and `answered_at` (ISO 8601 with milliseconds and
  offset), `first_token_s` and `response_s`, next to the receipt.
- **`usage.md`**, the full guide: what runs where, setup, verifying the model, `serve`, both modes, `.history`,
  receipts and how to check an answer, the canon and its ledger, testing, troubleshooting, and a reference of
  commands, ports and variables. The README gains a four-step *Using Savante*.

### Changed
- Interact mode uses one light professional theme: bordered panels and forced text contrast over Gradio's styles.
  The `bankml serve` panel was a raw JSON block that overflowed the side column; it is now a compact card that wraps
  inside its border.
- The Gradio queue runs 4 events at once, so the timer ticks while an answer streams.

## 0.0.6 — 2026-09-28

bankml answers on this laptop. The answers come through the reference engine (P0), behind the gate, with a receipt,
in a Savante UI that anyone can watch.

### Added
- **`bankml serve`** (P0, `serve.rs`, std only) is a loopback HTTP gateway in front of llama.cpp b11192's
  `llama-server`:
  - It starts only after `verify` passes (guard, then the sha256 pin to `FORK.json`).
  - The upstream must be serving the verified file. Either bankml launches it (`--spawn LLAMA_SERVER`), or a running
    server's `/props` must name the same canonical path. Otherwise it refuses.
  - Every `/v1/chat/completions` answer, streamed or not, carries a `bankml_receipt`. The receipt holds the version,
    the engine, the model sha256, the guard verdict, the tokens, the time to first token, the wall time and the
    **sha256 of the answer text**, so a client can check that what it shows is what the verified model wrote.
  - `GET /bankml` reports the verification.
  - What it is not: bankml's own forward pass (P3). The arithmetic is ggml's.
- **`ui/savante.py`** is the Savante chat in Gradio (3.x or newer), built from the Hugging Face template
  PYTHAI/savante. It has two modes:
  - **interact** (the operator): the chat, the `.prompt` picker, `.history` and the offline verifier. The `.prompt`
    choices are the persona's `system_prompt` (canon, ledgered, the default), `sAGI.prompt` (canon facet, ledgered) or
    the Space's `Savante.prompt` (not ledgered), each shown with its sha256. `.history` is a JSONL file outside the
    canon, and the last session reloads on start. Each answer is labelled a draft and carries its receipt.
  - **view** (anyone watching, read-only): the live testing log (`testing/live.log`, refreshed every 2 s), every
    release's gate record, CI status, the laptop's load, memory and swap (so a disturbed measurement shows as one),
    and Savante's office and ledger. It has no chat, no `.history`, and runs no commands.
- **iNFT compatibility.** Savante's canon is read and never written, not even a Python cache. At start the UI re-hashes
  every file that `savante.commitments.json` commits to: the nine artefacts, the card, the image and the thot bundle.
  If the persona does not verify, the UI refuses to speak as Savante. The Integrity tab shows the ledger, and
  `bind/savante_verify.py` runs from a button. Nothing mints; the card's status stays `not_yet_minted`.
- 2 new unit tests (the JSON reader and HTTP bodies) and 2 end-to-end tests of `serve` against a mock llama-server:
  the receipt and the answer hash, refusal of a wrong pin, and refusal of an upstream serving a different file.

### Measured on the laptop (Ryzen 3 3200U), the real canon and model
- `bankml serve` verified Bonsai-8B Q1_0 (`284a335a…`, the fork's pin) and the running llama-server serving it.
- A Savante turn under the persona's system prompt: 316 prompt + 28 completion tokens in 125 s, with the first token
  at 113 s. Prefill runs at 2.8 tok/s in llama.cpp. The answer's sha256 matched its receipt.
- The canon ledger verifies 12/12; `savante_verify.py` exits 0 (APPROVE); `~/savante` is unchanged afterwards.

## 0.0.5 — 2026-09-28

The third performance release, on the ternary kernel, with the testing shown live. **No kernel changed**: three
experiments were bit-exact, and none was reliably faster. Gate record: `testing/results/0.0.5.txt`. Every oracle is
bit-exact. Ternary stays at 9.2–9.6× ggml over a whole token, and 1-bit at parity.

### Measured and not adopted (bit-exact, not reliably faster), code in `testing/experiments/`
- **`Q2Packed`**: the ternary weights repacked once at load time, as llama.cpp's `repack.cpp` does for other types.
  Each quad of blocks becomes 64 contiguous code bytes followed by its four f16 scales: the same 72 bytes, with the
  arithmetic unchanged. It measured 1.00–1.075× in the first run and 0.90–1.05× in the release gate, on the real
  ffn_gate, ffn_down and output tensors. That is within noise.
- **Two-row decode tile**, loading the activation once for two rows: **0.80×**. Register pressure costs more than the
  shared loads save.
- **Software prefetch** of the weight row, 256–1024 bytes ahead: within ±3 % at 1 and 3 threads. The hardware
  prefetcher already follows a sequential stream.

### Added
- **`testing/live.sh`** and a live log: the gate and every experiment append to `testing/live.log`, which the UI's
  view mode shows as it runs (branch `ui`, next release).

### What the oracles and the stopwatch say after three performance releases
At 3 threads the ternary kernel moves about 9 GB/s against a 17 GB/s floor. Going from 1 to 3 threads gives 1.8×, and
the laptop has two physical cores. So it is bound by compute per core, not by memory. It already does one multiply
per weight, and moving the bytes around changes nothing measurable. On this Zen+ core, both kernels are at their
instruction-throughput limit for bit-exact results. The gains still available lie elsewhere:
- the forward pass (P3), which turns the 9.5× matmul lead into tokens;
- a Zen3 or AVX-512 core (the node).

## 0.0.4 — 2026-09-28

Where the time goes, and a faster 1-bit prefill. Gate record: `testing/results/0.0.4.txt`. Every oracle is bit-exact.
Two measurements in the gate were disturbed by other load on the laptop (swap full), so the record adds a re-run.

### Added
- **`bench_memory_floor`** (`par.rs`) measures streaming read bandwidth on the laptop at 1–4 threads: **14.9–17.2
  GB/s**. Streaming one token's weights therefore takes at least **0.12–0.14 s** (ternary) and **0.06–0.07 s** (1-bit).
  Against that floor, bankml's ternary matmuls (0.23–0.27 s at 3 threads) are about 2× above it, and the 1-bit ones
  (0.34–0.37 s) about **5.5× above it**. Both kernels are compute-bound, not memory-bound, which says where the
  remaining work is.
- **`q1_0::mat_mul_act` / `mat_mul_act_par`**: a 1-bit prefill over prepared `Q8Act` columns, using the selection
  kernel in a 1×4 tile. The ±1 expansion is shared by the four columns, and the scales are already f32, so nothing is
  converted once per column. A column that holds q = −128 falls back to the wrapping kernel. Measured against ggml's
  per-pair kernel (1024×4096 × 32 columns): **1.27–1.33× median** (8.25 against 10.09–10.28 ns per block·column,
  min). The q8-byte tile it replaces measured 1.08–1.18×. Bit-exact against the scalar model of ggml on 1,500 random
  cases, and against ggml itself in `ab_vs_ggml`.

### Measured and not adopted (the oracle said same bits; the stopwatch said no)
- **Two 1-bit blocks per iteration**, with their FMA chains interleaved: 1.00× on the GEMV, 1.05× on an L1-resident
  row. The kernel is not bound by latency.
- **A "pair-order" 1-bit kernel** that replaces a multiply (`madd`) with a 128-bit add, on the theory that Zen+'s
  single integer-multiply pipe is the limit. It was bit-exact and **0.84×** on decode: the cross-lane extract and the
  widening cost more than the multiply they save. The source is in `testing/experiments/q1_0_pair_order.rs` for anyone
  who wants to try it on another core.
- (Already on record from 0.0.1: 2- and 4-row decode tiles, 0.81–0.99×.)

1-bit decode stays at parity with ggml (0.95–1.03× over the whole token). Every bit-exact variant tried so far runs at
about ggml's speed on this core.

### Testing
- The gate now also runs `bench_q1_0_prefill_act` and `bench_memory_floor`.
- `testing/experiments/` holds the code of measured-and-rejected kernels.

## 0.0.3 — 2026-09-28

Threads, and more oracle evidence. Each speed figure below was measured with the oracles passing on the same code.
The full record is in `testing/results/0.0.3.txt`, written by the new release gate.

### Added
- **`par::Pool`**: a zero-dependency persistent thread pool. The caller's thread is worker 0. Each matmul wakes the
  pool instead of spawning threads: 12.6 µs per call against 88.8 µs for `thread::scope`, at 3 threads on the dev box.
  That is about 3 ms per token instead of 22 ms. The row scheduler hands out 16-row chunks from an atomic counter.
- **`mat_vec_par` / `mat_mul_par`** for `Q1_0` and `Q2_0`. Every row goes through the same single-thread kernel, so
  the output bits do not depend on the thread count. Unit tests check this at 1–4 threads, and the whole-model budgets
  check it against ggml.
- **An oracle on the real 8B 1-bit model**: `oracle_ggml_b11192_real_bonsai_8b_q1_0`. All 254 `Q1_0` tensors of
  Bonsai-8B (8,188,239,872 weights) dequantize bit-exact, 762/762 q8_0 rows are byte-exact, and 762/762 dot products
  are bit-exact against ggml AVX2 and generic. 0.0.2 covered only the 1.7B file for 1-bit.
- **`decode_budget_q1_0`**: one token's worth of all 253 `Q1_0` matmuls of Bonsai-8B, against ggml's kernel on the same
  pool and scheduler, at 1–4 threads.
- **`testing/release_gate.sh`** runs the unit tests, clippy, both guard checks, every oracle, both A/Bs and both
  budgets, and writes `testing/results/<version>.txt`. A kernel change counts only if every oracle still passes.

### Measured (dev box, Ryzen 3 3200U, 2 cores / 4 threads; all 253 matmuls of one token; min s/token over two runs)
| threads | ternary: ggml → bankml | 1-bit: ggml → bankml |
|---:|---:|---:|
| 1 | 4.22–4.65 → **0.45–0.47** (9.4–9.9×) | 0.62–0.67 → 0.61–0.62 (1.00–1.07×) |
| 2 | 2.65 → **0.31** (8.7×) | 0.45–0.46 → 0.43–0.44 (0.93–1.13×) |
| 3 | 2.27–2.36 → **0.23–0.25** (9.5–9.9×) | 0.34–0.35 → 0.34–0.35 (0.98–1.00×) |
| 4 | 2.11 → **0.23** (9.4×) | 0.34–0.35 → 0.36 (0.96–0.97×) |

bankml's ternary matmuls now take less time per token than ggml's **1-bit** ones: 0.23–0.25 s against 0.34–0.35 s at 3
threads, although the ternary file is twice the size. The matmul-only ceiling for ternary rises from 2.77 tok/s (0.0.2,
which spawned threads for each call) to 4.0–4.4 tok/s. The 1-bit kernel stays at parity with ggml. The 1-thread medians
are noisy (ternary 0.62 s against 0.47 s min): the model does not fit in page cache alongside everything else in 5.8
GB of RAM.

### Testing
- A new **`testing/`** folder holds every test that lives outside the modules. It contains the release gate, the
  oracle generator, the guard harnesses (moved from `tools/`) and **`testing/cli.rs`**, a new end-to-end suite that
  runs the `bankml` binary. The suite covers verdicts and exit codes, a hostile header, pin and verify, and runs as a
  cargo integration test.
- **`testing/results/`** holds each release's record. `testing/README.md` lists every test and where it lives.

### Changed
- The ternary budget and the A/B harness use `par::Pool` for both engines instead of spawning threads per call. The
  ggml side runs on the same scheduler, so the A/B compares kernel with kernel.

## 0.0.2 — 2026-09-28

An audit of 0.0.1. The kernels are unchanged and re-proven: the oracles again pass on all 8,188,239,872 ternary and
1,719,904,256 1-bit weights against llama.cpp b11192. The fixes below all concern untrusted input, or an API that let
safe code reach unsafe code with the wrong sizes.

### Fixed
- **The guard crashed on nested arrays.** GGUF arrays of arrays were parsed by recursion with no limit, so a 24 MB
  header aborted `bankml guard` with a stack overflow (exit 134) instead of refusing the file. Nesting beyond
  `MAX_ARRAY_DEPTH` (64) now refuses, with the reason given.
- **The guard's KV size could overflow.** `kv_f16_bytes_per_token` multiplied header integers in `i128` without
  checking. In release builds the value wrapped with no error and the verdict was `play`; debug builds panicked. An
  overflow now refuses, with the reason given. The Python guard, which uses big integers, still plays these files; the
  difference is listed in `gguf.rs` with the other deliberate differences.
- **Soundness: activation fields were public.** `q1_0::Q8Act` and `q2_0::Q8Act2` exposed `n`, `d`, `sum` and the
  other fields that the AVX2 kernels read through raw pointers. Safe code could set `n` beyond the buffers and cause
  an out-of-bounds read. The fields are now private, with read-only accessors (`n()`, `qs()`, `d()`).
- **Tensor spans were unchecked.** `Mmap::tensor` and `tensor_bytes` truncated a 128-bit element count to 64 bits
  and used unchecked arithmetic. `tensor_bytes` also allocated whatever size the header claimed before it read the
  file. A new function, `gguf::tensor_span`, checks the type, requires whole blocks per row and rejects overflow.
  `tensor_bytes` now checks the span against the file size before it allocates.
- **`bankml pin` with an unreadable `--fork` file** said "no sha256 record: unpinned, refused" (exit 2). It is an I/O
  error, and now says so (exit 1).

### Added
- `bankml::verify` and `bankml verify FILE --fork FORK.json [--json]` run the guard and then the pin as one gate: the
  check that P0 puts in front of every answer. `--json` prints the verdict, the model sha256 and the bankml version.
- `bankml version` and `bankml::VERSION`.
- CI (GitHub Actions) runs the build, the unit tests, `clippy -D warnings`, the Python guard suite and the Rust-vs-Python
  guard agreement check on every push.
- 4 new unit tests (31 in all): nested arrays, KV overflow, tensor spans, and `verify`.

### Changed
- The `Q8Act`/`Q8Act2` fields are no longer public (see above). This is an API break, allowed while the crate is
  `publish = false` and 0.0.x.

## 0.0.1 — 2026-09-26

First public cut: the P1 GGUF guard and sha256 pin, and the P2 `Q1_0` and `Q2_0_g64` kernels, all bit-exact against
llama.cpp b11192.
