---
name: bankml
description: bankml (github.com/cryptoAGI/bankml, local ~/cryptoAGI/bankml) — the zero-dependency Rust runtime for 1-bit (Q1_0) and ternary (Q2_0_g64) GGUF models, bit-exact against llama.cpp b11192, with `bankml serve` (a verifying loopback gateway that pins the model by sha256 and puts a receipt on every answer) and the Savante UI (interact on 127.0.0.1:7873, view on the LAN at :7874). Use when working on bankml's kernels, oracles, release gate, serve/receipts, the model importer, the Savante UI (chat, .history/.memory, Merkle commitments, agents, THOT, PostgreSQL, iNFT, voice, embeddings), cutting a release, or measuring speed. Triggers "bankml", "bankml serve", "Q1_0", "Q2_0", "Bonsai", "release gate", "oracle", "receipt", "Savante UI", "savante.py", "view.py", "models.py", "speak.py", "console.py", "serve --native", "Ollama API", "JSON schema", "logprobs", "slot".
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, WebFetch
---

# bankml

**Verified low-bit inference for the CPU you already have.** A Rust runtime with no crates: its own GGUF guard, sha256
pin, AVX2 kernels for ggml's `Q1_0` (1.125 bits per weight) and `Q2_0_g64` (ternary, 2.25 bpw) that are **bit-exact
against the compiled llama.cpp b11192 libraries**, a thread pool, its own forward pass (Qwen3 and Llama graphs; Q1_0, Q2_0_g64 and F16 weights; token-identical to
llama.cpp), and `bankml serve`, which answers behind that gate — through llama-server, or with `--native` from bankML's
own forward pass. Authors: Professor Codephreak and Gregory L. Magnusson (cryptoAGI). Licence: `MIT OR Apache-2.0`
(key handling `GPL-3.0-only`; AGPL only walled off — see *Licensing*). This file is published in the repo at
`.claude/skills/bankml/SKILL.md`; paths under `~` and the RAM figures describe the development laptop.

The rule that governs everything: **the same bits first, then the speed.** A result counts only if an external
oracle checked it; a speed-up counts only on code that passed every oracle in the same gate run, and only beyond the
run-to-run noise. Rejected experiments are kept with their numbers (`testing/experiments/`, 0.0.5).

## Where things are

| path | what |
|---|---|
| `bankML/bankml.rs` | `verify` (guard → pin), `Verified` (sha256, arch, name, types); the module list |
| `bankML/gguf.rs` | header parser, the guard (`judge`: the three low-bit traps, hostile headers), `type_name` for every ggml type, `Mmap` |
| `bankML/sha256.rs` | FIPS 180-4, the FORK.json pin scanner, SHA-NI path (0.1.8, 5.5×) |
| `bankML/q1_0.rs`, `q2_0.rs`, `f16.rs` | kernels: scalar models of ggml (`vec_dot_ref`), AVX2 paths, `*_par`; F16 = ggml's two products chosen by shape (`vec_dot_f16` for 1 column, tinyBLAS for ≥ 2) |
| `bankML/par.rs`, `sys.rs` | the zero-dependency thread pool (0.0.3); `/proc` memory, rss, CPU |
| `bankML/tokenizer.rs`, `chat.rs`, `unicode_letters.rs` | tokenizer (`qwen2`, `smollm` pre-tokenizers) and chat templates (per model by sha, `chat::TEMPLATES`), identical to llama.cpp's (4,346/4,346; 317/317 per template) |
| `bankML/forward.rs` | the forward pass: `forward::plan` (arch/type/tied from the header, refuses the rest), `Weights`, `KvCache` (f16, or q8_0 with the Hadamard rotation, 0.3.9), the three ggml attention kernels, `prefill`, `decode`, `logits` |
| `bankML/sampler.rs` | llama-server's whole default chain: penalties, DRY, top-n-σ, top-k, typical-p, top-p, min-p, XTC, temperature, dist (libstdc++ `std::sort`/`partial_sort` ports, mt19937) |
| `bankML/grammar.rs`, `schema.rs` | llama.cpp's GBNF engine, JSON mode's grammar, the content rule; `json_schema_to_grammar` + the chat parser's wrapping, per template |
| `bankML/native.rs` | the engine behind `serve --native`: one slot, llama-server's prompt-cache rule, slots save/restore, logprobs, the context limit; `Registry`, `Residency` (one resident model, verify on every load) |
| `bankML/prompt_cache.rs` | llama-server's host prompt cache (`--cache-ram`; `BANKML_CACHE_RAM`), 0.3.8 |
| `bankML/metrics.rs` | per-answer TTFT, pp/tg tokens/s, grammar time, RAPL joules per token (or `null`); `GET /bankml/metrics`, 0.3.7 |
| `bankML/ollama.rs` | Ollama's API on `serve --native`: `/api/chat` `/api/generate` (NDJSON) `/api/tags` `/api/ps` `/api/show`, options, refusals, `keep_alive`; native only, never proxied; roadmap `docs/OLLAMA.md` |
| `bankML/create.rs`, `convert.rs` | `bankml create` (Modelfile layer over a pinned base; `/api/create` `/api/delete` `/api/copy`); `bankml convert` (safetensors → GGUF F16, byte-identical to llama.cpp's converter) |
| `bankML/serve.rs` | the gateway: `/bankml`, `/bankml/usage`, `/bankml/metrics`, `/health` `/props` `/v1/models`, `/v1/chat/completions` with receipts; `--native` also answers `/tokenize`, `/apply-template`, `/slots` |
| `bankML/gpu/` | the video-card component: `mod.rs` registry, `vulkan.rs`, `hf.rs` (rented, listed only), `spirv.rs`, `kernels.rs`, `compute.rs`, `worker.rs` (`BANKML_GPU_LIMIT`, per-shape calibration) |
| `bankML/train/` | mindXtrain in Rust: `script.rs` (author), `imprint.rs` (score) |
| `bankML/main.rs` | CLI: `guard`, `sha256`, `pin`, `verify`, `serve [--native]`, `generate [--sample] [--json]`, `tokenize`, `chat-template`, `create`, `convert`, `gpu [--remote\|--verify]`, `usage`, `version` |
| `capi/` | the C API (`libbankml.so/.a`, `capi/include/bankml.h`; a second workspace member, `bankml-capi`) |
| `sAGI/savante.py` | interact UI (Gradio 3.37, **loopback only**, refuses `--host 0.0.0.0`, trusted-Host middleware) |
| `sAGI/view.py` | the LAN page (stdlib, fixed routes only, CSP, cached state) |
| `sAGI/console.py` | the bankML console (0.3.7, loopback :7875): Interaction, Admin, Logging, Infotags; persona `sAGI/personas/bankml.persona` |
| `sAGI/models.py` | importer: catalogue / Hugging Face / Ollama; sha256-pinned; OSS licences only; carrier switch with rollback; `native_for` picks the engine |
| `sAGI/embed.py` | bge-m3 via local Ollama (the model mindX uses); BM25 + RRF search; memory-guarded |
| `sAGI/speak.py` | Savante's voice (Piper Cori → Jaimla's 182 Hz), introduction + reading, `Savante.opus` exports |
| `sAGI/agents.py`, `thot.py`, `connectors.py`, `chain.py` | custom agents (keccak doctrine root), THOT manifests, PostgreSQL, iNFT (devnet only) |
| `testing/` | `release_gate.sh`, `cli.rs`, the oracles (`*_oracle.py`, `*.cpp`), the Python suites, `decode_ab.py`, `pinned.sh`, `results/<version>.txt`; every file and gate stage in `testing/README.md` |
| `tools/` | `cards.py` (README speed cards from a gate record), `makecards.py` + `seo.py` (share cards and their audit), `bashmoji.sh` |
| docs | root: `README.md` (the front door: a Documentation router and the Releases table), `CHANGELOG.md` (the source of truth), `LICENSING.md`; `docs/`: `why-bankml.md`, `thesis.md`, `usage.md`, `install.md` (every option and variable), `playback.md`, `TECHNICAL.md`, `oracles.md`, `PERFORMANCE.md`, `research.md`, `OLLAMA.md`, `CAPI.md`, `embedding.md`, `huggingface.md`, `BUILD_HISTORY.md`, `TODO.md`, `modules/` (one page per module), `llms.txt` (the method for agents), `index.html` (the docs reader), `cards/` |

Models: `.models` → `~/mindX/bankml/.models` (Bonsai-1.7B/8B Q1_0, Ternary-Bonsai-8B, imports). Pins:
`~/.local/share/bankml/forks/*.FORK.json`. Engine: `~/sAGI/bonsai/llama-b11192/llama-server`. UI state:
`~/.local/share/bankml/savante/`. Canon: `~/cryptoAGI/savante` (moved 2026-09-29; `~/savante` is a link to it) (**read-only, never write, not even a pycache**).

## Install

`./install.sh` (0.2.7+, commit a555995) does the usage steps in order: check, build, engine (llama.cpp b11192,
sha256-checked), python (Gradio 3 venv only if missing), canon (clone if absent, never modified), model
(`sAGI/models.py first-run`), start; also `stop`, `status`, `voice`. It keeps what it chose in
`~/.local/share/bankml/install.env`, and its glyphs come from `tools/bashmoji.sh`, vendored from cryptoAGI/bashmoji
(MIT OR Apache-2.0). Docs: `docs/usage.md` covers install and use; `docs/playback.md` is the simple new-to-Savante
guide to the page, her card and her voice.

**Layout (operator, 2026-09-29, commit ef65eb4):** the Rust runtime is in `bankML/` (Cargo.toml `[lib]` path
`bankML/bankml.rs`, `[[bin]]` path `bankML/main.rs`), and Savante's UI is in `sAGI/` (was `ui/`). Three strings still
say `ui/` on purpose, because they are data rather than paths: thot.py's `$comment` (in THOT manifest digests), speak.py's `"source"` label (in
the committed voice manifest), and the DreamKnob bundle banner.

## Ports and processes

18092 llama-server (spawned by serve) or the native engine address · 18093 `bankml serve` · 7873 interact (loopback) ·
7874 view (your network) · 7875 the bankML console (loopback). The gate's live stages use their own spare ports.
Start the carrier through the importer (`python3 sAGI/models.py use FILE`, or `_stop_carrier` + `_start_carrier`), which
spawns `bankml serve MODEL --fork FORK --spawn llama-server --ctx 2048 --threads 3`. Relaunch the UIs with
`setsid nohup python3 -B sAGI/savante.py --mode interact --port 7873 &` and `sAGI/view.py --host 0.0.0.0 --port 7874`.

## Oracles (see docs/oracles.md; every script and stage in testing/README.md)

Each oracle records llama.cpp b11192's own answers once (a script, usually `--record`), and a Rust test or a live
`--bankml` run replays them. Records live in `.models/oracle-*/`, outside git.
- **Kernels**: `testing/ggml_oracle.py` (ctypes on the shipped `libggml-base.so` / `libggml-cpu-haswell.so` / `-x64.so`);
  `oracle_ggml_b11192_real_*` re-derive every dequantized tensor, q8_0 row and dot product (8B: 8,188,239,872
  weights, 762 dots). Match the **binary**, not the source: haswell (FMA) and x64 builds differ on 19/762 ternary
  dots. `f16_oracle.py` (F16 products), `oracle_ggml_b11192_q8_0_kv_kernels` (the q8_0 KV kernels, dlopen).
- **Forward pass** (P3): `forward_oracle.py` (op by op), `model_oracle.py` (whole model per layer),
  `greedy_oracle.py` (`--long`, `--deep`), `sample_oracle.py`, `tokenizer_oracle.py`, `template_oracle.py`.
- **Serving**: `serve_oracle.py` (conversations from a FRESH llama-server), `json_oracle.py`, `json_schema_oracle.py`,
  `grammar_oracle.{cpp,py}`, `schema_oracle.{cpp,py}` and `content_oracle.{cpp,py}` (llama.cpp's own code in
  libllama/libllama-common, no model), `penalty_oracle.py` (`--kind sampler` for the rest of the chain),
  `sort_oracle.cpp` (libstdc++'s `std::sort`), `context_oracle.py`, `slot_oracle.py` (the engine is its own oracle),
  `session_oracle.py`, `kv_oracle.py`, `logprobs_oracle.py` (plain and streamed), `capi/capi_oracle.py`.
- **Models**: `convert_oracle.py` (llama.cpp's converter, byte for byte; name heuristics), `persona_oracle.py`
  (promote.py's persona layer end to end), `train_oracle.py` (mindXtrain's Python; run with
  `~/mindxtrain/.venv/bin/python`). **GPU**: `gpu_q1_0_mat_vec_bit_exact` and `bankml gpu --verify`.
- **Speed**: `ab_vs_ggml*`, `decode_budget_*`, `bench_*` in the gate; `decode_ab.py` (pairs against llama-server,
  identical answers or the round is refused) under `pinned.sh`, outside the gate.
- Also: scalar models vs AVX2 (3,500 / 3,300 cases), f16 all 65,536 values vs F16C, guard Rust == Python, FIPS
  vectors, RFC 6962 CT roots, Savante's doctrine root `0x92fe83eb…`, THOT spec vectors, the iNFT contract on anvil.
- Every oracle found something: FMA contractions read from the disassembly; a reduction fused only when the maximum
  is not in the first chunk; libm `expf` vs ggml's `v_expf`; `</s>` as CONTROL; b11192's Hadamard rotation around a
  quantized KV cache.

## The release routine

1. Do the work; tests: `cargo test --release` (and `-p bankml-capi`), `cargo clippy --release --workspace
   --all-targets -- -D warnings`, `python3 -B testing/test_ui.py`, `test_console.py`, `test_models.py`
   (`BANKML_TEST_CARRIER=1`), `test_connectors.py`, `test_chain.py`. A new behaviour gets an oracle and a gate stage.
2. Bump `Cargo.toml` version, build (updates `Cargo.lock`), turn the CHANGELOG's `Unreleased` heading into the
   version and date, **update the README Releases table and status rows**, `testing/README.md`, and the docs the
   change touches.
3. **Stage the release (`git add -A`) before running the gate**, then keep the next release's edits out of anything
   the gate compiles: the oracle stage runs `cargo test` again and **recompiles from the working tree** — park `.rs`
   edits for the next version outside the tree until the gate finishes.
4. `./install.sh stop`, then `BANKML_GGML_LIB=~/sAGI/bonsai/llama-b11192 testing/release_gate.sh` (add `LLAMA_SRC`
   and `BANKML_LLAMA_SRC` to re-record the schema/content grammars and the name heuristics). It runs every oracle in
   turn, so leave the laptop to it; don't run CPU- or memory-heavy work meanwhile (it skews the budgets and can stall the live stages).
   Record: `testing/results/<v>.txt`. Then `python3 tools/cards.py` redraws the README's speed cards from it.
5. `git add testing/results/<v>.txt`, commit (attribution: `Co-Authored-By: Professor Codephreak
   <codephreak@pythai.net>`), `git tag -a v<v>`, push with tags, `gh release create` with the changelog section and
   the record attached; release notes / PR bodies end with `made with luv.pythai.net`. In `docs/index.html` bump
   `FALLBACK_TAG` and the masthead's gate-record line.
6. Deploy: restart UIs only when llama-server is idle (`top -b -n 2 -d 1 -p <pid>`), as separate calls.

## Pitfalls learned the hard way

- **Never `pgrep -f`/`pkill -f` a pattern that appears in the same command line** — it matches the calling shell
  and kills it (exit 144). Find PIDs with `ps -eo pid,args | grep "[p]attern"` or `ss -ltnp`, then `kill <pid>` in
  a separate call from any relaunch.
- The development laptop has ~6 GB RAM, ~1 GB free, swap full, ~6 GB disk free: **one Piper render at a time** (render lock);
  bge-m3 (1.2 GB) loads only with ≥ 1.3 GB free; Qwen3-8B Q4_K_M does not fit; keep `--ctx 2048` for 8B (KV ~0.3 GB).
- Gradio 3.37 has path-traversal CVEs: interact stays on loopback; `/file=` refuses dot-directories; exports live in
  `sAGI/voice/export` (allowed path).
- `bankml serve` hashes the whole model before it binds a port — wait on the process, never on "no ports yet".
  A spawned engine that dies is detected at once (0.1.7).
- serve requires a loopback `Host` and `Content-Type: application/json` on POST (0.1.7); tests and clients must send
  them.
- Benchmark on fixed resources: `BANKML_PIN_CPUS=2,3 BANKML_PIN_MEM=2500M testing/pinned.sh <cmd>` (cores pinned,
  memory capped, load recorded). A test that `spawn`s serve must kill it on panic (a leaked child holds the test's
  pipes open and a `| grep` pipeline never ends).
- Benchmarks on this laptop are noisy (browsers, other sessions): interleave A/B runs, report min and median, never
  claim a gain inside the noise.
- Speculative decoding (draft Bonsai-1.7B for 8B) stayed token-identical at temperature 0 but showed no gain beyond
  noise (0.39–0.53 vs 0.47 tok/s, 2026-09-29); prompt caching works (70 → 28/31 prompt tokens on repeat turns).
- The chat window slides one exchange per turn once past 12, which changes the prompt right after the system
  prompt and defeats prefix caching — trim with slack (let it reach ~16, drop 4 at once).
- **The gate's live stages and the Vega 3**: the GPU worker's buffers come out of system RAM, which stalled the 0.3.5
  and 0.3.6 gates at `json_schema_oracle_live`; the remaining stages ran with `BANKML_GPU=off` and the record marks
  it. On a short-memory machine set `BANKML_GPU=off` (no token changes) or rely on `BANKML_GPU_LIMIT` (0.3.7).
- **Live oracles start each answer from an empty slot**: a cached prefix changes F16 product paths (column counts)
  and moves seeded answers. Interleaved sessions are the exception, and need the host prompt cache (0.3.8).
- `BANKML_LLAMA_THREADS` (default 3) must equal the reference run's `-t`: split-KV decode chunks by it, so the bits do.
- `bankml convert`'s bytes depend on the source **directory's name** (`general.name`): gen39 must sit in
  `mindx-gen39/` to give the pin.
- Refuse, never approximate: what bankML does not reproduce (`mirostat`, a custom `samplers` order, top-k 0 or above
  128) is refused with the reason; what llama-server itself refuses gets its 400 and its message, word for word.

## Licensing

Own code `MIT OR Apache-2.0` ("do what you want, rights preserved"; Apache for the patent grant). Anything that
makes, holds or checks a secret (e.g. future receipt signing) is `GPL-3.0-only` in its own opt-in module the core
never imports, so no black-box modification of key handling can ship. AGPL-derived code only walled off with its own
licence file and per-file SPDX (the bankon-vault pattern). Imported **models** must carry OSI licences (Gemma,
Llama refused). Never copy KoboldCpp code (AGPL); take llama.cpp/ggml code (MIT) from upstream; vLLM is Apache-2.0.

## Status (keep current; CHANGELOG.md is the source of truth)

**Released: 0.3.6** (2026-10-04). Since the milestone 0.3.0 (2026-09-29, Savante answered by bankML's own forward
pass), each release added one llama-server behaviour, token-identical, with its oracle in the gate:
- **0.3.1** (10-01) Ollama's API natively, a registry of pinned models, one resident, `keep_alive`.
- **0.3.2** (10-01) the C API (`libbankml`, `bankml_log` == `snprintf` on 47/47 formats); Rust 1.99 pinned; workspace.
- **0.3.3** (10-02) JSON mode: llama.cpp's GBNF engine ported; llama-server's `json_object` grammar, prefilled; redraw.
- **0.3.4** (10-02) O4: the Llama graph, F16 (`f16.rs`), tied embeddings — `mindx-gen39`, SmolLM2-135M-Instruct,
  Bonsai-1.7B; `</s>` is CONTROL (found by the grown tokenizer corpus).
- **0.3.5** (10-03) O6b + O5: JSON schemas per template (`schema.rs`), the content rule (`common_chat_parse`),
  `bankml convert` and `bankml create` (promote.py's persona layer end to end), `num_ctx` as Ollama applies it.
- **0.3.6** (10-04) O2: repeat, frequency and presence penalties; the whole prompt fills the window; `/v1` sampler
  refusals 400, not 500. mindX's default engine on its VPS since 2026-10-04 (see the mindx skill).

**Unreleased on the branch** (`Cargo.toml` still says 0.3.6; 0.3.7 and 0.3.8 built, 0.3.9 in progress):
- **0.3.7** the rest of the default chain (typical-p, top-n-σ, XTC, dynamic temperature, DRY; 76/76 on two models),
  libstdc++'s `std::sort` (876/876); `metrics.rs` (TTFT, pp/tg, RAPL joules per token, or `null`); `BANKML_GPU_LIMIT`
  and per-shape GPU calibration; `./install.sh power`; the console (`sAGI/console.py`, :7875) and `bankml.persona`;
  `mirostat` and a custom `samplers` order refused on `/v1`.
- **0.3.8** the context limit (8/8), slots save/restore/erase with `--slot-dir` (19/19), the single-slot contract and
  the host prompt cache (`prompt_cache.rs`, 14/14), `/v1` logprobs plain and streamed (14/14); `why-bankml.md`,
  `tools/makecards.py`, `tools/seo.py`.
- **0.3.9** the q8_0 KV cache with b11192's Hadamard rotation (`BANKML_CACHE_TYPE=q8_0`, 6/6; kernels 4,000/4,000),
  the grammar mask through a trie (median 2.94 ms against 38.8 ms, 13×); next, 1-bit decode measured with
  `decode_ab.py` on an idle machine.

Docs reader: **https://cryptoagi.github.io/bankml/** (GitHub Pages from `main` `/docs`; `docs/index.html` reads the
repo at the latest release through the GitHub API, falls back to its `FALLBACK_TAG`, and reads a document newer than
the release from `main`). **Bump that tag and the masthead's gate-record line at each release.**

What exists, each proven against llama.cpp b11192:
- **The forward pass** (`forward.rs`, `sampler.rs`): token-identical for the Q1_0 and Q2_0_g64 Qwen3 files and the
  Llama graph in F16 with tied embeddings; every ggml CPU attention kernel (reference, tiled for micro-batches of
  ≥ 64 rows, split-KV for decode over ≥ 512 padded cells, which depends on `BANKML_LLAMA_THREADS`); batched prefill;
  the whole sampler chain; `bankml generate [--sample]`. Converted models need torch only for b11192's converter.
- **`bankml serve --native`** (`native.rs`): llama-server's prompt-cache rule (the common prefix, less one when it is
  all cached; the rest in micro-batches of 512), its endpoints on the engine address so Savante works unchanged; the
  importer's `engine` setting `auto` = native for Q2_0 files only (`native_for`). It counts the EOG token in
  `completion_tokens`, as llama-server does.
- **GPU** (`bankML/gpu/`): Vulkan through dlopen, bankML's own SPIR-V assembler, Q1_0 kernels; `bankml gpu
  [--remote|--verify]`. ⚠ RADV on the Vega 3 does **not fuse** `Fma`, so `spirv.rs fma_exact` computes it exactly
  (Boldo–Melquiond); `--verify` must include layer-shaped data. Speed-neutral on the APU.
- **mindXtrain in Rust** (`bankML/train/`): the author and score stages, identical to mindXtrain's Python.

**The road to 1.0.0** is in docs/TODO.md (and docs/thesis.md §VII). 0.4.0 (native serve complete) needs three more
items: 1-bit decode at least at llama-server's speed, measured; `auto` choosing native for 1-bit files too; the
milestone gate. Then 0.5.0 (hardware: NEON, AVX-512, more cards) → 0.6.0 (models) → 0.7.0 (mindXtrain end to end) →
0.8.0 (trust: signed receipts in `crypto/`, GPL-3.0-only) → 0.9.0 (release candidate) → 1.0.0 (llama.cpp only as
the gate's oracle). The `upstream/` Q2_0 kernel awaits the authors' decision to open a llama.cpp PR.
