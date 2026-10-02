---
name: bankml
description: bankml (github.com/cryptoAGI/bankml, local ~/cryptoAGI/bankml) — the zero-dependency Rust runtime for 1-bit (Q1_0) and ternary (Q2_0_g64) GGUF models, bit-exact against llama.cpp b11192, with `bankml serve` (a verifying loopback gateway that pins the model by sha256 and puts a receipt on every answer) and the Savante UI (interact on 127.0.0.1:7873, view on the LAN at :7874). Use when working on bankml's kernels, oracles, release gate, serve/receipts, the model importer, the Savante UI (chat, .history/.memory, Merkle commitments, agents, THOT, PostgreSQL, iNFT, voice, embeddings), cutting a release, or measuring speed. Triggers "bankml", "bankml serve", "Q1_0", "Q2_0", "Bonsai", "release gate", "oracle", "receipt", "Savante UI", "savante.py", "view.py", "models.py", "speak.py".
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, WebFetch
---

# bankml

**Verified low-bit inference for the CPU you already have.** A Rust runtime with no crates: its own GGUF guard, sha256
pin, AVX2 kernels for ggml's `Q1_0` (1.125 bits per weight) and `Q2_0_g64` (ternary, 2.25 bpw) that are **bit-exact
against the compiled llama.cpp b11192 libraries**, a thread pool, its own Qwen3 forward pass (token-identical to
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
| `bankML/tokenizer.rs`, `chat.rs` | the tokenizer and the chat template, identical to llama.cpp's (4,258/4,258; 317/317) |
| `bankML/forward.rs` | the Qwen3 forward pass: `Weights`, `KvCache`, the three ggml attention kernels, `prefill` (micro-batches), `decode`, `logits` |
| `bankML/sampler.rs` | llama-server's sampler chain (libstdc++ `partial_sort` port, mt19937) |
| `bankML/native.rs` | the engine behind `serve --native`: one slot with llama-server's prompt-cache rule |
| `bankML/ollama.rs` | Ollama's API on `serve --native` (0.3.1): `/api/chat` `/api/generate` (NDJSON) `/api/tags` `/api/ps` `/api/show`, option mapping, refusals, `keep_alive`, stop strings; the registry and residency (`Registry`, `Residency`, one resident model, verify on every load) are in `native.rs`; native only, never proxied; roadmap `docs/OLLAMA.md` |
| `bankML/gpu/` | the video-card component: `mod.rs` registry, `vulkan.rs`, `hf.rs` (rented, listed only), `spirv.rs`, `kernels.rs`, `compute.rs`, `worker.rs` |
| `bankML/train/` | mindXtrain in Rust: `script.rs` (author), `imprint.rs` (score) |
| `bankML/gguf.rs` | header parser, the guard (`judge`: the three low-bit traps, hostile headers), `type_name` for every ggml type, `Mmap` |
| `bankML/q1_0.rs`, `q2_0.rs` | kernels: scalar models of ggml (`vec_dot_ref`), AVX2 paths, `*_par`; oracle and A/B tests (`#[ignore]`d, real models) |
| `bankML/par.rs` | the zero-dependency thread pool (0.0.3) |
| `bankML/sha256.rs` | FIPS 180-4, the FORK.json pin scanner, SHA-NI path (0.1.8, 5.5×) |
| `bankML/serve.rs` | the gateway: `/bankml`, `/bankml/usage`, `/health` `/props` `/v1/models`, `/v1/chat/completions` with receipts; `--native` also answers `/tokenize`, `/apply-template` and serves the engine address |
| `bankML/main.rs` | CLI: `guard`, `sha256`, `pin`, `verify`, `serve [--native]`, `generate [--sample]`, `tokenize`, `chat-template`, `gpu [--remote\|--verify]`, `usage`, `version` |
| `sAGI/savante.py` | interact UI (Gradio 3.37, **loopback only**, refuses `--host 0.0.0.0`, trusted-Host middleware) |
| `sAGI/view.py` | the LAN page (stdlib, fixed routes only, CSP, cached state) |
| `sAGI/models.py` | importer: catalogue / Hugging Face / Ollama; sha256-pinned; OSS licences only; carrier switch with rollback |
| `sAGI/embed.py` | bge-m3 via local Ollama (the model mindX uses); BM25 + RRF search; memory-guarded |
| `sAGI/speak.py` | Savante's voice (Piper Cori → Jaimla's 182 Hz), introduction + reading, `Savante.opus` exports |
| `sAGI/agents.py`, `thot.py`, `connectors.py`, `chain.py` | custom agents (keccak doctrine root), THOT manifests, PostgreSQL, iNFT (devnet only) |
| `testing/` | `release_gate.sh`, `cli.rs`, `ggml_oracle.py`, `guard_agree.py`, the Python suites, `results/<version>.txt` |
| docs | root: `README.md` (the front door: a Documentation router and the Releases table), `CHANGELOG.md`, `LICENSING.md`, `LICENSE-MIT`, `LICENSE-APACHE`; `docs/`: `usage.md`, `TECHNICAL.md`, `PERFORMANCE.md`, `oracles.md`, `research.md`, `embedding.md`, `TODO.md`, `cards/` |

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

18092 llama-server (spawned by serve) · 18093 `bankml serve` · 7873 interact (loopback) · 7874 view (your network).
Start the carrier through the importer (`python3 sAGI/models.py use FILE`, or `_stop_carrier` + `_start_carrier`), which
spawns `bankml serve MODEL --fork FORK --spawn llama-server --ctx 2048 --threads 3`. Relaunch the UIs with
`setsid nohup python3 -B sAGI/savante.py --mode interact --port 7873 &` and `sAGI/view.py --host 0.0.0.0 --port 7874`.

## Oracles (see oracles.md)

- **ggml oracle**: `testing/ggml_oracle.py` records llama.cpp b11192's own answers (via ctypes on the shipped
  `libggml-base.so` / `libggml-cpu-haswell.so` / `-x64.so`); `oracle_ggml_b11192_real_*` re-derive every dequantized
  tensor (sha256 of f32), every q8_0 row (bytes) and every dot product (`to_bits`). 1.7B: 197 tensors, 788 cases;
  8B Q1_0 and ternary: 254 tensors, 8,188,239,872 weights, 762 cases. Match the **binary**, not the source: the
  shipped kernels use fused multiply-adds; haswell (FMA) and x64 (no FMA) builds differ on 19/762 ternary dots.
- **Forward-pass oracles** (P3): `testing/forward_oracle.py` drives the shipped ggml op by op (embed, norms,
  Q/K/V with YaRN RoPE, the three attention kernels, SwiGLU sweep); `testing/model_oracle.py` covers the whole
  model per layer; `greedy_oracle.py` (`--long`, `--deep`) and `sample_oracle.py` record llama-server's own tokens;
  `serve_oracle.py` records whole conversations through `/v1/chat/completions` from a FRESH server (so its cache
  starts empty). Every one found something: FMA contractions read from the disassembly; a reduction fused only
  when the maximum is not in the first chunk; libm `expf` vs ggml's `v_expf`.
- **GPU oracle**: `gpu_q1_0_mat_vec_bit_exact` and `bankml gpu --verify`. **mindXtrain**: `testing/train_oracle.py`,
  run with `~/mindxtrain/.venv/bin/python`.
- Scalar models vs AVX2 (3,500 / 3,300 cases), f16 all 65,536 values vs F16C, guard Rust == Python (28/28), FIPS
  vectors, RFC 6962 CT roots, Savante's doctrine root `0x92fe83eb…`, THOT spec vectors, the iNFT contract on anvil.

## The release routine

1. Do the work; tests: `cargo test --release`, `cargo clippy --release --all-targets -- -D warnings`,
   `python3 -B testing/test_ui.py`, `test_models.py` (`BANKML_TEST_CARRIER=1`), `test_connectors.py`, `test_chain.py`.
2. Bump `Cargo.toml` version, build (updates `Cargo.lock`), write the CHANGELOG entry, **update the README Releases
   table and status rows**, update docs the change touches.
3. **Stage the release (`git add -A`) before running the gate**, then keep the next release's edits out of anything
   the gate compiles: the oracle stage runs `cargo test` again and **recompiles from the working tree** — park `.rs`
   edits for the next version outside the tree until the gate finishes.
4. `BANKML_GGML_LIB=~/sAGI/bonsai/llama-b11192 testing/release_gate.sh` (~20–30 min; oracles ~10 min, benchmarks
   last; don't run CPU-heavy work meanwhile — it skews the decode budgets). Record: `testing/results/<v>.txt`.
5. `git add testing/results/<v>.txt`, commit (attribution: `Co-Authored-By: Professor Codephreak
   <codephreak@pythai.net>`), `git tag -a v<v>`, push with tags, `gh release create` with the changelog section and
   the record attached; release notes / PR bodies end with `made with luv.pythai.net`.
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
  prompt and defeats prefix caching — trim with slack (let it reach ~16, drop 4 at once) and try `--cache-reuse`.

## Licensing

Own code `MIT OR Apache-2.0` ("do what you want, rights preserved"; Apache for the patent grant). Anything that
makes, holds or checks a secret (e.g. future receipt signing) is `GPL-3.0-only` in its own opt-in module the core
never imports, so no black-box modification of key handling can ship. AGPL-derived code only walled off with its own
licence file and per-file SPDX (the bankon-vault pattern). Imported **models** must carry OSI licences (Gemma,
Llama refused). Never copy KoboldCpp code (AGPL); take llama.cpp/ggml code (MIT) from upstream; vLLM is Apache-2.0.

## Status (keep current)

Released through **0.3.4** (2026-10-02: O4 — mindX's own `mindx-gen39` (SmolLM2-135M + LoRA, merged, F16), SmolLM2-135M-Instruct and Bonsai-1.7B served natively, token-identical to llama-server b11192 on every oracle family; `bankML/f16.rs` = ggml's two F16 products chosen by shape (`vec_dot_f16` for 1 column, llamafile tinyBLAS for ≥2 when rows%4==0 && k%8==0), oracle `oracle_ggml_b11192_f16` (testing/f16_oracle.py); `forward::plan` decides arch/type/tied from the header and refuses the rest; Llama graph = NORM RoPE, no QK-norm; only output rows leave the last layer (inp_out_ids — decides F16 bits); `smollm` pre-tokenizer (digits split, then GPT-2 regex per piece; bytes without tokens dropped); templates per model by sha (`chat::TEMPLATES`), JSON grammar per template; llama-vocab's type overrides by name (`</s>` is CONTROL — a real bug found by the grown tokenizer corpus); converted models pinned via `sAGI/models.py` CONVERTED/pin_converted; registry alias drops the type suffix (`mindx-gen39`); Q8_0 deferred: Qwen3-0.6B's own template differs on 4/317). **0.3.3** (2026-10-02: JSON mode, O6's first cut — `bankML/grammar.rs` ports llama.cpp's GBNF engine; llama-server's `json_object` grammar (generated from the template's PEG parser, root starts with the generation prompt, prefilled) is a constant checked against the server; `sampler.rs` draws, checks, and redraws under the mask as `common_sampler_sample`; end set = llama-vocab's 6 tokens; `/v1` response_format/json_schema/grammar, Ollama `format:"json"`, `bankml_chat`, `generate --json`; oracles `oracle_grammar_masks` (testing/grammar_oracle.{cpp,py}, libllama's public grammar sampler) and `oracle_json_mode{,_ternary}` + `json_oracle.py --bankml`; real JSON schemas refused until `json_schema_to_grammar` is ported). **0.3.2** (2026-10-02: Rust 1.99 pinned in rust-toolchain.toml; a C API, `capi/` → libbankml.so/.a + `capi/include/bankml.h`, docs/CAPI.md: open (verify) · chat (stream + receipt) · close · free · set_log, and the variadic `bankml_log(level, fmt, ...)` defined in Rust over VaList, byte-identical to snprintf on 47/47 formats; the root is now a workspace (`.`, `capi`). 3-thread benches read slower in the 0.3.2 gate under load; a quiet-machine 1.95-vs-1.99 A/B is owed). **0.3.1** (2026-10-01: Ollama's API natively, a registry of pinned models with one resident and `keep_alive`; docs/OLLAMA.md is the gap matrix against mindX and the O1–O8 track). Milestone **0.3.0** (2026-09-29): **Savante answered by bankML's own forward pass.** Docs reader:
**https://cryptoagi.github.io/bankml/** (GitHub Pages from `main` `/docs`; `docs/index.html` reads the repo at the
latest release through the GitHub API, falling back to its `FALLBACK_TAG`. **Bump that tag and the masthead's gate-record
line at each release.**)

What exists, each proven against llama.cpp b11192:
- **P3 forward pass** (`bankML/forward.rs`, `sampler.rs`): token-identical for the Q1_0 and Q2_0_g64 Qwen3 files,
  and (0.3.4, O4) the Llama graph in F16 (`f16.rs`) and tied embeddings: SmolLM2-135M-Instruct, `mindx-gen39`,
  Bonsai-1.7B. Converted models need torch only for b11192's converter; bankML never runs it.
  - Every ggml CPU attention kernel: the reference, the tiled one (micro-batches of ≥ 64 rows) and split-KV
    (decode over ≥ 512 padded cells; it depends on llama.cpp's `-t`, `BANKML_LLAMA_THREADS`, default 3).
  - Batched prefill; seeded sampling (a libstdc++ `partial_sort` port for tie order, mt19937).
  - `bankml generate [--sample]`.
- **`bankml serve --native`** (`native.rs`): llama-server's prompt-cache rule (the common prefix, less one when it
  is all cached; the rest in micro-batches of 512). It also answers llama-server's endpoints on the engine address,
  so Savante works unchanged. The importer's `engine` setting: `auto` = native for Q2_0 files. It counts the EOG
  token in `completion_tokens`, as llama-server does.
- **GPU** (`bankML/gpu/`):
  - Vulkan through dlopen, bankML's own SPIR-V assembler, Q1_0 kernels.
  - `bankml gpu [--remote|--verify]`; `worker.rs` gives a verified card a calibrated share of the rows.
  - ⚠ RADV on the Vega 3 does **not fuse** `Fma` (NoContraction does not help), so `spirv.rs fma_exact` computes
    it exactly (Boldo–Melquiond). `--verify` must include layer-shaped data, or it misses this.
  - Speed-neutral on the APU; per-call submit and wait is the bottleneck.
- **mindXtrain in Rust** (`bankML/train/`): the author and score stages, identical to mindXtrain's Python.
- **Layout** since ef65eb4: Rust in `bankML/`, UI in `sAGI/`. Savante's canon is `~/cryptoAGI/savante`
  (`~/savante` is a link).

**The road to 1.0.0** is in docs/TODO.md: what 1.0.0 means (llama.cpp only as the gate's oracle; full oracles per
model × format; parity or better everywhere; stable interfaces; signed receipts; Savante and mindX on bankML by
default), and the milestones 0.4.0 (native serve complete) → 0.5.0 (hardware) → 0.6.0 (models) → 0.7.0 (mindXtrain
end to end) → 0.8.0 (trust) → 0.9.0 (release candidate). Next (docs/TODO.md): slot save and restore in native serve; batched GPU submissions (Q/K/V and gate/up); the Q2_0
GPU kernel, then several cards; mindXtrain probe (Llama architecture, PEFT adapters), the verdicts, LoRA on the CPU;
signed receipts in `crypto/` (GPL-3.0-only); NEON. The `upstream/` Q2_0 kernel awaits the authors' decision to open
a llama.cpp PR.
