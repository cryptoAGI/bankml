<h1 align="center">bankml</h1>

<p align="center">
  <b>Verified low-bit inference for the CPU you already have.</b><br>
  A zero-dependency Rust runtime for 1-bit and ternary language models — bit-exact against llama.cpp, then faster.<br><br>
  <a href="https://github.com/Professor-Codephreak">Professor Codephreak</a> &middot; Gregory L. Magnusson &middot;
  <a href="https://github.com/cryptoAGI">cryptoAGI</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/dependencies-0-56D364?style=flat-square" alt="zero dependencies">
  <img src="https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-2563EB?style=flat-square" alt="MIT OR Apache-2.0">
  <img src="https://img.shields.io/badge/llama.cpp%20b11192-bit--exact-39D3C7?style=flat-square" alt="bit-exact vs llama.cpp b11192">
  <img src="https://img.shields.io/badge/ternary%20kernel-9.4%E2%80%9310.0%C3%97-D9A23A?style=flat-square" alt="ternary 9.4–10.0x">
  <img src="https://img.shields.io/badge/1--bit%20kernel-parity-5AD1FF?style=flat-square" alt="1-bit parity">
  <img src="https://img.shields.io/badge/status-P0%20serving%20%C2%B7%20Savante%20UI%20%C2%B7%20forward%20pass%20next-F59E0B?style=flat-square" alt="status">
  <a href="https://github.com/cryptoAGI/bankml/releases/latest"><img src="https://img.shields.io/github/v/release/cryptoAGI/bankml?style=flat-square&label=release&color=0ECB81" alt="latest release"></a>
</p>

---

## Why

An 8-billion-parameter model quantized to one bit per weight fits in 1.2 GB; its ternary sibling in 2.3 GB. Both run
on a laptop or a two-core server. On such machines the ternary model gives the **better answers** — and llama.cpp
serves it **five to seven times slower** than the 1-bit one.

bankml found why and fixed it at the kernel: **llama.cpp b11192 has no vectorised x86 kernel for ternary (`Q2_0`)
at all** — it runs scalar C with 64 integer multiplies per block. bankml's kernel computes the **same bits**, verified
against llama.cpp's own compiled library on all 8.19 billion weights of the model, **9.4–10.0× faster** per matrix
(the 0.2.2 gate record; every gate since 0.0.1 has measured 9.4–10.8×).

## Documentation

Start here, then go where your question is:

| if you want to… | read |
|---|---|
| install, run and use bankml and Savante (both modes, models, `.history`, receipts, voice, settings, troubleshooting) | **[docs/usage.md](docs/usage.md)** |
| understand the design, the method, the proofs and the literature | **[docs/TECHNICAL.md](docs/TECHNICAL.md)** (the technical report and thesis) |
| read the thesis: the design intent of bankml's authors, in their own words | **[the Thesis](docs/TECHNICAL.md#thesis--professor-codephreak-and-gregory-l-magnusson)**, in TECHNICAL.md (Savante reads it aloud: `Savante-reading.opus`) |
| meet Savante, the agent bankml runs | **[Using Savante](#using-savante)**, with her public places (Hugging Face, canon, sAGI) |
| see every speed number, the machines and the commands that produced them | **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** |
| know what bankml is checked against, and how | **[docs/oracles.md](docs/oracles.md)** |
| see where bankml stands among Rust engines, 1-bit/ternary kernels and verifiable inference (with papers) | **[docs/research.md](docs/research.md)** |
| use meaning search (bge-m3, the embedding model mindX uses) | **[docs/embedding.md](docs/embedding.md)** |
| see what comes next and what was rejected, and why | **[docs/TODO.md](docs/TODO.md)** |
| check what changed in each release | **[CHANGELOG.md](CHANGELOG.md)** and the [Releases](#releases) table below |
| know what you may do with the code | **[LICENSING.md](LICENSING.md)**: `MIT OR Apache-2.0`; key handling `GPL-3.0-only` |
| run or read the tests and each release's gate record | **[testing/README.md](testing/README.md)**, [testing/results/](testing/results/) |

## Results

<p align="center">
  <img src="docs/cards/ternary_speed.svg" alt="Ternary kernel vs llama.cpp b11192" width="720"><br>
  <img src="docs/cards/decode_budget.svg" alt="One ternary token: all 253 matmuls" width="720"><br>
  <img src="docs/cards/ternary_proof.svg" alt="Ternary bit-exact against ggml b11192" width="720"><br>
  <img src="docs/cards/q1_speed.svg" alt="1-bit kernel vs llama.cpp b11192" width="720"><br>
  <img src="docs/cards/q1_proof.svg" alt="1-bit bit-exact against ggml b11192" width="720">
</p>

All numbers on this page come from the 0.2.2 gate record ([testing/results/0.2.2.txt](testing/results/0.2.2.txt)),
where every oracle was bit-exact in the same run. On the zero-dependency thread pool, **one ternary token's matmuls
take 0.233 s on three threads against llama.cpp's 2.204 s (9.46×)**, comparing the fastest runs; by medians it is
0.287 s against 2.23 s (7.8×). That is less than llama.cpp needs for the *1-bit* model's matmuls (0.35 s). The figure
needs the 2.31 GB model resident in memory. On this 6 GB laptop that means closing other applications; when they hold
the memory, the gate measures the disk instead (1.4–5.2 s per token), as recorded in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

Every number above is measured and reproducible — the tables, machines and commands are in
**[PERFORMANCE.md](docs/PERFORMANCE.md)**. The design, the method and the literature are in the technical report,
**[TECHNICAL.md](docs/TECHNICAL.md)**. Where bankml stands among Rust engines, 1-bit and ternary kernels and verifiable
inference (with the papers) is in **[research.md](docs/research.md)**; every oracle it is checked against, in
**[oracles.md](docs/oracles.md)**.

## The three design goals

| goal | made testable as |
|---|---|
| **Optimization** | measured against llama.cpp's own kernels on the same bits, in one process — never quoted |
| **Succinct** | one crate, zero dependencies (SHA-256, GGUF parse, f16, memory map and every kernel are in-crate), one binary |
| **Verified response** | a model that cannot be verified does not answer: a header **guard**, a sha256 **pin** to the fork's provenance record, and an **oracle** that proves each kernel bit-exact against the compiled reference before its speed counts |

## Status

| phase | what | state |
|---|---|---|
| P1 | GGUF guard (the three low-bit traps) + sha256 pin | **done** — Rust guard == Python guard, JSON for JSON; `bankml verify` = guard + pin as one gate (0.0.2) |
| P2 | `Q1_0` kernel (1-bit) | **done** — bit-exact on 1.72 B + 8.19 B weights (1.7B and 8B); decode at parity (1.02×), prefill 1.23× (0.2.2 gate) |
| P2 | `Q2_0_g64` kernel (ternary) | **done** — bit-exact on 8.19 B weights; 9.4–10.0× decode per matrix, 12.9× prefill; one whole token 9.46× on three threads (0.2.2 gate) |
| P3 | Qwen3 forward pass (tokenizer, YaRN RoPE, GQA, `q8_0` KV cache, sampling) | **tokenizer (0.2.1), chat template (0.2.2), embedding and first RMS norm (0.2.3), Q/K/V with head norms and YaRN RoPE (0.2.4), attention with `wo` and the residual (0.2.5) done** — identical to llama.cpp on every oracle case (4,258; 317; 300; 140; 84); the feed-forward block next, each step against its oracle |
| P0 | serve answers now through the reference, behind guard + pin + receipts | **done (0.0.6)** — `bankml serve`, receipt with the answer's sha256; Savante UI (view / interact) |
| P4 | OpenAI-compatible endpoint; mindX provider; the sAGI engine on top | endpoint done (`bankml serve`); mindX provider next |
| UI | Savante: interact (chat, `.prompt`, `.history`, `.memory`, Responses, Metrics, RAGE search, custom agents, THOT, PostgreSQL) · view (LAN, read-only) · proof of data by commitments | **0.0.6–0.0.9**; aivatar card, her own voice, DreamKnobs 0.1.1–0.1.5 |
| models | import without friction: Bonsai-8B on first run; a pinned open-source catalogue, any Hugging Face GGUF, Ollama; open-source licences only; carrier switch with rollback | **0.1.5**, hardened 0.1.6 |
| memory | bge-m3 embeddings (the model mindX uses): meaning search fused with BM25, pgvector publishing | **0.1.7** ([embedding.md](docs/embedding.md)) |
| audits | two full audits, every finding fixed with a test; RFC 6962 commitments; a hardened gateway | **0.1.6, 0.1.7** |
| iNFT | mint an agent from its THOT bundle (prepared, simulated, signed by the owner), load one from a token | **0.1.0** — full path tested on a local devnet; the contract is not on a public chain yet |
| P5 | ARM / NEON, handheld | planned |

Since 0.0.6 bankml answers through the reference engine (P0), behind its gate and with a receipt. Its own forward
pass is the next phase. The whole-model budget says what that can reach: the ternary matrix work for one token takes
**0.233 s** against llama.cpp's **2.204 s** on three threads, a matmul-bound ceiling of about **4.3 tokens/s**
against **0.45** (0.2.2 gate, model resident). Five more bit-exact kernel variants were measured in 0.0.4–0.0.5, and
none was reliably faster. On the test laptop bankml's ternary token runs about twice the memory floor (0.124 s), and
the 1-bit kernel is at the core's instruction limit (TECHNICAL §IV.5).

## Releases

Every release passed the full gate (build, tests, clippy, every suite, every oracle, the A/Bs and the decode budgets)
before it was tagged; its record is `testing/results/<version>.txt`, and the details are in
**[CHANGELOG.md](CHANGELOG.md)**.

| version | what it brought |
|---|---|
| [0.2.5](https://github.com/cryptoAGI/bankml/releases/tag/v0.2.5) | P3 step five: layer 0's attention (f16 K/V cache, ggml's flash attention reference path, grouped-query), `wo` and the residual, bit-exact against the shipped ggml, 84 of 84 rows |
| [0.2.4](https://github.com/cryptoAGI/bankml/releases/tag/v0.2.4) | P3 step four: layer 0's Q, K and V (projections, head norms, YaRN RoPE) bit-exact against the shipped ggml, 140 of 140 rows at positions to 63,214; matching needed ggml's compiled FMAs, read from the binary |
| [0.2.3](https://github.com/cryptoAGI/bankml/releases/tag/v0.2.3) | P3 step three: the embedding lookup and the first RMS norm of bankml's own forward pass, bit-exact against the shipped ggml on 300 of 300 tokens |
| [0.2.2](https://github.com/cryptoAGI/bankml/releases/tag/v0.2.2) | P3 step two: the chat template, byte-identical to llama.cpp on 317 of 317 conversations; the ternary headline re-measured with the model resident: 0.231 s per token, 9.45× |
| [0.2.1](https://github.com/cryptoAGI/bankml/releases/tag/v0.2.1) | P3 step one: bankml's own tokenizer (no crates), token-identical to llama.cpp on 4,258 of 4,258 recorded cases, `bankml tokenize` |
| [**0.2.0**](https://github.com/cryptoAGI/bankml/releases/tag/v0.2.0) | **milestone**: every document checked against the code and records (23 corrections); the fourth audit fixed; bankml's ternary kernel prepared for llama.cpp (`upstream/`, bit-exact 200,000/200,000, 3.4× the shipped scalar path) |
| [0.1.9](https://github.com/cryptoAGI/bankml/releases/tag/v0.1.9) | the system prompt's KV saved across restarts: first answer after a restart 132 s → 15 s, identical; history counted in the engine's own tokens and never silently dropped; the third audit fixed; bge-m3 live |
| [0.1.8](https://github.com/cryptoAGI/bankml/releases/tag/v0.1.8) | speed on fixed resources: SHA-NI pin 5.5× (a server start 25 s → 7 s); a chat window that keeps the prompt cache warm (8 moves in 60 turns, not 48); CPU/RAM sliders with bankml's own psutil (`sys.rs`); n-gram speculation opt-in; `MIT OR Apache-2.0`; docs in `docs/` |
| [0.1.7](https://github.com/cryptoAGI/bankml/releases/tag/v0.1.7) | the second audit (gateway, chat path, commitments, agents, PostgreSQL, chain) fixed and tested; receipts bound to the request; RFC 6962 Merkle tree; bge-m3 embedding ([embedding.md](docs/embedding.md)); [research.md](docs/research.md) and [oracles.md](docs/oracles.md) |
| [0.1.6](https://github.com/cryptoAGI/bankml/releases/tag/v0.1.6) | the first audit of 0.1.5 fixed, each finding with a test: carrier switch, downloads, view mode, pronunciation |
| [0.1.5](https://github.com/cryptoAGI/bankml/releases/tag/v0.1.5) | the model importer (catalogue, Hugging Face, Ollama; open source only; sha256-pinned); readable metrics; the VOICE dock; Savante reads the thesis; Savante.opus |
| [0.1.4](https://github.com/cryptoAGI/bankml/releases/tag/v0.1.4) | DreamKnobs in view mode; a timer that counts real seconds |
| [0.1.3](https://github.com/cryptoAGI/bankml/releases/tag/v0.1.3) | Savante's own voice (open parts: Cori's body, Jaimla's pitch); PLAY that plays; a card with depth |
| 0.1.2 | Savante speaks: introduction and voice examples, pre-rendered |
| [0.1.1](https://github.com/cryptoAGI/bankml/releases/tag/v0.1.1) | the aivatar card, a chosen aivatar, a layout that moves both ways |
| [0.1.0](https://github.com/cryptoAGI/bankml/releases/tag/v0.1.0) | **milestone**: the iNFT path (prepare, simulate, owner signs, load from a token) and the full gate |
| [0.0.9](https://github.com/cryptoAGI/bankml/releases/tag/v0.0.9) | custom agents, THOT bundles, PostgreSQL connector |
| [0.0.8](https://github.com/cryptoAGI/bankml/releases/tag/v0.0.8) | `.history` search, Responses, `.memory`, Metrics, Merkle proofs |
| [0.0.7](https://github.com/cryptoAGI/bankml/releases/tag/v0.0.7) | the LAN view, response times, arrangeable layout, usage.md |
| [0.0.6](https://github.com/cryptoAGI/bankml/releases/tag/v0.0.6) | `bankml serve`: answers behind the gate, with receipts; the Savante UI |
| [0.0.5](https://github.com/cryptoAGI/bankml/releases/tag/v0.0.5) | three ternary kernel variants measured and rejected, with their numbers |
| [0.0.4](https://github.com/cryptoAGI/bankml/releases/tag/v0.0.4) | where the time goes; a faster 1-bit prefill |
| [0.0.3](https://github.com/cryptoAGI/bankml/releases/tag/v0.0.3) | the zero-dependency thread pool; the 8B 1-bit oracle; `testing/` |
| [0.0.2](https://github.com/cryptoAGI/bankml/releases/tag/v0.0.2) | the first audit: guard hardened, `verify`, soundness fix |
| [0.0.1](https://github.com/cryptoAGI/bankml/releases/tag/v0.0.1) | the guard and pin; the `Q1_0` and `Q2_0_g64` kernels, bit-exact against llama.cpp b11192 |

## Build and verify

```sh
cargo build --release
cargo test --release                       # unit + end-to-end tests, offline

# the oracles and the A/B against llama.cpp's own kernels (needs the model files and the b11192 release):
#   models  → .models/  from huggingface.co/PYTHAI/Bonsai-8B-gguf-fork and PYTHAI/Ternary-Bonsai-8B-gguf-fork
#             (check each file's sha256 against the fork's FORK.json: `bankml pin FILE --fork FORK.json`)
#   release → llama-b11192-bin-ubuntu-x64.tar.gz, sha256 34cf6fa5de9da0db3932c78fe15fed2fbca17451e665dac0a4f6a3c8fc881ec7
python3 testing/ggml_oracle.py .models/Ternary-Bonsai-8B-Q2_0_g64.gguf /path/to/llama-b11192 .models/oracle-ternary
BANKML_GGML_LIB=/path/to/llama-b11192 cargo test --release -- --ignored --nocapture --test-threads=1

python3 testing/guard_agree.py target/release/bankml     # the Rust guard == the Python guard
target/release/bankml guard MODEL.gguf                 # play | refuse (with the reason) | need more
target/release/bankml verify MODEL.gguf --fork FORK.json --json   # guard, then the sha256 pin: one gate
```

## Using Savante

Savante is the review office of the mindX DAIO: an agent defined by a signed canon, not by a model. With bankml she
runs on your own computer, and every answer she gives carries a receipt.

**Savante in public:**
- [Savante on Hugging Face](https://huggingface.co/spaces/PYTHAI/savante): the public office (chat, canon, integrity);
- [the sAGI skill](https://huggingface.co/spaces/PYTHAI/savante/blob/main/skills/sagi/SKILL.md) she runs;
- [Savante's loop](https://huggingface.co/datasets/PYTHAI/savante-loop): her public questions and answers, as a dataset;
- [the sAGI engine](https://github.com/cryptoAGI/sagi): the verdict contract, enforced in code;
- [Savante's canon](https://github.com/cryptoAGI/savante): persona, charter, facets and ledger.

Four steps (details in **[usage.md](docs/usage.md)**):

**1. Build bankml and verify the model**
```sh
cargo build --release
target/release/bankml verify .models/Bonsai-8B-Q1_0.gguf --fork .models/FORK.json    # → play, sha256 284a335a…
```

**2. Start the gate** (bankml launches llama.cpp b11192 on the verified file, or checks a running one via `--upstream`)
```sh
target/release/bankml serve .models/Bonsai-8B-Q1_0.gguf --fork .models/FORK.json --spawn /path/to/llama-server
```

**3. Talk to Savante**: `python3 ui/savante.py --mode interact`, then open **http://127.0.0.1:7873**.
Type a question and press **Send**. A timer runs from the press of Send until the answer is complete: the first turn on
a laptop spends about 2 minutes reading Savante's system prompt, then writes a few tokens a second. Under each answer
you see when it was sent, the time to the first token, the total time, and the receipt: the verified model's sha256
and the sha256 of the answer, checked ✓. To ask for a review, start with *"review:"*.
- **`.prompt`** chooses what carries the conversation: the persona's own system prompt (default), the `sAGI.prompt`
  facet, or the Hugging Face Space's template.
- **`.history`** (`~/.local/share/bankml/savante/savante.history`) records every exchange with its timestamps,
  response times and receipt. The **.history** tab shows them all, newest first, with a **ragebar**: type and the
  exchanges are ranked by RAGE retrieval (mindX's `rage.py` when present, the same BM25 built in otherwise).
- **Responses** steps through every answer (⤒ ▲ ▼ ⤓). **📋 copy** puts the answer on the clipboard, **➕ save to
  .memory** keeps it, and **🔏 proof** gives the inclusion proof for that one exchange.
- **`.memory`** (`savante.memory`) holds the notes you keep. With *use .memory* on, they go into the system prompt,
  labelled as your notes, never as evidence.
- **Metrics** is computed from `.history`: time to first token, response time, prefill and writing speed (median, p90,
  mean), how many answers match their receipt, and the commitments.
- **Proof of data, not the data.** `.history` and `.memory` never leave the laptop. What can be shared is their
  commitment (a Merkle root over the lines and a CIDv1 of the file), and an inclusion proof for any one exchange
  that checks against the root without revealing the rest.
- **Agents**: derive your own agent from Savante's template with its own `.persona`, `.prompt`, card and ledger
  (keccak256 doctrine root, built exactly as Savante's binder builds hers). Build its **THOT bundle** (the dataset an
  iNFT points to: history and memory committed by digest, lineage by generation), and **publish** it to PostgreSQL
  (pgvector/pgvectorscale) or **load** a published one back, verified byte for byte. Savante's canon is never written.
- **iNFT**: plan and simulate an `iNFT_7857` mint from the agent's THOT bundle, get the unsigned transaction (you sign
  it), mint on a local devnet, or load an agent back from a token, verified back to the minted generation.
- **Models, imported without friction**: Bonsai-8B arrives by itself on first run. The **Models** tab (or `python3
  ui/models.py`) imports from a curated catalogue (Bonsai 1-bit and ternary, Qwen3 0.6B–8B, SmolLM2/3, Granite), from
  any Hugging Face GGUF by URL, or from Ollama (search ollama.com, or adopt a local model with no download). Each file
  is kept only if its sha256 equals the publisher's, then guarded and pinned, and the carrier switches with rollback.
  Open-source licences only: Gemma and Llama are refused. Standard formats (Q4_K_M, Q8_0, …) play through the same
  verified gateway as bankml's 1-bit and ternary models.
- **Savante speaks**: in the card, her introduction for new participants (her canon, plus a note on SCIEN·TIFIC's
  2²⁵⁶ − 1 as the measure of accuracy), **the reading** (bankml's thesis and *can a binary computer perform a ternary
  operation?* from docs/TECHNICAL.md), and her voice examples, all pre-rendered in her own voice and committed to this
  repository. There is PLAY ALL, per chapter, and from any line, and **⤓ Savante.opus** / **⤓ Savante-reading.opus**
  export each set as one file with chapter marks. DreamKnob **SPEED**, **FM**, **GAIN** and **VOLUME** appear in a
  dockable, resizable VOICE dock when she speaks. View mode on the LAN plays and exports them too.
- **The aivatar**: click the portrait for the agent's card, with every aspect of its persona, its verifiable identity
  (hashes, doctrine root, THOT) and its ledgered files. Choose the portrait, or upload one for a derived agent.
- Panels resize from a refined corner handle. The side panel drags to either side (drop zones appear) or swaps with ⇄.

**4. Let others watch**: `python3 ui/view.py --host 0.0.0.0` gives a read-only page at
**http://&lt;your LAN address&gt;:7874**. It shows the live testing, the release records, CI, the machine's load and
Savante's ledger, and the commitments of `.history` and `.memory` (never their content), in draggable, resizable
panels. It is the standard library, not Gradio, so it is safe to put on a network.

Savante's canon (`~/savante`) is only ever read. Before she speaks, the UI re-hashes every file her iNFT ledger
(`savante.commitments.json`) commits to, and it refuses if her persona does not verify. Nothing is minted.

## Layout

| file | what |
|---|---|
| `bankml.rs` | the crate: the plan of record (header checklist), `guard`, `pin`, `verify`, `Receipt` |
| `gguf.rs` | header-only GGUF v3 parse, the guard, a read-only memory map |
| `q1_0.rs` | the 1-bit kernel: dequantize, `q8_0` activations, scalar model, AVX2, decode and prefill |
| `q2_0.rs` | the ternary kernel: the same, plus the per-token activation layout |
| `sha256.rs` | FIPS 180-4 SHA-256 and the `FORK.json` pin |
| `par.rs` | the thread pool and row scheduler (0.0.3) |
| `serve.rs` | P0: the verified loopback gateway with receipts (0.0.6) |
| `ui/savante.py` | the Savante UI, interact mode (Gradio, loopback) |
| `ui/view.py` | the read-only view page for the LAN (standard library) |
| `ui/agents.py` | custom agents: derive, edit, ledger; keccak256 and the doctrine root |
| `ui/thot.py` | THOT manifests (`sagi.thot_manifest/1`), checked against the spec's test vectors |
| `ui/connectors.py` | PostgreSQL (pgvector / pgvectorscale): publish and load agents, verified |
| `ui/models.py` | the model importer: catalogue, Hugging Face, Ollama; sha256 pins; the carrier switch |
| `ui/embed.py` | embeddings with bge-m3 (the model mindX uses) through the local Ollama — see [embedding.md](docs/embedding.md) |
| `ui/speak.py` | Savante's voice: rendering, the introduction and the reading, the exports |
| `ui/chain.py` | iNFT: ABI, JSON-RPC, mint planning, simulation, unsigned transactions, devnet, load from a token |
| `docs/usage.md` | the full guide: setup, both modes, `.history`, receipts, the canon, troubleshooting |
| `docs/embedding.md` | the embedding model: why, how, the cache, the ranking, PostgreSQL, settings |
| `docs/research.md` | the state of the field and bankml's place in it, with links and papers |
| `docs/oracles.md` | every oracle bankml is checked against: what, how, where, and what it last found |
| `docs/TODO.md` | what comes next and what was rejected, each item with its source (research, KoboldCpp, vLLM, Rust, audits) |
| `LICENSING.md` | the licence layers: `MIT OR Apache-2.0`, key handling `GPL-3.0-only`, AGPL walled off |
| `sys.rs` | bankml's psutil: memory, cores, a process's resident memory and CPU time, from `/proc` (0.1.8) |
| `testing/` | every test outside the modules: the release gate, the end-to-end CLI suite (`cli.rs`), the oracle generator (`ggml_oracle.py`), the guard agreement check and the Python guard it was ported from; `testing/results/` holds each release's gate record — see [testing/README.md](testing/README.md) |
| `upstream/` | the AVX2 `Q2_0` kernel prepared for llama.cpp, its bit-exact harness against the shipped library, and why ([upstream/README.md](upstream/README.md)) |
| `docs/cards/` | the result cards above, drawn from the measured numbers |

Changes by release are in **[CHANGELOG.md](CHANGELOG.md)**.

## Models

[PYTHAI/Bonsai-8B-gguf-fork](https://huggingface.co/PYTHAI/Bonsai-8B-gguf-fork) (1-bit, 1.16 GB) ·
[PYTHAI/Ternary-Bonsai-8B-gguf-fork](https://huggingface.co/PYTHAI/Ternary-Bonsai-8B-gguf-fork) (ternary, 2.31 GB) —
PrismML's Bonsai (Qwen3-8B dense), forked with provenance; Apache-2.0.

## Part of

[![mindX](https://img.shields.io/badge/mindX-56D364?style=flat-square&logoColor=black)](https://mindx.pythai.net)
[![sAGI](https://img.shields.io/badge/sAGI-6D28D9?style=flat-square&logoColor=white)](https://github.com/cryptoAGI/sagi)
[![Savante](https://img.shields.io/badge/Savante-D9A23A?style=flat-square&logoColor=black)](https://huggingface.co/spaces/PYTHAI/savante)
[![minaiml](https://img.shields.io/badge/minaiml-39D3C7?style=flat-square&logoColor=black)](https://github.com/minaiml)
[![DeltaVerse](https://img.shields.io/badge/DeltaVerse-5AD1FF?style=flat-square&logoColor=black)](https://deltaverse.pythai.net)

## Licence

**`MIT OR Apache-2.0`**, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)): open source, do
what you want with it, rights preserved. Key handling, when bankml has any, is `GPL-3.0-only` so no black-box
modification of it can ship; AGPL-derived code, if ever used, stays walled off under AGPL. The layers and rules are in
**[LICENSING.md](LICENSING.md)**. © 2026 Professor Codephreak and Gregory L. Magnusson, cryptoAGI.
