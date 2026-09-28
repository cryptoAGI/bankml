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
  <img src="https://img.shields.io/badge/licence-Apache--2.0-2563EB?style=flat-square" alt="Apache-2.0">
  <img src="https://img.shields.io/badge/llama.cpp%20b11192-bit--exact-39D3C7?style=flat-square" alt="bit-exact vs llama.cpp b11192">
  <img src="https://img.shields.io/badge/ternary%20kernel-9.5%E2%80%939.8%C3%97-D9A23A?style=flat-square" alt="ternary 9.5–9.8x">
  <img src="https://img.shields.io/badge/1--bit%20kernel-parity-5AD1FF?style=flat-square" alt="1-bit parity">
  <img src="https://img.shields.io/badge/status-kernels%20proven%20%C2%B7%20forward%20pass%20next-F59E0B?style=flat-square" alt="status">
</p>

---

## Why

An 8-billion-parameter model quantized to one bit per weight fits in 1.2 GB; its ternary sibling in 2.3 GB. Both run
on a laptop or a two-core server. On such machines the ternary model gives the **better answers** — and llama.cpp
serves it **five to seven times slower** than the 1-bit one.

bankml found why and fixed it at the kernel: **llama.cpp b11192 has no vectorised x86 kernel for ternary (`Q2_0`)
at all** — it runs scalar C with 64 integer multiplies per block. bankml's kernel computes the **same bits**, verified
against llama.cpp's own compiled library on all 8.19 billion weights of the model, **9.5–9.8× faster**.

## Results

<p align="center">
  <img src="docs/cards/ternary_speed.svg" alt="Ternary kernel vs llama.cpp b11192" width="720"><br>
  <img src="docs/cards/decode_budget.svg" alt="One ternary token: all 253 matmuls" width="720"><br>
  <img src="docs/cards/ternary_proof.svg" alt="Ternary bit-exact against ggml b11192" width="720"><br>
  <img src="docs/cards/q1_speed.svg" alt="1-bit kernel vs llama.cpp b11192" width="720"><br>
  <img src="docs/cards/q1_proof.svg" alt="1-bit bit-exact against ggml b11192" width="720">
</p>

Since 0.0.3 the kernels run on a zero-dependency thread pool: one ternary token's matmuls take **0.23–0.25 s** on
three threads, less than llama.cpp needs for the *1-bit* model's (0.34 s). Every release's test record is in
[testing/results/](testing/results/).

Every number above is measured and reproducible — the tables, machines and commands are in
**[PERFORMANCE.md](PERFORMANCE.md)**. The design, the method and the literature are in the technical report,
**[TECHNICAL.md](TECHNICAL.md)**.

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
| P2 | `Q1_0` kernel (1-bit) | **done** — bit-exact on 1.72 B weights; decode 1.03×, prefill 1.16× |
| P2 | `Q2_0_g64` kernel (ternary) | **done** — bit-exact on 8.19 B weights; 9.5–9.8× decode, 12.5× prefill |
| P3 | Qwen3 forward pass (tokenizer, YaRN RoPE, GQA, `q8_0` KV cache, sampling) | next — acceptance: token-identical to llama.cpp at temperature 0 |
| P0 | serve answers now through the reference, behind guard + pin + receipts | **done (0.0.6)** — `bankml serve`, receipt with the answer's sha256; Savante UI (view / interact) |
| P4 | OpenAI-compatible endpoint; mindX provider; the sAGI engine on top | planned |
| P5 | ARM / NEON, handheld | planned |

Since 0.0.6 bankml answers through the reference engine (P0), behind its gate. Its own forward pass is the next phase. The whole-model
budget says what it can reach: the ternary matrix work for one token in **0.36 s** against llama.cpp's **2.54 s** on
three threads, a matmul-bound ceiling of about **2.8 tokens/s** against **0.39**.

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

## Run it on this computer (0.0.6)

```sh
cargo build --release
# answers through llama.cpp b11192's llama-server, behind the guard, the pin and a receipt (P0)
target/release/bankml serve .models/Bonsai-8B-Q1_0.gguf --fork FORK.json --upstream 127.0.0.1:18092   # or --spawn /path/to/llama-server
python3 ui/savante.py --mode interact            # http://127.0.0.1:7873  Savante: chat, .prompt, .history, verifier
python3 ui/savante.py --mode view --port 7874    # http://127.0.0.1:7874  read-only: watch the testing live
```

`serve` refuses to start unless the file verifies and the upstream serves that very file. Every answer carries a
`bankml_receipt` with the sha256 of the text. The UI reads Savante's canon (`~/savante`, or `SAVANTE_CANON`) without
writing to it, and checks it against the iNFT ledger `savante.commitments.json` before it speaks as Savante.

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
| `ui/savante.py` | the Savante UI, view and interact modes (0.0.6) |
| `testing/` | every test outside the modules: the release gate, the end-to-end CLI suite (`cli.rs`), the oracle generator (`ggml_oracle.py`), the guard agreement check and the Python guard it was ported from; `testing/results/` holds each release's gate record — see [testing/README.md](testing/README.md) |
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

Apache-2.0 — see [LICENSE](LICENSE). © 2026 Professor Codephreak and Gregory L. Magnusson, cryptoAGI.
