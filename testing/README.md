# testing

Everything that decides whether a bankml version ships, and what each version produced. A speed figure counts only if
every oracle passed on the same code: the same bits first, then the speed.

## Run

```sh
cargo test --release                                  # unit tests (in the modules) + testing/cli.rs, offline
BANKML_GGML_LIB=/path/to/llama-b11192 testing/release_gate.sh   # the full gate → testing/results/<version>.txt
```

The gate runs the build, the unit and CLI tests, `clippy -D warnings`, the Python guard suite, the Rust-against-Python
guard agreement, and then, when the models (`.models/`) and the llama.cpp b11192 release are present, every oracle, both
kernel A/Bs and both whole-model decode budgets. It stops at the first failure.

## What is here

| file | what |
|---|---|
| `release_gate.sh` | the gate above (also appends to `live.log`) |
| `live.sh` | runs one step with its output appended to `live.log`, which `ui/savante.py --mode view` shows live |
| `cli.rs` | end-to-end tests of the `bankml` binary (a cargo integration test): verdicts and exit codes, hostile headers, pin, verify, and `serve` against a mock llama-server (receipts, answer hashes, refusals) |
| `ggml_oracle.py` | writes an oracle: llama.cpp b11192's own answers (its exported symbols, in-process) on a real GGUF — dequantized tensors, q8_0 rows, dot products |
| `gguf_guard.py`, `test_gguf_guard.py` | the Python guard the Rust one was ported from (vendored from minaiml), and its suite |
| `test_chain.py` | the iNFT path on a throwaway anvil devnet: deploy iNFT_7857, simulate, refusals, unsigned tx, mint, read-back, load, lineage |
| `test_connectors.py` | the PostgreSQL connector against a throwaway PostgreSQL 16 cluster with pgvector: publish, verified load, tamper refusal, injection as data, THOT generations |
| `test_models.py` | the model importer against a loopback server and a synthetic GGUF: the sha256 pin (tampered downloads discarded), the open-source licence gate, resume, the guard, Ollama adoption by link, URL and search parsing; with `BANKML_TEST_CARRIER=1`, a real carrier switch and rollback on spare ports |
| `test_ui.py` | the Savante UI's data layer, offline: CIDs, Merkle commitments, inclusion proofs (and their failures), RAGE search, metrics, `.memory`, and the view server's routes |
| `guard_agree.py` | runs both guards on every synthetic case and any real file given; exit 0 only if the JSON is identical |
| `results/<version>.txt` | each release's record |
| `experiments/` | kernels that were measured and not adopted, with their numbers, for re-running elsewhere |

## The tests, and where they live

Rust unit tests sit next to the code they test. The ones marked *real* need the model files and the b11192 release, so
they are `#[ignore]`d and run through the gate (`cargo test --release -- --ignored <name> --nocapture --test-threads=1`).

| test | where | checks |
|---|---|---|
| guard cases t1–t9, fail-closed extras, nested arrays, KV overflow, tensor spans | `gguf.rs` | the three low-bit traps, and hostile headers refuse instead of crashing |
| `real_bonsai_1_7b_q1_0` *(real)* | `gguf.rs` | the guard's report on the real file |
| FIPS vectors, FORK.json pin | `sha256.rs` | SHA-256 and the pin scanner |
| `verify_runs_guard_then_pin` | `bankml.rs` | the verify gate |
| JSON reader, chunked/sized bodies | `serve.rs` | what `serve` parses from llama-server |
| f16 (all 65,536 values, and against F16C hardware), layouts, q8_0 rounding, AVX2 == scalar model (3,500 cases), `mat_vec`/`mat_mul`/`*_par` == per-pair | `q1_0.rs` | the 1-bit kernel against its scalar model of ggml |
| `oracle_ggml_b11192_real_bonsai_1_7b`, `oracle_ggml_b11192_real_bonsai_8b_q1_0` *(real)* | `q1_0.rs` | every `Q1_0` weight of Bonsai-1.7B and Bonsai-8B dequantized bit-exact; q8_0 rows and dot products bit-exact against ggml |
| `ab_vs_ggml`, `decode_budget_q1_0` *(real)* | `q1_0.rs` | speed against ggml's own 1-bit kernel, in one process on the same bits; one token's 253 matmuls at 1–4 threads |
| layouts, portable and AVX2 paths == ggml model (3,300 cases), `*_par` == single thread | `q2_0.rs` | the ternary kernel against its scalar model |
| `oracle_ggml_b11192_real_ternary_bonsai_8b` *(real)* | `q2_0.rs` | every `Q2_0` weight of Ternary-Bonsai-8B bit-exact; dot products against ggml's haswell and x64 builds |
| `ab_vs_ggml_q2_0`, `decode_budget_q2_0` *(real)* | `q2_0.rs` | speed against ggml's ternary kernel |
| `act_tile_bit_exact`, `bench_q1_0_prefill_act` | `q1_0.rs` | the 0.0.4 prefill tile: bits, then speed |
| pool coverage, `bench_pool_overhead`, `bench_memory_floor` | `par.rs` | the thread pool, its wake-up cost, the machine's read bandwidth |

## Results

| version | record | headline |
|---|---|---|
| 0.0.1 | `results/0.0.1.txt` | kernels bit-exact; ternary 9.5–9.8× ggml per matmul |
| 0.0.2 | `results/0.0.2.txt` | audit: guard hardened (a crashing input now refuses), soundness fix, `verify` |
| 0.0.3 | `results/0.0.3.txt` | thread pool; 8B 1-bit oracle; ternary token matmuls 0.23–0.25 s at 3 threads (9.5–9.9× ggml) |
| 0.1.4 | `results/0.1.4.txt` | DreamKnobs in view mode (emerge on play, same voice chain); the timer counts real seconds |
| 0.1.3 | `results/0.1.3.txt` | Savante's own voice (Cori body, Jaimla's 182 Hz, SAVANTE resonance; slower, steadier); PLAY fixed and verified in a browser; lead PLAY, pause/resume, continue-until-stopped; 5 DreamKnobs emerging on play; oscilloscope; 3D depth |
| 0.1.2 | `results/0.1.2.txt` | Savante speaks: introduction (7 chapters) and voice examples in her voice; DreamKnob SPEED/FM; HF + GitHub links; view-mode Listen |
| 0.1.1 | `results/0.1.1.txt` | aivatar card (every persona aspect, verifiable identity), chosen aivatar (ledgered, THOT facet), drop zones + ⇄ + refined resize handles |
| 0.1.0 | `results/0.1.0.txt` | **milestone** — the full gate: every kernel oracle, A/Bs, budgets, and the UI, PostgreSQL and iNFT-devnet suites |
| 0.0.9 | `results/0.0.9.txt` | custom agents (keccak256 doctrine root = Savante's), THOT manifests (= the spec's 3 test vectors), PostgreSQL publish/load (throwaway-cluster tests) |
| 0.0.8 | `results/0.0.8.txt` | `.history` with ragebar (RAGE), Responses (copy, save to .memory, proof), `.memory`, Metrics, commitments + inclusion proofs; docs to 0.0.8 |
| 0.0.7 | `results/0.0.7.txt` | LAN view (stdlib), drag and resize layout, response timer and times in `.history`, usage.md (no Rust change) |
| 0.0.6 | `results/0.0.6.txt` | `bankml serve` (P0) with receipts; Savante UI, view / interact; iNFT ledger checked 12/12 |
| 0.0.5 | `results/0.0.5.txt` | three ternary experiments, bit-exact, none reliably faster (no kernel change); kernels at the Zen+ instruction limit; live log |
| 0.0.4 | `results/0.0.4.txt` | memory floor 15–17 GB/s (kernels are compute-bound); 1-bit prefill 1.27–1.33× ggml; two rejected decode experiments |
