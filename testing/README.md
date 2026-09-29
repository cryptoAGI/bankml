# testing

Everything that decides whether a bankml version ships, and what each version produced. A speed figure counts only if
every oracle passed on the same code: the same bits first, then the speed.

## Run

```sh
cargo test --release                                  # unit tests (in the modules) + testing/cli.rs, offline
BANKML_GGML_LIB=/path/to/llama-b11192 testing/release_gate.sh   # the full gate → testing/results/<version>.txt
```

The gate runs, and stops at the first failure: the build; the unit and CLI tests; `clippy -D warnings`; the licence
headers (`spdx_check.py`); the Python suites — the guard, the UI data layer (`test_ui.py`), the PostgreSQL connector
(`test_connectors.py`, a throwaway cluster), the iNFT path (`test_chain.py`, a throwaway anvil), the model importer
(`test_models.py` with a real carrier on spare ports); the Rust-against-Python guard agreement; and then, when the
models (`.models/`) and the llama.cpp b11192 release are present, every oracle, both kernel A/Bs, the prefill tile, the
memory floor and both whole-model decode budgets.

## What is here

| file | what |
|---|---|
| `release_gate.sh` | the gate above (also appends to `live.log`) |
| `pinned.sh` | runs a benchmark on a fixed amount of processor and memory: pinned cores (`taskset`), a hard RAM cap with no swap (a user cgroup), and the machine's load recorded before and after |
| `spdx_check.py` | every source file carries the SPDX licence of its layer (LICENSING.md); in the gate |
| `live.sh` | runs one step with its output appended to `live.log`, which `ui/savante.py --mode view` shows live |
| `cli.rs` | end-to-end tests of the `bankml` binary (a cargo integration test): verdicts and exit codes, hostile headers, pin, verify, and `serve` against a mock llama-server (receipts, answer hashes, refusals) |
| `forward_oracle.py` | the forward pass's operations from the SHIPPED ggml, driven through ctypes as llama.cpp's Qwen3 graph drives them (`get_rows` on the Q1_0 embedding, `rms_norm`, `mul`), on 300 real token ids, for `oracle_forward_*` |
| `template_oracle.py` | records llama.cpp's own chat-template rendering (`/apply-template`) of 317 conversations — every rule of the Bonsai / Qwen3 template in bankml's scope, and 300 random ones — for `oracle_chat_template` |
| `tokenizer_oracle.py` | records llama.cpp's own `/tokenize` answers (a running llama-server) on a corpus — every doc, Savante's canon, edge cases and a seeded 2,000-string Unicode fuzz set, with special tokens parsed and not — for `oracle_tokenizer` |
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
| `reads_this_machine_and_this_process`, `stat_with_spaces_and_parens_in_the_name`, `busy_loop_is_seen_as_cpu` | `sys.rs` | bankml's psutil: `/proc` memory, rss, CPU time and % |
| `heads_are_bounded`, `only_loopback_hosts`, `unpaired_surrogates_become_replacement_characters`, `file_identity_changes_are_seen` | `serve.rs` | the gateway's limits, Host check, JSON decoding and the model-identity re-check |
| `oracle_forward_attention` *(real)* | `forward.rs` | P3 steps five and six: layer 0's `kqv_out`, `attn_out`, `ffn_inp`, `l_out` (f16 cache, causal flash attention, `wo`, residual, the FFN block) bit-exact against the shipped ggml, 112 of 112 rows |
| `oracle_forward_swiglu_sweep` | `forward.rs` | SwiGLU with ggml's vectorized expf on 24,600 values over ±120 and the edges, every output's bits |
| `oracle_forward_qkv_rope` *(real)* | `forward.rs` | P3 step four: layer 0's `Qcur`/`Kcur`/`Vcur` (projections, head norms, YaRN RoPE) bit-exact against the shipped ggml, 140 of 140 rows, positions 0–27 and to 63,214 |
| `oracle_forward_embed_norm` *(real)*, `rms_norm_of_a_constant_row` | `forward.rs` | P3 step three: `inp_embd` and `attn_norm-0` bit-exact against the shipped ggml on 300 of 300 tokens |
| `oracle_chat_template` *(real)*, `a_short_conversation`, `python_split_semantics` | `chat.rs` | P3 step two: every recorded conversation byte-identical to llama.cpp b11192's rendering (317 of 317) |
| `oracle_tokenizer` *(real)*, `pretokenizer_shapes`, `byte_chars_are_gpt2s` | `tokenizer.rs` | P3 step one: every recorded case token-identical to llama.cpp b11192 (4,258 of 4,258) |
| `type_ids_match_mainline`, `verified_json_escapes_what_the_header_says` | `bankml.rs` | type ids; `/bankml`'s JSON stays valid for a hostile model name |
| `hardware_path_equals_portable_on_every_length` | `sha256.rs` | SHA-NI equals the portable rounds (lengths 0–1,000, 4 KiB, 64 KiB, split updates) |

## Results

| version | record | headline |
|---|---|---|
| 0.2.6 | `results/0.2.6.txt` | P3 step six: the FFN block; layer 0 complete and bit-exact (112 of 112 rows), SwiGLU sweep 24,600 of 24,600 |
| 0.2.5 | `results/0.2.5.txt` | P3 step five: layer 0's attention, `wo` and the residual bit-exact against the shipped ggml (84 of 84 rows) |
| 0.2.4 | `results/0.2.4.txt` | P3 step four: Q/K/V, head norms and YaRN RoPE bit-exact against the shipped ggml (140 of 140 rows) |
| 0.2.3 | `results/0.2.3.txt` | P3 step three: the embedding and the first RMS norm bit-exact against the shipped ggml (300 of 300 tokens) |
| 0.2.2 | `results/0.2.2.txt` | P3 step two: the chat template byte-identical to llama.cpp on 317 of 317 conversations; the ternary headline re-measured with the model resident (0.231 s, 9.45×) |
| 0.2.1 | `results/0.2.1.txt` | P3 step one: a Rust tokenizer token-identical to llama.cpp on 4,258 of 4,258 recorded cases |
| 0.2.0 | `results/0.2.0.txt` | **milestone** — docs pass (23 corrections), fourth audit fixed, the `Q2_0` kernel prepared for llama.cpp (bit-exact 200,000/200,000) |
| 0.1.9 | `results/0.1.9.txt` | the system prompt's KV saved across restarts (first answer 132 s → 15 s, identical); history counted in engine tokens, never silently dropped |
| 0.1.8 | `results/0.1.8.txt` | SHA-NI pin 5.5×; a history window that keeps the cache warm; CPU/RAM sliders; `sys.rs`; `pinned.sh`; `MIT OR Apache-2.0` |
| 0.1.7 | `results/0.1.7.txt` | the second audit; hardened `serve` (limits, Host, request hash); RFC 6962 commitments; bge-m3 embedding |
| 0.1.6 | `results/0.1.6.txt` | the first audit of 0.1.5 fixed, each finding with a test |
| 0.1.5 | `results/0.1.5.txt` | the model importer (catalogue, Hugging Face, Ollama; open source only; sha256-pinned); metrics charts; Savante reads the thesis |
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
| 0.0.3 | `results/0.0.3.txt` | thread pool; 8B 1-bit oracle; ternary token matmuls 0.23–0.25 s at 3 threads (9.5–9.9× ggml), with the model resident |
| 0.0.2 | `results/0.0.2.txt` | audit: guard hardened (a crashing input now refuses), soundness fix, `verify` |
| 0.0.1 | `results/0.0.1.txt` | kernels bit-exact; ternary 9.5–9.8× ggml per matmul |
