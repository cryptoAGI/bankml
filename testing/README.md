# testing

Everything that decides whether a bankml version ships, and what each version produced. A speed figure counts only if
every oracle passed on the same code: the same bits first, then the speed.

## Run

```sh
cargo test --release                                  # unit tests (in the modules) + testing/cli.rs, offline
BANKML_GGML_LIB=/path/to/llama-b11192 testing/release_gate.sh   # the full gate → testing/results/<version>.txt
```

The gate runs, and stops at the first failure: the build; the unit and CLI tests, and the C API's (`bankml-capi`);
`clippy -D warnings` over the workspace; the licence headers (`spdx_check.py`); the Python suites — the guard, the UI
data layer (`test_ui.py`), the console (`test_console.py`), the PostgreSQL connector (`test_connectors.py`, a
throwaway cluster), the iNFT path (`test_chain.py`, a throwaway anvil), the model importer (`test_models.py` with a
real carrier on spare ports); the Rust-against-Python guard agreement; the C library and the printf oracle. Then, when
`BANKML_GGML_LIB` points at the llama.cpp b11192 release and the models are in `.models/`: the schema and content
grammars re-recorded from llama.cpp's own code (with `LLAMA_SRC`), libstdc++'s sort orders (with `g++`), every
`#[ignore]`d oracle, both kernel A/Bs, the prefill tile, the memory floor and both whole-model decode budgets, and the
live checks against a running `serve --native`. Last, the converter against llama.cpp's `convert_hf_to_gguf.py` for
each directory under `.models/convert/`, and the name heuristics (with `BANKML_LLAMA_SRC`).

### The gate's stages, in order

Each stage prints a `## name` line in the record. The `#[ignore]`d tests in the `for t in …` loop run one at a time
(`cargo test --release -- --ignored --exact <path> --nocapture --test-threads=1`); each is described in
[the tests table](#the-tests-and-where-they-live).

| stage | runs | checks |
|---|---|---|
| the oracle loop | `oracle_tokenizer` … `oracle_ggml_b11192_real_ternary_bonsai_8b`, the O4 oracles, then `ab_vs_ggml`, `ab_vs_ggml_q2_0`, `bench_q1_0_prefill_act`, `bench_memory_floor`, `decode_budget_q1_0`, `decode_budget_q2_0` | bits first, then the speed figures |
| `serve_oracle_ollama_shape` | `serve_oracle.py --bankml` | `/api/chat` == `/v1` == llama-server's record, and again after unload and reload (Bonsai-8B) |
| `o4_live` | `serve_oracle.py`, `json_oracle.py`, `json_schema_oracle.py`, each `--bankml STEM [NAME]` | the same three live checks on Bonsai-1.7B, SmolLM2-135M-Instruct and mindx-gen39 (asked for as `mindx-gen39`) |
| `persona_oracle_live` | `persona_oracle.py --bankml` | the created mindx-gen39 with promote.py's persona layer: `/api/chat`, `/api/generate`, `/v1`, `/api/ps` |
| `penalty_oracle_live` | `penalty_oracle.py --bankml mindx-gen39-F16 mindx-gen39` | the penalties through `/v1` and `/api/chat`; refusals as 400s with llama-server's message |
| `sampler_oracle_live` | `penalty_oracle.py --kind sampler --bankml …` | typical-p, top-n-σ, XTC, dynamic temperature and DRY through `/v1` (typical-p also `/api`) |
| `context_oracle_live` | `context_oracle.py --bankml Bonsai-1.7B-Q1_0` | the context limit as llama-server's: stop at the full context, past it its 400 body |
| `slot_oracle_live` | `slot_oracle.py Bonsai-1.7B-Q1_0` | slot save, restore and erase: an answer after a restore equals an empty slot's, across a restart |
| `session_oracle_live` | `session_oracle.py --bankml Bonsai-1.7B-Q1_0` | interleaved conversations through llama-server's host prompt cache; simultaneous requests queued |
| `kv_oracle_live` | `kv_oracle.py --bankml Bonsai-1.7B-Q1_0` | the q8_0 KV cache against `llama-server --cache-type-k/v q8_0` |
| `logprobs_oracle_live` | `logprobs_oracle.py --bankml Bonsai-1.7B-Q1_0` | `/v1` logprobs, plain and streamed: every logprob the same float |
| `json_oracle_live` | `json_oracle.py --bankml` | JSON mode and grammars: `/v1` (once streamed) and `/api/chat` with `format: "json"` |
| `json_schema_oracle_live` | `json_schema_oracle.py --bankml` | JSON schemas: `/v1` (once streamed) and `/api/chat` with `format: <schema>` |
| `capi_chat_oracle` | `capi/capi_oracle.py --chat` | `bankml_chat` == `serve --native` (ternary) == llama-server's record (1-bit); the refusals |
| `oracle_convert_b11192 <dir>` | `convert::tests::oracle_convert_b11192` per `.models/convert/<dir>/` with `<dir>.oracle.gguf` | `bankml convert` byte-identical to llama.cpp's converter |
| `oracle_name_heuristics` | `convert_oracle.py --names`, then `convert::tests::oracle_name_heuristics` | the name heuristics against gguf-py's |

Each live stage starts `target/release/bankml serve --native` on spare ports and replays a record taken earlier from
llama-server b11192 (the oracle script's `--record` mode, or its default mode for the older scripts); the records live
in `.models/oracle-*/`, outside git.

## What is here

| file | what |
|---|---|
| `release_gate.sh` | the gate above (also appends to `live.log`) |
| `pinned.sh` | runs a benchmark on a fixed amount of processor and memory: pinned cores (`taskset`), a hard RAM cap with no swap (a user cgroup), and the machine's load recorded before and after |
| `spdx_check.py` | every source file carries the SPDX licence of its layer (LICENSING.md); in the gate |
| `live.sh` | runs one step with its output appended to `live.log`, which `sAGI/savante.py --mode view` shows live |
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
| `test_console.py` | 0.3.7: the bankML console (`sAGI/console.py`) with no engine: its routes, its security (loopback Host, same-origin JSON POSTs, CSP), the SELF block as the model reads it (every value with its unit, "not measured" for a null), the persona's doctrine root, and the Infotags metadata (ERC-721 attributes, an RFC 6962 root over the log's lines); `python3 testing/test_console.py` |
| `capi/printf_oracle.c` | 0.3.2: the C API's `bankml_log` (a C-variadic function defined in Rust) against libc `snprintf`, byte for byte, on every supported conversion; every unsupported one must give its marker and read no argument it cannot type |
| `capi/chat.c`, `capi/capi_oracle.py` | 0.3.2 (0.3.4: also the O4 models against their records): a C program embedding `libbankml`; `--chat` compares `bankml_chat` with a live `serve --native` (ternary) and llama-server's record (1-bit), and checks the refusals; `--printf` builds and runs the printf oracle |
| `grammar_oracle.cpp`, `grammar_oracle.py` | 0.3.3: llama.cpp b11192's own grammar sampler through libllama's public API on the pinned vocabulary: every token's piece and end flag, then the whole-vocabulary mask before every token of 196 runs (12 grammars — llama-server's JSON-mode grammar with its prefill, every grammar in llama.cpp's `grammars/`, three written for token terminals, edges and UTF-8 — × inputs × the tokenizer's tokens and one token per byte) and where llama.cpp rejects; for `oracle_grammar_masks` |
| `json_oracle.py` | 0.3.3: `--record MODEL` launches llama-server b11192 (Savante's flags, `--verbose`) and records its `/v1/chat/completions` answers under `response_format: json_object` (and two user `grammar`s on the 1-bit model), greedy and seeded, tokens and grammar included, for `oracle_json_mode*`; `--bankml` sends a subset through a live `serve --native` on `/v1` (streamed once) and Ollama's `/api/chat` with `format: "json"` |
| `f16_oracle.py` | 0.3.4: the shipped ggml's `mul_mat` on F16 weights through ctypes (one column → `ggml_vec_dot_f16`, two or more → llamafile's tinyBLAS, and the fallbacks): SmolLM2's real matrices by 1–28 columns and synthetic shapes for the tails; every F16 tensor widened by `ggml_fp16_to_fp32_row`; for `oracle_ggml_b11192_f16` |
| `schema_oracle.cpp`, `schema_oracle.py` | O6b (0.3.5: each template, `schemas-<stem>.jsonl` beside the Qwen3 `schemas.jsonl`): llama.cpp b11192's own `json_schema_to_grammar` and the chat path's schema grammar (`common_chat_templates_apply`, jinja, the pinned template read from the GGUF header, thinking off), called inside the release's `libllama-common.so` — no model loaded, no server — over 173 schemas: llama.cpp's own test cases (`json_schema_cases.json`, 81) and `json_schema_corpus.json` (Pydantic-shaped schemas like mindX's, the model oracle's schemas, edges and every refusal path); for `oracle_schema_grammars` |
| `json_schema_oracle.py` | O6b (0.3.5: every native model, `--bankml STEM [NAME]`, and answers cut by `max_tokens`): `--record MODEL` records llama-server b11192's answers under JSON schemas (`response_format: json_schema`, top-level `json_schema`, `json_object` with a schema; objects, enums, ranges, `$defs`, a pattern, top-level array / enum / integer), greedy and seeded, with each request's exact text, for `oracle_json_schema*`; `--bankml` sends a subset through a live `serve --native` on `/v1` (streamed once) and Ollama's `/api/chat` with `format: <schema>` |
| `convert_oracle.py` | O5: `--record DIR OUT` runs llama.cpp b11192's `convert_hf_to_gguf.py --outtype f16`; `--bankml DIR ORACLE` runs `bankml convert` and compares the sha256; `--compare A B` diffs two GGUFs key by key and tensor by tensor; `--chkhsh DIR` measures llama.cpp's pre-tokenizer hash and bankml's vocabulary+merges sha256 (the evidence for `convert.rs`'s `KNOWN_TOKENIZERS`); `--names OUT` writes gguf-py's name heuristics for `oracle_name_heuristics` |
| `pins/` | O5: the FORK.json of each conversion proven byte-identical (SmolLM2-135M-Instruct, mindx-gen39), with every input's sha256 and the source revision |
| `content_oracle.cpp`, `content_oracle.py` | 0.3.5: llama.cpp b11192's own `common_chat_parse` with the parser llama-server builds per request (libllama-common, no model), on every prefix of every recorded constrained answer and edge cases, per template; for `oracle_json_content` (the content rule, and the raw-text answer for an empty parse) |
| `persona_oracle.py` | 0.3.5 (O5 end to end): `--record MODELFILE DIR` creates mindx-gen39 from promote.py's persona Modelfile two ways (`FROM` the merged directory, which must be named `mindx-gen39` for the pin; and `FROM mindx-gen39` in place) in scratch registries under `.models/oracle-persona/`, then records llama-server b11192 given the persona as the system message; `--bankml` asks the created model the user's turns alone through `/api/chat`, `/api/generate` and `/v1` (each from an empty slot) and checks `/api/ps`; for `oracle_persona_layer` |
| `guard_agree.py` | runs both guards on every synthetic case and any real file given; exit 0 only if the JSON is identical |
| `model_oracle.py` | P3: the whole Qwen3 (and since 0.3.4 Llama, F16, tied-embedding) graph from the shipped ggml through ctypes, one layer at a time: per token and layer the sha256 of `l_out`, then `result_norm` and the logits; `python3 testing/model_oracle.py GGUF LIBDIR [OUT]`, for `oracle_forward_model*` |
| `greedy_oracle.py` | P3: llama-server b11192's greedy tokens for short chat prompts (`/apply-template`, `/tokenize`, `/completion`, cache off); `--long` (≥ 64-token prompts, the tiled kernel) and `--deep` (past 256 cells, the split-KV kernel); `python3 testing/greedy_oracle.py [URL] [N_PREDICT] [--long \| --deep]`, for `oracle_greedy_llama_server*` |
| `sample_oracle.py` | P3: llama-server's seeded sampling (top-k, top-p, min-p, temperature), each case with the parameters the server read back (`generation_settings`); `python3 testing/sample_oracle.py [URL] [N_PREDICT]`, for `oracle_sample_llama_server` and `oracle_llama_server_*` |
| `serve_oracle.py` | 0.3.0: Savante-style conversations through a FRESH llama-server's `/v1/chat/completions`, turn after turn (text, counts, `timings.cache_n`), for `oracle_native_serve*`; `--bankml [STEM [NAME]]` is the live Ollama-shape check (0.3.1, 0.3.4) |
| `train_oracle.py` | mindXtrain's own Python (author stage `data/scripts.py`, score stage `eval/imprint.py`) on every persona and 3,000 random sets, for `oracle_train_script` and `oracle_train_imprint`; run with mindXtrain's interpreter: `~/mindxtrain/.venv/bin/python testing/train_oracle.py [~/mindxtrain]` |
| `penalty_oracle.py` | O2 (0.3.6): llama-server's penalties sampler (repeat, frequency, presence over `repeat_last_n`, the prompt in the window), greedy and seeded, refusals kept with their message, for `oracle_penalties*`; 0.3.7: `--kind sampler` records the rest of the chain (typical-p, top-n-σ, XTC, dynamic temperature, DRY) for `oracle_samplers*`; `--bankml STEM [NAME]` replays every case live through `/v1` and every fourth through `/api/chat` |
| `sort_oracle.cpp` | 0.3.7: libstdc++'s own `std::sort` on 876 float-keyed index arrays (sizes 0–1,000, heavy ties, sorted, reversed, equal), the order typical-p leaves equal scores in; `g++ -O2 -o target/sort_oracle testing/sort_oracle.cpp && target/sort_oracle > .models/oracle-sort/cases.txt` (the gate does this), for `oracle_std_sort` |
| `context_oracle.py` | 0.3.8: the context limit with context shift off, at `--ctx 256`: a full context stops with `"length"`, a prompt that does not fit gets llama-server's 400 body; `--record STEM` (needs `BANKML_GGML_LIB`), then `--bankml STEM` |
| `slot_oracle.py` | 0.3.8: `POST /slots/0?action=save\|restore\|erase` on `serve --native`; the oracle is the engine itself (an answer after a restore == an empty slot's, `cache_n` > 0, across a restart) and llama-server's refusals; `python3 testing/slot_oracle.py [STEM]` |
| `session_oracle.py` | 0.3.8: the single-slot contract: three conversations taking turns through llama-server's host prompt cache (text, counts, `cache_n`, turn by turn), and four simultaneous requests all answered as if asked alone; `--record STEM`, then `--bankml STEM` |
| `kv_oracle.py` | 0.3.9: the q8_0 KV cache: llama-server `--cache-type-k q8_0 --cache-type-v q8_0` against `serve --native` with `BANKML_CACHE_TYPE=q8_0` (greedy, seeded, a long prefill, a two-turn conversation); `--record STEM`, then `--bankml STEM` |
| `logprobs_oracle.py` | 0.3.8: `/v1` `logprobs` and `top_logprobs`: ids, texts and bytes equal, every logprob the same 32-bit float; streamed cases (`stream-*`) compare every chunk's delta and entries; refusals with the same status and message; `--record STEM`, then `--bankml STEM` |
| `decode_ab.py` | 0.3.9: decode speed against llama-server b11192 in pairs (fresh servers each round, alternating order, each side's own `timings`); a round whose answers are not token-identical is refused; run pinned: `BANKML_PIN_CPUS=1,2,3 BANKML_PIN_MEM=3000M testing/pinned.sh python3 testing/decode_ab.py Bonsai-8B-Q1_0 [rounds] [threads]`. Not in the gate (it needs a quiet machine) |
| `json_schema_cases.json`, `json_schema_corpus.json` | the schema oracle's inputs: llama.cpp's own 81 test cases, and the Pydantic-shaped corpus with edges and refusals |
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
| Modelfile parser (Ollama's quoting, mindX's own Modelfiles), `quote` round trip, spec refusals, manifest digest, layer rules, create over a pin, create → tags → show → copy → delete through the HTTP router | `create.rs` | O5: `bankml create` and the Ollama endpoints, with no model loaded |
| name heuristics, regexes, permutation, metadata order; `oracle_convert_b11192` *(real)*, `oracle_name_heuristics` *(real)* | `convert.rs` | O5: `bankml convert` against llama.cpp b11192's `convert_hf_to_gguf.py`, byte for byte (`convert_oracle.py`) |
| `create_writes_a_layer_over_a_pin`, `convert_refuses_with_the_reason` | `cli.rs` | O5's CLI: exit codes, the manifest, the refusals |
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
| `keep_alive_as_ollama_reads_it`, `options_map_onto_the_engine`, `refusals_say_why`, `ndjson_framing`, `stop_strings_end_the_answer`, `names_resolve_to_pins`, `timestamps_are_rfc3339` | `bankML/ollama.rs` | 0.3.1: Ollama's API — keep_alive (`5m`, `1h`, seconds, 0, −1), option mapping and refusals, NDJSON lines, stop strings held back across pieces, registry names and aliases |
| `oracle_forward_attention` *(real)* | `forward.rs` | P3 steps five and six: layer 0's `kqv_out`, `attn_out`, `ffn_inp`, `l_out` (f16 cache, causal flash attention, `wo`, residual, the FFN block) bit-exact against the shipped ggml, 112 of 112 rows |
| `oracle_forward_model` *(real)* | `forward.rs` | P3 step seven: the whole model (36 layers' `l_out`, `result_norm`, all 151,669 logits × 28 tokens) bit-exact against the shipped ggml, 1,064 of 1,064 rows |
| `oracle_greedy_llama_server` *(real)* | `forward.rs` | end to end: bankml's greedy tokens == llama-server b11192's on 6 of 6 chat prompts (164 tokens) |
| `oracle_forward_model_ternary`, `oracle_greedy_llama_server_ternary` *(real)* | `forward.rs` | P3 step eight: the ternary model, whole graph bit-exact (1,064 of 1,064 rows) and llama-server's greedy tokens on 6 of 6 prompts |
| `oracle_forward_attention_tiled`, `oracle_greedy_llama_server_long` *(real)* | `bankML/forward.rs` | P3 step nine: ggml's tiled flash attention on a 150-row micro-batch (150 of 150 rows) and llama-server's tokens on 6 of 6 long prompts |
| `oracle_forward_attention_split`, `oracle_greedy_llama_server_deep` *(real)* | `bankML/forward.rs` | P3 step ten: ggml's split-KV decode kernel (14 of 14 rows, 257–1,000 cells, 3 and 4 threads) and llama-server's tokens on 3 continuations running to ~300 cells (600 tokens) |
| `oracle_sample_llama_server` *(real)* | `bankML/forward.rs`, `bankML/sampler.rs` | P3 step eleven: llama-server's seeded sampling, 40 of 40 continuations (1,175 tokens) identical |
| `gpu_q1_0_mat_vec_bit_exact` | `bankML/gpu/kernels.rs` | both Q1_0 GPU kernels on every usable card, bit-exact against the CPU kernel (skipped without a GPU) |
| `oracle_train_script`, `oracle_train_imprint` | `bankML/train/` | mindXtrain's author and score stages, identical to mindXtrain's Python (84 of 84; 3,000 of 3,000) |
| `oracle_native_serve` *(real)* | `bankML/native.rs` | 0.3.0: three Savante-style conversations through llama-server's own chat endpoint, replayed through bankML's native engine: text, counts and prompt-cache reuse identical on 9 of 9 turns |
| `serve_oracle_ollama_shape` *(real, gate)* | `testing/serve_oracle.py --bankml` | 0.3.1: the same conversations through a live `serve --native`: `/api/chat` (NDJSON for the first), `/v1/chat/completions`, and `/v1` again after unload and reload, each equal to the others and to llama-server's record, turn by turn |
| `grammar::tests::*` | `bankML/grammar.rs` | 0.3.3: the server's JSON grammar parses; parse errors; a toy vocabulary's masks and accepts; partial UTF-8 across tokens; the C decoder's reading; requests resolved as llama-server resolves `response_format`/`json_schema`/`grammar` and Ollama's `format`; the JSON content rule, growth-only |
| `oracle_grammar_masks` *(real)* | `bankML/grammar.rs` | 0.3.3: all 151,669 token pieces and the end set; then every recorded whole-vocabulary mask (1,645) and rejection (116) of 196 runs identical to llama.cpp b11192's grammar sampler; times one mask |
| `oracle_json_mode`, `oracle_json_mode_ternary` *(real)* | `bankML/native.rs` | 0.3.3: llama-server's answers under JSON mode (and two GBNF grammars on the 1-bit model), greedy and seeded, from an empty cache: the same tokens, raw text, content, finish and counts; the server's grammar and generation prompt are bankML's |
| `json_oracle_live` *(real, gate)* | `testing/json_oracle.py --bankml` | 0.3.3: a live `serve --native`: `/v1` with `response_format` (once streamed) and `grammar`, and Ollama's `/api/chat` with `format: "json"`, each equal to llama-server's record |
| `schema::tests::*` | `bankML/schema.rs` | O6b: nlohmann's JSON reading and printing (key order, duplicates, `1` vs `1.0`, exponents); `{"type": "object"}` converts to JSON mode's constant; llama.cpp's own 81 `test-json-schema-to-grammar` cases with their expected grammars (each also parsed); refusals with llama.cpp's messages |
| `oracle_schema_grammars` *(gate)* | `bankML/schema.rs` | O6b: 173 of 173 schemas give llama.cpp b11192's grammar text byte for byte, bare and wrapped for the chat path (148 grammars each), or its refusal with the same message (24 bare, 20 wrapped); the one schema llama.cpp turns into a grammar that is not UTF-8 is refused by bankML; every grammar parses |
| `oracle_json_schema`, `oracle_json_schema_ternary` *(real)* | `bankML/native.rs` | O6b: llama-server's answers under JSON schemas, replayed from an empty cache: tokens, raw text, content, finish and counts; the server's grammar and generation prompt are bankML's |
| `json_schema_oracle_live` *(real, gate)* | `testing/json_schema_oracle.py --bankml` | O6b: a live `serve --native`: `/v1` with each schema shape (once streamed) and Ollama's `/api/chat` with `format: <schema>`, each equal to llama-server's record |
| `printf::tests::*`, `bankml_log_equals_snprintf`, `null_safe_and_refusing` | `capi/src/` | 0.3.2: the formatter on a typed argument queue (a read of the wrong type fails), the Rust-defined variadic `bankml_log` against libc `snprintf` called from Rust, NULL safety and refusals |
| `printf oracle` *(gate)* | `testing/capi/printf_oracle.c` | 0.3.2: from C, `bankml_log` == libc `snprintf` on every supported case (47 of 47), every unsupported case marked (17 of 17), `%n` never written |
| `capi_chat_oracle` *(real, gate)* | `testing/capi/capi_oracle.py --chat` | 0.3.2: `bankml_chat` == `serve --native` turn by turn on Ternary-Bonsai-8B (text, streamed pieces, counts, cache reuse, receipt hashes) and == llama-server's record on Bonsai-8B Q1_0; Bonsai-1.7B and an unpinned file refused |
| `oracle_forward_swiglu_sweep` | `forward.rs` | SwiGLU with ggml's vectorized expf on 24,600 values over ±120 and the edges, every output's bits |
| `oracle_forward_qkv_rope` *(real)* | `forward.rs` | P3 step four: layer 0's `Qcur`/`Kcur`/`Vcur` (projections, head norms, YaRN RoPE) bit-exact against the shipped ggml, 140 of 140 rows, positions 0–27 and to 63,214 |
| `oracle_forward_embed_norm` *(real)*, `rms_norm_of_a_constant_row` | `forward.rs` | P3 step three: `inp_embd` and `attn_norm-0` bit-exact against the shipped ggml on 300 of 300 tokens |
| `oracle_chat_template` *(real)*, `a_short_conversation`, `python_split_semantics` | `chat.rs` | P3 step two: every recorded conversation byte-identical to llama.cpp b11192's rendering (317 of 317) |
| `oracle_tokenizer` *(real)*, `pretokenizer_shapes`, `byte_chars_are_gpt2s` | `tokenizer.rs` | P3 step one: every recorded case token-identical to llama.cpp b11192 (4,258 of 4,258; re-recorded in 0.3.4 on the grown corpus, 4,346 of 4,346, after the `</s>` fix) |
| `avx2_equals_the_scalar_models`, `the_two_reductions_differ`, `mad_and_scale_match_the_scalar_definitions` | `bankML/f16.rs` | 0.3.4: the F16 kernels' AVX2 paths equal their scalar definitions (tails, fallbacks, 1–6 columns); ggml's two F16 reductions give different bits, so the path matters |
| `oracle_ggml_b11192_f16` *(real)* | `bankML/f16.rs` | 0.3.4: 211 of 211 F16 tensors widened and 552,268 of 552,268 `mul_mat` elements bit-exact against the shipped ggml (61 tinyBLAS products, 26 `vec_dot_f16`) |
| `oracle_forward_model_bonsai_1_7b`, `oracle_forward_model_llama_f16` *(real)* | `bankML/forward.rs` | 0.3.4: the whole model bit-exact — Bonsai-1.7B (tied embeddings) 840 of 840 rows; SmolLM2-135M-Instruct and mindx-gen39 (Llama, F16, tied) 800 of 800 each |
| `oracle_llama_server_bonsai_1_7b`, `oracle_llama_server_llama_f16` *(real)* | `bankML/forward.rs` | 0.3.4: llama-server's tokens on each model: greedy short, long and deep (SmolLM2, mindx-gen39), short (Bonsai-1.7B), seeded 40 of 40 each |
| `oracle_native_serve_o4` *(real)* | `bankML/native.rs` | 0.3.4: the Savante conversations (9 of 9 turns) and JSON mode (23 of 23) on Bonsai-1.7B, SmolLM2-135M-Instruct and mindx-gen39 |
| `oracle_tokenizer_smollm`, `smollm_pretokenizer_shapes` | `bankML/tokenizer.rs` | 0.3.4: SmolLM2's vocabulary and `smollm` pre-tokenizer, 4,346 of 4,346 cases |
| `oracle_chat_template_chatml`, `chatml_templates` | `bankML/chat.rs` | 0.3.4: SmolLM2-Instruct's template and mindx-gen39's ChatML, 317 of 317 each |
| `o4_live` *(real, gate)* | `testing/serve_oracle.py --bankml STEM [NAME]`, `testing/json_oracle.py --bankml STEM [NAME]` | 0.3.4: the live Ollama-shape and JSON-mode checks on Bonsai-1.7B, SmolLM2-135M-Instruct and mindx-gen39 (asked for as `mindx-gen39`) |
| `type_ids_match_mainline`, `verified_json_escapes_what_the_header_says` | `bankml.rs` | type ids; `/bankml`'s JSON stays valid for a hostile model name |
| `hardware_path_equals_portable_on_every_length` | `sha256.rs` | SHA-NI equals the portable rounds (lengths 0–1,000, 4 KiB, 64 KiB, split updates) |
| `oracle_json_schema_o4` *(real)* | `bankML/native.rs` | 0.3.5: answers under JSON schemas on Bonsai-1.7B, SmolLM2-135M-Instruct and mindx-gen39 (each template's own schema grammar), 56 of 56 each |
| `oracle_json_content` *(gate)* | `bankML/grammar.rs` | 0.3.5: the content of a constrained answer, against llama.cpp's own `common_chat_parse` (`content_oracle.py`): every prefix of every recorded answer and edge cases, 30,063 of 30,063 texts per template |
| `oracle_persona_layer` *(real)* | `bankML/create.rs` | 0.3.5: mindx-gen39 created with promote.py's persona Modelfile, asked the user's turns alone, gives llama-server's tokens for the persona as the system message (27 of 27, two ways) |
| `oracle_penalties`, `oracle_penalties_8b` *(real)* | `bankML/native.rs` | 0.3.6: the repeat, frequency and presence penalties, greedy and seeded, token-identical to llama-server (56 of 56 on each of mindx-gen39, Bonsai-1.7B and Bonsai-8B), refusals with its message (12 of 12) |
| `oracle_samplers`, `oracle_samplers_8b` *(real)* | `bankML/native.rs` | 0.3.7: typical-p, top-n-σ, XTC, dynamic temperature and DRY, alone and together (76 of 76 on mindx-gen39 and on Bonsai-1.7B, 16 of 16 refusals each; Bonsai-8B in `oracle_samplers_8b`, whose count the CHANGELOG does not yet give) |
| `oracle_std_sort` *(gate)*, `partial_sort_orders_the_top`, `mt19937_reference_value` | `bankML/sampler.rs` | 0.3.7: bankML's port of libstdc++'s `std::sort` leaves 876 of 876 arrays in the same order as `sort_oracle.cpp`; the partial sort and the generator |
| `oracle_ggml_b11192_q8_0_kv_kernels` *(real)* | `bankML/forward.rs` | 0.3.9: the q8_0 KV cache's kernels against the shipped haswell library: `quantize_row_q8_0` byte for byte and `vec_dot_q8_0_q8_0` bit for bit (4,000 each, saturating cases included) |
| `rates_and_energy_are_measured_or_null`, `the_ring_keeps_the_last_ones` | `bankML/metrics.rs` | 0.3.7: a measurement is taken or `null`, never estimated; the ring of 256 |
| `interleaved_conversations_find_their_prefix_again` | `bankML/prompt_cache.rs` | 0.3.8: the host prompt cache hands back the state that keeps more of the prompt |
| `fit_messages_as_ollama` | `bankML/native.rs` | 0.3.5: `num_ctx` as Ollama's `chatPrompt` applies it (oldest messages dropped first) |
| `logprob_texts_as_llama_server_writes_them` | `bankML/serve.rs` | 0.3.8: logprob texts and bytes, incomplete UTF-8 cut as llama-server cuts it |

## Results

| version | record | headline |
|---|---|---|
| 0.3.6 | `results/0.3.6.txt` | O2: the repeat, frequency and presence penalties token-identical to llama-server (56 / 56 on each of three models; live 85 / 85); the last three stages run with `BANKML_GPU=off`, marked in the record |
| 0.3.5 | `results/0.3.5.txt` | O6b and O5: JSON schemas (173 schemas × 3 templates against llama.cpp's own code; answers 28 / 28, 11 / 11, 56 / 56 × 3), the content rule (30,063 / 30,063 per template), `bankml create` with mindX's persona layer (27 / 27, two ways); the last three stages run with `BANKML_GPU=off`, marked in the record |
| 0.3.4 | `results/0.3.4.txt` | O4: mindX's own `mindx-gen39`, SmolLM2-135M-Instruct (Llama, F16) and Bonsai-1.7B (tied embeddings) token-identical to llama-server on every oracle family; F16 products bit-exact against ggml |
| 0.3.3 | `results/0.3.3.txt` | JSON mode: llama-server's grammar, prefill and redraw; token-identical greedy and seeded |
| 0.3.2 | `results/0.3.2.txt` | the C API (`libbankml`): `bankml_chat` identical to `serve --native` and llama-server's record; `bankml_log` identical to libc `snprintf`; Rust 1.99 passing every bit-exact oracle |
| 0.3.1 | `results/0.3.1.txt` | Ollama's API on `serve --native`: `/api/chat` identical to `/v1` and llama-server's record, unload/reload unchanged |
| 0.3.0 | `results/0.3.0.txt` | milestone: Savante answered by bankML's own forward pass; conversations identical to llama-server's (9 of 9 turns) |
| 0.2.14 | `results/0.2.14.txt` | the GPU in the forward pass, every token oracle exact with it working; the exact FMA; the on-card oracle catches an unfused driver |
| 0.2.13 | `results/0.2.13.txt` | GPU kernels bit-exact on the Vega 3; mindXtrain author and score stages identical |
| 0.2.12 | `results/0.2.12.txt` | batched prefill (checked by every prefill oracle); the GPU component's discovery |
| 0.2.11 | `results/0.2.11.txt` | P3 step eleven: sampling — llama-server's chain, same seed, same tokens (40 of 40) |
| 0.2.10 | `results/0.2.10.txt` | P3 step ten: long contexts — the split-KV kernel (14 of 14) and 600 tokens past 256 cells identical |
| 0.2.9 | `results/0.2.9.txt` | P3 step nine: long prompts — the tiled kernel (150 of 150) and llama-server's tokens on 6 of 6 long prompts |
| 0.2.8 | `results/0.2.8.txt` | P3 step eight: the ternary model, bit-exact and token-identical; about 8× llama-server end to end |
| 0.2.7 | `results/0.2.7.txt` | P3 step seven: the whole model bit-exact (1,064 of 1,064 rows); greedy generation token-identical to llama-server (6 of 6 prompts) |
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

0.3.7, 0.3.8 and 0.3.9 are unreleased; their measurements are in [CHANGELOG.md](../CHANGELOG.md) until each has a
gate record.

## Tools beside the gate (`tools/`)

| file | what |
|---|---|
| `tools/cards.py` | draws the README's speed cards (`docs/cards/*.svg`) from a gate record, so a card cannot drift from what was measured; `python3 tools/cards.py [testing/results/<version>.txt]` (default: the newest) |
| `tools/makecards.py` | draws a page's share cards (1200×630, 1200×1200 and a 180×180 touch icon) with the words given as arguments; needs Pillow and the Noto Sans fonts |
| `tools/seo.py` | audits a page's search and sharing metadata as a crawler and a link preview read them, and checks each card's real size against the declared one; standard library only; `python3 tools/seo.py URL`, or `page.html --base URL` / `--offline` |
| `tools/bashmoji.sh` | the installer's glyphs and colours, vendored from cryptoAGI/bashmoji |
