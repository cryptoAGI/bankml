# TODO

What comes next, and where each item came from. Every item ends in a measured, reproducible result, or it is recorded
as rejected with its numbers (the house rule: the same bits first, then the speed). Sources:
**R** = [research.md](research.md) (the field, 2026-09-29) · **K** = the clean-room study of KoboldCpp (docs only) ·
**V** = the vLLM code review (main @ `36768d1`, Apache-2.0) · **Rs** = the Rust 1.95 study (the `rust` skill) ·
**A** = the audits of 0.1.5, 0.1.6, 0.1.7–0.1.8 and 0.1.9.

## 0.1.8 — speed and efficiency (done)

- [x] **SHA-NI for the model pin** (Rs). 5.5× (0.23 s vs 1.25 s per 248 MB); the 8B model in 2.9 s; bit-exact
  against the portable rounds and `sha256sum`.
- [x] **A history window that keeps the prompt cache warm** (K: Smart Context's reserve; V: chained block hashes show
  why a one-exchange slide defeats reuse). The window moves 8 times in 60 turns instead of 48; 39 of 47 turns reuse
  the previous prompt. The chat's footer (clock, receipt) is no longer sent to the model.
- [x] **Resources sliders** (operator): CPU threads and a RAM budget that becomes the engine's context, applied by a
  verified restart; usage read by bankml's own `sys.rs` (psutil's job, no crates) at `GET /bankml/usage`.
- [x] **Licensing layers** (operator): `MIT OR Apache-2.0`; `crypto/` GPL-3.0-only; `agpl/` walled off; SPDX gate.
- [x] **Speculative decoding verdict** (R, K, V). Draft model (Bonsai-1.7B): token-identical, no gain beyond noise.
  n-gram, unpinned and interleaved: paired +0.09, +0.11, −0.06 tok/s (medians 0.61 vs 0.59): within noise, not
  adopted. On fixed resources (`testing/pinned.sh`: cores 2–3, 2 threads, 2.5 GB): +0.01, +0.36, +0.16 while the
  baseline swung 0.36–1.40 tok/s with the machine's load (load average up to 3.8). Ahead in 5 of 6 pairs, exact,
  no memory cost → **opt-in** (`--spec-ngram`, a Resources checkbox), not the default. Re-measure on an idle machine
  ("The production server" below).
- [x] **Fixed resources for every benchmark** (operator): `testing/pinned.sh` pins cores, caps memory in a user
  cgroup, and records the load before and after.
- [x] **A refusal no longer resets the connection** (found by the CLI suite under load): serve read and discarded
  nothing before closing, so an unread body made the kernel send RST and the client could lose the 403/415. It now
  drains the declared body first; the suite passes 10/10 in a row.
- [ ] **Measure the prefill/decode knobs** (K, V): `-t 2/3/4` with `-tb 4`; `-ub 128/256/512`; `-fa on`;
  `-ctv q8_0` (then `-ctk q8_0`) with RSS and exactness recorded. Adopt only what is faster *and* token-identical,
  or clearly label what is not exact.
- [ ] **Is batched verification cheap here?** (V) `llama-bench -p 1,2,4,8,16`: speculation can only pay if a
  k-token verify costs much less than k single steps on this CPU.

## Rejected, with reasons (keep)

- **`--cache-reuse` / KV shifting** (K recommended it; V showed why not): re-rotating cached keys and keeping KV that
  depended on dropped tokens is **not token-identical**. bankml's window policy gets most of the benefit exactly.
- **vLLM's CPU backend as an engine** (V): no GGUF on x86, fast paths need AVX-512/AMX, ~2.9× slower than llama.cpp
  on CPU (arXiv:2608.23841).
- **KoboldCpp code** (K): AGPL-3.0; ideas only, from its documentation.
- **fp8 KV cache** (V): needs AVX-512/AMX on vLLM's CPU backend; q8_0 via llama.cpp is the equivalent to measure.

## 0.1.9 — done

- [x] Third audit (0.1.7–0.1.8): 13 findings fixed, including silent loss of history (HIGH).
- [x] **bge-m3 live**: 9.2 s first call, 1.4 s after, 1.14 GB loaded; hybrid search on the real `.history`.
- [x] **The system prompt's KV saved across restarts** (K): first answer after a restart 132 s → 15.4 s, identical.
- [x] ~~madvise for hashing~~ (Rs): measured, rejected — cold 364 MB/s vs warm 419 MB/s; the disk is not the limit.
- [x] ~~PGO for `bankml serve`~~ (Rs): measured, rejected — 0.71 ms per request against 15–130 s answers.
- [ ] Target-feature 1.1 clean-up (Rs): same bits, fewer `unsafe`; no speed at stake → after 0.2.0.

## 0.2.0 — milestone (done)

- [x] A full docs pass: 23 corrections from a line-by-line review against the code and the gate records. Among them,
  the whole-token ternary figures are dated, with the reason they cannot be reproduced on this laptop today.
- [x] The fourth audit (0.1.9): the window kept stable per session, the trimmed-history note only when the context
  forced it, a clear refusal when a prompt cannot fit, slot saves in the background, the 503 path off the accept
  thread.
- [x] **The plain-AVX2 `Q2_0` kernel prepared for llama.cpp** (`upstream/`): bit-exact against the shipped library on
  200,000 of 200,000 random cases, 3.4× its scalar path. Submitting the pull request is the authors' decision.
- [x] The oracle method written up (`docs/oracles.md`, `upstream/README.md`).

## The production server (and any machine with ≥ 3 GB free)

- [x] **Re-measure the whole-token ternary budget** with the 2.31 GB model resident (done 2026-09-29: 0.231 s at three threads, 9.45×; browser and chat engine closed, 2.25 of 2.31 GB cached) in the page cache (on this laptop
  it no longer fits: 1.04 GB stayed resident after a full read). The 0.0.3–0.0.6 records say 0.23–0.25 s per token at
  three threads; the gates since 0.1.0, disk-bound, say 4.5–5.2 s.
- [ ] Re-measure n-gram speculation on an idle machine (ahead in 5 of 6 pairs here, within the noise).

## After 0.2.0 — P3 and beyond

- [x] **P3, step one (0.2.1): the tokenizer.** GPT-2 byte-level BPE with the Qwen2 pre-tokenizer, no crates,
  token-identical to llama.cpp on 4,258 of 4,258 recorded cases (a 2,000-string Unicode fuzz set included).
- [x] **P3: the Qwen3 forward pass, token-identical to llama.cpp** (0.2.2–0.2.11, one verified step per release):
  the chat template, the embedding and norms, Q/K/V with YaRN RoPE, all three of ggml's CPU attention kernels, the
  FFN with ggml's own `expf`, the whole model's logits for the 1-bit and ternary files, llama.cpp's micro-batching,
  and seeded sampling. Batched prefill (0.2.12). **0.3.0: `bankml serve --native`**, Savante answered by bankML's
  own forward pass, with conversations identical to llama-server's (9 of 9 turns). The per-release record is in
  CHANGELOG.md; the road on from here is the next section.
- [ ] **A block-hashed prefix cache in P3** (V: `hash(parent, tokens[16], extra)`, a HashMap to blocks, ref counts,
  an intrusive LRU free list, ~300 lines). Exact by construction: only the unchanged prefix is reused.
- [ ] **Prompt-lookup speculation in P3** (V: the n-gram proposer, a KMP scan over the reversed history, ~60 lines,
  no model memory) with greedy verification (accept until the first mismatch, plus the bonus token). Exact only with
  batch-invariant kernels.
- [ ] **Signed receipts** (R: EigenAI and TEE stacks show the direction): an operator key signs each receipt, in
  `crypto/` under GPL-3.0-only (LICENSING.md).
- [ ] **NEON kernels** for ARM (R: upstream `Q1_0`/`Q2_0` already have NEON; P5, handheld).
- [ ] Real activations in the oracle (TECHNICAL §VI); a Zen 3 row; AVX-512 VNNI where present.
- [x] **The integrated GPU (Radeon Vega 3, shared DDR4)**, (b): Vulkan compute kernels for `Q1_0`, bit-exact against
  the CPU reference (0.2.13), working inside the forward pass (0.2.14). What's left is in the road below. (a),
  measuring llama.cpp's own Vulkan build for standard models, is still open as a comparison.
- [ ] Optional external engines at a process boundary (bitnet.cpp, KoboldCpp) as benchmark rivals and cross-checks,
  never linked (LICENSING.md).

## 0.3.1 — Ollama's API, natively (done)

- [x] **Phase O1 of [OLLAMA.md](OLLAMA.md)**: `/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`,
  `/api/generate` on `bankml serve --native`; a model registry (`--registry`) with one resident model, each load
  verified, and `keep_alive`. Native only: what the forward pass cannot do is refused with a reason, never proxied.
  Oracle: `/api/chat` gives the same turns as `/v1/chat/completions` and llama-server's record, and unload/reload
  changes nothing (`serve_oracle.py --bankml`, in the gate).
- [ ] **Bench Crane's CPU ternary path** against bankML on the same Ternary-Bonsai-8B file, pinned
  (`testing/pinned.sh`): the one Rust engine that loads the Bonsai formats (the field survey in OLLAMA.md).

**The Ollama track (O2–O8)** is folded into the milestones below; each item is marked with its phase. The gap matrix
of what mindX asks of Ollama, with the evidence, is in [OLLAMA.md](OLLAMA.md).

## 0.3.2 — toolchain and the C API (done)

- [x] **Rust 1.99, pinned** (`rust-toolchain.toml`, `rust-version = "1.99"`); the one new clippy lint
  (`chunks_exact_to_as_chunks`, 29 sites) fixed by `as_chunks`, every bit-exact oracle passing on the new compiler.
- [x] **The C API** (`capi/`, `libbankml.so`/`.a`, `capi/include/bankml.h`, [CAPI.md](CAPI.md)): `bankml_open` with
  `serve`'s verification, `bankml_chat` with `serve --native`'s answer and receipt (one code path), and a printf-style
  `bankml_log` defined in Rust as a C-variadic function (1.99), byte-identical to libc `snprintf`, marking what it
  does not support. Oracles in the gate. **This is the llama.h-shaped seam for embedding**: what an application that
  links llama.cpp would link instead, and the way into P5 (an NDK or iOS app embeds a C library).
- [ ] The C API's next surface, each with an oracle: tokenize/detokenize (against `/tokenize`), logits and
  `n_probs` (bit-exact), the slot's save/restore, more than one handle sharing one mapping, and a semver promise for
  the ABI (1.0.0).

## 0.3.4 — mindX's own model, natively (O4, done)

- [x] **Tied embeddings** (Bonsai-1.7B native): the logits from `token_embd`, as llama.cpp duplicates it. Whole model
  840 of 840 rows; llama-server's tokens greedy and seeded (6, 40 of 40); conversations 9 of 9; JSON mode 23 of 23.
- [x] **F16 weights** (`f16.rs`, O3's first kernel): ggml's `vec_dot_f16` for one column and llamafile's tinyBLAS for
  two or more, chosen by shape exactly as ggml chooses; 552,268 of 552,268 `mul_mat` elements bit-exact on real and
  synthetic shapes.
- [x] **The Llama graph** (NORM RoPE, no Q/K norms, GQA), SmolLM2's `smollm` pre-tokenizer (4,346 of 4,346) and two
  ChatML templates (317 of 317 each); JSON mode's grammar on ChatML.
- [x] **SmolLM2-135M-Instruct and `mindx-gen39`**, converted by b11192's own converter from pinned safetensors and
  pinned (`sAGI/models.py` `CONVERTED`): whole model 800 of 800 rows each; greedy short, long and deep, seeded 40 of
  40, conversations 9 of 9, JSON 23 of 23, live on `/v1`, `/api` and the C API.
- [x] Fixed on the way: the Qwen3 tokenizer took `</s>` as text (llama-vocab makes it CONTROL by name).
- [ ] **Q8_0** (the Qwen3-0.6B pin): the kernels (read, not built), and an oracle for Qwen's own template, which differs
  from the Bonsai template on 4 of 317 conversations.
- [ ] Decode on the F16 models at llama-server's speed: 38 vs 41 tok/s on SmolLM2 (the pool's condvar hand-off is the
  suspect; a spin was measured and rejected on this SMT laptop, PERFORMANCE.md).
- [x] mindX's seam for its lineage: convert-and-pin each new generation (O5's Rust converter and `bankml create`, with
  the persona `SYSTEM` as a derived model's layer) — **0.3.5**. Next, in mindX: `promote.py --to bankml`.

## 0.3.5 — JSON schemas, and `bankml create` (O6b, O5, done)

- [x] **JSON schemas** (`schema.rs`): the grammar on each template (173 schemas × 3 templates, llama.cpp's own code
  as the oracle); answers token-identical on all five native models (28, 11, 56 × 3), length cuts included; the
  content rule against llama.cpp's own parser (30,063 texts per template; two fixes: unfinished escapes, the raw-text
  fallback for an empty parse).
- [x] **`bankml convert`** (byte-identical to b11192's converter, run directly on gen39 and SmolLM2) and **`bankml
  create`** (`/api/create`, `/api/delete`, `/api/copy`): promote.py's persona Modelfile end to end, 27 of 27 two ways.
- [x] O5's gaps: `num_ctx` fitted as Ollama fits it (oldest messages first; the rest refused), `/api/ps` names the
  derived model.
- [ ] Tokens past `num_ctx` during an answer: Ollama's ContextShift (O2).
- [ ] `ADAPTER` in a Modelfile (LoRA merge in Rust, against PEFT's merge) — today: merge first, `FROM` the directory.
- [x] **O2's penalties (unreleased, branch `o2-penalties`)**: repeat, frequency and presence over `repeat_last_n`, the
  prompt in the window, token-identical to llama-server b11192 (`oracle_penalties`: mindx-gen39 56/56, Bonsai-1.7B
  56/56, Bonsai-8B 56/56, 12/12 refusals each; live 85/85). `no_repeat_ngram_size` is still transformers', not llama.cpp's.
- [ ] mindXtrain's bankml backend ([proposed](https://huggingface.co/PYTHAI/mindXtrain/discussions/1): `serve --to bankml`, `imprint-bankml`) needed **O2 first** (now there):
  `mindx-gen39` degenerates into repetition without a penalty, and the imprint gate uses `repetition_penalty 1.3` and
  `no_repeat_ngram_size 3` (the latter's oracle is transformers' `generate`).

## The road from 0.3.0 to 1.0.0

**What 1.0.0 means.** bankML is the engine, not a companion to one:
1. llama.cpp is needed only as the **oracle in the gate**, never at run time. The installer does not fetch
   llama-server unless you ask for it.
2. Every supported architecture × format has its whole-model, greedy, sampling and conversation oracles in the gate,
   and passes them.
3. On every supported format bankML is **at least at parity with llama.cpp**, and ahead where the kernels allow,
   measured on named reference machines. That covers CPU (x86 AVX2 and AVX-512, ARM NEON) and GPU (Vulkan).
4. Stable, documented interfaces under semver: the CLI, the HTTP endpoints, and the library API.
5. Signed receipts wherever the operator holds a key, and a verifier anyone can run.
6. Savante and mindX run on bankML by default.

Each milestone ends with a gated release, and nothing counts until its oracle passes.

### 0.4.0 — native serve complete (everything Savante and mindX ask of llama-server)
- [x] **Slot save and restore in `--native`** (**0.3.8**): the answer after a restore equals an empty slot's, across a
  restart; 19 / 19 (`slot_oracle_live`).
- [x] **A stated single-slot contract** (**0.3.8**), as llama-server `-np 1`, with its host prompt cache
  (`prompt_cache.rs`): interleaved conversations identical turn by turn, simultaneous requests queued; 14 / 14
  (`session_oracle_live`). More than one slot waits for continuous batching (O8).
- [x] The rest of llama-server's sampler chain, each with a seeded oracle: **repetition, presence and frequency
  penalties (`last_n`) — done (O2 first cut, `oracle_penalties`), retiring the refusal of the coach's
  `repeat_penalty: 1.3`**; **typical-p, DRY, XTC, top-n-σ and dynamic temperature — done (0.3.7, `oracle_samplers`,
  76 / 76 on two models)**. Not reproduced: mirostat, a custom sampler order, top-k above 128.
- [x] **JSON mode** (O6, first cut; **0.3.3**). It is not a JSON-only mask: the request turned out to need
  llama-server's own grammar. That grammar is generated from the chat template's parser, its root starts with the
  generation prompt, and it is prefilled. So llama.cpp's whole GBNF engine is ported (`grammar.rs`), with the
  sampler's redraw and llama-vocab's end set. The oracles: the grammar sampler's masks (196 of 196 runs, 1,645
  masks) and llama-server's answers, greedy and seeded (Bonsai-8B Q1_0 23 of 23; ternary 13 of 13, 534 tokens). The surfaces:
  `format: "json"` on `/api/*`, `response_format`, `json_schema` and `grammar` on `/v1`, `bankml_chat` and
  `bankml generate --json`.
- [x] **JSON schemas, the grammar** (O6b, **0.3.5**): `common/json-schema.cpp` +
  `json-schema-to-grammar.cpp` and the chat parser's wrapping ported (`bankML/schema.rs`); text oracle = llama.cpp's
  own code in libllama-common (`testing/schema_oracle.py`): 173 of 173 schemas identical, bare and wrapped, on each of
  the three templates; wired to
  `format: {schema}`, `response_format.json_schema`, `json_object` + `schema`, top-level `json_schema`, `bankml_chat`.
- [x] **JSON schemas, the answers** (**0.3.5**): llama-server's answers token-identical on both 8B models and the
  three O4 models (`oracle_json_schema*`, live in the gate). Next: tool calls.
- [ ] A faster whole-vocabulary mask: a byte trie over the token pieces, so each grammar stack walks shared prefixes
  once. Today's mask is a straight port: 24.2 ms (median) per redrawn token, 3.9 ms per token on average
  over real answers, about 1 % of a 1-bit decode step (PERFORMANCE.md). The oracle is unchanged: the same masks.
- [x] **Logprobs on `/v1/chat/completions`** (**0.3.8**): bit-exact, llama-server's entry rules (UTF-8 splits, stop
  words), 14 / 14 with five streamed (`logprobs_oracle_live`).
- [x] **Logprobs in streamed answers** (**0.3.8**): each chunk's delta and entry as llama-server's; 14 / 14 with the
  non-streamed cases. `/completion` (and `n_probs`) stays unserved natively: no client of bankML uses it.
- [x] **Behaviour at the context limit exactly as llama-server's** (**0.3.8**): stop at the full context, past it
  its 400 body; 8 / 8 (`context_oracle_live`).
- [ ] A `q8_0` KV cache, matching llama.cpp's `--cache-type-k/v q8_0` so the oracle exists; then a Hadamard-rotated
  4-bit KV (O8: what makes the boardroom's `num_ctx 8192` affordable).
- [ ] **1-bit decode at least at llama-server's speed**: 1.9–2.0 tokens/s against 2.8 before 0.3.4; after 0.3.4's
  attention work one loaded-machine pair read 2.30–2.59 against 2.33–2.48 (PERFORMANCE.md) — re-measure pinned and idle. Cache the norm
  weights, cut per-token allocations, share one quantized activation across Q/K/V and gate/up, and compute logits
  only where sampled. (O8)
- [ ] The engine setting's `auto` picks native for both the 1-bit and the ternary files.

### 0.5.0 — hardware

**Direction (operator, 2026-10-05): bankML must be GPU-ready across vendors — AMD and NVIDIA, Hugging Face's rented
NVIDIA cards included — because the models it is heading for (Qwen3.8, then IBM Granite and GLM) run on GPUs.** The
backend is already vendor-neutral (Vulkan through `dlopen`, bankML's own SPIR-V, no SDK); what it lacks is kernels
beyond Q1_0 and a verified run on anything but the integrated Vega 3. A vendor counts as supported only when its card
passes `bankml gpu --verify` bit-exact on layer-shaped data (RADV taught that a driver may not fuse `Fma`).

- [ ] **The vendor matrix, each proven by `--verify` and recorded in PERFORMANCE.md**: AMD integrated (Vega 3, done),
  AMD discrete (RDNA), NVIDIA (the first rented T4 or L4 through Hugging Face, **only on the owner's go-ahead and
  budget**), Intel. Per-vendor float behaviour (fused or unfused `Fma`, denormals) is detected by the verify step, as
  `spirv.rs fma_exact` already does for RADV.
- [ ] The F16 and Q2_0 (ternary) GPU kernels, bit-exact on the card, in `--verify`: the formats Qwen3.8 imprints and
  the ternary line need.
- [ ] Batched GPU submissions: Q/K/V and gate/up in one command buffer, one wait per group, persistent descriptor
  sets. Goal: a measured gain on the Vega 3.
- [ ] The Q2_0 (ternary) GPU kernel, bit-exact on the card, in `--verify`.
- [ ] Several cards: each takes a share of every matrix's rows, still one card's exact dot per element. Proven on a
  multi-card machine; a Hugging Face Job (`l4x4`, `a10g-largex2`) is the candidate, **only on the owner's go-ahead
  and budget**.
- [ ] A discrete-card measurement (the first rented T4 or L4 run, with the owner's approval): verify, then speed.
- [ ] AVX-512 and VNNI kernels where present (Zen 4, Intel), bit-exact; Zen 3 and Zen 4 rows in PERFORMANCE.md. (O8)
- [ ] **NEON kernels for ARM** (P5), with the oracle against llama.cpp's ARM build on an ARM machine: the first step of
  the handheld track below.
- [ ] The target-feature 1.1 clean-up: the same bits with fewer `unsafe`.
- [ ] CI builds and unit-tests across the x86 variants and aarch64.

### 0.6.0 — more models

**Direction (operator, 2026-10-05): SmolLM2 and `mindx-genN` stay as the *example* of the imprint (they prove the
dream → weights → proof-of-recall loop and that bankML serves it token-identically); their language is not the goal.
The imprint work moves to **Qwen3.8** and the **ternary** line, and from successful runs graduates to **IBM Granite**
and **GLM** models, on GPU.**

- [ ] **Qwen3.8** (`qwen3_5`, mindX `docs/QWEN38_IMPRINT_AGENDA.md`): Gated DeltaNet layers interleaved with gated
  attention, multi-token prediction; a new graph, oracle-exact against llama.cpp's own support for it before any
  imprint is served.
- [ ] The ternary line beyond Bonsai-8B: each new ternary model gets the full oracle set; imprints in ternary.
- [ ] **IBM Granite** (Granite 4.x; mindX already serves `granite4.1:3b` through Ollama) and **GLM**, on GPU, once
  the vendor matrix above stands.
- [x] The **Llama architecture** for SmolLM2 (O4, **0.3.4**): tied embeddings (Bonsai-1.7B too), SmolLM2's tokenizer
  and templates, `mindx-genN` served natively. Still open: Llama 3.x (rope factors, the `llama3` pre-tokenizer, its
  template), which are refused today.
- [ ] Qwen2 and Qwen2.5 (the coder models in the catalogue).
- [ ] ggml's standard formats as kernels bit-exact against ggml: **F16 done (0.3.4)**; `Q8_0` and BF16 next, then
  `Q4_K_M`, so that catalogue models run natively (O3).
- [ ] The encoder graph (XLM-R: LayerNorm, bidirectional attention, CLS pooling) with `/api/embed` and
  `/v1/embeddings`; the oracle is llama.cpp's bge-m3 embedding output (O7).
- [ ] Grammar beyond JSON: JSON schemas **done (0.3.5)**; tool calls through the template next (O6).
- [ ] A GGUF writer for repacked forks; the safetensors → GGUF converter for the merged SmolLM2 is **done (0.3.5,
  O5)**.
- [ ] Each architecture and format gets the full oracle set (the whole model, greedy, sampling, conversations) in the
  gate.

### 0.7.0 — mindXtrain in Rust, end to end
- [ ] The probe stage: PEFT adapters loaded from safetensors and merged, then recall probing with transformers'
  decoding rules (repetition penalty, no-repeat n-gram), against mindXtrain's own `probe_recall`.
- [ ] The classroom and boardroom verdicts, the dojo tie-break, the feedback ledger, the receipts (BLAKE3 manifest),
  each identical to mindXtrain's Python.
- [ ] **LoRA training on the CPU**: a backward pass and AdamW for the 135M–0.6B imprint recipe. Gradients are checked
  against PyTorch f32 within a stated bound, and the recall gate's verdict must agree.
- [x] `bankml create` (O5, **0.3.5**): a Modelfile subset as a derived model over a pinned base. Next, in mindX:
  `promote.py --to bankml` replaces `ollama create`.
- [ ] One generation end to end in Rust: author → imprint → probe → score → verdict → GGUF. Also use the exact
  pipeline to find out why the gate has refused every generation since 39.

### 0.8.0 — trust
- [ ] **Signed receipts** in `crypto/` (GPL-3.0-only, opt-in, a cargo feature): the operator's key through BANKON
  or Parsec custody, never an agent's. `bankml verify-receipt` for anyone.
- [ ] Reproducible builds: the release binary's sha256 published and rebuilt in CI; an SBOM (zero dependencies make
  it short).
- [ ] Fuzzing without crates (an in-crate harness) for the GGUF parser, the JSON parser and the HTTP reader; the
  findings fixed, each with a test.
- [ ] An independent review of `serve` (the loopback rules, request limits, receipts) before 1.0.

### 0.9.0 — release candidate
- [ ] The mindX provider (P4): mindX's inference discovery uses `bankml serve --native` with receipts.
- [ ] Savante defaults to native. The installer builds or downloads a verified bankML binary and fetches llama.cpp
  only for the oracle or on request.
- [ ] Release binaries for x86_64 and aarch64 Linux, sha256-pinned, with the verification steps in usage.md.
- [ ] The interfaces frozen and documented (CLI, HTTP, library) with a semver policy; deprecations stated.
- [ ] Soak tests: long conversations (4k–32k context with YaRN), memory ceilings, restarts with slot restore.
- [ ] Every document checked against the code and the records, as for 0.2.0.

### Handheld and distributed intelligence (github.com/minaiml): making smartphones actually smart

[minaiml](https://github.com/minaiml) is **our own project**: "min ai ml — language models in miniature", one of the
organizations of the estate mapped in [github.com/Professor-Codephreak/orgmap](https://github.com/Professor-Codephreak/orgmap)
([`map/minaiml.md`](https://github.com/Professor-Codephreak/orgmap/blob/main/map/minaiml.md), domain: machine
learning and local language models). It is the delivery layer, for models that run on hardware you already own, where
*sparsity, not size, decides what a small machine can run*. It holds 2 original works, its profile and
`brobot-private` (brobot, the Android model player, developed privately until release; the public `brobot` fork is
MIT), and 78 **research forks**, each studying an upstream project: phone apps and runtimes (pocketpal-ai, maid,
MNN, LiteRT-LM, mlc-llm), sparse delivery (flash-moe, Flash-iOS, flash-pi-dsv4, Anemll), and distributed inference
(distributed-llama, petals, exo). bankML's part is to be the **verified engine** inside that delivery: the same bits
as llama.cpp, a pinned model and a receipt, on a phone as on a laptop. Then many devices act as one.

**Handheld**
- [ ] bankML on aarch64 (Android and Linux phones, the Raspberry Pi): Rust's own aarch64 targets, still no crates;
  NEON kernels bit-exact (0.5.0); the oracle run on the device against llama.cpp's ARM build (MIT; minaiml's
  research fork of it).
- [ ] bankML as a library for apps: a small C ABI (verify, load, generate, receipt), so brobot plays pinned models
  through bankML. The core stays standalone and agnostic and imports no extension, per brobot's layering.
- [ ] Phone GPUs (Adreno, Mali, Apple through MoltenVK) through the same `gpu/` Vulkan backend, each card verified
  bit-exact on the device before it computes anything (`bankml gpu --verify`).
- [ ] Delivery rules on the handheld exactly as on the laptop: pin by sha256, OSI licences only, the guard before
  load, and memory planned for 4–12 GB phones (the importer's plan).
- [ ] **Sparse delivery:** mixture-of-experts Qwen3 (the A3B shape) in bankML's forward pass, with the shared weights
  resident and the experts read from flash on demand. The oracle is llama.cpp's MoE output. It is built from the
  models' published architectures (Apache-2.0) with llama.cpp (MIT) as the reference. minaiml studies the idea in
  its research forks of flash-moe, Flash-iOS, flash-pi-dsv4 and Anemll, but **their upstreams**
  (`danveloper/flash-moe`, `danveloper/flash-pi-dsv4`, `Anemll/Anemll`, `Anemll/Flash-iOS`) **declare no
  licence**. Open source or go away: bankML does not build on their code until they do.

**Distributed**
- [ ] Several devices as one engine: 0.5.0's multi-card row split, generalised to devices on a network. Each device
  computes a share of every matrix's rows, or a span of layers, with kernels verified on that device, and **the
  answer is bit-identical to one machine's**. The transport is authenticated with device keys: Parsec custody, and
  [pmVPN](https://pmvpn.pythai.net)'s keyring (one wallet signature, eight port-scoped SSH keys; v0.1.1,
  github.com/poormanvpn/pmVPN).
- [ ] Per-device receipts, signed (0.8.0): which device computed which share, so a distributed answer is as
  accountable as a local one.
- [ ] The distributed research forks, by their upstreams' licences:
  - **distributed-llama and petals** (both MIT) can be studied and built on, with attribution.
  - **exo:** the upstream `exo-explore/exo` is **Apache-2.0** today, but minaiml's fork holds an older snapshot that
    reports **GPL-3.0**. Take the licence of the revision studied; a GPL revision stays at a process boundary, never
    linked into the MIT / Apache core (LICENSING.md).
  - **MNN, mlc-llm, LiteRT-LM and cutile-rs** are Apache-2.0.
  - **executorch and ncnn** show `NOASSERTION`: read their licence files before touching either.
- [ ] The first deployment is mindX's own mesh: the VPS nodes, the laptops, and the phones running brobot.

### 1.0.0
- [ ] The six conditions at the top of this section are all met, each with a record in `testing/results/1.0.0.txt`.

