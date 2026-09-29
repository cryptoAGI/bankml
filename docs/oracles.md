# Oracles in bankml

## What an oracle is here

An **oracle** is an answer bankml did not compute and cannot influence, against which its own answer is compared.
bankml's rule since 0.0.1 is that a result counts only if an oracle has checked it: *the same bits first, then the
speed*. A faster kernel that is not bit-identical to its oracle is not a result; a construction (a hash, a Merkle
root, an ABI encoding) that does not reproduce a published value is not trusted.

The strongest oracle is the **compiled reference itself**: llama.cpp b11192's own shared libraries, called in-process
through their exported symbols on the same bytes bankml reads. The others are published values: test vectors from
standards (FIPS 180-4, RFC 6962), values published by the systems bankml interoperates with (Savante's doctrine root,
the THOT spec's vectors, Hugging Face's and Ollama's sha256 for a model file), and independent implementations
(pycryptodome, Foundry's `cast`, the Python guard bankml's Rust guard was ported from).

This file lists every oracle bankml uses, what it checks, how to run it, and what it last found. The results are in
[`testing/results/`](../testing/results/), one record per release, written by
[`testing/release_gate.sh`](../testing/release_gate.sh).

## 1. The ggml oracle: kernels bit-exact against the compiled llama.cpp

### What it checks

For each real model, three things, every one to the bit:

| # | quantity | bankml side | ggml side (llama.cpp b11192, shipped binaries) |
|---|---|---|---|
| 1 | every weight of every low-bit tensor, dequantized to f32 | `q1_0::dequantize_row`, `q2_0::dequantize_row` | `dequantize_row_q1_0` / `dequantize_row_q2_0` in `libggml-base.so` |
| 2 | activation rows quantized to `q8_0` (32 signed bytes and a half-precision scale per block) | `quantize_row_q8_0` | `quantize_row_q8_0` in `libggml-cpu-haswell.so` |
| 3 | dot products of a weight row with a `q8_0` row | `vec_dot_ref` (scalar model), `vec_dot` / `vec_dot_act` (AVX2), `vec_dot_act_scalar` | `ggml_vec_dot_q1_0_q8_0` (AVX2 path), `ggml_vec_dot_q1_0_q8_0_generic`, `ggml_vec_dot_q2_0_q8_0` |

Dequantized tensors are compared by the sha256 of their little-endian f32 output (a whole 8-billion-weight model,
tensor by tensor, without holding it in memory); quantized rows byte for byte; dot products by their f32 bit
patterns (`to_bits()`), not within a tolerance.

### Why "the compiled library" and not "the source"

The first port of ggml's generic C, done faithfully from the source, disagreed with the shipped library in the last
bit. Reading the disassembly showed why: the compiler had fused a multiply and an add into one FMA instruction, which
rounds once instead of twice. A source-level port cannot see that; the binary oracle catches it. bankml's kernels
reproduce the float order and the fused multiply-adds of the **shipped** `libggml-cpu-haswell.so` (the variant ggml's
backend scorer loads on an AVX2 CPU without AVX-512), and the oracle proves it on real tensors.

It also shows the oracle can tell float orders apart. For the ternary format ggml ships two builds of the same generic
C: the haswell variant (FMA) and the baseline x64 variant (SSE, no FMA). They disagree with each other on 19 of the
762 recorded ternary dot products. bankml carries a model of each (`vec_dot_ref` for haswell, `vec_dot_ref_nofma` for
x64) and matches each one in 762 of 762.

### How it runs

1. **Record** (once per model): [`testing/ggml_oracle.py`](../testing/ggml_oracle.py) loads the release's
   `libggml-base.so` and `libggml-cpu-haswell.so` (and, for Q2_0, `libggml-cpu-x64.so`) with `ctypes`, calls
   `ggml_cpu_init()`, memory-maps the GGUF and writes:
   - `dequant.tsv`: tensor name · element count · sha256 of ggml's f32 output, for every tensor of the type;
   - `vecdot.bin`: for sampled rows of every tensor (3 per tensor by default): the f32 activation `x`, ggml's `q8_0`
     of it, and ggml's dot products (AVX2 and generic).

   The release tarball is checked by sha256 before use (`llama-b11192-bin-ubuntu-x64.tar.gz`, `34cf6fa5…81ec7`).
2. **Compare** (every gate): the Rust tests `oracle_ggml_b11192_real_*` (in [`q1_0.rs`](../bankML/q1_0.rs) and
   [`q2_0.rs`](../bankML/q2_0.rs), `#[ignore]`d because they need the models) re-derive every recorded quantity from the same
   file through bankml's own memory map and assert equality.
3. **A/B against the live library** (every gate): `ab_vs_ggml` and `ab_vs_ggml_q2_0` `dlopen` the shipped haswell
   library directly (no crate, `dlopen`/`dlsym` declared by hand) and time ggml's kernel and bankml's on the same
   real weights in one process, after first checking they agree.

### What it found (0.1.7–0.2.0 gates; identical in each)

| model | tensors | weights | q8_0 rows | dot products |
|---|---|---|---|---|
| Bonsai-1.7B, `Q1_0` | 197 | 1,719,904,256 | 788 byte-exact | 788 bit-exact (AVX2 and `vec_dot_act`); generic port == ggml generic 788/788 |
| Bonsai-8B, `Q1_0` | 254 | 8,188,239,872 | 762 byte-exact | 762 bit-exact; generic 762/762 |
| Ternary-Bonsai-8B, `Q2_0_g64` | 254 | 8,188,239,872 | 762 byte-exact | 762 bit-exact vs haswell (reference, AVX2 and scalar paths); no-FMA model == x64 build 762/762 |

### Run it

```sh
# once: the oracle files (numpy + the b11192 release)
python3 testing/ggml_oracle.py .models/Bonsai-1.7B-Q1_0.gguf /path/to/llama-b11192 .models/oracle
python3 testing/ggml_oracle.py .models/Bonsai-8B-Q1_0.gguf /path/to/llama-b11192 .models/oracle-8b-q1
python3 testing/ggml_oracle.py .models/Ternary-Bonsai-8B-Q2_0_g64.gguf /path/to/llama-b11192 .models/oracle-ternary
# every time
cargo test --release -- --ignored oracle_ggml_b11192 --nocapture --test-threads=1
BANKML_GGML_LIB=/path/to/llama-b11192 cargo test --release -- --ignored ab_vs_ggml --nocapture --test-threads=1
```

## 1b. The tokenizer oracle: token-identical to llama.cpp (0.2.1)

P3, bankml's own forward pass, starts with the tokenizer. [`testing/tokenizer_oracle.py`](../testing/tokenizer_oracle.py)
asks a running llama-server (b11192, the Bonsai / Qwen3 vocabulary) to tokenize a corpus, with special tokens parsed
and not, and records every answer. The corpus is:
- every document in the repository and Savante's canon texts;
- the chat template's markers;
- hand-picked edge cases: contractions, CRLF, every kind of whitespace, digits in several scripts, combining marks,
  CJK, right-to-left scripts, emoji with joiners, mathematical alphanumerics;
- a seeded fuzz set of 2,000 strings drawn from 17 Unicode ranges.

`tokenizer::tests::oracle_tokenizer` re-derives every case with bankml's tokenizer (`tokenizer.rs`, no crates) and
requires the same ids in the same order. Last result: **4,258 of 4,258 cases token-identical**.

Two details the oracle settles:
- **Which special tokens split the text.** Qwen3's `<think>` markers are USER_DEFINED and split the text even when
  special tokens are not parsed; CONTROL tokens such as `<|im_start|>` do not.
- **The pre-tokenizer's letter class.** It is Unicode general category L, which is not `char::is_alphabetic`. It is
  generated into `unicode_letters.rs`.

## 1c. The chat-template oracle: byte-identical prompts (0.2.2)

A conversation becomes a prompt through the model's chat template, a Jinja program inside the GGUF. bankml does not
run Jinja. `chat.rs` writes the Bonsai / Qwen3 template's rules out, and `check_template` accepts only that template,
identified by the sha256 of its text. [`testing/template_oracle.py`](../testing/template_oracle.py) records
llama-server's own `/apply-template` for 317 conversations. They cover:
- a system prompt first, later, or absent;
- assistant turns with and without `<think>` blocks, before and after the last real user query;
- `reasoning_content`;
- runs of tool results;
- user messages that look like tool responses;
- special markers and Unicode inside content;
- 300 random conversations.

`oracle_chat_template` requires every prompt **byte-identical**: **317 of 317**. The oracle found one server behaviour
that the template alone would not predict: an empty `reasoning_content` is dropped before templating.

## 1d. The forward-pass oracle: llama.cpp's own operations, from the shipped ggml (0.2.3)

The release has no tool that prints a model's intermediate values, so the oracle builds them the way the kernel oracle
does. [`testing/forward_oracle.py`](../testing/forward_oracle.py) drives the **shipped** `libggml` through ctypes with
the same operations llama.cpp's Qwen3 graph uses and computes them with the release's own CPU backend. For 300 real
token ids (the chat markers, the table's edges, and tokens of the tokenizer's corpus) it records the sha256 of each
row's f32 bytes:
- `get_rows` on the Q1_0 token embedding: llama.cpp's `inp_embd`;
- `rms_norm` with the model's epsilon, then `mul` by `blk.0.attn_norm.weight`: llama.cpp's `attn_norm-0`.

bankml's `forward.rs` computes the same in the float order read from b11192's `ops.cpp`:
- the sum of squares accumulated in double, one float product at a time, in order;
- the mean rounded to float;
- `scale = 1 / sqrtf(mean + eps)`;
- each output `(x · scale) · w`, with no FMA.

**300 of 300 rows bit-exact for both.**

**Step four (0.2.4): Q, K, V, the head norms and RoPE.** The oracle also builds layer 0's attention inputs on a
28-token chat prompt:
- `mul_mat` of the Q1_0 `attn_q`, `attn_k` and `attn_v` weights with the normed rows;
- `reshape` to 128-wide heads, `rms_norm` and `mul` by `attn_q_norm` and `attn_k_norm`;
- `rope_ext` in NEOX mode with the YaRN parameters llama.cpp's context derives (`freq_scale` 0.25, `ext_factor` 1,
  `attn_factor` 1.0, whose bits the record carries, beta 32 and 1, `n_ctx_orig` 16,384, base 10⁶).

RoPE runs twice: at positions 0–27, and at positions 7 to 63,214, where theta is a long product.
`oracle_forward_qkv_rope` requires **140 of 140 rows bit-exact**.

This step showed why the oracle is the shipped binary and not the source. Written as `ops.cpp` reads, bankml's RoPE
matched only 30 of 84 rows, while the projections and norms before it were exact. The disassembly of
`libggml-cpu-haswell.so` shows GCC's FMA contraction of the rotation, the YaRN mix and the magnitude term. With those
three written as the same `mul_add`s, every row matches.

**Step five (0.2.5): attention.** The oracle stores K and V through `cpy` to f16, as llama.cpp's cache does. It
builds the causal f16 mask and runs `flash_attn_ext`, which is llama-server's default attention on the CPU, with F32
accumulation set as llama-graph sets it. The output then goes through `wo` (`mul_mat`) and the residual `add`.
`oracle_forward_attention` requires **84 of 84 rows bit-exact** (`kqv_out`, `attn_out`, `ffn_inp`, 28 tokens).

The reproduction follows ggml's reference path:
- the f16 dot's four 8-lane FMA accumulators and their reduction tree;
- an online softmax whose V accumulator is rounded to f16 at every step.

The oracle rejects a near miss: with the softmax sum written as an FMA, only 59 of 84 rows match. Two other kernels
are not yet covered, and each will get its own oracle:
- the tiled kernel, for 64 or more query rows;
- the split-KV kernel, for a decode whose padded KV length reaches 512 (llama.cpp pads to multiples of 256, so from
  257 cells in use).

**Step six (0.2.6): the feed-forward block; layer 0 complete.** The oracle continues from `ffn_inp`:
- `rms_norm` and `mul` by `ffn_norm`;
- `mul_mat` by `ffn_gate` and `ffn_up`;
- `swiglu_split`;
- `mul_mat` by `ffn_down`;
- `add`, giving `l_out`, the layer's output.

`oracle_forward_attention` requires 112 of 112 rows through `l_out`. SiLU in the shipped build uses ggml's own
vectorized `expf`, a polynomial in FMAs, not libm. bankml reproduces it lane by lane.

A second check, `oracle_forward_swiglu_sweep`, feeds 24,600 values across ±120 and the edges straight to
`ggml_swiglu_split` and requires every output's bits (**24,600 of 24,600**). The sweep exists because the layer check
alone was not enough: with libm's `expf`, `l_out` still matched 27 of 28 rows, since q8_0 quantization absorbs most
of the difference, while the sweep matched only 19,245 of 24,600.

**Step seven (0.2.7): the whole model, and llama-server itself.**
- [`testing/model_oracle.py`](../testing/model_oracle.py) computes the whole Qwen3 graph in the shipped ggml: every
  layer, `output_norm` and `mul_mat` by `output.weight`. It runs one layer per ggml context and carries the residual
  stream between contexts as f32 bytes, which is the same arithmetic as one graph in a fraction of the memory.
  `oracle_forward_model` requires **1,064 of 1,064 rows bit-exact**: 36 × 28 `l_out` rows, 28 `result_norm` rows,
  and 28 logit rows of 151,669 values each.
- [`testing/greedy_oracle.py`](../testing/greedy_oracle.py) steps outside the graph. It asks the running
  llama-server to render 6 chat prompts (`/apply-template`), tokenize them (`/tokenize`) and continue them greedily
  (`/completion`, top-k 1, prompt cache off). `oracle_greedy_llama_server` requires bankml's own forward pass to
  produce the **same tokens on every prompt (6 of 6, 164 tokens)**, and to end the turn where the server did.

Both stay inside the range bankml reproduces: prompts under 64 tokens, contexts up to 256 cells (the largest case
uses 80). llama.cpp pads the KV length to multiples of 256, so a decode beyond 256 cells takes the split-KV kernel.

**Step eight (0.2.8): the ternary model.** Both whole-model checks run again on Ternary-Bonsai-8B (Q2_0_g64), whose
every matrix, including the embedding and the output, is ternary:
- `oracle_forward_model_ternary` checks bankml against the shipped ggml graph: **1,064 of 1,064 rows**.
- `oracle_greedy_llama_server_ternary` checks against llama-server running the ternary model, with Savante's flags,
  on a spare port: **6 of 6 prompts, 140 tokens**.

The greedy comparison now covers exactly the tokens the server generated. Its list includes the end-of-turn token
when it produced one. Where it stopped otherwise, that is its stopping policy, not a token choice.

**Step nine (0.2.9): long prompts and ggml's tiled kernel.** llama.cpp computes a prompt in micro-batches of up to
512 tokens. A micro-batch of 64 rows or more takes `flash_attn_ext_tiled`, a different algorithm from the reference
path: f32 Q, a SIMD GEMM per 64-cell KV tile, a vectorized softmax summed in double, and an f32 accumulator.
- The forward oracle runs layer 0 on a 150-row micro-batch in the shipped ggml. `oracle_forward_attention_tiled`
  requires **150 of 150 rows bit-exact**. The reference kernel would match only 1, the first row, so the kernel
  choice is itself checked.
- `greedy_oracle.py --long` records llama-server's continuations for 6 prompts of 111–116 tokens.
  `oracle_greedy_llama_server_long` requires **the same tokens on all 6**.

**Step ten (0.2.10): long contexts and ggml's split-KV kernel.** A single-token decode over 512 or more padded cells
cuts the cells into one chunk per llama.cpp thread and reduces the partial softmaxes, so the thread count shapes the
bits.
- The forward oracle records one decode row at 7 context lengths (257–1,000 cells) and at 3 and 4 threads.
  `oracle_forward_attention_split` requires **14 of 14**.
- `greedy_oracle.py --deep` records 3 continuations of 200 tokens that run to about 300 cells.
  `oracle_greedy_llama_server_deep` requires **the same 600 tokens**.

The 4-thread cases did double duty. Besides showing the dependence on the thread count, they exposed the reduction's
FMA contraction: at 3 threads the first chunk always held the maximum, where the fused and plain forms agree.

Each later step of the forward pass (RoPE, attention, the feed-forward block,
the logits) is added to the same oracle before it counts.

## 2. Scalar models: every fast path against its own reference

Between the real-model oracle runs, the kernels are held to a scalar model of ggml, on synthetic inputs, in the
ordinary `cargo test`:

| check | where | cases |
|---|---|---|
| half-precision conversion, all 65,536 values, and against the CPU's F16C instruction | `q1_0.rs` | 65,536 |
| `q8_0` rounding (round-half-to-even, scale = 127 / max\|x\|) | `q1_0.rs` | edge cases |
| AVX2 `Q1_0` dot == scalar model of ggml | `q1_0.rs` | 3,500 |
| AVX2 and portable `Q2_0` dot == ggml model | `q2_0.rs` | 3,300 |
| matrix–vector, matrix–matrix and threaded paths == per-pair dots | `q1_0.rs`, `q2_0.rs`, `par.rs` | every shape tested |
| the 0.0.4 prefill tile == per-pair | `q1_0.rs` (`act_tile_bit_exact`) | every tile |
| the generic port == f64 arithmetic on dequantized data (the mathematical definition) | `q1_0.rs` | random blocks |

A speed-up is admitted only after these and §1 pass on the same build. Kernels that were faster but not exact, or
exact but not faster beyond noise, are kept with their numbers in [`testing/experiments/`](../testing/experiments/)
(0.0.5).

## 3. The guard: Rust against the Python it was ported from

bankml's header guard (`gguf.rs`: play, refuse or need more bytes; the three low-bit traps; hostile headers) was
ported from minaiml's Python `gguf_guard.py`, vendored in [`testing/gguf_guard.py`](../testing/gguf_guard.py) with its
suite. [`testing/guard_agree.py`](../testing/guard_agree.py) runs **both** on every synthetic case and on every real
model present and requires identical JSON. Last result: **28/28 agree**.

## 4. Cryptographic constructions against published values

| construction | oracle | where |
|---|---|---|
| SHA-256 (the model pin) | FIPS 180-4 test vectors; since 0.1.8 the SHA-NI path must also equal the portable rounds on every length 0–1,000 (and 4 KiB, 64 KiB, split updates), and a real 1.16 GB model must hash to coreutils `sha256sum`'s value and its published pin | `sha256.rs` (`fips_vectors`, `hardware_path_equals_portable_on_every_length`) |
| the model pin | the sha256 the publisher lists: the FORK.json of the PYTHAI fork, a Hugging Face repository's LFS sha256 at a fixed revision, or the Ollama registry's layer digest; every import is hashed as it streams and kept only if equal | `bankml.rs` (`pin`), `sAGI/models.py` |
| keccak256 (pure Python) | pycryptodome's keccak on every input length 0–400 bytes; and Savante's **published doctrine root** `0x92fe83eb…ae137d0`, reproduced from her persona | `sAGI/agents.py`, `testing/test_ui.py` |
| THOT manifests (`sagi.thot_manifest/1`) | the spec's own test vectors (`THOT_MANIFEST.md` §5: savante@1fcca89, jaimla@8b57ccf, luvai@0c1eef7): bundle root, Merkle root, identity CID | `sAGI/thot.py`, `testing/test_ui.py` |
| CIDv1 (raw, sha2-256, base32) | the CID of `"abc"` that mindX's `rage.py` and Savante's ledger construction give | `testing/test_ui.py` |
| `.history` Merkle tree (RFC 6962) | the Certificate Transparency reference roots for 1, 3 and 8 leaves | `testing/test_ui.py` |
| iNFT ABI encoding | the compiled `iNFT_7857` contract itself, deployed from its artifact on a throwaway anvil devnet: it must accept the calldata bankml encodes (simulate and mint), refuse what it should (missing role, a repeated content root), and read back exactly the values encoded | `sAGI/chain.py`, `testing/test_chain.py` |
| Savante's canon | her own offline verifier `bind/savante_verify.py`: 12 of 12 commitments, doctrine root | the Verifier tab |

## 5. The gateway: receipts against the text

`bankml serve`'s receipts are checked end to end against a mock llama-server whose answers are known
([`testing/cli.rs`](../testing/cli.rs)): the `response_sha256` equals the sha256 of the text the mock sent (streamed and
not), `request_sha256` equals the sha256 of the request body, and a model file changed after verification produces a
503 and no receipt. In use, the UI recomputes every answer's sha256 and marks ✓ or `≠ received!`.

## 6. Oracles planned

- **P3, bankml's own forward pass.** The criterion is already fixed: at temperature 0, on the same prompts, an answer
  must be **token-identical** to llama.cpp b11192's. Every kernel it will use already passes §1, and its tokenizer
  passes §1b (0.2.1).
- **Speculative decoding (measured in 0.1.8).** A draft model (Bonsai-1.7B) and n-gram speculation both left the
  output **token-identical** at temperature 0 on every run; neither was faster beyond this laptop's noise, so neither
  is the default (n-gram is an opt-in). The same criterion applies to any future speed-up that changes how tokens are
  computed.
- **The upstream `Q2_0` kernel (0.2.0).** `upstream/test_q2_0_avx2.c` holds the C drop-in to the shipped
  `ggml_vec_dot_q2_0_q8_0`, bit for bit: 200,000 of 200,000 random cases.


## The rule, restated

- An oracle is **external**: a compiled library, a standard's vectors, a published value, an independent
  implementation. A test that compares bankml with bankml is a regression test, not an oracle.
- A comparison is **exact** where the quantity is exact: bits, bytes, hashes. Tolerances are used for nothing that an
  oracle can check exactly.
- A **speed claim** is made only on code that passed every oracle in the same gate run, and only if the gain is beyond
  the run-to-run noise measured on the same machine.
- Every gate run is **kept** (`testing/results/<version>.txt`), including the rejected experiments' numbers.
