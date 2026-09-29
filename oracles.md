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
[`testing/results/`](testing/results/), one record per release, written by
[`testing/release_gate.sh`](testing/release_gate.sh).

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

1. **Record** (once per model): [`testing/ggml_oracle.py`](testing/ggml_oracle.py) loads the release's
   `libggml-base.so` and `libggml-cpu-haswell.so` (and, for Q2_0, `libggml-cpu-x64.so`) with `ctypes`, calls
   `ggml_cpu_init()`, memory-maps the GGUF and writes:
   - `dequant.tsv`: tensor name · element count · sha256 of ggml's f32 output, for every tensor of the type;
   - `vecdot.bin`: for sampled rows of every tensor (3 per tensor by default): the f32 activation `x`, ggml's `q8_0`
     of it, and ggml's dot products (AVX2 and generic).

   The release tarball is checked by sha256 before use (`llama-b11192-bin-ubuntu-x64.tar.gz`, `34cf6fa5…81ec7`).
2. **Compare** (every gate): the Rust tests `oracle_ggml_b11192_real_*` (in [`q1_0.rs`](q1_0.rs) and
   [`q2_0.rs`](q2_0.rs), `#[ignore]`d because they need the models) re-derive every recorded quantity from the same
   file through bankml's own memory map and assert equality.
3. **A/B against the live library** (every gate): `ab_vs_ggml` and `ab_vs_ggml_q2_0` `dlopen` the shipped haswell
   library directly (no crate, `dlopen`/`dlsym` declared by hand) and time ggml's kernel and bankml's on the same
   real weights in one process, after first checking they agree.

### What it found (0.1.7 gate)

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
exact but not faster beyond noise, are kept with their numbers in [`testing/experiments/`](testing/experiments/)
(0.0.5).

## 3. The guard: Rust against the Python it was ported from

bankml's header guard (`gguf.rs`: play, refuse or need more bytes; the three low-bit traps; hostile headers) was
ported from minaiml's Python `gguf_guard.py`, vendored in [`testing/gguf_guard.py`](testing/gguf_guard.py) with its
suite. [`testing/guard_agree.py`](testing/guard_agree.py) runs **both** on every synthetic case and on every real
model present and requires identical JSON. Last result: **28/28 agree**.

## 4. Cryptographic constructions against published values

| construction | oracle | where |
|---|---|---|
| SHA-256 (the model pin) | FIPS 180-4 test vectors | `sha256.rs` (`fips_vectors`) |
| the model pin | the sha256 the publisher lists: the FORK.json of the PYTHAI fork, a Hugging Face repository's LFS sha256 at a fixed revision, or the Ollama registry's layer digest; every import is hashed as it streams and kept only if equal | `bankml.rs` (`pin`), `ui/models.py` |
| keccak256 (pure Python) | pycryptodome's keccak on every input length 0–400 bytes; and Savante's **published doctrine root** `0x92fe83eb…ae137d0`, reproduced from her persona | `ui/agents.py`, `testing/test_ui.py` |
| THOT manifests (`sagi.thot_manifest/1`) | the spec's own test vectors (`THOT_MANIFEST.md` §5: savante@1fcca89, jaimla@8b57ccf, luvai@0c1eef7): bundle root, Merkle root, identity CID | `ui/thot.py`, `testing/test_ui.py` |
| CIDv1 (raw, sha2-256, base32) | the CID of `"abc"` that mindX's `rage.py` and Savante's ledger construction give | `testing/test_ui.py` |
| `.history` Merkle tree (RFC 6962) | the Certificate Transparency reference roots for 1, 3 and 8 leaves | `testing/test_ui.py` |
| iNFT ABI encoding | the compiled `iNFT_7857` contract itself, deployed from its artifact on a throwaway anvil devnet: it must accept the calldata bankml encodes (simulate and mint), refuse what it should (missing role, a repeated content root), and read back exactly the values encoded | `ui/chain.py`, `testing/test_chain.py` |
| Savante's canon | her own offline verifier `bind/savante_verify.py`: 12 of 12 commitments, doctrine root | the Verifier tab |

## 5. The gateway: receipts against the text

`bankml serve`'s receipts are checked end to end against a mock llama-server whose answers are known
([`testing/cli.rs`](testing/cli.rs)): the `response_sha256` equals the sha256 of the text the mock sent (streamed and
not), `request_sha256` equals the sha256 of the request body, and a model file changed after verification produces a
503 and no receipt. In use, the UI recomputes every answer's sha256 and marks ✓ or `≠ received!`.

## 6. Oracles planned

- **P3, bankml's own forward pass.** The criterion is already fixed: at temperature 0, on the same prompts, an answer
  must be **token-identical** to llama.cpp b11192's. Every kernel it will use already passes §1.
- **Speculative decoding (0.1.8).** A small draft model (Bonsai-1.7B) proposing tokens for Bonsai-8B must leave the
  output **token-identical** at temperature 0; a speed-up counts only then.
- **SHA-NI hashing (0.1.8).** The hardware SHA-256 path must equal the portable one on every length and on real model
  files, and equal `sha256sum`.

## The rule, restated

- An oracle is **external**: a compiled library, a standard's vectors, a published value, an independent
  implementation. A test that compares bankml with bankml is a regression test, not an oracle.
- A comparison is **exact** where the quantity is exact: bits, bytes, hashes. Tolerances are used for nothing that an
  oracle can check exactly.
- A **speed claim** is made only on code that passed every oracle in the same gate run, and only if the gain is beyond
  the run-to-run noise measured on the same machine.
- Every gate run is **kept** (`testing/results/<version>.txt`), including the rejected experiments' numbers.
