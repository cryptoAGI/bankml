# bankml: verified low-bit inference on commodity CPUs — a technical report

*Professor Codephreak and Gregory L. Magnusson · cryptoAGI · bankml v0.1.0 · 2026. Companion to [PERFORMANCE.md](PERFORMANCE.md), which holds every measurement cited
here with the command that reproduces it.*

## Abstract

Language models quantized to one bit or to ternary values per weight shrink an 8-billion-parameter network to
1.2–2.3 GB, small enough for a laptop or a two-core server. Whether such models are *useful* there depends less on the
arithmetic than on two properties the literature treats separately: the speed of the matrix kernels that consume the
packed weights, and the confidence that a fast kernel computes exactly what the reference computes. This report argues
that the two must be established together, and that **bit-exactness against the reference implementation's own
compiled code is the correctness criterion a low-bit kernel should meet before its speed is reported at all**. We
describe bankml, a zero-dependency Rust runtime for the ggml `Q1_0` (1-bit) and `Q2_0_g64` (ternary) formats; its
verification discipline (a header guard, a sha256 pin, and an oracle that calls the reference library's exported
symbols in-process); and its measured results against llama.cpp b11192. For `Q1_0` the kernel is bit-exact on all
1.72 billion weights of a real model and matches or exceeds the reference speed (1.01× decode, 1.10× prefill, single
thread; 1.2–1.3× in prefill over prepared activations since 0.0.4, and bit-exact on all 8.19 billion weights of the 8B
1-bit model). For `Q2_0_g64` the reference ships no vectorised x86 kernel at all; bankml's kernel is bit-exact on all
8.19 billion weights of Ternary-Bonsai-8B and 9.5–9.8× faster per projection (12.5× in prefill). Since these products
are about 98 % of the reference's time per token, the gap between the better ternary model and the faster 1-bit one is
a kernel gap, now closed at the kernel level: with a zero-dependency thread pool, one ternary token's products take
0.23–0.25 s on three threads of a laptop, less than the reference needs for the *1-bit* model's (0.34 s). A measured
memory floor (15–17 GB/s) shows both kernels are compute-bound, and five further bit-exact variants measured no reliable
gain, which places them at the instruction-throughput limit of the test core (§IV.5). Since 0.0.6 the runtime answers
through the reference behind its gates (phase P0). Every answer carries a receipt with the sha256 of its text, and the
conversation's history is committed by a Merkle root and a CID, so it can be proven without being disclosed (§III.6).
Custom agents derived from Savante's template are bundled as THOT manifests and minted as iNFTs, the mint prepared and
simulated by the runtime and signed by the owner, and loaded back from a token with their lineage verified (§III.7).
Every cryptographic construction is checked against a published value.

## Thesis — Professor Codephreak and Gregory L. Magnusson

*The design intent this report tests is that of bankml's authors, Professor Codephreak and Gregory L. Magnusson,
stated in their own words over the course of the project (Codephreak and Magnusson 2026). It is recorded here as
their thesis; the sections that follow measure the work against it.*

**Port what works, and only what works.** "We prototyped in python and we switch to rust from working architecture"
(2026-07-04). bankml was not begun from a design document. It was begun, on the instruction to "create llama.cpp rust
version todo as bankml.rs" (2026-09-25), after a working architecture existed and had been measured: a 1-bit Bonsai-8B
served by llama.cpp b11192 on a two-core server. The Rust runtime inherits a proven system's behaviour as its
specification, which is why exactness against that system's compiled code (§III.4) is its first criterion.

**Three design goals: "optimization, succinct, and verified response"** (2026-09-25). Optimization is measured,
never quoted; succinctness is a budget (one crate, no runtime dependencies, one static binary); and a verified
response is a property of every answer — a model that cannot be verified does not answer (§III.1).

**The standing machine is the CPU you already have.** One server has been shown to be enough to power one
autonomous AI system, at a running cost of about one dollar a day, with inference drawn from the local CPU and free
tiers. A runtime that makes a commodity CPU sufficient is therefore not an optimization of the system; it is the
condition of its economics. Accelerators are rented for events, not owned (§I.1).

**Consume the inference, and prefer the better answer.** "If there is inference mindX needs to consume it, however
long it takes" (2026-09-26); and, for the review office that runs on these models, "a better answer is more important
than speed where speed under one minute is fine for now" (2026-09-26). This ordering — quality first, bounded by a
latency the use can bear — is why the ternary model, the better and the slower of the two, is the one this runtime
must make fast (§I.2).

**Know the limit, and code around it.** "They should know limits and be able to code around them" (2026-09-26). For a
runtime this reads two ways, both implemented: it knows the limits of what it may load — the guard refuses, with its
reason, a file that would load and answer in nonsense, rather than serving it — and where the reference engine is
slow, the answer is not to accept the limit but to write the kernel that removes it.

**Improve by increments, with the oracle in the loop.** "Audit and improve with three more incremental release pushes
focusing on optimization and performance including oracle feedback" (2026-09-28). Each release ran a gate whose record
is published (`testing/results/`). A speed counts only when every oracle has passed on the same code, and a variant that
is bit-exact but not reliably faster is recorded and not shipped (`testing/experiments/`). The negative results are
part of the evidence: they are how the limit of §IV.5 is known rather than assumed.

**Use the machine at hand.** "Use this laptop hardware" (2026-09-28). Every measurement in this report was taken on
a two-core laptop, and the runtime serves Savante there. Where the machine disturbs a measurement (swap full, other
load), the record says so and the measurement is re-run rather than smoothed.

**Prove the data; keep the data.** "Localstorage and hash and CID can work as data reference for proof of data while
keeping local data private and still verifiable" (2026-09-28). The transcript, the operator's notes and a viewer's
layout stay where they were made: on disk, or in the browser's local storage. What travels is a commitment: a sha256,
a CIDv1, a Merkle root. Whoever holds the root can check any single exchange they are shown against it, without holding
the rest (§III.6). The same construction binds Savante's canon to her iNFT ledger.

### Contributions

What this work contributes, stated so each can be checked against the code (file names are in this repository)
and the measurements (PERFORMANCE.md).

1. **A verified-response runtime architecture.** A model is admitted only through three gates: a header guard, a
   provenance pin, and a bit-exactness oracle. Since 0.0.6 every answer carries a receipt: model hash, guard verdict,
   token counts, timings, and the sha256 of the answer text (`serve.rs`). The contribution is the ordering: correctness
   is established before speed is measured, and a model that cannot be verified does not answer (§III.1).
2. **A header-only GGUF guard for the three low-bit traps.** From the file header alone, without reading tensor data,
   the guard returns *play*, *refuse* or *need more* against a named engine. It refuses (1) fork-only types
   (`PQ2_0` = 142, `PTQ1_0` = 143), which mainline rejects safely; (2) legacy `Q2_0` whose group-128 bytes are filed
   under type 42, which mainline reads as group-64 — detectable as a size mismatch; and (3) Bonsai 2 `Q2_0` in a rotated
   basis, which mainline loads and answers in fluent nonsense — the dangerous case, visible only by name. The Python
   original (vendored as `testing/gguf_guard.py`) and the Rust port (`gguf.rs`) agree JSON-for-JSON on every case (20 of 20 with the
   real file; 18 of 18 synthetic in the published tree).
3. **Provenance forks and the pin.** The models are forked with provenance to `PYTHAI/Bonsai-8B-gguf-fork` and
   `PYTHAI/Ternary-Bonsai-8B-gguf-fork`, each with a `FORK.json` recording the upstream revision and every file's
   sha256; `bankml pin` refuses a file whose hash differs and names both hashes (`sha256.rs`, FIPS 180-4 vectors).
4. **Exactness against the compiled reference, not its source.** The oracle loads the sha256-checked llama.cpp b11192
   release and calls its exported symbols in-process on the same bytes (`testing/ggml_oracle.py`). Reading the shipped
   binary exposed that its compiler contracts multiply-add pairs into single FMA instructions; a source-faithful port
   disagrees in the last bit. bankml models both reference builds — the haswell build and the baseline x64 build,
   which themselves disagree on 19 of 762 ternary cases — so the oracle demonstrably distinguishes float orders.
5. **A bit-exact 1-bit (`Q1_0`) kernel at reference speed.** Every weight of two real models (1.72 billion in the 1.7B,
   8.19 billion in the 8B) and every recorded dot product match. Decode is level with the reference (0.93–1.13× across
   releases and thread counts). Prefill is 1.2–1.3× faster through a 1×4 selection tile over prepared activations
   (`q1_0.rs`, 0.0.4).
6. **The finding that explains the ternary gap.** llama.cpp b11192 ships no vectorised x86 kernel for `Q2_0`: the
   generic C is renamed to the x86 symbol in `arch-fallback.h`, there is no repack or sgemm case, and the shipped
   library runs scalar code with 64 integer multiplies per block. The better ternary model is slow for want of a
   kernel, not because ternary weights are expensive (§III.5).
7. **A bit-exact ternary (`Q2_0_g64`) kernel, 9.5–9.8× the reference.** Two blocks per 256-bit register, the
   activation re-laid-out once per token so the inner loop has no shuffles, and ggml's float chain kept serial for
   exactness: all 8.19 billion weights of Ternary-Bonsai-8B and 762 of 762 dot products match, at 9.5–9.8× the
   reference per projection and 12.5× in prefill (`q2_0.rs`).
8. **A whole-model decode budget and its floor.** Timing one token's worth of every matrix product at the same clock
   as the reference's own end-to-end benchmark shows those products are about 98 % of its time per token. On a
   zero-dependency thread pool (`par.rs`) bankml's ternary products take 0.23–0.25 s at three threads, bounding a
   complete forward pass at about 4 tokens per second against the reference's 0.4 (§IV.3). A measured read-bandwidth
   floor (§IV.5) shows the remaining time is compute, not memory.
9. **Succinctness as a budget.** One crate, zero dependencies — SHA-256, the GGUF parse, half-precision conversion,
   a read-only memory map and all kernels are in-crate — so the whole runtime is auditable end to end.
10. **Measurement that travels with the model.** End-to-end baselines on three machines (a laptop, a one-core server,
    a sixteen-core container), stamped with an independent clock (chronos) and sequenced per request (kairos), and a
    *carrier test* that grades form and substance separately. It is that test which showed the ternary model gives
    the better answers — the reason this runtime targets it.
11. **The runtime's first consumers.** The sAGI engine (a code-enforced verdict contract that any OpenAI-compatible
    backend can carry) and the improve.skill loop (every measured answer an iteration toward a published skill
    version) are built to run on this runtime; they define what "verified response" must mean in production.
12. **Savante on the runtime, with proof of data** (0.0.6–0.0.8). A gateway (`bankml serve`) answers only for a
    verified file, and only through an upstream that is serving that file. A Savante UI reads her canon without
    writing to it and checks it against her iNFT ledger before she speaks. It records each exchange with its
    timings and receipt, searches the history with RAGE, keeps the operator's notes (`.memory`), and publishes only
    commitments, with an inclusion proof per exchange (§III.6).
13. **The testing kept with the code.** A release gate, its record for every version, the end-to-end tests, the UI's
    data-layer tests, and the measured-and-rejected kernels (`testing/`).
14. **Agents from a template, bound for an iNFT, checked against published values** (0.0.9–0.1.0). Custom agents
    are derived from Savante's template without writing to her canon. Each carries a ledger whose keccak256 doctrine
    root is built exactly as her binder builds hers (reproducing her published root). A THOT manifest to
    `sagi.thot_manifest/1` reproduces the spec's three test vectors. A PostgreSQL connector loads an agent only if
    every byte matches its manifest. The iNFT path mints from the manifest's contentRoot, loads a token back, and
    walks the THOT lineage to the minted generation, tested end to end on a local EVM (§III.7).

## I. The problem

### I.1 Low-bit models are memory-bound on a CPU

In autoregressive decoding each generated token multiplies every weight matrix by one activation vector. The
operation is a matrix–vector product whose arithmetic intensity — operations per byte moved — is fixed by the weight
format and is low. By the roofline argument (Williams, Waterman and Patterson 2009), a kernel of low arithmetic
intensity is bound by memory bandwidth, not by peak arithmetic, until its working set fits a cache. Quantization to one
or 1.58 bits per weight raises the intensity by moving fewer bytes per multiply-accumulate; it does not change the
bound's nature. A CPU runtime for such models therefore wins by moving the packed weights through the core with as
little overhead per byte as possible, and loses whenever the unpacking or the bookkeeping per block costs more than
the bytes saved.

### I.2 The reference implementation leaves a measured gap

llama.cpp and its tensor library ggml are the de-facto reference for CPU inference of quantized language models
(Gerganov et al., software). Under its release b11192 we measured the two Bonsai-8B variants — the same network
quantized two ways — on identical prompts and hardware (PERFORMANCE.md). The ternary model generates **about five to
seven times slower** than the 1-bit model (≈0.4 vs ≈2 tokens/s on a four-thread laptop; ≈3.8 vs ≈26 tokens/s on a
sixteen-core container), although its file is only twice as large. A factor of two would be explained by bytes; the
remainder is not. The ternary model, meanwhile, is the one that gives the better answers in our application: in a
structured review task its verdict agreed with its own reasoning where the 1-bit model's did not (PERFORMANCE.md,
"Ternary vs 1-bit"). The practical problem is therefore concrete: the more useful model is the one the reference
runtime serves poorly.

### I.3 Speed claims without exactness are not comparable

A faster kernel that rounds differently is not the same computation. In a quantized network small differences in
accumulation order or activation rounding change logits, and a changed logit changes a sampled token, after which two
runs diverge completely. Benchmarks that compare kernels without first establishing equality of results compare
different functions. We take the stricter position: a kernel is admitted for comparison only when its outputs are
*bit-identical* to the reference library's on the same inputs.

## II. Literature

### II.1 Binary and ternary weights

The idea of constraining weights to {−1, +1} during training, while keeping real-valued weights for the gradient
update, was established by BinaryConnect (Courbariaux, Bengio and David 2015) and extended to binarized activations in
Binarized Neural Networks (Hubara et al. 2016). XNOR-Net (Rastegari et al. 2016) introduced a real-valued scaling
factor per filter, recovering much of the accuracy lost to pure signs — the ancestor of the per-group scale every
modern low-bit format carries. Ternary Weight Networks (Li, Zhang and Liu 2016) and Trained Ternary Quantization
(Zhu et al. 2017) added the zero, arguing that {−1, 0, +1} is a far better approximation of a Gaussian weight
distribution than two values at a small additional cost in bits.

The strongest version of the binary school's claim is not that one bit is enough in general, but that with a learned
scale and training in the loop the network adapts to the constraint. Its strain appears at scale: post-training
binarization of large networks loses too much, which is why the language-model work that followed trains *through*
the constraint.

### II.2 Low-bit language models

For transformers the post-training route matured first: LLM.int8() (Dettmers et al. 2022) isolated outlier features
to make 8-bit inference lossless at scale, and GPTQ (Frantar et al. 2023) reached 3–4 bits per weight with
second-order error compensation. Below that, training-aware methods dominate. BitNet (Wang et al. 2023) trained
transformers with binary weights from scratch, and its successor (Ma et al. 2024) argued that ternary weights —
"1.58 bits" — match full-precision models of the same size at scale while replacing multiplications with additions.
The Bonsai models studied here belong to this lineage: dense Qwen3-8B networks (Qwen Team 2025) distributed as ggml
`Q1_0` (1.125 bits per weight, 1.16 GB) and `Q2_0_g64` (2.25 bits per weight on disk, 2.31 GB) files.

### II.3 Runtimes

Runtime engineering for these models has taken two directions. Server engines such as vLLM (Kwon et al. 2023)
maximize throughput across many concurrent requests on accelerators, chiefly by managing the key–value cache in pages;
their gains come from batching and do not transfer to one user on a CPU. The ggml/llama.cpp direction targets exactly
that case: a single process, CPU kernels specialized per quantization format and per instruction set, weights
memory-mapped from the file. bankml is a contribution to the second direction, with one change of emphasis: exactness
against the reference is proven per kernel before speed is claimed, and every answer is meant to carry the evidence
of what produced it.

### II.4 The architecture being served

Qwen3-8B is a decoder-only transformer using root-mean-square layer normalization (Zhang and Sennrich 2019), rotary
position embeddings (Su et al. 2024), grouped-query attention (Ainslie et al. 2023) and a SwiGLU feed-forward block
(Shazeer 2020). Of its per-token work, the projection matrices dominate the bytes moved; they are the tensors the
low-bit formats compress, and the subject of this report.

## III. The contribution

### III.1 Verified response: three gates before an answer

bankml's design goal is stated as a property of every response rather than of the program: *a model that cannot be
verified does not answer.* Three gates enforce it.

1. **The guard.** Before any tensor data is read, the GGUF header is parsed and checked against three traps observed
   in practice — among them files that load in mainline llama.cpp and then answer in fluent nonsense because a tensor
   type is misidentified. The guard returns *play*, *refuse* (with the reason) or *need more* (bytes); it never
   falls back silently. Its Rust implementation agrees with the original Python guard on 20 of 20 cases, JSON-identical.
2. **The pin.** The file's sha256 (FIPS 180-4) must equal the value recorded in the provenance manifest of the model's
   fork. A mislabelled file is refused with both hashes named.
3. **The oracle.** A kernel is admitted only when it is bit-exact against the reference library on real tensors
   (§III.4).

Since 0.0.6 a fourth element closes the loop at the answer. `bankml serve` starts only after the guard and the pin
pass. It binds the upstream engine to the verified path (it launches it, or checks that the running engine's `/props`
names the same file). It attaches a receipt to each answer: model hash, guard verdict, token counts, time to first
token, wall time, and the sha256 of the answer text, which the client re-computes. The arithmetic in P0 is still the
reference's; P3 replaces it with bankml's kernels behind the same receipt.

### III.2 The Q1_0 format and its arithmetic

ggml's `Q1_0` stores weights in blocks of 128: a 16-bit IEEE 754 half-precision scale *d* followed by 16 bytes of
signs, least-significant bit first. The storage cost is (16 + 2) × 8 / 128 = **1.125 bits per weight**. A block's
weights are *w*ᵢ = *d* · *s*ᵢ with *s*ᵢ ∈ {−1, +1}. The dot product of a block with an activation segment *x* is

  *d* · (Σ_{sᵢ=+1} *x*ᵢ − Σ_{sᵢ=−1} *x*ᵢ),

a signed sum requiring no multiplication by weights at all. ggml does not compute this on real activations: it first
quantizes the activation row to `q8_0` (blocks of 32 signed bytes with their own half-precision scale, computed as
127 / max|*x*| with round-half-to-even), and the kernel `ggml_vec_dot_q1_0_q8_0` accumulates integer products of
±1 and signed bytes, applying both scales at the end. To be bit-exact, bankml reproduces both steps: the same
activation quantizer, the same integer accumulation, and the same order of floating-point operations on the scales.
The last point is not cosmetic. Reading the disassembly of the shipped generic kernel showed that the compiler had
fused a multiply and an add into a single FMA instruction, which rounds once instead of twice; a "faithful" C port
that rounded twice disagreed with the shipped library in the last bit. The port was made to match the binary, not
the source.

### III.3 Mapping the arithmetic to AVX2

On x86-64 with AVX2 the 32 sign bits that meet one `q8_0` activation block are expanded to 32 bytes by a byte
shuffle (`vpshufb`), a per-lane bit mask and a byte compare, then combined with the activation bytes and accumulated
with `vpmaddubsw` (unsigned × signed bytes, adjacent pairs summed to 16-bit lanes) and `vpmaddwd` (widened to 32-bit
lanes). The integer sum of each activation block is converted to single precision and folded into eight float lanes
with one fused multiply-add per `q8_0` block, then per `Q1_0` block, and reduced horizontally at the end — the float
order of ggml's own AVX2 kernel, which a scalar model (`vec_dot_ref`) reproduces and the AVX2 path must equal bit for
bit (`q1_0.rs`, module header). For decoding, the activation block scales are converted
from half to single precision once per token rather than once per weight block (`vec_dot_act`), which is where the
decode kernel recovers parity with the reference. For prefill — many activation columns against the same weights —
a 1×4 tile loads each weight block once and applies it to four columns, amortizing the sign expansion; this is the
source of the 1.10–1.16× prefill gain.

### III.4 The oracle: exactness against the compiled reference

The oracle loads the reference library's own shared objects (`libggml-base.so`, `libggml-cpu-haswell.so`) from the
sha256-checked b11192 release and calls their exported functions through a foreign-function interface on the same
bytes bankml reads. It compares: every weight dequantized (all 197 `Q1_0` tensors of a real 1.7-billion-parameter
model, 1,719,904,256 weights); activation rows quantized to `q8_0` (788 rows, byte for byte); and dot products (788,
bit for bit, against both the AVX2 and the generic paths). Comparing against the *binary* rather than a re-implementation
from the source is the methodological core of this work: it catches exactly the class of discrepancy (§III.2, fused
rounding) that a source-level port misses.

### III.5 The Q2_0_g64 (ternary) kernel

ggml b11192 defines `Q2_0` (read from the tag's source, not inferred) as blocks of 64 weights: a half-precision scale
*d* first, then 16 bytes of 2-bit codes, lowest bits first. A code *c* ∈ {0, 1, 2, 3} decodes to (*c* − 1) · *d*, so the
format admits {−1, 0, +1, +2} — not strictly ternary; in the real Ternary-Bonsai-8B file the code for +2 never occurs
(the observed split is 31 % / 38 % / 31 % / 0 %). The storage cost is (16 + 2) × 8 / 64 = 2.25 bits per weight. As with
`Q1_0`, activations are first quantized to `q8_0`, so one `Q2_0` block meets two activation blocks, and the dot product
is an integer sum of small codes times signed bytes, scaled at the end.

The decisive finding came from reading the build rather than the kernel: **llama.cpp b11192 ships no x86 kernel for
`Q2_0`.** In `ggml/src/ggml-cpu/arch-fallback.h` the generic C function is renamed to the x86 symbol, and neither the
weight-repacking path nor the sgemm path has a case for the type; the disassembly of the shipped haswell library is
scalar code with 64 integer multiplies per block, used for decoding and prefill alike. The five-to-seven-fold gap of
§I.2 is therefore not a property of ternary weights but of a missing kernel.

bankml's AVX2 kernel places two weight blocks in one 256-bit register and extracts the codes with a shift and a mask.
Because integer sums are exact in any order, the activation — not the weights — is rearranged once per token into the
layout the weight codes need, removing every shuffle from the inner loop; the float sequence that combines the per-block
integer sums keeps ggml's order exactly, including the fused multiply-adds its compiler emits, which is what keeps the
result bit-identical. A read-only memory map (`gguf::Mmap`, no dependency) lets the oracle and the benchmarks walk the
2.3 GB file without copying it. The oracle (§III.4) was extended rather than forked: it selects `Q1_0` or `Q2_0` by the
file's tensor types.

### III.6 Proof of data without the data

The runtime's records are private by default and verifiable by construction. Every exchange is one line in a local
JSONL file (`.history`); the operator's notes are another (`.memory`). Each line is a leaf, hashed as
sha256(0x00 ‖ line). The leaves form an RFC 6962 Merkle tree: interior nodes are sha256(0x01 ‖ left ‖ right), and
the tree is split at the largest power of two, so an odd node is promoted and never paired with itself. Leaves and
nodes cannot be confused, and trees of different sizes do not share a root; the tree before 0.1.7 had neither
property. The whole file also has a sha256 and a CIDv1: raw codec, sha2-256,
base32, the construction Savante's iNFT ledger uses for her canon files. Only these commitments leave the machine. The
LAN view shows the root, the count and the CID, never a line. For any single exchange, the UI emits an inclusion proof:
the leaf, its index, the tree size, the sibling path and the root, checked as RFC 9162 §2.1.3.2 prescribes. A holder
of the root and the count can check that exchange (and detect a changed byte in it)
without the rest of the history. The browser's local storage plays the same part on the client side: the viewer's
layout stays in their own browser. The tests (`testing/test_ui.py`) check that every proof verifies, that a changed byte
or a proof for another record fails, that the CID equals the house implementation's, and that the view's state carries
the commitment and none of the content. These commitments are what a THOT dataset bundle and an iNFT's storage
reference will point to (§VI).

### III.7 From a template to a token

A custom agent is derived, never copied into place. Savante's canon is read as a template. The derived persona gets
a new identity, and its token bindings start empty. The ledger commits each file by sha256 and CIDv1, and commits the
fifteen clauses its owner may not edit by a keccak256 doctrine root. The construction is the one Savante's binder
uses: pointer, 0x1f, canonical JSON, 0x1e, in a fixed order. The keccak256 is written here in the standard library.
Two published values anchor it: it reproduces Savante's doctrine root (`0x92fe83eb…`) from her persona, and it
agrees with an independent implementation on every input length from 0 to 400 bytes.

The agent's dataset is its THOT manifest (`sagi.thot_manifest/1`). The persona and the prompt are core facets. The
card, the history and the notes are declared custom facets, the last two committed by digest only, so the manifest
can be published without disclosing a conversation. The bundle root, the 64-leaf keccak Merkle tree and the identity
follow the specification to the byte: the builder reproduces the specification's own test vectors for three published
bundles. Every facet change is a new generation whose parent is the previous manifest's CID.

The mint uses the house ERC-7857 contract's `mintOpenAgent`. The token's `contentRoot` is the manifest's contentRoot,
which the contract accepts once and never again, and its `metadataRoot` is the keccak256 of the card. The runtime
builds and simulates the call, decodes the contract's own errors, and hands the owner an unsigned transaction. It
sends only on a local development chain. Loading reads the token's payload, finds the manifest generation with that
root, verifies the bundle, and walks the parent links from the agent's current generation back to the minted one.
The genesis stays attached; what came after is proven to descend from it. The whole path is tested on a throwaway
local chain (deploy, simulate, refuse without the role, mint, read back, refuse a second mint, load, evolve twice and
load again). The contract is not deployed on a public chain, and its audit is not yet cleared: a real mint waits for
both, and for a wallet with the minter role.

### III.8 Can a binary computer perform a ternary operation?

Yes. It does so exactly, because a trit is stored in bits and the operation on it is ordinary integer arithmetic. It
cannot store a trit natively, so it pays for the encoding in space.

**Information.** A trit carries log₂ 3 ≈ 1.585 bits. A binary memory must spend a whole number of bits on it (two,
leaving one of the four codes unused) or pack several trits into one integer. Since 3⁵ = 243 ≤ 256, five trits fit in
one byte (1.6 bits per trit), within 1 % of the bound. `Q2_0_g64` takes the simple route. It uses two bits per weight
plus a 16-bit scale per 64 weights, 2.25 bits per weight in all, and the fourth code (+2) is legal but never occurs
in Ternary-Bonsai-8B (§III.5). The cost is about 0.66 bits per weight over the information bound, which buys decoding
by one shift and one mask.

**Operation.** A weight *w* ∈ {−1, 0, +1} times an activation *x* is not really a multiplication. It is a selection:
add *x*, skip it, or subtract it. The ternary dot product is therefore Σ_{w=+1} *x* − Σ_{w=−1} *x*, the same signed
sum as the binary one (§III.2) with a third case that contributes nothing. Binary hardware can compute it in any of
several exact ways:

- sign-and-add, in which the sign of the code negates *x* and the zero code masks it out;
- two bit-planes, a "+1" mask and a "−1" mask, each a popcount or masked sum, subtracted;
- an offset code, which is bankml's. The stored code is *c* = *w* + 1 ∈ {0, 1, 2}, so Σ *w*·*x* = Σ *c*·*x* − Σ *x*.
  The AVX2 kernel multiplies the unsigned codes by the signed activation bytes with `vpmaddubsw`, and then subtracts
  Σ *x*. That sum depends only on the activation, so it is computed once per token and shared by every weight row
  (`q2_0.rs`).

Every path is integer arithmetic, so the result is not an approximation of the ternary operation; it *is* the ternary
operation, and the oracle confirms it to the bit on 762 of 762 products (§IV.2).

**What native ternary would add.** Balanced-ternary machines have been built. Setun (Moscow State University, 1958)
used {−1, 0, +1} natively, and it is the precedent for the idea. A ternary cell would store 1.585 bits of information
where a binary cell stores one, and ternary logic makes negation free. Neither is a question of *possibility*: binary
and ternary machines compute the same functions, and each can simulate the other with constant overhead. The choice is
one of cost. On a binary CPU, what limits a ternary model is memory bandwidth (§I.1, §IV.5). Packing closer to 1.585
bits per weight would therefore be the next ternary speed-up. Native ternary gates would not. bankml keeps 2 bits
because it must read the reference's file format bit for bit.

**The binary choice, compared.** `Q1_0` spends 1.125 bits per weight and has no zero: every weight votes, + or −.
`Q2_0_g64` spends 2.25 bits and lets a weight abstain. In this application the extra state gives better answers
(PERFORMANCE.md, "Ternary vs 1-bit"). Doubling the bits need not double the time, and on this CPU it does not. Once the
missing kernel is written, both formats run close to the rate at which memory can deliver their weights (§IV.5).

## IV. Results and testable propositions

### IV.1 Q1_0

On a single thread of an AMD Ryzen 3 3200U the bankml decode kernel runs a 12288×4096 matrix–vector product at
14.32 ns per block against the reference's 14.43 (1.01×, minima; 1.03× at the median), and the prefill tile at 12.95
against 14.22 ns per block-column (1.10×; 1.16× at the median), with every result bit-identical (§III.4). (Later runs
on the same laptop at a higher boost clock read about 10 ns per block for both kernels: the ratio is the result.) Since
0.0.4 a 1×4 selection tile over prepared activations (`mat_mul_act`) brings prefill to 1.2–1.3× the reference. The
oracle now also covers the 8B 1-bit model: all 254 tensors (8,188,239,872 weights) are bit-exact, as are 762 of 762
rows and dot products.

### IV.2 Q2_0_g64

All 254 `Q2_0` tensors of Ternary-Bonsai-8B — 8,188,239,872 weights — dequantize bit-identically to the reference;
762 of 762 activation rows and 762 of 762 dot products match its haswell build exactly, and a model without fused
multiply-adds matches its baseline x64 build in 762 of 762. The two reference builds disagree with each other on 19 of
the 762 cases, which shows the oracle distinguishes float orders rather than passing loosely.

On a single core, on real layer-0 weights, alternating the two kernels over the same bytes, bankml computes each
projection of the model **9.5 to 9.8 times faster** than the reference (e.g. 4.98 against 48.98 ns per 64-weight block
for the attention query projection, minima; 9.1–10.6× across four runs), and the prefill tile **12.5 times faster**
(3.96 against 48.90 ns per block-column). Full tables are in PERFORMANCE.md. Proposition P2 of §IV.4 is supported: the
reference's ternary slowdown is recoverable in the kernel, because it *is* the kernel.

### IV.3 Where the time goes, end to end

Timing one token's worth of all 253 ternary matrix products (2.13 GB of weights) in file order, bracketed by the
reference's own end-to-end benchmark at the same clock: on three threads the reference spends 2.54–2.64 s of its
2.59–2.68 s per token in these products — about 98 % of the wall — and bankml computes the same products in 0.36–0.37 s,
a matmul-bound ceiling of about 2.8 tokens per second against the reference's 0.37–0.39. On one thread the figures are
4.2–4.4 s against 0.49 s. With the persistent thread pool of 0.0.3 (both engines on the same scheduler), the three-thread
figure is 0.23–0.25 s against the reference's 2.18–2.36 s (9.2–9.9×), a ceiling of about 4 tokens per second. That
is less time than the reference spends on the *1-bit* model's products (0.34–0.37 s). What remains — attention with a
key–value cache, normalisation, rotary embedding, sampling — is the forward pass (§IV.4), and the ceiling stands until
it exists.

### IV.4 What bankml answers with today, and what remains

Since 0.0.6 bankml answers, through the reference (phase P0). `bankml serve` verifies the file and binds the reference
engine to it, and every answer carries a receipt. Its own transformer forward pass (normalization, rotary embedding,
attention with a quantized key–value cache, the feed-forward block, sampling) is phase P3. On the laptop a Savante
turn under her roughly 300-token system prompt took 125 s, 113 s of it prefill at 2.8 tokens per second in the
reference. That is the wall P3 and the prefill kernels are aimed at.

The gate does not depend on the format. The guard reads any mainline type (it names all of ggml's standard types,
from `Q4_0` and the K-quants to the `IQ` family and `MXFP4`), and no architecture is allow-listed. So a standard
Qwen3, SmolLM or Granite GGUF at `Q4_K_M` or `Q8_0` is served on exactly the terms of a 1-bit Bonsai: it is guarded,
pinned to its publisher's sha256, and carries a receipt. Since 0.1.5 the verification reports the architecture and the
weight types it checked. The importer (`ui/models.py`) brings such files in, and admits only open-source licences.
Only `Q1_0` and `Q2_0_g64` have bankml kernels proven against the oracle; other formats are answered by the reference
engine alone, which is what P0 means. The propositions this report makes testable are
stated at the kernel level:

- **P1.** A low-bit kernel proven bit-exact against the compiled reference can match its speed without giving up
  exactness (supported for `Q1_0`, §IV.1).
- **P2.** The reference runtime's ternary slowdown relative to 1-bit exceeds the ratio of their file sizes, so part of
  it is recoverable in the kernel (§IV.2–IV.3).
- **P3.** An end-to-end bankml answer will be token-identical to the reference's at temperature zero on the same
  prompts, because every kernel it uses is bit-exact — the criterion for phase P3.

### IV.5 The floor, and the limit of the test core

A read of a 768 MiB buffer on the same pool and scheduler as the matrix products measures the laptop's floor: 14.9–17.3
GB/s. Streaming one token's weights therefore takes at least 0.12–0.14 s (ternary) and 0.06–0.07 s (1-bit). Against it,
bankml's ternary products run about 2× above the floor at three threads and the 1-bit products (like the reference's)
about 5.5× above it. Both are bound by compute, not memory: the ternary kernel moves about 9 GB/s at three threads, and
gains 1.8× from one to three threads on two physical cores.

Releases 0.0.4 and 0.0.5 tried five bit-exact variants aimed at that compute; all passed the oracle and none shipped:

| variant | kernel | speed | reason |
|---|---|---:|---|
| two blocks per iteration, interleaved FMA chains | 1-bit decode | 1.00× | not latency-bound |
| "pair order": one multiply fewer per block | 1-bit decode | 0.84× | the cross-lane extract costs more than the multiply |
| two rows sharing the activation | ternary decode | 0.80× | register pressure |
| software prefetch, 256–1024 bytes ahead | ternary decode | ±3 % | the hardware prefetcher already follows the stream |
| repacked quads (codes contiguous, scales adjacent) | ternary decode | 0.90–1.075× | within noise across two runs |

(With the 2- and 4-row 1-bit tiles of 0.0.1, at 0.81–0.99×.) On this Zen+ core the kernels are therefore at their
instruction-throughput limit for bit-exact results. The gains still available lie elsewhere: the forward pass, and a
core with wider or better-fed vector units. The code of every rejected variant is kept in `testing/experiments/`.

## V. Objections

**"Bit-exactness is the wrong target; accuracy is what matters."** The model's accuracy is fixed by its weights; the
runtime's only freedom is to compute the same function faster. Any deviation, however small, is a different
function whose accuracy is unmeasured. Exactness is what lets speed be compared at all (§I.3).

**"A kernel benchmark is not an inference benchmark."** Agreed, which is why PERFORMANCE.md reports the reference's
end-to-end numbers beside the kernel numbers, and why §IV.3 measures how much of a token's wall time the matrix
products account for. A kernel win that is invisible end to end is reported as such.

**"One machine proves little."** The kernel results are from one laptop core; the deciding measurement for the
production server (a Zen3 core) is named as open in the crate's checklist. Relative results between two kernels on
the same core are less sensitive to the machine than absolute rates, but they are not independent of it.

**"Why not adopt the reference and contribute upstream?"** The oracle makes that path easy: every bankml kernel is,
by construction, a drop-in replacement for a reference function on the same bits. Owning the runtime serves the
second goal — a small, single-binary player with receipts — which upstream does not set out to be.

## VI. Future work

The plan continues the phases in `bankml.rs`; each step ends in a measured, reproducible row or is not done.

- **P3 — the forward pass, token-identical.** A Qwen3 forward in Rust: reading GGUF arrays (the tokenizer's
  vocabulary and merges), a byte-pair tokenizer and chat template, the embedding lookup, RMSNorm including Qwen3's
  query/key norms, rotary embedding — with the file's YaRN scaling (factor 4, original context 16,384) reproduced
  exactly as the reference applies it — grouped-query attention (32 query heads, 8 key/value heads, head size 128)
  over a `q8_0` key–value cache, SwiGLU, greedy sampling. The acceptance test is proposition P3: token-identical
  output to the reference at temperature zero on fixed prompts.
- **P0, done (0.0.6).** `bankml serve` puts the reference behind bankml's guard, pin and receipts. Next: swap bankml's
  own kernels in behind the same interface, one tensor type at a time, verified by the oracle.
- **Commitments on chain.** Receipts carry the answer's sha256 today, and the history is committed by a Merkle root
  and a CID (§III.6). Next: an optional THOT8 ternary commitment of each output, whose Keccak-256 leaf matches the
  on-chain `THOTLib.sol`, and THOT dataset bundles of the history and notes for inclusion with an iNFT.
- **Connectors, done on this machine (0.0.9–0.1.0); next, the public ends.** The agent path is done end to end:
  THOT bundles, PostgreSQL publish and load, and the iNFT mint and load on a local chain. Next: store bundles where
  a `storageURI` can name them (Arweave through the permanence rungs), compute embeddings for the vector column, and
  mint on a public chain once the contract is deployed and audited.
- **Faster 1-bit, if a core allows it.** On the test core every bit-exact 1-bit variant tried (§IV.5) ran at the
  reference's speed or slower. A core with AVX-512 VNNI (a single-instruction byte dot product) is where the next
  measurement belongs.
- **Real activations in the oracle.** Dump activations from the reference's own forward pass and verify against them,
  replacing today's real-weights × synthetic-activations cases.
- **The production machine and other CPUs.** A Zen3 row on the one-core server — it loads the same haswell build of the
  reference, so the same scalar ternary path is expected there, which the row will confirm or refute — NEON for ARM (P5, handheld devices), and AVX-512 where present.
- **The mindX seam (P4).** The OpenAI-compatible endpoint on loopback exists (`bankml serve`). Next: register it as a
  provider so mindX's inference budget records real usage, and run the sAGI engine and the improve.skill loop on it.
- **Upstream.** Because every bankml kernel is, by construction, a bit-exact drop-in for a reference function, the
  ternary kernel is offered back to llama.cpp as a contribution; the oracle is the evidence that accompanies it.
- **Voice (voaicey).** A licence-clean text-to-speech path for the model's answers — espeak-ng in the browser by
  default, with a template of the advanced voice settings — released alongside the runtime.

## References

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F. and Sanghai, S. (2023). "GQA: Training
Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *Proceedings of EMNLP 2023*.

Codephreak, Professor and Magnusson, G. L. (2026). Design directives for bankml and the mindX runtime, recorded in
the project (personal communication and project record, 2026-07-04 to 2026-09-26).

Courbariaux, M., Bengio, Y. and David, J.-P. (2015). "BinaryConnect: Training Deep Neural Networks with binary weights
during propagations." *Advances in Neural Information Processing Systems 28*.

Dettmers, T., Lewis, M., Belkada, Y. and Zettlemoyer, L. (2022). "LLM.int8(): 8-bit Matrix Multiplication for
Transformers at Scale." *Advances in Neural Information Processing Systems 35*.

Frantar, E., Ashkboos, S., Hoefler, T. and Alistarh, D. (2023). "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers." *International Conference on Learning Representations (ICLR 2023)*.

Gerganov, G. et al. *llama.cpp* and *ggml* (software). github.com/ggml-org/llama.cpp, release b11192.

Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R. and Bengio, Y. (2016). "Binarized Neural Networks." *Advances in
Neural Information Processing Systems 29*.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H. and Stoica, I. (2023).
"Efficient Memory Management for Large Language Model Serving with PagedAttention." *Proceedings of the 29th Symposium
on Operating Systems Principles (SOSP 2023)*.

Li, F., Zhang, B. and Liu, B. (2016). "Ternary Weight Networks." arXiv:1605.04711.

Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J. and Wei, F. (2024). "The Era of
1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv:2402.17764.

National Institute of Standards and Technology (2015). *Secure Hash Standard (SHS)*. FIPS PUB 180-4.

Qwen Team (2025). "Qwen3 Technical Report." arXiv:2505.09388.

Rastegari, M., Ordonez, V., Redmon, J. and Farhadi, A. (2016). "XNOR-Net: ImageNet Classification Using Binary
Convolutional Neural Networks." *European Conference on Computer Vision (ECCV 2016)*.

Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W. and Liu, Y. (2024). "RoFormer: Enhanced Transformer with Rotary Position
Embedding." *Neurocomputing* 568.

Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y. and Wei, F. (2023). "BitNet:
Scaling 1-bit Transformers for Large Language Models." arXiv:2310.11453.

Williams, S., Waterman, A. and Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore
Architectures." *Communications of the ACM* 52(4): 65–76.

Zhang, B. and Sennrich, R. (2019). "Root Mean Square Layer Normalization." *Advances in Neural Information Processing
Systems 32*.

Zhu, C., Han, S., Mao, H. and Dally, W. J. (2017). "Trained Ternary Quantization." *International Conference on
Learning Representations (ICLR 2017)*.
