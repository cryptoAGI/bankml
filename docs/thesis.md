# The bankML thesis: exactness before speed, and inference you can check on the computer you already have

*cryptoAGI · bankML · 6 October 2026. The design intent quoted in §0 is that of bankML's authors, Professor
Codephreak and Gregory L. Magnusson, in their own words and dated. The argument around it is assembled from the
project's record: its code, its [CHANGELOG](../CHANGELOG.md), its measurements ([PERFORMANCE.md](PERFORMANCE.md)) and
its oracles ([oracles.md](oracles.md)). Written while v0.3.6 is the latest release and 0.3.7–0.3.9 are being gated on
the way to 0.4.0. The plain-language version is [why-bankml.md](why-bankml.md); the technical report is
[TECHNICAL.md](TECHNICAL.md).*

---

## Abstract

Low-bit language models make an eight-billion-parameter network small enough for a laptop. The engines that run
them report speed, but a speed figure says nothing about whether the engine computes what the model's reference
computes, and in floating point two correct-looking programs rarely produce the same bits. This thesis argues that
**for a low-bit inference engine, bit-exactness against the reference implementation's compiled code is the
correctness criterion, and it must be met before any speed is reported**. It argues further that the criterion is
not a cost paid against speed but an instrument for finding speed: holding every kernel to the reference's bits is
how bankML found that the reference has no vectorised x86 kernel for ternary weights, and how it closed a 9.4–10×
gap without changing a single answer. We define the terms, place the work in its lineage, give the evidence the
project has gathered (every claim tied to an oracle that can be re-run), answer the strongest objections, state the
limits, and set out what remains between the present work and a 1.0 in which the reference is needed only as the
oracle.

---

## Thesis

> An inference engine for low-bit models should be **exact first and fast second**: every kernel, the tokenizer, the
> templates, the samplers and whole conversations reproduce the reference's own compiled output bit for bit; only
> then is speed measured; and every answer carries the evidence of the file and the arithmetic that produced it. On
> commodity CPUs this discipline does not cost speed. It finds it.

---

## 0. The design intent, in the authors' words

These are recorded in [TECHNICAL.md](TECHNICAL.md#thesis--professor-codephreak-and-gregory-l-magnusson), quoted here
exactly and dated. The rest of this document measures the work against them.

- **Port what works.** "We prototyped in python and we switch to rust from working architecture" (2026-07-04). The
  instruction that began the runtime: "create llama.cpp rust version todo as bankml.rs" (2026-09-25).
- **Three goals.** "optimization, succinct, and verified response" (2026-09-25).
- **Consume the inference, prefer the better answer.** "If there is inference mindX needs to consume it, however long
  it takes" (2026-09-26); "a better answer is more important than speed where speed under one minute is fine for now"
  (2026-09-26).
- **Know the limit, code around it.** "They should know limits and be able to code around them" (2026-09-26).
- **Increments, with the oracle in the loop.** "Audit and improve with three more incremental release pushes focusing
  on optimization and performance including oracle feedback" (2026-09-28).
- **The machine at hand.** "Use this laptop hardware" (2026-09-28).
- **Prove the data, keep the data.** "Localstorage and hash and CID can work as data reference for proof of data while
  keeping local data private and still verifiable" (2026-09-28).

---

## I. The problem

### I.1 Small models, unverified engines

A 1-bit (`Q1_0`) Bonsai-8B occupies 1.2 GB and its ternary (`Q2_0_g64`) sibling 2.3 GB, so a laptop with 6 GB of memory
or a two-core server can hold either. On such a machine, inference is limited by how fast the matrix kernels consume
packed weights, and every engine reports a tokens-per-second figure for it.

That figure is not, by itself, evidence of anything. Floating-point addition is not associative, so a kernel that
sums the same products in another order, fuses a multiply and an add, or rounds an intermediate to half precision
produces different bits (Goldberg 1991). Usually the difference is invisible. Sometimes it changes a greedy token, and
from then on the answer diverges. An engine that is "faster" by a different arithmetic has answered a different
question: whether its outputs are as good as the reference's must then be argued statistically, model by model, and
it rarely is.

### I.2 The reference has gaps a user cannot see

The reference engine here is llama.cpp at release b11192 (Gerganov et al.), the engine the Bonsai files were made
for. On x86, b11192 ships **no vectorised kernel for `Q2_0`**: the generic C is renamed to the x86 symbol in
`arch-fallback.h`, and the shipped library runs scalar code with 64 integer multiplies per block
([q2_0.md](modules/q2_0.md)). So the ternary model, which gives the better answers on the project's own carrier test
([TECHNICAL.md, contribution 10](TECHNICAL.md#contributions)), ran five to seven times slower than the 1-bit one. Nothing
in the reference's output says so. It was found by measuring kernel against kernel on the same bits.

### I.3 The question

Can a runtime be built that (a) gives exactly the reference's answers, (b) is at least as fast and faster where the
reference is weak, (c) refuses a model it cannot verify, and (d) lets anyone check, after the fact, which file and
which arithmetic produced an answer, all on a commodity CPU and with nothing to trust beyond its own source?

---

## II. Lineage

### II.1 Low-bit networks

Binary weights with a real-valued training copy come from BinaryConnect (Courbariaux, Bengio and David 2015) and
binarized activations from Hubara et al. (2016). XNOR-Net added the per-filter scale (Rastegari et al. 2016), the
ancestor of every per-group scale in today's formats. Ternary Weight Networks (Li, Zhang and Liu 2016) and Trained
Ternary Quantization (Zhu et al. 2017) added the zero. For transformers, post-training methods reached 8 bits
(Dettmers et al. 2022) and 3–4 bits (Frantar et al. 2023); below that, models are trained through the constraint:
BitNet (Wang et al. 2023) and the ternary "1.58-bit" models (Ma et al. 2024), to which Bonsai belongs, as dense Qwen3
networks (Qwen Team 2025).

The strongest form of this school's claim is that, trained in the loop, a network adapts to one or two bits with
little loss. bankML takes that claim as given. Its subject is the step after: running such a model, and knowing it
was run correctly.

### II.2 Runtimes

Server engines such as vLLM maximise throughput over many requests on accelerators, chiefly by paging the key–value
cache (Kwon et al. 2023). Their gains come from batching and do not transfer to one user on a CPU. The ggml/llama.cpp
lineage targets that case: one process, kernels per format and instruction set, weights memory-mapped. CPU kernels
for ternary models specifically include bitnet.cpp (Wang, Zhou, Song et al. 2024; 2025) and table-lookup methods
such as T-MAC (Wei et al. 2025). bankML belongs to the ggml lineage and keeps its file format, its kernels' float
order and its server's behaviour as the specification.

### II.3 Floating point and the trust problem

Two older lines of thought meet here. The first is numerical: identical results across implementations require
identical operation order and rounding (Goldberg 1991). The second is about trust in compiled code: what a program
does is fixed by the instructions it was compiled to, not by its source (Thompson 1984), which is why reproducible
builds compare binaries rather than intentions (Lamb and Zacchiroli 2022). bankML applies both to inference. Its
oracle calls the reference's **shipped, compiled library** in-process on the same bytes, and that is how the project
found that the shipped binary contracts multiply–add pairs into fused instructions, so that a port faithful to the
C source disagrees in the last bit ([TECHNICAL.md §III.4](TECHNICAL.md#iii4-the-oracle-exactness-against-the-compiled-reference)).

### II.4 Verifiable inference

Work on verifiable inference spans zero-knowledge proofs of a model's computation (e.g. zkLLM, Sun, Li and Zhang
2024), optimistic re-execution that lets a dispute be settled by running the computation again (opML, Conway et al.
2024), and attested hardware that signs results (EigenAI, Alves et al. 2026), which also stresses deterministic
inference. Bit-exact verification has also been pursued for GPU engines (Cankaya 2026). bankML's relation to these is
specific. Re-execution and determinism both presuppose that two runs of the same model produce the same bits, and
bankML's discipline is to establish exactly that, against an independent reference, per kernel. Its receipts are
integrity, not proof: they bind the answer's text to a pinned file and to the request, between a client and its own
gateway ([research.md §3](research.md#3-verifiable-and-attested-inference)).

### II.5 What bankML inherits, extends and breaks

It **inherits** ggml's formats, its kernels' semantics and llama-server's protocol. It **extends** them with an x86
ternary kernel the reference lacks, a gate in front of every answer, and receipts. It **breaks** with the convention
that a new engine reports speed first and argues quality afterwards: here speed is reported only for arithmetic
already shown identical.

---

## III. The contribution

### III.1 Definitions

- **Reference.** A named, sha256-pinned release of another engine (here llama.cpp b11192), its compiled libraries and
  its server, run on the same machine.
- **Bit-exact.** A function's output equals the reference's output on the same input, compared as bit patterns, not
  within a tolerance.
- **Oracle.** A test that obtains the reference's output by running the reference itself, in-process through its
  exported symbols or as its own server, and compares bankML's output with it. An oracle's count is reported as
  matched / total ([oracles.md](oracles.md)).
- **Gate.** The ordered set of checks that decide whether a version ships ([`testing/release_gate.sh`](../testing/release_gate.sh));
  its output is kept as the release's record ([`testing/results/`](../testing/results/)).
- **Verified response.** An answer produced only from a file that passed the guard and the pin, by arithmetic the gate
  has shown bit-exact, and carrying a receipt of the file's sha256, the request's and the answer's.

### III.2 Axioms the runtime is built on

1. **Exactness precedes speed.** No speed figure is recorded for a kernel until its oracle passes in the same run
   ([PERFORMANCE.md](PERFORMANCE.md)). A variant that is bit-exact but not reliably faster is kept as a negative
   result and not shipped ([`testing/experiments/`](../testing/experiments/)).
2. **The compiled reference is the specification, not its source.** Where the shipped library and its C source
   disagree, the library wins (§II.3).
3. **A model that cannot be verified does not answer.** The guard ([`bankML/gguf.rs`](../bankML/gguf.rs)) refuses,
   from the header alone and with a reason, the three low-bit traps, including a file mainline loads and answers in
   fluent nonsense. The pin ([`bankML/sha256.rs`](../bankML/sha256.rs)) refuses a file whose hash differs from its
   provenance record.
4. **Refuse rather than approximate.** A sampler, option or request the runtime cannot reproduce exactly is refused
   with the reason, not served approximately ([modules/sampler.md](modules/sampler.md)).
5. **Succinctness is a budget.** One crate, zero external dependencies, one binary
   ([`Cargo.toml`](../Cargo.toml)): everything that touches an answer can be read.

### III.3 The mechanism: port, prove, then exceed

The method repeats for each component. Port the reference's behaviour; build an oracle that runs the reference;
iterate until the oracle passes in full; only then optimise, re-running the oracle on every change. For the kernels
the optimisation keeps ggml's floating-point chain and reorders only integer sums, which are exact in any order
([q2_0.md](modules/q2_0.md)). The same method, applied above the kernels, carried the runtime from a gateway in front
of the reference (0.0.6) to a forward pass and server of its own (0.2.7–0.3.0), and then through each behaviour a
client meets: the Ollama API, JSON grammars and schemas, penalties, the whole sampler chain, the context limit, slots,
the prompt cache and logprobs ([CHANGELOG](../CHANGELOG.md)).

### III.4 The instrument finds what it is pointed at

The oracle's value shows most clearly in what it has found that was not being looked for:

- **The missing ternary kernel** (§I.2): found by timing kernel against kernel on identical bits.
- **Fused multiply–add in the shipped binary**: found because a source-faithful port missed the last bit; the oracle
  also distinguishes the haswell and baseline x64 builds, which disagree on 19 of 762 ternary cases
  ([TECHNICAL.md, contribution 4](TECHNICAL.md#contributions)).
- **The Hadamard rotation around a quantized cache** (0.3.9): bankML's first q8_0 KV cache matched llama-server on 2 of
  6 answers, each agreeing for dozens of tokens and then drifting. The cause was that b11192 rotates queries, keys and
  values through a Hadamard transform whenever the cache is quantized, an outlier-smoothing technique of the QuaRot
  and QuIP# line (Ashkboos et al. 2024; Tseng et al. 2024). With the rotation reproduced, 6 of 6
  ([CHANGELOG 0.3.9](../CHANGELOG.md)).

None of these is visible in a tokens-per-second figure, and none would have been found by a tolerance-based test.

---

## IV. Evidence and testable propositions

Each proposition below is a claim anyone can re-test: the oracle named is in the gate, and its record is published.

| # | proposition | evidence (oracle · figure) | where |
|---|---|---|---|
| P1 | The 1-bit kernel computes the reference's bits on real models | every weight of Bonsai-1.7B (1.72 B) and Bonsai-8B (8.19 B) | [q1_0.md](modules/q1_0.md) |
| P2 | The ternary kernel computes the reference's bits and is 9.4–10× faster per matrix | all 8.19 B weights of Ternary-Bonsai-8B; 9.4–10.0× in every gate since 0.0.1 (9.4–10.8×); one token's 253 matmuls 0.233 s against 2.204 s at three threads | [q2_0.md](modules/q2_0.md), [PERFORMANCE.md](PERFORMANCE.md) |
| P3 | Whole answers on the ternary model are the reference's tokens, about 8× faster end to end | greedy and seeded oracles; 2.3–2.4 against 0.30 tokens/s on the same laptop | [README](../README.md), [PERFORMANCE.md](PERFORMANCE.md) |
| P4 | The whole model is bit-exact, layer by layer | every row of every layer and all 151,669 logits; three CPU attention kernels | [forward.md](modules/forward.md) |
| P5 | Conversations served by bankML equal llama-server's turn by turn, text, counts and cache reuse | `oracle_native_serve` 9 / 9 turns; `session_oracle_live` 14 / 14 interleaved | [serve.md](modules/serve.md), [prompt_cache.md](modules/prompt_cache.md) |
| P6 | Constrained answers are the reference's: JSON mode, grammars, schemas | grammar masks 196 / 196 runs, 1,645 masks; schema grammars 173 / 173 per template | [grammar.md](modules/grammar.md), [schema.md](modules/schema.md) |
| P7 | The sampler chain is the reference's, seed for seed | penalties 56 / 56; the whole chain 76 / 76 on three models | [sampler.md](modules/sampler.md) |
| P8 | Token probabilities are the reference's floats, streamed or not | logprobs 14 / 14, every value the same 32-bit float | [serve.md](modules/serve.md) |
| P9 | A half-size q8_0 cache keeps the reference's answers | kernels 4,000 / 4,000 bit-exact against the shipped library; answers 6 / 6 | [forward.md](modules/forward.md) |
| P10 | The kernels sit at the hardware's limit, not the memory floor | a measured 15–17 GB/s floor; five further bit-exact variants with no reliable gain | [TECHNICAL.md §IV.5](TECHNICAL.md#iv5-the-floor-and-the-limit-of-the-test-core) |
| P11 | Exactness costs nothing on the grammar mask either | a trie mask 13× faster at the median (2.94 against 38.8 ms), every mask identical by both paths | [grammar.md](modules/grammar.md) |

**Open proposition.** P12: *1-bit decode is at least at the reference's speed.* A loaded-machine pair read 2.30–2.59
against 2.33–2.48 tokens/s after 0.3.4; the claim waits for the pinned, idle-machine measurement
([`testing/decode_ab.py`](../testing/decode_ab.py)) and is not made here.

---

## V. Objections

**1. "Bit-exactness is too strict. A different but equally accurate arithmetic is just as good."** In its strongest
form: an engine with a better summation order may be *more* accurate than the reference, so binding it to the
reference's rounding forbids improvement. The answer is that "equally good" is a statistical claim about a
distribution of answers, which must be argued model by model and is rarely argued at all, while "identical" is a
claim anyone can check on any input. A runtime that wants to differ can still do so, but then its answers are its
own. bankML chose the claim that can be checked. The cost has so far been zero: P2 and P11 are faster with identical
bits.

**2. "It binds the runtime to one version of one engine."** True, and stated: b11192 is pinned by sha256. Moving to a
newer reference means re-recording the oracles and passing them again, which is the same work any port needs to know
it is still right. The pin makes the change visible instead of silent.

**3. "The 1-bit kernel is not faster, so exactness has a ceiling."** The 1-bit kernel is at parity
(0.93–1.13× decode across releases) and §IV.5 of the technical report places it at the instruction-throughput limit
of the test core: five bit-exact variants gained nothing reliable. That is a property of the core, not of exactness;
the next measurement belongs on a core with a single-instruction byte dot product (AVX-512 VNNI)
([TECHNICAL.md §VI](TECHNICAL.md#vi-future-work)).

**4. "Receipts prove nothing to a third party."** Correct, and stated in [research.md](research.md#3-verifiable-and-attested-inference):
today's receipts are unsigned integrity between a client and its own gateway. What bankML adds to the verifiability
literature is the precondition the stronger schemes need. Re-execution (opML) and deterministic attested inference
(EigenAI) both require that a re-run gives the same bits; bankML establishes that against an independent reference,
per kernel. Signed receipts are on the road to 1.0.

**5. "CPU-only inference is a niche."** It is the case the project exists for: one server shown sufficient to run one
autonomous AI system, accelerators rented for events rather than owned (TECHNICAL.md, Thesis). The design already
admits a GPU where one is present, on the same terms: a card takes part only after it proves on the card that it
gives the CPU's bits ([gpu.md](modules/gpu.md)).

**6. "The reference could simply add the missing kernel."** It could, and bankML offers it: a drop-in in ggml's own C,
bit-exact on 200,000 of 200,000 cases and 3.4× the shipped scalar path ([TECHNICAL.md §VI](TECHNICAL.md#vi-future-work)).
The thesis does not depend on the gap staying open. It depends on the method that found it.

---

## VI. Limits, stated

- Architectures: Qwen3 and Llama. Weight types: `Q1_0`, `Q2_0_g64`, F16. Others are refused with the reason.
- One conversation slot, as llama-server `-np 1`; more slots wait for continuous batching.
- Receipts are unsigned. The commitments of the conversation (a Merkle root and a CID) prove data to whoever holds
  the root, not to the world.
- Every speed figure is from one laptop core class (Ryzen 3 3200U, Zen+) unless stated. A one-core server and a
  sixteen-core container were measured for baselines, not for the kernels' claims.
- Bit-exactness is against b11192's haswell build on x86 with AVX2. Other builds and other instruction sets are
  separate oracles.

---

## VII. What remains: from 0.4.0 to 1.0

**0.4.0, native serving complete.** Everything Savante and mindX ask of llama-server, answered by bankML's own engine.
What remains is the open proposition P12 (1-bit decode at the reference's speed, measured pinned and idle), the
engine setting's `auto` choosing bankML for both 1-bit and ternary files, and the milestone gate
([TODO.md](TODO.md)).

**0.5.0, hardware.** NEON for ARM (phones and tablets), AVX-512 where present, and more graphics cards, each under its
own oracle.

**1.0.0.** The definition the project has set itself ([TODO.md](TODO.md)):

1. llama.cpp is needed only as the oracle in the gate, never at run time;
2. every supported architecture and format has its whole-model, greedy, sampling and conversation oracles, and passes
   them;
3. on every supported format bankML is at least at the reference's speed, and ahead where the kernels allow, on named
   machines;
4. stable, documented interfaces under semver;
5. signed receipts wherever the operator holds a key, and a verifier anyone can run;
6. Savante and mindX run on bankML by default.

When that holds, the thesis will have become a property of a released system rather than a claim about one: an
engine whose every answer can be reproduced bit for bit by an independent reference, and that is faster than that
reference where it matters.

---

## References

Alves, P., Patankar, A., Pereira, B. et al. (2026). "EigenAI: Deterministic Inference, Verifiable Results."
arXiv:2602.00182.

Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Cameron, P., Jaggi, M., Alistarh, D., Hoefler, T. and Hensman, J.
(2024). "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." *Advances in Neural Information Processing Systems
37*. arXiv:2404.00456.

Cankaya, E. (2026). "Bit-Exact AI Inference Verification Without Performance Tradeoffs." arXiv:2606.00279.

Codephreak, Professor and Magnusson, G. L. (2026). Design directives for bankML and the mindX runtime, recorded in the
project (project record, 2026-07-04 to 2026-09-28); quoted in [TECHNICAL.md](TECHNICAL.md#thesis--professor-codephreak-and-gregory-l-magnusson).

Conway, K. D., So, C., Yu, X. et al. (2024). "opML: Optimistic Machine Learning on Blockchain." arXiv:2401.17555.

Courbariaux, M., Bengio, Y. and David, J.-P. (2015). "BinaryConnect: Training Deep Neural Networks with binary weights
during propagations." *Advances in Neural Information Processing Systems 28*.

Dettmers, T., Lewis, M., Belkada, Y. and Zettlemoyer, L. (2022). "LLM.int8(): 8-bit Matrix Multiplication for
Transformers at Scale." *Advances in Neural Information Processing Systems 35*.

Frantar, E., Ashkboos, S., Hoefler, T. and Alistarh, D. (2023). "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers." *International Conference on Learning Representations (ICLR 2023)*.

Gerganov, G. et al. *llama.cpp* and *ggml* (software). github.com/ggml-org/llama.cpp, release b11192.

Goldberg, D. (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic." *ACM Computing
Surveys* 23(1): 5–48.

Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R. and Bengio, Y. (2016). "Binarized Neural Networks." *Advances in
Neural Information Processing Systems 29*.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H. and Stoica, I. (2023).
"Efficient Memory Management for Large Language Model Serving with PagedAttention." *Proceedings of the 29th Symposium
on Operating Systems Principles (SOSP 2023)*.

Lamb, C. and Zacchiroli, S. (2022). "Reproducible Builds: Increasing the Integrity of Software Supply Chains." *IEEE
Software* 39(2).

Li, F., Zhang, B. and Liu, B. (2016). "Ternary Weight Networks." arXiv:1605.04711.

Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J. and Wei, F. (2024). "The Era of
1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv:2402.17764.

Qwen Team (2025). "Qwen3 Technical Report." arXiv:2505.09388.

Rastegari, M., Ordonez, V., Redmon, J. and Farhadi, A. (2016). "XNOR-Net: ImageNet Classification Using Binary
Convolutional Neural Networks." *European Conference on Computer Vision (ECCV 2016)*.

Sun, H., Li, J. and Zhang, H. (2024). "zkLLM: Zero Knowledge Proofs for Large Language Models." *Proceedings of the ACM
Conference on Computer and Communications Security (CCS 2024)*. arXiv:2404.16109.

Thompson, K. (1984). "Reflections on Trusting Trust." *Communications of the ACM* 27(8): 761–763.

Tseng, A., Chee, J., Sun, Q., Kuleshov, V. and De Sa, C. (2024). "QuIP#: Even Better LLM Quantization with Hadamard
Incoherence and Lattice Codebooks." *International Conference on Machine Learning (ICML 2024)*.

Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y. and Wei, F. (2023). "BitNet:
Scaling 1-bit Transformers for Large Language Models." arXiv:2310.11453.

Wang, J., Zhou, H., Song, T. et al. (2024). "1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on
CPUs." arXiv:2410.16144.

Wang, J., Zhou, H., Song, T. et al. (2025). "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs."
arXiv:2502.11880.

Wei, J. et al. (2025). "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge." *Proceedings of
EuroSys 2025*. arXiv:2407.00088.

Zhu, C., Han, S., Mao, H. and Dally, W. J. (2017). "Trained Ternary Quantization." *International Conference on
Learning Representations (ICLR 2017)*.

*Notes on the references.* Author lists for the 2024–2026 preprints follow [research.md](research.md), which records
which details were re-fetched and which are as commonly cited. QuaRot and QuIP# are cited for the technique of
rotating activations by a Hadamard transform before quantization; the attribution of llama.cpp's own rotation to
their influence is ours, not a statement by llama.cpp's authors.
