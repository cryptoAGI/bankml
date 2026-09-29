# Research: where bankml stands

*A survey of Rust-native LLM inference, 1-bit and ternary CPU inference, and verifiable inference, made on
2026-09-28/29 to answer one question: is bankml at the cutting edge, and where is it not? Every link below was
checked to resolve on those dates unless marked. bankml's own numbers are its own measurements
([TECHNICAL.md](TECHNICAL.md), [PERFORMANCE.md](PERFORMANCE.md), [`testing/results/`](../testing/results/)); no third
party has benchmarked bankml.*

## The answer in brief

**Where bankml is at or near the edge**

1. **Bit-exact reproduction of the compiled llama.cpp library, tested by an oracle, from Rust.** bankml's `Q1_0` and
   `Q2_0_g64` kernels are compared against the shipped ggml shared objects of llama.cpp b11192. The check covers every
   weight, every `q8_0` activation row and every dot product, and matches the compiler's fused multiply-add rounding.
   No other project was found that does this against the *binary*. The nearest are:
   - [OxiLLaMa](https://github.com/cool-japan/oxillama), which claims logit parity within a tolerance;
   - [Frink](https://github.com/antonellof/frink), whose quantizer output is byte-identical but whose matmul is not
     claimed to be;
   - [bitnet-rs](https://github.com/lilyco-42/bitnet-rs), which has byte-level unit tests against its C++ baseline.

   llama.cpp's own [PR #26348](https://github.com/ggml-org/llama.cpp/pull/26348) shows why the FMA detail matters:
   its kernel matched bit for bit in isolation, but perplexity still shifted once the compiler contracted it
   differently.
2. **A plain-AVX2 kernel for `Q2_0_g64`, a gap upstream has not closed.** At b11192, llama.cpp has only scalar code
   for `Q2_0` on x86:
   - [PR #24448](https://github.com/ggml-org/llama.cpp/pull/24448) (merged 2026-07-07) is "CPU only, ARM NEON +
     generic scalar fallback";
   - the x86 kernel in [PR #26348](https://github.com/ggml-org/llama.cpp/pull/26348) needs AVX-VNNI and is still
     open.

   bankml's AVX2 kernel is 9.5–9.8× the reference per projection, bit-exact. **Caveat:** that speedup is measured
   against a scalar baseline. PrismML's own fork has AVX2 kernels for its *other* ternary format (`PQ2_0`, [PR
   #206](https://github.com/PrismML-Eng/llama.cpp/pull/206), about 3× over its own baseline), which is a different
   format and not directly comparable.
3. **A pinned-model gateway with a receipt on every answer, in one zero-dependency static binary.** Model hashes
   and answer hashes are common inside TEE and attestation stacks, but rare as a simple local gate. **Caveat:**
   bankml's receipts are not signed or hardware-attested. They prove what this gateway served, not where or by whom
   (§3 below).

**Where others are clearly ahead**

- **A full forward pass in Rust.** [candle](https://github.com/huggingface/candle),
  [mistral.rs](https://github.com/EricLBuehler/mistral.rs), [Frink](https://github.com/antonellof/frink),
  [OxiLLaMa](https://github.com/cool-japan/oxillama) and [bitnet-rs](https://github.com/lilyco-42/bitnet-rs) run
  whole models natively. bankml answers through llama-server; its own Qwen3 forward pass is phase P3.
- **GPUs.** mistral.rs, Frink and PrismML's fork run on CUDA and/or Metal; bankml is CPU-only by design.
- **Breadth of formats.** OxiLLaMa lists K, IQ, TQ1_0, TQ2_0 and Q1_0_G128 (all "Alpha"); bankml has its own kernels
  for two formats and serves the rest through the reference.
- **ARM.** Upstream `Q1_0`/`Q2_0` have NEON; bankml's kernels are x86 AVX2 only.
- **End-to-end speed leadership.** It is not shown: bankml's gains are per kernel, and at the whole-token level
  the decode-budget ratio against the reference ranges from 0.87× to 1.62× across thread counts in the 0.1.5 gate
  record, measured on a laptop that was also serving a model, so it is noisy and not a claim of end-to-end leadership.
- **Attestation.** [EigenAI](https://arxiv.org/abs/2602.00182) (bit-exact GPU inference on a modified llama.cpp,
  re-executed in a TEE) and TEE serving stacks give receipts a third party can trust. bankml's receipts do not.

## 1. Rust-native inference engines

| project | what it runs on CPU | own kernels? | vs llama.cpp on CPU | exactness testing |
|---|---|---|---|---|
| [candle](https://github.com/huggingface/candle) (Hugging Face) | GGUF K-quants ([k_quants.rs](https://github.com/huggingface/candle/blob/main/candle-core/src/quantized/k_quants.rs)); x86 repack in progress, aarch64 in [PR #3697](https://github.com/huggingface/candle/pull/3697) | yes (AVX2/NEON) | no published parity numbers | none found |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | GGUF Q/K, ISQ, GPTQ, AWQ, HQQ, FP8 | yes (on candle) | published wins are CUDA; no CPU comparison | none found |
| [burn](https://github.com/tracel-ai/burn) / [models](https://github.com/tracel-ai/models) | a general deep-learning framework | yes | not a GGUF-quant CPU engine | — |
| [rustformers/llm](https://github.com/rustformers/llm) | archived, "Unmaintained" (last push 2024-06) | called ggml | — | — |
| [lm.rs](https://github.com/samuel-vitorino/lm.rs) | a minimal engine (last push 2024-10) | yes | — | — |
| [ratchet](https://github.com/huggingface/ratchet) | browser / WebGPU | — | — | — |
| [Crane](https://github.com/lucasjinreal/Crane), [kalosm](https://github.com/floneum/kalosm) | built on candle | — | — | — |
| [OxiLLaMa](https://github.com/cool-japan/oxillama) (2026) | K, IQ, TQ1_0/TQ2_0, Q1_0_G128, all "Alpha" | yes (AVX2/AVX-512/NEON) | targets ≥ 80 % of llama.cpp | top-1 logit parity 32/32, within tolerance, not bit-exact |
| [Frink](https://github.com/antonellof/frink) (2026; [blog](https://www.fratepietro.com/2026/frink-rust-gguf-inference-engine/)) | K, IQ, MXFP4, MoE; CPU, Metal, CUDA | yes | 1.41–5.06× **slower** than llama.cpp on CPU, by its author | quantizer bytes identical on Q8_0, IQ4_NL |
| [llama-gguf](https://github.com/Lexmata/llama-gguf) | GGUF | yes | 0.3 tok/s on Mistral-7B; "correctness over speed" | — |

## 2. 1-bit and ternary inference

**Upstream llama.cpp** ([releases](https://github.com/ggml-org/llama.cpp/releases); latest b11243):
- `Q1_0` arrived in [PR #21273](https://github.com/ggml-org/llama.cpp/pull/21273) (merged 2026-04-06, NEON and scalar
  only), and x86 SSSE3→AVX2+FMA in [PR #21636](https://github.com/ggml-org/llama.cpp/pull/21636) (merged 2026-04-20).
- `Q2_0` is in [PR #24448](https://github.com/ggml-org/llama.cpp/pull/24448) (merged 2026-07-07; x86 "later").
- The only x86 `Q2_0` kernel is VNNI-only, in [PR #26348](https://github.com/ggml-org/llama.cpp/pull/26348)
  (open; 2.1–3.6×), which does not help a plain-AVX2 CPU such as bankml's test Ryzen 3200U.
- The older ternary types `TQ1_0`/`TQ2_0` come from [PR #10010](https://github.com/ggml-org/llama.cpp/pull/10010),
  and the formats are discussed in [#22019](https://github.com/ggml-org/llama.cpp/discussions/22019).
- [Issue #29351](https://github.com/ggml-org/llama.cpp/issues/29351) (2026-09-24) found an x86 `Q8_0` sign bug by
  checking against the generic C kernel, which is the method bankml's oracle uses.

**PrismML** (the Bonsai models): [Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf), [announcement](https://prismml.com/news/bonsai-8b),
[docs](https://docs.prismml.com/run/llamacpp), and the [fork](https://github.com/PrismML-Eng/llama.cpp), which has
x86 kernels for PrismML's newer formats:
- `PQ2_0` AVX2/AVX-VNNI: [PR #206](https://github.com/PrismML-Eng/llama.cpp/pull/206), merged 2026-09-21, about 3×
  decode on Ternary-Bonsai-2-27B;
- `Q1_0` 4×8 repack: [PR #233](https://github.com/PrismML-Eng/llama.cpp/pull/233);
- `PQ1_0`: [PR #255](https://github.com/PrismML-Eng/llama.cpp/pull/255);
- `PTQ1_0`: [PR #250](https://github.com/PrismML-Eng/llama.cpp/pull/250).

**Microsoft:**
- [bitnet.cpp](https://github.com/microsoft/BitNet) has kernels I2_S, TL1 (2 weights → 4-bit index) and TL2
  (3 weights → 5-bit). It reports 2.37–6.17× on x86 and 72–82 % less energy.
- [T-MAC](https://github.com/microsoft/T-MAC) does lookup-table mixed-precision GEMM, up to 6.6× per kernel and 2.8×
  end to end over llama.cpp.
- Both work on BitNet-style ternary weights, a different format from ggml's `Q2_0`.

**In Rust:**
- [bitnet-rs](https://github.com/lilyco-42/bitnet-rs) (NEON);
- [alice-aegis](https://github.com/Aefinity-AI/alice-aegis) (2026), a `no_std` UEFI BitNet engine with frozen
  integer semantics, bit-identical digests across ISAs and **SHA-256 receipts chaining the logits**. It is the
  closest peer to bankml's receipts, but it checks against itself, not against llama.cpp;
- [bitnet-toy](https://github.com/tidynest/bitnet-toy);
- [bitnet-llm](https://crates.io/crates/bitnet-llm) (FFI bindings).

No Rust engine other than bankml was found with `Q2_0_g64`; OxiLLaMa lists `Q1_0_G128`.

## 3. Verifiable and attested inference

| approach | projects | what it proves | bankml's relation |
|---|---|---|---|
| zero-knowledge proofs | [EZKL](https://github.com/zkonduit/ezkl) (Rust, small models); [zkLLM](https://arxiv.org/abs/2404.16109); [DeepProve](https://eprint.iacr.org/2026/1112); [OpenLLM](https://eprint.iacr.org/2026/1578) | that a computation was done, with no trust in the prover; [Hollow-LLM](https://arxiv.org/abs/2607.28884) shows ZK correctness alone does not prove a real large model ran | none; far heavier |
| optimistic re-execution | [opML](https://github.com/ora-io/opml) ([paper](https://arxiv.org/abs/2401.17555)); [Optimistic TEE-Rollups](https://arxiv.org/abs/2512.20176) | fraud is challengeable by re-running | bankml's bit-exactness is what re-execution needs, but it has no dispute protocol |
| TEEs | [EigenAI](https://arxiv.org/abs/2602.00182); [OpenPCC](https://arxiv.org/abs/2606.11145); [Phala private AI inference](https://phala.com/solutions/private-ai-inference) | signed receipts from attested hardware | bankml's receipts are unsigned, with no hardware root |
| model pin + answer hash | bankml | which verified file served, and that the text and request are unchanged | honest scope: integrity between a client and its own gateway |

## 4. What this means for bankml's roadmap

- **P3 (its own forward pass) is where the field already is**, so it is not optional if bankml is to stand beside
  candle and OxiLLaMa. Its distinct contribution would be a forward pass *bit-exact against the compiled reference*,
  which nobody else claims.
- **A plain-AVX2 `Q2_0` kernel is worth upstreaming** (or offering to PrismML): the gap is real, current and
  documented upstream.
- **Receipts:** signing them (an operator key) and binding them to a re-executable bit-exact run would move them
  from integrity toward verifiability, the direction EigenAI shows.
- **NEON** would bring the kernels to ARM, where upstream `Q1_0`/`Q2_0` already are.

## Papers

**Binary and ternary networks**
1. Courbariaux, Bengio, David. [*BinaryConnect: Training Deep Neural Networks with Binary Weights during Propagations*](https://arxiv.org/abs/1511.00363). NeurIPS 2015.
2. Rastegari, Ordonez, Redmon, Farhadi. [*XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks*](https://arxiv.org/abs/1603.05279). ECCV 2016.
3. Li, Zhang, Liu. [*Ternary Weight Networks*](https://arxiv.org/abs/1605.04711). 2016.
4. Zhu, Han, Mao, Dally. [*Trained Ternary Quantization*](https://arxiv.org/abs/1612.01064). ICLR 2017.

**Quantized and 1-bit language models**
5. Dettmers, Lewis, Belkada, Zettlemoyer. [*LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*](https://arxiv.org/abs/2208.07339). NeurIPS 2022.
6. Frantar, Ashkboos, Hoefler, Alistarh. [*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*](https://arxiv.org/abs/2210.17323). ICLR 2023.
7. Wang et al. [*BitNet: Scaling 1-bit Transformers for Large Language Models*](https://arxiv.org/abs/2310.11453). 2023.
8. Ma et al. [*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/abs/2402.17764). 2024.
9. Ma et al. [*BitNet b1.58 2B4T Technical Report*](https://arxiv.org/abs/2504.12285). 2025.

**CPU kernels for 1-bit and ternary inference**
10. Wang, Zhou, Song et al. [*1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs*](https://arxiv.org/abs/2410.16144). 2024.
11. Wang, Zhou, Song et al. [*Bitnet.cpp: Efficient Edge Inference for Ternary LLMs*](https://arxiv.org/abs/2502.11880). 2025.
12. Wei et al. [*T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge*](https://arxiv.org/abs/2407.00088). EuroSys 2025.
13. Li, Yin, Wang et al. [*Vec-LUT: Vector Table Lookup for Parallel Ultra-Low-Bit LLM Inference*](https://arxiv.org/abs/2512.06443). 2025.

**Verifiable and attested inference**
14. Sun, Li, Zhang. [*zkLLM: Zero Knowledge Proofs for Large Language Models*](https://arxiv.org/abs/2404.16109). CCS 2024.
15. Conway, So, Yu et al. [*opML: Optimistic Machine Learning on Blockchain*](https://arxiv.org/abs/2401.17555). 2024.
16. Chan, Ding, Chen et al. [*Optimistic TEE-Rollups*](https://arxiv.org/abs/2512.20176). 2025.
17. Alves, Patankar, Pereira et al. [*EigenAI: Deterministic Inference, Verifiable Results*](https://arxiv.org/abs/2602.00182). 2026.
18. Cankaya. [*Bit-Exact AI Inference Verification Without Performance Tradeoffs*](https://arxiv.org/abs/2606.00279). 2026 (vLLM and Hugging Face on GPU, not llama.cpp).
19. Gailly et al. [*DeepProve: Verifiable End-to-End LLM Inference*](https://eprint.iacr.org/2026/1112). IACR ePrint 2026/1112.
20. Yang, Ren, Xu, Zhang et al. [*OpenLLM: Modular and Scalable zkSNARKs for Verifiable LLM Inference*](https://eprint.iacr.org/2026/1578). IACR ePrint 2026/1578.
21. Gong, Liu, Li. [*Hollow-LLM Attack*](https://arxiv.org/abs/2607.28884). 2026.
22. Zhou, Zhao, Wang et al. [*OpenPCC: Open and Confidential LLM Serving on Commodity TEEs*](https://arxiv.org/abs/2606.11145). 2026.

**Retrieval, embedding and commitments used by bankml**
23. Cormack, Clarke, Büttcher. [*Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). SIGIR 2009. [doi:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114) (the DOI refuses scripts; the author's PDF resolves).
24. Chen, Xiao, Zhang et al. [*M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*](https://arxiv.org/abs/2402.03216) (BGE-M3). 2024.
25. Laurie, Langley, Kasper. [*RFC 6962: Certificate Transparency*](https://www.rfc-editor.org/rfc/rfc6962). 2013.
26. Laurie, Messeri, Stradling. [*RFC 9162: Certificate Transparency Version 2.0*](https://www.rfc-editor.org/rfc/rfc9162). 2021.

*Unverified details, stated as found:*
- The author lists of papers 3, 4, 12, 25 and 26 are as commonly cited, not re-fetched.
- The venues of papers 12 and 14 come from secondary pages.
- Every project's own speed claims are the project's, not reproduced here.
