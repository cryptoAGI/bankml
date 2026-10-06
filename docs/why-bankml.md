# Why bankML — what it is good for, why Rust, and where it is going

*Written 2026-10-06, while bankML 0.3.6 is the latest release, 0.3.7 is in its release gate, and 0.3.8 and 0.3.9
are being built on the way to the 0.4.0 milestone.*

This page is the short version. Each idea links to the page that explains it in full, and to the code that does it,
for when you want to go deeper.

---

## bankML in one paragraph

bankML runs language models on the computer you already have: a laptop, a small server, no graphics card needed. It
specialises in **1-bit and ternary models**, which store each weight in one or two bits instead of sixteen. An
8-billion-parameter model then fits in 1.2 GB (1-bit) or 2.3 GB (ternary). bankML gives **exactly the same answers as
llama.cpp**, the reference engine, and checks that it does, bit for bit, before it is allowed to be faster. Every
answer carries a **receipt** that says which model file produced it.
→ the full story: [README](../README.md) · the technical report: [TECHNICAL.md](TECHNICAL.md)

---

## The advantages

### 1. It is fast where it matters most
llama.cpp has no fast x86 code for ternary weights; it falls back to plain C. bankML wrote the missing kernel. Each
matrix is **9.4–10× faster**, and whole answers on the ternary model come out at **about 8× llama-server's speed**
(2.3–2.4 tokens/s against 0.30 on the same laptop). The ternary model gives the better answers of the two, so this
is the speed that counts.
→ [PERFORMANCE.md](PERFORMANCE.md) · the kernel: [modules/q2_0.md](modules/q2_0.md),
[`bankML/q2_0.rs`](../bankML/q2_0.rs)

### 2. It is exact, and proves it
Speed only counts if the numbers are the same. Every kernel, the tokenizer, the chat templates, the samplers and whole
conversations are checked against llama.cpp b11192's own compiled code. These checks are called **oracles**, and a
release ships only when every oracle passes in its **release gate**.
→ [oracles.md](oracles.md) · the gate: [`testing/release_gate.sh`](../testing/release_gate.sh) and its records in
[`testing/results/`](../testing/results/)

### 3. It refuses to answer from a model it cannot verify
Before a model answers, three gates run. A **guard** reads the file's header and refuses known-bad files. A **pin**
checks the file's sha256 against the record of where it came from. The **oracle**, run ahead of time, has proven the
arithmetic. Each answer then carries a **receipt** with the model's sha256 and the hashes of the request and the
answer.
→ [TECHNICAL.md §III.1](TECHNICAL.md#iii1-verified-response-three-gates-before-an-answer) ·
[usage.md §9, receipts](usage.md#9-receipts-and-how-to-check-an-answer) · the guard:
[`bankML/gguf.rs`](../bankML/gguf.rs) · the pin: [`bankML/sha256.rs`](../bankML/sha256.rs)

### 4. It is small and has nothing to install around it
bankML is one Rust crate with **zero external dependencies**. The GGUF reader, sha256, f16 maths, the thread pool,
the HTTP server and every kernel are written in the crate. You get one binary and nothing else to trust.
→ [modules/README.md](modules/README.md) (one page per source file) · [`Cargo.toml`](../Cargo.toml)

### 5. It speaks the languages your tools already use
- the **OpenAI** chat API (`/v1/chat/completions`), streamed or not
  → [modules/serve.md](modules/serve.md)
- the **Ollama** API (`/api/chat`, `/api/generate`, a registry of pinned models), so programs written for Ollama work
  unchanged → [OLLAMA.md](OLLAMA.md)
- a **C library** (`libbankml`) for C, C++, Python, Go or Swift programs → [CAPI.md](CAPI.md)
- **JSON mode and JSON schemas**: answers forced into a shape you choose → [modules/grammar.md](modules/grammar.md),
  [modules/schema.md](modules/schema.md)

### 6. It measures itself
bankML reports its own time to first token, tokens per second and, where the machine allows, the energy each token
costs. It also limits how much of a graphics card it may use.
→ [modules/metrics.md](modules/metrics.md) · [modules/console.md](modules/console.md) · the GPU:
[modules/gpu.md](modules/gpu.md)

---

## What people use it for

| use | how | read more |
|---|---|---|
| **A private assistant on your own machine** | Savante, a chat page on `127.0.0.1:7873`, answered by bankML | [usage.md §4](usage.md#4-talk-to-savante) · [playback.md](playback.md) |
| **The engine behind mindX** | mindX's default model server on its VPS since 2026-10-04, in place of Ollama | [OLLAMA.md](OLLAMA.md) |
| **A drop-in Ollama or OpenAI server** | `bankml serve --native --registry` | [usage.md §6a](usage.md#6a-ollamas-api) · [install.md](install.md) |
| **Your own model, verified** | `bankml convert` (safetensors → GGUF, byte-identical to llama.cpp's) and `bankml create` (a persona layer from a Modelfile) | [modules/convert.md](modules/convert.md) · [modules/create.md](modules/create.md) |
| **Structured answers for programs** | JSON mode, JSON schemas and grammars | [usage.md, JSON mode](usage.md#json-mode-and-grammars-033) |
| **Inference inside another program** | `libbankml` and `bankml.h` | [CAPI.md](CAPI.md) |
| **Agents with a provable history** | receipts, Merkle commitments of the conversation, THOT bundles and iNFTs | [usage.md §8a](usage.md#8a-proof-of-data-without-the-data) · [§8e](usage.md#8e-inft-mint-an-agent-load-one-from-a-token) |
| **Meaning search over notes** | bge-m3 embeddings fused with keyword search | [embedding.md](embedding.md) |

New here? Start with `./install.sh`: [usage.md §1](usage.md#1-install).

---

## How Rust helps bankML go fast and stay exact

Rust lets bankML write code as close to the hardware as C, while the compiler checks much of what C leaves to the
programmer. Here is how that plays out in practice:

- **Direct access to the CPU's vector instructions.** The kernels use AVX2 through `std::arch` and are compiled with
  `#[target_feature(enable = "avx2,fma,f16c")]`. bankML checks at run time which instructions the CPU has, so one
  binary runs everywhere and uses the fast path where it can. → [modules/q1_0.md](modules/q1_0.md),
  [modules/q2_0.md](modules/q2_0.md)
- **No hidden costs.** Rust has no garbage collector and no runtime pauses. Abstractions such as iterators and
  fixed-size chunks (`as_chunks`) compile down to plain loops, which lets the compiler drop bounds checks inside the
  hot kernels.
- **Safe threads, the same bits on any thread count.** The thread pool hands out rows from an atomic counter.
  Ownership rules guarantee that each output chunk has exactly one writer, so threads can never race, and the answer
  does not depend on how many threads ran. → [modules/par.md](modules/par.md), [`bankML/par.rs`](../bankML/par.rs)
- **`unsafe` kept in small, named places.** The few places that need it (the memory map, the vector intrinsics, the
  GPU loader) carry documented safety contracts behind safe functions. The rest of the code is checked by the
  compiler. → [modules/gguf.md](modules/gguf.md)
- **Zero-copy model loading.** The model file is memory-mapped, and the kernels read the weights straight from the
  mapping without copying them. → [`bankML/gguf.rs`](../bankML/gguf.rs)
- **Hostile files handled safely.** All arithmetic on file headers is checked (`checked_add`, `checked_mul`), so a
  malformed file is refused with a reason instead of crashing. → [modules/gguf.md](modules/gguf.md)
- **Whole-program optimisation.** Release builds use link-time optimisation, one code-generation unit and
  abort-on-panic, which gives a single small, fast binary. → [`Cargo.toml`](../Cargo.toml)
- **A pinned toolchain.** Rust 1.99 is pinned, so a build today gives the same machine code as a build next month,
  which an exactness project needs. → [`rust-toolchain.toml`](../rust-toolchain.toml)

---

## Where it is going

The road is laid out release by release in [TODO.md](TODO.md), and each step counts only once its oracle passes.

- **0.3.7** (in its release gate): llama-server's whole sampler chain, bankML measuring itself, a GPU limiter, and
  the bankML console.
- **0.3.8** (built): llama-server's behaviour at the context limit, saved and restored conversations (slots), a
  prompt cache so conversations taking turns each keep their context, and token probabilities (logprobs), streamed
  or not. → [modules/prompt_cache.md](modules/prompt_cache.md)
- **0.3.9** (in progress): a `q8_0` memory cache that halves the conversation memory, a faster grammar mask, and 1-bit
  speed at or above llama-server's.
- **0.4.0, the milestone:** native serving complete. bankML answers everything Savante and mindX ask of llama-server,
  and its own engine is chosen by default for both 1-bit and ternary models.
- **0.5.0, hardware:** ARM phones and tablets (NEON), newer x86 instructions (AVX-512), and more graphics cards.
- **1.0.0:** llama.cpp is needed only as the oracle in the gate, never at run time. bankML runs at llama.cpp's speed
  or better on every supported format, its interfaces are stable, and receipts are signed.

### Our hope for bankML

We hope bankML shows that a capable, private AI does not need a data centre or a graphics card: that an
eight-billion-parameter model can run on the laptop in front of you and give answers you can check. We want every
speed claim to come with proof that the numbers are the same, so that "faster" always means faster *and* correct.
We hope the ternary kernel goes back upstream to llama.cpp, so everyone benefits from it. We want bankML to become
the dependable, verifiable engine under Savante, mindX and the agents built on them. And we want it to stay small
enough that one person can read all of it.

→ the authors' own words: [the Thesis](TECHNICAL.md#thesis--professor-codephreak-and-gregory-l-magnusson) · where
bankML stands among other engines and papers: [research.md](research.md) · how it was built, step by step:
[BUILD_HISTORY.md](BUILD_HISTORY.md)
