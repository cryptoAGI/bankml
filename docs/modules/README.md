# bankML's source, module by module

One page per module of `bankML/` (and the C API in `capi/`). Each says what the module does and who calls it
(**Summary**), its public interface with real signatures (**Technical usage**), the oracles and tests that prove it
(**How it is verified**), why it is useful and how it keeps processing efficient (**Advantages and efficiency**), and
what it refuses or does not do yet (**Limitations**).

The code's own comments stay short and definitive; these pages are where the explanation lives. The build history —
how each phase was reached, with its evidence — is in [../BUILD_HISTORY.md](../BUILD_HISTORY.md).

## The path of a request

```
 client ──► serve.rs (OpenAI /v1)  ─┐
        └─► ollama.rs (Ollama /api) ─┼─► native.rs (registry, residency, the slot) ─► forward.rs (the graph)
 C API ──► capi (libbankml) ────────┘        │  gate: bankml.rs verify = gguf.rs guard + sha256.rs pin   │
                                             │  text: tokenizer.rs, chat.rs                              ▼
                                             └─ draw: sampler.rs, grammar.rs, schema.rs      q1_0 · q2_0 · f16 · gpu
                                                                                               (par.rs threads)
```

## Pages

| page | module | in one line |
|---|---|---|
| **The gate** | | |
| [bankml.md](bankml.md) | `bankml.rs` | the crate root: `verify` (guard, then pin), `Verified`, the ggml types, the log sink |
| [gguf.md](gguf.md) | `gguf.rs` | the GGUF header parser, the guard (`play \| refuse \| need_more`), `Mmap` |
| [sha256.md](sha256.md) | `sha256.rs` | FIPS 180-4 sha256 with SHA-NI, the `FORK.json` pin |
| **The arithmetic** | | |
| [q1_0.md](q1_0.md) | `q1_0.rs` | ggml's 1-bit format and its product, bit-exact, AVX2 |
| [q2_0.md](q2_0.md) | `q2_0.rs` | ggml's ternary format, and the kernel llama.cpp lacks on x86 (~9–10×) |
| [f16.md](f16.md) | `f16.rs` | ggml's two F16 products, chosen by shape as ggml chooses them |
| [par.md](par.md) | `par.rs` | the zero-dependency thread pool; bits independent of thread count |
| [gpu.md](gpu.md) | `gpu/` | Vulkan through dlopen, bankML's own SPIR-V, a verified card's share of the rows |
| [sys.md](sys.md) | `sys.rs` | machine facts and the process's usage (`bankml usage`, `/bankml/usage`): CPU, memory, GPU busy and memory, package power |
| [metrics.md](metrics.md) | `metrics.rs` | bankML's own measurements of its answers: TTFT, pp and tg tokens/s, energy per token (`/bankml/metrics`) |
| **The model** | | |
| [forward.md](forward.md) | `forward.rs` | the Qwen3 and Llama graphs, the KV cache, ggml's three attention kernels |
| [tokenizer.md](tokenizer.md) | `tokenizer.rs`, `unicode_letters.rs` | llama.cpp's tokenizer and pre-tokenizers |
| [chat.md](chat.md) | `chat.rs` | the chat templates, byte-identical, chosen by the template's sha |
| **The answer** | | |
| [sampler.md](sampler.md) | `sampler.rs` | llama-server's sampler chain, penalties included, same seed same tokens |
| [grammar.md](grammar.md) | `grammar.rs` | llama.cpp's GBNF engine, JSON mode and the content rule |
| [schema.md](schema.md) | `schema.rs` | `json_schema_to_grammar` and the chat wrapping, per template |
| **Serving** | | |
| [native.md](native.md) | `native.rs` | the engine: the slot and prompt cache, the registry, one resident model |
| [serve.md](serve.md) | `serve.rs` | the gateway: OpenAI's API, receipts, the loopback rules |
| [ollama.md](ollama.md) | `ollama.rs` | Ollama's API, natively |
| [create.md](create.md) | `create.rs` | `bankml create`: Modelfiles as verified layers over pinned models |
| [convert.md](convert.md) | `convert.rs` | `bankml convert`: safetensors → GGUF, byte-identical to llama.cpp's converter |
| [main.md](main.md) | `main.rs` | the `bankml` command line |
| [capi.md](capi.md) | `capi/` | `libbankml` and the printf-style log; the full reference is [../CAPI.md](../CAPI.md) |
| **Training** | | |
| [train.md](train.md) | `train/` | mindXtrain's author and score stages, identical to its Python |
| **The console** | | |
| [console.md](console.md) | `sAGI/console.py` | bankML as itself: four tabs (Interaction, Admin, Logging, Infotags), the measured SELF block |

Install and configure: [../install.md](../install.md). Use: [../usage.md](../usage.md). The checks:
[../oracles.md](../oracles.md). Speed: [../PERFORMANCE.md](../PERFORMANCE.md).
