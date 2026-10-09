# browseML — bankML in the browser

bankML's own engine, compiled to WebAssembly and run by a web page on the visitor's CPU. There's nothing to install
and no account to create, and no server does the work. Free for everyone, on Hugging Face's free static hosting.

- **Verified before it speaks.** The engine checks the model exactly as `bankml serve` does: the GGUF guard, then the
  sha256 of the whole file against its FORK.json. The model is Bonsai-1.7B Q1_0 (248 MB), downloaded once from
  `prism-ml/Bonsai-1.7B-gguf` at a pinned revision and kept in the browser's Cache Storage.
- **A receipt on every answer.** It's `serve`'s own `bankml_receipt`, and the page recomputes the answer's sha256.
- **Exact before it is fast.** `testing/oracle.mjs` runs the WebAssembly engine on llama-server b11192's recorded
  answers (`oracle-forward/serve-Bonsai-1.7B-Q1_0.jsonl`, sampled at temperature 0.3 with fixed seeds), and every
  text, finish reason and token count must be identical.

## How it fits together

| piece | what it does |
|---|---|
| `browseML/src/lib.rs` | the WebAssembly exports (`browseml_open`, `browseml_chat`). It only moves bytes across the boundary, as `capi/` does for C. |
| `hf/space/browseml-wasi.js` | the few WASI calls the engine makes, answered in memory: one read-only `/models` directory holding the downloaded file, the clocks, random bytes, environment and stderr. Anything else is refused, never faked. |
| `hf/space/browseml-worker.js` | runs the engine in a Web Worker, so the page stays responsive |
| `hf/space/browseml.js` | downloads the model with progress, caches it, loads it and chats with it |
| `hf/space/browseML/*.FORK.json` | the model's pin |

The engine needed four small changes for WebAssembly, and none of them changes the native build:
- **`gguf::Mmap`** reads the file into memory where there is no `mmap`, one shared copy per path.
- **`serve::ident`** has a non-unix form.
- **The Vulkan loader** reports "no card" on WebAssembly.
- **The prompt cache's 8 GiB default** fits a 32-bit `usize`.

Q1_0's dot product has a WebAssembly SIMD kernel that is bit-identical to the scalar reference:
- The integer sums are vectorised.
- The float operations stay the reference's own fused multiply-adds, one lane at a time.

## Build and check

```sh
tools/browseml.sh            # → hf/space/browseml.wasm
tools/browseml.sh oracle     # and the oracle (Node 20+, .models/Bonsai-1.7B-Q1_0.gguf and its FORK.json)
```

## Limits, stated

- **One thread.** Threads in the browser need cross-origin isolation, which a static Space cannot set by header.
- **Q1_0 models only, so far.** The page offers Bonsai-1.7B; the 8B models are for `bankml serve`.
- **Memory.** A tab holds about 2× the model while it loads, then the model once.
- **No stopping mid-answer.** An answer cannot be interrupted, because the engine runs to `max_tokens` or end of turn.
