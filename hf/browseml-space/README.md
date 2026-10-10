---
title: browseML
emoji: 🌐
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: bankML in your browser, verified, a receipt on every answer
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
tags:
  - webassembly
  - browser-inference
  - 1-bit
  - gguf
  - verified-inference
  - rust
models:
  - prism-ml/Bonsai-1.7B-gguf
---

# browseML — bankML in your browser

This page runs [bankML](https://github.com/cryptoAGI/bankml)'s own engine, compiled to WebAssembly, on your CPU.
Nothing to install, no account, and no server answers for you: the Space only serves files.

- **The model, verified before it speaks.** Bonsai-1.7B Q1_0 (248 MB) is downloaded once from
  `prism-ml/Bonsai-1.7B-gguf` at a pinned revision and kept in your browser's Cache Storage. The engine checks it as
  `bankml serve` does: the GGUF guard, then the sha256 of the whole file against its pin
  (`browseML/Bonsai-1.7B-Q1_0.gguf.FORK.json`). It does not answer from a file that fails either check.
- **A receipt on every answer, checked in the page.** The engine returns `serve`'s own `bankml_receipt`. The page
  recomputes the sha256 of the text it received and shows the verdict under the answer: ✓ when it matches, ✗ when
  it does not or when no receipt came.
- **Three things kept apart.** The answer is only the model's text. A note the page adds is labelled as the page's.
  The delivery line gives the engine's own token counts, the time to the first token, tokens per second, where it
  ran and the receipt's verdict. If the engine counts no completion tokens, the page says *No delivery from the
  model* rather than showing text. Only a delivered answer joins the conversation.

## Three builds, one engine

| file | target | used when |
|---|---|---|
| `browseml-mt-relaxed.wasm` | `wasm32-wasip1-threads`, SIMD + relaxed SIMD | the page is cross-origin isolated, the browser compiles relaxed SIMD (Chrome, Edge), and `relaxed_madd` is fused on this machine (checked when the engine starts) |
| `browseml-mt.wasm` | `wasm32-wasip1-threads`, SIMD | cross-origin isolated, otherwise (Firefox, or `relaxed_madd` not fused) |
| `browseml.wasm` | `wasm32-wasip1`, SIMD | not isolated, or no `SharedArrayBuffer`: one thread |

All three are exact, not approximate. The SIMD kernels match the scalar reference bit for bit. The relaxed build
is used only where its multiply-add is fused, which makes it the reference's `mul_add`; where it is not fused, the
page falls back to `browseml-mt.wasm`. The answers are the same with any build. Only the speed changes.

**The oracle.** `browseML/testing/oracle.mjs` runs the WebAssembly engine on llama-server b11192's recorded answers
(nine conversations, temperature 0.3, fixed seeds). Every text, finish reason and token count must be identical.
`browseml.wasm` gives 9 of 9, and `browseml-mt-relaxed.wasm` on 2 threads gives 9 of 9.

## Limits

- **Threads need cross-origin isolation.** This Space sends `cross-origin-opener-policy: same-origin` and
  `cross-origin-embedder-policy: credentialless` (above). Without them, the engine runs on one thread.
- **Safari runs on one thread.**
- **The model's first download is 248 MB.** The page asks before it starts, then shows the download's progress.
- **Memory.** A tab holds about twice the model while it loads, then the model once.
- **An answer cannot be stopped midway.** The engine runs until the turn ends or the answer reaches its token limit.
- **Speed is your CPU's.** On a 2-core laptop the engine writes about 1 to 3 tokens a second.

## The files

`index.html`, `style.css` and `app.js` are the page. `browseml.js`, `browseml-worker.js`, `browseml-thread.js`,
`browseml-wasi.js` and `browseML/*.FORK.json` are copies of the files in
[PYTHAI/bankml](https://huggingface.co/spaces/PYTHAI/bankml). `COPIED.json` records their sha256, so a copy that
drifts from its source shows up. The three `.wasm` files are built by `tools/browseml.sh` and copied in by
`tools/browseml-space.sh`, which prints their sha256. Source of record: `hf/browseml-space/` in
[github.com/cryptoAGI/bankml](https://github.com/cryptoAGI/bankml).

## More bankML

- [PYTHAI/bankml](https://huggingface.co/spaces/PYTHAI/bankml): the full Space. It offers bankML in this browser,
  your own `bankml serve`, or a Hugging Face provider (labelled as not bankML), plus diagnostics, FAQ and logs.
- [PYTHAI/bankML-UIF](https://huggingface.co/spaces/PYTHAI/bankML-UIF): the experimental edition.
- [github.com/cryptoAGI/bankml](https://github.com/cryptoAGI/bankml): the source. The runtime for 1-bit and ternary
  GGUF models is bit-exact against llama.cpp b11192.

Licence: bankML is `MIT OR Apache-2.0`, at your choice. This Space's files are under either licence, and the Hub
lists it as `mit`. Bonsai-1.7B is Apache-2.0 (prism-ml).
