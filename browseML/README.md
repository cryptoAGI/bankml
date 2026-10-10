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
| `hf/space/browseml-worker.js` | runs the engine in a Web Worker, so the page stays responsive. On a cross-origin-isolated page it chooses the fastest threaded build that compiles here and is exact here (below) and starts the pool's threads first |
| `hf/space/browseml-thread.js` | one thread of the engine's pool: a Web Worker running the same module on the same shared memory (`wasi.thread-spawn` → `wasi_thread_start`) |
| `hf/space/browseml.js` | downloads the model with progress, caches it, loads it (one load at a time) and chats with it |
| `hf/space/browseML/*.FORK.json` | the model's pin |
| `hf/space/bankml-chat.js`, `space-dashboard.js` | the Space's engine for the ultimate input field, the activity line, and this computer's controls and diagnostics |
| `testing/oracle.mjs`, `testing/bench.mjs` | exactness against llama-server's record; timing at any thread count, with the pool on Node worker threads |

### Three builds, one engine

| file | target | where the page uses it |
|---|---|---|
| `browseml-mt-relaxed.wasm` | `wasm32-wasip1-threads`, SIMD + relaxed SIMD | cross-origin isolated, the browser compiles relaxed SIMD (Chrome, Edge) and `relaxed_madd` is fused on this machine |
| `browseml-mt.wasm` | `wasm32-wasip1-threads`, SIMD | cross-origin isolated, otherwise (Firefox; or `relaxed_madd` not fused) |
| `browseml.wasm` | `wasm32-wasip1`, SIMD | not isolated, or no `SharedArrayBuffer`: one thread |

The Space sends `cross-origin-opener-policy: same-origin` and `cross-origin-embedder-policy: credentialless`
(`custom_headers` in its README), and huggingface.co embeds it with `allow="cross-origin-isolated"`, so threads work
both on the Space page and at pythai-bankml.static.hf.space. `credentialless` rather than `require-corp`, because the
page loads scripts from deltaverse.pythai.net and jsDelivr.

**Threads.** The engine's pool (`par::Pool`, whose results do not depend on the thread count) runs on Web Workers
sharing the module's memory. A thread that waits cannot start a worker, so the page creates the pool's workers
before the engine starts; `wasi.thread-spawn` then hands each one a thread. The WASI shim copies out of shared
memory where `TextDecoder` and `getRandomValues` refuse a `SharedArrayBuffer` view.

**Relaxed SIMD, exactly.** `f32x4.relaxed_madd` may be fused or not, by the implementation; fused, it *is* the
reference's `mul_add`. So the relaxed build checks at open (`q1_0::relaxed_madd_is_fused`:
(1 + 2⁻²³)·(1 − 2⁻²³) − 1 is −2⁻⁴⁶ fused and 0 unfused) and refuses to open where it is not fused; the page then
loads `browseml-mt.wasm`. Its kernel (`q1_0::relaxed_core`):
- **Sign sums:** computed as llama.cpp's AVX2 "sel" kernel does, Σ±q = 2·Σ₊q − Σq. Σ₊q comes from
  `relaxed_dot_i8x16_i7x16` against the weights' bits as 0 or 1, deterministic everywhere because the second operand
  never has its top bit set. (With ±1 it is not: V8 on x86 reads that operand as unsigned.) Σq is precomputed per
  activation. This applies only to activations without a −128, where ggml's i8 negation wraps; those keep
  `simd128_core`.
- **Inner multiply-adds:** `d1·s` is exact in f32 (11 + 10 significant bits), so they are the same bits fused or not.
- **Outer `d0·ab + acc`:** one fused instruction instead of the f64 emulation.

The engine needed six small changes for WebAssembly, and none of them changes the native build:
- **`gguf::Mmap`** reads the file into memory where there is no `mmap`, one shared copy per path.
- **`serve::ident`** has a non-unix form.
- **The Vulkan loader** reports "no card" on WebAssembly.
- **The prompt cache's 8 GiB default** fits a 32-bit `usize`.
- **An unseeded request's seed** comes from the clock alone, because WebAssembly has no process id.
- **Rope's `cosf` and `sinf` are glibc's algorithm** (`bankML/sincosf.rs`, ported from ARM's optimized-routines,
  which glibc adopted). The reference builds its rope cache with glibc's functions. Rust's WebAssembly libm
  differs from them in the last bit on about 1 % of the rope's angles: 1,987 cosines and 2,917 sines out of 262,144,
  for positions 0–4095. That was enough to part one of the oracle's nine conversations from the reference late in
  the answer. The port matches glibc's x86-64 FMA build on all 2³² inputs (`sincosf_matches_glibc_on_every_float`).
  `exp`, `ln` and `powf` already agree bit for bit.

Q1_0's dot product has a WebAssembly SIMD kernel (`q1_0::simd128_core`, used by `vec_dot` and the forward pass's
`vec_dot_act`) that is bit-identical to the scalar reference:
- **The integer sums are vectorised.** Signs are applied as `(q ^ m) − m`, then two pairwise widening adds.
- **The inner multiply-adds are plain multiply then add.** The q8 scale is f16-exact (11 bits) and each lane sum
  is an integer of at most 10 bits, so their product is exact in f32. A fused and an unfused multiply-add are then
  the same bits.
- **The outer fused multiply-adds go through f64 SIMD.** A product of two f32 is exact in f64; when the f64 sum
  is exact (TwoSum error zero), one rounding to f32 is the fused result. The rare inexact lanes that land on an
  f32 halfway point, or are subnormal or non-finite, use the scalar `mul_add`. WebAssembly has no deterministic
  SIMD FMA, and the software `fmaf` was a third of the time.

The **f16 paths and the tiled attention kernel** have WebAssembly SIMD too (`bankML/wasm_simd.rs`), each bit-identical
to its scalar definition:
- **`f16::vec_dot`** (the KV cache's dot products) keeps the reference's four 8-lane accumulators, as eight f32x4, and
  its reduction order. Both factors are f16 (11 significant bits), so each product is exact in f32 and the fused
  multiply-add is a multiply then an add.
- **`f16::widen`** widens four f16 at a time exactly (the bits shifted into place, scaled by 2¹¹²; infinities and NaNs
  set apart); `f16::mad` and `f16::scale` (the reference kernel's V accumulator) use it, then round each lane to f16 as
  the scalar path does.
- **The tiled kernel's score chains and V update** run four lanes at a time through `wasm_simd::fmadd`: the fused
  `relaxed_madd` in the relaxed build, otherwise the f64 route above, with the scalar `mul_add` for the rare lanes
  where rounding twice could differ.

`tools/browseml.sh test` runs these kernels' unit tests, and `q1_0`'s, on `wasm32-wasip1` under Node's WASI
(`testing/wasi-run.mjs`), with SIMD and with relaxed SIMD: all 65,536 f16 values through `widen4`, and about a million
`fmadd` lanes against `mul_add`. Those lanes include sums that land on an f32 midpoint after the f64 rounding, which is
exactly where rounding twice goes wrong; with the fix removed, the test fails.

### Measured

Bonsai-1.7B Q1_0 on a Ryzen 3 3200U (2 cores, 4 threads), in Node (`testing/bench.mjs`: a 20-token prompt, 48 tokens
written; the machine otherwise idle):

| build | threads | prompt | writing |
|---|---|---|---|
| scalar reference, first build | 1 | ~0.2 tok/s | ~0.2 tok/s |
| `browseml.wasm` (SIMD) | 1 | 1.43 tok/s | 1.11 tok/s |
| `browseml-mt.wasm` | 2 | 2.59 | 1.57 |
| `browseml-mt-relaxed.wasm` | 2 | 3.53 | 2.19 |
| `browseml-mt.wasm` | 4 | 2.96 | 1.91 |
| `browseml-mt-relaxed.wasm` | 4 | **5.16** | **2.71** |

The f16 and attention SIMD (`wasm_simd.rs`), measured old against new, interleaved (3 rounds in both orders, load
average 1.4–2.3): writing on 2 threads with relaxed SIMD, median 2.76 against 2.31 tok/s (+19 %, ahead in 6 of 6);
on one thread, 1.62 against 1.52 (+6 %, ahead in 5 of 6, at the edge of the noise). Reading a 20-token prompt
did not change beyond the noise: attention is a small share at that length.

Every build gives the oracle's answers, the same text: `browseml.wasm` 9 of 9 (`testing/oracle.mjs`),
`browseml-mt-relaxed.wasm` on 2 threads 9 of 9 (`testing/bench.mjs … oracle.jsonl 9`), and through 4 threads in
Chrome the first three likewise.

### Where the time goes, and what is next

From V8's profiles of every thread (`node --cpu-prof … testing/bench.mjs`):
- **The 1-bit kernel** (`vec_dot_act`) is 73–79 % of each thread.
- **Serial parts** — attention (`attend_head_partial`, 5 %), the f16 dot products, the pool's dispatch — leave the
  other threads idle for about a fifth of the time.
- **This laptop has 2 physical cores.** Beyond 2 threads, its hyperthreads share the SIMD units, so a visitor with
  more cores gains more from threads.

Next:
- spread attention across the pool, as the native engine's split-KV attention does;
- read the activation's signs once per 128 weights instead of once per 32.

## What a visitor sees while it works

Nothing is a silent wait (`bankml-chat.js`):
- **An activity line** at the top of the page, with a bar wherever it can measure:
  - the download, with MB, MB/s and time left;
  - verification, with a running clock;
  - reading the prompt, with elapsed time against an estimate from the last measured reading speed and the prompt's
    size; it says when it runs past the estimate;
  - writing, with the token count and speed.
- **The same line in the response window**, where it stays as the answer's timing record ("read 446 tokens in
  3 min 22 s (2.2 tok/s) · wrote 55 in 69 s (0.80 tok/s) · 4 threads").
- **The dashboard's "now" line** shows it too.
- **A second question during a download** waits for the download in progress instead of starting another.

## Build and check

```sh
tools/browseml.sh            # → hf/space/browseml.wasm, browseml-mt.wasm, browseml-mt-relaxed.wasm
tools/browseml.sh test       # the kernels' unit tests under Node's WASI, SIMD and relaxed SIMD (no model needed)
tools/browseml.sh oracle     # and the oracle on browseml.wasm (Node 20+, .models/Bonsai-1.7B-Q1_0.gguf and its FORK.json)
node browseML/testing/bench.mjs hf/space/browseml-mt-relaxed.wasm .models/Bonsai-1.7B-Q1_0.gguf FORK.json 4 \
  .models/oracle-forward/serve-Bonsai-1.7B-Q1_0.jsonl 9    # 4 threads: speed, and the 9 answers against the record
```

## Limits, stated

- **Threads need cross-origin isolation.** Safari, and pages that are not isolated, run on one thread.
- **Relaxed SIMD only where it is exact.** Firefox does not compile it yet, and a machine whose `relaxed_madd` is
  not fused gets the plain threaded build: same answers, slower.
- **Q1_0 models only, so far.** The page offers Bonsai-1.7B; the 8B models are for `bankml serve`.
- **Memory.** A tab holds about 2× the model while it loads, then the model once.
- **No stopping mid-answer.** An answer cannot be interrupted, because the engine runs to `max_tokens` or end of turn.
