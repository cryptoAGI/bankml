# Changelog

## 0.0.8 — 2026-09-28

Savante's memory, made searchable, measurable and provable, with the documents brought up to date. No Rust code
changed. Record: `testing/results/0.0.8.txt`.

### Added
- **`.history` tab with a ragebar.** Every exchange, newest first: its time, session, time to first token, response
  time, tokens, and whether the answer matches its receipt. Before this it showed nothing until a button was pressed.
  The search bar (the GATERAGE ragebar's look) ranks exchanges as you type, using mindX's RAGE `rage.py` when present
  (`RAGE_PATH`) and the same BM25 built in otherwise.
- **Responses tab**: ⤒ first, ▲ previous, ▼ next, ⤓ latest through every answer. **📋 copy** puts the answer on the
  clipboard, **➕ save to .memory** keeps it, and **🔏 proof** gives its inclusion proof.
- **`.memory`** (`savante.memory`, JSONL, outside the canon): notes typed or saved from responses, with their source.
  A side-panel switch appends them to the system prompt (newest first, 2,400 characters at most), labelled as the
  operator's notes, not evidence. The footer names how many went in.
- **Metrics tab**, computed from `.history`: time to first token, response time, prefill and writing speed (n, median,
  p90, mean, min, max), a bar per exchange, receipt-hash agreement, the last 25 exchanges, and each row's timing source
  (the press of Send, or the receipt for older records).
- **Proof of data without the data.** Each `.history`/`.memory` line's sha256 is a leaf of a Merkle tree, and the file
  has a sha256 and a CIDv1 (raw, sha2-256, base32: the construction of Savante's iNFT ledger; equal to mindX
  `rage.cid_v1_raw`). The view page shows only these commitments. An inclusion proof lets one exchange be checked
  against the root without disclosing the rest.
- **`testing/test_ui.py`**: 16 offline checks of the UI's data layer. They cover the CIDs, commitments, every
  inclusion proof, tamper and cross-record failure, the search, the metrics and `.memory`, and the view server
  (commitments present, content absent, traversal 404, POST 405). Run in CI and in the gate.

### Changed
- **TECHNICAL.md** brought to 0.0.8:
  - The Abstract now includes the threads, the floor and P0.
  - The Thesis gains the principles stated on 2026-09-28, quoted and dated: incremental improvement with the oracle
    in the loop, the machine at hand, and proof of data while keeping the data.
  - Contributions 1, 5 and 8 are updated, and 12–13 are new.
  - New §III.6 (proof of data) and §IV.5 (the floor, and the five rejected variants).
  - §IV.4 no longer says bankml cannot answer.
  - Future work drops the 1-bit layout port (tried and rejected) and adds connectors, THOT and custom agents.
- **README**: status badge and phase table (P0 done; P2 with the 8B oracle and threads; UI row); the budget numbers
  (0.23–0.25 s, about 4 tok/s); the Savante section covers the new tabs and proofs.
- **usage.md**: the new tabs, `.memory`, and §8a on proofs (with a hand check in Python).
- View: the live log is capped at 60 % of the window, so the other panels stay in reach.

## 0.0.7 — 2026-09-28

Savante's UI, made to be watched and used: a LAN view, a professional look, response times, layout you can arrange,
and a guide. No Rust code changed (the kernels and their 0.0.6 gate records stand). Record: `testing/results/0.0.7.txt`.

### Added
- **View mode on the LAN: `ui/view.py`**, the Python standard library, not Gradio. The installed Gradio 3.37 has
  path-traversal bugs that let a client read files from the host (e.g. CVE-2023-51449, fixed in 4.11), so it stays
  on loopback. The view server has four fixed GET routes: the page; `/api/state`; `/api/result?name=`, for a name the
  results directory lists; and `/savante.png`. Every other path gets 404, POST gets 405, the page renders all data as
  text, and a strict Content-Security-Policy applies. It shows the live testing, the release records, CI, the
  laptop's load and swap, `bankml serve`'s verification and Savante's office and ledger. Light and dark themes.
- **Modular layout.** In view mode every panel can be dragged by its title to a new position and resized from its
  corner; the layout is kept per browser, with a *reset layout* button. In interact mode the chat and side panels
  resize, and the side panel drags to either side of the chat.
- **A response timer** that starts at the press of Send and ticks every second: *reading the prompt (prefill)*, then
  *writing · first token at N s*. Each answer's footer reads `⏱ sent HH:MM:SS · first token N s · answered in N s`.
- **Response times in `.history`**: every record carries `sent_at` and `answered_at` (ISO 8601 with milliseconds and
  offset), `first_token_s` and `response_s`, next to the receipt.
- **`usage.md`**, the full guide: what runs where, setup, verifying the model, `serve`, both modes, `.history`,
  receipts and how to check an answer, the canon and its ledger, testing, troubleshooting, and a reference of
  commands, ports and variables. The README gains a four-step *Using Savante*.

### Changed
- Interact mode uses one light professional theme: bordered panels and forced text contrast over Gradio's styles.
  The `bankml serve` panel was a raw JSON block that overflowed the side column; it is now a compact card that wraps
  inside its border.
- The Gradio queue runs 4 events at once, so the timer ticks while an answer streams.

## 0.0.6 — 2026-09-28

bankml answers on this laptop. The answers come through the reference engine (P0), behind the gate, with a receipt,
in a Savante UI that anyone can watch.

### Added
- **`bankml serve`** (P0, `serve.rs`, std only) is a loopback HTTP gateway in front of llama.cpp b11192's
  `llama-server`:
  - It starts only after `verify` passes (guard, then the sha256 pin to `FORK.json`).
  - The upstream must be serving the verified file. Either bankml launches it (`--spawn LLAMA_SERVER`), or a running
    server's `/props` must name the same canonical path. Otherwise it refuses.
  - Every `/v1/chat/completions` answer, streamed or not, carries a `bankml_receipt`. The receipt holds the version,
    the engine, the model sha256, the guard verdict, the tokens, the time to first token, the wall time and the
    **sha256 of the answer text**, so a client can check that what it shows is what the verified model wrote.
  - `GET /bankml` reports the verification.
  - What it is not: bankml's own forward pass (P3). The arithmetic is ggml's.
- **`ui/savante.py`** is the Savante chat in Gradio (3.x or newer), built from the Hugging Face template
  PYTHAI/savante. It has two modes:
  - **interact** (the operator): the chat, the `.prompt` picker, `.history` and the offline verifier. The `.prompt`
    choices are the persona's `system_prompt` (canon, ledgered, the default), `sAGI.prompt` (canon facet, ledgered) or
    the Space's `Savante.prompt` (not ledgered), each shown with its sha256. `.history` is a JSONL file outside the
    canon, and the last session reloads on start. Each answer is labelled a draft and carries its receipt.
  - **view** (anyone watching, read-only): the live testing log (`testing/live.log`, refreshed every 2 s), every
    release's gate record, CI status, the laptop's load, memory and swap (so a disturbed measurement shows as one),
    and Savante's office and ledger. It has no chat, no `.history`, and runs no commands.
- **iNFT compatibility.** Savante's canon is read and never written, not even a Python cache. At start the UI re-hashes
  every file that `savante.commitments.json` commits to: the nine artefacts, the card, the image and the thot bundle.
  If the persona does not verify, the UI refuses to speak as Savante. The Integrity tab shows the ledger, and
  `bind/savante_verify.py` runs from a button. Nothing mints; the card's status stays `not_yet_minted`.
- 2 new unit tests (the JSON reader and HTTP bodies) and 2 end-to-end tests of `serve` against a mock llama-server:
  the receipt and the answer hash, refusal of a wrong pin, and refusal of an upstream serving a different file.

### Measured on the laptop (Ryzen 3 3200U), the real canon and model
- `bankml serve` verified Bonsai-8B Q1_0 (`284a335a…`, the fork's pin) and the running llama-server serving it.
- A Savante turn under the persona's system prompt: 316 prompt + 28 completion tokens in 125 s, with the first token
  at 113 s. Prefill runs at 2.8 tok/s in llama.cpp. The answer's sha256 matched its receipt.
- The canon ledger verifies 12/12; `savante_verify.py` exits 0 (APPROVE); `~/savante` is unchanged afterwards.

## 0.0.5 — 2026-09-28

The third performance release, on the ternary kernel, with the testing shown live. **No kernel changed**: three
experiments were bit-exact, and none was reliably faster. Gate record: `testing/results/0.0.5.txt`. Every oracle is
bit-exact. Ternary stays at 9.2–9.6× ggml over a whole token, and 1-bit at parity.

### Measured and not adopted (bit-exact, not reliably faster), code in `testing/experiments/`
- **`Q2Packed`**: the ternary weights repacked once at load time, as llama.cpp's `repack.cpp` does for other types.
  Each quad of blocks becomes 64 contiguous code bytes followed by its four f16 scales: the same 72 bytes, with the
  arithmetic unchanged. It measured 1.00–1.075× in the first run and 0.90–1.05× in the release gate, on the real
  ffn_gate, ffn_down and output tensors. That is within noise.
- **Two-row decode tile**, loading the activation once for two rows: **0.80×**. Register pressure costs more than the
  shared loads save.
- **Software prefetch** of the weight row, 256–1024 bytes ahead: within ±3 % at 1 and 3 threads. The hardware
  prefetcher already follows a sequential stream.

### Added
- **`testing/live.sh`** and a live log: the gate and every experiment append to `testing/live.log`, which the UI's
  view mode shows as it runs (branch `ui`, next release).

### What the oracles and the stopwatch say after three performance releases
At 3 threads the ternary kernel moves about 9 GB/s against a 17 GB/s floor. Going from 1 to 3 threads gives 1.8×, and
the laptop has two physical cores. So it is bound by compute per core, not by memory. It already does one multiply
per weight, and moving the bytes around changes nothing measurable. On this Zen+ core, both kernels are at their
instruction-throughput limit for bit-exact results. The gains still available lie elsewhere:
- the forward pass (P3), which turns the 9.5× matmul lead into tokens;
- a Zen3 or AVX-512 core (the node).

## 0.0.4 — 2026-09-28

Where the time goes, and a faster 1-bit prefill. Gate record: `testing/results/0.0.4.txt`. Every oracle is bit-exact.
Two measurements in the gate were disturbed by other load on the laptop (swap full), so the record adds a re-run.

### Added
- **`bench_memory_floor`** (`par.rs`) measures streaming read bandwidth on the laptop at 1–4 threads: **14.9–17.2
  GB/s**. Streaming one token's weights therefore takes at least **0.12–0.14 s** (ternary) and **0.06–0.07 s** (1-bit).
  Against that floor, bankml's ternary matmuls (0.23–0.27 s at 3 threads) are about 2× above it, and the 1-bit ones
  (0.34–0.37 s) about **5.5× above it**. Both kernels are compute-bound, not memory-bound, which says where the
  remaining work is.
- **`q1_0::mat_mul_act` / `mat_mul_act_par`**: a 1-bit prefill over prepared `Q8Act` columns, using the selection
  kernel in a 1×4 tile. The ±1 expansion is shared by the four columns, and the scales are already f32, so nothing is
  converted once per column. A column that holds q = −128 falls back to the wrapping kernel. Measured against ggml's
  per-pair kernel (1024×4096 × 32 columns): **1.27–1.33× median** (8.25 against 10.09–10.28 ns per block·column,
  min). The q8-byte tile it replaces measured 1.08–1.18×. Bit-exact against the scalar model of ggml on 1,500 random
  cases, and against ggml itself in `ab_vs_ggml`.

### Measured and not adopted (the oracle said same bits; the stopwatch said no)
- **Two 1-bit blocks per iteration**, with their FMA chains interleaved: 1.00× on the GEMV, 1.05× on an L1-resident
  row. The kernel is not bound by latency.
- **A "pair-order" 1-bit kernel** that replaces a multiply (`madd`) with a 128-bit add, on the theory that Zen+'s
  single integer-multiply pipe is the limit. It was bit-exact and **0.84×** on decode: the cross-lane extract and the
  widening cost more than the multiply they save. The source is in `testing/experiments/q1_0_pair_order.rs` for anyone
  who wants to try it on another core.
- (Already on record from 0.0.1: 2- and 4-row decode tiles, 0.81–0.99×.)

1-bit decode stays at parity with ggml (0.95–1.03× over the whole token). Every bit-exact variant tried so far runs at
about ggml's speed on this core.

### Testing
- The gate now also runs `bench_q1_0_prefill_act` and `bench_memory_floor`.
- `testing/experiments/` holds the code of measured-and-rejected kernels.

## 0.0.3 — 2026-09-28

Threads, and more oracle evidence. Each speed figure below was measured with the oracles passing on the same code.
The full record is in `testing/results/0.0.3.txt`, written by the new release gate.

### Added
- **`par::Pool`**: a zero-dependency persistent thread pool. The caller's thread is worker 0. Each matmul wakes the
  pool instead of spawning threads: 12.6 µs per call against 88.8 µs for `thread::scope`, at 3 threads on the dev box.
  That is about 3 ms per token instead of 22 ms. The row scheduler hands out 16-row chunks from an atomic counter.
- **`mat_vec_par` / `mat_mul_par`** for `Q1_0` and `Q2_0`. Every row goes through the same single-thread kernel, so
  the output bits do not depend on the thread count. Unit tests check this at 1–4 threads, and the whole-model budgets
  check it against ggml.
- **An oracle on the real 8B 1-bit model**: `oracle_ggml_b11192_real_bonsai_8b_q1_0`. All 254 `Q1_0` tensors of
  Bonsai-8B (8,188,239,872 weights) dequantize bit-exact, 762/762 q8_0 rows are byte-exact, and 762/762 dot products
  are bit-exact against ggml AVX2 and generic. 0.0.2 covered only the 1.7B file for 1-bit.
- **`decode_budget_q1_0`**: one token's worth of all 253 `Q1_0` matmuls of Bonsai-8B, against ggml's kernel on the same
  pool and scheduler, at 1–4 threads.
- **`testing/release_gate.sh`** runs the unit tests, clippy, both guard checks, every oracle, both A/Bs and both
  budgets, and writes `testing/results/<version>.txt`. A kernel change counts only if every oracle still passes.

### Measured (dev box, Ryzen 3 3200U, 2 cores / 4 threads; all 253 matmuls of one token; min s/token over two runs)
| threads | ternary: ggml → bankml | 1-bit: ggml → bankml |
|---:|---:|---:|
| 1 | 4.22–4.65 → **0.45–0.47** (9.4–9.9×) | 0.62–0.67 → 0.61–0.62 (1.00–1.07×) |
| 2 | 2.65 → **0.31** (8.7×) | 0.45–0.46 → 0.43–0.44 (0.93–1.13×) |
| 3 | 2.27–2.36 → **0.23–0.25** (9.5–9.9×) | 0.34–0.35 → 0.34–0.35 (0.98–1.00×) |
| 4 | 2.11 → **0.23** (9.4×) | 0.34–0.35 → 0.36 (0.96–0.97×) |

bankml's ternary matmuls now take less time per token than ggml's **1-bit** ones: 0.23–0.25 s against 0.34–0.35 s at 3
threads, although the ternary file is twice the size. The matmul-only ceiling for ternary rises from 2.77 tok/s (0.0.2,
which spawned threads for each call) to 4.0–4.4 tok/s. The 1-bit kernel stays at parity with ggml. The 1-thread medians
are noisy (ternary 0.62 s against 0.47 s min): the model does not fit in page cache alongside everything else in 5.8
GB of RAM.

### Testing
- A new **`testing/`** folder holds every test that lives outside the modules. It contains the release gate, the
  oracle generator, the guard harnesses (moved from `tools/`) and **`testing/cli.rs`**, a new end-to-end suite that
  runs the `bankml` binary. The suite covers verdicts and exit codes, a hostile header, pin and verify, and runs as a
  cargo integration test.
- **`testing/results/`** holds each release's record. `testing/README.md` lists every test and where it lives.

### Changed
- The ternary budget and the A/B harness use `par::Pool` for both engines instead of spawning threads per call. The
  ggml side runs on the same scheduler, so the A/B compares kernel with kernel.

## 0.0.2 — 2026-09-28

An audit of 0.0.1. The kernels are unchanged and re-proven: the oracles again pass on all 8,188,239,872 ternary and
1,719,904,256 1-bit weights against llama.cpp b11192. The fixes below all concern untrusted input, or an API that let
safe code reach unsafe code with the wrong sizes.

### Fixed
- **The guard crashed on nested arrays.** GGUF arrays of arrays were parsed by recursion with no limit, so a 24 MB
  header aborted `bankml guard` with a stack overflow (exit 134) instead of refusing the file. Nesting beyond
  `MAX_ARRAY_DEPTH` (64) now refuses, with the reason given.
- **The guard's KV size could overflow.** `kv_f16_bytes_per_token` multiplied header integers in `i128` without
  checking. In release builds the value wrapped with no error and the verdict was `play`; debug builds panicked. An
  overflow now refuses, with the reason given. The Python guard, which uses big integers, still plays these files; the
  difference is listed in `gguf.rs` with the other deliberate differences.
- **Soundness: activation fields were public.** `q1_0::Q8Act` and `q2_0::Q8Act2` exposed `n`, `d`, `sum` and the
  other fields that the AVX2 kernels read through raw pointers. Safe code could set `n` beyond the buffers and cause
  an out-of-bounds read. The fields are now private, with read-only accessors (`n()`, `qs()`, `d()`).
- **Tensor spans were unchecked.** `Mmap::tensor` and `tensor_bytes` truncated a 128-bit element count to 64 bits
  and used unchecked arithmetic. `tensor_bytes` also allocated whatever size the header claimed before it read the
  file. A new function, `gguf::tensor_span`, checks the type, requires whole blocks per row and rejects overflow.
  `tensor_bytes` now checks the span against the file size before it allocates.
- **`bankml pin` with an unreadable `--fork` file** said "no sha256 record: unpinned, refused" (exit 2). It is an I/O
  error, and now says so (exit 1).

### Added
- `bankml::verify` and `bankml verify FILE --fork FORK.json [--json]` run the guard and then the pin as one gate: the
  check that P0 puts in front of every answer. `--json` prints the verdict, the model sha256 and the bankml version.
- `bankml version` and `bankml::VERSION`.
- CI (GitHub Actions) runs the build, the unit tests, `clippy -D warnings`, the Python guard suite and the Rust-vs-Python
  guard agreement check on every push.
- 4 new unit tests (31 in all): nested arrays, KV overflow, tensor spans, and `verify`.

### Changed
- The `Q8Act`/`Q8Act2` fields are no longer public (see above). This is an API break, allowed while the crate is
  `publish = false` and 0.0.x.

## 0.0.1 — 2026-09-26

First public cut: the P1 GGUF guard and sha256 pin, and the P2 `Q1_0` and `Q2_0_g64` kernels, all bit-exact against
llama.cpp b11192.
