# The C API (0.3.2)

bankML as a library a C, C++, Python (`ctypes`), Go (`cgo`) or Swift program links: **`libbankml.so`** and
**`libbankml.a`**, declared in one hand-written header, [`capi/include/bankml.h`](../capi/include/bankml.h). The
shape is llama.h's (open a model, run a chat, close it), with bankML's gate in front of it:

- a model opens only after the GGUF guard plays it and its sha256 equals its `FORK.json` record. This is the same
  `verify` that `bankml serve` runs;
- every answer carries the receipt that `bankml serve --native` gives: the model's sha256, the guard verdict, the
  counts, the times, and the sha256 of the answer and of the request.

The crate is `capi/` (package `bankml-capi`), a second member of the workspace. Its only dependency is the bankml
crate itself, by path, so **the workspace still has no external crate**. Its licence is `MIT OR Apache-2.0`, like the
core.

## Build and link

```sh
cargo build --release -p bankml-capi        # target/release/libbankml.so and target/release/libbankml.a
```

Shared:
```sh
cc app.c -I capi/include -L target/release -lbankml -Wl,-rpath,$PWD/target/release -o app
```

Static (one binary, no `.so` to ship; the system libraries are the ones `cargo rustc -p bankml-capi --crate-type
staticlib -- --print native-static-libs` lists):
```sh
cc app.c -I capi/include target/release/libbankml.a -lgcc_s -lutil -lrt -lpthread -lm -ldl -o app
```

`cargo build --release` at the root builds the `bankml` binary only, as before. The root package is the default
member, so the C library is built only when it is asked for. The toolchain is pinned in `rust-toolchain.toml` to
**Rust 1.99.0**, because `bankml_log` is a C-variadic function defined in Rust, which was stabilized in 1.99. rustup reads
that file and fetches 1.99 by itself; the machine's default toolchain is not changed.

## The API

```c
const char *bankml_version(void);
bankml_t   *bankml_open(const char *model_path, const char *fork_json_path, uint32_t n_ctx, char **err);
int         bankml_chat(bankml_t *h, const char *request_json, bankml_piece_cb cb, void *user, char **result_json);
void        bankml_close(bankml_t *h);
void        bankml_free(char *s);
void        bankml_set_log(bankml_log_cb cb, void *user);
void        bankml_log(int level, const char *fmt, ...);

typedef int  (*bankml_piece_cb)(const char *piece, size_t len, void *user);          /* return 0 to stop */
typedef void (*bankml_log_cb)(int level, const char *msg, size_t len, void *user);
```

### `bankml_open`

`bankml_open` runs the full verification before anything is mapped for inference:
1. the GGUF guard: it refuses fork-only types, legacy group-128 Q2_0 and Bonsai 2 on mainline;
2. the sha256 pin to the `FORK.json` at `fork_json_path`;
3. the file-identity check: the file must not change while it is being hashed;
4. the native check `serve --native` makes: Qwen3, Q1_0 or Q2_0_g64 weights, its own output matrix, and a chat
   template bankML renders.

On a refusal it returns NULL, with the reason in `*err` in `serve`'s words. Two examples:
- `refuse: Bonsai-1.7B-Q1_0.gguf: bankML's native forward pass does not play it: tied embeddings (no output.weight) …`
- `refuse: X.gguf has no sha256 record in FORK.json: unpinned, refused`

`n_ctx` is the context in tokens. 0 means 4096, `serve`'s default.

### `bankml_chat`

`request_json` is the chat request that `/v1/chat/completions` takes on `serve --native`:
- `messages`;
- the sampling keys: `temperature`, `top_k`, `top_p`, `min_p`, `min_keep` and `seed`. A key that is not set takes
  the model's GGUF default;
- `max_tokens` (or llama-server's `n_predict`);
- `stop`.

A sampler bankML does not reproduce is refused with a reason (`BANKML_E_REQUEST`), never approximated: the penalties,
DRY, typical-p, XTC, top-n-σ and dynamic temperature. `model` and `stream` are ignored: the handle names the model,
and the callback is the stream.

The answer streams to `cb` as whole UTF-8 pieces, each `len` bytes long and NUL-terminated. To stop, return 0. A NULL
`cb` is allowed.

`*result_json` receives the object that the non-streamed `/v1/chat/completions` returns:
- `choices`, `usage`, and llama-server's `timings.cache_n`;
- `bankml_receipt`, made by the same code that `serve` uses (`serve::NativeChat`, `serve::completion_json` and
  `serve::Tally`).

The receipt's `ttft_ms` is set, because the C API always streams. On an error, `*result_json` is
`{"error": {"code", "message"}}`.

The return code is one of these:

| code | meaning |
|---|---|
| `BANKML_OK` (0) | answered |
| `BANKML_E_ARG` (−1) | a NULL handle or request, or a request that is not UTF-8 |
| `BANKML_E_REQUEST` (−2) | the request is refused: not JSON, no messages, a sampler that is not reproduced, or a bad `stop` |
| `BANKML_E_CHANGED` (−3) | the model file changed since it was verified. There is no answer; open it again to verify it again |
| `BANKML_E_ENGINE` (−4) | the engine failed, for example because the prompt does not fit the context |
| `BANKML_E_PANIC` (−5) | a panic was caught (unwinding builds only; see below) |

The handle keeps one slot, as llama-server's `-np 1` does. The next request reuses the longest common prefix of the
tokens already in its KV cache. A conversation sent turn by turn therefore pays only for its new tokens, and
`timings.cache_n` says how many were reused.

### Ownership

- The strings bankML returns through `char **` (`err` and `result_json`) belong to the caller. Free each one once
  with `bankml_free`.
- `bankml_version()` returns a static string. Do not free it.
- Strings passed in are only read during the call.
- `bankml_close` releases the model's mapping and its KV cache. NULL is a no-op for `bankml_close` and
  `bankml_free`.

### Threads

- **One handle has one slot, so calls on it serialize.** A second thread's `bankml_chat` waits for the first one to
  finish.
- Different handles are independent. Each maps its model, so two handles on the same 8B file share the page cache
  but not their KV caches.
- `bankml_close` must not race a call on the same handle.
- The streaming callback runs on the calling thread. It must not close the handle.
- The log sink is process-wide. It may be called from any thread that calls into bankML.

### Panics and errors

Nothing unwinds into C. Every entry point checks its pointers and returns NULL or an error code with a message. Each
one also runs its work inside `catch_unwind`. That catches in an unwinding build, such as `cargo test`. The release
profile is built with `panic = "abort"`, so there a panic aborts the process, as a failed C `assert` does; it is never
undefined behaviour. The C API's own code returns errors rather than panicking (no `unwrap` on input, NULL checked, a NUL inside a
message replaced rather than refused). The engine underneath is not proven panic-free, so a panic there is a bug to
report.

## The log: a C-variadic function defined in Rust

```c
void bankml_set_log(bankml_log_cb cb, void *user);   /* NULL: standard error */
void bankml_log(int level, const char *fmt, ...);    /* BANKML_LOG_ERROR 0 · WARN 1 · INFO 2 · DEBUG 3 */
```

`bankml_log` is defined in Rust with Rust 1.99's C-variadic function definitions:

```rust
pub unsafe extern "C" fn bankml_log(level: c_int, fmt: *const c_char, mut args: ...)
```

Its formatter (`capi/src/printf.rs`) reads each argument with `args.next_arg::<T>()`, at exactly the C type that the
conversion names.

The library's own messages go to the same sink: a model verified, the GPU joining or not, a refusal. The core has a
small log hook for this (`bankml::log`, `bankml::set_log_sink`), which writes to standard error as before unless an
embedder installs a sink. The callback receives the message, its length in bytes (it may contain a NUL if a `%c`
printed one) and a terminating NUL.

**Supported**, byte-identical to glibc's `snprintf` (proven by the printf oracle below):

| conversion | lengths | notes |
|---|---|---|
| `%d %i` | none, `hh`, `h`, `l`, `ll`, `z` | `int`, `signed char`, `short`, `long`, `long long`, `ssize_t` |
| `%u %x %X` | none, `hh`, `h`, `l`, `ll`, `z` | `#` gives `0x`/`0X` before a non-zero value; `+` and space are ignored, as C ignores them |
| `%f %F` | none, `l` | precision (default 6), exact decimal expansion, ties to even; `inf`/`nan` (`INF`/`NAN`), a sign on `-nan` and `-0.0` |
| `%c` | none | width and `-` |
| `%s` | none | width, `-`, precision (reads at most that many bytes, so the string need not be NUL-terminated); NULL prints `(null)` |
| `%p` | none | `0x…`, NULL prints `(nil)`; width and `-` |
| `%%` | | |

The flags are `-`, `+`, space, `#` and `0`. A width or a precision is a number or `*` (an `int` argument; a negative
width means `-`, and a negative precision means none).

**Everything else is written literally, never guessed:**

- **An unknown conversion or length** (`%e %g %o %a %n %Lf %ls %jd`, …) is written `%<unsupported:SPEC>`, and **no
  argument is read for it**. Its type is unknown, so no later argument can be located either: every later conversion
  is written `%<skipped:SPEC>`. A `%%` still prints `%`.
- **A known conversion with flags it does not vouch for** is written `%<unsupported:SPEC>`. Examples: `%+s`, `%05c`,
  `%#p`, `%#d`, `%.3c`, and a width or precision over 65,536. Its argument has a known type, so it is read and
  dropped, and the rest of the format goes on.
- **`%n` never writes** through its argument.
- A `%` at the very end is written `%<incomplete:>`.

## The oracles (in the release gate)

- **`printf oracle`** (`testing/capi/printf_oracle.c`, compiled with the system `cc`). For every case, the message
  that `bankml_log` delivers to a sink installed with `bankml_set_log` must equal what libc `snprintf` writes for the
  same format and arguments. The comparison is byte for byte, with the length, so an embedded NUL counts. The cases
  cover:
  - integers at their limits in every length;
  - `%f` precision and rounding (ties, `%.0f` of 0.5/1.5/2.5, `%.1100f` of the smallest subnormal, `DBL_MAX`);
  - `inf` and `nan` with signs;
  - strings, including a precision-bounded unterminated buffer and NULL;
  - `%c` of 0 and 255, `%p`, `%%`;
  - widths, precisions and `*`.

  Then every unsupported case must give its exact marker and read no argument it cannot type, and `%n` must leave its
  target untouched. The Rust unit tests (`cargo test -p bankml-capi`) run the same comparison from Rust, calling the
  Rust-defined variadic function, plus a typed-queue model that fails if the formatter reads a type other than the
  one passed.
- **`capi chat oracle`** (`testing/capi/chat.c` + `capi_oracle.py --chat`). A C program embeds `libbankml` the way an
  application would. It has three cases:
  - **Ternary-Bonsai-8B.** `serve_oracle.py`'s three Savante conversations (9 turns) go through `bankml serve
    --native` `/v1/chat/completions`. Then the very same request bytes go through `bankml_chat` in a fresh process.
    These must be equal, turn by turn: the text, the streamed pieces joined, the prompt and completion counts, the
    prompt-cache reuse, the finish reason, and the receipt's `response_sha256`, `request_sha256` and `model_sha256`.
  - **Bonsai-8B Q1_0.** llama-server b11192's recorded conversations go through `bankml_chat`, and the same text and
    counts must come back, with `response_sha256` equal to the sha256 of llama-server's text. The gate's
    `serve_oracle --bankml` ties the same record to `serve --native`.
  - **Refusals.** Bonsai-1.7B is refused (tied embeddings, as `serve --native` refuses it), and so is a file that its
    `FORK.json` does not pin. The library's own log reaches the installed sink.

## Why it exists

This C API is the llama.h-shaped seam for embedding: a program that today links llama.cpp's `llama.h` can link
bankML instead and get the same answers (the oracles above), with the verification and the receipt built in. It is
also the path to the handheld targets (P5): an Android NDK or iOS app embeds a C library, not an HTTP server. What it
does not have yet is listed in [TODO.md](TODO.md): tokenizer and logits access, more than one slot, and a stable ABI
promise (semver for the C API comes with 1.0.0).
