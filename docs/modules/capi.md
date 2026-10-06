# `capi/src/` — the C API: `libbankml.so` / `libbankml.a`

## Summary

`capi/` (package `bankml-capi`) is bankML as a library for C, C++, Python, Go or Swift programs, declared in
`capi/include/bankml.h` (since 0.3.2). It is the llama.h-shaped seam for embedding bankML in another program, with
bankML's verification gate in front. `lib.rs` holds the entry points; `printf.rs` is the formatter behind
`bankml_log`. The full reference (build and link, every function, ownership, threads, error codes, oracles) is **[CAPI.md](../CAPI.md)**;
this page only maps it to the two source files.

## Technical usage

- **`lib.rs`**: `bankml_version`, `bankml_open` (the same `verify` as `bankml serve`, then the native forward pass),
  `bankml_chat` (the `serve --native` chat request, streamed to a callback, with the same receipt through
  `serve::NativeChat`, `serve::completion_json` and `serve::Tally`), `bankml_close`, `bankml_free`, `bankml_set_log`
  and `bankml_log`. Every entry point returns an error code or NULL with a message and runs inside `catch_unwind`.
  One handle has one slot, so calls on it serialize.

  ```rust
  pub unsafe extern "C" fn bankml_open(model_path: *const c_char, fork_json_path: *const c_char, n_ctx: u32,
                                       err: *mut *mut c_char) -> *mut Bankml        // n_ctx 0: 4096
  pub unsafe extern "C" fn bankml_chat(h: *mut Bankml, request_json: *const c_char, cb: Option<PieceCb>,
                                       user: *mut c_void, result_json: *mut *mut c_char) -> c_int
  ```

  `bankml_open` reads the FORK.json, records the file's identity, runs `verify` (the guard, then the sha256 pin),
  and refuses the file if its identity changed while it was being hashed, as `bankml serve` does. A model that
  bankML's native forward pass does not play is refused with serve's reason. The refusal goes to `*err` and is also
  logged at error level. Messages use the file name as given; the `model` field of each answer uses the canonical
  file's name, as serve does.

  `bankml_chat` checks before every answer, as `serve --native` does, that the file is still the one that was
  verified (otherwise `BANKML_E_CHANGED`: reopen to verify again). The request takes `messages`, `temperature`,
  `top_k`, `top_p`, `min_p`, `seed`, `max_tokens` / `n_predict` and `stop`; since 0.3.3 `response_format`,
  `json_schema` and `grammar` as `serve --native` takes them. The answer streams to the callback as whole, non-empty
  UTF-8 pieces, each NUL-terminated. The handle's lock is held while the callback runs, so the callback must not
  close the handle or call `bankml_chat` on it.

  `bankml_chat` returns `BANKML_OK` (0) or `BANKML_E_ARG` (−1), `_REQUEST` (−2), `_CHANGED` (−3, the file changed
  since it was verified), `_ENGINE` (−4) or `_PANIC` (−5). Because it parses the request with `serve --native`'s
  `NativeChat::parse`, it takes what `/v1` takes: the whole sampler chain (0.3.7) and `logprobs` / `top_logprobs`
  (0.3.8), whose entries come back in `*result_json` as `choices[0].logprobs.content`. The callback receives text
  only.
- **`bankml_log` is a C-variadic function defined in Rust** (stabilized in Rust 1.99, pinned in `rust-toolchain.toml`):

  ```rust
  pub unsafe extern "C" fn bankml_log(level: c_int, fmt: *const c_char, mut args: ...)
  ```

  It passes `&mut args` (a `VaList`) to `printf::format` and delivers the result to the sink `bankml_set_log`
  installed (standard error by default). The library's own messages reach the same sink through `bankml::log`.
  The sink is process-wide; the callback is called outside the sink's lock, so it may run concurrently on any
  thread that calls into bankML. A NUL inside a message is replaced by U+2400 so that C does not truncate it.
- **`printf.rs`**: `pub unsafe fn format(fmt: &[u8], a: &mut impl Args) -> Vec<u8>`. The `Args` trait is the typed
  reads the formatter needs; `VaList` implements it with `next_arg::<T>()` at the C type each conversion names, and
  the unit tests implement it with a typed queue.
  - Supported, byte-identical to glibc's `snprintf`: `%d %i %u %x %X` (with `hh h l ll z`), `%f %F %lf`, `%c`, `%s`,
    `%p`, `%%`; the flags `- + space # 0`; a width and a precision, each a number or `*` (an `int` argument, as C
    reads it).
  - Anything else is written `%<unsupported:SPEC>` and never guessed. `%n` never writes through its argument.
  - An unknown conversion or length (`%e`, `%g`, `%o`, `%n`, `%Lf`, `%ls`, `%jd`, …) reads no argument. Since the
    formatter cannot know that argument's type, it reads no argument after it either: every later conversion is
    written `%<skipped:SPEC>` (`%%` still prints `%`).
  - A known conversion with a flag combination it does not vouch for (`%+s`, `%05c`, `%#p`, `%#d`), or with a width
    or precision above 65,536 (`LIMIT`), is written `%<unsupported:SPEC>`. Its argument has a known type, so it is
    read and dropped and the rest of the format goes on. The limit exists because this is a log line, not a way to
    make the library allocate gigabytes.
  - A `%` with anything between it and the next `%` (`%5%`) is marked but reads no argument; a format that ends
    inside a conversion is written `%<incomplete:SPEC>`.
  - `%f` relies on Rust printing the exact decimal expansion of the double, rounded to nearest with ties to even,
    as glibc does in the default rounding mode. A NULL `%s` prints `(null)` when the precision allows six bytes,
    otherwise nothing; a `%s` with a precision never reads past it, so an unterminated buffer is legal, as in C.

## How it is verified

The printf oracle (`testing/capi/printf_oracle.c`: `bankml_log` against libc `snprintf`, byte for byte, 47 of 47
formats identical, 17 of 17 unsupported cases marked), the capi chat oracle (`bankml_chat` against `serve --native`
and llama-server b11192's records), and `cargo test -p bankml-capi`. Details in [CAPI.md](../CAPI.md#the-oracles-in-the-release-gate).

## Advantages and efficiency

A program that links llama.cpp's `llama.h` can link bankML instead and get the same answers with verification and a
receipt built in. The crate's only dependency is the bankml crate itself, so the workspace still has no external
crate. `printf.rs` states a `# Safety` contract for each typed argument read and for `format`.

## Limitations

No tokenizer or raw logits access (only the request's `logprobs`), one slot per handle (calls on a handle
serialize), and no stable ABI promise before 1.0.0 ([TODO.md](../TODO.md)). The capi chat oracle does not yet cover
the 0.3.7 samplers or logprobs through `bankml_chat`; they share `serve --native`'s code and its oracles. The release
profile aborts on a panic rather than unwinding, as a C library's failed assertion does; `catch_unwind` (and
`BANKML_E_PANIC`) only takes effect in an unwinding build.

## See also

[CAPI.md](../CAPI.md) · [usage.md](../usage.md) (6b, the C API) · [oracles.md](../oracles.md) · [TODO.md](../TODO.md)
