# `capi/src/` — the C API: `libbankml.so` / `libbankml.a`

## Summary

`capi/` (package `bankml-capi`) is bankML as a library for C, C++, Python, Go or Swift programs, declared in
`capi/include/bankml.h`. `lib.rs` holds the entry points; `printf.rs` is the formatter behind `bankml_log`. The full
reference (build and link, every function, ownership, threads, error codes, oracles) is **[CAPI.md](../CAPI.md)**;
this page only maps it to the two source files.

## Technical usage

- **`lib.rs`**: `bankml_version`, `bankml_open` (the same `verify` as `bankml serve`, then the native forward pass),
  `bankml_chat` (the `serve --native` chat request, streamed to a callback, with the same receipt through
  `serve::NativeChat`, `serve::completion_json` and `serve::Tally`), `bankml_close`, `bankml_free`, `bankml_set_log`
  and `bankml_log`. Every entry point returns an error code or NULL with a message and runs inside `catch_unwind`.
- **`bankml_log` is a C-variadic function defined in Rust** (stabilized in Rust 1.99, pinned in `rust-toolchain.toml`):

  ```rust
  pub unsafe extern "C" fn bankml_log(level: c_int, fmt: *const c_char, mut args: ...)
  ```

  It passes `&mut args` (a `VaList`) to `printf::format` and delivers the result to the sink `bankml_set_log`
  installed (standard error by default). The library's own messages reach the same sink through `bankml::log`.
- **`printf.rs`**: `pub unsafe fn format(fmt: &[u8], a: &mut impl Args) -> Vec<u8>`. The `Args` trait is the typed
  reads the formatter needs; `VaList` implements it with `next_arg::<T>()` at the C type each conversion names, and
  the unit tests implement it with a typed queue.
  - Supported, byte-identical to glibc's `snprintf`: `%d %i %u %x %X` (with `hh h l ll z`), `%f %F %lf`, `%c`, `%s`,
    `%p`, `%%`; the flags `- + space # 0`; a width and a precision, each a number or `*`.
  - Anything else is written `%<unsupported:SPEC>` and never guessed. After an unknown conversion no further argument
    is read (`%<skipped:SPEC>`). `%n` never writes.

## How it is verified

The printf oracle (`testing/capi/printf_oracle.c`: `bankml_log` against libc `snprintf`, byte for byte, 47 of 47
formats identical, 17 of 17 unsupported cases marked), the capi chat oracle (`bankml_chat` against `serve --native`
and llama-server b11192's records), and `cargo test -p bankml-capi`. Details in [CAPI.md](../CAPI.md#the-oracles-in-the-release-gate).

## Advantages and efficiency

A program that links llama.cpp's `llama.h` can link bankML instead and get the same answers with verification and a
receipt built in. The crate's only dependency is the bankml crate itself, so the workspace still has no external
crate. `printf.rs` states a `# Safety` contract for each typed argument read and for `format`.

## Limitations

No tokenizer or logits access, one slot per handle, and no stable ABI promise before 1.0.0 ([TODO.md](../TODO.md)).
The release profile aborts on a panic rather than unwinding.

## See also

[CAPI.md](../CAPI.md) · [usage.md](../usage.md) (6b, the C API) · [oracles.md](../oracles.md) · [TODO.md](../TODO.md)
