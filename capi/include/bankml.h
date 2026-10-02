/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/*
 * bankml.h — bankML's C API (0.3.2): verified low-bit inference behind one small header.
 *
 *   cargo build --release -p bankml-capi      → target/release/libbankml.so and libbankml.a
 *   cc app.c -Icapi/include -Ltarget/release -lbankml                      (shared; run with LD_LIBRARY_PATH or rpath)
 *   cc app.c -Icapi/include target/release/libbankml.a -lgcc_s -lutil -lrt -lpthread -lm -ldl  (static)
 *
 * The shape of llama.h, with bankML's gate in front: a model is opened only after the GGUF guard plays it and its
 * sha256 equals its FORK.json record (the same `verify` as `bankml serve`), and every answer carries the receipt
 * `bankml serve --native` gives. docs/CAPI.md is the full reference.
 *
 * Ownership: strings bankML returns through `char**` (err, result_json) are the caller's and are freed with
 * bankml_free(); bankml_version()'s string is static. Strings passed in are only read during the call.
 * Threads: a handle has one slot; calls on one handle serialize. Different handles are independent (each maps its
 * model). bankml_close() must not race a call on the same handle. Nothing unwinds across this boundary: functions
 * return NULL or an error code; a panic in a release build aborts the process, as a failed C assert does.
 */
#ifndef BANKML_H
#define BANKML_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* An open, verified model and its one slot (KV cache + prompt cache). Opaque. */
typedef struct bankml bankml_t;

/* bankml_chat()'s return codes. */
#define BANKML_OK          0
#define BANKML_E_ARG      (-1) /* a NULL handle or request, or a request that is not UTF-8 */
#define BANKML_E_REQUEST  (-2) /* refused: not JSON, no messages, a sampler bankML does not reproduce, a bad stop */
#define BANKML_E_CHANGED  (-3) /* the model file changed since it was verified: no answer; open it again */
#define BANKML_E_ENGINE   (-4) /* the engine failed (e.g. the prompt does not fit the context) */
#define BANKML_E_PANIC    (-5) /* a panic was caught (unwinding builds only) */

/* Log levels. */
#define BANKML_LOG_ERROR 0
#define BANKML_LOG_WARN  1
#define BANKML_LOG_INFO  2
#define BANKML_LOG_DEBUG 3

/* Streaming: one whole-UTF-8 piece of the answer (`len` bytes, NUL-terminated). Return 0 to stop generating. */
typedef int (*bankml_piece_cb)(const char *piece, size_t len, void *user);

/* Logging: one message (`len` bytes, NUL-terminated; it may contain a NUL if a %c printed one). */
typedef void (*bankml_log_cb)(int level, const char *msg, size_t len, void *user);

/* The library version, e.g. "0.3.2". Static: do not free. */
const char *bankml_version(void);

/*
 * Verify `model_path` against the FORK.json at `fork_json_path` — the GGUF guard, then the sha256 pin — and open it
 * in bankML's own forward pass with a context of `n_ctx` tokens (0 = 4096). A model bankML does not play natively
 * (another architecture, other weight types, tied embeddings) is refused with the reason `bankml serve --native`
 * gives. Returns NULL on refusal or error, with the reason in *err (if err is not NULL; free it with bankml_free).
 */
bankml_t *bankml_open(const char *model_path, const char *fork_json_path, uint32_t n_ctx, char **err);

/*
 * One chat completion. `request_json` is the OpenAI / llama-server chat request `bankml serve --native` takes:
 *   {"messages": [{"role": "system"|"user"|"assistant", "content": "…"}, …],
 *    "temperature", "top_k", "top_p", "min_p", "min_keep", "seed", "max_tokens" (or "n_predict"), "stop"}
 * Unset sampling keys take the model's GGUF defaults; a sampler bankML does not reproduce (penalties, dry,
 * typical-p, xtc, top-n-sigma, dynamic temperature) is refused, never approximated. "model" and "stream" are ignored.
 * The answer streams to `cb` (may be NULL); *result_json (if not NULL) receives the non-streamed
 * /v1/chat/completions object — choices, usage, timings.cache_n, bankml_receipt — or {"error": {"code", "message"}}.
 * The slot keeps its KV cache between calls and reuses the longest common prefix, as llama-server's -np 1 does.
 * Returns BANKML_OK or a BANKML_E_* code.
 */
int bankml_chat(bankml_t *h, const char *request_json, bankml_piece_cb cb, void *user, char **result_json);

/* Close a handle: the model's mapping and KV cache are released. NULL is a no-op. */
void bankml_close(bankml_t *h);

/* Free a string bankML returned through err or result_json. NULL is a no-op. */
void bankml_free(char *s);

/*
 * Send every message — bankml_log()'s and the library's own (verification, the GPU) — to `cb` with `user`.
 * NULL restores the default, standard error. The sink is process-wide; `cb` must not call bankml_set_log().
 */
void bankml_set_log(bankml_log_cb cb, void *user);

/*
 * A printf-style message at `level` to the sink. Byte-identical to glibc's snprintf for:
 *   %d %i (with hh h l ll z)   %u %x %X (with hh h l ll z)   %f %F %lf   %c   %s   %p   %%
 *   flags - + space # 0, width and precision as numbers or * (an int argument).
 * Anything else is written literally and never guessed: an unknown conversion (%e %g %o %n %Lf %ls %jd …) becomes
 * "%<unsupported:SPEC>" and, since its argument's type is unknown, no later argument is read ("%<skipped:SPEC>");
 * a known conversion with flags it does not vouch for (%+s %05c %#p %#d, width or precision over 65536) becomes
 * "%<unsupported:SPEC>" with its argument read and dropped. %n never writes. Defined in Rust (a C-variadic function).
 */
void bankml_log(int level, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#ifdef __cplusplus
}
#endif

#endif /* BANKML_H */
