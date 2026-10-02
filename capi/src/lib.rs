// SPDX-License-Identifier: MIT OR Apache-2.0
//! bankML's C API (0.3.2): `libbankml.so` / `libbankml.a`, declared in `capi/include/bankml.h`, documented in
//! docs/CAPI.md. The llama.h-shaped seam for embedding bankML in another program, with bankML's gate in front:
//!
//! - `bankml_open` runs the same `verify` as `bankml serve` (the guard, then the sha256 pin to FORK.json), then opens
//!   the model in bankML's own forward pass; a model it does not play natively is refused with serve's reason;
//! - `bankml_chat` takes the OpenAI / llama-server chat request `serve --native` takes, refuses what it refuses, streams
//!   the answer to a callback, and returns the object `/v1/chat/completions` returns — usage, `timings.cache_n` and
//!   the same `bankml_receipt` (one code path: `serve::NativeChat`, `serve::completion_json`, `serve::Tally`);
//! - `bankml_log` is a printf-style C-variadic function **defined in Rust** (`mut args: ...`, Rust 1.99), formatting
//!   with `printf.rs`; it and the library's own messages go to the one sink `bankml_set_log` installs.
//!
//! Nothing here unwinds into C: every entry point returns an error code or NULL with a message, and wraps its work
//! in `catch_unwind` (which catches in an unwinding build; the release profile aborts on a panic, as a C library's
//! failed assertion does). One handle has one slot: calls on it serialize.

use engine::native::{engine_name, header_info, Native};
use engine::serve::{completion_json, ident, FileIdent, Json, NativeChat, Tally};
use engine::Verified;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub mod printf;

/// `bankml_chat`'s return codes (`BANKML_OK`, `BANKML_E_*` in bankml.h).
pub const BANKML_OK: c_int = 0;
/// a NULL handle or request, or a request that is not UTF-8
pub const BANKML_E_ARG: c_int = -1;
/// the request is refused: not JSON, no messages, a sampler bankML does not reproduce, a malformed `stop`
pub const BANKML_E_REQUEST: c_int = -2;
/// the model file changed since `bankml_open` verified it: no answer; open it again to verify it again
pub const BANKML_E_CHANGED: c_int = -3;
/// the engine failed (the prompt does not fit the context, …)
pub const BANKML_E_ENGINE: c_int = -4;
/// a panic was caught (only in an unwinding build)
pub const BANKML_E_PANIC: c_int = -5;

/// A verified model, open in bankML's forward pass, with its one slot. Opaque to C (`bankml_t`).
pub struct Bankml {
    native: Native,
    verified: Verified,
    engine: String,
    /// the canonical path that was hashed, and its identity then
    model: PathBuf,
    ident: FileIdent,
    model_id: String,
    run: Mutex<()>,
}

/// The streaming callback, `int (*)(const char* piece, size_t len, void* user)`: returns 0 to stop.
pub type PieceCb = unsafe extern "C" fn(piece: *const c_char, len: usize, user: *mut c_void) -> c_int;
/// The log sink: the level, the message (NUL-terminated, `len` bytes before the NUL), the user pointer.
pub type LogCb = unsafe extern "C" fn(level: c_int, msg: *const c_char, len: usize, user: *mut c_void);

fn to_c(s: &str) -> *mut c_char {
    // a NUL inside a message would cut it short in C: it is shown as U+2400 instead
    CString::new(s.replace('\0', "\u{2400}")).map(CString::into_raw).unwrap_or(std::ptr::null_mut())
}

unsafe fn path_arg(p: *const c_char, what: &str) -> Result<PathBuf, String> {
    use std::os::unix::ffi::OsStrExt;
    if p.is_null() {
        return Err(format!("{what} is NULL"));
    }
    let b = unsafe { CStr::from_ptr(p) }.to_bytes();
    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(b)))
}

/// The library's version, e.g. `"0.3.2"`. Static; do not free.
#[no_mangle]
pub extern "C" fn bankml_version() -> *const c_char {
    static V: OnceLock<CString> = OnceLock::new();
    V.get_or_init(|| CString::new(engine::VERSION).unwrap_or_default()).as_ptr()
}

fn open(model: &Path, fork_json: &Path, n_ctx: u32) -> Result<Bankml, String> {
    let fork = std::fs::read_to_string(fork_json).map_err(|e| format!("cannot read {}: {e}", fork_json.display()))?;
    // as `bankml serve`: the file's identity before the hash must be its identity after it
    let before = ident(model).map_err(|e| format!("{}: {e}", model.display()))?;
    let verified = engine::verify(model, &fork, engine::gguf::Engine::Mainline).map_err(|e| format!("refuse: {e}"))?;
    let canon = model.canonicalize().map_err(|e| format!("{}: {e}", model.display()))?;
    let id = ident(&canon).map_err(|e| format!("{}: {e}", canon.display()))?;
    if id != before {
        return Err(format!("{} changed while it was being hashed: refused", canon.display()));
    }
    // the name the caller gave (as serve's messages use it), and the canonical file's name (as serve's `model` field)
    let given = model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let model_id = canon.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if let Err(why) = header_info(&canon).native {
        return Err(format!("refuse: {given}: bankML's native forward pass does not play it: {why}"));
    }
    let n_ctx = if n_ctx == 0 { 4096 } else { n_ctx as usize }; // serve's default
    let native = Native::open(&canon, n_ctx)?;
    engine::log(engine::LOG_INFO, &format!("bankml: {given} verified (sha256 {}); open in bankML's forward pass, context {n_ctx}", verified.model_sha256));
    Ok(Bankml { engine: engine_name(&native), native, verified, model: canon, ident: id, model_id, run: Mutex::new(()) })
}

/// Verify `model_path` against the FORK.json at `fork_json_path` (the guard, then the sha256 pin) and open it in
/// bankML's forward pass with a context of `n_ctx` tokens (0: 4096). NULL on refusal or error, with the reason in
/// `*err` (when `err` is not NULL; free it with `bankml_free`).
///
/// # Safety
/// `model_path` and `fork_json_path` must be NULL or NUL-terminated strings; `err` must be NULL or point to a
/// writable `char*`.
#[no_mangle]
pub unsafe extern "C" fn bankml_open(model_path: *const c_char, fork_json_path: *const c_char, n_ctx: u32, err: *mut *mut c_char) -> *mut Bankml {
    if !err.is_null() {
        unsafe { *err = std::ptr::null_mut() };
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let model = unsafe { path_arg(model_path, "model_path") }?;
        let fork = unsafe { path_arg(fork_json_path, "fork_json_path") }?;
        open(&model, &fork, n_ctx)
    }));
    let why = match r {
        Ok(Ok(h)) => return Box::into_raw(Box::new(h)),
        Ok(Err(e)) => e,
        Err(_) => "bankml_open: a panic was caught".to_string(),
    };
    engine::log(engine::LOG_ERROR, &format!("bankml_open: {why}"));
    if !err.is_null() {
        unsafe { *err = to_c(&why) };
    }
    std::ptr::null_mut()
}

fn chat(h: &Bankml, body: &[u8], cb: Option<PieceCb>, user: *mut c_void) -> Result<String, (c_int, String)> {
    let _run = h.run.lock().unwrap_or_else(|e| e.into_inner());
    // as `serve --native` before every answer: the file must still be the one that was verified
    if ident(&h.model).ok() != Some(h.ident) {
        return Err((BANKML_E_CHANGED, format!("{} changed since bankml verified it: refused; bankml_open it again to verify it again", h.model_id)));
    }
    let text = std::str::from_utf8(body).map_err(|_| (BANKML_E_ARG, "request_json is not UTF-8".to_string()))?;
    let req = Json::parse(text).ok_or((BANKML_E_REQUEST, "request_json is not JSON".to_string()))?;
    let nc = NativeChat::parse(&h.native, &req).map_err(|e| (BANKML_E_REQUEST, e))?;
    let mut t = Tally::new(body);
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mut buf = Vec::new();
    let d = nc
        .run(&h.native, &mut t, true, |piece| {
            let Some(f) = cb else { return true };
            if piece.is_empty() {
                return true;
            }
            buf.clear();
            buf.extend_from_slice(piece.as_bytes());
            buf.push(0);
            unsafe { f(buf.as_ptr().cast(), piece.len(), user) != 0 }
        })
        .map_err(|e| (BANKML_E_ENGINE, e))?;
    Ok(completion_json(created, &h.model_id, &d, &t, &h.engine, &h.verified))
}

/// One chat completion: `request_json` in the OpenAI / llama-server shape (`messages`, and `temperature`, `top_k`,
/// `top_p`, `min_p`, `seed`, `max_tokens`/`n_predict`, `stop`, and since 0.3.3 `response_format` / `json_schema` /
/// `grammar` as `serve --native` takes them; a sampler bankML does not reproduce is refused). The
/// answer streams to `cb` (may be NULL) as whole UTF-8 pieces; `cb` returns 0 to stop. `*result_json` (when not NULL)
/// receives the `/v1/chat/completions` object with `usage`, `timings.cache_n` and `bankml_receipt` — or, on an error,
/// `{"error": {"code": …, "message": …}}`; free it with `bankml_free`. Returns `BANKML_OK` or a `BANKML_E_*` code.
///
/// # Safety
/// `h` must be NULL or a handle from `bankml_open` not yet closed; `request_json` NULL or a NUL-terminated string;
/// `result_json` NULL or a writable `char*`; `cb` (if any) must be callable with `user` and must not close `h`.
#[no_mangle]
pub unsafe extern "C" fn bankml_chat(h: *mut Bankml, request_json: *const c_char, cb: Option<PieceCb>, user: *mut c_void, result_json: *mut *mut c_char) -> c_int {
    if !result_json.is_null() {
        unsafe { *result_json = std::ptr::null_mut() };
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        if h.is_null() {
            return Err((BANKML_E_ARG, "the handle is NULL".to_string()));
        }
        if request_json.is_null() {
            return Err((BANKML_E_ARG, "request_json is NULL".to_string()));
        }
        let body = unsafe { CStr::from_ptr(request_json) }.to_bytes();
        chat(unsafe { &*h }, body, cb, user)
    }));
    let (code, json) = match r {
        Ok(Ok(j)) => (BANKML_OK, j),
        Ok(Err((code, e))) => (code, format!("{{\"error\": {{\"code\": {code}, \"message\": {}}}}}", engine::gguf::jstr(&e))),
        Err(_) => (BANKML_E_PANIC, format!("{{\"error\": {{\"code\": {BANKML_E_PANIC}, \"message\": \"a panic was caught\"}}}}")),
    };
    if !result_json.is_null() {
        unsafe { *result_json = to_c(&json) };
    }
    code
}

/// Close a handle (NULL is a no-op): the model's mapping and its KV cache are released.
///
/// # Safety
/// `h` must be NULL or a handle from `bankml_open`, not used again, and not in use by another thread.
#[no_mangle]
pub unsafe extern "C" fn bankml_close(h: *mut Bankml) {
    if !h.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(h) })));
    }
}

/// Free a string bankML returned (`err`, `result_json`). NULL is a no-op.
///
/// # Safety
/// `s` must be NULL or a string from this library, freed once.
#[no_mangle]
pub unsafe extern "C" fn bankml_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

// ---------------------------------------------------------------- the log ----------------------------------------

struct Sink {
    cb: Option<LogCb>,
    user: usize,
}

static SINK: Mutex<Sink> = Mutex::new(Sink { cb: None, user: 0 });

fn deliver(level: c_int, msg: &[u8]) {
    let (cb, user) = {
        let s = SINK.lock().unwrap_or_else(|e| e.into_inner());
        (s.cb, s.user)
    };
    match cb {
        Some(f) => {
            let mut v = Vec::with_capacity(msg.len() + 1);
            v.extend_from_slice(msg);
            v.push(0);
            unsafe { f(level, v.as_ptr().cast(), msg.len(), user as *mut c_void) }
        }
        None => {
            use std::io::Write;
            let mut e = std::io::stderr().lock();
            let _ = e.write_all(msg);
            let _ = e.write_all(b"\n");
        }
    }
}

fn from_engine(level: i32, msg: &str) {
    deliver(level, msg.as_bytes());
}

/// Send every message — `bankml_log`'s and the library's own (verification, the GPU, …) — to `cb` with `user`;
/// NULL restores standard error. The sink is process-wide.
///
/// # Safety
/// `cb` (if any) must stay callable with `user` until it is replaced; it may be called from any thread that calls
/// into bankML, and must not call `bankml_set_log` itself.
#[no_mangle]
pub unsafe extern "C" fn bankml_set_log(cb: Option<LogCb>, user: *mut c_void) {
    *SINK.lock().unwrap_or_else(|e| e.into_inner()) = Sink { cb, user: user as usize };
    engine::set_log_sink(if cb.is_some() { Some(from_engine) } else { None });
}

/// A printf-style message at `level` to the sink: `%d %i %u %x %X` (with `hh h l ll z`), `%f %F`, `%c %s %p %%`,
/// flags `- + space # 0`, width and precision (numbers or `*`), byte-identical to glibc's `snprintf`. Anything else
/// is written as `%<unsupported:SPEC>` and never guessed (see `printf.rs`). Defined in Rust as a C-variadic function.
///
/// # Safety
/// `fmt` must be NULL (nothing is logged) or a NUL-terminated string, and the arguments must match it as for
/// `printf`.
#[no_mangle]
pub unsafe extern "C" fn bankml_log(level: c_int, fmt: *const c_char, mut args: ...) {
    if fmt.is_null() {
        return;
    }
    let f = unsafe { CStr::from_ptr(fmt) }.to_bytes();
    if let Ok(out) = catch_unwind(AssertUnwindSafe(|| unsafe { printf::format(f, &mut args) })) {
        deliver(level, &out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{c_double, c_long, c_longlong, c_uint, c_ulong, c_ulonglong};

    extern "C" {
        fn snprintf(buf: *mut c_char, n: usize, fmt: *const c_char, ...) -> c_int;
    }

    static GOT: Mutex<Vec<u8>> = Mutex::new(Vec::new());
    /// the sink is process-wide: the tests that log take turns
    static SERIAL: Mutex<()> = Mutex::new(());

    unsafe extern "C" fn capture(level: c_int, msg: *const c_char, len: usize, _user: *mut c_void) {
        let b = unsafe { std::slice::from_raw_parts(msg.cast::<u8>(), len) };
        assert_eq!(unsafe { *msg.add(len) }, 0, "NUL-terminated");
        let mut g = GOT.lock().unwrap();
        g.clear();
        g.extend_from_slice(format!("{level}:").as_bytes());
        g.extend_from_slice(b);
    }

    /// The same format and arguments through `bankml_log` (the Rust-defined C-variadic function) and libc's
    /// `snprintf`: the bytes must be equal.
    macro_rules! same {
        ($fmt:expr $(, $a:expr)*) => {{
            let fmt: &CStr = $fmt;
            let mut buf = vec![0u8; 512];
            let n = unsafe { snprintf(buf.as_mut_ptr().cast(), buf.len(), fmt.as_ptr() $(, $a)*) };
            buf.truncate(n as usize);
            unsafe { bankml_log(2, fmt.as_ptr() $(, $a)*) };
            let got = GOT.lock().unwrap().clone();
            assert_eq!(String::from_utf8_lossy(&got[2..]), String::from_utf8_lossy(&buf), "{:?}", fmt);
            assert_eq!(&got[..2], b"2:");
        }};
    }

    #[test]
    fn bankml_log_equals_snprintf() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { bankml_set_log(Some(capture), std::ptr::null_mut()) };
        let s = c"bankml";
        same!(c"plain text, no conversions");
        same!(c"%d %i %u %d %d", 0 as c_int, -1 as c_int, c_uint::MAX, c_int::MIN, c_int::MAX);
        same!(c"[%5d|%-5d|%05d|%+d|% d|%.3d|%.0d|%8.3d]", 42 as c_int, 42 as c_int, -42 as c_int, 7 as c_int, 7 as c_int, 5 as c_int, 0 as c_int, -5 as c_int);
        same!(c"%ld %lu %lld %llu %zu", c_long::MIN, c_ulong::MAX, c_longlong::MIN, c_ulonglong::MAX, usize::MAX);
        same!(c"%x %X %#x %#X %08x %#010x %lx %llX", 255 as c_uint, 255 as c_uint, 255 as c_uint, 255 as c_uint, 0xbeef as c_uint, 0xbeef as c_uint, c_ulong::MAX, c_ulonglong::MAX);
        same!(c"%f %.3f %.0f %.0f %.0f %#.0f %10.2f %-10.2f| %010.2f %+.1f %.20f", 3.14258 as c_double, -2.0005 as c_double, 0.5 as c_double,
              1.5 as c_double, 2.5 as c_double, 3.0 as c_double, -1.005 as c_double, 1.0 as c_double, -3.25 as c_double, 0.0 as c_double, 0.1 as c_double);
        same!(c"%f %F %f %5f %f %.2f %f", f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::INFINITY, 1e300 as c_double, -0.0 as c_double, f64::MIN_POSITIVE);
        same!(c"[%s|%8s|%-8s|%.4s|%.*s|%*s]", s.as_ptr(), s.as_ptr(), s.as_ptr(), s.as_ptr(), 2 as c_int, s.as_ptr(), -9 as c_int, s.as_ptr());
        same!(c"%c%c|%3c|%-3c|", b'o' as c_int, b'k' as c_int, b'x' as c_int, b'y' as c_int);
        same!(c"%p %p %20p %-20p|", 0x1234usize as *const c_void, std::ptr::null::<c_void>(), s.as_ptr(), s.as_ptr());
        same!(c"100%% and %hhd %hd %hhu %hx", 300 as c_int, 70000 as c_int, 511 as c_uint, 0x12345 as c_uint);
        // an unsupported conversion: marked, and nothing read after it
        unsafe { bankml_log(1, c"%d %e %d".as_ptr(), 5 as c_int, 2.0 as c_double, 6 as c_int) };
        assert_eq!(GOT.lock().unwrap().as_slice(), b"1:5 %<unsupported:e> %<skipped:d>");
        // the library's own messages reach the same sink
        engine::log(engine::LOG_WARN, "from the engine");
        assert_eq!(GOT.lock().unwrap().as_slice(), b"1:from the engine");
        unsafe { bankml_set_log(None, std::ptr::null_mut()) };
    }

    #[test]
    fn null_safe_and_refusing() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(unsafe { CStr::from_ptr(bankml_version()) }.to_str().unwrap(), engine::VERSION);
        let mut err = std::ptr::null_mut();
        assert!(unsafe { bankml_open(std::ptr::null(), std::ptr::null(), 0, &mut err) }.is_null());
        assert_eq!(unsafe { CStr::from_ptr(err) }.to_str().unwrap(), "model_path is NULL");
        unsafe { bankml_free(err) };
        // a file that is not a pinned GGUF is refused, with the reason
        let dir = std::env::temp_dir().join(format!("bankml-capi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.gguf"), b"GGUX").unwrap();
        std::fs::write(dir.join("FORK.json"), b"{\"files\": []}").unwrap();
        let (m, f) = (CString::new(dir.join("m.gguf").to_str().unwrap()).unwrap(), CString::new(dir.join("FORK.json").to_str().unwrap()).unwrap());
        assert!(unsafe { bankml_open(m.as_ptr(), f.as_ptr(), 0, &mut err) }.is_null());
        assert!(unsafe { CStr::from_ptr(err) }.to_str().unwrap().starts_with("refuse: guard refused"));
        unsafe { bankml_free(err) };
        assert!(unsafe { bankml_open(m.as_ptr(), f.as_ptr(), 0, std::ptr::null_mut()) }.is_null()); // err may be NULL
        std::fs::remove_dir_all(&dir).unwrap();
        let mut res = std::ptr::null_mut();
        assert_eq!(unsafe { bankml_chat(std::ptr::null_mut(), c"{}".as_ptr(), None, std::ptr::null_mut(), &mut res) }, BANKML_E_ARG);
        assert!(unsafe { CStr::from_ptr(res) }.to_str().unwrap().contains("\"code\": -1"));
        unsafe { bankml_free(res) };
        unsafe { bankml_close(std::ptr::null_mut()) };
        unsafe { bankml_free(std::ptr::null_mut()) };
        unsafe { bankml_log(0, std::ptr::null()) };
    }
}
