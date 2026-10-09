// SPDX-License-Identifier: MIT OR Apache-2.0
//! browseML — bankML in the browser.
//!
//! The same engine as `bankml serve --native` and the C API, compiled to WebAssembly (`wasm32-wasip1`), so a visitor's
//! own CPU answers: the model is verified before it plays (the GGUF guard, then the sha256 pin in its FORK.json), and
//! every answer carries `bankml_receipt`, exactly as served. Nothing here re-implements the engine: this crate only
//! moves bytes across the WebAssembly boundary, as `capi/` moves them across C's.
//!
//! The page gives the module a WASI filesystem holding the model file (see `hf/space/browseml.js`); the model is
//! opened by path, as the native engine opens it. One thread (the pool's results do not depend on the thread count).
//!
//! Exports (all strings are UTF-8 pointer + length in the module's memory; results are read from the out buffer):
//!   browseml_alloc(n) -> ptr, browseml_free(ptr, n)
//!   browseml_open(model_path, fork_json, n_ctx) -> 0 or an error code; out = the verified model as JSON, or the reason
//!   browseml_chat(request_json) -> 0 or an error code; pieces stream to the import `browseml.piece`; out = the
//!     `/v1/chat/completions` object (usage, timings, bankml_receipt) or {"error": …}
//!   browseml_out_ptr() / browseml_out_len(), browseml_version() (into out)
//! Imports (module "browseml"): piece(ptr, len) -> i32 (0 stops the answer), log(level, ptr, len).

use engine::native::{engine_name, header_info, Native};
use engine::serve::{completion_json, Json, NativeChat, Tally};
use engine::Verified;
use std::path::Path;
use std::sync::Mutex;

pub const OK: i32 = 0;
pub const E_ARG: i32 = -1;
pub const E_REQUEST: i32 = -2;
pub const E_NOT_OPEN: i32 = -3;
pub const E_ENGINE: i32 = -4;
pub const E_REFUSED: i32 = -6;

#[link(wasm_import_module = "browseml")]
extern "C" {
    fn piece(ptr: *const u8, len: usize) -> i32;
    fn log(level: i32, ptr: *const u8, len: usize);
}

struct Open {
    native: Native,
    verified: Verified,
    engine: String,
    model_id: String,
}

static OPEN: Mutex<Option<Open>> = Mutex::new(None);
static OUT: Mutex<Vec<u8>> = Mutex::new(Vec::new());

fn set_out(s: &str) {
    let mut o = OUT.lock().unwrap_or_else(|e| e.into_inner());
    o.clear();
    o.extend_from_slice(s.as_bytes());
}

fn to_page(level: i32, msg: &str) {
    // SAFETY: the page's import reads `len` bytes at `ptr` during the call and keeps nothing.
    unsafe { log(level, msg.as_ptr(), msg.len()) }
}

/// # Safety
/// `ptr` must point to `len` readable bytes in this module's memory (or `len` be 0).
unsafe fn arg<'a>(ptr: *const u8, len: usize, what: &str) -> Result<&'a str, String> {
    if len == 0 {
        return Ok("");
    }
    if ptr.is_null() {
        return Err(format!("{what} is NULL"));
    }
    std::str::from_utf8(unsafe { std::slice::from_raw_parts(ptr, len) }).map_err(|_| format!("{what} is not UTF-8"))
}

#[no_mangle]
pub extern "C" fn browseml_alloc(n: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(n.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` and `n` must come from one `browseml_alloc(n)`, freed once.
#[no_mangle]
pub unsafe extern "C" fn browseml_free(ptr: *mut u8, n: usize) {
    if !ptr.is_null() {
        drop(unsafe { Vec::from_raw_parts(ptr, 0, n.max(1)) });
    }
}

#[no_mangle]
pub extern "C" fn browseml_out_ptr() -> *const u8 {
    OUT.lock().unwrap_or_else(|e| e.into_inner()).as_ptr()
}

#[no_mangle]
pub extern "C" fn browseml_out_len() -> usize {
    OUT.lock().unwrap_or_else(|e| e.into_inner()).len()
}

#[no_mangle]
pub extern "C" fn browseml_version() -> usize {
    set_out(engine::VERSION);
    engine::VERSION.len()
}

/// As `bankml_open`: the guard, the sha256 pin, the native forward pass, or a refusal with serve's reason.
fn open(model: &Path, fork: &str, n_ctx: u32) -> Result<Open, String> {
    // the relaxed build is the reference's arithmetic only where relaxed_madd is fused: refuse it elsewhere (the page
    // then loads browseml-mt.wasm)
    #[cfg(target_feature = "relaxed-simd")]
    if !engine::q1_0::relaxed_madd_is_fused() {
        return Err("refuse: this machine's relaxed_madd is not fused, so the relaxed build would not be exact: load browseml-mt.wasm".into());
    }
    let verified = engine::verify(model, fork, engine::gguf::Engine::Mainline).map_err(|e| format!("refuse: {e}"))?;
    let model_id = model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if let Err(why) = header_info(model).native {
        return Err(format!("refuse: {model_id}: bankML's native forward pass does not play it: {why}"));
    }
    let n_ctx = if n_ctx == 0 { 4096 } else { n_ctx as usize };
    let native = Native::open(model, n_ctx)?;
    engine::log(engine::LOG_INFO, &format!("browseml: {model_id} verified (sha256 {}); open in bankML's forward pass, context {n_ctx}", verified.model_sha256));
    Ok(Open { engine: engine_name(&native), native, verified, model_id })
}

/// # Safety
/// The pointers must be valid for their lengths in this module's memory.
#[no_mangle]
pub unsafe extern "C" fn browseml_open(model_ptr: *const u8, model_len: usize, fork_ptr: *const u8, fork_len: usize, n_ctx: u32) -> i32 {
    engine::set_log_sink(Some(to_page));
    let r = (|| {
        let model = unsafe { arg(model_ptr, model_len, "model_path") }?;
        let fork = unsafe { arg(fork_ptr, fork_len, "fork_json") }?;
        *OPEN.lock().unwrap_or_else(|e| e.into_inner()) = None; // the previous model's memory first
        open(Path::new(model), fork, n_ctx)
    })();
    match r {
        Ok(o) => {
            let v = &o.verified;
            set_out(&format!(
                "{{\"model\": {}, \"sha256\": {}, \"guard\": {}, \"engine\": {}, \"arch\": {}, \"version\": {}}}",
                engine::gguf::jstr(&o.model_id),
                engine::gguf::jstr(&v.model_sha256),
                engine::gguf::jstr(v.guard),
                engine::gguf::jstr(&o.engine),
                v.arch.as_deref().map(engine::gguf::jstr).unwrap_or_else(|| "null".into()),
                engine::gguf::jstr(engine::VERSION)
            ));
            *OPEN.lock().unwrap_or_else(|e| e.into_inner()) = Some(o);
            OK
        }
        Err(e) => {
            set_out(&e);
            if e.starts_with("refuse") { E_REFUSED } else { E_ENGINE }
        }
    }
}

/// # Safety
/// `req_ptr` must be valid for `req_len` bytes in this module's memory.
#[no_mangle]
pub unsafe extern "C" fn browseml_chat(req_ptr: *const u8, req_len: usize) -> i32 {
    let r = (|| {
        let guard = OPEN.lock().unwrap_or_else(|e| e.into_inner());
        let o = guard.as_ref().ok_or((E_NOT_OPEN, "no model is open: browseml_open first".to_string()))?;
        let text = unsafe { arg(req_ptr, req_len, "request_json") }.map_err(|e| (E_ARG, e))?;
        let req = Json::parse(text).ok_or((E_REQUEST, "request_json is not JSON".to_string()))?;
        let nc = NativeChat::parse(&o.native, &req, text).map_err(|e| (E_REQUEST, e))?;
        let mut t = Tally::new(text.as_bytes());
        let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let d = nc
            .run(&o.native, &mut t, true, |p| {
                // SAFETY: the page's import reads `len` bytes at `ptr` during the call and keeps nothing.
                p.is_empty() || unsafe { piece(p.as_ptr(), p.len()) } != 0
            })
            .map_err(|e| (E_ENGINE, e))?;
        Ok(completion_json(created, &o.model_id, &d, &t, &o.engine, &o.verified))
    })();
    match r {
        Ok(j) => {
            set_out(&j);
            OK
        }
        Err((code, e)) => {
            set_out(&format!("{{\"error\": {{\"code\": {code}, \"message\": {}}}}}", engine::gguf::jstr(&e)));
            code
        }
    }
}
