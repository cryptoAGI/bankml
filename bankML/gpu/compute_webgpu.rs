// SPDX-License-Identifier: MIT OR Apache-2.0
//! The GPU in a browser (browseML): [`compute`](super::compute)'s API — `Gpu`, `Buffer`, `Pipeline` — on WebGPU,
//! through the host's `bankml_gpu_*` imports (browseML's GPU worker holds the device, the pipelines and the buffers).
//!
//! The worker, its verification ([`super::kernels::verify_q1_0`]) and its calibration run unchanged on top of it, so a
//! WebGPU device computes rows only after it has reproduced the CPU kernel's bits. Each SPIR-V kernel is mapped to its
//! WGSL twin ([`super::wgsl`], bankML's own), which performs the same float operations in the same order. The engine
//! hands the host the shader's text and the vendor's workgroup size: the host holds no kernel of its own.
//!
//! Calls are few per product: writes are staged here and posted without waiting, a submit is posted, and one
//! [`Gpu::wait`] blocks (the host's `Atomics.wait`) until the result is in the output buffer's mirror in this memory, so
//! [`Gpu::read_f32`] is a copy here. Uploads wait (once per matrix).

use std::cell::RefCell;

#[link(wasm_import_module = "env")]
extern "C" {
    /// The adapter, as text "vendor\tarchitecture\tdescription\tkind\theap\tsubgroup" into `out` (at most `cap`
    /// bytes; the largest subgroup size last, 0 when unknown); its length, or 0 when the host has no WebGPU (or no bridge).
    fn bankml_gpu_adapter(out: *mut u8, cap: u32) -> u32;
    /// Open the device; 1 on success.
    fn bankml_gpu_open() -> u32;
    /// A storage buffer of `bytes`; its id (0 on failure).
    fn bankml_gpu_buffer(bytes: u32) -> u32;
    /// Copy `len` bytes at `ptr` into buffer `id` from its start: `wait` 1 blocks until copied, 0 posts and returns.
    fn bankml_gpu_write(id: u32, ptr: *const u8, len: u32, wait: u32) -> u32;
    fn bankml_gpu_free(id: u32);
    /// A compute pipeline from the WGSL text at `code` (`len` bytes, entry point `main`), its `WG` override set to
    /// `workgroup`, `bindings` storage buffers and a uniform after them; its id (0: refused).
    fn bankml_gpu_pipeline(code: *const u8, len: u32, workgroup: u32, bindings: u32) -> u32;
    /// Dispatch pipeline `pipe` on buffers `ids[..n]` with push constants `push[..np]`; the last buffer's first
    /// `out_len` bytes are copied to `out_ptr` here when it finishes. Returns at once.
    fn bankml_gpu_submit(pipe: u32, ids: *const u32, n: u32, push: *const u32, np: u32, out_ptr: *mut u8, out_len: u32) -> u32;
    /// Block until every submit since the last wait finished and its result is copied; 1 on success.
    fn bankml_gpu_wait() -> u32;
    /// Bytes the device's buffers hold now.
    fn bankml_gpu_allocated() -> f64;
}

/// The host's adapter, when it offers one: (vendor, architecture, description, kind, heap bytes, largest subgroup).
pub fn adapter() -> Option<(String, String, String, String, u64, u32)> {
    let mut b = vec![0u8; 512];
    // SAFETY: the host writes at most `cap` bytes into `b`.
    let n = unsafe { bankml_gpu_adapter(b.as_mut_ptr(), b.len() as u32) } as usize;
    if n == 0 || n > b.len() {
        return None;
    }
    let s = String::from_utf8_lossy(&b[..n]).into_owned();
    let f: Vec<&str> = s.split('\t').collect();
    let get = |i: usize| f.get(i).map(|x| x.to_string()).unwrap_or_default();
    Some((get(0), get(1), get(2), get(3), get(4).parse().unwrap_or(0), get(5).parse().unwrap_or(0)))
}

/// One WebGPU device, through the host.
pub struct Gpu {
    pub name: String,
    pub heap_bytes: u64,
    /// The workgroup size for this vendor ([`super::wgsl::workgroup_for`]).
    pub workgroup: u32,
}

/// A device buffer: its id, its size, the bytes staged for its next write, and (once it has been an output) the
/// mirror its results land in.
pub struct Buffer {
    id: u32,
    pub bytes: usize,
    stage: RefCell<Vec<u8>>,
    mirror: RefCell<Vec<u8>>,
}

pub struct Pipeline {
    id: u32,
    bindings: u32,
    push_bytes: u32,
}

/// The WGSL twin of a SPIR-V kernel: both Q1_0 kernels (one invocation per row, or eight) have the same one, which
/// takes one activation or many.
fn twin_of(spirv: &[u32]) -> Option<&'static str> {
    (spirv == super::kernels::q1_0_mat_vec().as_slice() || spirv == super::kernels::q1_0_mat_vec8().as_slice()
        || spirv == super::kernels::q1_0_mat_mul8().as_slice()).then_some(super::wgsl::Q1_0_MM)
}

impl Gpu {
    /// Many activations in one dispatch: push constants (rows, blocks per row, columns), output `O[col · rows + row]`
    /// (the WGSL kernel's token dimension).
    pub const COLUMNS: bool = true;
    /// Several submits in flight, one [`Gpu::wait`] for all: each output lands in its own mirror.
    pub const MANY_IN_FLIGHT: bool = true;

    /// Open the host's WebGPU device (`index` is 0: the one adapter the page was given).
    pub fn open(_index: usize) -> Result<Gpu, String> {
        let (vendor, arch, desc, _, heap, subgroup) = adapter().ok_or("no WebGPU adapter")?;
        let workgroup = super::wgsl::workgroup_for(&vendor, subgroup);
        // SAFETY: an import without arguments.
        if unsafe { bankml_gpu_open() } != 1 {
            return Err("the WebGPU device did not open".into());
        }
        Ok(Gpu { name: [vendor, arch, desc].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" · "), heap_bytes: heap, workgroup })
    }

    pub fn buffer(&self, bytes: usize) -> Result<Buffer, String> {
        let bytes = bytes.max(4);
        // SAFETY: an import with a plain argument.
        let id = unsafe { bankml_gpu_buffer(bytes as u32) };
        if id == 0 {
            return Err(format!("a WebGPU buffer of {bytes} bytes was refused"));
        }
        Ok(Buffer { id, bytes, stage: RefCell::new(Vec::new()), mirror: RefCell::new(Vec::new()) })
    }

    /// A buffer holding `data` (waits until the host has copied it).
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buffer, String> {
        let bytes = std::mem::size_of_val(data);
        let b = self.buffer(bytes)?;
        // SAFETY: `data` is `bytes` long and lives until the call returns, which waits for the copy.
        if unsafe { bankml_gpu_write(b.id, data.as_ptr() as *const u8, bytes as u32, 1) } != 1 {
            return Err("a WebGPU upload failed".into());
        }
        Ok(b)
    }

    /// Overwrite the start of `b` with `data`: staged in `b` and posted (the host copies it before the next submit).
    pub fn write<T: Copy>(&self, b: &Buffer, data: &[T]) {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= b.bytes);
        let mut st = b.stage.borrow_mut();
        st.clear();
        // SAFETY: `T` is plain data, `bytes` long.
        st.extend_from_slice(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) });
        // SAFETY: the stage lives in `b` until the next write to `b`, which comes after the submit that uses it.
        unsafe { bankml_gpu_write(b.id, st.as_ptr(), bytes as u32, 0) };
    }

    /// The first `n` f32 of `b`, from its mirror (call after [`Gpu::wait`]).
    pub fn read_f32(&self, b: &Buffer, n: usize) -> Vec<f32> {
        let m = b.mirror.borrow();
        assert!(n * 4 <= m.len(), "this buffer was not the output of the last submit");
        m[..n * 4].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    }

    pub fn allocated(&self) -> u64 {
        // SAFETY: an import without arguments.
        unsafe { bankml_gpu_allocated() as u64 }
    }

    pub fn free(&self, b: Buffer) {
        drop(b);
    }

    pub fn pipeline(&self, spirv: &[u32], bindings: u32, push_bytes: u32) -> Result<Pipeline, String> {
        let code = twin_of(spirv).ok_or("no WGSL twin for this kernel")?;
        // SAFETY: the host copies the text before the call returns.
        let id = unsafe { bankml_gpu_pipeline(code.as_ptr(), code.len() as u32, self.workgroup, bindings) };
        if id == 0 {
            return Err("the WebGPU pipeline was refused".into());
        }
        Ok(Pipeline { id, bindings, push_bytes })
    }

    pub fn run(&self, p: &Pipeline, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<(), String> {
        self.submit(p, bufs, push, groups)?;
        self.wait()
    }

    pub fn wait(&self) -> Result<(), String> {
        // SAFETY: an import without arguments; it blocks this thread (a worker's) until the result is copied.
        if unsafe { bankml_gpu_wait() } != 1 {
            return Err("the WebGPU product failed".into());
        }
        Ok(())
    }

    /// Start the kernel; the host computes the workgroups for its own workgroup size (the vendor's) from the rows (and
    /// the columns, when a third push constant gives them) in `push`, so `groups` is not used. The last buffer is the output: its result lands in its mirror here.
    pub fn submit(&self, p: &Pipeline, bufs: &[&Buffer], push: &[u32], _groups: u32) -> Result<(), String> {
        assert_eq!(bufs.len() as u32, p.bindings);
        assert_eq!(push.len() as u32 * 4, p.push_bytes);
        let out = bufs.last().ok_or("no buffers")?;
        let mut m = out.mirror.borrow_mut();
        m.resize(out.bytes, 0);
        let ids: Vec<u32> = bufs.iter().map(|b| b.id).collect();
        // SAFETY: `ids`, `push` and the mirror outlive the call; the host copies `ids` and `push` at once, and writes
        // the mirror only before `wait` returns, while `out` (and so its mirror) is borrowed by the caller.
        let ok = unsafe { bankml_gpu_submit(p.id, ids.as_ptr(), ids.len() as u32, push.as_ptr(), push.len() as u32, m.as_mut_ptr(), m.len() as u32) };
        if ok != 1 {
            return Err("the WebGPU submit was refused".into());
        }
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: an import with a plain argument; the host frees the buffer after any work queued before it.
        unsafe { bankml_gpu_free(self.id) };
    }
}

// SAFETY: one engine thread drives the GPU (the forward pass holds the worker behind a Mutex); ids are plain numbers
// and the host serialises the calls.
unsafe impl Send for Gpu {}
unsafe impl Send for Buffer {}
unsafe impl Send for Pipeline {}
