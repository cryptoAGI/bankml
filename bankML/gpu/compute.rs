// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vulkan compute on one device: a compute queue, host-visible coherent buffers mapped for their lifetime,
//! pipelines from bankML's own SPIR-V, and a dispatch completed by a fence.
//!
//! Entry points come from the run-time loader (`vulkan.rs`); the C structs are declared here from the Vulkan 1.1
//! headers. Every call's result is checked.
//! Details: docs/modules/gpu.md.

use super::vulkan;
use std::ffi::{c_char, c_void, CStr};

type Pfn = unsafe extern "C" fn();
type H = u64; // non-dispatchable handle
type P = *mut c_void; // dispatchable handle

macro_rules! vk_struct { ($name:ident { $($f:ident : $t:ty),* $(,)? }) => { #[repr(C)] struct $name { $($f: $t),* } } }

vk_struct!(DeviceQueueCreateInfo { s_type: i32, p_next: *const c_void, flags: u32, family: u32, count: u32, priorities: *const f32 });
vk_struct!(DeviceCreateInfo { s_type: i32, p_next: *const c_void, flags: u32, queue_info_count: u32, queue_infos: *const DeviceQueueCreateInfo,
    layer_count: u32, layers: *const *const c_char, ext_count: u32, exts: *const *const c_char, features: *const c_void });
vk_struct!(BufferCreateInfo { s_type: i32, p_next: *const c_void, flags: u32, size: u64, usage: u32, sharing: i32, family_count: u32, families: *const u32 });
vk_struct!(MemoryRequirements { size: u64, alignment: u64, type_bits: u32 });
vk_struct!(MemoryAllocateInfo { s_type: i32, p_next: *const c_void, size: u64, type_index: u32 });
vk_struct!(ShaderModuleCreateInfo { s_type: i32, p_next: *const c_void, flags: u32, code_size: usize, code: *const u32 });
vk_struct!(DescriptorSetLayoutBinding { binding: u32, dtype: i32, count: u32, stages: u32, samplers: *const c_void });
vk_struct!(DescriptorSetLayoutCreateInfo { s_type: i32, p_next: *const c_void, flags: u32, count: u32, bindings: *const DescriptorSetLayoutBinding });
vk_struct!(PushConstantRange { stages: u32, offset: u32, size: u32 });
vk_struct!(PipelineLayoutCreateInfo { s_type: i32, p_next: *const c_void, flags: u32, set_count: u32, sets: *const H, pc_count: u32, pcs: *const PushConstantRange });
vk_struct!(PipelineShaderStageCreateInfo { s_type: i32, p_next: *const c_void, flags: u32, stage: u32, module: H, name: *const c_char, spec: *const c_void });
vk_struct!(ComputePipelineCreateInfo { s_type: i32, p_next: *const c_void, flags: u32, stage: PipelineShaderStageCreateInfo, layout: H, base: H, base_index: i32 });
vk_struct!(DescriptorPoolSize { dtype: i32, count: u32 });
vk_struct!(DescriptorPoolCreateInfo { s_type: i32, p_next: *const c_void, flags: u32, max_sets: u32, size_count: u32, sizes: *const DescriptorPoolSize });
vk_struct!(DescriptorSetAllocateInfo { s_type: i32, p_next: *const c_void, pool: H, count: u32, layouts: *const H });
vk_struct!(DescriptorBufferInfo { buffer: H, offset: u64, range: u64 });
vk_struct!(WriteDescriptorSet { s_type: i32, p_next: *const c_void, set: H, binding: u32, element: u32, count: u32, dtype: i32,
    images: *const c_void, buffers: *const DescriptorBufferInfo, views: *const c_void });
vk_struct!(CommandPoolCreateInfo { s_type: i32, p_next: *const c_void, flags: u32, family: u32 });
vk_struct!(CommandBufferAllocateInfo { s_type: i32, p_next: *const c_void, pool: H, level: i32, count: u32 });
vk_struct!(CommandBufferBeginInfo { s_type: i32, p_next: *const c_void, flags: u32, inheritance: *const c_void });
vk_struct!(SubmitInfo { s_type: i32, p_next: *const c_void, wait_count: u32, waits: *const H, wait_stages: *const u32, cb_count: u32,
    cbs: *const P, signal_count: u32, signals: *const H });
vk_struct!(FenceCreateInfo { s_type: i32, p_next: *const c_void, flags: u32 });
vk_struct!(QueueFamilyProperties { flags: u32, count: u32, ts_bits: u32, gran: [u32; 3] });
vk_struct!(MemoryType { flags: u32, heap: u32 });
vk_struct!(MemoryHeap { size: u64, flags: u32 });
vk_struct!(MemoryProperties { type_count: u32, types: [MemoryType; 32], heap_count: u32, heaps: [MemoryHeap; 16] });

const STORAGE_BUFFER: i32 = 7;
const STAGE_COMPUTE: u32 = 0x20;
const BIND_POINT_COMPUTE: i32 = 1;
const MEM_HOST_VISIBLE: u32 = 0x2;
const MEM_HOST_COHERENT: u32 = 0x4;
const MEM_DEVICE_LOCAL: u32 = 0x1;

fn check(r: i32, what: &str) -> Result<(), String> {
    if r == 0 { Ok(()) } else { Err(format!("{what} failed ({r})")) }
}

/// Device-level entry points, loaded once.
#[allow(clippy::type_complexity)]
struct Fns {
    create_buffer: unsafe extern "C" fn(P, *const BufferCreateInfo, *const c_void, *mut H) -> i32,
    buffer_reqs: unsafe extern "C" fn(P, H, *mut MemoryRequirements),
    alloc: unsafe extern "C" fn(P, *const MemoryAllocateInfo, *const c_void, *mut H) -> i32,
    bind: unsafe extern "C" fn(P, H, H, u64) -> i32,
    map: unsafe extern "C" fn(P, H, u64, u64, u32, *mut *mut c_void) -> i32,
    destroy_buffer: unsafe extern "C" fn(P, H, *const c_void),
    free: unsafe extern "C" fn(P, H, *const c_void),
    shader: unsafe extern "C" fn(P, *const ShaderModuleCreateInfo, *const c_void, *mut H) -> i32,
    dsl: unsafe extern "C" fn(P, *const DescriptorSetLayoutCreateInfo, *const c_void, *mut H) -> i32,
    playout: unsafe extern "C" fn(P, *const PipelineLayoutCreateInfo, *const c_void, *mut H) -> i32,
    pipelines: unsafe extern "C" fn(P, H, u32, *const ComputePipelineCreateInfo, *const c_void, *mut H) -> i32,
    dpool: unsafe extern "C" fn(P, *const DescriptorPoolCreateInfo, *const c_void, *mut H) -> i32,
    dsets: unsafe extern "C" fn(P, *const DescriptorSetAllocateInfo, *mut H) -> i32,
    update: unsafe extern "C" fn(P, u32, *const WriteDescriptorSet, u32, *const c_void),
    cpool: unsafe extern "C" fn(P, *const CommandPoolCreateInfo, *const c_void, *mut H) -> i32,
    cbufs: unsafe extern "C" fn(P, *const CommandBufferAllocateInfo, *mut P) -> i32,
    begin: unsafe extern "C" fn(P, *const CommandBufferBeginInfo) -> i32,
    end: unsafe extern "C" fn(P) -> i32,
    reset_cb: unsafe extern "C" fn(P, u32) -> i32,
    bind_pipeline: unsafe extern "C" fn(P, i32, H),
    bind_sets: unsafe extern "C" fn(P, i32, H, u32, u32, *const H, u32, *const u32),
    push: unsafe extern "C" fn(P, H, u32, u32, u32, *const c_void),
    dispatch: unsafe extern "C" fn(P, u32, u32, u32),
    fence: unsafe extern "C" fn(P, *const FenceCreateInfo, *const c_void, *mut H) -> i32,
    submit: unsafe extern "C" fn(P, u32, *const SubmitInfo, H) -> i32,
    wait: unsafe extern "C" fn(P, u32, *const H, u32, u64) -> i32,
    reset_fence: unsafe extern "C" fn(P, u32, *const H) -> i32,
    idle: unsafe extern "C" fn(P) -> i32,
    destroy_shader: unsafe extern "C" fn(P, H, *const c_void),
    destroy_dsl: unsafe extern "C" fn(P, H, *const c_void),
    destroy_playout: unsafe extern "C" fn(P, H, *const c_void),
    destroy_pipeline: unsafe extern "C" fn(P, H, *const c_void),
    destroy_dpool: unsafe extern "C" fn(P, H, *const c_void),
    destroy_cpool: unsafe extern "C" fn(P, H, *const c_void),
    destroy_fence: unsafe extern "C" fn(P, H, *const c_void),
    destroy_device: unsafe extern "C" fn(P, *const c_void),
}

/// The logical device and its entry points, shared by the `Gpu` and every object made on it: whichever is dropped
/// last destroys the device, so nothing is ever used after its device is gone.
struct Device {
    dev: P,
    fns: Fns,
    /// The Vulkan instance the device was made from; destroyed after the device.
    inst: Instance,
    /// Bytes bankML's live buffers hold on this device.
    allocated: std::sync::atomic::AtomicU64,
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: the last reference is going: no `Gpu`, `Buffer` or `Pipeline` of this device is left, so no work is
        // in flight on it and no object of it remains.
        unsafe {
            (self.fns.idle)(self.dev);
            (self.fns.destroy_device)(self.dev, std::ptr::null());
        }
        // then `inst` drops, and the instance with it
    }
}

/// A Vulkan instance, destroyed on drop: every `Gpu::open` makes one, and before 0.5.0 none was ever destroyed.
struct Instance {
    inst: P,
    destroy: unsafe extern "C" fn(P, *const c_void),
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: the instance was created by `vulkan::instance` and is destroyed once, here, after its device.
        unsafe { (self.destroy)(self.inst, std::ptr::null()) }
    }
}

/// One GPU, ready for compute.
///
/// `Send` but not `Sync`: its Vulkan calls come from one thread at a time. On drop it waits for its work and destroys
/// its command pool and fence; the device itself goes with the last object made on it.
pub struct Gpu {
    pub name: String,
    dev: std::sync::Arc<Device>,
    queue: P,
    mem_type: u32,
    /// Size of the heap bankML's buffers come from; the GPU limiter's budget, against [`Gpu::allocated`].
    pub heap_bytes: u64,
    cpool: H,
    cb: P,
    fence: H,
}

/// A host-visible storage buffer, mapped for its whole life; destroyed, and its memory freed, on drop.
///
/// `Send`, not `Sync`: it is written and read only through a `Gpu` of its own device.
pub struct Buffer {
    dev: std::sync::Arc<Device>,
    buf: H,
    mem: H,
    ptr: *mut u8,
    pub bytes: usize,
    /// The memory the driver allocated for it (≥ `bytes`).
    alloc: u64,
}

/// A compute pipeline: its layout, single descriptor set and push-constant size; destroyed on drop.
pub struct Pipeline {
    dev: std::sync::Arc<Device>,
    pipe: H,
    layout: H,
    dsl: H,
    pool: H,
    set: H,
    bindings: u32,
    push_bytes: u32,
}

impl Gpu {
    /// Open Vulkan physical device `index` (the index `bankml gpu` reports) for compute.
    pub fn open(index: usize) -> Result<Gpu, String> {
        // SAFETY: each entry point is cast to its Vulkan signature; create-info structs and out-pointers outlive
        // each call; `pd` comes from this instance and `dev` from `pd`.
        unsafe {
            let (gipa, inst) = vulkan::instance()?;
            // owned from here: an early return below destroys it
            let instance = Instance { inst, destroy: std::mem::transmute_copy(&gipa(inst, c"vkDestroyInstance".as_ptr()).ok_or("no vkDestroyInstance")?) };
            let get = |name: &CStr| -> Result<Pfn, String> {
                gipa(inst, name.as_ptr()).ok_or_else(|| format!("no {}", name.to_string_lossy()))
            };
            macro_rules! f { ($n:literal) => { std::mem::transmute_copy(&get($n)?) } }
            let enum_pd: unsafe extern "C" fn(P, *mut u32, *mut P) -> i32 = f!(c"vkEnumeratePhysicalDevices");
            let qfp: unsafe extern "C" fn(P, *mut u32, *mut QueueFamilyProperties) = f!(c"vkGetPhysicalDeviceQueueFamilyProperties");
            let memp: unsafe extern "C" fn(P, *mut MemoryProperties) = f!(c"vkGetPhysicalDeviceMemoryProperties");
            let create_device: unsafe extern "C" fn(P, *const DeviceCreateInfo, *const c_void, *mut P) -> i32 = f!(c"vkCreateDevice");
            let get_queue: unsafe extern "C" fn(P, u32, u32, *mut P) = f!(c"vkGetDeviceQueue");
            let mut n = 0u32;
            enum_pd(inst, &mut n, std::ptr::null_mut());
            let mut pds = vec![std::ptr::null_mut(); n as usize];
            check(enum_pd(inst, &mut n, pds.as_mut_ptr()), "vkEnumeratePhysicalDevices")?;
            let pd = *pds.get(index).ok_or_else(|| format!("no Vulkan device {index}"))?;
            let name = vulkan::devices()?.into_iter().find(|d| d.index == index).map(|d| d.name).unwrap_or_default();
            let mut nq = 0u32;
            qfp(pd, &mut nq, std::ptr::null_mut());
            let mut fams: Vec<QueueFamilyProperties> = (0..nq).map(|_| QueueFamilyProperties { flags: 0, count: 0, ts_bits: 0, gran: [0; 3] }).collect();
            qfp(pd, &mut nq, fams.as_mut_ptr());
            let family = fams.iter().position(|q| q.flags & 0x2 != 0).ok_or("no compute queue")? as u32;
            let mut mp: MemoryProperties = std::mem::zeroed();
            memp(pd, &mut mp);
            let want = |need: u32| (0..mp.type_count).find(|&i| mp.types[i as usize].flags & need == need);
            let mem_type = want(MEM_DEVICE_LOCAL | MEM_HOST_VISIBLE | MEM_HOST_COHERENT)
                .or_else(|| want(MEM_HOST_VISIBLE | MEM_HOST_COHERENT))
                .ok_or("no host-visible coherent memory")?;
            let prio = 1.0f32;
            let qci = DeviceQueueCreateInfo { s_type: 2, p_next: std::ptr::null(), flags: 0, family, count: 1, priorities: &prio };
            let dci = DeviceCreateInfo { s_type: 3, p_next: std::ptr::null(), flags: 0, queue_info_count: 1, queue_infos: &qci, layer_count: 0,
                                         layers: std::ptr::null(), ext_count: 0, exts: std::ptr::null(), features: std::ptr::null() };
            let fns = Fns {
                create_buffer: f!(c"vkCreateBuffer"), buffer_reqs: f!(c"vkGetBufferMemoryRequirements"), alloc: f!(c"vkAllocateMemory"),
                bind: f!(c"vkBindBufferMemory"), map: f!(c"vkMapMemory"), destroy_buffer: f!(c"vkDestroyBuffer"), free: f!(c"vkFreeMemory"),
                shader: f!(c"vkCreateShaderModule"), dsl: f!(c"vkCreateDescriptorSetLayout"), playout: f!(c"vkCreatePipelineLayout"),
                pipelines: f!(c"vkCreateComputePipelines"), dpool: f!(c"vkCreateDescriptorPool"), dsets: f!(c"vkAllocateDescriptorSets"),
                update: f!(c"vkUpdateDescriptorSets"), cpool: f!(c"vkCreateCommandPool"), cbufs: f!(c"vkAllocateCommandBuffers"),
                begin: f!(c"vkBeginCommandBuffer"), end: f!(c"vkEndCommandBuffer"), reset_cb: f!(c"vkResetCommandBuffer"),
                bind_pipeline: f!(c"vkCmdBindPipeline"), bind_sets: f!(c"vkCmdBindDescriptorSets"), push: f!(c"vkCmdPushConstants"),
                dispatch: f!(c"vkCmdDispatch"), fence: f!(c"vkCreateFence"), submit: f!(c"vkQueueSubmit"), wait: f!(c"vkWaitForFences"),
                reset_fence: f!(c"vkResetFences"), idle: f!(c"vkDeviceWaitIdle"),
                destroy_shader: f!(c"vkDestroyShaderModule"), destroy_dsl: f!(c"vkDestroyDescriptorSetLayout"),
                destroy_playout: f!(c"vkDestroyPipelineLayout"), destroy_pipeline: f!(c"vkDestroyPipeline"),
                destroy_dpool: f!(c"vkDestroyDescriptorPool"), destroy_cpool: f!(c"vkDestroyCommandPool"),
                destroy_fence: f!(c"vkDestroyFence"), destroy_device: f!(c"vkDestroyDevice"),
            };
            let mut dev = std::ptr::null_mut();
            check(create_device(pd, &dci, std::ptr::null(), &mut dev), "vkCreateDevice")?;
            let mut queue = std::ptr::null_mut();
            get_queue(dev, family, 0, &mut queue);
            // from here on the device is owned: an early return below destroys it with `dv`
            let dv = std::sync::Arc::new(Device { dev, fns, inst: instance, allocated: Default::default() });
            let fns = &dv.fns;
            let mut cpool = 0;
            let cpci = CommandPoolCreateInfo { s_type: 39, p_next: std::ptr::null(), flags: 0x2, family }; // RESET_COMMAND_BUFFER
            check((fns.cpool)(dev, &cpci, std::ptr::null(), &mut cpool), "vkCreateCommandPool")?;
            let mut cb = std::ptr::null_mut();
            let cbai = CommandBufferAllocateInfo { s_type: 40, p_next: std::ptr::null(), pool: cpool, level: 0, count: 1 };
            check((fns.cbufs)(dev, &cbai, &mut cb), "vkAllocateCommandBuffers")?;
            let mut fence = 0;
            check((fns.fence)(dev, &FenceCreateInfo { s_type: 8, p_next: std::ptr::null(), flags: 0 }, std::ptr::null(), &mut fence), "vkCreateFence")?;
            let heap_bytes = mp.heaps[mp.types[mem_type as usize].heap as usize].size;
            Ok(Gpu { name, dev: dv, queue, mem_type, heap_bytes, cpool, cb, fence })
        }
    }

    /// A mapped buffer of at least `bytes` (minimum 4), counted in [`Gpu::allocated`].
    pub fn buffer(&self, bytes: usize) -> Result<Buffer, String> {
        let bytes = bytes.max(4);
        // SAFETY: `self.dev` is live; the memory type is checked against the buffer's requirements before allocating,
        // and the mapping covers the whole allocation.
        unsafe {
            let mut buf = 0;
            let bci = BufferCreateInfo { s_type: 12, p_next: std::ptr::null(), flags: 0, size: bytes as u64, usage: 0x20, sharing: 0, family_count: 0, families: std::ptr::null() };
            check((self.dev.fns.create_buffer)(self.dev.dev, &bci, std::ptr::null(), &mut buf), "vkCreateBuffer")?;
            let mut req = MemoryRequirements { size: 0, alignment: 0, type_bits: 0 };
            (self.dev.fns.buffer_reqs)(self.dev.dev, buf, &mut req);
            if req.type_bits & (1 << self.mem_type) == 0 {
                return Err("the buffer cannot live in host-visible memory".into());
            }
            let mut mem = 0;
            let mai = MemoryAllocateInfo { s_type: 5, p_next: std::ptr::null(), size: req.size, type_index: self.mem_type };
            check((self.dev.fns.alloc)(self.dev.dev, &mai, std::ptr::null(), &mut mem), "vkAllocateMemory")?;
            check((self.dev.fns.bind)(self.dev.dev, buf, mem, 0), "vkBindBufferMemory")?;
            let mut ptr = std::ptr::null_mut();
            check((self.dev.fns.map)(self.dev.dev, mem, 0, u64::MAX, 0, &mut ptr), "vkMapMemory")?;
            self.dev.allocated.fetch_add(req.size, std::sync::atomic::Ordering::Relaxed);
            Ok(Buffer { dev: self.dev.clone(), buf, mem, ptr: ptr as *mut u8, bytes, alloc: req.size })
        }
    }

    /// A buffer holding `data`.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buffer, String> {
        let bytes = std::mem::size_of_val(data);
        let b = self.buffer(bytes)?;
        // SAFETY: the mapping is at least `bytes` long and `T` is plain data.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, b.ptr, bytes) };
        Ok(b)
    }

    /// Overwrite the start of `b` with `data`. Panics if `data` is larger than `b`.
    ///
    /// Must not be called while a pending submission uses `b`.
    pub fn write<T: Copy>(&self, b: &Buffer, data: &[T]) {
        assert!(std::sync::Arc::ptr_eq(&b.dev, &self.dev), "a buffer of another device");
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= b.bytes);
        // SAFETY: the mapping is at least `b.bytes` ≥ `bytes` long (asserted) and `T` is plain data.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, b.ptr, bytes) };
    }

    /// The first `n` f32 of `b`. Panics if `b` is shorter; call after [`Gpu::wait`] for the results.
    pub fn read_f32(&self, b: &Buffer, n: usize) -> Vec<f32> {
        assert!(n * 4 <= b.bytes);
        // SAFETY: the mapping holds at least `n` f32 (asserted), the memory is coherent, and any bits are a valid f32.
        let mut v = vec![0.0f32; n];
        unsafe { std::ptr::copy_nonoverlapping(b.ptr as *const f32, v.as_mut_ptr(), n) };
        v
    }

    /// Bytes bankML's buffers hold on this device now.
    pub fn allocated(&self) -> u64 {
        self.dev.allocated.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Destroy `b` and free its memory, which also unmaps it (what dropping it does). `b` must not be in use by a
    /// pending submission.
    pub fn free(&self, b: Buffer) {
        drop(b);
    }

    /// A compute pipeline from SPIR-V with `bindings` storage buffers (bindings 0…) and `push_bytes` of constants.
    pub fn pipeline(&self, spirv: &[u32], bindings: u32, push_bytes: u32) -> Result<Pipeline, String> {
        // SAFETY: `spirv` and every create-info struct outlive the calls that read them; handles come from `self.dev`.
        unsafe {
            let mut module = 0;
            let smci = ShaderModuleCreateInfo { s_type: 16, p_next: std::ptr::null(), flags: 0, code_size: spirv.len() * 4, code: spirv.as_ptr() };
            check((self.dev.fns.shader)(self.dev.dev, &smci, std::ptr::null(), &mut module), "vkCreateShaderModule")?;
            let bs: Vec<DescriptorSetLayoutBinding> = (0..bindings)
                .map(|b| DescriptorSetLayoutBinding { binding: b, dtype: STORAGE_BUFFER, count: 1, stages: STAGE_COMPUTE, samplers: std::ptr::null() })
                .collect();
            let mut dsl = 0;
            let dslci = DescriptorSetLayoutCreateInfo { s_type: 32, p_next: std::ptr::null(), flags: 0, count: bindings, bindings: bs.as_ptr() };
            check((self.dev.fns.dsl)(self.dev.dev, &dslci, std::ptr::null(), &mut dsl), "vkCreateDescriptorSetLayout")?;
            let pcr = PushConstantRange { stages: STAGE_COMPUTE, offset: 0, size: push_bytes };
            let mut layout = 0;
            let plci = PipelineLayoutCreateInfo { s_type: 30, p_next: std::ptr::null(), flags: 0, set_count: 1, sets: &dsl,
                                                  pc_count: (push_bytes > 0) as u32, pcs: &pcr };
            check((self.dev.fns.playout)(self.dev.dev, &plci, std::ptr::null(), &mut layout), "vkCreatePipelineLayout")?;
            let stage = PipelineShaderStageCreateInfo { s_type: 18, p_next: std::ptr::null(), flags: 0, stage: STAGE_COMPUTE, module,
                                                        name: c"main".as_ptr(), spec: std::ptr::null() };
            let cpci = ComputePipelineCreateInfo { s_type: 29, p_next: std::ptr::null(), flags: 0, stage, layout, base: 0, base_index: -1 };
            let mut pipe = 0;
            check((self.dev.fns.pipelines)(self.dev.dev, 0, 1, &cpci, std::ptr::null(), &mut pipe), "vkCreateComputePipelines")?;
            let size = DescriptorPoolSize { dtype: STORAGE_BUFFER, count: bindings };
            let mut pool = 0;
            let dpci = DescriptorPoolCreateInfo { s_type: 33, p_next: std::ptr::null(), flags: 0, max_sets: 1, size_count: 1, sizes: &size };
            check((self.dev.fns.dpool)(self.dev.dev, &dpci, std::ptr::null(), &mut pool), "vkCreateDescriptorPool")?;
            let mut set = 0;
            let dsai = DescriptorSetAllocateInfo { s_type: 34, p_next: std::ptr::null(), pool, count: 1, layouts: &dsl };
            check((self.dev.fns.dsets)(self.dev.dev, &dsai, &mut set), "vkAllocateDescriptorSets")?;
            // the module is compiled into the pipeline and no longer needed
            (self.dev.fns.destroy_shader)(self.dev.dev, module, std::ptr::null());
            Ok(Pipeline { dev: self.dev.clone(), pipe, layout, dsl, pool, set, bindings, push_bytes })
        }
    }

    /// Bind `bufs` to the pipeline's bindings in order, push `push`, dispatch `groups` workgroups, and wait.
    pub fn run(&self, p: &Pipeline, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<(), String> {
        self.submit(p, bufs, push, groups)?;
        self.wait()
    }

    /// Wait for the work `submit` started and reset the fence for the next submit.
    ///
    /// Call only after `submit`: the fence is otherwise never signalled and this blocks indefinitely.
    pub fn wait(&self) -> Result<(), String> {
        // SAFETY: the fence belongs to `self.dev`.
        unsafe {
            check((self.dev.fns.wait)(self.dev.dev, 1, &self.fence, 1, u64::MAX), "vkWaitForFences")?;
            check((self.dev.fns.reset_fence)(self.dev.dev, 1, &self.fence), "vkResetFences")
        }
    }

    /// Start `groups` workgroups of `p` on `bufs` and return at once; [`Gpu::wait`] completes it.
    ///
    /// One submission is in flight at a time (one command buffer, one fence, one descriptor set per pipeline): call
    /// `wait` before the next `submit`. Panics if `bufs` or `push` do not match the pipeline.
    pub fn submit(&self, p: &Pipeline, bufs: &[&Buffer], push: &[u32], groups: u32) -> Result<(), String> {
        assert_eq!(bufs.len() as u32, p.bindings);
        assert!(std::sync::Arc::ptr_eq(&p.dev, &self.dev) && bufs.iter().all(|b| std::sync::Arc::ptr_eq(&b.dev, &self.dev)),
                "a pipeline or buffer of another device");
        assert_eq!(push.len() as u32 * 4, p.push_bytes);
        // SAFETY: the buffers and pipeline belong to this device; the previous submission has been waited for, so the
        // command buffer may be reset and the descriptor set rewritten; `infos`, `writes` and `si` outlive the calls.
        unsafe {
            let infos: Vec<DescriptorBufferInfo> = bufs.iter().map(|b| DescriptorBufferInfo { buffer: b.buf, offset: 0, range: u64::MAX }).collect();
            let writes: Vec<WriteDescriptorSet> = infos.iter().enumerate()
                .map(|(i, info)| WriteDescriptorSet { s_type: 35, p_next: std::ptr::null(), set: p.set, binding: i as u32, element: 0, count: 1,
                                                      dtype: STORAGE_BUFFER, images: std::ptr::null(), buffers: info, views: std::ptr::null() })
                .collect();
            (self.dev.fns.update)(self.dev.dev, writes.len() as u32, writes.as_ptr(), 0, std::ptr::null());
            check((self.dev.fns.reset_cb)(self.cb, 0), "vkResetCommandBuffer")?;
            check((self.dev.fns.begin)(self.cb, &CommandBufferBeginInfo { s_type: 42, p_next: std::ptr::null(), flags: 1, inheritance: std::ptr::null() }), "vkBeginCommandBuffer")?;
            (self.dev.fns.bind_pipeline)(self.cb, BIND_POINT_COMPUTE, p.pipe);
            (self.dev.fns.bind_sets)(self.cb, BIND_POINT_COMPUTE, p.layout, 0, 1, &p.set, 0, std::ptr::null());
            if p.push_bytes > 0 {
                (self.dev.fns.push)(self.cb, p.layout, STAGE_COMPUTE, 0, p.push_bytes, push.as_ptr() as *const c_void);
            }
            (self.dev.fns.dispatch)(self.cb, groups, 1, 1);
            check((self.dev.fns.end)(self.cb), "vkEndCommandBuffer")?;
            let si = SubmitInfo { s_type: 4, p_next: std::ptr::null(), wait_count: 0, waits: std::ptr::null(), wait_stages: std::ptr::null(),
                                  cb_count: 1, cbs: &self.cb, signal_count: 0, signals: std::ptr::null() };
            check((self.dev.fns.submit)(self.queue, 1, &si, self.fence), "vkQueueSubmit")
        }
    }
}

// SAFETY: Vulkan objects are not tied to the thread that made them. Every call on a `Gpu` comes from one thread at a
// time (`Gpu` is not `Sync`; the forward pass holds the worker behind a Mutex), and a `Buffer` is reached only through a
// `Gpu` of its own device (asserted), so it is `Send` but not `Sync`. `Device` is shared by `Arc`: its handle and
// entry points never change after creation, its counter is atomic, and Vulkan lets different objects of one device be
// created and destroyed from different threads.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}
unsafe impl Send for Gpu {}
unsafe impl Send for Buffer {}
unsafe impl Send for Pipeline {}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: waiting for the device first means the command buffer and fence are no longer in use; both were made
        // on this device and are destroyed once, here. The device itself goes with its last object.
        unsafe {
            (self.dev.fns.idle)(self.dev.dev);
            (self.dev.fns.destroy_fence)(self.dev.dev, self.fence, std::ptr::null());
            (self.dev.fns.destroy_cpool)(self.dev.dev, self.cpool, std::ptr::null());
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.dev.allocated.fetch_sub(self.alloc, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: made on `self.dev`, destroyed exactly once, here; freeing the memory also unmaps it. The caller keeps
        // a buffer alive while a submission uses it (`Gpu::submit` takes it by reference until `wait`).
        unsafe {
            (self.dev.fns.destroy_buffer)(self.dev.dev, self.buf, std::ptr::null());
            (self.dev.fns.free)(self.dev.dev, self.mem, std::ptr::null());
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // SAFETY: made on `self.dev` and destroyed once, here; destroying the pool frees its descriptor set.
        unsafe {
            (self.dev.fns.destroy_pipeline)(self.dev.dev, self.pipe, std::ptr::null());
            (self.dev.fns.destroy_playout)(self.dev.dev, self.layout, std::ptr::null());
            (self.dev.fns.destroy_dpool)(self.dev.dev, self.pool, std::ptr::null());
            (self.dev.fns.destroy_dsl)(self.dev.dev, self.dsl, std::ptr::null());
        }
    }
}
