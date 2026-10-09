// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Vulkan backend: enumerates every GPU a Vulkan driver exposes.
//!
//! The loader is opened at run time with `dlopen` and every entry point comes from `vkGetInstanceProcAddr`; the
//! few C structs used are declared here from the Vulkan 1.x headers. Without a loader there are no Vulkan devices.
//! Details: docs/modules/gpu.md.

use super::{Device, Kind};
use std::ffi::{c_char, c_void, CStr};

type VkResult = i32;
type Pfn = unsafe extern "C" fn();
pub(super) type GetInstanceProcAddr = unsafe extern "C" fn(*mut c_void, *const c_char) -> Option<Pfn>;

#[cfg(not(target_family = "wasm"))]
#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
// WebAssembly (browseML) has no Vulkan loader to open: no card, and the CPU path stays as it is
#[cfg(target_family = "wasm")]
unsafe fn dlopen(_filename: *const c_char, _flag: i32) -> *mut c_void {
    std::ptr::null_mut()
}
#[cfg(target_family = "wasm")]
unsafe fn dlsym(_handle: *mut c_void, _symbol: *const c_char) -> *mut c_void {
    std::ptr::null_mut()
}
const RTLD_NOW: i32 = 2;

#[repr(C)]
struct ApplicationInfo {
    s_type: i32,
    p_next: *const c_void,
    app_name: *const c_char,
    app_version: u32,
    engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct InstanceCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    app_info: *const ApplicationInfo,
    layer_count: u32,
    layers: *const *const c_char,
    ext_count: u32,
    exts: *const *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct QueueFamilyProperties {
    flags: u32,
    count: u32,
    timestamp_bits: u32,
    granularity: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryType {
    flags: u32,
    heap: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryHeap {
    size: u64,
    flags: u32,
}

#[repr(C)]
struct MemoryProperties {
    type_count: u32,
    types: [MemoryType; 32],
    heap_count: u32,
    heaps: [MemoryHeap; 16],
}

const STYPE_APPLICATION_INFO: i32 = 0;
const STYPE_INSTANCE_CREATE_INFO: i32 = 1;
const QUEUE_COMPUTE: u32 = 0x2;
const MEM_DEVICE_LOCAL: u32 = 0x1;
const MEM_HOST_VISIBLE: u32 = 0x2;
const HEAP_DEVICE_LOCAL: u32 = 0x1;
const fn make_version(major: u32, minor: u32) -> u32 {
    (major << 22) | (minor << 12)
}

/// The loader's `vkGetInstanceProcAddr`, or why it is not there.
fn loader() -> Result<GetInstanceProcAddr, String> {
    for name in [c"libvulkan.so.1", c"libvulkan.so", c"libvulkan.1.dylib", c"libMoltenVK.dylib"] {
        // SAFETY: dlopen and dlsym get NUL-terminated names; `vkGetInstanceProcAddr` has the documented signature.
        let h = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            let f = unsafe { dlsym(h, c"vkGetInstanceProcAddr".as_ptr()) };
            if !f.is_null() {
                return Ok(unsafe { std::mem::transmute::<*mut c_void, GetInstanceProcAddr>(f) });
            }
        }
    }
    Err("no Vulkan loader (libvulkan.so.1) on this machine".into())
}

/// An instance-level function by name, cast to its signature.
///
/// # Safety
/// `T` must be the function-pointer type of `name`'s Vulkan signature; `inst` must be NULL (for global functions)
/// or a live instance.
unsafe fn proc<T>(gipa: GetInstanceProcAddr, inst: *mut c_void, name: &CStr) -> Result<T, String> {
    let f = gipa(inst, name.as_ptr()).ok_or_else(|| format!("the Vulkan loader has no {}", name.to_string_lossy()))?;
    Ok(std::mem::transmute_copy::<Pfn, T>(&f))
}

/// A new Vulkan instance for compute, never destroyed, and the loader's `vkGetInstanceProcAddr`.
pub(super) fn instance() -> Result<(GetInstanceProcAddr, *mut c_void), String> {
    let gipa = loader()?;
    // SAFETY: each pointer is cast to its Vulkan signature; the create-info structs outlive the call.
    unsafe {
        let create: unsafe extern "C" fn(*const InstanceCreateInfo, *const c_void, *mut *mut c_void) -> VkResult =
            proc(gipa, std::ptr::null_mut(), c"vkCreateInstance")?;
        let app = ApplicationInfo { s_type: STYPE_APPLICATION_INFO, p_next: std::ptr::null(), app_name: c"bankml".as_ptr(), app_version: 1,
                                    engine_name: c"bankml".as_ptr(), engine_version: 1, api_version: make_version(1, 1) };
        let ci = InstanceCreateInfo { s_type: STYPE_INSTANCE_CREATE_INFO, p_next: std::ptr::null(), flags: 0, app_info: &app,
                                      layer_count: 0, layers: std::ptr::null(), ext_count: 0, exts: std::ptr::null() };
        let mut inst = std::ptr::null_mut();
        let r = create(&ci, std::ptr::null(), &mut inst);
        if r != 0 || inst.is_null() {
            return Err(format!("vkCreateInstance failed ({r})"));
        }
        Ok((gipa, inst))
    }
}

/// Every physical device the Vulkan loader enumerates, from a temporary instance.
pub fn devices() -> Result<Vec<Device>, String> {
    let gipa = loader()?;
    // SAFETY: as in `instance`; the instance is destroyed only after `enumerate` is done with it.
    unsafe {
        let create: unsafe extern "C" fn(*const InstanceCreateInfo, *const c_void, *mut *mut c_void) -> VkResult =
            proc(gipa, std::ptr::null_mut(), c"vkCreateInstance")?;
        let app = ApplicationInfo { s_type: STYPE_APPLICATION_INFO, p_next: std::ptr::null(), app_name: c"bankml".as_ptr(), app_version: 1,
                                    engine_name: c"bankml".as_ptr(), engine_version: 1, api_version: make_version(1, 1) };
        let ci = InstanceCreateInfo { s_type: STYPE_INSTANCE_CREATE_INFO, p_next: std::ptr::null(), flags: 0, app_info: &app,
                                      layer_count: 0, layers: std::ptr::null(), ext_count: 0, exts: std::ptr::null() };
        let mut inst = std::ptr::null_mut();
        let r = create(&ci, std::ptr::null(), &mut inst);
        if r != 0 || inst.is_null() {
            return Err(format!("vkCreateInstance failed ({r})"));
        }
        let destroy: unsafe extern "C" fn(*mut c_void, *const c_void) = proc(gipa, inst, c"vkDestroyInstance")?;
        let out = enumerate(gipa, inst);
        destroy(inst, std::ptr::null());
        out
    }
}

/// Describe each physical device of `inst`.
///
/// # Safety
/// `inst` must be a live instance obtained through `gipa`.
unsafe fn enumerate(gipa: GetInstanceProcAddr, inst: *mut c_void) -> Result<Vec<Device>, String> {
    let enum_pd: unsafe extern "C" fn(*mut c_void, *mut u32, *mut *mut c_void) -> VkResult = proc(gipa, inst, c"vkEnumeratePhysicalDevices")?;
    let props: unsafe extern "C" fn(*mut c_void, *mut u8) = proc(gipa, inst, c"vkGetPhysicalDeviceProperties")?;
    let mem: unsafe extern "C" fn(*mut c_void, *mut MemoryProperties) = proc(gipa, inst, c"vkGetPhysicalDeviceMemoryProperties")?;
    let qf: unsafe extern "C" fn(*mut c_void, *mut u32, *mut QueueFamilyProperties) = proc(gipa, inst, c"vkGetPhysicalDeviceQueueFamilyProperties")?;
    let mut n = 0u32;
    enum_pd(inst, &mut n, std::ptr::null_mut());
    let mut pds = vec![std::ptr::null_mut(); n as usize];
    let r = enum_pd(inst, &mut n, pds.as_mut_ptr());
    if r < 0 {
        return Err(format!("vkEnumeratePhysicalDevices failed ({r})"));
    }
    let mut out = Vec::new();
    for (index, &pd) in pds.iter().take(n as usize).enumerate() {
        // VkPhysicalDeviceProperties begins apiVersion, driverVersion, vendorID, deviceID, deviceType,
        // deviceName[256]; the 4 KiB buffer is larger than the whole struct.
        let mut p = vec![0u8; 4096];
        props(pd, p.as_mut_ptr());
        let u = |o: usize| u32::from_le_bytes(p[o..o + 4].try_into().unwrap());
        let (api, vendor, device, dtype) = (u(0), u(8), u(12), u(16) as i32);
        let name_bytes = &p[20..276];
        let name = String::from_utf8_lossy(&name_bytes[..name_bytes.iter().position(|&b| b == 0).unwrap_or(256)]).into_owned();
        let mut m: MemoryProperties = std::mem::zeroed();
        mem(pd, &mut m);
        let heaps = &m.heaps[..(m.heap_count as usize).min(16)];
        let device_local_bytes = heaps.iter().filter(|h| h.flags & HEAP_DEVICE_LOCAL != 0).map(|h| h.size).sum();
        let host_visible_device_local = m.types[..(m.type_count as usize).min(32)]
            .iter()
            .any(|t| t.flags & (MEM_DEVICE_LOCAL | MEM_HOST_VISIBLE) == (MEM_DEVICE_LOCAL | MEM_HOST_VISIBLE));
        let mut nq = 0u32;
        qf(pd, &mut nq, std::ptr::null_mut());
        let mut fams = vec![QueueFamilyProperties::default(); nq as usize];
        qf(pd, &mut nq, fams.as_mut_ptr());
        let compute_queues = fams.iter().filter(|f| f.flags & QUEUE_COMPUTE != 0).map(|f| f.count).sum();
        out.push(Device {
            backend: "vulkan",
            index,
            name,
            vendor,
            device,
            kind: match dtype {
                1 => Kind::Integrated,
                2 => Kind::Discrete,
                3 => Kind::Virtual,
                4 => Kind::Cpu,
                _ => Kind::Other,
            },
            api: format!("{}.{}.{}", api >> 22, (api >> 12) & 0x3ff, api & 0xfff),
            device_local_bytes,
            host_visible_device_local,
            compute_queues,
            sysfs: None,
            count: 1,
            usd_per_hour: None,
        });
    }
    Ok(out)
}
