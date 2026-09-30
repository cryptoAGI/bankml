// SPDX-License-Identifier: MIT OR Apache-2.0
//! The video-card component: find every GPU on the machine, describe it, and (in later steps) run bankml's kernels
//! on all of them. Modular and extensible: each backend is a module with one entry point, `devices()`, listed in
//! [`BACKENDS`]; a new one (CUDA, ROCm, Metal, …) is a new module and one line there. Nothing is linked at build
//! time: a backend opens its driver library at run time (`dlopen`), so bankml builds and runs everywhere and a
//! machine without the library simply has no devices from that backend.
//!
//! What bankml asks of a card, in order: it must be a real GPU (integrated or discrete; software renderers such as
//! Mesa's llvmpipe run on the CPU and are refused), it must have a compute queue, and — when kernels land — its
//! results must be the CPU kernels' bits, checked on the card before it is trusted (the oracle rule: same bits
//! first, then speed). Several cards are used together by giving each a share of every matrix's rows, which keeps
//! each output element one card's exact dot product.
//!
//! Remote cards are a separate kind. The Hugging Face backend (`hf.rs`) lists the GPU hardware Hugging Face rents
//! by the minute — Nvidia T4 to 8× H200, provisioned on AWS, Azure and GCP — with each flavor's card count, memory
//! and price. A remote card is never selected automatically (it costs money): using one means running bankml there
//! (a Hugging Face Job), where the Vulkan backend finds the card like any local one.
//!
//! The kernel's own view (`/sys/class/drm/card*/device`: driver, VRAM, GTT, PCI address) is merged with the
//! backend's, so a card is described by what the driver says, not only by what the API reports.

pub mod compute;
pub mod hf;
pub mod kernels;
pub mod spirv;
pub mod vulkan;
pub mod worker;

use std::path::PathBuf;

/// What kind of device a backend reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Integrated,
    Discrete,
    Virtual,
    /// a software renderer running on the CPU (Mesa llvmpipe, lavapipe): never used
    Cpu,
    /// hardware rented from a provider (Hugging Face Jobs); listed, never started automatically
    Remote,
    Other,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Integrated => "integrated",
            Kind::Discrete => "discrete",
            Kind::Virtual => "virtual",
            Kind::Cpu => "cpu (software renderer)",
            Kind::Remote => "remote (rented)",
            Kind::Other => "other",
        }
    }
}

/// The kernel's description of a DRM card (Linux sysfs).
#[derive(Debug, Clone, Default)]
pub struct Sysfs {
    pub card: String,
    pub pci: String,
    pub driver: String,
    pub vram_total: Option<u64>,
    pub gtt_total: Option<u64>,
    pub numa_node: Option<i32>,
}

/// One GPU as bankml sees it.
#[derive(Debug, Clone)]
pub struct Device {
    pub backend: &'static str,
    /// the backend's own index (Vulkan: the physical-device index)
    pub index: usize,
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub kind: Kind,
    /// the API version the device supports, e.g. "1.3.255"
    pub api: String,
    /// memory the device can address fastest (device-local heaps), in bytes
    pub device_local_bytes: u64,
    /// device-local memory the CPU can map directly (integrated memory, resizable BAR): weights need no copy
    pub host_visible_device_local: bool,
    pub compute_queues: u32,
    pub sysfs: Option<Sysfs>,
    /// cards in this device (a multi-card flavor such as 8× H200 is one entry with count 8); 1 for a local card
    pub count: u32,
    /// the price per hour in US dollars, for rented hardware
    pub usd_per_hour: Option<f64>,
}

impl Device {
    /// A device bankml will put to work: a real GPU with a compute queue.
    pub fn usable(&self) -> bool {
        matches!(self.kind, Kind::Integrated | Kind::Discrete) && self.compute_queues > 0
    }

    pub fn vendor_name(&self) -> &'static str {
        match self.vendor {
            0x1002 => "AMD",
            0x10de => "NVIDIA",
            0x8086 => "Intel",
            0x13b5 => "Arm",
            0x5143 => "Qualcomm",
            0x106b => "Apple",
            0x10005 => "Mesa",
            _ => "unknown",
        }
    }
}

/// A backend's discovery function: every device it finds, or why it found none.
pub type Discover = fn() -> Result<Vec<Device>, String>;

/// Every local backend bankml knows: a name and its discovery function. Add a backend here.
pub const BACKENDS: &[(&str, Discover)] = &[("vulkan", vulkan::devices)];

/// Remote backends: asked only on request (they use the network), never selected automatically.
pub const REMOTE_BACKENDS: &[(&str, Discover)] = &[("huggingface", hf::devices)];

/// Every device every backend finds, with the kernel's view attached, and the reasons a backend found none.
pub fn discover() -> (Vec<Device>, Vec<String>) {
    let cards = sysfs_cards();
    let (mut all, mut notes) = (Vec::new(), Vec::new());
    for (name, f) in BACKENDS {
        match f() {
            Ok(mut ds) => {
                for d in ds.iter_mut() {
                    d.sysfs = cards.iter().find(|(v, id, _)| *v == d.vendor && *id == d.device).map(|c| c.2.clone());
                }
                all.extend(ds);
            }
            Err(e) => notes.push(format!("{name}: {e}")),
        }
    }
    (all, notes)
}

/// The devices bankml will use, in the order it will use them: discrete cards first (largest memory first), then
/// integrated ones. `BANKML_GPU=off` turns the component off; `BANKML_GPU=0,2` picks backend indices.
pub fn selected(devs: &[Device]) -> Vec<Device> {
    let env = std::env::var("BANKML_GPU").unwrap_or_default();
    if matches!(env.trim(), "off" | "none" | "cpu") {
        return Vec::new();
    }
    let pick: Option<Vec<usize>> = (!env.trim().is_empty()).then(|| env.split(',').filter_map(|s| s.trim().parse().ok()).collect());
    let mut v: Vec<Device> = devs.iter().filter(|d| d.usable() && pick.as_ref().is_none_or(|p| p.contains(&d.index))).cloned().collect();
    v.sort_by_key(|d| (d.kind != Kind::Discrete, std::cmp::Reverse(d.device_local_bytes)));
    v
}

/// DRM cards from sysfs: (vendor id, device id, description).
fn sysfs_cards() -> Vec<(u32, u32, Sysfs)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir("/sys/class/drm") else { return out };
    let mut names: Vec<String> = rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    for card in names.into_iter().filter(|n| n.starts_with("card") && !n.contains('-')) {
        let dev = PathBuf::from("/sys/class/drm").join(&card).join("device");
        let read = |f: &str| std::fs::read_to_string(dev.join(f)).ok().map(|s| s.trim().to_string());
        let hex = |f: &str| read(f).and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok());
        let (Some(vendor), Some(device)) = (hex("vendor"), hex("device")) else { continue };
        let link = |f: &str| std::fs::read_link(dev.join(f)).ok().and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()));
        let pci = std::fs::canonicalize(&dev).ok().and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned())).unwrap_or_default();
        out.push((vendor, device, Sysfs {
            card,
            pci,
            driver: link("driver").unwrap_or_default(),
            vram_total: read("mem_info_vram_total").and_then(|s| s.parse().ok()),
            gtt_total: read("mem_info_gtt_total").and_then(|s| s.parse().ok()),
            numa_node: read("numa_node").and_then(|s| s.parse().ok()),
        }));
    }
    out
}

/// A JSON report of what was found and what will be used (`bankml gpu`); with `remote`, the rented hardware too.
pub fn report_json(remote: bool) -> String {
    let (mut devs, mut notes) = discover();
    if remote {
        for (name, f) in REMOTE_BACKENDS {
            match f() {
                Ok(ds) => devs.extend(ds),
                Err(e) => notes.push(format!("{name}: {e}")),
            }
        }
    }
    let sel = selected(&devs);
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let opt = |v: Option<u64>| v.map(|x| x.to_string()).unwrap_or_else(|| "null".into());
    let one = |d: &Device| {
        let sys = d.sysfs.as_ref().map(|s| format!(
            "{{\"card\": \"{}\", \"pci\": \"{}\", \"driver\": \"{}\", \"vram_total\": {}, \"gtt_total\": {}, \"numa_node\": {}}}",
            esc(&s.card), esc(&s.pci), esc(&s.driver), opt(s.vram_total), opt(s.gtt_total),
            s.numa_node.map(|n| n.to_string()).unwrap_or_else(|| "null".into()))).unwrap_or_else(|| "null".into());
        format!("{{\"backend\": \"{}\", \"index\": {}, \"name\": \"{}\", \"vendor\": \"{} (0x{:04x})\", \"device\": \"0x{:04x}\", \"kind\": \"{}\", \"api\": \"{}\", \
                 \"device_local_bytes\": {}, \"host_visible_device_local\": {}, \"compute_queues\": {}, \"count\": {}, \"usd_per_hour\": {}, \"usable\": {}, \"sysfs\": {}}}",
                d.backend, d.index, esc(&d.name), d.vendor_name(), d.vendor, d.device, d.kind.name(), esc(&d.api), d.device_local_bytes,
                d.host_visible_device_local, d.compute_queues, d.count, d.usd_per_hour.map(|p| format!("{p:.2}")).unwrap_or_else(|| "null".into()),
                d.usable(), sys)
    };
    format!("{{\"devices\": [{}], \"selected\": [{}], \"notes\": [{}], \"kernels\": \"not yet: bankml's matmuls run on the CPU until the GPU kernels pass their bit-exact oracle\"}}",
            devs.iter().map(one).collect::<Vec<_>>().join(", "),
            sel.iter().map(|d| format!("\"{}:{} {}\"", d.backend, d.index, esc(&d.name))).collect::<Vec<_>>().join(", "),
            notes.iter().map(|n| format!("\"{}\"", esc(n))).collect::<Vec<_>>().join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(kind: Kind, mem: u64, index: usize) -> Device {
        Device { backend: "test", index, name: format!("d{index}"), vendor: 0x1002, device: 1, kind, api: "1.3".into(),
                 device_local_bytes: mem, host_visible_device_local: false, compute_queues: 1, sysfs: None, count: 1, usd_per_hour: None }
    }

    #[test]
    fn software_renderers_are_never_selected_and_discrete_cards_come_first() {
        let ds = vec![dev(Kind::Cpu, 1 << 34, 0), dev(Kind::Integrated, 2 << 30, 1), dev(Kind::Discrete, 8 << 30, 2), dev(Kind::Discrete, 24 << 30, 3),
                      dev(Kind::Remote, 1128 << 30, 4)];
        let s = selected(&ds);
        assert_eq!(s.iter().map(|d| d.index).collect::<Vec<_>>(), vec![3, 2, 1], "a rented card is never selected automatically");
    }

    #[test]
    fn discovery_never_panics() {
        let (ds, _) = discover();
        for d in &ds {
            assert!(!d.name.is_empty());
        }
    }
}
