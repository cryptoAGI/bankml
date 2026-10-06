// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Hugging Face backend: the GPU flavors Hugging Face Jobs rents, as remote devices.
//!
//! The public hardware list (no token) is fetched with the system `curl`, since bankML has no TLS stack; without
//! `curl` or the network the backend returns the reason. Nothing is started, rented or paid for.
//! Details: docs/modules/gpu.md.

use super::{Device, Kind};
use crate::serve::Json;

pub const HARDWARE_URL: &str = "https://huggingface.co/api/jobs/hardware";

/// Fetch the hardware list and return its GPU flavors as remote devices.
pub fn devices() -> Result<Vec<Device>, String> {
    let out = std::process::Command::new("curl")
        .args(["-fsS", "--max-time", "20", "-A", "bankml", HARDWARE_URL])
        .output()
        .map_err(|e| format!("curl is needed to list Hugging Face hardware: {e}"))?;
    if !out.status.success() {
        return Err(format!("{HARDWARE_URL}: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    parse(&String::from_utf8_lossy(&out.stdout))
}

/// Parse the hardware list's JSON into devices, GPU flavors only.
///
/// A multi-card flavor is one device with its `count`; the price per hour is set only for a per-minute price.
pub fn parse(text: &str) -> Result<Vec<Device>, String> {
    let Some(Json::Arr(items)) = Json::parse(text) else { return Err("the hardware list is not a JSON array".into()) };
    let s = |v: Option<&Json>| v.and_then(Json::as_str).unwrap_or("").to_string();
    let mut out = Vec::new();
    for (index, it) in items.iter().enumerate() {
        let Some(acc) = it.get("accelerator").filter(|a| !matches!(a, Json::Null)) else { continue };
        if s(acc.get("type")) != "gpu" {
            continue;
        }
        let count: u32 = s(acc.get("quantity")).parse().unwrap_or(1);
        let vram_gb: f64 = s(acc.get("vram")).split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let per_min = match it.get("unitCostUSD") { Some(Json::Num(n)) => Some(*n), _ => None };
        let per_hour = per_min.filter(|_| s(it.get("unitLabel")) == "minute").map(|m| m * 60.0);
        let maker = s(acc.get("manufacturer"));
        out.push(Device {
            backend: "huggingface",
            index,
            name: format!("{} · {} ({})", s(it.get("name")), s(it.get("prettyName")), s(acc.get("model"))),
            vendor: match maker.as_str() { "Nvidia" | "NVIDIA" => 0x10de, "AMD" => 0x1002, "Intel" => 0x8086, _ => 0 },
            device: 0,
            kind: Kind::Remote,
            api: "Hugging Face Jobs".into(),
            device_local_bytes: (vram_gb * 1e9) as u64,
            host_visible_device_local: false,
            compute_queues: 0,
            sysfs: None,
            count,
            usd_per_hour: per_hour,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_multi_card_flavor_is_one_device_with_its_count() {
        let j = r#"[{"name":"cpu-basic","accelerator":null,"unitCostUSD":0.000167,"unitLabel":"minute"},
                    {"name":"h200x8","prettyName":"Nvidia H200","accelerator":{"type":"gpu","model":"H200","quantity":"8","vram":"1128 GB","manufacturer":"Nvidia"},
                     "unitCostUSD":0.6666667,"unitLabel":"minute"}]"#;
        let d = parse(j).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!((d[0].count, d[0].vendor, d[0].kind), (8, 0x10de, Kind::Remote));
        assert_eq!(d[0].device_local_bytes, 1_128_000_000_000);
        assert!((d[0].usd_per_hour.unwrap() - 40.0).abs() < 1e-3);
        assert!(!d[0].usable());
    }
}
