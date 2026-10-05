// SPDX-License-Identifier: MIT OR Apache-2.0
//! System and process readings, the part of psutil (or Rust's `sysinfo`) bankml needs, with no crates: on Linux
//! everything comes from `/proc`, which is what those libraries read too. Used by `bankml serve` (`GET
//! /bankml/usage`) and `bankml usage`, and by the Savante UI's Resources sliders.
//!
//! - memory: `/proc/meminfo` (MemTotal, MemAvailable, SwapTotal, SwapFree);
//! - a process: `/proc/<pid>/status` (VmRSS) and `/proc/<pid>/stat` (utime + stime, in clock ticks);
//! - CPU %: the change in a process's ticks over a sampling interval, divided by the ticks per second (`sysconf`),
//!   so 100 % is one core busy and `cores × 100 %` is the machine;
//! - energy (0.3.7): the CPU package's RAPL counter (`/sys/class/powercap/intel-rapl:0/energy_uj`, which AMD exposes
//!   too; it includes an APU's GPU), its wrap at `max_energy_range_uj` handled; root-only unless `./install.sh power`
//!   has run, and then `null`, never estimated;
//! - GPUs (0.3.7): `/sys/class/drm/card*/device` — busy %, VRAM and GTT used and total, where the driver exposes them
//!   (amdgpu does; others read as `null`).

use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Memory {
    pub total: u64,
    pub available: u64,
    pub swap_total: u64,
    pub swap_free: u64,
}

/// One kB-valued field of a `/proc` key: value table.
fn field_kb(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|l| {
        let rest = l.strip_prefix(key)?.strip_prefix(':')?;
        rest.split_whitespace().next()?.parse::<u64>().ok().map(|kb| kb * 1024)
    })
}

pub fn memory() -> Option<Memory> {
    let t = std::fs::read_to_string("/proc/meminfo").ok()?;
    Some(Memory {
        total: field_kb(&t, "MemTotal")?,
        available: field_kb(&t, "MemAvailable")?,
        swap_total: field_kb(&t, "SwapTotal").unwrap_or(0),
        swap_free: field_kb(&t, "SwapFree").unwrap_or(0),
    })
}

pub fn cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Resident memory of a process in bytes.
pub fn rss(pid: u32) -> Option<u64> {
    field_kb(&std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?, "VmRSS")
}

/// utime + stime of a process, in clock ticks. The command name (field 2) may contain spaces and parentheses, so
/// fields are counted after the last ')'.
pub fn cpu_ticks(pid: u32) -> Option<u64> {
    parse_ticks(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

fn parse_ticks(stat: &str) -> Option<u64> {
    let rest = &stat[stat.rfind(')')? + 1..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    // after ')': state(0) ppid(1) … utime is field 14 of the full line = index 11 here, stime index 12
    Some(f.get(11)?.parse::<u64>().ok()? + f.get(12)?.parse::<u64>().ok()?)
}

/// Clock ticks per second (`sysconf(_SC_CLK_TCK)`; 100 on Linux in practice).
pub fn ticks_per_second() -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::os::raw::{c_int, c_long};
        extern "C" {
            fn sysconf(name: c_int) -> c_long;
        }
        let v = unsafe { sysconf(2) } as i64; // _SC_CLK_TCK
        if v > 0 {
            return v as u64;
        }
    }
    100
}

pub fn alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// CPU % of a set of processes over `interval` (100 % = one core).
pub fn cpu_percent(pids: &[u32], interval: Duration) -> f64 {
    let sum = |pids: &[u32]| pids.iter().filter_map(|&p| cpu_ticks(p)).sum::<u64>();
    let (a, t) = (sum(pids), Instant::now());
    std::thread::sleep(interval);
    let b = sum(pids);
    let secs = t.elapsed().as_secs_f64().max(1e-3);
    b.saturating_sub(a) as f64 / ticks_per_second() as f64 / secs * 100.0
}

/// The RAPL package energy counter in µJ and its wrap value, when readable.
pub fn energy_uj() -> Option<(u64, u64)> {
    let base = Path::new("/sys/class/powercap/intel-rapl:0");
    let read = |f: &str| std::fs::read_to_string(base.join(f)).ok()?.trim().parse::<u64>().ok();
    Some((read("energy_uj")?, read("max_energy_range_uj").unwrap_or(u64::MAX)))
}

/// Joules between two `energy_uj` readings, across one wrap of the counter.
pub fn joules(a: (u64, u64), b: (u64, u64)) -> f64 {
    let d = if b.0 >= a.0 { b.0 - a.0 } else { b.0 + (a.1 - a.0) };
    d as f64 / 1e6
}

/// One GPU's readings from sysfs; a field the driver does not expose is `None`.
#[derive(Debug, Clone, Default)]
pub struct Gpu {
    pub card: String,
    pub driver: String,
    pub busy_percent: Option<u64>,
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
    pub gtt_used: Option<u64>,
    pub gtt_total: Option<u64>,
}

/// Every DRM card with a device (`card0`, `card1`, …; not its connectors).
pub fn gpus() -> Vec<Gpu> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir("/sys/class/drm") else { return out };
    let mut names: Vec<String> = rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("card") && n[4..].chars().all(|c| c.is_ascii_digit())).collect();
    names.sort();
    for n in names {
        let dev = Path::new("/sys/class/drm").join(&n).join("device");
        let num = |f: &str| std::fs::read_to_string(dev.join(f)).ok()?.trim().parse::<u64>().ok();
        let driver = std::fs::read_link(dev.join("driver")).ok().and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned())).unwrap_or_default();
        out.push(Gpu { card: n, driver, busy_percent: num("gpu_busy_percent"), vram_used: num("mem_info_vram_used"), vram_total: num("mem_info_vram_total"),
                       gtt_used: num("mem_info_gtt_used"), gtt_total: num("mem_info_gtt_total") });
    }
    out
}

fn opt(v: Option<u64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "null".into())
}

pub fn gpu_json(g: &Gpu) -> String {
    format!("{{\"card\": {}, \"driver\": {}, \"busy_percent\": {}, \"vram_used_bytes\": {}, \"vram_total_bytes\": {}, \"gtt_used_bytes\": {}, \"gtt_total_bytes\": {}}}",
            crate::gguf::jstr(&g.card), crate::gguf::jstr(&g.driver), opt(g.busy_percent), opt(g.vram_used), opt(g.vram_total), opt(g.gtt_used), opt(g.gtt_total))
}

/// A snapshot for the named processes: JSON with memory, cores, per process rss and CPU %, the package power over the
/// same interval (RAPL; `null` when unreadable) and each GPU's readings.
pub fn usage_json(procs: &[(&str, u32)], interval: Duration) -> String {
    let live: Vec<(&str, u32)> = procs.iter().copied().filter(|&(_, p)| p > 0 && alive(p)).collect();
    let before: Vec<Option<u64>> = live.iter().map(|&(_, p)| cpu_ticks(p)).collect();
    let e0 = energy_uj();
    let t = Instant::now();
    std::thread::sleep(interval);
    let secs = t.elapsed().as_secs_f64().max(1e-3);
    let watts = match (e0, energy_uj()) {
        (Some(a), Some(b)) => format!("{:.2}", joules(a, b) / secs),
        _ => "null".into(),
    };
    let hz = ticks_per_second() as f64;
    let mut rows = Vec::new();
    let (mut rss_sum, mut cpu_sum) = (0u64, 0.0);
    for (i, &(name, pid)) in live.iter().enumerate() {
        let r = rss(pid).unwrap_or(0);
        let c = match (before[i], cpu_ticks(pid)) {
            (Some(a), Some(b)) => b.saturating_sub(a) as f64 / hz / secs * 100.0,
            _ => 0.0,
        };
        rss_sum += r;
        cpu_sum += c;
        rows.push(format!("{{\"name\": {}, \"pid\": {pid}, \"rss_bytes\": {r}, \"cpu_percent\": {c:.1}}}", crate::gguf::jstr(name)));
    }
    let m = memory().unwrap_or_default();
    format!(
        "{{\"source\": \"bankml sys.rs (/proc)\", \"cores\": {}, \"mem_total_bytes\": {}, \"mem_available_bytes\": {}, \"swap_total_bytes\": {}, \
         \"swap_free_bytes\": {}, \"rss_bytes\": {rss_sum}, \"cpu_percent\": {cpu_sum:.1}, \"interval_ms\": {}, \"processes\": [{}], \
         \"package_watts\": {watts}, \"gpus\": [{}]}}",
        cores(), m.total, m.available, m.swap_total, m.swap_free, interval.as_millis(), rows.join(", "),
        gpus().iter().map(gpu_json).collect::<Vec<_>>().join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_this_machine_and_this_process() {
        let m = memory().expect("/proc/meminfo");
        assert!(m.total > 0 && m.available <= m.total);
        assert!(cores() >= 1);
        let me = std::process::id();
        assert!(rss(me).unwrap() > 0 && cpu_ticks(me).is_some() && alive(me));
        assert!(ticks_per_second() >= 1);
    }

    #[test]
    fn energy_counter_wraps() {
        assert_eq!(joules((1_000_000, 10_000_000), (3_500_000, 10_000_000)), 2.5);
        // wrapped: from 9.5 J up to the 10 J range, then on to 0.5 J
        assert_eq!(joules((9_500_000, 10_000_000), (500_000, 10_000_000)), 1.0);
    }

    #[test]
    fn gpus_and_power_are_reported_or_null() {
        let j = usage_json(&[("self", std::process::id())], Duration::from_millis(20));
        assert!(j.contains("\"package_watts\": ") && j.contains("\"gpus\": ["));
        for g in gpus() {
            assert!(g.card.starts_with("card"));
            if let (Some(u), Some(t)) = (g.vram_used, g.vram_total) {
                assert!(u <= t);
            }
        }
    }

    #[test]
    fn stat_with_spaces_and_parens_in_the_name() {
        // pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime …
        let s = "1234 (a (b) c) S 1 1 1 0 -1 4194560 100 0 0 0 250 50 0 0 20 0 1 0 100 0 0";
        assert_eq!(parse_ticks(s), Some(300));
    }

    #[test]
    fn busy_loop_is_seen_as_cpu() {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s2 = stop.clone();
        let h = std::thread::spawn(move || {
            let mut x = 0u64;
            while !s2.load(std::sync::atomic::Ordering::Relaxed) {
                x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
            }
        });
        let pct = cpu_percent(&[std::process::id()], Duration::from_millis(400));
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        h.join().unwrap();
        assert!(pct > 30.0, "a spinning thread should read as busy, got {pct:.1} %");
        let j = usage_json(&[("self", std::process::id())], Duration::from_millis(50));
        assert!(j.contains("\"processes\": [{\"name\": \"self\"") && j.starts_with('{') && j.ends_with('}'));
    }
}
