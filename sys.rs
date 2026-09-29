// SPDX-License-Identifier: MIT OR Apache-2.0
//! System and process readings, the part of psutil (or Rust's `sysinfo`) bankml needs, with no crates: on Linux
//! everything comes from `/proc`, which is what those libraries read too. Used by `bankml serve` (`GET
//! /bankml/usage`) and `bankml usage`, and by the Savante UI's Resources sliders.
//!
//! - memory: `/proc/meminfo` (MemTotal, MemAvailable, SwapTotal, SwapFree);
//! - a process: `/proc/<pid>/status` (VmRSS) and `/proc/<pid>/stat` (utime + stime, in clock ticks);
//! - CPU %: the change in a process's ticks over a sampling interval, divided by the ticks per second (`sysconf`),
//!   so 100 % is one core busy and `cores × 100 %` is the machine.

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
        extern "C" {
            fn sysconf(name: i32) -> i64;
        }
        let v = unsafe { sysconf(2) }; // _SC_CLK_TCK
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

/// A snapshot for the named processes: JSON with memory, cores and, per process, rss and CPU %.
pub fn usage_json(procs: &[(&str, u32)], interval: Duration) -> String {
    let live: Vec<(&str, u32)> = procs.iter().copied().filter(|&(_, p)| p > 0 && alive(p)).collect();
    let before: Vec<Option<u64>> = live.iter().map(|&(_, p)| cpu_ticks(p)).collect();
    let t = Instant::now();
    std::thread::sleep(interval);
    let secs = t.elapsed().as_secs_f64().max(1e-3);
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
         \"swap_free_bytes\": {}, \"rss_bytes\": {rss_sum}, \"cpu_percent\": {cpu_sum:.1}, \"interval_ms\": {}, \"processes\": [{}]}}",
        cores(), m.total, m.available, m.swap_total, m.swap_free, interval.as_millis(), rows.join(", ")
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
