// SPDX-License-Identifier: MIT OR Apache-2.0
//! The machine under bankML, measured: what psutil, `ss`/`netstat`, `ip -s link`, `uptime` and a ping would tell
//! you, read from `/proc` and `/sys` with no crates. Used by `bankml diag` and `bankml serve`
//! (`GET /bankml/diagnostics`).
//!
//! - **system:** kernel, uptime, boot time;
//! - **cpu:** busy share over the sample (whole machine and per core, psutil's rule: iowait counts as idle), load
//!   averages, runnable and total tasks, clocks, temperatures (`/sys/class/hwmon`), pressure (PSI);
//! - **memory:** total, available, used, buffers, cached, dirty, swap, pressure;
//! - **processes:** state, RSS, virtual size, threads, open file descriptors, context switches, page faults, CPU %
//!   over the sample, storage I/O, OOM score, age;
//! - **disk:** space for each path, its block device, whether it rotates, and the device's I/O counters
//!   (`/proc/diskstats`, 512-byte sectors as the kernel counts them); I/O pressure;
//! - **network:** each interface's counters (`/proc/net/dev`); TCP and UDP sockets by state (`/proc/net/tcp*`,
//!   `udp*`); listening sockets, and each one's owner where its `/proc/<pid>/fd` is readable (as `ss -p` for a user
//!   who is not root); the sockets of bankML's own processes;
//! - **ping:** TCP connect round trips (min/avg/max over three), the unprivileged measure; ICMP needs a raw socket;
//! - **gpu** and **energy:** `sys.rs`'s readings.
//!
//! Over HTTP (`full = false`) the report keeps to bankML's own picture: no remote addresses, no other process's
//! sockets, no host name, because `--allow-origin` lets one web origin read every route. `bankml diag --full` adds
//! them, for the operator at the terminal. A reading that cannot be taken is `null`, never estimated.
//!
//! The oracle is psutil, `ss` and `ip` on the same machine (`testing/diag_oracle.py`). Details: docs/modules/diag.md.

use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A JSON value, kept in insertion order, that also renders as indented text for the terminal.
#[derive(Debug, Clone, PartialEq)]
pub enum V {
    Null,
    Bool(bool),
    Int(i64),
    Num(f64, usize),
    Str(String),
    Arr(Vec<V>),
    Obj(Vec<(String, V)>),
}

impl V {
    fn obj(kv: Vec<(&str, V)>) -> V {
        V::Obj(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    fn s(x: impl Into<String>) -> V {
        V::Str(x.into())
    }
    fn u(x: Option<u64>) -> V {
        x.map_or(V::Null, |v| V::Int(v as i64))
    }
    fn f(x: Option<f64>, d: usize) -> V {
        x.filter(|v| v.is_finite()).map_or(V::Null, |v| V::Num(v, d))
    }
    pub fn get(&self, k: &str) -> Option<&V> {
        match self {
            V::Obj(kv) => kv.iter().find(|(n, _)| n == k).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn to_json(&self) -> String {
        match self {
            V::Null => "null".into(),
            V::Bool(b) => b.to_string(),
            V::Int(i) => i.to_string(),
            V::Num(x, d) => format!("{x:.d$}"),
            V::Str(s) => crate::gguf::jstr(s),
            V::Arr(a) => format!("[{}]", a.iter().map(V::to_json).collect::<Vec<_>>().join(", ")),
            V::Obj(kv) => format!("{{{}}}", kv.iter().map(|(k, v)| format!("{}: {}", crate::gguf::jstr(k), v.to_json())).collect::<Vec<_>>().join(", ")),
        }
    }

    /// Indented `key: value` lines; a list of objects is one line per object.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        self.text_into(0, &mut out);
        out
    }
    fn inline(&self) -> String {
        match self {
            V::Str(s) => s.clone(),
            V::Obj(kv) => kv.iter().filter(|(_, v)| *v != V::Null && *v != V::Str(String::new())).map(|(k, v)| format!("{k}={}", v.inline())).collect::<Vec<_>>().join(" "),
            V::Arr(a) => a.iter().map(V::inline).collect::<Vec<_>>().join(", "),
            v => v.to_json(),
        }
    }
    fn text_into(&self, depth: usize, out: &mut String) {
        let pad = "  ".repeat(depth);
        let V::Obj(kv) = self else {
            out.push_str(&format!("{pad}{}\n", self.inline()));
            return;
        };
        for (k, v) in kv {
            match v {
                V::Obj(_) => {
                    out.push_str(&format!("{pad}{k}\n"));
                    v.text_into(depth + 1, out);
                }
                V::Arr(a) if a.iter().any(|x| matches!(x, V::Obj(_))) => {
                    out.push_str(&format!("{pad}{k} ({})\n", a.len()));
                    for x in a {
                        out.push_str(&format!("{pad}  - {}\n", x.inline()));
                    }
                }
                _ => out.push_str(&format!("{pad}{k}: {}\n", v.inline())),
            }
        }
    }
}

fn read(p: impl AsRef<Path>) -> Option<String> {
    std::fs::read_to_string(p).ok()
}

/// The value of `key:` in a `/proc` `key: value` table, its first word as an integer (kB fields stay in kB).
fn table_u64(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|l| l.strip_prefix(key)?.strip_prefix(':')?.split_whitespace().next()?.parse().ok())
}

// ---- cpu ----

/// One `cpu` line of `/proc/stat`: (total, busy) ticks. Busy is psutil's: everything but idle and iowait (guest
/// time is already inside user and nice, so it is not added twice).
fn cpu_line(l: &str) -> Option<(u64, u64)> {
    let f: Vec<u64> = l.split_whitespace().skip(1).map(|x| x.parse().ok()).collect::<Option<_>>()?;
    if f.len() < 4 {
        return None;
    }
    let total: u64 = f.iter().take(8).sum(); // user nice system idle iowait irq softirq steal
    let idle = f[3] + f.get(4).copied().unwrap_or(0);
    Some((total, total - idle))
}

/// The whole machine first, then each core, as (total, busy) ticks.
pub fn parse_stat(text: &str) -> Vec<(u64, u64)> {
    text.lines().filter(|l| l.starts_with("cpu")).filter_map(cpu_line).collect()
}

fn busy_percent(a: (u64, u64), b: (u64, u64)) -> Option<f64> {
    let dt = b.0.checked_sub(a.0)?;
    (dt > 0).then(|| (b.1.saturating_sub(a.1)) as f64 / dt as f64 * 100.0)
}

/// `/proc/loadavg`: the three averages, runnable and total scheduling entities.
pub fn parse_loadavg(text: &str) -> Option<([f64; 3], u64, u64)> {
    let w: Vec<&str> = text.split_whitespace().collect();
    let l = [w.first()?.parse().ok()?, w.get(1)?.parse().ok()?, w.get(2)?.parse().ok()?];
    let (r, t) = w.get(3)?.split_once('/')?;
    Some((l, r.parse().ok()?, t.parse().ok()?))
}

/// `/proc/pressure/<what>`: for "some" and "full", the 10 s, 60 s and 300 s averages (percent) and the total stall (µs).
pub fn parse_pressure(text: &str) -> V {
    let mut kv = Vec::new();
    for l in text.lines() {
        let mut w = l.split_whitespace();
        let Some(kind) = w.next() else { continue };
        let f: Vec<(&str, &str)> = w.filter_map(|x| x.split_once('=')).collect();
        let g = |k: &str| f.iter().find(|(n, _)| *n == k).and_then(|(_, v)| v.parse::<f64>().ok());
        kv.push((kind, V::obj(vec![("avg10", V::f(g("avg10"), 2)), ("avg60", V::f(g("avg60"), 2)), ("avg300", V::f(g("avg300"), 2)), ("total_us", V::u(g("total").map(|x| x as u64)))])));
    }
    if kv.is_empty() { V::Null } else { V::obj(kv) }
}

fn pressure(what: &str) -> V {
    read(format!("/proc/pressure/{what}")).map_or(V::Null, |t| parse_pressure(&t))
}

/// Every temperature under `/sys/class/hwmon`: the chip's name, the sensor's label, degrees Celsius.
pub fn temperatures(root: &Path) -> Vec<V> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else { return out };
    let mut dirs: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    dirs.sort();
    for d in dirs {
        let chip = read(d.join("name")).map(|s| s.trim().to_string()).unwrap_or_default();
        let mut inputs: Vec<_> = std::fs::read_dir(&d).into_iter().flatten().flatten().map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("temp") && n.ends_with("_input")).collect();
        inputs.sort_by_key(|n| n[4..n.len() - 6].parse::<u32>().unwrap_or(0));
        for n in inputs {
            let Some(milli) = read(d.join(&n)).and_then(|s| s.trim().parse::<i64>().ok()) else { continue };
            let label = read(d.join(n.replace("_input", "_label"))).map(|s| s.trim().to_string()).unwrap_or_default();
            out.push(V::obj(vec![("chip", V::s(chip.clone())), ("label", V::s(label)), ("celsius", V::Num(milli as f64 / 1000.0, 1))]));
        }
    }
    out
}

// ---- memory ----

fn memory_v() -> V {
    let Some(t) = read("/proc/meminfo") else { return V::Null };
    let kb = |k: &str| table_u64(&t, k).map(|v| v * 1024);
    let (total, avail) = (kb("MemTotal"), kb("MemAvailable"));
    // psutil's "used": total − free − buffers − cached (cached includes SReclaimable)
    let cached = kb("Cached").zip(kb("SReclaimable")).map(|(c, s)| c + s);
    let used = (|| Some(total?.saturating_sub(kb("MemFree")? + kb("Buffers")? + cached?)))();
    V::obj(vec![
        ("total_bytes", V::u(total)), ("available_bytes", V::u(avail)), ("used_bytes", V::u(used)), ("free_bytes", V::u(kb("MemFree"))),
        ("buffers_bytes", V::u(kb("Buffers"))), ("cached_bytes", V::u(cached)), ("dirty_bytes", V::u(kb("Dirty"))),
        ("swap_total_bytes", V::u(kb("SwapTotal"))), ("swap_free_bytes", V::u(kb("SwapFree"))),
        ("available_percent", V::f(total.zip(avail).map(|(t, a)| a as f64 / t as f64 * 100.0), 1)),
        ("pressure", pressure("memory")),
    ])
}

// ---- processes ----

/// Fields of `/proc/<pid>/stat` after the command name, which may hold spaces and parentheses: index 0 is the state.
fn stat_fields(stat: &str) -> Option<Vec<&str>> {
    Some(stat[stat.rfind(')')? + 1..].split_whitespace().collect())
}

/// The inodes of a process's sockets (`/proc/<pid>/fd/* -> socket:[inode]`), and how many descriptors it holds;
/// `None` when its fd directory cannot be read (another user's process).
fn fds(pid: u32) -> Option<(usize, Vec<u64>)> {
    let rd = std::fs::read_dir(format!("/proc/{pid}/fd")).ok()?;
    let (mut n, mut socks) = (0, Vec::new());
    for e in rd.flatten() {
        n += 1;
        if let Ok(l) = std::fs::read_link(e.path()) {
            if let Some(i) = l.to_str().and_then(|s| s.strip_prefix("socket:[")?.strip_suffix(']')?.parse().ok()) {
                socks.push(i);
            }
        }
    }
    Some((n, socks))
}

struct ProcSample {
    name: String,
    pid: u32,
    ticks: Option<u64>,
}

fn proc_v(p: &ProcSample, secs: f64, uptime: Option<f64>) -> V {
    let pid = p.pid;
    let status = read(format!("/proc/{pid}/status")).unwrap_or_default();
    let stat = read(format!("/proc/{pid}/stat")).unwrap_or_default();
    let f = stat_fields(&stat).unwrap_or_default();
    let fu = |i: usize| f.get(i).and_then(|x| x.parse::<u64>().ok());
    let hz = crate::sys::ticks_per_second() as f64;
    let cpu = match (p.ticks, crate::sys::cpu_ticks(pid)) {
        (Some(a), Some(b)) => Some(b.saturating_sub(a) as f64 / hz / secs * 100.0),
        _ => None,
    };
    // field 22 (index 19 here) is the start time in ticks after boot
    let age = uptime.zip(fu(19)).map(|(up, st)| (up - st as f64 / hz).max(0.0));
    let io = crate::sys::io(pid);
    V::obj(vec![
        ("name", V::s(p.name.clone())), ("pid", V::Int(pid as i64)),
        ("command", V::s(status.lines().find_map(|l| l.strip_prefix("Name:")).unwrap_or("").trim())),
        ("state", V::s(f.first().copied().unwrap_or(""))),
        ("rss_bytes", V::u(table_u64(&status, "VmRSS").map(|k| k * 1024))), ("vm_bytes", V::u(table_u64(&status, "VmSize").map(|k| k * 1024))),
        ("threads", V::u(table_u64(&status, "Threads"))), ("fds", V::u(fds(pid).map(|(n, _)| n as u64))),
        ("ctx_voluntary", V::u(table_u64(&status, "voluntary_ctxt_switches"))), ("ctx_involuntary", V::u(table_u64(&status, "nonvoluntary_ctxt_switches"))),
        ("minor_faults", V::u(fu(7))), ("major_faults", V::u(fu(9))),
        ("cpu_percent", V::f(cpu, 1)), ("read_bytes", V::u(io.map(|x| x.0))), ("write_bytes", V::u(io.map(|x| x.1))),
        ("oom_score", V::u(read(format!("/proc/{pid}/oom_score")).and_then(|s| s.trim().parse().ok()))),
        ("age_s", V::f(age, 1)),
        ("cgroup", cgroup_v(pid)),
    ])
}

/// A cgroup v2 limit file: a number, or `max` (no limit).
fn limit(text: Option<String>) -> V {
    match text.as_deref().map(str::trim) {
        Some("max") => V::s("max"),
        Some(t) => t.parse::<u64>().map_or(V::Null, |n| V::Int(n as i64)),
        None => V::Null,
    }
}

/// The process's cgroup (v2: `/proc/<pid>/cgroup` is `0::/path`) and what it limits and counts: memory (max, high,
/// current, peak, swap, the OOM events), CPU (`cpu.max` as quota and period, and the CPUs that quota allows; weight;
/// throttling) and tasks. A systemd service's `MemoryMax=` and `CPUQuota=` are these files.
pub fn cgroup_v(pid: u32) -> V {
    let Some(path) = read(format!("/proc/{pid}/cgroup")).and_then(|t| t.lines().find_map(|l| l.strip_prefix("0::").map(str::to_string))) else { return V::Null };
    let dir = Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/'));
    let f = |n: &str| read(dir.join(n));
    let events = f("memory.events").unwrap_or_default();
    let cpu_stat = f("cpu.stat").unwrap_or_default();
    let ev = |k: &str| events.lines().find_map(|l| l.strip_prefix(k)?.strip_prefix(' ')?.trim().parse::<u64>().ok());
    let cs = |k: &str| cpu_stat.lines().find_map(|l| l.strip_prefix(k)?.strip_prefix(' ')?.trim().parse::<u64>().ok());
    // cpu.max: "<quota|max> <period>"
    let cpu_max = f("cpu.max").map(|t| t.split_whitespace().map(str::to_string).collect::<Vec<_>>());
    let (quota, period) = match cpu_max.as_deref() {
        Some([q, p]) => (Some(q.clone()), p.parse::<u64>().ok()),
        _ => (None, None),
    };
    let allowed = quota.as_deref().and_then(|q| q.parse::<u64>().ok()).zip(period).map(|(q, p)| q as f64 / p as f64);
    V::obj(vec![
        ("path", V::s(path.trim())),
        ("memory_max", limit(f("memory.max"))), ("memory_high", limit(f("memory.high"))),
        ("memory_current_bytes", limit(f("memory.current"))), ("memory_peak_bytes", limit(f("memory.peak"))),
        ("swap_current_bytes", limit(f("memory.swap.current"))), ("swap_max", limit(f("memory.swap.max"))),
        ("oom_events", V::u(ev("oom"))), ("oom_kills", V::u(ev("oom_kill"))), ("memory_high_events", V::u(ev("high"))),
        ("cpu_quota_us", quota.map_or(V::Null, |q| q.parse::<u64>().map_or(V::s(q), |n| V::Int(n as i64)))), ("cpu_period_us", V::u(period)),
        ("cpu_quota_cpus", V::f(allowed, 2)), ("cpu_weight", limit(f("cpu.weight"))),
        ("cpu_usage_us", V::u(cs("usage_usec"))), ("nr_throttled", V::u(cs("nr_throttled"))), ("throttled_us", V::u(cs("throttled_usec"))),
        ("pids_current", limit(f("pids.current"))), ("pids_max", limit(f("pids.max"))),
        ("pressure", read(dir.join("memory.pressure")).map_or(V::Null, |t| V::obj(vec![("memory", parse_pressure(&t)), ("cpu", f("cpu.pressure").map_or(V::Null, |t| parse_pressure(&t))), ("io", f("io.pressure").map_or(V::Null, |t| parse_pressure(&t)))]))),
    ])
}

// ---- disk ----

/// One device's line of `/proc/diskstats`: reads, read bytes, writes, written bytes, milliseconds doing I/O.
pub fn parse_diskstats(text: &str, dev: &str) -> Option<[u64; 5]> {
    text.lines().find_map(|l| {
        let w: Vec<&str> = l.split_whitespace().collect();
        if w.get(2) != Some(&dev) {
            return None;
        }
        let g = |i: usize| w.get(i).and_then(|x| x.parse::<u64>().ok());
        Some([g(3)?, g(5)? * 512, g(7)?, g(9)? * 512, g(12)?])
    })
}

/// The block device holding `path`: the longest mount point in `/proc/self/mountinfo` that contains it, its
/// source's last component (`/dev/nvme0n1p2` → `nvme0n1p2`).
fn device_of(path: &Path) -> Option<String> {
    let path = std::fs::canonicalize(path).ok()?;
    let mi = read("/proc/self/mountinfo")?;
    let mut best: Option<(usize, String)> = None;
    for l in mi.lines() {
        let w: Vec<&str> = l.split_whitespace().collect();
        let (Some(mp), Some(sep)) = (w.get(4), w.iter().position(|x| *x == "-")) else { continue };
        let src = w.get(sep + 2).copied().unwrap_or("");
        if path.starts_with(mp) && best.as_ref().is_none_or(|(n, _)| mp.len() > *n) && src.starts_with("/dev/") {
            best = Some((mp.len(), src.rsplit('/').next().unwrap_or("").to_string()));
        }
    }
    best.map(|(_, d)| d)
}

/// A partition's disk (`nvme0n1p2` → `nvme0n1`, `sda1` → `sda`): the name under `/sys/block`.
fn disk_of(dev: &str) -> String {
    let mut d = dev.to_string();
    while !d.is_empty() && !Path::new(&format!("/sys/block/{d}")).exists() {
        d.pop();
    }
    if d.ends_with('p') && !Path::new(&format!("/sys/block/{d}")).exists() {
        d.pop();
    }
    if d.is_empty() { dev.to_string() } else { d }
}

fn disk_v(name: &str, path: &Path, stats: &str) -> V {
    let space = crate::sys::disk(path);
    let dev = device_of(path);
    let disk = dev.as_deref().map(disk_of);
    let io = dev.as_deref().and_then(|d| parse_diskstats(stats, d));
    let sys = |f: &str| disk.as_ref().and_then(|d| read(format!("/sys/block/{d}/{f}"))).map(|s| s.trim().to_string());
    V::obj(vec![
        ("name", V::s(name)), ("path", V::s(path.display().to_string())),
        ("total_bytes", V::u(space.map(|s| s.0))), ("available_bytes", V::u(space.map(|s| s.1))),
        ("device", dev.map_or(V::Null, V::Str)), ("disk", disk.clone().map_or(V::Null, V::Str)),
        ("model", sys("device/model").map_or(V::Null, V::Str)), ("rotational", sys("queue/rotational").map_or(V::Null, |r| V::Bool(r == "1"))),
        ("reads", V::u(io.map(|x| x[0]))), ("read_bytes", V::u(io.map(|x| x[1]))), ("writes", V::u(io.map(|x| x[2]))),
        ("written_bytes", V::u(io.map(|x| x[3]))), ("io_ms", V::u(io.map(|x| x[4]))),
    ])
}

// ---- network ----

/// An interface's counters from `/proc/net/dev`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nic {
    pub name: String,
    /// bytes, packets, errors, drops received; then the same sent
    pub rx: [u64; 4],
    pub tx: [u64; 4],
}

pub fn parse_net_dev(text: &str) -> Vec<Nic> {
    text.lines().skip(2).filter_map(|l| {
        let (name, rest) = l.split_once(':')?;
        let f: Vec<u64> = rest.split_whitespace().map(|x| x.parse().ok()).collect::<Option<_>>()?;
        (f.len() >= 12).then(|| Nic { name: name.trim().to_string(), rx: [f[0], f[1], f[2], f[3]], tx: [f[8], f[9], f[10], f[11]] })
    }).collect()
}

/// One row of `/proc/net/{tcp,tcp6,udp,udp6}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sock {
    pub proto: &'static str,
    pub local: SocketAddr,
    pub remote: SocketAddr,
    pub state: u8,
    pub inode: u64,
}

/// `0100007F:1F90` (IPv4) or 32 hex digits and a port (IPv6). The kernel prints each 32-bit word of the address as
/// a host-order integer, so each word's native bytes are the address's bytes.
fn hex_addr(s: &str) -> Option<SocketAddr> {
    let (a, p) = s.split_once(':')?;
    let port = u16::from_str_radix(p, 16).ok()?;
    let word = |i: usize| u32::from_str_radix(a.get(i * 8..i * 8 + 8)?, 16).ok().map(u32::to_ne_bytes);
    match a.len() {
        8 => Some(SocketAddr::from((word(0)?, port))),
        32 => {
            let mut b = [0u8; 16];
            for i in 0..4 {
                b[i * 4..i * 4 + 4].copy_from_slice(&word(i)?);
            }
            Some(SocketAddr::from((b, port)))
        }
        _ => None,
    }
}

pub fn parse_net_socks(text: &str, proto: &'static str) -> Vec<Sock> {
    text.lines().skip(1).filter_map(|l| {
        let w: Vec<&str> = l.split_whitespace().collect();
        Some(Sock { proto, local: hex_addr(w.get(1)?)?, remote: hex_addr(w.get(2)?)?, state: u8::from_str_radix(w.get(3)?, 16).ok()?, inode: w.get(9)?.parse().ok()? })
    }).collect()
}

/// The kernel's TCP state names (`include/net/tcp_states.h`), as `ss` prints them.
pub fn tcp_state(s: u8) -> &'static str {
    match s {
        1 => "ESTAB", 2 => "SYN-SENT", 3 => "SYN-RECV", 4 => "FIN-WAIT-1", 5 => "FIN-WAIT-2", 6 => "TIME-WAIT", 7 => "UNCONN",
        8 => "CLOSE-WAIT", 9 => "LAST-ACK", 10 => "LISTEN", 11 => "CLOSING", 12 => "NEW-SYN-RECV", _ => "UNKNOWN",
    }
}

fn sockets() -> Vec<Sock> {
    let mut out = Vec::new();
    for (f, p) in [("tcp", "tcp"), ("tcp6", "tcp6"), ("udp", "udp"), ("udp6", "udp6")] {
        if let Some(t) = read(format!("/proc/net/{f}")) {
            out.extend(parse_net_socks(&t, p));
        }
    }
    out
}

/// Which readable process holds each socket inode: (inode, pid, command). Another user's processes are not
/// readable without root, as with `ss -p`.
fn socket_owners(only: Option<&[u32]>) -> Vec<(u64, u32, String)> {
    let pids: Vec<u32> = match only {
        Some(p) => p.to_vec(),
        None => std::fs::read_dir("/proc").into_iter().flatten().flatten().filter_map(|e| e.file_name().to_str()?.parse().ok()).collect(),
    };
    let mut out = Vec::new();
    for pid in pids {
        if let Some((_, socks)) = fds(pid) {
            let comm = read(format!("/proc/{pid}/comm")).map(|s| s.trim().to_string()).unwrap_or_default();
            out.extend(socks.into_iter().map(|i| (i, pid, comm.clone())));
        }
    }
    out
}

fn sock_v(s: &Sock, owners: &[(u64, u32, String)], remote: bool) -> V {
    let o = owners.iter().find(|(i, _, _)| *i == s.inode && s.inode != 0);
    let mut kv = vec![("proto", V::s(s.proto)), ("state", V::s(tcp_state(s.state))), ("local", V::s(s.local.to_string()))];
    if remote {
        kv.push(("remote", V::s(s.remote.to_string())));
    }
    kv.push(("pid", o.map_or(V::Null, |o| V::Int(o.1 as i64))));
    kv.push(("process", o.map_or(V::Null, |o| V::s(o.2.clone()))));
    V::obj(kv)
}

fn network_v(own: &[u32], full: bool) -> V {
    let nics = read("/proc/net/dev").map(|t| parse_net_dev(&t)).unwrap_or_default();
    let socks = sockets();
    let mut counts: Vec<(String, i64)> = Vec::new();
    for s in &socks {
        let k = format!("{}_{}", s.proto.trim_end_matches('6'), tcp_state(s.state).to_lowercase().replace('-', "_"));
        match counts.iter_mut().find(|(n, _)| *n == k) {
            Some(c) => c.1 += 1,
            None => counts.push((k, 1)),
        }
    }
    counts.sort();
    let owners = socket_owners(if full { None } else { Some(own) });
    let own_inodes: Vec<u64> = owners.iter().filter(|o| own.contains(&o.1)).map(|o| o.0).collect();
    let listening: Vec<V> = socks.iter().filter(|s| s.state == 10 || (s.proto.starts_with("udp") && s.state == 7))
        .filter(|s| full || own_inodes.contains(&s.inode)).map(|s| sock_v(s, &owners, false)).collect();
    let mine: Vec<V> = socks.iter().filter(|s| own_inodes.contains(&s.inode) && s.state != 10).map(|s| sock_v(s, &owners, full || s.remote.ip().is_loopback())).collect();
    let mut kv = vec![
        ("interfaces", V::Arr(nics.iter().map(|n| V::obj(vec![
            ("name", V::s(n.name.clone())), ("rx_bytes", V::Int(n.rx[0] as i64)), ("rx_packets", V::Int(n.rx[1] as i64)), ("rx_errors", V::Int(n.rx[2] as i64)),
            ("rx_drops", V::Int(n.rx[3] as i64)), ("tx_bytes", V::Int(n.tx[0] as i64)), ("tx_packets", V::Int(n.tx[1] as i64)),
            ("tx_errors", V::Int(n.tx[2] as i64)), ("tx_drops", V::Int(n.tx[3] as i64)),
        ])).collect())),
        ("sockets", V::Obj(counts.into_iter().map(|(k, n)| (k, V::Int(n))).collect())),
        (if full { "listening" } else { "listening_bankml" }, V::Arr(listening)),
        ("bankml_connections", V::Arr(mine)),
    ];
    if full {
        let rest: Vec<V> = socks.iter().filter(|s| s.state != 10 && !(s.proto.starts_with("udp") && s.state == 7)).map(|s| sock_v(s, &owners, true)).collect();
        kv.push(("connections", V::Arr(rest)));
    }
    V::obj(kv)
}

// ---- ping ----

/// TCP connect round trips to `target` (HOST:PORT): three tries, each at most `timeout`; min, avg and max over the
/// ones that connected. Connecting is the unprivileged ping: ICMP echo needs a raw socket.
pub fn ping(target: &str, timeout: Duration) -> V {
    let addr: Option<SocketAddr> = target.to_socket_addrs().ok().and_then(|mut a| a.next());
    let Some(addr) = addr else {
        return V::obj(vec![("target", V::s(target)), ("error", V::s("cannot resolve"))]);
    };
    let (mut ms, mut err) = (Vec::new(), None);
    for _ in 0..3 {
        let t = Instant::now();
        match TcpStream::connect_timeout(&addr, timeout) {
            Ok(_) => ms.push(t.elapsed().as_secs_f64() * 1000.0),
            Err(e) => err = Some(e.kind().to_string()),
        }
    }
    let avg = (!ms.is_empty()).then(|| ms.iter().sum::<f64>() / ms.len() as f64);
    V::obj(vec![
        ("target", V::s(target)), ("address", V::s(addr.to_string())), ("method", V::s("tcp connect")),
        ("sent", V::Int(3)), ("connected", V::Int(ms.len() as i64)),
        ("min_ms", V::f(ms.iter().copied().reduce(f64::min), 3)), ("avg_ms", V::f(avg, 3)), ("max_ms", V::f(ms.iter().copied().reduce(f64::max), 3)),
        ("error", err.filter(|_| ms.len() < 3).map_or(V::Null, V::Str)),
    ])
}

// ---- the report ----

/// What to measure and how much of it to show.
pub struct Opts<'a> {
    /// bankML's processes: (name, pid); pid 0 or not alive is left out
    pub procs: &'a [(&'a str, u32)],
    /// paths whose disks to report: (name, path)
    pub paths: &'a [(&'a str, &'a Path)],
    /// HOST:PORT targets for `ping`
    pub ping: &'a [String],
    /// the CPU, process and energy readings are rates over this interval
    pub sample: Duration,
    /// remote addresses, every process's sockets and the host name (the terminal); off over HTTP
    pub full: bool,
}

pub fn report(o: &Opts) -> V {
    let live: Vec<(&str, u32)> = o.procs.iter().copied().filter(|&(_, p)| p > 0 && crate::sys::alive(p)).collect();
    let samples: Vec<ProcSample> = live.iter().map(|&(n, p)| ProcSample { name: n.to_string(), pid: p, ticks: crate::sys::cpu_ticks(p) }).collect();
    let stat0 = read("/proc/stat").map(|t| parse_stat(&t)).unwrap_or_default();
    let e0 = crate::sys::energy_uj();
    let t = Instant::now();
    std::thread::sleep(o.sample);
    let stat_text = read("/proc/stat").unwrap_or_default();
    let stat1 = parse_stat(&stat_text);
    let e1 = crate::sys::energy_uj();
    let secs = t.elapsed().as_secs_f64().max(1e-3);

    let uptime = read("/proc/uptime").and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok());
    let btime = table_u64(&stat_text.replace(' ', ":"), "btime"); // "btime 1791…" read as a key: value line
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    let mut system = vec![
        ("kernel", read("/proc/sys/kernel/osrelease").map_or(V::Null, |s| V::s(s.trim()))),
        ("uptime_s", V::f(uptime, 1)), ("boot_time", V::u(btime)), ("at", V::Num(now, 3)),
    ];
    if o.full {
        system.insert(0, ("hostname", read("/proc/sys/kernel/hostname").map_or(V::Null, |s| V::s(s.trim()))));
    }

    let busy: Vec<Option<f64>> = stat0.iter().zip(&stat1).map(|(a, b)| busy_percent(*a, *b)).collect();
    let (model, mhz) = crate::sys::cpu();
    let load = read("/proc/loadavg").and_then(|t| parse_loadavg(&t));
    let cpu = V::obj(vec![
        ("model", model.map_or(V::Null, V::Str)), ("logical", V::u(crate::sys::online_cpus().map(|n| n as u64))),
        ("allowed", V::Int(crate::sys::cores() as i64)),
        ("busy_percent", V::f(busy.first().copied().flatten(), 1)),
        ("per_core_busy_percent", V::Arr(busy.iter().skip(1).map(|b| V::f(*b, 1)).collect())),
        ("mhz", V::Arr(mhz.iter().map(|m| V::Num(*m, 0)).collect())),
        ("load", load.map_or(V::Null, |(l, _, _)| V::Arr(l.iter().map(|x| V::Num(*x, 2)).collect()))),
        ("runnable", V::u(load.map(|x| x.1))), ("tasks", V::u(load.map(|x| x.2))),
        ("temperatures", V::Arr(temperatures(Path::new("/sys/class/hwmon")))),
        ("pressure", pressure("cpu")),
    ]);

    let stats = read("/proc/diskstats").unwrap_or_default();
    let watts = e0.zip(e1).map(|(a, b)| crate::sys::joules(a, b) / secs);
    let own: Vec<u32> = live.iter().map(|x| x.1).collect();
    V::obj(vec![
        ("source", V::s("bankml diag.rs (/proc, /sys)")), ("bankml", V::s(crate::VERSION)), ("sample_ms", V::Int(o.sample.as_millis() as i64)),
        ("full", V::Bool(o.full)),
        ("system", V::obj(system)),
        ("cpu", cpu),
        ("memory", memory_v()),
        ("processes", V::Arr(samples.iter().map(|p| proc_v(p, secs, uptime)).collect())),
        ("disks", V::Arr(o.paths.iter().map(|(n, p)| disk_v(n, p, &stats)).collect())),
        ("io_pressure", pressure("io")),
        ("network", network_v(&own, o.full)),
        ("ping", V::Arr(o.ping.iter().map(|t| ping(t, Duration::from_secs(1))).collect())),
        ("gpus", V::Arr(crate::sys::gpus().iter().map(|g| V::s(crate::sys::gpu_json(g))).collect())),
        ("package_watts", V::f(watts, 2)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_busy_is_psutils_rule() {
        let a = parse_stat("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 50 0 25 400 25 0 0 0 0 0\nintr 1\n");
        let b = parse_stat("cpu  160 0 70 880 90 0 0 0 0 0\ncpu0 80 0 35 440 45 0 0 0 0 0\n");
        assert_eq!(a.len(), 2);
        // Δtotal 200, Δidle+iowait 120 → 40 % busy
        assert_eq!(busy_percent(a[0], b[0]), Some(40.0));
        assert_eq!(busy_percent(a[1], b[1]), Some(40.0));
        assert_eq!(busy_percent(a[0], a[0]), None);
    }

    #[test]
    fn loadavg_pressure_and_diskstats() {
        assert_eq!(parse_loadavg("0.52 0.58 0.59 2/913 12345\n"), Some(([0.52, 0.58, 0.59], 2, 913)));
        let p = parse_pressure("some avg10=1.50 avg60=0.75 avg300=0.10 total=123456\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=7\n");
        assert_eq!(p.get("some").and_then(|s| s.get("avg10")), Some(&V::Num(1.5, 2)));
        assert_eq!(p.get("full").and_then(|s| s.get("total_us")), Some(&V::Int(7)));
        let d = " 259       0 nvme0n1 1000 5 20000 300 400 6 8000 900 0 1500 1200 0 0 0 0\n 259       2 nvme0n1p2 900 5 18000 280 390 6 7900 890 0 1400 1170\n";
        assert_eq!(parse_diskstats(d, "nvme0n1p2"), Some([900, 18000 * 512, 390, 7900 * 512, 1400]));
        assert_eq!(parse_diskstats(d, "sda"), None);
    }

    #[test]
    fn net_dev_and_sockets() {
        let dev = "Inter-|   Receive                                                |  Transmit\n face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n    lo: 1000 10 0 0 0 0 0 0 1000 10 0 0 0 0 0 0\n  wlo1: 5000 50 1 2 0 0 0 3 4000 40 0 1 0 0 0 0\n";
        let n = parse_net_dev(dev);
        assert_eq!(n[1], Nic { name: "wlo1".into(), rx: [5000, 50, 1, 2], tx: [4000, 40, 0, 1] });
        let tcp = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n   0: 0100007F:46AD 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 98765 1 0000000000000000 100 0 0 10 0\n";
        let s = parse_net_socks(tcp, "tcp");
        assert_eq!(s[0].local.to_string(), "127.0.0.1:18093");
        assert_eq!((tcp_state(s[0].state), s[0].inode), ("LISTEN", 98765));
        // ::1 port 7873, as /proc/net/tcp6 prints it on a little-endian machine
        let tcp6 = "  sl  local_address                         remote_address                        st\n   0: 00000000000000000000000001000000:1EC1 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 4242 1\n";
        let s6 = parse_net_socks(tcp6, "tcp6");
        if cfg!(target_endian = "little") {
            assert_eq!(s6[0].local.to_string(), "[::1]:7873");
        }
    }

    #[test]
    fn text_and_json_render_the_same_tree() {
        let v = V::obj(vec![("a", V::Int(1)), ("b", V::obj(vec![("c", V::Null), ("d", V::s("x\"y"))])), ("e", V::Arr(vec![V::obj(vec![("f", V::Num(1.5, 1))])]))]);
        assert_eq!(v.to_json(), r#"{"a": 1, "b": {"c": null, "d": "x\"y"}, "e": [{"f": 1.5}]}"#);
        assert_eq!(v.to_text(), "a: 1\nb\n  c: null\n  d: x\"y\ne (1)\n  - f=1.5\n");
    }

    /// The whole report on this machine: it parses as JSON, sees this process, its own listening socket and a ping
    /// to it, and over HTTP (`full: false`) carries no remote address and no host name.
    #[test]
    fn report_sees_this_machine_and_keeps_to_its_own_over_http() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let me = std::process::id();
        let target = [format!("127.0.0.1:{port}")];
        let accept = std::thread::spawn(move || for _ in 0..3 { let _ = l.accept(); });
        let v = report(&Opts { procs: &[("self", me)], paths: &[("root", Path::new("/"))], ping: &target, sample: Duration::from_millis(50), full: false });
        accept.join().unwrap();
        let j = v.to_json();
        assert!(crate::serve::Json::parse(&j).is_some(), "not JSON: {j}");
        assert!(v.get("system").unwrap().get("hostname").is_none());
        let p = &v.get("ping").unwrap();
        assert!(matches!(p, V::Arr(a) if a[0].get("connected") == Some(&V::Int(3))), "{}", p.to_json());
        let net = v.get("network").unwrap();
        let listen = net.get("listening_bankml").unwrap().to_json();
        assert!(listen.contains(&format!("127.0.0.1:{port}")) && listen.contains(&format!("\"pid\": {me}")), "{listen}");
        assert!(net.get("connections").is_none() && !j.contains("\"remote\": \"") || j.contains("\"remote\": \"127.0.0.1"));
        let procs = v.get("processes").unwrap().to_json();
        assert!(procs.contains("\"threads\": ") && procs.contains("\"fds\": "), "{procs}");
        let mem = v.get("memory").unwrap();
        assert!(matches!(mem.get("total_bytes"), Some(V::Int(t)) if *t > 0));
    }
}
