# `bankML/diag.rs` — the machine under bankML, measured

## Summary

`diag.rs` is what psutil, `ss` (or `netstat`), `ip -s link`, `uptime` and a ping would tell you about the machine
bankML runs on, read from `/proc` and `/sys` with no crates, and checked against those tools. It builds on `sys.rs`
(memory, cores, per-process CPU time, storage I/O, RAPL energy, GPUs) and adds what `sys.rs` did not read:

| section | readings | source |
|---|---|---|
| system | kernel, uptime, boot time (and the host name with `--full`) | `/proc/sys/kernel`, `/proc/uptime`, `/proc/stat` |
| cpu | busy share over the sample, whole machine and per core (psutil's rule: iowait is idle); load averages; runnable and total tasks; clocks; every temperature sensor; CPU pressure | `/proc/stat`, `/proc/loadavg`, `cpufreq`, `/sys/class/hwmon`, `/proc/pressure/cpu` |
| memory | total, available, used (psutil's definition), free, buffers, cached, dirty, swap; memory pressure | `/proc/meminfo`, `/proc/pressure/memory` |
| processes | state, RSS, virtual size, threads, open descriptors, voluntary and involuntary context switches, minor and major page faults, CPU % over the sample, storage I/O, OOM score, age | `/proc/<pid>/{status,stat,fd,io,oom_score}` |
| disks | space for each path, its block device and disk, model, whether it rotates, the device's reads, writes, bytes and I/O time; I/O pressure | `statvfs`, `/proc/self/mountinfo`, `/sys/block`, `/proc/diskstats`, `/proc/pressure/io` |
| network | each interface's bytes, packets, errors and drops; TCP and UDP sockets counted by state; listening sockets with their owners; bankML's own connections | `/proc/net/dev`, `/proc/net/{tcp,tcp6,udp,udp6}`, `/proc/<pid>/fd` |
| ping | TCP connect round trips: three tries, min / avg / max, or why they failed | `connect(2)` |
| gpus, energy | `sys.rs`'s readings, package watts over the sample | `/sys/class/drm`, RAPL |

Callers: `bankml diag` (CLI) and `bankml serve`'s `GET /bankml/diagnostics` (serve, its spawned engine, the model's
disk, and a ping to serve's own address and the engine's). Added in 0.4.4.

## Technical usage

```rust
pub struct Opts<'a> {
    pub procs: &'a [(&'a str, u32)],   // bankML's processes; pid 0 or not alive is left out
    pub paths: &'a [(&'a str, &'a Path)],
    pub ping: &'a [String],            // HOST:PORT
    pub sample: Duration,              // CPU, process and energy readings are rates over this
    pub full: bool,                    // remote addresses, every process's sockets, the host name
}
pub fn report(o: &Opts) -> V           // V: JSON in insertion order; .to_json(), .to_text()
pub fn ping(target: &str, timeout: Duration) -> V
// parsers, each over the file's text: parse_stat, parse_loadavg, parse_pressure, parse_diskstats, parse_net_dev,
// parse_net_socks, tcp_state
```

```sh
bankml diag                          # text, this process, a ping to 127.0.0.1:18093
bankml diag --json --full PID …      # JSON; every socket and remote address; the named processes
bankml diag --ping 1.1.1.1:443 --ping 127.0.0.1:18092 --sample 1000
curl -s 127.0.0.1:18093/bankml/diagnostics     # serve's own picture (not full), at most every 2 s
```

## How it is verified

- **Unit tests** on fixed text: psutil's busy rule (`stat_busy_is_psutils_rule`), load, pressure and diskstats lines,
  `/proc/net/dev`, an IPv4 and an IPv6 socket row (the kernel prints each address word in host order), the JSON and
  text renderings of one tree; and the whole report on the running machine, which must see its own listening socket
  and its owner, ping it three times out of three, parse as JSON, and carry no host name when not `full`.
- **The oracle**, `testing/diag_oracle.py` (in the gate): `bankml diag --json --full` on the oracle's own process
  against psutil, `ss -ltn` and `ip -s -j link` on the same machine. Fixed values must be equal (cores, memory and
  swap totals, boot time, host name, the set of interfaces, the set of listening TCP sockets and their owners, the
  root file system's size, the sensors' names). Counters must lie between bankML's reading and psutil's later one
  (interface bytes and packets, disk I/O, context switches). Levels that move must agree within a stated tolerance
  (available and used memory 64 MB, load 0.15, temperatures 3 °C, RSS 4 MB). A ping to a listening socket connects
  three times; one to a closed port is refused three times.
- **Not checked by an independent source**, and said so in the oracle's output: pressure (psutil does not read it),
  the busy share of one sample (only its range), and the ping times (only their count and order).

## Advantages and efficiency

- One read of each file per report; the parsers take text, so they are tested without the machine. The sample is
  one sleep for every rate (CPU, processes, energy). `serve` caches the report for two seconds for all pollers.
- No crates, no subprocesses: the readings `ss` and psutil take, without either.
- The ping is unprivileged (a TCP connect), so it measures exactly what a client of `serve` meets.

## Limitations

- Linux only (`/proc`, `/sys`); elsewhere the readings are `null`.
- Owners of other users' sockets are unknown without root, as with `ss -p`.
- No ICMP echo: that needs a raw socket (`CAP_NET_RAW`) or the `ping_group_range` sysctl.
- Over HTTP the report is never `full`: `--allow-origin` lets one web origin read every route, so remote addresses,
  other processes' sockets and the host name stay at the terminal.

## See also

[sys.md](sys.md) (the readings this builds on) · [serve.md](serve.md) (`/bankml/usage`, `/bankml/status`,
`/bankml/diagnostics`) · [metrics.md](metrics.md) (what each answer measured) · `sAGI/sysdiag.py` (the console's
Python view of the same machine) · `testing/diag_oracle.py`
