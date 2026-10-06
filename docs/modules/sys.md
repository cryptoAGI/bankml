# `bankML/sys.rs` — memory, cores and per-process usage from `/proc`

## Summary

`sys.rs` is the part of psutil (or Rust's `sysinfo`) that bankml needs, with no crates. On Linux everything comes from
`/proc`, which is what those libraries read too: total and available memory, swap, the number of cores, and each
process's resident memory and CPU time.

Callers: `bankml usage [PID …]` (CLI), `bankml serve`'s `GET /bankml/usage` (serve itself and the engine it launched),
and through that endpoint the Savante UI's Resources sliders (threads and a RAM budget). It was added in 0.1.8.

## Technical usage

```rust
pub struct Memory { pub total: u64, pub available: u64, pub swap_total: u64, pub swap_free: u64 }

pub fn memory() -> Option<Memory>
pub fn cores() -> usize
pub fn rss(pid: u32) -> Option<u64>
pub fn cpu_ticks(pid: u32) -> Option<u64>
pub fn ticks_per_second() -> u64
pub fn alive(pid: u32) -> bool
pub fn cpu_percent(pids: &[u32], interval: Duration) -> f64
pub fn usage_json(procs: &[(&str, u32)], interval: Duration) -> String
```

| function | source | notes |
|---|---|---|
| `memory` | `/proc/meminfo`: MemTotal, MemAvailable, SwapTotal, SwapFree | bytes; swap fields default to 0 |
| `cores` | `std::thread::available_parallelism` | 1 if unknown |
| `rss` | `/proc/<pid>/status` VmRSS | bytes |
| `cpu_ticks` | `/proc/<pid>/stat` utime + stime | fields counted after the last `)`, so a command name with spaces or parentheses is safe |
| `ticks_per_second` | `sysconf(_SC_CLK_TCK)` on Linux | 100 if unavailable |
| `alive` | `/proc/<pid>` exists | |
| `cpu_percent` | change in ticks over `interval` | 100 % is one core busy; `cores × 100 %` is the machine |

`usage_json` samples the named live processes over `interval` and returns one JSON object: `source`, `cores`,
`mem_total_bytes`, `mem_available_bytes`, `swap_total_bytes`, `swap_free_bytes`, `rss_bytes` and `cpu_percent` (sums),
`interval_ms`, and `processes` (`name`, `pid`, `rss_bytes`, `cpu_percent` each). Process names are escaped with
`gguf::jstr`. Processes with pid 0 or not alive are left out.

```sh
bankml usage            # bankml itself, sampled over 0.5 s
bankml usage 1234 5678  # the given pids; a non-numeric argument is an error (exit 1)
curl -s 127.0.0.1:18093/bankml/usage   # on a running bankml serve (default listen address)
```

For the endpoint, `serve` samples itself and its `llama-server` child over 250 ms, and takes at most one sample per
second however many clients poll, so a poll never holds a connection for the sampling time.

## How it is verified

- `reads_this_machine_and_this_process`: memory total positive and available ≤ total, at least one core, this
  process's RSS and ticks readable, `alive`, ticks per second ≥ 1.
- `stat_with_spaces_and_parens_in_the_name`: a `/proc/<pid>/stat` line whose command is `a (b) c` parses to the right
  utime + stime.
- `busy_loop_is_seen_as_cpu`: a spinning thread reads above 30 % over 400 ms; `usage_json` produces a well-formed
  object naming the process.

## Advantages and efficiency

- **No crate for a small job.** The readings are a handful of `/proc` files and one libc call (`sysconf`), declared
  by hand, so `Cargo.toml`'s `[dependencies]` stays empty.
- **One sampling window for all processes.** `usage_json` reads every process's ticks before and after a single
  sleep, rather than sleeping per process.
- **Robust parsing.** Missing files or fields give `None` (or 0 in the JSON) rather than a panic; the `stat` parser is
  anchored on the last `)`, as the kernel's format requires.
- **Practice.** `unsafe` is a single `sysconf` call behind `ticks_per_second`, with a fallback when it fails.

## Limitations

- Linux only in practice: every reading comes from `/proc`. On other systems `memory`, `rss` and `cpu_ticks` return
  `None`, and `ticks_per_second` falls back to 100.
- CPU % is a sample over the given interval, so each call blocks for that interval.
- It reports what the processes use; it does not set limits. Fixed-resource benchmarks use `testing/pinned.sh`
  (cores pinned, memory capped in a cgroup).

## See also

- [bankml.md](bankml.md), [par.md](par.md)
- [../usage.md](../usage.md) §13 (commands, ports, environment), [../PERFORMANCE.md](../PERFORMANCE.md),
  [../TODO.md](../TODO.md) (Resources sliders)

## Power and GPUs (0.3.7)

- `energy_uj() -> Option<(u64, u64)>` reads `/sys/class/powercap/intel-rapl:0/energy_uj` and its wrap value;
  `joules(a, b)` takes the difference across one wrap. Root-only unless `./install.sh power` has run; then `null`.
- `gpus() -> Vec<Gpu>` reads each DRM card's `gpu_busy_percent`, `mem_info_vram_*` and `mem_info_gtt_*` (amdgpu exposes
  them; other drivers read as `null`). `usage_json` adds `package_watts` over its interval, `gpus` and `gpu_limiter`.
