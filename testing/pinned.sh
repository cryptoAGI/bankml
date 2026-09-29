#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Run a benchmark on a FIXED amount of processor and memory, so runs are comparable:
#   CPU  — pinned to a fixed set of cores (taskset; default the last N logical CPUs)
#   RAM  — a hard cap with no swap (a user cgroup via systemd-run: MemoryMax, MemorySwapMax=0)
# and the machine's state recorded before and after (load average, free memory), because other programs can still
# run on the same cores — without root, bankml cannot evict them, only pin itself and say what else was running.
#   BANKML_PIN_CPUS=2,3 BANKML_PIN_MEM=2500M testing/pinned.sh <command …>
set -euo pipefail
# default: the last two CPUs this shell is ALLOWED to use (a cpuset may start anywhere, e.g. 4-7)
allowed=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
cpus=${BANKML_PIN_CPUS:-$(echo "$allowed" | tr ',' '\n' | tail -2 | paste -sd,)}
mem=${BANKML_PIN_MEM:-2500M}
state() { echo "# $1: load $(cut -d' ' -f1-3 /proc/loadavg) · MemAvailable $(awk '/MemAvailable/{printf "%.2f GB", $2/1e6}' /proc/meminfo) · cpus $cpus · mem cap $mem"; }
state before
# the cap is claimed only if a probe scope really shows it in its cgroup's memory.max (the controller is delegated)
capped=$(systemd-run --user --scope -q -p MemoryMax="$mem" sh -c 'cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max' 2>/dev/null || true)
if [ -n "$capped" ] && [ "$capped" != "max" ]; then
  echo "# memory cap enforced: memory.max = $capped bytes"
  systemd-run --user --scope -q -p MemoryMax="$mem" -p MemorySwapMax=0 taskset -c "$cpus" "$@"
else
  echo "# no user cgroup: CPU pinned, memory NOT capped"
  taskset -c "$cpus" "$@"
fi
state after
