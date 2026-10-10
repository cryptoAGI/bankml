#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""bankml diag against psutil, ss and ip on the same machine.

    python3 testing/diag_oracle.py [target/release/bankml]

Runs `bankml diag --json --full` on this oracle's own process, with a TCP ping to a socket this script listens on and
to a port nothing listens on, then reads the same things through psutil (an independent implementation of the same
/proc and /sys readings), `ss -ltnH` / `ss -lunH` and `ip -s -j link`. A value that is fixed (cores, memory total,
boot time, the listening sockets, interface names, disk size) must be equal; a counter that only grows (bytes, packets,
I/O, context switches) must lie between bankml's reading and psutil's later one; a level that moves (available memory,
load, temperatures) must agree within a stated tolerance. Readings that nothing independent reports here (pressure,
the CPU busy share of one instant, the ping times themselves) are listed as unchecked, not counted as passed.
"""
import json
import os
import socket
import subprocess
import sys
import threading

import psutil

BIN = sys.argv[1] if len(sys.argv) > 1 else "target/release/bankml"
MB = 1 << 20
checks, unchecked = [], []


def check(name, ok, seen=""):
    checks.append((name, bool(ok), seen))


def main():
    me = psutil.Process()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    dead_port = closed.getsockname()[1]
    closed.close()  # a port nothing listens on
    stop = threading.Event()

    def accept():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                srv.accept()[0].close()
            except OSError:
                pass

    threading.Thread(target=accept, daemon=True).start()
    out = subprocess.run([BIN, "diag", "--json", "--full", "--sample", "300", "--ping", f"127.0.0.1:{port}", "--ping", f"127.0.0.1:{dead_port}", str(os.getpid())],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    d = json.loads(out.stdout)
    # psutil's readings, taken after bankml's: counters can only have grown
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    nics = psutil.net_io_counters(pernic=True)
    conns = psutil.net_connections("inet")
    temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
    io = psutil.disk_io_counters(perdisk=True)
    load = os.getloadavg()
    stop.set()

    # system and cpu
    check("cores = psutil.cpu_count()", d["cpu"]["logical"] == psutil.cpu_count(), f'{d["cpu"]["logical"]} / {psutil.cpu_count()}')
    check("one busy share per core", len(d["cpu"]["per_core_busy_percent"]) == psutil.cpu_count())
    check("busy shares within 0..100", all(b is None or 0 <= b <= 100 for b in [d["cpu"]["busy_percent"], *d["cpu"]["per_core_busy_percent"]]))
    check("boot time = psutil.boot_time()", abs(d["system"]["boot_time"] - psutil.boot_time()) <= 1, f'{d["system"]["boot_time"]} / {psutil.boot_time():.0f}')
    check("load averages = os.getloadavg() (±0.15, they move every 5 s)", all(abs(a - b) <= 0.15 for a, b in zip(d["cpu"]["load"], load)), f'{d["cpu"]["load"]} / {[round(x, 2) for x in load]}')
    check("host name = socket.gethostname()", d["system"]["hostname"] == socket.gethostname())
    t_ours = [(t["chip"], t["label"], t["celsius"]) for t in d["cpu"]["temperatures"]]
    t_ps = [(chip, s.label, s.current) for chip, ss in temps.items() for s in ss]
    check("temperatures: the same sensors as psutil", sorted((c, l) for c, l, _ in t_ours) == sorted((c, l) for c, l, _ in t_ps), f"{len(t_ours)} / {len(t_ps)}")
    check("temperatures within 3 °C of psutil's", all(any(c == c2 and l == l2 and abs(v - v2) <= 3 for c2, l2, v2 in t_ps) for c, l, v in t_ours))

    # memory
    m = d["memory"]
    check("memory total = psutil's", m["total_bytes"] == vm.total, f'{m["total_bytes"]} / {vm.total}')
    check("swap total = psutil's", m["swap_total_bytes"] == sw.total)
    check("memory available within 64 MB of psutil's", abs(m["available_bytes"] - vm.available) <= 64 * MB, f'{m["available_bytes"] // MB} / {vm.available // MB} MB')
    check("memory used within 64 MB of psutil's", abs(m["used_bytes"] - vm.used) <= 64 * MB, f'{m["used_bytes"] // MB} / {vm.used // MB} MB')
    check("cached within 64 MB of psutil's", abs(m["cached_bytes"] - vm.cached) <= 64 * MB)
    check("buffers within 16 MB of psutil's", abs(m["buffers_bytes"] - vm.buffers) <= 16 * MB)

    # this process
    p = next(x for x in d["processes"] if x["pid"] == os.getpid())
    check("threads = psutil num_threads()", p["threads"] == me.num_threads(), f'{p["threads"]} / {me.num_threads()}')
    check("open descriptors = psutil num_fds() (±2: the subprocess pipes)", abs(p["fds"] - me.num_fds()) <= 2, f'{p["fds"]} / {me.num_fds()}')
    cs = me.num_ctx_switches()
    check("context switches ≤ psutil's later reading", p["ctx_voluntary"] <= cs.voluntary and p["ctx_involuntary"] <= cs.involuntary)
    check("RSS within 4 MB of psutil's", abs(p["rss_bytes"] - me.memory_info().rss) <= 4 * MB, f'{p["rss_bytes"] // MB} / {me.memory_info().rss // MB} MB')
    check("virtual size = psutil vms (±4 MB)", abs(p["vm_bytes"] - me.memory_info().vms) <= 4 * MB)
    check("command = psutil name()", p["command"] == me.name())
    check("age within 2 s of psutil's create_time()", abs(p["age_s"] - (psutil.time.time() - me.create_time())) <= 2)

    # disks
    root = next(x for x in d["disks"] if x["name"] == "root")
    check("/ total = psutil disk_usage('/').total", root["total_bytes"] == psutil.disk_usage("/").total)
    dev = root["device"]
    if dev in io:
        c = io[dev]
        check(f"{dev} I/O counters ≤ psutil's later reading (+64 MB)", root["read_bytes"] <= c.read_bytes <= root["read_bytes"] + 64 * MB and root["written_bytes"] <= c.write_bytes <= root["written_bytes"] + 64 * MB,
              f'read {root["read_bytes"] // MB} / {c.read_bytes // MB} MB')
        check(f"{dev} reads and writes ≤ psutil's later reading", root["reads"] <= c.read_count and root["writes"] <= c.write_count)
    else:
        check(f"root device {dev} known to psutil", False, str(list(io)))

    # network: interfaces
    ours = {n["name"]: n for n in d["network"]["interfaces"]}
    check("interfaces = psutil's", set(ours) == set(nics), f"{sorted(ours)} / {sorted(nics)}")
    ip = {i["ifname"]: i["stats64"] for i in json.loads(subprocess.run(["ip", "-s", "-j", "link"], capture_output=True, text=True).stdout)}
    check("interfaces = ip link's", set(ours) == set(ip))
    grow = lambda a, b, slack: a <= b <= a + slack
    check("interface counters ≤ psutil's later reading (+16 MB, +20,000 packets)", all(
        grow(ours[k]["rx_bytes"], v.bytes_recv, 16 * MB) and grow(ours[k]["tx_bytes"], v.bytes_sent, 16 * MB)
        and grow(ours[k]["rx_packets"], v.packets_recv, 20000) and grow(ours[k]["tx_packets"], v.packets_sent, 20000)
        and ours[k]["rx_errors"] <= v.errin and ours[k]["rx_drops"] <= v.dropin for k, v in nics.items() if k in ours))

    # network: listening sockets against ss and psutil
    def norm(addr):  # "[::1]:7873", "127.0.0.53%lo:53", "*:22" → (ip, port)
        h, _, pt = addr.rpartition(":")
        h = h.split("%")[0].strip("[]")  # the scope (%wlo1) first: /proc/net/*6 carries none
        return ("0.0.0.0" if h == "*" else h), int(pt)
    def ss(flag):
        return {norm(r.split()[3]) for r in subprocess.run(["ss", flag, "-H"], capture_output=True, text=True).stdout.splitlines() if r.strip()}
    ours_l = {norm(s["local"]) for s in d["network"]["listening"] if s["proto"].startswith("tcp")}
    ss_l = ss("-ltn")
    check("listening TCP = ss -ltn", ours_l == ss_l, f"{len(ours_l)} / {len(ss_l)}; only bankml {sorted(ours_l - ss_l)}, only ss {sorted(ss_l - ours_l)}")
    ours_u = {norm(s["local"]) for s in d["network"]["listening"] if s["proto"].startswith("udp")}
    ss_u = ss("-lun")
    check("unconnected UDP = ss -lun", ours_u == ss_u, f"{len(ours_u)} / {len(ss_u)}; only bankml {sorted(ours_u - ss_u)}, only ss {sorted(ss_u - ours_u)}")
    ps_l = {(c.laddr.ip, c.laddr.port) for c in conns if c.status == psutil.CONN_LISTEN}
    check("listening TCP = psutil net_connections()", ours_l == ps_l, f"{len(ours_l)} / {len(ps_l)}")
    check(f"this oracle's socket 127.0.0.1:{port} listed as its own", any(s["local"] == f"127.0.0.1:{port}" and s["pid"] == os.getpid() for s in d["network"]["listening"]))
    owners_ps = {(c.laddr.ip, c.laddr.port): c.pid for c in conns if c.status == psutil.CONN_LISTEN and c.pid}
    owners_ours = {norm(s["local"]): s["pid"] for s in d["network"]["listening"] if s["proto"].startswith("tcp") and s["pid"]}
    check("listening sockets' owners = psutil's (where readable)", all(owners_ours.get(k) == v for k, v in owners_ps.items()), f"{len(owners_ours)} / {len(owners_ps)} owned")
    est_ours = d["network"]["sockets"].get("tcp_estab", 0)
    est_ps = sum(1 for c in conns if c.type == socket.SOCK_STREAM and c.status == psutil.CONN_ESTABLISHED)
    check("established TCP count = psutil's (±4, connections come and go)", abs(est_ours - est_ps) <= 4, f"{est_ours} / {est_ps}")

    # ping
    pg = {x["target"]: x for x in d["ping"]}
    live, dead = pg[f"127.0.0.1:{port}"], pg[f"127.0.0.1:{dead_port}"]
    check("ping to a listening socket: 3 of 3 connected", live["connected"] == 3 and live["min_ms"] <= live["avg_ms"] <= live["max_ms"], json.dumps(live))
    check("ping to a closed port: 0 of 3, refused", dead["connected"] == 0 and "refused" in (dead["error"] or "").lower(), json.dumps(dead))

    unchecked.extend(["pressure (PSI): psutil does not read it", "the busy share of one 300 ms sample (only its range)", "ping times (only their order and count)",
                      "GPU and package power: sys.rs's, checked in its own tests"])
    passed = sum(ok for _, ok, _ in checks)
    for name, ok, seen in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name}{'  — ' + seen if seen else ''}")
    for u in unchecked:
        print(f"     unchecked: {u}")
    print(f"diag oracle: {passed} of {len(checks)} readings of bankml diag agree with psutil {psutil.__version__}, ss and ip on this machine")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
