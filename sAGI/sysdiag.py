#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""What the machine under bankML is doing now: CPU, memory, disk and GPU, read from /proc, /sys and statvfs.

Measured here, so it answers without an engine; the engine's own view (`GET /bankml/status`, `/bankml/usage`) is
merged in by the caller when serve answers. Every value is what was read, or absent; nothing is estimated. Each
section carries a level — ok, warn or bad — from fixed thresholds (THRESHOLDS), and the lines a person reads.
Linux only; elsewhere the sections say what could not be read. Details: docs/modules/console.md
"""
from __future__ import annotations

import html
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

THRESHOLDS = {"temp_warn_c": 85.0, "temp_bad_c": 95.0, "mem_warn_bytes": 1_000_000_000, "mem_bad_bytes": 400_000_000,
              "disk_warn_bytes": 5_000_000_000, "disk_bad_bytes": 1_500_000_000, "swap_warn_frac": 0.8}
GB = 1e9


def _read(p: str | Path, default: str | None = None) -> str | None:
    try:
        return Path(p).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return default


def _int(p: str | Path) -> int | None:
    v = _read(p)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def _gb(b: float | None) -> str:
    return "not read" if b is None else f"{b / GB:.2f} GB"


def _level(*levels: str) -> str:
    return "bad" if "bad" in levels else "warn" if "warn" in levels else "ok"


def temps(root: str = "/sys/class/hwmon") -> dict:
    """{sensor name: degrees C} from hwmon's temp1_input (k10temp/coretemp = CPU, amdgpu = GPU, nvme = disk)."""
    out = {}
    for h in sorted(Path(root).glob("hwmon*")):
        name, t = _read(h / "name"), _int(h / "temp1_input")
        if name and t is not None:
            out[name] = t / 1000.0
    return out


def cpu_busy(sample_s: float = 0.2, stat: str = "/proc/stat") -> float | None:
    """The whole machine's CPU busy share over `sample_s`, in percent, from two reads of /proc/stat."""
    def read():
        line = (_read(stat) or "").splitlines()[:1]
        if not line or not line[0].startswith("cpu "):
            return None
        v = [int(x) for x in line[0].split()[1:]]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        return sum(v), idle
    a = read()
    if a is None:
        return None
    time.sleep(sample_s)
    b = read()
    if b is None or b[0] == a[0]:
        return None
    return 100.0 * (1 - (b[1] - a[1]) / (b[0] - a[0]))


def cpu(t: dict, busy: float | None) -> dict:
    info = _read("/proc/cpuinfo") or ""
    model = next((l.split(":", 1)[1].strip() for l in info.splitlines() if l.startswith("model name")), None)
    mhz = [float(l.split(":", 1)[1]) for l in info.splitlines() if l.startswith("cpu MHz")]
    load = (_read("/proc/loadavg") or "").split()[:3]
    temp = t.get("k10temp", t.get("coretemp", t.get("cpu_thermal")))
    lvl = "ok" if temp is None else "bad" if temp >= THRESHOLDS["temp_bad_c"] else "warn" if temp >= THRESHOLDS["temp_warn_c"] else "ok"
    lines = [f"model {model or 'not read'} · {os.cpu_count()} logical cores",
             "clock " + (" · ".join(f"{m:.0f}" for m in mhz) + " MHz" if mhz else "not read"),
             "busy " + (f"{busy:.0f} % of the machine" if busy is not None else "not read")
             + (f" · load {' / '.join(load)} (1, 5, 15 min)" if load else ""),
             "temperature " + (f"{temp:.1f} °C" if temp is not None else "not read")
             + (" — throttling likely" if lvl == "bad" else " — hot" if lvl == "warn" else "")]
    return {"title": "CPU", "level": lvl, "lines": lines,
            "data": {"model": model, "logical": os.cpu_count(), "mhz": mhz, "busy_percent": busy, "load": load, "temp_c": temp}}


def memory() -> dict:
    m = {}
    for l in (_read("/proc/meminfo") or "").splitlines():
        k, _, v = l.partition(":")
        try:
            m[k] = int(v.split()[0]) * 1024
        except (ValueError, IndexError):
            pass
    avail, total = m.get("MemAvailable"), m.get("MemTotal")
    st, sf = m.get("SwapTotal"), m.get("SwapFree")
    used_swap = (st - sf) if st is not None and sf is not None else None
    lvl = "ok"
    if avail is not None:
        lvl = "bad" if avail < THRESHOLDS["mem_bad_bytes"] else "warn" if avail < THRESHOLDS["mem_warn_bytes"] else "ok"
    if st and used_swap is not None and used_swap / st >= THRESHOLDS["swap_warn_frac"]:
        lvl = _level(lvl, "warn")
    lines = [f"available {_gb(avail)} of {_gb(total)}",
             f"swap used {_gb(used_swap)} of {_gb(st)}" if st else "no swap"]
    return {"title": "Memory", "level": lvl, "lines": lines,
            "data": {"total_bytes": total, "available_bytes": avail, "swap_total_bytes": st, "swap_used_bytes": used_swap}}


def _device_of(path: str) -> str | None:
    """The block device (e.g. nvme0n1) holding `path`, from /proc/self/mountinfo's longest matching mount point."""
    best, dev = "", None
    for l in (_read("/proc/self/mountinfo") or "").splitlines():
        parts = l.split(" - ")
        if len(parts) != 2:
            continue
        mnt, src = parts[0].split()[4], parts[1].split()[1]
        if path.startswith(mnt) and len(mnt) > len(best) and src.startswith("/dev/"):
            best, dev = mnt, src[5:]
    return dev


def disk(paths: dict, t: dict) -> dict:
    """Free space for each named path (the models, the UI's state), the device under the models: SSD or spinning disk,
    and the drive's temperature."""
    lines, data, lvl = [], {}, "ok"
    for name, p in paths.items():
        try:
            s = os.statvfs(p)
        except OSError:
            lines.append(f"{name}: {p} not read")
            continue
        free, total = s.f_bavail * s.f_frsize, s.f_blocks * s.f_frsize
        l = "bad" if free < THRESHOLDS["disk_bad_bytes"] else "warn" if free < THRESHOLDS["disk_warn_bytes"] else "ok"
        lvl = _level(lvl, l)
        lines.append(f"{name}: {_gb(free)} free of {_gb(total)} ({100 * (1 - free / total):.0f} % used) · {p}"
                     + (" — nearly full" if l != "ok" else ""))
        data[name] = {"path": str(p), "free_bytes": free, "total_bytes": total}
    first = next(iter(paths.values()), None)
    dev = _device_of(str(Path(first).resolve())) if first else None
    if dev:
        base = dev
        while base and not Path(f"/sys/block/{base}").exists():
            base = base[:-1]  # nvme0n1p2 → nvme0n1, sda1 → sda
        rota = _read(f"/sys/block/{base}/queue/rotational") if base else None
        kind = {"0": "solid state", "1": "spinning disk"}.get(rota or "", "kind not read")
        model = _read(f"/sys/block/{base}/device/model") if base else None
        lines.append(f"device {dev} · {kind}" + (f" · {model}" if model else ""))
        data["device"] = {"name": dev, "rotational": rota == "1" if rota else None, "model": model}
    if "nvme" in t:
        lines.append(f"drive temperature {t['nvme']:.1f} °C")
        data["temp_c"] = t["nvme"]
    return {"title": "Disk", "level": lvl, "lines": lines, "data": data}


def gpu(t: dict, root: str = "/sys/class/drm") -> dict:
    """Each card's busy share and its memory (VRAM, and GTT — system memory it maps) from amdgpu's sysfs, and its
    temperature. Other drivers expose less; what was not read says so."""
    lines, cards = [], []
    for c in sorted(Path(root).glob("card[0-9]*")):
        if "-" in c.name or not (c / "device").exists():
            continue  # connectors (card1-eDP-1) are not cards
        d = c / "device"
        vendor = {"0x1002": "AMD", "0x10de": "NVIDIA", "0x8086": "Intel"}.get(_read(d / "vendor") or "", _read(d / "vendor"))
        busy, vu, vt, gu = _int(d / "gpu_busy_percent"), _int(d / "mem_info_vram_used"), _int(d / "mem_info_vram_total"), _int(d / "mem_info_gtt_used")
        card = {"card": c.name, "vendor": vendor, "device": _read(d / "device"), "busy_percent": busy,
                "vram_used_bytes": vu, "vram_total_bytes": vt, "gtt_used_bytes": gu}
        cards.append(card)
        lines.append(f"{c.name}: {vendor or 'vendor not read'} {card['device'] or ''} · busy "
                     + (f"{busy} %" if busy is not None else "not read")
                     + (f" · VRAM {_gb(vu)} of {_gb(vt)}" if vt else "") + (f" · GTT {_gb(gu)}" if gu is not None else ""))
    temp = t.get("amdgpu", t.get("nouveau", t.get("i915")))
    lvl = "ok" if temp is None else "bad" if temp >= THRESHOLDS["temp_bad_c"] else "warn" if temp >= THRESHOLDS["temp_warn_c"] else "ok"
    if temp is not None:
        lines.append(f"temperature {temp:.1f} °C" + (" — hot" if lvl != "ok" else ""))
    if not cards:
        lines.append("no card found under /sys/class/drm")
    return {"title": "GPU", "level": lvl, "lines": lines, "data": {"cards": cards, "temp_c": temp}}


def collect(paths: dict, sample_s: float = 0.2) -> list:
    """The sections, in reading order: CPU, memory, disk, GPU."""
    t = temps()
    return [cpu(t, cpu_busy(sample_s)), memory(), disk(paths, t), gpu(t)]


def ping(base: str, path: str = "/health", timeout: float = 5.0) -> dict:
    """One round trip to bankml serve (`GET /health` runs no model), timed here: {"engine", "engine_ok", "engine_ms",
    "why", "at"}."""
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            r.read()
            ok, why = r.status == 200, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        ok, why = False, f"HTTP {e.code}"
    except OSError as e:
        ok, why = False, str(getattr(e, "reason", e))
    return {"engine": base, "engine_ok": ok, "engine_ms": round((time.perf_counter() - t0) * 1000, 2), "why": why, "at": time.time()}


def engine(st: dict, u: dict, p: dict) -> dict:
    """The engine's own view (`GET /bankml/status`, `/bankml/usage`) and a ping, as a section."""
    if not st:
        return {"title": "Engine", "level": "bad", "lines": [f"bankml serve does not answer at {p.get('engine')} ({p.get('why')})"], "data": {"ping": p}}
    v, sv, disk = st.get("verified") or {}, st.get("serve") or {}, st.get("disk") or {}
    lim = u.get("gpu_limiter") or {}
    lines = [f"{v.get('name') or 'model'} · sha256 {str(v.get('model_sha256') or '')[:16]}… · verified {v.get('verdict') or v.get('guard') or '—'}",
             f"bankML {v.get('bankml') or st.get('bankml') or '—'} · {'native' if sv.get('native') else 'llama-server behind it'}"
             f" · pid {sv.get('pid')} · up {sv.get('uptime_s')} s · threads {sv.get('threads')}",
             f"ping {p.get('engine_ms')} ms ({p.get('why')})",
             "engine CPU " + (f"{u['cpu_percent']:.0f} % of one core" if u.get("cpu_percent") is not None else "not measured")
             + " · memory held " + (f"{u['rss_bytes'] / GB:.2f} GB" if u.get("rss_bytes") is not None else "not measured"),
             "engine reads " + (f"{disk['read_bytes'] / GB:.2f} GB" if disk.get("read_bytes") is not None else "not measured")
             + " from disk · model file " + (f"{disk['model_bytes'] / GB:.2f} GB" if disk.get("model_bytes") is not None else "not read"),
             "GPU limit " + (f"{lim['limit'] * 100:.0f} %" if lim.get("limit") is not None else "none")
             + " · GPU memory held " + (f"{lim['allocated_bytes'] / 1e6:.0f} MB" if lim.get("allocated_bytes") is not None else "not measured")]
    return {"title": "Engine", "level": "ok" if p.get("engine_ok") else "bad", "lines": lines, "data": {"ping": p}}


def to_html(sections: list, cls: str = "sv-diag") -> str:
    """The sections as an accordion for a page: <details> per section, its mark and level, those that need a look
    open. Every text escaped."""
    mark = {"ok": "✓", "warn": "!", "bad": "✗"}
    out = []
    for s in sections:
        lv = s.get("level", "ok")
        lines = "".join(f"<div class='{cls}-l'>{html.escape(str(l))}</div>" for l in s.get("lines", []))
        out.append(f"<details class='{cls}-s is-{lv}'{' open' if lv != 'ok' else ''}><summary><span class='{cls}-m'>{mark.get(lv, '?')}</span>"
                   f"{html.escape(str(s.get('title', '')))}<span class='{cls}-lv'>{lv}</span></summary>{lines}</details>")
    return f"<div class='{cls}'>" + "".join(out) + "</div>"
