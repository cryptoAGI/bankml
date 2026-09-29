#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Draw the README's speed cards from a release-gate record, so they cannot drift from what was measured.
Each card names the record it was drawn from. The proof cards (bit-exact counts) are unchanged by speed and are left
as they are.
usage: python3 tools/cards.py [testing/results/<version>.txt]   (default: the newest record)"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
GOLD, BLUE, GREEN, INK, DIM, BG, EDGE = "#D9A23A", "#5AD1FF", "#56D364", "#e6edf3", "#8b949e", "#0d1117", "#30363d"


def vkey(p: Path):
    return tuple(int(x) for x in p.stem.split("."))


def section(text: str, name: str) -> str:
    m = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1) if m else ""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def card(title: str, rows: list, foot: list, unit: str) -> str:
    """rows: (label, bankml value, llama.cpp value, ratio text); bars share one scale per card."""
    h = 54 + 46 * len(rows) + 18 * len(foot) + 10
    top = max(max(b, g) for _, b, g, _ in rows)
    scale = 280 / top
    t = lambda x, y, s, size=12, fill=INK, w="400", anchor="start": (  # noqa: E731
        f"<text x='{x}' y='{y}' font-family='{MONO}' font-size='{size}' font-weight='{w}' text-anchor='{anchor}' fill='{fill}'>{esc(s)}</text>")
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='720' height='{h}' viewBox='0 0 720 {h}' role='img' aria-label='{esc(title)}'>",
           f"<rect x='0.5' y='0.5' width='719' height='{h - 1}' rx='10' fill='{BG}' stroke='{EDGE}'/>", t(20, 32, title, 15, INK, "700")]
    y = 70
    for label, b, g, ratio in rows:
        out.append(t(20, y, label, 12, DIM))
        for dy, v, col, who in ((-2, b, GOLD, "bankml"), (10, g, BLUE, "llama.cpp")):
            w = max(2.0, v * scale)
            out.append(f"<rect x='200' y='{y + dy}' width='{w:.1f}' height='9' rx='2' fill='{col}'/>")
            out.append(t(round(208 + w, 1), y + dy + 8, f"{v:g} {unit} · {who}", 10.5))
        out.append(t(700, y, ratio, 12, GREEN, "700", "end"))
        y += 46
    for line in foot:
        out.append(t(20, y, line, 10.5, DIM))
        y += 18
    return "".join(out) + "</svg>\n"


def main():
    rec = Path(sys.argv[1]) if len(sys.argv) > 1 else max((ROOT / "testing/results").glob("*.txt"), key=vkey)
    text = rec.read_text()
    v = rec.stem
    cpu = re.search(r"· (.+?) · (\d+) threads", text)
    machine = cpu.group(1).replace(" with Radeon Vega Mobile Gfx", "") if cpu else "this machine"
    # 1. ternary kernel, per projection (decode) and the prefill tile
    q2 = section(text, "ab_vs_ggml_q2_0")
    rows = []
    for m in re.finditer(r"blk\.0\.(\w+)\.weight\s+(\d+x\d+)\s+ggml\s+[\d.]+/\s*([\d.]+).*?bankml\s+[\d.]+/\s*([\d.]+).*?\|\s*([\d.]+)x", q2):
        rows.append((f"{m.group(1)} {m.group(2).replace('x', ' × ')}", float(m.group(4)), float(m.group(3)), f"{float(m.group(5)):.2f}×"))
    pf = re.search(r"prefill .*?ggml per-pair [\d.]+/([\d.]+).*?1x4 tile [\d.]+/([\d.]+) \(([\d.]+)x\)", q2)
    if pf:
        rows.append(("prefill tile 1×4", float(pf.group(2)), float(pf.group(1)), f"{float(pf.group(3)):.2f}×"))
    if rows:
        (ROOT / "docs/cards/ternary_speed.svg").write_text(card(
            "Ternary Q2_0_g64 kernel vs llama.cpp b11192 — same bits, 1 thread", rows,
            [f"median ns per 64-weight block (prefill: per block·column) · {machine} · 1 thread",
             f"gate record testing/results/{v}.txt · every oracle bit-exact in the same run"], "ns"))
    # 2. 1-bit kernel: decode GEMV and the prefill tile
    q1 = section(text, "ab_vs_ggml")
    g = re.search(r"ggml b11192 [\d.]+/([\d.]+) \|.*?vec_dot_act [\d.]+/([\d.]+) \(([\d.]+)x\)", q1)
    p = re.search(r"ggml per-pair [\d.]+/([\d.]+).*?Q8Act tile [\d.]+/([\d.]+) \(([\d.]+)x\)", q1)
    if g and p:
        (ROOT / "docs/cards/q1_speed.svg").write_text(card(
            "1-bit Q1_0 kernel vs llama.cpp b11192 — same bits, 1 thread",
            [("decode GEMV 12288 × 4096", float(g.group(2)), float(g.group(1)), f"{float(g.group(3)):.2f}×"),
             ("prefill tile 1×4", float(p.group(2)), float(p.group(1)), f"{float(p.group(3)):.2f}×")],
            [f"median ns per 128-weight block (prefill: per block·column) · {machine} · 1 thread",
             f"gate record testing/results/{v}.txt · decode at parity: the 1-bit kernel is at the core's instruction limit"], "ns"))
    # 3. one ternary token, all matmuls, 3 threads
    db = section(text, "decode_budget_q2_0")
    m = re.search(r"3 thread\(s\): ggml b11192 ([\d.]+)/([\d.]+) s/token.*?bankml ([\d.]+)/([\d.]+) s/token \| ([\d.]+)x \| matmul-only ceiling ggml ([\d.]+), bankml ([\d.]+)", db)
    floor = re.search(r"3 thread\(s\): [\d.]+ GB/s read → floor for one ternary token \(([\d.]+) GB\) ([\d.]+) s", section(text, "bench_memory_floor"))
    if m:
        gmin, gmed, bmin, bmed = (float(m.group(k)) for k in (1, 2, 3, 4))
        b, gg = bmin, gmin
        note = ("the model stayed resident in memory" if bmed < 0.6 else
                "NOT resident: too little free memory to cache the 2.31 GB file, so this run measured the disk")
        (ROOT / "docs/cards/decode_budget.svg").write_text(card(
            "One ternary token: all 253 matmuls, 3 threads", [("253 matmuls, 1 token", b, gg, f"{gg / b:.2f}×")],
            [f"fastest of the runs (min) · median {bmed:g} s vs {gmed:g} s ({gmed / bmed:.2f}×) · {machine}",
             note,
             f"matmul ceiling {m.group(7)} tok/s vs {m.group(6)}" + (f" · memory floor {floor.group(2)} s/token ({floor.group(1)} GB read)" if floor else ""),
             f"gate record testing/results/{v}.txt · the forward pass (P3) is not built yet"], "s"))
    print(f"cards drawn from testing/results/{v}.txt")


if __name__ == "__main__":
    main()
