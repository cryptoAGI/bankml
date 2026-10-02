#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Every tracked source file names its licence in an SPDX header, and the header matches its layer (LICENSING.md):
crypto/ → GPL-3.0-only, agpl/ → AGPL-3.0-only (alone or with MIT), upstream/ → MIT (llama.cpp's licence),
everything else → MIT OR Apache-2.0. C sources and headers count since 0.3.2 (the C API and its oracles).
run: python3 testing/spdx_check.py   (exit 0 = every file agrees)"""
import re, subprocess, sys
from pathlib import Path

SRC = re.compile(r"\.(rs|py|mjs|jsx|sh|c|h)$")
files = [f for f in subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.split() if SRC.search(f)]
root = Path(__file__).resolve().parents[1]
bad = []
for f in files:
    head = "\n".join((root / f).read_text(encoding="utf-8", errors="replace").splitlines()[:3])
    m = re.search(r"SPDX-License-Identifier:\s*(.+?)\s*(?:\*/)?\s*$", head, re.M)
    want = ("GPL-3.0-only",) if f.startswith("crypto/") else ("AGPL-3.0-only", "MIT AND AGPL-3.0-only") if f.startswith("agpl/") \
        else ("MIT",) if f.startswith("upstream/") else ("MIT OR Apache-2.0",)
    if not m:
        bad.append(f"{f}: no SPDX header")
    elif m.group(1) not in want:
        bad.append(f"{f}: {m.group(1)!r}, expected {' or '.join(want)}")
for b in bad:
    print("FAIL", b)
print(f"spdx: {len(files) - len(bad)}/{len(files)} files carry the licence of their layer")
sys.exit(1 if bad else 0)
