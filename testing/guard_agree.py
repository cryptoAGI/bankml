#!/usr/bin/env python3
"""Rust guard == Python guard, JSON for JSON. stdlib only.
Runs `bankml guard F --json` and `gguf_guard.py F --json` on every file given plus the synthetic
GGUFs of minaiml's test_gguf_guard.py (written to a temp dir), for both engines, and compares the
parsed objects. Exit 0 only if every pair is identical (verdict, arch, types, reasons, kv bytes/token…).
usage: guard_agree.py [BANKML_BIN] [FILE ...]"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "gguf_guard.py")   # vendored from minaiml (tools/gguf_guard.py + test_gguf_guard.py)
sys.path.insert(0, os.path.dirname(PY))
import types
T = types.SimpleNamespace(__file__=os.path.join(os.path.dirname(PY), "test_gguf_guard.py"), __name__="test_gguf_guard")
try:  # the suite runs at module level and ends in sys.exit(); keep its namespace
    exec(compile(open(os.path.join(os.path.dirname(PY), "test_gguf_guard.py")).read(), "test_gguf_guard.py", "exec"), T.__dict__)
except SystemExit as e:
    if e.code:
        sys.exit("python guard suite failed; not comparing")

bin_ = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "target", "release", "bankml")
files = sys.argv[2:]
tmp = tempfile.mkdtemp(prefix="guard_agree_")
for i, (name, ts, blk, eng, want) in enumerate(T.cases):
    p = os.path.join(tmp, f"case{i}-{name.replace(' ', '_')}.gguf")
    open(p, "wb").write(T.gguf(name, ts, blk))
    files.append(p)
p = os.path.join(tmp, "kv-Bonsai-8B.gguf"); open(p, "wb").write(T.with_kv(T.gguf("Bonsai-8B", [("a", T.N, 41)], (128, 18)))); files.append(p)
p = os.path.join(tmp, "truncated.gguf"); open(p, "wb").write(T.gguf("x", [("a", T.N, 42)], (64, 18))[:40]); files.append(p)

bad = 0
for f in files:
    for eng in ("mainline", "prism"):
        py = json.loads(subprocess.run([sys.executable, PY, f, "--engine", eng, "--json"], capture_output=True, text=True).stdout)
        rs = json.loads(subprocess.run([bin_, "guard", f, "--engine", eng, "--json"], capture_output=True, text=True).stdout)
        ok = py == rs
        bad += not ok
        print("agree" if ok else "DIFFER", f"{eng:8} {py.get('verdict'):9} {os.path.basename(f)}", "" if ok else f"\n  py={py}\n  rs={rs}")
print(f"{len(files) * 2 - bad}/{len(files) * 2} agree")
sys.exit(1 if bad else 0)
