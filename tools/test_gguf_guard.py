#!/usr/bin/env python3
# Vendored into bankml from minaiml (same authors); the Rust guard in gguf.rs is its port — keep them in step.
# synthetic GGUFs, one per trap. stdlib only. run: python3 test_gguf_guard.py
import struct, os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from gguf_guard import judge

def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b

def gguf(name, tensors, pad_blocks):
    """tensors: [(name, n_elements, type_id)]; pad_blocks: (elems_per_block, bytes_per_block) actually written."""
    kv = [("general.architecture", "qwen3"), ("general.name", name), ("general.alignment", None)]
    h = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, v in kv:
        h += s(k)
        h += struct.pack("<I", 4) + struct.pack("<I", 32) if v is None else struct.pack("<I", 8) + s(v)
    off, sizes = 0, []
    for tn, n, ty in tensors:
        per, bb = pad_blocks if ty == 42 or ty >= 142 else (1, 2)
        sz = n // per * bb
        h += s(tn) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<I", ty) + struct.pack("<Q", off)
        off += (sz + 31) // 32 * 32
        sizes.append(sz)
    h += b"\0" * ((-len(h)) % 32)
    return h + b"\0" * off

N = 4096 * 4
cases = [
    ("Bonsai-1.7B",          [("a", N, 41), ("b", N, 1)],           (128, 18), "mainline", "play"),
    ("Ternary-Bonsai-1.7B",  [("a", N, 42), ("b", N, 42)],          (64, 18),  "mainline", "play"),
    ("Ternary-Bonsai-1.7B",  [("a", N, 42), ("b", N, 42)],          (128, 34), "mainline", "refuse"),  # legacy g128
    ("Ternary-Bonsai-1.7B",  [("a", N, 142)],                       (128, 34), "mainline", "refuse"),  # PQ2_0
    ("Ternary-Bonsai-1.7B",  [("a", N, 142)],                       (128, 34), "prism",    "play"),
    ("Ternary Bonsai 2 27B", [("a", N, 42)],                        (64, 18),  "mainline", "refuse"),  # rotated
    ("Ternary Bonsai 2 27B", [("a", N, 42)],                        (64, 18),  "prism",    "play"),
]
fail = 0
for name, ts, blk, eng, want in cases:
    b = gguf(name, ts, blk)
    got = judge(b, eng, len(b), name)
    ok = got["verdict"] == want
    fail += not ok
    print(("ok  " if ok else "FAIL"), f"{name:22} {eng:8} {want:6} -> {got['verdict']}", got.get("reasons", []))
# KV bytes/token (Qwen3-8B shape)
def with_kv(b):
    extra = (s("general.architecture") + struct.pack("<I", 8) + s("qwen3")
             + s("qwen3.block_count") + struct.pack("<II", 4, 36)
             + s("qwen3.attention.head_count_kv") + struct.pack("<II", 4, 8)
             + s("qwen3.attention.key_length") + struct.pack("<II", 4, 128))
    # patch kv count (+4) and splice extra KV after the header counts; offsets are data-relative
    nt, nkv = struct.unpack_from("<QQ", b, 8)
    hdr = b[:8] + struct.pack("<QQ", nt, nkv + 4)
    rest = b[24:]
    raw = hdr + extra + rest
    return raw
b = with_kv(gguf("Bonsai-8B", [("a", N, 41)], (128, 18)))
r = judge(b, "mainline", None, "Bonsai-8B")
ok = r.get("kv_f16_bytes_per_token") == 147456
print("ok   kv 147456" if ok else f"FAIL kv {r}")
fail += not ok
# truncated prefix
b = gguf("x", [("a", N, 42)], (64, 18))
r = judge(b[:40], "mainline")
print("ok   need_more" if r["verdict"] == "need_more" else "FAIL need_more")
fail += r["verdict"] != "need_more"
sys.exit(1 if fail else 0)
