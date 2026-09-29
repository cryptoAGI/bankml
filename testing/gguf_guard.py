#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Vendored into bankml from minaiml (same authors); the Rust guard in gguf.rs is its port — keep them in step.
# gguf_guard.py — decide whether a GGUF can play on a given llama.cpp build, from its header alone.
# stdlib only. Reads a prefix of the file (default 32 MiB); tensor data is never read.
#
# verdicts
#   play          known type ids, sizes consistent with mainline layout
#   refuse        will not load, or will load and emit garbage, on the named engine
#   need_more     header larger than the prefix supplied; rerun with --prefix larger
#
# the three traps it exists for (PrismML docs, 2026-09):
#   1. PQ2_0 (142) / PTQ1_0 (143): fork-only ggml types. mainline refuses — safe failure.
#   2. legacy Q2_0: group-128 bytes stored under type 42, which mainline reads as group-64. size mismatch.
#   3. Bonsai 2 Q2_0: correct group-64 bytes, rotated basis. mainline loads it and outputs gibberish.
#      nothing in the tensor table reveals this; detected by name only. the dangerous one.
#
# usage: gguf_guard.py FILE [--engine mainline|prism] [--file-size N] [--json]

import json, struct, sys, re, argparse

GGUF_MAGIC = b"GGUF"
# value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
          F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}

# ggml types this guard needs to size. (block elements, block bytes)
# Q2_0 g64: 64 trits * 2 bit = 16 B + f16 scale = 18 B  -> 2.25 bpw (matches PrismML table)
# legacy g128: 128 * 2 bit = 32 B + f16 = 34 B            -> 2.125 bpw
BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 42: (64, 18)}
NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
         10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16",
         34: "TQ1_0", 35: "TQ2_0", 41: "Q1_0", 42: "Q2_0", 142: "PQ2_0", 143: "PTQ1_0"}
MAINLINE_COUNT = 43   # GGML_TYPE_COUNT in ggml-org/llama.cpp and llama.rn 0.13.0-rc.3 (read 2026-09-24)
FORK_ONLY = {142, 143}
BONSAI2 = re.compile(r"bonsai[\s_-]*2|prism-fork-required", re.I)


class NeedMore(Exception):
    pass


class R:
    def __init__(s, b):
        s.b, s.o = b, 0

    def take(s, n):
        if s.o + n > len(s.b):
            raise NeedMore(s.o + n)
        v = s.b[s.o:s.o + n]
        s.o += n
        return v

    def fmt(s, f):
        return struct.unpack(f, s.take(struct.calcsize(f)))[0]

    def str(s):
        return s.take(s.fmt("<Q")).decode("utf-8", "replace")

    def val(s, t):
        if t in SCALAR:
            return s.fmt(SCALAR[t])
        if t == STR:
            return s.str()
        if t == ARR:
            et, n = s.fmt("<I"), s.fmt("<Q")
            if et in SCALAR:                       # skip bulk scalar arrays without decoding
                s.take(n * struct.calcsize(SCALAR[et]))
                return f"<array {n}>"
            for _ in range(n):
                s.val(et)
            return f"<array {n}>"
        raise ValueError(f"unknown gguf value type {t}")


def parse(b):
    r = R(b)
    if r.take(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    ver, nt, nkv = r.fmt("<I"), r.fmt("<Q"), r.fmt("<Q")
    kv = {}
    for _ in range(nkv):
        k = r.str()
        kv[k] = r.val(r.fmt("<I"))
    tensors = []
    for _ in range(nt):
        name = r.str()
        dims = [r.fmt("<Q") for _ in range(r.fmt("<I"))]
        tensors.append({"name": name, "dims": dims, "type": r.fmt("<I"), "offset": r.fmt("<Q")})
    align = int(kv.get("general.alignment", 32))
    data_start = (r.o + align - 1) // align * align
    return ver, kv, tensors, align, data_start


def nelem(d):
    n = 1
    for x in d:
        n *= x
    return n


def judge(b, engine="mainline", file_size=None, filename=""):
    try:
        ver, kv, ts, align, data_start = parse(b)
    except NeedMore as e:
        return {"verdict": "need_more", "bytes_needed": e.args[0]}
    types = sorted({t["type"] for t in ts})
    out = {"gguf_version": ver, "arch": kv.get("general.architecture"), "name": kv.get("general.name"),
           "tensors": len(ts), "types": {NAMES.get(t, str(t)): sum(1 for x in ts if x["type"] == t) for t in types},
           "engine": engine, "reasons": []}
    why = out["reasons"]

    # KV bytes/token at f16, for the RAM plan (R2). None when the header uses per-layer arrays.
    a = kv.get("general.architecture")
    L, H = kv.get(f"{a}.block_count"), kv.get(f"{a}.attention.head_count_kv")
    K, V = kv.get(f"{a}.attention.key_length"), kv.get(f"{a}.attention.value_length")
    if K is None and isinstance(kv.get(f"{a}.embedding_length"), int) and isinstance(kv.get(f"{a}.attention.head_count"), int):
        K = V = kv[f"{a}.embedding_length"] // kv[f"{a}.attention.head_count"]
    if all(isinstance(x, int) for x in (L, H, K)):
        out["kv_f16_bytes_per_token"] = L * H * (K + (V if isinstance(V, int) else K)) * 2

    unknown = [t for t in types if t >= MAINLINE_COUNT and t not in FORK_ONLY]
    if unknown:
        why.append(f"type ids {unknown} unknown to every build this guard knows")
    if engine == "mainline" and set(types) & FORK_ONLY:
        why.append("PQ2_0/PTQ1_0 are PrismML-fork types; mainline and llama.rn refuse them")

    # legacy Q2_0: compare bytes between offsets with the group-64 expectation
    ordered = sorted(ts, key=lambda t: t["offset"])
    ends = [x["offset"] for x in ordered[1:]] + ([file_size - data_start] if file_size else [None])
    short = 0
    for t, end in zip(ordered, ends):
        if t["type"] != 42 or end is None:
            continue
        per, bb = BLOCK[42]
        want = nelem(t["dims"]) // per * bb
        if end - t["offset"] < want:
            short += 1
    if short:
        why.append(f"{short} Q2_0 tensors smaller than group-64 layout: legacy group-128 Prism file, "
                   f"use *-Q2_0_g64.gguf or *-PQ2_0.gguf")

    label = " ".join(str(x) for x in (kv.get("general.name"), kv.get("general.basename"), filename) if x)
    if BONSAI2.search(label) and engine == "mainline":
        why.append("Bonsai 2 weights are in a rotated basis and need the fork's Hadamard transform; "
                   "mainline loads the Q2_0 band silently and outputs gibberish")

    out["verdict"] = "refuse" if why else "play"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--engine", default="mainline", choices=["mainline", "prism"])
    ap.add_argument("--prefix", type=int, default=32 << 20)
    ap.add_argument("--file-size", type=int)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    import os
    with open(a.file, "rb") as f:
        b = f.read(a.prefix)
    size = a.file_size or os.path.getsize(a.file)
    if size < len(b):
        size = len(b)
    res = judge(b, a.engine, size, os.path.basename(a.file))
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(res["verdict"], json.dumps(res.get("types", {})))
        for w in res.get("reasons", []):
            print("  -", w)
    sys.exit({"play": 0, "refuse": 2, "need_more": 3}[res["verdict"]])


if __name__ == "__main__":
    main()
