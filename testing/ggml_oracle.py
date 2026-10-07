#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""ggml's own answers on a real Q1_0 or Q2_0 (ternary, group 64) GGUF, for bankml's bit-exactness
tests.  stdlib + numpy.  The quant type is the one the file carries (Q1_0 = 41 wins if both).

Calls the *exported* symbols of the prebuilt llama.cpp b11192 ubuntu-x64 release
(llama-b11192-bin-ubuntu-x64.tar.gz, sha256 34cf6fa5…81ec7 — check it before use):
  libggml-base.so        dequantize_row_q1_0 / dequantize_row_q2_0
  libggml-cpu-haswell.so quantize_row_q8_0, ggml_vec_dot_q1_0_q8_0 (AVX2 path),
                         ggml_vec_dot_q1_0_q8_0_generic
                         ggml_vec_dot_q2_0_q8_0 — on x86 this IS the generic C (arch-fallback.h renames
                         it); there is no second symbol, so for Q2_0 the "generic" column is the same
                         C built by the baseline libggml-cpu-x64.so variant (SSE only, no FMA).
haswell is the variant ggml's backend scorer loads on an AVX2 CPU without AVX-512 (Zen 2/3 EPYC,
this dev box). Writes to OUT:
  dequant.tsv  name · n_elements · sha256 of ggml's f32 output (little-endian), every tensor of the type
  vecdot.bin   cases: u32 name_len · name · u32 row · u32 n · f32[n] x · u8[n/32*34] q8 (ggml's) ·
               f32 dot_avx2 · f32 dot_generic
Tensors are read through a memmap, a chunk of rows at a time (a 2.3 GB file never sits in RAM).
usage: ggml_oracle.py GGUF LIBDIR OUT [rows_per_tensor]"""
import ctypes, hashlib, os, struct, sys
import numpy as np

gguf_path, libdir, out = sys.argv[1:4]
rows_per = int(sys.argv[4]) if len(sys.argv) > 4 else 3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # testing/gguf_guard.py (vendored from minaiml)
from gguf_guard import parse  # the same header parse the guard uses

base = ctypes.CDLL(os.path.join(libdir, "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
cpu = ctypes.CDLL(os.path.join(libdir, "libggml-cpu-haswell.so"), mode=ctypes.RTLD_GLOBAL)
cpu.ggml_cpu_init()
P, I64, SZ = ctypes.c_void_p, ctypes.c_int64, ctypes.c_size_t
cpu.quantize_row_q8_0.argtypes = [P, P, I64]

with open(gguf_path, "rb") as fh:
    ver, kv, tensors, align, data_start = parse(fh.read(32 << 20))
TY = 41 if any(t["type"] == 41 for t in tensors) else 42
QK = {41: 128, 42: 64}[TY]
dequant = getattr(base, {41: "dequantize_row_q1_0", 42: "dequantize_row_q2_0"}[TY])
dequant.argtypes = [P, P, I64]
if TY == 41:
    dot_fast, dot_gen = cpu.ggml_vec_dot_q1_0_q8_0, cpu.ggml_vec_dot_q1_0_q8_0_generic
else:
    x64 = ctypes.CDLL(os.path.join(libdir, "libggml-cpu-x64.so"))  # RTLD_LOCAL: its own copy of the symbol
    x64.ggml_cpu_init()
    dot_fast, dot_gen = cpu.ggml_vec_dot_q2_0_q8_0, x64.ggml_vec_dot_q2_0_q8_0
for f in (dot_fast, dot_gen):
    f.argtypes = [ctypes.c_int, P, SZ, P, SZ, P, SZ, ctypes.c_int]
    f.restype = None
mm = np.memmap(gguf_path, dtype=np.uint8, mode="r")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20260925)


def act(n):
    """activation-like: heavy-tailed, a few outliers, plus exact half-way values that separate
    round-half-even (AVX2 quantizer) from round-half-away (_ref)."""
    x = (rng.standard_normal(n) * rng.standard_normal(n)).astype(np.float32)
    x[rng.integers(0, n, max(1, n // 97))] *= 60
    blk = x.reshape(-1, 32)
    blk[:, 0] = np.float32(127.0)                   # amax 127 → id exactly 1.0 …
    blk[:, 1] = np.float32(2.5)                     # … so these sit on .5 boundaries
    blk[:, 2] = np.float32(-3.5)
    blk[::7] = 0                                    # all-zero blocks (d = 0 path)
    return x


q1 = [t for t in tensors if t["type"] == TY]
cases = 0
with open(os.path.join(out, "dequant.tsv"), "w") as tsv, open(os.path.join(out, "vecdot.bin"), "wb") as vb:
    for t in q1:
        ne0, n = t["dims"][0], int(np.prod(t["dims"]))
        nrows, rb = n // ne0, ne0 // QK * 18
        start = data_start + t["offset"]
        h = hashlib.sha256()
        chunk = max(1, (1 << 24) // ne0)
        for r0 in range(0, nrows, chunk):
            r1 = min(nrows, r0 + chunk)
            src = np.ascontiguousarray(mm[start + r0 * rb: start + r1 * rb])
            dst = np.empty((r1 - r0) * ne0, dtype=np.float32)
            dequant(src.ctypes.data, dst.ctypes.data, dst.size)
            h.update(dst.astype("<f4").tobytes())
        tsv.write(f"{t['name']}\t{n}\t{h.hexdigest()}\n")

        rows = sorted({0, nrows - 1, *rng.integers(0, nrows, max(0, rows_per - 2)).tolist()})
        for r in rows:
            x = act(ne0)
            q8 = np.zeros(ne0 // 32 * 34, dtype=np.uint8)
            cpu.quantize_row_q8_0(x.ctypes.data, q8.ctypes.data, ne0)
            w = np.ascontiguousarray(mm[start + r * rb: start + (r + 1) * rb])
            s_avx, s_gen = ctypes.c_float(), ctypes.c_float()
            dot_fast(ne0, ctypes.byref(s_avx), 0, w.ctypes.data, 0, q8.ctypes.data, 0, 1)
            dot_gen(ne0, ctypes.byref(s_gen), 0, w.ctypes.data, 0, q8.ctypes.data, 0, 1)
            nm = t["name"].encode()
            vb.write(struct.pack("<I", len(nm)) + nm + struct.pack("<II", r, ne0) + x.astype("<f4").tobytes()
                     + q8.tobytes() + struct.pack("<ff", s_avx.value, s_gen.value))
            cases += 1
print(f"{len(q1)} {'Q1_0' if TY == 41 else 'Q2_0'} tensors dequantized+hashed, {cases} vec_dot cases -> {out}")
