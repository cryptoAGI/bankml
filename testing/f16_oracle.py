#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The F16 weight oracle (0.3.4): the SHIPPED ggml (llama.cpp b11192, libggml-cpu-haswell.so) multiplying F16
weights, for bankML's f16.rs to equal bit for bit. stdlib + numpy + ctypes.

  dequant.tsv  every F16 tensor of the GGUF widened to f32 by ggml's own `ggml_fp16_to_fp32_row`: name · sha256
  matmul.bin   `ggml_mul_mat` graphs computed by the CPU backend (3 threads), the path llama.cpp takes for each shape:
               one column → ggml_vec_dot_f16; two or more → llamafile_sgemm's tinyBLAS (when rows % 4 == 0 and
               k % 8 == 0; otherwise ggml's vec_dot again). Cases: real matrices (layers 0 and 29, the token table)
               by 1, 2, 3, 5 and 28 columns, and synthetic shapes that reach the tails (k % 32 != 0) and the
               fallbacks (rows % 4 != 0, k % 8 != 0). Record: u32 name_len · name · u32 m · u32 k · u32 n ·
               [f16 weights m·k, only for "synthetic"] · f32 activations n·k · f32 ggml output n·m.
usage: f16_oracle.py GGUF LIBDIR [OUT]   (OUT defaults to .models/oracle-f16)"""
import ctypes as C, hashlib, mmap, os, struct, sys
from pathlib import Path
import numpy as np

gguf, libdir = sys.argv[1], sys.argv[2]
out = Path(sys.argv[3] if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / ".models" / "oracle-f16")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_guard import parse  # noqa: E402

base = C.CDLL(os.path.join(libdir, "libggml-base.so"), mode=C.RTLD_GLOBAL)
cpu = C.CDLL(os.path.join(libdir, "libggml-cpu-haswell.so"), mode=C.RTLD_GLOBAL)
cpu.ggml_cpu_init()


class InitParams(C.Structure):
    _fields_ = [("mem_size", C.c_size_t), ("mem_buffer", C.c_void_p), ("no_alloc", C.c_bool)]


P = C.c_void_p
base.ggml_init.restype = P
base.ggml_init.argtypes = [InitParams]
for f, args in (("ggml_mul_mat", [P, P, P]), ("ggml_new_tensor_2d", [P, C.c_int, C.c_int64, C.c_int64]), ("ggml_new_graph", [P]),
                ("ggml_get_data", [P])):
    getattr(base, f).restype = P
    getattr(base, f).argtypes = args
base.ggml_build_forward_expand.argtypes = [P, P]
base.ggml_free.argtypes = [P]
base.ggml_fp16_to_fp32_row.argtypes = [P, P, C.c_int64]
cpu.ggml_graph_compute_with_ctx.argtypes = [P, P, C.c_int]
F32, F16 = 0, 1

fh = open(gguf, "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
ver, kv, tensors, align, data_start = parse(mm[:64 << 20])
T = {t["name"]: t for t in tensors}
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261002)

n16 = 0
with open(out / "dequant.tsv", "w") as tsv:
    for t in tensors:
        if t["type"] != F16:
            continue
        n = int(np.prod(t["dims"]))
        src = np.frombuffer(mm, dtype=np.uint8, count=2 * n, offset=data_start + t["offset"])
        dst = np.empty(n, dtype=np.float32)
        base.ggml_fp16_to_fp32_row(src.ctypes.data, dst.ctypes.data, n)
        tsv.write(f"{t['name']}\t{hashlib.sha256(dst.astype('<f4').tobytes()).hexdigest()}\n")
        n16 += 1


def act(n, k):
    """activation-like rows: heavy-tailed, a few outliers, some values that round to f16 halfway cases"""
    x = (rng.standard_normal((n, k)) * rng.standard_normal((n, k))).astype(np.float32)
    x[:, rng.integers(0, k, max(1, k // 97))] *= 30
    x[:, 0] = np.float32(1.0 + 2.0 ** -11)  # exactly halfway between two f16 values: ties to even
    return x


def mul_mat(w16, m, k, x):
    """ggml_mul_mat(W [k × m, F16], X [k × n, F32]) by the shipped CPU backend; returns the n × m output."""
    n = x.shape[0]
    ctx = base.ggml_init(InitParams(m * k * 2 + n * k * 4 + n * m * 4 + (64 << 20), None, False))
    W = base.ggml_new_tensor_2d(ctx, F16, k, m)
    C.memmove(base.ggml_get_data(W), w16.ctypes.data, m * k * 2)
    X = base.ggml_new_tensor_2d(ctx, F32, k, n)
    C.memmove(base.ggml_get_data(X), np.ascontiguousarray(x).ctypes.data, n * k * 4)
    Y = base.ggml_mul_mat(ctx, W, X)
    gf = base.ggml_new_graph(ctx)
    base.ggml_build_forward_expand(gf, Y)
    cpu.ggml_graph_compute_with_ctx(ctx, gf, 3)
    y = np.frombuffer(C.string_at(base.ggml_get_data(Y), n * m * 4), dtype="<f4").copy()
    base.ggml_free(ctx)
    return y


cases = 0
with open(out / "matmul.bin", "wb") as f:
    def write(name, m, k, x, y, w16=None):
        global cases
        nm = name.encode()
        f.write(struct.pack("<I", len(nm)) + nm + struct.pack("<III", m, k, x.shape[0]))
        if w16 is not None:
            f.write(w16.astype("<u2").tobytes())
        f.write(x.astype("<f4").tobytes() + y.astype("<f4").tobytes())
        cases += 1

    real = [f"blk.{il}.{w}.weight" for il in (0, 29) for w in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")]
    for name in real + ["token_embd.weight"]:
        t = T[name]
        k, m = t["dims"][0], t["dims"][1]
        w16 = np.frombuffer(mm, dtype=np.uint16, count=m * k, offset=data_start + t["offset"]).copy()
        for n in ((1, 2) if name == "token_embd.weight" else (1, 2, 3, 5, 28)):
            x = act(n, k)
            write(name, m, k, x, mul_mat(w16, m, k, x))
    for m, k in ((6, 72), (8, 36), (12, 100), (16, 40), (4, 8)):
        w16 = (rng.standard_normal(m * k) * 0.1).astype(np.float16).view(np.uint16)
        for n in (1, 2, 7):
            x = act(n, k)
            write("synthetic", m, k, x, mul_mat(w16, m, k, x), w16)
print(f"{n16} F16 tensors widened by ggml_fp16_to_fp32_row, {cases} mul_mat cases from the shipped ggml b11192 -> {out}")
