#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""P3's whole-model oracle, from the SHIPPED ggml (llama.cpp b11192): every layer of the Qwen3 graph, output_norm and
the logits, built through ctypes with the ops llama.cpp's graph uses and computed by the release's own CPU backend
(libggml-cpu-haswell.so, 3 threads), for a chat prompt of fewer than 64 tokens (so attention takes ggml's reference
flash-attention path, as testing/forward_oracle.py's layer-0 checks do).

The graph is computed one layer at a time, each in its own context, carrying the residual stream between them as
f32 bytes, so memory stays near one layer's weights; the arithmetic is the same as one graph's.

Runs Q1_0 (1-bit) and Q2_0_g64 (ternary) models. Writes .models/oracle-forward/model-<gguf stem>.tsv: a header line (the tokens), then per token and layer the sha256 of l_out's
f32 row, per token the sha256 of result_norm and of the logits row, the argmax, and the top five ids with their
logits (for eyes).
usage: python3 testing/model_oracle.py GGUF LIBDIR [OUT]"""
import ctypes as C, hashlib, mmap, os, struct, sys
from pathlib import Path

gguf, libdir = sys.argv[1], sys.argv[2]
out = Path(sys.argv[3] if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / ".models" / "oracle-forward")
out.mkdir(parents=True, exist_ok=True)
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
for f, args in (("ggml_mul_mat", [P, P, P]), ("ggml_new_tensor_1d", [P, C.c_int, C.c_int64]),
                ("ggml_new_tensor_2d", [P, C.c_int, C.c_int64, C.c_int64]), ("ggml_new_tensor_3d", [P, C.c_int, C.c_int64, C.c_int64, C.c_int64]),
                ("ggml_get_rows", [P, P, P]), ("ggml_rms_norm", [P, P, C.c_float]), ("ggml_mul", [P, P, P]), ("ggml_add", [P, P, P]),
                ("ggml_new_graph", [P]), ("ggml_get_data", [P]), ("ggml_reshape_2d", [P, P, C.c_int64, C.c_int64]),
                ("ggml_reshape_3d", [P, P, C.c_int64, C.c_int64, C.c_int64]), ("ggml_permute", [P, P, C.c_int, C.c_int, C.c_int, C.c_int]),
                ("ggml_cpy", [P, P, P]), ("ggml_swiglu_split", [P, P, P]),
                ("ggml_rope_ext", [P, P, P, P, C.c_int, C.c_int, C.c_int, C.c_float, C.c_float, C.c_float, C.c_float, C.c_float, C.c_float]),
                ("ggml_flash_attn_ext", [P, P, P, P, P, C.c_float, C.c_float, C.c_float])):
    getattr(base, f).restype = P
    getattr(base, f).argtypes = args
base.ggml_build_forward_expand.argtypes = [P, P]
base.ggml_free.argtypes = [P]
base.ggml_prec_set_acc.argtypes = [P, C.c_int]
cpu.ggml_graph_compute_with_ctx.argtypes = [P, P, C.c_int]
F32, F16, I32, Q1_0, Q2_0 = 0, 1, 26, 41, 42
QK = {Q1_0: 128, Q2_0: 64}

fh = open(gguf, "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
ver, kv, tensors, align, data_start = parse(mm[:64 << 20])
T = {t["name"]: t for t in tensors}
arch = kv["general.architecture"]
assert arch == "qwen3", arch
eps = float(kv["qwen3.attention.layer_norm_rms_epsilon"])
n_layer = int(kv["qwen3.block_count"])
n_embd = int(kv["qwen3.embedding_length"])
n_ff = int(kv["qwen3.feed_forward_length"])
n_head, n_head_kv, head = int(kv["qwen3.attention.head_count"]), int(kv["qwen3.attention.head_count_kv"]), int(kv["qwen3.attention.key_length"])
n_vocab = T["token_embd.weight"]["dims"][1]

libm = C.CDLL("libm.so.6")
libm.logf.restype = C.c_float
libm.logf.argtypes = [C.c_float]
f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]  # noqa: E731
factor = f32(float(kv.get("qwen3.rope.scaling.factor", 1.0)))
freq_scale = f32(1.0 / factor)
logfac = libm.logf(f32(1.0 / freq_scale))
mscale = f32(f32(f32(0.1) * logfac) + 1.0) if factor > 1.0 else 1.0
attn_factor = f32(mscale * f32(1.0 / f32(1.0 + f32(0.1 * logfac))))
n_ctx_orig = int(kv.get("qwen3.rope.scaling.original_context_length", kv["qwen3.context_length"]))
freq_base = float(kv["qwen3.rope.freq_base"])
kq_scale = f32(1.0 / f32(head ** 0.5))

prompt = [151644, 8948, 198, 2610, 525, 20739, 5409, 13, 151645, 198, 151644, 872, 198, 3838, 1558, 264, 22725, 12118, 30, 151645, 198,
          151644, 77091, 198, 151667, 271, 151668, 271]
n = len(prompt)
assert n < 64, "the reference attention path needs fewer than 64 query rows"


def q1_bytes(t):  # a quantized matrix's bytes: 18-byte blocks of 128 (Q1_0) or 64 (Q2_0_g64) weights
    return t["dims"][1] * (t["dims"][0] // QK[t["type"]]) * 18


def load(ctx, name):
    t = T[name]
    if t["type"] == F32:
        x = base.ggml_new_tensor_1d(ctx, F32, t["dims"][0])
        nb = 4 * t["dims"][0]
    else:
        assert t["type"] in QK, (name, t["type"])
        x = base.ggml_new_tensor_2d(ctx, t["type"], t["dims"][0], t["dims"][1])
        nb = q1_bytes(t)
    C.memmove(base.ggml_get_data(x), C.c_char_p(mm[data_start + t["offset"]: data_start + t["offset"] + nb]), nb)
    return x


def rows(t, width):
    raw = C.string_at(base.ggml_get_data(t), width * 4 * n)
    return [raw[p * width * 4:(p + 1) * width * 4] for p in range(n)]


def new_ctx(weight_bytes):
    return base.ggml_init(InitParams(weight_bytes + (192 << 20), None, False))


def compute(ctx, t):
    gf = base.ggml_new_graph(ctx)
    base.ggml_build_forward_expand(gf, t)
    cpu.ggml_graph_compute_with_ctx(ctx, gf, 3)


# the embedding
ctx = new_ctx(q1_bytes(T["token_embd.weight"]))
tok = base.ggml_new_tensor_1d(ctx, I32, n)
C.memmove(base.ggml_get_data(tok), (C.c_int32 * n)(*prompt), 4 * n)
x = base.ggml_get_rows(ctx, load(ctx, "token_embd.weight"), tok)
compute(ctx, x)
stream = C.string_at(base.ggml_get_data(x), n_embd * 4 * n)
base.ggml_free(ctx)

lines = []
mask_vals = (C.c_uint16 * (n * n))(*[0 if j <= i else 0xFC00 for i in range(n) for j in range(n)])
for il in range(n_layer):
    names = [f"blk.{il}.{w}.weight" for w in ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "attn_output",
                                              "ffn_norm", "ffn_gate", "ffn_up", "ffn_down")]
    ctx = new_ctx(sum(q1_bytes(T[m]) if T[m]["type"] in QK else 4 * T[m]["dims"][0] for m in names))
    W = {m.split(".")[2]: load(ctx, m) for m in names}
    inp = base.ggml_new_tensor_2d(ctx, F32, n_embd, n)
    C.memmove(base.ggml_get_data(inp), C.c_char_p(stream), len(stream))
    pos = base.ggml_new_tensor_1d(ctx, I32, n)
    C.memmove(base.ggml_get_data(pos), (C.c_int32 * n)(*range(n)), 4 * n)
    mask = base.ggml_new_tensor_2d(ctx, F16, n, n)
    C.memmove(base.ggml_get_data(mask), mask_vals, 2 * n * n)
    xn = base.ggml_mul(ctx, base.ggml_rms_norm(ctx, inp, C.c_float(eps)), W["attn_norm"])

    def qk(wname, nname, nh):
        r = base.ggml_reshape_3d(ctx, base.ggml_mul_mat(ctx, W[wname], xn), head, nh, n)
        r = base.ggml_mul(ctx, base.ggml_rms_norm(ctx, r, C.c_float(eps)), W[nname])
        return base.ggml_rope_ext(ctx, r, pos, None, head, 2, n_ctx_orig, C.c_float(freq_base), C.c_float(freq_scale),
                                  C.c_float(1.0), C.c_float(attn_factor), C.c_float(32.0), C.c_float(1.0))
    q = qk("attn_q", "attn_q_norm", n_head)
    k = qk("attn_k", "attn_k_norm", n_head_kv)
    v = base.ggml_reshape_3d(ctx, base.ggml_mul_mat(ctx, W["attn_v"], xn), head, n_head_kv, n)
    k16 = base.ggml_cpy(ctx, k, base.ggml_new_tensor_3d(ctx, F16, head, n_head_kv, n))
    v16 = base.ggml_cpy(ctx, v, base.ggml_new_tensor_3d(ctx, F16, head, n_head_kv, n))
    fa = base.ggml_flash_attn_ext(ctx, base.ggml_permute(ctx, q, 0, 2, 1, 3), base.ggml_permute(ctx, k16, 0, 2, 1, 3),
                                  base.ggml_permute(ctx, v16, 0, 2, 1, 3), mask, C.c_float(kq_scale), C.c_float(0.0), C.c_float(0.0))
    base.ggml_prec_set_acc(fa, 10)
    ffn_inp = base.ggml_add(ctx, base.ggml_mul_mat(ctx, W["attn_output"], base.ggml_reshape_2d(ctx, fa, n_head * head, n)), inp)
    fn = base.ggml_mul(ctx, base.ggml_rms_norm(ctx, ffn_inp, C.c_float(eps)), W["ffn_norm"])
    sw = base.ggml_swiglu_split(ctx, base.ggml_mul_mat(ctx, W["ffn_gate"], fn), base.ggml_mul_mat(ctx, W["ffn_up"], fn))
    l_out = base.ggml_add(ctx, base.ggml_mul_mat(ctx, W["ffn_down"], sw), ffn_inp)
    compute(ctx, l_out)
    for p, r in enumerate(rows(l_out, n_embd)):
        lines.append(f"l_out\t{il}\t{p}\t{hashlib.sha256(r).hexdigest()}\t{' '.join(f'{v:.6g}' for v in struct.unpack('<3f', r[:12]))}")
    stream = C.string_at(base.ggml_get_data(l_out), n_embd * 4 * n)
    base.ggml_free(ctx)
    print(f"layer {il + 1}/{n_layer}", end="\r", flush=True)

# output_norm and the logits
ctx = new_ctx(q1_bytes(T["output.weight"]) + 4 * n_embd + 4 * n_vocab * n)
inp = base.ggml_new_tensor_2d(ctx, F32, n_embd, n)
C.memmove(base.ggml_get_data(inp), C.c_char_p(stream), len(stream))
rn = base.ggml_mul(ctx, base.ggml_rms_norm(ctx, inp, C.c_float(eps)), load(ctx, "output_norm.weight"))
logits = base.ggml_mul_mat(ctx, load(ctx, "output.weight"), rn)
compute(ctx, logits)
for p, r in enumerate(rows(rn, n_embd)):
    lines.append(f"result_norm\t-\t{p}\t{hashlib.sha256(r).hexdigest()}\t{' '.join(f'{v:.6g}' for v in struct.unpack('<3f', r[:12]))}")
for p, r in enumerate(rows(logits, n_vocab)):
    vals = struct.unpack(f"<{n_vocab}f", r)
    top = sorted(range(n_vocab), key=lambda i: -vals[i])[:5]
    lines.append(f"logits\t{top[0]}\t{p}\t{hashlib.sha256(r).hexdigest()}\t{' '.join(f'{i}:{vals[i]:.6g}' for i in top)}")
base.ggml_free(ctx)
dest = out / f"model-{Path(gguf).stem}.tsv"
with open(dest, "w") as f:
    f.write(f"# {n_layer} layers · tokens {' '.join(map(str, prompt))}\n")
    f.write("\n".join(lines) + "\n")
print(f"\n{n_layer} layers, result_norm and logits for {n} tokens → {dest}")
