#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""P3's oracle for the first operations of the forward pass, from the SHIPPED ggml (llama.cpp b11192): the same ops
llama.cpp's Qwen3 graph uses — get_rows on the Q1_0 token embedding (inp_embd), then rms_norm(eps) and mul by
blk.0.attn_norm.weight (attn_norm-0) — built as a ggml graph through ctypes and computed by the release's own CPU
backend (libggml-cpu-haswell.so), on real token ids. Writes one line per token: id, sha256 of the embedding row's f32
bytes, sha256 of the normed row's f32 bytes, and the first four normed values (for eyes).
usage: python3 testing/forward_oracle.py GGUF LIBDIR [OUT]   (OUT default .models/oracle-forward)"""
import ctypes as C, hashlib, mmap, os, struct, sys
from pathlib import Path

gguf, libdir = sys.argv[1], sys.argv[2]
out = Path(sys.argv[3] if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / ".models" / "oracle-forward")
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
for f, args in (("ggml_mul_mat", [P, P, P]), ("ggml_new_tensor_1d", [P, C.c_int, C.c_int64]), ("ggml_new_tensor_2d", [P, C.c_int, C.c_int64, C.c_int64]),
                ("ggml_get_rows", [P, P, P]), ("ggml_rms_norm", [P, P, C.c_float]), ("ggml_mul", [P, P, P]), ("ggml_new_graph", [P]),
                ("ggml_get_data", [P])):
    getattr(base, f).restype = P
    getattr(base, f).argtypes = args
base.ggml_build_forward_expand.argtypes = [P, P]
base.ggml_nbytes.restype = C.c_size_t
base.ggml_nbytes.argtypes = [P]
base.ggml_free.argtypes = [P]
cpu.ggml_graph_compute_with_ctx.argtypes = [P, P, C.c_int]
F32, I32, Q1_0 = 0, 26, 41

fh = open(gguf, "rb")
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
ver, kv, tensors, align, data_start = parse(mm[:64 << 20])
T = {t["name"]: t for t in tensors}
eps = float(kv["qwen3.attention.layer_norm_rms_epsilon"])
emb, norm = T["token_embd.weight"], T["blk.0.attn_norm.weight"]
n_embd, n_vocab = emb["dims"] if "dims" in emb else emb["ne"]
assert emb["type"] == Q1_0 and norm["type"] == F32
emb_bytes = n_vocab * (n_embd // 128) * 18

# the ids: a real prompt through the tokenizer's oracle cases, the chat markers, and the table's edges
ids = [0, 1, 2, 13, 198, 220, 271, 872, 13048, 151643, 151644, 151645, 151667, 151668, n_vocab - 1]
cases = Path(__file__).resolve().parents[1] / ".models" / "oracle-tokenizer" / "cases.jsonl"
if cases.exists():
    import json
    for line in cases.read_text().splitlines()[:40]:
        ids += json.loads(line)["ids"][:24]
ids = list(dict.fromkeys(ids))  # unique, in order

ctx = base.ggml_init(InitParams(emb_bytes + n_embd * 4 * (3 * len(ids) + 8) + (64 << 20), None, False))
w = base.ggml_new_tensor_2d(ctx, Q1_0, n_embd, n_vocab)
C.memmove(base.ggml_get_data(w), C.c_char_p(mm[data_start + emb["offset"]: data_start + emb["offset"] + emb_bytes]), emb_bytes)
nw = base.ggml_new_tensor_1d(ctx, F32, n_embd)
C.memmove(base.ggml_get_data(nw), C.c_char_p(mm[data_start + norm["offset"]: data_start + norm["offset"] + n_embd * 4]), n_embd * 4)
tok = base.ggml_new_tensor_1d(ctx, I32, len(ids))
C.memmove(base.ggml_get_data(tok), (C.c_int32 * len(ids))(*ids), 4 * len(ids))
inp = base.ggml_get_rows(ctx, w, tok)                 # inp_embd
cur = base.ggml_mul(ctx, base.ggml_rms_norm(ctx, inp, C.c_float(eps)), nw)   # attn_norm-0
gf = base.ggml_new_graph(ctx)
base.ggml_build_forward_expand(gf, cur)
cpu.ggml_graph_compute_with_ctx(ctx, gf, 3)
row = n_embd * 4
a = C.string_at(base.ggml_get_data(inp), row * len(ids))
b = C.string_at(base.ggml_get_data(cur), row * len(ids))
out.mkdir(parents=True, exist_ok=True)
with open(out / "embed_norm.tsv", "w") as f:
    for k, i in enumerate(ids):
        ra, rb = a[k * row:(k + 1) * row], b[k * row:(k + 1) * row]
        first = " ".join(f"{x:.6g}" for x in struct.unpack("<4f", rb[:16]))
        f.write(f"{i}\t{hashlib.sha256(ra).hexdigest()}\t{hashlib.sha256(rb).hexdigest()}\t{first}\n")
base.ggml_free(ctx)
print(f"{len(ids)} tokens: inp_embd and attn_norm-0 rows from the shipped ggml (eps {eps:g}, 3 threads) → {out / 'embed_norm.tsv'}")

# ---- step four: layer 0's Q, K, V — projections, per-head RMS norms, YaRN RoPE (NEOX), as llama.cpp's qwen3 graph ----
# llama.cpp's own rope parameters for this model (llama-context.cpp, b11192), computed in float as it computes them
libm = C.CDLL("libm.so.6")
libm.logf.restype = C.c_float
libm.logf.argtypes = [C.c_float]
f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]  # noqa: E731
factor = f32(float(kv.get("qwen3.rope.scaling.factor", 1.0)))
freq_scale = f32(1.0 / factor)
logfac = libm.logf(f32(1.0 / freq_scale))
mscale = f32(f32(f32(0.1 * 1.0) * logfac) + 1.0) if factor > 1.0 else 1.0          # get_mscale(factor, 1)
attn_factor = f32(mscale * f32(1.0 / f32(1.0 + f32(0.1 * logfac))))                  # *= 1/(1 + 0.1·logf(factor))
n_ctx_orig = int(kv.get("qwen3.rope.scaling.original_context_length", kv["qwen3.context_length"]))
freq_base = float(kv["qwen3.rope.freq_base"])
n_head, n_head_kv, head = int(kv["qwen3.attention.head_count"]), int(kv["qwen3.attention.head_count_kv"]), int(kv["qwen3.attention.key_length"])

prompt = [151644, 8948, 198, 2610, 525, 20739, 5409, 13, 151645, 198, 151644, 872, 198, 3838, 1558, 264, 22725, 12118, 30, 151645, 198,
          151644, 77091, 198, 151667, 271, 151668, 271]   # "<|im_start|>system\nYou are Savante.<|im_end|>\n<|im_start|>user\n…"
n = len(prompt)


def tensor_2d(name, typ, ne0, ne1, bpr):
    t = T[name]
    b = mm[data_start + t["offset"]: data_start + t["offset"] + bpr * ne1]
    x = base.ggml_new_tensor_2d(ctx2, typ, ne0, ne1)
    C.memmove(base.ggml_get_data(x), C.c_char_p(b), len(b))
    return x


def tensor_1d(name, ne0):
    t = T[name]
    x = base.ggml_new_tensor_1d(ctx2, F32, ne0)
    C.memmove(base.ggml_get_data(x), C.c_char_p(mm[data_start + t["offset"]: data_start + t["offset"] + 4 * ne0]), 4 * ne0)
    return x


for f, args in (("ggml_reshape_3d", [P, P, C.c_int64, C.c_int64, C.c_int64]),
                ("ggml_rope_ext", [P, P, P, P, C.c_int, C.c_int, C.c_int, C.c_float, C.c_float, C.c_float, C.c_float, C.c_float, C.c_float])):
    getattr(base, f).restype = P
    getattr(base, f).argtypes = args
bpr = n_embd // 128 * 18
ctx2 = base.ggml_init(InitParams(emb_bytes + 2 * (n_embd * bpr) + 2 * (n_head_kv * head * bpr) + (256 << 20), None, False))
w2 = base.ggml_new_tensor_2d(ctx2, Q1_0, n_embd, n_vocab)
C.memmove(base.ggml_get_data(w2), C.c_char_p(mm[data_start + emb["offset"]: data_start + emb["offset"] + emb_bytes]), emb_bytes)
tok2 = base.ggml_new_tensor_1d(ctx2, I32, n)
C.memmove(base.ggml_get_data(tok2), (C.c_int32 * n)(*prompt), 4 * n)
pos = base.ggml_new_tensor_1d(ctx2, I32, n)
C.memmove(base.ggml_get_data(pos), (C.c_int32 * n)(*range(n)), 4 * n)
far = [7 + 2341 * i for i in range(n)]   # out to 63 214: long-context positions, where theta is a long product
pos_far = base.ggml_new_tensor_1d(ctx2, I32, n)
C.memmove(base.ggml_get_data(pos_far), (C.c_int32 * n)(*far), 4 * n)
xn = base.ggml_mul(ctx2, base.ggml_rms_norm(ctx2, base.ggml_get_rows(ctx2, w2, tok2), C.c_float(eps)), tensor_1d("blk.0.attn_norm.weight", n_embd))
outs = {}
for nm, wname, nh, qn in (("Q", "blk.0.attn_q.weight", n_head, "blk.0.attn_q_norm.weight"), ("K", "blk.0.attn_k.weight", n_head_kv, "blk.0.attn_k_norm.weight"),
                          ("V", "blk.0.attn_v.weight", n_head_kv, None)):
    proj = base.ggml_mul_mat(ctx2, tensor_2d(wname, Q1_0, n_embd, nh * head, bpr), xn) if hasattr(base, "ggml_mul_mat") else None
    outs[nm] = proj
    if qn:
        r = base.ggml_reshape_3d(ctx2, proj, head, nh, n)
        normed = base.ggml_mul(ctx2, base.ggml_rms_norm(ctx2, r, C.c_float(eps)), tensor_1d(qn, head))
        outs[nm + "_normed"] = normed
        outs[nm + "_rope"] = base.ggml_rope_ext(ctx2, normed, pos, None, head, 2, n_ctx_orig, C.c_float(freq_base), C.c_float(freq_scale),
                                                C.c_float(1.0), C.c_float(attn_factor), C.c_float(32.0), C.c_float(1.0))
        outs[nm + "_rope_far"] = base.ggml_rope_ext(ctx2, normed, pos_far, None, head, 2, n_ctx_orig, C.c_float(freq_base), C.c_float(freq_scale),
                                                    C.c_float(1.0), C.c_float(attn_factor), C.c_float(32.0), C.c_float(1.0))
# ---- step five: layer 0's attention — the K/V cache in f16, causal mask, flash attention (the CPU reference path:
# fewer than 64 query rows and fewer than 512 KV cells), then wo and the residual (llama.cpp's kqv_out, attn_out, ffn_inp)
F16 = 1
for f, args in (("ggml_permute", [P, P, C.c_int, C.c_int, C.c_int, C.c_int]), ("ggml_cpy", [P, P, P]),
                ("ggml_new_tensor_3d", [P, C.c_int, C.c_int64, C.c_int64, C.c_int64]), ("ggml_reshape_2d", [P, P, C.c_int64, C.c_int64]),
                ("ggml_add", [P, P, P]),
                ("ggml_flash_attn_ext", [P, P, P, P, P, C.c_float, C.c_float, C.c_float])):
    getattr(base, f).restype = P
    getattr(base, f).argtypes = args
base.ggml_prec_set_acc.argtypes = [P, C.c_int]
k16 = base.ggml_cpy(ctx2, outs["K_rope"], base.ggml_new_tensor_3d(ctx2, F16, head, n_head_kv, n))
v16 = base.ggml_cpy(ctx2, base.ggml_reshape_3d(ctx2, outs["V"], head, n_head_kv, n), base.ggml_new_tensor_3d(ctx2, F16, head, n_head_kv, n))
mask = base.ggml_new_tensor_2d(ctx2, F16, n, n)
C.memmove(base.ggml_get_data(mask), (C.c_uint16 * (n * n))(*[0 if j <= i else 0xFC00 for i in range(n) for j in range(n)]), 2 * n * n)
kq_scale = f32(1.0 / f32(head ** 0.5))
fa = base.ggml_flash_attn_ext(ctx2, base.ggml_permute(ctx2, outs["Q_rope"], 0, 2, 1, 3), base.ggml_permute(ctx2, k16, 0, 2, 1, 3),
                              base.ggml_permute(ctx2, v16, 0, 2, 1, 3), mask, C.c_float(kq_scale), C.c_float(0.0), C.c_float(0.0))
base.ggml_prec_set_acc(fa, 10)  # GGML_PREC_F32, as llama-graph sets it
kqv = base.ggml_reshape_2d(ctx2, fa, n_head * head, n)
attn_out = base.ggml_mul_mat(ctx2, tensor_2d("blk.0.attn_output.weight", Q1_0, n_head * head, n_embd, (n_head * head) // 128 * 18), kqv)
emb0 = base.ggml_get_rows(ctx2, w2, tok2)
outs["kqv_out"] = kqv
outs["attn_out"] = attn_out
outs["ffn_inp"] = base.ggml_add(ctx2, attn_out, emb0)
gf2 = base.ggml_new_graph(ctx2)
for k in outs:
    base.ggml_build_forward_expand(gf2, outs[k])
cpu.ggml_graph_compute_with_ctx(ctx2, gf2, 3)
with open(out / "qkv.tsv", "w") as f:
    f.write(f"# attn_factor {attn_factor!r} bits {struct.unpack('<I', struct.pack('<f', attn_factor))[0]:08x} · freq_scale {freq_scale} · n_ctx_orig {n_ctx_orig} · freq_base {freq_base} · kq_scale bits {struct.unpack('<I', struct.pack('<f', kq_scale))[0]:08x} · far {' '.join(map(str, far))} · tokens {' '.join(map(str, prompt))}\n")
    for k in outs:
        width = n_embd if k in ("kqv_out", "attn_out", "ffn_inp") else (n_head if k.startswith("Q") else n_head_kv) * head
        raw = C.string_at(base.ggml_get_data(outs[k]), width * 4 * n)
        for p_ in range(n):
            row = raw[p_ * width * 4:(p_ + 1) * width * 4]
            f.write(f"{k}\t{p_}\t{hashlib.sha256(row).hexdigest()}\t{' '.join(f'{x:.6g}' for x in struct.unpack('<3f', row[:12]))}\n")
print(f"layer 0 Q/K/V for {n} tokens (projections, normed, roped; attn_factor {attn_factor!r}) → {out / 'qkv.tsv'}")
