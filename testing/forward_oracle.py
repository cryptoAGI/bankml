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
for f, args in (("ggml_new_tensor_1d", [P, C.c_int, C.c_int64]), ("ggml_new_tensor_2d", [P, C.c_int, C.c_int64, C.c_int64]),
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
