#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The JSON-schema grammar oracle (O6b): llama.cpp b11192's own `json_schema_to_grammar` and the grammar llama-server
builds for a response-format schema on the jinja chat path, both called inside the b11192 release's
`libllama-common.so` (`testing/schema_oracle.cpp`; no model is loaded, no server runs). The template is read from the
pinned model's GGUF metadata (Bonsai / Qwen3, sha256 30a75d10e60b57e2…); the request is fed as
`oaicompat_chat_params_parse` feeds it with Savante's flags (`--jinja --reasoning off`: thinking off, reasoning format
deepseek, an empty schema made `{"type": "object"}`).

The corpus: llama.cpp's own `tests/test-json-schema-to-grammar.cpp` cases (`testing/json_schema_cases.json`, 81) and
`testing/json_schema_corpus.json` (Pydantic v2 schemas shaped like mindX's models — nested `$defs`/`$ref`, Optional,
Literal, enums, constrained ints and strings, formats, tuples, recursion, `extra="forbid"`/`"allow"` — and edge
cases: odd property names, duplicate keys, float literals, every refusal path of the schema reader and the pattern
translator, non-object schemas). Each record keeps the grammar or the error of both paths.

    LLAMA_SRC=<llama.cpp checkout at tag b11192> BANKML_GGML_LIB=<b11192 release dir> python3 testing/schema_oracle.py
    → .models/oracle-schema/schemas.jsonl, replayed by `oracle_schema_grammars` (cargo test --release -- --ignored)"""
import json, os, struct, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-schema"
src = Path(os.environ.get("LLAMA_SRC", ""))
lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
if not (src / "common" / "chat.h").exists() or not (lib / "libllama-common.so").exists():
    sys.exit("set LLAMA_SRC (llama.cpp source at tag b11192) and BANKML_GGML_LIB (the b11192 release dir)")
# 0.3.5: every reproduced template, each read from a model that carries it → its own record file
if os.environ.get("BANKML_ORACLE_MODEL"):
    jobs = [(Path(os.environ["BANKML_ORACLE_MODEL"]), "schemas.jsonl")]
else:
    bonsai = next((m for m in (root / ".models" / n for n in ("Bonsai-8B-Q1_0.gguf", "Bonsai-1.7B-Q1_0.gguf")) if m.exists()), None) \
        or sys.exit("no pinned Bonsai model to read the chat template from (BANKML_ORACLE_MODEL)")
    jobs = [(bonsai, "schemas.jsonl")] + [(root / ".models" / f"{s}.gguf", f"schemas-{s}.jsonl")
                                          for s in ("SmolLM2-135M-Instruct-F16", "mindx-gen39-F16") if (root / ".models" / f"{s}.gguf").exists()]


def chat_template(path):
    """tokenizer.chat_template from the GGUF header (metadata only; the tensors are not read)"""
    f = open(path, "rb")
    u32 = lambda: struct.unpack("<I", f.read(4))[0]
    u64 = lambda: struct.unpack("<Q", f.read(8))[0]
    s = lambda: f.read(u64()).decode()
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    assert f.read(4) == b"GGUF"
    u32(), u64()
    for _ in range(u64()):
        k, t = s(), u32()
        if t == 8:
            v = s()
            if k == "tokenizer.chat_template":
                return v
        elif t == 9:
            et, n = u32(), u64()
            for _ in range(n):
                f.seek(u64(), 1) if et == 8 else f.seek(size[et], 1)
        else:
            f.seek(size[t], 1)
    sys.exit(f"{path}: no chat template")


out.mkdir(parents=True, exist_ok=True)
exe = out / "schema_oracle"
subprocess.run(["g++", "-O1", "-std=c++17", f"-I{src}/common", f"-I{src}/include", f"-I{src}/ggml/include",
                str(root / "testing" / "schema_oracle.cpp"), f"-L{lib}", "-lllama-common", "-lllama", f"-Wl,-rpath,{lib}", "-o", str(exe)], check=True)
corpus = [{"name": "llama.cpp: " + c["name"], "schema": c["schema"]} for c in json.loads((root / "testing" / "json_schema_cases.json").read_text())["cases"]]
corpus += json.loads((root / "testing" / "json_schema_corpus.json").read_text())
for model, fname in jobs:
    tmpl = out / "template.jinja"
    tmpl.write_text(chat_template(model))
    p = subprocess.run([str(exe), str(tmpl)], input="".join(c["schema"].encode().hex() + "\n" for c in corpus), capture_output=True, text=True, check=True)
    recs, cur = [], {}
    for line in p.stdout.splitlines():
        if line == ".":
            recs.append(cur)
            cur = {}
            continue
        k, _, h = line.partition(" ")
        key = {"R": "raw", "RE": "raw_err", "C": "chat", "CE": "chat_err", "P": "prompt"}[k]
        try:
            cur[key] = bytes.fromhex(h).decode()
        except UnicodeDecodeError:  # llama.cpp wrote a grammar that is not UTF-8 (bankML refuses those): kept as hex
            cur[key + "_not_utf8"] = h
    assert len(recs) == len(corpus)
    with open(out / fname, "w") as f:
        for c, r in zip(corpus, recs):
            f.write(json.dumps({**c, **r}, ensure_ascii=False) + "\n")
    print(f"schema oracle: {len(recs)} schemas recorded from libllama-common b11192 ({sum('raw' in r for r in recs)} converted, "
          f"{sum('chat' in r and r['chat'] != '' for r in recs)} chat grammars; the template of {model.name}) → {out / fname}")
