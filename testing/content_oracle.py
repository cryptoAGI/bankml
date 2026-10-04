#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The JSON-content oracle (0.3.5): how llama-server b11192 turns a constrained answer's raw text into the message
`content`, from llama.cpp's own code — `testing/content_oracle.cpp` calls `common_chat_parse(raw, false, …)` with the
PEG chat parser `common_chat_templates_apply` builds for the request's schema (`--jinja --reasoning off`), inside the
b11192 release's `libllama-common.so`; no model is loaded. The server then answers with that content, or with the raw
text when the parse gives an empty message (`server_task_result_cmpl_final::to_json_oaicompat_chat`).

The corpus: every character-boundary prefix of every answer llama-server gave under JSON mode and JSON schemas
(`.models/oracle-json/{json,schema}-*.jsonl`, each with the schema it was asked under) — the texts a length cut can
leave — and edge cases (fences, whitespace, a reasoning block, scalars, escapes). One file per reproduced template.

    LLAMA_SRC=<llama.cpp at tag b11192> BANKML_GGML_LIB=<b11192 release dir> python3 testing/content_oracle.py
    → .models/oracle-content/content-<template>.jsonl, replayed by `oracle_json_content` (cargo test --release -- --ignored)"""
import json, os, struct, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-content"
src = Path(os.environ.get("LLAMA_SRC", ""))
lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
if not (src / "common" / "chat.h").exists() or not (lib / "libllama-common.so").exists():
    sys.exit("set LLAMA_SRC (llama.cpp source at tag b11192) and BANKML_GGML_LIB (the b11192 release dir)")
sys.path.insert(0, str(root / "testing"))


def chat_template(path):
    """tokenizer.chat_template from the GGUF header (metadata only)"""
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


def schema_text(req):
    """the schema the request asked under, as the chat path gets it ("" = JSON mode, {"type": "object"})"""
    s = req.get("json_schema")
    rf = req.get("response_format") or {}
    if rf.get("type") == "json_schema":
        s = rf["json_schema"]["schema"]
    elif rf.get("type") == "json_object":
        s = rf.get("schema", s)
    return "" if not s else json.dumps(s, separators=(",", ":"))


EDGES = ["", " ", "\n\n", "`", "``", "```", "```j", "```json", "```json\n", "```json\n\n", "```json\n{}\n```", "```json\n{}\n``", "```json\n{} ",
         "{", "{\n ", "{}", "{} \n", "{\"a\": \"}\\\"\"} \n", "[", "[1, 2] ", "\"", "\"a\\\"b", "\"a\\\"b\" ", "\"é\"", "-", "-1", "-1\n\n  ", "1e",
         "true", "tr", "null ", "<think>", "<think>\n", "<think>\n\n</think>", "<think>\n\n</think>\n\n", "<think>\nx\n</think>\n\n{\"a\": 1}",
         "<think>\n\n</think>\n\n{\"a\"", "  ```json\n[1]\n```",
         # 0.3.5: cuts inside escapes (llama.cpp ends the string before an unfinished one)
         "\"a\\", "\"a\\u", "\"a\\u0", "\"a\\u00e", "\"a\\u00e9", "{\"k\\", "{\"k\": \"v\\u12", "{\"k\": \"v\\\\", "{\"k\": \"v\\\\\\",
         "[\"\\", "\"\\ud83d\\", "\"\\ud83d\\ude", "{\"k\": 1.", "{\"k\": tr", "{\"k\": \"\\\""]
SCHEMAS = ["", '{"type":"number"}', '{"type":"string"}', '{"type":"array"}', '{"enum":["yes","no"]}', '{"type":"boolean"}']

out.mkdir(parents=True, exist_ok=True)
exe = out / "content_oracle"
subprocess.run(["g++", "-O1", "-std=c++17", f"-I{src}/common", f"-I{src}/include", f"-I{src}/ggml/include",
                str(root / "testing" / "content_oracle.cpp"), f"-L{lib}", "-lllama-common", "-lllama", f"-Wl,-rpath,{lib}", "-o", str(exe)], check=True)
recs = sorted((root / ".models" / "oracle-json").glob("*.jsonl"))
templates = {"qwen3": "Bonsai-1.7B-Q1_0", "smollm2": "SmolLM2-135M-Instruct-F16", "chatml": "mindx-gen39-F16"}
for tname, stem in templates.items():
    gguf = root / ".models" / f"{stem}.gguf"
    if not gguf.exists():
        gguf = root / ".models" / "Bonsai-8B-Q1_0.gguf" if tname == "qwen3" else None
    if gguf is None or not gguf.exists():
        print(f"content oracle: {tname} skipped (no model to read its template from)")
        continue
    tmpl = out / "template.jinja"
    tmpl.write_text(chat_template(gguf))
    cases = set()
    for r in recs:
        for line in r.read_text().splitlines():
            c = json.loads(line)
            if c.get("grammar_lazy") is not False or "raw" not in c:
                continue
            if "grammar" in c["request"]:  # a user GBNF: no chat parse applies
                continue
            st, raw = schema_text(c["request"]), c["raw"]
            for i in range(len(raw) + 1):
                cases.add((st, raw[:i]))
    for st in SCHEMAS:
        for e in EDGES:
            cases.add((st, e))
    cases = sorted(cases)
    p = subprocess.run([str(exe), str(tmpl)], input="".join(f"{s.encode().hex()} {r.encode().hex()}\n" for s, r in cases),
                       capture_output=True, text=True, check=True)
    lines = p.stdout.splitlines()
    assert len(lines) == len(cases)
    dest = out / f"content-{tname}.jsonl"
    with open(dest, "w") as f:
        for (s, r), l in zip(cases, lines):
            k, _, h = l.partition(" ")
            v = bytes.fromhex(h).decode()
            f.write(json.dumps({"schema": s, "raw": r, **({"content": v} if k == "C" else {"error": v})}, ensure_ascii=False) + "\n")
    print(f"content oracle ({tname}): {len(cases)} raw texts parsed by libllama-common b11192 "
          f"({sum(l.startswith('E') for l in lines)} refused) → {dest}")
