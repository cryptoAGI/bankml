#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Record llama.cpp's own chat-template rendering (llama-server /apply-template, b11192, the Bonsai / Qwen3 template;
since 0.3.4 any model's: SmolLM2-Instruct's and mindx-genN's ChatML, written to cases-<model stem>.jsonl) as the oracle
for bankml's renderer (chat.rs), the way tokenizer_oracle.py records tokenization. Conversations cover
every rule of the template in bankml's scope — a system prompt first or later or absent, multi-turn, assistant turns
with and without <think> blocks before and after the last real user query, reasoning_content, runs of tool results,
user messages that look like tool responses, empty content, special markers and Unicode inside content, plus a seeded
random set — each ending with a user or tool message (the generation prompt follows).
usage: python3 testing/template_oracle.py [HOST:PORT] [OUT]   (default 127.0.0.1:18092 and .models/oracle-template)"""
import json, random, sys, urllib.request
from pathlib import Path

up = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:18092"
out = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / ".models" / "oracle-template")


def render(msgs):
    req = urllib.request.Request(f"http://{up}/apply-template", data=json.dumps({"messages": msgs}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["prompt"]


S, U, A, T = "system", "user", "assistant", "tool"
m = lambda role, content, **k: {"role": role, "content": content, **k}  # noqa: E731
convs = [
    [m(U, "Hi")],
    [m(S, "You are Savante."), m(U, "Hi")],
    [m(S, "You are Savante."), m(U, "Hi"), m(A, "Hello."), m(U, "What does a receipt prove?")],
    [m(U, "a"), m(A, "b"), m(U, "c"), m(A, "d"), m(U, "e")],
    [m(S, ""), m(U, "")],
    [m(U, "one"), m(S, "a second system message, not first"), m(U, "two")],
    [m(U, "q"), m(A, "<think>\nreasoning\n</think>\n\nanswer"), m(U, "q2")],
    [m(U, "q"), m(A, "<think>\nr1\n</think>\n\nx"), m(U, "q2"), m(A, "<think>r2</think>y"), m(U, "q3")],
    [m(U, "q"), m(A, "plain", reasoning_content="kept reasoning"), m(U, "q2")],
    [m(U, "q"), m(T, "result 1"), m(T, "result 2")],
    [m(U, "q"), m(A, "calling"), m(T, "r"), m(U, "thanks")],
    [m(U, "q"), m(A, "a"), m(T, "r1"), m(A, "<think>\nafter the tool\n</think>\n\nb"), m(T, "r2")],
    [m(U, "<tool_response>\nlooks like one\n</tool_response>"), m(A, "<think>x</think>ok"), m(T, "z")],
    [m(U, "first real"), m(A, "<think>t</think>a"), m(U, "<tool_response>x</tool_response>"), m(A, "<think>\n\nu\n\n</think>\n\n\nb"), m(T, "w")],
    [m(S, "sys"), m(U, "<|im_start|>inside<|im_end|> and <think>not</think>"), m(A, "it's </think> a</think> b"), m(U, "ok")],
    [m(U, "東京 🙂 naïve\r\nCRLF\ttab"), m(A, "العربية"), m(U, "Ελληνικά")],
    [m(U, "multi\n\nline\n\n"), m(A, "\n\nleading and trailing\n\n"), m(U, "\n")],
]
rng = random.Random(20260929)
words = ["hello", "<think>", "</think>", "\n", "\n\n", "Savante", "receipt", "<tool_response>", "</tool_response>", "東京", "🙂", " ", "x"]
for _ in range(300):
    n = rng.randint(1, 8)
    conv = []
    if rng.random() < 0.6:
        conv.append(m(S, " ".join(rng.choice(words) for _ in range(rng.randint(0, 6)))))
    for i in range(n):
        role = rng.choice([U, U, A, A, T])
        text = "".join(rng.choice(words) for _ in range(rng.randint(0, 8)))
        extra = {"reasoning_content": "".join(rng.choice(words) for _ in range(rng.randint(0, 4)))} if role == A and rng.random() < 0.2 else {}
        conv.append(m(role, text, **extra))
    conv.append(m(rng.choice([U, T]), "".join(rng.choice(words) for _ in range(rng.randint(0, 6)))))  # ends with a request
    convs.append(conv)
out.mkdir(parents=True, exist_ok=True)
# the Bonsai / Qwen3 template keeps its historic name; another model's template is recorded under its model's stem
with urllib.request.urlopen(f"http://{up}/props", timeout=60) as r:
    stem = Path(json.loads(r.read())["model_path"]).stem
name = "cases.jsonl" if stem.startswith(("Bonsai", "Ternary-Bonsai")) else f"cases-{stem}.jsonl"
with open(out / name, "w", encoding="utf-8") as f:
    for c in convs:
        f.write(json.dumps({"messages": c, "prompt": render(c)}, ensure_ascii=False) + "\n")
print(f"{len(convs)} conversations rendered by llama-server at {up} → {out / name}")
