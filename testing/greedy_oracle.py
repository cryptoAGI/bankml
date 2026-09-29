#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""P3's end-to-end oracle: the tokens llama.cpp b11192's own llama-server generates, greedily (top-k 1), for short chat
prompts. The model's graph checks (forward_oracle.py, model_oracle.py) compare bankml with the shipped ggml op by op;
this one compares with the server Savante actually runs, as a whole: the prompt rendered by /apply-template,
tokenized by /tokenize, and the continuation from /completion with cache_prompt off.

Prompts stay under 64 tokens and prompt + continuation under 512 cells, the range whose attention bankml
reproduces (ggml's reference flash-attention path); the tiled and split-KV kernels are later steps.
Writes .models/oracle-forward/greedy.jsonl: {"messages", "prompt_ids", "ids"} per case.
usage: python3 testing/greedy_oracle.py [http://127.0.0.1:18092] [N_PREDICT]"""
import json, sys, urllib.request
from pathlib import Path

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18092"
n_predict = int(sys.argv[2]) if len(sys.argv) > 2 else 48
out = Path(__file__).resolve().parents[1] / ".models" / "oracle-forward"


def post(path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


SYSTEM = "You are Savante."
questions = ["What does a bonsai need?", "Name three prime numbers.", "What is 2 + 2?", "Say hello in French.",
             "What colour is the sky on a clear day?", "Write one short sentence about the sea."]
cases = []
for q in questions:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    prompt = post("/apply-template", {"messages": msgs})["prompt"]
    ids = post("/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
    assert len(ids) < 64, (q, len(ids))
    r = post("/completion", {"prompt": ids, "n_predict": n_predict, "temperature": 0, "top_k": 1, "samplers": ["top_k"],
                             "cache_prompt": False, "return_tokens": True})
    cases.append({"messages": msgs, "prompt_ids": ids, "ids": r["tokens"], "text": r["content"]})
    print(f"{len(ids)} + {len(r['tokens'])} tokens: {r['content'][:70]!r}")
out.mkdir(parents=True, exist_ok=True)
(out / "greedy.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
print(f"{len(cases)} greedy continuations from llama-server b11192 → {out / 'greedy.jsonl'}")
