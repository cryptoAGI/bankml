#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""P3's end-to-end oracle: the tokens llama.cpp b11192's own llama-server generates, greedily (top-k 1), for short chat
prompts. The model's graph checks (forward_oracle.py, model_oracle.py) compare bankml with the shipped ggml op by op;
this one compares with the server Savante actually runs, as a whole: the prompt rendered by /apply-template,
tokenized by /tokenize, and the continuation from /completion with cache_prompt off.

Prompts stay under 64 tokens and prompt + continuation within 256 cells (llama.cpp pads the KV length to 256s), the range whose attention bankml
reproduces (ggml's reference flash-attention path); the tiled and split-KV kernels are later steps.
Writes .models/oracle-forward/greedy-<model stem>.jsonl ({"messages", "prompt_ids", "ids"} per case), the model
named by the server's own /props.
With --long, the prompts are 64 tokens or more (a longer system prompt), so llama.cpp computes them with ggml's tiled
flash attention; prompt + continuation still stay within 256 cells. Written to greedy-long-<stem>.jsonl.
With --deep, the continuations run past 256 cells (a ~110-token prompt asking for a long answer, 200 tokens), where
llama.cpp decodes with its split-KV kernel, whose chunks follow its thread count (-t; Savante runs 3). Written to
greedy-deep-<stem>.jsonl.
usage: python3 testing/greedy_oracle.py [http://127.0.0.1:18092] [N_PREDICT] [--long | --deep]"""
import json, sys, urllib.request
from pathlib import Path

deep = "--deep" in sys.argv
long = "--long" in sys.argv or deep
argv = [a for a in sys.argv[1:] if a not in ("--long", "--deep")]
url = argv[0] if argv else "http://127.0.0.1:18092"
n_predict = int(argv[1]) if len(argv) > 1 else (200 if deep else 48)
out = Path(__file__).resolve().parents[1] / ".models" / "oracle-forward"


def post(path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


SYSTEM = "You are Savante."
if long:
    SYSTEM = ("You are Savante, an autonomous research agent. You answer plainly and briefly, you say when you do not know, "
              "and you never invent sources. You were built to run on a laptop with a one-bit or ternary model, so you keep "
              "your answers short: one to three sentences unless the user asks for more. When a question has a single "
              "correct answer, give it first and then, if it helps, one sentence of explanation.")
if deep:
    SYSTEM = ("You are Savante, an autonomous research agent and a patient teacher. When the user asks for an explanation or "
              "a story, you write at length: at least three hundred words, in full paragraphs, with concrete examples and "
              "no summary list at the end. You never invent sources, and you keep a warm, clear voice throughout.")
deep_questions = ["Explain in detail how to care for a bonsai tree through the four seasons.",
                  "Describe the water cycle step by step, with an example for each step.",
                  "Write a short story about a lighthouse keeper who finds a message in a bottle."]
questions = ["What does a bonsai need?", "Name three prime numbers.", "What is 2 + 2?", "Say hello in French.",
             "What colour is the sky on a clear day?", "Write one short sentence about the sea."]
cases = []
for q in (deep_questions if deep else questions):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    prompt = post("/apply-template", {"messages": msgs})["prompt"]
    ids = post("/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
    assert (64 <= len(ids) if long else len(ids) < 64), (q, len(ids))
    assert len(ids) + n_predict <= (1024 if deep else 256), (q, len(ids))
    r = post("/completion", {"prompt": ids, "n_predict": n_predict, "temperature": 0, "top_k": 1, "samplers": ["top_k"],
                             "cache_prompt": False, "return_tokens": True})
    # the server's list includes the end-of-turn token when it produced one; why it stopped is kept for the record
    cases.append({"messages": msgs, "prompt_ids": ids, "ids": r["tokens"], "text": r["content"],
                  "stop_type": r.get("stop_type"), "stopping_word": r.get("stopping_word")})
    print(f"{len(ids)} + {len(r['tokens'])} tokens ({r.get('stop_type')}): {r['content'][:70]!r}")
    if deep:
        assert len(ids) + len(r["tokens"]) > 256, "the deep case must run past 256 cells"

with urllib.request.urlopen(url + "/props", timeout=60) as r:
    stem = Path(json.load(r)["model_path"]).stem
out.mkdir(parents=True, exist_ok=True)
dest = out / f"greedy-{'deep-' if deep else 'long-' if long else ''}{stem}.jsonl"
dest.write_text("".join(json.dumps(c) + "\n" for c in cases))
print(f"{len(cases)} greedy continuations from llama-server b11192 ({stem}) → {dest}")
