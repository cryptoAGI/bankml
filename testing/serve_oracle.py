#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The 0.3.0 oracle: Savante-style conversations through llama-server b11192's own chat endpoint
(`/v1/chat/completions`, the path Savante uses), turn after turn in one server process, so its prompt cache carries
each conversation forward exactly as it does for Savante. Sampling at Savante's temperature (0.3) with a fixed seed
per turn; the model's own defaults for the rest. Each turn records the request, the answer's text, and the server's
counts: prompt tokens, completion tokens, and how many prompt tokens it took from its cache (`timings.cache_n`).
bankml's native engine must give the same text and the same counts, turn by turn (`oracle_native_serve`).

Start a FRESH llama-server (an empty cache) with Savante's flags on a spare port, then:
    python3 testing/serve_oracle.py http://127.0.0.1:18094
Writes .models/oracle-forward/serve-<model stem>.jsonl."""
import json, sys, urllib.request
from pathlib import Path

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18094"
out = Path(__file__).resolve().parents[1] / ".models" / "oracle-forward"


def post(path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


SYSTEM = ("You are Savante, an autonomous research agent. You answer plainly and briefly, you say when you do not know, "
          "and you never invent sources.")
conversations = [
    ["What is a bonsai?", "How often should I water one?", "And in winter?"],
    ["Name a prime number.", "Is 91 prime?", "Why not?"],
    ["Say hello in French.", "Now in Spanish.", "Which of the two did you find easier to write, and why?"],
]
turns = []
seed = 3000
for ci, conv in enumerate(conversations):
    messages = [{"role": "system", "content": SYSTEM}]
    for ti, q in enumerate(conv):
        messages.append({"role": "user", "content": q})
        body = {"messages": list(messages), "temperature": 0.3, "seed": seed, "max_tokens": 64, "stream": False}
        r = post("/v1/chat/completions", body)
        text = r["choices"][0]["message"].get("content") or ""
        u, t = r.get("usage", {}), r.get("timings", {})
        turns.append({"conversation": ci, "turn": ti, "request": body, "text": text, "finish_reason": r["choices"][0].get("finish_reason"),
                      "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "cache_n": t.get("cache_n")})
        print(f"c{ci}t{ti}: prompt {u.get('prompt_tokens')} (cached {t.get('cache_n')}), +{u.get('completion_tokens')}: {text[:60]!r}")
        messages.append({"role": "assistant", "content": text})
        seed += 1
with urllib.request.urlopen(url + "/props", timeout=60) as r:
    stem = Path(json.load(r)["model_path"]).stem
out.mkdir(parents=True, exist_ok=True)
dest = out / f"serve-{stem}.jsonl"
dest.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in turns), encoding="utf-8")
print(f"{len(turns)} turns from llama-server b11192's /v1/chat/completions ({stem}) → {dest}")
