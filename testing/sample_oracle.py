#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""P3's sampling oracle: the tokens llama.cpp b11192's llama-server samples, with a fixed seed, under the sampler chain
it builds for a request (top-k, top-p, min-p, temperature and the draw; the model's GGUF defaults where a request says
nothing). Several prompts × temperatures × top-k / top-p / min-p × seeds; each case keeps the resolved parameters,
read back from the server's own reply (`generation_settings`), so bankml replays exactly what the server ran.
Writes .models/oracle-forward/sample-<model stem>.jsonl.
usage: python3 testing/sample_oracle.py [http://127.0.0.1:18092] [N_PREDICT]"""
import json, sys, urllib.request
from pathlib import Path

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18092"
n_predict = int(sys.argv[2]) if len(sys.argv) > 2 else 32
out = Path(__file__).resolve().parents[1] / ".models" / "oracle-forward"


def post(path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


questions = ["Tell me a fact about the moon.", "Invent a name for a bakery.", "Describe a colour to someone who cannot see.",
             "Write a haiku about rain."]
variants = [{}, {"temperature": 0.3}, {"temperature": 1.0}, {"temperature": 1.5, "top_k": 40}, {"top_k": 5},
            {"top_k": 128, "top_p": 0.95}, {"top_p": 1.0, "temperature": 0.8}, {"min_p": 0.05, "temperature": 1.2},
            {"min_p": 0.2, "top_p": 1.0}, {"temperature": 0.0}]
cases = []
for qi, q in enumerate(questions):
    msgs = [{"role": "system", "content": "You are Savante."}, {"role": "user", "content": q}]
    prompt = post("/apply-template", {"messages": msgs})["prompt"]
    ids = post("/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
    for vi, v in enumerate(variants):
        body = {"prompt": ids, "n_predict": n_predict, "cache_prompt": False, "return_tokens": True, "seed": 1000 + 17 * qi + vi, **v}
        r = post("/completion", body)
        g = r["generation_settings"]
        params = {k: g[k] for k in ("temperature", "top_k", "top_p", "min_p", "min_keep", "seed", "dynatemp_range", "typical_p",
                                     "xtc_probability", "dry_multiplier", "repeat_penalty", "presence_penalty", "frequency_penalty",
                                     "top_n_sigma", "samplers")}
        cases.append({"prompt_ids": ids, "params": params, "ids": r["tokens"], "text": r["content"], "stop_type": r.get("stop_type")})
        print(f"q{qi} {v or 'defaults'} seed {params['seed']}: {len(r['tokens'])} tokens {r['content'][:50]!r}")
with urllib.request.urlopen(url + "/props", timeout=60) as r:
    stem = Path(json.load(r)["model_path"]).stem
out.mkdir(parents=True, exist_ok=True)
dest = out / f"sample-{stem}.jsonl"
dest.write_text("".join(json.dumps(c) + "\n" for c in cases))
print(f"{len(cases)} sampled continuations from llama-server b11192 ({stem}) → {dest}")
