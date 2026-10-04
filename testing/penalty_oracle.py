#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""O2's penalty oracle: the tokens llama.cpp b11192's llama-server samples under its penalties sampler
(`llama_sampler_penalties`: the repeat, frequency and presence penalties over the last `repeat_last_n` tokens), greedy
and seeded, for bankML to equal token for token.

llama-server feeds every prompt token into that window before the first draw (`server-context.cpp`:
`common_sampler_accept(.., false)` over the whole prompt, cached or not), so the prompt counts. The cases reach into it
(`repeat_last_n` smaller than, equal to and larger than the prompt) and make the models repeat themselves, which is
where a penalty decides the answer: mindx-gen39 degenerates into `,,,,` without one, and mindXtrain's imprint gate
decodes with a repetition penalty of 1.3. A request the server refuses is kept with its status and message, so bankML
can refuse the same.

Record (once per model): launches llama-server from BANKML_GGML_LIB with Savante's flags (`-t 3 -c 2048 -np 1 --jinja
--reasoning off`) on a spare port, and sends `/completion` with the chat prompt's token ids and `cache_prompt: false`;
each case keeps the parameters the server read back (`generation_settings`).
    BANKML_GGML_LIB=<b11192 release dir> python3 testing/penalty_oracle.py Bonsai-1.7B-Q1_0
    → .models/oracle-forward/penalty-<model>.jsonl, replayed by `oracle_penalties*` (cargo test --release -- --ignored)
usage: python3 testing/penalty_oracle.py STEM [N_PREDICT]"""
import json, os, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import SAVANTE, wait  # noqa: E402  (Savante's system prompt and the readiness wait)

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-forward"
PORT = 18304

QUESTIONS = ["Repeat the word moon forty times.", "List the days of the week, then list them again, three times.",
             "Tell me a fact about the moon.", "Write a haiku about rain."]
VARIANTS = [
    # the coach's ollama_predict (temperature 0, repeat_penalty 1.3), and mindXtrain's imprint penalty
    {"temperature": 0.0, "repeat_penalty": 1.3},
    {"temperature": 0.0, "repeat_penalty": 1.1},
    {"temperature": 0.0, "repeat_penalty": 1.3, "repeat_last_n": 0},
    {"temperature": 0.0, "repeat_penalty": 1.5, "repeat_last_n": 1},
    {"temperature": 0.0, "repeat_penalty": 2.0, "repeat_last_n": 8},
    {"temperature": 0.0, "repeat_penalty": 1.3, "repeat_last_n": 4096},
    {"temperature": 0.0, "repeat_penalty": 0.8},
    {"temperature": 0.0, "frequency_penalty": 0.5},
    {"temperature": 0.0, "presence_penalty": 1.0},
    {"temperature": 0.0, "frequency_penalty": -0.4, "presence_penalty": 0.6, "repeat_last_n": 16},
    {"repeat_penalty": 1.3},
    {"temperature": 1.0, "repeat_penalty": 1.2, "frequency_penalty": 0.3},
    {"temperature": 0.8, "top_k": 40, "presence_penalty": 1.5, "repeat_last_n": 128},
    {"temperature": 1.2, "min_p": 0.05, "repeat_penalty": 1.15, "repeat_last_n": 32},
    # limits: what the server does with them is the oracle
    {"temperature": 0.0, "repeat_penalty": 1.3, "repeat_last_n": -1},
    {"temperature": 0.0, "repeat_penalty": 0.0},
    {"temperature": 0.0, "repeat_penalty": -1.0},
]
KEYS = ("temperature", "top_k", "top_p", "min_p", "min_keep", "seed", "repeat_last_n", "repeat_penalty", "presence_penalty",
        "frequency_penalty", "dry_multiplier", "typical_p", "xtc_probability", "dynatemp_range", "top_n_sigma", "samplers")


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return 200, json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read() or b"{}"
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw.decode("utf-8", "replace")}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    stem = sys.argv[1]
    n_predict = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
    model = root / ".models" / f"{stem}.gguf"
    if not (lib / "llama-server").exists() or not model.exists():
        sys.exit("needs BANKML_GGML_LIB (b11192 release dir) and " + str(model))
    base = f"http://127.0.0.1:{PORT}"
    proc = subprocess.Popen([str(lib / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", str(PORT), "-t", "3",
                             "-c", "2048", "-np", "1", "--jinja", "--reasoning", "off", "--no-webui"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(base, proc)
        cases = []
        for qi, q in enumerate(QUESTIONS):
            msgs = [{"role": "system", "content": SAVANTE}, {"role": "user", "content": q}]
            _, t = post(base, "/apply-template", {"messages": msgs})
            _, tk = post(base, "/tokenize", {"content": t["prompt"], "add_special": False, "parse_special": True})
            ids = tk["tokens"]
            for vi, v in enumerate(VARIANTS):
                body = {"prompt": ids, "n_predict": n_predict, "cache_prompt": False, "return_tokens": True,
                        "seed": 2000 + 19 * qi + vi, **v}
                t0 = time.time()
                code, r = post(base, "/completion", body)
                if code != 200:
                    cases.append({"prompt_ids": ids, "request": body, "status": code, "error": r.get("error", r)})
                    print(f"q{qi} {v}: refused {code} {json.dumps(r)[:140]}", flush=True)
                    continue
                g = r["generation_settings"]
                cases.append({"prompt_ids": ids, "request": body, "params": {k: g.get(k) for k in KEYS}, "ids": r["tokens"],
                              "text": r["content"], "stop_type": r.get("stop_type")})
                print(f"q{qi} {v} seed {g['seed']}: {len(r['tokens'])} tokens, {time.time() - t0:.0f} s: {r['content'][:50]!r}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"penalty-{stem}.jsonl"
        dest.write_text("".join(json.dumps(c) + "\n" for c in cases))
        print(f"{len(cases)} penalty cases from llama-server b11192 ({stem}) → {dest}")
    finally:
        proc.terminate()
        proc.wait(timeout=60)


if __name__ == "__main__":
    main()
