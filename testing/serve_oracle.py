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
Writes .models/oracle-forward/serve-<model stem>.jsonl.

The 0.3.1 Ollama-shape case (`--bankml`): bankml serve --native is started on spare ports with the 8B 1-bit model and
the same conversations go through it three ways, each from an empty slot (the model unloaded and loaded again):
OpenAI's `/v1/chat/completions`, Ollama's `/api/chat` (options, NDJSON streaming for the first conversation), and
`/v1/chat/completions` once more after load → unload → reload. All three must give the same text and counts turn by
turn, and the first must equal llama-server's recorded answers above, so `/api/chat` is tied to llama-server too.
    python3 testing/serve_oracle.py --bankml        (needs target/release/bankml, the model and its FORK.json)
0.3.4: `--bankml STEM [NAME]` runs the same case on another pinned model against its own record (Bonsai-1.7B-Q1_0;
SmolLM2-135M-Instruct-F16; mindx-gen39-F16 asked for by Ollama's tag `mindx-gen39`, the alias without the type)."""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
url = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "--bankml" else "http://127.0.0.1:18094"
out = root / ".models" / "oracle-forward"


def post(path, body, base=None):
    req = urllib.request.Request((base or url) + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


SYSTEM = ("You are Savante, an autonomous research agent. You answer plainly and briefly, you say when you do not know, "
          "and you never invent sources.")
conversations = [
    ["What is a bonsai?", "How often should I water one?", "And in winter?"],
    ["Name a prime number.", "Is 91 prime?", "Why not?"],
    ["Say hello in French.", "Now in Spanish.", "Which of the two did you find easier to write, and why?"],
]


def bankml_case(stem="Bonsai-8B-Q1_0", name=None):
    """/api/chat == /v1/chat/completions == llama-server's record, and load → unload → reload changes nothing."""
    model = root / ".models" / f"{stem}.gguf"
    forks = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks"))
    fork = Path(os.environ.get("BANKML_FORK", forks / f"{stem}.gguf.FORK.json")) if stem == "Bonsai-8B-Q1_0" else forks / f"{stem}.gguf.FORK.json"
    record = out / f"serve-{stem}.jsonl"
    binary = root / "target" / "release" / "bankml"
    missing = [str(p) for p in (model, fork, record, binary) if not p.exists()]
    if missing:
        print("ollama-shape oracle skipped: missing " + ", ".join(missing))
        return 0
    base, name = "http://127.0.0.1:18195", name or stem.lower()
    proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18195",
                             "--upstream", "127.0.0.1:18196", "--ctx", "2048"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                break
            except Exception:
                if proc.poll() is not None:
                    print("FAILED: bankml serve exited"); return 1
                time.sleep(1)

        def fresh(load=True):
            assert post("/api/generate", {"model": name, "keep_alive": 0}, base)["done_reason"] == "unload"
            assert json.load(urllib.request.urlopen(base + "/api/ps"))["models"] == []
            if load:
                assert post("/api/generate", {"model": name, "keep_alive": -1, "stream": False}, base)["done_reason"] == "load"

        def v1(body):
            r = post("/v1/chat/completions", body, base)
            return (r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"], r["timings"]["cache_n"])

        def api(body, stream):
            o = {"model": name + ":latest", "messages": body["messages"], "stream": stream, "keep_alive": -1,
                 "options": {"temperature": body["temperature"], "seed": body["seed"], "num_predict": body["max_tokens"]}}
            req = urllib.request.Request(base + "/api/chat", json.dumps(o).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                lines = [json.loads(x) for x in r.read().decode().splitlines() if x.strip()]
            assert all(not x["done"] for x in lines[:-1]) and lines[-1]["done"] and "bankml_receipt" in lines[-1], "NDJSON framing"
            assert stream or len(lines) == 1
            text = "".join(x["message"]["content"] for x in lines)
            f = lines[-1]
            return (text, f["prompt_eval_count"], f["eval_count"], f["bankml_cache_n"])

        def run(ask, convs, stream_first=False):
            got, seed = [], 3000
            for ci, conv in enumerate(conversations):
                messages = [{"role": "system", "content": SYSTEM}]
                for q in conv:
                    messages.append({"role": "user", "content": q})
                    body = {"messages": list(messages), "temperature": 0.3, "seed": seed, "max_tokens": 64, "stream": False}
                    seed += 1
                    if ci not in convs:
                        continue
                    r = ask(body) if ask is v1 else ask(body, stream_first and ci == 0)
                    got.append((ci, r))
                    messages.append({"role": "assistant", "content": r[0]})
                if ci not in convs:
                    break
            return got

        want = [(t["text"], t["prompt_tokens"], t["completion_tokens"], t["cache_n"]) for t in map(json.loads, record.read_text().splitlines())]
        t0 = time.time()
        fresh(); a = run(v1, {0, 1, 2})
        fresh(); b = run(api, {0, 1, 2}, stream_first=True)
        fresh(load=False); c = run(v1, {0})  # nothing resident: /v1 loads (and verifies) the startup model itself
        ok = 0
        for i, ((_, ra), (_, rb)) in enumerate(zip(a, b)):
            good = ra == rb == want[i] and (i >= len(c) or c[i][1] == ra)
            ok += good
            if not good:
                print(f"  turn {i}: v1 {ra!r}\n          api {rb!r}\n          llama-server {want[i]!r}\n          reload {c[i][1] if i < len(c) else None!r}")
        n = len(want)
        print(f"ollama-shape oracle ({stem} as {name!r}): {ok} of {n} turns identical through /api/chat (NDJSON for conversation 1), /v1/chat/completions "
              f"and llama-server b11192's record (text, prompt/completion counts, cache reuse); {len(c)} turns unchanged after "
              f"unload and reload — {time.time() - t0:.0f} s")
        return 0 if ok == n and len(a) == len(b) == n else 1
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if "--bankml" in sys.argv:
    rest = sys.argv[sys.argv.index("--bankml") + 1:]
    sys.exit(bankml_case(*rest[:2]))
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
