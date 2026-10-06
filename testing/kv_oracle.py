#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""0.3.9: the q8_0 KV cache. llama-server b11192 run with `--cache-type-k q8_0 --cache-type-v q8_0` (flash attention,
as its default picks on CPU) answers a set of requests in one process: greedy and seeded, short and long prompts (a
prefill of several micro-batches), and a conversation whose turns reuse the cache. Recorded: the text and the counts,
`timings.cache_n` included. `bankml serve --native` with `BANKML_CACHE_TYPE=q8_0` must give the same, request by request.
    BANKML_GGML_LIB=<b11192 release dir> python3 testing/kv_oracle.py --record Bonsai-1.7B-Q1_0
    python3 testing/kv_oracle.py --bankml Bonsai-1.7B-Q1_0
→ .models/oracle-kv/kv-q8_0-<model>.jsonl"""
import json, os, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import SAVANTE, wait  # noqa: E402

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-kv"
LONG = " ".join(f"Item {i}: the {['red', 'green', 'blue', 'amber'][i % 4]} lantern number {i * 7 % 101} hangs by door {i % 13}." for i in range(120))


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"raw": (e.read() or b"").decode("utf-8", "replace")}


def requests():
    q = lambda text: [{"role": "system", "content": SAVANTE}, {"role": "user", "content": text}]
    conv = [{"role": "system", "content": SAVANTE}, {"role": "user", "content": "What is a lantern?"}]
    return [
        ("greedy", {"messages": q("Name three planets and one fact about each."), "max_tokens": 64, "temperature": 0}),
        ("seeded", {"messages": q("Invent a short poem about rain."), "max_tokens": 64, "temperature": 0.8, "seed": 11}),
        ("long-prompt", {"messages": q(LONG + "\n\nWhich door does lantern 5 hang by? Answer briefly."), "max_tokens": 48, "temperature": 0}),
        ("long-answer", {"messages": q("Explain in detail how a bonsai tree is pruned and wired over a year."), "max_tokens": 320, "temperature": 0}),
        ("conversation-1", {"messages": conv, "max_tokens": 48, "temperature": 0}),
        ("conversation-2", {"messages": conv, "max_tokens": 48, "temperature": 0, "_then": "And who lights it?"}),
    ]


def run(ask):
    """Every request in order through `ask(body)` → (status, response); conversation-2 continues conversation-1."""
    rows, last = [], None
    for name, body in requests():
        body = json.loads(json.dumps(body))
        then = body.pop("_then", None)
        if then:
            body["messages"] += [{"role": "assistant", "content": last}, {"role": "user", "content": then}]
        code, r = ask(body)
        if code != 200:
            rows.append({"name": name, "status": code, "error": r})
            continue
        last = r["choices"][0]["message"]["content"]
        rows.append({"name": name, "status": 200, "text": last, "finish_reason": r["choices"][0]["finish_reason"],
                     "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
                     "cache_n": r["timings"]["cache_n"]})
    return rows


def record(stem):
    lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
    model = root / ".models" / f"{stem}.gguf"
    if not (lib / "llama-server").exists() or not model.exists():
        sys.exit("needs BANKML_GGML_LIB (b11192 release dir) and " + str(model))
    base = "http://127.0.0.1:18311"
    proc = subprocess.Popen([str(lib / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", "18311", "-t", "3", "-c", "4096",
                             "-np", "1", "--jinja", "--reasoning", "off", "--no-webui", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(base, proc)
        rows = run(lambda b: post(base, "/v1/chat/completions", b))
        for t in rows:
            print(t["name"], t["status"], t.get("cache_n"), t.get("prompt_tokens"), t.get("completion_tokens"), repr(t.get("text", ""))[:70], flush=True)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"kv-q8_0-{stem}.jsonl"
        dest.write_text("".join(json.dumps(t) + "\n" for t in rows))
        print(f"{len(rows)} answers from llama-server b11192 ({stem}, q8_0 K and V) → {dest}")
    finally:
        proc.terminate()
        proc.wait(timeout=60)


def live(stem):
    binary = root / "target" / "release" / "bankml"
    model = root / ".models" / f"{stem}.gguf"
    fork = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks")) / f"{stem}.gguf.FORK.json"
    rec = out / f"kv-q8_0-{stem}.jsonl"
    for f in (binary, model, fork, rec):
        if not f.exists():
            sys.exit(f"needs {f}")
    base = "http://127.0.0.1:18213"
    env = {**os.environ, "BANKML_CACHE_TYPE": "q8_0"}
    proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18213",
                             "--upstream", "127.0.0.1:18214", "--ctx", "4096"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    n = ok = tokens = 0
    try:
        wait(base, proc)
        post(base, "/api/generate", {"model": stem.lower(), "keep_alive": 0})  # an empty slot, as the fresh llama-server
        want = [json.loads(l) for l in rec.read_text().splitlines()]
        got = run(lambda b: post(base, "/v1/chat/completions", b))
        for g, w in zip(got, want):
            n += 1
            ok += g == w
            tokens += w.get("completion_tokens", 0)
            if g != w:
                print(f"  FAIL {w['name']}\n    got  {json.dumps(g)[:300]}\n    want {json.dumps(w)[:300]}")
        if len(got) != len(want):
            n += 1
            print(f"  FAIL {len(got)} answers, {len(want)} recorded")
    finally:
        proc.terminate()
        proc.wait()
    print(f"kv oracle ({stem}, q8_0 K and V): {ok} of {n} answers through bankml serve --native identical to llama-server b11192 "
          f"(text, finish, counts, cache_n; {tokens} tokens)")
    return ok == n


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--record", "--bankml"):
        sys.exit(__doc__)
    if sys.argv[1] == "--record":
        record(sys.argv[2])
    else:
        sys.exit(0 if live(sys.argv[2]) else 1)
