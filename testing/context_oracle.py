#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""0.3.8: behaviour at the context limit, as llama-server b11192 behaves with context shift off (its default):
a generation that fills the context stops with finish_reason "length"; a request whose prompt does not fit is refused
with HTTP 400 and `{"error": {"code": 400, "message": "request (N tokens) exceeds the available context size (M tokens),
try increasing it", "type": "exceed_context_size_error", "n_prompt_tokens": N, "n_ctx": M}}`.

Both servers run with a small context (256 tokens) so the limit is cheap to reach. Record llama-server's answers, then
replay the same requests through `bankml serve --native --ctx 256`: the text, the counts and the finish reason must be
the same, and a refusal must have the same status and body.
    BANKML_GGML_LIB=<b11192 release dir> python3 testing/context_oracle.py --record Bonsai-1.7B-Q1_0
    python3 testing/context_oracle.py --bankml Bonsai-1.7B-Q1_0
→ .models/oracle-context/context-<model>.jsonl"""
import json, os, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import SAVANTE, wait  # noqa: E402

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-context"
CTX = 256


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read() or b"{}"
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw.decode("utf-8", "replace")}


def requests():
    """Prompts from a few tokens to past the context, each greedy with a generous max_tokens."""
    cases = []
    for name, words, max_tokens in [("short-fills", 4, 400), ("mid-fills", 60, 400), ("near", 150, 400), ("edge-a", 190, 400),
                                    ("edge-b", 200, 400), ("over", 240, 32), ("far-over", 400, 32), ("short-capped", 4, 24)]:
        text = "Count these words and then keep talking: " + " ".join(["moon"] * words)
        cases.append((name, {"messages": [{"role": "system", "content": SAVANTE}, {"role": "user", "content": text}],
                             "max_tokens": max_tokens, "temperature": 0}))
    return cases


def shape(code, r):
    if code != 200:
        return {"status": code, "error": r}
    c = r["choices"][0]
    return {"status": 200, "content": c["message"]["content"], "finish_reason": c["finish_reason"],
            "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"]}


def record(stem):
    lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
    model = root / ".models" / f"{stem}.gguf"
    if not (lib / "llama-server").exists() or not model.exists():
        sys.exit("needs BANKML_GGML_LIB (b11192 release dir) and " + str(model))
    base = "http://127.0.0.1:18305"
    proc = subprocess.Popen([str(lib / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", "18305", "-t", "3",
                             "-c", str(CTX), "-np", "1", "--jinja", "--reasoning", "off", "--no-webui"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(base, proc)
        cases = []
        for name, body in requests():
            code, r = post(base, "/v1/chat/completions", {**body, "cache_prompt": False})
            s = shape(code, r)
            cases.append({"name": name, "request": body, **s})
            print(name, s.get("status"), s.get("finish_reason"), s.get("prompt_tokens"), s.get("completion_tokens"), str(s.get("error", ""))[:120], flush=True)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"context-{stem}.jsonl"
        dest.write_text("".join(json.dumps(c) + "\n" for c in cases))
        print(f"{len(cases)} context-limit cases from llama-server b11192 ({stem}, -c {CTX}) → {dest}")
    finally:
        proc.terminate()
        proc.wait(timeout=60)


def live(stem):
    binary = root / "target" / "release" / "bankml"
    model = root / ".models" / f"{stem}.gguf"
    fork = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks")) / f"{stem}.gguf.FORK.json"
    rec = out / f"context-{stem}.jsonl"
    for f in (binary, model, fork, rec):
        if not f.exists():
            sys.exit(f"needs {f}")
    base = "http://127.0.0.1:18197"
    proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18197",
                             "--upstream", "127.0.0.1:18198", "--ctx", str(CTX)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = ok = 0
    try:
        wait(base, proc)
        for c in map(json.loads, rec.read_text().splitlines()):
            post(base, "/api/generate", {"model": stem.lower(), "keep_alive": 0})  # an empty slot, as cache_prompt false
            code, r = post(base, "/v1/chat/completions", c["request"])
            got = shape(code, r)
            want = {k: v for k, v in c.items() if k not in ("name", "request")}
            n += 1
            good = got == want
            ok += good
            if not good:
                print(f"  {c['name']}: got {json.dumps(got)[:240]}\n  {' ' * len(c['name'])}  want {json.dumps(want)[:240]}")
    finally:
        proc.terminate()
        proc.wait()
    print(f"context oracle ({stem}, ctx {CTX}): {ok} of {n} requests through bankml serve --native identical to llama-server b11192 "
          f"(the text, counts and finish at the limit; the refusal's status and body past it)")
    return ok == n


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--record", "--bankml"):
        sys.exit(__doc__)
    if sys.argv[1] == "--record":
        record(sys.argv[2])
    else:
        sys.exit(0 if live(sys.argv[2]) else 1)
