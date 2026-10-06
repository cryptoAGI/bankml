#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""0.3.8: log-probabilities on /v1 (`logprobs`, `top_logprobs`), as llama-server b11192 reports them: per generated
token its id, text, bytes, `logprob` (logf of the raw-logit softmax), and the top tokens with theirs. Record
llama-server's answers, then replay the same requests through `bankml serve --native`: ids, texts and bytes must be
equal and every logprob the same 32-bit float; a refused request must be refused with the same status and message.
    BANKML_GGML_LIB=<b11192 release dir> python3 testing/logprobs_oracle.py --record Bonsai-1.7B-Q1_0
    python3 testing/logprobs_oracle.py --bankml Bonsai-1.7B-Q1_0
→ .models/oracle-logprobs/logprobs-<model>.jsonl"""
import json, os, struct, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import SAVANTE, wait  # noqa: E402

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-logprobs"


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
    q = lambda text: [{"role": "system", "content": SAVANTE}, {"role": "user", "content": text}]
    return [
        ("greedy-top5", {"messages": q("Name three planets."), "max_tokens": 16, "temperature": 0, "logprobs": True, "top_logprobs": 5}),
        ("greedy-top1", {"messages": q("What is two plus two?"), "max_tokens": 12, "temperature": 0, "logprobs": True, "top_logprobs": 1}),
        ("greedy-default20", {"messages": q("Say hello."), "max_tokens": 8, "temperature": 0, "logprobs": True}),
        ("seeded-top5", {"messages": q("Invent a name for a cat."), "max_tokens": 16, "temperature": 0.9, "seed": 7, "logprobs": True, "top_logprobs": 5}),
        ("top0", {"messages": q("Say yes."), "max_tokens": 6, "temperature": 0, "logprobs": True, "top_logprobs": 0}),
        ("stop-string", {"messages": q("Count from one to ten in words."), "max_tokens": 30, "temperature": 0, "logprobs": True, "top_logprobs": 3, "stop": ["five"]}),
        ("multibyte", {"messages": q("Write the word 'café' and a heart emoji."), "max_tokens": 16, "temperature": 0, "logprobs": True, "top_logprobs": 3}),
        ("no-logprobs", {"messages": q("Say hi."), "max_tokens": 6, "temperature": 0}),
        ("top-without-logprobs", {"messages": q("Say hi."), "max_tokens": 6, "temperature": 0, "top_logprobs": 3}),
    ]


def shape(code, r):
    if code != 200:
        return {"status": code, "message": (r.get("error") or {}).get("message", r.get("raw"))}
    c = r["choices"][0]
    lp = c.get("logprobs")
    return {"status": 200, "content": c["message"]["content"], "finish_reason": c["finish_reason"],
            "logprobs": lp["content"] if lp else None}


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def same(got, want):
    if (got["status"], got.get("content"), got.get("finish_reason"), got.get("message")) != (want["status"], want.get("content"), want.get("finish_reason"), want.get("message")):
        return False, "status/content/finish/message"
    a, b = got.get("logprobs"), want.get("logprobs")
    if (a is None) != (b is None):
        return False, f"logprobs present: {a is not None} vs {b is not None}"
    for i, (x, y) in enumerate(zip(a or [], b or [])):
        for k in ("id", "token", "bytes"):
            if x[k] != y[k]:
                return False, f"token {i} {k}: {x[k]!r} vs {y[k]!r}"
        if f32(x["logprob"]) != f32(y["logprob"]):
            return False, f"token {i} logprob {x['logprob']!r} vs {y['logprob']!r}"
        if len(x["top_logprobs"]) != len(y["top_logprobs"]):
            return False, f"token {i} top count {len(x['top_logprobs'])} vs {len(y['top_logprobs'])}"
        for j, (u, v) in enumerate(zip(x["top_logprobs"], y["top_logprobs"])):
            if (u["id"], u["token"], u["bytes"]) != (v["id"], v["token"], v["bytes"]) or f32(u["logprob"]) != f32(v["logprob"]):
                return False, f"token {i} top {j}: {u} vs {v}"
    if len(a or []) != len(b or []):
        return False, f"{len(a or [])} tokens vs {len(b or [])}"
    return True, ""


def record(stem):
    lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
    model = root / ".models" / f"{stem}.gguf"
    base = "http://127.0.0.1:18307"
    proc = subprocess.Popen([str(lib / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", "18307", "-t", "3", "-c", "2048",
                             "-np", "1", "--jinja", "--reasoning", "off", "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(base, proc)
        cases = []
        for name, body in requests():
            code, r = post(base, "/v1/chat/completions", {**body, "cache_prompt": False})
            s = shape(code, r)
            cases.append({"name": name, "request": body, **s})
            print(name, s["status"], len(s.get("logprobs") or []), "tokens", str(s.get("message") or s.get("content"))[:60], flush=True)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"logprobs-{stem}.jsonl"
        dest.write_text("".join(json.dumps(c) + "\n" for c in cases))
        print(f"{len(cases)} logprob cases from llama-server b11192 ({stem}) → {dest}")
    finally:
        proc.terminate()
        proc.wait(timeout=60)


def live(stem):
    binary = root / "target" / "release" / "bankml"
    model = root / ".models" / f"{stem}.gguf"
    fork = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks")) / f"{stem}.gguf.FORK.json"
    rec = out / f"logprobs-{stem}.jsonl"
    base = "http://127.0.0.1:18177"
    proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18177",
                             "--upstream", "127.0.0.1:18178", "--ctx", "2048"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = ok = 0
    try:
        wait(base, proc)
        for c in map(json.loads, rec.read_text().splitlines()):
            post(base, "/api/generate", {"model": stem.lower(), "keep_alive": 0})
            code, r = post(base, "/v1/chat/completions", c["request"])
            good, why = same(shape(code, r), c)
            n += 1
            ok += good
            if not good:
                print(f"  {c['name']}: {why}")
    finally:
        proc.terminate()
        proc.wait()
    print(f"logprobs oracle ({stem}): {ok} of {n} requests through bankml serve --native identical to llama-server b11192 "
          f"(ids, texts, bytes, every logprob the same float; refusals with its message)")
    return ok == n


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--record", "--bankml"):
        sys.exit(__doc__)
    if sys.argv[1] == "--record":
        record(sys.argv[2])
    else:
        sys.exit(0 if live(sys.argv[2]) else 1)
