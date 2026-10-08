#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""0.4.2: `max_tokens: 0` as llama-server b11192 answers it, and Ollama's `num_predict: 0`.

llama-server checks the generation limit only once a token is generated (`n_gen > 0`, server-context.cpp "check the
limits"), so `max_tokens: 0` still samples one token — with its logprobs when asked for: the next-token distribution
after the prompt, which is what a yes/no forecast reads. bankML stopped before the first token. Record llama-server's
answers, then replay them through `bankml serve --native` with `logprobs_oracle.py`'s comparisons (text, finish reason,
ids, bytes, every logprob the same 32-bit float; streamed chunk by chunk).
Ollama's runner limits only when `numPredict > 0` (runner/ollamarunner, 0.20.0), so `num_predict: 0` is no limit:
`/api/chat` with 0 must answer as with -1.
    BANKML_GGML_LIB=<b11192 release dir> python3 testing/maxzero_oracle.py --record Bonsai-1.7B-Q1_0
    python3 testing/maxzero_oracle.py --bankml Bonsai-1.7B-Q1_0
→ .models/oracle-logprobs/maxzero-<model>.jsonl"""
import json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import SAVANTE, wait  # noqa: E402
from logprobs_oracle import ask, post, same, same_stream  # noqa: E402

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-logprobs"


def requests():
    q = lambda text: [{"role": "system", "content": SAVANTE}, {"role": "user", "content": text}]
    yes_no = q("Is water wet? Answer yes or no.")
    return [
        ("zero-top5", {"messages": yes_no, "max_tokens": 0, "temperature": 0, "logprobs": True, "top_logprobs": 5}),
        ("zero-top1", {"messages": yes_no, "max_tokens": 0, "temperature": 0, "logprobs": True, "top_logprobs": 1}),
        ("zero-plain", {"messages": yes_no, "max_tokens": 0, "temperature": 0}),
        ("zero-seeded", {"messages": q("Invent a name for a cat."), "max_tokens": 0, "temperature": 0.9, "seed": 7, "logprobs": True, "top_logprobs": 3}),
        ("stream-zero-top3", {"messages": yes_no, "max_tokens": 0, "temperature": 0, "logprobs": True, "top_logprobs": 3, "stream": True}),
        ("one-top5", {"messages": yes_no, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5}),
    ]


def record(stem):
    lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
    model = root / ".models" / f"{stem}.gguf"
    base = "http://127.0.0.1:18317"
    proc = subprocess.Popen([str(lib / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", "18317", "-t", "3", "-c", "2048",
                             "-np", "1", "--jinja", "--reasoning", "off", "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(base, proc)
        cases = []
        for name, body in requests():
            s = ask(base, {**body, "cache_prompt": False})
            cases.append({"name": name, "request": body, **s})
            n = len(s.get("logprobs") or s.get("chunks") or [])
            print(name, s["status"], s.get("finish_reason"), n, "entries/chunks", repr(str(s.get("content") or "")[:40]), flush=True)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"maxzero-{stem}.jsonl"
        dest.write_text("".join(json.dumps(c) + "\n" for c in cases))
        print(f"{len(cases)} max_tokens-0 cases from llama-server b11192 ({stem}) → {dest}")
    finally:
        proc.terminate()
        proc.wait(timeout=60)


def live(stem):
    binary = root / "target" / "release" / "bankml"
    model = root / ".models" / f"{stem}.gguf"
    fork = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks")) / f"{stem}.gguf.FORK.json"
    rec = out / f"maxzero-{stem}.jsonl"
    base = "http://127.0.0.1:18197"
    proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18197",
                             "--upstream", "127.0.0.1:18198", "--ctx", "2048"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = ok = 0
    try:
        wait(base, proc)
        for c in map(json.loads, rec.read_text().splitlines()):
            post(base, "/api/generate", {"model": stem.lower(), "keep_alive": 0})
            got = ask(base, c["request"])
            good, why = (same_stream if "chunks" in c else same)(got, c)
            n += 1
            ok += good
            if not good:
                print(f"  {c['name']}: {why}")
        # Ollama: num_predict 0 is no limit, the same answer as -1
        msgs = requests()[0][1]["messages"]
        answers = []
        for k in (0, -1):
            post(base, "/api/generate", {"model": stem.lower(), "keep_alive": 0})
            code, r = post(base, "/api/chat", {"model": stem.lower(), "messages": msgs, "stream": False,
                                                "options": {"temperature": 0, "num_predict": k, "num_ctx": 2048}})
            answers.append((code, (r.get("message") or {}).get("content"), r.get("eval_count")))
        n += 1
        good = answers[0] == answers[1] and answers[0][0] == 200 and (answers[0][2] or 0) > 1
        ok += good
        if not good:
            print(f"  /api/chat num_predict 0 vs -1: {answers}")
    finally:
        proc.terminate()
        proc.wait()
    print(f"max_tokens 0 ({stem}): {ok} of {n} identical to llama-server b11192 (one token, its logprobs the same float), "
          f"and Ollama's num_predict 0 answered as no limit")
    return ok == n


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--record", "--bankml"):
        sys.exit(__doc__)
    if sys.argv[1] == "--record":
        record(sys.argv[2])
    else:
        sys.exit(0 if live(sys.argv[2]) else 1)
