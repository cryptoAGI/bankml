#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""0.3.8: the single-slot contract. `bankml serve --native` has one slot, as llama-server run with `-np 1` (Savante's
flags): requests are served one at a time, and each reuses whatever prefix of its prompt the slot still holds from the
request before it — another conversation's, when sessions interleave.

1. Interleaved sessions: three Savante-style conversations take turns (A1 B1 C1 A2 B2 C2 A3 B3 C3) through
   llama-server b11192 with its prompt cache on, so every turn finds the previous conversation's tokens in the slot.
   Recorded: the text, the counts and `timings.cache_n`. bankML must give the same, turn by turn.
2. The queue (bankML alone; llama-server's order among simultaneous arrivals is not deterministic): four requests sent
   at once are all answered, none refused or mixed, each with the text it gets when asked alone.
    BANKML_GGML_LIB=<b11192 release dir> python3 testing/session_oracle.py --record Bonsai-1.7B-Q1_0
    python3 testing/session_oracle.py --bankml Bonsai-1.7B-Q1_0
→ .models/oracle-sessions/sessions-<model>.jsonl"""
import json, os, subprocess, sys, threading, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import SAVANTE, wait  # noqa: E402

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-sessions"
CONVERSATIONS = [
    ["What is a bonsai?", "How often should I water one?", "And in winter?"],
    ["Name a prime number.", "Is 91 prime?", "Why not?"],
    ["What is the moon made of?", "How far away is it?", "Could we live there?"],
]
QUEUE = ["Say one word for blue.", "Name a fruit.", "What is 3 times 4?", "Name a river."]


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"raw": (e.read() or b"").decode("utf-8", "replace")}


def body(history, seed):
    return {"messages": [{"role": "system", "content": SAVANTE}] + history, "max_tokens": 40, "temperature": 0.3, "seed": seed}


def interleave(ask):
    """The turns in interleaved order; `ask(body)` → (status, response). Each conversation keeps its own history."""
    hist = [[] for _ in CONVERSATIONS]
    turns = []
    for ti in range(3):
        for ci, conv in enumerate(CONVERSATIONS):
            hist[ci].append({"role": "user", "content": conv[ti]})
            b = body(hist[ci], 100 * ci + ti)
            code, r = ask(b)
            if code != 200:
                turns.append({"conversation": ci, "turn": ti, "status": code, "error": r})
                continue
            text = r["choices"][0]["message"]["content"]
            hist[ci].append({"role": "assistant", "content": text})
            turns.append({"conversation": ci, "turn": ti, "status": 200, "text": text, "finish_reason": r["choices"][0]["finish_reason"],
                          "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
                          "cache_n": r["timings"]["cache_n"]})
    return turns


def record(stem):
    lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
    model = root / ".models" / f"{stem}.gguf"
    if not (lib / "llama-server").exists() or not model.exists():
        sys.exit("needs BANKML_GGML_LIB (b11192 release dir) and " + str(model))
    base = "http://127.0.0.1:18309"
    proc = subprocess.Popen([str(lib / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", "18309", "-t", "3", "-c", "2048",
                             "-np", "1", "--jinja", "--reasoning", "off", "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(base, proc)
        turns = interleave(lambda b: post(base, "/v1/chat/completions", b))
        for t in turns:
            print(t["conversation"], t["turn"], t["status"], t.get("cache_n"), t.get("prompt_tokens"), repr(t.get("text", ""))[:60], flush=True)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"sessions-{stem}.jsonl"
        dest.write_text("".join(json.dumps(t) + "\n" for t in turns))
        print(f"{len(turns)} interleaved turns from llama-server b11192 ({stem}, -np 1) → {dest}")
    finally:
        proc.terminate()
        proc.wait(timeout=60)


def live(stem):
    binary = root / "target" / "release" / "bankml"
    model = root / ".models" / f"{stem}.gguf"
    fork = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks")) / f"{stem}.gguf.FORK.json"
    rec = out / f"sessions-{stem}.jsonl"
    for f in (binary, model, fork, rec):
        if not f.exists():
            sys.exit(f"needs {f}")
    base = "http://127.0.0.1:18207"
    proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18207",
                             "--upstream", "127.0.0.1:18208", "--ctx", "2048"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = ok = 0

    def check(name, good, detail=""):
        nonlocal n, ok
        n += 1
        ok += bool(good)
        if not good:
            print(f"  FAIL {name} {detail}")

    try:
        wait(base, proc)
        post(base, "/api/generate", {"model": stem.lower(), "keep_alive": 0})  # an empty slot, as the fresh llama-server
        want = [json.loads(l) for l in rec.read_text().splitlines()]
        got = interleave(lambda b: post(base, "/v1/chat/completions", b))
        for g, w in zip(got, want):
            check(f"conversation {w['conversation']} turn {w['turn']}", g == w, f"\n    got  {json.dumps(g)[:300]}\n    want {json.dumps(w)[:300]}")
        check("as many turns as recorded", len(got) == len(want))
        # the queue: each question alone, then all at once
        ask = lambda q: post(base, "/v1/chat/completions", {"messages": [{"role": "system", "content": SAVANTE}, {"role": "user", "content": q}],
                                                             "max_tokens": 24, "temperature": 0})
        alone = {}
        for q in QUEUE:
            code, r = ask(q)
            alone[q] = r["choices"][0]["message"]["content"] if code == 200 else None
        together = {}

        def go(q):
            together[q] = ask(q)

        threads = [threading.Thread(target=go, args=(q,)) for q in QUEUE]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for q in QUEUE:
            code, r = together[q]
            check(f"queued {q!r}: answered, and as when asked alone", code == 200 and r["choices"][0]["message"]["content"] == alone[q],
                  f"{code} {json.dumps(r)[:200]} vs {alone[q]!r}")
    finally:
        proc.terminate()
        proc.wait()
    print(f"session oracle ({stem}): {ok} of {n} checks — interleaved conversations through one slot identical to llama-server b11192 "
          f"(text, counts, cache_n), turn by turn; {len(QUEUE)} simultaneous requests queued and each answered as alone")
    return ok == n


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--record", "--bankml"):
        sys.exit(__doc__)
    if sys.argv[1] == "--record":
        record(sys.argv[2])
    else:
        sys.exit(0 if live(sys.argv[2]) else 1)
