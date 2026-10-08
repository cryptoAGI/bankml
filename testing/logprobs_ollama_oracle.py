#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""0.4.2: log-probabilities on Ollama's API (`/api/chat`, `/api/generate`; `logprobs`, `top_logprobs` as Ollama 0.20
asks for them) and the receipt's `logprobs_sha256`, against llama-server b11192's own record.

The record is `logprobs_oracle.py --record`'s (`.models/oracle-logprobs/logprobs-<model>.jsonl`): llama-server's
`/v1/chat/completions` answers with logprobs. Each recorded request is replayed through `bankml serve --native` as an
Ollama request (the messages; `temperature`, `seed`, `num_predict`, `stop` in `options`, as serve_oracle.py maps them),
and Ollama's entries must carry llama-server's: per token the same text and bytes and the same 32-bit float, the same
top tokens in the same order (Ollama's shape: no `id`; `bytes` and `top_logprobs` left out when empty).
- `top_logprobs: 0` (Ollama: the chosen token only) is compared with the record's `top_logprobs: 1` answer, whose
  chosen-token float bankML computes the same way (`sampler::token_probs` sums in the order its partial sort leaves).
- Streamed: every entry the NDJSON lines carry, in order, against every entry the record's chunks carry.
- `/api/generate` with `system` + `prompt` is the same conversation as `/api/chat` with those two messages.
- Refusal: `top_logprobs` 21 is a 400 with Ollama's message ("top_logprobs must be between 0 and 20").
- Receipts: `logprobs_sha256` is the sha256 of the logprobs array exactly as received — the body's substring when not
  streamed, the lines' arrays joined when streamed — on `/api/*` and on `/v1`; absent when logprobs were not asked for.
    python3 testing/logprobs_ollama_oracle.py --bankml Bonsai-1.7B-Q1_0"""
import hashlib, json, os, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import wait  # noqa: E402
from logprobs_oracle import f32  # noqa: E402

root = Path(__file__).resolve().parents[1]
rec_dir = root / ".models" / "oracle-logprobs"


def raw_post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, (e.read() or b"").decode("utf-8", "replace")


def array_at(text, key='"logprobs": '):
    """The raw JSON array after `key` (a key, not text inside a string: quotes there are escaped), or None."""
    i = text.find(key)
    if i < 0:
        return None
    i += len(key)
    if text[i] != "[":
        return None
    depth, in_str, esc = 0, False, False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            # the character after a backslash is escaped (\" and \\ included); an unescaped quote ends the string
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str, esc = True, False
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return None


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def ollama_body(name, req, top, chat=True):
    o = {"temperature": req.get("temperature", 0.8), "num_predict": req["max_tokens"]}
    if "seed" in req:
        o["seed"] = req["seed"]
    if "stop" in req:
        o["stop"] = req["stop"]
    b = {"model": name, "stream": bool(req.get("stream")), "logprobs": True, "top_logprobs": top, "options": o}
    if chat:
        b["messages"] = req["messages"]
    else:
        b["system"] = req["messages"][0]["content"]
        b["prompt"] = req["messages"][1]["content"]
    return b


def same_entries(got, want, top):
    """Ollama entries (`got`) against llama-server's (`want`): text, bytes, float, and `top` top tokens (none at 0)."""
    if len(got) != len(want):
        return False, f"{len(got)} entries vs {len(want)}"
    for i, (x, y) in enumerate(zip(got, want)):
        if "id" in x:
            return False, f"entry {i} has an id (not Ollama's shape)"
        if (x["token"], x.get("bytes", [])) != (y["token"], y["bytes"]) or f32(x["logprob"]) != f32(y["logprob"]):
            return False, f"entry {i}: {x['token']!r} {x['logprob']!r} vs {y['token']!r} {y['logprob']!r}"
        tops = x.get("top_logprobs", [])
        if top == 0:
            if tops:
                return False, f"entry {i}: top_logprobs at top 0"
            continue
        if len(tops) != top or len(y["top_logprobs"]) < top:
            return False, f"entry {i}: {len(tops)} top tokens, wanted {top}"
        for j, (u, v) in enumerate(zip(tops, y["top_logprobs"][:top])):
            if (u["token"], u.get("bytes", [])) != (v["token"], v["bytes"]) or f32(u["logprob"]) != f32(v["logprob"]):
                return False, f"entry {i} top {j}: {u} vs {v}"
    return True, ""


def run_case(base, name, case, top, chat=True, top_from=None):
    """One recorded case through Ollama's API; `top_from` names the record whose `top` it is compared with."""
    req = case["request"]
    stream = bool(req.get("stream"))
    path = "/api/chat" if chat else "/api/generate"
    raw_post(base, "/api/generate", {"model": name, "keep_alive": 0})  # an empty slot, as the record started from
    code, text = raw_post(base, path, ollama_body(name, req, top, chat))
    if code != 200:
        return False, f"status {code}: {text[:200]}"
    if stream:
        lines = [json.loads(l) for l in text.splitlines() if l.strip()]
        final = lines[-1]
        got = [e for l in lines[:-1] for e in l.get("logprobs") or []]
        want = [e for ch in case["chunks"] for e in ch.get("logprobs") or []]
        arrays = [array_at(l) for l in text.splitlines()[:-1]]
        joined = "[" + ", ".join(a[1:-1] for a in arrays if a) + "]"
        content = "".join((l.get("message") or {}).get("content", "") if chat else l.get("response", "") for l in lines[:-1])
        want_content = "".join(ch["delta"].get("content") or "" for ch in case["chunks"])
        hashed = joined
    else:
        final = json.loads(text)
        got = final.get("logprobs") or []
        want = case["logprobs"]
        content = (final.get("message") or {}).get("content") if chat else final.get("response")
        want_content = case["content"]
        hashed = array_at(text)
    if content != want_content:
        return False, f"text {content!r} vs {want_content!r}"
    if final.get("done_reason") != case["finish_reason"]:
        return False, f"done_reason {final.get('done_reason')} vs {case['finish_reason']}"
    ok, why = same_entries(got, want, top)
    if not ok:
        return False, why
    r = final.get("bankml_receipt") or {}
    if r.get("logprobs_sha256") != sha(hashed or ""):
        return False, f"receipt logprobs_sha256 {r.get('logprobs_sha256')} is not the sha256 of what was received"
    return True, ""


def live(stem):
    binary = root / "target" / "release" / "bankml"
    model = root / ".models" / f"{stem}.gguf"
    fork = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks")) / f"{stem}.gguf.FORK.json"
    rec = {c["name"]: c for c in map(json.loads, (rec_dir / f"logprobs-{stem}.jsonl").read_text().splitlines())}
    base = "http://127.0.0.1:18187"
    name = stem.lower()
    proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18187",
                             "--upstream", "127.0.0.1:18188", "--ctx", "2048"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    try:
        wait(base, proc)
        top = lambda c: c["request"].get("top_logprobs", 20)
        for case in ["greedy-top5", "greedy-top1", "greedy-default20", "seeded-top5", "stop-string", "multibyte",
                     "stream-greedy-top3", "stream-seeded-top2", "stream-stop-string", "stream-multibyte", "stream-to-the-end"]:
            results.append((f"/api/chat {case}", *run_case(base, name, rec[case], top(rec[case]))))
        # Ollama's top_logprobs 0: the chosen token only, compared with the record's top-1 answer
        results.append(("/api/chat greedy-top1 as top 0", *run_case(base, name, rec["greedy-top1"], 0)))
        for case in ["greedy-top5", "multibyte", "stream-greedy-top3"]:
            results.append((f"/api/generate {case}", *run_case(base, name, rec[case], top(rec[case]), chat=False)))
        # Ollama's refusal, before any model is loaded
        code, text = raw_post(base, "/api/chat", {**ollama_body(name, rec["greedy-top5"]["request"], 21)})
        msg = (json.loads(text) if text.strip().startswith("{") else {}).get("error")
        results.append(("/api/chat top_logprobs 21 refused", code == 400 and msg == "top_logprobs must be between 0 and 20", f"{code} {msg!r}"))
        # without logprobs: no entries, no hash
        b = {**ollama_body(name, rec["greedy-top5"]["request"], 0)}
        b.pop("logprobs")
        code, text = raw_post(base, "/api/chat", b)
        j = json.loads(text) if code == 200 else {}
        results.append(("/api/chat without logprobs", code == 200 and "logprobs" not in j and "logprobs_sha256" not in (j.get("bankml_receipt") or {}),
                        f"{code} keys {sorted(j)}"))
        # /v1: the receipt's hash covers the content array as received (its values are logprobs_oracle.py's)
        for case in ["greedy-top5", "multibyte"]:
            raw_post(base, "/api/generate", {"model": name, "keep_alive": 0})
            code, text = raw_post(base, "/v1/chat/completions", {**rec[case]["request"], "cache_prompt": False})
            arr = array_at(text, '"logprobs": {"content": ') if code == 200 else None
            h = (json.loads(text).get("bankml_receipt") or {}).get("logprobs_sha256") if code == 200 else None
            results.append((f"/v1 {case} receipt", bool(arr) and h == sha(arr), f"{code} {h}"))
    finally:
        proc.terminate()
        proc.wait()
    ok = sum(1 for _, good, _ in results if good)
    for what, good, why in results:
        if not good:
            print(f"  {what}: {why}")
    print(f"logprobs on Ollama's API ({stem}): {ok} of {len(results)} through bankml serve --native carry llama-server b11192's "
          f"entries (text, bytes, every logprob the same float) in Ollama's shape, with receipts that hash what was received")
    return ok == len(results)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--bankml":
        sys.exit(__doc__)
    sys.exit(0 if live(sys.argv[2]) else 1)
