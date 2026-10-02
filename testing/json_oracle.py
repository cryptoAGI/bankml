#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The JSON-mode oracle (0.3.3): llama-server b11192's own answers under a grammar, token by token, for bankML to equal.

Record (once per model): launches llama-server from BANKML_GGML_LIB with Savante's flags (`-t 3 -c 2048 -np 1 --jinja
--reasoning off`, plus `--verbose` so each answer carries its tokens and the grammar it ran) on a spare port and sends
`/v1/chat/completions` requests with `response_format: {"type": "json_object"}` — prompts that invite JSON, prompts
that tempt prose or code (so the grammar has to reject what the model wants), a long nested object, unicode — greedy
and seeded at several temperatures; on the 1-bit model also two user `grammar` (GBNF) requests. Every request is sent
with `cache_prompt: false`, so each answer starts from an empty cache. Each record keeps the request, the tokens
(with the end token), the raw text, the message content, the finish reason, the counts, the grammar and the
generation prompt the server used.
    BANKML_GGML_LIB=<b11192 release dir> python3 testing/json_oracle.py --record Bonsai-8B-Q1_0
    → .models/oracle-json/json-<model>.jsonl
`oracle_json_mode` / `oracle_json_mode_ternary` (cargo, --ignored) replay the records through bankML's engine.

Live (`--bankml`): bankml serve --native with the 1-bit model on spare ports; a subset of the records goes through
`/v1/chat/completions` (`response_format`, streamed once) and through Ollama's `/api/chat` with `format: "json"`
(the Ollama shape is mapped onto the same request), each from an empty slot; every answer must equal the record."""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-json"

SAVANTE = ("You are Savante, an autonomous research agent. You answer plainly and briefly, you say when you do not know, "
           "and you never invent sources.")
PROMPTS = [
    ("cat", [{"role": "user", "content": "Give me a JSON object describing a cat: its name, age and colour."}], 64),
    ("haiku", [{"role": "user", "content": "Write a haiku about autumn leaves."}], 48),
    ("yes", [{"role": "user", "content": "Reply with just the word yes, nothing else."}], 32),
    ("planets", [{"role": "system", "content": SAVANTE}, {"role": "user", "content": "List three planets and one fact about each."}], 96),
    ("library", [{"role": "user", "content": "Describe a small library as JSON: its name, its address (street and city), and a list of three "
                                             "books, each with a title, an author and a year."}], 200),
    ("code", [{"role": "user", "content": "Explain recursion in one sentence, then show a short Python example."}], 80),
    ("greet", [{"role": "user", "content": "How do you say 'good morning' in French, Japanese and Arabic?"}], 64),
]
SEEDED = [{"temperature": 0.7, "seed": 11}, {"temperature": 1.0, "seed": 22}, {"temperature": 1.3, "top_k": 40, "seed": 33}]
GRAMMARS = [("yesno", 'root ::= ("yes" | "no") "." [ ]? [A-Za-z ]{0,40} "."\n', [{"role": "user", "content": "Is the sun a star?"}]),
            ("list", 'root ::= ("- " [a-z][a-z ]{1,30} "\\n"){2,3}\n', [{"role": "user", "content": "Name some fruits."}])]


def post(base, path, body, timeout=7200):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait(base, proc):
    for _ in range(900):
        try:
            urllib.request.urlopen(base + "/health", timeout=2).read()
            return
        except Exception:
            if proc.poll() is not None:
                sys.exit("the server exited")
            time.sleep(1)
    sys.exit("the server never became healthy")


def requests(stem):
    ternary = "Ternary" in stem
    reqs = []
    for i, (name, msgs, n) in enumerate(PROMPTS):
        reqs.append((f"{name}-greedy", {"messages": msgs, "response_format": {"type": "json_object"}, "temperature": 0.0, "max_tokens": n}))
        if i in ((0, 1, 3) if ternary else (0, 1, 3, 5)):
            for v in (SEEDED[:2] if ternary else SEEDED):
                reqs.append((f"{name}-t{v['temperature']}-s{v['seed']}", {"messages": msgs, "response_format": {"type": "json_object"}, "max_tokens": n, **v}))
    if not ternary:
        for name, g, msgs in GRAMMARS:
            reqs.append((f"gbnf-{name}-greedy", {"messages": msgs, "grammar": g, "temperature": 0.0, "max_tokens": 48}))
            reqs.append((f"gbnf-{name}-t1.0-s7", {"messages": msgs, "grammar": g, "temperature": 1.0, "seed": 7, "max_tokens": 48}))
    return reqs


def record(stem):
    lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
    model = root / ".models" / f"{stem}.gguf"
    if not (lib / "llama-server").exists() or not model.exists():
        sys.exit("needs BANKML_GGML_LIB (b11192 release dir) and " + str(model))
    base = "http://127.0.0.1:18302"
    proc = subprocess.Popen([str(lib / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", "18302", "-t", "3", "-c", "2048",
                             "-np", "1", "--jinja", "--reasoning", "off", "--no-webui", "--verbose"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(base, proc)
        cases = []
        for name, body in requests(stem):
            t0 = time.time()
            r = post(base, "/v1/chat/completions", {**body, "cache_prompt": False, "return_tokens": True})
            v = r["__verbose"]
            g = v["generation_settings"]
            cases.append({"name": name, "request": body, "tokens": v["tokens"], "raw": v["content"], "content": r["choices"][0]["message"]["content"],
                          "finish_reason": r["choices"][0]["finish_reason"], "prompt_tokens": r["usage"]["prompt_tokens"],
                          "completion_tokens": r["usage"]["completion_tokens"], "grammar": g["grammar"], "grammar_lazy": g["grammar_lazy"],
                          "generation_prompt": g["generation_prompt"], "seed": g["seed"], "top_k": g["top_k"], "top_p": g["top_p"],
                          "min_p": g["min_p"], "temperature": g["temperature"]})
            print(f"{name}: {len(v['tokens'])} tokens, {r['choices'][0]['finish_reason']}, {time.time() - t0:.0f} s: {r['choices'][0]['message']['content'][:70]!r}",
                  flush=True)
    finally:
        proc.terminate()
        proc.wait()
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"json-{stem}.jsonl"
    dest.write_text("".join(json.dumps(c) + "\n" for c in cases))
    print(f"{len(cases)} constrained answers from llama-server b11192 ({stem}) → {dest}")


def bankml():
    stem = "Bonsai-8B-Q1_0"
    model = root / ".models" / f"{stem}.gguf"
    fork = Path(os.environ.get("BANKML_FORK", Path.home() / ".local/share/bankml/forks/Bonsai-8B-Q1_0.gguf.FORK.json"))
    rec = out / f"json-{stem}.jsonl"
    binary = root / "target" / "release" / "bankml"
    missing = [str(p) for p in (model, fork, rec, binary) if not p.exists()]
    if missing:
        print("json live oracle skipped: missing " + ", ".join(missing))
        return 0
    cases = {c["name"]: c for c in map(json.loads, rec.read_text().splitlines())}
    pick = [cases[n] for n in ("cat-greedy", "haiku-t0.7-s11", "yes-greedy", "gbnf-yesno-t1.0-s7")]
    base, name = "http://127.0.0.1:18197", "bonsai-8b-q1_0"
    proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18197",
                             "--upstream", "127.0.0.1:18198", "--ctx", "2048"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0, n, ok = time.time(), 0, 0
    try:
        wait(base, proc)

        def fresh():
            assert post(base, "/api/generate", {"model": name, "keep_alive": 0})["done_reason"] == "unload"
            assert post(base, "/api/generate", {"model": name, "keep_alive": -1, "stream": False})["done_reason"] == "load"

        for c in pick:
            body = c["request"]
            paths = [("v1", False), ("v1", True)] if c is pick[0] else [("v1", False)]
            if "response_format" in body:
                paths.append(("api", False))
            for path, stream in paths:
                fresh()
                if path == "v1":
                    if stream:
                        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps({**body, "stream": True}).encode(), {"Content-Type": "application/json"})
                        text, fin, usage = "", None, None
                        with urllib.request.urlopen(req, timeout=7200) as r:
                            for line in r.read().decode().splitlines():
                                if line.startswith("data: {") and '"choices"' in line:
                                    d = json.loads(line[6:])
                                    text += d["choices"][0]["delta"].get("content", "")
                                    fin = d["choices"][0]["finish_reason"] or fin
                                    usage = d.get("usage") or usage
                        got = (text, fin, usage["prompt_tokens"], usage["completion_tokens"])
                    else:
                        r = post(base, "/v1/chat/completions", body)
                        got = (r["choices"][0]["message"]["content"], r["choices"][0]["finish_reason"], r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"])
                else:
                    o = {"model": name, "messages": body["messages"], "format": "json", "stream": False, "keep_alive": -1,
                         "options": {k: body[k] for k in ("temperature", "seed", "top_k") if k in body} | {"num_predict": body["max_tokens"]}}
                    r = post(base, "/api/chat", o)
                    got = (r["message"]["content"], r["done_reason"], r["prompt_eval_count"], r["eval_count"])
                want = (c["content"], c["finish_reason"], c["prompt_tokens"], c["completion_tokens"])
                n += 1
                ok += got == want
                if got != want:
                    print(f"  {c['name']} via {path}{' (stream)' if stream else ''}: got {got!r:.160} want {want!r:.160}")
    finally:
        proc.terminate()
        proc.wait()
    print(f"json live oracle: {ok} of {n} answers through bankml serve --native (/v1 response_format and grammar, streamed once; /api/chat "
          f"format \"json\") identical to llama-server b11192's record (content, finish, prompt/completion counts) — {time.time() - t0:.0f} s")
    return 0 if ok == n else 1


if __name__ == "__main__":
    if "--record" in sys.argv:
        record(sys.argv[sys.argv.index("--record") + 1])
    elif "--bankml" in sys.argv:
        sys.exit(bankml())
    else:
        sys.exit(__doc__)
