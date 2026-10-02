#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The JSON-schema oracle (O6b): llama-server b11192's own answers under JSON schemas, token by token, for bankML to
equal. The grammar text itself is checked without a model by `testing/schema_oracle.py` (`oracle_schema_grammars`);
this checks the whole path — the grammar, its prefill, the redraw, the end set and the content rule — on real answers.

Record (once per model): launches llama-server from BANKML_GGML_LIB with Savante's flags (`-t 3 -c 2048 -np 1 --jinja
--reasoning off --verbose`) on a spare port and sends `/v1/chat/completions` requests carrying schemas the way mindX
and OpenAI clients send them — `response_format: {"type": "json_schema", "json_schema": {"name", "strict", "schema"}}`,
the top-level `json_schema` field, and `response_format: {"type": "json_object", "schema": …}` — over schemas with
required and optional fields, enums, integer ranges, nested `$defs`/`$ref`, `anyOf` with null, a pattern and a date
format, string lengths, a top-level array, a top-level enum and a top-level integer; greedy and seeded. Each request
is sent from an empty cache (`cache_prompt: false`) and recorded with its exact text (`request_text`, so a float
literal stays one), the tokens with the end token, the raw text, the content, the finish reason, the counts, the
grammar and the generation prompt the server used.
    BANKML_GGML_LIB=<b11192 release dir> python3 testing/json_schema_oracle.py --record Bonsai-8B-Q1_0
    BANKML_GGML_LIB=<b11192 release dir> python3 testing/json_schema_oracle.py --record Ternary-Bonsai-8B-Q2_0_g64
    → .models/oracle-json/schema-<model>.jsonl, replayed by `oracle_json_schema` / `oracle_json_schema_ternary`
      (cargo test --release -- --ignored)

Live (`--bankml`): bankml serve --native with the 1-bit model on spare ports; a subset of the records goes through
`/v1/chat/completions` (each request shape, one streamed) and through Ollama's `/api/chat` with `format: <schema>`,
each from an empty slot; every answer must equal the record (content, finish, prompt and completion counts)."""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import SAVANTE, SEEDED, wait  # noqa: E402  (the same server flags, seeds and readiness wait)

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-json"

CAT = {"type": "object", "properties": {"name": {"type": "string", "maxLength": 24}, "age": {"type": "integer", "minimum": 0, "maximum": 30},
       "colour": {"enum": ["black", "white", "ginger", "grey", "tabby"]}, "indoor": {"type": "boolean"}},
       "required": ["name", "age", "colour"], "additionalProperties": False}
VERDICT = {"type": "object", "properties": {"verdict": {"enum": ["verified", "unverified", "refuted", "unknown"]}, "claim": {"type": "string"},
           "score": {"type": "integer", "minimum": 0, "maximum": 100}, "sources": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
           "required": ["verdict", "claim", "score", "sources"], "additionalProperties": False}
LIBRARY = {"$defs": {"Book": {"type": "object", "properties": {"title": {"type": "string"}, "author": {"type": "string"},
                              "year": {"type": "integer", "minimum": 1000, "maximum": 2100}}, "required": ["title", "author", "year"]},
                     "Address": {"type": "object", "properties": {"street": {"type": "string"}, "city": {"type": "string"}}, "required": ["street", "city"]}},
           "type": "object", "properties": {"name": {"type": "string"}, "address": {"$ref": "#/$defs/Address"},
                                            "books": {"type": "array", "items": {"$ref": "#/$defs/Book"}, "minItems": 3, "maxItems": 3}},
           "required": ["name", "address", "books"]}
PLAN = {"title": "Plan", "type": "object", "properties": {
            "goal": {"title": "Goal", "type": "string"},
            "steps": {"title": "Steps", "type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "object", "properties": {
                "action": {"enum": ["search", "write", "ask_user", "done"]}, "why": {"type": "string"}}, "required": ["action", "why"]}},
            "priority": {"title": "Priority", "type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            "owner": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "kind": {"const": "plan"}},
        "required": ["goal", "steps", "kind"]}
PLANETS = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "properties": {"planet": {"type": "string"}, "fact": {"type": "string"}},
           "required": ["planet", "fact"], "additionalProperties": False}}
YESNO = {"enum": ["yes", "no"]}
APOLLO = {"type": "object", "properties": {"date": {"type": "string", "format": "date"}, "code": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{2}$"},
          "crew": {"type": "array", "items": {"type": "string", "minLength": 2}, "minItems": 1, "maxItems": 3}}, "required": ["date", "code", "crew"]}
NUMBER = {"type": "integer", "minimum": 1, "maximum": 100}
HAIKU = {"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "string", "maxLength": 40}, "minItems": 3, "maxItems": 3},
         "score": {"enum": [0.5, 1.0, 2]}}, "required": ["lines"]}


def openai_rf(schema, name):
    return {"response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}}


u = lambda t: [{"role": "user", "content": t}]
# (name, messages, max_tokens, how the schema is sent)
CASES = [
    ("cat", u("Give me a JSON object describing a cat: its name, age and colour."), 64, openai_rf(CAT, "cat")),
    ("verdict", [{"role": "system", "content": SAVANTE}, {"role": "user", "content": "Is the Great Wall of China visible from the Moon with the naked eye?"}],
     96, {"json_schema": VERDICT}),
    ("library", u("Describe a small library as JSON: its name, its address (street and city), and a list of three books, each with a title, "
                  "an author and a year."), 200, openai_rf(LIBRARY, "library")),
    ("plan", u("Plan how to write a short report on bees."), 128, {"response_format": {"type": "json_object", "schema": PLAN}}),
    ("planets", u("List three planets and one fact about each."), 96, openai_rf(PLANETS, "planets")),
    ("yesno", u("Is the sun a star?"), 16, {"json_schema": YESNO}),
    ("apollo", u("When did Apollo 11 land on the Moon? Give the date, a mission code and the crew."), 80, openai_rf(APOLLO, "apollo")),
    ("number", u("Pick a number between 1 and 100."), 16, {"response_format": {"type": "json_object", "schema": NUMBER}}),
    ("haiku", u("Write a haiku about autumn leaves."), 64, {"json_schema": HAIKU}),
]


def requests(stem):
    ternary = "Ternary" in stem
    reqs = []
    for i, (name, msgs, n, how) in enumerate(CASES):
        if ternary and name in ("library", "plan", "apollo", "haiku"):
            continue
        reqs.append((f"{name}-greedy", {"messages": msgs, **how, "temperature": 0.0, "max_tokens": n}))
        if name in (("cat", "yesno") if ternary else ("cat", "verdict", "planets", "yesno", "haiku")):
            for v in (SEEDED[:1] if ternary else SEEDED[:2]):
                reqs.append((f"{name}-t{v['temperature']}-s{v['seed']}", {"messages": msgs, **how, "max_tokens": n, **v}))
    return reqs


def post_text(base, path, text, timeout=7200):
    req = urllib.request.Request(base + path, text.encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def record(stem):
    lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
    model = root / ".models" / f"{stem}.gguf"
    if not (lib / "llama-server").exists() or not model.exists():
        sys.exit("needs BANKML_GGML_LIB (b11192 release dir) and " + str(model))
    base = "http://127.0.0.1:18303"
    proc = subprocess.Popen([str(lib / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", "18303", "-t", "3", "-c", "2048",
                             "-np", "1", "--jinja", "--reasoning", "off", "--no-webui", "--verbose"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(base, proc)
        cases = []
        for name, body in requests(stem):
            t0 = time.time()
            text = json.dumps(body)
            r = post_text(base, "/v1/chat/completions", json.dumps({**body, "cache_prompt": False, "return_tokens": True}))
            v = r["__verbose"]
            g = v["generation_settings"]
            cases.append({"name": name, "request": body, "request_text": text, "tokens": v["tokens"], "raw": v["content"],
                          "content": r["choices"][0]["message"]["content"], "finish_reason": r["choices"][0]["finish_reason"],
                          "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
                          "grammar": g["grammar"], "grammar_lazy": g["grammar_lazy"], "generation_prompt": g["generation_prompt"],
                          "seed": g["seed"], "top_k": g["top_k"], "top_p": g["top_p"], "min_p": g["min_p"], "temperature": g["temperature"]})
            print(f"{name}: {len(v['tokens'])} tokens, {r['choices'][0]['finish_reason']}, {time.time() - t0:.0f} s: "
                  f"{r['choices'][0]['message']['content'][:70]!r}", flush=True)
    finally:
        proc.terminate()
        proc.wait()
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"schema-{stem}.jsonl"
    dest.write_text("".join(json.dumps(c) + "\n" for c in cases))
    print(f"{len(cases)} schema-constrained answers from llama-server b11192 ({stem}) → {dest}")


def schema_of(body):
    if "json_schema" in body:
        return body["json_schema"]
    rf = body["response_format"]
    return rf["json_schema"]["schema"] if rf["type"] == "json_schema" else rf["schema"]


def bankml():
    stem = "Bonsai-8B-Q1_0"
    model = root / ".models" / f"{stem}.gguf"
    fork = Path(os.environ.get("BANKML_FORK", Path.home() / ".local/share/bankml/forks/Bonsai-8B-Q1_0.gguf.FORK.json"))
    rec = out / f"schema-{stem}.jsonl"
    binary = root / "target" / "release" / "bankml"
    missing = [str(p) for p in (model, fork, rec, binary) if not p.exists()]
    if missing:
        print("json schema live oracle skipped: missing " + ", ".join(missing))
        return 0
    cases = {c["name"]: c for c in map(json.loads, rec.read_text().splitlines())}
    # one per request shape, a seeded one, the scalar contents; the first is also streamed and sent through /api/chat
    pick = [cases[n] for n in ("cat-greedy", "verdict-t0.7-s11", "plan-greedy", "yesno-greedy", "number-greedy", "haiku-t1.0-s22")]
    base, name = "http://127.0.0.1:18199", "bonsai-8b-q1_0"
    proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18199",
                             "--upstream", "127.0.0.1:18200", "--ctx", "2048"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0, n, ok = time.time(), 0, 0
    try:
        wait(base, proc)

        def fresh():
            assert post_text(base, "/api/generate", json.dumps({"model": name, "keep_alive": 0}))["done_reason"] == "unload"
            assert post_text(base, "/api/generate", json.dumps({"model": name, "keep_alive": -1, "stream": False}))["done_reason"] == "load"

        for c in pick:
            body, text = c["request"], c["request_text"]
            paths = [("v1", False), ("v1", True), ("api", False)] if c is pick[0] else [("v1", False), ("api", False)]
            for path, stream in paths:
                fresh()
                if path == "v1" and stream:
                    req = urllib.request.Request(base + "/v1/chat/completions", (text[:-1] + ', "stream": true}').encode(), {"Content-Type": "application/json"})
                    acc, fin, usage = "", None, None
                    with urllib.request.urlopen(req, timeout=7200) as r:
                        for line in r.read().decode().splitlines():
                            if line.startswith("data: {") and '"choices"' in line:
                                d = json.loads(line[6:])
                                acc += d["choices"][0]["delta"].get("content", "")
                                fin = d["choices"][0]["finish_reason"] or fin
                                usage = d.get("usage") or usage
                    got = (acc, fin, usage["prompt_tokens"], usage["completion_tokens"])
                elif path == "v1":
                    r = post_text(base, "/v1/chat/completions", text)
                    got = (r["choices"][0]["message"]["content"], r["choices"][0]["finish_reason"], r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"])
                else:
                    o = {"model": name, "messages": body["messages"], "format": schema_of(body), "stream": False, "keep_alive": -1,
                         "options": {k: body[k] for k in ("temperature", "seed", "top_k") if k in body} | {"num_predict": body["max_tokens"]}}
                    r = post_text(base, "/api/chat", json.dumps(o))
                    got = (r["message"]["content"], r["done_reason"], r["prompt_eval_count"], r["eval_count"])
                want = (c["content"], c["finish_reason"], c["prompt_tokens"], c["completion_tokens"])
                n += 1
                ok += got == want
                if got != want:
                    print(f"  {c['name']} via {path}{' (stream)' if stream else ''}: got {got!r:.160} want {want!r:.160}")
    finally:
        proc.terminate()
        proc.wait()
    print(f"json schema live oracle: {ok} of {n} answers through bankml serve --native (/v1 response_format json_schema, top-level "
          f"json_schema and json_object with a schema, one streamed; /api/chat format <schema>) identical to llama-server b11192's record "
          f"(content, finish, prompt/completion counts) — {time.time() - t0:.0f} s")
    return 0 if ok == n else 1


if __name__ == "__main__":
    if "--record" in sys.argv:
        record(sys.argv[sys.argv.index("--record") + 1])
    elif "--bankml" in sys.argv:
        sys.exit(bankml())
    else:
        sys.exit(__doc__)
