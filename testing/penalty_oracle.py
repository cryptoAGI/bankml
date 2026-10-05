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

Live (`--bankml STEM [NAME]`): bankml serve --native with the model on a spare port; every recorded case goes through
`/v1/chat/completions` (the chat the prompt was made from, the case's parameters as top-level fields) and every fourth
also through Ollama's `/api/chat` (as `options`), each from an empty slot. An answer must equal the record (the text,
the finish, the completion count, the end token included as llama-server counts it); a refused case must be a 400
carrying llama-server's message.
0.3.7: `--kind sampler` records and replays the rest of the default chain (typical-p, top-n-σ, XTC, dynamic temperature,
DRY) the same way → .models/oracle-forward/sampler-<model>.jsonl, replayed by `oracle_samplers*`.
usage: python3 testing/penalty_oracle.py [--kind penalty|sampler] STEM [N_PREDICT] | [--kind …] --bankml STEM [NAME]"""
import json, os, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import SAVANTE, wait  # noqa: E402  (Savante's system prompt and the readiness wait)

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-forward"
PORT = 18304

QUESTIONS = ["Repeat the word moon forty times.", "List the days of the week, then list them again, three times.",
             "Tell me a fact about the moon.", "Write a haiku about rain."]
PENALTY_VARIANTS = [
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
# 0.3.7: the rest of llama-server's default chain — typical-p, top-n-σ, XTC, dynamic temperature and DRY — alone and
# where they change each other's path (typical-p leaves the set unsorted for top-p and min-p; XTC draws from its own
# generator; DRY's breakers cap a repeat), seeded where the draw decides, and llama-server's refusals and clamps
SAMPLER_VARIANTS = [
    {"temperature": 0.8, "typical_p": 0.9},
    {"temperature": 1.0, "typical_p": 0.5, "top_p": 0.9},
    {"temperature": 1.0, "typical_p": 0.95, "top_p": 1.0, "min_p": 0.1},
    {"temperature": 0.0, "typical_p": 0.7},
    {"temperature": 1.0, "top_n_sigma": 1.0},
    {"temperature": 1.2, "top_n_sigma": 2.0, "top_k": 80},
    {"temperature": 0.0, "top_n_sigma": 0.5},
    {"temperature": 1.0, "xtc_probability": 0.5, "xtc_threshold": 0.1},
    {"temperature": 0.9, "xtc_probability": 1.0, "xtc_threshold": 0.05, "min_keep": 2},
    {"temperature": 1.0, "xtc_probability": 0.3, "xtc_threshold": 0.6},
    {"temperature": 1.0, "xtc_probability": 2.0, "xtc_threshold": 0.2},
    {"temperature": 0.8, "dynatemp_range": 0.5},
    {"temperature": 1.0, "dynatemp_range": 1.0, "dynatemp_exponent": 2.0},
    {"temperature": 0.0, "dynatemp_range": 0.3},
    {"temperature": 0.0, "dry_multiplier": 0.8},
    {"temperature": 0.0, "dry_multiplier": 1.0, "dry_base": 1.5, "dry_allowed_length": 1, "dry_penalty_last_n": 32},
    {"temperature": 0.7, "dry_multiplier": 0.8, "dry_sequence_breakers": ["\n", ".", " moon"]},
    {"temperature": 0.0, "dry_multiplier": 0.5, "dry_base": 0.5, "repeat_penalty": 1.1},
    {"temperature": 1.0, "typical_p": 0.9, "top_n_sigma": 1.5, "xtc_probability": 0.5, "dynatemp_range": 0.4, "dry_multiplier": 0.6},
    # limits: what the server does with them is the oracle
    {"temperature": 0.0, "dry_multiplier": 0.8, "dry_penalty_last_n": -1},
    {"temperature": 0.0, "dry_multiplier": 0.8, "dry_allowed_length": -1},
    {"temperature": 0.0, "dry_multiplier": 0.8, "dry_sequence_breakers": []},
    {"temperature": 0.0, "dry_multiplier": 0.8, "dry_sequence_breakers": "\n"},
]
SETS = {"penalty": PENALTY_VARIANTS, "sampler": SAMPLER_VARIANTS}
OLLAMA_OPTIONS = {"temperature", "top_k", "top_p", "min_p", "min_keep", "seed", "repeat_penalty", "repeat_last_n", "presence_penalty",
                  "frequency_penalty", "typical_p"}
KEYS = ("temperature", "top_k", "top_p", "min_p", "min_keep", "seed", "repeat_last_n", "repeat_penalty", "presence_penalty",
        "frequency_penalty", "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "dry_sequence_breakers",
        "typical_p", "xtc_probability", "xtc_threshold", "dynatemp_range", "dynatemp_exponent", "top_n_sigma", "samplers")


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


def live(stem, name, kind="penalty"):
    binary = root / "target" / "release" / "bankml"
    model = root / ".models" / f"{stem}.gguf"
    forks = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks"))
    fork = forks / f"{stem}.gguf.FORK.json"
    rec = out / f"{kind}-{stem}.jsonl"
    variants = SETS[kind]
    for f in (binary, model, fork, rec):
        if not f.exists():
            sys.exit(f"needs {f}")
    cases = [json.loads(line) for line in rec.read_text().splitlines()]
    base, name = "http://127.0.0.1:18198", name or stem.lower()
    proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18198",
                             "--upstream", "127.0.0.1:18201", "--ctx", "2048"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0, n, ok = time.time(), 0, 0
    try:
        wait(base, proc)

        def fresh():
            assert post(base, "/api/generate", {"model": name, "keep_alive": 0})[1]["done_reason"] == "unload"
            assert post(base, "/api/generate", {"model": name, "keep_alive": -1, "stream": False})[1]["done_reason"] == "load"

        for i, c in enumerate(cases):
            q = QUESTIONS[i // len(variants)]
            msgs = [{"role": "system", "content": SAVANTE}, {"role": "user", "content": q}]
            params = {k: v for k, v in c["request"].items() if k not in ("prompt", "n_predict", "cache_prompt", "return_tokens")}
            want_err = (c.get("error") or {}).get("message")
            if not want_err:
                want = (c["text"], "stop" if c["stop_type"] == "eos" else "length", len(c["ids"]))
            # Ollama's options carry only some samplers; a case with others goes through /v1 alone
            api_ok = set(params) <= OLLAMA_OPTIONS
            for path in (("v1", "api") if i % 4 == 0 and api_ok else ("v1",)):
                fresh()
                if path == "v1":
                    code, r = post(base, "/v1/chat/completions", {"model": name, "messages": msgs, "max_tokens": c["request"]["n_predict"], **params})
                    got = (r["choices"][0]["message"]["content"], r["choices"][0]["finish_reason"], r["usage"]["completion_tokens"]) if code == 200 else None
                else:
                    code, r = post(base, "/api/chat", {"model": name, "messages": msgs, "stream": False, "keep_alive": -1,
                                                       "options": params | {"num_predict": c["request"]["n_predict"]}})
                    got = (r["message"]["content"], r["done_reason"], r["eval_count"]) if code == 200 else None
                n += 1
                if want_err:
                    err = json.dumps(r)
                    good = code == 400 and want_err in err
                    if not good:
                        print(f"  case {i} via {path}: llama-server refused ({want_err}); bankML {code} {err[:160]}")
                else:
                    good = got == want
                    if not good:
                        print(f"  case {i} {params} via {path}: got {got!r:.160} want {want!r:.160}")
                ok += good
    finally:
        proc.terminate()
        proc.wait()
    print(f"{kind} live oracle ({stem}): {ok} of {n} answers through bankml serve --native (/v1 top-level fields; /api/chat options) "
          f"identical to llama-server b11192's record (text, finish, completion count; refusals with its message) — {time.time() - t0:.0f} s")
    return ok == n


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    args = sys.argv[1:]
    kind = "penalty"
    if "--kind" in args:
        i = args.index("--kind")
        kind = args[i + 1]
        del args[i:i + 2]
    if kind not in SETS:
        sys.exit(f"--kind {kind}: one of {', '.join(SETS)}")
    if args[0] == "--bankml":
        sys.exit(0 if live(args[1], args[2] if len(args) > 2 else None, kind) else 1)
    stem = args[0]
    n_predict = int(args[1]) if len(args) > 1 else 48
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
            for vi, v in enumerate(SETS[kind]):
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
        dest = out / f"{kind}-{stem}.jsonl"
        dest.write_text("".join(json.dumps(c) + "\n" for c in cases))
        print(f"{len(cases)} {kind} cases from llama-server b11192 ({stem}) → {dest}")
    finally:
        proc.terminate()
        proc.wait(timeout=60)


if __name__ == "__main__":
    main()
