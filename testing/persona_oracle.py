#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The O5 end-to-end oracle (0.3.5): mindX's own model with its persona, created by `bankml create` exactly as mindX's
`promote.py` layers it on Ollama, must answer as llama-server b11192 answers the same GGUF given the same system
message and stops.

The layer is promote.py's `persona_modelfile_layer`, byte for byte — `FROM <model>`, `SYSTEM` with the persona in triple quotes,
`PARAMETER stop <|im_end|>`, `PARAMETER num_ctx 2048` — written by mindX itself (the persona is mindX's text, so it is
read from that file, never copied into this repository). Two creates from it, each in its own scratch registry under
.models/oracle-persona/:
  - `dir`: FROM the merged safetensors directory mindXtrain writes (PYTHAI/mindXascension weights/gen39/ollama_push/
    merged @ 4bd31b9d): `bankml create` converts it (the GGUF's sha256 must be the pin's, 6b64c748…, so the converter
    is checked again on the downloaded directory) and layers the persona — what `promote.py --to bankml` would run;
  - `alias`: promote.py's own Modelfile, unchanged (`FROM mindx-gen39`, re-created in place under the same name) over
    the pinned mindx-gen39-F16.gguf — Ollama's shape.

Record (`--record MODELFILE MERGED_DIR`): the two creates, then llama-server b11192 (Savante's flags) on the GGUF, asked
the conversations with the persona as the system message and `stop: ["<|im_end|>"]`, greedy and seeded, each from an
empty cache: tokens, text, finish, counts → .models/oracle-persona/persona-mindx-gen39.jsonl. `oracle_persona_layer`
(cargo, --ignored) replays them through each manifest's layer applied to the user's turns alone.

Live (`--bankml`): `bankml serve --native --registry <scratch>` asked for `mindx-gen39` (the derived model) with the
user's turns only, through Ollama's `/api/chat`, `/api/generate` (single turns) and `/v1/chat/completions`, each from
an empty slot (the record has no prompt cache: a cached prefix changes F16's product paths, so seeded answers move); every
answer must equal the record, `/api/ps` must name `mindx-gen39:latest`, and `/api/show` must give the layer back."""
import hashlib, json, os, re, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-persona"
binary = root / "target" / "release" / "bankml"
PIN = "6b64c748d96ad26fd72402299bd27b2ae82f489bd0469498dd18eb6054058266"
GGUF = "mindx-gen39-F16.gguf"
CONVERSATIONS = [
    ["Who are you?"],
    ["What is a Gödel machine?"],
    ["What does mindX mean by augmented intelligence?"],
    ["Say hello in one sentence."],
    ["Name three things you do every day.", "Which of those matters most, and why?"],
    ["What is a belief in a BDI agent?", "And an intention?", "Give an example of each."],
]
SEEDED = [{"temperature": 0.7, "seed": 11}, {"temperature": 1.0, "seed": 22}]


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait(base, proc):
    for _ in range(600):
        try:
            urllib.request.urlopen(base + "/health", timeout=2).read()
            return
        except Exception:
            if proc.poll() is not None:
                sys.exit("the server exited")
            time.sleep(1)
    sys.exit("the server never became healthy")


def layer_of(modelfile):
    """promote.py's Modelfile: (FROM, the persona, the stops, num_ctx) — read back for the llama-server side"""
    t = Path(modelfile).read_text()
    m = re.search(r'(?s)^SYSTEM """(.*?)"""\n', t, re.M)
    stops = re.findall(r"(?m)^PARAMETER stop (.+)$", t)
    ctx = re.search(r"(?m)^PARAMETER num_ctx (\d+)$", t)
    frm = re.search(r"(?m)^FROM (.+)$", t).group(1)
    assert m and stops and ctx, "not promote.py's persona Modelfile"
    return t, frm, m.group(1), stops, int(ctx.group(1))


def requests():
    reqs = []
    for ci, conv in enumerate(CONVERSATIONS):
        for v in [{"temperature": 0.0}] + SEEDED:
            reqs.append((ci, v))
    return reqs


def create(variant, text, frm_line, models=None):
    reg = out / f"registry-{variant}"
    shutil.rmtree(reg, ignore_errors=True)
    reg.mkdir(parents=True)
    mf = reg / "Modelfile"
    mf.write_text(re.sub(r"(?m)^FROM .+$", lambda _: f"FROM {frm_line}", text, count=1))
    cmd = [str(binary), "create", "mindx-gen39", "-f", str(mf), "--registry", str(reg)] + (["--models", str(models)] if models else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(f"bankml create ({variant}): exit {p.returncode}\n  " + "\n  ".join((p.stdout + p.stderr).strip().splitlines()))
    if p.returncode != 0:
        sys.exit(f"bankml create ({variant}) failed")
    return reg


def record(modelfile, merged):
    lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
    if not (lib / "llama-server").exists() or not binary.exists():
        sys.exit("needs BANKML_GGML_LIB (b11192 release dir) and target/release/bankml")
    text, frm, persona, stops, n_ctx = layer_of(modelfile)
    # dir: FROM the merged safetensors directory (converted, pinned, layered)
    reg = create("dir", text, str(Path(merged).resolve()))
    sha = hashlib.sha256((reg / GGUF).read_bytes()).hexdigest()
    print(f"converted {GGUF}: sha256 {sha} — {'the pin' if sha == PIN else 'NOT the pin ' + PIN}")
    if sha != PIN:
        sys.exit(1)
    # alias: promote.py's Modelfile unchanged (FROM mindx-gen39), over the pinned file and its FORK.json
    forks = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks"))
    areg = out / "registry-alias"
    shutil.rmtree(areg, ignore_errors=True)
    areg.mkdir(parents=True)
    shutil.copy(forks / f"{GGUF}.FORK.json", areg)
    mf = areg / "Modelfile"
    mf.write_text(text)
    p = subprocess.run([str(binary), "create", "mindx-gen39", "-f", str(mf), "--registry", str(areg), "--models", str(root / ".models")],
                       capture_output=True, text=True)
    print(f"bankml create (alias, promote.py's Modelfile as written, FROM {frm}): exit {p.returncode}\n  " + "\n  ".join((p.stdout + p.stderr).strip().splitlines()))
    if p.returncode != 0:
        sys.exit(1)
    model = reg / GGUF
    base = "http://127.0.0.1:18305"
    proc = subprocess.Popen([str(lib / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", "18305", "-t", "3", "-c", str(n_ctx),
                             "-np", "1", "--jinja", "--reasoning", "off", "--no-webui", "--verbose"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cases = []
    try:
        wait(base, proc)
        for ci, v in requests():
            msgs = [{"role": "system", "content": persona}]
            tag = "greedy" if v["temperature"] == 0.0 else f"t{v['temperature']}-s{v['seed']}"
            for ti, q in enumerate(CONVERSATIONS[ci]):
                msgs.append({"role": "user", "content": q})
                body = {"messages": list(msgs), "max_tokens": 96, "stop": stops, **v}
                r = post(base, "/v1/chat/completions", {**body, "cache_prompt": False, "return_tokens": True})
                vb = r["__verbose"]
                content = r["choices"][0]["message"]["content"]
                cases.append({"name": f"c{ci}-t{ti}-{tag}", "conversation": ci, "turn": ti, "request": body,
                              "user_messages": [m for m in msgs if m["role"] != "system"], "tokens": vb["tokens"], "text": content,
                              "finish_reason": r["choices"][0]["finish_reason"], "prompt_tokens": r["usage"]["prompt_tokens"],
                              "completion_tokens": r["usage"]["completion_tokens"]})
                msgs.append({"role": "assistant", "content": content})
                print(f"{cases[-1]['name']}: {len(vb['tokens'])} tokens, {cases[-1]['finish_reason']}: {content[:60]!r}", flush=True)
    finally:
        proc.terminate()
        proc.wait()
    dest = out / "persona-mindx-gen39.jsonl"
    dest.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    print(f"{len(cases)} answers from llama-server b11192 with promote.py's persona as the system message → {dest}")


def bankml():
    rec = out / "persona-mindx-gen39.jsonl"
    missing = [str(p) for p in (rec, out / "registry-dir" / GGUF, binary) if not p.exists()]
    if missing:
        print("persona live oracle skipped: missing " + ", ".join(missing))
        return 0
    cases = [json.loads(x) for x in rec.read_text().splitlines()]
    t0, n, ok = time.time(), 0, 0
    for variant in ("dir", "alias"):
        reg = out / f"registry-{variant}"
        model = reg / GGUF if variant == "dir" else root / ".models" / GGUF
        base = "http://127.0.0.1:18306"
        proc = subprocess.Popen([str(binary), "serve", str(model), "--fork", str(reg / f"{GGUF}.FORK.json"), "--native", "--registry", str(reg),
                                 "--listen", "127.0.0.1:18306", "--upstream", "127.0.0.1:18307", "--ctx", "2048"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait(base, proc)
            tags = [m["name"] for m in json.load(urllib.request.urlopen(base + "/api/tags"))["models"]]
            show = post(base, "/api/show", {"model": "mindx-gen39"})
            assert "mindx-gen39:latest" in tags, tags
            assert show.get("system") == cases[0]["request"]["messages"][0]["content"], "the layer's SYSTEM through /api/show"
            for c in cases:
                body, want = c["request"], (c["text"], c["finish_reason"], c["prompt_tokens"], c["completion_tokens"])
                opts = {k: body[k] for k in ("temperature", "seed") if k in body} | {"num_predict": body["max_tokens"]}
                paths = ["api", "v1"] + (["generate"] if c["turn"] == 0 else [])
                for path in paths:
                    # from an empty cache, as the record was made (cache_prompt false): a cached prefix changes the
                    # micro-batches of the prompt, hence F16's product path and the logits' last bits
                    assert post(base, "/api/generate", {"model": "mindx-gen39", "keep_alive": 0})["done_reason"] == "unload"
                    if path == "api":
                        r = post(base, "/api/chat", {"model": "mindx-gen39", "messages": c["user_messages"], "stream": False, "options": opts})
                        got = (r["message"]["content"], r["done_reason"], r["prompt_eval_count"], r["eval_count"])
                    elif path == "generate":
                        r = post(base, "/api/generate", {"model": "mindx-gen39", "prompt": c["user_messages"][0]["content"], "stream": False, "options": opts})
                        got = (r["response"], r["done_reason"], r["prompt_eval_count"], r["eval_count"])
                    else:
                        b = {"model": "mindx-gen39", "messages": c["user_messages"], "max_tokens": body["max_tokens"]} | {k: body[k] for k in ("temperature", "seed") if k in body}
                        r = post(base, "/v1/chat/completions", b)
                        got = (r["choices"][0]["message"]["content"], r["choices"][0]["finish_reason"], r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"])
                    n += 1
                    ok += got == want
                    if got != want:
                        print(f"  {variant} {c['name']} via {path}: got {got!r:.200}\n    want {want!r:.200}")
            ps = json.load(urllib.request.urlopen(base + "/api/ps"))["models"]
            assert ps and ps[0]["name"] == "mindx-gen39:latest" and ps[0]["context_length"] == 2048, f"/api/ps names {ps}"
        finally:
            proc.terminate()
            proc.wait()
    print(f"persona live oracle: {ok} of {n} answers from the created mindx-gen39 (FROM the merged directory, and promote.py's Modelfile "
          f"FROM mindx-gen39 in place) through /api/chat, /api/generate and /v1 — the user's turns only, the persona from the layer — "
          f"identical to llama-server b11192 given the persona as the system message (text, finish, prompt and completion counts); "
          f"/api/ps names mindx-gen39:latest — {time.time() - t0:.0f} s")
    return 0 if ok == n else 1


if __name__ == "__main__":
    if "--record" in sys.argv:
        i = sys.argv.index("--record")
        record(sys.argv[i + 1], sys.argv[i + 2])
    elif "--bankml" in sys.argv:
        sys.exit(bankml())
    else:
        sys.exit(__doc__)
