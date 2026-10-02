#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The C API's oracles (0.3.2), driven from C programs compiled with the system `cc` against target/release/libbankml.so
(build it first: cargo build --release -p bankml-capi).

    python3 testing/capi/capi_oracle.py --printf   bankml_log (a C-variadic function defined in Rust) == libc snprintf,
                                                   byte for byte, on every supported conversion; the rest marked
    python3 testing/capi/capi_oracle.py --chat     bankml_chat == bankml serve --native == llama-server b11192:
      - Ternary-Bonsai-8B-Q2_0_g64: serve_oracle.py's three Savante conversations (temperature 0.3, a seed per turn)
        through `bankml serve --native` /v1/chat/completions, then the very same request bytes through bankml_chat in
        a fresh process: text, streamed pieces, prompt/completion counts, prompt-cache reuse, finish reason and the
        receipt's response_sha256, request_sha256 and model_sha256 must all be equal, turn by turn;
      - Bonsai-8B-Q1_0: the recorded llama-server b11192 conversations (serve-Bonsai-8B-Q1_0.jsonl, which
        serve_oracle.py --bankml ties to serve --native) through bankml_chat: the same text and counts, and
        response_sha256 = sha256 of llama-server's text; since 0.3.4 the same for the models O4 opened, each against
        llama-server running it: Bonsai-1.7B-Q1_0 (tied embeddings), SmolLM2-135M-Instruct-F16 and mindx-gen39-F16
        (the Llama graph in F16);
      - refusals: Qwen3-0.6B-Q8_0 (Q8_0 weights, O3) is refused with serve's reason, and a file its FORK.json does not
        pin is refused; the library's own messages reach the sink bankml_set_log installed.
Models in .models/, pins in $BANKML_FORKS (default ~/.local/share/bankml/forks). A missing model skips its case."""
import hashlib, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[2]
lib = root / "target" / "release"
build = root / "target" / "capi"
models = root / ".models"
forks = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks"))

# as testing/serve_oracle.py
SYSTEM = ("You are Savante, an autonomous research agent. You answer plainly and briefly, you say when you do not know, "
          "and you never invent sources.")
conversations = [
    ["What is a bonsai?", "How often should I water one?", "And in winter?"],
    ["Name a prime number.", "Is 91 prime?", "Why not?"],
    ["Say hello in French.", "Now in Spanish.", "Which of the two did you find easier to write, and why?"],
]


def cc(src, out, extra=()):
    build.mkdir(parents=True, exist_ok=True)
    cmd = ["cc", "-O2", "-Wall", "-Wextra", "-Werror", *extra, str(src), "-I", str(root / "capi" / "include"),
           "-L", str(lib), "-lbankml", f"-Wl,-rpath,{lib}", "-o", str(out)]
    subprocess.run(cmd, check=True)
    return out


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pinned(fork_path, name):
    return next(f["sha256"] for f in json.loads(Path(fork_path).read_text())["files"] if f["path"].rsplit("/", 1)[-1] == name)


class Chat:
    """The C program as a line server: one request per line in, one result per line out."""

    def __init__(self, exe, model, fork, n_ctx=2048):
        self.p = subprocess.Popen([str(exe), str(model), str(fork), str(n_ctx)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        self.opened = json.loads(self.p.stdout.readline())

    def ask(self, body: bytes):
        self.p.stdin.write(body + b"\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        _, err = self.p.communicate(timeout=120)  # closes stdin: the program reads EOF and closes its handle
        return self.p.returncode, err.decode(errors="replace")


def same_turn(i, a, b, keys):
    bad = [k for k in keys if a.get(k) != b.get(k)]
    if bad:
        print(f"  turn {i}: differs in {', '.join(bad)}")
        for k in bad:
            print(f"    {k}: {a.get(k)!r}\n    {' ' * len(k)}  {b.get(k)!r}")
    return not bad


def facts(result, streamed=None):
    """The comparable facts of a /v1/chat/completions object (and, for bankml_chat, what was streamed)."""
    r = result
    f = {"text": r["choices"][0]["message"]["content"], "finish": r["choices"][0]["finish_reason"],
         "prompt": r["usage"]["prompt_tokens"], "completion": r["usage"]["completion_tokens"], "cache_n": r["timings"]["cache_n"]}
    rc = r.get("bankml_receipt", {})
    for k in ("response_sha256", "request_sha256", "model_sha256", "guard", "prompt_tokens", "completion_tokens"):
        f["receipt." + k] = rc.get(k)
    if streamed is not None:
        f["streamed"] = streamed
    return f


def printf_case():
    exe = cc(root / "testing" / "capi" / "printf_oracle.c", build / "printf_oracle", ["-Wno-format"])
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    return r.returncode


def chat_case():
    exe = cc(root / "testing" / "capi" / "chat.c", build / "chat")
    failed, t0 = 0, time.time()

    # refusals, before any model is loaded
    small, small_fork = models / "Qwen3-0.6B-Q8_0.gguf", forks / "Qwen3-0.6B-Q8_0.gguf.FORK.json"
    if small.exists() and small_fork.exists():
        c = Chat(exe, small, small_fork)
        rc, err = c.close()
        ok = c.opened.get("open") is False and rc == 2 and "native forward pass does not play it: weights are Q8_0" in c.opened.get("error", "") \
            and "[bankml log 0] bankml_open: refuse:" in err
        failed += not ok
        print(f"capi refusal: Qwen3-0.6B-Q8_0 {'refused' if ok else 'NOT refused as serve refuses it'}: {c.opened.get('error')}")
    q1, q1_fork = models / "Bonsai-8B-Q1_0.gguf", forks / "Bonsai-8B-Q1_0.gguf.FORK.json"
    tern, tern_fork = models / "Ternary-Bonsai-8B-Q2_0_g64.gguf", forks / "Ternary-Bonsai-8B-gguf-fork.FORK.json"
    if q1.exists() and tern_fork.exists():
        c = Chat(exe, q1, tern_fork)
        c.close()
        ok = c.opened.get("open") is False and "unpinned, refused" in c.opened.get("error", "")
        failed += not ok
        print(f"capi refusal: a file its FORK.json does not pin {'refused' if ok else 'NOT refused'}: {c.opened.get('error')}")

    # Ternary-Bonsai-8B: bankml serve --native, then the same bytes through bankml_chat
    binary = root / "target" / "release" / "bankml"
    if tern.exists() and tern_fork.exists() and binary.exists():
        base = "http://127.0.0.1:18197"
        proc = subprocess.Popen([str(binary), "serve", str(tern), "--fork", str(tern_fork), "--native", "--listen", "127.0.0.1:18197",
                                 "--upstream", "127.0.0.1:18198", "--ctx", "2048"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        served = []
        try:
            for _ in range(600):
                try:
                    urllib.request.urlopen(base + "/health", timeout=2).read()
                    break
                except Exception:
                    if proc.poll() is not None:
                        print("FAILED: bankml serve --native exited"); return 1
                    time.sleep(1)
            seed = 3000
            for conv in conversations:
                messages = [{"role": "system", "content": SYSTEM}]
                for q in conv:
                    messages.append({"role": "user", "content": q})
                    body = json.dumps({"messages": list(messages), "temperature": 0.3, "seed": seed, "max_tokens": 64, "stream": False}).encode()
                    seed += 1
                    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=3600) as r:
                        res = json.load(r)
                    served.append((body, res))
                    messages.append({"role": "assistant", "content": res["choices"][0]["message"]["content"]})
        finally:
            proc.terminate()
            proc.wait(timeout=60)
        t_serve = time.time() - t0
        c = Chat(exe, tern, tern_fork)
        if not c.opened.get("open"):
            print(f"FAILED: bankml_open refused Ternary-Bonsai-8B: {c.opened.get('error')}"); return 1
        ok = 0
        for i, (body, res) in enumerate(served):
            got = c.ask(body)
            want = facts(res)
            want["streamed"] = want["text"]
            have = facts(got["result"], got["streamed"]) if got["rc"] == 0 else {"rc": got["rc"], "result": got["result"]}
            good = got["rc"] == 0 and got["pieces"] > 0 and same_turn(i, have, want, list(want)) and want["receipt.request_sha256"] == sha(body) \
                and want["receipt.response_sha256"] == sha(want["text"].encode()) and want["receipt.model_sha256"] == pinned(tern_fork, tern.name)
            ok += good
        rc, err = c.close()
        logged = f"[bankml log 2] bankml: {tern.name} verified (sha256 {pinned(tern_fork, tern.name)})" in err \
            and "[bankml log 2] chat oracle: libbankml" in err
        failed += (ok != len(served)) + (not logged) + (rc != 0)
        if not logged:
            print("  the C program's stderr (the sink) was:\n" + "".join("    " + x + "\n" for x in err.splitlines()[:20]))
        print(f"capi chat oracle: Ternary-Bonsai-8B-Q2_0_g64: {ok} of {len(served)} turns through bankml_chat identical to bankml serve "
              f"--native's /v1/chat/completions (text, streamed pieces, prompt/completion counts, cache reuse, finish, receipt "
              f"response/request/model sha256); library log via bankml_set_log: {'yes' if logged else 'NO'} — serve {t_serve:.0f} s, "
              f"C {time.time() - t0 - t_serve:.0f} s")
    else:
        print("capi chat oracle: Ternary-Bonsai-8B skipped: model, FORK.json or target/release/bankml missing")

    # llama-server b11192's recorded conversations, each model against the server that ran it
    for stem in ("Bonsai-8B-Q1_0", "Bonsai-1.7B-Q1_0", "SmolLM2-135M-Instruct-F16", "mindx-gen39-F16"):
        failed += recorded_case(exe, stem)
    return 1 if failed else 0


def recorded_case(exe, stem):
    q1, q1_fork = models / f"{stem}.gguf", forks / f"{stem}.gguf.FORK.json"
    record = models / "oracle-forward" / f"serve-{stem}.jsonl"
    failed = 0
    if q1.exists() and q1_fork.exists() and record.exists():
        t1 = time.time()
        c = Chat(exe, q1, q1_fork)
        if not c.opened.get("open"):
            print(f"FAILED: bankml_open refused {stem}: {c.opened.get('error')}"); return 1
        turns = [json.loads(x) for x in record.read_text().splitlines()]
        ok = 0
        for i, t in enumerate(turns):
            body = json.dumps(t["request"]).encode()
            got = c.ask(body)
            want = {"text": t["text"], "finish": t["finish_reason"], "prompt": t["prompt_tokens"], "completion": t["completion_tokens"],
                    "cache_n": t["cache_n"], "streamed": t["text"], "receipt.response_sha256": sha(t["text"].encode()),
                    "receipt.request_sha256": sha(body), "receipt.model_sha256": pinned(q1_fork, q1.name), "receipt.guard": "play",
                    "receipt.prompt_tokens": t["prompt_tokens"], "receipt.completion_tokens": t["completion_tokens"]}
            have = facts(got["result"], got["streamed"]) if got["rc"] == 0 else {"rc": got["rc"]}
            ok += got["rc"] == 0 and same_turn(i, have, want, list(want))
        rc, _ = c.close()
        failed += (ok != len(turns)) + (rc != 0)
        print(f"capi chat oracle: {stem}: {ok} of {len(turns)} turns through bankml_chat identical to llama-server b11192's "
              f"record (text, streamed pieces, counts, cache reuse, finish; response_sha256 = sha256 of llama-server's text) — "
              f"{time.time() - t1:.0f} s")
    else:
        print(f"capi chat oracle: {stem} skipped: model, FORK.json or its serve record missing")
    return failed


if __name__ == "__main__":
    if "--printf" in sys.argv:
        sys.exit(printf_case())
    if "--chat" in sys.argv:
        sys.exit(chat_case())
    print(__doc__)
    sys.exit(1)
