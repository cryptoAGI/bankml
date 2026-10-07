#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""0.4.1: a native run stops when its client goes away (`Native::complete_while`, `serve::client_alive`).

Against a fresh `bankml serve --native` on a small model:
  1. a long prompt, the client closes after a second: serve logs that it stopped during the prompt, and records no
     answer in /bankml/metrics;
  2. the same prompt again, read to the end: it reuses what the abandoned run computed (`cache_n` > 0) and its answer
     is the answer an empty slot gives (asked once more with `cache_prompt: false`);
  3. a streamed answer of up to 400 tokens, the client closes after its first pieces: serve logs that it stopped
     during generation, long before 400 tokens;
  4. /api/generate (Ollama's shape), abandoned the same way: stopped as well.
    python3 testing/disconnect_oracle.py [MODEL_STEM]"""
import json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import wait  # noqa: E402

root = Path(__file__).resolve().parents[1]
PORT, UP = 18611, 18612
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def abandon(path, body, after):
    """Send a request and close the connection `after` seconds later, without reading the answer to the end."""
    s = socket.create_connection(("127.0.0.1", PORT))
    data = json.dumps(body).encode()
    s.sendall(f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data)
    time.sleep(after)
    s.close()


def log_until(log, needle, timeout):
    t = time.time()
    while time.time() - t < timeout:
        text = log.read_text(errors="replace")
        if needle in text:
            return text
        time.sleep(0.5)
    return log.read_text(errors="replace")


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "Bonsai-1.7B-Q1_0"
    fork = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks")) / f"{stem}.gguf.FORK.json"
    log = Path(os.environ.get("TMPDIR", "/tmp")) / f"bankml-disconnect-{os.getpid()}.log"
    proc = subprocess.Popen([str(root / "target/release/bankml"), "serve", str(root / ".models" / f"{stem}.gguf"), "--fork", str(fork), "--native",
                             "--listen", f"127.0.0.1:{PORT}", "--upstream", f"127.0.0.1:{UP}", "--ctx", "4096"],
                            stdout=open(log, "w"), stderr=subprocess.STDOUT, env={**os.environ, "BANKML_GPU": "off"})
    ok = n = 0

    def check(name, good, why=""):
        nonlocal ok, n
        n += 1
        ok += bool(good)
        print(("ok   " if good else "FAIL ") + name + ("" if good else f" — {why}"))

    try:
        wait(BASE, proc)
        # a prompt of about 1,500 tokens: three micro-batches, several seconds of prefill on the laptop
        long = " ".join(f"Fact {i}: the number {i} is followed by {i + 1}." for i in range(130))
        msgs = [{"role": "user", "content": long + " Which fact comes first? Answer in five words."}]
        before = len(get("/bankml/metrics").get("records") or [])
        t0 = time.time()
        abandon("/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "temperature": 0}, 1.0)
        text = log_until(log, "the client went away; stopped after", 120)
        check("1. an abandoned prompt stops during the prefill, and says how far it got",
              "the client went away; stopped after" in text and "prompt tokens (kept in the slot)" in text, text[-300:])
        time.sleep(1)
        check("1. no answer is recorded for it", len(get("/bankml/metrics").get("records") or []) == before)
        r = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "temperature": 0})
        cache_n = r["timings"]["cache_n"]
        check("2. the next request reuses what the abandoned run computed", cache_n > 0, f"cache_n {cache_n}")
        fresh = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "temperature": 0, "cache_prompt": False})
        check("2. and its answer is an empty slot's", r["choices"][0]["message"]["content"] == fresh["choices"][0]["message"]["content"],
              f'{r["choices"][0]["message"]["content"]!r} vs {fresh["choices"][0]["message"]["content"]!r}')
        recs = len(get("/bankml/metrics").get("records") or [])
        abandon("/v1/chat/completions", {"messages": [{"role": "user", "content": "Count from one to two hundred in words."}],
                                         "max_tokens": 400, "temperature": 0, "stream": True}, 4.0)
        # a stream notices either way: its next write fails (the run ends there, recorded) or the peek before a token
        # sees the closed socket (the run stops with GONE, logged); both must come long before 400 tokens
        t, k = time.time(), None
        while time.time() - t < 180 and k is None:
            text = log.read_text(errors="replace")
            gone = [l for l in text.splitlines() if "stopped after" in l and "generated tokens" in l]
            rs = get("/bankml/metrics").get("records") or []
            if gone:
                k = int(gone[-1].split("stopped after ")[1].split()[0])
            elif len(rs) > recs:
                k = rs[-1]["completion_tokens"]
            time.sleep(1)
        check("3. an abandoned stream stops during generation, long before its 400 tokens", k is not None and k < 300, f"stopped at {k}")
        mark = text.count("the client went away")
        abandon("/api/generate", {"model": stem.lower(), "prompt": long, "stream": False, "options": {"num_predict": 16}}, 1.0)
        t = time.time()
        while time.time() - t < 120 and log.read_text(errors="replace").count("the client went away") <= mark:
            time.sleep(0.5)
        check("4. /api/generate abandoned stops as well", log.read_text(errors="replace").count("the client went away") > mark)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    print(f"disconnect oracle ({stem}): {ok} of {n} checks — a run whose client goes away stops there, keeps what it computed, "
          f"and changes no answer")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
