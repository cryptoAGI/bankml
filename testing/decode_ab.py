#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""0.3.9: bankML against llama-server b11192, decode speed, in pairs. Each round starts a fresh llama-server and a
fresh `bankml serve --native` (alternating which goes first), sends both the same greedy request with the prompt
cache off, and reads each side's own `timings` (prompt and predicted tokens per second, llama-server's fields, which
bankML reports since 0.3.7). The answers must be token-identical, or the speeds are not comparable and the round is
refused. Run it on a quiet machine, pinned, so both get the same cores and memory:
    BANKML_GGML_LIB=<b11192 release dir> BANKML_PIN_CPUS=1,2,3 BANKML_PIN_MEM=3000M \\
        testing/pinned.sh python3 testing/decode_ab.py Bonsai-8B-Q1_0 [rounds] [threads]
Prints every round and the medians; the ratio is bankML ÷ llama-server (above 1: bankML faster)."""
import json, os, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import SAVANTE, wait  # noqa: E402

root = Path(__file__).resolve().parents[1]
PROMPT = "Explain in a short paragraph why the sky is blue, and what changes at sunset."


def ask(base, max_tokens):
    body = {"messages": [{"role": "system", "content": SAVANTE}, {"role": "user", "content": PROMPT}], "max_tokens": max_tokens,
            "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    t = d["timings"]
    return d["choices"][0]["message"]["content"], t["prompt_per_second"], t["predicted_per_second"], t["predicted_n"]


def llama(stem, threads, max_tokens):
    lib = Path(os.environ["BANKML_GGML_LIB"])
    base = "http://127.0.0.1:18321"
    p = subprocess.Popen([str(lib / "llama-server"), "-m", str(root / ".models" / f"{stem}.gguf"), "--host", "127.0.0.1", "--port", "18321",
                          "-t", str(threads), "-c", "2048", "-np", "1", "--jinja", "--reasoning", "off", "--no-webui"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(base, p)
        ask(base, 4)  # warm the pages: both sides read a memory-mapped file
        return ask(base, max_tokens)
    finally:
        p.terminate()
        p.wait(timeout=60)


def bankml(stem, threads, max_tokens):
    fork = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks")) / f"{stem}.gguf.FORK.json"
    base = "http://127.0.0.1:18323"
    env = {**os.environ, "BANKML_THREADS": str(threads), "BANKML_GPU": os.environ.get("BANKML_GPU", "off")}
    p = subprocess.Popen([str(root / "target" / "release" / "bankml"), "serve", str(root / ".models" / f"{stem}.gguf"), "--fork", str(fork),
                          "--native", "--listen", "127.0.0.1:18323", "--upstream", "127.0.0.1:18324", "--ctx", "2048"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        wait(base, p)
        ask(base, 4)
        return ask(base, max_tokens)
    finally:
        p.terminate()
        p.wait(timeout=60)


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "Bonsai-8B-Q1_0"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    max_tokens = 64
    print(f"# {stem}, {rounds} rounds, {threads} threads, {max_tokens} tokens greedy; load {open('/proc/loadavg').read().split()[:3]}")
    rows = []
    for r in range(rounds):
        order = [("llama-server", llama), ("bankML", bankml)] if r % 2 == 0 else [("bankML", bankml), ("llama-server", llama)]
        got = {}
        for name, run in order:
            got[name] = run(stem, threads, max_tokens)
            time.sleep(2)
        (lt, lp, ld, ln), (bt, bp, bd, bn) = got["llama-server"], got["bankML"]
        if (lt, ln) != (bt, bn):
            sys.exit(f"round {r + 1}: the answers differ, the speeds are not comparable\n  bankML {bt!r}\n  llama  {lt!r}")
        rows.append((bp, bd, lp, ld))
        print(f"round {r + 1} ({order[0][0]} first): decode bankML {bd:.2f} tok/s, llama-server {ld:.2f} ({bd / ld:.3f}×); "
              f"prompt {bp:.1f} against {lp:.1f} tok/s; {bn} tokens, identical", flush=True)
    med = lambda i: statistics.median(x[i] for x in rows)
    print(f"median: decode bankML {med(1):.2f} tok/s, llama-server {med(3):.2f} ({med(1) / med(3):.3f}×); "
          f"prompt {med(0):.1f} against {med(2):.1f} tok/s ({med(0) / med(2):.3f}×)")


if __name__ == "__main__":
    main()
