#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""0.3.8: slot save and restore on `bankml serve --native` (llama-server's `POST /slots/{id}?action=save|restore|erase`
on the engine address, Savante's warm start). The oracle is the engine itself: an answer after a restore must be the
answer computed from an empty slot — the same text and counts — and the restore must actually skip the prompt
(`timings.cache_n` > 0); the same across a restart of the server. Refusals are llama-server's: 501 without a slot
directory, "Invalid slot ID", "Invalid action", "Invalid filename" (path traversal included), and a missing or
damaged file refused with "Unable to restore slot: …".
    python3 testing/slot_oracle.py [STEM]          (default Bonsai-1.7B-Q1_0)"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json_oracle import SAVANTE, wait  # noqa: E402

root = Path(__file__).resolve().parents[1]
stem = sys.argv[1] if len(sys.argv) > 1 else "Bonsai-1.7B-Q1_0"
binary = root / "target" / "release" / "bankml"
model = root / ".models" / f"{stem}.gguf"
fork = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks")) / f"{stem}.gguf.FORK.json"
GW, ENG = "http://127.0.0.1:18187", "http://127.0.0.1:18188"
n = ok = 0


def check(name, good, detail=""):
    global n, ok
    n += 1
    ok += bool(good)
    if not good:
        print(f"  FAIL {name} {detail}")


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def start(slot_dir):
    args = [str(binary), "serve", str(model), "--fork", str(fork), "--native", "--listen", GW[7:], "--upstream", ENG[7:], "--ctx", "2048"]
    if slot_dir:
        args += ["--slot-dir", str(slot_dir)]
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait(GW, p)
    return p


ASK = {"messages": [{"role": "system", "content": SAVANTE + " " + "Keep this context in mind: the moon is far away. " * 20},
                    {"role": "user", "content": "In one sentence, why does the moon look larger near the horizon?"}],
       "max_tokens": 40, "temperature": 0}


def answer():
    code, r = post(GW, "/v1/chat/completions", ASK)
    return (r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]), r["timings"]["cache_n"]


for f in (binary, model, fork):
    if not f.exists():
        sys.exit(f"needs {f}")
d = Path(tempfile.mkdtemp(prefix="bankml-slots-"))
t0 = time.time()
try:
    # without --slot-dir: llama-server's 501
    p = start(None)
    code, r = post(ENG, "/slots/0?action=save", {"filename": "x.bin"})
    check("no slot directory → 501 not_supported_error", code == 501 and r["error"]["type"] == "not_supported_error"
          and r["error"]["message"] == "This server does not support slots action. Start it with `--slot-save-path`", r)
    p.terminate(); p.wait()

    p = start(d)
    post(ENG, "/slots/0?action=erase", {})
    fresh, cache0 = answer()                                   # from an empty slot
    code, r = post(ENG, "/slots/0?action=save", {"filename": "warm.bin"})
    # the slot holds the prompt and every generated token but the last, which is sampled and never computed: llama-server
    # b11192 reports n_saved 341 for this request's 302 + 40 tokens, and so must bankML
    held = fresh[1] + fresh[2] - 1
    check("save: 200 with n_saved = the slot's tokens, as llama-server counts them", code == 200 and r["n_saved"] == held and r["n_written"] > 0 and "save_ms" in r["timings"], r)
    code, r = post(ENG, "/slots/0?action=erase", {})
    check("erase: 200 with n_erased", code == 200 and r["n_erased"] == held, r)
    code, r = post(ENG, "/slots/0?action=restore", {"filename": "warm.bin"})
    check("restore: 200 with n_restored", code == 200 and r["n_restored"] == held and r["n_read"] > 0, r)
    again, cache1 = answer()
    check("the answer after a restore is the answer from an empty slot", again == fresh, f"{again} vs {fresh}")
    check("the restore skipped the prompt (cache_n > 0)", cache1 > 0 and cache0 == 0, f"cache_n {cache0} then {cache1}")
    # refusals, as llama-server's
    for path, body, want in [("/slots/abc?action=save", {"filename": "a.bin"}, "Invalid slot ID"), ("/slots/1?action=save", {"filename": "a.bin"}, "Invalid slot ID"),
                             ("/slots/0?action=bogus", {"filename": "a.bin"}, "Invalid action"), ("/slots/0?action=save", {"filename": "../escape.bin"}, "Invalid filename"),
                             ("/slots/0?action=save", {"filename": "a/b.bin"}, "Invalid filename"), ("/slots/0?action=save", {"filename": "x."}, "Invalid filename"),
                             ("/slots/0?action=save", {}, "Invalid filename")]:
        code, r = post(ENG, path, body)
        check(f"{path} {body} → 400 {want}", code == 400 and r["error"]["message"] == want, r)
    check("nothing escaped the slot directory", not (d.parent / "escape.bin").exists())
    code, r = post(ENG, "/slots/0?action=restore", {"filename": "missing.bin"})
    check("a missing file is refused", code == 400 and r["error"]["message"].startswith("Unable to restore slot: "), r)
    bad = d / "damaged.bin"
    b = bytearray((d / "warm.bin").read_bytes())
    b[len(b) // 2] ^= 0xFF
    bad.write_bytes(bytes(b))
    code, r = post(ENG, "/slots/0?action=restore", {"filename": "damaged.bin"})
    check("a damaged file is refused (its sha256)", code == 400 and "damaged" in r["error"]["message"], r)
    after_bad, _ = answer()
    check("after a refused restore the engine still answers the same", after_bad == fresh, after_bad)
    p.terminate(); p.wait()

    # a new server: the warm start Savante uses
    p = start(d)
    code, r = post(ENG, "/slots/0?action=restore", {"filename": "warm.bin"})
    check("restore into a fresh server", code == 200, r)
    warm, cache2 = answer()
    check("after a restart and a restore the answer is the same, without recomputing the prompt", warm == fresh and cache2 > 0, f"{warm} cache_n {cache2}")
    p.terminate(); p.wait()
finally:
    shutil.rmtree(d, ignore_errors=True)
print(f"slot oracle ({stem}): {ok} of {n} checks — save, erase and restore as llama-server's slot API; answers after a restore "
      f"identical to an empty slot's, across a restart; refusals as llama-server's — {time.time() - t0:.0f} s")
sys.exit(0 if ok == n else 1)
