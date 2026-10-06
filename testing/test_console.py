#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The bankML console (sAGI/console.py) without an engine: its routes, its security (loopback Host, same-origin JSON
POSTs, CSP), the SELF block as the model reads it (every value with its unit, "not measured" for a null), the
persona's doctrine root, and the Infotags metadata (ERC-721 attributes, an RFC 6962 root over the log's lines).
run: python3 testing/test_console.py"""
import json, os, sys, tempfile, threading, urllib.error, urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="bankml-console-"))
os.environ.update(BANKML_UI_STATE=str(tmp), BANKML_SERVE_LISTEN="127.0.0.1:9")  # no engine: every reading is absent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "sAGI"))
import console as C  # noqa: E402
import savante as S  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    fails += not ok
    print(("ok   " if ok else "FAIL ") + name)


srv = ThreadingHTTPServer(("127.0.0.1", 0), C.H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"


def req(path, data=None, headers=None, method=None):
    r = urllib.request.Request(base + path, data, headers or {}, method=method)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


try:
    for p in ("/", "/app.js", "/style.css", "/vendor/d3.v7.min.js", "/vendor/d3.LICENSE"):
        code, body, h = req(p)
        check(f"GET {p} is served", code == 200 and len(body) > 0)
    code, _, h = req("/")
    check("the page carries a strict CSP (no network but itself)", "default-src 'none'" in h.get("Content-Security-Policy", ""))
    check("an unknown path is 404, never a file from the disk", req("/../README.md")[0] == 404 and req("/console.py")[0] == 404)
    code, body, _ = req("/api/state")
    st = json.loads(body)
    check("/api/state answers without an engine (serve null, persona present)", code == 200 and st["serve"] is None and st["persona"]["name"] == "bankML")
    check("the persona's doctrine root is computed", str(st["persona"]["doctrine_root"]).startswith("0x"))
    check("a foreign Host is refused", req("/api/state", headers={"Host": "evil.example"})[0] == 403)
    check("a cross-origin POST is refused", req("/api/resources", b"{}", {"Content-Type": "application/json", "Origin": "http://evil.example"})[0] == 403)
    check("a POST that is not JSON is refused", req("/api/ask", b"x=1", {"Content-Type": "application/x-www-form-urlencoded"})[0] == 400)
    check("an empty question is refused", req("/api/ask", b'{"message": " "}', {"Content-Type": "application/json"})[0] == 400)
    # SELF as the model reads it: units on every value, "not measured" for a null (a bare key was read as seconds)
    sb = C.self_block()
    t = C.self_text(sb)
    check("SELF without an engine says not measured, never a number", t.count("not measured") >= 8 and "tokens per second" not in t)
    t2 = C.self_text({**sb, "last_eval_tps": 8.56, "completion_tokens_total": 75, "package_watts": 12.25, "gpu_limit": 0.8})
    check("SELF names each value with its unit", "8.6 tokens per second" in t2 and "75" in t2 and "12.2 watts" in t2 and "80 percent" in t2)
    # Infotags: ERC-721 attributes and a Merkle root over the log's exact lines
    C.STATE.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"at": 1, "question": "a", "answer": "b"}), json.dumps({"at": 2, "question": "c", "answer": "d"})]
    C.LOG.write_text("\n".join(lines) + "\n")
    info = json.loads(req("/api/infotags")[1])
    root = S.merkle_root([S._leaf(l.encode()) for l in lines])
    attrs = {a["trait_type"]: a["value"] for a in info["attributes"]}
    check("Infotags: the exchanges' RFC 6962 root over the log's lines", info["bankml"]["exchanges_root"] == root and attrs["exchanges"] == 2)
    check("Infotags: ERC-721 shape (name, description, attributes) and CIDs", info["name"] and info["description"] and info["bankml"]["persona_cid"].startswith("b"))
    log = json.loads(req("/api/log")[1])
    check("/api/log returns the exchanges", len(log["exchanges"]) == 2)
    # public mode (a hosted demo): its one host name is served, read-only, and no visitor's question is shown
    C.PUBLIC = "demo.example"
    pub = {"Host": "demo.example"}
    code, body, _ = req("/api/state", headers=pub)
    check("public: the named host is served and says it is public", code == 200 and json.loads(body)["public"] is True)
    check("public: any other host is still refused", req("/api/state", headers={"Host": "evil.example"})[0] == 403)
    check("public: the resource controls are refused (read-only)",
          req("/api/resources", b'{"threads": 1, "ram_gb": 1, "gpu_limit": 0}', {**pub, "Content-Type": "application/json", "Origin": "https://demo.example"})[0] == 403)
    check("public: no exchange is shown, though a log exists", json.loads(req("/api/log", headers=pub)[1])["exchanges"] == []
          and {a["trait_type"]: a["value"] for a in json.loads(req("/api/infotags", headers=pub)[1])["attributes"]}["exchanges"] == 0)
    C.PUBLIC = None
finally:
    srv.shutdown()

print("all ok" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
