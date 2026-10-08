#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The bankML console (sAGI/console.py) without an engine: its routes, its security (loopback Host, same-origin JSON
POSTs, CSP), the SELF block as the model reads it (every value with its unit, "not measured" for a null), the
persona's doctrine root, and the Infotags metadata (ERC-721 attributes, an RFC 6962 root over the log's lines).
run: python3 testing/test_console.py"""
import json, os, sys, tempfile, threading, time, urllib.error, urllib.request
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
    for p in ("/", "/app.js", "/style.css", "/vendor/d3.v7.min.js", "/vendor/d3.LICENSE", "/engine.js", "/engine.css", "/thesis.css"):
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
    # the first answer: the persona's prefix first and unchanged, SELF last, so the engine's prompt cache keeps the rest
    m1 = C.messages("PERSONA", [], "q1", "self one")
    m2 = C.messages("PERSONA", [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}], "q2", "self two")
    check("the prompt: persona first, SELF as a system message just before the question",
          [m["role"] for m in m1] == ["system", "system", "user"] and m1[0]["content"] == "PERSONA" and m1[1]["content"].startswith(C.SELF_HEAD)
          and m1[-1]["content"] == "q1")
    check("the prompt: the persona and the conversation so far are the same prefix turn after turn (SELF never in it)",
          m2[0] == m1[0] and m2[1:3] == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}] and "self" not in m2[0]["content"])
    convo = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(40)]
    starts = [C.window(convo[:n])[0]["content"] if C.window(convo[:n]) else None for n in range(2, 41, 2)]
    check("the window keeps at least 12 messages and starts on a question", all(len(C.window(convo[:n])) >= min(n, 12) and int(C.window(convo[:n])[0]["content"]) % 2 == 0 for n in range(2, 41, 2)))
    check("the window's start moves once every four exchanges, not every turn", len(set(starts)) <= 1 + (40 - 12) // 8 and starts[:6] == ["0"] * 6)
    check("a max_tokens that is not a number is refused (400), not a dropped connection",
          req("/api/ask", b'{"message": "hi", "max_tokens": "lots"}', {"Content-Type": "application/json"})[0] == 400)
    check("/api/state says what the warmer did", "warm" in json.loads(req("/api/state")[1]))
    # the warmer against a fake engine: restore a saved slot, else prefill the persona alone and save its slot; never on an
    # engine that has answered; once per engine; a question closes a prefill in progress
    import http.server as hs
    seen, calls = [], []
    class Fake(hs.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if seen[-1].get("hang"):
                time.sleep(5)
            b = b'{"choices": [{"message": {"content": "x"}}]}'
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    fake = hs.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    engine = {"verified": {"model_sha256": "ab" * 32}, "hashed_at": 1}
    metrics = {"records": []}
    real = (C._get, C.S._slot_call, C.models.LISTEN, C.models.SLOTS, C.S.engine_ctx)
    C._get = lambda path, timeout=5.0: engine if path == "/bankml" else metrics if path == "/bankml/metrics" else None
    C.S._slot_call = lambda action, name: calls.append((action, name)) or {}
    C.models.LISTEN = f"127.0.0.1:{fake.server_address[1]}"
    C.models.SLOTS = tmp / "slots"
    C.models.SLOTS.mkdir(parents=True, exist_ok=True)
    C.S.engine_ctx = lambda: 2048
    try:
        C.WARM.update(key=None, state="idle", conn=None)
        st1 = C.warm()
        check("warm: a fresh engine with no saved slot is prefilled with the persona alone (one token), then its slot saved",
              st1 == "prefilled and saved" and len(seen) == 1 and seen[0]["messages"] == [{"role": "system", "content": C.persona()["system_prompt"]}]
              and seen[0]["max_tokens"] == 1 and calls == [("save", C._slot_key(engine)[1])])
        check("warm: once per engine (nothing sent again)", C.warm() == st1 and len(seen) == 1)
        (C.models.SLOTS / C._slot_key(engine)[1]).write_bytes(b"kv")
        engine["hashed_at"] = 2; calls.clear()
        check("warm: a restarted engine restores the saved slot, no prefill", C.warm() == "restored" and calls[0][0] == "restore" and len(seen) == 1)
        engine["hashed_at"] = 3; metrics["records"] = [{"ttft_ms": 1}]; calls.clear()
        check("warm: an engine that has answered is left alone (its slot is a conversation)", C.warm() == "engine in use" and not calls and len(seen) == 1)
        # a question during a prefill closes it
        engine["hashed_at"] = 4; metrics["records"] = []; (C.models.SLOTS / C._slot_key(engine)[1]).unlink()
        real_persona = C.persona
        C.persona = lambda: {**real_persona(), "hang": True}
        orig_dumps = C.json.dumps
        C.json.dumps = lambda o, *a, **k: orig_dumps({**o, "hang": True} if isinstance(o, dict) and "messages" in o else o, *a, **k)
        out = {}
        th = threading.Thread(target=lambda: out.update(s=C.warm()), daemon=True); th.start()
        for _ in range(50):
            if C.WARM.get("conn") is not None and getattr(C.WARM["conn"], "sock", None) is not None:
                break
            time.sleep(0.05)
        t0 = time.time(); C.cancel_warm(); th.join(10)
        C.json.dumps, C.persona = orig_dumps, real_persona
        check("warm: a question closes a prefill in progress at once", out.get("s") == "cancelled by a question" and time.time() - t0 < 3)
        C.cancel_warm()  # nothing in progress: no error
    finally:
        C._get, C.S._slot_call, C.models.LISTEN, C.models.SLOTS, C.S.engine_ctx = real
        fake.shutdown()
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
    th = json.loads(req("/api/thesis")[1])
    check("/api/thesis: the Thesis section of docs/TECHNICAL.md, whole, with its anchor",
          th["markdown"].startswith("## Thesis — Professor Codephreak and Gregory L. Magnusson") and "### Contributions" in th["markdown"]
          and "## I. The problem" not in th["markdown"] and th["url"].endswith("#thesis--professor-codephreak-and-gregory-l-magnusson"))
    code, body, _ = req("/api/engine")
    check("/api/engine without an engine is a 502 that says where it looked", code == 502 and "127.0.0.1:9" in json.loads(body)["error"])
    check("the Engine tab draws with serve's own renderer", req("/engine.js")[1] == (ROOT / "bankML" / "status.js").read_bytes())
    # Diagnostics: spans after LlamaIndex's SimpleSpanHandler, and measured checks
    import diagnostics as D  # noqa: E402
    h = D.SpanHandler(keep=3)
    D.HANDLER, saved = h, D.HANDLER
    with D.span("outer", n=1) as o:
        with D.span("inner") as i:
            i.event("first piece")
        try:
            with D.span("broken"):
                raise ValueError("no engine")
        except ValueError:
            pass
    trees = h.trees()
    t0 = trees[0]
    check("diagnostics: a span nests under the current one, children in start order",
          t0["name"] == "outer" and [c["name"] for c in t0["children"]] == ["inner", "broken"] and o.duration >= i.duration)
    check("diagnostics: a span left by an exception is kept with its error (span_drop)", t0["children"][1]["error"] == "ValueError: no engine")
    check("diagnostics: events carry their time inside the span", t0["children"][0]["events"][0]["name"] == "first piece")
    orphan = D.Span(id_="b-1", name="b", parent_id="gone-0")  # its parent was never kept (or kept no longer)
    h.enter(orphan); h.exit("b-1")
    stand = [t for t in h.trees() if t["id"] == "gone-0-MISSING"]
    check("diagnostics: a span whose parent is not kept hangs under a MISSING stand-in, not lost",
          len(stand) == 1 and stand[0]["children"][0]["id"] == "b-1" and len(h.done) == 3)
    check("diagnostics: the trees render as text with durations", "outer (" in D.render(trees) and "└── broken" in D.render(trees))
    D.HANDLER = saved
    secret = "what is my private question"
    req("/api/ask", json.dumps({"message": secret}).encode(), {"Content-Type": "application/json"})
    dg = json.loads(req("/api/diagnostics")[1])
    ask = dg["traces"][0]
    kids = [c["name"] for c in ask["children"]]
    check("/api/diagnostics: an answer is traced span by span", ask["name"] == "ask" and kids[:2] == ["self_block", "engine.stream"] and "receipt.verify" in kids)
    check("/api/diagnostics: the engine's failure is on its span", bool(ask["children"][1]["error"]) and ask["tags"]["ok"] is False)
    check("/api/diagnostics: no question text in any trace", secret not in json.dumps(dg))
    check("/api/diagnostics: without an engine the first check is bad, and says where it looked",
          dg["checks"][0]["level"] == "bad" and "127.0.0.1:9" in dg["checks"][0]["seen"])
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
