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
              st1 == "prefilled and saved" and len(seen) == 1 and seen[0]["messages"] == [{"role": "system", "content": C.base_system()}]
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
    check("/api/diagnostics: an answer is traced span by span (SELF, the window's memory, the engine)",
          ask["name"] == "ask" and kids[:2] == ["self_block", "memory"] and "engine.stream" in kids and "receipt.verify" in kids)
    check("/api/diagnostics: the engine's failure is on its span", bool(ask["children"][kids.index("engine.stream")]["error"]) and ask["tags"]["ok"] is False)
    check("/api/diagnostics: no question text in any trace", secret not in json.dumps(dg))
    check("/api/diagnostics: without an engine the first check is bad, and says where it looked",
          dg["checks"][0]["level"] == "bad" and "127.0.0.1:9" in dg["checks"][0]["seen"])
    # the machine, measured (sAGI/sysdiag.py): fake /sys trees for the thresholds, then this machine through the console
    import sysdiag as SD
    hw = tmp / "hwmon"
    for i, (name, mc) in enumerate([("k10temp", 96000), ("amdgpu", 70000), ("nvme", 41000), ("BAT0", None)]):
        (hw / f"hwmon{i}").mkdir(parents=True)
        (hw / f"hwmon{i}" / "name").write_text(name + "\n")
        if mc is not None:
            (hw / f"hwmon{i}" / "temp1_input").write_text(f"{mc}\n")
    tt = SD.temps(str(hw))
    check("sysdiag: temperatures by sensor, a sensor without one skipped", tt == {"k10temp": 96.0, "amdgpu": 70.0, "nvme": 41.0})
    check("sysdiag: a CPU at 96 °C is bad (throttling), at 70 °C ok",
          SD.cpu(tt, 50.0)["level"] == "bad" and "throttling" in SD.cpu(tt, 50.0)["lines"][3] and SD.cpu({"k10temp": 70.0}, None)["level"] == "ok")
    drm = tmp / "drm"
    (drm / "card0" / "device").mkdir(parents=True); (drm / "card0-eDP-1" / "device").mkdir(parents=True)
    for f, v in (("vendor", "0x1002"), ("device", "0x15d8"), ("gpu_busy_percent", "37"), ("mem_info_vram_used", "500000000"), ("mem_info_vram_total", "2000000000")):
        (drm / "card0" / "device" / f).write_text(v + "\n")
    g = SD.gpu(tt, str(drm))
    check("sysdiag: a card's busy share and memory; a connector is not a card",
          len(g["data"]["cards"]) == 1 and g["data"]["cards"][0]["busy_percent"] == 37 and "AMD" in g["lines"][0] and "VRAM 0.50 GB of 2.00 GB" in g["lines"][0])
    check("sysdiag: no card says so", "no card found" in SD.gpu({}, str(tmp / "nodrm"))["lines"][-1])
    dk = SD.disk({"here": str(tmp)}, {"nvme": 41.0})
    check("sysdiag: free space of each path, the drive's temperature", "here" in dk["data"] and "drive temperature 41.0" in dk["lines"][-1])
    check("sysdiag: a path that cannot be read says so", "not read" in SD.disk({"gone": str(tmp / "nope")}, {})["lines"][0])
    C.SAVANTE_URL = "http://127.0.0.1:9/"
    st = json.loads(req("/api/state")[1])
    check("the switch: where Savante lives, and that it does not answer (so the page marks it, not a dead link)",
          st["savante"] == {"url": "http://127.0.0.1:9/", "up": False} and C.answers(base + "/") is True)
    pg = json.loads(req("/api/ping")[1])
    check("/api/ping: one timed round trip to the engine, and why it failed when it did", pg["engine_ok"] is False and pg["engine_ms"] >= 0 and pg["why"])
    sd = json.loads(req("/api/sysdiag")[1])
    check("/api/sysdiag: CPU, memory, disk, GPU and the engine, each with a level and lines",
          [x["title"] for x in sd["sections"]] == ["CPU", "Memory", "Disk", "GPU", "Engine"]
          and all(x["level"] in ("ok", "warn", "bad") and x["lines"] for x in sd["sections"]) and sd["sections"][-1]["level"] == "bad")
    # .context: bankML's own codebase (tools/context.py) — the summary in the cached prefix, passages per question
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import context as CX  # noqa: E402
    ctxf = C.context()
    check(".context: generated, with a summary and chunks that each name their GitHub and Hugging Face source",
          ctxf.get("summary") and len(ctxf.get("chunks") or []) > 20
          and all(c["github"].startswith("https://github.com/cryptoAGI/bankml/blob/main/") and c["huggingface"].startswith("https://huggingface.co/spaces/PYTHAI/bankml/blob/main/") for c in ctxf["chunks"]))
    check(".context: fresh — every chunk's source unchanged since it was built (else: python3 tools/context.py)", CX.stale(ctxf) == [])
    check(".context: the summary follows the persona in the first system message (the cached prefix)",
          C.base_system().startswith(C.persona()["system_prompt"]) and ctxf["summary"] in C.base_system())
    t1, ids1 = C.context_passages("how does the sampler draw tokens, and what are the penalties?", 2)
    check(".context: a question about a module brings that module's passage, with its sources", "sampler" in ids1 and "github.com" in t1 and "huggingface.co" in t1)
    check(".context: an unrelated question brings none, and k 0 none", C.context_passages("a recipe for banana bread", 2) == ("", []) and C.context_passages("sampler", 0) == ("", []))
    # .memory per response window, the collection, review of .history, recall (sAGI/console_memory.py)
    CM = C.CM
    check("memory: a window's name from its title (survives a reload; the field's ids do not)",
          CM.slug("Output 1") == "output-1" and CM.slug("") == "main" and CM.slug("../../etc/passwd") == "etc-passwd" and CM.slug(CM.COLLECTION) == CM.COLLECTION)
    M = CM.Store(tmp / "mem-unit")
    for bad in ("", "   "):
        try:
            M.add("w", bad); check("memory: an empty note is refused", False)
        except CM.MemoryError_:
            check("memory: an empty note is refused", True)
    M.add("w", "first note"); M.add("w", "second note"); M.add("w", "first note")
    check("memory: the same note twice is one note", [n["text"] for n in M.notes("w")] == ["first note", "second note"])
    M.add("w", "x" * 5000)
    check("memory: a long note is cut to the limit", len(M.notes("w")[-1]["text"]) == CM.LIMITS["note_chars"])
    M.remove("w", 3)
    b1 = M.block("w", 2400, "H")
    M.add("w", "third note")
    check("memory: listed oldest first, a new note appended (the cached prefix holds)",
          b1 == "H\n- first note\n- second note" and M.block("w", 2400, "H").startswith(b1))
    check("memory: over budget the newest are kept", M.block("w", 30, "H") == "H\n- second note\n- third note")
    M.add(CM.COLLECTION, "collected idea")
    ctx = M.context("w", CM.clamp({}))
    check("memory: the collection first, then the window's notes", 0 < ctx.index("collected idea") < ctx.index("first note"))
    check("memory: each can be switched off", "collected" not in M.context("w", CM.clamp({"use_collection": False}))
          and M.context("w", CM.clamp({"use_collection": False, "use_memory": False})) == "")
    check("memory: windows listed without the collection", [w["name"] for w in M.windows()] == ["w"])
    lim = dict(CM.LIMITS); CM.LIMITS["notes_per_window"] = 3
    try:
        M.add("w", "fourth"); check("memory: a full window refuses a note", False)
    except CM.MemoryError_:
        check("memory: a full window refuses a note", True)
    CM.LIMITS.update(lim)
    st = M.save_settings({"recall_k": 99, "memory_budget": -5, "use_memory": "off", "recall_source": "nowhere", "junk": 1})
    check("memory: options clamped to their limits and types, unknown keys dropped",
          st == {**CM.DEFAULTS, "recall_k": 4, "memory_budget": 0, "use_memory": False} and M.settings() == st)
    check("memory: the defaults are sane (memory and collection on, recall off)",
          CM.DEFAULTS["use_memory"] and CM.DEFAULTS["use_collection"] and CM.DEFAULTS["recall_k"] == 0 and CM.DEFAULTS["memory_budget"] <= 3000)
    recs = [{"at": 1790000000, "question": "how fast is ternary decode", "answer": "about 0.2 s a token"},
            {"at": 1790000100, "question": "what is a receipt", "answer": "the sha256 of the weights, request and answer"},
            {"at": 1790000200, "question": "unanswered", "answer": ""}]
    txt, ids = CM.recall(recs, "ternary decode speed", 2, set(), S._BM25)
    check("recall: the matching exchange, dated, as context", ids == [0] and "RECALL" in txt and "0.2 s a token" in txt)
    check("recall: skips what the conversation already holds, and is off at 0",
          CM.recall(recs, "ternary decode", 2, {"how fast is ternary decode"}, S._BM25) == ("", []) and CM.recall(recs, "ternary", 0, set(), S._BM25) == ("", []))
    # the endpoints
    J = {"Content-Type": "application/json"}
    post = lambda body: req("/api/memory", json.dumps(body).encode(), J)  # noqa: E731
    check("/api/memory: add to a window by its title", json.loads(post({"action": "add", "window": "Output 1", "text": "keep answers short"})[1])["window"] == "output-1")
    got = json.loads(req("/api/memory?window=Output%201")[1])
    check("/api/memory: the window's notes and the context a question carries",
          got["notes"][0]["text"] == "keep answers short" and "keep answers short" in got["context"] and got["settings"] == C.MEM.settings())
    hist = json.loads(req("/api/history?q=")[1])
    check("/api/history: the console's exchanges, newest first", hist["source"] == "console" and hist["of"] == len(hist["exchanges"]) >= 2)
    i0 = hist["exchanges"][0]["i"]
    check("/api/memory collect: exchanges of .history into the collection",
          json.loads(post({"action": "collect", "window": CM.COLLECTION, "source": "console", "indices": [i0]})[1])["ok"]
          and len(C.MEM.notes(CM.COLLECTION)) == 1 and "asked" in C.MEM.notes(CM.COLLECTION)[0]["text"])
    check("/api/memory: refusals are 400 with the reason", post({"action": "collect", "indices": []})[0] == 400 and post({"action": "nope"})[0] == 400
          and post({"action": "remove", "window": "Output 1", "n": 9})[0] == 400)
    check("/api/memory settings: saved, clamped", json.loads(post({"action": "settings", "settings": {"recall_k": 1}})[1])["settings"]["recall_k"] == 1)
    # what a question carries: the persona, the collection, the window's notes; recall and SELF just before the question
    import http.server as hs2
    sent = []
    class Eng(hs2.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            sent.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            b = b"data: [DONE]\n\n"
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    eng = hs2.ThreadingHTTPServer(("127.0.0.1", 0), Eng)
    threading.Thread(target=eng.serve_forever, daemon=True).start()
    real_serve = C.SERVE
    C.SERVE = f"http://127.0.0.1:{eng.server_address[1]}"
    try:
        req("/api/ask", json.dumps({"message": "and the ternary one?", "window": "Output 1"}).encode(), J)
        m = sent[-1]["messages"]
        check("ask: the window's memory and the collection follow the persona in the first system message",
              m[0]["role"] == "system" and m[0]["content"].startswith(C.persona()["system_prompt"])
              and "keep answers short" in m[0]["content"] and "asked" in m[0]["content"])
        check("ask: another window does not carry this window's notes",
              (req("/api/ask", json.dumps({"message": "hello", "window": "Output 2"}).encode(), J) and "keep answers short" not in sent[-1]["messages"][0]["content"]))
        check("ask: the exchange is logged with its window", json.loads(C.log_lines()[-1])["window"] == "output-2")
        check("ask: per-request options override the saved ones (no memory)",
              (req("/api/ask", json.dumps({"message": "x", "window": "Output 1", "options": {"use_memory": False, "use_collection": False}}).encode(), J)
               and sent[-1]["messages"][0]["content"] == C.base_system()))
    finally:
        C.SERVE = real_serve
        eng.shutdown()
        post({"action": "settings", "settings": CM.DEFAULTS})
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
    check("public: no Savante link to this computer", json.loads(req("/api/state", headers=pub)[1])["savante"] is None)
    check("public: the machine is not described, but ping answers", req("/api/sysdiag", headers=pub)[0] == 403 and req("/api/ping", headers=pub)[0] == 200)
    check("public: no memory and no history (they are the operator's)",
          req("/api/memory", headers=pub)[0] == 403 and req("/api/history", headers=pub)[0] == 403
          and req("/api/memory", b'{"action": "add", "text": "x"}', {**pub, **{"Content-Type": "application/json", "Origin": "https://demo.example"}})[0] == 403)
    C.PUBLIC = None
finally:
    srv.shutdown()

print("all ok" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
