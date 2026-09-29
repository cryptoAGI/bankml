#!/usr/bin/env python3
"""The Savante UI's data layer, offline (stdlib only; Gradio is not imported): proofs of data, CIDs, .history
search, .memory, metrics, and the view server's routes. A temporary state directory; nothing touches the canon.
run: python3 testing/test_ui.py"""
import json, os, sys, tempfile, threading, urllib.request, urllib.error
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="bankml-ui-"))
os.environ.update(BANKML_UI_STATE=str(tmp), RAGE_PATH=str(tmp / "no-rage"), SAVANTE_CANON=str(tmp / "no-canon"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import savante as u  # noqa: E402

fails = 0
def check(name, ok):
    global fails
    fails += not ok
    print(("ok   " if ok else "FAIL ") + name)

# CIDv1 raw of "abc" — the value mindX rage.py and the canon's ledger construction give
check("cid_v1_raw('abc')", u.cid_v1_raw(b"abc") == "bafkreif2pall7dybz7vecqka3zo24irdwabwdi4wc55jznaq75q7eaavvu")

# .history: records, commitment, inclusion proofs (odd and even sizes), tamper detection
for i in range(5):
    u.history_append({"ts": 1790000000 + i, "sent_at": f"2026-09-28T10:0{i}:00.000-0700", "first_token_s": 10.0 + i, "response_s": 20.0 + 2 * i,
                      "session": "s1", "user": f"question {i} about verification" if i % 2 else f"question {i} about ternary kernels",
                      "assistant": f"answer {i}", "receipt": {"prompt_tokens": 300, "completion_tokens": 20, "response_sha256": u.sha256(f"answer {i}".encode())}})
c = u.commitment(u.HISTORY)
check("commitment counts 5 records, has a root and a CID", c["records"] == 5 and len(c["merkle_root"]) == 64 and c["file_cid"].startswith("bafkrei"))
lines = u._lines(u.HISTORY)
check("every record's inclusion proof verifies", all(u.verify_inclusion(lines[k], u.inclusion_proof(u.HISTORY, k)) for k in range(5)))
check("a changed byte fails its proof", not u.verify_inclusion(lines[2].replace(b"answer 2", b"answer X"), u.inclusion_proof(u.HISTORY, 2)))
check("a proof for another record fails", not u.verify_inclusion(lines[1], u.inclusion_proof(u.HISTORY, 3)))
check("out-of-range proof is empty", u.inclusion_proof(u.HISTORY, 9) == {})

# search (built-in BM25 here: RAGE_PATH points nowhere)
hits, engine = u.history_search("ternary kernels")
check("history search finds the ternary questions first (built-in engine)", engine.startswith("RAGE-shaped") and {h[1] for h in hits[:3]} == {0, 2, 4})
check("empty query returns nothing", u.history_search("  ")[0] == [])

# metrics from recorded timings
m = u.metrics()
check("metrics: 5 exchanges, median response 24 s, all hashes match", m["exchanges"] == 5 and abs(m["response_s"]["median"] - 24.0) < 1e-9 and m["hash_ok"] == 5)
check("metrics: prefill = prompt tokens / first token", abs(m["prefill_tok_s"]["median"] - 300 / 12.0) < 1e-9)

# .memory
u.memory_add("the operator prefers verdicts", {"kind": "typed"})
u.memory_add("second note", {"kind": "response"})
check(".memory holds 2 notes and a prompt block", len(u.memory_all()) == 2 and "second note" in u.memory_block() and "not evidence" in u.memory_block())
u.memory_remove(1)
check(".memory remove by number", [x["text"] for x in u.memory_all()] == ["second note"])

# the view server: fixed routes, commitments only, no traversal
import view  # noqa: E402
from http.server import ThreadingHTTPServer
srv = ThreadingHTTPServer(("127.0.0.1", 0), view.H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"
def get(path, method="GET"):
    try:
        with urllib.request.urlopen(urllib.request.Request(base + path, method=method), timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
st, body = get("/api/state")
state = json.loads(body)
check("view /api/state carries the .history commitment", st == 200 and state["private"]["history"]["merkle_root"] == c["merkle_root"])
check("view /api/state carries no .history content", b"question 1" not in body and b"answer 3" not in body)
check("view refuses traversal and unknown paths", get("/api/result?name=../../etc/passwd")[0] == 404 and get("/etc/passwd")[0] == 404)
check("view refuses POST", get("/api/state", "POST")[0] == 405)
srv.shutdown()

print(f"{'all ok' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
