#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Savante UI's data layer, offline (stdlib only; Gradio is not imported): proofs of data, CIDs, .history
search, .memory, metrics, and the view server's routes. A temporary state directory; nothing touches the canon.
run: python3 testing/test_ui.py"""
import json, os, sys, tempfile, threading, urllib.request, urllib.error
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="bankml-ui-"))
os.environ.update(BANKML_UI_STATE=str(tmp), RAGE_PATH=str(tmp / "no-rage"), SAVANTE_CANON=str(tmp / "no-canon"), BANKML_VOICE_DIR=str(tmp / "voice"), BANKML_EXPORT_DIR=str(tmp / "export"))
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
# RFC 6962: the certificate-transparency reference roots, and the second-preimage cases the old tree allowed
ct = [bytes.fromhex(x) for x in ["", "00", "10", "2021", "3031", "40414243", "5051525354555657", "606162636465666768696a6b6c6d6e6f"]]
check("Merkle roots equal RFC 6962's reference vectors (1, 2, 3 and 8 leaves)",
      u.merkle_root([u._leaf(x) for x in ct[:1]]) == "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
      and u.merkle_root([u._leaf(x) for x in ct[:3]]) == "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77"
      and u.merkle_root([u._leaf(x) for x in ct]) == "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328")
lv = [u._leaf(l) for l in lines]
node = bytes.fromhex(lv[0]) + bytes.fromhex(lv[1])  # an interior node's preimage, offered as a "line"
fake = {"scheme": u.MERKLE_SCHEME, "record": 0, "records": 4, "leaf": u._leaf(node), "path": [u._node(lv[2], lv[3])], "merkle_root": u.merkle_root(lv[:4])}
check("an interior node cannot pass as a record (leaves and nodes are domain-separated)", not u.verify_inclusion(node, fake))
check("3 records and 4 records with the last repeated have different roots", u.merkle_root(lv[:3]) != u.merkle_root(lv[:3] + lv[2:3]))
check("a proof moved to another index fails", not u.verify_inclusion(lines[1], {**u.inclusion_proof(u.HISTORY, 1), "record": 0}))

# a torn last line (a crash mid-write) must not blank the history, and the next record must not be glued onto it
import shutil as _sh
torn = tmp / "torn.history"
_sh.copy(u.HISTORY, torn)
with open(torn, "ab") as f:
    f.write(b'{"ts": 1, "user": "half a rec')
u.jsonl_append(torn, {"ts": 2, "user": "after the crash"})
recs = u.jsonl_read(torn)
check("a torn line is skipped and counted, the rest survives, and the next append starts on a new line",
      len(recs) == 6 and recs[-1]["user"] == "after the crash" and u.JSONL_BAD[str(torn)] == 1)
check("a receipt is checked against the raw answer (think block and whitespace included)",
      u._hash_ok({"assistant": "hi", "assistant_raw": "<think>x</think>\n hi ", "receipt": {"response_sha256": u.sha256("<think>x</think>\n hi ".encode())}})
      and u._hash_ok({"assistant": "hi", "receipt": {"response_sha256": u.sha256(b"hi")}}))
u.INFLIGHT.clear()
u.INFLIGHT["lost"] = __import__("time").time() - u.STALE_S - 1
check("a lost request (a closed tab) expires instead of blocking model switches forever", u.chat_busy() is False and not u.INFLIGHT)

# embeddings (bge-m3 through Ollama), offline: a fake Ollama whose vectors are bags of words hashed into 1,024 dims
import http.server as _hs, threading as _th, embed as _em, math as _m
def _vec(t):
    v = [0.0] * _em.DIMS
    for w in __import__("re").findall(r"[a-z0-9]+", t.lower()):
        v[int(__import__("hashlib").sha256(w.encode()).hexdigest(), 16) % _em.DIMS] += 1.0
    return v
class _FakeOllama(_hs.BaseHTTPRequestHandler):
    def do_GET(self):
        b = json.dumps({"models": []}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        b = json.dumps({"embeddings": [_vec(t) for t in req["input"]]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a):
        pass
_fo = _hs.ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
_th.Thread(target=_fo.serve_forever, daemon=True).start()
_em.OLLAMA = f"http://127.0.0.1:{_fo.server_address[1]}"
_em.provenance = lambda: {"model": "bge-m3", "digest": "d1" * 32, "bytes": 1, "licence": "mit", "open": True, "dims": _em.DIMS}
_em._mem_available = lambda: 8 * 10 ** 9
r_idx = _em.index(u.HISTORY, u.history_all())
check("embed.index: every exchange embedded once, cached beside .history with the model digest",
      r_idx["embedded"] == 5 and _em.index(u.HISTORY, u.history_all())["embedded"] == 0 and len(_em.cached(u.HISTORY)) == 5
      and all(abs(sum(x * x for x in v) - 1) < 1e-5 for v in _em.cached(u.HISTORY).values()))
_em.provenance = lambda: {"model": "bge-m3", "digest": "e2" * 32, "bytes": 1, "licence": "mit", "open": True, "dims": _em.DIMS}
check("a cache written by other weights is not used (vectors are tied to the model's digest)", _em.cached(u.HISTORY) == {})
_em.provenance = lambda: {"model": "bge-m3", "digest": "d1" * 32, "bytes": 1, "licence": "mit", "open": True, "dims": _em.DIMS}
check("reciprocal rank fusion: first in both beats first in one", [i for i, _ in _em.fuse([1, 2, 3], [2, 3, 1])][0] == 2
      and _em.fuse([7], [])[0][0] == 7)
hits_h, eng_h = u.history_search("ternary kernels")
check("the ragebar fuses BM25 and bge-m3 and says so", "bge-m3" in eng_h and "reciprocal rank" in eng_h and hits_h)
_mem = _em._mem_available
_em._mem_available = lambda: 10 ** 8
_em._QCACHE.clear()
hits_b, eng_b = u.history_search("verification")
check("without the memory to load bge-m3, the ragebar is BM25 alone (and still answers)", "bge-m3" not in eng_b and hits_b)
_em._mem_available = _mem
_em.provenance = lambda: {}
check("without bge-m3 in Ollama, embed says why and search stands on BM25", not _em.status()["ready"] and "bge-m3" not in u.history_search("verification")[1])
_fo.shutdown()

# the history window moves in steps, so the engine's prompt cache is reused (KoboldCpp's Smart Context idea; vLLM's
# chained block hashes show a one-exchange slide defeats it)
starts = [u.window_start(n) for n in range(1, 61)]
check("the window moves 8 times in 60 turns (a one-exchange slide moves 48 times)", sum(1 for a, b in zip(starts, starts[1:]) if a != b) == 8
      and all(12 <= n - s <= 17 for n, s in zip(range(13, 61), starts[12:])))
tt = [(f"q{i}", f"a{i}\n\n<sub>⏱ sent 12:00:00 · receipt</sub>") for i in range(30)]
mm = u.build_messages("S", tt, "Q", False)
check("the model never sees the chat's footer (clock, receipt)", "<sub>" not in json.dumps(mm) and mm[2]["content"] == "a18")
cnt = lambda t: max(1, len(t) // 4)  # noqa: E731 — a stand-in tokenizer, offline
big = [("q" * 100, "a" * 1200) for _ in range(13)]
u.build_messages("s" * 3600, big, "Q", False, ctx_tokens=2048, reserve_tokens=256, count=cnt)
check("never silent amnesia: when history cannot all fit, as much as fits is sent (0.1.8 sent none), and it is recorded",
      u.LAST_WINDOW["sent"] >= 2 and u.LAST_WINDOW["of"] == 13 and u.LAST_WINDOW["ctx"] == 2048)
check("an answer's own <sub> survives; only the chat's clock footer is removed",
      u.model_text("H\n\n<sub>2</sub>O\n\n<sub>⏱ sent 12:00</sub>") == "H\n\n<sub>2</sub>O")
check("the window fits the engine's context, moving in whole steps", len(u.build_messages("S", tt, "Q", False, ctx_tokens=60, reserve_tokens=10)) < len(mm))

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

# custom agents: keccak256, the doctrine root, derive / verify / save, preflight, the template untouched
os.environ["BANKML_AGENTS"] = str(tmp / "agents")
import importlib, agents  # noqa: E402
importlib.reload(agents)
check("keccak256('') and ('abc') vectors", agents.keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
      and agents.keccak256(b"abc").hex() == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45")
try:  # an independent implementation, where installed: every length 0..400 crosses the 136-byte rate twice
    from Crypto.Hash import keccak as _k
    check("keccak256 == pycryptodome for every length 0..400", all(agents.keccak256((bytes(range(256)) * 2)[:n]) ==
          _k.new(digest_bits=256, data=(bytes(range(256)) * 2)[:n]).digest() for n in range(401)))
except ImportError:
    print("skip  keccak256 vs pycryptodome (not installed)")
real = Path.home() / "savante"
if (real / "savante.persona").is_file():
    sp, sl = json.loads((real / "savante.persona").read_bytes()), json.loads((real / "savante.commitments.json").read_bytes())
    check("doctrine root reproduces Savante's published ledger value", agents.doctrine_root(sp, sl["doctrine_root"]["pointers"]) == sl["doctrine_root"]["value"])
else:
    print("skip  doctrine root vs Savante (no ~/savante here)")
tpl = {"persona": "tpl", "name": "Template", "source": "s", "format": "f", "system_prompt": "You are the template.", "mantra": "m", "oath": "o",
       "bdi": {"beliefs": [{"id": "b", "belief": "x"}]}, "skills": {"primary": "p", "taxonomy": [], "defer_triggers": [], "validation": []},
       "safety": {"scope": "s"}, "embodiment": {}, "token": {"intelligence": {"tool_allowlist": ["read"]}, "bindings": {"erc7857": "SECRET"}}}
tpl_bytes = json.dumps(tpl, sort_keys=True).encode()
slug = agents.derive(tpl, {"artifacts": {"identity": {"sha256": "ab"}}, "doctrine_root": {"value": "0x01"}}, "Auditor Ada", system_prompt="You are Ada.", description="an auditor")
f = agents.files(slug)
check("derive writes persona, prompt, card, ledger; it verifies", slug == "auditor_ada" and all(f[k].is_file() for k in ("persona", "prompt", "agentcard.json", "commitments.json"))
      and all(ok for _, ok, _ in agents.verify(slug)))
pd = json.loads(f["persona"].read_text())
check("derived identity is new; the template's bindings are not copied", pd["name"] == "Auditor Ada" and pd["token"]["bindings"]["erc7857"] is None
      and json.dumps(tpl, sort_keys=True).encode() == tpl_bytes)
card = json.loads(f["agentcard.json"].read_text())
check("agent card: ERC-8004 registration-v1, not_yet_minted, derived_from recorded", card["type"].endswith("#registration-v1") and card["bankml"]["status"] == "not_yet_minted"
      and card["bankml"]["derived_from"]["persona_sha256"] == "ab")
r0 = json.loads(f["commitments.json"].read_text())["doctrine_root"]["value"]
led = agents.save(slug, prompt_text="You are Ada, revised.")
check("editing .prompt re-ledgers (prompt hash changes; the doctrine root, over the persona, does not)", led["doctrine_root"]["value"] == r0
      and led["artifacts"]["prompt"]["sha256"] == u.sha256(b"You are Ada, revised.\n"))
pd["voice_examples"] = ["I check before I speak."]
led = agents.save(slug, persona_text=json.dumps(pd))
check("editing a field outside the doctrine re-ledgers with the same doctrine root", led["doctrine_root"]["value"] == r0
      and led["doctrine_root"]["fixed_at_derivation"] == r0 and all(ok for _, ok, _ in agents.verify(slug)))
try:
    agents.save(slug, persona_text=json.dumps(dict(pd, mantra="a new mantra")))
    check("a doctrine clause cannot be edited after derivation (the docs' promise, now enforced)", False)
except ValueError as e:
    check("a doctrine clause cannot be edited after derivation (the docs' promise, now enforced)", "/mantra" in str(e)
          and json.loads(f["persona"].read_text())["mantra"] != "a new mantra")
led_p = f["commitments.json"]
lj = json.loads(led_p.read_text())
lj["doctrine_root"]["pointers"] = ["/name"]  # a ledger that narrows its own check
led_p.write_text(json.dumps(lj))
check("verify uses bankml's pointer list, not the one in the ledger it checks", not dict((n, ok) for n, ok, _ in agents.verify(slug))["doctrine root"])
agents.rebind(slug)
f["prompt"].write_text("tampered\n")
check("a hand edit outside save() is caught by verify", not dict((n, ok) for n, ok, _ in agents.verify(slug))["prompt"])
for bad, why in ((dict(pd, weight=0.5), "float"), ({k: v for k, v in pd.items() if k != "oath"}, "missing clause")):
    try:
        agents.save(slug, persona_text=json.dumps(bad))
        check(f"preflight refuses a {why}", False)
    except ValueError:
        check(f"preflight refuses a {why}", True)
try:
    agents.agent_dir("../etc")
    check("agent_dir rejects path traversal", False)
except ValueError:
    check("agent_dir rejects path traversal", True)

# the chosen aivatar: validated by content and size, ledgered, and part of the THOT bundle
png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
led = agents.set_aivatar(slug, png)
check("a PNG aivatar is accepted and ledgered", led["artifacts"]["aivatar"]["sha256"] == u.sha256(png) and all(ok for _, ok, _ in agents.verify(slug)))
for bad, why in ((b"not an image at all", "a non-image"), (b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024), "an image over 2 MB")):
    try:
        agents.set_aivatar(slug, bad)
        check(f"the aivatar refuses {why}", False)
    except ValueError:
        check(f"the aivatar refuses {why}", True)

# THOT manifests: the spec's test vectors (where those repositories are here), and a bankml bundle's lineage
import subprocess, thot  # noqa: E402
VEC = [("savante", Path.home() / "savante", "1fcca89", "savante.thot.json", "0x235da8e993dc8af2c077f50d698962446b872b17b1e5b033d5c5d976532b8880",
        "0xdc1d80957cf831aee6638fd569e22cf0f6e5a1ec99ddde91cfecb5a15408fbe1", "bafkreieerglzkjrkmwrxhmrk53bf3tvhp3aut3qlatwpdtoutmspuxrh4i"),
       ("jaimla", Path.home() / "cryptoAGI" / "jaimla", "8b57ccf", "jaimla.thot.json", "0x7f5bcc69dc9d50854189762dba0a97b3a48090ee546a0423aa9722917740e086",
        "0xd7cb5363324645b4f6d0df19902fe9cd15dcb1c36ae925ee5ff2278315f6a258", "bafkreidczapw72pldcf5pxyoz2hhdleauto7ajvjgun7w2iiq7ccwwhvku"),
       ("luvai", Path.home() / "cryptoAGI" / "luvai", "0c1eef7", "luvai.thot.json", "0x2feb7ecfcffe91c035951298e5d977d880ec475edb87fff5f3038efc8d44f94c",
        "0xf594d1b2ab3f98b20ca203227e71a437ac86d82f6c4d4942227ecf8ee018309b", "bafkreiefvcfiw2rkrcjzblcj5ivrux7mtpsha4fqaulgvteycpr4zbwqtq")]
for name, repo, commit, fn, br, mr, cid in VEC:
    g = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{fn}"], capture_output=True, text=True)
    if g.returncode:
        print(f"skip  THOT test vector {name} (repository not here)")
        continue
    m = json.loads(g.stdout)
    recs = thot.records(m["facets"])
    check(f"THOT spec vector {name}: bundle_root, Merkle root and identity CID", thot.bundle_root(recs)[0] == br and thot.merkle_root(recs) == mr
          and thot.identity(m)["cid"] == cid)
m1 = thot.build(slug)
check("a bankml THOT bundle verifies, with the aivatar as a facet", thot.verify(slug) == [] and m1["merkle"]["populated"] == 6
      and any(x["facet"] == "x-bankml.aivatar" for x in m1["facets"]))
check("the bundle commits history by digest and holds none of its text", "question" not in (agents.agent_dir(slug) / f"{slug}.thot.json").read_text()
      and any(x["facet"] == "x-bankml.history" for x in m1["facets"]))
m1b = thot.build(slug)
f["history"].write_text(f["history"].read_text() + '{"user": "new"}\n')
m2 = thot.build(slug)
check("re-bind keeps the generation; a facet change is n+1 with parent = the previous CID",
      m1b["bundle"]["generation"] == m1["bundle"]["generation"] and m2["bundle"]["generation"] == m1b["bundle"]["generation"] + 1
      and m2["bundle"]["parent"] == m1b["identity"]["cid"] and thot.verify(slug) == [])

# Savante's voice: the pronunciation table, speech from markdown, one render, the view's audio gate
import speak  # noqa: E402
t2 = {"format": "voaice-pronunciation/1", "version": 2, "entries": [{"match": "PYTHAIML", "say": "Pith AI M L"}, {"match": "SAVANTE", "say": "Sav ont"},
                                                                  {"match": "PYTHAI", "say": "Pith AI"}]}
check("pronunciation: one pass, longest match first, case-insensitive", speak.say("savante and PYTHAIML and pythai", t2) == "Sav ont and Pith AI M L and Pith AI")
check("pronunciation is idempotent (a spoken form holds no match)", speak.say(speak.say("Savante", t2), t2) == "Sav ont")
sp = speak.speech("# Title\n\nSee [the docs](https://x.y) and `code`. Second sentence!\n\n```sh\nrm -rf /\n```\n\n| a | b |\n|---|---|\n\n- a **bold** item")
check("speech(): code and tables dropped, links read as text, markers gone", "rm -rf" not in " ".join(sp) and "https" not in " ".join(sp)
      and "See the docs and code." in sp and "Second sentence!" in sp and "a bold item" in sp)
check("speech() keeps savante_sagi as written; say() speaks it as words", speak.speech("one seat of core_command, savante_sagi.") == ["one seat of core_command, savante_sagi."]
      and speak.say("APPROVE_WITH_CONDITIONS", t2) == "APPROVE WITH CONDITIONS")
ok_v, why_v = speak.available()
if ok_v:
    it = speak.render(["I am Savante."])
    check("a statement renders to Opus in Savante's voice, cached by key", len(it) == 1 and it[0]["file"].is_file() and it[0]["file"].read_bytes()[:4] == b"OggS"
          and speak.cached(["I am Savante."])[0]["key"] == it[0]["key"])
else:
    print(f"skip  voice render ({why_v})")
check("say(): SCIEN·TIFIC, bankml and a pythai address are said, not spelled", speak.say("SCIEN·TIFIC at scientific.pythai.net runs bankml", t2)
      == "Sci-en, Tiffic at scientific dot Pith AI dot net runs bank M L")
check("say(): TECHNICAL.md notation read aloud", speak.say("w ∈ {−1, 0, +1}; 3⁵ = 243 ≤ 256 (§III.5)", t2)
      == "w in minus one, zero, or plus one; three to the fifth equals 243 at most 256")
check("say(): names keep their separators; files, formats and references are read properly",
      speak.say("Professor / OVERLORD", t2) == "Professor / OVERLORD"
      and speak.say("mindX DAIO · savante_sagi", t2) == "mindX DAIO · Sav ont sagi"
      and speak.say("see (q1_0.rs, 0.0.4) now", t2) == "see now" and speak.say("bankml.rs", t2) == "bank M L dot R S"
      and speak.say("PQ2_0 = 142 and Q4_K_M", t2) == "P Q two zero equals 142 and Q four K M"
      and speak.say("Bankml, as in §III.4.", t2) == "bank M L, as in section three point four."
      and speak.say("(16 + 2) × 8 / 128", t2) == "(16 plus 2) times 8 over 128")
check("speech(): list markers only at line starts; an inline + survives", speak.speech("- item one\n\nThe cost is (16 + 2) bits.") == ["item one", "The cost is (16 + 2) bits."])
rc = speak.reading_chapters()
check("the reading: thesis, II.1, III.2, III.5 and III.8 from TECHNICAL.md", [t for t, _ in rc][0] == "The thesis" and len(rc) == 5
      and rc[-1][0] == "Can a binary computer perform a ternary operation?" and rc[-1][1][0] == "Yes.")
try:
    speak.export("t-incomplete", [("c", ["never rendered, never exported"])])
    check("an export is refused until every sentence is rendered", False)
except RuntimeError:
    check("an export is refused until every sentence is rendered", not (speak.EXPORT_DIR / "t-incomplete.opus").exists())
(speak.EXPORT_DIR).mkdir(parents=True, exist_ok=True)
(speak.EXPORT_DIR / "t-broken.opus").write_bytes(b"OggS")
(speak.EXPORT_DIR / "t-broken.json").write_text('{"sig": "abc', encoding="utf-8")
check("a damaged export record means not current, never an error", speak.export_state("t-broken", [("c", ["x"])])["current"] is False)
if ok_v:
    speak.render(["It ends."])
    e = speak.export("t-set", [("one", ["I am Savante."]), ("two", ["It ends."])])
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(e["file"])], capture_output=True, text=True)
    check("a complete export: one Ogg Opus file, one chapter mark per chapter, current until a clip changes",
          e["current"] and e["file"].read_bytes()[:4] == b"OggS" and len(json.loads(pr.stdout or "{}").get("chapters", [])) == 2
          and not speak.export_state("t-set", [("one", ["I am Savante."]), ("two", ["It ends.", "A new line."])])["current"])
import view  # noqa: E402
check("view's audio route: only a 24-hex key the manifest lists", view.audio_file("../../etc/passwd") is None and view.audio_file("0" * 24) is None
      and (not ok_v or view.audio_file(it[0]["key"]) is not None))

# the view server: fixed routes, commitments only, no traversal
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
st_k, body_k = get("/knobs.js")
check("view serves the DreamKnob bundle at /knobs.js", st_k == 200 and b"SavanteKnobs" in body_k)
check("view's export route: only a named, complete export", get("/export/t-set.opus")[0] == 404 and get("/export/..%2Fmanifest.opus")[0] == 404
      and view.export_file("Savante") is None)
srv.shutdown()

print(f"{'all ok' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
