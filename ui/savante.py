#!/usr/bin/env python3
"""bankml · Savante — the local chat UI, in Gradio, built from the Hugging Face template PYTHAI/savante.

What it takes from the template (the Space's `local:bonsai-8b` carrier): the system prompt is the canon
persona's own `system_prompt`, the history is the last 12 exchanges (each ≤ 4000 characters), answers are
labelled drafts, `<think>` blocks are stripped and Qwen3 carriers get ` /no_think`.

What it adds:
  · **iNFT compatibility.** Savante's canon (~/savante, or SAVANTE_CANON) is read and never written. At start
    every file the ledger `savante.commitments.json` commits to — the nine artefacts, the agent card, the image —
    is re-hashed (sha256) and compared; if the persona does not verify, the UI refuses to speak as Savante.
    `bind/savante_verify.py` (the full offline verifier: doctrine root, thot, mirror) runs from a button.
    Nothing here mints, and nothing is written into the canon — not even a Python cache.
  · **.prompt.** Which system prompt carries the conversation: the persona's `system_prompt` (canon, ledgered;
    the default), `sAGI.prompt` (canon facet, ledgered) or the Space's `Savante.prompt` (template, not ledgered).
    Each is shown with its sha256 and whether the ledger vouches for it.
  · **.history.** Every exchange is one JSONL line in `savante.history` under BANKML_UI_STATE
    (default ~/.local/share/bankml/savante/), outside the canon; the last session reloads on start.
  · **bankml.** Answers come through `bankml serve` — the guard and the sha256 pin in front of the model, and a
    receipt (model sha256, tokens, timings, sha256 of the answer) on every response.

  · **Two modes.** `--mode interact` (the operator, loopback: chat, .prompt, .history with response times, the
    verifier; Gradio) and view (read-only, for anyone on the LAN: the live testing log, every release's results, CI,
    the laptop's load, Savante's office and ledger — served by `ui/view.py`, the standard library, not Gradio).

  bankml serve .models/Bonsai-8B-Q1_0.gguf --fork FORK.json --upstream 127.0.0.1:18092 --listen 127.0.0.1:18093
  python3 ui/savante.py --mode interact               # http://127.0.0.1:7873
  python3 ui/view.py --host 0.0.0.0 --port 7874       # http://<this laptop's LAN address>:7874

stdlib + gradio (3.x or newer).
"""
from __future__ import annotations

import argparse
import html
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.dont_write_bytecode = True  # never leave a cache next to anything we import

CANON = Path(os.environ.get("SAVANTE_CANON", Path.home() / "savante")).expanduser()
SERVE = os.environ.get("BANKML_SERVE", "http://127.0.0.1:18093")
STATE = Path(os.environ.get("BANKML_UI_STATE", Path.home() / ".local" / "share" / "bankml" / "savante")).expanduser()
HISTORY = STATE / "savante.history"
MEMORY = STATE / "savante.memory"
ACTIVE = {"slug": None}  # None = Savante (the canon); else a custom agent from ui/agents.py


def use_agent(slug):
    """Point .history and .memory (and the prompt choices) at Savante or at a custom agent."""
    global HISTORY, MEMORY
    import agents
    if slug:
        f = agents.files(slug)
        HISTORY, MEMORY = f["history"], f["memory"]
    else:
        HISTORY, MEMORY = STATE / "savante.history", STATE / "savante.memory"
    ACTIVE["slug"] = slug or None
RAGE_PATH = Path(os.environ.get("RAGE_PATH", Path.home() / "mindX" / "mindx" / "godel" / "mindxtrain" / "hf" / "space_ui")).expanduser()
REPO = Path(os.environ.get("BANKML_REPO", Path(__file__).resolve().parents[1])).expanduser()
LIVE = REPO / "testing" / "live.log"
CI_API = "https://api.github.com/repos/cryptoAGI/bankml/actions/runs?per_page=5"
SPACE_PROMPT_URL = "https://huggingface.co/spaces/PYTHAI/savante/resolve/main/Savante.prompt"
KEEP_EXCHANGES, KEEP_CHARS = 12, 4000  # the template's buildMessages limits


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── the canon, read-only, checked against its ledger ─────────────────────────────
class Canon:
    def __init__(self, root: Path):
        self.root = root
        self.ledger = self._json("savante.commitments.json") or {}
        self.persona = self._json("savante.persona") or {}
        self.card = self._json("savante.agentcard.json") or {}
        self.rows = self._check()

    def _json(self, rel):
        try:
            return json.loads((self.root / rel).read_bytes())
        except Exception:
            return None

    def _check(self):
        """(name, path, ok, detail) for every file the ledger commits to."""
        want = [(k, v.get("path"), v.get("sha256")) for k, v in (self.ledger.get("artifacts") or {}).items()]
        card = self.ledger.get("card") or {}
        v0 = (card.get("versions") or {}).get("v0") or {}
        want.append(("card v0", card.get("path"), v0.get("sha256")))
        img = self.ledger.get("image_candidate") or {}
        want.append(("image (named, not pinned)", img.get("path"), img.get("sha256")))
        bundle = self.ledger.get("bundle") or {}
        want.append(("bundle (thot)", bundle.get("path"), (bundle.get("file") or {}).get("sha256")))
        rows = []
        for name, rel, h in want:
            if not rel or not h:
                rows.append((name, rel or "?", False, "not in the ledger"))
                continue
            try:
                got = sha256((self.root / rel).read_bytes())
                rows.append((name, rel, got == h, "sha256 ✓" if got == h else f"sha256 {got[:12]}… ≠ ledger {h[:12]}…"))
            except OSError as e:
                rows.append((name, rel, False, f"unreadable: {e.strerror}"))
        return rows

    @property
    def persona_ok(self):
        return any(n == "identity" and ok for n, _, ok, _ in self.rows)

    def ledgered(self, rel: str) -> bool:
        return any(r == rel and ok for _, r, ok, _ in self.rows)

    def file(self, rel):
        try:
            return (self.root / rel).read_bytes()
        except OSError:
            return None


def space_prompt() -> bytes | None:
    """The template's Savante.prompt: cached under STATE, fetched from the Space once."""
    cache = STATE / "Savante.prompt"
    if cache.is_file():
        return cache.read_bytes()
    try:
        with urllib.request.urlopen(SPACE_PROMPT_URL, timeout=10) as r:
            b = r.read()
        STATE.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b)
        return b
    except Exception:
        return None


PROMPTS = ("persona · system_prompt (ledgered)", "the agent's .prompt file (ledgered; Savante: sAGI.prompt)", "Savante.prompt (Space template, not ledgered)")


def system_prompt(canon: Canon, which: str):
    """(text, provenance line) — or (None, why) when the choice cannot be used."""
    if ACTIVE["slug"]:
        import agents
        slug = ACTIVE["slug"]
        bad = [r for r in agents.verify(slug) if not r[1]]
        if bad:
            return None, f"agent {slug} does not verify against its ledger ({', '.join(r[0] for r in bad)}): refusing to speak"
        f = agents.files(slug)
        led = json.loads(f["commitments.json"].read_text(encoding="utf-8"))
        if which == PROMPTS[1]:
            b = f["prompt"].read_bytes()
            return b.decode("utf-8", "replace"), f"{slug}.prompt · sha256 {sha256(b)[:12]}… (ledgered)"
        if which == PROMPTS[0]:
            sp = json.loads(f["persona"].read_text(encoding="utf-8")).get("system_prompt") or ""
            return sp, f"{slug}.persona system_prompt · persona sha256 {led['artifacts']['persona']['sha256'][:12]}… (ledgered)"
    if which == PROMPTS[0]:
        if not canon.persona_ok:
            return None, "savante.persona does not verify against the ledger: refusing to speak as Savante"
        sp = canon.persona.get("system_prompt") or ""
        return sp, f"persona.system_prompt · persona sha256 {canon.ledger['artifacts']['identity']['sha256'][:12]}… (ledgered)"
    if which == PROMPTS[1]:
        b = canon.file("sAGI.prompt")
        if b is None or not canon.ledgered("sAGI.prompt"):
            return None, "sAGI.prompt missing or does not verify against the ledger"
        return b.decode("utf-8", "replace"), f"sAGI.prompt · sha256 {sha256(b)[:12]}… (ledgered)"
    b = space_prompt()
    if b is None:
        return None, f"Savante.prompt not cached and {SPACE_PROMPT_URL} unreachable"
    return b.decode("utf-8", "replace"), f"Savante.prompt · sha256 {sha256(b)[:12]}… (Space template; not in the ledger)"


# ── .history ─────────────────────────────────────────────────────────────────────
def history_append(rec: dict):
    STATE.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def history_load():
    """(session id, [[user, answer], …]) of the most recent session, or a new session."""
    try:
        lines = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, ValueError):
        lines = []
    if not lines:
        return uuid.uuid4().hex[:12], []
    sid = lines[-1].get("session")
    return sid, [[r.get("user", ""), r.get("shown", r.get("assistant", ""))] for r in lines if r.get("session") == sid]


def history_all() -> list:
    """Every record in .history, oldest first (all sessions)."""
    try:
        return [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, ValueError):
        return []


def _timing(r: dict):
    """(first_token_s, response_s, source) — measured at the press of Send when recorded (0.0.7+), else the receipt's."""
    if r.get("response_s") is not None:
        return r.get("first_token_s"), r.get("response_s"), "ui"
    rc = r.get("receipt") or {}
    if rc.get("wall_ms") is not None:
        return (rc["ttft_ms"] / 1000 if rc.get("ttft_ms") is not None else None), rc["wall_ms"] / 1000, "receipt"
    return None, None, None


def _when(r: dict) -> str:
    return r.get("sent_at") or (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(r["ts"])) if r.get("ts") else "?")


def _hash_ok(r: dict):
    h = (r.get("receipt") or {}).get("response_sha256")
    return None if not h else h == sha256((r.get("assistant") or "").encode())


# ── proof of data without the data: sha256 leaves, a Merkle root, CIDv1 ─────────────
# The content stays here (files outside the canon, a browser's localStorage); what travels is a commitment. A
# holder of the root can check any one exchange shown to them — its line and its sibling path — without the rest.
def cid_v1_raw(b: bytes) -> str:
    """CIDv1, raw codec (0x55), sha2-256 multihash, base32 lower with the 'b' prefix — as the canon's ledger uses."""
    import base64
    return "b" + base64.b32encode(bytes([0x01, 0x55, 0x12, 0x20]) + hashlib.sha256(b).digest()).decode().lower().rstrip("=")


def _lines(path: Path) -> list:
    try:
        return [l for l in path.read_bytes().split(b"\n") if l.strip()]
    except OSError:
        return []


def _pair(a: str, b: str) -> str:
    return hashlib.sha256(bytes.fromhex(a) + bytes.fromhex(b)).hexdigest()


def merkle(leaves: list) -> list:
    """All levels, leaves first; an odd node is paired with itself."""
    levels = [leaves]
    while len(levels[-1]) > 1:
        lv = levels[-1]
        levels.append([_pair(lv[i], lv[i + 1] if i + 1 < len(lv) else lv[i]) for i in range(0, len(lv), 2)])
    return levels


def commitment(path: Path) -> dict:
    """{records, merkle_root, file_sha256, file_cid} for a JSONL file — shareable; reveals no content."""
    lines = _lines(path)
    leaves = [sha256(l) for l in lines]
    raw = b"".join(l + b"\n" for l in lines)
    return {"file": path.name, "records": len(lines), "merkle_root": merkle(leaves)[-1][0] if leaves else None,
            "file_sha256": sha256(raw) if lines else None, "file_cid": cid_v1_raw(raw) if lines else None}


def inclusion_proof(path: Path, k: int) -> dict:
    """The proof that record k (0-based) is in the file whose root is `merkle_root`."""
    lines = _lines(path)
    leaves = [sha256(l) for l in lines]
    if not 0 <= k < len(leaves):
        return {}
    levels, i, sib = merkle(leaves), k, []
    for lv in levels[:-1]:
        j = i ^ 1
        sib.append({"side": "right" if j > i else "left", "hash": lv[j] if j < len(lv) else lv[i]})
        i //= 2
    return {"record": k, "leaf": leaves[k], "line_sha256_of": "the exact JSONL line, without its newline",
            "path": sib, "merkle_root": levels[-1][0], "records": len(leaves)}


def verify_inclusion(line: bytes, proof: dict) -> bool:
    h = sha256(line)
    if h != proof.get("leaf"):
        return False
    for step in proof.get("path", []):
        h = _pair(h, step["hash"]) if step["side"] == "right" else _pair(step["hash"], h)
    return h == proof.get("merkle_root")


# ── RAGE over .history: the house index (mindX rage.py) when present, else the same BM25 shape here ──
class _BM25:
    def __init__(self):
        self.docs, self.df = [], {}

    @staticmethod
    def toks(t):
        return re.findall(r"[a-z0-9]+", t.lower())

    def add(self, text, source):
        tf = {}
        for w in self.toks(text):
            tf[w] = tf.get(w, 0) + 1
        for w in tf:
            self.df[w] = self.df.get(w, 0) + 1
        self.docs.append((source, text, tf, sum(tf.values())))

    def search(self, q, k=8):
        import math
        n = len(self.docs) or 1
        avg = sum(d[3] for d in self.docs) / n if self.docs else 1
        out = []
        for src, text, tf, ln in self.docs:
            sc = 0.0
            for w in set(self.toks(q)):
                if w in tf:
                    idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    sc += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * ln / avg))
            if sc > 0:
                out.append((sc, src, text))
        return sorted(out, reverse=True)[:k]


def rage_engine():
    """The house RAGE module if importable (no cache written), else None."""
    try:
        if str(RAGE_PATH) not in sys.path:
            sys.path.append(str(RAGE_PATH))
        from mindxhfgradio import rage  # noqa: WPS433 (stdlib-only module)
        return rage
    except Exception:
        return None


def history_search(query: str, k: int = 8):
    """[(score, record_index, record)] best first; engine name."""
    recs = history_all()
    if not query.strip() or not recs:
        return [], "—"
    rage = rage_engine()
    docs = [(i, f"{r.get('user', '')}\n{r.get('assistant', '')}") for i, r in enumerate(recs)]
    if rage is not None:
        idx = rage.Index()
        for i, t in docs:
            idx.add(t, source=str(i))
        hits = [(sc, int(d.source), recs[int(d.source)]) for sc, d in idx.search(query, k=k * 3)]
        engine = "RAGE (mindX rage.py, BM25)"
    else:
        idx = _BM25()
        for i, t in docs:
            idx.add(t, str(i))
        hits = [(sc, int(src), recs[int(src)]) for sc, src, _ in idx.search(query, k=k * 3)]
        engine = "RAGE-shaped BM25 (built in)"
    seen, out = set(), []
    for sc, i, r in hits:  # one hit per exchange (the house index chunks long ones)
        if i not in seen:
            seen.add(i)
            out.append((sc, i, r))
    return out[:k], engine


# ── .memory: notes the operator keeps, outside the canon ──────────────────────────
def memory_all() -> list:
    try:
        return [json.loads(l) for l in MEMORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, ValueError):
        return []


def memory_add(text: str, source: dict | None = None) -> int:
    text = (text or "").strip()
    if not text:
        return len(memory_all())
    STATE.mkdir(parents=True, exist_ok=True)
    with MEMORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": round(time.time(), 3), "at": iso(time.time()), "text": text, "sha256": sha256(text.encode()),
                            "source": source or {"kind": "typed"}}, ensure_ascii=False) + "\n")
    return len(memory_all())


def memory_remove(n: int) -> int:
    m = memory_all()
    if 1 <= n <= len(m):
        del m[n - 1]
        STATE.mkdir(parents=True, exist_ok=True)
        MEMORY.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in m), encoding="utf-8")
    return len(memory_all())


MEMORY_BUDGET = 2400


def memory_block() -> str:
    """The notes as a system-prompt addendum, newest first, within MEMORY_BUDGET characters."""
    out, used = [], 0
    for m in reversed(memory_all()):
        line = "- " + m["text"].replace("\n", " ")
        if used + len(line) > MEMORY_BUDGET:
            break
        out.append(line)
        used += len(line)
    if not out:
        return ""
    return ("\n\nMEMORY — notes the operator kept from earlier conversations. They are context, not evidence: "
            "cite them as the operator's notes, never as findings.\n" + "\n".join(out))


# ── metrics, computed from .history ───────────────────────────────────────────────
def _pct(v, q):
    v = sorted(v)
    if not v:
        return None
    i = (len(v) - 1) * q
    lo = int(i)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (i - lo)


def metrics() -> dict:
    recs = history_all()
    rows, first, total, pre, wr, ok, bad = [], [], [], [], [], 0, 0
    for r in recs:
        ft, rs, src = _timing(r)
        rc = r.get("receipt") or {}
        pt, ct = rc.get("prompt_tokens"), rc.get("completion_tokens")
        h = _hash_ok(r)
        ok += h is True
        bad += h is False
        if rs is not None:
            total.append(rs)
        if ft is not None:
            first.append(ft)
            if pt:
                pre.append(pt / ft if ft > 0 else 0)
            if ct and rs and rs > ft:
                wr.append(ct / (rs - ft))
        rows.append({"when": _when(r), "session": r.get("session"), "first": ft, "total": rs, "src": src, "pt": pt, "ct": ct, "hash": h,
                     "q": (r.get("user") or "")[:80]})
    days = {}
    for x in rows:
        days[x["when"][:10]] = days.get(x["when"][:10], 0) + 1
    stat = lambda v: {"n": len(v), "median": _pct(v, .5), "p90": _pct(v, .9), "mean": (sum(v) / len(v)) if v else None,
                      "min": min(v) if v else None, "max": max(v) if v else None}
    return {"exchanges": len(recs), "sessions": len({r.get("session") for r in recs}), "first_token_s": stat(first),
            "response_s": stat(total), "prefill_tok_s": stat(pre), "write_tok_s": stat(wr), "hash_ok": ok, "hash_bad": bad,
            "hash_missing": len(recs) - ok - bad, "per_day": days, "rows": rows}


# ── the carrier: bankml serve ─────────────────────────────────────────────────────
def serve_status():
    try:
        with urllib.request.urlopen(SERVE + "/bankml", timeout=3) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": f"bankml serve not reachable at {SERVE}: {e}"}


def build_messages(system, turns, question, qwen3):
    msgs = [{"role": "system", "content": system}]
    for u, a in turns[-KEEP_EXCHANGES:]:
        msgs += [{"role": "user", "content": u[:KEEP_CHARS]}, {"role": "assistant", "content": (a or "")[:KEEP_CHARS]}]
    msgs.append({"role": "user", "content": question + (" /no_think" if qwen3 else "")})
    return msgs


def stream(messages, max_tokens, temperature):
    """OpenAI SSE through bankml serve; yields (text, receipt-or-None)."""
    body = {"messages": messages, "stream": True, "max_tokens": int(max_tokens), "temperature": float(temperature),
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(SERVE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    acc, receipt = "", None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    ev = json.loads(line[5:])
                except ValueError:
                    continue
                if "bankml_receipt" in ev:
                    receipt = ev["bankml_receipt"]
                    continue
                for ch in ev.get("choices") or []:
                    acc += (ch.get("delta") or {}).get("content") or ""
                yield acc, None
    except urllib.error.HTTPError as e:
        yield f"bankml serve refused: {e.read().decode('utf-8', 'replace')[:500]}", {"error": True}
        return
    except urllib.error.URLError as e:
        yield (f"Cannot reach bankml serve at `{SERVE}` ({e.reason}).\n\nStart it:\n\n```sh\n"
               "bankml serve .models/Bonsai-8B-Q1_0.gguf --fork FORK.json --upstream http://127.0.0.1:18092\n```"), {"error": True}
        return
    yield acc, receipt or {}


def show_answer(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def receipt_line(rc: dict, text: str) -> str:
    if not rc or rc.get("error"):
        return ""
    bits = [f"{rc.get('prompt_tokens', 0)} prompt + {rc.get('completion_tokens', 0)} completion tokens"]
    if rc.get("wall_ms") is not None:
        bits.append(f"{rc['wall_ms'] / 1000:.1f} s")
    if rc.get("ttft_ms") is not None:
        bits.append(f"first token {rc['ttft_ms'] / 1000:.1f} s")
    ans = rc.get("response_sha256")
    match = ans and ans == sha256(text.encode())
    bits.append(f"answer sha256 {ans[:12]}…{' ✓' if match else ' (≠ received!)'}" if ans else "no answer hash")
    return (f"draft · not a finding — receipt: bankml {rc.get('bankml', '?')} · {rc.get('engine', '?')} · "
            f"model sha256 {str(rc.get('model_sha256', '?'))[:12]}… · " + " · ".join(bits))


# ── the page ─────────────────────────────────────────────────────────────────────
def office_md(canon: Canon) -> str:
    p, card = canon.persona, canon.card
    beliefs = [b.get("belief", "") for b in ((p.get("bdi") or {}).get("beliefs") or [])[:8] if isinstance(b, dict)]
    ok = sum(1 for r in canon.rows if r[2])
    status = (card.get("savante") or {}).get("status", "?")
    lines = [f"### {p.get('name', 'Savante')} — {p.get('kind', '')}",
             f"*{p.get('mantra', '')}*", "", f"**Oath.** {p.get('oath', '')}", "",
             f"**Card:** {card.get('type', '?')} · status `{status}` (DEFER on minting; no mint here) · "
             f"doctrine root `{(canon.ledger.get('doctrine_root') or {}).get('value', '?')[:18]}…` · "
             f"persona digest `{((canon.ledger.get('artifacts') or {}).get('identity') or {}).get('sha256', '?')[:18]}…` "
             "(on-chain slots unwritten)",
             f"**Ledger:** {ok}/{len(canon.rows)} files verify (sha256 against `savante.commitments.json`)", ""]
    lines += [f"- {b}" for b in beliefs]
    return "\n".join(lines)


def integrity_md(canon: Canon) -> str:
    rows = ["| ledger entry | path | check |", "|---|---|---|"]
    rows += [f"| {n} | `{p}` | {'✓' if ok else '✗'} {d} |" for n, p, ok, d in canon.rows]
    return "\n".join(rows)


# ── view mode: watching the testing ──────────────────────────────────────────────
def live_tail(n=60) -> str:
    try:
        lines = LIVE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"(no live log yet at {LIVE})"
    return "\n".join(lines[-n:])


def live_state() -> dict:
    """{running, title, step, age_s} of the testing log."""
    try:
        lines = LIVE.read_text(encoding="utf-8", errors="replace").splitlines()
        age = time.time() - LIVE.stat().st_mtime
    except OSError:
        return {"running": False, "title": "no testing log yet", "step": "—", "age_s": None}
    heads = [l for l in lines if l.startswith("# bankml")][-1:]
    step = [l for l in lines if l.startswith("## ")]
    running = age < 90 and not any("gate exit" in l or "gate done" in l for l in lines[-3:])
    return {"running": running, "title": heads[0].lstrip("# ") if heads else "", "step": step[-1][3:] if step else "—", "age_s": int(age)}


def live_stage() -> str:
    st = live_state()
    state = "**running**" if st["running"] else ("**idle**" + (f" (last write {st['age_s'] // 60} min ago)" if st["age_s"] is not None else ""))
    return f"{state} · {st['title']} · current step: `{st['step']}`"


def results_list():
    d = REPO / "testing" / "results"
    return sorted((p.name for p in d.glob("*.txt")), key=lambda n: [int(x) if x.isdigit() else x for x in re.split(r"[.]", n)], reverse=True) if d.is_dir() else []


def results_read(name):
    if not name or "/" in name:
        return ""
    try:
        return (REPO / "testing" / "results" / name).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return str(e)


_CI = {"t": 0.0, "runs": [], "error": None}


def ci_runs() -> dict:
    """The last five CI runs of cryptoAGI/bankml (public API, cached 60 s)."""
    if time.time() - _CI["t"] >= 60:
        try:
            req = urllib.request.Request(CI_API, headers={"Accept": "application/vnd.github+json", "User-Agent": "bankml-ui"})
            with urllib.request.urlopen(req, timeout=5) as r:
                runs = json.loads(r.read()).get("workflow_runs", [])
            _CI.update(t=time.time(), error=None, runs=[{"title": x.get("display_title", "")[:90], "result": x.get("conclusion") or x.get("status"),
                                                           "when": x.get("created_at", ""), "sha": (x.get("head_sha") or "")[:7]} for x in runs])
        except Exception as e:
            _CI.update(t=time.time(), error=str(e))
    return {"runs": _CI["runs"], "error": _CI["error"]}


def ci_status() -> str:
    c = ci_runs()
    if c["error"] and not c["runs"]:
        return f"CI status unavailable: {c['error']}"
    rows = ["| CI run | result | when |", "|---|---|---|"]
    return "\n".join(rows + [f"| {x['title'][:70]} | {x['result']} | {x['when']} |" for x in c["runs"]])


def machine_state() -> dict:
    try:
        la = open("/proc/loadavg").read().split()[:3]
        mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}
        cpu = open("/proc/cpuinfo").read().split("model name")[1].split(":")[1].split("\n")[0].strip()
        return {"cpu": cpu, "threads": os.cpu_count(), "load": [float(x) for x in la],
                "ram_available_gb": round(mem["MemAvailable"] / 1048576, 1), "ram_total_gb": round(mem["MemTotal"] / 1048576, 1),
                "swap_used_mb": round((mem["SwapTotal"] - mem["SwapFree"]) / 1024),
                "swap_full": bool(mem["SwapTotal"] and mem["SwapFree"] < mem["SwapTotal"] * 0.05)}
    except Exception as e:
        return {"error": str(e)}


def machine() -> str:
    m = machine_state()
    if "error" in m:
        return f"machine status unavailable: {m['error']}"
    warn = " · **swap full: measurements may be disturbed**" if m["swap_full"] else ""
    return (f"{m['cpu']} · {m['threads']} threads · load {' '.join(map(str, m['load']))} · RAM available {m['ram_available_gb']} "
            f"of {m['ram_total_gb']} GB · swap used {m['swap_used_mb']} MB{warn}")


def carrier_md() -> str:
    """bankml serve's verification as a compact card that wraps inside the side column."""
    st = serve_status()
    if "error" in st:
        return f"<div class='bk-card bk-bad'><b>bankml serve</b><br>not reachable — start it (see usage.md)<br><code>{SERVE}</code></div>"
    v = st.get("verified") or {}
    sha = v.get("model_sha256", "")
    return ("<div class='bk-card'><b>bankml serve</b> <span class='bk-ok'>● verified · play</span>"
            f"<dl><dt>model</dt><dd>{Path(st.get('model', '')).name}</dd>"
            f"<dt>sha256</dt><dd class='bk-mono'>{sha}</dd>"
            f"<dt>bankml</dt><dd>{v.get('bankml', '?')} · guard {v.get('guard', '?')}</dd>"
            f"<dt>engine</dt><dd>{st.get('engine', '?')}</dd></dl></div>")


# ── the timer: from the press of Send to the last token ──────────────────────────
PENDING: dict = {"t0": None, "first": None}


def timer_md() -> str:
    t0, first = PENDING["t0"], PENDING["first"]
    if t0 is None:
        return "<div class='bk-timer bk-idle'>⏱ ready</div>"
    el = time.time() - t0
    phase = f"writing · first token at {first - t0:.1f} s" if first else "reading the prompt (prefill)"
    return f"<div class='bk-timer bk-live'>⏱ {el:5.1f} s · {phase}</div>"


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}" + time.strftime("%z", time.localtime(t))


CSS = """
.gradio-container{max-width:1320px!important}
#bk-head{border:1px solid #0b1220;border-left:5px solid #d9a23a;border-radius:10px;padding:14px 18px;background:#0f172a}
#bk-head, #bk-head *{color:#e2e8f0!important}#bk-head h2{margin:0 0 4px 0;color:#ffffff!important}
#bk-head strong{color:#fbbf24!important}
#bk-chat{border:1px solid #94a3b8!important;border-radius:10px!important;box-shadow:0 1px 2px rgba(15,23,42,.08)}
#bk-side{border:1px solid #94a3b8;border-radius:10px;padding:10px;background:#f8fafc;box-shadow:0 1px 2px rgba(15,23,42,.08)}
.bk-card{border:1px solid #cbd5e1;border-left:4px solid #0f766e;border-radius:8px;padding:10px 12px;background:#ffffff;font-size:13px;
 overflow-wrap:anywhere;word-break:break-word;max-width:100%}
.bk-card, .bk-card *{color:#0f172a!important}
.bk-card dl{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;margin:8px 0 0 0}.bk-card dt{color:#475569!important}.bk-card dd{margin:0;min-width:0}
.bk-mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.bk-card .bk-ok{color:#15803d!important;font-weight:700}.bk-bad{border-left-color:#b91c1c!important}.bk-bad b{color:#b91c1c!important}
.bk-timer{border:1px solid #cbd5e1;border-radius:8px;padding:9px 12px;font:700 16px ui-monospace,Menlo,Consolas,monospace;text-align:center;background:#ffffff}
.bk-live{border:2px solid #0f766e;color:#0f766e!important;background:#f0fdfa}.bk-idle{color:#475569!important}
#bk-prov, #bk-prov *{font-size:12px;color:#334155!important;overflow-wrap:anywhere}
#bk-row{flex-wrap:nowrap;align-items:flex-start;position:relative}
#bk-side,#bk-chat{position:relative}
#bk-side.bk-sized{flex:none!important}
.bk-grip{display:flex;align-items:center;gap:8px;cursor:grab;user-select:none;font:600 11px/1 ui-sans-serif,system-ui;letter-spacing:.08em;
 text-transform:uppercase;color:#64748b;padding:6px 8px;margin:-2px 0 8px;border-radius:8px;border:1px solid transparent;transition:all .15s}
.bk-grip:hover{border-color:#cbd5e1;color:#0f766e;background:rgba(15,118,110,.05)}.bk-grip:active{cursor:grabbing}
.bk-grip svg{flex:none;opacity:.8}.bk-grip .bk-sp{flex:1}
.bk-swap{border:1px solid #cbd5e1;background:transparent;border-radius:6px;padding:2px 7px;font:600 12px ui-sans-serif,system-ui;color:#475569;cursor:pointer}
.bk-swap:hover{border-color:#0f766e;color:#0f766e}
.bk-rz{position:absolute;right:3px;bottom:3px;width:16px;height:16px;cursor:nwse-resize;z-index:5;opacity:.45;transition:opacity .15s;touch-action:none}
.bk-rz:hover,.bk-rz.on{opacity:1}.bk-rz svg{display:block}
.bk-zone{position:fixed;z-index:9999;border:2px dashed rgba(15,118,110,.55);border-radius:12px;background:rgba(15,118,110,.06);
 display:flex;align-items:center;justify-content:center;font:600 13px ui-sans-serif,system-ui;letter-spacing:.06em;color:#0f766e;transition:background .12s}
.bk-zone.hot{background:rgba(15,118,110,.18);border-style:solid}
.dark .bk-grip{color:#94a3b8}.dark .bk-grip:hover{border-color:rgba(148,163,184,.35);color:#2dd4bf;background:rgba(45,212,191,.06)}
.dark .bk-swap{border-color:rgba(148,163,184,.35);color:#cbd5e1}.dark .bk-zone{color:#2dd4bf;border-color:rgba(45,212,191,.55);background:rgba(45,212,191,.06)}
/* the aivatar and its card */
.bk-av-t,input.bk-av-t[type=checkbox]{position:absolute!important;opacity:0!important;width:1px!important;height:1px!important;margin:0!important;pointer-events:none!important;border:0!important}
.bk-av-pic{display:block;position:relative;cursor:pointer;border-radius:12px;overflow:hidden;border:1px solid #cbd5e1;transition:transform .2s,box-shadow .2s}
.bk-av-pic img,.bk-av-pic .bk-glyph{display:block;width:100%;height:230px;object-fit:cover}
.bk-av-pic:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(15,118,110,.25)}
.bk-av-cap{position:absolute;left:0;right:0;bottom:0;padding:18px 10px 8px;font:600 12px ui-sans-serif,system-ui;letter-spacing:.06em;color:#fff!important;
 background:linear-gradient(transparent,rgba(2,6,23,.85))}
.bk-glyph{display:flex!important;align-items:center;justify-content:center;font:700 64px ui-sans-serif,system-ui;color:#2dd4bf!important;
 background:radial-gradient(circle at 30% 30%,#134e4a,#020617)}
.bk-modal{display:none;position:fixed;inset:0;z-index:10000;align-items:center;justify-content:center;padding:24px}
.bk-av-t:checked ~ .bk-modal{display:flex}
.bk-modal-bg{position:absolute;inset:0;background:rgba(2,6,23,.72);backdrop-filter:blur(4px);cursor:zoom-out}
.bk-holo{position:relative;max-width:920px;width:100%;max-height:88vh;overflow:auto;border-radius:16px;padding:22px 24px 16px;color:#e2e8f0;
 background:linear-gradient(rgba(45,212,191,.05) 1px,transparent 1px) 0 0/100% 22px,linear-gradient(90deg,rgba(45,212,191,.05) 1px,transparent 1px) 0 0/22px 100%,
 linear-gradient(160deg,rgba(15,23,42,.97),rgba(2,6,23,.97));border:1px solid rgba(45,212,191,.45);
 box-shadow:0 0 0 1px rgba(217,162,58,.25),0 0 40px rgba(45,212,191,.18),inset 0 0 60px rgba(45,212,191,.05);animation:bk-glow 4s ease-in-out infinite}
.bk-holo{scrollbar-width:thin;scrollbar-color:rgba(45,212,191,.45) transparent}
.bk-holo::-webkit-scrollbar{width:8px}.bk-holo::-webkit-scrollbar-track{background:transparent}
.bk-holo::-webkit-scrollbar-thumb{background:rgba(45,212,191,.4);border-radius:8px}
@keyframes bk-glow{50%{box-shadow:0 0 0 1px rgba(217,162,58,.4),0 0 60px rgba(45,212,191,.28),inset 0 0 60px rgba(45,212,191,.07)}}
@media (prefers-reduced-motion:reduce){.bk-holo{animation:none}}
.bk-holo *{color:#e2e8f0}
.bk-x{position:absolute;top:12px;right:14px;cursor:pointer;font-size:18px;color:#94a3b8!important;border:1px solid rgba(148,163,184,.3);border-radius:8px;padding:2px 9px}
.bk-x:hover{color:#2dd4bf!important;border-color:#2dd4bf}
.bk-holo-top{display:flex;gap:18px;align-items:center}
.bk-holo-pic{flex:none;width:120px;height:120px;border-radius:50%;overflow:hidden;border:2px solid #d9a23a;box-shadow:0 0 24px rgba(217,162,58,.35)}
.bk-holo-pic img,.bk-holo-pic .bk-glyph{width:120px;height:120px;object-fit:cover;font-size:40px}
.bk-kicker{font:600 11px ui-monospace,Menlo,monospace;letter-spacing:.2em;text-transform:uppercase;color:#2dd4bf!important}
.bk-holo h3{margin:4px 0 2px;font:700 28px ui-sans-serif,system-ui;letter-spacing:.02em;color:#fff!important}
.bk-mantra{font-style:italic;color:#fbbf24!important}
.bk-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.bk-chip{font:11px ui-monospace,Menlo,monospace;padding:2px 8px;border-radius:999px;border:1px solid rgba(45,212,191,.35);color:#99f6e4!important;background:rgba(45,212,191,.06)}
.bk-desc{margin:16px 0 10px;line-height:1.55;color:#cbd5e1!important}
.bk-cols{display:grid;grid-template-columns:1fr 1.15fr;gap:22px}@media (max-width:760px){.bk-cols{grid-template-columns:1fr}}
.bk-h{font:600 11px ui-monospace,Menlo,monospace;letter-spacing:.18em;text-transform:uppercase;color:#d9a23a!important;margin:10px 0 4px;
 border-bottom:1px solid rgba(217,162,58,.25);padding-bottom:3px}
.bk-holo p,.bk-holo li{font-size:13px;line-height:1.5;color:#cbd5e1!important}.bk-holo ul{margin:4px 0;padding-left:18px}
.bk-hash{display:grid;grid-template-columns:auto 1fr;gap:6px 12px;margin:6px 0}
.bk-hash dt{font:600 11px ui-monospace,Menlo,monospace;color:#94a3b8!important;white-space:nowrap}
.bk-hash dd{margin:0;font:12px ui-monospace,Menlo,monospace;color:#5eead4!important;overflow-wrap:anywhere}
.bk-aspects details{border:1px solid rgba(45,212,191,.22);border-radius:10px;margin:6px 0;background:rgba(45,212,191,.03)}
.bk-aspects summary{cursor:pointer;padding:7px 12px;font:600 12px ui-monospace,Menlo,monospace;letter-spacing:.12em;text-transform:uppercase;color:#5eead4!important}
.bk-aspects summary:hover{background:rgba(45,212,191,.07)}.bk-aspects details>div{padding:4px 14px 10px}
.bk-asp{margin:2px 0;padding-left:16px}dl.bk-asp{display:grid;grid-template-columns:minmax(90px,auto) 1fr;gap:3px 12px;padding-left:0}
dl.bk-asp>dt{font:600 11px ui-monospace,Menlo,monospace;color:#94a3b8!important}dl.bk-asp>dd{margin:0;font-size:12.5px;line-height:1.45;color:#cbd5e1!important;overflow-wrap:anywhere}
ol.bk-asp>li{font-size:12.5px;line-height:1.45;color:#cbd5e1!important;margin:2px 0}
.bk-sp{white-space:pre-wrap;font:12px/1.5 ui-monospace,Menlo,monospace;color:#cbd5e1!important;max-height:40vh;overflow:auto;margin:0}
.bk-ft{width:100%;border-collapse:collapse;font-size:12px}.bk-ft th,.bk-ft td{border-bottom:1px solid rgba(148,163,184,.18);padding:4px 6px;text-align:left}
.bk-ft th{font:600 11px ui-monospace,Menlo,monospace;color:#94a3b8!important}.bk-ft td.bk-mono{color:#5eead4!important;overflow-wrap:anywhere}
.bk-foot{margin-top:14px;padding-top:8px;border-top:1px solid rgba(148,163,184,.2);font:11px ui-monospace,Menlo,monospace;letter-spacing:.06em;color:#64748b!important}
button.primary,button.lg.primary{background:#0f766e!important;border-color:#0f766e!important;color:#fff!important}
.tabs button.selected{border-bottom:3px solid #0f766e!important;font-weight:700}
#ragebar textarea,#ragebar input{background:#05070a!important;color:#e9ffe9!important;border:1px solid #1f3a2b!important;border-radius:8px!important;
 font:15px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace!important;caret-color:#3ddc84;padding:14px 16px!important}
#ragebar textarea:focus,#ragebar input:focus{border-color:#3ddc84!important;box-shadow:0 0 0 3px rgba(61,220,132,.18)!important}
#ragebar label span{color:#0f766e!important;font-weight:700;letter-spacing:.12em;text-transform:lowercase}
.rb-stat{font:12px ui-monospace,Menlo,Consolas,monospace;color:#334155;margin:6px 2px}.rb-stat.rb-hot{color:#b91c1c}
.rb-meter{height:4px;border-radius:2px;background:#e2e8f0;overflow:hidden;margin:0 0 6px}.rb-meter i{display:block;height:100%;background:#16a34a}
.bk-ex{border:1px solid #cbd5e1;border-radius:8px;padding:10px 12px;margin:8px 0;background:#fff}
.bk-ex, .bk-ex *{color:#0f172a}
.bk-exh{font:12px ui-monospace,Menlo,Consolas,monospace;color:#475569!important;margin-bottom:6px;overflow-wrap:anywhere}
.bk-q{font-weight:700;margin-bottom:6px;white-space:pre-wrap}.bk-a{white-space:pre-wrap;line-height:1.5}
.bk-okb{color:#15803d!important;font-weight:700}.bk-badb{color:#b91c1c!important;font-weight:700}.bk-dimb{color:#64748b!important}
.bk-note{font-size:12px;color:#475569!important;margin:6px 2px}
.bk-t{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}.bk-t th,.bk-t td{border:1px solid #cbd5e1;padding:5px 8px;text-align:left;color:#0f172a}
.bk-t th{background:#f1f5f9}.bk-spark{border:1px solid #cbd5e1;border-radius:8px;background:#fff}.bk-spark rect{fill:#0f766e}
#bk-nav button{min-width:0}
/* dark mode: transparent surfaces over the page, light text and borders — never white panels */
.dark #bk-side,.dark .bk-card,.dark .bk-ex,.dark .bk-timer,.dark .bk-grip,.dark .bk-spark,.dark .bk-t th,.dark .bk-t td{background:transparent!important}
.dark #bk-side{border-color:rgba(148,163,184,.35)!important;box-shadow:none}
.dark .bk-card,.dark .bk-ex,.dark .bk-timer,.dark .bk-grip,.dark .bk-spark,.dark .bk-t th,.dark .bk-t td{border-color:rgba(148,163,184,.35)!important}
.dark .bk-card,.dark .bk-card *,.dark .bk-ex,.dark .bk-ex *,.dark .bk-t th,.dark .bk-t td,.dark .bk-grip{color:#e2e8f0!important}
.dark .bk-card dt,.dark .bk-exh,.dark .bk-note,.dark #bk-prov,.dark #bk-prov *,.dark .bk-idle,.dark .rb-stat{color:#94a3b8!important}
.dark .bk-card .bk-ok,.dark .bk-okb{color:#4ade80!important}.dark .bk-badb,.dark .bk-bad b{color:#f87171!important}
.dark .bk-live{border-color:#2dd4bf!important;color:#2dd4bf!important;background:rgba(45,212,191,.08)!important}
.dark .bk-t th{background:rgba(148,163,184,.08)!important}.dark .rb-meter{background:rgba(148,163,184,.2)}
.dark .bk-spark rect{fill:#2dd4bf}
"""


# ── renderers for the .history, Responses, .memory and Metrics tabs (all text escaped) ──
E = html.escape


def _fmt(v, unit="s", nd=1):
    return "—" if v is None else f"{v:.{nd}f} {unit}".strip()


def _exchange_html(i: int, r: dict, n: int, score: float | None = None, top: float = 1.0) -> str:
    ft, rs, src = _timing(r)
    h = _hash_ok(r)
    rc = r.get("receipt") or {}
    meter = f"<div class='rb-meter'><i style='width:{max(4, int(100 * score / top))}%'></i></div>" if score is not None else ""
    badge = {True: "<span class='bk-okb'>answer sha256 ✓</span>", False: "<span class='bk-badb'>answer sha256 ≠ receipt</span>",
             None: "<span class='bk-dimb'>no receipt hash</span>"}[h]
    return (f"<div class='bk-ex'>{meter}<div class='bk-exh'>#{i + 1} of {n} · {E(_when(r))} · session {E(str(r.get('session', '?')))}"
            f" · first token {_fmt(ft)} · answered {_fmt(rs)}{' (receipt)' if src == 'receipt' else ''}"
            f" · {rc.get('prompt_tokens', '?')}+{rc.get('completion_tokens', '?')} tokens · {badge}</div>"
            f"<div class='bk-q'>{E(r.get('user') or '')}</div><div class='bk-a'>{E(r.get('assistant') or '')}</div></div>")


def history_html() -> str:
    recs = history_all()
    if not recs:
        return f"<div class='bk-card'>No exchanges yet. They are written to <code>{E(str(HISTORY))}</code>.</div>"
    return (f"<div class='bk-note'>{len(recs)} exchanges · newest first · <code>{E(str(HISTORY))}</code></div>"
            + "".join(_exchange_html(i, r, len(recs)) for i, r in reversed(list(enumerate(recs)))))


def search_html(q: str) -> str:
    if not q.strip():
        return "<div class='rb-stat'>type to search every question and answer in .history</div>"
    hits, engine = history_search(q)
    n = len(history_all())
    if not hits:
        return f"<div class='rb-stat rb-hot'>no match for “{E(q)}” · {E(engine)} over {n} exchanges</div>"
    top = hits[0][0] or 1.0
    return (f"<div class='rb-stat'>{len(hits)} of {n} exchanges · {E(engine)}</div>"
            + "".join(_exchange_html(i, r, n, sc, top) for sc, i, r in hits))


def response_at(k: int):
    """(clamped index, html, plain answer) for the k-th response (0 = oldest)."""
    recs = history_all()
    if not recs:
        return 0, "<div class='bk-card'>No responses yet.</div>", ""
    k = max(0, min(int(k), len(recs) - 1))
    return k, _exchange_html(k, recs[k], len(recs)), recs[k].get("assistant") or ""


def memory_html() -> str:
    m = memory_all()
    if not m:
        return f"<div class='bk-card'>.memory is empty. Save a response from <b>Responses</b>, or type a note. File: <code>{E(str(MEMORY))}</code></div>"
    rows = "".join(f"<div class='bk-ex'><div class='bk-exh'>note {j + 1} · {E(x.get('at', ''))} · "
                   f"{E((x.get('source') or {}).get('kind', ''))} {E(str((x.get('source') or {}).get('sent_at', '') or ''))}</div>"
                   f"<div class='bk-a'>{E(x['text'])}</div></div>" for j, x in enumerate(m))
    blk = memory_block()
    return (f"<div class='bk-note'>{len(m)} notes · {len(blk)} of {MEMORY_BUDGET} characters go into the prompt when "
            f".memory is on · <code>{E(str(MEMORY))}</code></div>" + rows)


def metrics_html() -> str:
    m = metrics()
    if not m["exchanges"]:
        return "<div class='bk-card'>No exchanges in .history yet.</div>"
    def row(name, st, unit, nd=1):
        return (f"<tr><td>{name}</td><td>{st['n']}</td><td>{_fmt(st['median'], unit, nd)}</td><td>{_fmt(st['p90'], unit, nd)}</td>"
                f"<td>{_fmt(st['mean'], unit, nd)}</td><td>{_fmt(st['min'], unit, nd)}</td><td>{_fmt(st['max'], unit, nd)}</td></tr>")
    tot = [x["total"] for x in m["rows"] if x["total"] is not None]
    w, hgt = 640, 90
    bars = ""
    if tot:
        mx = max(tot) or 1
        bw = w / len(tot)
        bars = "".join(f"<rect x='{j * bw + 1:.1f}' y='{hgt - v / mx * (hgt - 6):.1f}' width='{max(bw - 2, 1):.1f}' height='{v / mx * (hgt - 6):.1f}' rx='2'><title>#{j + 1}: {v:.1f} s</title></rect>"
                       for j, v in enumerate(tot))
    recent = "".join(f"<tr><td>{E(x['when'])}</td><td>{_fmt(x['first'])}</td><td>{_fmt(x['total'])}</td><td>{x['src'] or '—'}</td>"
                     f"<td>{x['pt'] or '—'}+{x['ct'] or '—'}</td><td>{'✓' if x['hash'] else ('✗' if x['hash'] is False else '—')}</td>"
                     f"<td>{E(x['q'])}</td></tr>" for x in reversed(m["rows"][-25:]))
    days = " · ".join(f"{d}: {c}" for d, c in sorted(m["per_day"].items()))
    ch, cm = commitment(HISTORY), commitment(MEMORY)
    proofs = (f"<div class='bk-card'><b>Commitments</b> (shareable; reveal no content) · .history: {ch['records']} records, "
              f"Merkle root <span class='bk-mono'>{ch['merkle_root']}</span>, CID <span class='bk-mono'>{ch['file_cid']}</span>"
              f" · .memory: {cm['records']} notes, root <span class='bk-mono'>{cm['merkle_root']}</span></div>")
    return (f"<div class='bk-card'><b>{m['exchanges']} exchanges · {m['sessions']} sessions</b> · answer hash matches its receipt "
            f"in {m['hash_ok']}, differs in {m['hash_bad']}, no receipt in {m['hash_missing']} · per day: {E(days)}</div>"
            "<table class='bk-t'><tr><th>measure</th><th>n</th><th>median</th><th>p90</th><th>mean</th><th>min</th><th>max</th></tr>"
            + row("time to first token (from Send)", m["first_token_s"], "s") + row("response time (from Send)", m["response_s"], "s")
            + row("prompt reading (prefill)", m["prefill_tok_s"], "tok/s", 2) + row("writing", m["write_tok_s"], "tok/s", 2) + "</table>"
            f"<div class='bk-note'>response time per exchange, oldest → newest (hover for the value)</div>"
            f"<svg class='bk-spark' viewBox='0 0 {w} {hgt}' width='100%' height='{hgt}' preserveAspectRatio='none'>{bars}</svg>"
            "<div class='bk-note'>Times are measured from the press of Send where recorded (ui, 0.0.7+); older exchanges use the "
            "receipt's gateway times. Prefill = prompt tokens ÷ time to first token; writing = completion tokens ÷ the rest.</div>"
            "<table class='bk-t'><tr><th>sent</th><th>first token</th><th>answered</th><th>source</th><th>tokens</th><th>hash</th><th>question</th></tr>"
            + recent + "</table>" + proofs)


def _aspect(v, depth=0) -> str:
    """Any persona value as escaped HTML: objects as definition lists, lists as lists, text as text."""
    if isinstance(v, dict):
        if not v:
            return "<i>—</i>"
        return "<dl class='bk-asp'>" + "".join(f"<dt>{E(str(k))}</dt><dd>{_aspect(x, depth + 1)}</dd>" for k, x in v.items()) + "</dl>"
    if isinstance(v, list):
        if not v:
            return "<i>—</i>"
        return "<ol class='bk-asp'>" + "".join(f"<li>{_aspect(x, depth + 1)}</li>" for x in v) + "</ol>"
    if v is None:
        return "<i>null</i>"
    return E(str(v))


IMG_KINDS = {b"\x89PNG\r\n\x1a\n": "png", b"\xff\xd8\xff": "jpeg", b"RIFF": "webp"}


def image_kind(b: bytes):
    for magic, kind in IMG_KINDS.items():
        if b.startswith(magic) and (kind != "webp" or b[8:12] == b"WEBP"):
            return kind
    return None


def savante_images() -> list:
    d = CANON / "gfx"
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")) if d.is_dir() else []


def chosen_image(canon: Canon):
    """(bytes, kind, label) of the portrait in use: a custom agent's own x-bankml.aivatar, or Savante's chosen canon image."""
    slug = ACTIVE["slug"]
    if slug:
        import agents
        for ext in ("png", "jpeg", "webp"):
            q = agents.agent_dir(slug) / f"{slug}.aivatar.{ext}"
            if q.is_file():
                b = q.read_bytes()
                return b, image_kind(b), q.name
        return None, None, None
    pref = STATE / "savante.aivatar"
    name = pref.read_text(encoding="utf-8").strip() if pref.is_file() else ""
    rel = f"gfx/{name}" if name in savante_images() else ((canon.ledger.get("image_candidate") or {}).get("path") or "gfx/Savante3.png")
    b = canon.file(rel)
    return (b, image_kind(b) if b else None, rel)


# ── the aivatar: click the portrait for the agent's card (pure CSS toggle; the image is embedded, never served) ──
def aivatar_html(canon: Canon) -> str:
    import base64
    slug = ACTIVE["slug"]
    if slug:
        import agents
        f = agents.files(slug)
        p = json.loads(f["persona"].read_text(encoding="utf-8"))
        card = json.loads(f["agentcard.json"].read_text(encoding="utf-8"))
        led = json.loads(f["commitments.json"].read_text(encoding="utf-8"))
        tp = agents.agent_dir(slug) / f"{slug}.thot.json"
        man = json.loads(tp.read_text(encoding="utf-8")) if tp.is_file() else {}
        ledger_ok = f"{sum(ok for _, ok, _ in agents.verify(slug))}/{len(agents.verify(slug))}"
        persona_sha, droot = led["artifacts"]["persona"]["sha256"], led["doctrine_root"]["value"]
        status = (card.get("bankml") or {}).get("status", "not_yet_minted")
        arts = [(k, a["path"], a["bytes"], a["sha256"]) for k, a in led["artifacts"].items()] + [("card", led["card"]["path"], led["card"]["bytes"], led["card"]["sha256"])]
    else:
        p, card, led = canon.persona, canon.card, canon.ledger
        try:
            man = json.loads((CANON / "savante.thot.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            man = {}
        ledger_ok = f"{sum(r[2] for r in canon.rows)}/{len(canon.rows)}"
        persona_sha = ((led.get("artifacts") or {}).get("identity") or {}).get("sha256", "")
        droot = (led.get("doctrine_root") or {}).get("value", "")
        status = (card.get("savante") or {}).get("status", "?")
        arts = [(k, a.get("path"), a.get("bytes"), a.get("sha256")) for k, a in (led.get("artifacts") or {}).items()]
    name = p.get("name", "agent")
    img, kind, img_label = chosen_image(canon)
    pic = (f"<img src='data:image/{kind};base64,{base64.b64encode(img).decode()}' alt='{E(name)}'>" if img and kind
           else f"<div class='bk-glyph'>{E(''.join(w[0] for w in name.split()[:2]).upper())}</div>")
    ident = man.get("identity") or {}
    beliefs = [b.get("belief", "") for b in ((p.get("bdi") or {}).get("beliefs") or [])[:5] if isinstance(b, dict)]
    attrs = "".join(f"<span class='bk-chip'>{E(str(a.get('trait_type')))}: {E(str(a.get('value')))}</span>" for a in (card.get("attributes") or [])[:8])
    facets = " · ".join(E(x.get("facet", "")) for x in (man.get("facets") or []))
    rows = [("persona sha256", persona_sha), ("doctrine root", droot), ("THOT", ident.get("thot", "—")), ("THOT CID", ident.get("cid", "—")),
            ("contentRoot", ident.get("contentRoot", "—")), ("generation", str((man.get("bundle") or {}).get("generation", "—"))),
            ("facets", facets or "—"), ("ledger", f"{ledger_ok} verify · status {status}")]
    dl = "".join(f"<dt>{E(k)}</dt><dd>{E(v) if k != 'facets' else v}</dd>" for k, v in rows)
    shown = {"persona", "name", "system_prompt"}  # name is the title; the system prompt gets its own section below
    aspects = "".join(f"<details{' open' if k in ('bdi', 'skills') else ''}><summary>{E(k)}</summary><div>{_aspect(v)}</div></details>"
                      for k, v in p.items() if k not in shown)
    aspects += f"<details><summary>system_prompt</summary><div><pre class='bk-sp'>{E(p.get('system_prompt', ''))}</pre></div></details>"
    files = "".join(f"<tr><td>{E(str(k))}</td><td>{E(str(pth))}</td><td>{E(str(n))}</td><td class='bk-mono'>{E(str(h))}</td></tr>" for k, pth, n, h in arts)
    return f"""<div class='bk-av'>
<input type='checkbox' id='bk-av-open' class='bk-av-t'>
<label for='bk-av-open' class='bk-av-pic' title='open {E(name)}'s card'>{pic}<span class='bk-av-cap'>{E(name)} · open the card</span></label>
<div class='bk-modal'><label for='bk-av-open' class='bk-modal-bg'></label>
<div class='bk-holo' role='dialog' aria-label='{E(name)}'>
<label for='bk-av-open' class='bk-x' title='close'>✕</label>
<div class='bk-holo-top'><div class='bk-holo-pic'>{pic}</div><div>
<div class='bk-kicker'>{E(p.get('kind', ''))} · {E(card.get('type', '').split('#')[-1] or 'agent')}</div>
<h3>{E(name)}</h3><div class='bk-mantra'>{E(p.get('mantra', ''))}</div>
<div class='bk-chips'>{attrs}</div></div></div>
<p class='bk-desc'>{E(card.get('description') or '')}</p>
<div class='bk-cols'><div><div class='bk-h'>oath</div><p>{E(p.get('oath', ''))}</p>
<div class='bk-h'>office</div><p>primary skill: {E(str((p.get('skills') or {}).get('primary', '—')))}<br>scope: {E(str((p.get('safety') or {}).get('scope', '—')))}</p>
<div class='bk-h'>beliefs</div><ul>{''.join(f'<li>{E(b)}</li>' for b in beliefs)}</ul></div>
<div><div class='bk-h'>identity — verifiable, not asserted</div><dl class='bk-hash'>{dl}</dl></div></div>
<div class='bk-h'>every aspect of the persona</div>
<div class='bk-aspects'>{aspects}</div>
<div class='bk-h'>ledgered files</div>
<table class='bk-ft'><tr><th>facet</th><th>file</th><th>bytes</th><th>sha256</th></tr>{files}</table>
<div class='bk-foot'>portrait: {E(str(img_label or 'none'))} · the canon is read-only · every value above is re-derived from its files · nothing here mints</div>
</div></div></div>"""


# Gradio 3 has no layout API: this runs once in the page. Drag the side panel's grip to either side of the chat;
# drag any panel's corner to resize it. The choice is kept in this browser (localStorage).
LAYOUT_JS = """() => {
  const K = 'bankml-interact-layout-v2';
  const row = document.getElementById('bk-row'), side = document.getElementById('bk-side'), chat = document.getElementById('bk-chat');
  if (!row || !side || side.dataset.bk) return [];
  side.dataset.bk = '1';
  let st = {}; try { st = JSON.parse(localStorage.getItem(K) || '{}') } catch (e) {}
  const save = () => { try { localStorage.setItem(K, JSON.stringify(st)) } catch (e) {} };
  const place = where => { if (where === 'left') row.prepend(side); else row.append(side); st.side = where; save() };
  const svgDots = '<svg width="10" height="16" viewBox="0 0 10 16"><g fill="currentColor">' +
    [2, 8].map(x => [3, 8, 13].map(y => '<circle cx="' + x + '" cy="' + y + '" r="1.4"/>').join('')).join('') + '</g></svg>';
  const svgRz = '<svg width="16" height="16" viewBox="0 0 16 16"><g stroke="currentColor" stroke-width="1.4" stroke-linecap="round">' +
    '<path d="M14 6 6 14"/><path d="M14 10 10 14"/><path d="M14 13.5 13.5 14"/></g></svg>';
  // the grip: drag to move, or the swap button
  const grip = document.createElement('div'); grip.className = 'bk-grip'; grip.draggable = true;
  grip.innerHTML = svgDots + '<span>panel</span><span class="bk-sp"></span>';
  const swap = document.createElement('button'); swap.className = 'bk-swap'; swap.type = 'button'; swap.title = 'move to the other side'; swap.textContent = '⇄';
  swap.addEventListener('click', e => { e.stopPropagation(); place(row.firstElementChild === side ? 'right' : 'left') });
  grip.append(swap); side.prepend(grip);
  // drop zones above everything while dragging, so nothing underneath can swallow the drop
  let zones = [];
  const clear = () => { zones.forEach(z => z.remove()); zones = [] };
  grip.addEventListener('dragstart', e => {
    e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', 'bk-side'); side.style.opacity = .55;
    const r = row.getBoundingClientRect();
    for (const [where, x] of [['left', r.left], ['right', r.left + r.width / 2]]) {
      const z = document.createElement('div'); z.className = 'bk-zone'; z.textContent = 'drop ' + where;
      Object.assign(z.style, { left: (x + 4) + 'px', top: (r.top + 4) + 'px', width: (r.width / 2 - 8) + 'px', height: Math.min(r.height, innerHeight - r.top) - 8 + 'px' });
      z.addEventListener('dragover', ev => { ev.preventDefault(); z.classList.add('hot') });
      z.addEventListener('dragleave', () => z.classList.remove('hot'));
      z.addEventListener('drop', ev => { ev.preventDefault(); place(where); clear() });
      document.body.append(z); zones.push(z);
    }
  });
  grip.addEventListener('dragend', () => { side.style.opacity = 1; clear() });
  // resize handles (a refined corner, the same in every browser)
  const handle = (el, key, axis) => {
    const h = document.createElement('div'); h.className = 'bk-rz'; h.innerHTML = svgRz; h.title = 'drag to resize'; el.append(h);
    h.addEventListener('pointerdown', e => {
      e.preventDefault(); h.setPointerCapture(e.pointerId); h.classList.add('on');
      const r = el.getBoundingClientRect(), x0 = e.clientX, y0 = e.clientY;
      const mv = ev => {
        if (axis !== 'y') { const w = Math.max(260, r.width + (ev.clientX - x0) * (key === 'side' && row.firstElementChild !== side ? -1 : 1)); el.style.width = w + 'px'; el.classList.add('bk-sized') }
        const hh = Math.max(220, r.height + ev.clientY - y0); el.style.height = hh + 'px';
        for (const c of el.querySelectorAll('[style*="height"]')) if (c !== el && c.offsetHeight > 150) c.style.height = (hh - 40) + 'px';
      };
      const up = () => { h.removeEventListener('pointermove', mv); h.classList.remove('on'); st[key] = { w: el.style.width, h: el.style.height }; save() };
      h.addEventListener('pointermove', mv); h.addEventListener('pointerup', up, { once: true });
    });
    const v = st[key];
    if (v) { if (v.w) { el.style.width = v.w; el.classList.add('bk-sized') } if (v.h) el.style.height = v.h }
  };
  handle(side, 'side', 'xy');
  if (chat) handle(chat, 'chat', 'y');
  if (st.side === 'left') row.prepend(side);
  return [];
}"""


def view_tabs(gr, canon: Canon):
    with gr.Tab("Testing (live)"):
        stage = gr.Markdown(live_stage())
        mach = gr.Markdown(machine())
        log = gr.Textbox(value=live_tail(), lines=28, max_lines=28, label=f"live — {LIVE}", interactive=False)
    with gr.Tab("Results"):
        names = results_list()
        pick = gr.Dropdown(names, value=names[0] if names else None, label="release record (testing/results/)")
        body = gr.Textbox(value=results_read(names[0]) if names else "", lines=30, max_lines=40, label="gate record", interactive=False)
        pick.change(results_read, pick, body)
        ci = gr.Markdown(ci_status())
    with gr.Tab("Office"):
        gr.Markdown(office_md(canon))
    with gr.Tab("Integrity"):
        gr.Markdown(integrity_md(canon))
    return stage, mach, log, ci


def build(canon: Canon, mode: str):
    import gradio as gr

    sid0, turns0 = history_load()
    theme = gr.themes.Base(primary_hue="teal", secondary_hue="amber", neutral_hue="slate") if hasattr(gr, "themes") else None
    with gr.Blocks(title="bankml · Savante", css=CSS, theme=theme) as demo:
        session = gr.State({"id": sid0})
        gr.Markdown("## bankml · Savante — verified low-bit inference on this computer\n"
                    "Every answer is a **draft**, carried by a local model behind bankml's guard and sha256 pin, "
                    "with a receipt. Savante's canon is read-only and checked against its ledger.", elem_id="bk-head")
        with gr.Tab("Interaction"):
            with gr.Row(elem_id="bk-row"):
                with gr.Column(scale=3, elem_id="bk-main"):
                    chat = gr.Chatbot(value=turns0, height=540, label="Savante (draft)", elem_id="bk-chat")
                    msg = gr.Textbox(placeholder="Ask Savante — e.g. review: is this model ready to serve?", show_label=False)
                    with gr.Row():
                        send, stop, new = gr.Button("Send", variant="primary"), gr.Button("Stop"), gr.Button("New session")
                with gr.Column(scale=1, elem_id="bk-side"):
                    timer = gr.HTML(timer_md())
                    aiv = gr.HTML(aivatar_html(canon))
                    which = gr.Dropdown(list(PROMPTS), value=PROMPTS[0], label=".prompt")
                    prov = gr.Markdown(system_prompt(canon, PROMPTS[0])[1], elem_id="bk-prov")
                    use_mem = gr.Checkbox(value=True, label="use .memory (the operator's notes, appended to the system prompt)")
                    max_tokens = gr.Slider(16, 1024, value=256, step=16, label="max tokens")
                    temperature = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="temperature")
                    carrier = gr.HTML(carrier_md())
                    gr.Button("refresh carrier").click(carrier_md, None, carrier)
            gr.Markdown(f"`.history` → `{HISTORY}` (outside the canon) · session `{sid0}`")

            def add(m, h):
                if not m.strip():
                    return "", h
                PENDING.update(t0=time.time(), first=None)  # the clock starts at the press of Send
                return "", (h or []) + [[m, None]]

            def respond(h, which, max_tokens, temperature, sess, use_mem):
                if not h or h[-1][1] is not None:
                    yield h
                    return
                system, why = system_prompt(canon, which)
                t0 = PENDING["t0"] or time.time()
                if system is None:
                    h[-1][1] = f"refused: {why}"
                    PENDING.update(t0=None, first=None)
                    yield h
                    return
                mem = memory_block() if use_mem else ""
                if mem:
                    system, why = system + mem, why + f" + .memory ({len(memory_all())} notes, {len(mem)} chars)"
                st = serve_status()
                qwen3 = "qwen3" in json.dumps(st).lower() or "bonsai" in json.dumps(st).lower()
                msgs = build_messages(system, [t for t in h[:-1] if t[1] is not None], h[-1][0], qwen3)
                text, rc = "", {}
                try:
                    for text, r in stream(msgs, max_tokens, temperature):
                        if text and PENDING["first"] is None:
                            PENDING["first"] = time.time()
                        h[-1][1] = show_answer(text)
                        rc = r if r is not None else rc
                        yield h
                    t1, first = time.time(), PENDING["first"]
                finally:
                    PENDING.update(t0=None, first=None)
                answer = show_answer(text)
                timing = {"sent_at": iso(t0), "first_token_s": round(first - t0, 2) if first else None,
                          "response_s": round(t1 - t0, 2), "answered_at": iso(t1)}
                clock = (f"⏱ sent {time.strftime('%H:%M:%S', time.localtime(t0))} · first token "
                         f"{timing['first_token_s'] if first else '—'} s · answered in {timing['response_s']} s")
                foot = receipt_line(rc, text)
                h[-1][1] = answer + f"\n\n<sub>{clock}" + (f"<br>{foot}<br>{why}" if foot else "") + "</sub>"
                history_append({"ts": round(t0, 3), **timing, "agent": ACTIVE["slug"] or "savante", "session": sess["id"], "user": h[-1][0], "assistant": answer,
                                "shown": h[-1][1], "prompt": which, "prompt_provenance": why, "receipt": rc})
                yield h

            ev = msg.submit(add, [msg, chat], [msg, chat]).then(respond, [chat, which, max_tokens, temperature, session, use_mem], chat)
            ev2 = send.click(add, [msg, chat], [msg, chat]).then(respond, [chat, which, max_tokens, temperature, session, use_mem], chat)
            stop.click(lambda: PENDING.update(t0=None, first=None), None, None, cancels=[ev, ev2])
            which.change(lambda w: system_prompt(canon, w)[1], which, prov)
            new.click(lambda: ([], {"id": uuid.uuid4().hex[:12]}), None, [chat, session])
        stage, mach, log, ci = view_tabs(gr, canon)
        demo.load(lambda: (live_stage(), machine(), live_tail(), ci_status()), None, [stage, mach, log, ci], every=2, show_progress=False)
        demo.load(timer_md, None, timer, every=1, show_progress=False)
        demo.load(None, None, None, _js=LAYOUT_JS)
        with gr.Tab("Verifier"):
            out = gr.Textbox(label="bind/savante_verify.py (offline; exit 0 = APPROVE)", lines=14)

            def verify():
                v = CANON / "bind" / "savante_verify.py"
                if not v.is_file():
                    return f"{v} not found"
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                p = subprocess.run([sys.executable, "-B", str(v), str(CANON)], capture_output=True, text=True, env=env, timeout=300)
                return f"exit {p.returncode}\n" + (p.stdout + p.stderr)[-6000:]

            gr.Button("run the verifier").click(verify, None, out)
        with gr.Tab(".history"):
            rq = gr.Textbox(label="ragebar", placeholder="search .history — every question and answer (RAGE)", elem_id="ragebar", lines=1)
            hits = gr.HTML(search_html(""))
            gr.Markdown("---")
            hall = gr.HTML(history_html())
            gr.Button("refresh").click(history_html, None, hall)
            rq.change(search_html, rq, hits, show_progress=False)
        with gr.Tab("Responses"):
            n0 = max(len(history_all()) - 1, 0)
            pos = gr.State(n0)
            with gr.Row(elem_id="bk-nav"):
                b_first, b_up, b_down, b_last = gr.Button("⤒ first"), gr.Button("▲ previous"), gr.Button("▼ next"), gr.Button("⤓ latest")
                b_copy, b_mem = gr.Button("📋 copy"), gr.Button("➕ save to .memory", variant="primary")
                b_proof = gr.Button("🔏 proof")
            where = gr.Markdown()
            card = gr.HTML(response_at(n0)[1])
            plain = gr.Textbox(visible=False)
            said = gr.Markdown()

            def show(k):
                k, htm, txt = response_at(k)
                return k, htm, txt, f"response **{k + 1} of {len(history_all())}**", ""

            outs = [pos, card, plain, where, said]
            b_first.click(lambda: show(0), None, outs)
            b_last.click(lambda: show(10 ** 9), None, outs)
            b_up.click(lambda k: show(k - 1), pos, outs)
            b_down.click(lambda k: show(k + 1), pos, outs)
            b_copy.click(None, plain, said, _js="(t) => { navigator.clipboard.writeText(t || ''); return 'copied to the clipboard'; }")

            def to_memory(k):
                recs = history_all()
                if not recs:
                    return "nothing to save"
                k = max(0, min(int(k), len(recs) - 1))
                r = recs[k]
                n = memory_add(r.get("assistant") or "", {"kind": "response", "session": r.get("session"), "sent_at": _when(r),
                                                            "response_sha256": (r.get("receipt") or {}).get("response_sha256")})
                return f"saved response {k + 1} to .memory ({n} notes)"

            b_mem.click(to_memory, pos, said)
            proof_box = gr.Code(language="json", label="inclusion proof — share this with the one exchange; the rest of .history stays here")

            def proof(k):
                c = commitment(HISTORY)
                pr = inclusion_proof(HISTORY, int(k))
                line = _lines(HISTORY)[int(k)] if pr else b""
                return json.dumps({"commitment": c, "proof": pr, "verifies": verify_inclusion(line, pr) if pr else False}, indent=1)

            b_proof.click(proof, pos, proof_box)
        with gr.Tab(".memory"):
            mview = gr.HTML(memory_html())
            with gr.Row():
                note = gr.Textbox(label="add a note", placeholder="a fact or preference Savante should keep in mind", scale=4)
                b_add = gr.Button("add", variant="primary", scale=1)
            with gr.Row():
                which_n = gr.Number(label="note number", precision=0, value=1, scale=1)
                b_del = gr.Button("remove that note", scale=1)
                gr.Button("refresh", scale=1).click(memory_html, None, mview)
            b_add.click(lambda t: ("", memory_add(t, {"kind": "typed"}) and memory_html()), note, [note, mview])
            b_del.click(lambda n: (memory_remove(int(n or 0)), memory_html())[1], which_n, mview)
        with gr.Tab("Metrics"):
            mt = gr.HTML(metrics_html())
            gr.Button("refresh").click(metrics_html, None, mt)
        # everything that reads .history or .memory refreshes after each answer and on open
        for e in (ev, ev2):
            e.then(history_html, None, hall).then(metrics_html, None, mt).then(lambda: show(10 ** 9), None, outs)
        demo.load(lambda: (history_html(), metrics_html(), memory_html()), None, [hall, mt, mview])
        with gr.Tab("Agents"):
            import agents
            SAV = "Savante (the canon · read-only template)"
            gr.Markdown("**Custom agents from the Savante template.** Savante's canon is never written: *derive* makes a new agent "
                        f"with its own `.persona`, `.prompt`, agent card and ledger in `{agents.AGENTS}`. The ledger (sha256 + CIDv1 "
                        "per file, and a keccak256 doctrine root built exactly as Savante's binder builds hers) is regenerated on every "
                        "save; an agent that does not verify does not speak. The chat, .history, .memory, Responses and Metrics follow "
                        "the agent in use. Nothing here mints.")
            with gr.Row():
                apick = gr.Dropdown([SAV] + agents.list_agents(), value=SAV, label="agent", scale=3)
                b_use = gr.Button("use this agent", variant="primary", scale=1)
            active_md = gr.Markdown("in use: **Savante** (the canon)")
            with gr.Accordion("derive a new agent from Savante", open=False):
                with gr.Row():
                    d_name = gr.Textbox(label="name", placeholder="e.g. Auditor Ada")
                    d_kind = gr.Textbox(label="kind", value="governance")
                d_mantra = gr.Textbox(label="mantra", value=canon.persona.get("mantra", ""))
                d_oath = gr.Textbox(label="oath", value=canon.persona.get("oath", ""), lines=2)
                d_desc = gr.Textbox(label="public description (the agent card)", lines=2)
                d_sp = gr.Textbox(label="system prompt (.prompt) — start from Savante's and make it this agent's", value=canon.persona.get("system_prompt", ""), lines=10)
                b_derive = gr.Button("derive", variant="primary")
                d_out = gr.Markdown()
            with gr.Accordion("edit the agent in use (custom agents only): .persona and .prompt", open=False):
                e_persona = gr.Code(language="json", label=".persona (JSON; preflight: ASCII keys, integers only; every doctrine clause present)")
                e_prompt = gr.Textbox(label=".prompt", lines=10)
                b_save = gr.Button("save and re-ledger", variant="primary")
                e_out = gr.Markdown()
            with gr.Accordion("aivatar — the portrait of the agent in use (click it in the side panel for the full card)", open=False):
                gr.Markdown("Savante: choose one of her canon's images (the choice is kept outside the canon). A derived agent: "
                            "upload its own (PNG, JPEG or WebP, at most 2 MB); it becomes a ledgered facet, `x-bankml.aivatar`, "
                            "and part of its THOT bundle.")
                with gr.Row():
                    av_pick = gr.Dropdown(savante_images(), label="Savante's canon images", scale=2)
                    b_avpick = gr.Button("use this image for Savante", scale=1)
                with gr.Row():
                    av_up = gr.File(label="an image for the derived agent in use", file_types=["image"], scale=2)
                    b_avup = gr.Button("set as the agent's aivatar", variant="primary", scale=1)
                av_out = gr.Markdown()
            ledger = gr.HTML()

            def ledger_html():
                slug = ACTIVE["slug"]
                if not slug:
                    return "<div class='bk-card'>Savante's ledger: see the <b>Integrity</b> tab (every committed file, re-hashed at start).</div>"
                led = json.loads(agents.files(slug)["commitments.json"].read_text(encoding="utf-8"))
                rows = "".join(f"<tr><td>{E(n)}</td><td>{'✓' if ok else '✗'}</td><td class='bk-mono'>{E(str(d))}</td></tr>" for n, ok, d in agents.verify(slug))
                arts = "".join(f"<tr><td>{E(k)}</td><td class='bk-mono'>{a['sha256']}</td><td class='bk-mono'>{a['cid']}</td></tr>"
                               for k, a in list(led["artifacts"].items()) + [("card", led["card"])])
                return (f"<div class='bk-card'><b>{E(slug)}</b> · derived from {E(str((led.get('derived_from') or {}).get('agent')))} · "
                        f"status not_yet_minted · doctrine root <span class='bk-mono'>{led['doctrine_root']['value']}</span></div>"
                        f"<table class='bk-t'><tr><th>check</th><th></th><th>detail</th></tr>{rows}</table>"
                        f"<table class='bk-t'><tr><th>file</th><th>sha256</th><th>CIDv1</th></tr>{arts}</table>")

            def editors():
                slug = ACTIVE["slug"]
                if not slug:
                    return "", ""
                f = agents.files(slug)
                return f["persona"].read_text(encoding="utf-8"), f["prompt"].read_text(encoding="utf-8")

            def switch(choice):
                slug = None if choice == SAV else choice
                use_agent(slug)
                sid, turns = history_load()
                who = f"**{choice}**" + ("" if slug else " (the canon)")
                pe, pr = editors()
                return (f"in use: {who} · .history `{HISTORY}`", turns, {"id": sid}, history_html(), metrics_html(), memory_html(),
                        pe, pr, ledger_html(), system_prompt(canon, PROMPTS[0])[1], aivatar_html(canon), *show(10 ** 9))

            b_use.click(switch, apick, [active_md, chat, session, hall, mt, mview, e_persona, e_prompt, ledger, prov, aiv] + outs)

            def do_derive(name, kind, mantra, oath, desc, sp):
                if not name.strip():
                    return "give the agent a name", gr.update()
                try:
                    slug = agents.derive(canon.persona, canon.ledger, name.strip(), system_prompt=sp, mantra=mantra, oath=oath, kind=kind, description=desc)
                except (FileExistsError, ValueError) as e:
                    return f"not derived: {e}", gr.update()
                root = json.loads(agents.files(slug)["commitments.json"].read_text(encoding="utf-8"))["doctrine_root"]["value"]
                return (f"derived **{slug}** in `{agents.agent_dir(slug)}` · doctrine root `{root}` — choose it above and press *use this agent*",
                        gr.update(choices=[SAV] + agents.list_agents(), value=slug))

            b_derive.click(do_derive, [d_name, d_kind, d_mantra, d_oath, d_desc, d_sp], [d_out, apick])

            def do_save(pe, pr):
                slug = ACTIVE["slug"]
                if not slug:
                    return "Savante's canon is read-only: derive an agent to edit one.", ledger_html()
                try:
                    led = agents.save(slug, pe, pr)
                except (ValueError, json.JSONDecodeError) as e:
                    return f"not saved: {e}", ledger_html()
                return f"saved and re-ledgered · doctrine root `{led['doctrine_root']['value']}`", ledger_html()

            b_save.click(do_save, [e_persona, e_prompt], [e_out, ledger])

            def pick_savante_image(name):
                if name not in savante_images():
                    return "choose an image", gr.update()
                STATE.mkdir(parents=True, exist_ok=True)
                (STATE / "savante.aivatar").write_text(name + "\n", encoding="utf-8")
                return f"Savante's portrait: `gfx/{name}`" + ("" if not ACTIVE["slug"] else " (shown when Savante is in use)"), aivatar_html(canon)

            def set_agent_image(fobj):
                slug = ACTIVE["slug"]
                if not slug:
                    return "Savante's canon is read-only: choose one of her images above, or derive an agent", gr.update(), gr.update()
                if fobj is None:
                    return "upload an image first", gr.update(), gr.update()
                path = fobj if isinstance(fobj, str) else getattr(fobj, "name", None)
                try:
                    led = agents.set_aivatar(slug, Path(path).read_bytes())
                except (ValueError, OSError) as e:
                    return f"not set: {e}", gr.update(), gr.update()
                return (f"set · sha256 `{led['artifacts']['aivatar']['sha256'][:16]}…` · ledgered; the next THOT build includes it",
                        aivatar_html(canon), ledger_html())

            b_avpick.click(pick_savante_image, av_pick, [av_out, aiv])
            b_avup.click(set_agent_image, av_up, [av_out, aiv, ledger])
            demo.load(ledger_html, None, ledger)

            import connectors
            import thot
            with gr.Accordion("THOT bundle — the dataset an iNFT points to (sagi.thot_manifest/1)", open=False):
                gr.Markdown("Builds `<agent>.thot.json`: persona and prompt (core facets), the card, and **.history / .memory "
                            "committed by digest only** (`x-bankml.*` custom facets) — publishable, holding no conversation text. "
                            "A facet change makes a new generation whose `parent` is the previous manifest's CID. Checked against "
                            "the spec's own test vectors (savante, jaimla, luvai).")
                b_thot = gr.Button("build and verify the THOT bundle of the agent in use", variant="primary")
                thot_out = gr.Code(language="json", label="manifest (identity, bundle root, Merkle root, lineage)")

                def do_thot():
                    slug = ACTIVE["slug"]
                    if not slug:
                        st = json.loads((CANON / "savante.thot.json").read_text(encoding="utf-8"))
                        return json.dumps({"savante (read-only)": st["identity"], "generation": st["bundle"]["generation"],
                                           "findings": thot.check_structure(st) + thot.check_facets(st, CANON)}, indent=1)
                    tpl = json.loads((CANON / "savante.thot.json").read_text(encoding="utf-8"))
                    m = thot.build(slug, tpl)
                    return json.dumps({"identity": m["identity"], "bundle": m["bundle"], "bundle_root": m["bundle_root"]["value"],
                                       "merkle_root": m["merkle"]["root"], "rung": m["rung"]["evidence"], "relations": m.get("relations"),
                                       "findings": thot.verify(slug)}, indent=1, ensure_ascii=False)

                b_thot.click(do_thot, None, thot_out)
            with gr.Accordion("PostgreSQL (pgvector / pgvectorscale) — publish the agent, or load a published one", open=False):
                pg_st = gr.JSON(label="connection (BANKML_PG_DSN)")
                with gr.Row():
                    b_pgst = gr.Button("check the connection")
                    priv = gr.Checkbox(value=False, label="include .history and .memory lines (private; off = commitments only)")
                    b_pub = gr.Button("publish the agent in use", variant="primary")
                with gr.Row():
                    pub_pick = gr.Dropdown([], label="published agents", scale=3)
                    b_list = gr.Button("list", scale=1)
                    b_load = gr.Button("load (verified) as a local agent", scale=1)
                pg_out = gr.Markdown()

                def pg_status():
                    st = connectors.status()
                    return st

                def do_pub(p):
                    slug = ACTIVE["slug"]
                    if not slug:
                        return "Savante's canon is not published from here: derive an agent first."
                    try:
                        r = connectors.publish(slug, include_private=bool(p))
                    except Exception as e:  # noqa: BLE001 — shown to the operator
                        return f"not published: {e}"
                    return (f"published **{r['slug']}** · THOT generation {r['generation']} · `{r['thot_cid']}` · "
                            + (f"{r['exchanges']} exchanges and {r['memory']} notes included" if r["private_included"] else "commitments only (no private lines)"))

                def do_list():
                    try:
                        items = connectors.published()
                    except Exception as e:  # noqa: BLE001
                        return gr.update(choices=[]), f"cannot list: {e}"
                    return gr.update(choices=[x["slug"] for x in items]), f"{len(items)} published agents"

                def do_load(slug):
                    if not slug:
                        return "choose a published agent", gr.update()
                    target = slug if slug not in agents.list_agents() else f"{slug}_loaded"[:40]
                    try:
                        r = connectors.load(slug, as_slug=target)
                    except Exception as e:  # noqa: BLE001
                        return f"refused: {e}", gr.update()
                    return (f"loaded **{r['slug']}** from `{r['from']}` · verified against THOT `{r['thot_cid']}` · {r['private_lines']} private lines — "
                            "choose it above and press *use this agent*", gr.update(choices=[SAV] + agents.list_agents()))

                b_pgst.click(pg_status, None, pg_st)

            import chain
            with gr.Accordion("iNFT — mint the agent (prepared here, signed by the owner), or load one from a token", open=False):
                gr.Markdown("The house ERC-7857 contract `iNFT_7857` (`mintOpenAgent`). contentRoot = the agent's THOT manifest "
                            "contentRoot; metadataRoot = keccak256 of its card. **bankml never signs off a devnet**: it simulates, "
                            "and hands you the unsigned transaction and the `cast send` line for your own wallet. The contract "
                            "is not deployed on any public chain today, its audit is not cleared, and minting needs MINTER_ROLE. "
                            "*Start a local devnet* runs anvil (chain 31337) and deploys the contract, to try the whole path here.")
                with gr.Row():
                    c_rpc = gr.Textbox(label="RPC URL", value="http://127.0.0.1:8545", scale=2)
                    c_contract = gr.Textbox(label="iNFT_7857 contract", scale=2)
                    c_from = gr.Textbox(label="minter (from)", scale=2)
                with gr.Row():
                    c_to = gr.Textbox(label="owner of the new token (to)", scale=2)
                    c_dim = gr.Dropdown([str(d) for d in chain.VALID_DIMENSIONS], value="768", label="dimensions", scale=1)
                    b_dev = gr.Button("start a local devnet (anvil + iNFT_7857)", scale=2)
                with gr.Row():
                    b_plan = gr.Button("plan and simulate", variant="primary")
                    b_utx = gr.Button("unsigned transaction (owner signs)")
                    b_mint = gr.Button("mint on the local devnet (31337 only)")
                with gr.Row():
                    c_tok = gr.Number(label="token id", precision=0, value=1, scale=1)
                    b_read = gr.Button("load the agent from this token (verified)", scale=2)
                c_out = gr.Code(language="json", label="result")

                def _plan(to, dim):
                    slug = ACTIVE["slug"]
                    if not slug:
                        raise ValueError("Savante is not minted from here (her verdict on minting is DEFER): use a derived agent")
                    thot.build(slug, json.loads((CANON / "savante.thot.json").read_text(encoding="utf-8")))
                    return chain.plan_mint(slug, to=to.strip(), dimensions=int(dim))

                def _json(fn):
                    def run(*a):
                        try:
                            return json.dumps(fn(*a), indent=1, ensure_ascii=False)
                        except Exception as e:  # noqa: BLE001 — shown to the operator
                            return json.dumps({"error": f"{type(e).__name__}: {e}"}, indent=1)
                    return run

                def dev():
                    d = chain.devnet_up()
                    return d["url"], d["contract"], d["accounts"][0], d["accounts"][1], json.dumps({"devnet": d}, indent=1)

                b_dev.click(dev, None, [c_rpc, c_contract, c_from, c_to, c_out])
                b_plan.click(_json(lambda r, c, f, t, d: {"plan": (p := _plan(t, d)), "simulation": chain.simulate(r, c.strip(), f.strip(), p)}),
                             [c_rpc, c_contract, c_from, c_to, c_dim], c_out)
                b_utx.click(_json(lambda r, c, f, t, d: chain.unsigned_tx(r, c.strip(), f.strip(), _plan(t, d))), [c_rpc, c_contract, c_from, c_to, c_dim], c_out)
                b_mint.click(_json(lambda r, c, f, t, d: (lambda p: {"simulation": (sm := chain.simulate(r, c.strip(), f.strip(), p)),
                                                                     "sent": chain.send_devnet(r, c.strip(), f.strip(), p) if sm["ok"] else None,
                                                                     "token": chain.read_token(r, c.strip(), sm["tokenId"]) if sm["ok"] else None})(_plan(t, d))),
                             [c_rpc, c_contract, c_from, c_to, c_dim], c_out)
                b_read.click(_json(lambda r, c, k: chain.load_from_chain(r, c.strip(), int(k))), [c_rpc, c_contract, c_tok], c_out)
                b_pub.click(do_pub, priv, pg_out)
                b_list.click(do_list, None, [pub_pick, pg_out])
                b_load.click(do_load, pub_pick, [pg_out, apick])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=7873)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mode", choices=("interact", "view"), default="interact")
    a = ap.parse_args()
    if a.mode == "view":  # the LAN-safe stdlib server (not Gradio): ui/view.py
        import view
        sys.argv = [sys.argv[0], "--host", a.host if a.host != "127.0.0.1" else "0.0.0.0", "--port", str(a.port if a.port != 7873 else 7874)]
        return view.main()
    canon = Canon(CANON)
    bad = [r for r in canon.rows if not r[2]]
    print(f"canon {CANON}: {len(canon.rows) - len(bad)}/{len(canon.rows)} ledger files verify" + (f"; FAILING: {bad}" if bad else ""))
    demo = build(canon, a.mode)
    try:  # concurrency > 1 so the 1 s timer ticks while an answer streams (Gradio 3: concurrency_count)
        demo.queue(concurrency_count=4)
    except TypeError:
        demo.queue(default_concurrency_limit=4)
    demo.launch(server_name=a.host, server_port=a.port, show_api=False)


if __name__ == "__main__":
    main()
