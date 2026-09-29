#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""bankml · Savante — the local chat UI, in Gradio, built from the Hugging Face template PYTHAI/savante.

What it takes from the template (the Space's `local:bonsai-8b` carrier): the system prompt is the canon
persona's own `system_prompt`, the history is the last 12 exchanges (each ≤ 4000 characters), answers are
labelled drafts, `<think>` blocks are stripped and Qwen3 carriers get ` /no_think`.

What it adds:
  · **iNFT compatibility.** Savante's canon (~/cryptoAGI/savante, or SAVANTE_CANON) is read and never written. At start
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
    the laptop's load, Savante's office and ledger — served by `sAGI/view.py`, the standard library, not Gradio).

  bankml serve .models/Bonsai-8B-Q1_0.gguf --fork FORK.json --upstream 127.0.0.1:18092 --listen 127.0.0.1:18093
  python3 sAGI/savante.py --mode interact             # http://127.0.0.1:7873
  python3 sAGI/view.py --host 0.0.0.0 --port 7874     # http://<this laptop's LAN address>:7874

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
import embed  # noqa: E402 — bge-m3 via the local Ollama (optional; BM25 stands alone without it)

def _canon_default() -> Path:
    """Savante's canon: ~/cryptoAGI/savante (beside jaimla and luvai), or an older ~/savante if that is the only one."""
    new, old = Path.home() / "cryptoAGI" / "savante", Path.home() / "savante"
    return old if (not new.exists() and old.exists()) else new


CANON = Path(os.environ.get("SAVANTE_CANON", _canon_default())).expanduser()
SERVE = os.environ.get("BANKML_SERVE", "http://127.0.0.1:18093")
STATE = Path(os.environ.get("BANKML_UI_STATE", Path.home() / ".local" / "share" / "bankml" / "savante")).expanduser()
HISTORY = STATE / "savante.history"
MEMORY = STATE / "savante.memory"
ACTIVE = {"slug": None}  # None = Savante (the canon); else a custom agent from sAGI/agents.py


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
def jsonl_read(path: Path) -> list:
    """Every record of a JSONL file, oldest first. A damaged line (a crash mid-write) is skipped and counted in
    JSONL_BAD, never allowed to blank the whole file."""
    out, bad = [], 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for l in text.splitlines():
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except ValueError:
            bad += 1
            continue
        if isinstance(r, dict):
            out.append(r)
        else:
            bad += 1
    JSONL_BAD[str(path)] = bad
    return out


JSONL_BAD: dict = {}


def jsonl_append(path: Path, rec: dict) -> None:
    """One record as one write, under a lock, fsynced. If the file ends mid-line (a torn write), a newline goes first,
    so the new record is never glued onto the damaged one."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        size = os.fstat(fd).st_size
        if size:
            with open(path, "rb") as f:
                f.seek(size - 1)
                if f.read(1) != b"\n":
                    data = b"\n" + data
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def history_append(rec: dict, path: Path | None = None):
    jsonl_append(path or HISTORY, rec)


def history_load():
    """(session id, [[user, answer], …]) of the most recent session, or a new session."""
    lines = jsonl_read(HISTORY)
    if not lines:
        return uuid.uuid4().hex[:12], []
    sid = lines[-1].get("session")
    return sid, [[r.get("user", ""), r.get("shown", r.get("assistant", ""))] for r in lines if r.get("session") == sid]


def history_all() -> list:
    """Every record in .history, oldest first (all sessions); damaged lines skipped."""
    return jsonl_read(HISTORY)


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
    """The receipt hashes the raw text the model wrote (think block and whitespace included): `assistant_raw` since
    0.1.7; older records only kept the cleaned `assistant`, which matches whenever cleaning changed nothing."""
    h = (r.get("receipt") or {}).get("response_sha256")
    return None if not h else h == sha256((r.get("assistant_raw", r.get("assistant")) or "").encode())


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


# RFC 6962 / RFC 9162 Merkle trees: leaf = H(0x00 ‖ line), node = H(0x01 ‖ left ‖ right), the tree split at the largest
# power of two below n (an odd node is promoted, never paired with itself). Leaves and nodes cannot be confused, and a
# proof binds the record's index and the tree's size.
MERKLE_SCHEME = "rfc6962-sha256"


def _leaf(line: bytes) -> str:
    return hashlib.sha256(b"\x00" + line).hexdigest()


def _node(a: str, b: str) -> str:
    return hashlib.sha256(b"\x01" + bytes.fromhex(a) + bytes.fromhex(b)).hexdigest()


def _split(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def merkle_root(leaves: list) -> str | None:
    if not leaves:
        return None
    if len(leaves) == 1:
        return leaves[0]
    k = _split(len(leaves))
    return _node(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


def _audit_path(m: int, leaves: list) -> list:
    """RFC 6962 PATH(m, D[n]): sibling hashes from the leaf up."""
    n = len(leaves)
    if n <= 1:
        return []
    k = _split(n)
    if m < k:
        return _audit_path(m, leaves[:k]) + [merkle_root(leaves[k:])]
    return _audit_path(m - k, leaves[k:]) + [merkle_root(leaves[:k])]


def commitment(path: Path) -> dict:
    """{records, merkle_root, scheme, file_sha256, file_cid} for a JSONL file — shareable; reveals no content."""
    lines = _lines(path)
    raw = b"".join(l + b"\n" for l in lines)
    return {"file": path.name, "records": len(lines), "merkle_root": merkle_root([_leaf(l) for l in lines]), "scheme": MERKLE_SCHEME,
            "file_sha256": sha256(raw) if lines else None, "file_cid": cid_v1_raw(raw) if lines else None}


def inclusion_proof(path: Path, k: int) -> dict:
    """The proof that record k (0-based) of `records` is in the file whose root is `merkle_root`."""
    lines = _lines(path)
    leaves = [_leaf(l) for l in lines]
    if not 0 <= k < len(leaves):
        return {}
    return {"scheme": MERKLE_SCHEME, "record": k, "records": len(leaves), "leaf": leaves[k],
            "line_hashed_as": "sha256(0x00 ‖ the exact JSONL line, without its newline)",
            "path": _audit_path(k, leaves), "merkle_root": merkle_root(leaves)}


def verify_inclusion(line: bytes, proof: dict) -> bool:
    """RFC 9162 §2.1.3.2: recompute the root from the line, its index and the tree size; all three are bound."""
    try:
        fn, sn = int(proof["record"]), int(proof["records"]) - 1
        path = list(proof["path"])
    except (KeyError, TypeError, ValueError):
        return False
    if proof.get("scheme") != MERKLE_SCHEME or not 0 <= fn <= sn:
        return False
    r = _leaf(line)
    if r != proof.get("leaf"):
        return False
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = _node(p, r)
            while not fn & 1 and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            r = _node(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == proof.get("merkle_root")


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
    sem = _semantic(query, recs, k * 3)
    if sem is None:
        return out[:k], engine
    ranked, note = sem
    fused = embed.fuse([i for _, i, _ in out], ranked)
    return [(sc, i, recs[i]) for i, sc in fused[:k]], f"{engine} + {embed.MODEL} (meaning, {note}), fused by reciprocal rank"


def _semantic(query: str, recs: list, n: int):
    """([record indices by meaning], note) from bge-m3, or None when it cannot run now (then BM25 stands alone).
    Exchanges not yet embedded are indexed in the background; the ragebar never waits for them."""
    try:
        st = embed.status()
    except Exception:  # noqa: BLE001
        return None
    if not st.get("ready"):
        return None
    have = embed.cached(HISTORY)
    vecs = {i: have.get(hashlib.sha256(embed.exchange_text(r).encode()).hexdigest()) for i, r in enumerate(recs)}
    missing = sum(1 for v in vecs.values() if v is None)
    if missing:
        embed.index_async(HISTORY, recs)
    if len(query.strip()) < 3 or missing == len(recs):
        return None
    qv = embed.query_vector(query)
    if qv is None:
        return None
    ranked = sorted((i for i, v in vecs.items() if v is not None), key=lambda i: -embed.cosine(qv, vecs[i]))[:n]
    return ranked, f"{len(recs) - missing} of {len(recs)} embedded"


# ── .memory: notes the operator keeps, outside the canon ──────────────────────────
def memory_all() -> list:
    return jsonl_read(MEMORY)


def memory_add(text: str, source: dict | None = None) -> int:
    text = (text or "").strip()
    if not text:
        return len(memory_all())
    jsonl_append(MEMORY, {"ts": round(time.time(), 3), "at": iso(time.time()), "text": text, "sha256": sha256(text.encode()),
                          "source": source or {"kind": "typed"}})
    return len(memory_all())


def memory_remove(n: int) -> int:
    m = memory_all()
    if 1 <= n <= len(m):
        del m[n - 1]
        STATE.mkdir(parents=True, exist_ok=True)
        tmp = MEMORY.with_name(MEMORY.name + ".tmp")
        tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in m), encoding="utf-8")
        os.replace(tmp, MEMORY)  # never half a .memory
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
def serve_status(timeout=3):
    try:
        with urllib.request.urlopen(SERVE + "/bankml", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": f"bankml serve not reachable at {SERVE}: {e}"}


WINDOW_STEP = 6  # the history window moves in steps, not one exchange per turn


def window_start(n: int, lo: int = KEEP_EXCHANGES, step: int = WINDOW_STEP) -> int:
    """Where the history the model sees starts. It moves only every `step` turns, so between moves the prompt is
    the previous prompt plus one exchange and the engine's prompt cache reuses all of it (a window that slides one
    exchange per turn changes the text right after the system prompt and forces the whole history to be re-read).
    The model sees between `lo` and `lo + step - 1` exchanges once the chat is longer than `lo`."""
    return 0 if n <= lo else (n - lo) // step * step


def model_text(a: str) -> str:
    """An answer as the model wrote it, without the footer the chat adds (clock, receipt, provenance). The footer
    always starts with the clock, so an answer's own `<sub>` (a chemical formula, say) is kept."""
    return (a or "").rsplit("\n\n<sub>⏱", 1)[0]


_TOK: dict = {}
_CTX = {"t": 0.0, "n": None}
LAST_WINDOW: dict = {}  # what the last build_messages() sent: {"sent", "of", "ctx"} (the answer's footer says it)


def n_tokens(text: str) -> int:
    """Tokens in `text`, counted by the engine's own tokenizer (llama-server /tokenize, loopback), cached per text.
    If the engine cannot be asked: characters ÷ 3, an overestimate for English but not for CJK or emoji (≈ 1 token per
    character) — the engine then refuses an overfull prompt with its own message."""
    k = hashlib.sha256(text.encode()).digest()
    if k in _TOK:
        return _TOK[k]
    try:
        import models
        req = urllib.request.Request(f"http://{models.UPSTREAM}/tokenize", data=json.dumps({"content": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            n = len(json.loads(r.read())["tokens"])
    except (OSError, ValueError, KeyError):
        return -(-len(text) // 3)
    if len(_TOK) > 8192:  # drop the oldest half, not everything (a long session is not re-counted each turn)
        for old in list(_TOK)[:4096]:
            _TOK.pop(old, None)
    _TOK[k] = n
    return n


def engine_ctx() -> int | None:
    """The context the running engine actually has (its /props through bankml serve), cached per engine start (a
    restart with a new context is seen at once); if the engine cannot say, the saved setting (Resources)."""
    st = serve_status(1)
    key = st.get("hashed_at")
    if key is not None and _CTX.get("key") == key and _CTX["n"]:
        return _CTX["n"]
    if _CTX.get("key") is None and time.time() - _CTX["t"] < 10:  # no engine a moment ago: do not wait on it every turn
        return _CTX["n"]
    try:
        with urllib.request.urlopen(SERVE + "/props", timeout=3) as r:
            n = (json.loads(r.read()).get("default_generation_settings") or {}).get("n_ctx")
    except (OSError, ValueError):
        n = None
    if not n:
        import models
        n = models.resources()["ctx"]
    _CTX.update(t=time.time(), n=n, key=key)
    return n


# ── the system prompt's KV, kept across engine restarts (llama-server slot save / restore) ──────────────────────
_WARM: dict = {}  # (engine start, slot file) -> "restored" | "saved" | "none"


def _slot(system: str):
    """(engine instance, slot file name) for this model, context and system prompt; None without a verified engine."""
    st = serve_status()
    v = st.get("verified") or {}
    if not v.get("model_sha256"):
        return None
    h = hashlib.sha256(f"{v['model_sha256']}|{engine_ctx()}|{system}".encode()).hexdigest()[:24]
    return st.get("hashed_at"), f"savante-{h}.bin"


def _slot_call(action: str, name: str) -> dict:
    import models
    req = urllib.request.Request(f"http://{models.UPSTREAM}/slots/0?action={action}", data=json.dumps({"filename": name}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def slot_restore(system: str) -> str:
    """Before the first question to a newly started engine: restore the saved KV of this system prompt, if any. The
    same tokens give the same KV, so the answer is unchanged; only the prefill is skipped."""
    import models
    k = _slot(system)
    if not k or k in _WARM:
        return ""  # nothing done on this turn (the slot field records actions, not the engine's state)
    if (models.SLOTS / k[1]).is_file():
        try:
            _slot_call("restore", k[1])
            _WARM[k] = "restored"
        except (OSError, ValueError):
            _WARM[k] = "none"  # an engine without --slot-save-path, or a file it will not take: prefill as usual
    else:
        _WARM[k] = "none"
    return _WARM[k]


def slot_save(system: str) -> str:
    """After the first answer of an engine's life with this system prompt: save the slot in the background (newest
    three kept). Only when no other answer is being written — the engine's one slot then holds this conversation, not
    another tab's — and never in the way of the answer or its .history line. Returns "saving" when it started one."""
    import models, threading
    k = _slot(system)
    if not k or k in _WARM or len(INFLIGHT) > 1:
        return ""
    _WARM[k] = "saving"

    def run():
        try:
            _slot_call("save", k[1])
            _WARM[k] = "saved"
            for old in sorted(models.SLOTS.glob("savante-*.bin"), key=lambda p: p.stat().st_mtime)[:-3]:
                old.unlink(missing_ok=True)
        except (OSError, ValueError):
            _WARM[k] = "none"  # not retried every turn
    threading.Thread(target=run, daemon=True).start()
    return "saving"


class ContextTooSmall(ValueError):
    """The system prompt and the question alone do not fit the engine's context."""


_WSTART: dict = {}  # session id -> where its history window started last turn (kept while it still fits)


def build_messages(system, turns, question, qwen3, ctx_tokens: int | None = None, reserve_tokens: int = 256, count=None,
                   info: dict | None = None, session: str | None = None):
    """The messages for one turn: the system prompt, a windowed history, the question.

    The window starts at `window_start` (12–17 exchanges, moving in steps of six). With `ctx_tokens` it must also fit
    the engine's context, counted in the engine's own tokens (`count`, default `n_tokens`). When it does not:
    - the previous turn's start (per `session`) is kept while it still fits, so the prompt is the previous prompt plus
      one exchange and the engine's cache is reused;
    - when it stops fitting, the start jumps so that half of what fits is kept, leaving room for the next few turns;
    - never all history when some fits; and if the system prompt and question alone do not fit, ContextTooSmall.
    `info` (the caller's own dict — no shared state between tabs) receives {"sent", "of", "base", "ctx", "trimmed"}."""
    count = count or n_tokens
    turns = [(u, model_text(a)) for u, a in turns]
    n = len(turns)
    base = window_start(n)
    start = base
    per_msg = 8  # the chat template's markers around each message
    if ctx_tokens:
        budget = ctx_tokens - count(system) - count(question) - reserve_tokens - 3 * per_msg
        if budget < 0:
            raise ContextTooSmall(f"the system prompt and the question need {ctx_tokens - budget} tokens, more than the "
                                  f"engine's {ctx_tokens}-token context (with {reserve_tokens} kept for the answer)")
        cost = {}

        def c(k):  # tokens of exchange k, counted only when it could be sent
            if k not in cost:
                u, a = turns[k]
                cost[k] = count(u[:KEEP_CHARS]) + count((a or "")[:KEEP_CHARS]) + 2 * per_msg
            return cost[k]
        suffix = [0] * (n + 1)
        for k in range(n - 1, base - 1, -1):
            suffix[k] = suffix[k + 1] + c(k)
        size = lambda k: suffix[k]  # noqa: E731
        prev = _WSTART.get(session) if session else None
        if prev is not None and base <= prev < n and size(prev) <= budget:
            start = prev  # the same start as last turn: the whole previous prompt is reused
        elif size(base) > budget:
            fit_start = next((k for k in range(base, n) if size(k) <= budget), n)
            fits = n - fit_start
            # keep half of what fits (room to grow, so the next turns reuse this prompt) — but only when that still
            # leaves at least two exchanges; with so little room, history is worth more than a warm cache
            start = fit_start + fits // 2 if fits >= 4 else fit_start
    if session:
        _WSTART[session] = start
    msgs = [{"role": "system", "content": system}]
    for u, a in turns[start:]:
        msgs += [{"role": "user", "content": u[:KEEP_CHARS]}, {"role": "assistant", "content": (a or "")[:KEEP_CHARS]}]
    msgs.append({"role": "user", "content": question + (" /no_think" if qwen3 else "")})
    w = {"sent": n - start, "of": n, "base": base, "ctx": ctx_tokens, "trimmed": start > base}
    if info is not None:
        info.update(w)
    LAST_WINDOW.clear()
    LAST_WINDOW.update(w)
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
                ch = (ev.get("choices") or [{}])[0] or {}  # the receipt hashes choices[0], so the text is choices[0] only
                # a lone surrogate (\ud800 without its pair) becomes U+FFFD, as bankml's JSON decoder makes it
                acc += re.sub("[\ud800-\udfff]", "\ufffd", (ch.get("delta") or {}).get("content") or "")
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


# ── Models: import (catalogue, Hugging Face, Ollama) and choose the carrier ──────────────────────
def _job_bar(j: dict) -> str:
    pct = int(100 * j["done"] / j["total"]) if j.get("total") else 0
    size = f" · {j['done'] / 1e9:.2f} / {j['total'] / 1e9:.2f} GB" if j.get("total") else ""
    return (f"<div class='bk-jobbar'><i style='width:{pct}%'></i></div><div class='bk-note'>{pct}%{size} · "
            f"{int(time.time() - (j.get('started') or time.time()))} s</div>")


def job_html() -> str:
    import models
    j = models.JOB
    if j["state"] == "idle":
        return "<div class='bk-note'>No import running.</div>"
    if j["state"] == "running":
        return f"<div class='bk-card'><b>working</b> · {E(j['what'])}{_job_bar(j)}</div>"
    if j["state"] == "error":
        return f"<div class='bk-card bk-bad'><b>refused / failed</b><br>{E(j['error'])}</div>"
    r = j.get("result") or {}
    what = (r.get("verified") or {}).get("name") or r.get("file") or "done"
    note = f"<br>{E(r['note'])}" if r.get("note") else ""
    return f"<div class='bk-card'><b class='bk-okb'>done</b> · {E(str(what))}{note}</div>"


def models_html() -> str:
    import models
    st = serve_status()
    cur = (st.get("verified") or {}).get("model_sha256")
    pins = models.forks()
    rows = []
    for m in models.installed():
        sha = pins[m["file"]][1]["sha256"] if m["file"] in pins else None
        here = cur and sha == cur
        rows.append(f"<tr class='{'bk-cur' if here else ''}'><td>{'● ' if here else ''}<b>{E(m['file'])}</b></td><td>{m['bytes'] / 1e9:.2f} GB</td>"
                    f"<td>{E(m['source'] or '?')}</td><td>{E(m['licence'] or '?')}</td>"
                    f"<td>{'<span class=bk-okb>pinned</span>' if m['pinned'] else ('adopt on use' if m['catalog'] else '<span class=bk-badb>unpinned</span>')}</td></tr>")
    free = models.free_bytes() / 1e9
    return (f"<div class='bk-note'>{len(rows)} models in <code>{E(str(models.MODELS))}</code> · {free:.1f} GB free on disk · pins in "
            f"<code>{E(str(models.FORKS))}</code> · ● is the carrier now</div>"
            "<table class='bk-t'><tr><th>model</th><th>size</th><th>source</th><th>licence</th><th>pin</th></tr>" + "".join(rows) + "</table>")


def catalog_choices() -> list:
    import models
    out = []
    for c in models.CATALOG:
        have = (models.MODELS / c["file"]).exists()
        ok, why = models.fits(c["bytes"], have)
        out.append(f"{c['id']} — {c['title']} · {c['bytes'] / 1e9:.2f} GB · " + ("here" if have else ("fits" if ok else "does not fit: " + why)))
    return out


def resources_default_gb() -> float:
    """The budget the current settings imply for the carrier's model (weights + KV + overhead), for the slider."""
    import models
    st = serve_status()
    p = Path(st.get("model", "")) if "model" in st else None
    if not p or not p.exists():
        return 2.0
    pl = models.plan(p, 64.0)
    return round((pl["weights_gb"] * 1e9 + models.OVERHEAD + models.resources()["ctx"] * pl["kv_bytes_per_token"]) / 1e9, 1)


def resources_plan_html(ram_gb: float) -> str:
    import models
    st = serve_status()
    p = Path(st.get("model", "")) if "model" in st else None
    if not p or not p.exists():
        return "<div class='bk-note'>no carrier running: the budget is saved and used when a model starts</div>"
    pl = models.plan(p, float(ram_gb))
    if not pl["fits"]:
        return (f"<div class='bk-card bk-bad'>{ram_gb:.1f} GB cannot hold {E(p.name)}: weights {pl['weights_gb']} GB + engine "
                f"{pl['overhead_gb']:.2f} GB + a {models.CTX_MIN}-token KV cache needs at least <b>{pl['min_ram_gb']} GB</b></div>")
    ex = pl["ctx"] // 900  # a chat exchange is roughly 900 tokens here (question, answer, template)
    return (f"<div class='bk-note'>{E(p.name)}: weights {pl['weights_gb']} GB + engine {pl['overhead_gb']:.2f} GB + KV cache "
            f"{pl['kv_gb']} GB → a <b>{pl['ctx']}-token</b> context (about {ex} exchanges of history). Now: "
            f"{models.resources()['ctx']} tokens, {models.resources()['threads']} threads.</div>")


def resources_usage_html() -> str:
    import models
    u = models.usage()
    rows = "".join(f"<dt>{E(x['name'])} · pid {x['pid']}</dt><dd>{x['rss_bytes'] / 1e9:.2f} GB · {x['cpu_percent']:.0f} % CPU</dd>"
                   for x in u.get("processes") or [])
    return (f"<div class='bk-card'><b>engine now</b> · {u['rss_gb']} GB resident · {u['cpu_pct']} % CPU "
            f"(100 % = one of {u['cores']} cores) · machine: {u['mem_available_gb']} of {u['mem_total_gb']} GB free"
            f"<dl>{rows}</dl><div class='bk-note'>measured by {E(u.get('source', '?'))}</div></div>")


def installed_choices() -> list:
    """Chat models only: an embedding model (bge-m3 and the like) is listed in the table, not offered as a carrier."""
    import models
    return [m["file"] for m in models.installed() if (m["pinned"] or m["catalog"]) and models.guard(Path(m["path"])).get("arch") not in models.EMBEDDING_ARCHS]


def carrier_md() -> str:
    """bankml serve's verification as a compact card that wraps inside the side column."""
    st = serve_status()
    if "error" in st:
        import models
        j = models.JOB
        if j["state"] == "running":
            return f"<div class='bk-card'><b>bankml serve</b><br>starting: {E(j['what'])}{_job_bar(j)}</div>"
        why = f"<br>{E(j['error'])}" if j["state"] == "error" else ""
        return (f"<div class='bk-card bk-bad'><b>bankml serve</b><br>not reachable — choose a model in the Models tab, or start it (docs/usage.md)"
                f"<br><code>{SERVE}</code>{why}</div>")
    v = st.get("verified") or {}
    sha = v.get("model_sha256", "")
    kinds = ", ".join(f"{k}×{n}" for k, n in sorted((v.get("types") or {}).items(), key=lambda x: -x[1]) if k not in ("F32",))
    return ("<div class='bk-card'><b>bankml serve</b> <span class='bk-ok'>● verified · play</span>"
            f"<dl><dt>model</dt><dd>{E(v.get('name') or Path(st.get('model', '')).name)}</dd>"
            f"<dt>arch · weights</dt><dd>{E(v.get('arch') or '?')} · {E(kinds or '?')}</dd>"
            f"<dt>sha256</dt><dd class='bk-mono'>{sha}</dd>"
            f"<dt>bankml</dt><dd>{v.get('bankml', '?')} · guard {v.get('guard', '?')}</dd>"
            f"<dt>engine</dt><dd>{st.get('engine', '?')}</dd></dl></div>")


# ── the timer: from the press of Send to the last token ──────────────────────────
PENDING: dict = {"t0": None, "first": None}  # the timer's view of the latest Send
INFLIGHT: dict = {}  # request id -> start time: every answer being written, in any tab
STALE_S = 1800  # an entry older than this is a lost request (a closed tab), not an answer


def chat_busy() -> bool:
    now = time.time()
    for k, t in list(INFLIGHT.items()):
        if now - t > STALE_S:
            INFLIGHT.pop(k, None)
    return bool(INFLIGHT)


def timer_md() -> str:
    t0, first = PENDING["t0"], PENDING["first"]
    if t0 is not None and not INFLIGHT and time.time() - t0 > 120:  # Send was pressed but no answer started (a closed tab)
        PENDING.update(t0=None, first=None)
        t0 = None
    if t0 is None:
        return "<div class='bk-timer bk-idle'>⏱ ready</div>"
    el = time.time() - t0
    phase = f"writing · first token at {first - t0:.1f} s" if first else "reading the prompt (prefill)"
    # the browser counts the seconds from the moment of Send (data-t0); the server only changes the phase
    return (f"<div class='bk-timer bk-live' data-t0='{int(t0 * 1000)}'>⏱ <span class='bk-el'>{el:.1f}</span> s · {E(phase)}</div>")


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
.bk-el{display:inline-block;min-width:4.2ch;text-align:right;font-variant-numeric:tabular-nums}
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
.bk-knobs-wrap{border:1px solid #cbd5e1;border-radius:12px;padding:8px 8px 6px;background:#0b1220}
.bk-kh{font:700 11px ui-monospace,Menlo,monospace;letter-spacing:.2em;color:#2dd4bf!important;margin:0 2px 6px}
.bk-kn{font:10.5px ui-sans-serif,system-ui;color:#94a3b8!important;margin:4px 2px 0}
.dark .bk-knobs-wrap{background:transparent!important;border-color:rgba(148,163,184,.35)!important}
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
.bk-3d{position:relative;max-width:920px;width:100%;max-height:88vh;border-radius:16px;transform-style:preserve-3d;
 display:grid;grid-template-areas:"holo";grid-template-columns:minmax(0,1fr);
 transform:perspective(1600px) rotateX(var(--rx,0deg)) rotateY(var(--ry,0deg));transition:transform .25s ease-out;will-change:transform;
 box-shadow:0 30px 80px rgba(0,0,0,.65),0 12px 24px rgba(0,0,0,.45)}
.bk-scope{position:absolute;inset:0;width:100%;height:100%;border-radius:16px;pointer-events:none;z-index:0;
 background:radial-gradient(120% 90% at 50% 40%,#04211f 0%,#020a10 60%,#01050a 100%)}
.bk-sheen{position:absolute;inset:0;border-radius:16px;pointer-events:none;z-index:2;
 background:radial-gradient(60% 50% at var(--gx,50%) var(--gy,30%),rgba(255,255,255,.07),transparent 60%);mix-blend-mode:screen}
.bk-holo{grid-area:holo;min-width:0;box-sizing:border-box;position:relative;z-index:1;max-width:920px;width:100%;max-height:88vh;overflow:auto;border-radius:16px;padding:22px 24px 16px;color:#e2e8f0;
 background:linear-gradient(rgba(45,212,191,.05) 1px,transparent 1px) 0 0/100% 22px,linear-gradient(90deg,rgba(45,212,191,.05) 1px,transparent 1px) 0 0/22px 100%,
 linear-gradient(160deg,rgba(15,23,42,.72),rgba(2,6,23,.80));border:1px solid rgba(45,212,191,.45);backdrop-filter:blur(1.5px);
 box-shadow:0 0 0 1px rgba(217,162,58,.25),0 0 40px rgba(45,212,191,.18),inset 0 0 60px rgba(45,212,191,.05);animation:bk-glow 4s ease-in-out infinite}
.bk-holo{scrollbar-width:thin;scrollbar-color:rgba(45,212,191,.45) transparent}
.bk-holo::-webkit-scrollbar{width:8px}.bk-holo::-webkit-scrollbar-track{background:transparent}
.bk-holo::-webkit-scrollbar-thumb{background:rgba(45,212,191,.4);border-radius:8px}
@keyframes bk-glow{50%{box-shadow:0 0 0 1px rgba(217,162,58,.4),0 0 60px rgba(45,212,191,.28),inset 0 0 60px rgba(45,212,191,.07)}}
@media (prefers-reduced-motion:reduce){.bk-holo{animation:none}}
.bk-holo *{color:#e2e8f0}
/* depth: raised sections with a bevel — light from above, shadow below */
.bk-holo-top,.bk-cols>div,.bk-voice,.bk-intro,.bk-links .bk-link,.bk-knobs-card,.bk-aspects details,.bk-ft{
 box-shadow:inset 0 1px 0 rgba(255,255,255,.07),inset 0 -1px 0 rgba(0,0,0,.35),0 6px 18px rgba(0,0,0,.35)}
.bk-cols>div{border-radius:12px;padding:8px 12px;background:rgba(2,6,23,.35);border:1px solid rgba(45,212,191,.12)}
.bk-holo-top{border-radius:14px;padding:10px 12px;background:rgba(2,6,23,.35);border:1px solid rgba(217,162,58,.18)}
.bk-holo-pic{box-shadow:0 0 24px rgba(217,162,58,.35),0 10px 26px rgba(0,0,0,.55)}
@media (prefers-reduced-motion:reduce){.bk-3d{transform:none!important;transition:none}}
.bk-x{position:absolute;top:12px;right:14px;cursor:pointer;font-size:18px;color:#94a3b8!important;border:1px solid rgba(148,163,184,.3);border-radius:8px;padding:2px 9px}
.bk-x:hover{color:#2dd4bf!important;border-color:#2dd4bf}
.bk-holo-top{display:flex;gap:18px;align-items:center}
.bk-holo-pic{flex:none;width:120px;height:120px;border-radius:50%;overflow:hidden;border:2px solid #d9a23a;box-shadow:0 0 24px rgba(217,162,58,.35)}
.bk-holo-pic img,.bk-holo-pic .bk-glyph{width:120px;height:120px;object-fit:cover;font-size:40px}
.bk-kicker{font:600 11px ui-monospace,Menlo,monospace;letter-spacing:.2em;text-transform:uppercase;color:#2dd4bf!important}
.bk-holo h3{margin:4px 0 2px;font:700 28px ui-sans-serif,system-ui;letter-spacing:.02em;color:#fff!important;display:flex;align-items:center;gap:14px}
.bk-lead{cursor:pointer;border-radius:999px;border:1px solid #2dd4bf;background:rgba(45,212,191,.12);color:#5eead4!important;
 font:700 12px ui-monospace,Menlo,monospace;letter-spacing:.1em;padding:6px 14px;box-shadow:0 0 16px rgba(45,212,191,.25);transition:all .15s}
.bk-lead:hover{background:rgba(45,212,191,.24);box-shadow:0 0 22px rgba(45,212,191,.4)}
[data-bk].bk-on{border-color:#d9a23a!important;color:#fde68a!important;background:rgba(217,162,58,.16)!important;box-shadow:0 0 18px rgba(217,162,58,.35)!important}
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
/* the knob dock: hidden until she speaks; a side column (vertical stack), or a strip at the top or the bottom */
.bk-kdock{grid-area:dock;display:none;box-sizing:border-box;position:relative;z-index:3;border:1px solid rgba(45,212,191,.35);border-radius:14px;
 background:linear-gradient(160deg,rgba(15,23,42,.88),rgba(2,6,23,.92));box-shadow:0 0 24px rgba(45,212,191,.18),inset 0 1px 0 rgba(255,255,255,.07);
 padding:6px 10px 14px;animation:bk-kin .5s cubic-bezier(.2,.9,.3,1.2)}
@keyframes bk-kin{from{opacity:0;transform:scale(.94) translateY(-6px)}to{opacity:1;transform:none}}
.bk-3d.kd-on .bk-kdock{display:block}
.bk-3d.kd-on.kd-side{grid-template-areas:"holo dock";grid-template-columns:minmax(0,1fr) auto;gap:10px}
.bk-3d.kd-on.kd-top{grid-template-areas:"dock" "holo";grid-template-rows:auto minmax(0,1fr);gap:8px}
.bk-3d.kd-on.kd-bottom{grid-template-areas:"holo" "dock";grid-template-rows:minmax(0,1fr) auto;gap:8px}
.bk-3d.kd-on.kd-top .bk-holo,.bk-3d.kd-on.kd-bottom .bk-holo{max-height:calc(88vh - var(--kdh,150px))}
.kd-side .bk-kdock{align-self:start;max-height:88vh;overflow:auto}
.bk-kgrip{display:flex;align-items:center;gap:6px;cursor:grab;user-select:none;padding:2px 0 6px;border-bottom:1px solid rgba(45,212,191,.18);margin-bottom:6px}
.bk-kgrip b{font:700 11px ui-monospace,Menlo,monospace;letter-spacing:.2em;color:#5eead4!important}.bk-kdots{color:#94a3b8!important}.bk-ksp{flex:1}
.bk-kgrip button{cursor:pointer;border:1px solid rgba(148,163,184,.35);background:transparent;color:#cbd5e1!important;border-radius:6px;padding:0 5px;font-size:12px;line-height:18px}
.bk-kgrip button:hover,.bk-kgrip button.on{border-color:#2dd4bf;color:#5eead4!important}
.bk-kresize{position:absolute;right:3px;bottom:3px;width:14px;height:14px;cursor:nwse-resize;opacity:.55;
 background:linear-gradient(135deg,transparent 45%,#5eead4 46%,#5eead4 54%,transparent 55%,transparent 70%,#5eead4 71%,#5eead4 79%,transparent 80%)}
.bk-kresize:hover{opacity:1}
.bk-kzone{position:fixed;z-index:10001;border:2px dashed rgba(45,212,191,.6);border-radius:12px;background:rgba(45,212,191,.08);display:flex;
 align-items:center;justify-content:center;font:600 12px ui-monospace,Menlo,monospace;letter-spacing:.12em;color:#5eead4;pointer-events:auto}
.bk-kzone.hot{background:rgba(45,212,191,.22);border-style:solid}
@media (max-width:760px){.bk-3d.kd-on.kd-side{grid-template-areas:"holo" "dock";grid-template-columns:minmax(0,1fr)}}
.bk-knobs-card{max-width:520px;border:1px solid rgba(45,212,191,.22);border-radius:12px;background:rgba(45,212,191,.03);
 max-height:0;opacity:0;overflow:hidden;padding:0 10px;margin:0;transform:translateY(-6px) scale(.97);
 transition:max-height .45s ease,opacity .35s ease,transform .45s cubic-bezier(.2,.9,.3,1.2),padding .3s,margin .3s}
.bk-knobs-card.bk-emerge{max-height:140px;opacity:1;padding:8px 10px 4px;margin:6px 0 8px;transform:none;
 box-shadow:0 0 22px rgba(45,212,191,.18),inset 0 1px 0 rgba(255,255,255,.07)}
.bk-pend .bk-line{opacity:.45}.bk-dim{opacity:.35;border-color:transparent!important;cursor:default}
@media (prefers-reduced-motion:reduce){.bk-knobs-card{transition:none}}
.bk-links{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;margin:4px 0 6px}
.bk-link{display:block;text-decoration:none;border:1px solid rgba(45,212,191,.28);border-radius:10px;padding:8px 10px;background:rgba(45,212,191,.04);transition:all .15s}
.bk-link b{display:block;font:600 13px ui-sans-serif,system-ui;color:#5eead4!important}.bk-link span{display:block;font-size:11.5px;color:#94a3b8!important;margin-top:2px}
.bk-link:hover{border-color:#d9a23a;background:rgba(217,162,58,.08);box-shadow:0 0 14px rgba(217,162,58,.18)}
.bk-intro{border:1px solid rgba(217,162,58,.3);border-radius:12px;padding:10px 12px;background:rgba(217,162,58,.03);margin-bottom:6px}
.bk-chap{border:1px solid rgba(45,212,191,.18);border-radius:10px;margin:6px 0;padding:0}
.bk-chap summary,.bk-wait{display:flex;align-items:center;gap:10px;padding:7px 10px;cursor:pointer;list-style:none}
.bk-chap summary::-webkit-details-marker{display:none}.bk-chap summary b,.bk-wait b{flex:1;font:600 13px ui-sans-serif,system-ui;color:#e2e8f0!important}
.bk-cn{flex:none;width:22px;height:22px;border-radius:50%;border:1px solid #d9a23a;display:flex;align-items:center;justify-content:center;font:700 11px ui-monospace,Menlo,monospace;color:#fbbf24!important}
.bk-chap .bk-vlist{padding:0 8px 8px}.bk-wait{opacity:.55;cursor:default}
.bk-voice{border:1px solid rgba(45,212,191,.22);border-radius:12px;padding:10px 12px;background:rgba(45,212,191,.03);margin-bottom:6px}
.bk-vbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}
.bk-playall,.bk-stop,.bk-play1{cursor:pointer;border-radius:8px;font:700 12px ui-monospace,Menlo,monospace;letter-spacing:.08em}
.bk-playall{border:1px solid #2dd4bf;background:rgba(45,212,191,.12);color:#5eead4!important;padding:6px 14px}
.bk-playall:hover{background:rgba(45,212,191,.25);box-shadow:0 0 16px rgba(45,212,191,.3)}
.bk-export{margin-left:8px;padding:3px 10px;border:1px solid rgba(255,209,102,.55);border-radius:14px;color:#ffd166;text-decoration:none;font:600 12px ui-monospace,monospace;white-space:nowrap}.bk-export:hover{background:rgba(255,209,102,.12)}.bk-export small{opacity:.7;font-weight:400}.bk-export.off{opacity:.45;border-style:dashed;cursor:default}.bk-stop{border:1px solid rgba(148,163,184,.4);background:transparent;color:#cbd5e1!important;padding:6px 12px}
.bk-vmeta{font:11px ui-monospace,Menlo,monospace;color:#94a3b8!important}.bk-vmeta code{color:#fbbf24!important}
.bk-vlist{list-style:none;margin:0;padding:0;counter-reset:v}
.bk-vlist li{display:flex;align-items:baseline;gap:10px;padding:5px 6px;border-radius:8px;transition:background .2s}
.bk-vlist li.bk-now{background:rgba(217,162,58,.14);box-shadow:inset 3px 0 0 #d9a23a}
.bk-vlist li.bk-now .bk-line{color:#fde68a!important}
.bk-play1{flex:none;border:1px solid rgba(45,212,191,.35);background:transparent;color:#5eead4!important;padding:1px 7px;font-size:10px}
.bk-play1:hover{background:rgba(45,212,191,.15)}
.bk-line{flex:1;font-size:13px;line-height:1.5;color:#e2e8f0!important}.bk-dur{flex:none;font:11px ui-monospace,Menlo,monospace;color:#64748b!important}
.bk-foot{margin-top:14px;padding-top:8px;border-top:1px solid rgba(148,163,184,.2);font:11px ui-monospace,Menlo,monospace;letter-spacing:.06em;color:#64748b!important}
button.primary,button.lg.primary{background:#0f766e!important;border-color:#0f766e!important;color:#fff!important}
.tabs button.selected{border-bottom:3px solid #0f766e!important;font-weight:700}
/* input ready: a slow-blinking, light-green, semi-transparent cursor while the field is empty */
#bk-input label{position:relative}
#bk-input textarea{caret-color:rgba(134,239,172,.9)}
#bk-input textarea:placeholder-shown{text-indent:16px}
#bk-input label:has(textarea:placeholder-shown)::after{content:"";position:absolute;left:14px;top:50%;width:9px;height:1.25em;
 transform:translateY(-50%);border-radius:2px;background:rgba(134,239,172,.55);box-shadow:0 0 10px rgba(134,239,172,.45);
 animation:bk-ready 1.6s ease-in-out infinite;pointer-events:none}
@keyframes bk-ready{0%,100%{opacity:.9}50%{opacity:.08}}
@media (prefers-reduced-motion:reduce){#bk-input label:has(textarea:placeholder-shown)::after{animation:none;opacity:.6}}
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
.bk-t th{background:#f1f5f9}
.bk-commit dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:3px 12px;margin:6px 0 0}.bk-commit dt{font-weight:600}.bk-commit dd{margin:0;overflow-wrap:anywhere}
.bk-jobbar{height:8px;border-radius:5px;background:rgba(100,116,139,.2);overflow:hidden;margin:8px 0 2px}.bk-jobbar i{display:block;height:100%;background:linear-gradient(90deg,#0d9488,#d9a23a)}
.bk-cur td{background:rgba(13,148,136,.08)!important}
.bk-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:8px;margin:4px 0 10px}
.bk-kpi{border:1px solid #cbd5e1;border-radius:10px;padding:8px 10px;display:flex;flex-direction:column;gap:1px;min-width:0}
.bk-kpi b{font:700 20px ui-monospace,Menlo,Consolas,monospace;color:#0f172a;white-space:nowrap}
.bk-kpi span{font-size:12px;color:#334155}.bk-kpi small{font-size:11px;color:#64748b}
.bk-chart{margin:8px 0 12px;border:1px solid #cbd5e1;border-radius:10px;padding:8px 10px 4px}
.bk-chart figcaption{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;font-size:12px;color:#334155;margin-bottom:2px}
.bk-chart figcaption b{margin-right:auto;color:#0f172a}
.bk-chart svg{display:block;width:100%;height:auto;max-height:240px;overflow:visible}
.bk-lg{display:inline-flex;align-items:center;gap:5px}.bk-lg i{display:inline-block;width:10px;height:10px;border-radius:3px}
.bk-chart .g{stroke:rgba(100,116,139,.22);stroke-width:1}.bk-chart .ax{stroke:rgba(100,116,139,.6);stroke-width:1}
.bk-chart text{font:11px ui-monospace,Menlo,Consolas,monospace;fill:#64748b}.bk-chart .yl{text-anchor:end}.bk-chart .xl{text-anchor:middle}
.bk-chart .yu{text-anchor:middle}.bk-chart .ml{text-anchor:end;fill:#b45309;paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round}.bk-chart .clip{text-anchor:middle;fill:#b91c1c}
.bk-chart .med{stroke:#d97706;stroke-width:1.2;stroke-dasharray:5 4}
.bk-chart .hit{fill:transparent}.bk-chart .hit:hover{fill:rgba(100,116,139,.10)}
.bk-chart rect.s1,.bk-lg i.s1{fill:rgba(13,148,136,.85);background:rgba(13,148,136,.85)}
.bk-chart rect.s2,.bk-lg i.s2{fill:rgba(217,162,58,.9);background:rgba(217,162,58,.9)}
.bk-chart rect.s3,.bk-lg i.s3{fill:rgba(100,116,139,.6);background:rgba(100,116,139,.6)}
.bk-chart circle.s1{fill:#0d9488}.bk-chart circle.s2{fill:#d9a23a}
.bk-chart .ln{fill:none;stroke-width:1.5;opacity:.7}.bk-chart .ln.s1{stroke:#0d9488}.bk-chart .ln.s2{stroke:#d9a23a}
#bk-nav button{min-width:0}
/* dark mode: transparent surfaces over the page, light text and borders — never white panels */
.dark #bk-side,.dark .bk-card,.dark .bk-ex,.dark .bk-timer,.dark .bk-grip,.dark .bk-t th,.dark .bk-t td{background:transparent!important}
.dark #bk-side{border-color:rgba(148,163,184,.35)!important;box-shadow:none}
.dark .bk-card,.dark .bk-ex,.dark .bk-timer,.dark .bk-grip,.dark .bk-t th,.dark .bk-t td{border-color:rgba(148,163,184,.35)!important}
.dark .bk-card,.dark .bk-card *,.dark .bk-ex,.dark .bk-ex *,.dark .bk-t th,.dark .bk-t td,.dark .bk-grip{color:#e2e8f0!important}
.dark .bk-card dt,.dark .bk-exh,.dark .bk-note,.dark #bk-prov,.dark #bk-prov *,.dark .bk-idle,.dark .rb-stat{color:#94a3b8!important}
.dark .bk-card .bk-ok,.dark .bk-okb{color:#4ade80!important}.dark .bk-badb,.dark .bk-bad b{color:#f87171!important}
.dark .bk-live{border-color:#2dd4bf!important;color:#2dd4bf!important;background:rgba(45,212,191,.08)!important}
.dark .bk-t th{background:rgba(148,163,184,.08)!important}.dark .rb-meter{background:rgba(148,163,184,.2)}
.dark .bk-kpi,.dark .bk-chart{border-color:rgba(148,163,184,.35)!important;background:transparent!important}
.dark .bk-kpi b,.dark .bk-chart figcaption b{color:#e2e8f0!important}.dark .bk-kpi span,.dark .bk-chart figcaption{color:#cbd5e1!important}
.dark .bk-kpi small,.dark .bk-chart text{color:#94a3b8!important;fill:#94a3b8}.dark .bk-chart .ml{fill:#fbbf24;stroke:#0b0f19}
.dark .bk-chart .g{stroke:rgba(148,163,184,.16)}.dark .bk-chart rect.s1,.dark .bk-lg i.s1{fill:rgba(45,212,191,.75);background:rgba(45,212,191,.75)}
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


def _nice(mx: float) -> tuple:
    """A round axis top and step for 0..mx (1, 2 or 5 × 10ⁿ), about four gridlines."""
    import math
    if mx <= 0:
        return 1.0, 0.25
    raw = mx / 4
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    return step * math.ceil(mx / step), step


def _chart(rows: list, series: list, unit: str, title: str, stacked: bool = False, cap: float | None = None) -> str:
    """A small SVG chart that keeps its proportions: y axis with round gridlines, one slot per exchange (bars at most
    18 px wide, however few exchanges there are), values above `cap` clipped with a ▲ mark. series: [(key, label, css)]."""
    W, H, L, R, T, B = 640, 190, 46, 10, 14, 26
    pw, ph = W - L - R, H - T - B
    n = len(rows)
    tops = [sum((r.get(k) or 0) for k, _, _ in series) if stacked else max([(r.get(k) or 0) for k, _, _ in series] or [0]) for r in rows]
    top, step = _nice(min(max(tops or [0]), cap) if cap else max(tops or [0]))
    y = lambda v: T + ph - min(v, top) / top * ph
    g = [f"<line x1='{L}' x2='{W - R}' y1='{y(k * step):.1f}' y2='{y(k * step):.1f}' class='g'/>"
         f"<text x='{L - 6}' y='{y(k * step) + 4:.1f}' class='yl'>{k * step:g}</text>" for k in range(int(round(top / step)) + 1)]
    slot = pw / max(n, 1)
    bw = max(3.0, min(18.0, slot * 0.6))
    bars = []
    for j, r in enumerate(rows):
        cx = L + slot * (j + 0.5)
        base = 0.0
        tip = [f"#{j + 1} · {r.get('when', '')}"]
        for k, (key, label, css) in enumerate(series):
            v = r.get(key)
            if v is None:
                continue
            tip.append(f"{label}: {v:.2f} {unit}")
            lo, hi = (base, base + v) if stacked else (0.0, v)
            if stacked:
                base = hi
                bars.append(f"<rect x='{cx - bw / 2:.1f}' y='{y(hi):.1f}' width='{bw:.1f}' height='{max(y(lo) - y(hi), 0.5):.1f}' class='{css}'/>")
            else:
                bars.append(f"<circle cx='{cx:.1f}' cy='{y(v):.1f}' r='3.2' class='{css}'/>")
        if (tops[j] or 0) > top:
            bars.append(f"<text x='{cx:.1f}' y='{T - 3}' class='clip'>▲</text>")
        bars.append(f"<rect x='{cx - slot / 2:.1f}' y='{T}' width='{slot:.1f}' height='{ph}' class='hit'><title>{E(chr(10).join(tip))}</title></rect>")
    if not stacked:  # connect each series with a thin line
        for key, _, css in series:
            pts = [(L + slot * (j + 0.5), y(r[key])) for j, r in enumerate(rows) if r.get(key) is not None]
            if len(pts) > 1:
                bars.insert(0, f"<polyline points='{' '.join(f'{a:.1f},{b:.1f}' for a, b in pts)}' class='{css} ln'/>")
    med = sorted(t for t in tops if t) if stacked else []  # a median of mixed series would mean nothing
    if med:
        mv = med[len(med) // 2] if len(med) % 2 else (med[len(med) // 2 - 1] + med[len(med) // 2]) / 2
        bars.append(f"<line x1='{L}' x2='{W - R}' y1='{y(mv):.1f}' y2='{y(mv):.1f}' class='med'/>"
                    f"<text x='{W - R}' y='{y(mv) - 4:.1f}' class='ml'>median {mv:.1f} {unit}</text>")
    every = max(1, n // 8)
    ticks = sorted({0, n - 1} | {j for j in range(0, n, every) if n - 1 - j >= every / 2}) if n else []
    xl = [f"<text x='{L + slot * (j + 0.5):.1f}' y='{H - 8}' class='xl'>#{j + 1}</text>" for j in ticks]
    legend = "".join(f"<span class='bk-lg'><i class='{css}'></i>{E(label)}</span>" for key, label, css in series
                     if any(r.get(key) is not None for r in rows))
    return (f"<figure class='bk-chart'><figcaption><b>{E(title)}</b>{legend}</figcaption>"
            f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='{E(title)}'>"
            f"<line x1='{L}' x2='{L}' y1='{T}' y2='{T + ph}' class='ax'/><line x1='{L}' x2='{W - R}' y1='{T + ph}' y2='{T + ph}' class='ax'/>"
            + "".join(g) + "".join(bars) + "".join(xl) + f"<text x='8' y='{T + ph / 2:.0f}' class='yu' transform='rotate(-90 8 {T + ph / 2:.0f})'>{E(unit)}</text></svg></figure>")


def metrics_html() -> str:
    m = metrics()
    if not m["exchanges"]:
        return "<div class='bk-card'>No exchanges in .history yet.</div>"
    def row(name, st, unit, nd=1):
        return (f"<tr><td>{name}</td><td>{st['n']}</td><td>{_fmt(st['median'], unit, nd)}</td><td>{_fmt(st['p90'], unit, nd)}</td>"
                f"<td>{_fmt(st['mean'], unit, nd)}</td><td>{_fmt(st['min'], unit, nd)}</td><td>{_fmt(st['max'], unit, nd)}</td></tr>")
    rows = m["rows"][-60:]  # the last 60 exchanges; the tables below keep every statistic
    tl = [{"when": x["when"], "first": x["first"], "write": (x["total"] - x["first"]) if x["total"] is not None and x["first"] is not None else None,
           "total": x["total"] if x["first"] is None else None} for x in rows]
    tp = [{"when": x["when"], "pre": (x["pt"] / x["first"]) if x["pt"] and x["first"] else None,
           "wr": (x["ct"] / (x["total"] - x["first"])) if x["ct"] and x["total"] and x["first"] is not None and x["total"] > x["first"] else None} for x in rows]
    tots = sorted(x["total"] for x in rows if x["total"] is not None)
    cap = tots[int(0.95 * (len(tots) - 1))] * 1.15 if len(tots) >= 8 else None  # one slow outlier must not flatten the rest
    def kpi(v, label, sub=""):
        return f"<div class='bk-kpi'><b>{v}</b><span>{E(label)}</span>{f'<small>{E(sub)}</small>' if sub else ''}</div>"
    fs, rs, ws = m["first_token_s"], m["response_s"], m["write_tok_s"]
    verified = m["hash_ok"] + m["hash_bad"]
    tiles = ("<div class='bk-kpis'>"
             + kpi(m["exchanges"], "exchanges", f"{m['sessions']} sessions")
             + kpi(_fmt(fs["median"]), "first token", f"median · p90 {_fmt(fs['p90'])}")
             + kpi(_fmt(rs["median"]), "response", f"median · p90 {_fmt(rs['p90'])}")
             + kpi(_fmt(ws["median"], "tok/s", 2), "writing", "median")
             + kpi(f"{m['hash_ok']}/{verified}" if verified else "—", "answers match receipt", f"{m['hash_missing']} without receipt")
             + "</div>")
    recent = "".join(f"<tr><td>{E(x['when'])}</td><td>{_fmt(x['first'])}</td><td>{_fmt(x['total'])}</td><td>{x['src'] or '—'}</td>"
                     f"<td>{x['pt'] or '—'}+{x['ct'] or '—'}</td><td>{'✓' if x['hash'] else ('✗' if x['hash'] is False else '—')}</td>"
                     f"<td>{E(x['q'])}</td></tr>" for x in reversed(m["rows"][-25:]))
    days = " · ".join(f"{d}: {c}" for d, c in sorted(m["per_day"].items()))
    ch, cm = commitment(HISTORY), commitment(MEMORY)
    val = lambda v: f"<span class='bk-mono'>{E(str(v))}</span>" if v else "<i>— (empty)</i>"
    proofs = ("<div class='bk-card bk-commit'><b>Commitments</b> <span class='bk-dimb'>(shareable; they reveal no content)</span><dl>"
              f"<dt>.history</dt><dd>{ch['records']} records</dd>"
              f"<dt>Merkle root</dt><dd>{val(ch['merkle_root'])}</dd>"
              f"<dt>CID</dt><dd>{val(ch['file_cid'])}</dd>"
              f"<dt>.memory</dt><dd>{cm['records']} notes</dd>"
              f"<dt>Merkle root</dt><dd>{val(cm['merkle_root'])}</dd></dl></div>")
    return (tiles
            + _chart(tl, [("first", "waiting for the first token", "s1"), ("write", "writing the answer", "s2"), ("total", "total (no first-token time)", "s3")],
                     "s", f"response time per exchange{' (last 60)' if len(m['rows']) > 60 else ''}", stacked=True, cap=cap)
            + _chart(tp, [("pre", "prompt reading (prefill)", "s1"), ("wr", "writing", "s2")], "tok/s", "throughput per exchange")
            + f"<div class='bk-note'>per day: {E(days)} · hover a column for its values</div>"
            "<table class='bk-t'><tr><th>measure</th><th>n</th><th>median</th><th>p90</th><th>mean</th><th>min</th><th>max</th></tr>"
            + row("time to first token (from Send)", m["first_token_s"], "s") + row("response time (from Send)", m["response_s"], "s")
            + row("prompt reading (prefill)", m["prefill_tok_s"], "tok/s", 2) + row("writing", m["write_tok_s"], "tok/s", 2) + "</table>"
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


LINKS = [("Savante on Hugging Face", "https://huggingface.co/spaces/PYTHAI/savante", "the public office: chat, canon, integrity"),
         ("sAGI on Hugging Face", "https://huggingface.co/spaces/PYTHAI/savante/blob/main/skills/sagi/SKILL.md", "the sAGI skill Savante runs"),
         ("Savante's loop", "https://huggingface.co/datasets/PYTHAI/savante-loop", "public questions and answers (dataset)"),
         ("sAGI engine", "https://github.com/cryptoAGI/sagi", "the verdict contract, enforced in code"),
         ("Savante's canon", "https://github.com/cryptoAGI/savante", "persona, charter, facets, ledger"),
         ("bankml", "https://github.com/cryptoAGI/bankml", "the runtime this page runs on")]


def links_html(derived: bool) -> str:
    head = "derived from Savante — her public places" if derived else "Savante in public"
    a = "".join(f"<a class='bk-link' href='{E(u)}' target='_blank' rel='noopener noreferrer'><b>{E(t)}</b><span>{E(d)}</span></a>" for t, u, d in LINKS)
    return f"<div class='bk-h'>{E(head)}</div><div class='bk-links'>{a}</div>"


def _clip_li(text: str, item, lead: bool = False) -> str:
    if item:
        return (f"<li{' class=bk-lead-line' if lead else ''}><button type='button' class='bk-play1' data-bk='from' title='play from here'>▶</button>"
                f"<span class='bk-line'>{E(text)}</span><audio preload='none' src='/file={E(str(item['file']))}'></audio></li>")
    return f"<li class='bk-pend'><span class='bk-play1 bk-dim'>·</span><span class='bk-line'>{E(text)}</span><span class='bk-dur'>rendering</span></li>"


def _chapters_html(chapters: list, lead_n: int = 0) -> tuple:
    """(html, ready, total) for a list of chapters: whatever is rendered plays now."""
    import speak
    total = sum(len(x) for _, x in chapters)
    ready, parts = 0, []
    for n, (title, sents) in enumerate(chapters, 1):
        items = speak.cached(sents)
        k = sum(1 for i in items if i)
        ready += k
        secs = sum((i["seconds"] or 0) for i in items if i)
        state = f"{secs / 60:.1f} min" if k == len(sents) else f"{k}/{len(sents)} ready"
        nlead = lead_n if n == 1 else 0
        lis = "".join(_clip_li(t, i, j < nlead) for j, (t, i) in enumerate(zip(sents, items)))
        play = "<button type='button' class='bk-play1' data-bk='all' data-scope='details'>▶ chapter</button>" if k else ""
        parts.append(f"<details class='bk-chap bk-voice'><summary><span class='bk-cn'>{n}</span><b>{E(title)}</b>"
                     f"<span class='bk-dur'>{state}</span>{play}</summary><ol class='bk-vlist'>{lis}</ol></details>")
    return "".join(parts), ready, total


def export_html(name: str, chapters: list, lead: list = ()) -> str:
    """⤓ <name>.opus: the whole set as one file, offered only when it is complete and current."""
    import speak
    st = speak.export_state(name, ([("Voice examples", list(lead))] if lead else []) + list(chapters))
    if st["current"]:
        f = st["file"]
        return (f"<a class='bk-export' download='{E(name)}.opus' href='/file={E(str(f))}' title='Ogg Opus, one chapter mark per chapter'>"
                f"⤓ {E(name)}.opus <small>{f.stat().st_size / 1e6:.1f} MB</small></a>")
    return f"<span class='bk-export off' title='an export is complete or it is not made'>⤓ {E(name)}.opus · {st['ready']}/{st['total']} rendered</span>"


def intro_html(canon: Canon) -> str:
    """Savante reads herself to a new participant: chapters from her canon; whatever is rendered plays now."""
    import speak
    chapters = speak.intro_chapters(CANON, canon.persona, canon.card)
    body, ready, total = _chapters_html(chapters, len(speak.speech(canon.card.get("description") or "")))
    reading = speak.reading_chapters()
    rbody, rready, rtotal = _chapters_html(reading)
    sets = speak.export_sets(CANON, canon.persona, canon.card)
    if ready + rready < total + rtotal or not all(speak.export_state(n, ([("Voice examples", l)] if l else []) + c)["current"] for n, (l, c) in sets.items()):
        speak.render_intro_async(CANON, canon.persona, canon.card)
    note = "" if ready == total else f" · {ready}/{total} sentences rendered so far — refresh the page for more"
    rnote = "" if rready == rtotal else f" · {rready}/{rtotal} rendered so far"
    return (f"<div class='bk-h'>introduction — Savante reads herself to a new participant</div>"
            f"<div class='bk-intro'><div class='bk-vbar'><button type='button' class='bk-playall' data-bk='all' data-scope='.bk-intro'>▶ PLAY THE INTRODUCTION</button>"
            f"<button type='button' class='bk-stop' data-bk='stop'>■ STOP</button>{export_html('Savante', *sets['Savante'][::-1])}"
            f"<span class='bk-vmeta'>{len(chapters)} chapters from her canon (persona, .prompt, explanation, manifesto, Savante.md) read verbatim in her own voice, "
            f"with one note of bankml's own (SCIEN·TIFIC){note}</span></div>" + body + "</div>"
            + (f"<div class='bk-h'>the reading — Savante reads bankml's thesis, and binary and ternary</div>"
               f"<div class='bk-intro bk-reading'><div class='bk-vbar'><button type='button' class='bk-playall' data-bk='all' data-scope='.bk-reading'>▶ PLAY THE READING</button>"
               f"<button type='button' class='bk-stop' data-bk='stop'>■ STOP</button>{export_html('Savante-reading', reading)}"
               f"<span class='bk-vmeta'>{len(reading)} chapters from TECHNICAL.md, verbatim, the notation read aloud{rnote}</span></div>" + rbody + "</div>"
               if reading else ""))


def voice_html(examples: list, name: str) -> str:
    """The persona's voice examples in Savante's voice: whatever is rendered plays now; PLAY ALL plays them in order."""
    import speak
    lines = [x for x in examples if isinstance(x, str) and x.strip()]
    if not lines:
        return (f"<div class='bk-h'>{E(name)} speaks</div><p>No voice examples in this persona yet — add them to "
                "<code>voice_examples</code> in the .persona editor (Agents).</p>")
    ok, why = speak.available()
    if not ok:
        return f"<div class='bk-h'>{E(name)} speaks</div><p>Not rendered on this computer ({E(why)}).</p>"
    items = speak.cached(lines)
    k = sum(1 for i in items if i)
    if k < len(lines):
        speak.render_async(lines)
    v = speak.savante_voice()
    total = sum((i["seconds"] or 0) for i in items if i)
    who = ("Savante's own voice (Cori's body, Jaimla's pitch, SAVANTE's resonance; slower, steadier)" if v.get("model")
           else f"the house stand-in {E(v['voice'])} at {v['wpm']} wpm")
    return (f"<div class='bk-h'>{E(name)} speaks — every voice example, pre-rendered</div>"
            f"<div class='bk-voice'><div class='bk-vbar'>"
            f"<button type='button' class='bk-playall' data-bk='all' data-scope='.bk-voice'>▶ PLAY ALL</button>"
            f"<button type='button' class='bk-stop' data-bk='stop'>■ STOP</button>"
            f"<span class='bk-vmeta'>{k}/{len(lines)} ready · {total:.0f} s · {who} · said sav-ont</span></div>"
            f"<ol class='bk-vlist'>{''.join(_clip_li(t, i) for t, i in zip(lines, items))}</ol></div>")


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
    voice = voice_html(p.get("voice_examples") or [], name)
    lead_btn = ("<button type='button' class='bk-lead' data-bk='lead' title='hear who she is'>▶ PLAY</button>" if not slug else "")
    intro = intro_html(canon) if not slug else ""
    links = links_html(bool(slug))
    return f"""<div class='bk-av'>
<input type='checkbox' id='bk-av-open' class='bk-av-t'>
<label for='bk-av-open' class='bk-av-pic' title='open {E(name)}'s card'>{pic}<span class='bk-av-cap'>{E(name)} · open the card</span></label>
<div class='bk-modal'><label for='bk-av-open' class='bk-modal-bg'></label>
<div class='bk-3d'><canvas class='bk-scope' aria-hidden='true'></canvas><div class='bk-sheen'></div>
<div class='bk-holo' role='dialog' aria-label='{E(name)}'>
<label for='bk-av-open' class='bk-x' title='close'>✕</label>
<div class='bk-holo-top'><div class='bk-holo-pic'>{pic}</div><div>
<div class='bk-kicker'>{E(p.get('kind', ''))} · {E(card.get('type', '').split('#')[-1] or 'agent')}</div>
<h3>{E(name)}{lead_btn}</h3><div class='bk-mantra'>{E(p.get('mantra', ''))}</div>
<div class='bk-chips'>{attrs}</div></div></div>
<p class='bk-desc'>{E(card.get('description') or '')}</p>
<div class='bk-cols'><div><div class='bk-h'>oath</div><p>{E(p.get('oath', ''))}</p>
<div class='bk-h'>office</div><p>primary skill: {E(str((p.get('skills') or {}).get('primary', '—')))}<br>scope: {E(str((p.get('safety') or {}).get('scope', '—')))}</p>
<div class='bk-h'>beliefs</div><ul>{''.join(f'<li>{E(b)}</li>' for b in beliefs)}</ul></div>
<div><div class='bk-h'>identity — verifiable, not asserted</div><dl class='bk-hash'>{dl}</dl></div></div>
{links}
{intro}
{voice}
<div class='bk-h'>every aspect of the persona</div>
<div class='bk-aspects'>{aspects}</div>
<div class='bk-h'>ledgered files</div>
<table class='bk-ft'><tr><th>facet</th><th>file</th><th>bytes</th><th>sha256</th></tr>{files}</table>
<div class='bk-foot'>portrait: {E(str(img_label or 'none'))} · the canon is read-only · every value above is re-derived from its files · nothing here mints</div>
</div>
<div class='bk-kdock' aria-label='voice controls'>
<div class='bk-kgrip' draggable='true' title='drag to dock: side, top or bottom'><span class='bk-kdots'>⠿</span><b>VOICE</b>
<span class='bk-ksp'></span><button type='button' data-kd='top' title='dock to the top'>⤒</button><button type='button' data-kd='side' title='dock to the side'>⇥</button><button type='button' data-kd='bottom' title='dock to the bottom'>⤓</button></div>
<div class='bk-kbody'></div><div class='bk-kresize' title='drag to resize the knobs'></div></div>
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
  // Savante's voice: the DreamKnob controls, and the player every card's PLAY buttons use
  try {
    if (!window.SavanteKnobs) (0, eval)(__KNOBS__);
    // the knobs live in the card and emerge when she starts to speak (mounted then, not before)
    // the knob dock: appears when she speaks; side column (vertical stack), top or bottom; dragged, docked, resized
    const KD = 'bankml-knob-dock-v1';
    let kd = { dock: 'side', size: 56 }; try { kd = Object.assign(kd, JSON.parse(localStorage.getItem(KD) || '{}')) } catch (e) {}
    const kdSave = () => { try { localStorage.setItem(KD, JSON.stringify(kd)) } catch (e) {} };
    const kdLay = (s3) => {
      s3.classList.remove('kd-side', 'kd-top', 'kd-bottom'); s3.classList.add('kd-on', 'kd-' + kd.dock);
      const dock = s3.querySelector('.bk-kdock'), body = dock && dock.querySelector('.bk-kbody');
      if (body && window.SavanteKnobs) SavanteKnobs.mount(body, kd.size, kd.dock === 'side' ? 'column' : 'row');
      dock.querySelectorAll('[data-kd]').forEach(b => b.classList.toggle('on', b.dataset.kd === kd.dock));
      requestAnimationFrame(() => s3.style.setProperty('--kdh', (dock.offsetHeight + 12) + 'px'));
    };
    window.bkEmerge = (from) => { const s3 = from && from.closest('.bk-3d'); if (s3) kdLay(s3) };
    const kdSet = (s3, where) => { kd.dock = where; kdSave(); kdLay(s3) };
    document.addEventListener('click', (e) => {
      const b = e.target.closest && e.target.closest('[data-kd]'); if (!b) return;
      e.preventDefault(); e.stopPropagation(); kdSet(b.closest('.bk-3d'), b.dataset.kd);
    }, true);
    // drag the grip: three drop zones over the card — top, side, bottom
    let kzones = [];
    const kclear = () => { kzones.forEach(z => z.remove()); kzones = [] };
    document.addEventListener('dragstart', (e) => {
      const g = e.target.closest && e.target.closest('.bk-kgrip'); if (!g) return;
      const s3 = g.closest('.bk-3d'), r = s3.getBoundingClientRect(); e.dataTransfer.setData('text/plain', 'bk-kdock'); e.dataTransfer.effectAllowed = 'move';
      const band = Math.min(90, r.height * 0.18);
      for (const [where, box] of [['top', [r.left, r.top, r.width, band]], ['bottom', [r.left, r.bottom - band, r.width, band]],
                                  ['side', [r.right - Math.min(160, r.width * 0.25), r.top + band + 6, Math.min(160, r.width * 0.25), r.height - 2 * band - 12]]]) {
        const z = document.createElement('div'); z.className = 'bk-kzone'; z.textContent = 'dock ' + where;
        Object.assign(z.style, { left: box[0] + 'px', top: box[1] + 'px', width: box[2] + 'px', height: box[3] + 'px' });
        z.addEventListener('dragover', ev => { ev.preventDefault(); z.classList.add('hot') });
        z.addEventListener('dragleave', () => z.classList.remove('hot'));
        z.addEventListener('drop', ev => { ev.preventDefault(); kclear(); kdSet(s3, where) });
        document.body.append(z); kzones.push(z);
      }
    });
    document.addEventListener('dragend', (e) => { if (e.target.closest && e.target.closest('.bk-kgrip')) kclear() });
    // resize: drag the corner to scale the knobs (40–96 px)
    document.addEventListener('pointerdown', (e) => {
      const h = e.target.closest && e.target.closest('.bk-kresize'); if (!h) return;
      e.preventDefault(); h.setPointerCapture(e.pointerId);
      const s3 = h.closest('.bk-3d'), x0 = e.clientX, y0 = e.clientY, s0 = kd.size;
      const mv = (ev) => { const d = kd.dock === 'side' ? ev.clientX - x0 : ev.clientY - y0;
        const n = Math.round(Math.max(40, Math.min(96, s0 + d / (kd.dock === 'side' ? 1.5 : 3)))); if (n !== kd.size) { kd.size = n; kdLay(s3) } };
      h.addEventListener('pointermove', mv);
      h.addEventListener('pointerup', () => { h.removeEventListener('pointermove', mv); kdSave() }, { once: true });
    });
  } catch (e) { console.warn('voice knobs', e) }
  if (!window.bkPlayer) window.bkPlayer = (() => {
    let ctx = null, lfo, depth, delay, pre, vol, cur = null, queue = [];
    const src = new WeakMap();
    const v = () => ({ speed: 1, fmRate: 5, fmDepth: 0, gain: 0, volume: 80, ...(window.bkVoice || {}) });
    const chain = () => {
      if (ctx) return;
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      delay = ctx.createDelay(0.05); delay.delayTime.value = 0.006;       // the FM line: a short delay...
      lfo = ctx.createOscillator(); depth = ctx.createGain();              // ...whose length an LFO sweeps
      lfo.connect(depth); depth.connect(delay.delayTime); lfo.start();
      window.bkAnalyser = ctx.createAnalyser(); bkAnalyser.fftSize = 2048; bkAnalyser.smoothingTimeConstant = 0;
      pre = ctx.createGain(); pre.connect(delay);                          // GAIN, into the chain
      vol = ctx.createGain();                                              // VOLUME, the master out
      delay.connect(bkAnalyser); bkAnalyser.connect(vol); vol.connect(ctx.destination);
    };
    const apply = () => {
      const x = v();
      if (ctx) { lfo.frequency.setTargetAtTime(Math.max(0.01, +x.fmRate || 0), ctx.currentTime, 0.02);
                 depth.gain.setTargetAtTime(0.0025 * Math.max(0, Math.min(100, +x.fmDepth || 0)) / 100, ctx.currentTime, 0.02);
                 pre.gain.setTargetAtTime(Math.pow(10, Math.max(-12, Math.min(12, +x.gain || 0)) / 20), ctx.currentTime, 0.02);
                 vol.gain.setTargetAtTime(Math.max(0, Math.min(100, +x.volume)) / 100, ctx.currentTime, 0.02); }
      else if (cur) cur.volume = Math.max(0, Math.min(1, (+x.volume) / 100));   // no Web Audio: volume still works
      if (cur) { cur.preservesPitch = true; cur.mozPreservesPitch = true; cur.webkitPreservesPitch = true; cur.playbackRate = +x.speed || 1; }
    };
    window.addEventListener('bk-voice', apply);
    const mark = (a, on) => { const li = a && a.closest('li'); if (li) li.classList.toggle('bk-now', on) };
    const play = (a, then) => {
      try {
        chain(); if (ctx.state === 'suspended') ctx.resume();
        if (!src.has(a)) { const s = ctx.createMediaElementSource(a); s.connect(pre); src.set(a, s) }
      } catch (err) { console.warn('bankml voice: Web Audio unavailable, playing plain', err) }
      cur = a; apply(); a.currentTime = 0; mark(a, true);
      window.bkPlaying = true;
      a.onended = () => { mark(a, false); window.bkPlaying = false; if (then) then() };
      const d = a.closest('details'); if (d && !d.open) d.open = true;
      const li = a.closest('li'); if (li && li.scrollIntoView) li.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      a.play().catch(() => mark(a, false));
    };
    const stop = () => { queue = []; window.bkPlaying = false; if (cur) { cur.onended = null; cur.pause(); mark(cur, false) } cur = null };
    // one delegated listener: PLAY buttons work however the card's HTML was inserted
    let btn = null, paused = false, lastDone = null;
    const label = (b, on) => {
      if (!b) return;
      if (!b.dataset.label) b.dataset.label = b.textContent;
      const small = b.dataset.label.trim() === '▶';
      b.textContent = on === 'play' ? (small ? '❚❚' : '❚❚ PAUSE') : on === 'paused' ? (small ? '▶' : '▶ RESUME') : b.dataset.label;
      b.classList.toggle('bk-on', !!on);
    };
    const finish = () => { label(btn, null); btn = null; paused = false; cur = null; window.bkPlaying = false };
    const run = (b, list) => {
      stop(); label(btn, null); btn = b; label(b, 'play'); if (window.bkEmerge) window.bkEmerge(b);
      queue = list.filter(Boolean);
      const next = () => { const a = queue.shift(); if (a) play(a, () => { lastDone = a; next() }); else finish() };
      next();
    };
    const holoOf = (b) => b.closest('.bk-holo');
    const introAll = (h) => [...h.querySelectorAll('.bk-intro audio')];
    const exAll = (h) => [...h.querySelectorAll('.bk-voice:not(.bk-chap) audio')];
    document.addEventListener('click', (e) => {
      const b = e.target.closest && e.target.closest('[data-bk]');
      if (!b || !b.closest('.bk-holo')) return;
      e.preventDefault(); e.stopPropagation();
      const act = b.dataset.bk, h = holoOf(b);
      if (act === 'stop') { stop(); label(btn, null); btn = null; return }
      if (b === btn && cur) {                       // the playing button is a pause button
        if (paused) { paused = false; cur.play(); label(b, 'play'); window.bkPlaying = true }
        else { paused = true; cur.pause(); label(b, 'paused'); window.bkPlaying = false }
        return;
      }
      if (act === 'lead') return run(b, [...h.querySelectorAll('.bk-lead-line audio')]);   // who she is, then it ends
      if (act === 'all' && b.dataset.scope === 'details') return run(b, [...b.closest('details').querySelectorAll('audio')]);
      if (act === 'all' && b.dataset.scope === '.bk-voice') return run(b, exAll(h));
      if (act === 'all' && b.dataset.scope === '.bk-reading') {   // the reading alone; resumes where it was left
        const all = [...h.querySelectorAll('.bk-reading audio')], k = lastDone ? all.indexOf(lastDone) : -1;
        return run(b, all.slice(k >= 0 && k < all.length - 1 ? k + 1 : 0));
      }
      if (act === 'all') {                          // the introduction: continue after what was heard, on until stopped
        const all = introAll(h).concat(exAll(h)), k = lastDone ? all.indexOf(lastDone) : -1;
        return run(b, all.slice(k >= 0 && k < all.length - 1 ? k + 1 : 0));
      }
      if (act === 'from') { const all = introAll(h).concat(exAll(h)), a = b.closest('li').querySelector('audio'); return run(b, all.slice(Math.max(0, all.indexOf(a)))) }
    }, true);
    return { stop: () => { stop(); label(btn, null); btn = null } };
  })();
  // the oscilloscope behind the card: her waveform while she speaks, a slow idle sweep otherwise
  if (!window.bkScope) window.bkScope = (() => {
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    let buf = null; const t0 = performance.now();
    const draw = () => {
      for (const c of document.querySelectorAll('.bk-scope')) {
        if (!c.offsetParent) continue;                      // the card is closed: nothing to draw
        const r = c.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
        if (c.width !== Math.round(r.width * dpr)) { c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr) }
        const g = c.getContext('2d'), W = c.width, H = c.height;
        g.fillStyle = 'rgba(1,8,12,0.22)'; g.fillRect(0, 0, W, H);          // phosphor persistence
        g.strokeStyle = 'rgba(45,212,191,0.06)'; g.lineWidth = 1;          // graticule
        g.beginPath(); for (let i = 1; i < 10; i++) { const x = W * i / 10; g.moveTo(x, 0); g.lineTo(x, H) }
        for (let j = 1; j < 8; j++) { const y = H * j / 8; g.moveTo(0, y); g.lineTo(W, y) } g.stroke();
        const live = window.bkAnalyser && window.bkPlaying;
        g.lineWidth = 2.2 * dpr; g.strokeStyle = live ? 'rgba(94,234,212,0.95)' : 'rgba(45,212,191,0.35)';
        g.shadowColor = live ? 'rgba(45,212,191,0.9)' : 'rgba(45,212,191,0.4)'; g.shadowBlur = 14 * dpr;
        g.beginPath();
        if (live) {
          buf = buf || new Float32Array(bkAnalyser.fftSize); bkAnalyser.getFloatTimeDomainData(buf);
          for (let i = 0; i < buf.length; i++) { const x = W * i / (buf.length - 1), y = H / 2 - buf[i] * H * 0.42; i ? g.lineTo(x, y) : g.moveTo(x, y) }
        } else {
          const t = (performance.now() - t0) / 1000;
          for (let i = 0; i <= 240; i++) { const u = i / 240, x = W * u;
            const y = H / 2 + Math.sin(u * 14 + t * 1.4) * H * 0.035 * (0.6 + 0.4 * Math.sin(t * 0.7 + u * 3)); i ? g.lineTo(x, y) : g.moveTo(x, y) }
        }
        g.stroke(); g.shadowBlur = 0;
      }
      if (!still) requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
    // depth: the card tilts toward the pointer, and the sheen follows
    document.addEventListener('pointermove', (e) => {
      if (still) return;
      for (const s3 of document.querySelectorAll('.bk-3d')) {
        if (!s3.offsetParent) continue;
        const r = s3.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
        const cx = Math.max(-0.5, Math.min(0.5, x - 0.5)), cy = Math.max(-0.5, Math.min(0.5, y - 0.5));
        s3.style.setProperty('--ry', (cx * 9).toFixed(2) + 'deg'); s3.style.setProperty('--rx', (-cy * 7).toFixed(2) + 'deg');
        s3.style.setProperty('--gx', (x * 100).toFixed(1) + '%'); s3.style.setProperty('--gy', (y * 100).toFixed(1) + '%');
      }
    }, { passive: true });
    return true;
  })();
  // the timer: real seconds going by, counted here from the moment Send was pressed
  if (!window.bkClock) window.bkClock = setInterval(() => {
    for (const t of document.querySelectorAll('.bk-timer[data-t0]')) {
      const el = t.querySelector('.bk-el'); if (el) el.textContent = ((Date.now() - +t.dataset.t0) / 1000).toFixed(1);
    }
  }, 100);
  handle(side, 'side', 'xy');
  if (chat) handle(chat, 'chat', 'y');
  if (st.side === 'left') row.prepend(side);
  return [];
}"""


KNOBS_JS = Path(__file__).resolve().parent / "voice" / "knobs" / "savante_knobs.js"


def layout_js() -> str:
    """LAYOUT_JS with the DreamKnob bundle embedded (Gradio 3 serves no page scripts of ours)."""
    try:
        src = KNOBS_JS.read_text(encoding="utf-8")
    except OSError:
        src = ""
    return LAYOUT_JS.replace("__KNOBS__", json.dumps(src))


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
                    msg = gr.Textbox(placeholder="Ask Savante — e.g. review: is this model ready to serve?", show_label=False, elem_id="bk-input")
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
                    with gr.Accordion("Resources · CPU and RAM for the engine", open=False):
                        import models as _M
                        _r, _mt = _M.resources(), _M.mem_total() / 1e9
                        cpu_s = gr.Slider(1, os.cpu_count() or 1, value=_r["threads"], step=1, label=f"CPU threads (of {os.cpu_count()})")
                        ram_s = gr.Slider(0.5, round(_mt, 1), value=_r["ram_gb"] or resources_default_gb(), step=0.1, label="RAM budget for the engine (GB)")
                        spec_c = gr.Checkbox(value=bool(_r.get("spec_ngram")), label="n-gram speculation (exact at temperature 0, no extra memory; "
                                             "ahead in 5 of 6 paired runs here but not beyond this laptop's noise, so off by default)")
                        res_plan = gr.HTML(resources_plan_html(_r["ram_gb"] or resources_default_gb()))
                        with gr.Row():
                            res_apply = gr.Button("Apply (restarts the engine)", variant="primary")
                            res_use = gr.Button("usage now")
                        res_usage = gr.HTML("<div class='bk-note'>press “usage now” to read what the engine uses</div>")
                        ram_s.change(resources_plan_html, ram_s, res_plan, show_progress=False)
                        res_use.click(resources_usage_html, None, res_usage)
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
                rid = uuid.uuid4().hex
                INFLIGHT[rid] = time.time()
                t0 = PENDING["t0"] or time.time()
                hist, agent = HISTORY, ACTIVE["slug"] or "savante"  # where this exchange belongs, fixed at its start
                try:
                    system, why = system_prompt(canon, which)
                    if system is None:
                        h[-1][1] = f"refused: {why}"
                        yield h
                        return
                    mem = memory_block() if use_mem else ""
                    if mem:
                        system, why = system + mem, why + f" + .memory ({len(memory_all())} notes, {len(mem)} chars)"
                    st = serve_status()
                    arch = str((st.get("verified") or {}).get("arch") or "").lower()
                    qwen3 = arch in ("qwen3", "smollm3") or any(k in json.dumps(st).lower() for k in ("qwen3", "bonsai", "smollm3"))
                    warm = slot_restore(system)
                    win = {}
                    try:
                        msgs = build_messages(system, [t for t in h[:-1] if t[1] is not None], h[-1][0], qwen3,
                                              ctx_tokens=engine_ctx(), reserve_tokens=int(max_tokens), info=win, session=sess["id"])
                    except ContextTooSmall as e:
                        h[-1][1] = f"refused: {e}. Raise the RAM budget in Resources (a larger context), shorten the question, or lower max tokens."
                        yield h
                        return
                    text, rc, first = "", {}, None
                    for text, r in stream(msgs, max_tokens, temperature):
                        if text and first is None:
                            first = time.time()
                            if PENDING["t0"] == t0:
                                PENDING["first"] = first
                        h[-1][1] = show_answer(text)
                        rc = r if r is not None else rc
                        yield h
                    t1 = time.time()
                    answer = show_answer(text)
                    timing = {"sent_at": iso(t0), "first_token_s": round(first - t0, 2) if first else None,
                              "response_s": round(t1 - t0, 2), "answered_at": iso(t1)}
                    clock = (f"⏱ sent {time.strftime('%H:%M:%S', time.localtime(t0))} · first token "
                             f"{timing['first_token_s'] if first else '—'} s · answered in {timing['response_s']} s")
                    foot = receipt_line(rc, text)
                    trimmed = (f"<br>history: {win['sent']} of {win['of']} exchanges fit the engine's {win['ctx']}-token context — "
                               "raise the RAM budget in Resources for more") if win.get("trimmed") else ""
                    h[-1][1] = answer + f"\n\n<sub>{clock}" + (f"<br>{foot}<br>{why}" if foot else "") + trimmed + "</sub>"
                    history_append({"ts": round(t0, 3), **timing, "agent": agent, "slot": warm or None, "session": sess["id"], "user": h[-1][0], "assistant": answer,
                                    "assistant_raw": text, "shown": h[-1][1], "prompt": which, "prompt_provenance": why, "receipt": rc}, hist)
                    slot_save(system)  # background; never in the way of the answer or its .history line
                    yield h
                finally:
                    INFLIGHT.pop(rid, None)
                    if PENDING["t0"] == t0:  # only this request's clock; another tab's stays
                        PENDING.update(t0=None, first=None)

            ev = msg.submit(add, [msg, chat], [msg, chat]).then(respond, [chat, which, max_tokens, temperature, session, use_mem], chat)
            ev2 = send.click(add, [msg, chat], [msg, chat]).then(respond, [chat, which, max_tokens, temperature, session, use_mem], chat)
            stop.click(lambda: PENDING.update(t0=None, first=None), None, None, cancels=[ev, ev2])  # a cancelled respond runs its finally
            which.change(lambda w: system_prompt(canon, w)[1], which, prov)
            new.click(lambda: ([], {"id": uuid.uuid4().hex[:12]}), None, [chat, session])
        stage, mach, log, ci = view_tabs(gr, canon)
        demo.load(lambda: (live_stage(), machine(), live_tail(), ci_status()), None, [stage, mach, log, ci], every=2, show_progress=False)
        demo.load(timer_md, None, timer, every=1, show_progress=False)
        demo.load(lambda: aivatar_html(canon), None, aiv, show_progress=False)  # the card reflects whatever has rendered by now
        demo.load(None, None, None, _js=layout_js())
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
        with gr.Tab("Models"):
            import models as MD
            gr.Markdown("**Models.** Savante speaks through one carrier at a time: `bankml serve` in front of llama.cpp, verified "
                        "before it answers (the header guard, then the file's sha256 against its FORK.json pin). Bring models in "
                        "from the catalogue, from any Hugging Face GGUF by URL, or from Ollama. Every file is hashed as it arrives "
                        "and kept only if it equals the sha256 its publisher lists. **Open source only**: a model whose licence is "
                        "not open source (Gemma, Llama, …) is refused.")
            mlist = gr.HTML(models_html())
            with gr.Row():
                use_dd = gr.Dropdown(installed_choices(), label="installed models", scale=3)
                use_btn = gr.Button("Use this model", variant="primary", scale=1)
            with gr.Row():
                then_use = gr.Checkbox(value=True, label="switch to a model when its import finishes", scale=3)
                cancel_btn = gr.Button("Cancel download", scale=1)
            mjob = gr.HTML(job_html())
            seen = gr.State(-1)
            with gr.Accordion("Catalogue: open-source models, pinned to their publishers' sha256", open=True):
                with gr.Row():
                    cat_dd = gr.Dropdown(catalog_choices(), label="catalogue", scale=3)
                    cat_btn = gr.Button("Import", scale=1)
            with gr.Accordion("Hugging Face: any GGUF by URL", open=False):
                with gr.Row():
                    hf_url = gr.Textbox(label="URL", placeholder="https://huggingface.co/OWNER/REPO  or  …/blob/main/FILE.gguf", scale=3)
                    hf_look = gr.Button("Look up", scale=1)
                hf_info = gr.HTML()
                hf_state = gr.State({})
                with gr.Row():
                    hf_dd = gr.Dropdown([], label="GGUF file", scale=3)
                    hf_btn = gr.Button("Import", scale=1)
            with gr.Accordion("Ollama: search ollama.com, or adopt what the local Ollama holds", open=False):
                with gr.Row():
                    ol_q = gr.Textbox(label="search ollama.com", placeholder="qwen3, granite, smollm …", scale=3)
                    ol_go = gr.Button("Search", scale=1)
                ol_res = gr.HTML()
                with gr.Row():
                    ol_dd = gr.Dropdown([], label="model", scale=2)
                    ol_tag = gr.Dropdown([], label="tag", scale=2)
                    ol_btn = gr.Button("Import", scale=1)
                loc = [f"{m['name']}:{m['tag']}" for m in MD.ollama_local()]
                with gr.Row():
                    loc_dd = gr.Dropdown(loc, label=f"in the local Ollama ({len(loc)}; adopted by link, no download)", scale=3)
                    loc_btn = gr.Button("Adopt", scale=1)

            def _busy():
                return chat_busy()

            def _started(ok):
                # the handler returns at once; the poll below follows the job (no queue worker held for a download)
                msg = job_html() if ok else "<div class='bk-card bk-bad'>another import or switch is running; wait for it</div>"
                yield msg, gr.update(), gr.update(), gr.update()

            def _watch(started):
                yield from _started(started)

            def poll(last):
                j = MD.JOB
                if j["state"] == "running":
                    return job_html(), gr.update(), gr.update(), gr.update(), last
                if j["seq"] != last:
                    return job_html(), models_html(), gr.update(choices=installed_choices()), carrier_md(), j["seq"]
                return gr.update(), gr.update(), gr.update(), gr.update(), last

            def _import_then(spec_fn, use):
                def run():
                    r = MD.import_spec(spec_fn())
                    if use and _busy():  # the import stands; the switch waits for the answer being written
                        return {**r, "note": "imported; an answer is being written, so press Use this model when it is done"}
                    return MD.switch(r["file"], _busy) if use else r
                return run

            def do_use(f):
                if not f:
                    yield "<div class='bk-card bk-bad'>choose an installed model</div>", models_html(), gr.update(), carrier_md()
                    return
                yield from _watch(MD.start_job(f"switching to {f}", MD.switch, f, _busy))

            def do_cat(c, use):
                if not c:
                    yield "<div class='bk-card bk-bad'>choose a catalogue model</div>", models_html(), gr.update(), carrier_md()
                    return
                cid = c.split(" — ")[0]
                yield from _watch(MD.start_job(f"importing {cid}", _import_then(lambda: MD.spec_catalog(cid), use)))

            def do_hf_look(url):
                try:
                    r = MD.hf_resolve(url)
                except Exception as e:  # noqa: BLE001
                    return f"<div class='bk-card bk-bad'>{E(str(e))}</div>", {}, gr.update(choices=[], value=None)
                lic = r["licence"] if isinstance(r["licence"], str) else ", ".join(r["licence"]) or "none stated"
                if r.get("gated"):
                    return (f"<div class='bk-card bk-bad'><b>{E(r['repo'])}</b> is gated (it needs an account and an accepted "
                            "agreement); bankml imports only openly downloadable models.</div>", {}, gr.update(choices=[], value=None))
                if not r["open"]:
                    return (f"<div class='bk-card bk-bad'><b>{E(r['repo'])}</b>: licence <b>{E(lic)}</b> is not open source; bankml does not "
                            "import it.</div>", {}, gr.update(choices=[], value=None))
                ch = [f"{f['file']} · {f['bytes'] / 1e9:.2f} GB · {MD.fits(f['bytes'])[1]}" for f in r["files"]]
                pick = next((c for c in ch if c.split(" · ")[0] == r["chosen"]), ch[0] if len(ch) == 1 else None)
                return (f"<div class='bk-card'><b>{E(r['repo'])}</b> @ <span class='bk-mono'>{r['revision'][:12]}</span> · licence "
                        f"<span class='bk-okb'>{E(lic)}</span> · {len(ch)} GGUF files, each pinned to the repository's LFS sha256</div>",
                        r, gr.update(choices=ch, value=pick))

            def do_hf(r, f, use):
                if not r or not f:
                    yield "<div class='bk-card bk-bad'>look up a repository and choose a file</div>", models_html(), gr.update(), carrier_md()
                    return
                file = f.split(" · ")[0]
                yield from _watch(MD.start_job(f"importing {file}", _import_then(lambda: MD.spec_hf(r, file), use)))

            def do_ol_search(q):
                try:
                    res = MD.ollama_search(q)
                except Exception as e:  # noqa: BLE001
                    return f"<div class='bk-card bk-bad'>{E(str(e))}</div>", gr.update(choices=[])
                rows = "".join(f"<tr><td><b>{E(x['name'])}</b></td><td>{E(', '.join(x['sizes']))}</td><td>{E(x['pulls'])}</td><td>{E(x['description'][:160])}</td></tr>"
                               for x in res[:20])
                return ("<table class='bk-t'><tr><th>model</th><th>sizes</th><th>pulls</th><th>description</th></tr>" + rows + "</table>"
                        "<div class='bk-note'>The licence is read from each model's registry layer at import; one that is not open source is refused.</div>",
                        gr.update(choices=[x["name"] for x in res[:20]], value=None))

            def do_ol_tags(name):
                if not name:
                    return gr.update(choices=[], value=None)
                try:
                    tags = [t for t in MD.ollama_tags(name) if "cloud" not in t]
                except Exception:  # noqa: BLE001
                    tags = ["latest"]
                return gr.update(choices=tags, value=next((t for t in tags if re.fullmatch(r"[0-9.]+[mb]", t)), tags[0] if tags else None))

            def do_ol(name, tag, use):
                if not name or not tag:
                    yield "<div class='bk-card bk-bad'>choose a model and a tag</div>", models_html(), gr.update(), carrier_md()
                    return
                yield from _watch(MD.start_job(f"importing {name}:{tag}", _import_then(lambda: MD.spec_ollama(MD.ollama_resolve(f"{name}:{tag}")), use)))

            outs_m = [mjob, mlist, use_dd, carrier]

            def do_resources(threads, ram, spec):
                ok = MD.start_job(f"applying {int(threads)} threads and {ram:.1f} GB", MD.apply_resources, threads, ram, chat_busy, spec)
                return ("<div class='bk-card'>restarting the engine with the new settings — the carrier card follows it</div>" if ok
                        else "<div class='bk-card bk-bad'>another import or switch is running; wait for it</div>")
            res_apply.click(do_resources, [cpu_s, ram_s, spec_c], res_usage)
            demo.load(poll, seen, outs_m + [seen], every=2)
            cancel_btn.click(lambda: ("<div class='bk-card'>cancelling at the next chunk; the partial file is kept for resume</div>"
                                      if MD.cancel_job() else job_html()), None, mjob)
            use_btn.click(do_use, use_dd, outs_m)
            cat_btn.click(do_cat, [cat_dd, then_use], outs_m)
            hf_look.click(do_hf_look, hf_url, [hf_info, hf_state, hf_dd])
            hf_btn.click(do_hf, [hf_state, hf_dd, then_use], outs_m)
            ol_go.click(do_ol_search, ol_q, [ol_res, ol_dd])
            ol_q.submit(do_ol_search, ol_q, [ol_res, ol_dd])
            ol_dd.change(do_ol_tags, ol_dd, ol_tag)
            ol_btn.click(do_ol, [ol_dd, ol_tag, then_use], outs_m)
            def do_loc(nt, use):
                yield from do_ol(*(nt or ":").rsplit(":", 1), use)
            loc_btn.click(do_loc, [loc_dd, then_use], outs_m)
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


LOOPBACK = {"127.0.0.1", "localhost"}  # (::1 is not offered: the Host check's parser cannot read a bracketed address)


def _trusted_hosts():
    """Refuse any request whose Host is not loopback (a DNS-rebinding page cannot drive the UI): the middleware is
    added where Gradio builds its app, before the server starts."""
    from gradio import routes
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    orig = routes.App.create_app
    if getattr(orig, "_bankml", False):
        return

    def create_app(*a, **k):
        app = orig(*a, **k)
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "::1"])
        return app
    create_app._bankml = True
    routes.App.create_app = staticmethod(create_app)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=7873)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mode", choices=("interact", "view"), default="interact")
    a = ap.parse_args()
    if a.mode == "view":  # the LAN-safe stdlib server (not Gradio): sAGI/view.py
        import view
        sys.argv = [sys.argv[0], "--host", a.host if a.host != "127.0.0.1" else "0.0.0.0", "--port", str(a.port if a.port != 7873 else 7874)]
        return view.main()
    if a.mode == "interact" and a.host not in LOOPBACK:
        # Gradio 3.37 has published path-traversal CVEs, and interact runs the Verifier, devnet mints and Postgres
        # publishing: it is never offered to the network. view mode is the LAN page.
        sys.exit(f"refused: interact mode serves loopback only ({', '.join(sorted(LOOPBACK))}); use --mode view for the LAN")
    _trusted_hosts()
    canon = Canon(CANON)
    bad = [r for r in canon.rows if not r[2]]
    print(f"canon {CANON}: {len(canon.rows) - len(bad)}/{len(canon.rows)} ledger files verify" + (f"; FAILING: {bad}" if bad else ""))
    demo = build(canon, a.mode)
    try:  # concurrency > 1 so the 1 s timer ticks while an answer streams (Gradio 3: concurrency_count)
        demo.queue(concurrency_count=4)
    except TypeError:
        demo.queue(default_concurrency_limit=4)
    import models
    if a.mode == "interact" and "error" in serve_status() and os.environ.get("BANKML_FIRST_RUN", "1") != "0":
        # first run (or the carrier is down): bring in Bonsai-8B if absent, verify it, start it; the UI opens meanwhile
        models.start_job("first run: Bonsai-8B as the carrier", models.first_run, chat_busy)
        print("bankml serve not reachable: starting the first-run carrier in the background (Models tab shows progress)")
    import speak
    speak.VOICE_DIR.mkdir(parents=True, exist_ok=True)
    demo.launch(server_name=a.host, server_port=a.port, show_api=False, allowed_paths=[str(speak.VOICE_DIR), str(speak.EXPORT_DIR)])


if __name__ == "__main__":
    main()
