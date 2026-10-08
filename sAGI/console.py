#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""bankML · console — talk to bankML as itself (sAGI/personas/bankml.persona), and watch what it measures.

Six tabs: **Ask** (the landing: a question, its streamed answer and receipt), **Admin** (CPU threads, RAM budget and
GPU limit as sliders; D3 charts of what bankml measures), **Engine** (bankml serve's own status: checks, CPU, memory, disk,
GPU, the answers measured and the engine's log, drawn by serve's renderer, `bankML/status.js`), **Receipts** (every exchange's receipt, and the iNFT
commitments: a Merkle root over the exchanges), **Logs** (the engine's log) and **Diagnostics** (measured checks of
the engine, and every answer's trace: its spans and their durations — `diagnostics.py`, after LlamaIndex's
instrumentation). A Savante | bankML switch links to
Savante's page on :7873.

The persona knows its own use only from measurement: each question goes to `bankml serve` with a SELF block built
from `GET /bankml/usage` and `GET /bankml/metrics` at that moment; a value that was not measured is null.

The Python standard library only, loopback only (it refuses another host), a fixed set of routes, a strict CSP
(no network but itself), and every POST must be JSON from this page's own origin.

Public mode (`--public HOST`, for a hosted demo such as a Hugging Face Space): it also answers as HOST and may bind any
address, but it is read-only — the resource controls are refused — and it keeps no record of visitors' questions:
nothing is logged, and the Receipts tab shows no exchange.

  python3 sAGI/console.py                 # http://127.0.0.1:7875 — bankml serve on 127.0.0.1:18093
  python3 sAGI/console.py --public bankml.example --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostics as D  # noqa: E402  (stdlib-only)
import models  # noqa: E402  (stdlib-only)
import savante as S  # noqa: E402  (stdlib-only at import; its Merkle tree and CIDv1)
import console_memory as CM  # noqa: E402  (stdlib-only: .memory per response window, the collection, recall)
import sysdiag  # noqa: E402  (stdlib-only: CPU, memory, disk and GPU from /proc and /sys)

SERVE = "http://" + models.LISTEN
STATIC = HERE / "console"
PERSONA = HERE / "personas" / "bankml.persona"
STATE = Path(os.environ.get("BANKML_UI_STATE", Path.home() / ".local" / "share" / "bankml" / "savante")).expanduser()
LOG = STATE / "console.jsonl"
MEM = CM.Store(STATE / "console-memory")
# where Savante answers on this computer (the switch in the bar); install.sh starts it on 7873
SAVANTE_URL = os.environ.get("BANKML_SAVANTE_URL", "http://127.0.0.1:7873/")
FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
         "/style.css": ("style.css", "text/css; charset=utf-8"), "/vendor/d3.v7.min.js": ("vendor/d3.v7.min.js", "text/javascript"),
         "/vendor/d3.LICENSE": ("vendor/d3.LICENSE", "text/plain; charset=utf-8"),
         # the Engine tab draws bankml serve's status with serve's own renderer (one source for both pages)
         "/engine.js": (HERE.parent / "bankML" / "status.js", "text/javascript; charset=utf-8"),
         "/engine.css": (HERE.parent / "bankML" / "status.css", "text/css; charset=utf-8"),
         # the Ask landing: the ultimate input field, built (console/uif/README.md); the plain box when absent
         "/uif.js": ("uif/uif.js", "text/javascript; charset=utf-8"), "/uif.css": ("uif/uif.css", "text/css; charset=utf-8"),
         # the Thesis tab: a scroll whose sections open as an accordion
         "/thesis.css": ("thesis.css", "text/css; charset=utf-8"),
         # the theme: depth, glass, the field's contrast, T mode, and the light/dark switch
         "/theme.css": ("theme.css", "text/css; charset=utf-8"), "/theme.js": ("theme.js", "text/javascript; charset=utf-8"),
         # the DeltaVerse $ (deltaverse.pythai.net/favicon.ico, MIT), the one Savante and view use
         "/favicon.ico": (HERE / "favicon.ico", "image/x-icon"),
         # the Diagnostics desk (drag, resize, the lights) and its Inspector (what was chosen, what to do)
         "/desk.js": ("desk.js", "text/javascript; charset=utf-8"), "/inspect.js": ("inspect.js", "text/javascript; charset=utf-8")}
LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
JOB = {"busy": False, "what": "", "error": "", "done": None}
# --public HOST: the one extra host name the console answers as; None = loopback only
PUBLIC: str | None = None
# the port this console answers on (for its own component check)
PORT: int | None = None
# public mode bounds each answer, as the hosted machine is shared
PUBLIC_MAX_TOKENS = 384
_LOCK = threading.Lock()


def persona() -> dict:
    return json.loads(PERSONA.read_text(encoding="utf-8"))


def _get(path: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(SERVE + path, timeout=timeout) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def self_block() -> dict:
    """What bankML knows of itself now, measured: null where it was not."""
    u, m, b = _get("/bankml/usage") or {}, _get("/bankml/metrics") or {}, _get("/bankml") or {}
    last = (m.get("records") or [{}])[-1]
    gpu = (u.get("gpus") or [{}])[0] if u.get("gpus") else {}
    lim = u.get("gpu_limiter") or {}
    v = b.get("verified") or {}
    return {"model": (v.get("name") or b.get("resident")), "model_sha256": v.get("model_sha256"), "bankml": v.get("bankml"),
            "prompt_tokens_total": m.get("prompt_tokens"), "completion_tokens_total": m.get("completion_tokens"),
            "last_ttft_ms": last.get("ttft_ms"), "last_prompt_tps": last.get("prompt_tps"), "last_eval_tps": last.get("eval_tps"),
            "cpu_percent": u.get("cpu_percent"), "rss_bytes": u.get("rss_bytes"), "mem_available_bytes": u.get("mem_available_bytes"),
            "gpu_busy_percent": gpu.get("busy_percent"), "gpu_limit": lim.get("limit"), "gpu_allocated_bytes": lim.get("allocated_bytes"),
            "package_watts": u.get("package_watts"), "joules_per_token": m.get("joules_per_token")}


def self_text(sb: dict) -> str:
    """The SELF block as the model reads it: two lines. The first says who is speaking, so the model file is never
    taken for the speaker (an A/B on Bonsai-8B, three fresh conversations each: "I am bankML" 3 of 3 with this line,
    1 of 3 with "the model I am running: …", which twice answered "I am Bonsai-8B"). The second carries every
    measurement, each with its unit (a bare key such as `last_eval_tps` was read as seconds in testing) and "not
    measured" where it was not: one line instead of nine, so it costs the prompt little whichever turn it rides on."""
    def v(x, unit="", scale=1.0, d=1):
        return "not measured" if x is None else f"{x * scale:.{d}f}{unit}"
    sha = str(sb["model_sha256"] or "")[:16]
    who = ("- I am bankML" + (f" {sb['bankml']}" if sb["bankml"] else "") + f". The model file I run is {sb['model'] or 'none'}"
           + (f" (sha256 {sha}…)" if sha else "") + ": it is what I compute with, not who I am")
    now = ("- measured now: tokens read in total " + v(sb["prompt_tokens_total"], d=0)
           + ", written " + v(sb["completion_tokens_total"], d=0)
           + "; my last answer: first token after " + v(sb["last_ttft_ms"], " milliseconds", d=0)
           + ", reading at " + v(sb["last_prompt_tps"], " tokens per second")
           + ", writing at " + v(sb["last_eval_tps"], " tokens per second")
           + "; CPU " + v(sb["cpu_percent"], " percent of one core")
           + "; memory I hold " + v(sb["rss_bytes"], " GB", 1e-9, 2) + ", still available " + v(sb["mem_available_bytes"], " GB", 1e-9, 2)
           + "; GPU busy " + v(sb["gpu_busy_percent"], " percent", d=0) + ", my GPU limit " + v(sb["gpu_limit"], " percent of its memory and time", 100, 0)
           + ", GPU memory I hold " + v(sb["gpu_allocated_bytes"], " MB", 1e-6, 0)
           + "; power " + v(sb["package_watts"], " watts") + ", energy per token I write " + v(sb["joules_per_token"], " joules", d=3))
    return who + "\n" + now

def answers(url: str, timeout: float = 1.5) -> bool:
    """Whether a page answers at `url` (any HTTP status counts: something is listening and speaking HTTP)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (OSError, ValueError):
        return False


def state() -> dict:
    p = persona()
    return {"serve": _get("/bankml"), "usage": _get("/bankml/usage"), "metrics": _get("/bankml/metrics"), "resources": models.resources(),
            "persona": {"name": p["name"], "mantra": p["mantra"], "doctrine_root": _doctrine_root(p)}, "job": JOB, "self": self_block(),
            "public": PUBLIC is not None, "warm": WARM["state"],
            "savante": None if PUBLIC else {"url": SAVANTE_URL, "up": answers(SAVANTE_URL)}}


def _doctrine_root(p: dict) -> str | None:
    try:
        import agents
        return agents.doctrine_root(p)
    except Exception:  # noqa: BLE001 — shown as unknown, never guessed
        return None


def log_lines() -> list:
    try:
        return [l for l in LOG.read_bytes().split(b"\n") if l.strip()]
    except OSError:
        return []


def engine_tail(n: int = 60) -> list:
    try:
        return models.LOG.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def diagnostics() -> dict:
    """The Diagnostics tab: measured checks of the engine now, and the newest answers' traces (spans, durations,
    events). A trace's tags carry counts and statuses, never a question's text, so public mode shows them too."""
    trees = D.HANDLER.trees()
    sci = None
    try:  # the newest scientific.diagnostic (testing/scientific_diagnostic.py): its verdict and summary, every number to 18 decimals
        d = json.loads((models.LOG.parent / "scientific.diagnostic").read_text())
        sci = {k: d.get(k) for k in ("at", "verdict", "bankml", "reference", "model", "threads", "precision", "summary", "timing", "answer")}
    except (OSError, ValueError):
        pass
    comps = D.components(SERVE, PORT, PUBLIC is not None)
    if PUBLIC:  # a public console says how each part is, not where its files are or what its agents are called
        comps = [{k: c[k] for k in ("component", "file", "role", "level", "ms")} | {"seen": ""} for c in comps]
    return {"components": comps, "checks": D.checks(SERVE, engine_tail()), "traces": trees, "text": D.render(trees), "scientific": sci,
            "source": "sAGI/diagnostics.py — after LlamaIndex's instrumentation (MIT): SimpleSpan, SimpleSpanHandler"}


THESIS_ANCHOR = "thesis--professor-codephreak-and-gregory-l-magnusson"


def thesis() -> dict:
    """The Thesis tab: the authors' thesis and the contributions measured against it, read from docs/TECHNICAL.md
    (from its `## Thesis` heading to the next `## `), so the page and the report never drift apart."""
    try:
        text = (HERE.parent / "docs" / "TECHNICAL.md").read_text(encoding="utf-8")
    except OSError as e:
        return {"error": str(e)}
    start = text.find("\n## Thesis")
    end = text.find("\n## ", start + 5)
    if start < 0:
        return {"error": "docs/TECHNICAL.md has no Thesis section"}
    return {"markdown": text[start + 1:end if end > 0 else None].strip(), "source": "docs/TECHNICAL.md",
            "url": f"https://github.com/cryptoAGI/bankml/blob/main/docs/TECHNICAL.md#{THESIS_ANCHOR}",
            "base": "https://github.com/cryptoAGI/bankml/blob/main/docs/"}


def infotags() -> dict:
    """The metadata an iNFT publication of this session carries (ERC-721 metadata shape, `attributes`), with the
    commitments that let a holder check any one exchange: an RFC 6962 Merkle root over the log's lines and its CIDv1."""
    st, p, lines = state(), persona(), ([] if PUBLIC else log_lines())
    v = (st["serve"] or {}).get("verified") or {}
    root = S.merkle_root([S._leaf(l) for l in lines])
    m = st["metrics"] or {}
    attrs = [("engine", "bankML " + str(v.get("bankml"))), ("model", v.get("name")), ("model_sha256", v.get("model_sha256")),
             ("persona", p["name"]), ("doctrine_root", st["persona"]["doctrine_root"]), ("exchanges", len(lines)),
             ("prompt_tokens", m.get("prompt_tokens")), ("completion_tokens", m.get("completion_tokens")),
             ("joules_per_token", m.get("joules_per_token")), ("merkle_scheme", S.MERKLE_SCHEME), ("exchanges_root", root)]
    return {"name": f"bankML session · {v.get('name') or 'no model'}", "description": p["token"]["public_metadata"]["description"],
            "external_url": "https://github.com/cryptoAGI/bankml",
            "attributes": [{"trait_type": k, "value": val} for k, val in attrs if val is not None],
            "bankml": {"exchanges_root": root, "exchanges_cid": S.cid_v1_raw(b"\n".join(lines)) if lines else None,
                       "persona_cid": S.cid_v1_raw(PERSONA.read_bytes()), "generated_at": int(time.time())}}


def apply(threads: int, ram_gb: float, gpu_limit: float) -> None:
    with _LOCK:
        if JOB["busy"]:
            raise RuntimeError("a restart is already running")
        JOB.update(busy=True, what=f"restarting with {threads} threads, {ram_gb:.1f} GB, GPU {gpu_limit:.0%}", error="", done=None)

    def run():
        with D.span("apply", threads=threads, ram_gb=ram_gb, gpu_limit=gpu_limit) as sp:
            try:
                models.apply_resources(threads, ram_gb, engine="native", gpu_limit=gpu_limit)
                JOB["done"] = time.time()
            except Exception as e:  # noqa: BLE001 — reported to the page
                JOB["error"] = sp.error = str(e)
            finally:
                JOB["busy"] = False
    threading.Thread(target=run, daemon=True).start()


# ── the first answer: the persona's prefix kept warm ────────────────────────────────────────────────────────────
SELF_HEAD = "SELF (measured by bankML just now):\n"
# what the warmer did for the engine running now: its key (engine instance, slot file), its state, and the connection of
# a prefill in progress (closed by a question that arrives meanwhile)
WARM: dict = {"key": None, "state": "idle", "conn": None}


def messages(system: str, hist: list, q: str, self_txt: str, recall: str = "") -> list:
    """The prompt as sent. The persona's system prompt comes first and never changes, so the engine reuses it, and the
    conversation before this question, from its prompt cache; SELF changes with every question, so it comes last, as a
    second system message just before the question. Inside the first system message it made the engine recompute
    everything after it on every turn (376 of 551 tokens reused, 66 s to the first token, Bonsai-8B on the laptop).
    Recall from .history, when asked for, rides in the same late message: it changes with every question too."""
    late = SELF_HEAD + self_txt + ("\n\n" + recall if recall else "")
    return [{"role": "system", "content": system}, *hist, {"role": "system", "content": late}, {"role": "user", "content": q}]


def ping() -> dict:
    """One round trip from the console to bankml serve (`GET /health`, which runs no model), timed here."""
    return sysdiag.ping(SERVE)


def sysdiag_sections() -> list:
    """The machine now (sysdiag: CPU, memory, disk, GPU), then the engine's own view when bankml serve answers."""
    st, u = _get("/bankml/status") or {}, _get("/bankml/usage") or {}
    disk = st.get("disk") or {}
    paths = {"models": disk.get("path") or str(models.MODELS), "state": str(STATE)}
    return sysdiag.collect({k: v for k, v in paths.items() if v}) + [sysdiag.engine(st, u, ping())]


def records(source: str) -> list:
    """The exchanges of a .history as {"at", "question", "answer", "window"}: the console's own log, or Savante's."""
    if source == "savante":
        return [{"at": r.get("ts"), "question": r.get("user", ""), "answer": r.get("assistant", ""), "window": r.get("session")}
                for r in S.history_all()]
    out = []
    for l in log_lines():
        try:
            r = json.loads(l)
        except ValueError:
            continue
        out.append({"at": r.get("at"), "question": r.get("question", ""), "answer": r.get("answer", ""), "window": r.get("window")})
    return out


def window(hist: list, keep: int = 12, step: int = 8) -> list:
    """The conversation sent with a question: at least the last `keep` messages, trimmed `step` at a time (four
    exchanges), so the start of the window, and with it the cached prefix, moves once every four turns instead of on
    every turn once the conversation is longer than `keep` (a window that slid each turn changed the prompt right after
    the system prompt, and the engine recomputed everything)."""
    h = [m for m in hist if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    start = max(0, len(h) - keep) // step * step
    return h[start:]


def _slot_key(st: dict | None):
    """(engine instance, slot file) for the persona's prefix on the verified engine now running; None without one."""
    v = (st or {}).get("verified") or {}
    if not v.get("model_sha256"):
        return None
    h = hashlib.sha256(f"{v['model_sha256']}|{S.engine_ctx()}|{persona()['system_prompt']}".encode()).hexdigest()[:24]
    return st.get("hashed_at"), f"console-{h}.bin"


def warm() -> str:
    """Before the first question to a newly started engine: restore the persona's saved slot, or else prefill the
    persona and save its slot, so the first answer does not pay for the persona at prefill speed (about 300 tokens, two
    minutes on Bonsai-8B here). Only on an engine that has answered nothing: its one slot then holds no one's
    conversation. A question that arrives during the prefill closes it (`cancel_warm`): the engine stops when its
    client goes away (0.4.1) and keeps what it computed, so the question starts from there."""
    st = _get("/bankml")
    k = _slot_key(st)
    if not k or WARM["key"] == k:
        return WARM["state"]
    WARM.update(key=k, state="checking")
    if ((_get("/bankml/metrics") or {}).get("records")):
        WARM["state"] = "engine in use"  # it has answered: its slot is a conversation, not ours to replace
        return WARM["state"]
    if (models.SLOTS / k[1]).is_file():
        try:
            S._slot_call("restore", k[1])
            WARM["state"] = "restored"
            return WARM["state"]
        except (OSError, ValueError):
            pass  # an engine without --slot-dir, or a file it will not take: prefill instead
    WARM["state"] = "prefilling"
    host, port = models.LISTEN.rsplit(":", 1)
    conn = http.client.HTTPConnection(host, int(port), timeout=1800)
    WARM["conn"] = conn
    body = {"messages": [{"role": "system", "content": persona()["system_prompt"]}], "max_tokens": 1, "temperature": 0, "stream": False}
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        conn.getresponse().read()
    except (OSError, http.client.HTTPException):
        WARM["state"] = "cancelled by a question"
        return WARM["state"]
    finally:
        WARM["conn"] = None
        conn.close()
    try:
        S._slot_call("save", k[1])
        WARM["state"] = "prefilled and saved"
        for old in sorted(models.SLOTS.glob("console-*.bin"), key=lambda f: f.stat().st_mtime)[:-3]:
            old.unlink(missing_ok=True)
    except (OSError, ValueError):
        WARM["state"] = "prefilled"
    return WARM["state"]


def cancel_warm() -> None:
    """A question has come: close a prefill in progress, so the engine turns to the question."""
    conn = WARM.get("conn")
    sock = getattr(conn, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(2)
        except OSError:
            pass


def warmer(every: float = 15.0) -> None:
    """Watch for a newly started engine (its `hashed_at` changes) and warm it; quiet when nothing changes."""
    while True:
        try:
            warm()
        except Exception as e:  # noqa: BLE001 — never take the console down; tried again on the next engine
            WARM["state"] = f"error: {e}"
        time.sleep(every)


class H(BaseHTTPRequestHandler):
    server_version = "bankml-console"

    def log_message(self, *a):
        pass

    def _host_ok(self) -> bool:
        h = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return h in LOOPBACK or (PUBLIC is not None and h == PUBLIC)

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, b"loopback only", "text/plain")
        path = self.path.split("?", 1)[0]
        if path in FILES:
            name, ctype = FILES[path]
            try:
                return self._send(200, (STATIC / name).read_bytes(), ctype)
            except OSError:
                return self._send(404, b"not built", "text/plain")
        if path == "/api/state":
            return self._json(state())
        if path == "/api/log":
            rows = [] if PUBLIC else [json.loads(l) for l in log_lines()[-200:]]
            return self._json({"exchanges": rows, "engine_log": engine_tail()})
        if path == "/api/infotags":
            return self._json(infotags())
        if path == "/api/diagnostics":
            return self._json(diagnostics())
        if path == "/api/thesis":
            return self._json(thesis())
        if path == "/api/ping":
            return self._json(ping())
        if path == "/api/sysdiag":
            if PUBLIC:
                return self._send(403, b"a public console does not describe its machine", "text/plain")
            with D.span("sysdiag"):
                return self._json({"at": time.time(), "host": os.uname().nodename, "sections": sysdiag_sections()})
        if path in ("/api/memory", "/api/history"):
            if PUBLIC:
                return self._send(403, b"a public console keeps no memory and shows no history", "text/plain")
            qs = dict(p.split("=", 1) for p in (self.path.split("?", 1)[1].split("&") if "?" in self.path else []) if "=" in p)
            from urllib.parse import unquote_plus
            qs = {k: unquote_plus(v) for k, v in qs.items()}
            if path == "/api/memory":
                w = qs.get("window")
                body = {"settings": MEM.settings(), "defaults": CM.DEFAULTS, "limits": CM.LIMITS, "windows": MEM.windows(),
                        "collection": len(MEM.notes(CM.COLLECTION))}
                if w:
                    body.update(window=CM.slug(w), notes=MEM.notes(w), context=MEM.context(w, MEM.settings()))
                return self._json(body)
            src = qs.get("source") if qs.get("source") in CM.SOURCES else "console"
            recs = records(src)
            try:
                lim = max(1, min(int(qs.get("limit") or 40), 200))
            except ValueError:
                lim = 40
            hits = CM.search(recs, qs.get("q", ""), lim, S._BM25)
            return self._json({"source": src, "of": len(recs), "exchanges": [{"i": i, **r} for i, r in hits]})
        if path == "/api/engine":
            s = _get("/bankml/status", 10)
            return self._json(s) if s is not None else self._json({"error": f"bankml serve does not answer at {SERVE}"}, 502)
        self._send(404, b"not found", "text/plain")

    def _body(self):
        if not self._host_ok():
            return None, "loopback only"
        origin = self.headers.get("Origin")
        if origin and origin.split("://", 1)[-1].rsplit(":", 1)[0] not in (*LOOPBACK, *([PUBLIC] if PUBLIC else [])):
            return None, "cross-origin"
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return None, "POST bodies must be application/json"
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1 << 20:
            return None, "too large"
        try:
            return json.loads(self.rfile.read(n) or b"{}"), None
        except ValueError:
            return None, "not JSON"

    def do_POST(self):
        req, err = self._body()
        if err:
            return self._send(403 if err in ("loopback only", "cross-origin") else 400, err.encode(), "text/plain")
        path = self.path.split("?", 1)[0]
        if path == "/api/resources":
            if PUBLIC:
                return self._send(403, b"read-only: this is a public console", "text/plain")
            try:
                apply(int(req["threads"]), float(req["ram_gb"]), float(req["gpu_limit"]))
                return self._json({"ok": True, "job": JOB})
            except (KeyError, ValueError, RuntimeError) as e:
                return self._json({"ok": False, "error": str(e)}, 400)
        if path == "/api/ask":
            return self._ask(req)
        if path == "/api/memory":
            if PUBLIC:
                return self._send(403, b"a public console keeps no memory", "text/plain")
            return self._memory(req)
        self._send(404, b"not found", "text/plain")

    def _memory(self, req: dict):
        """.memory per window: add, remove, the options, and collect exchanges of a .history into a window or the
        collection."""
        a, w = req.get("action"), req.get("window") or "main"
        try:
            if a == "add":
                n = MEM.add(w, req.get("text", ""))
            elif a == "remove":
                n = MEM.remove(w, int(req.get("n", 0)))
            elif a == "settings":
                return self._json({"ok": True, "settings": MEM.save_settings(req.get("settings") or {})})
            elif a == "collect":
                src = req.get("source") if req.get("source") in CM.SOURCES else "console"
                recs = records(src)
                picked = [recs[i] for i in req.get("indices") or [] if isinstance(i, int) and 0 <= i < len(recs)]
                if not picked:
                    raise CM.MemoryError_("no exchange chosen")
                for r in picked:
                    n = MEM.add(w, CM.exchange_note(r), {"kind": "history", "source": src, "at": r.get("at")})
            else:
                raise CM.MemoryError_("action: add, remove, settings or collect")
        except (CM.MemoryError_, ValueError, TypeError) as e:
            return self._json({"ok": False, "error": str(e)}, 400)
        return self._json({"ok": True, "window": CM.slug(w), "notes": n})

    def _ask(self, req: dict):
        """Stream an answer as NDJSON lines {"piece"} … then {"done", "receipt", "timings", "self"}; log the exchange.
        Traced: the span `ask` holds `self_block`, `engine.stream` (its first piece as an event), `metrics`,
        `receipt.verify` and `log.write`."""
        q = str(req.get("message", "")).strip()
        if not q:
            return self._json({"error": "an empty question"}, 400)
        try:
            int(req.get("max_tokens") or 256)
        except (TypeError, ValueError):
            return self._json({"error": "max_tokens: a whole number"}, 400)
        cancel_warm()
        with D.span("ask", chars=len(q), history=len(req.get("history") or []), public=PUBLIC is not None) as root:
            self._ask_traced(req, q, root)

    def _ask_traced(self, req: dict, q: str, root):
        with D.span("self_block"):
            sb = self_block()
        hist = window(req.get("history") or [])
        win = CM.slug(req.get("window"))
        system, recalled, recall_text = persona()["system_prompt"], [], ""
        if not PUBLIC:  # a public console uses no memory and recalls nothing: they are the operator's
            opts = CM.clamp({**MEM.settings(), **{k: v for k, v in (req.get("options") or {}).items() if k in CM.DEFAULTS}})
            with D.span("memory", window=win):
                system += MEM.context(win, opts)
            if opts["recall_k"]:
                with D.span("recall", k=opts["recall_k"], source=opts["recall_source"]):
                    recall_text, recalled = CM.recall(records(opts["recall_source"]), q, opts["recall_k"],
                                                      {m.get("content") for m in hist if m.get("role") == "user"}, S._BM25)
        body = {"messages": messages(system, hist, q, self_text(sb), recall_text), "stream": True,
                "max_tokens": min(int(req.get("max_tokens") or 256), PUBLIC_MAX_TOKENS if PUBLIC else 1 << 30)}
        for k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty", "seed"):
            if req.get(k) is not None:
                body[k] = req[k]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        text, receipt, err, t0, pieces = "", None, None, time.time(), 0
        with D.span("engine.stream", max_tokens=body["max_tokens"], messages=len(body["messages"])) as es:
            try:
                r = urllib.request.Request(SERVE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(r, timeout=1800) as resp:
                    es.event("headers", status=resp.status)
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        d = json.loads(line[5:])
                        if "bankml_receipt" in d:
                            receipt = d["bankml_receipt"]
                            es.event("receipt")
                            continue
                        piece = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                        if piece:
                            if not pieces:
                                es.event("first piece")
                            pieces += 1
                            text += piece
                            self.wfile.write((json.dumps({"piece": piece}) + "\n").encode())
                            self.wfile.flush()
            except urllib.error.HTTPError as e:
                err = e.read().decode("utf-8", "replace")[:500] or str(e)
            except (OSError, ValueError) as e:
                err = str(e)
            es.tags.update(pieces=pieces, chars=len(text))
            es.error = err and err[:300]
        with D.span("metrics"):
            last = ((_get("/bankml/metrics") or {}).get("records") or [{}])[-1]
        if not PUBLIC:  # a public console keeps no record of its visitors' questions
            with D.span("log.write"):
                rec = {"at": int(t0), "question": q, "answer": text, "error": err, "receipt": receipt, "metrics": last, "self_before": sb,
                       "window": win, **({"recall": recalled} if recalled else {})}
                STATE.mkdir(parents=True, exist_ok=True)
                with LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with D.span("receipt.verify") as rv:
            ok = receipt is not None and hashlib.sha256(text.encode()).hexdigest() == receipt.get("response_sha256")
            rv.tags.update(receipt=receipt is not None, ok=ok)
        root.tags.update(ok=ok, ttft_ms=last.get("ttft_ms"), eval_tps=last.get("eval_tps"))
        self.wfile.write((json.dumps({"done": True, "error": err, "receipt": receipt, "metrics": last,
                                      "answer_sha256_ok": ok}) + "\n").encode())
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7875)
    ap.add_argument("--public", metavar="HOST", help="also answer as HOST, read-only and keeping no questions (a hosted demo)")
    a = ap.parse_args()
    global PUBLIC, PORT
    PUBLIC, PORT = a.public, a.port
    if a.host not in ("127.0.0.1", "localhost", "::1") and not PUBLIC:
        sys.exit("the console is loopback only: it can restart the engine (use view.py for the LAN, --public for a hosted demo)")
    print(f"bankML console on http://{a.host}:{a.port} — bankml serve at {SERVE}")
    threading.Thread(target=warmer, daemon=True).start()  # the first answer: the persona's prefix kept warm
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
