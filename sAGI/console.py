#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""bankML · console — talk to bankML as itself (sAGI/personas/bankml.persona), and watch what it measures.

Four tabs: **Interaction** (a question and its streamed answer, with the receipt), **Admin** (the CPU threads, the RAM
budget and the GPU limit as sliders, and D3 charts of what bankml measures), **Logging** (every exchange with its
timings and receipt, refusals, the engine's log) and **Infotags** (the metadata an iNFT publication of this session
carries: the model and persona it ran, and a Merkle root over its exchanges).

The persona knows its own use only from measurement: each question goes to `bankml serve` with a SELF block built
from `GET /bankml/usage` and `GET /bankml/metrics` at that moment; a value that was not measured is null.

The Python standard library only, loopback only (it refuses another host), a fixed set of routes, a strict CSP
(no network but itself), and every POST must be JSON from this page's own origin.

  python3 sAGI/console.py                 # http://127.0.0.1:7875 — bankml serve on 127.0.0.1:18093
"""
from __future__ import annotations

import argparse
import hashlib
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
import models  # noqa: E402  (stdlib-only)
import savante as S  # noqa: E402  (stdlib-only at import; its Merkle tree and CIDv1)

SERVE = "http://" + models.LISTEN
STATIC = HERE / "console"
PERSONA = HERE / "personas" / "bankml.persona"
STATE = Path(os.environ.get("BANKML_UI_STATE", Path.home() / ".local" / "share" / "bankml" / "savante")).expanduser()
LOG = STATE / "console.jsonl"
FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
         "/style.css": ("style.css", "text/css; charset=utf-8"), "/vendor/d3.v7.min.js": ("vendor/d3.v7.min.js", "text/javascript"),
         "/vendor/d3.LICENSE": ("vendor/d3.LICENSE", "text/plain; charset=utf-8")}
LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
JOB = {"busy": False, "what": "", "error": "", "done": None}
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
    """The SELF block as the model reads it: one plain sentence per measurement, with its unit (a bare key such as
    `last_eval_tps` was read as seconds in testing); a value not measured says so."""
    def v(x, unit="", scale=1.0, d=1):
        return "not measured" if x is None else f"{x * scale:.{d}f}{unit}"
    return "\n".join([
        f"- the model I am running: {sb['model'] or 'none'} (sha256 {str(sb['model_sha256'] or '')[:16]}…), bankML {sb['bankml']}",
        f"- tokens I have read in total (prompts): {v(sb['prompt_tokens_total'], d=0)}",
        f"- tokens I have written in total (answers): {v(sb['completion_tokens_total'], d=0)}",
        f"- time to first token of my last answer: {v(sb['last_ttft_ms'], ' milliseconds', d=0)}",
        f"- prompt reading speed of my last answer: {v(sb['last_prompt_tps'], ' tokens per second')}",
        f"- generation speed of my last answer: {v(sb['last_eval_tps'], ' tokens per second')}",
        f"- CPU use right now: {v(sb['cpu_percent'], ' percent of one core')}",
        f"- memory I hold (RSS): {v(sb['rss_bytes'], ' GB', 1e-9, 2)}; memory still available on this machine: {v(sb['mem_available_bytes'], ' GB', 1e-9, 2)}",
        f"- GPU busy right now: {v(sb['gpu_busy_percent'], ' percent', d=0)}; my GPU limit: {v(sb['gpu_limit'], ' percent of its memory and time', 100, 0)}; GPU memory I hold: {v(sb['gpu_allocated_bytes'], ' MB', 1e-6, 0)}",
        f"- power the CPU package draws: {v(sb['package_watts'], ' watts')}; energy per token I write: {v(sb['joules_per_token'], ' joules', d=3)}",
    ])


def state() -> dict:
    p = persona()
    return {"serve": _get("/bankml"), "usage": _get("/bankml/usage"), "metrics": _get("/bankml/metrics"), "resources": models.resources(),
            "persona": {"name": p["name"], "mantra": p["mantra"], "doctrine_root": _doctrine_root(p)}, "job": JOB, "self": self_block()}


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


def infotags() -> dict:
    """The metadata an iNFT publication of this session carries (ERC-721 metadata shape, `attributes`), with the
    commitments that let a holder check any one exchange: an RFC 6962 Merkle root over the log's lines and its CIDv1."""
    st, p, lines = state(), persona(), log_lines()
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
        try:
            models.apply_resources(threads, ram_gb, engine="native", gpu_limit=gpu_limit)
            JOB["done"] = time.time()
        except Exception as e:  # noqa: BLE001 — reported to the page
            JOB["error"] = str(e)
        finally:
            JOB["busy"] = False
    threading.Thread(target=run, daemon=True).start()


class H(BaseHTTPRequestHandler):
    server_version = "bankml-console"

    def log_message(self, *a):
        pass

    def _host_ok(self) -> bool:
        h = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return h in LOOPBACK

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
            return self._send(200, (STATIC / name).read_bytes(), ctype)
        if path == "/api/state":
            return self._json(state())
        if path == "/api/log":
            rows = [json.loads(l) for l in log_lines()[-200:]]
            tail = []
            try:
                tail = models.LOG.read_text(errors="replace").splitlines()[-60:]
            except OSError:
                pass
            return self._json({"exchanges": rows, "engine_log": tail})
        if path == "/api/infotags":
            return self._json(infotags())
        self._send(404, b"not found", "text/plain")

    def _body(self):
        if not self._host_ok():
            return None, "loopback only"
        origin = self.headers.get("Origin")
        if origin and origin.split("://", 1)[-1].rsplit(":", 1)[0] not in LOOPBACK:
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
            try:
                apply(int(req["threads"]), float(req["ram_gb"]), float(req["gpu_limit"]))
                return self._json({"ok": True, "job": JOB})
            except (KeyError, ValueError, RuntimeError) as e:
                return self._json({"ok": False, "error": str(e)}, 400)
        if path == "/api/ask":
            return self._ask(req)
        self._send(404, b"not found", "text/plain")

    def _ask(self, req: dict):
        """Stream an answer as NDJSON lines {"piece"} … then {"done", "receipt", "timings", "self"}; log the exchange."""
        q = str(req.get("message", "")).strip()
        if not q:
            return self._json({"error": "an empty question"}, 400)
        sb = self_block()
        system = persona()["system_prompt"] + "\n\nSELF (measured by bankML just now):\n" + self_text(sb)
        hist = [m for m in (req.get("history") or [])[-12:] if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        body = {"messages": [{"role": "system", "content": system}, *hist, {"role": "user", "content": q}], "stream": True,
                "max_tokens": int(req.get("max_tokens") or 256)}
        for k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty", "seed"):
            if req.get(k) is not None:
                body[k] = req[k]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        text, receipt, err, t0 = "", None, None, time.time()
        try:
            r = urllib.request.Request(SERVE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=1800) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    if "bankml_receipt" in d:
                        receipt = d["bankml_receipt"]
                        continue
                    piece = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                    if piece:
                        text += piece
                        self.wfile.write((json.dumps({"piece": piece}) + "\n").encode())
                        self.wfile.flush()
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "replace")[:500] or str(e)
        except (OSError, ValueError) as e:
            err = str(e)
        last = ((_get("/bankml/metrics") or {}).get("records") or [{}])[-1]
        rec = {"at": int(t0), "question": q, "answer": text, "error": err, "receipt": receipt, "metrics": last, "self_before": sb}
        STATE.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.wfile.write((json.dumps({"done": True, "error": err, "receipt": receipt, "metrics": last,
                                      "answer_sha256_ok": receipt is not None and hashlib.sha256(text.encode()).hexdigest() == receipt.get("response_sha256")}) + "\n").encode())
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7875)
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        sys.exit("the console is loopback only: it can restart the engine (use view.py for the LAN)")
    print(f"bankML console on http://{a.host}:{a.port} — bankml serve at {SERVE}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
