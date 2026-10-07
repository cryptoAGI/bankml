# SPDX-License-Identifier: MIT OR Apache-2.0
# The span model is ported from LlamaIndex (MIT, Copyright (c) Jerry Liu; LICENSING.md, "Other material").
"""Diagnostics for the bankML console: spans, events and trace trees, and a set of measured checks.

The span model is LlamaIndex's instrumentation (run-llama/llama_index, MIT; llama-index-instrumentation at ec837e5):
`span/simple.py` (SimpleSpan: start, end, duration, metadata), `span_handlers/base.py` (BaseSpanHandler: open,
completed and dropped spans under one lock; enter / exit / drop) and `span_handlers/simple.py` (SimpleSpanHandler:
the duration on exit, the error on drop, `_build_tree_by_parent`, and the repair of a span whose parent is missing).
Ported to the standard library — no pydantic, no treelib — with three changes: a contextvar holds the current span
(the dispatcher's job there), a span carries timed events (the event handlers' job there), and only the newest
traces are kept, so a long-running console does not grow without bound.

Nothing a visitor typed is ever put in a span: tags hold counts, sizes and statuses only.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import threading
import time
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field

KEEP = 400  # finished spans kept (newest); a trace whose root has been dropped from here is repaired, not lost


@dataclass
class Span:
    """SimpleSpan: when it began and ended, how long it took, what went wrong if it did."""
    id_: str
    name: str
    parent_id: str | None = None
    tags: dict = field(default_factory=dict)
    start: float = field(default_factory=time.time)
    t0: float = field(default_factory=time.perf_counter)
    end: float | None = None
    duration: float = 0.0
    error: str | None = None
    events: list = field(default_factory=list)  # (milliseconds after the start, name, data)

    def event(self, name: str, **data) -> None:
        self.events.append((round((time.perf_counter() - self.t0) * 1000, 1), name, data))

    def as_dict(self) -> dict:
        return {"id": self.id_, "name": self.name, "parent": self.parent_id, "tags": self.tags, "start": self.start,
                "duration_ms": round(self.duration * 1000, 1), "error": self.error,
                "events": [{"at_ms": t, "name": n, **d} for t, n, d in self.events]}


class SpanHandler:
    """BaseSpanHandler + SimpleSpanHandler: open spans by id; finished ones (completed or dropped) newest last."""

    def __init__(self, keep: int = KEEP):
        self.lock = threading.Lock()
        self.open: dict[str, Span] = {}
        self.done: deque[Span] = deque(maxlen=keep)

    def enter(self, span: Span) -> None:
        with self.lock:
            self.open.setdefault(span.id_, span)

    def exit(self, id_: str) -> None:
        with self.lock:
            s = self.open.pop(id_, None)
            if s:
                s.end = time.time()
                s.duration = time.perf_counter() - s.t0
                self.done.append(s)

    def drop(self, id_: str, err: BaseException) -> None:
        with self.lock:
            s = self.open.pop(id_, None)
            if s:
                s.end = time.time()
                s.duration = time.perf_counter() - s.t0
                s.error = f"{type(err).__name__}: {err}"
                self.done.append(s)

    def trees(self, limit: int = 20) -> list[dict]:
        """The trace trees, newest first: each root with its children nested, in start order. A span whose parent is
        no longer kept gets a stand-in parent `<id>-MISSING` (SimpleSpanHandler's repair), so nothing disappears."""
        with self.lock:
            spans = [s.as_dict() for s in self.done] + [{**s.as_dict(), "open": True} for s in self.open.values()]
        ids = {s["id"] for s in spans}
        for s in list(spans):
            if s["parent"] is not None and s["parent"] not in ids:
                s["parent"] += "-MISSING"
                if s["parent"] not in ids:
                    ids.add(s["parent"])
                    spans.append({"id": s["parent"], "name": "(older, not kept)", "parent": None, "tags": {}, "start": s["start"],
                                  "duration_ms": None, "error": None, "events": []})
        kids: dict[str | None, list] = {}
        for s in spans:
            kids.setdefault(s["parent"], []).append(s)

        def build(s: dict) -> dict:
            return {**s, "children": [build(c) for c in sorted(kids.get(s["id"], []), key=lambda c: c["start"])]}

        roots = sorted(kids.get(None, []), key=lambda s: s["start"], reverse=True)[:limit]
        return [build(r) for r in roots]


HANDLER = SpanHandler()
_CURRENT: contextvars.ContextVar[str | None] = contextvars.ContextVar("bankml_span", default=None)


@contextlib.contextmanager
def span(name: str, **tags):
    """Time a block as a span under the current one (the dispatcher's span_enter / span_exit / span_drop)."""
    s = Span(id_=f"{name}-{uuid.uuid4().hex[:8]}", name=name, parent_id=_CURRENT.get(), tags=tags)
    HANDLER.enter(s)
    token = _CURRENT.set(s.id_)
    try:
        yield s
    except BaseException as e:
        HANDLER.drop(s.id_, e)
        raise
    else:
        HANDLER.exit(s.id_)
    finally:
        _CURRENT.reset(token)


def render(trees: list[dict]) -> str:
    """The trees as text, one span per line with its duration, as SimpleSpanHandler.print_trace_trees shows them."""
    out: list[str] = []

    def walk(n: dict, pre: str, last: bool, root: bool) -> None:
        d = "open" if n.get("open") else ("—" if n["duration_ms"] is None else f"{n['duration_ms']} ms")
        out.append(("" if root else pre + ("└── " if last else "├── ")) + f"{n['name']} ({d})" + (f"  ✗ {n['error']}" if n["error"] else ""))
        for e in n["events"]:
            out.append(("" if root else pre + ("    " if last else "│   ")) + f"   · {e['name']} @ {e['at_ms']} ms")
        ch = n["children"]
        for i, c in enumerate(ch):
            walk(c, pre if root else pre + ("    " if last else "│   "), i == len(ch) - 1, False)

    for t in trees:
        walk(t, "", True, True)
        out.append("")
    return "\n".join(out)


def _probe(url: str, timeout: float = 5.0):
    """GET a JSON endpoint: (milliseconds, the JSON or None, the error or None)."""
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = json.loads(r.read())
        return round((time.perf_counter() - t) * 1000, 1), body, None
    except (OSError, ValueError) as e:
        return round((time.perf_counter() - t) * 1000, 1), None, str(e)


def checks(serve: str, engine_log: list[str]) -> list[dict]:
    """What can be measured about the engine right now; each check is ok, warn or bad, with what was seen."""
    out = []

    def add(name: str, level: str, seen: str):
        out.append({"check": name, "level": level, "seen": seen})

    ms, b, err = _probe(serve + "/bankml")
    if b is None:
        add("bankml serve answers", "bad", f"{serve}/bankml: {err}")
        return out
    add("bankml serve answers", "ok" if ms < 500 else "warn", f"GET /bankml in {ms} ms")
    v = b.get("verified") or {}
    add("the model is verified", "ok" if v.get("guard") == "play" else "bad",
        f"guard {v.get('guard')} · {v.get('name') or b.get('resident')} · sha256 {str(v.get('model_sha256') or '')[:16]}…")
    add("the engine", "ok", f"bankML {v.get('bankml')} · {v.get('engine')} · {v.get('arch') or ''}".strip(" ·"))
    ms, m, err = _probe(serve + "/bankml/metrics")
    if m is None:
        add("metrics", "warn", f"/bankml/metrics: {err}")
    else:
        last = (m.get("records") or [{}])[-1]
        n = len(m.get("records") or [])
        add("metrics", "ok", "no answer measured yet" if not n else
            f"{n} answers measured · last: TTFT {last.get('ttft_ms')} ms, prompt {last.get('prompt_tps')} tok/s, "
            f"generation {last.get('eval_tps')} tok/s")
    ms, u, err = _probe(serve + "/bankml/usage")
    if u is None:
        add("usage", "warn", f"/bankml/usage: {err}")
    else:
        avail = u.get("mem_available_bytes")
        add("memory available", "bad" if avail is not None and avail < 0.5e9 else "warn" if avail is not None and avail < 1.0e9 else "ok",
            "not measured" if avail is None else f"{avail / 1e9:.2f} GB available · bankML holds {(u.get('rss_bytes') or 0) / 1e9:.2f} GB")
        cpu = u.get("cpu_percent")
        add("CPU", "ok", "not measured" if cpu is None else f"{cpu:.0f} % of one core")
    bad = [l for l in engine_log if any(w in l.lower() for w in ("panicked", "error", "refuse", "failed"))]
    add("the engine log", "warn" if bad else "ok", bad[-1][:300] if bad else f"{len(engine_log)} recent lines, no error")
    with HANDLER.lock:
        dropped = [s for s in HANDLER.done if s.error]
    add("console spans", "warn" if dropped else "ok",
        f"{len(dropped)} of the last {len(HANDLER.done)} spans failed: {dropped[-1].name}: {dropped[-1].error}" if dropped
        else f"{len(HANDLER.done)} spans, none failed")
    return out


# ── every component of sAGI, each asked in its own words ──────────────────────────────────────────────────────────

def _component(name: str, file: str, role: str, fn) -> dict:
    """Run one component's check, timed; an exception is the component's own failure, said as it was raised."""
    t = time.perf_counter()
    try:
        level, seen = fn()
    except Exception as e:  # noqa: BLE001 — reported, never hidden
        level, seen = "bad", f"{type(e).__name__}: {e}"
    return {"component": name, "file": file, "role": role, "level": level, "seen": seen, "ms": round((time.perf_counter() - t) * 1000, 1)}


def _http(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read(4096)


def components(serve: str, here_port: int | None = None, public: bool = False, timeout: float = 8.0) -> list[dict]:
    """Every part of sAGI, checked now and in parallel: what each is for, whether it answers or is ready, and what it
    said. A check that does not finish within `timeout` is reported as such. Nothing is started, installed or written."""
    import concurrent.futures as cf
    import pathlib
    import shutil
    here = pathlib.Path(__file__).resolve().parent

    def serve_c():
        ms, b, err = _probe(serve + "/bankml")
        if b is None:
            return "bad", f"{serve}: {err}"
        v = b.get("verified") or {}
        return ("ok" if v.get("guard") == "play" else "warn"), f"{v.get('name') or b.get('resident')} · bankML {v.get('bankml')} · {ms} ms"

    def page(port: int, start: str):
        def f():
            try:
                code, _ = _http(f"http://127.0.0.1:{port}/")
                return ("ok" if code == 200 else "warn"), f"answering on 127.0.0.1:{port} (HTTP {code})"
            except OSError:
                return "info", f"not running — start it with: {start}"
        return f

    def models_c():
        import models
        pins = models.installed()
        pinned = sum(1 for m in pins if m.get("pinned"))
        bin_ok = models.BANKML.is_file()
        llama = models.LLAMA.is_file()
        lvl = "ok" if bin_ok and pinned else "warn"
        return lvl, (f"{pinned} pinned of {len(pins)} model files in {models.MODELS} · bankml binary {'present' if bin_ok else 'MISSING at ' + str(models.BANKML)}"
                     f" · llama-server {'present' if llama else 'absent (only the native engine)'} · engine setting {models.resources().get('engine', 'auto')}")

    def personas_c():
        import agents
        out, bad = [], 0
        for p in sorted((here / "personas").glob("*.persona")):
            d = json.loads(p.read_text())
            pf = agents.preflight(d)
            bad += bool(pf)
            out.append(f"{p.stem} {'✓' if not pf else '✗ ' + '; '.join(pf)[:80]} root {agents.doctrine_root(d)[:12]}…")
        return ("bad" if bad else "ok"), " · ".join(out) or "no persona files"

    def agents_c():
        import agents
        a = agents.list_agents()
        return "ok", (f"{len(a)} agents in {agents.AGENTS}: " + ", ".join(a[:8]) + (" …" if len(a) > 8 else "")) if a else f"no agents yet in {agents.AGENTS} (derive one from Savante)"

    def thot_c():
        import agents
        import thot
        a = agents.list_agents()
        bound = [x for x in a if (agents.AGENTS / x / f"{x}.thot.json").is_file()]
        if not bound:
            return "info", f"ready; none of {len(a)} agents bound into a THOT manifest yet (sagi.thot_manifest/1)"
        bad = {x: thot.verify(x) for x in bound}
        bad = {x: v for x, v in bad.items() if v}
        return ("ok" if not bad else "warn"), (f"{len(bound)} of {len(a)} agents bound; every manifest verifies" if not bad
                                               else "; ".join(f"{x}: {'; '.join(map(str, v))}" for x, v in bad.items())[:240])

    def chain_c():
        import chain
        art = chain.ARTIFACT
        return ("ok" if art.is_file() else "info"), (f"iNFT_7857 artifact present ({art}); a mint is prepared here, signed by its owner"
                                                    if art.is_file() else f"no iNFT_7857 artifact at {art} (forge build in DeltaVerse/deploy/iNFT4): mints cannot be prepared")

    def connectors_c():
        if not shutil.which("psql"):
            return "info", "no psql client: PostgreSQL publishing is off"
        import connectors
        s = connectors.status()
        if not s.get("ok"):
            return "info", f"PostgreSQL not reachable ({s.get('error')})"
        return "ok", f"PostgreSQL {s.get('server')} · db {s.get('db')} · pgvector {s.get('vector') or 'not installed'} · vectorscale {s.get('vectorscale') or 'not installed'}"

    def embed_c():
        import embed
        s = embed.status()
        return ("ok" if s.get("ready") else "info"), (f"{embed.MODEL} ready in Ollama{' (loaded)' if s.get('loaded') else ''}" if s.get("ready") else s.get("why", "not ready"))

    def speak_c():
        import speak
        ok, why = speak.available()
        return ("ok" if ok else "info"), ("the voice renders" if ok else f"no voice: {why}")

    def diag_c():
        with HANDLER.lock:
            n, failed = len(HANDLER.done), sum(1 for s in HANDLER.done if s.error)
        return ("warn" if failed else "ok"), f"{n} spans kept, {failed} failed"

    checks = [
        ("bankml serve", "bankML/serve.rs", "the verified engine every page talks to", serve_c),
        ("console", "sAGI/console.py", "this page: bankML as itself, its receipts, logs and diagnostics",
         lambda: ("ok", f"answering{' on port ' + str(here_port) if here_port else ''}{' · public, read-only' if public else ''}")),
        ("Savante", "sAGI/savante.py", "the chat UI (Gradio), Savante's persona over bankML", page(7873, "python3 sAGI/savante.py --mode interact")),
        ("view", "sAGI/view.py", "the read-only page for the LAN", page(7874, "python3 sAGI/view.py --host 0.0.0.0 --port 7874")),
        ("models", "sAGI/models.py", "the importer: pinned models, the carrier, its resources", models_c),
        ("personas", "sAGI/personas/", "who speaks: each .persona checked, its doctrine root", personas_c),
        ("agents", "sAGI/agents.py", "custom agents derived from the Savante template", agents_c),
        ("thot", "sAGI/thot.py", "THOT manifests: the dataset bundle an iNFT points to", thot_c),
        ("chain", "sAGI/chain.py", "prepares an iNFT mint the owner signs", chain_c),
        ("connectors", "sAGI/connectors.py", "PostgreSQL (pgvector): publish an agent, load it back", connectors_c),
        ("embed", "sAGI/embed.py", "embeddings (bge-m3 via the local Ollama) for history and memory", embed_c),
        ("voice", "sAGI/speak.py", "Savante's voice, rendered from open parts", speak_c),
        ("diagnostics", "sAGI/diagnostics.py", "spans and traces (after LlamaIndex's instrumentation)", diag_c),
    ]
    with cf.ThreadPoolExecutor(max_workers=len(checks)) as ex:
        futs = [(c, ex.submit(_component, *c)) for c in checks]
        out = []
        for (name, file, role, _), f in futs:
            try:
                out.append(f.result(timeout=timeout))
            except cf.TimeoutError:
                out.append({"component": name, "file": file, "role": role, "level": "warn", "seen": f"no answer within {timeout:.0f} s", "ms": timeout * 1000})
    return out
