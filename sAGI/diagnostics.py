# SPDX-License-Identifier: MIT
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
