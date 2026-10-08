#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""A persona's .context: what it knows of its own making, beside its .persona and .prompt.

A .context (built by tools/context.py) holds a `summary`, sent after the persona's system prompt so it sits in the
engine's cached prefix, and `chunks`, of which the few that match a question ride just before it, each with where it
came from. The bankML console reads bankml.context; Savante reads savante.context. Scoring is BM25 over the
question's own words (common words dropped) with a match required, the same as the Space's page does in JavaScript,
so a question brings the same passages everywhere. Stdlib only. Details: docs/modules/console.md
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

THRESHOLD = 2.5  # a passage about the question scores 3.5 and more; common words alone near 1.3
STOP = set("the and for are was were this that with from what which who whom how why when where does did doing done "
           "can could would should will shall may might must have has had having been being into onto over under about "
           "your yours you you're its it's they them their there here then than also just only very more most some any "
           "all not but yes our ours ask tell me my mine please".split())
_CACHE: dict = {}


def load(path: Path) -> dict:
    """The .context at `path`, read again when the file changes; {} when there is none or it does not parse."""
    try:
        m = Path(path).stat().st_mtime
    except OSError:
        return {}
    hit = _CACHE.get(str(path))
    if not hit or hit[0] != m:
        try:
            hit = (m, json.loads(Path(path).read_text(encoding="utf-8")))
        except ValueError:
            hit = (m, {})
        _CACHE[str(path)] = hit
    return hit[1] or {}


def summary_block(ctx: dict, heading: str) -> str:
    s = (ctx or {}).get("summary")
    return f"\n\n{heading}\n{s}" if s else ""


def _toks(t: str) -> list:
    return re.findall(r"[a-z0-9]+", t.lower())


def passages(ctx: dict, q: str, k: int, heading: str) -> tuple[str, list]:
    """The `k` chunks of `ctx` that best match `q`, as text for the late system message, and their ids;
    ("", []) when none matches, when `k` is 0, or when the question has no words of its own."""
    chunks = (ctx or {}).get("chunks") or []
    words = [w for w in _toks(" ".join(w for w in re.findall(r"[a-z0-9_.]+", (q or "").lower()) if w not in STOP and len(w) > 2))]
    if k <= 0 or not chunks or not words:
        return "", []
    docs = []
    for c in chunks:
        tf: dict = {}
        for w in _toks(f"{c['title']} {c['title']} {c['text']}"):
            tf[w] = tf.get(w, 0) + 1
        docs.append((c, tf, sum(tf.values())))
    n = len(docs)
    df: dict = {}
    for _, tf, _ in docs:
        for w in tf:
            df[w] = df.get(w, 0) + 1
    avg = sum(d[2] for d in docs) / n
    scored = []
    for c, tf, ln in docs:
        sc = sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * ln / avg))
                 for w in set(words) if w in tf)
        if sc >= THRESHOLD:
            scored.append((sc, c))
    scored.sort(key=lambda x: -x[0])
    hits = [c for _, c in scored[:k]]
    if not hits:
        return "", []

    def src(c):
        links = " · ".join(f"{k} {c[k.lower().replace(' ', '')]}" for k in ("GitHub", "Hugging Face") if c.get(k.lower().replace(" ", "")))
        return f" ({links})" if links else ""
    body = "\n".join(f"- [{c.get('source', c['id'])}] {c['title']}: {c['text']}{src(c)}" for c in hits)
    return f"{heading}\n{body}", [c["id"] for c in hits]
