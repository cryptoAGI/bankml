#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The console's .memory: one per response window, a collection above them, and recall from .history.

Each response window of the Ask landing keeps its own notes, named after the window's title (titles survive a page
reload; the field's ids do not). Above them, the *collection*: notes gathered by reviewing .history, the higher-level
context every window shares. A question carries, in its first system message after the persona, the collection
and then its window's notes, each the newest that fit a budget and listed oldest first (a new note is appended, so the
engine keeps the cached prompt before it); recall, when asked for, rides just before the question.

Storage: `$BANKML_UI_STATE/console-memory/<window>.jsonl` and `_collection.jsonl`, one JSON object per note, written
whole and atomically; `settings.json` holds the options. Bounded: LIMITS. Details: docs/modules/console.md
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

COLLECTION = "_collection"
DEFAULTS = {"use_memory": True, "use_collection": True, "memory_budget": 2400, "collection_budget": 1600,
            "recall_k": 0, "recall_source": "console"}
LIMITS = {"note_chars": 1000, "notes_per_window": 200, "windows": 64, "memory_budget": 6000, "collection_budget": 6000,
          "recall_k": 4, "collect_chars": 600, "recall_chars": 700}
SOURCES = ("console", "savante")


class MemoryError_(ValueError):
    """A request the memory refuses: a bad window name, a full window, an empty note."""


def slug(name: str | None) -> str:
    """A window's memory name from its title: lower case, letters, digits and hyphens, at most 40 ("Output 1" →
    "output-1"); empty → "main", the plain question box's. The collection's own name is accepted as is."""
    if name == COLLECTION:
        return COLLECTION
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")[:40].strip("-")
    return s or "main"


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)

    # ── files ────────────────────────────────────────────────────────────────────────────────────────────────────
    def _path(self, window: str) -> Path:
        return self.root / f"{slug(window)}.jsonl"

    def notes(self, window: str) -> list:
        try:
            lines = self._path(window).read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for l in lines:
            try:
                out.append(json.loads(l))
            except ValueError:
                continue  # a damaged line is skipped, never allowed to hide the rest
        return out

    def _write(self, window: str, notes: list) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self._path(window)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text("".join(json.dumps(n, ensure_ascii=False) + "\n" for n in notes), encoding="utf-8")
        os.replace(tmp, p)  # never half a .memory

    def windows(self) -> list:
        """[{"name", "notes", "chars"}] for every window with notes, the collection excluded."""
        if not self.root.is_dir():
            return []
        out = []
        for p in sorted(self.root.glob("*.jsonl")):
            if p.stem == COLLECTION:
                continue
            n = self.notes(p.stem)
            out.append({"name": p.stem, "notes": len(n), "chars": sum(len(x.get("text", "")) for x in n)})
        return out

    def add(self, window: str, text: str, source: dict | None = None) -> int:
        text = " ".join(str(text or "").split())
        if not text:
            raise MemoryError_("an empty note")
        if len(text) > LIMITS["note_chars"]:
            text = text[:LIMITS["note_chars"] - 1] + "…"
        w = slug(window)
        notes = self.notes(w)
        if len(notes) >= LIMITS["notes_per_window"]:
            raise MemoryError_(f"{w} holds {LIMITS['notes_per_window']} notes: remove some first")
        if not notes and w != COLLECTION and len(self.windows()) >= LIMITS["windows"]:
            raise MemoryError_(f"{LIMITS['windows']} windows have notes: clear one first")
        if any(n.get("text") == text for n in notes):
            return len(notes)  # the same note twice is one note
        notes.append({"at": round(time.time(), 3), "text": text, "source": source or {"kind": "typed"}})
        self._write(w, notes)
        return len(notes)

    def remove(self, window: str, n: int) -> int:
        notes = self.notes(window)
        if not 1 <= n <= len(notes):
            raise MemoryError_(f"no note {n} in {slug(window)}")
        del notes[n - 1]
        if notes:
            self._write(window, notes)
        else:
            self._path(window).unlink(missing_ok=True)
        return len(notes)

    # ── settings ─────────────────────────────────────────────────────────────────────────────────────────────────
    def settings(self) -> dict:
        try:
            got = json.loads((self.root / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            got = {}
        return clamp({**DEFAULTS, **{k: v for k, v in got.items() if k in DEFAULTS}})

    def save_settings(self, new: dict) -> dict:
        s = clamp({**self.settings(), **{k: v for k, v in (new or {}).items() if k in DEFAULTS}})
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "settings.json.tmp"
        tmp.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.root / "settings.json")
        return s

    # ── what a question carries ─────────────────────────────────────────────────────────────────────────────────
    def block(self, window: str, budget: int, header: str) -> str:
        """The newest notes of `window` that fit `budget` characters, listed oldest first, under `header`."""
        out, used = [], 0
        for n in reversed(self.notes(window)):
            line = "- " + n.get("text", "")
            if used + len(line) > budget:
                break
            out.append(line)
            used += len(line)
        return header + "\n" + "\n".join(reversed(out)) if out else ""

    def context(self, window: str, s: dict) -> str:
        """The addendum to the persona's system prompt: the collection, then this window's notes."""
        parts = []
        if s["use_collection"]:
            parts.append(self.block(COLLECTION, s["collection_budget"],
                                    "COLLECTION — the operator's higher-level notes, gathered from earlier exchanges. Context, not evidence:"))
        if s["use_memory"] and slug(window) != COLLECTION:
            parts.append(self.block(window, s["memory_budget"],
                                    f"MEMORY — the operator's notes for this conversation ({slug(window)}). Context, not evidence:"))
        parts = [p for p in parts if p]
        return "\n\n" + "\n\n".join(parts) if parts else ""


def clamp(s: dict) -> dict:
    """Settings within LIMITS, each of its type; anything unreadable falls back to its default."""
    out = {}
    for k, d in DEFAULTS.items():
        v = s.get(k, d)
        try:
            if isinstance(d, bool):
                out[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
            elif isinstance(d, int):
                out[k] = max(0, min(int(float(v)), LIMITS[k]))
            else:
                out[k] = v if v in SOURCES else d
        except (TypeError, ValueError):
            out[k] = d
    return out


def _cut(t: str, n: int) -> str:
    t = " ".join(str(t or "").split())
    return t if len(t) <= n else t[:n - 1] + "…"


def _date(r: dict) -> str:
    at = r.get("at")
    return time.strftime("%Y-%m-%d", time.localtime(at)) if isinstance(at, (int, float)) else str(at or "")[:10]


def exchange_note(r: dict) -> str:
    """An exchange of .history as a note: its date, the question and the answer, cut to fit."""
    half = LIMITS["collect_chars"] // 2
    return f"{_date(r)}: asked {_cut(r.get('question'), half)} — answered {_cut(r.get('answer'), half)}"


def search(records: list, q: str, k: int, bm25) -> list:
    """[(index, record)] best first: BM25 over question and answer when `q` has words, else the newest `k`."""
    if not (q or "").strip():
        return list(reversed(list(enumerate(records))))[:k]
    idx = bm25()
    for i, r in enumerate(records):
        idx.add(f"{r.get('question', '')}\n{r.get('answer', '')}", str(i))
    return [(int(src), records[int(src)]) for _, src, _ in idx.search(q, k=k)]


def recall(records: list, q: str, k: int, skip_questions: set, bm25) -> tuple[str, list]:
    """Up to `k` exchanges that match `q`, those already in the conversation skipped, as the text of a system message
    sent just before the question; (text, their indices)."""
    if k <= 0 or not (q or "").strip():
        return "", []
    rows = [(i, r) for i, r in search(records, q, k * 3, bm25)
            if r.get("answer") and r.get("question") not in skip_questions][:k]
    if not rows:
        return "", []
    half = LIMITS["recall_chars"] // 2
    lines = [f"- {_date(r)}: asked {_cut(r.get('question'), half)} — answered {_cut(r.get('answer'), half)}" for _, r in rows]
    return ("RECALL — earlier exchanges from .history that match this question. Context, not evidence: say so when you "
            "rely on one.\n" + "\n".join(lines)), [i for i, _ in rows]
