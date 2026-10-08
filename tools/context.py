#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Build sAGI/personas/bankml.context: what bankML knows of its own codebase, generated from the repository.

The persona says who bankML is; the prompt what it is asked; the context what it is made of. Two parts:
  · summary — always sent, after the persona's system prompt (so it sits in the engine's cached prefix): the version,
    the path of a request through the modules, where everything lives. Kept short: on a small CPU every token of a
    system prompt is read at prefill speed once.
  · chunks — retrieved per question (the best few for a question about bankML itself): each module page's Summary,
    the module map, and the opening of the main documents; each with its GitHub and Hugging Face links and the sha256
    of the file it came from, so a stale context is detected (testing/test_console.py) and regenerated.
Run it after changing docs/modules or the documents it reads, and at each release:
    python3 tools/context.py            # writes sAGI/personas/bankml.context
    python3 tools/context.py --check    # exit 1 if any chunk's source changed since
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sAGI" / "personas" / "bankml.context"
GITHUB = "https://github.com/cryptoAGI/bankml/blob/main/"
SPACE = "https://huggingface.co/spaces/PYTHAI/bankml/blob/main/"
CHUNK = 650
DOCS = [("docs/why-bankml.md", "why bankML"), ("docs/oracles.md", "how bankML is verified: the oracles"),
        ("docs/PERFORMANCE.md", "bankML's speed, measured"), ("docs/install.md", "installing bankML"),
        ("docs/usage.md", "using bankML"), ("docs/OLLAMA.md", "Ollama's API in bankML"), ("docs/CAPI.md", "the C API, libbankml"),
        ("docs/TECHNICAL.md", "the technical report and the thesis"), ("docs/TODO.md", "the road to 1.0.0")]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def plain(md: str) -> str:
    """Markdown to running text: links to their words, code marks and emphasis dropped, whitespace folded."""
    md = re.sub(r"```.*?```", " ", md, flags=re.S)
    md = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", md)
    md = re.sub(r"[`*_>#|]", "", md)
    return " ".join(md.split())


def cut(text: str, n: int = CHUNK) -> str:
    if len(text) <= n:
        return text
    t = text[:n]
    end = max(t.rfind(". "), t.rfind("; "))
    return (t[:end + 1] if end > n * 0.6 else t.rstrip() + "…")


def section(md: str, name: str) -> str:
    m = re.search(rf"^## {re.escape(name)}\s*$(.*?)(?=^## |\Z)", md, flags=re.M | re.S)
    return m.group(1) if m else ""


def first_prose(md: str) -> str:
    """The first paragraphs after the title, skipping badges, tables and headings."""
    body = re.sub(r"^#.*$", "", md, count=1, flags=re.M)
    paras = [p for p in re.split(r"\n\s*\n", body) if p.strip() and not p.lstrip().startswith(("|", "#", "<", "[!", "```", "- [", "!["))]
    return " ".join(plain(p) for p in paras[:3])


def chunk(cid: str, title: str, src: str, text: str) -> dict:
    p = ROOT / src
    return {"id": cid, "title": title, "source": src, "github": GITHUB + src, "huggingface": SPACE + src,
            "sha256": sha(p), "text": cut(text)}


def build() -> dict:
    version = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(), flags=re.M).group(1)
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        commit = None
    idx = (ROOT / "docs/modules/README.md").read_text(encoding="utf-8")
    diagram = re.search(r"## The path of a request\s*```(.*?)```", idx, flags=re.S)
    rows = re.findall(r"^\| \[(\w+)\.md\]\(\w+\.md\) \| `([^`]+)` \| (.+?) \|\s*$", idx, flags=re.M)
    summary = (
        f"bankML {version}: a Rust runtime with no dependencies for 1-bit (Q1_0) and ternary (Q2_0) GGUF language models on the CPU, "
        "bit-exact against llama.cpp b11192 and checked by oracles in every release; `bankml serve` verifies the model "
        "(guard, then its sha256 pin) before it answers and puts a receipt on every answer. Its source: bankML/ (the "
        "runtime), capi/ (libbankml), sAGI/ (Savante and this console, Python), testing/ (oracles, the release gate), "
        "docs/modules/ (one page per module). The path of a request:"
        + ("\n" + "\n".join(l.rstrip() for l in diagram.group(1).strip("\n").splitlines()) if diagram else "")
        + f"\nSource: {GITHUB.rsplit('/blob/', 1)[0]} · mirrored at {SPACE.rsplit('/blob/', 1)[0]}. "
        "Answer questions about yourself and your architecture from this and from the CONTEXT passages a question brings; "
        "name the file; say what is not in them as not known."
    )
    chunks = [chunk("modules", "bankML's architecture: its modules, one line each", "docs/modules/README.md",
                    " · ".join(f"{f} — {plain(r)}" for _, f, r in rows))]
    for page in sorted((ROOT / "docs/modules").glob("*.md")):
        if page.name == "README.md":
            continue
        md = page.read_text(encoding="utf-8")
        head = re.search(r"^# (.+)$", md, flags=re.M)
        chunks.append(chunk(page.stem, plain(head.group(1)) if head else page.stem, f"docs/modules/{page.name}",
                            plain(section(md, "Summary")) or first_prose(md)))
    for src, title in DOCS:
        p = ROOT / src
        if p.is_file():
            chunks.append(chunk("doc-" + Path(src).stem.lower(), title, src, first_prose(p.read_text(encoding="utf-8"))))
    return {"context": "bankml.context/1", "persona": "bankml.persona", "version": version, "commit": commit,
            "repo": GITHUB.rsplit("/blob/", 1)[0], "space": SPACE.rsplit("/blob/", 1)[0],
            "summary": summary, "chunks": [c for c in chunks if c["text"]]}


def stale(ctx: dict) -> list:
    """The chunks whose source file changed (or vanished) since the context was built."""
    out = []
    for c in ctx.get("chunks", []):
        p = ROOT / c["source"]
        if not p.is_file() or sha(p) != c["sha256"]:
            out.append(c["source"])
    return sorted(set(out))


if __name__ == "__main__":
    if "--check" in sys.argv:
        s = stale(json.loads(OUT.read_text(encoding="utf-8")))
        print("context: fresh" if not s else "context: stale — regenerate (python3 tools/context.py): " + ", ".join(s))
        sys.exit(1 if s else 0)
    ctx = build()
    OUT.write_text(json.dumps(ctx, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: summary {len(ctx['summary'])} chars, {len(ctx['chunks'])} chunks, "
          f"{sum(len(c['text']) for c in ctx['chunks'])} chars of passages")
