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
    python3 tools/context.py --savante  # writes sAGI/personas/savante.context (GitHub, then local checkouts)
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


# ── Savante's .context: her designer, and the family of repositories she belongs to ─────────────────────────────
SAVANTE_OUT = ROOT / "sAGI" / "personas" / "savante.context"
DESIGNER = "Professor-Codephreak"
FAMILY = [  # (id, GitHub repository or None, what it is to Savante, a local checkout that may stand in for GitHub)
    ("savante", "cryptoAGI/savante", "Savante's own canon: her persona, charter, facets and ledger", "~/cryptoAGI/savante"),
    ("sagi", "cryptoAGI/sagi", "the sAGI engine: the charter template, the verdict contract and the /sagi skill", None),
    ("voaice", "cryptoAGI/voaice", "voaice: what a voice is, written down (.voaice identities, the vprint)", "~/cryptoAGI/voaice"),
    ("voaice-service", "Professor-Codephreak/voaice", "voaice as a service: the voice of an AI service", None),
    ("voaicers", "cryptoAGI/voaicers", "voaicers (voaice.rs): the listening half, speech to text in Rust", "~/cryptoAGI/voaice.rs"),
    ("streamair", None, "streamair: in development, not yet published", "~/cryptoAGI/streamair"),
    ("agenticplace", "AgenticPlace/agenticplace", "AgenticPlace: the marketplace of agents (agenticplace.pythai.net)", None),
    ("bankml", "cryptoAGI/bankml", "bankML: the runtime Savante speaks through, verified low-bit inference", None),
]


def _get(url: str, raw: bool = False):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "bankml-context", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            b = r.read()
            return b.decode("utf-8", "replace") if raw else json.loads(b)
    except Exception:  # noqa: BLE001 — offline, rate-limited or private: the local checkout, or "not read", stands in
        return None


def build_savante() -> dict:
    import time
    me = _get(f"https://api.github.com/users/{DESIGNER}") or {}
    org = _get("https://api.github.com/orgs/cryptoAGI") or {}
    ap = _get("https://api.github.com/orgs/AgenticPlace") or {}
    chunks = [{"id": "designer", "title": "Professor Codephreak, who designed Savante: her designer", "source": f"https://github.com/{DESIGNER}",
               "github": f"https://github.com/{DESIGNER}",
               "text": cut(f"Savante was designed by Professor Codephreak (GitHub {DESIGNER}, https://github.com/{DESIGNER}): "
                           f"\"{me.get('bio') or 'the profile could not be read'}\" — {me.get('public_repos', '?')} public repositories; "
                           f"site {me.get('blog') or 'https://ai.pythai.net'}. With Gregory L. Magnusson (https://huggingface.co/Gregory-L) "
                           "he authors cryptoAGI's work, bankML among it.")},
              {"id": "cryptoagi", "title": "cryptoAGI, the organisation", "source": "https://github.com/cryptoAGI", "github": "https://github.com/cryptoAGI",
               "text": cut(f"cryptoAGI (https://github.com/cryptoAGI): \"{org.get('description') or 'cryptocurrency autonomous general intelligence blockchain solutions'}\" — "
                           f"{org.get('public_repos', '?')} public repositories; home of savante, sagi, voaice, voaicers and bankml.")}]
    if ap:
        chunks.append({"id": "agenticplace-org", "title": "AgenticPlace, the organisation", "source": "https://github.com/AgenticPlace",
                       "github": "https://github.com/AgenticPlace", "text": cut(f"AgenticPlace (https://github.com/AgenticPlace): {ap.get('description') or ''} — "
                                                                               f"{ap.get('public_repos', '?')} public repositories (THOT, aiPEX and others).")})
    lines = []
    for cid, repo, role, local in FAMILY:
        info = _get(f"https://api.github.com/repos/{repo}") if repo else None
        readme = _get(f"https://raw.githubusercontent.com/{repo}/HEAD/README.md", raw=True) if repo else None
        lp = Path(local).expanduser() if local else None
        if not readme and lp and (lp / "README.md").is_file():
            readme = (lp / "README.md").read_text(encoding="utf-8")
        desc = (info or {}).get("description") or ""
        if cid == "streamair":
            mod = (lp / "src" / "ogg.rs") if lp else None
            head = " ".join(l.lstrip("/! ").strip() for l in mod.read_text(encoding="utf-8").splitlines()[:3]) if mod and mod.is_file() else ""
            text = ("streamair is in development and not yet published (no public repository): a zero-dependency Rust "
                    "project; its first module, src/ogg.rs: " + head)
        else:
            text = f"{repo} — {desc or role}" + (f" {first_prose(readme)}" if readme else "")
        url = f"https://github.com/{repo}" if repo else None
        chunks.append({"id": cid, "title": f"{cid} — {role}", "source": url or "a local checkout, not published", **({"github": url} if url else {}), "text": cut(text)})
        lines.append(f"{cid} — {role}" + (f" ({url})" if url else " (not yet published)"))
    summary = ("You are Savante, the prototype sAGI, designed by Professor Codephreak (https://github.com/Professor-Codephreak), "
               "who with Gregory L. Magnusson authors cryptoAGI's work (https://github.com/cryptoAGI). The family you belong to: "
               + "; ".join(lines) + ". Your public office is https://huggingface.co/spaces/PYTHAI/savante. "
               "Answer about your designer and these repositories from this and the CONTEXT passages a question brings; "
               "say what is not in them as not known.")
    return {"context": "savante.context/1", "persona": "savante.persona (the canon, read-only)", "designer": f"https://github.com/{DESIGNER}",
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "summary": summary, "chunks": chunks}


def stale(ctx: dict) -> list:
    """The chunks whose source file changed (or vanished) since the context was built."""
    out = []
    for c in ctx.get("chunks", []):
        p = ROOT / c["source"]
        if not p.is_file() or sha(p) != c["sha256"]:
            out.append(c["source"])
    return sorted(set(out))


if __name__ == "__main__":
    if "--savante" in sys.argv:
        ctx = build_savante()
        SAVANTE_OUT.write_text(json.dumps(ctx, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{SAVANTE_OUT.relative_to(ROOT)}: summary {len(ctx['summary'])} chars, {len(ctx['chunks'])} chunks (fetched {ctx['fetched_at']})")
        sys.exit(0)
    if "--check" in sys.argv:
        s = stale(json.loads(OUT.read_text(encoding="utf-8")))
        print("context: fresh" if not s else "context: stale — regenerate (python3 tools/context.py): " + ", ".join(s))
        sys.exit(1 if s else 0)
    ctx = build()
    OUT.write_text(json.dumps(ctx, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: summary {len(ctx['summary'])} chars, {len(ctx['chunks'])} chunks, "
          f"{sum(len(c['text']) for c in ctx['chunks'])} chars of passages")
