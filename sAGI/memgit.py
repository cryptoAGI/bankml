# SPDX-License-Identifier: MIT OR Apache-2.0
"""Versions of .memory: every add and remove is a git commit in a private repository, so a removed note can be
brought back and a rewrite can be undone. gitmind (github.com/Professor-Codephreak/gitmind, MIT, vendored in
sAGI/gitmind/) watches each repository and records every change in its ledger (<repo>/.gitmind/ledger.jsonl),
marking a move backwards as a rollback.

What it is for: .memory is the one file the UI rewrites (a removal writes the file again without the note); .history
is append-only and already committed by its Merkle root, so it is not versioned here.

What it is not: required, or a backup off this machine. Without git, or with BANKML_MEMORY_GIT=0, nothing is
versioned and nothing fails. gitmind runs with no backup sources: .memory never leaves the laptop, and gitmind's
IPFS source would publish every version of it. Restoring a version is a new commit (forward-only); history is never
rewritten.

Where the repositories are: <state>/memgit/<name>-<8 hex of the file's path>/, one per .memory file, each holding a
copy of that file. Never in the file's own directory: an agent's directory is already its own git repository (the
THOT rung's locator, thot._commit_locator), where .memory is git-ignored on purpose and must stay out."""
import hashlib, os, shutil, subprocess
from pathlib import Path

STATE = Path(os.environ.get("BANKML_UI_STATE", Path.home() / ".local" / "share" / "bankml" / "savante")).expanduser()
# the operator's global git settings must not change these commits: no signing prompt, no hooks, a fixed identity
GIT = ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-c", "user.name=bankml .memory",
       "-c", "user.email=memory@bankml.local"]


def enabled() -> bool:
    return os.environ.get("BANKML_MEMORY_GIT", "1") != "0" and shutil.which("git") is not None


def _git(d: Path, *args: str, timeout: int = 15):
    try:
        r = subprocess.run([*GIT, *args], cwd=str(d), capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _gitmind():
    try:
        import gitmind  # vendored beside this file (sAGI/gitmind), or an installed one
        return gitmind
    except Exception:
        return None


def repo_dir(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    return STATE / "memgit" / f"{path.name}-{hashlib.sha256(str(path).encode()).hexdigest()[:8]}"


def _repo(path: Path) -> Path | None:
    """The repository for `path`, made on first use."""
    d = repo_dir(path)
    if not (d / ".git").exists():
        d.mkdir(parents=True, exist_ok=True)
        if _git(d, "init", "-q") is None:
            return None
        (d / ".gitignore").write_text(".gitmind/\n", encoding="utf-8")
        (d / "SOURCE").write_text(str(Path(path).expanduser().resolve()) + "\n", encoding="utf-8")
    return d


def _snapshot(d: Path) -> dict | None:
    gm = _gitmind()
    if gm is None:
        return None
    try:
        return gm.GitMind(repo_root=d, sources=[]).snapshot()  # no sources: nothing leaves the machine
    except Exception:
        return None


def commit(path: Path, message: str) -> str | None:
    """Commit the current contents of `path` with `message`; the new commit's sha, or None when off, failed, or
    nothing changed."""
    if not enabled():
        return None
    path = Path(path)
    d = _repo(path)
    if d is None:
        return None
    try:
        data = path.read_bytes() if path.exists() else b""
        tmp = d / (path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, d / path.name)
    except OSError:
        return None
    if _git(d, "add", "--", ".gitignore", "SOURCE", path.name) is None:
        return None
    if _git(d, "diff", "--cached", "--quiet") is not None:  # exit 0: nothing staged
        return None
    if _git(d, "commit", "-q", "-m", message) is None:
        return None
    _snapshot(d)
    return _git(d, "rev-parse", "HEAD")


def versions(path: Path, n: int = 20) -> list:
    """The newest `n` versions of `path`: [{"sha", "short", "at", "subject"}], newest first; [] when there are none."""
    path = Path(path)
    d = repo_dir(path)
    if not enabled() or not (d / ".git").exists():
        return []
    out = _git(d, "log", f"-{int(n)}", "--pretty=%H\x1f%h\x1f%cI\x1f%s", "--", path.name) or ""
    return [dict(zip(("sha", "short", "at", "subject"), line.split("\x1f"))) for line in out.splitlines() if line]


def restore(path: Path, rev: str) -> str | None:
    """Put `path` back as it was at `rev` (a sha or a prefix of one), as a new commit. The new sha, or None when `rev`
    is not a version of `path` or that version is the current one."""
    path = Path(path)
    d = repo_dir(path)
    rev = (rev or "").strip()
    if not enabled() or not rev or not (d / ".git").exists() or not all(c in "0123456789abcdef" for c in rev.lower()):
        return None
    full = _git(d, "rev-parse", "--verify", "-q", rev + "^{commit}")
    if full is None or full not in {v["sha"] for v in versions(path, n=10_000)}:
        return None
    try:
        data = subprocess.run([*GIT, "show", f"{full}:{path.name}"], cwd=str(d), capture_output=True, timeout=15)
    except Exception:
        return None
    if data.returncode != 0 or (path.exists() and path.read_bytes() == data.stdout):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data.stdout)
    os.replace(tmp, path)  # never half a .memory
    return commit(path, f"restore {path.name} to {full[:7]}")


def ingest_bundle(path: Path, bundle_path: Path, name: str = "browseml.memory") -> dict:
    """Bring a browser's .memory into this one: a git bundle from browseML's GITMIND (EXPORT BUNDLE) is cloned to a
    temporary directory, its newest `name` read, and every note this .memory does not hold yet (same sha256 of its text)
    appended — delivery records (how an answer was delivered, never sent to a model) left out — then committed as one
    version. Returns {head, notes, added, skipped}. The bundle is checked by git (`git bundle verify`) before it is read."""
    import json as _json, tempfile
    path, bundle_path = Path(path), Path(bundle_path)
    with tempfile.TemporaryDirectory(prefix="bankml-ingest-") as t:
        d = Path(t)
        if _git(d, "init", "-q") is None or _git(d, "bundle", "verify", str(bundle_path)) is None:
            raise ValueError("not a git bundle git accepts")
        if _git(d, "clone", "-q", str(bundle_path), "c") is None:
            raise ValueError("the bundle could not be cloned")
        c = d / "c"
        head = _git(c, "rev-parse", "HEAD") or ""
        src = c / name
        if not src.is_file():
            raise ValueError(f"the bundle has no {name}")
        lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    have = set()
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            try:
                have.add(hashlib.sha256(str(_json.loads(l).get("text", "")).encode()).hexdigest())
            except ValueError:
                continue
    added, skipped = [], 0
    for l in lines:
        try:
            n = _json.loads(l)
        except ValueError:
            skipped += 1
            continue
        text = str(n.get("text", "")).strip()
        h = hashlib.sha256(text.encode()).hexdigest()
        if not text or (n.get("source") or {}).get("kind") == "delivery" or h in have:
            skipped += 1
            continue
        have.add(h)
        added.append(_json.dumps({"ts": n.get("ts"), "at": n.get("at"), "text": text, "sha256": h,
                                  "source": {"kind": (n.get("source") or {}).get("kind", "typed"), "from": f"browseML bundle {head[:7]}"}},
                                 ensure_ascii=False))
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write("".join(a + "\n" for a in added))
        commit(path, f"ingest {len(added)} note{'s' if len(added) != 1 else ''} from a browseML bundle ({head[:7]})")
    return {"head": head, "notes": len(lines), "added": len(added), "skipped": skipped}
