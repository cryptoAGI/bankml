# SPDX-License-Identifier: MIT OR Apache-2.0
# gitmind/core.py — git state monitor + multi-source backup/rollback (standalone)
"""GitMind: watch a git repo, record rollbacks (and self-rollbacks), snapshot the
repo as git bundles, and replicate those bundles across variable backup sources.

Self-contained: standard library + `git`. No LLM dependency — gitmind is plain
infrastructure, so it works identically whatever model you run (GLM-4.5/5.2,
Llama, Qwen, Claude, …). Backup sources are pluggable behind the `Source` ABC;
LocalSource ships built-in, IPFS/Arweave are optional adapters you wire up.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _git_toplevel(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(start),
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except Exception:
        pass
    return start


class Source:
    """A backup destination. Implement `put(data, label) -> dict`."""
    name = "base"
    async def put(self, data: bytes, label: str) -> Dict[str, Any]:
        raise NotImplementedError


class LocalSource(Source):
    """Copy the bundle into a local backup directory (newest `keep` retained)."""
    name = "local"
    def __init__(self, root: Path, keep: int = 3):
        self.dir = root / ".gitmind" / "backups"
        self.keep = keep
    async def put(self, data: bytes, label: str) -> Dict[str, Any]:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            p = self.dir / label
            await asyncio.to_thread(p.write_bytes, data)
            try:
                for old in sorted(self.dir.glob("*.bundle"),
                                  key=lambda f: f.stat().st_mtime, reverse=True)[self.keep:]:
                    old.unlink(missing_ok=True)
            except Exception:
                pass
            return {"name": self.name, "ok": True, "ref": str(p), "bytes": len(data)}
        except Exception as e:
            return {"name": self.name, "ok": False, "error": str(e)[:160]}


class IPFSSource(Source):
    """Add the bundle to an IPFS node via its HTTP API (e.g. local Kubo daemon at
    http://127.0.0.1:5001, or a pinning gateway). Set GITMIND_IPFS_API."""
    name = "ipfs"
    def __init__(self, api: Optional[str] = None):
        self.api = api or os.getenv("GITMIND_IPFS_API")
    async def put(self, data: bytes, label: str) -> Dict[str, Any]:
        if not self.api:
            return {"name": self.name, "ok": False, "error": "not_configured (set GITMIND_IPFS_API)"}
        try:
            import urllib.request
            boundary = "----gitmind"
            body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                    f"filename=\"{label}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(self.api.rstrip("/") + "/api/v0/add",
                data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            def _do():
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read().decode().strip().splitlines()[-1])
            res = await asyncio.to_thread(_do)
            cid = res.get("Hash")
            return {"name": self.name, "ok": bool(cid), "ref": cid,
                    "gateway": f"https://ipfs.io/ipfs/{cid}" if cid else None}
        except Exception as e:
            return {"name": self.name, "ok": False, "error": str(e)[:160]}


class GitMind:
    """Monitor git state, record rollbacks, replicate bundles to variable sources."""

    def __init__(self, repo_root: Optional[Path] = None, sources: Optional[List[Source]] = None):
        self.root = Path(repo_root) if repo_root else _git_toplevel()
        self.dir = self.root / ".gitmind"
        self.ledger = self.dir / "ledger.jsonl"
        self.state_file = self.dir / "last_state.json"
        self.self_flag = self.dir / "self_rollback.flag"
        self.sources = sources if sources is not None else [LocalSource(self.root), IPFSSource()]

    # ── git plumbing ──
    def _git(self, *args: str, timeout: int = 20) -> Optional[str]:
        try:
            r = subprocess.run(["git", *args], cwd=str(self.root),
                               capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    def is_repo(self) -> bool:
        return self._git("rev-parse", "--is-inside-work-tree") == "true"

    def state(self) -> Dict[str, Any]:
        last = (self._git("log", "-1", "--pretty=%h\x1f%s\x1f%cI\x1f%ae") or "").split("\x1f")
        return {
            "ts": time.time(),
            "head": self._git("rev-parse", "HEAD"),
            "short": self._git("rev-parse", "--short", "HEAD"),
            "branch": self._git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(self._git("status", "--porcelain")),
            "commit_count": int(self._git("rev-list", "--count", "HEAD") or 0),
            "last_commit": {"short": last[0] if last and last[0] else None,
                            "subject": last[1] if len(last) > 1 else None,
                            "committed": last[2] if len(last) > 2 else None,
                            "author": last[3] if len(last) > 3 else None},
            "reflog_op": (self._git("reflog", "-1", "--pretty=%gs") or "").split(":")[0] or None,
        }

    def _is_ancestor(self, a: str, b: str) -> bool:
        try:
            return subprocess.run(["git", "merge-base", "--is-ancestor", a, b],
                                  cwd=str(self.root), capture_output=True, timeout=15).returncode == 0
        except Exception:
            return False

    # ── ledger ──
    def _append(self, event: Dict[str, Any]) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), **event}) + "\n")
        except Exception:
            pass

    def _read(self, limit: int = 50, kinds: Optional[set] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            if self.ledger.exists():
                for line in self.ledger.read_text(encoding="utf-8").splitlines():
                    try:
                        e = json.loads(line)
                        if kinds is None or e.get("kind") in kinds:
                            out.append(e)
                    except Exception:
                        continue
        except Exception:
            pass
        return out[-limit:][::-1]

    # ── monitor + rollback detection ──
    def snapshot(self) -> Dict[str, Any]:
        st = self.state()
        prev = {}
        try:
            if self.state_file.exists():
                prev = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        p, c = prev.get("head"), st.get("head")
        transition = "none"
        if p and c and p != c:
            transition = "rollback" if self._is_ancestor(c, p) else ("advance" if self._is_ancestor(p, c) else "reset")
        st["transition"], st["prev_head"] = transition, p
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(st), encoding="utf-8")
        except Exception:
            pass
        if transition in ("rollback", "reset"):
            selfi = self.self_flag.exists()
            try:
                if selfi:
                    self.self_flag.unlink()
            except Exception:
                pass
            self._append({"kind": "self_rollback" if selfi else "rollback", "transition": transition,
                          "from": p, "to": c, "reflog_op": st.get("reflog_op"),
                          "branch": st.get("branch"), "self_initiated": selfi})
        else:
            self._append({"kind": "snapshot", "transition": transition, "head": c, "branch": st.get("branch")})
        return st

    def mark_self_rollback(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.self_flag.write_text(str(time.time()), encoding="utf-8")
        except Exception:
            pass

    # ── backup: bundle + replicate ──
    def _bundle(self) -> Optional[bytes]:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / f".b_{int(time.time())}.tmp"
            r = subprocess.run(["git", "bundle", "create", str(tmp), "--all"],
                               cwd=str(self.root), capture_output=True, timeout=300)
            if r.returncode != 0 or not tmp.exists():
                return None
            data = tmp.read_bytes()
            tmp.unlink(missing_ok=True)
            return data
        except Exception:
            return None

    async def backup(self) -> Dict[str, Any]:
        if not self.is_repo():
            return {"ok": False, "error": "not a git repo"}
        st = self.state()
        data = await asyncio.to_thread(self._bundle)
        if not data:
            return {"ok": False, "error": "git bundle failed"}
        label = f"{st.get('short') or 'HEAD'}-{int(time.time())}.bundle"
        results = await asyncio.gather(*[s.put(data, label) for s in self.sources])
        entry = {"kind": "backup", "head": st.get("head"), "short": st.get("short"),
                 "branch": st.get("branch"), "bytes": len(data), "label": label,
                 "sources": results, "replicas": len([r for r in results if r.get("ok")])}
        self._append(entry)
        return {"ok": entry["replicas"] > 0, **entry}

    # ── rollback / restore ──
    def rollback_to(self, sha: str, *, hard: bool = False) -> Dict[str, Any]:
        if not self.is_repo():
            return {"ok": False, "error": "not a git repo"}
        if not self._git("cat-file", "-t", sha):
            return {"ok": False, "error": f"unknown commit {sha}"}
        self.mark_self_rollback()
        before = self._git("rev-parse", "HEAD")
        if hard and os.getenv("GITMIND_ALLOW_HARD") == "1":
            self._git("reset", "--hard", sha); op = "reset --hard"
        else:
            self._git("revert", "--no-edit", f"{sha}..HEAD"); op = "revert --no-edit"
        after = self._git("rev-parse", "HEAD")
        ok = bool(after) and after != before
        self._append({"kind": "self_rollback", "transition": "rollback", "op": op,
                      "from": before, "to": after, "target": sha, "self_initiated": True, "ok": ok})
        return {"ok": ok, "op": op, "from": before, "to": after, "target": sha}

    def report(self) -> Dict[str, Any]:
        backups = self._read(10, {"backup"})
        rollbacks = self._read(20, {"rollback", "self_rollback"})
        return {"ts": time.time(), "state": self.state(),
                "sources": [s.name for s in self.sources],
                "last_backup": backups[0] if backups else None,
                "backups_recent": backups, "rollbacks_recent": rollbacks,
                "self_rollback_count": sum(1 for r in rollbacks if r.get("kind") == "self_rollback"),
                "rollback_count": len(rollbacks)}
