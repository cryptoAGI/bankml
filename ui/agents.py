#!/usr/bin/env python3
"""Custom agents from the Savante template — each with its own .persona, .prompt, card and ledger.

Savante's canon (~/savante) is the template and is only ever read. `derive()` writes a new agent into
BANKML_AGENTS (default ~/.local/share/bankml/agents/<slug>/):

  <slug>.persona            the persona (mindX .persona v1 shape), a fresh identity; the token bindings start empty
  <slug>.prompt             the system prompt, as plain text (the .prompt file the chat uses)
  <slug>.agentcard.json     EIP-721 metadata ∪ ERC-8004 registration-v1, status not_yet_minted, derived_from Savante
  <slug>.commitments.json   the ledger: sha256 + CIDv1 of each file, and the doctrine root (keccak256 over the same
                            15 RFC 6901 pointers as Savante's ledger, same canonical bytes)
  <slug>.history            its conversations (JSONL), and <slug>.memory, its notes

The ledger is recomputed on every save, so a custom agent is always internally consistent; comparing its doctrine
root with one written on chain is what would catch an edit after publication (as for Savante).

stdlib only: keccak256 is implemented here and checked against Savante's published doctrine root.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path

AGENTS = Path(os.environ.get("BANKML_AGENTS", Path.home() / ".local" / "share" / "bankml" / "agents")).expanduser()

# ── keccak256 (the Ethereum variant: original Keccak padding 0x01, not SHA3's 0x06) ───────────────────────────
_RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
       0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
       0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
       0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
       0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
_ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
_M = (1 << 64) - 1


def _rol(v, n):
    return ((v << n) | (v >> (64 - n))) & _M if n else v


def _f(a):
    for rc in _RC:
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rol(c[(x + 1) % 5], 1) for x in range(5)]
        a = [[a[x][y] ^ d[x] for y in range(5)] for x in range(5)]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rol(a[x][y], _ROT[x][y])
        a = [[b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
        a[0][0] ^= rc
    return a


def keccak256(data: bytes) -> bytes:
    rate = 136
    msg = bytearray(data) + b"\x01"
    msg += b"\x00" * ((-len(msg)) % rate)
    msg[-1] |= 0x80
    a = [[0] * 5 for _ in range(5)]
    for off in range(0, len(msg), rate):
        blk = msg[off:off + rate]
        for i in range(rate // 8):
            a[i % 5][i // 5] ^= int.from_bytes(blk[8 * i:8 * i + 8], "little")
        a = _f(a)
    return b"".join(a[i % 5][i // 5].to_bytes(8, "little") for i in range(4))


# ── the ledger's constructions (as Savante's bind/savante_bind.py defines them) ───────────────────────────────
DOCTRINE_POINTERS = ["/persona", "/name", "/source", "/format", "/system_prompt", "/mantra", "/oath", "/bdi/beliefs",
                     "/skills/primary", "/skills/taxonomy", "/skills/defer_triggers", "/skills/validation", "/safety",
                     "/embodiment", "/token/intelligence/tool_allowlist"]


def canonical_bytes(v) -> bytes:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def pointer_get(doc, ptr: str):
    cur = doc
    for part in ptr.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def preflight(v, path="") -> list:
    """The binder's preflight: P1 every key ASCII; P2 every number an integer within ±2^53 (no floats), so the
    canonical bytes are the same in every JSON implementation. Returns the violations."""
    bad = []
    if isinstance(v, dict):
        for k, x in v.items():
            if not k.isascii():
                bad.append(f"P1: key {path}/{k} is not ASCII")
            bad += preflight(x, f"{path}/{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            bad += preflight(x, f"{path}/{i}")
    elif isinstance(v, float):
        bad.append(f"P2: {path} is a float ({v}); use an integer or a string")
    elif isinstance(v, int) and not isinstance(v, bool) and abs(v) > 2 ** 53:
        bad.append(f"P2: {path} exceeds ±2^53")
    return bad


def doctrine_root(persona: dict, pointers=DOCTRINE_POINTERS) -> str:
    """keccak256 over the clauses an owner may not edit. A missing pointer is a hard error, never a skipped term."""
    pre = b""
    for p in pointers:
        try:
            v = pointer_get(persona, p)
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"doctrine clause {p} is missing from the persona") from e
        pre += p.encode() + b"\x1f" + canonical_bytes(v) + b"\x1e"
    return "0x" + keccak256(pre).hex()


def cid_v1_raw(b: bytes) -> str:
    import base64
    return "b" + base64.b32encode(bytes([0x01, 0x55, 0x12, 0x20]) + hashlib.sha256(b).digest()).decode().lower().rstrip("=")


def file_commitment(b: bytes) -> dict:
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "cid": cid_v1_raw(b)}


# ── agents ─────────────────────────────────────────────────────────────────────────────────────────────────────
def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:40] or "agent"


def agent_dir(slug: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_]{1,40}", slug or ""):
        raise ValueError(f"bad agent name {slug!r}")
    return AGENTS / slug


def list_agents() -> list:
    return sorted(p.name for p in AGENTS.iterdir() if (p / f"{p.name}.persona").is_file()) if AGENTS.is_dir() else []


def files(slug: str) -> dict:
    d = agent_dir(slug)
    return {k: d / f"{slug}.{k}" for k in ("persona", "prompt", "agentcard.json", "commitments.json", "history", "memory")}


def derive(template_persona: dict, template_ledger: dict, name: str, *, system_prompt: str | None = None,
           mantra: str | None = None, oath: str | None = None, kind: str | None = None, description: str | None = None) -> str:
    """A new agent from the template. Never touches the template's files."""
    slug = slugify(name)
    d = agent_dir(slug)
    if d.exists():
        raise FileExistsError(f"agent {slug!r} exists")
    p = copy.deepcopy(template_persona)
    t_sha = ((template_ledger.get("artifacts") or {}).get("identity") or {}).get("sha256")
    t_root = (template_ledger.get("doctrine_root") or {}).get("value")
    p.update(persona=slug, name=name, kind=kind or p.get("kind", "agent"), source="bankml custom agent",
             template=f"derived from {template_persona.get('name', 'template')} (persona sha256 {t_sha}, doctrine root {t_root}) by bankml",
             system_prompt=system_prompt or p.get("system_prompt", ""), mantra=mantra if mantra is not None else p.get("mantra", ""),
             oath=oath if oath is not None else p.get("oath", ""), voice_examples=[], exchanges=[])
    tok = p.get("token") if isinstance(p.get("token"), dict) else {}
    p["token"] = {"schema": "bankml.agent.token v1", "intelligence": {"tool_allowlist": ((tok.get("intelligence") or {}).get("tool_allowlist") or [])},
                  "bindings": {"erc8004": None, "erc7857": None, "erc6551": None}, "public_metadata": {"description": description or ""}}
    d.mkdir(parents=True)
    f = files(slug)
    f["persona"].write_text(json.dumps(p, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    f["prompt"].write_text(p["system_prompt"] + "\n", encoding="utf-8")
    rebind(slug, derived_from={"agent": template_persona.get("name"), "persona_sha256": t_sha, "doctrine_root": t_root})
    return slug


def save(slug: str, persona_text: str | None = None, prompt_text: str | None = None) -> dict:
    """Write edited .persona / .prompt (validated), then re-ledger."""
    f = files(slug)
    if persona_text is not None:
        p = json.loads(persona_text)
        if not isinstance(p, dict) or not p.get("name"):
            raise ValueError("the persona must be a JSON object with a name")
        bad = preflight(p)
        if bad:
            raise ValueError("; ".join(bad[:5]))
        new_root = doctrine_root(p)  # every doctrine clause present
        fixed = fixed_root(slug)
        if fixed and new_root != fixed:
            changed = [ptr for ptr in DOCTRINE_POINTERS
                       if canonical_bytes(pointer_get(p, ptr)) != canonical_bytes(pointer_get(json.loads(f["persona"].read_text(encoding="utf-8")), ptr))]
            raise ValueError("the doctrine clauses are fixed when an agent is derived (" + ", ".join(changed[:6]) +
                             " changed); derive a new agent to change them. The .prompt, voice examples and other fields stay editable.")
        f["persona"].write_text(json.dumps(p, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if prompt_text is not None:
        f["prompt"].write_text(prompt_text.rstrip("\n") + "\n", encoding="utf-8")
    return rebind(slug)


AIVATAR_MAX = 2 * 1024 * 1024
_MAGIC = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpeg"), (b"RIFF", "webp"))


def aivatar_path(slug: str):
    for ext in ("png", "jpeg", "webp"):
        q = agent_dir(slug) / f"{slug}.aivatar.{ext}"
        if q.is_file():
            return q
    return None


def set_aivatar(slug: str, b: bytes) -> dict:
    """The agent's chosen portrait: PNG, JPEG or WebP by content (not by name), at most 2 MB; then re-ledger."""
    if len(b) > AIVATAR_MAX:
        raise ValueError(f"the image is {len(b)} bytes; at most {AIVATAR_MAX}")
    kind = next((k for m, k in _MAGIC if b.startswith(m) and (k != "webp" or b[8:12] == b"WEBP")), None)
    if not kind:
        raise ValueError("not a PNG, JPEG or WebP image")
    for ext in ("png", "jpeg", "webp"):
        (agent_dir(slug) / f"{slug}.aivatar.{ext}").unlink(missing_ok=True)
    (agent_dir(slug) / f"{slug}.aivatar.{kind}").write_bytes(b)
    return rebind(slug)


def fixed_root(slug: str) -> str | None:
    """The doctrine root recorded when the agent was derived (carried forward by every rebind, never recomputed)."""
    try:
        led = json.loads(files(slug)["commitments.json"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    d = led.get("doctrine_root") or {}
    return d.get("fixed_at_derivation") or d.get("value")


def rebind(slug: str, derived_from: dict | None = None) -> dict:
    """Regenerate the agent card and the ledger from the files as they are now."""
    f = files(slug)
    p = json.loads(f["persona"].read_text(encoding="utf-8"))
    old = json.loads(f["commitments.json"].read_text(encoding="utf-8")) if f["commitments.json"].is_file() else {}
    derived_from = derived_from or old.get("derived_from")
    fixed = (old.get("doctrine_root") or {}).get("fixed_at_derivation") or (old.get("doctrine_root") or {}).get("value") or doctrine_root(p)
    card = {"type": "https://eips.ethereum.org/EIPS/eip-8004#registration-v1", "name": p.get("name"),
            "description": ((p.get("token") or {}).get("public_metadata") or {}).get("description") or p.get("mantra", ""),
            "image": None, "external_url": None,
            "attributes": [{"trait_type": "kind", "value": p.get("kind")}, {"trait_type": "derived_from", "value": (derived_from or {}).get("agent")},
                           {"trait_type": "runtime", "value": "bankml"}],
            "services": [], "x402Support": False, "active": True, "registrations": [], "supportedTrust": [],
            "bankml": {"status": "not_yet_minted", "derived_from": derived_from,
                       "note": "derived output, regenerated by bankml on every save; never hand-edited"}}
    f["agentcard.json"].write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    arts = {k: {"path": f[k].name, **file_commitment(f[k].read_bytes())} for k in ("persona", "prompt")}
    av = aivatar_path(slug)
    if av:
        arts["aivatar"] = {"path": av.name, **file_commitment(av.read_bytes())}
    ledger = {"schema": "bankml.agent.commitments v1", "agent": slug, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "derived_from": derived_from, "artifacts": arts,
              "card": {"path": f["agentcard.json"].name, **file_commitment(f["agentcard.json"].read_bytes())},
              "doctrine_root": {"value": doctrine_root(p), "fixed_at_derivation": fixed, "hash": "keccak256 (bankml, pure Python)", "pointers": DOCTRINE_POINTERS,
                                "canonicalization": "json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=False), utf-8; "
                                                    "preimage = Σ pointer ‖ 0x1f ‖ canonical(value) ‖ 0x1e"},
              "onchain_slots": {"personaDigest": arts["persona"]["sha256"], "doctrineRoot": doctrine_root(p), "written": False},
              "mint": None, "mint_reason": "not minted: the owner signs a mint; bankml prepares and verifies it"}
    f["commitments.json"].write_text(json.dumps(ledger, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return ledger


def verify(slug: str) -> list:
    """[(entry, ok, detail)]: every ledgered file re-hashed, and the doctrine root recomputed."""
    f = files(slug)
    led = json.loads(f["commitments.json"].read_text(encoding="utf-8"))
    rows = []
    for k, a in list(led["artifacts"].items()) + [("card", led["card"])]:
        b = (agent_dir(slug) / a["path"]).read_bytes()
        rows.append((k, hashlib.sha256(b).hexdigest() == a["sha256"], a["path"]))
    p = json.loads(f["persona"].read_text(encoding="utf-8"))
    dr = led.get("doctrine_root") or {}
    now = doctrine_root(p)  # always over bankml's own pointer list, never the list inside the ledger being checked
    rows.append(("doctrine root", now == dr.get("value") and list(dr.get("pointers") or []) == DOCTRINE_POINTERS, dr.get("value")))
    rows.append(("doctrine unchanged since derivation", now == (dr.get("fixed_at_derivation") or dr.get("value")), dr.get("fixed_at_derivation") or dr.get("value")))
    return rows
