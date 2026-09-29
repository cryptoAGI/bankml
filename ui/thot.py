#!/usr/bin/env python3
"""THOT manifests (sagi.thot_manifest/1) for bankml agents — the dataset bundle an iNFT points to.

Implements ~/sagi/engine/THOT_MANIFEST.md §2, §3, §3a, §5 and §7 exactly, and is checked against that spec's own
test vectors (savante @1fcca89, jaimla @8b57ccf, luvai @0c1eef7): the same bundle_root and Merkle root recomputed
from each manifest's facets, and the same identity CID the spec's lineage table records.

A bankml agent's bundle holds:
  persona  <slug>.persona        (core, the ternary head, leaf 0)
  prompt   <slug>.prompt         (core)
  x-bankml.agentcard  <slug>.agentcard.json   the card (EIP-721 ∪ ERC-8004)
  x-bankml.history    <slug>.history          its conversations — committed by hash; the bytes stay on this computer
  x-bankml.memory     <slug>.memory           its notes — likewise
Proof of data without the data: the manifest is publishable (it holds digests, never content), and any one
exchange can later be shown with its line and the history's commitment.

Each agent directory is a git repository (history and memory are git-ignored), so the rung's locator
`localhost/<user>/<slug>@<commit>` names real bytes, and a later push can move it to a public host.

stdlib only (keccak256 from agents.py).
"""
from __future__ import annotations

import getpass
import hashlib
import json
import subprocess
from pathlib import Path

from agents import agent_dir, canonical_bytes, cid_v1_raw, doctrine_root, files, keccak256

SCHEMA = "sagi.thot_manifest/1"
CORE = ["persona", "agent", "model", "prompt", "tool", "skill", "voaice", "faice", "attribute", "verse.xml", "aiml", "reputation", "agency"]
PAD = keccak256(b"")
ALGORITHMS = {
    "facet_digest": "sha256",
    "cid": "cidv1-raw-sha2-256-base32",
    "bundle_root": "keccak256",
    "merkle_leaf": "keccak256",
    "merkle_pad": "keccak256",
    "doctrine_root": "keccak256",
    "identity_thot": "sha256",
    "identity_content_root": "keccak256",
    "canonicalisation": "json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False).encode('utf-8')",
    "note": "bankml agent bundle; algorithms per sagi.thot_manifest/1 §3a",
}
CUSTOM = {
    "x-bankml.agentcard": {"suffix": "agentcard.json", "media": "application/json",
                           "role": "the agent card: EIP-721 metadata ∪ ERC-8004 registration-v1, derived by bankml on every save"},
    "x-bankml.history": {"suffix": "history", "media": "application/x-ndjson",
                         "role": "the agent's conversations, one JSON object per exchange; committed by digest, the bytes stay local"},
    "x-bankml.memory": {"suffix": "memory", "media": "application/x-ndjson",
                        "role": "the operator's notes for the agent; committed by digest, the bytes stay local"},
}
SPEC_URL = "https://github.com/cryptoAGI/bankml/blob/main/usage.md"


# ── §5: order, records, bundle_root, the 64-leaf Merkle tree ─────────────────────────────────────────────────
def order(labels) -> list:
    core = [c for c in CORE if c in labels]
    return core + sorted(l for l in labels if l not in CORE)


def records(facets: list) -> list:
    present = {f["facet"]: f["sha256"] for f in facets if f.get("state", "present") == "present"}
    return [l.encode("utf-8") + b"\x1f" + present[l].encode("ascii") for l in order(present)]


def bundle_root(recs: list) -> tuple:
    pre = b"".join(r + b"\x1e" for r in recs)
    return "0x" + keccak256(pre).hex(), len(pre)


def merkle_root(recs: list) -> str:
    if len(recs) > 64:
        raise ValueError("more than 64 present facets: a binder fails, it never truncates (§5)")
    level = [keccak256(r) for r in recs] + [PAD] * (64 - len(recs))
    while len(level) > 1:
        level = [keccak256(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return "0x" + level[0].hex()


# ── §2: identity, over the document without its identity block ──────────────────────────────────────────────
def identity(manifest: dict) -> dict:
    body = {k: v for k, v in manifest.items() if k != "identity"}
    c = canonical_bytes(body)
    cid = cid_v1_raw(c)
    return {"thot": "thot:" + hashlib.sha256(c).hexdigest(), "cid": cid, "name": "thot-" + cid, "contentRoot": "0x" + keccak256(c).hex()}


def check_structure(m: dict) -> list:
    """§8 steps 2–4 and 7 from the manifest alone (no facet bytes): the findings, empty when it holds."""
    out = []
    recs = records(m["facets"])
    br, n = bundle_root(recs)
    labels = order([f["facet"] for f in m["facets"] if f.get("state", "present") == "present"])
    want = {"bundle_root.value": br, "bundle_root.preimage_bytes": n, "bundle_root.order": labels, "merkle.root": merkle_root(recs),
            "merkle.leaves": 64, "merkle.populated": len(recs), "merkle.ternary_head": "persona", "merkle.ternary_head_index": 0}
    for k, v in want.items():
        a, b = k.split(".")
        if m.get(a, {}).get(b) != v:
            out.append(f"{k}: {m.get(a, {}).get(b)!r} != recomputed {v!r}")
    got, want_id = m.get("identity") or {}, identity(m)
    for k in ("thot", "cid", "name", "contentRoot"):  # §8 step 3: these four; other identity keys are descriptive
        if got.get(k) != want_id[k]:
            out.append(f"identity.{k} does not recompute")
    b = m.get("bundle", {})
    g = b.get("generation")
    if not (isinstance(g, int) and not isinstance(g, bool) and g >= 1):
        out.append("V1: generation is not an integer >= 1")
    elif (g == 1) != (b.get("parent") is None):
        out.append("V2: generation 1 exactly when parent is null")
    if not (isinstance(b.get("parent_reason"), str) and b.get("parent_reason")):
        out.append("V4: parent_reason missing")
    return out


def check_facets(m: dict, root: Path) -> list:
    """§8 step 1: every present facet's bytes, size, sha256 and CID."""
    out = []
    for f in m["facets"]:
        if f.get("state", "present") != "present":
            continue
        try:
            b = (root / f["path"]).read_bytes()
        except OSError as e:
            out.append(f"{f['facet']}: {f['path']} unreadable ({e.strerror})")
            continue
        if (len(b), hashlib.sha256(b).hexdigest(), cid_v1_raw(b)) != (f["bytes"], f["sha256"], f["cid"]):
            out.append(f"{f['facet']}: bytes differ from the manifest")
    return out


# ── the agent's bundle ─────────────────────────────────────────────────────────────────────────────────────
def _git(d: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(d), *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit_locator(slug: str) -> tuple:
    """Commit the agent's public files in its own git repository; (locator, held labels)."""
    d = agent_dir(slug)
    if not (d / ".git").is_dir():
        _git(d, "init", "-q")
        (d / ".gitignore").write_text(f"{slug}.history\n{slug}.memory\n*.thot.prev.json\n", encoding="utf-8")
    _git(d, "add", "-A")
    if _git(d, "status", "--porcelain"):
        _git(d, "-c", "user.name=bankml", "-c", "user.email=bankml@localhost", "commit", "-q", "-m", "bankml: agent files")
    head = _git(d, "rev-parse", "HEAD")
    tracked = set(_git(d, "ls-files").splitlines())
    return f"localhost/{getpass.getuser()}/{slug}@{head}", tracked


def build(slug: str, template_manifest: dict | None = None, reason: str | None = None) -> dict:
    """Write <slug>.thot.json: generation 1 at genesis, or n+1 with parent = the previous identity.cid when a facet
    changed (§7 G2, G3); a re-bind with no facet change keeps the generation (G4)."""
    d, f = agent_dir(slug), files(slug)
    for k in ("history", "memory"):
        f[k].touch(exist_ok=True)  # an empty file is a real, committed state
    facets, custom = [], []
    for label, path in (("persona", f["persona"]), ("prompt", f["prompt"])):
        b = path.read_bytes()
        facets.append({"facet": label, "path": path.name, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "cid": cid_v1_raw(b),
                       "state": "present", "custom": False, "added_in": 1})
    locator, tracked = _commit_locator(slug)
    out = d / f"{slug}.thot.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
    prev_added = {x["facet"]: x.get("added_in", 1) for x in (prev or {}).get("facets", [])}
    gen = (prev or {}).get("bundle", {}).get("generation", 1)
    custom_now = dict(CUSTOM)
    from agents import aivatar_path
    av = aivatar_path(slug)
    if av:  # the chosen portrait, when there is one
        custom_now["x-bankml.aivatar"] = {"suffix": av.name[len(slug) + 1:], "media": "image/" + av.suffix[1:],
                                          "role": "the agent's chosen portrait (its aivatar); PNG, JPEG or WebP by content"}
    for label, spec in sorted(custom_now.items()):
        p = d / f"{slug}.{spec['suffix']}"
        b = p.read_bytes()
        added = prev_added.get(label, gen if prev else 1)
        facets.append({"facet": label, "path": p.name, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "cid": cid_v1_raw(b),
                       "state": "present", "custom": True, "added_in": added})
        custom.append({"facet": label, "path": p.name, "owner": "bankml", "spec_url": SPEC_URL, "media": spec["media"], "added_in": added, "role": spec["role"]})
    for x in facets:
        x["at_locator"] = x["path"] in tracked
    changed = prev is None or {x["facet"]: x["sha256"] for x in prev["facets"]} != {x["facet"]: x["sha256"] for x in facets}
    if prev is None:
        bundle = {"id": slug, "officer": json.loads(f["persona"].read_text(encoding="utf-8")).get("name", slug), "generation": 1, "parent": None,
                  "parent_reason": "genesis: a new agent derived by bankml; no earlier manifest of this bundle"}
    elif changed:
        diff = sorted({x["facet"] for x in facets} ^ {x["facet"] for x in prev["facets"]} |
                      {x["facet"] for x in facets for y in prev["facets"] if x["facet"] == y["facet"] and x["sha256"] != y["sha256"]})
        bundle = dict(prev["bundle"], generation=prev["bundle"]["generation"] + 1, parent=prev["identity"]["cid"],
                      parent_reason=reason or f"facets changed: {', '.join(diff)}")
        for x in facets:  # a facet first present in this generation is added_in it
            if x["facet"] not in prev_added:
                x["added_in"] = bundle["generation"]
        (d / f"{slug}.thot.prev.json").write_text(json.dumps(prev, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        bundle = prev["bundle"]
    recs = records(facets)
    br, n = bundle_root(recs)
    labels = order([x["facet"] for x in facets])
    persona = json.loads(f["persona"].read_text(encoding="utf-8"))
    m = {"$comment": "DERIVED OUTPUT — regenerable by bankml (ui/thot.py), never hand-edited. Contains no digest of itself.",
         "schema": SCHEMA, "algorithms": ALGORITHMS, "bundle": bundle,
         "facets": facets, "absent": [{"facet": c, "state": "absent", "reason": "not authored for this bankml agent"} for c in CORE if c not in ("persona", "prompt")],
         "custom": custom, "custom_rule": "bankml's facets are namespaced x-bankml.* and declared here (registry custom_rule)",
         "doctrine_root": doctrine_root(persona),
         "bundle_root": {"value": br, "hash": "keccak256", "preimage_bytes": n, "order": labels,
                         "construction": "concat over present facets — core in registry order, then custom by name — of (facet_utf8 + 0x1f + sha256_hex_ascii + 0x1e)"},
         "merkle": {"leaves": 64, "leaf_rule": "keccak256(facet_utf8 || 0x1f || sha256_hex_ascii), in bundle_root order", "padding": "keccak256(b'')",
                    "populated": len(recs), "root": merkle_root(recs), "ternary_head": "persona", "ternary_head_index": 0},
         "rung": {"value": "referenced", "derived_by": "permanence/lib/rungs.js rungOf(evidence) — derived from evidence, never asserted",
                  "ladder": ["referenced", "committed", "stored", "attested"],
                  "evidence": {"locator": locator, "locator_holds": [x["facet"] for x in facets if x["at_locator"]],
                               "locator_lacks": [x["facet"] for x in facets if not x["at_locator"]],
                               "commitTx": None, "dataTx": None, "attestation": None}},
         "license": "MIT"}
    if template_manifest and template_manifest.get("identity"):
        m["relations"] = [{"relation": "derived_from", "bundle": template_manifest.get("bundle", {}).get("id"), "repository": "github.com/cryptoAGI/savante",
                           "manifest_cid": template_manifest["identity"]["cid"]}]
    elif prev and prev.get("relations"):
        m["relations"] = prev["relations"]
    m["identity"] = identity(m)
    text = json.dumps(m, indent=1, ensure_ascii=False) + "\n"
    out.write_text(text, encoding="utf-8")
    arch = d / "thot"  # every manifest ever built, by its CID: a token commits to one generation forever
    arch.mkdir(exist_ok=True)
    (arch / f"{m['identity']['cid']}.json").write_text(text, encoding="utf-8")
    return m


def generations(slug: str) -> list:
    """All archived manifests of the agent, oldest generation first."""
    arch = agent_dir(slug) / "thot"
    ms = [json.loads(p.read_text(encoding="utf-8")) for p in arch.glob("*.json")] if arch.is_dir() else []
    return sorted(ms, key=lambda m: (m["bundle"]["generation"], m["identity"]["cid"]))


def verify(slug: str) -> list:
    m = json.loads((agent_dir(slug) / f"{slug}.thot.json").read_text(encoding="utf-8"))
    return check_structure(m) + check_facets(m, agent_dir(slug))
