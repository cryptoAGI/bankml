#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The chain side of a bankml agent: prepare an iNFT mint the owner signs, and load an agent back from a token.

Target: the house ERC-7857 contract `iNFT_7857` (DeltaVerse/deploy/iNFT4), open (unsealed) agents:

  mintOpenAgent(address to, bytes32 contentRoot, string storageURI, bytes32 metadataRoot,
                uint256 dimensions, uint8 parallelUnits, string tokenURI_) returns (uint256 tokenId)   — MINTER_ROLE
  getPayload(uint256) → (bytes32 contentRoot, string storageURI, bytes32 metadataRoot, uint256 dimensions,
                         uint8 parallelUnits, uint40 mintedAt, bytes32 sealedKeyHash, bool verified)

From an agent's THOT bundle: contentRoot = the manifest's identity.contentRoot (keccak256 of its canonical bytes; the
contract accepts a content root once, ever); metadataRoot = keccak256 of the agent card's canonical bytes. The storage
and token URIs are `local://…` references until the bundle is stored somewhere (Arweave via the permanence rungs is
the house path) — said so, never dressed up.

bankml prepares, simulates (eth_call) and hands over an unsigned transaction and the equivalent `cast send`. It sends
only to a local devnet (chain id 31337, anvil's unlocked accounts) — on any other chain the owner signs, in their
own wallet. The contract is not deployed on a public chain today, its audit is not cleared, and minting needs
MINTER_ROLE.

stdlib only; keccak256 from agents.py.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import agents
from agents import canonical_bytes, keccak256

DEVNET_CHAIN_IDS = {31337}
MINT_SIG = "mintOpenAgent(address,bytes32,string,bytes32,uint256,uint8,string)"
ERRORS = ["TransferRequiresSealedKey()", "InvalidDimension(uint256)", "ZeroAddress()", "ZeroBytes32()", "EmptyString()",
          "StringTooLong(uint256)", "ContentRootAlreadyMinted(bytes32)", "TokenDoesNotExist(uint256)", "NotAuthorized(address)",
          "AccessControlUnauthorizedAccount(address,bytes32)", "EnforcedPause()", "TokenIsOpen(uint256)"]
VALID_DIMENSIONS = (8, 64, 256, 384, 512, 768, 1024, 2048, 4096, 8192, 65536, 1048576)


# ── ABI (the subset these calls need) ──────────────────────────────────────────────────────────────────────────
def selector(sig: str) -> bytes:
    return keccak256(sig.encode())[:4]


def _word(t: str, v) -> bytes:
    if t == "address":
        b = bytes.fromhex(v[2:] if v.startswith("0x") else v)
        if len(b) != 20:  # a short or long address would shift every later argument in calldata the owner signs
            raise ValueError(f"an address is 20 bytes, not {len(b)}: {v}")
        return bytes(12) + b
    if t == "bytes32":
        b = bytes.fromhex(v[2:] if isinstance(v, str) else v.hex())
        assert len(b) == 32, "bytes32 must be 32 bytes"
        return b
    if t == "bool":
        return int(bool(v)).to_bytes(32, "big")
    if t.startswith("uint"):
        n = int(v)
        assert 0 <= n < 1 << int(t[4:] or 256), f"{t} out of range"
        return n.to_bytes(32, "big")
    raise ValueError(t)


def encode(types: list, values: list) -> bytes:
    """Head/tail ABI encoding; `string` is the only dynamic type used here."""
    head, tail = b"", b""
    base = 32 * len(types)
    for t, v in zip(types, values):
        if t == "string":
            b = v.encode("utf-8")
            head += (base + len(tail)).to_bytes(32, "big")
            tail += len(b).to_bytes(32, "big") + b + bytes((-len(b)) % 32)
        else:
            head += _word(t, v)
    return head + tail


def _u(b: bytes, i: int) -> int:
    return int.from_bytes(b[32 * i:32 * i + 32], "big")


def _str_at(b: bytes, off: int) -> str:
    n = int.from_bytes(b[off:off + 32], "big")
    return b[off + 32:off + 32 + n].decode("utf-8", "replace")


def decode_payload(ret: bytes) -> dict:
    t = _u(ret, 0)  # the returned tuple is dynamic: an offset to it first
    b = ret[t:]
    return {"contentRoot": "0x" + b[0:32].hex(), "storageURI": _str_at(b, _u(b, 1)), "metadataRoot": "0x" + b[64:96].hex(),
            "dimensions": _u(b, 3), "parallelUnits": _u(b, 4), "mintedAt": _u(b, 5), "sealedKeyHash": "0x" + b[192:224].hex(),
            "verified": bool(_u(b, 7))}


def decode_revert(data: str) -> str:
    b = bytes.fromhex(data[2:]) if data and data.startswith("0x") else b""
    if len(b) < 4:
        return "reverted without a reason"
    if b[:4] == selector("Error(string)"):
        return "Error: " + _str_at(b[4:], _u(b[4:], 0))
    for e in ERRORS:
        if b[:4] == selector(e):
            args = [("0x" + b[4 + 32 * i:36 + 32 * i].hex()) for i in range((len(b) - 4) // 32)]
            return e.split("(")[0] + ("(" + ", ".join(args) + ")" if args else "")
    return "reverted: 0x" + b.hex()


# ── JSON-RPC ───────────────────────────────────────────────────────────────────────────────────────────────────
class RPCError(Exception):
    def __init__(self, err):
        super().__init__(err.get("message", str(err)))
        self.data = err.get("data")


def rpc(url: str, method: str, params: list):
    req = urllib.request.Request(url, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read())
    if "error" in out:
        raise RPCError(out["error"])
    return out["result"]


def call(url, to, data: bytes, frm=None):
    tx = {"to": to, "data": "0x" + data.hex()}
    if frm:
        tx["from"] = frm
    return bytes.fromhex(rpc(url, "eth_call", [tx, "latest"])[2:])


def receipt(url, h, timeout=60.0):
    """Wait for a transaction's receipt."""
    import time
    t = time.time()
    while time.time() - t < timeout:
        rc = rpc(url, "eth_getTransactionReceipt", [h])
        if rc:
            return rc
        time.sleep(0.2)
    raise TimeoutError(f"no receipt for {h} after {timeout} s")


def chain_id(url) -> int:
    return int(rpc(url, "eth_chainId", []), 16)


# ── the mint, prepared from a THOT bundle ─────────────────────────────────────────────────────────────────────
def plan_mint(slug: str, to: str, dimensions: int = 768, parallel_units: int = 1, storage_uri: str | None = None, token_uri: str | None = None) -> dict:
    d = agents.agent_dir(slug)
    m = json.loads((d / f"{slug}.thot.json").read_text(encoding="utf-8"))
    card = json.loads((d / f"{slug}.agentcard.json").read_text(encoding="utf-8"))
    if dimensions not in VALID_DIMENSIONS:
        raise ValueError(f"dimensions must be one of {VALID_DIMENSIONS}")
    args = {"to": to, "contentRoot": m["identity"]["contentRoot"],
            "storageURI": storage_uri or f"local://thot/{m['identity']['cid']}",
            "metadataRoot": "0x" + keccak256(canonical_bytes(card)).hex(), "dimensions": dimensions, "parallelUnits": parallel_units,
            "tokenURI": token_uri or f"local://card/{agents.file_commitment((d / f'{slug}.agentcard.json').read_bytes())['cid']}"}
    data = selector(MINT_SIG) + encode(["address", "bytes32", "string", "bytes32", "uint256", "uint8", "string"],
                                       [args["to"], args["contentRoot"], args["storageURI"], args["metadataRoot"], args["dimensions"],
                                        args["parallelUnits"], args["tokenURI"]])
    return {"agent": slug, "thot": {"cid": m["identity"]["cid"], "generation": m["bundle"]["generation"]}, "function": MINT_SIG,
            "args": args, "calldata": "0x" + data.hex(),
            "note": "storageURI/tokenURI are local references until the bundle and card are stored; the rung stays 'referenced'"}


def simulate(url: str, contract: str, frm: str, plan: dict) -> dict:
    """eth_call the mint as `frm`: the token id it would get, or the decoded revert."""
    try:
        ret = call(url, contract, bytes.fromhex(plan["calldata"][2:]), frm)
        return {"ok": True, "tokenId": int.from_bytes(ret[:32], "big")}
    except RPCError as e:
        d = e.data if isinstance(e.data, str) else (e.data or {}).get("data") if isinstance(e.data, dict) else None
        return {"ok": False, "error": decode_revert(d) if d else str(e)}


def unsigned_tx(url: str, contract: str, frm: str, plan: dict) -> dict:
    """What the owner signs, and the same as a `cast send` line (their key, their wallet)."""
    cid = chain_id(url)
    tx = {"chainId": cid, "from": frm, "to": contract, "value": "0x0", "data": plan["calldata"]}
    try:
        tx["gas"] = rpc(url, "eth_estimateGas", [{k: tx[k] for k in ("from", "to", "value", "data")}])
    except RPCError as e:
        tx["gas_estimate_error"] = str(e)
    import shlex
    a = plan["args"]
    q = shlex.quote  # every argument quoted: a storage or token URI cannot break out of the command the owner pastes
    cast = (f'cast send {q(contract)} {q(MINT_SIG)} {q(a["to"])} {q(a["contentRoot"])} {q(a["storageURI"])} {q(a["metadataRoot"])} '
            f'{q(str(a["dimensions"]))} {q(str(a["parallelUnits"]))} {q(a["tokenURI"])} --rpc-url <RPC> --account <your keystore>')
    return {"tx": tx, "cast": cast}


def send_devnet(url: str, contract: str, frm: str, plan: dict) -> dict:
    """Send on a local devnet only (anvil's unlocked accounts). Refused on every other chain."""
    from urllib.parse import urlsplit
    host = (urlsplit(url).hostname or "").lower()
    if host not in ("127.0.0.1", "localhost", "::1"):  # a remote node can report any chain id; a devnet is on this machine
        raise PermissionError(f"{url} is not on this machine: bankml sends only to a local devnet; the owner signs everything else")
    cid = chain_id(url)
    if cid not in DEVNET_CHAIN_IDS:
        raise PermissionError(f"chain {cid} is not a local devnet: the owner signs this mint in their own wallet (unsigned_tx)")
    h = rpc(url, "eth_sendTransaction", [{"from": frm, "to": contract, "data": plan["calldata"]}])
    rc = receipt(url, h)
    return {"tx": h, "status": int(rc["status"], 16), "block": int(rc["blockNumber"], 16)}


# ── loading an agent back from a token ────────────────────────────────────────────────────────────────────────
def read_token(url: str, contract: str, token_id: int) -> dict:
    tid = encode(["uint256"], [token_id])
    p = decode_payload(call(url, contract, selector("getPayload(uint256)") + tid))
    uri = _str_at(ret := call(url, contract, selector("tokenURI(uint256)") + tid), _u(ret, 0))
    owner = "0x" + call(url, contract, selector("ownerOf(uint256)") + tid)[12:32].hex()
    is_open = bool(_u(call(url, contract, selector("openMint(uint256)") + tid), 0))
    return {"tokenId": token_id, "owner": owner, "open": is_open, "tokenURI": uri, **p}


def find_local_agent(content_root: str) -> str | None:
    """The local agent whose THOT manifest's contentRoot is this token's (current or any recorded previous generation)."""
    import thot
    for slug in agents.list_agents():
        if any(m["identity"]["contentRoot"] == content_root for m in thot.generations(slug)):
            return slug
    return None


def _descends(gens: list, root: dict, head: dict) -> bool:
    """Walk head's parent links back through the archived manifests; True if it reaches root (THOT §7)."""
    by_cid = {g["identity"]["cid"]: g for g in gens}
    cur = head
    while cur is not None:
        if cur["identity"]["cid"] == root["identity"]["cid"]:
            return True
        cur = by_cid.get(cur["bundle"].get("parent"))
    return False


def load_from_chain(url: str, contract: str, token_id: int) -> dict:
    """Read the token; find the agent it commits to (locally); verify the bundle. The chain holds roots, not files:
    the files come from the owner's copy (or a store the storageURI names), and are accepted only if they hash to it."""
    t = read_token(url, contract, token_id)
    slug = find_local_agent(t["contentRoot"])
    if not slug:
        return {"token": t, "agent": None, "verified": False,
                "reason": "no local bundle has this contentRoot; fetch the bundle its storageURI names, then load it (the files must hash to the root)"}
    import thot
    findings = thot.verify(slug)
    m = json.loads((agents.agent_dir(slug) / f"{slug}.thot.json").read_text(encoding="utf-8"))
    current = m["identity"]["contentRoot"] == t["contentRoot"]
    minted = next(g for g in thot.generations(slug) if g["identity"]["contentRoot"] == t["contentRoot"])
    lineage_ok = _descends(thot.generations(slug), minted, m)
    return {"token": t, "agent": slug, "verified": not findings and lineage_ok, "findings": findings,
            "minted_generation": minted["bundle"]["generation"], "current_generation": m["bundle"]["generation"],
            "descends_from_minted": lineage_ok, "generation_on_chain_is_current": current,
            "note": None if current else "the token commits to an earlier generation (the genesis stays attached; later ones are reached by metadata)"}


# ── a local devnet, for trying the whole path without any public chain ───────────────────────────────────────
DEVNET = {"proc": None, "url": None, "contract": None, "accounts": None}
ARTIFACT = Path.home() / "DeltaVerse" / "deploy" / "iNFT4" / "out" / "iNFT_7857.sol" / "iNFT_7857.json"


def devnet_up(port: int = 8545, artifact: Path = ARTIFACT) -> dict:
    """Start anvil (chain id 31337) on loopback and deploy iNFT_7857 from its compiled artifact; admin = account 0."""
    import shutil
    import subprocess
    import time
    if DEVNET["proc"] and DEVNET["proc"].poll() is None:
        return {k: v for k, v in DEVNET.items() if k != "proc"}
    anvil = shutil.which("anvil") or str(Path.home() / ".foundry" / "bin" / "anvil")
    if not Path(anvil).is_file() or not artifact.is_file():
        raise FileNotFoundError("needs Foundry's anvil and the iNFT4 build (forge build in DeltaVerse/deploy/iNFT4)")
    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen([anvil, "--host", "127.0.0.1", "--port", str(port), "--silent"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            chain_id(url)
            break
        except OSError:
            time.sleep(0.1)
    accts = rpc(url, "eth_accounts", [])
    art = json.loads(artifact.read_text())
    ctor = encode(["string", "string", "address", "address", "uint96", "address", "address", "uint256"],
                  ["bankml agents (devnet)", "BKA", accts[0], accts[0], 0, "0x" + "00" * 20, "0x" + "00" * 20, 0])
    h = rpc(url, "eth_sendTransaction", [{"from": accts[0], "data": art["bytecode"]["object"] + ctor.hex(), "gas": hex(12_000_000)}])
    DEVNET.update(proc=proc, url=url, contract=receipt(url, h)["contractAddress"], accounts=accts[:3])
    return {k: v for k, v in DEVNET.items() if k != "proc"}

