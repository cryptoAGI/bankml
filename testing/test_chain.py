#!/usr/bin/env python3
"""The chain side against a throwaway local EVM: anvil on a free port, the house iNFT_7857 (DeltaVerse iNFT4) deployed
from its compiled artifact, a bankml agent minted from its THOT bundle and loaded back. Nothing leaves this computer.
Skips (exit 0) when anvil or the artifact is absent (CI).
run: python3 testing/test_chain.py"""
import json, os, shutil, socket, subprocess, sys, tempfile, time
from pathlib import Path

ART = Path(os.environ.get("BANKML_INFT_ARTIFACT", Path.home() / "DeltaVerse/deploy/iNFT4/out/iNFT_7857.sol/iNFT_7857.json"))
ANVIL = shutil.which("anvil") or str(Path.home() / ".foundry/bin/anvil")
if not ART.is_file() or not Path(ANVIL).is_file():
    print("skip  anvil or the iNFT_7857 artifact is not here")
    sys.exit(0)

tmp = Path(tempfile.mkdtemp(prefix="bankml-chain-"))
os.environ.update(BANKML_AGENTS=str(tmp / "agents"), BANKML_UI_STATE=str(tmp / "state"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import agents, chain, thot  # noqa: E402

s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
url = f"http://127.0.0.1:{port}"
proc = subprocess.Popen([ANVIL, "--port", str(port), "--silent"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
fails = 0


def check(name, ok):
    global fails
    fails += not ok
    print(("ok   " if ok else "FAIL ") + name)


try:
    for _ in range(100):
        try:
            chain.chain_id(url)
            break
        except OSError:
            time.sleep(0.1)
    accts = chain.rpc(url, "eth_accounts", [])
    minter, holder, stranger = accts[0], accts[1], accts[2]
    check("anvil is a devnet (chain id 31337)", chain.chain_id(url) == 31337)

    art = json.loads(ART.read_text())
    ctor = chain.encode(["string", "string", "address", "address", "uint96", "address", "address", "uint256"],
                        ["bankml agents (devnet)", "BKA", minter, minter, 0, "0x" + "00" * 20, "0x" + "00" * 20, 0])
    h = chain.rpc(url, "eth_sendTransaction", [{"from": minter, "data": art["bytecode"]["object"] + ctor.hex(), "gas": hex(12_000_000)}])
    rc = chain.receipt(url, h)
    inft = rc.get("contractAddress")
    check("iNFT_7857 deployed from its artifact", int(rc["status"], 16) == 1 and inft)

    tpl = {"persona": "tpl", "name": "Template", "source": "s", "format": "f", "system_prompt": "You are the template.", "mantra": "m", "oath": "o",
           "bdi": {"beliefs": []}, "skills": {"primary": "p", "taxonomy": [], "defer_triggers": [], "validation": []},
           "safety": {}, "embodiment": {}, "token": {"intelligence": {"tool_allowlist": []}}}
    slug = agents.derive(tpl, {}, "Minted Mira", system_prompt="You are Mira.")
    m1 = thot.build(slug)
    plan = chain.plan_mint(slug, to=holder)
    check("the plan's contentRoot is the THOT manifest's", plan["args"]["contentRoot"] == m1["identity"]["contentRoot"])

    sim = chain.simulate(url, inft, minter, plan)
    check(f"simulate as the minter: token {sim.get('tokenId')}", sim["ok"])
    bad = chain.simulate(url, inft, stranger, plan)
    check(f"simulate without MINTER_ROLE is refused, decoded: {bad.get('error', '')[:60]}", not bad["ok"] and "AccessControl" in bad.get("error", ""))
    ut = chain.unsigned_tx(url, inft, minter, plan)
    check("unsigned tx for the owner: chain id, calldata, gas estimate, cast line", ut["tx"]["chainId"] == 31337 and ut["tx"]["data"] == plan["calldata"]
          and "gas" in ut["tx"] and ut["cast"].startswith("cast send"))

    sent = chain.send_devnet(url, inft, minter, plan)
    check("minted on the devnet", sent["status"] == 1)
    tok = chain.read_token(url, inft, sim["tokenId"])
    check("the token reads back: contentRoot, storageURI, open, owner", tok["contentRoot"] == m1["identity"]["contentRoot"]
          and tok["storageURI"] == plan["args"]["storageURI"] and tok["open"] and tok["owner"].lower() == holder.lower()
          and tok["dimensions"] == 768 and tok["metadataRoot"] == plan["args"]["metadataRoot"])
    again = chain.simulate(url, inft, minter, plan)
    check(f"the same content root cannot be minted twice: {again.get('error', '')[:40]}", not again["ok"] and "ContentRootAlreadyMinted" in again.get("error", ""))

    ld = chain.load_from_chain(url, inft, sim["tokenId"])
    check("load from chain finds the agent by contentRoot and verifies it", ld["agent"] == slug and ld["verified"] and ld["generation_on_chain_is_current"])

    for i in range(2):  # the agent evolves; the token keeps pointing at the genesis
        agents.files(slug)["history"].write_text(agents.files(slug)["history"].read_text() + json.dumps({"n": i}) + "\n")
        thot.build(slug)
    ld2 = chain.load_from_chain(url, inft, sim["tokenId"])
    check("after two generations the token still loads: lineage walks back to the minted genesis", ld2["agent"] == slug and ld2["verified"]
          and ld2["minted_generation"] == 1 and ld2["current_generation"] == 3 and ld2["descends_from_minted"] and not ld2["generation_on_chain_is_current"])

    saved = chain.DEVNET_CHAIN_IDS
    chain.DEVNET_CHAIN_IDS = set()  # pretend this is a public chain
    try:
        chain.send_devnet(url, inft, minter, chain.plan_mint(slug, to=holder))
        check("send is refused off a devnet (the owner signs)", False)
    except PermissionError:
        check("send is refused off a devnet (the owner signs)", True)
    finally:
        chain.DEVNET_CHAIN_IDS = saved
    try:
        chain.send_devnet(url.replace("127.0.0.1", "devnet.example.com"), inft, minter, plan)
        check("send is refused to a node that is not on this machine, whatever chain id it reports", False)
    except PermissionError:
        check("send is refused to a node that is not on this machine, whatever chain id it reports", True)
    try:
        chain.encode(["address"], ["0x" + "11" * 19])
        check("a 19-byte address is refused (it would shift every later argument)", False)
    except ValueError:
        check("a 19-byte address is refused (it would shift every later argument)", True)
    ut2 = chain.unsigned_tx(url, inft, minter, {**plan, "args": {**plan["args"], "tokenURI": "x\"; rm -rf ~; echo \""}})
    check("the cast line quotes every argument", "'x\"; rm -rf ~; echo \"'" in ut2["cast"])
    try:
        chain.plan_mint(slug, to=holder, dimensions=100)
        check("an invalid dimension is refused before any call", False)
    except ValueError:
        check("an invalid dimension is refused before any call", True)
finally:
    proc.terminate()
    proc.wait(timeout=10)
    shutil.rmtree(tmp, ignore_errors=True)

print("all ok" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
