#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PostgreSQL connector against a throwaway cluster: initdb in a temp directory, a unix socket there, pgvector
enabled as that cluster's superuser. Nothing touches the system's databases. Skips (exit 0) when initdb is absent.
Covers: schema (HNSW index on vector(1024)), publish public-only and with private lines, load verified byte for
byte, a tampered row refused, the THOT manifest and its lineage across a publish.
run: python3 testing/test_connectors.py"""
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

BIN = Path("/usr/lib/postgresql/16/bin")
if not (BIN / "initdb").is_file() or not shutil.which("psql"):
    print("skip  no PostgreSQL 16 binaries here")
    sys.exit(0)
if not Path("/usr/share/postgresql/16/extension/vector.control").is_file():
    print("skip  pgvector is not installed for PostgreSQL 16 here")
    sys.exit(0)

tmp = Path(tempfile.mkdtemp(prefix="bankml-pg-"))
os.environ.update(BANKML_AGENTS=str(tmp / "agents"), BANKML_UI_STATE=str(tmp / "state"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import agents, connectors, thot  # noqa: E402

data, sock, port = tmp / "pgdata", tmp / "sock", "5499"
sock.mkdir()
subprocess.run([BIN / "initdb", "-D", data, "-A", "trust", "-U", "bankml", "--no-sync"], check=True, capture_output=True)
subprocess.run([BIN / "pg_ctl", "-D", data, "-o", f"-k {sock} -p {port} -c listen_addresses=''", "-l", tmp / "pg.log", "-w", "start"], check=True, capture_output=True)
dsn = f"host={sock} port={port} user=bankml dbname=postgres"
fails = 0


def check(name, ok):
    global fails
    fails += not ok
    print(("ok   " if ok else "FAIL ") + name)


try:
    connectors.psql("CREATE DATABASE bankml;", dsn=dsn)
    dsn = f"host={sock} port={port} user=bankml dbname=bankml"
    connectors.psql("CREATE EXTENSION vector;", dsn=dsn)
    st = connectors.ensure_schema(dsn)
    check(f"schema ready with pgvector {st.get('vector')} and an {st.get('index')} index", st["ok"] and st["index"].startswith("hnsw"))
    cols = connectors.psql("SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid='bankml_exchanges'::regclass AND attname='embedding';", dsn=dsn).strip()
    check("bankml_exchanges.embedding is vector(1024)", cols == "vector(1024)")

    tpl = {"persona": "tpl", "name": "Template", "source": "s", "format": "f", "system_prompt": "You are the template.", "mantra": "m", "oath": "o",
           "bdi": {"beliefs": [{"id": "b", "belief": "x"}]}, "skills": {"primary": "p", "taxonomy": [], "defer_triggers": [], "validation": []},
           "safety": {"scope": "s"}, "embodiment": {}, "token": {"intelligence": {"tool_allowlist": []}}, "notes": "ünïcode ✓ and a tab\there"}
    slug = agents.derive(tpl, {}, "Auditor Ada", system_prompt="You are Ada.\nLine two, with \\ and 'quotes' and $$ dollars;")
    f = agents.files(slug)
    f["history"].write_text('{"ts": 1, "user": "q1 \'x\'", "assistant": "a1\\ttab"}\n{"ts": 2, "user": "q2", "assistant": "a2 ✓"}\n', encoding="utf-8")
    f["memory"].write_text('{"ts": 3, "text": "a note; DROP TABLE bankml_agents; --"}\n', encoding="utf-8")

    r = connectors.publish(slug, dsn=dsn)
    n = connectors.psql("SELECT (SELECT count(*) FROM bankml_exchanges) || ',' || (SELECT count(*) FROM bankml_memory);", dsn=dsn).strip()
    check("publish (public only): the agent row, and no private lines", r["private_included"] is False and n == "0,0")
    row = json.loads(connectors.psql("SELECT row_to_json(a) FROM bankml_agents a;", dsn=dsn))
    check("the database holds the history's commitment, not its content", "q1" not in json.dumps(row) and
          any(x["facet"] == "x-bankml.history" for x in row["manifest"]["facets"]))

    r2 = connectors.publish(slug, include_private=True, dsn=dsn)
    check("publish with private lines: 2 exchanges, 1 note; the injection text is stored as data", r2["exchanges"] == 2 and r2["memory"] == 1
          and connectors.psql("SELECT count(*) FROM bankml_agents;", dsn=dsn).strip() == "1")
    check("published() lists it", [p["slug"] for p in connectors.published(dsn)] == [slug])

    l = connectors.load(slug, as_slug="ada_copy", dsn=dsn)
    g = agents.files("ada_copy")
    check("load verifies and restores persona and prompt byte for byte", l["verified"] and g["persona"].read_bytes() == f["persona"].read_bytes()
          and g["prompt"].read_bytes() == f["prompt"].read_bytes())
    check("load restores history and memory byte for byte (so their proofs still hold)", g["history"].read_bytes() == f["history"].read_bytes()
          and g["memory"].read_bytes() == f["memory"].read_bytes())
    check("the loaded agent's ledger verifies", all(ok for _, ok, _ in agents.verify("ada_copy")))

    connectors.psql("UPDATE bankml_agents SET prompt = prompt || ' tampered';", dsn=dsn)
    try:
        connectors.load(slug, as_slug="ada_bad", dsn=dsn)
        check("a tampered prompt in the database is refused on load", False)
    except ValueError:
        check("a tampered prompt in the database is refused on load", True)
    connectors.publish(slug, include_private=True, dsn=dsn)  # restore
    connectors.psql("UPDATE bankml_exchanges SET line = replace(line, 'a2', 'A2') WHERE seq = 1;", dsn=dsn)
    try:
        connectors.load(slug, as_slug="ada_bad2", dsn=dsn)
        check("a tampered history line in the database is refused on load", False)
    except ValueError:
        check("a tampered history line in the database is refused on load", True)

    g1 = json.loads((agents.agent_dir(slug) / f"{slug}.thot.json").read_text())["bundle"]["generation"]
    f["memory"].write_text(f["memory"].read_text() + '{"ts": 4, "text": "another"}\n', encoding="utf-8")
    r3 = connectors.publish(slug, dsn=dsn)
    check("a facet change makes the next publish a new THOT generation", r3["generation"] == g1 + 1 and not thot.verify(slug))
    connectors.publish(slug, include_private=True, dsn=dsn)
    before = connectors.psql("SELECT generation, (SELECT count(*) FROM bankml_exchanges) FROM bankml_agents;", dsn=dsn).strip()
    f["history"].write_text(f["history"].read_text() + "not json at all\n", encoding="utf-8")  # the line's ::jsonb cast fails mid-publish
    try:
        connectors.publish(slug, include_private=True, dsn=dsn)
        check("a publish that fails part-way changes nothing (one transaction)", False)
    except RuntimeError:
        after = connectors.psql("SELECT generation, (SELECT count(*) FROM bankml_exchanges) FROM bankml_agents;", dsn=dsn).strip()
        check("a publish that fails part-way changes nothing (one transaction)", before == after)
    cr = json.loads((agents.agent_dir(slug) / f"{slug}.thot.json").read_text())["identity"]["contentRoot"]
    try:
        connectors.load(slug, as_slug="ada_anchor", dsn=dsn, expect_content_root="0x" + "00" * 32)
        check("load anchored to a contentRoot refuses a row that does not match it", False)
    except ValueError:
        check("load anchored to a contentRoot refuses a row that does not match it", True)
finally:
    subprocess.run([BIN / "pg_ctl", "-D", data, "-m", "fast", "-w", "stop"], capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)

print("all ok" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
