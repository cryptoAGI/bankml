#!/usr/bin/env python3
"""Connectors for bankml agents: PostgreSQL (pgvector / pgvectorscale) — publish an agent, load it back, verified.

Through the `psql` client and the standard library (no driver). Values never enter SQL text: they travel as COPY
data (text format, escaped) into a temporary table, and the statements read them from there.

Publish writes the agent's public parts — persona, prompt, card, ledger, THOT manifest — and, only when asked
(`include_private=True`), its .history and .memory lines. Without that, the database holds the commitments (the THOT
manifest's digests of history and memory) and not the content: proof of data without the data.

Load rebuilds the agent from the row and refuses it unless every public file re-hashes to the digest the stored
THOT manifest commits to (and the manifest's own identity recomputes).

  BANKML_PG_DSN   libpq connection string (default "dbname=bankml": the local socket, the operator's role)
Schema: bankml_agents, bankml_exchanges (with embedding vector(1024) — bge-m3's width, as mindX uses — when the
vector extension is present; DiskANN index when pgvectorscale is, HNSW otherwise), bankml_memory.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import agents
import thot

DSN = os.environ.get("BANKML_PG_DSN", "dbname=bankml")
SETUP = ('sudo -u postgres psql -c "CREATE ROLE $USER LOGIN CREATEDB" -c "CREATE DATABASE bankml OWNER $USER" '
         '&& sudo -u postgres psql -d bankml -c "CREATE EXTENSION IF NOT EXISTS vector"')
EMBED_DIM = 1024


def _esc(v) -> str:
    """COPY text format: backslash, tab, newline and carriage return escaped; None is \\N."""
    if v is None:
        return r"\N"
    return str(v).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def psql(script: str, rows: list | None = None, dsn: str | None = None) -> str:
    """Run a script; `rows` ([(k, v), …]) are COPY-ed into temp table _in(k text, v text) first. Returns stdout."""
    head = ""
    if rows is not None:
        head = "CREATE TEMP TABLE _in(k text, v text);\nCOPY _in FROM STDIN;\n" + "".join(f"{_esc(k)}\t{_esc(v)}\n" for k, v in rows) + "\\.\n"
    p = subprocess.run(["psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-d", dsn or DSN, "-f", "-"],
                       input=head + script, capture_output=True, text=True, timeout=120)
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or f"psql exit {p.returncode}")
    return p.stdout


def status(dsn: str | None = None) -> dict:
    try:
        out = psql("SELECT json_build_object('user', current_user, 'db', current_database(), 'server', current_setting('server_version'),"
                   " 'vector', (SELECT extversion FROM pg_extension WHERE extname='vector'),"
                   " 'vectorscale', (SELECT extversion FROM pg_extension WHERE extname='vectorscale'),"
                   " 'vector_available', EXISTS (SELECT 1 FROM pg_available_extensions WHERE name='vector'),"
                   " 'vectorscale_available', EXISTS (SELECT 1 FROM pg_available_extensions WHERE name='vectorscale'));", dsn=dsn)
        return {"ok": True, **json.loads(out.strip())}
    except (RuntimeError, OSError, subprocess.TimeoutExpired, ValueError) as e:
        return {"ok": False, "error": str(e).splitlines()[-1] if str(e) else repr(e), "setup": SETUP}


def ensure_schema(dsn: str | None = None) -> dict:
    st = status(dsn)
    if not st["ok"]:
        return st
    psql("""
CREATE TABLE IF NOT EXISTS bankml_agents (slug text PRIMARY KEY, name text NOT NULL, persona jsonb NOT NULL, persona_raw text NOT NULL, prompt text NOT NULL,
  card jsonb NOT NULL, commitments jsonb NOT NULL, manifest jsonb NOT NULL, thot_cid text NOT NULL, content_root text NOT NULL,
  generation int NOT NULL, private_included boolean NOT NULL, published_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS bankml_exchanges (agent text NOT NULL REFERENCES bankml_agents(slug) ON DELETE CASCADE, seq int NOT NULL,
  line_sha256 text NOT NULL, line text NOT NULL, rec jsonb NOT NULL, PRIMARY KEY (agent, seq));
CREATE TABLE IF NOT EXISTS bankml_memory (agent text NOT NULL REFERENCES bankml_agents(slug) ON DELETE CASCADE, seq int NOT NULL,
  line_sha256 text NOT NULL, line text NOT NULL, rec jsonb NOT NULL, PRIMARY KEY (agent, seq));
""", dsn=dsn)
    if st.get("vector"):
        psql(f"ALTER TABLE bankml_exchanges ADD COLUMN IF NOT EXISTS embedding vector({EMBED_DIM});", dsn=dsn)
        if st.get("vectorscale"):
            psql("CREATE INDEX IF NOT EXISTS bankml_exchanges_emb ON bankml_exchanges USING diskann (embedding vector_cosine_ops);", dsn=dsn)
        else:
            psql("CREATE INDEX IF NOT EXISTS bankml_exchanges_emb ON bankml_exchanges USING hnsw (embedding vector_cosine_ops);", dsn=dsn)
    return {**status(dsn), "schema": "ready", "index": "diskann (pgvectorscale)" if st.get("vectorscale") else ("hnsw (pgvector)" if st.get("vector") else "none")}


def _lines(p):
    try:
        return [l for l in p.read_bytes().split(b"\n") if l.strip()]
    except OSError:
        return []


def publish(slug: str, include_private: bool = False, dsn: str | None = None) -> dict:
    """Upsert the agent (and, if asked, its history and memory lines). The THOT manifest is rebuilt first."""
    ensure_schema(dsn)
    f = agents.files(slug)
    m = thot.build(slug)
    bad = thot.verify(slug)
    if bad:
        raise ValueError(f"the agent does not verify, not published: {bad}")
    led = json.loads(f["commitments.json"].read_text(encoding="utf-8"))
    rows = [("slug", slug), ("name", m["bundle"]["officer"]), ("persona", f["persona"].read_text(encoding="utf-8")),
            ("prompt", f["prompt"].read_text(encoding="utf-8")), ("card", f["agentcard.json"].read_text(encoding="utf-8")),
            ("commitments", json.dumps(led, ensure_ascii=False)), ("manifest", json.dumps(m, ensure_ascii=False)),
            ("thot_cid", m["identity"]["cid"]), ("content_root", m["identity"]["contentRoot"]), ("generation", str(m["bundle"]["generation"])),
            ("private", "true" if include_private else "false")]
    get = lambda k: f"(SELECT v FROM _in WHERE k='{k}')"  # noqa: E731 — fixed column names only; values stay in _in
    psql(f"""
INSERT INTO bankml_agents (slug, name, persona, persona_raw, prompt, card, commitments, manifest, thot_cid, content_root, generation, private_included)
VALUES ({get('slug')}, {get('name')}, {get('persona')}::jsonb, {get('persona')}, {get('prompt')}, {get('card')}::jsonb, {get('commitments')}::jsonb,
        {get('manifest')}::jsonb, {get('thot_cid')}, {get('content_root')}, {get('generation')}::int, {get('private')}::boolean)
ON CONFLICT (slug) DO UPDATE SET name=EXCLUDED.name, persona=EXCLUDED.persona, persona_raw=EXCLUDED.persona_raw, prompt=EXCLUDED.prompt, card=EXCLUDED.card,
  commitments=EXCLUDED.commitments, manifest=EXCLUDED.manifest, thot_cid=EXCLUDED.thot_cid, content_root=EXCLUDED.content_root,
  generation=EXCLUDED.generation, private_included=EXCLUDED.private_included, published_at=now();
""", rows, dsn)
    n_ex = n_mem = 0
    for table, key in (("bankml_exchanges", "history"), ("bankml_memory", "memory")):
        psql(f"DELETE FROM {table} WHERE agent = (SELECT v FROM _in WHERE k='slug');", [("slug", slug)], dsn)
        if not include_private:
            continue
        lines = _lines(f[key])
        # k = the line's position; v = its exact text (the bytes its sha256 and the Merkle proofs are over)
        psql(f"""
INSERT INTO {table} (agent, seq, line_sha256, line, rec)
SELECT (SELECT v FROM _in WHERE k='slug'), k::int, encode(sha256(convert_to(v, 'UTF8')), 'hex'), v, v::jsonb FROM _in WHERE k <> 'slug';
""", [("slug", slug)] + [(str(i), l.decode("utf-8")) for i, l in enumerate(lines)], dsn)
        n_ex, n_mem = (len(lines), n_mem) if key == "history" else (n_ex, len(lines))
    return {"slug": slug, "thot_cid": m["identity"]["cid"], "content_root": m["identity"]["contentRoot"], "generation": m["bundle"]["generation"],
            "private_included": include_private, "exchanges": n_ex, "memory": n_mem}


def published(dsn: str | None = None) -> list:
    out = psql("SELECT coalesce(json_agg(json_build_object('slug', slug, 'name', name, 'generation', generation, 'thot_cid', thot_cid,"
               " 'private_included', private_included, 'published_at', published_at) ORDER BY published_at DESC), '[]') FROM bankml_agents;", dsn=dsn)
    return json.loads(out.strip() or "[]")


def load(slug: str, as_slug: str | None = None, dsn: str | None = None) -> dict:
    """Rebuild a published agent locally — refused unless every public file matches the stored THOT manifest."""
    out = psql("SELECT row_to_json(a) FROM bankml_agents a WHERE slug = (SELECT v FROM _in WHERE k='slug');", [("slug", slug)], dsn).strip()
    if not out:
        raise KeyError(f"no published agent {slug!r}")
    row = json.loads(out)
    m = row["manifest"]
    findings = thot.check_structure(m)
    persona_b = row["persona_raw"].encode()  # the exact bytes; the jsonb copy is for queries
    want = {x["facet"]: x["sha256"] for x in m["facets"]}
    got = {"persona": hashlib.sha256(persona_b).hexdigest(), "prompt": hashlib.sha256(row["prompt"].encode()).hexdigest()}
    findings += [f"{k}: stored bytes do not match the manifest" for k in got if got[k] != want.get(k)]
    if findings:
        raise ValueError(f"refused: {findings}")
    target = as_slug or slug
    d = agents.agent_dir(target)
    d.mkdir(parents=True, exist_ok=True)
    f = agents.files(target)
    f["persona"].write_bytes(persona_b)
    f["prompt"].write_text(row["prompt"], encoding="utf-8")
    (d / f"{target}.thot.json").write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    agents.rebind(target, derived_from=row["commitments"].get("derived_from"))
    n = 0
    if row.get("private_included"):
        for table, key, facet in (("bankml_exchanges", "history", "x-bankml.history"), ("bankml_memory", "memory", "x-bankml.memory")):
            lines = json.loads(psql(f"SELECT coalesce(json_agg(line ORDER BY seq), '[]') FROM {table} WHERE agent = (SELECT v FROM _in WHERE k='slug');",
                                    [("slug", slug)], dsn).strip())
            b = "".join(l + "\n" for l in lines).encode()
            if hashlib.sha256(b).hexdigest() != want.get(facet):
                raise ValueError(f"refused: the stored {key} does not match the manifest's {facet} digest")
            f[key].write_bytes(b)
            n += len(lines)
    return {"slug": target, "from": slug, "thot_cid": row["thot_cid"], "generation": row["generation"], "verified": True, "private_lines": n}
