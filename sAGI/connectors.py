#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Connectors for bankml agents: PostgreSQL (pgvector / pgvectorscale) — publish an agent, load it back, verified.

Through the `psql` client and the standard library (no driver). Values never enter SQL text: they travel as COPY
data (text format, escaped) into a temporary table, and the statements read them from there.

Publish writes the agent's public parts — persona, prompt, card, ledger, THOT manifest — and, only when asked
(`include_private=True`), its .history and .memory lines. Without that, the database holds the commitments (the THOT
manifest's digests of history and memory) and not the content: proof of data without the data.

Load rebuilds the agent from the row and refuses it unless every public file re-hashes to the digest the stored
THOT manifest commits to (and the manifest's own identity recomputes).

  BANKML_PG_DSN   libpq connection string (default "dbname=bankml": the local socket, the operator's role)
Schema: bankml_agents, bankml_exchanges and bankml_memory (each with embedding vector(1024) — bge-m3's width, as mindX
uses — when the vector extension is present, DiskANN index when pgvectorscale is, HNSW otherwise; and with embed_model
and content_hash, sha256(model ‖ digest ‖ dims ‖ text), so a vector is tied to the exact weights that made it),
bankml_memory_versions (.memory's memgit commits).

Publishing is incremental: the stored lines are read first; only lines that changed are written, a line keeps its
vector while its text is unchanged, and a vector is sent only where a row's content_hash is missing or different —
copied from the embedding cache beside the file (embed.py), so publishing twice embeds nothing the second time.
A browseML gitmind bundle (EXPORT BUNDLE) can be ingested into an agent's .memory (ingest_bundle), notes deduplicated.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import agents
import memgit
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


def psql(script: str, rows: list | None = None, dsn: str | None = None, atomic: bool = False) -> str:
    """Run a script; `rows` ([(k, v), …]) are COPY-ed into temp table _in(k text, v text) first. Returns stdout.
    atomic: the whole script, COPY included, is one transaction (psql --single-transaction): all of it or none."""
    head = ""
    if rows is not None:
        head = "CREATE TEMP TABLE _in(k text, v text);\nCOPY _in FROM STDIN;\n" + "".join(f"{_esc(k)}\t{_esc(v)}\n" for k, v in rows) + "\\.\n"
    p = subprocess.run(["psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1"] + (["--single-transaction"] if atomic else []) + ["-d", dsn or DSN, "-f", "-"],
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
    psql("""
CREATE TABLE IF NOT EXISTS bankml_memory_versions (agent text NOT NULL REFERENCES bankml_agents(slug) ON DELETE CASCADE,
  commit text NOT NULL, at text NOT NULL, subject text NOT NULL, PRIMARY KEY (agent, commit));
ALTER TABLE bankml_exchanges ADD COLUMN IF NOT EXISTS embed_model text, ADD COLUMN IF NOT EXISTS content_hash text;
ALTER TABLE bankml_memory ADD COLUMN IF NOT EXISTS embed_model text, ADD COLUMN IF NOT EXISTS content_hash text;
""", dsn=dsn)
    if st.get("vector"):
        for t in ("bankml_exchanges", "bankml_memory"):
            psql(f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS embedding vector({EMBED_DIM});", dsn=dsn)
            how = "diskann" if st.get("vectorscale") else "hnsw"
            psql(f"CREATE INDEX IF NOT EXISTS {t}_emb ON {t} USING {how} (embedding vector_cosine_ops);", dsn=dsn)
    return {**status(dsn), "schema": "ready", "index": "diskann (pgvectorscale)" if st.get("vectorscale") else ("hnsw (pgvector)" if st.get("vector") else "none")}


def _lines(p):
    try:
        return [l for l in p.read_bytes().split(b"\n") if l.strip()]
    except OSError:
        return []


def publish(slug: str, include_private: bool = False, dsn: str | None = None) -> dict:
    """Upsert the agent (and, if asked, its history and memory lines). The THOT manifest is rebuilt first."""
    sch = ensure_schema(dsn)
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
    hist = _lines(f["history"]) if include_private else []
    mem = _lines(f["memory"]) if include_private else []
    # k = "h:<position>" / "m:<position>"; v = the line's exact text (the bytes its sha256 and the Merkle proofs are over).
    # Only what changed is sent: the stored lines are read first (seq -> line sha256, content_hash).
    stored = _stored(slug, sch, dsn) if include_private else {}
    plan = {"h": (hist, f["history"], "exchange_text"), "m": (mem, f["memory"], "note_text")}
    stats = {"written": 0, "kept": 0, "embedded": 0, "vectors_sent": 0}
    emb_note = "no private lines published"
    vec_on = include_private and sch.get("vector")
    if vec_on:
        import embed
    for p, (lines, path, how) in plan.items():
        have = stored.get(p, {})
        recs = []
        for i, l in enumerate(lines):
            sha = hashlib.sha256(l).hexdigest()
            text = l.decode("utf-8")
            try:
                recs.append(json.loads(text))
            except ValueError:  # not a record: it gets no vector, and the ::jsonb cast refuses the whole publish, as before
                recs.append({})
            if have.get(i, (None, None))[0] == sha:
                stats["kept"] += 1
            else:
                rows.append((f"{p}:{i}", text)); stats["written"] += 1
        if not vec_on or not lines:
            continue
        try:  # vectors: computed only for texts the cache lacks, sent only where the row's content_hash differs
            text_of = getattr(embed, how)
            r = embed.index(path, recs, text_of=text_of)
            stats["embedded"] += r["embedded"]
            cache = embed.cached(path)
            for i, rec in enumerate(recs):
                t = text_of(rec)
                if not t.strip() or (rec.get("source") or {}).get("kind") == "delivery":
                    continue
                ch = embed.content_hash(t)
                if have.get(i, (None, None))[1] == ch and have.get(i, (None, None))[0] == hashlib.sha256(lines[i]).hexdigest():
                    continue
                v = cache.get(hashlib.sha256(t.encode()).hexdigest())
                if v is not None:
                    rows.append((f"{p}e:{i}", "[" + ",".join(f"{x:.7g}" for x in v) + "]")); rows.append((f"{p}c:{i}", ch))
                    stats["vectors_sent"] += 1
            emb_note = f"{stats['vectors_sent']} vectors sent, {stats['embedded']} computed, with {embed.MODEL} ({embed.provenance().get('digest', '')[:12]}…)"
        except (embed.Unavailable, ValueError) as e:
            emb_note = f"not embedded: {e}"
    if include_private and not sch.get("vector") and (hist or mem):
        emb_note = "not embedded: the database has no pgvector"
    model = None
    if vec_on:
        model = embed.MODEL
        rows.append(("model", model))
    vec_sql = lambda table, p: (f"""
UPDATE {table} x SET embedding = i.v::vector, embed_model = (SELECT v FROM _in WHERE k='model'),
  content_hash = (SELECT v FROM _in c WHERE c.k = '{p}c:' || substr(i.k, 4))
 FROM _in i WHERE i.k LIKE '{p}e:%' AND x.agent = (SELECT v FROM _in WHERE k='slug') AND x.seq = substr(i.k, 4)::int;
""" if vec_on else "")  # noqa: E731
    lines_sql = "".join(f"""
DELETE FROM {table} WHERE agent = {get('slug')} AND seq >= {len(lines)};
INSERT INTO {table} (agent, seq, line_sha256, line, rec)
SELECT {get('slug')}, substr(k, 3)::int, encode(sha256(convert_to(v, 'UTF8')), 'hex'), v, v::jsonb FROM _in WHERE k LIKE '{p}:%'
ON CONFLICT (agent, seq) DO UPDATE SET line_sha256 = EXCLUDED.line_sha256, line = EXCLUDED.line, rec = EXCLUDED.rec,
  {"embedding = NULL, " if sch.get("vector") else ""}embed_model = NULL, content_hash = NULL;
{vec_sql(table, p)}""" for table, p, lines in (("bankml_exchanges", "h", hist), ("bankml_memory", "m", mem)))
    # .memory's versions (memgit): each commit once
    vers = memgit.versions(f["memory"], n=100_000) if include_private else []
    for j, v in enumerate(vers):
        rows += [(f"v:{j}:c", v["sha"]), (f"v:{j}:a", v["at"]), (f"v:{j}:s", v["subject"])]
    if vers:
        lines_sql += """
INSERT INTO bankml_memory_versions (agent, commit, at, subject)
SELECT (SELECT v FROM _in WHERE k='slug'), c.v, a.v, s.v FROM _in c
  JOIN _in a ON a.k = replace(c.k, ':c', ':a') JOIN _in s ON s.k = replace(c.k, ':c', ':s')
 WHERE c.k LIKE 'v:%:c' ON CONFLICT DO NOTHING;
"""
    # one transaction: the agent row and its lines change together or not at all
    psql(f"""
INSERT INTO bankml_agents (slug, name, persona, persona_raw, prompt, card, commitments, manifest, thot_cid, content_root, generation, private_included)
VALUES ({get('slug')}, {get('name')}, {get('persona')}::jsonb, {get('persona')}, {get('prompt')}, {get('card')}::jsonb, {get('commitments')}::jsonb,
        {get('manifest')}::jsonb, {get('thot_cid')}, {get('content_root')}, {get('generation')}::int, {get('private')}::boolean)
ON CONFLICT (slug) DO UPDATE SET name=EXCLUDED.name, persona=EXCLUDED.persona, persona_raw=EXCLUDED.persona_raw, prompt=EXCLUDED.prompt, card=EXCLUDED.card,
  commitments=EXCLUDED.commitments, manifest=EXCLUDED.manifest, thot_cid=EXCLUDED.thot_cid, content_root=EXCLUDED.content_root,
  generation=EXCLUDED.generation, private_included=EXCLUDED.private_included, published_at=now();
{lines_sql}""", rows, dsn, atomic=True)
    return {"slug": slug, "thot_cid": m["identity"]["cid"], "content_root": m["identity"]["contentRoot"], "generation": m["bundle"]["generation"],
            "private_included": include_private, "exchanges": len(hist), "memory": len(mem), "versions": len(vers), **stats, "embedding": emb_note}


def _stored(slug: str, sch: dict, dsn: str | None) -> dict:
    """What the database holds for this agent: {"h"|"m": {seq: (sha256 of the stored line, content_hash)}} — hashed from
    the line itself, so a line changed in the database is rewritten by the next publish."""
    out = psql("SELECT json_build_object('h', (SELECT coalesce(json_agg(json_build_array(seq, encode(sha256(convert_to(line, 'UTF8')), 'hex'), content_hash)), '[]') FROM bankml_exchanges WHERE agent = (SELECT v FROM _in WHERE k='slug')),"
               " 'm', (SELECT coalesce(json_agg(json_build_array(seq, encode(sha256(convert_to(line, 'UTF8')), 'hex'), content_hash)), '[]') FROM bankml_memory WHERE agent = (SELECT v FROM _in WHERE k='slug')));",
               [("slug", slug)], dsn)
    j = json.loads(out.strip() or "{}")
    return {p: {int(r[0]): (r[1], r[2]) for r in j.get(p, [])} for p in ("h", "m")}


def published(dsn: str | None = None) -> list:
    out = psql("SELECT coalesce(json_agg(json_build_object('slug', slug, 'name', name, 'generation', generation, 'thot_cid', thot_cid,"
               " 'private_included', private_included, 'published_at', published_at) ORDER BY published_at DESC), '[]') FROM bankml_agents;", dsn=dsn)
    return json.loads(out.strip() or "[]")


def load(slug: str, as_slug: str | None = None, dsn: str | None = None, expect_content_root: str | None = None) -> dict:
    """Rebuild a published agent locally — refused unless every public file matches the stored THOT manifest.
    The manifest comes from the same row, so on its own that is self-consistency: whoever can write the database can
    write a consistent row. Pass `expect_content_root` (an iNFT's contentRoot, or one published elsewhere) to anchor
    the manifest outside the database; the result says which kind of check was made."""
    out = psql("SELECT row_to_json(a) FROM bankml_agents a WHERE slug = (SELECT v FROM _in WHERE k='slug');", [("slug", slug)], dsn).strip()
    if not out:
        raise KeyError(f"no published agent {slug!r}")
    row = json.loads(out)
    m = row["manifest"]
    findings = thot.check_structure(m)
    if expect_content_root and (m.get("identity") or {}).get("contentRoot", "").lower() != expect_content_root.lower():
        raise ValueError(f"refused: the stored manifest's contentRoot is not the expected {expect_content_root}")
    persona_b = row["persona_raw"].encode()  # the exact bytes; the jsonb copy is for queries
    want = {x["facet"]: x["sha256"] for x in m["facets"]}
    got = {"persona": hashlib.sha256(persona_b).hexdigest(), "prompt": hashlib.sha256(row["prompt"].encode()).hexdigest()}
    findings += [f"{k}: stored bytes do not match the manifest" for k in got if got[k] != want.get(k)]
    if findings:
        raise ValueError(f"refused: {findings}")
    # every private blob fetched and checked BEFORE anything is written: a refusal leaves the target untouched
    private = {}
    if row.get("private_included"):
        for table, key, facet in (("bankml_exchanges", "history", "x-bankml.history"), ("bankml_memory", "memory", "x-bankml.memory")):
            lines = json.loads(psql(f"SELECT coalesce(json_agg(line ORDER BY seq), '[]') FROM {table} WHERE agent = (SELECT v FROM _in WHERE k='slug');",
                                    [("slug", slug)], dsn).strip())
            b = "".join(l + "\n" for l in lines).encode()
            if hashlib.sha256(b).hexdigest() != want.get(facet):
                raise ValueError(f"refused: the stored {key} does not match the manifest's {facet} digest")
            private[key] = (b, len(lines))
    target = as_slug or slug
    d = agents.agent_dir(target)
    d.mkdir(parents=True, exist_ok=True)
    f = agents.files(target)
    f["persona"].write_bytes(persona_b)
    f["prompt"].write_text(row["prompt"], encoding="utf-8")
    (d / f"{target}.thot.json").write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    agents.rebind(target, derived_from=row["commitments"].get("derived_from"))
    n = 0
    for key, (b, k) in private.items():
        f[key].write_bytes(b)
        n += k
    return {"slug": target, "from": slug, "thot_cid": row["thot_cid"], "generation": row["generation"], "verified": True, "private_lines": n,
            "anchored": bool(expect_content_root), "check": "against the expected contentRoot" if expect_content_root else
            "self-consistency (files against the manifest stored beside them)"}


def recall(slug: str, vector: list, k: int = 4, table: str = "bankml_memory", dsn: str | None = None) -> list:
    """The `k` rows of an agent nearest `vector` by meaning (cosine, through the DiskANN or HNSW index):
    [{seq, text, similarity, model}]. Only rows embedded with the same model's identity are comparable; the caller passes
    a vector from the model the rows were made with (embed.query_vector)."""
    if table not in ("bankml_memory", "bankml_exchanges"):
        raise ValueError("table: bankml_memory or bankml_exchanges")
    vec = "[" + ",".join(f"{x:.7g}" for x in vector) + "]"
    out = psql(f"""SELECT coalesce(json_agg(r), '[]') FROM (
  SELECT seq, coalesce(rec->>'text', concat_ws(E'\\n', rec->>'user', rec->>'assistant')) AS text,
         round((1 - (embedding <=> (SELECT v FROM _in WHERE k='vec')::vector))::numeric, 4) AS similarity, embed_model AS model
  FROM {table} WHERE agent = (SELECT v FROM _in WHERE k='slug') AND embedding IS NOT NULL
  ORDER BY embedding <=> (SELECT v FROM _in WHERE k='vec')::vector LIMIT {int(k)}) r;""", [("slug", slug), ("vec", vec)], dsn)
    return json.loads(out.strip() or "[]")
