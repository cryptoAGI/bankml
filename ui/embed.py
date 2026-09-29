# SPDX-License-Identifier: MIT OR Apache-2.0
"""Embeddings for Savante: the model mindX uses (bge-m3, 1024 dimensions), served by the local Ollama, the same way
mindX's memory_pgvector.py calls it (/api/embed, inputs cut at 4,000 characters).

What it is for:
  - the .history ragebar: BM25 (words) and bge-m3 (meaning) ranked together by reciprocal rank fusion, so "how fast is
    the ternary model" finds an exchange that says "Q2_0 decode took 2.4 s per token";
  - PostgreSQL publishing: bankml_exchanges.embedding (vector(1024)) is filled, so a published agent's history can be
    searched by meaning in the database, as mindX's own memories are.

What it is not: required. Without Ollama, without bge-m3, or without the memory to load it, search is BM25 alone and
publishing leaves the column empty. Nothing is downloaded by this module; `ollama pull bge-m3` (or the Models tab)
brings the model in.

Privacy: vectors are derived from private text, so they stay beside the .history they come from (<history>.emb,
one JSON line per exchange, keyed by the sha256 of the embedded text) and leave the machine only if the operator
publishes private lines. Provenance: every vector records the model's digest (the sha256 of the GGUF Ollama loaded,
read from its local manifest), so a vector can be traced to the exact weights that made it."""
import base64, hashlib, json, math, os, re, struct, threading, urllib.request
from pathlib import Path

MODEL = os.environ.get("BANKML_EMBED_MODEL", "bge-m3")
OLLAMA = os.environ.get("BANKML_OLLAMA", "http://127.0.0.1:11434").rstrip("/")
DIMS = 1024           # bge-m3; the width of bankml_exchanges.embedding and of mindX's VECTOR(1024)
MAX_CHARS = 4000      # mindX's cut for bge-m3 (≈ 1,000 tokens, well inside its 8,192-token window)
KEEP_ALIVE = os.environ.get("BANKML_EMBED_KEEP_ALIVE", "60s")  # Ollama unloads it after this: the laptop needs the memory back
NEED_FREE = int(float(os.environ.get("BANKML_EMBED_NEED_GB", "1.3")) * 1e9)  # to load bge-m3 (1.16 GB of weights) safely
RRF_K = 60            # reciprocal rank fusion constant (Cormack, Clarke and Büttcher 2009)

_LOCK = threading.Lock()   # one embedding call at a time; a search that finds it busy uses BM25 alone
_QCACHE: dict = {}         # query text -> vector (the ragebar searches on every keystroke)


class Unavailable(RuntimeError):
    pass


def _post(path: str, body: dict, timeout=120) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path: str, timeout=3) -> dict:
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as r:
        return json.loads(r.read())


def _mem_available() -> int:
    try:
        return int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) * 1024
    except (OSError, AttributeError):
        return 0


def provenance() -> dict:
    """{model, digest, bytes, licence, open, dims} from the local Ollama's own manifest; {} if it is not here."""
    try:
        import models
        s = models.ollama_local_spec(MODEL if ":" in MODEL else f"{MODEL}:latest")
    except Exception:  # noqa: BLE001 — not pulled, no Ollama store, unreadable
        return {}
    return {"model": MODEL, "digest": s["sha256"], "bytes": s["bytes"], "licence": s["licence"], "open": s["open"], "dims": DIMS}


def status() -> dict:
    """What the UI shows: ready or not, and why."""
    pv = provenance()
    if not pv:
        return {"ready": False, "why": f"{MODEL} is not in the local Ollama (ollama pull {MODEL})"}
    if not pv["open"]:
        return {"ready": False, "why": f"{MODEL}'s licence ({pv['licence']}) is not open source", **pv}
    try:
        loaded = any(m.get("name", "").split(":")[0] == MODEL.split(":")[0] for m in _get("/api/ps").get("models") or [])
    except OSError as e:
        return {"ready": False, "why": f"Ollama is not answering at {OLLAMA} ({e})", **pv}
    return {"ready": True, "loaded": loaded, "why": "ready", **pv}


def embed(texts: list, block: bool = True) -> list:
    """Unit-length 1024-d vectors for `texts` (each cut at MAX_CHARS). Raises Unavailable when it cannot run safely."""
    st = status()
    if not st["ready"]:
        raise Unavailable(st["why"])
    if not st.get("loaded") and _mem_available() < NEED_FREE:
        raise Unavailable(f"{_mem_available() / 1e9:.1f} GB free; loading {MODEL} needs {NEED_FREE / 1e9:.1f} GB")
    if not _LOCK.acquire(blocking=block):
        raise Unavailable("an embedding is already running")
    try:
        r = _post("/api/embed", {"model": MODEL, "input": [t[:MAX_CHARS] for t in texts], "keep_alive": KEEP_ALIVE, "truncate": True})
    except OSError as e:
        raise Unavailable(f"Ollama /api/embed: {e}") from e
    finally:
        _LOCK.release()
    out = []
    for v in r.get("embeddings") or []:
        if len(v) != DIMS:
            raise Unavailable(f"{MODEL} returned {len(v)} dimensions, not {DIMS}")
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    if len(out) != len(texts):
        raise Unavailable(f"{MODEL} returned {len(out)} vectors for {len(texts)} texts")
    return out


# ── the private cache beside .history ────────────────────────────────────────────────────────────────────────
def exchange_text(r: dict) -> str:
    """What is embedded for one exchange: the question and the answer, as the ragebar's BM25 indexes them."""
    return f"{r.get('user', '')}\n{r.get('assistant', '')}"[:MAX_CHARS]


def _pack(v: list) -> str:
    return base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode()


def _unpack(s: str) -> list:
    b = base64.b64decode(s)
    return list(struct.unpack(f"<{len(b) // 4}f", b))


def cache_path(history: Path) -> Path:
    return history.with_name(history.name + ".emb")


def cached(history: Path) -> dict:
    """text sha256 -> vector, for the current model digest only (a changed model means a new index)."""
    pv = provenance()
    out = {}
    try:
        lines = cache_path(history).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for l in lines:
        try:
            j = json.loads(l)
        except ValueError:
            continue
        if j.get("digest") == pv.get("digest") and j.get("dims") == DIMS:
            out[j["text_sha256"]] = _unpack(j["vec"])
    return out


def index(history: Path, records: list, batch: int = 8) -> dict:
    """Embed every exchange not yet in the cache. Returns {embedded, cached, total}."""
    have = cached(history)
    todo = []
    for r in records:
        t = exchange_text(r)
        h = hashlib.sha256(t.encode()).hexdigest()
        if h not in have and h not in {x[0] for x in todo}:
            todo.append((h, t))
    pv = provenance()
    done = 0
    for k in range(0, len(todo), batch):
        part = todo[k:k + batch]
        vecs = embed([t for _, t in part])
        with open(cache_path(history), "a", encoding="utf-8") as f:
            for (h, _), v in zip(part, vecs):
                f.write(json.dumps({"text_sha256": h, "model": MODEL, "digest": pv["digest"], "dims": DIMS, "vec": _pack(v)}) + "\n")
        done += len(part)
    return {"embedded": done, "cached": len(have), "total": len(records)}


INDEXING = {"running": False, "error": None}


def index_async(history: Path, records: list) -> None:
    """index() in the background, once at a time (the ragebar never waits for it)."""
    if INDEXING["running"]:
        return

    def run():
        INDEXING.update(running=True, error=None)
        try:
            index(history, records)
        except Unavailable as e:
            INDEXING["error"] = str(e)
        finally:
            INDEXING["running"] = False
    threading.Thread(target=run, daemon=True).start()


def query_vector(q: str):
    """The query's vector, cached; None if the model cannot answer right now (busy, absent, no memory)."""
    q = q.strip()[:MAX_CHARS]
    if q in _QCACHE:
        return _QCACHE[q]
    try:
        v = embed([q], block=False)[0]
    except Unavailable:
        return None
    if len(_QCACHE) > 256:
        _QCACHE.clear()
    _QCACHE[q] = v
    return v


def cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))  # both unit length


def fuse(bm25: list, semantic: list, k: int = RRF_K) -> list:
    """Reciprocal rank fusion of two rankings of record indices: score = Σ 1 / (k + rank)."""
    sc: dict = {}
    for ranking in (bm25, semantic):
        for rank, i in enumerate(ranking, 1):
            sc[i] = sc.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(sc.items(), key=lambda x: -x[1])
