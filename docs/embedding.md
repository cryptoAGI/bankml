# Embedding in bankml and Savante

## Introduction

A language model answers; an **embedding model** measures meaning. It turns a piece of text into a list of numbers
(a vector) so that texts which mean similar things get vectors that point in similar directions. Search by
embedding finds what a question is *about*, not only the words it shares.

mindX, the system Savante belongs to, keeps its memories this way. Its memory store (`agents/memory_pgvector.py`)
embeds every memory and document with **bge-m3** into 1,024 numbers and stores them in PostgreSQL with pgvector.
bankml now uses the same model, so Savante's local history and mindX's memory can be searched by meaning in the same
way.

The embedding model is **optional**. Everything in bankml works without it. With it, the `.history` ragebar also
understands paraphrase, and a published agent's history becomes searchable by meaning in PostgreSQL.

**bankML does not compute embeddings itself yet.** bge-m3 is an XLM-R encoder, and bankML has no encoder graph:
`bankml serve --native` refuses `/api/embed` with HTTP 400 and the reason. The encoder graph, with `/api/embed`,
`/v1/embeddings` and an oracle against llama.cpp's bge-m3 output, is phase O7 of [OLLAMA.md](OLLAMA.md), planned for
0.6.0 ([TODO.md](TODO.md#060--more-models)). Until then Savante asks the local Ollama, as described below.

## Summary

| | |
|---|---|
| model | **bge-m3** (BAAI), the model mindX uses by default (`MINDX_EMBED_MODEL`, default `bge-m3`) |
| licence | **MIT** (read from the model's licence layer in the local Ollama store) |
| weights | 1.16 GB GGUF, served by the local **Ollama** (`ollama pull bge-m3`) |
| pinned by | the sha256 of that GGUF (Ollama's layer digest `daec91ff…3062c` on this laptop), recorded with every vector |
| output | 1,024 dimensions, normalized to unit length; the width of mindX's `VECTOR(1024)` and bankml's `bankml_exchanges.embedding` |
| input | each text cut at 4,000 characters, as mindX cuts it (≈ 1,000 tokens; bge-m3 reads up to 8,192) |
| used for | 1. the `.history` ragebar: words (BM25) and meaning (bge-m3) ranked together; 2. `embedding` in PostgreSQL publishing |
| without it | BM25 alone, and an empty `embedding` column; nothing fails |
| privacy | vectors of private text stay beside `.history` (`<history>.emb`) and leave the machine only if you publish private lines |

## Explanation

### Why an embedding model at all

The ragebar has always searched `.history` with BM25, which ranks exchanges by the words they share with the query,
weighted by how rare each word is. It is fast, exact and explainable, and it is blind to paraphrase: "how fast is
the ternary model" does not match an answer that says "Q2_0 decode took 2.4 s per token".

An embedding model closes that gap. bge-m3 maps the query and every exchange into the same 1,024-dimensional space,
where the angle between two vectors reflects how close their meanings are. bankml does not replace BM25 with it. It
**fuses** the two rankings, so an exact word match still counts and a match in meaning is added to it.

### Why bge-m3

- **It is what mindX uses.** Savante is one of mindX's offices. Using the same model and width means her vectors and
  mindX's live in the same space: a history published from bankml can sit in the same database and be compared with
  mindX's memories without re-embedding either.
- **It is open.** MIT-licensed, like the rest of what bankml admits ("open source or go away").
- **It reads long text.** Its 8,192-token window holds a whole question and answer; mindX cuts at 4,000 characters
  and bankml does the same, so the same text yields the same vector.
- **It is already on this computer.** The local Ollama holds it; bankml adds no new runtime and downloads nothing.

### How it stays out of the way

This laptop has about 1 GB of free memory while the chat model runs, and loading bge-m3 needs about 1.2 GB. So:

- bankml asks for an embedding only when the ragebar or a publish needs one, and Ollama unloads the model 60 seconds
  after the last use (`keep_alive`), giving the memory back;
- before loading it, bankml checks free memory (1.3 GB by default) and, if there is not enough, uses BM25 alone
  rather than push the machine into swap;
- one embedding call runs at a time; a keystroke that finds it busy is answered by BM25, and the query's vector is
  cached so the next keystroke with the same text is instant;
- exchanges are embedded once, in the background, and cached by the sha256 of the embedded text. The ragebar never
  waits for indexing; it uses what is ready and says how much is.

## Technical

### The model and its identity

bge-m3 is served by Ollama (0.13.3 here) from its store (`/usr/share/ollama/.ollama/models`). bankml reads the
model's **local manifest** and takes two things from it: the model layer's digest, which is the sha256 of the GGUF
Ollama loads, and the licence layer, which it classifies (MIT). It never asks the registry: what is identified is
what is on disk. Every cached vector records that digest, and a cache written by different weights is ignored.

### Calling it

`POST http://127.0.0.1:11434/api/embed` with `{"model": "bge-m3", "input": [...], "keep_alive": "60s", "truncate":
true}`, the endpoint mindX uses (`/api/embed`, Ollama ≥ 0.3.4). The reply's vectors must be 1,024 long, or the call
is refused. bankml divides each by its length so that the cosine of two vectors is their dot product.

### The private cache

`<history>.emb` beside `.history` (for Savante, `~/.local/share/bankml/savante/savante.history.emb`), one JSON line per
exchange:

```json
{"text_sha256": "…", "model": "bge-m3", "digest": "daec91ff…", "dims": 1024, "vec": "<1,024 float32, little-endian, base64>"}
```

The key is the sha256 of the exact text embedded: `user + "\n" + assistant`, cut at 4,000 characters. An edited or
new exchange gets a new key; nothing is ever re-embedded twice. The file is derived data: delete it and it is
rebuilt.

### Ranking: reciprocal rank fusion

For a query, bankml ranks exchanges twice: by BM25 (mindX's `rage.py` when present, otherwise the built-in BM25),
and by cosine similarity to the query's vector. The rankings are fused by **reciprocal rank fusion** (Cormack, Clarke
and Büttcher 2009): each exchange scores Σ 1 / (60 + rank) over the rankings it appears in. RRF needs no tuning and
no score calibration between the two systems; an exchange near the top of either list rises, one near the top of both
rises most. The ragebar's status line names the engines and how many exchanges are embedded.

### PostgreSQL

When an agent is published **with its private lines** (`connectors.publish(…, include_private=True)`), and the
database has pgvector, every exchange's vector is written to `bankml_exchanges.embedding` (`vector(1024)`) in the
same transaction as the lines themselves, and indexed by pgvectorscale's DiskANN when it is installed, else HNSW. A
query can then find an agent's exchanges by meaning in SQL:

```sql
SELECT seq, line FROM bankml_exchanges WHERE agent = 'ada'
ORDER BY embedding <=> '[…1,024 numbers…]'::vector LIMIT 5;
```

Public-only publishing sends no lines and so no vectors: a vector of private text is itself private.

### Code

| file | what |
|---|---|
| `sAGI/embed.py` | status and provenance, `embed()`, the cache (`index`, `index_async`, `cached`), `query_vector`, `fuse` |
| `sAGI/savante.py` | `history_search()` fuses BM25 and bge-m3; `_semantic()` indexes in the background |
| `sAGI/connectors.py` | `publish()` writes `embedding` with the lines, in one transaction |
| `testing/test_ui.py` | fusion, the cache, and the fallback to BM25 (offline: a fake Ollama) |

## Measured on this laptop (2026-09-29, Ryzen 3 3200U, Ollama 0.13.3)

| | |
|---|---|
| first call (load + embed one text) | 9.2 s |
| next call (three texts, model loaded) | 1.40 s |
| memory while loaded | 1.14 GB; free memory fell from 1.82 GB to 0.43 GB, which is why the 1.3 GB guard exists |
| indexing the three real exchanges in `.history` | 11.5 s (once; cached afterwards) |
| meaning check (cosine to "How fast is the ternary model on this laptop?") | a fact about decode speed 0.448 · the oversight office's oath 0.299 |
| ragebar status | `RAGE (mindX rage.py, BM25) + bge-m3 (meaning, 3 of 3 embedded), fused by reciprocal rank` |

The earlier attempt the same day was refused by the guard (1.0 GB free), as designed; this one ran when other
applications had released memory.

## Usage

**Have it.** bge-m3 is in the local Ollama:

```sh
ollama pull bge-m3          # once; 1.16 GB. On this laptop it is already there.
ollama list | grep bge-m3
```

Or adopt it through the Models tab's **Ollama** section like any other model (it is an embedding model, so it is
not offered as a chat carrier).

**Use it.** Nothing to switch on. Open **.history** and type in the ragebar. The status line says, for example:

```
3 of 3 exchanges · RAGE-shaped BM25 (built in) + bge-m3 (meaning, 3 of 3 embedded), fused by reciprocal rank
```

The first search starts indexing the history in the background; results improve as it completes. If bge-m3 cannot
run (not pulled, Ollama down, not enough free memory), the status line names BM25 alone.

**Publish with it.** Agents tab → PostgreSQL → publish **with private lines**. The result reports
`embedded: N of M exchanges embedded with bge-m3 (daec91ff…)`.

**Settings** (environment):

| variable | default | meaning |
|---|---|---|
| `BANKML_EMBED_MODEL` | `bge-m3` | the Ollama model (must produce 1,024 dimensions) |
| `BANKML_OLLAMA` | `http://127.0.0.1:11434` | the Ollama server (keep it local: history text is sent to it) |
| `BANKML_EMBED_KEEP_ALIVE` | `60s` | how long Ollama keeps it loaded after a call |
| `BANKML_EMBED_NEED_GB` | `1.3` | free memory required before loading it |

**Check it.** From the repository root, `python3 -B -c "import sys; sys.path.insert(0, 'sAGI'); import embed;
print(embed.status())"` prints the model, its digest and licence, and whether it is ready.
