# Using bankml and Savante

How to install bankml, start it, and use it on your own computer. Everything runs on this machine: the model, the
gate that verifies it, and Savante's page. New to Savante's page and her voice? Read **[playback.md](playback.md)**.

- [1. Install](#1-install)
- [2. What you need](#2-what-you-need)
- [3. Start, stop, check](#3-start-stop-check)
- [4. Talk to Savante](#4-talk-to-savante)
- [5. Models](#5-models)
- [6. By hand: build, verify, serve](#6-by-hand-build-verify-serve)
- [7. Let others watch (view mode, on the LAN)](#7-let-others-watch-view-mode-on-the-lan)
- [8. The files Savante keeps: `.history`, `.memory`, `.prompt`](#8-the-files-savante-keeps-history-memory-prompt)
- [8a. Proof of data without the data](#8a-proof-of-data-without-the-data)
- [8b. Custom agents from the Savante template](#8b-custom-agents-from-the-savante-template)
- [8c. THOT bundles: the dataset an iNFT points to](#8c-thot-bundles-the-dataset-an-inft-points-to)
- [8d. PostgreSQL: publish an agent, load one back](#8d-postgresql-publish-an-agent-load-one-back)
- [8e. iNFT: mint an agent, load one from a token](#8e-inft-mint-an-agent-load-one-from-a-token)
- [9. Receipts, and how to check an answer](#9-receipts-and-how-to-check-an-answer)
- [10. Savante's canon and the iNFT ledger](#10-savantes-canon-and-the-inft-ledger)
- [11. Testing and the release gate](#11-testing-and-the-release-gate)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Reference: commands, ports, environment](#13-reference-commands-ports-environment)

---

## 1. Install

One command, from a fresh machine to Savante answering on your own CPU:

```sh
git clone https://github.com/cryptoAGI/bankml && cd bankml
./install.sh
```

or, without cloning first (the installer clones bankml to `~/bankml`, or to `$BANKML_DIR`):

```sh
curl -fsSL https://raw.githubusercontent.com/cryptoAGI/bankml/main/install.sh | bash
```

It runs these steps in order, and stops at the first one that fails, saying why:

| step | what it does |
|---|---|
| `check` | Rust 1.95+, Python 3.10+, git, curl, tar, ss, sha256sum; reports AVX2, RAM and free disk |
| `build` | `cargo build --release` and `cargo test --release` |
| `engine` | downloads llama.cpp b11192 (17 MB) and refuses it unless its sha256 is the published one |
| `python` | uses your Python if it has Gradio and numpy; otherwise makes a venv with Gradio 3 |
| `canon` | clones Savante's canon to `~/cryptoAGI/savante` (beside jaimla and luvai) if it is not there; an existing canon is left as it is, and a machine that still has the older `~/savante` keeps using it |
| `model` | imports Bonsai-8B (1.16 GB) if absent, verifies it, and starts `bankml serve` with llama-server behind it |
| `start` | starts Savante on `http://127.0.0.1:7873` |

Run any steps on their own: `./install.sh build`, `./install.sh model start`. Options: `--skip-tests`, `--no-start`,
`--view` (also start view mode, §7), `--voice` (also install Savante's voice; see [playback.md](playback.md)).
Nothing needs sudo. What the installer chose is kept in `~/.local/share/bankml/install.env`, and logs in
`~/.local/share/bankml/logs/`.

## 2. What you need

| what | where it comes from |
|---|---|
| Linux on x86_64 (AVX2 for the fast kernels; without it bankml is correct, just slower) | — |
| Rust 1.95 or newer | [rustup](https://rustup.rs) |
| Python 3.10+ | your system |
| about 2 GB of free RAM and 3 GB of free disk | the 1-bit 8B model maps 1.16 GB |

The installer fetches the rest: llama.cpp b11192 (`llama-b11192-bin-ubuntu-x64.tar.gz`, sha256
`34cf6fa5de9da0db3932c78fe15fed2fbca17451e665dac0a4f6a3c8fc881ec7`), Gradio 3 if needed, the model from
[PYTHAI/Bonsai-8B-gguf-fork](https://huggingface.co/PYTHAI/Bonsai-8B-gguf-fork), and Savante's canon (§10). On
another platform, build llama.cpp b11192 yourself and set `BANKML_LLAMA_SERVER` to its `llama-server`.

The ternary model (2.31 GB) also works and gives better answers, but llama.cpp runs it about 5× slower on a CPU (see
[PERFORMANCE.md](PERFORMANCE.md)).

## 3. Start, stop, check

```sh
./install.sh start           # bankml serve (if not up) is started by `model`; this starts Savante's page
./install.sh start --view    # also the read-only page for your network
./install.sh status          # what is running, and which model bankml verified
./install.sh stop            # stops the page(s), bankml serve and llama-server
```

What runs where:

```
 you (browser) ──► Savante's page (127.0.0.1:7873) ──► bankml serve (127.0.0.1:18093) ──► llama-server (127.0.0.1:18092)
                                                        verifies the model file            runs the verified file
 anyone on the LAN ──► view mode (0.0.0.0:7874, read-only)
```

- **bankml serve** is the gate. It refuses to start unless the model passes the guard, its sha256 equals the pin in
  the model's `FORK.json`, and llama-server serves that same file. Every answer carries a receipt (§9).
- **Savante's page** is reachable only from this computer.
- **View mode** is a read-only page for others on your network (§7).

## 4. Talk to Savante

Open **http://127.0.0.1:7873**, type a question, press **Send**.

- The first answer takes about **2 minutes** on a laptop CPU, while the model reads Savante's instructions. Later
  answers start sooner.
- Every answer is labelled **draft · not a finding**, and carries a receipt (§9).
- To ask for a review, say so: *"review: is Bonsai-8B ready to serve mindX?"*. She then answers as FINDINGS /
  VERDICT / RATIONALE / CONDITIONS / RISKS WATCHED. The 1-bit carrier was graded REJECT for review duty in the sAGI
  carrier test, so treat its verdicts as drafts.
- **Stop** cancels an answer; **New session** starts a fresh conversation.

The side panel sets the system prompt (`.prompt`, §8), max tokens, temperature, and the engine's CPU threads and RAM
budget (**Apply** restarts the engine with them, and restores your previous settings if it will not start). The tabs
hold your history, notes, metrics, integrity checks and the verifier. [playback.md](playback.md) walks through the
page, her card and her voice.

## 5. Models

The **Models** tab and `sAGI/models.py` import a model, verify it, and switch to it:

```sh
python3 sAGI/models.py list                 # what is here, pinned or not, and which one is running
python3 sAGI/models.py catalog              # the curated catalogue, and whether each fits this machine
python3 sAGI/models.py import qwen3-1.7b    # a catalogue id, a Hugging Face URL, or ollama:NAME:TAG
python3 sAGI/models.py use Qwen3-1.7B-Q8_0.gguf   # switch (rolls back if the new model fails)
```

An import is kept only if its sha256 equals the one its publisher lists, `bankml guard` passes, and a `FORK.json`
pin is written; `bankml serve` checks that pin before every start. Only open-source models are accepted (Gemma and
Llama are refused), and an import that would not fit this machine's RAM or disk is refused with the numbers.

## 6. By hand: build, verify, serve

What the installer does, step by step:

```sh
cargo build --release && cargo test --release
target/release/bankml guard  .models/Bonsai-8B-Q1_0.gguf                   # play | refuse (reason) | need_more
target/release/bankml verify .models/Bonsai-8B-Q1_0.gguf --fork FORK.json  # guard, then the sha256 pin
target/release/bankml serve  .models/Bonsai-8B-Q1_0.gguf --fork FORK.json \
    --spawn /path/to/llama-b11192/llama-server --threads 3 --ctx 2048
python3 sAGI/savante.py --mode interact
```

`verify` exits 0 on play, 2 on refuse, 1 on an I/O error, 3 on a truncated file. `serve` can also sit in front of a
llama-server you started (`--upstream 127.0.0.1:18092`); it then checks through `/props` that the server runs the
verified file. It listens on `127.0.0.1:18093` and speaks the OpenAI API:

```sh
curl -s 127.0.0.1:18093/bankml     # what was verified, when, and what is upstream
curl -s 127.0.0.1:18093/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello. /no_think"}],"max_tokens":32}'
```

## 7. Let others watch (view mode, on the LAN)

```sh
./install.sh start --view          # or: python3 sAGI/view.py --host 0.0.0.0 --port 7874
```

Others open `http://<this computer's LAN address>:7874` (`ip -4 addr` shows it). The page is read-only: the live
testing log, this machine's load, the verified model, the release records, CI, Savante's office and ledger, and her
recordings (see [playback.md](playback.md)). It never shows the content of `.history` or `.memory`, only their
commitments (§8a). It is the Python standard library, not Gradio, because the Gradio installed for the chat has
path-traversal bugs and must stay on this computer.

## 8. The files Savante keeps: `.history`, `.memory`, `.prompt`

Nothing is ever written into the canon (`~/cryptoAGI/savante`). What the UI writes lives in `BANKML_UI_STATE` (default
`~/.local/share/bankml/savante/`):

- **`savante.history`**: one JSON object per exchange (JSONL):

  ```json
  {"ts": 1790633549.123, "sent_at": "2026-09-28T15:12:29.123-0700", "first_token_s": 113.02,
   "response_s": 125.11, "answered_at": "2026-09-28T15:14:34.233-0700", "session": "e2e-test",
   "user": "In one sentence: …", "assistant": "Savante knows …", "assistant_raw": "Savante knows …", "shown": "… (with the footer)",
   "prompt": "persona · system_prompt (canon, ledgered)", "prompt_provenance": "persona.system_prompt · persona sha256 d96556b11989… (ledgered)",
   "receipt": {"bankml": "0.0.7", "model_sha256": "284a335a…", "prompt_tokens": 316, "completion_tokens": 28,
               "ttft_ms": 113000, "wall_ms": 125100, "response_sha256": "80010408…"}}
  ```

  `sent_at` is the press of Send; `first_token_s` and `response_s` are measured from it. The receipt's
  `ttft_ms`/`wall_ms` are measured by bankml serve from when it forwarded the request. `assistant_raw` (0.1.7+) is
  the text exactly as the model wrote it, which is what the receipt hashes; `assistant` is the same text with any
  `<think>` block and surrounding whitespace removed. Each record is appended as one locked, fsynced write; a line
  damaged by a crash is skipped (and counted), never allowed to hide the rest. The history is plain text: back it
  up, grep it, or delete it.
- **`savante.memory`**: one JSON object per note: `{"ts", "at", "text", "sha256", "source": {"kind": "typed" |
  "response", "session", "sent_at", "response_sha256"}}`.
- **`Savante.prompt`**: the Space template's prompt, cached the first time it is chosen.

**The history the model sees.**
- **Without a context limit:** the last 12–17 exchanges, each cut to 4,000 characters. The window starts at 12
  exchanges and moves in steps of six, so between moves each prompt is the previous prompt plus one exchange, and
  the engine's prompt cache reuses all of it.
- **Within the engine's context.** The window must also fit the context the engine actually runs with (`n_ctx` from
  `/props`), counted by its own tokenizer.
  - The start stays where it was last turn while that still fits, so the cache is reused.
  - When it no longer fits, the start jumps so that the newest half of what fits is kept, leaving room for the next
    turns. When only two or three exchanges fit, all of them are kept.
  - The answer's footer then says so: "history: 4 of 13 exchanges fit the engine's 2048-token context — raise the
    RAM budget in Resources for more".
  - Exchanges older than the 12–17 window are dropped without a note, as they always were.
- **If the system prompt and the question alone do not fit,** the answer is refused with the numbers.
- On a 2048-token context the persona prompt plus `.memory` leaves room for only a few exchanges; 4096 tokens (about
  0.3 GB more RAM for the 8B model) gives the window room to stay put for several turns. **After an engine
restart**, the first question restores the saved KV of the system prompt (a 51 MB file per model, context and system
prompt, in `~/.local/share/bankml/savante/slots/`), so it skips the system prompt's prefill: on this laptop 132 s
became 15 s, with the same answer.
(A window that slid by one exchange per turn, as before 0.1.8, changed the text right after the system prompt and
forced the whole history to be re-read every turn; over 60 turns the stable window moves 8 times instead of 48.) The
chat's footer (clock, receipt) is never sent to the model.

## 8a. Proof of data without the data

`.history` and `.memory` stay on this computer. What can leave it is a **commitment**:

| part | what |
|---|---|
| leaf | sha256(0x00 ‖ one JSONL line, exactly as written, without its newline) |
| Merkle root | RFC 6962 / RFC 9162: node = sha256(0x01 ‖ left ‖ right), the tree split at the largest power of two below n (an odd node is promoted, never paired with itself). Scheme `rfc6962-sha256`; 0.1.6 and earlier used an unprefixed tree, so their roots differ |
| file sha256 / CIDv1 | of the whole file; CIDv1 raw (0x55), sha2-256, base32 lower: the construction Savante's iNFT ledger uses |

**🔏 proof** (Responses tab) produces, for one exchange, `{"commitment": …, "proof": {"scheme", "record", "records",
"leaf", "path": [hash, …], "merkle_root"}, "verifies": true}`. Give someone the exchange's line and the proof: they
verify it as RFC 9162 §2.1.3.2 describes (the index and the tree size decide at each step which side the sibling is
on) and compare the result with the root **and record count** you published; the pair is the commitment. That proves the exchange is in your history, unchanged,
and reveals nothing else. The same works for any data kept in a browser's local storage: keep the data, publish the
hash or CID. Metrics and the view page show the commitments; the view page never serves a line.

To check a proof by hand, in Python:

```python
import hashlib, json
line = b'...the exact JSONL line...'
p = json.load(open("proof.json"))["proof"]
h = hashlib.sha256(line).hexdigest()
for s in p["path"]:
    a, b = (h, s["hash"]) if s["side"] == "right" else (s["hash"], h)
    h = hashlib.sha256(bytes.fromhex(a) + bytes.fromhex(b)).hexdigest()
print(h == p["merkle_root"])
```

## 8b. Custom agents from the Savante template

**Agents** tab. Savante's canon is the template and is never written. **Derive a new agent** creates, in
`~/.local/share/bankml/agents/<name>/` (`BANKML_AGENTS`):

| file | what |
|---|---|
| `<name>.persona` | the persona (mindX `.persona v1` shape): a new name and identity, Savante's BDI/skills/safety as a starting point; the `token` bindings start empty (Savante's are never copied) |
| `<name>.prompt` | the system prompt, as text: the `.prompt` the chat uses |
| `<name>.agentcard.json` | EIP-721 metadata ∪ ERC-8004 registration-v1; status `not_yet_minted`; `derived_from` names Savante's persona sha256 and doctrine root |
| `<name>.commitments.json` | the ledger: sha256 + CIDv1 of each file, and the **doctrine root** (keccak256 over the same 15 JSON pointers as Savante's ledger) |
| `<name>.history`, `<name>.memory` | the agent's own conversations and notes |

**Use this agent** switches the chat, `.history`, `.memory`, Responses and Metrics to it. **Edit** (custom agents
only) opens the `.persona` as JSON and the `.prompt` as text. **Save and re-ledger** refuses a persona that breaks
the binder's preflight (non-ASCII keys, floats, integers beyond ±2⁵³), lacks a doctrine clause, or **changes one**:
the 15 doctrine clauses (name, source, format, system prompt, mantra, oath, beliefs, skills, safety, embodiment, tool
allowlist …) are fixed when the agent is derived, and their root is recorded then and carried forward. To change them,
derive a new agent. Everything else (the `.prompt` file, voice examples, description, the aivatar) stays editable.
Otherwise it regenerates the card and ledger. An agent whose files do not match its ledger refuses to speak.

The keccak256 is written in pure Python. It reproduces Savante's published doctrine root
(`0x92fe83eb…ae137d0`) from her persona, and it equals pycryptodome's keccak for every input length from 0 to 400
bytes (tested).

## 8c. THOT bundles: the dataset an iNFT points to

**Agents → THOT bundle** builds `<name>.thot.json` to `sagi.thot_manifest/1` (`~/sagi/engine/THOT_MANIFEST.md`):

- **Facets.** Core facets `persona` (the ternary head, leaf 0) and `prompt`, plus declared custom facets
  `x-bankml.agentcard`, `x-bankml.history` and `x-bankml.memory`. The history and memory are committed **by digest
  only**: the manifest holds no conversation text and can be published.
- **Bundle root.** `bundle_root` = keccak256 over `label ‖ 0x1f ‖ sha256_hex ‖ 0x1e`, core facets in registry order,
  then custom facets by name.
- **Merkle root.** A 64-leaf keccak256 tree, padded with `keccak256(b"")`, with unsorted pairs and no prefixes.
- **Identity.** `thot:<sha256>`, a CIDv1, `thot-<cid>`, and `contentRoot` (keccak256), all over the canonical
  manifest without its identity block.
- **Lineage.** Genesis is generation 1. Every facet change (a new chat, a new note, an edited prompt) makes the next
  build generation n+1, with `parent` = the previous manifest's CID. A re-bind with no change keeps the generation.
  `relations` records `derived_from` Savante's bundle.
- **Rung.** Each agent directory is its own git repository, so the rung's locator
  (`localhost/<user>/<name>@<commit>`) names real bytes. History and memory are git-ignored and listed as
  `locator_lacks`.

The builder is checked against the spec's own test vectors: Savante @1fcca89, Jaimla @8b57ccf and LuvAI @0c1eef7 give
the same bundle roots, Merkle roots and identity CIDs. Savante's current generation-9 manifest verifies with no findings.

## 8d. PostgreSQL: publish an agent, load one back

**Agents → PostgreSQL**. The connection is `BANKML_PG_DSN` (default `dbname=bankml`, the local socket, your role).
One-time setup, which needs sudo:

```sh
sudo -u postgres psql -c "CREATE ROLE $USER LOGIN CREATEDB" -c "CREATE DATABASE bankml OWNER $USER"
sudo -u postgres psql -d bankml -c "CREATE EXTENSION IF NOT EXISTS vector"
```

- **Publish the agent in use.** This rebuilds and verifies its THOT bundle, then upserts `bankml_agents`: persona
  (the exact bytes, plus a jsonb copy for queries), prompt, card, ledger, manifest, THOT CID, contentRoot and
  generation.
  - **Off by default:** `.history` and `.memory` lines go in (`bankml_exchanges`, `bankml_memory`, exact line text,
    sequence and sha256) only when *include .history and .memory lines* is ticked. Otherwise the database holds only
    their commitments, inside the manifest.
- **Load.** Rebuilds a published agent as a local one. It is **refused** unless the stored manifest's identity
  recomputes and every restored file (persona, prompt, and history and memory when included) re-hashes to the
  manifest's digest. A tampered row cannot be loaded.
- **Vectors.** `bankml_exchanges.embedding` is `vector(1024)` (bge-m3's width, as mindX uses), indexed with DiskANN
  when pgvectorscale is installed and HNSW (pgvector) otherwise. It is filled with bge-m3 vectors when private lines are
  published and bge-m3 can run (see [embedding.md](embedding.md)); otherwise it stays empty.
- **Safety.** Values travel to psql as COPY data into a temporary table; no value is ever part of SQL text.
  `testing/test_connectors.py` runs every step against a throwaway cluster (initdb in a temp dir, pgvector, its own
  socket), including a tampered prompt, a tampered history line and an SQL-injection string, all handled.

## 8e. iNFT: mint an agent, load one from a token

**Agents → iNFT**, for the agent in use. Savante herself is not minted from here: her verdict on minting is DEFER.

The contract is the house ERC-7857 `iNFT_7857` (DeltaVerse iNFT4), and an open (unsealed) agent is minted with:

```
mintOpenAgent(address to, bytes32 contentRoot, string storageURI, bytes32 metadataRoot,
              uint256 dimensions, uint8 parallelUnits, string tokenURI)        — MINTER_ROLE only
```

| argument | from the agent |
|---|---|
| `contentRoot` | its THOT manifest's `identity.contentRoot` (keccak256 of the canonical manifest). The contract accepts a content root once, ever |
| `metadataRoot` | keccak256 of the agent card's canonical bytes |
| `storageURI`, `tokenURI` | `local://thot/<cid>` and `local://card/<cid>` until the bundle and card are stored somewhere (the rung stays `referenced`) |
| `dimensions`, `parallelUnits` | one of 8 … 1048576 (768 by default), and 1 |

- **Plan and simulate** builds the calldata and runs it as an `eth_call` from the minter. You get the token id it
  would receive, or the decoded revert (`AccessControlUnauthorizedAccount`, `ContentRootAlreadyMinted`,
  `InvalidDimension`, …).
- **Unsigned transaction** gives the transaction (chain id, to, data, gas estimate) and the same call as a
  `cast send … --account <your keystore>` line. **You sign it**, in your own wallet: bankml never signs on a
  public chain.
- **Mint on the local devnet** sends only when the chain id is 31337 (anvil). On any other chain it refuses.
- **Load the agent from this token** reads `getPayload`, `tokenURI`, `ownerOf` and `openMint`, finds the agent whose
  THOT generation has that `contentRoot` (every manifest is archived by CID in `<agent>/thot/`), verifies its
  bundle, and walks the THOT parent links from the current generation back to the minted one. The genesis stays
  attached to the token forever; later generations are proven to descend from it.
- **Start a local devnet** runs anvil (127.0.0.1:8545, chain 31337), deploys `iNFT_7857` from its compiled artifact
  (`DeltaVerse/deploy/iNFT4/out`) with account 0 as admin and minter, and fills in the fields. The whole path then
  runs on this computer.

Where it stands: `iNFT_7857` is **not deployed on any public chain**, its audit (`iNFT4/audit.json`) is not cleared,
and a real mint needs a wallet holding MINTER_ROLE. `testing/test_chain.py` runs the whole path on a throwaway
anvil: deploy, simulate (and the refusal without the role), unsigned transaction, mint, read-back, the double-mint
refusal, load, two more generations and load again, and the refusal to send off a devnet.

## 9. Receipts, and how to check an answer

Every answer from `bankml serve` carries:

| field | meaning |
|---|---|
| `bankml` | the version that served it |
| `engine` | what computed it (today: llama.cpp b11192 behind bankml P0) |
| `model_sha256`, `guard` | the file that was verified and pinned, and the guard's verdict |
| `prompt_tokens`, `completion_tokens` | the engine's own counts |
| `ttft_ms`, `wall_ms` | time to first token and total, at the gateway |
| `response_sha256` | sha256 of the answer text exactly as the model produced it (`choices[0]`, UTF-8; an unpaired `\u` surrogate counts as U+FFFD) |
| `request_sha256` | (0.1.7+) sha256 of the request body bankml forwarded: which prompt this answer is to |

The UI recomputes the answer's sha256 and shows ✓ when it matches. A mismatch shows `(≠ received!)`. To check an
answer yourself, hash `assistant_raw` from `.history` (older records: `assistant`) and compare it with
`receipt.response_sha256`.

**What a receipt proves, and what it does not.** It binds an answer to the request it answered and to the sha256 of
the model file bankml verified before serving, and it catches any change to the text afterwards. It is **not
signed**: anyone can write a receipt, so it shows integrity between you and your own bankml serve, not to a third
party who does not trust your machine. bankml serve also re-checks, before each answer, that the model file is the
one it verified (device, inode, size and modification time) and that the engine still serves that file; if either
changed, it refuses to answer rather than issue a receipt for weights it did not verify.

## 10. Savante's canon and the iNFT ledger

Savante is defined by her canon, `~/cryptoAGI/savante` (github.com/cryptoAGI/savante), bound for an iNFT by the ledger
`savante.commitments.json`. The ledger commits by sha256 (and CIDv1) to:
- the persona;
- the charter;
- the sagi skill;
- six `sAGI.*` facets;
- the agent card;
- the image;
- the thot bundle;
- a keccak256 **doctrine root** over 15 clauses an owner may not edit.

At start the UI re-hashes every one of those files and shows the result in **Integrity**. If the persona does not
verify, the UI refuses to speak as Savante. The **Verifier** tab runs the full offline check. Nothing here mints: the
card's status is `not_yet_minted`, and the decision to mint belongs to the owner's signature. Point the UI at another
checkout with `SAVANTE_CANON=/path`.

## 11. Testing and the release gate

```sh
cargo test --release                                             # offline
BANKML_GGML_LIB=/path/to/llama-b11192 testing/release_gate.sh    # everything → testing/results/<version>.txt
testing/live.sh "title" <command…>                               # one step, shown live in view mode
```

The gate runs, in order:
1. the build, the unit tests and the end-to-end CLI tests;
2. clippy (`-D warnings`);
3. the licence headers (`testing/spdx_check.py`);
4. the Python suites: the guard, the UI data layer, the PostgreSQL connector (a throwaway cluster), the iNFT path (a
   throwaway anvil devnet), the model importer (with a real carrier on spare ports);
5. the Rust and Python guards agreeing on every case;
6. every oracle: bankml's kernels must equal llama.cpp's compiled kernels, bit for bit, on every weight of the real
   models;
7. the A/B speed tests, the prefill tile, the memory floor and the whole-token budgets.

A speed counts only if every oracle passed on the same code. See `testing/README.md`.

## 12. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| chat says *Cannot reach bankml serve* | serve not running | `./install.sh status`, then `./install.sh model` (§3) |
| the installer stops at a step | that step's prerequisite is missing | it says which; fix it and run `./install.sh` again (steps already done are quick) |
| serve prints `refuse: … != pinned` | the file is not the one the fork pinned | re-download; `bankml sha256 FILE` |
| serve prints `upstream … serves X, not the verified Y` | the llama-server on that port runs another file | restart it with the verified file, or use `--spawn` |
| serve prints `upstream … not healthy` | llama-server not up (a cold load can take a minute) | wait; check its log |
| the first answer takes minutes | CPU prefill of the ~300-token system prompt (≈3 tok/s on a laptop) | expected; later turns reuse the cache |
| *refused: savante.persona does not verify* | the canon was edited or is incomplete | `git -C ~/cryptoAGI/savante status`; run the Verifier |
| view mode says *swap full* | the machine is under memory pressure | close other work before measuring |
| view from another computer does not load | firewall, or bound to 127.0.0.1 | `--host 0.0.0.0`; allow TCP 7874 |

## 13. Reference: commands, ports, environment

| command | what |
|---|---|
| `./install.sh [step …] [--skip-tests] [--no-start] [--view] [--voice]` | install and start (§1); steps: check build engine python canon model voice start stop status |
| `bankml guard FILE [--engine mainline\|prism] [--json]` | header check: play / refuse / need_more |
| `bankml sha256 FILE` | the file's sha256 |
| `bankml pin FILE --fork FORK.json` | sha256 against the fork's record |
| `bankml verify FILE --fork FORK.json [--engine mainline\|prism] [--json]` | guard, then pin |
| `bankml serve FILE --fork FORK.json [--upstream H:P \| --spawn BIN] [--listen H:P] [--threads N] [--ctx N] [--spec-ngram] [--slot-dir DIR]` | the gate in front of llama-server (n-gram speculation opt-in; slot save/restore directory) |
| `bankml chat-template MODEL.gguf < messages.json` | the prompt a conversation becomes, as llama.cpp's `/apply-template` (P3; byte-identical on its oracle; tools and assistant prefills refused) |
| `bankml gpu [--remote]` | every video card found (Vulkan, merged with `/sys/class/drm`) and which bankml will use; `--remote` adds the GPUs Hugging Face rents (22 NVIDIA flavors with card counts and prices; listed, never started); `BANKML_GPU=off` turns the component off, `BANKML_GPU=0,2` picks cards (0.2.12; the GPU kernels are the next steps) |
| `bankml generate MODEL.gguf [--max N] [--sample [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S]] < messages.json` (or plain text) | bankml's own forward pass, greedy, or with `--sample` llama-server's sampler chain (the model's defaults unless given; same seed, same tokens as llama-server, 0.2.11), streamed (P3, 0.2.7): token-identical to llama-server b11192 on its oracle for the 1-bit and ternary models, prompts of any length and contexts of any length (all three of ggml's CPU attention kernels, since 0.2.10); `BANKML_LLAMA_THREADS` (default 3) must equal the `-t` of the llama.cpp being matched, because its long-context decode kernel chunks by thread; `BANKML_THREADS` sets the threads. Ternary: 2.3–2.4 tokens/s against llama-server's 0.30; 1-bit: 1.8 against 2.8 |
| `bankml tokenize MODEL.gguf [--no-special] < text` | token ids, as llama.cpp's `/tokenize` (P3's tokenizer; token-identical on its oracle) |
| `bankml usage [PID …]` | memory, cores, and each process's resident memory and CPU % (bankml's psutil, from `/proc`); `bankml serve` answers the same at `GET /bankml/usage` |
| `python3 sAGI/savante.py --mode interact [--port 7873]` | talk to Savante (loopback) |
| `python3 sAGI/view.py [--host 0.0.0.0] [--port 7874]` | the read-only page for the LAN |
| `python3 sAGI/models.py list \| catalog \| search Q \| import ID\|URL\|ollama:NAME:TAG \| use FILE \| first-run` | the model importer and carrier switch |
| `python3 sAGI/speak.py [--prune]` | render every voice clip, write both exports (`--prune`: drop unused clips) |

| port | service |
|---|---|
| 18092 | llama-server b11192 (upstream, loopback) |
| 18093 | bankml serve (loopback) |
| 7873 | interact mode (loopback) |
| 7874 | view mode (LAN) |

| variable | default | meaning |
|---|---|---|
| `SAVANTE_CANON` | `~/cryptoAGI/savante` | Savante's canon (read-only) |
| `BANKML_SERVE` | `http://127.0.0.1:18093` | where the UI finds bankml serve |
| `BANKML_UI_STATE` | `~/.local/share/bankml/savante` | `.history` and caches |
| `BANKML_REPO` | the checkout | where view mode reads `testing/` |
| `BANKML_AGENTS` | `~/.local/share/bankml/agents` | custom agents (one git repository each) |
| `BANKML_PG_DSN` | `dbname=bankml` | PostgreSQL for publishing and loading agents |
| `RAGE_PATH` | `~/mindX/mindx/godel/mindxtrain/hf/space_ui` | where the ragebar finds mindX's `rage.py` (built-in BM25 otherwise) |
| `BANKML_GGML_LIB` | — | llama.cpp b11192 release dir, for the oracles |
| `BANKML_THREADS` | pool: all cores; budgets: `1,2,3,4` (Q1_0), `1,3` (Q2_0) | the thread pool's size, or a comma list of thread counts for the decode budgets |
| `BANKML_NO_SHANI` | unset | set to hash with the portable SHA-256 instead of the CPU's SHA extensions |
| `BANKML_OLLAMA_MODELS` | `/usr/share/ollama/.ollama/models` | the local Ollama store the importer adopts from |
| `BANKML_PIPER`, `BANKML_PRONUNCIATION` | `~/.local/share/bankml/piper`, the house table | Savante's voice engine and pronunciation table |
| `BANKML_ESPEAK`, `BANKML_ESPEAK_VOICES` | the DeltaVerse vendor dirs | the eSpeak fallback voice |
| `BANKML_OLLAMA`, `BANKML_EMBED_MODEL`, `BANKML_EMBED_KEEP_ALIVE`, `BANKML_EMBED_NEED_GB` | see [embedding.md](embedding.md) | the embedding model |
| `BANKML_MODELS` | `.models` in the checkout | where imported models go |
| `BANKML_FORKS` | `~/.local/share/bankml/forks` | FORK.json pins, one per imported model |
| `BANKML_LLAMA_SERVER` | `~/sAGI/bonsai/llama-b11192/llama-server` | the engine the carrier spawns |
| `BANKML_BIN` | `target/release/bankml` | the bankml binary the importer and switch call |
| `BANKML_FIRST_RUN` | `1` | `0` stops interact from starting the Bonsai-8B carrier by itself |
| `BANKML_CTX`, `BANKML_THREADS_SERVE` | `2048`, `3` | the engine's context and threads when the importer starts the carrier |
| `BANKML_SERVE_LISTEN`, `BANKML_UPSTREAM` | `127.0.0.1:18093`, `127.0.0.1:18092` | the ports the switch manages |
| `BANKML_VOICE_DIR`, `BANKML_EXPORT_DIR` | `sAGI/voice/cache`, `sAGI/voice/export` | voice clips and the two exports |
| `BANKML_VOICE_ASYNC` | `1` | `0` stops the UI rendering missing clips in the background |
