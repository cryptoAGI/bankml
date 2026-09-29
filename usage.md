# Using bankml and Savante

This guide goes from a fresh checkout to asking Savante a question on your own computer, and to letting other people
watch the testing on your network. It also says what each part does, what it checks, and where it keeps what.

- [1. What runs where](#1-what-runs-where)
- [2. What you need](#2-what-you-need)
- [3. Build and check bankml](#3-build-and-check-bankml)
- [4. Verify the model](#4-verify-the-model)
- [5. Start `bankml serve`](#5-start-bankml-serve)
- [6. Talk to Savante (interact mode)](#6-talk-to-savante-interact-mode)
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

## 1. What runs where

```
 you (browser) ──► ui/savante.py  (interact, 127.0.0.1:7873, Gradio)  ──► bankml serve (127.0.0.1:18093)
                     │ reads ~/savante (read-only)                        │ verified: guard + sha256 pin
                     │ writes ~/.local/share/bankml/savante/              ▼
                     │        savante.history                         llama-server b11192 (127.0.0.1:18092)
                                                                       Bonsai-8B Q1_0 — the file bankml verified
 anyone on the LAN ──► ui/view.py (view, 0.0.0.0:7874, stdlib, read-only) ──► testing/live.log, testing/results/, CI
```

- **bankml serve** is the gate. It refuses to start unless the model file passes the guard, its sha256 equals the
  pin in the model fork's `FORK.json`, and the llama-server behind it serves *that* file. Every answer carries a
  receipt.
- **Interact mode** is where you talk to Savante. It is only reachable from this computer (loopback), because Gradio
  3.x must not face a network (see §7).
- **View mode** is a small read-only page for everyone else: the live testing log, the release records, CI, this
  machine's load, and Savante's office and ledger. It has no chat, no history, and no way to run anything.
- **The answers come from llama.cpp b11192's kernels** (bankml phase P0). bankml vouches for the file, the path and
  the transcript. bankml's own forward pass is phase P3.

## 2. What you need

| what | where it comes from | check |
|---|---|---|
| Rust 1.95 (or newer) | rustup | `cargo --version` |
| Python 3.10+ with Gradio 3.x or newer | `pip install gradio` | `python3 -c "import gradio; print(gradio.__version__)"` |
| llama.cpp b11192 release (`llama-server`) | `llama-b11192-bin-ubuntu-x64.tar.gz`, sha256 `34cf6fa5…81ec7` | `sha256sum` against that value |
| The model | [PYTHAI/Bonsai-8B-gguf-fork](https://huggingface.co/PYTHAI/Bonsai-8B-gguf-fork) → `Bonsai-8B-Q1_0.gguf` (1.16 GB) | step 4 |
| The fork's `FORK.json` | the same repository | step 4 |
| Savante's canon | `git clone https://github.com/cryptoAGI/savante ~/savante` | §10 |

RAM: the 1-bit 8B model maps 1.16 GB; the whole stack runs in about 2 GB. The ternary model (2.31 GB) also works and
gives better answers, but llama.cpp runs it about 5× slower on a CPU (see PERFORMANCE.md).

## 3. Build and check bankml

```sh
git clone https://github.com/cryptoAGI/bankml && cd bankml
cargo build --release
cargo test --release          # unit + end-to-end tests, offline, a few seconds
target/release/bankml version
```

## 4. Verify the model

```sh
mkdir -p .models
# put Bonsai-8B-Q1_0.gguf and the fork's FORK.json in .models/ (or anywhere)
target/release/bankml guard  .models/Bonsai-8B-Q1_0.gguf                  # play | refuse (reason) | need_more
target/release/bankml verify .models/Bonsai-8B-Q1_0.gguf --fork .models/FORK.json --json
```

`verify` prints `{"verdict": "play", …, "model_sha256": "284a335a…"}` and exits 0. Anything else refuses, gives its
reason, and exits 2 (1 for an I/O error, 3 for a truncated file).

## 5. Start `bankml serve`

Either let bankml launch llama-server on the verified file (recommended: the identity holds by construction):

```sh
target/release/bankml serve .models/Bonsai-8B-Q1_0.gguf --fork .models/FORK.json \
    --spawn /path/to/llama-b11192/llama-server --upstream 127.0.0.1:18092 --threads 3 --ctx 4096
```

or put it in front of a llama-server that is already running. bankml then checks, through the server's `/props`,
that it serves the same file:

```sh
/path/to/llama-b11192/llama-server -m .models/Bonsai-8B-Q1_0.gguf --host 127.0.0.1 --port 18092 \
    -c 4096 -t 3 -np 1 --jinja --reasoning off --no-webui &
target/release/bankml serve .models/Bonsai-8B-Q1_0.gguf --fork .models/FORK.json --upstream 127.0.0.1:18092
```

It listens on `127.0.0.1:18093` (`--listen` to change). Check it:

```sh
curl -s 127.0.0.1:18093/bankml          # what was verified, when, and what is upstream
```

It is an OpenAI-compatible endpoint, so any client works:

```sh
curl -s 127.0.0.1:18093/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello. /no_think"}],"max_tokens":32}'
```

## 6. Talk to Savante (interact mode)

```sh
python3 ui/savante.py --mode interact          # then open http://127.0.0.1:7873
```

The page, left to right. Every panel can be resized from its bottom-right corner, and the side panel can be dragged
by its grip to either side of the chat. Your choice is remembered in your browser.

- **The chat.** Type a question and press **Send** (or Enter). **Stop** cancels the answer being written; **New
  session** starts a fresh conversation. The last session's history reloads on start.
- **The timer.** It starts at the press of Send and ticks every second. It shows first *reading the prompt
  (prefill)* and then *writing · first token at N s*. On a laptop CPU the first turn spends most of its time reading
  Savante's roughly 300-token system prompt: expect about 2 minutes, then a few tokens per second.
- **Under every answer:** `⏱ sent HH:MM:SS · first token N s · answered in N s`, then the receipt (§9), then which
  system prompt carried the turn. Every answer is labelled **draft · not a finding**, because Savante is an office and
  a draft is not a verdict.
- **`.prompt`** picks the system prompt:

  | choice | what it is | ledgered? |
  |---|---|---|
  | persona · system_prompt (the default) | `savante.persona` → `system_prompt`, exactly as in the canon | yes — refused if it does not verify |
  | sAGI.prompt | the canon's sAGI facet | yes |
  | Savante.prompt | the Hugging Face Space's template prompt, fetched once and cached | no — said so under each answer |

- **max tokens** and **temperature** are the usual sampling controls. Low temperature suits verdicts.
- **The bankml serve card** shows the verified model, its full sha256, the bankml version and the engine. **Refresh
  carrier** re-reads it.

- **use .memory** (side panel, on by default) appends your `.memory` notes to the system prompt, newest first, within
  2,400 characters. They are labelled as the operator's notes, not evidence, and the footer says how many went in.

The other tabs:
- **.history**: every exchange, newest first: its time, session, time to first token, response time, tokens, and
  whether the answer's sha256 matches its receipt. At the top is the **ragebar**: type, and the exchanges are
  ranked by RAGE retrieval as you type, each with a relevance meter. It uses mindX's `rage.py` when present
  (`RAGE_PATH`) and the same BM25 built in otherwise; the engine is named above the results.
- **Responses**: one answer at a time. **⤒ first**, **▲ previous**, **▼ next**, **⤓ latest** step through every
  response in `.history`. **📋 copy** puts the answer on the clipboard, **➕ save to .memory** keeps it as a note
  (with its source: session, send time, answer sha256), and **🔏 proof** gives its inclusion proof (§8a).
- **.memory**: your notes, numbered. Add one by typing, remove one by number.
- **Metrics**: computed from `.history`. It shows the number of exchanges and sessions, and the median, p90, mean,
  minimum and maximum of the time to first token, the response time, the prompt-reading (prefill) speed and the
  writing speed. It also shows a bar per exchange, how many answers match their receipt, the last 25 exchanges, and
  the commitments (§8a). Times come from the press of Send where recorded (0.0.7+); older exchanges use the receipt's
  gateway times, and each row says which.
- **Testing (live), Results, Office, Integrity** are the same as view mode (§7).
- **Verifier** runs `~/savante/bind/savante_verify.py` offline. Exit 0 means APPROVE: the ledger, the doctrine root,
  the thot bundle and the mirror all agree.

To ask for a review, say so: *"review: is Bonsai-8B ready to serve mindX?"*. Savante then answers under her verdict
contract (FINDINGS / VERDICT / RATIONALE / CONDITIONS / RISKS WATCHED). A plain question gets a short answer in the same
voice. The 1-bit Bonsai-8B carrier was graded REJECT for review duty in the sAGI carrier test, so treat its verdicts as
drafts.

## 7. Let others watch (view mode, on the LAN)

```sh
python3 ui/view.py --host 0.0.0.0 --port 7874   # others open http://<this computer's LAN address>:7874
ip -4 addr | grep inet                          # find the LAN address
```

- **Testing — live**: the running step (a green *running* pill, or *idle*) and the last 80 lines of
  `testing/live.log`, which the release gate and `testing/live.sh` append to. It refreshes every 2 s and follows the
  bottom unless you scroll up.
- **This laptop**: CPU, load, free RAM, swap. When swap is full the page says measurements may be disturbed.
- **bankml serve**: verified or not, the model, its sha256.
- **Release records**: every `testing/results/<version>.txt`.
- **CI**: the last five GitHub Actions runs.
- **Savante**: name, mantra, card status (`not_yet_minted`), doctrine root, and the ledger check, file by file.
- **Private data — commitments only**: the count, Merkle root and CID of `.history`, and the count and root of
  `.memory`. Never their content (§8a).

Every panel can be dragged by its title to a new place and resized from its corner. **Reset layout** puts them back.
The layout lives in each viewer's own browser.

**Why view mode is not Gradio:** the Gradio installed here (3.37) has path-traversal bugs that let a client read files
from the host (e.g. CVE-2023-51449, fixed in 4.11). `ui/view.py` is the Python standard library with four fixed GET
routes (`/`, `/api/state`, `/api/result?name=` for a listed record only, `/savante.png`). Every other path is a
404, POST is refused, and the page renders all data as text under a strict Content-Security-Policy. Keep interact mode
on `127.0.0.1`.

## 8. The files Savante keeps: `.history`, `.memory`, `.prompt`

Nothing is ever written into the canon (`~/savante`). What the UI writes lives in `BANKML_UI_STATE` (default
`~/.local/share/bankml/savante/`):

- **`savante.history`**: one JSON object per exchange (JSONL):

  ```json
  {"ts": 1790633549.123, "sent_at": "2026-09-28T15:12:29.123-0700", "first_token_s": 113.02,
   "response_s": 125.11, "answered_at": "2026-09-28T15:14:34.233-0700", "session": "e2e-test",
   "user": "In one sentence: …", "assistant": "Savante knows …", "shown": "… (with the footer)",
   "prompt": "persona · system_prompt (canon, ledgered)", "prompt_provenance": "persona.system_prompt · persona sha256 d96556b11989… (ledgered)",
   "receipt": {"bankml": "0.0.7", "model_sha256": "284a335a…", "prompt_tokens": 316, "completion_tokens": 28,
               "ttft_ms": 113000, "wall_ms": 125100, "response_sha256": "80010408…"}}
  ```

  `sent_at` is the press of Send; `first_token_s` and `response_s` are measured from it. The receipt's
  `ttft_ms`/`wall_ms` are measured by bankml serve from when it forwarded the request. The history is plain text:
  back it up, grep it, or delete it.
- **`savante.memory`**: one JSON object per note: `{"ts", "at", "text", "sha256", "source": {"kind": "typed" |
  "response", "session", "sent_at", "response_sha256"}}`.
- **`Savante.prompt`**: the Space template's prompt, cached the first time it is chosen.

The model sees the last 12 exchanges (each cut to 4,000 characters), as in the Hugging Face template.

## 8a. Proof of data without the data

`.history` and `.memory` stay on this computer. What can leave it is a **commitment**:

| part | what |
|---|---|
| leaf | sha256 of one JSONL line, exactly as written (without its newline) |
| Merkle root | pairwise sha256 up the tree; an odd node is paired with itself |
| file sha256 / CIDv1 | of the whole file; CIDv1 raw (0x55), sha2-256, base32 lower: the construction Savante's iNFT ledger uses |

**🔏 proof** (Responses tab) produces, for one exchange, `{"commitment": …, "proof": {"leaf", "path": [{"side",
"hash"}, …], "merkle_root"}, "verifies": true}`. Give someone the exchange's line and the proof: they hash the line,
fold in the path, and compare with the root you published. That proves the exchange is in your history, unchanged,
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
the binder's preflight (non-ASCII keys, floats, integers beyond ±2⁵³) or lacks a doctrine clause, and otherwise
regenerates the card and ledger. An agent whose files do not match its ledger refuses to speak.

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
  when pgvectorscale is installed and HNSW (pgvector) otherwise. Embeddings are not computed yet (no local embedding
  model is served); the column is ready for them.
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
| `response_sha256` | sha256 of the answer text exactly as the model produced it |

The UI recomputes the answer's sha256 and shows ✓ when it matches. A mismatch shows `(≠ received!)`. To check an
answer yourself, hash the text from `.history` (`assistant` before `<think>` stripping; for Bonsai with reasoning off
they are the same) and compare it with `receipt.response_sha256`.

## 10. Savante's canon and the iNFT ledger

Savante is defined by her canon, `~/savante` (github.com/cryptoAGI/savante), bound for an iNFT by the ledger
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
1. the unit tests and the end-to-end CLI tests;
2. clippy;
3. both guard checks (Python and Rust must agree);
4. every oracle: bankml's kernels must equal llama.cpp's compiled kernels, bit for bit, on every weight of the real
   models;
5. the A/B speed tests and the whole-token budgets.

A speed counts only if every oracle passed on the same code. See `testing/README.md`.

## 12. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| chat says *Cannot reach bankml serve* | serve not running | §5; `curl 127.0.0.1:18093/bankml` |
| serve prints `refuse: … != pinned` | the file is not the one the fork pinned | re-download; `bankml sha256 FILE` |
| serve prints `upstream … serves X, not the verified Y` | the llama-server on that port runs another file | restart it with the verified file, or use `--spawn` |
| serve prints `upstream … not healthy` | llama-server not up (a cold load can take a minute) | wait; check its log |
| the first answer takes minutes | CPU prefill of the ~300-token system prompt (≈3 tok/s on a laptop) | expected; later turns reuse the cache |
| *refused: savante.persona does not verify* | the canon was edited or is incomplete | `git -C ~/savante status`; run the Verifier |
| view mode says *swap full* | the machine is under memory pressure | close other work before measuring |
| view from another computer does not load | firewall, or bound to 127.0.0.1 | `--host 0.0.0.0`; allow TCP 7874 |

## 13. Reference: commands, ports, environment

| command | what |
|---|---|
| `bankml guard FILE [--engine mainline\|prism] [--json]` | header check: play / refuse / need_more |
| `bankml sha256 FILE` | the file's sha256 |
| `bankml pin FILE --fork FORK.json` | sha256 against the fork's record |
| `bankml verify FILE --fork FORK.json [--json]` | guard, then pin |
| `bankml serve FILE --fork FORK.json [--upstream H:P \| --spawn BIN] [--listen H:P] [--threads N] [--ctx N]` | the gate in front of llama-server |
| `python3 ui/savante.py --mode interact [--port 7873]` | talk to Savante (loopback) |
| `python3 ui/view.py [--host 0.0.0.0] [--port 7874]` | the read-only page for the LAN |

| port | service |
|---|---|
| 18092 | llama-server b11192 (upstream, loopback) |
| 18093 | bankml serve (loopback) |
| 7873 | interact mode (loopback) |
| 7874 | view mode (LAN) |

| variable | default | meaning |
|---|---|---|
| `SAVANTE_CANON` | `~/savante` | Savante's canon (read-only) |
| `BANKML_SERVE` | `http://127.0.0.1:18093` | where the UI finds bankml serve |
| `BANKML_UI_STATE` | `~/.local/share/bankml/savante` | `.history` and caches |
| `BANKML_REPO` | the checkout | where view mode reads `testing/` |
| `BANKML_AGENTS` | `~/.local/share/bankml/agents` | custom agents (one git repository each) |
| `BANKML_PG_DSN` | `dbname=bankml` | PostgreSQL for publishing and loading agents |
| `RAGE_PATH` | `~/mindX/mindx/godel/mindxtrain/hf/space_ui` | where the ragebar finds mindX's `rage.py` (built-in BM25 otherwise) |
| `BANKML_GGML_LIB` | — | llama.cpp b11192 release dir, for the oracles |
| `BANKML_THREADS` | all cores | threads for the kernels' benchmarks |
