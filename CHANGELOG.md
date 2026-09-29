# Changelog

## 0.1.5 — 2026-09-28

Models come in without friction, and only open-source, sha256-pinned ones: Bonsai-8B on first run, then the
catalogue, any Hugging Face GGUF, or Ollama. The metrics become readable charts. The voice controls get a dock. Savante
reads bankml's thesis and the binary / ternary argument, and her audio is exported and committed. Record:
`testing/results/0.1.5.txt`.

### Added — models
- **The importer** (`ui/models.py`, the **Models** tab, `python3 ui/models.py`). Its sources:
  - **First run.** If no carrier answers when interact starts, Bonsai-8B is imported (only if absent), pinned and
    verified, and `bankml serve` is started in the background with progress shown.
  - **Catalogue.** Eleven open-source models: Bonsai-8B (default), Ternary-Bonsai-8B, Bonsai-4B and 1.7B, Qwen3
    0.6B / 1.7B / 4B / 8B, SmolLM2-1.7B, SmolLM3-3B and Granite-3.3-2B. Each is pinned to its repository's revision,
    size and sha256, read 2026-09-28, and an import refuses if the repository no longer agrees.
  - **Hugging Face.** Any repository or file URL. The licence is read from the repository, and the pin is its LFS
    sha256 at the resolved revision.
  - **Ollama.** Search ollama.com; import from the registry, where the model layer's digest is the GGUF's sha256 and
    the licence layer is read; or adopt a model the local Ollama holds, by link with no download.
- **The trust chain.** Every file is hashed as it streams, and kept only if it equals the published sha256; an
  interrupted download resumes. `bankml guard` must then say play, and a FORK.json pin is written, so `bankml serve`
  verifies it before every start.
- **Open source only.** A licence outside the OSI list is refused before any byte is fetched: Gemma and Llama on
  Hugging Face and on Ollama, and anything unrecognised.
- **The machine's limits.** An import that would leave less than 1.5 GB free on disk, or whose weights exceed about
  70 % of RAM, is refused with the numbers.
- **Carrier switch.** Stop, then start `bankml serve --spawn llama-server` on the chosen model. It is refused while an
  answer is being written. If the new model does not come up, the previous one is restored; it is found by its
  verified sha256, because serve reports canonical paths.
- **Verified on this laptop:**
  - Qwen3-0.6B Q8_0 was downloaded from Hugging Face, and qwen3:0.6b (a Q4_K_M) was adopted from the local Ollama;
    both are pinned, and both answered through bankml with receipts;
  - Bonsai-1.7B, already here, was pinned against the published sha256;
  - a switch, a refusal while busy, and a rollback from a bad pin all ran on spare ports.
- `testing/test_models.py`: 19 checks, in the gate. The source is a loopback server with a synthetic GGUF, and a
  real carrier runs on spare ports.

### Changed — bankml (Rust)
- **Standard models, named and reported.** `gguf::type_name` names all of ggml's standard types (Q8_1, Q8_K, the
  IQ family, I8–I64, F64, MXFP4), and a new test confirms a K-quant model plays on mainline. The guard already played
  such files; nothing about them was allow-listed or refused.
- **`Verified` carries what was verified:** `arch`, `name` and the tensor types. `/bankml` reports them, so the UI
  shows "qwen3 · Q4_K×155, Q6_K×15" and sends `/no_think` only to families that honour it (Qwen3, SmolLM3, Bonsai).

### Changed — metrics
- **Readable charts** replace the old bar strip, whose bars became "giant blue blobs" when there were few exchanges:
  one bar was stretched across a third of the width.
  - Summary tiles: exchanges and sessions, median and p90 first-token and response times, writing speed, and answers
    matching their receipts.
  - Response time per exchange: stacked bars (waiting, then writing), at most 18 px wide, on a round-numbered axis
    with a median line. One slow outlier is clipped with a ▲ so the rest stay readable.
  - Throughput per exchange: prefill and writing tok/s as lines.
  - The charts keep their proportions at any width, and dark mode is transparent.
- **Commitments** read one per line: the .history record count, Merkle root, CID, and .memory. An empty root reads
  "— (empty)", not "None".

### Added
- **The VOICE dock.** The DreamKnobs no longer sit inside each section. They appear only when a PLAY is pressed, in
  one dock beside the card: a vertical stack in the side column by default. ⤒, ⤓ and ⇥ dock it to the top of the
  card, the bottom, or back to the side (top and bottom lay the knobs in a row). It can also be dragged by its grip
  onto a drop zone and resized from its corner (40–96 px). Place and size are remembered per browser. Verified in
  headless Firefox:
  - hidden before play, and a vertical stack to the right of the card after it, with no overlap;
  - above the card when docked top and below it when docked bottom, horizontal both times;
  - back at the side, with the choice remembered.
- **SCIEN·TIFIC in the introduction.** A new chapter, marked as bankml's own note rather than her canon
  (`ui/voice/readings/scientific.md`), follows "Why I exist". It takes 2²⁵⁶ − 1 (SCIEN·TIFIC's whole supply, the
  largest value one EVM word holds, about 1.16 × 10⁷⁷; scientific.pythai.net) first as the finest resolution a
  single word allows, the maximum measurement of accuracy, and then as a measure of size. The atom count is stated as
  measured: estimates for the observable universe run from 10⁷⁸ to 10⁸², so by a strict count 2²⁵⁶ falls short, but
  it is of the same order of scale.
- **The reading.** Savante reads bankml's thesis and the binary / ternary sections of `TECHNICAL.md` (§II.1, §III.2,
  §III.5, §III.8) verbatim, as its own section of the card with **▶ PLAY THE READING**. It is also in the view
  mode's Listen panel.
- **TECHNICAL.md §III.8, "Can a binary computer perform a ternary operation?"** Yes, exactly. A trit is stored in bits
  (log₂ 3 ≈ 1.585 bits of information; `Q2_0_g64` spends 2 bits plus the scale). The ternary product is a signed
  sum, which bankml computes as Σ c·x − Σ x with the offset code c = w + 1 (`vpmaddubsw`, then one subtraction of an
  activation sum shared by every row). The section also covers Setun and why packing, not native ternary gates, is
  the next ternary speed-up on a bandwidth-bound CPU.
- **Export.** `⤓ Savante.opus` holds the voice examples and the whole introduction; `⤓ Savante-reading.opus` holds
  the reading. Each is one Ogg Opus file (40 kbit/s) with a chapter mark per chapter. It is made only when every
  sentence is rendered, carries a signature of the exact clips it was built from, and is rebuilt when one changes.
  Both are offered in the card, and in view mode at the fixed routes `/export/Savante.opus` and
  `/export/Savante-reading.opus`, which serve only a named export that is complete and current. `python3
  ui/speak.py` renders everything and writes both files; `--prune` drops clips no current text uses.
- **The audio is in the repository.** `ui/voice/cache/savante/` (every clip plus its manifest) and `ui/voice/export/`
  are committed, so a fresh clone plays and downloads without rendering. The voice is built from public-domain Cori
  (see 0.1.3).

### Changed
- **Speech.** List markers are removed only at the start of a line, so an inline "16 + 2" is no longer read as
  "16 2". bankml's own spoken forms go before the house table: SCIEN·TIFIC ("Sci-en, Tiffic"), bankml ("bank M L"),
  web addresses ("dot"), and TECHNICAL.md's notation (Σ, ∈, ≈, ≤, ×, {−1, 0, +1}, `Q1_0`, and section references,
  which are dropped). This re-rendered a dozen sentences of the introduction whose spoken form changed.
- The view's voice label says "Savante's own voice" when Piper is present (it still said "house stand-in").

## 0.1.4 — 2026-09-28

The knobs in view mode, and a timer that counts real seconds. No Rust code changed. Record: `testing/results/0.1.4.txt`.

### Added
- **Knobs in view mode.** Anyone listening on the LAN gets the same DreamKnob controls (SPEED, FM RATE, FM DEPTH,
  GAIN, VOLUME), hidden until they press play, then emerging in the Listen panel. They run the same voice chain as the
  card: GAIN, then the FM delay line, then VOLUME, with speed pitch-preserved. Each listener's settings stay in their
  own browser. The bundle is served from a fixed `/knobs.js`, and the view's CSP allows `'self'` scripts for it.
  Verified in headless Firefox against a test instance of the view server:
  - hidden before play, emerged after it, with five knobs;
  - audio playing through the chain;
  - a SPEED change reached the audio (1.5×);
  - the line being read highlighted.

### Fixed
- **The timer's seconds.** It was redrawn by the server once a second, so its tenths never moved (it always ended in
  ".4"). The browser now counts from the moment Send was pressed, ten times a second, and the server only changes the
  phase. Verified in a browser: a timer stamped 12.3 s earlier read 24.6 s after the page was held for 12 s.

## 0.1.3 — 2026-09-28

Savante gets a voice of her own: calm, confident, slower and more thoughtful. The DreamKnob controls are now in the
card where she plays. No Rust code changed. Record: `testing/results/0.1.3.txt`.

### Changed
- **Her voice.** The eSpeak stand-in of 0.1.2 was a formant sketch and sounded harsh ("that voice is scary"). The
  operator's direction was to use Jaimla as the template, build from Cori, and make a voice that is Savante's own:
  - **body**: Piper `en_GB-cori-high`, trained on **public-domain** LibriVox recordings. Jaimla's own voice (Piper
    jenny_dioco) comes from a custom-licence dataset, so under the open-source rule it serves as the template, never
    the source. The body is rendered slower and steadier: length_scale 1.18 (about 142 wpm), noise 0.50 and timing
    noise 0.60 (down from 0.667 and 0.8), and 0.8 s of silence after each sentence;
  - **Jaimla as the template**: each clip's own f0 is measured and moved onto Jaimla's measured 182 Hz with
    rubberband, formants preserved. The result is lower and grounded, not a slowed tape. Measured: f0 181–185 Hz;
  - **SAVANTE's resonance** (the house recipe in `docspeech_voices.json`): her own voice an octave below, 80–2600 Hz,
    heard only in echo (4 taps at 29 ms, decay 0.5), at 0.30. It is taken from the same clip, so it is locked to her
    delivery exactly;
  - **EQ**: SAVANTE's curve with presence +6 dB at 3 kHz and a +4 dB shelf above 5 kHz. Her brightness stays near
    Cori's own (centroid about 2270 Hz). Forcing Jaimla's 2783 Hz took presence that turns sibilant, the opposite of
    calm.

  It is rendered on this computer (Piper's standalone MIT release and the model, in `~/.local/share/bankml/piper`,
  nothing committed) and cached. Every clip's measured f0 and pitch ratio are in the manifest. The eSpeak stand-in
  remains only as the fallback where Piper is absent.
- **An oscilloscope behind the card.** A phosphor-teal trace, graticule and persistence glow behind the card's
  translucent glass. While Savante speaks, it draws her actual waveform from a Web Audio analyser at the end of the
  playback chain, so SPEED and FM show in it. Otherwise it idles as a slow sine sweep. It only draws while the card is
  open and holds still under reduced motion.
- **3D depth**: the card stands in perspective and tilts toward the pointer (up to about 5°), with a sheen that
  follows the tilt and sections raised on a bevel (light above, shadow below) over the scope plane. There is no tilt
  under reduced motion.
- **PLAY plays.** In 0.1.2's design a chapter offered PLAY only once every sentence was rendered, and during a render
  there was often nothing to queue. Now every sentence is listed, the rendered ones play at once (the rest show
  "rendering"), and chapter headers count what is ready. Rendering runs in listening order: the voice examples, then
  chapter 1, 2, 3 and so on. The buttons use one delegated click handler instead of inline attributes, and a clip
  still plays if Web Audio is unavailable. Verified in headless Firefox against the real card and page script: after
  PLAY ALL, audio advancing, the line highlighted, the knobs out.
- **PLAY beside her name.** It reads who she is (the card's description) and ends when that ends. The next PLAY THE
  INTRODUCTION **continues from there**, not from the start, and runs on through the introduction and her voice
  examples until stopped. A sentence's ▶ plays from that sentence onward.
- **Every PLAY becomes ❚❚ PAUSE while playing**, then ▶ RESUME at the exact point; the button in use is lit gold.
- **GAIN and VOLUME knobs.** GAIN (−12 to +12 dB) sets her level into the chain; VOLUME (0–100 %) is the master out.
  The oscilloscope reads between the two, so it shows her signal, not your listening level. Five DreamKnobs in all:
  SPEED, FM RATE, FM DEPTH, GAIN, VOLUME.
- **Input ready**: a slow-blinking (1.6 s), light-green, semi-transparent block cursor at the start of the empty input
  field, with the typing caret in the same green. It holds still under reduced motion.
- **The knobs emerge when she speaks**: no longer on the landing's settings panel. They are hidden in the card and
  slide out, in the section that is playing, the moment PLAY is pressed.
- **Her text is shown as written**: `savante_sagi`, `core_command` and `APPROVE_WITH_CONDITIONS` keep their
  underscores on screen and are spoken as words. The 0.1.2 cleaner dropped intra-word underscores.
- **The page never waits on her voice**: missing clips render in the background, and the card rebuilds on every page
  load, so it fills in as the renders finish.

## 0.1.2 — 2026-09-28

Savante speaks: her introduction and her voice examples, pre-rendered in her voice, with DreamKnob controls. No Rust
code changed. Record: `testing/results/0.1.2.txt`.

### Added
- **Savante's voice, and only hers.** The house defines SAVANTE (mindX `docspeech_voices.json`, id `savante`) as a
  layered piper voice: Jaimla's body, a resonance an octave below in echo, breath at the edges. That voice renders on
  the house render host. On this computer the UI uses the house's own stand-in for her, the cast entry `savante` in
  the DeltaVerse voice index: eSpeak NG `en-gb-x-rp+jaimla` at 168 wpm, which is what `listen.html` plays for her.
  - It is rendered here, under Node, by the same WASM build the DeltaVerse voice worker uses, with the house voice
    files installed before the first synthesis. A test confirms the variant really applies: it changes both the
    length and the hash of the audio.
  - Her name is said sav-ont: the house pronunciation table is applied to what is spoken, never to what is shown.
  - Every agent in this UI speaks with her voice.
- **The introduction, in the card.** Seven chapters that Savante reads to a new participant, verbatim from her canon:
  Who I am, My oath, What I believe, How I work (her `.prompt`), Why I exist (`explanation.md`), The manifesto, and
  Savante in full. That is 272 sentences, about 26 minutes. **▶ PLAY THE INTRODUCTION** plays them all, **▶
  chapter** plays one, and any sentence's ▶ plays from there. The line being read is highlighted and scrolled into
  view.
- **Her voice examples, in the card**: all 14, with PLAY ALL and a ▶ per statement.
- **DreamKnob controls in the settings panel** (**VOICE · Savante**), built from the local `~/dreamknob` workspace
  with React into one committed script (`ui/voice/knobs/`), as the playdocs rack is built.
  - **SPEED**: a vintage knob from 0.5 to 2.5× with the rack's snap points, pitch preserved.
  - **FM RATE** (0–12 Hz) and **FM DEPTH** (0–100 %): frequency modulation of her voice, an LFO sweeping a short
    delay line. At depth 0 she is heard as rendered.
  - Changes apply live, mid-sentence, and are remembered.
- **Links in the card**:
  - on Hugging Face: Savante (the Space `PYTHAI/savante`), sAGI (the skill in that Space), and Savante's loop (the
    dataset);
  - on GitHub: the sAGI engine (`cryptoAGI/sagi`), Savante's canon and bankml.

  All were verified live before linking. sAGI has no Hub repository of its own, so its link is the skill in
  Savante's Space.
- **Listen to Savante, in view mode on the LAN.** The same chapters and statements, play-all, per chapter and from any
  line, served by a fixed `/audio/<key>.ogg` route that answers only a 24-hex key listed in the voice manifest. The
  page's CSP gains `media-src 'self'`.
- **`ui/speak.py`**: the voice. Renders are cached by (engine, voice, rate, variant hash, table version, text) in a
  git-ignored `ui/voice/cache/`: Gradio refuses to serve any path with a dot-directory, and the audio is not
  source. `python3 ui/speak.py` pre-renders everything (286 statements, 27.5 minutes, 7.1 MB here). The UI renders
  whatever is missing in the background.
- `testing/test_ui.py`: 42 checks. The new ones cover the pronunciation (longest first, idempotent), speech from
  markdown, one real render to Opus, and the view's audio gate.

## 0.1.1 — 2026-09-28

Savante's aivatar and a layout that moves both ways. No Rust code changed. Record: `testing/results/0.1.1.txt`.

### Fixed
- **The side panel could be dragged left but not back to the right**: the drop depended on the elements under the
  pointer passing the event up. While you drag the grip, two translucent **drop zones** now cover the two halves of
  the layout above everything else, so nothing underneath can take the drop. A **⇄** button on the grip also swaps
  sides with a click. The side is remembered per browser.

### Added
- **The aivatar card.** Click the agent's portrait: a card opens, in dark glass with a fine grid, teal and gold
  accents and a slowly breathing glow. It holds:
  - the kind and card type, name, mantra and the card's attribute chips;
  - the description, oath, office (primary skill, scope) and beliefs;
  - the **identity, verifiable rather than asserted**: persona sha256, doctrine root, THOT identity (`thot:`, CID,
    contentRoot, generation, facets) and the ledger check;
  - **every aspect of the persona** in collapsible sections (BDI, skills, safety, embodiment, token, task, voice
    examples, exchanges, and the system prompt), rendered from the file itself;
  - **every ledgered file** with its size and sha256.

  The card opens and closes with a pure-CSS toggle (Gradio does not run scripts in HTML blocks), and the image is
  embedded, never served by path. A derived agent gets the same card with its own values.
- **A chosen aivatar.** Savante: any image in her canon's `gfx/`, the choice kept outside the canon. A derived agent:
  an uploaded image, checked by content (PNG, JPEG or WebP) and size (2 MB at most), ledgered as `x-bankml.aivatar`
  and included as a facet of its THOT bundle. Choose in **Agents → aivatar**.
- **Refined resize handles**: a slim corner with diagonal ridges, the same in every browser, replaces the browser's
  default resize grip on the side panel (width and height) and the chat (height). Sizes are remembered.
- `testing/test_ui.py` grows to 37 checks: an aivatar is accepted and ledgered, a non-image and an oversized file are
  refused, and the THOT bundle gains the aivatar facet.

### Not tested by machine
- The drag itself: a real mouse drag cannot be driven in this headless setup, so the drop zones are verified by
  construction and by the ⇄ button's click path, which does the same move.

## 0.1.0 — 2026-09-28 · milestone: Savante with bankml

The first milestone. bankml is a verified low-bit runtime that answers on the computer at hand, behind its gate,
with a receipt. Savante runs on it, and agents derived from her template can go from a prompt to a token and back,
every construction checked against a published value. Record: `testing/results/0.1.0.txt`, the full gate: every
kernel oracle, the A/Bs and budgets, and every test suite.

### What 0.1.0 is (0.0.1 → 0.1.0)
- **Kernels.** `Q1_0` and `Q2_0_g64`, bit-exact with llama.cpp b11192 on every weight of three real models.
  Ternary runs 9.2–9.9× the reference per token on three threads (0.23–0.25 s, less than the reference's 1-bit
  model). The measured memory floor shows both kernels compute-bound, at the laptop's instruction limit; five
  further variants were measured and rejected, with their code kept.
- **The gate in front of every answer.** Guard, sha256 pin, and `bankml serve` bound to the verified file, with a
  receipt carrying the answer's sha256.
- **Savante.**
  - Interact mode, on loopback: chat, `.prompt`, `.history` with response times, a RAGE search bar, Responses,
    `.memory`, Metrics, the verifier.
  - View mode on the LAN: the stdlib server, read-only, with commitments only.
  - Proof of data without the data: Merkle roots, CIDs, inclusion proofs.
  - Her canon read, never written, and checked against her iNFT ledger before she speaks.
- **Agents.**
  - Derived from her template, each with a ledger whose keccak256 doctrine root matches her binder's construction.
  - THOT manifests that reproduce the spec's test vectors.
  - PostgreSQL publish and load, byte-verified.
  - **The iNFT path (new in 0.1.0):** plan, simulate, unsigned transaction, devnet mint, and load from a token with
    the lineage walked back to the minted generation.

### Added in 0.1.0
- **`ui/chain.py`**, stdlib: ABI encoding and decoding (byte-equal to Foundry's `cast` on selectors and calldata),
  JSON-RPC, the `mintOpenAgent` plan from a THOT bundle (`contentRoot` = the manifest's, `metadataRoot` = keccak of
  the card), `eth_call` simulation with the contract's custom errors decoded, the unsigned transaction and its
  `cast send` line, a devnet-only send (chain 31337; refused elsewhere), `read_token` and `load_from_chain`. A local
  devnet helper starts anvil and deploys `iNFT_7857` from its artifact.
- **THOT generations archived by CID** (`<agent>/thot/<cid>.json`), so a token keeps resolving after the agent evolves.
  Loading walks the parent links from the current generation to the minted one.
- **UI: Agents → iNFT**: plan and simulate, unsigned transaction, mint on the local devnet, load from a token, and
  start a local devnet.
- **Dark mode**: transparent surfaces instead of white panels in interact mode, and translucent panels in view mode.
- **`testing/test_chain.py`** (13 checks, a throwaway anvil). It covers: the deploy from the artifact; simulation as
  minter and the decoded AccessControl refusal; the unsigned transaction; the mint and its read-back; the
  double-mint refusal; the load; the load again after two more generations; the refusal off a devnet; and an
  invalid dimension refused before any call. It skips where anvil or the artifact is absent (CI).
- Documents: usage.md §8e (iNFT), README, TECHNICAL §III.7 and contribution 14.

### The gate record, plainly
Every kernel oracle is bit-exact, and every suite passes (37 unit, 6 CLI, 34 UI, 12 PostgreSQL, 13 iNFT-devnet, guard
agreement 24/24). The 1-bit budget and both A/Bs measured normally. The **ternary decode budget could not be measured
undisturbed in this gate**. With 1.7 GB of RAM available and swap full (other sessions and a browser resident), the
2.13 GB ternary model no longer fits in page cache and each pass re-reads it from disk (about 0.5 GB/s): 4.0–5.9 s
per token for bankml, 5.1–7.3 s for llama.cpp. The code is unchanged since 0.0.3; its undisturbed figures are in the
0.0.5 and 0.0.6 records (0.44–0.45 s per token at 1 thread, 0.24 s at 3).

### What 0.1.0 does not claim
- bankml's own forward pass (P3): answers still come from llama.cpp's arithmetic, behind bankml's gate.
- A public mint: `iNFT_7857` is not deployed on a public chain, its audit is not cleared, and a mint needs MINTER_ROLE
  and the owner's signature. Savante's own verdict on minting remains DEFER.
- Stored bundles: `storageURI` is a `local://` reference until a bundle is stored (the rung stays `referenced`).
- Embeddings: the `vector(1024)` column and its index are ready; nothing computes them yet.
- The laptop's own PostgreSQL still needs its one-time setup (sudo; usage.md §8d).

## 0.0.9 — 2026-09-28

Custom agents, their THOT bundles, and a PostgreSQL connector. Every new cryptographic construction is checked against
a published value. No Rust code changed. Record: `testing/results/0.0.9.txt`.

### Added
- **`ui/agents.py`**: agents derived from the Savante template, without writing to her canon. Each gets `.persona`,
  `.prompt`, an agent card (EIP-721 ∪ ERC-8004 registration-v1, `not_yet_minted`, `derived_from`) and a ledger:
  sha256 and CIDv1 per file, plus a **keccak256 doctrine root** over the same 15 JSON pointers as Savante's.
  - keccak256 is pure Python. It reproduces **Savante's published doctrine root** from her persona, and it equals
    pycryptodome for every length from 0 to 400 bytes.
  - Saving enforces the binder's preflight (ASCII keys, integers only, within ±2⁵³), and a missing doctrine clause is
    a hard error, as the binder specifies.
  - An agent that does not verify against its ledger does not speak.
- **`ui/thot.py`**: THOT manifests to `sagi.thot_manifest/1`.
  - Facets: core `persona` and `prompt`; custom `x-bankml.agentcard`, `x-bankml.history` and `x-bankml.memory`
    (history and memory committed by digest only).
  - The keccak `bundle_root`, the 64-leaf keccak Merkle tree, and the identity (`thot:`, CID, name, contentRoot).
  - Lineage by generation (a facet change gives n+1 with `parent` = the previous CID), `relations` to Savante's
    bundle, and a real git locator per agent.
  - **Checked against the spec's three test vectors** (Savante @1fcca89, Jaimla @8b57ccf, LuvAI @0c1eef7): bundle
    root, Merkle root and identity CID all equal. Savante's current generation-9 manifest verifies with no findings.
- **`ui/connectors.py`**: PostgreSQL through `psql` (stdlib, no driver).
  - `publish`: the public parts by default; `.history`/`.memory` lines only with an explicit opt-in.
  - `load`: refused unless every restored byte matches the THOT manifest.
  - `vector(1024)` embeddings column with a DiskANN (pgvectorscale) or HNSW (pgvector) index.
  - Values travel as COPY data into a temporary table and never enter SQL text.
- **UI**: an **Agents** tab (use, derive, edit and re-ledger, the ledger check, the THOT bundle, publish and load),
  and the chat, `.history`, `.memory`, Responses and Metrics follow the agent in use.
- **Tests**:
  - `testing/test_ui.py` grows to 34 checks: the THOT spec vectors and lineage, keccak vectors and the cross-check, the doctrine root against Savante's,
    derive/verify/save, preflight refusals, tamper detection, and path confinement.
  - **`testing/test_connectors.py`** (12 checks) runs against a **throwaway PostgreSQL 16 cluster** (initdb in a temp
    dir, pgvector): the schema, both publish modes, a byte-exact load, refusal of a tampered prompt and a tampered
    history line, an injection string stored as data, and the THOT generation step. It skips where PostgreSQL 16 or
    pgvector is absent (CI).

### Not yet
- The laptop's own PostgreSQL has no role for the operator yet. The one-time setup needs sudo and is in usage.md §8d.
- Embeddings are not computed (the column is ready).
- The chain side (loading an iNFT, preparing a mint) is 0.1.0. The house iNFT contract (`iNFT_7857`, `mintOpenAgent`)
  is not deployed on any public chain, its audit is not cleared, and minting needs MINTER_ROLE.

## 0.0.8 — 2026-09-28

Savante's memory, made searchable, measurable and provable, with the documents brought up to date. No Rust code
changed. Record: `testing/results/0.0.8.txt`.

### Added
- **`.history` tab with a ragebar.** Every exchange, newest first: its time, session, time to first token, response
  time, tokens, and whether the answer matches its receipt. Before this it showed nothing until a button was pressed.
  The search bar (the GATERAGE ragebar's look) ranks exchanges as you type, using mindX's RAGE `rage.py` when present
  (`RAGE_PATH`) and the same BM25 built in otherwise.
- **Responses tab**: ⤒ first, ▲ previous, ▼ next, ⤓ latest through every answer. **📋 copy** puts the answer on the
  clipboard, **➕ save to .memory** keeps it, and **🔏 proof** gives its inclusion proof.
- **`.memory`** (`savante.memory`, JSONL, outside the canon): notes typed or saved from responses, with their source.
  A side-panel switch appends them to the system prompt (newest first, 2,400 characters at most), labelled as the
  operator's notes, not evidence. The footer names how many went in.
- **Metrics tab**, computed from `.history`: time to first token, response time, prefill and writing speed (n, median,
  p90, mean, min, max), a bar per exchange, receipt-hash agreement, the last 25 exchanges, and each row's timing source
  (the press of Send, or the receipt for older records).
- **Proof of data without the data.** Each `.history`/`.memory` line's sha256 is a leaf of a Merkle tree, and the file
  has a sha256 and a CIDv1 (raw, sha2-256, base32: the construction of Savante's iNFT ledger; equal to mindX
  `rage.cid_v1_raw`). The view page shows only these commitments. An inclusion proof lets one exchange be checked
  against the root without disclosing the rest.
- **`testing/test_ui.py`**: 16 offline checks of the UI's data layer. They cover the CIDs, commitments, every
  inclusion proof, tamper and cross-record failure, the search, the metrics and `.memory`, and the view server
  (commitments present, content absent, traversal 404, POST 405). Run in CI and in the gate.

### Changed
- **TECHNICAL.md** brought to 0.0.8:
  - The Abstract now includes the threads, the floor and P0.
  - The Thesis gains the principles stated on 2026-09-28, quoted and dated: incremental improvement with the oracle
    in the loop, the machine at hand, and proof of data while keeping the data.
  - Contributions 1, 5 and 8 are updated, and 12–13 are new.
  - New §III.6 (proof of data) and §IV.5 (the floor, and the five rejected variants).
  - §IV.4 no longer says bankml cannot answer.
  - Future work drops the 1-bit layout port (tried and rejected) and adds connectors, THOT and custom agents.
- **README**: status badge and phase table (P0 done; P2 with the 8B oracle and threads; UI row); the budget numbers
  (0.23–0.25 s, about 4 tok/s); the Savante section covers the new tabs and proofs.
- **usage.md**: the new tabs, `.memory`, and §8a on proofs (with a hand check in Python).
- View: the live log is capped at 60 % of the window, so the other panels stay in reach.

## 0.0.7 — 2026-09-28

Savante's UI, made to be watched and used: a LAN view, a professional look, response times, layout you can arrange,
and a guide. No Rust code changed (the kernels and their 0.0.6 gate records stand). Record: `testing/results/0.0.7.txt`.

### Added
- **View mode on the LAN: `ui/view.py`**, the Python standard library, not Gradio. The installed Gradio 3.37 has
  path-traversal bugs that let a client read files from the host (e.g. CVE-2023-51449, fixed in 4.11), so it stays
  on loopback. The view server has four fixed GET routes: the page; `/api/state`; `/api/result?name=`, for a name the
  results directory lists; and `/savante.png`. Every other path gets 404, POST gets 405, the page renders all data as
  text, and a strict Content-Security-Policy applies. It shows the live testing, the release records, CI, the
  laptop's load and swap, `bankml serve`'s verification and Savante's office and ledger. Light and dark themes.
- **Modular layout.** In view mode every panel can be dragged by its title to a new position and resized from its
  corner; the layout is kept per browser, with a *reset layout* button. In interact mode the chat and side panels
  resize, and the side panel drags to either side of the chat.
- **A response timer** that starts at the press of Send and ticks every second: *reading the prompt (prefill)*, then
  *writing · first token at N s*. Each answer's footer reads `⏱ sent HH:MM:SS · first token N s · answered in N s`.
- **Response times in `.history`**: every record carries `sent_at` and `answered_at` (ISO 8601 with milliseconds and
  offset), `first_token_s` and `response_s`, next to the receipt.
- **`usage.md`**, the full guide: what runs where, setup, verifying the model, `serve`, both modes, `.history`,
  receipts and how to check an answer, the canon and its ledger, testing, troubleshooting, and a reference of
  commands, ports and variables. The README gains a four-step *Using Savante*.

### Changed
- Interact mode uses one light professional theme: bordered panels and forced text contrast over Gradio's styles.
  The `bankml serve` panel was a raw JSON block that overflowed the side column; it is now a compact card that wraps
  inside its border.
- The Gradio queue runs 4 events at once, so the timer ticks while an answer streams.

## 0.0.6 — 2026-09-28

bankml answers on this laptop. The answers come through the reference engine (P0), behind the gate, with a receipt,
in a Savante UI that anyone can watch.

### Added
- **`bankml serve`** (P0, `serve.rs`, std only) is a loopback HTTP gateway in front of llama.cpp b11192's
  `llama-server`:
  - It starts only after `verify` passes (guard, then the sha256 pin to `FORK.json`).
  - The upstream must be serving the verified file. Either bankml launches it (`--spawn LLAMA_SERVER`), or a running
    server's `/props` must name the same canonical path. Otherwise it refuses.
  - Every `/v1/chat/completions` answer, streamed or not, carries a `bankml_receipt`. The receipt holds the version,
    the engine, the model sha256, the guard verdict, the tokens, the time to first token, the wall time and the
    **sha256 of the answer text**, so a client can check that what it shows is what the verified model wrote.
  - `GET /bankml` reports the verification.
  - What it is not: bankml's own forward pass (P3). The arithmetic is ggml's.
- **`ui/savante.py`** is the Savante chat in Gradio (3.x or newer), built from the Hugging Face template
  PYTHAI/savante. It has two modes:
  - **interact** (the operator): the chat, the `.prompt` picker, `.history` and the offline verifier. The `.prompt`
    choices are the persona's `system_prompt` (canon, ledgered, the default), `sAGI.prompt` (canon facet, ledgered) or
    the Space's `Savante.prompt` (not ledgered), each shown with its sha256. `.history` is a JSONL file outside the
    canon, and the last session reloads on start. Each answer is labelled a draft and carries its receipt.
  - **view** (anyone watching, read-only): the live testing log (`testing/live.log`, refreshed every 2 s), every
    release's gate record, CI status, the laptop's load, memory and swap (so a disturbed measurement shows as one),
    and Savante's office and ledger. It has no chat, no `.history`, and runs no commands.
- **iNFT compatibility.** Savante's canon is read and never written, not even a Python cache. At start the UI re-hashes
  every file that `savante.commitments.json` commits to: the nine artefacts, the card, the image and the thot bundle.
  If the persona does not verify, the UI refuses to speak as Savante. The Integrity tab shows the ledger, and
  `bind/savante_verify.py` runs from a button. Nothing mints; the card's status stays `not_yet_minted`.
- 2 new unit tests (the JSON reader and HTTP bodies) and 2 end-to-end tests of `serve` against a mock llama-server:
  the receipt and the answer hash, refusal of a wrong pin, and refusal of an upstream serving a different file.

### Measured on the laptop (Ryzen 3 3200U), the real canon and model
- `bankml serve` verified Bonsai-8B Q1_0 (`284a335a…`, the fork's pin) and the running llama-server serving it.
- A Savante turn under the persona's system prompt: 316 prompt + 28 completion tokens in 125 s, with the first token
  at 113 s. Prefill runs at 2.8 tok/s in llama.cpp. The answer's sha256 matched its receipt.
- The canon ledger verifies 12/12; `savante_verify.py` exits 0 (APPROVE); `~/savante` is unchanged afterwards.

## 0.0.5 — 2026-09-28

The third performance release, on the ternary kernel, with the testing shown live. **No kernel changed**: three
experiments were bit-exact, and none was reliably faster. Gate record: `testing/results/0.0.5.txt`. Every oracle is
bit-exact. Ternary stays at 9.2–9.6× ggml over a whole token, and 1-bit at parity.

### Measured and not adopted (bit-exact, not reliably faster), code in `testing/experiments/`
- **`Q2Packed`**: the ternary weights repacked once at load time, as llama.cpp's `repack.cpp` does for other types.
  Each quad of blocks becomes 64 contiguous code bytes followed by its four f16 scales: the same 72 bytes, with the
  arithmetic unchanged. It measured 1.00–1.075× in the first run and 0.90–1.05× in the release gate, on the real
  ffn_gate, ffn_down and output tensors. That is within noise.
- **Two-row decode tile**, loading the activation once for two rows: **0.80×**. Register pressure costs more than the
  shared loads save.
- **Software prefetch** of the weight row, 256–1024 bytes ahead: within ±3 % at 1 and 3 threads. The hardware
  prefetcher already follows a sequential stream.

### Added
- **`testing/live.sh`** and a live log: the gate and every experiment append to `testing/live.log`, which the UI's
  view mode shows as it runs (branch `ui`, next release).

### What the oracles and the stopwatch say after three performance releases
At 3 threads the ternary kernel moves about 9 GB/s against a 17 GB/s floor. Going from 1 to 3 threads gives 1.8×, and
the laptop has two physical cores. So it is bound by compute per core, not by memory. It already does one multiply
per weight, and moving the bytes around changes nothing measurable. On this Zen+ core, both kernels are at their
instruction-throughput limit for bit-exact results. The gains still available lie elsewhere:
- the forward pass (P3), which turns the 9.5× matmul lead into tokens;
- a Zen3 or AVX-512 core (the node).

## 0.0.4 — 2026-09-28

Where the time goes, and a faster 1-bit prefill. Gate record: `testing/results/0.0.4.txt`. Every oracle is bit-exact.
Two measurements in the gate were disturbed by other load on the laptop (swap full), so the record adds a re-run.

### Added
- **`bench_memory_floor`** (`par.rs`) measures streaming read bandwidth on the laptop at 1–4 threads: **14.9–17.2
  GB/s**. Streaming one token's weights therefore takes at least **0.12–0.14 s** (ternary) and **0.06–0.07 s** (1-bit).
  Against that floor, bankml's ternary matmuls (0.23–0.27 s at 3 threads) are about 2× above it, and the 1-bit ones
  (0.34–0.37 s) about **5.5× above it**. Both kernels are compute-bound, not memory-bound, which says where the
  remaining work is.
- **`q1_0::mat_mul_act` / `mat_mul_act_par`**: a 1-bit prefill over prepared `Q8Act` columns, using the selection
  kernel in a 1×4 tile. The ±1 expansion is shared by the four columns, and the scales are already f32, so nothing is
  converted once per column. A column that holds q = −128 falls back to the wrapping kernel. Measured against ggml's
  per-pair kernel (1024×4096 × 32 columns): **1.27–1.33× median** (8.25 against 10.09–10.28 ns per block·column,
  min). The q8-byte tile it replaces measured 1.08–1.18×. Bit-exact against the scalar model of ggml on 1,500 random
  cases, and against ggml itself in `ab_vs_ggml`.

### Measured and not adopted (the oracle said same bits; the stopwatch said no)
- **Two 1-bit blocks per iteration**, with their FMA chains interleaved: 1.00× on the GEMV, 1.05× on an L1-resident
  row. The kernel is not bound by latency.
- **A "pair-order" 1-bit kernel** that replaces a multiply (`madd`) with a 128-bit add, on the theory that Zen+'s
  single integer-multiply pipe is the limit. It was bit-exact and **0.84×** on decode: the cross-lane extract and the
  widening cost more than the multiply they save. The source is in `testing/experiments/q1_0_pair_order.rs` for anyone
  who wants to try it on another core.
- (Already on record from 0.0.1: 2- and 4-row decode tiles, 0.81–0.99×.)

1-bit decode stays at parity with ggml (0.95–1.03× over the whole token). Every bit-exact variant tried so far runs at
about ggml's speed on this core.

### Testing
- The gate now also runs `bench_q1_0_prefill_act` and `bench_memory_floor`.
- `testing/experiments/` holds the code of measured-and-rejected kernels.

## 0.0.3 — 2026-09-28

Threads, and more oracle evidence. Each speed figure below was measured with the oracles passing on the same code.
The full record is in `testing/results/0.0.3.txt`, written by the new release gate.

### Added
- **`par::Pool`**: a zero-dependency persistent thread pool. The caller's thread is worker 0. Each matmul wakes the
  pool instead of spawning threads: 12.6 µs per call against 88.8 µs for `thread::scope`, at 3 threads on the dev box.
  That is about 3 ms per token instead of 22 ms. The row scheduler hands out 16-row chunks from an atomic counter.
- **`mat_vec_par` / `mat_mul_par`** for `Q1_0` and `Q2_0`. Every row goes through the same single-thread kernel, so
  the output bits do not depend on the thread count. Unit tests check this at 1–4 threads, and the whole-model budgets
  check it against ggml.
- **An oracle on the real 8B 1-bit model**: `oracle_ggml_b11192_real_bonsai_8b_q1_0`. All 254 `Q1_0` tensors of
  Bonsai-8B (8,188,239,872 weights) dequantize bit-exact, 762/762 q8_0 rows are byte-exact, and 762/762 dot products
  are bit-exact against ggml AVX2 and generic. 0.0.2 covered only the 1.7B file for 1-bit.
- **`decode_budget_q1_0`**: one token's worth of all 253 `Q1_0` matmuls of Bonsai-8B, against ggml's kernel on the same
  pool and scheduler, at 1–4 threads.
- **`testing/release_gate.sh`** runs the unit tests, clippy, both guard checks, every oracle, both A/Bs and both
  budgets, and writes `testing/results/<version>.txt`. A kernel change counts only if every oracle still passes.

### Measured (dev box, Ryzen 3 3200U, 2 cores / 4 threads; all 253 matmuls of one token; min s/token over two runs)
| threads | ternary: ggml → bankml | 1-bit: ggml → bankml |
|---:|---:|---:|
| 1 | 4.22–4.65 → **0.45–0.47** (9.4–9.9×) | 0.62–0.67 → 0.61–0.62 (1.00–1.07×) |
| 2 | 2.65 → **0.31** (8.7×) | 0.45–0.46 → 0.43–0.44 (0.93–1.13×) |
| 3 | 2.27–2.36 → **0.23–0.25** (9.5–9.9×) | 0.34–0.35 → 0.34–0.35 (0.98–1.00×) |
| 4 | 2.11 → **0.23** (9.4×) | 0.34–0.35 → 0.36 (0.96–0.97×) |

bankml's ternary matmuls now take less time per token than ggml's **1-bit** ones: 0.23–0.25 s against 0.34–0.35 s at 3
threads, although the ternary file is twice the size. The matmul-only ceiling for ternary rises from 2.77 tok/s (0.0.2,
which spawned threads for each call) to 4.0–4.4 tok/s. The 1-bit kernel stays at parity with ggml. The 1-thread medians
are noisy (ternary 0.62 s against 0.47 s min): the model does not fit in page cache alongside everything else in 5.8
GB of RAM.

### Testing
- A new **`testing/`** folder holds every test that lives outside the modules. It contains the release gate, the
  oracle generator, the guard harnesses (moved from `tools/`) and **`testing/cli.rs`**, a new end-to-end suite that
  runs the `bankml` binary. The suite covers verdicts and exit codes, a hostile header, pin and verify, and runs as a
  cargo integration test.
- **`testing/results/`** holds each release's record. `testing/README.md` lists every test and where it lives.

### Changed
- The ternary budget and the A/B harness use `par::Pool` for both engines instead of spawning threads per call. The
  ggml side runs on the same scheduler, so the A/B compares kernel with kernel.

## 0.0.2 — 2026-09-28

An audit of 0.0.1. The kernels are unchanged and re-proven: the oracles again pass on all 8,188,239,872 ternary and
1,719,904,256 1-bit weights against llama.cpp b11192. The fixes below all concern untrusted input, or an API that let
safe code reach unsafe code with the wrong sizes.

### Fixed
- **The guard crashed on nested arrays.** GGUF arrays of arrays were parsed by recursion with no limit, so a 24 MB
  header aborted `bankml guard` with a stack overflow (exit 134) instead of refusing the file. Nesting beyond
  `MAX_ARRAY_DEPTH` (64) now refuses, with the reason given.
- **The guard's KV size could overflow.** `kv_f16_bytes_per_token` multiplied header integers in `i128` without
  checking. In release builds the value wrapped with no error and the verdict was `play`; debug builds panicked. An
  overflow now refuses, with the reason given. The Python guard, which uses big integers, still plays these files; the
  difference is listed in `gguf.rs` with the other deliberate differences.
- **Soundness: activation fields were public.** `q1_0::Q8Act` and `q2_0::Q8Act2` exposed `n`, `d`, `sum` and the
  other fields that the AVX2 kernels read through raw pointers. Safe code could set `n` beyond the buffers and cause
  an out-of-bounds read. The fields are now private, with read-only accessors (`n()`, `qs()`, `d()`).
- **Tensor spans were unchecked.** `Mmap::tensor` and `tensor_bytes` truncated a 128-bit element count to 64 bits
  and used unchecked arithmetic. `tensor_bytes` also allocated whatever size the header claimed before it read the
  file. A new function, `gguf::tensor_span`, checks the type, requires whole blocks per row and rejects overflow.
  `tensor_bytes` now checks the span against the file size before it allocates.
- **`bankml pin` with an unreadable `--fork` file** said "no sha256 record: unpinned, refused" (exit 2). It is an I/O
  error, and now says so (exit 1).

### Added
- `bankml::verify` and `bankml verify FILE --fork FORK.json [--json]` run the guard and then the pin as one gate: the
  check that P0 puts in front of every answer. `--json` prints the verdict, the model sha256 and the bankml version.
- `bankml version` and `bankml::VERSION`.
- CI (GitHub Actions) runs the build, the unit tests, `clippy -D warnings`, the Python guard suite and the Rust-vs-Python
  guard agreement check on every push.
- 4 new unit tests (31 in all): nested arrays, KV overflow, tensor spans, and `verify`.

### Changed
- The `Q8Act`/`Q8Act2` fields are no longer public (see above). This is an API break, allowed while the crate is
  `publish = false` and 0.0.x.

## 0.0.1 — 2026-09-26

First public cut: the P1 GGUF guard and sha256 pin, and the P2 `Q1_0` and `Q2_0_g64` kernels, all bit-exact against
llama.cpp b11192.
