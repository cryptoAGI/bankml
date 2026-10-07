# The bankML persona as a template

*2026-10-06. It describes `sAGI/personas/bankml.persona` as of this pull request, which is built on PR #1's version of that file. The tagline is the project owner's.*

**cryptoAGI, powered by minaiml, made with LUV.** `bankml.persona` is how the bankML engine speaks of itself. The file keeps every field it had, so the doctrine root stays `0x4d5ef2d9…e7d5`. It adds five blocks so the file can also be used as a template:

| block | what it holds |
|---|---|
| `template_spec` | how to derive a persona from this one: which fields are doctrine, which are editable, which shapes the code depends on, and where this file stands as a sAGI facet |
| `conformance` | bankML against cypherpunk4096, one commitment at a time: met, partial, not yet, or not applicable, each with the files that show it |
| `algorithms` | every hash function used for pins, receipts, CIDs, the doctrine root and the history Merkle tree, so a successor can be named rather than assumed |
| `provenance` | who made it, the tagline, and what "powered by minaiml" and "made with LUV" mean, with sources |
| `licence` | `MIT OR Apache-2.0`; key handling `GPL-3.0-only` (none yet) |

It also adds three sourced `awareness` facts, which `ultimate-bankml-ui` reads to bankML as "What I know about myself":
- the guard's minaiml origin;
- receipts are unsigned;
- bankML does not carry the cypherpunk4096 mark.

## Deriving a new persona

1. **Copy** the file to `<slug>.persona`. The slug must match `[a-z0-9_]{1,40}`.
2. **Write the doctrine.** These are the 15 pointers in `sAGI/agents.py` `DOCTRINE_POINTERS`: `/persona /name /source /format /system_prompt /mantra /oath /bdi/beliefs /skills/primary /skills/taxonomy /skills/defer_triggers /skills/validation /safety /embodiment /token/intelligence/tool_allowlist`.
   - Give each one a source in your own project's documents.
   - The keccak256 root is computed over these pointers.
   - After adoption the root is fixed: `agents.save()` refuses any change to them. To change one, adopt a new agent.
3. **Fill the editable fields:** `kind`, `doctrine`, the `*_source` fields, desires and intentions, capabilities, `awareness.facts`, `task`, `token.public_metadata`, `runtime`, `provenance` and `licence`.
4. **Keep the shapes the code reads.** `template_spec.code_contract` lists them: `name`, `mantra` and `system_prompt` as strings; `awareness.facts[].fact`; `token.public_metadata.description`.
   - `self_block.fields` names what the console measures. A persona for another engine names its own fields, and leaves a value null when it was not measured.
5. **Reset `conformance`.** Set every status to `not assessed` and empty its evidence. Conformance belongs to a subject at a commit, so it is never inherited.
6. **Set `token.bindings` to null.** Only a mint pipeline writes them.
7. **Check:**
   - `agents.preflight(p)` returns nothing: every key is ASCII, and there are no floats.
   - `agents.doctrine_root(p)` computes.
8. **Install** with `agents.adopt(p, source=…)`. `sAGI/thot.py` can then bind the agent into a `sagi.thot_manifest/1` bundle.

Do not name a block `template`. `agents.derive()` writes a `template` string into every persona it derives.

## What conformance means here

[cypherpunk4096](https://github.com/cypherpunk4096/standard) (CC0, read at `030f5c3c`) is **all-or-nothing**. An artifact carries the mark only if it meets every commitment (C1 to C5) and ships none of the patterns the standard does not accept.

**bankML does not carry the mark, and the draft says so.**

The standard is written for units deployed on chain:
- C1 is a deterministic CREATE2/CREATE3 address.
- C3 is source verification on a public explorer.
- C5 covers signature surfaces in contracts.

bankML deploys nothing, so for C1 and C3 the block says *not applicable* rather than *met*. C5 is met only vacuously, because receipts are unsigned. The standard's reference implementation, SCIEN·TIFIC, meets C5 the same way.

### Claimed, with evidence

- **No external crate.** The Rust core and the C API have none (`Cargo.toml`).
- **Nothing to custody.** bankML holds no keys and serves on loopback.
- **llama.cpp's bits.** Kernels, tokens, templates and samplers reproduce llama.cpp b11192's output. A gate record is published for each release (`docs/oracles.md`, `testing/results/`).
- **Hashes only, no signatures.** sha256 (an in-crate implementation, checked against FIPS vectors) for pins, receipts and CIDs, and keccak256 for the doctrine root. No signature surface exists.
- **No floats.** The persona and ledgers refuse them (preflight P2).

### Not yet true

- **CDN scripts.** The public pages load code from other hosts:
  - `hf/space/index.html` loads scripts from `deltaverse.pythai.net`.
  - `docs/index.html` loads cdnjs and Google Fonts.

  This is on the standard's not-accepted list.
- **Gradio.** `sAGI/savante.py` depends on it.
- **Offline rebuild.** No byte-identical offline rebuild has been recorded.
- **Rounding.** Served rates are rounded to three decimals, and no display rule documents that.
- **The word "verified".** cypherpunk4096 reserves it for green-checkmark explorer verification. bankML's doctrine uses it in 6 of its 15 clauses. Rewording it changes the doctrine root, so the owner has to decide.
- **Signed receipts** are planned but not built. When they come, the signature has to be bytes, its scheme has to be named in `algorithms`, and a CP2048-QR tier has to be declared for that rail.
- **Facet bundle.** bankML's own persona is not a full sAGI facet bundle: there is no `.agent` and no `.model` file, and no THOT manifest.
