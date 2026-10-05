# `bankML/train/` — mindXtrain in Rust, one verified stage at a time

## Summary

`bankML/train/` is bankML's training module. It ports mindXtrain (github.com/Professor-Codephreak/mindXtrain,
continued at huggingface.co/PYTHAI/mindXtrain; Apache-2.0), which imprints a persona onto a model and proves it with a
recall gate:

```
author (persona + exchanges → a chat-JSONL script) → imprint (LoRA SFT) → probe (the same inquiries before and
after) → score (did the voice move toward the persona, and did the utterances change) → classroom → boardroom
```

Each stage reproduces mindXtrain's Python exactly, checked by an oracle that runs mindXtrain's own functions
(`testing/train_oracle.py`), before anything new is built on it. Two stages are ported: **author** (`script.rs`, from
`mindxtrain/data/scripts.py`) and **score** (`imprint.rs`, the lexical path of `mindxtrain/eval/imprint.py`). The
module is a library API (`bankml::train`); it has no CLI command yet. It is exercised by its unit tests and its two
oracles in the release gate.

## Technical usage

### `mod.rs` — the stage registry

```rust
pub const STAGES: &[(&str, &str, &str)]
pub fn py_json_str(s: &str, out: &mut String)
```

- `STAGES` lists each stage of the proof loop as (stage, module, status). Today: `author` → `script` (ported),
  `imprint` (not yet: LoRA SFT needs a backward pass), `probe` (not yet: needs the Llama architecture and adapter
  loading in the forward pass), `score` → `imprint` (ported), `classroom` and `boardroom` (not yet). A new stage is a
  new module and one line there.
- `py_json_str` writes a JSON string as Python's `json.dumps(…, ensure_ascii=False)` does: `\"`, `\\`, `\n`, `\r`,
  `\t`, `\b`, `\f` escaped, other control characters as `\u00XX`, everything else as is.

### `script.rs` — the author stage

```rust
pub struct Persona { pub name: String, pub system_prompt: String, pub voice_examples: Vec<String> }
pub struct Exchange { pub user: String, pub assistant: String }

pub fn persona_from_json(raw: &Json) -> Result<Persona, String>
pub fn system_prompt(p: &Persona) -> String
pub fn script_jsonl(p: &Persona, exchanges: &[Exchange], seed_voice: bool) -> String
pub fn training_params(rows: usize) -> (u32, u32, u32)
```

- `persona_from_json` is `persona_from_dict`. It reads the same keys, clean-room, from any persona JSON object:
  - the name is the first truthy value of `name`, `persona`, `id`, `title` (default `actor`);
  - the system prompt is the first truthy value of `system_prompt`, `system`, `description`, `bio`, `summary`,
    `prompt`;
  - the voice examples are collected from `voice_examples`, `examples`, `utterances`, `samples`, `voice`: a list adds
    its strings, numbers and booleans as Python's `str()` writes them, and a string adds itself.
- `system_prompt` is `persona_system_prompt`: the persona's own prompt stripped as Python's `str.strip()` strips, or
  `You are <name>. Stay in character and answer in your own voice.`
- `script_jsonl` is `build_script_rows` + `write_script_jsonl`: one row per exchange, then, with `seed_voice`, one
  row per voice example answering `Say something as <name>.`. Each line is byte-identical to Python's
  `json.dumps(row, ensure_ascii=False)`.
- `training_params` is `derive_training_params`: (epochs, grad_accum, per_device) by row count — 24 epochs up to 8
  rows, 16 up to 32, 8 up to 128, 4 with grad_accum 2 up to 512, then 2 with grad_accum 4. Small scripts must overfit
  to imprint.

```rust
use bankml::serve::Json;
use bankml::train::script::{persona_from_json, script_jsonl, training_params, Exchange};

let p = persona_from_json(&Json::parse(r#"{"name": "Savante", "voice": "Verification beats permission."}"#).unwrap())?;
let jsonl = script_jsonl(&p, &[Exchange { user: "Who are you?".into(), assistant: "Savante.".into() }], true);
let (epochs, grad_accum, per_device) = training_params(jsonl.lines().count());
```

### `imprint.rs` — the score stage

```rust
pub fn default_inquiries(name: &str) -> Vec<String>
pub fn jaccard(a: &str, b: &str) -> f64
pub fn round4(x: f64) -> f64
pub struct ImprintReport { pub before_voice: f64, pub after_voice: f64, pub imprint_delta: f64, pub shift: f64,
                           pub method: &'static str, pub imprinted: bool }
pub fn score(before: &[String], after: &[String], baseline: &[String]) -> ImprintReport
```

- `default_inquiries` returns mindXtrain's five persona-agnostic recall probes.
- Tokens are `[a-z0-9']+` of the Unicode-lowercased text; similarity is token Jaccard (0 when either side has no
  tokens).
- The voice score is the mean, over utterances, of the best similarity to any voice example. The shift is the mean
  `1 − similarity(before, after)`. Means are summed left to right, then divided once, as Python's
  `sum(xs) / len(xs)`.
- Every figure is rounded to 4 decimals as Python's `round` rounds (`round4`: the exact binary value, half to even).
- The verdict is `imprinted = delta > 0 and shift > 0`, on the unrounded values. `method` is `lexical`, or `none`
  when there are no utterances or no voice examples.

## How it is verified

`testing/train_oracle.py` runs mindXtrain's own Python with mindXtrain's interpreter and writes the cases to
`.models/oracle-train/`:

```sh
~/mindxtrain/.venv/bin/python testing/train_oracle.py [~/mindxtrain]
```

- **`oracle_train_script`** (`script.rs`, in the release gate) reads `script.jsonl`: every persona in mindX and
  cryptoAGI plus edge cases (nameless, whitespace prompt, control characters and quotes, falsy name, a number as
  name), each with and without `seed_voice`. It requires the name, the system prompt, the JSONL text byte for byte,
  and the epochs. **84 of 84 scripts byte-identical** to `scripts.py` ([oracles.md](../oracles.md), 0.2.13).
- **`oracle_train_imprint`** (`imprint.rs`, in the release gate) reads `imprint.jsonl`: 3,000 randomized
  before/after/baseline sets, including empty utterances and non-ASCII words. It requires every field of the report.
  **3,000 of 3,000 reports identical** to `score_imprint`.
- Unit tests: `a_nameless_persona_gets_the_synthesised_prompt` and `jaccard_and_rounding_as_python` (including
  `round4(0.03125) == 0.0312`, a tie at the fourth decimal).

## Advantages and efficiency

- **Identical, not similar.** Each stage is proved against mindXtrain's own functions on real personas and thousands
  of random cases before the next is built. A Rust stage can replace the Python one without changing a verdict.
- **No Python, no packages.** The author and score stages run inside bankML with no interpreter and no external
  crate. The JSON is bankML's own (`serve::Json`), and Python's string escaping, truthiness, `str()`, `strip()` and
  `round` are reproduced explicitly where they matter to the bytes.
- **Lexical path only, on purpose.** mindXtrain prefers sentence-transformer cosine when that package is installed;
  the dependency-free lexical path is the one ported, so the score needs no model.
- **Rust practice visible in the code.** No `unsafe` in the module. Malformed input returns `Err` with a reason
  (`a persona is a JSON object`, `persona field "name" is not text`). Sets are `BTreeSet`, so token sets are
  deterministic. The oracles fail closed: every case must match. The toolchain is pinned to Rust 1.99.0 in
  `rust-toolchain.toml`.
- **Where it goes next** ([TODO.md](../TODO.md), 0.7.0): the probe stage (PEFT adapters loaded from safetensors and
  merged, then recall probing with transformers' decoding rules, against mindXtrain's `probe_recall`); the classroom
  and boardroom verdicts, the dojo tie-break, the feedback ledger and the receipts; **LoRA training on the CPU** (a
  backward pass and AdamW for the 135M–0.6B imprint recipe, gradients checked against PyTorch f32); then one
  generation end to end in Rust: author → imprint → probe → score → verdict → GGUF.

## Limitations

- Only the author and score stages are ported. Imprint (LoRA SFT), probe, classroom and boardroom are not.
- The score stage ports only the lexical path; the sentence-transformer cosine path is not ported.
- A persona's identity fields are expected to be strings. A number in a voice list is written as Python writes an
  int, so a float literal such as `3.0` would differ, because bankML's JSON keeps numbers as f64.
- No CLI command; the module is reached through the library.

## See also

- [oracles.md](../oracles.md) — 0.2.13, the mindXtrain oracles
- [TODO.md](../TODO.md) — 0.7.0, mindXtrain in Rust end to end
- [usage.md](../usage.md) — testing and the release gate
- [forward.md](forward.md) — the forward pass the probe stage will run on
