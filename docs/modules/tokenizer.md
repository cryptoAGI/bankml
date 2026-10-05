# `bankML/tokenizer.rs` — byte-level BPE, token-identical to llama.cpp b11192

This page also covers `bankML/unicode_letters.rs`, the tokenizer's letter table.

## Summary

`tokenizer.rs` turns text into token ids exactly as llama.cpp b11192's `llama_tokenize(…, add_special = false,
parse_special)` does, for GGUF vocabularies of the `gpt2` model (byte-level BPE) with one of two pre-tokenizers:
`qwen2` (Qwen3, Bonsai) and `smollm` (SmolLM2 and mindX's `mindx-genN`). It also turns ids back into bytes, and
collects the tokens that end generation as llama-vocab.cpp does.

It follows llama.cpp's three steps:
1. **Special tokens first.** CONTROL (3) and USER_DEFINED (4) tokens are cut out of the text, longest first. Without
   `parse_special` only USER_DEFINED ones are.
2. **Pre-tokenize** each remaining span. `qwen2`'s pattern is written out alternative by alternative (ECMAScript
   semantics). `smollm` runs two passes: digits cut out one by one, then GPT-2's pattern inside each piece.
3. **Byte-level BPE** on each piece: the adjacent pair with the lowest merge rank is merged, the leftmost on ties.

`unicode_letters.rs` holds Unicode general category L (Lu, Ll, Lt, Lm, Lo) as 622 sorted inclusive ranges, generated
from Python's `unicodedata` (Unicode 13.0.0). It exists because `\p{L}` is category L, and Rust's
`char::is_alphabetic` is the Alphabetic property, which is not the same set.

Callers: the native engine (`native.rs`: prompts, grammar pieces, end-of-generation ids); `bankml serve --native`
(`POST /tokenize`, and every chat request); `bankml tokenize` and `bankml generate` (`main.rs`); and `native.rs`'s
model check, which calls `check_gguf`.

## Technical usage

```rust
pub fn check_gguf(h: &crate::gguf::Header) -> Result<(), String>
pub enum Pre { Qwen2, Smollm }

impl Tokenizer {
    pub fn from_gguf(path: &Path) -> Result<Self, String>
    pub fn new(tokens: Vec<String>, mut types: Vec<i32>, merges: &[String]) -> Result<Self, String>
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32>
    pub fn token_bytes(&self, id: u32) -> Vec<u8>
    pub fn piece(&self, id: u32) -> Vec<u8>
    pub fn eog_ids(&self, kv_ids: &[u32]) -> Vec<u32>
    pub fn n_tokens(&self) -> usize
    pub fn token_type(&self, id: u32) -> Option<i32>
    pub fn id(&self, text: &str) -> Option<u32>
}

pub fn pretokenize(text: &str) -> Vec<&str>          // qwen2
pub fn pretokenize_smollm(text: &str) -> Vec<&str>   // smollm
pub fn is_letter(c: char) -> bool                    // unicode_letters.rs
```

- `from_gguf` reads `tokenizer.ggml.{model,pre,tokens,token_type,merges}` straight from the GGUF header through
  bankml's memory map, and refuses any vocabulary other than `gpt2/qwen2` and `gpt2/smollm`.
- `new` applies llama-vocab.cpp's type overrides by text: every end-of-generation name and each fill-in-the-middle
  role's token become CONTROL; gpt-oss's four channel markers become USER_DEFINED. So `</s>`, NORMAL in the Qwen3
  file, is parsed as a special token.
- `token_bytes(id)` is the detokenizer without special tokens: a control token gives nothing.
- `piece(id)` is `token_to_piece(…, special = true)`: the bytes a grammar matches (`grammar::Vocab` is built from it).
- `eog_ids(kv_ids)` is llama-vocab.cpp's end set: the FIM pad / repo / separator tokens, every name in its end-of-turn
  list, and the GGUF's own eos / eot / eom ids, with b11192's two exceptions.

From the command line and over HTTP:

```sh
echo -n "Hello world" | bankml tokenize .models/Bonsai-8B-Q1_0.gguf               # {"tokens": [...]}
echo -n "<|im_start|>" | bankml tokenize .models/Bonsai-8B-Q1_0.gguf --no-special
curl -s 127.0.0.1:PORT/tokenize -d '{"content": "Hello", "parse_special": true}'  # serve --native
```

`parse_special` defaults to true on `/tokenize`; `bankml tokenize` parses specials unless `--no-special` is given.

## How it is verified

- Unit tests: `pretokenizer_shapes`, `smollm_pretokenizer_shapes`, `byte_chars_are_gpt2s`.
- `oracle_tokenizer` (`#[ignore]`, in the gate): `testing/tokenizer_oracle.py` records llama-server b11192's own
  `/tokenize` on a corpus (the repository's documents and Savante's canon, the chat markers, edge cases: contractions,
  CRLF, every kind of whitespace, digits in several scripts, combining marks, CJK, right-to-left scripts, emoji with
  joiners, a seeded fuzz set of 2,000 strings from 17 Unicode ranges), with special tokens parsed and not. The test
  re-derives every case and requires the same ids in the same order: **4,346 of 4,346** in the 0.3.6 gate record.
- `oracle_tokenizer_smollm`: the same corpus on SmolLM2's vocabulary against llama-server running SmolLM2,
  **4,346 of 4,346** (docs/oracles.md §5d).
- Indirectly, every greedy, sampling, conversation and JSON oracle: they compare token ids, so a tokenizer error
  shows there too. `oracle_grammar_masks` checks `piece` and the end set: 151,669 of 151,669 tokens and an end set of 6.

The oracle has caught real differences: which special tokens split the text (Qwen3's `<think>` markers are
USER_DEFINED and split even without `parse_special`), the letter class, and in 0.3.4 `</s>` taken as text in 2 of
4,346 cases before the type override was added.

## Advantages and efficiency

- **The same ids as llama.cpp, without llama.cpp.** No Python tokenizer, no regex engine, no crate. A prompt's token
  count, the prompt cache's reuse and every downstream oracle depend on this being exact.
- **Header only.** `from_gguf` reads the vocabulary keys and skips the rest of the header; the tensors are not touched.
  Every length read is checked against the file's size before use (`Rd::take`, `Rd::len`), so a truncated or hostile
  header gives an `Err`, not a panic or an over-read.
- **Precomputed tables.** Merges are keyed by token-id pairs `(left, right) → (rank, merged id)`; each byte's token
  is a 256-entry array; special tokens are sorted longest first once, at load.
- **The letter table.** `is_letter` is a binary search over 622 ranges.
- **Rust practice.** Zero dependencies, no `unsafe`, `Result` errors that say what is not reproduced. Where llama.cpp's
  behaviour depends on hash-map order, the vocabulary is refused rather than guessed.
- **Next** (docs/TODO.md 0.6.0): the `llama3` pre-tokenizer, part of Llama 3.x support, which is refused today.

## Limitations

- Only `gpt2/qwen2` and `gpt2/smollm` are reproduced. Any other model or pre-tokenizer is refused.
- A header with any `tokenizer.ggml.fim_*` key is refused: llama.cpp then skips its detection by text for that role.
- A vocabulary with two candidates for one fill-in-the-middle role is refused (llama.cpp picks one in hash order).
- A merge naming a string outside the vocabulary is refused (llama.cpp merges by text; none of the pinned
  vocabularies has one).
- `encode` never adds BOS or EOS (`add_special = false`).
- A byte with no token of its own (SmolLM2 lacks 21) is dropped, as llama-vocab's fallback drops it.
- The letter table is Unicode 13.0.0 data.

## See also

- [../oracles.md](../oracles.md) §1b, §5d — the tokenizer oracle
- [../usage.md](../usage.md) §13 — `bankml tokenize`
- [../OLLAMA.md](../OLLAMA.md) — O4 (SmolLM2's tokenizer) and the converter's pre-tokenizer naming
- [../TODO.md](../TODO.md) — 0.6.0, more models
- Sibling pages: [chat.md](chat.md), [grammar.md](grammar.md), [forward.md](forward.md), [native.md](native.md),
  [serve.md](serve.md), [gguf.md](gguf.md)
