# `bankML/grammar.rs` — llama.cpp's GBNF engine, JSON mode and the content rule

## Summary

`grammar.rs` constrains sampling the way llama-server b11192 does. It is a port of llama.cpp's
`src/llama-grammar.cpp` (MIT, attributed in the file): the GBNF parser, the pushdown stacks, `advance_stack`,
`reject_candidates`, the partial UTF-8 decoder, `accept_token` and `apply`. The semantics are llama.cpp's; only the
data layout differs (a stack element is an index into one flat element array instead of a pointer).

It also holds what llama-server does around the engine:
- **Which constraint a request asks for** (`Constraint`), resolved as `oaicompat_chat_params_parse` resolves the
  OpenAI fields, and from Ollama's `format`.
- **JSON mode's grammar.** With `response_format: {"type": "json_object"}` on the jinja chat path, llama-server does
  not use `grammars/json.gbnf`. Its chat layer turns `{"type": "object"}` into a PEG parser and that into GBNF whose
  root starts with the generation prompt, and the sampler accepts the generation prompt's tokens into the grammar
  before the first draw. `JSON_OBJECT_GRAMMAR` is that text byte for byte (Qwen3 template); `JSON_OBJECT_GRAMMAR_CHATML`
  is the ChatML templates' version (no `<think>` block).
- **The content rule.** The answer's `content` is the JSON value alone, without the fence (`json_content`,
  `json_message`, `ContentStream`).

Where the grammar sits in sampling: the unconstrained chain draws first; if the token passes the grammar it stands.
Otherwise every rejected token is set to −∞ and the chain draws again from the same generator
(`Sampler::sample_constrained`, see [sampler.md](sampler.md)).

Callers: the native engine (`native.rs`: `Native::grammar` builds and prefills a `Grammar`, `complete` checks, masks
and accepts per token); `serve.rs` (`NativeChat::parse` resolves `from_openai_text`; the stream uses
`ContentStream`); `ollama.rs` (`from_ollama`, `from_ollama_text`); `bankml generate --json`; and `schema.rs`, whose
grammars this engine parses.

## Technical usage

### Resolving a request

```rust
pub enum Constraint { None, JsonObject, Schema(Value), Gbnf(String) }
pub fn from_openai(req: &Json) -> Result<Constraint, String>
pub fn from_openai_text(req: &Json, body: &str) -> Result<Constraint, String>
pub fn from_ollama(format: Option<&Json>) -> Result<Constraint, String>
pub fn from_ollama_text(req: &Json, body: &str) -> Result<Constraint, String>
```

| request | `Constraint` |
|---|---|
| `/v1`: `response_format` `json_object` (no schema), or a schema `{}` / `{"type": "object"}` | `JsonObject` |
| `/v1`: `response_format` `json_schema`, `json_object` + `schema`, top-level `json_schema` (an object) | `Schema(value)` (converted at parse time, so a refusal is a 400) |
| `/v1`: `grammar` (a GBNF string) | `Gbnf(text)`: a user grammar, root `root`, no prefill |
| Ollama: `format: "json"` or `{}` | `JsonObject` |
| Ollama: `format: <schema object>` | `Schema(value)`, as `response_format: {"type": "json_schema", …}` |
| none, `text`, a non-object schema inside `response_format` | `None` |

The `_text` variants read the schema from the request's own text, so a literal such as `2.0` stays a float.

### The engine

```rust
impl Rules { pub fn parse(text: &str, tokenize: &dyn Fn(&[u8]) -> Vec<u32>) -> Result<Rules, String> }
impl Vocab {
    pub fn new(pieces: Vec<Vec<u8>>, eog_ids: &[u32]) -> Vocab
    pub fn piece(&self, id: u32) -> &[u8]
    pub fn is_eog(&self, id: u32) -> bool
}
impl Grammar {
    pub fn new(r: std::sync::Arc<Rules>) -> Grammar
    pub fn apply(&self, v: &Vocab, cands: &mut [Cand])
    pub fn allows(&self, v: &Vocab, id: u32) -> bool
    pub fn accept(&mut self, v: &Vocab, id: u32) -> Result<(), String>
    pub fn complete(&self) -> bool
}
pub fn json_object_grammar(t: crate::chat::Template) -> (&'static str, &'static str)
pub const JSON_OBJECT_GRAMMAR: &str;  pub const JSON_OBJECT_GRAMMAR_CHATML: &str;  pub const JSON_OBJECT_PREFILL: &str;
pub fn mask_hash(allowed: impl Iterator<Item = u32>) -> (usize, u64)
```

- `Rules::parse` is `llama_grammar_init_impl(vocab, text, "root")`: parse, check every rule reference, refuse left
  recursion. `tokenize` resolves `<token>` terminals (special tokens parsed); `<[id]>` and `!<…>` are supported.
- `Vocab` holds every token's piece as the grammar reads it (`Tokenizer::piece`), decoded to code points once, and
  the end-of-generation set.
- `apply` sets every rejected candidate to −∞. An end-of-generation token passes only when a stack is empty (the
  grammar is complete); an empty piece never passes. `allows` is the single-candidate check.
- `accept` advances the stacks. A token that leaves no stack is an `Err` (llama.cpp throws).
- `json_object_grammar(t)` gives the JSON-mode grammar and its prefill for a template.

### The content rule

```rust
pub fn json_content(raw: &str) -> &str
pub fn json_message(raw: &str) -> &str
impl ContentStream {
    pub fn new(c: &Constraint) -> ContentStream
    pub fn push(&mut self, piece: &str) -> String
    pub fn content(&self, raw: &str, streamed: bool) -> String
}
```

- `json_content` returns the value inside `space ("```json" space VALUE space "```" | space VALUE space)`: the
  object once complete, the open part while it is still open. A string ends at its closing quote, a number or
  literal at the first space or the fence. An escape the cut left unfinished is not content (0.3.5).
- `json_message` is the non-streamed answer: the parsed content, or the raw text when that parse is empty, as
  llama-server answers.
- `ContentStream::push` sends only the growth of `json_content`; without a JSON constraint it passes pieces through.

```sh
echo "Describe a cat as JSON." | bankml generate .models/Bonsai-8B-Q1_0.gguf --json
curl -s 127.0.0.1:PORT/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"A cat, as JSON"}],"response_format":{"type":"json_object"}}'
```

## How it is verified

- Unit tests: `the_server_grammar_parses`, `parse_errors_say_where`, `a_small_grammar_masks_and_accepts`,
  `partial_utf8_across_tokens`, `decode_matches_the_c_reading`, `requests_resolve_as_llama_server_does`,
  `json_content_is_the_value`.
- `oracle_grammar_masks` (`#[ignore]`, in the gate): `testing/grammar_oracle.cpp` drives libllama b11192's own
  grammar sampler (`llama_sampler_init_grammar`, `_apply`, `_accept`) on the pinned vocabulary. Pieces and end flags:
  **151,669 of 151,669** tokens, end set of 6. Masks over 12 grammars (JSON mode with its prefill, the 8 in
  llama.cpp's `grammars/`, 3 written for token terminals, `.`, `{m,n}` edges, comments, CRLF, astral and CJK ranges),
  each input tokenized normally and one token per byte: **196 of 196 runs; 1,645 of 1,645 masks and 116 of 116
  rejection points identical** (masks compared by count and FNV-1a hash, `mask_hash`).
- `oracle_json_mode`, `oracle_json_mode_ternary` (`native.rs`): `testing/json_oracle.py --record` against
  llama-server b11192: the same token ids, raw text, content, finish reason and counts, and the server's reported
  grammar and generation prompt equal to bankml's constant and prefill. Bonsai-8B **23 of 23** (860 tokens, 152
  redrawn), ternary **13 of 13** (534 tokens, 49 redrawn); 23 of 23 on each O4 model. Live (`--bankml`): 8 of 8.
- `oracle_json_content`: llama.cpp's own `common_chat_parse` (`testing/content_oracle.{cpp,py}`) on every prefix of
  every recorded constrained answer and edge cases: **30,063 of 30,063** texts on each template.
- Schema answers (`oracle_json_schema*`) run through this engine too; see [schema.md](schema.md).

## Advantages and efficiency

- **The same JSON as llama-server.** Not a JSON-only mask: the same grammar text, the same prefill, the same redraw
  and the same content rule, so constrained answers, seeded or greedy, are token-identical and can be checked.
- **Cheap when the model already writes valid JSON.** The first draw is checked as one candidate (microseconds). The
  whole-vocabulary mask runs only when that token breaks the grammar.
- **Work done once.** Each vocabulary's pieces are decoded to code points once (`Vocab::new`; the native engine keeps
  one per model in a `OnceLock`). The JSON-mode `Rules` are parsed once per engine and shared through an `Arc`.
- **Measured** (docs/PERFORMANCE.md, gate 0.3.3, laptop): one whole-vocabulary mask over 151,669 tokens, median
  **24.2 ms** (p90 47.0, max 101.0). Grammar time per generated token, everything included: **3.90 ms** (Bonsai-8B
  Q1_0) and **3.95 ms** (ternary), about 1 % of a 1-bit step and 1.4 % of a ternary step. An answer written as valid
  JSON on its own spent 0.4 ms of grammar per token.
- **Rust practice.** Zero crates; no `unsafe`. Parse errors say where (`expecting name at …`). Where llama.cpp
  would loop practically forever (`{m,n}` with n below m) or throw (an empty stack after accept), bankml returns an
  `Err`. The repetition limits of llama.cpp (2,000) are kept.
- **Next** (docs/TODO.md 0.4.0): a byte trie over the token pieces, so each grammar stack walks shared prefixes once;
  the oracle stays the same masks. Then tool calls through the template (O6).

## Limitations

- A user `grammar` gets no prefill, as llama-server does. `response_format` and `grammar` together are refused
  (llama-server would keep only the format's grammar).
- A top-level `json_schema` that is `null` or not an object is refused (llama-server fails the request).
- Ollama's `format` accepts only `"json"`, `""` or a schema object.
- The mask is a straight port of `reject_candidates` (cost above).
- Tool calls are not supported yet.

## See also

- [../oracles.md](../oracles.md) §5c, §5d, §5e
- [../PERFORMANCE.md](../PERFORMANCE.md) — JSON mode's cost
- [../OLLAMA.md](../OLLAMA.md) — O6, what llama-server does with `response_format`
- [../TODO.md](../TODO.md) — 0.4.0, the byte-trie mask
- [../usage.md](../usage.md) — `bankml generate --json`, the `/v1` and `/api` fields
- Sibling pages: [schema.md](schema.md), [sampler.md](sampler.md), [tokenizer.md](tokenizer.md),
  [chat.md](chat.md), [native.md](native.md), [serve.md](serve.md), [ollama.md](ollama.md)
