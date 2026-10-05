# `bankML/schema.rs` — JSON schema to GBNF, as llama-server b11192 builds it

## Summary

`schema.rs` turns a JSON schema into the GBNF text llama-server b11192 would build for it, byte for byte. It is a port
(MIT, attributed in the file) of:
- `common/json-schema.cpp`: the schema tree, which keywords decide a node's kind, `$ref` resolution, the errors;
- `common/json-schema-to-grammar.cpp` (`common_chat_schema_converter`): rule naming and de-duplication, objects with
  required / optional / additional properties, `_not_strings`, arrays and tuples, integer ranges, the regex → GBNF
  pattern translation, string formats, the primitive rules;
- `common/trie.cpp`, which `_not_strings` walks;
- the parts of nlohmann's `ordered_json` the grammar text depends on: object key order and duplicate keys, the
  integer / float distinction, `dump()`;
- the GBNF the PEG chat parser adds around the schema (`common/chat-auto-parser-generator.cpp`,
  `common/peg-parser.cpp`): the `json-*` rules, `response-format`, `root`, and on the Qwen3 template the `until-13`
  rules for the reasoning block.

The grammar text matters because the answer depends on it: the grammar engine ([grammar.md](grammar.md)) masks the
same tokens only if it parses the same rules.

Callers: `grammar.rs` (`from_openai*` and `from_ollama*` convert a schema at request parse time to accept or refuse
it); `native.rs` (`Native::grammar` builds the grammar for the model's template); `serve.rs` (`NativeChat::parse`).
The surfaces are `/v1/chat/completions` (`response_format` `json_schema`, `json_object` + `schema`, the top-level
`json_schema`), Ollama's `/api/chat` and `/api/generate` with `format: <schema>`, and the C API's `bankml_chat`.

## Technical usage

```rust
pub enum Value { Null, Bool(bool), Int(i64), UInt(u64), Float(f64), Str(String), Arr(Vec<Value>), Obj(Vec<(String, Value)>) }
impl Value {
    pub fn parse(s: &str) -> Option<Value>
    pub fn from_json(j: &crate::serve::Json) -> Value
    pub fn get(&self, k: &str) -> Option<&Value>
    pub fn dump(&self) -> String
}
pub fn json_schema_to_grammar(schema: &Value) -> Result<(String, Vec<String>), String>
pub fn chat_grammar(schema: &Value, t: crate::chat::Template) -> Result<(String, Vec<String>), String>
pub fn format_literal(s: &str) -> String
```

- `Value` is JSON as nlohmann's `ordered_json` sees it: integers apart from floats (`1` vs `1.0`), unsigned apart
  from signed, object keys in first-seen order with the last duplicate's value. `Value::parse` is the exact reading
  (nesting deeper than 512 levels is refused). `from_json` converts bankml's request JSON, whose numbers are f64, so
  a schema written `2.0` reads as `2`; the servers re-read the body with `parse` to keep it a float.
- `json_schema_to_grammar` is llama.cpp's `json_schema_to_grammar(schema)` (`force_gbnf`): the grammar whose root is
  the schema. The second value holds warnings, for example a pattern it reads but cannot translate, which becomes
  "any string" as in llama.cpp.
- `chat_grammar(schema, template)` is what llama-server builds on the jinja path with thinking off: the schema's rules
  and the parser's rules in one converter, with `response-format`, `root`, and on Qwen3 the `until-13` rules. On the
  ChatML templates (SmolLM2-Instruct, `mindx-genN`) there is no reasoning block. Its errors carry llama-server's
  prefix: `Unable to generate parser for this template. Automatic parser generation failed: `.
- `{"type": "object"}` gives exactly each template's JSON-mode grammar (`grammar::JSON_OBJECT_GRAMMAR` /
  `_CHATML`); `grammar.rs` then treats the request as plain JSON mode.

Supported schema shapes include `$ref` into the same document (a `#/…` pointer, such as `#/$defs/…`), `anyOf` /
`oneOf`, `allOf`, `const`, `enum`, `null`, `boolean`, `number`, `integer` with `minimum` / `maximum` (and the
exclusive forms), `string` with `pattern`, `minLength` / `maxLength` and the formats `date`, `time`, `date-time` and
`uuid`, arrays with `items`, `minItems` / `maxItems`, tuples (`prefixItems`), and objects with `properties`,
`required` and `additionalProperties`.

```sh
curl -s 127.0.0.1:PORT/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "A cat"}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "cat", "schema":
    {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
     "required": ["name"]}}}}'
```

## How it is verified

- Unit tests: `json_reads_and_prints_as_nlohmann`, `the_object_schema_is_the_json_mode_grammar`, `refusals_say_where`,
  and `llama_cpp_test_cases`, which carries llama.cpp's own `tests/test-json-schema-to-grammar.cpp` cases with their
  expected grammars (`testing/json_schema_cases.json`, at least 80 cases) and checks that each grammar parses.
- `oracle_schema_grammars` (`#[ignore]`, in the gate): `testing/schema_oracle.{cpp,py}` calls llama.cpp b11192's own
  `json_schema_to_grammar` and `common_chat_templates_apply` inside the release's `libllama-common.so` (no model) on
  173 schemas: llama.cpp's 81 test cases, Pydantic-shaped schemas like mindX's, edge cases and every refusal path,
  with the template read from each model that carries it. **148 grammars byte-identical bare and on the chat path,
  on each of the three templates; every refusal with llama.cpp's message** (24 bare, 20 chat). Every grammar must
  also parse in `grammar.rs`.
- `oracle_json_schema`, `_ternary`, `_o4` (`native.rs`): `testing/json_schema_oracle.py --record` against
  llama-server b11192, greedy and seeded, from an empty cache, including answers cut by `max_tokens`: Bonsai-8B
  **28 of 28**, Ternary-Bonsai-8B **11 of 11**, Bonsai-1.7B, SmolLM2-135M-Instruct and mindx-gen39 **56 of 56** each.
  Live in the gate through `/v1` (one streamed) and `/api/chat` with `format: <schema>`.

## Advantages and efficiency

- **Any schema, llama-server's grammar.** mindX's Pydantic-shaped schemas get the exact grammar llama-server would
  use, so the answers are token-identical and checkable, without llama.cpp at run time.
- **Refusals before generation.** A schema is converted when the request is parsed, so a schema b11192 refuses is a
  400 with its message before any token is computed.
- **Bounded on hostile input.** JSON nesting is capped at 512 levels and regex group nesting at 100
  (`MAX_PATTERN_DEPTH`); errors are `Result`s with a path (`JSON schema error at #/properties/a: …`), not panics.
- **Rust practice.** Zero crates (nlohmann's number printing and the trie are written out); no `unsafe`.
- **Next** (docs/TODO.md, docs/OLLAMA.md): tool calls through the template (O6), which build on the schema converter.

## Limitations

- Refused as b11192 refuses, with its message: an unknown type, an empty `enum`, a `$ref` outside the document, a
  pattern no regex reads, and the other paths `oracle_schema_grammars` records.
- One deliberate divergence: a `pattern` with a non-ASCII character before a quantifier (for example `^é+$`).
  llama.cpp splits the character byte by byte into a grammar that is not UTF-8; bankml refuses it and suggests a
  group, `(é)+`.
- Unsupported regex features (for example `\d`) are not refused: the string accepts anything, with a warning, as in
  llama.cpp.
- Known limit, measured: a float in an `enum` or `const` is printed with Rust's shortest round-trip digits, which agree
  with nlohmann's Grisu2 except on rare doubles where Grisu2 is not shortest.
- `chat_grammar` reproduces the jinja path with thinking off on the three reproduced templates only.

## See also

- [../oracles.md](../oracles.md) §5e — the schema oracles
- [../OLLAMA.md](../OLLAMA.md) — O6b, the chat path's wrapping per template
- [../TODO.md](../TODO.md) — 0.4.0, 0.6.0
- [../usage.md](../usage.md) — `/v1` and `/api` request fields
- Sibling pages: [grammar.md](grammar.md), [chat.md](chat.md), [sampler.md](sampler.md), [native.md](native.md),
  [serve.md](serve.md), [ollama.md](ollama.md)
