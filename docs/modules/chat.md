# `bankML/chat.rs` — chat templates, byte-identical to llama.cpp b11192

## Summary

A conversation becomes a prompt through the model's chat template, a Jinja program stored in the GGUF
(`tokenizer.chat_template`). bankml does not run Jinja. `chat.rs` writes out the rules of each template it
reproduces, and identifies the model's template by the sha256 of its text. A template it does not know is refused.

Three templates are reproduced (`TEMPLATES`):

| sha256 prefix | `Template` | carried by |
|---|---|---|
| `30a75d10e60b57e2` | `Qwen3` | Bonsai 1.7B and 8B (Qwen3), thinking off |
| `872be49dbb638044` | `SmolLm2` | SmolLM2-Instruct: ChatML with a default system message |
| `9fe579a2c222698c` | `ChatMl` | plain ChatML, no default system message (`mindx-genN`) |

On the Qwen3 template the generation prompt ends in an empty `<think>` block (thinking off). On the two ChatML
templates every message renders as `<|im_start|>{role}\n{content}<|im_end|>\n` whatever its role, and
`reasoning_content` is dropped, as llama-server renders them.

Callers: the native engine (`native.rs`: `template_of` at open, `render` for every prompt, `generation_prompt` for
the grammar's prefill); `bankml serve --native` (`POST /apply-template`, `/v1/chat/completions`, `/api/chat`); the
Ollama layer (`ollama.rs`); `bankml chat-template` and `bankml generate` (`main.rs`); and `grammar.rs` / `schema.rs`,
which choose the JSON grammar's root by template.

## Technical usage

```rust
pub const TEMPLATE_SHA256: &str = "30a75d10e60b57e2";
pub enum Template { Qwen3, SmolLm2, ChatMl }
pub const TEMPLATES: [(&str, Template, &str); 3];

impl Template {
    pub fn render(self, msgs: &[Message]) -> Result<String, String>
    pub fn generation_prompt(self) -> &'static str
}

pub struct Message { pub role: String, pub content: String, pub reasoning: Option<String> }
impl Message { pub fn new(role: &str, content: &str) -> Self }

pub fn template_of(gguf: &std::path::Path) -> Result<Template, String>
pub fn check_template(gguf: &std::path::Path) -> Result<(), String>
pub fn messages_from_json(v: &Json) -> Result<Vec<Message>, String>
pub fn render(msgs: &[Message]) -> Result<String, String>   // the Qwen3 template
```

- `template_of` guards the file, hashes `tokenizer.chat_template` and matches the first 16 hex digits against
  `TEMPLATES`. The error names the hash it found and the templates it knows.
- `messages_from_json` reads OpenAI-style `[{"role", "content", "reasoning_content"?}, …]`. An empty
  `reasoning_content` is dropped before templating, as llama-server does.
- `render` (Qwen3) reproduces the template's details: a first system message; the last real user query (scanning
  back, the first user message that is not a wrapped `<tool_response>`); assistant turns after it keep their
  `<think>` block, those before it lose it; runs of tool messages grouped into one user turn of
  `<tool_response>` blocks; Python's `split` and `strip` semantics where the template uses them.
- `generation_prompt()` is what the template appends for the assistant's turn:
  `"<|im_start|>assistant\n<think>\n\n</think>\n\n"` on Qwen3, `"<|im_start|>assistant\n"` on the ChatML templates.

```sh
echo '[{"role":"system","content":"You are Savante."},{"role":"user","content":"Hi"}]' \
  | bankml chat-template .models/Bonsai-8B-Q1_0.gguf
```

prints:

```text
<|im_start|>system
You are Savante.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
<think>

</think>

```

## How it is verified

- Unit tests: `a_short_conversation`, `python_split_semantics`, `chatml_templates`.
- `oracle_chat_template` (`#[ignore]`, in the gate): `testing/template_oracle.py` records llama-server's own
  `/apply-template` for 317 conversations (system prompts first, later or absent; assistant turns with and without
  `<think>` blocks around the last real query; `reasoning_content`; runs of tool results; user messages that look
  like tool responses; special markers and Unicode in content; 300 random conversations). Every prompt must be
  byte-identical: **317 of 317** (0.3.6 gate record).
- `oracle_chat_template_chatml`: SmolLM2-Instruct's template and mindx-gen39's ChatML, each against llama-server
  rendering that model's own, **317 of 317** each (docs/oracles.md §5d).
- Indirectly: the conversation oracle (`oracle_native_serve`, 9 of 9 turns, including the prompt cache's reuse) and
  every greedy, sampling and JSON oracle start from a rendered prompt.

The oracle found one server behaviour the template alone would not predict: an empty `reasoning_content` is dropped.

## Advantages and efficiency

- **No template engine.** No Jinja interpreter and no crate: each template is a short, readable Rust function, and
  the sha256 pin means a model whose template changed is refused instead of rendered wrongly.
- **Exact prompts.** A byte-identical prompt is what lets the prompt cache reuse what llama-server would reuse, and
  lets every token-level oracle compare like with like.
- **Cheap.** Rendering is string concatenation over the messages; the template is identified once, when the engine
  opens the model.
- **Rust practice.** Zero dependencies, no `unsafe`; out-of-scope input is an `Err` with the reason, not a guess.
- **Next** (docs/TODO.md, docs/OLLAMA.md): tool calls through the template (O6, after JSON schemas), and the Llama 3.x
  template with that architecture (0.6.0).

## Limitations

- Only the three templates above. Any other is refused by its hash.
- Refused rather than guessed: tool definitions and `tool_calls` (the template serialises them with its own
  `tojson`); non-text content; a conversation ending with an assistant message (llama-server treats that as a
  prefill, which is server logic beyond the template); an empty message list.
- On Qwen3, a role other than system, user, assistant and tool renders nothing, as in the template.
- Ollama's persona `SYSTEM` for `mindx-genN` lives in its Modelfile, not in the GGUF: here it is the caller's system
  message (`bankml create` handles the Modelfile).

## See also

- [../oracles.md](../oracles.md) §1c, §5d — the chat-template oracle
- [../usage.md](../usage.md) §13 — `bankml chat-template`
- [../TECHNICAL.md](../TECHNICAL.md) §III.7 — from a template to a token
- [../OLLAMA.md](../OLLAMA.md) — O4 templates, O6 tool calls
- [../TODO.md](../TODO.md)
- Sibling pages: [tokenizer.md](tokenizer.md), [grammar.md](grammar.md), [schema.md](schema.md),
  [native.md](native.md), [serve.md](serve.md), [ollama.md](ollama.md), [create.md](create.md)
