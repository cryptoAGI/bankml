// SPDX-License-Identifier: MIT OR Apache-2.0
//! mindXtrain in Rust: ports of mindXtrain's stages (github.com/Professor-Codephreak/mindXtrain, continued at
//! huggingface.co/PYTHAI/mindXtrain; Apache-2.0), which imprint a persona onto a model and prove it with a recall
//! gate. Each ported stage reproduces mindXtrain's Python exactly, checked by an oracle that runs mindXtrain's own
//! functions (`testing/train_oracle.py`). [`STAGES`] lists the stages and their status.
//!
//! Details: docs/modules/train.md.

pub mod imprint;
pub mod script;

/// The stages of the proof loop, with where each stands in the Rust port.
pub const STAGES: &[(&str, &str, &str)] = &[
    ("author", "script", "ported: persona + exchanges → chat JSONL, byte-identical to mindXtrain"),
    ("imprint", "", "not yet: LoRA SFT needs a backward pass"),
    ("probe", "", "not yet: needs adapter loading in bankml's forward pass (the Llama graph runs since 0.3.4)"),
    ("score", "imprint", "ported: the lexical voice metric and the imprinted verdict, identical to mindXtrain"),
    ("classroom", "", "not yet"),
    ("boardroom", "", "not yet"),
];

/// A JSON string as Python's `json.dumps(…, ensure_ascii=False)` writes it.
pub fn py_json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}
