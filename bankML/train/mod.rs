// SPDX-License-Identifier: MIT OR Apache-2.0
//! mindXtrain in Rust — bankml's training module, grown one verified stage at a time. mindXtrain
//! (github.com/Professor-Codephreak/mindXtrain, continued at huggingface.co/PYTHAI/mindXtrain; Apache-2.0) imprints
//! a persona onto a model and proves it with a recall gate:
//!
//!   author (persona + exchanges → a chat-JSONL script) → imprint (LoRA SFT) → probe (the same inquiries before and
//!   after) → score (did the voice move toward the persona, and did the utterances change) → classroom → boardroom
//!
//! Each stage here reproduces mindXtrain's Python exactly, checked by an oracle that runs mindXtrain's own functions
//! (`testing/train_oracle.py`), before anything new is built on it. The stages are modules listed in [`STAGES`]; a
//! new stage is a new module and one line there.
//!
//! Ported so far: `script` (author: `mindxtrain/data/scripts.py`) and `imprint` (score: the lexical path of
//! `mindxtrain/eval/imprint.py`). Next: the probe on bankml's own forward pass (it needs the Llama architecture and
//! adapter loading), then the classroom and boardroom verdicts, then LoRA training on the CPU.

pub mod imprint;
pub mod script;

/// The stages of the proof loop, with where each stands in the Rust port.
pub const STAGES: &[(&str, &str, &str)] = &[
    ("author", "script", "ported: persona + exchanges → chat JSONL, byte-identical to mindXtrain"),
    ("imprint", "", "not yet: LoRA SFT needs a backward pass"),
    ("probe", "", "not yet: needs the Llama architecture and adapter loading in bankml's forward pass"),
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
