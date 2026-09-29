// SPDX-License-Identifier: MIT OR Apache-2.0
//! P3, step two — the chat template: a conversation rendered into the prompt text exactly as llama.cpp b11192 renders
//! it with the Bonsai / Qwen3 template carried in the GGUF (`tokenizer.chat_template`, sha256 `30a75d10e60b57e2…`;
//! thinking is off: the generation prompt ends in an empty `<think>` block). The oracle is llama-server's own
//! `/apply-template` (`testing/template_oracle.py` → `oracle_chat_template`).
//!
//! In scope: system, user, assistant and tool messages whose content is text, with or without `reasoning_content`,
//! ending with a user or tool message (the generation prompt follows). Out of scope, refused rather than guessed:
//! tool definitions and tool calls (the template serialises them with its own `tojson`), and a conversation ending
//! with an assistant message (llama-server treats that as a prefill, which is server logic beyond the template).

use crate::serve::Json;

/// The template this module reproduces, by the sha256 of its text (the 1.7B and 8B Bonsai files carry the same one).
pub const TEMPLATE_SHA256: &str = "30a75d10e60b57e2";

#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub reasoning: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Message { role: role.into(), content: content.into(), reasoning: None }
    }
}

/// Messages from OpenAI-style JSON (`[{"role", "content", "reasoning_content"?}, …]`); refuses what is out of scope.
/// The model's own template, checked: only the Bonsai / Qwen3 template this module reproduces is accepted.
pub fn check_template(gguf: &std::path::Path) -> Result<(), String> {
    let h = crate::gguf::guard_file(gguf, crate::gguf::Engine::Mainline).map_err(|e| format!("{}: {e}", gguf.display()))?;
    let t = match h.header.as_ref().and_then(|h| h.kv.get("tokenizer.chat_template")) {
        Some(crate::gguf::Val::S(t)) => t.clone(),
        _ => return Err("the model carries no chat template".into()),
    };
    let mut s = crate::sha256::Sha256::default();
    s.update(t.as_bytes());
    let got = crate::sha256::hex(&s.finish());
    if got.starts_with(TEMPLATE_SHA256) { Ok(()) } else { Err(format!("chat template sha256 {} is not the one reproduced here ({TEMPLATE_SHA256}…)", &got[..16])) }
}

pub fn messages_from_json(v: &Json) -> Result<Vec<Message>, String> {
    let Json::Arr(a) = v else { return Err("messages must be a JSON array".into()) };
    a.iter()
        .map(|m| {
            if m.get("tool_calls").is_some() {
                return Err("tool calls are not in scope".to_string());
            }
            let role = m.get("role").and_then(Json::as_str).ok_or("a message without a role")?.to_string();
            let content = match m.get("content") {
                Some(Json::Str(s)) => s.clone(),
                None | Some(Json::Null) => String::new(),
                _ => return Err("only text content is in scope".to_string()),
            };
            // an empty reasoning_content is dropped before templating, as llama-server does (the oracle shows it: the
            // content's own <think> block is then split out, which a present-but-empty string would have prevented)
            let reasoning = m.get("reasoning_content").and_then(Json::as_str).filter(|r| !r.is_empty()).map(str::to_string);
            Ok(Message { role, content, reasoning })
        })
        .collect()
}

// Python's str methods, exactly as the template uses them
fn rstrip_nl(s: &str) -> &str {
    s.trim_end_matches('\n')
}
fn lstrip_nl(s: &str) -> &str {
    s.trim_start_matches('\n')
}
fn strip_nl(s: &str) -> &str {
    s.trim_matches('\n')
}
/// `s.split(sep)[0]`
fn before_first<'a>(s: &'a str, sep: &str) -> &'a str {
    s.split(sep).next().unwrap_or(s)
}
/// `s.split(sep)[-1]`
fn after_last<'a>(s: &'a str, sep: &str) -> &'a str {
    s.rsplit(sep).next().unwrap_or(s)
}

/// The prompt, with the generation prompt appended. Refuses what is out of scope (see the module docs).
pub fn render(msgs: &[Message]) -> Result<String, String> {
    match msgs.last() {
        None => return Err("no messages".into()),
        Some(m) if m.role == "assistant" => return Err("a conversation ending with an assistant message (a prefill) is not in scope".into()),
        _ => {}
    }
    let mut out = String::new();
    if msgs[0].role == "system" {
        out += &format!("<|im_start|>system\n{}<|im_end|>\n", msgs[0].content);
    }
    // the last real user query: scanning back, the first user message that is not a wrapped tool response
    let last_query = msgs
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| m.role == "user" && !(m.content.starts_with("<tool_response>") && m.content.ends_with("</tool_response>")))
        .map(|(i, _)| i)
        .unwrap_or(msgs.len() - 1);
    for (i, m) in msgs.iter().enumerate() {
        let (first, last) = (i == 0, i + 1 == msgs.len());
        match m.role.as_str() {
            "user" => out += &format!("<|im_start|>user\n{}<|im_end|>\n", m.content),
            "system" if !first => out += &format!("<|im_start|>system\n{}<|im_end|>\n", m.content),
            "assistant" => {
                let mut content = m.content.as_str();
                let reasoning: String = match &m.reasoning {
                    Some(r) => r.clone(),
                    None if content.contains("</think>") => {
                        let r = lstrip_nl(after_last(rstrip_nl(before_first(content, "</think>")), "<think>")).to_string();
                        content = lstrip_nl(after_last(content, "</think>"));
                        r
                    }
                    None => String::new(),
                };
                if i > last_query && (last || !reasoning.is_empty()) {
                    out += &format!("<|im_start|>assistant\n<think>\n{}\n</think>\n\n{}", strip_nl(&reasoning), lstrip_nl(content));
                } else {
                    out += &format!("<|im_start|>assistant\n{content}");
                }
                out += "<|im_end|>\n";
            }
            "tool" => {
                if first || msgs[i - 1].role != "tool" {
                    out += "<|im_start|>user";
                }
                out += &format!("\n<tool_response>\n{}\n</tool_response>", m.content);
                if last || msgs[i + 1].role != "tool" {
                    out += "<|im_end|>\n";
                }
            }
            _ => {} // the first system message is already out; any other role renders nothing, as in the template
        }
    }
    out += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_conversation() {
        let p = render(&[Message::new("system", "You are Savante."), Message::new("user", "Hi")]).unwrap();
        assert_eq!(p, "<|im_start|>system\nYou are Savante.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert!(render(&[Message::new("user", "a"), Message::new("assistant", "b")]).is_err());
    }

    #[test]
    fn python_split_semantics() {
        assert_eq!((before_first("a</think>b</think>c", "</think>"), after_last("a</think>b</think>c", "</think>")), ("a", "c"));
        assert_eq!((before_first("abc", "x"), after_last("abc", "x")), ("abc", "abc"));
    }

    /// Every conversation llama.cpp b11192 rendered (testing/template_oracle.py), byte for byte.
    #[test]
    #[ignore = "needs .models/oracle-template (testing/template_oracle.py)"]
    fn oracle_chat_template() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models");
        let cases = std::fs::read_to_string(dir.join("oracle-template/cases.jsonl")).unwrap();
        let (mut n, mut bad) = (0, Vec::new());
        for line in cases.lines() {
            let v = Json::parse(line).unwrap();
            let msgs = messages_from_json(v.get("messages").unwrap()).unwrap();
            let want = v.get("prompt").and_then(Json::as_str).unwrap();
            let got = render(&msgs).unwrap();
            n += 1;
            if got != want {
                let at = got.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                bad.push(format!("case {n}: first difference at byte {at}: got {:?} want {:?}", &got[at.min(got.len())..(at + 40).min(got.len())],
                                 &want[at.min(want.len())..(at + 40).min(want.len())]));
            }
        }
        eprintln!("chat-template oracle: {} of {n} conversations byte-identical to llama.cpp b11192", n - bad.len());
        for b in bad.iter().take(8) {
            eprintln!("  {b}");
        }
        assert!(bad.is_empty());
    }
}
