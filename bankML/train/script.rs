// SPDX-License-Identifier: MIT OR Apache-2.0
//! The author stage: a persona and a few exchanges become a training *script* (OpenAI-chat JSONL), as
//! mindXtrain's (github.com/Professor-Codephreak/mindXtrain, continued at huggingface.co/PYTHAI/mindXtrain; Apache-2.0)
//! `mindxtrain/data/scripts.py` builds it — the same recognised keys read clean-room from any persona JSON, the same
//! synthesised system prompt, the same voice-seed rows, and each line byte-identical to Python's
//! `json.dumps(row, ensure_ascii=False)`.
//!
//! Scope: a persona's identity fields are strings (as every persona in mindX is); a number in a voice list is written
//! as Python writes an int, so a float literal such as `3.0` would differ (bankml's JSON keeps numbers as f64).

use super::py_json_str;
use crate::serve::Json;

const NAME_KEYS: [&str; 4] = ["name", "persona", "id", "title"];
const SYSTEM_KEYS: [&str; 6] = ["system_prompt", "system", "description", "bio", "summary", "prompt"];
const VOICE_KEYS: [&str; 5] = ["voice_examples", "examples", "utterances", "samples", "voice"];

#[derive(Debug, Clone, PartialEq)]
pub struct Persona {
    pub name: String,
    pub system_prompt: String,
    pub voice_examples: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Exchange {
    pub user: String,
    pub assistant: String,
}

/// Python's truthiness, for the values a persona's identity keys hold.
fn truthy(v: &Json) -> bool {
    match v {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Num(n) => *n != 0.0,
        Json::Str(s) => !s.is_empty(),
        Json::Arr(a) => !a.is_empty(),
        Json::Obj(o) => !o.is_empty(),
    }
}

/// Python's `str()` of a scalar.
fn py_str(v: &Json) -> Option<String> {
    match v {
        Json::Str(s) => Some(s.clone()),
        Json::Bool(b) => Some(if *b { "True" } else { "False" }.into()),
        Json::Num(n) if n.fract() == 0.0 && n.abs() < 1e16 => Some(format!("{}", *n as i64)),
        Json::Num(n) => Some(format!("{n}")),
        _ => None,
    }
}

/// `persona_from_dict`: the first truthy name key, the first truthy system key, and every voice list or string.
pub fn persona_from_json(raw: &Json) -> Result<Persona, String> {
    let Json::Obj(_) = raw else { return Err("a persona is a JSON object".into()) };
    let first = |keys: &[&str], dflt: &str| -> Result<String, String> {
        for k in keys {
            if let Some(v) = raw.get(k).filter(|v| truthy(v)) {
                return py_str(v).ok_or_else(|| format!("persona field {k:?} is not text"));
            }
        }
        Ok(dflt.to_string())
    };
    let (name, system_prompt) = (first(&NAME_KEYS, "actor")?, first(&SYSTEM_KEYS, "")?);
    let mut voice_examples = Vec::new();
    for k in VOICE_KEYS {
        match raw.get(k) {
            Some(Json::Arr(a)) => voice_examples.extend(a.iter().filter(|x| matches!(x, Json::Str(_) | Json::Num(_) | Json::Bool(_))).filter_map(py_str)),
            Some(Json::Str(s)) => voice_examples.push(s.clone()),
            _ => {}
        }
    }
    Ok(Persona { name, system_prompt, voice_examples })
}

/// `persona_system_prompt`: the persona's own, stripped, or one synthesised from its name.
pub fn system_prompt(p: &Persona) -> String {
    let s = p.system_prompt.trim_matches(py_space);
    if s.is_empty() { format!("You are {}. Stay in character and answer in your own voice.", p.name) } else { s.to_string() }
}

/// Python's `str.strip()` whitespace (Unicode `White_Space` plus the ASCII separators it also strips).
fn py_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// `build_script_rows` + `write_script_jsonl`: the script as JSONL text, one row per line.
pub fn script_jsonl(p: &Persona, exchanges: &[Exchange], seed_voice: bool) -> String {
    let sys = system_prompt(p);
    let mut out = String::new();
    let mut row = |user: &str, assistant: &str| {
        out.push_str("{\"messages\": [{\"role\": \"system\", \"content\": ");
        py_json_str(&sys, &mut out);
        out.push_str("}, {\"role\": \"user\", \"content\": ");
        py_json_str(user, &mut out);
        out.push_str("}, {\"role\": \"assistant\", \"content\": ");
        py_json_str(assistant, &mut out);
        out.push_str("}]}\n");
    };
    for e in exchanges {
        row(&e.user, &e.assistant);
    }
    if seed_voice {
        let ask = format!("Say something as {}.", p.name);
        for v in &p.voice_examples {
            row(&ask, v);
        }
    }
    out
}

/// `derive_training_params`: (epochs, grad_accum, per_device) for a script of `rows` rows — small scripts must
/// overfit to imprint.
pub fn training_params(rows: usize) -> (u32, u32, u32) {
    let (e, g) = match rows.max(1) {
        0..=8 => (24, 1),
        9..=32 => (16, 1),
        33..=128 => (8, 1),
        129..=512 => (4, 2),
        _ => (2, 4),
    };
    (e, g, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nameless_persona_gets_the_synthesised_prompt() {
        let p = persona_from_json(&Json::parse(r#"{"voice": "hi", "examples": ["a", 3, true, {"x": 1}]}"#).unwrap()).unwrap();
        assert_eq!(p.name, "actor");
        assert_eq!(p.voice_examples, vec!["a", "3", "True", "hi"]);
        assert_eq!(system_prompt(&p), "You are actor. Stay in character and answer in your own voice.");
        assert_eq!(training_params(0), (24, 1, 1));
        assert_eq!(training_params(513), (2, 4, 1));
    }

    /// Every persona and script mindXtrain itself built (testing/train_oracle.py), byte for byte.
    #[test]
    #[ignore = "needs .models/oracle-train/script.jsonl (testing/train_oracle.py)"]
    fn oracle_train_script() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/oracle-train");
        let cases = std::fs::read_to_string(dir.join("script.jsonl")).unwrap();
        let (mut n, mut ok) = (0, 0);
        for line in cases.lines() {
            let c = Json::parse(line).unwrap();
            let p = persona_from_json(c.get("persona_raw").unwrap()).unwrap();
            let want = c.get("persona").unwrap();
            let exch: Vec<Exchange> = match c.get("exchanges") {
                Some(Json::Arr(a)) => a.iter().map(|e| Exchange { user: e.get("user").and_then(Json::as_str).unwrap().into(),
                                                                   assistant: e.get("assistant").and_then(Json::as_str).unwrap().into() }).collect(),
                _ => vec![],
            };
            let seed = c.get("seed_voice").and_then(Json::as_bool).unwrap();
            let rows = script_jsonl(&p, &exch, seed);
            let params = training_params(rows.lines().count());
            n += 1;
            let good = Some(p.name.as_str()) == want.get("name").and_then(Json::as_str)
                && Some(p.system_prompt.as_str()) == want.get("system_prompt").and_then(Json::as_str)
                && rows == c.get("jsonl").and_then(Json::as_str).unwrap()
                && params.0 as f64 == match c.get("params").and_then(|p| p.get("epochs")) { Some(Json::Num(x)) => *x, _ => -1.0 };
            ok += good as usize;
            if !good {
                eprintln!("  case {n} ({}) differs", c.get("label").and_then(Json::as_str).unwrap_or("?"));
            }
        }
        eprintln!("train oracle: author stage {ok} of {n} scripts byte-identical to mindXtrain's scripts.py");
        assert_eq!(ok, n);
    }
}
