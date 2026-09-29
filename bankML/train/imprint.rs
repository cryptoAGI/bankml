// SPDX-License-Identifier: MIT OR Apache-2.0
//! The score stage: did the persona take? `mindxtrain/eval/imprint.py` poses the same inquiries to the model before
//! and after training and scores the utterances against the persona's voice examples. This is its dependency-free
//! path, exactly: tokens are `[a-z0-9']+` of the Unicode-lowercased text; similarity is token Jaccard; the voice
//! score is the mean over utterances of the best similarity to any voice example; the shift is the mean
//! `1 − similarity(before, after)`; every figure is rounded to 4 decimals as Python's `round` rounds; and the
//! verdict is `imprinted = delta > 0 and shift > 0` on the unrounded values.
//! (mindXtrain prefers sentence-transformer cosine when that package is installed; that path is not ported.)

use std::collections::BTreeSet;

/// `default_inquiries`: the recall probes, persona-agnostic.
pub fn default_inquiries(name: &str) -> Vec<String> {
    vec!["Who are you?".into(), "What do you do?".into(), format!("Describe {name} in one sentence."), "What matters most to you?".into(), "Say hello.".into()]
}

fn tokens(text: &str) -> BTreeSet<String> {
    let lower = text.to_lowercase();
    let mut out = BTreeSet::new();
    let mut cur = String::new();
    for c in lower.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'' {
            cur.push(c);
        } else if !cur.is_empty() {
            out.insert(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.insert(cur);
    }
    out
}

/// Token Jaccard in [0, 1] (0 when either side has no tokens).
pub fn jaccard(a: &str, b: &str) -> f64 {
    let (ta, tb) = (tokens(a), tokens(b));
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count();
    inter as f64 / (ta.len() + tb.len() - inter) as f64
}

/// Python's `sum(xs) / len(xs)`: left to right, then one division.
fn mean(xs: impl Iterator<Item = f64>) -> Option<f64> {
    let (mut s, mut n) = (0.0f64, 0usize);
    for x in xs {
        s += x;
        n += 1;
    }
    (n > 0).then(|| s / n as f64)
}

/// Python's `round(x, 4)`: the exact binary value rounded half to even at the fourth decimal.
pub fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap()
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImprintReport {
    pub before_voice: f64,
    pub after_voice: f64,
    pub imprint_delta: f64,
    pub shift: f64,
    pub method: &'static str,
    pub imprinted: bool,
}

fn voice(utterances: &[String], baseline: &[String]) -> (f64, &'static str) {
    if utterances.is_empty() || baseline.is_empty() {
        return (0.0, "none");
    }
    let per = utterances.iter().map(|u| baseline.iter().map(|r| jaccard(u, r)).fold(f64::NEG_INFINITY, f64::max));
    (mean(per).unwrap(), "lexical")
}

/// `score_imprint` on its lexical path.
pub fn score(before: &[String], after: &[String], baseline: &[String]) -> ImprintReport {
    let (bv, m1) = voice(before, baseline);
    let (av, m2) = voice(after, baseline);
    let method = if m1 != "none" { m1 } else { m2 };
    let shift = mean(before.iter().zip(after).map(|(b, a)| 1.0 - jaccard(b, a))).unwrap_or(0.0);
    let delta = av - bv;
    ImprintReport { before_voice: round4(bv), after_voice: round4(av), imprint_delta: round4(delta), shift: round4(shift), method,
                    imprinted: delta > 0.0 && shift > 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::Json;

    #[test]
    fn jaccard_and_rounding_as_python() {
        assert_eq!(jaccard("Hello, WORLD", "hello there"), 1.0 / 3.0);
        assert_eq!(jaccard("", "x"), 0.0);
        assert_eq!(round4(0.03125), 0.0312); // an exact tie at the fourth decimal: half to even, as Python
        assert_eq!(round4(2.0 / 3.0), 0.6667);
    }

    /// Every score mindXtrain's own score_imprint gave (testing/train_oracle.py), value for value.
    #[test]
    #[ignore = "needs .models/oracle-train/imprint.jsonl (testing/train_oracle.py)"]
    fn oracle_train_imprint() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/oracle-train");
        let cases = std::fs::read_to_string(dir.join("imprint.jsonl")).unwrap();
        let strs = |v: &Json, k: &str| -> Vec<String> {
            match v.get(k) { Some(Json::Arr(a)) => a.iter().map(|x| x.as_str().unwrap().to_string()).collect(), _ => panic!("{k}") }
        };
        let num = |v: &Json, k: &str| match v.get(k) { Some(Json::Num(n)) => *n, _ => panic!("{k}") };
        let (mut n, mut ok) = (0, 0);
        for line in cases.lines() {
            let c = Json::parse(line).unwrap();
            let r = score(&strs(&c, "before"), &strs(&c, "after"), &strs(&c, "baseline"));
            let w = c.get("report").unwrap();
            n += 1;
            let good = r.before_voice == num(w, "before_voice") && r.after_voice == num(w, "after_voice") && r.imprint_delta == num(w, "imprint_delta")
                && r.shift == num(w, "shift") && Some(r.method) == w.get("method").and_then(Json::as_str) && Some(r.imprinted) == w.get("imprinted").and_then(Json::as_bool);
            ok += good as usize;
            if !good && n - ok <= 5 {
                eprintln!("  case {n}: got {r:?}");
            }
        }
        eprintln!("train oracle: score stage {ok} of {n} imprint reports identical to mindXtrain's score_imprint (lexical path)");
        assert_eq!(ok, n);
    }
}
