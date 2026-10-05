// SPDX-License-Identifier: MIT OR Apache-2.0
//! bankML's own measurements of its answers (0.3.7): one record per completion, taken inside the engine
//! (`Native::complete`), kept in a bounded ring and served at `GET /bankml/metrics`. The definitions are the
//! conventional ones, so the numbers compare with llama.cpp's: time to first token (TTFT), prompt processing in tokens
//! per second (llama-bench's `pp`), generation in tokens per second (`tg`), and — when the RAPL counter is readable —
//! the request's package energy and joules per generated token. Nothing is estimated: a field that was not measured
//! is `null`.

use std::collections::VecDeque;
use std::sync::Mutex;

/// The records kept (the oldest goes first).
pub const KEEP: usize = 256;

/// One completion, as the engine measured it.
#[derive(Debug, Clone, Default)]
pub struct Record {
    /// seconds since the Unix epoch when the completion ended
    pub at: f64,
    pub model: String,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    /// the uncached prompt's computation, the time to the first generated token, and the generation after it
    pub prompt_ms: f64,
    pub ttft_ms: Option<f64>,
    pub eval_ms: f64,
    pub finish: String,
    pub grammar_ms: f64,
    pub resampled: usize,
    /// the CPU package's energy over the completion (includes an APU's GPU), when RAPL is readable
    pub energy_j: Option<f64>,
}

impl Record {
    /// Prompt tokens per second over the part that was computed (`pp`).
    pub fn prompt_tps(&self) -> Option<f64> {
        let n = self.prompt_tokens.saturating_sub(self.cached_tokens);
        (self.prompt_ms > 0.0 && n > 0).then(|| n as f64 / (self.prompt_ms / 1e3))
    }

    /// Generated tokens per second (`tg`).
    pub fn eval_tps(&self) -> Option<f64> {
        (self.eval_ms > 0.0 && self.completion_tokens > 0).then(|| self.completion_tokens as f64 / (self.eval_ms / 1e3))
    }

    pub fn joules_per_token(&self) -> Option<f64> {
        self.energy_j.filter(|_| self.completion_tokens > 0).map(|e| e / self.completion_tokens as f64)
    }

    pub fn json(&self) -> String {
        let f = |v: Option<f64>, d: usize| v.map(|x| format!("{x:.d$}")).unwrap_or_else(|| "null".into());
        format!("{{\"at\": {:.3}, \"model\": {}, \"prompt_tokens\": {}, \"cached_tokens\": {}, \"completion_tokens\": {}, \"prompt_ms\": {:.1}, \
                 \"ttft_ms\": {}, \"eval_ms\": {:.1}, \"prompt_tps\": {}, \"eval_tps\": {}, \"finish\": {}, \"grammar_ms\": {:.1}, \"resampled\": {}, \
                 \"energy_j\": {}, \"joules_per_token\": {}}}",
                self.at, crate::gguf::jstr(&self.model), self.prompt_tokens, self.cached_tokens, self.completion_tokens, self.prompt_ms,
                f(self.ttft_ms, 1), self.eval_ms, f(self.prompt_tps(), 2), f(self.eval_tps(), 2), crate::gguf::jstr(&self.finish),
                self.grammar_ms, self.resampled, f(self.energy_j, 3), f(self.joules_per_token(), 4))
    }
}

static RING: Mutex<VecDeque<Record>> = Mutex::new(VecDeque::new());

/// Keep a record (the oldest is dropped past `KEEP`).
pub fn push(r: Record) {
    let mut q = RING.lock().unwrap_or_else(|e| e.into_inner());
    if q.len() == KEEP {
        q.pop_front();
    }
    q.push_back(r);
}

/// The records, oldest first.
pub fn records() -> Vec<Record> {
    RING.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect()
}

/// `GET /bankml/metrics`: the records and their totals (tokens in and out, energy where measured).
pub fn json() -> String {
    let rs = records();
    let tin: usize = rs.iter().map(|r| r.prompt_tokens).sum();
    let tout: usize = rs.iter().map(|r| r.completion_tokens).sum();
    let measured: Vec<&Record> = rs.iter().filter(|r| r.energy_j.is_some()).collect();
    let energy = (!measured.is_empty()).then(|| measured.iter().filter_map(|r| r.energy_j).sum::<f64>());
    let energy_out: usize = measured.iter().map(|r| r.completion_tokens).sum();
    let jpt = energy.filter(|_| energy_out > 0).map(|e| e / energy_out as f64);
    let f = |v: Option<f64>, d: usize| v.map(|x| format!("{x:.d$}")).unwrap_or_else(|| "null".into());
    format!("{{\"source\": \"bankml metrics.rs (measured in the engine)\", \"kept\": {}, \"keep\": {KEEP}, \"prompt_tokens\": {tin}, \
             \"completion_tokens\": {tout}, \"energy_j\": {}, \"joules_per_token\": {}, \"records\": [{}]}}",
            rs.len(), f(energy, 3), f(jpt, 4), rs.iter().map(Record::json).collect::<Vec<_>>().join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_and_energy_are_measured_or_null() {
        let r = Record { prompt_tokens: 120, cached_tokens: 20, completion_tokens: 50, prompt_ms: 500.0, eval_ms: 2500.0,
                         energy_j: Some(10.0), ..Default::default() };
        assert_eq!(r.prompt_tps(), Some(200.0));
        assert_eq!(r.eval_tps(), Some(20.0));
        assert_eq!(r.joules_per_token(), Some(0.2));
        let none = Record { completion_tokens: 0, ..r.clone() };
        assert_eq!((none.eval_tps(), none.joules_per_token()), (None, None));
        let unmeasured = Record { energy_j: None, ..r };
        assert!(unmeasured.json().contains("\"energy_j\": null, \"joules_per_token\": null"));
    }

    #[test]
    fn the_ring_keeps_the_last_ones() {
        for i in 0..KEEP + 3 {
            push(Record { completion_tokens: i, ..Default::default() });
        }
        let rs = records();
        assert_eq!(rs.len(), KEEP);
        assert_eq!(rs.last().unwrap().completion_tokens, KEEP + 2);
        assert!(json().starts_with('{') && json().contains(&format!("\"kept\": {KEEP}")));
    }
}
