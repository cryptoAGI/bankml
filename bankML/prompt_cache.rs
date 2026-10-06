// SPDX-License-Identifier: MIT OR Apache-2.0
//! llama-server's host prompt cache (b11192 `server_prompt_cache`, `--cache-ram`), for one slot.
//!
//! When a request shares little of its prompt with the slot, the slot's state (its tokens and KV rows) is copied to
//! RAM before it is overwritten, and the cached state that keeps more of the new prompt is moved back into the slot.
//! Interleaved conversations then each find their own prefix again, and the answers equal llama-server's: the prefix
//! reused (`cache_n`) decides where prefill starts, and with it the arithmetic.
//!
//! The rules, as `server_context::get_available_slot` and `server_prompt_cache::{alloc, load, update}`:
//! - update the cache when the slot's similarity to the prompt (LCP / prompt length) is at most 0.1
//!   (`--slot-prompt-similarity`; an empty slot included), or above it but the slot would keep under half of its
//!   tokens (`f_keep < 0.5`);
//! - save: skip an empty slot or one already held whole by an entry; drop entries the slot holds whole; evict the
//!   oldest until the new entry fits the byte limit;
//! - load: the entry with both a larger kept fraction and a larger similarity than the slot's own, among entries that
//!   would keep at least a quarter of themselves, moves into the slot;
//! - then evict the oldest while over the byte limit, then while over the token limit (n_ctx, raised to what the byte
//!   limit holds at the cache's average bytes per token).

use std::collections::VecDeque;

/// llama-server's `--slot-prompt-similarity` default.
const SLOT_PROMPT_SIMILARITY: f32 = 0.1;

struct Entry<S> {
    tokens: Vec<u32>,
    state: S,
    bytes: usize,
}

pub struct PromptCache<S> {
    entries: VecDeque<Entry<S>>,
    limit_bytes: usize,
    limit_tokens: usize,
}

/// The length of the longest common prefix (`server_tokens::get_common_prefix`).
pub fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl<S> PromptCache<S> {
    /// `limit_bytes` 0: no byte limit (llama-server's `--cache-ram -1`); `limit_tokens`: the context size.
    pub fn new(limit_bytes: usize, limit_tokens: usize) -> Self {
        PromptCache { entries: VecDeque::new(), limit_bytes, limit_tokens }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    pub fn n_tokens(&self) -> usize {
        self.entries.iter().map(|e| e.tokens.len()).sum()
    }

    /// Whether a request with `prompt` updates the cache, given what the slot holds (single-slot selection).
    pub fn wants_update(slot: &[u32], prompt: &[u32]) -> bool {
        if slot.is_empty() {
            return true; // no similarity: selected by LRU, which always updates
        }
        let f_sim = lcp(slot, prompt) as f32 / prompt.len() as f32;
        if f_sim > SLOT_PROMPT_SIMILARITY {
            let f_keep = (f_sim * prompt.len() as f32) / slot.len() as f32;
            f_keep < 0.5
        } else {
            true
        }
    }

    /// Copy the slot into the cache (`alloc`); `state` is called only when the copy is kept.
    pub fn save(&mut self, slot: &[u32], bytes: usize, state: impl FnOnce() -> S) -> bool {
        if slot.is_empty() || self.entries.iter().any(|e| lcp(&e.tokens, slot) == slot.len()) {
            return false;
        }
        if self.limit_bytes > 0 && bytes > self.limit_bytes {
            return false;
        }
        self.entries.retain(|e| lcp(&e.tokens, slot) != e.tokens.len());
        if self.limit_bytes > 0 {
            while !self.entries.is_empty() && self.bytes() + bytes > self.limit_bytes {
                self.entries.pop_front();
            }
        }
        self.entries.push_back(Entry { tokens: slot.to_vec(), state: state(), bytes });
        true
    }

    /// The cached state that serves `prompt` better than the slot does, removed from the cache (`load`).
    pub fn take_better(&mut self, slot: &[u32], prompt: &[u32]) -> Option<(Vec<u32>, S)> {
        let base = lcp(slot, prompt);
        let mut f_keep_best = if slot.is_empty() { -1.0 } else { base as f32 / slot.len() as f32 };
        let mut f_sim_best = base as f32 / prompt.len() as f32;
        let mut best = None;
        for (i, e) in self.entries.iter().enumerate() {
            let l = lcp(&e.tokens, prompt);
            let f_keep = l as f32 / e.tokens.len() as f32;
            let f_sim = l as f32 / prompt.len() as f32;
            if f_keep < 0.25 {
                continue; // don't trash large prompts
            }
            if f_keep_best < f_keep && f_sim_best < f_sim {
                (f_keep_best, f_sim_best, best) = (f_keep, f_sim, Some(i));
            }
        }
        let e = self.entries.remove(best?)?;
        Some((e.tokens, e.state))
    }

    /// Evict the oldest entries over the limits (`update`).
    pub fn update(&mut self) {
        if self.limit_bytes > 0 {
            while !self.entries.is_empty() && self.bytes() > self.limit_bytes {
                self.entries.pop_front();
            }
        }
        let per_token = (self.bytes() as f32 / self.n_tokens().max(1) as f32).max(1.0);
        let limit = if self.limit_bytes > 0 { self.limit_tokens.max((self.limit_bytes as f32 / per_token) as usize) } else { self.limit_tokens };
        if self.limit_tokens > 0 {
            while !self.entries.is_empty() && self.n_tokens() > limit {
                self.entries.pop_front();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_conversations_find_their_prefix_again() {
        let sys = [1, 2, 3, 4];
        let a1: Vec<u32> = sys.iter().copied().chain([10, 11, 12, 13]).collect();
        let b1: Vec<u32> = sys.iter().copied().chain([20, 21, 22, 23]).collect();
        let a2: Vec<u32> = a1.iter().copied().chain([14, 15]).collect();
        let mut c: PromptCache<&str> = PromptCache::new(0, 4096);
        // the slot holds A; B shares the system prefix, which keeps exactly half of the slot: no update (not < 0.5)
        assert!(!PromptCache::<&str>::wants_update(&a1, &b1));
        let b_long: Vec<u32> = sys.iter().copied().chain([20, 21, 22, 23, 24, 25]).collect();
        let a_long: Vec<u32> = a1.iter().copied().chain([16, 17, 18]).collect();
        assert!(PromptCache::<&str>::wants_update(&a_long, &b_long)); // keeps 4 of 11
        assert!(c.save(&a_long, 11, || "A"));
        assert!(c.take_better(&a_long, &b_long).is_none()); // A itself is no better than the slot (also A)
        // later the slot holds B; A's next turn takes A back
        let got = c.take_better(&b_long, &a2);
        assert_eq!(got, Some((a_long.clone(), "A")));
        assert!(c.is_empty());
        // a saved state held whole by an entry is not saved twice; an entry the slot holds whole is replaced
        assert!(c.save(&a1, 8, || "A1"));
        assert!(!c.save(&a1[..6], 6, || "A1 part"));
        assert!(c.save(&a_long, 11, || "A long"));
        assert_eq!(c.len(), 1);
        // the byte limit evicts the oldest
        let mut c: PromptCache<u8> = PromptCache::new(20, 4096);
        assert!(c.save(&a1, 8, || 1));
        assert!(c.save(&b1, 8, || 2));
        assert!(c.save(&[7, 7, 7], 8, || 3));
        assert_eq!((c.len(), c.bytes()), (2, 16));
        assert!(!c.save(&[9], 21, || 4)); // over the limit on its own: not cached
    }
}
