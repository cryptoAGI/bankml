# `bankML/prompt_cache.rs` — llama-server's host prompt cache, for the native engine's one slot

## Summary

`prompt_cache.rs` is llama-server b11192's `server_prompt_cache` (`--cache-ram`, on by default at 8192 MiB). When a
request shares little of its prompt with the slot, the slot's state (its tokens and every layer's K and V) is copied
to RAM before it is overwritten. If a cached state would keep more of the new prompt, it moves back into the slot.
Interleaved conversations then each find their own prefix again. That changes `cache_n`, the point where prefill
starts, and with it the arithmetic. So it is needed for the answers to equal llama-server's, not only for speed.

Caller: `native.rs` (`Native::complete_with`, which `complete` calls), before the longest-common-prefix reuse. It was
added in 0.3.8 (unreleased).

## Technical usage

```rust
pub struct PromptCache<S>                              // S: the slot state; Vec<KvCache> in native.rs
pub fn lcp(a: &[u32], b: &[u32]) -> usize
impl<S> PromptCache<S> {
    pub fn new(limit_bytes: usize, limit_tokens: usize) -> Self   // 0 bytes: no byte limit
    pub fn wants_update(slot: &[u32], prompt: &[u32]) -> bool
    pub fn save(&mut self, slot: &[u32], bytes: usize, state: impl FnOnce() -> S) -> bool
    pub fn take_better(&mut self, slot: &[u32], prompt: &[u32]) -> Option<(Vec<u32>, S)>
    pub fn update(&mut self)
    pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool
    pub fn bytes(&self) -> usize; pub fn n_tokens(&self) -> usize   // what the entries hold
}
```

How `native.rs` uses it, once per request:

```rust
if PromptCache::<Vec<KvCache>>::wants_update(&slot.tokens, prompt) {
    pc.save(&slot.tokens, bytes, || slot.caches.clone());        // the copy is made only if it is kept
    if let Some((t, c)) = pc.take_better(&slot.tokens, prompt) { // a cached state that serves the prompt better
        (slot.tokens, slot.caches) = (t, c);
    }
    pc.update();                                                 // evict over the limits
}
```

`bytes` is the slot's real size (`KvCache::bytes`), so with `BANKML_CACHE_TYPE=q8_0` (0.4.0) each saved state is
53 % of an f16 one's bytes (CHANGELOG 0.4.0) and the same limit holds more of them.

The rules, from `server_context::get_available_slot` and `server_prompt_cache::{alloc, load, update}`:

| step | rule |
|---|---|
| when | similarity (LCP / prompt length) ≤ 0.1 (`--slot-prompt-similarity`), an empty slot included; or above it, but the slot would keep under half its tokens |
| save | skip an empty slot or one an entry already holds whole; drop entries the slot holds whole; evict the oldest until the new entry fits the byte limit; an entry larger than the limit alone is not cached |
| load | among entries that keep at least a quarter of themselves, the one with both a larger kept fraction and a larger similarity than the slot's own moves into the slot |
| evict | the oldest while over the byte limit, then while over the token limit (n_ctx, raised to what the byte limit holds at the cache's average bytes per token) |

The byte limit is `BANKML_CACHE_RAM` in MiB, as `--cache-ram` (0 off, -1 no limit). Unset, it is llama-server's
8192 MiB, but at most a quarter of the memory available when the model loads.

## How it is verified

- `testing/session_oracle.py` (`session_oracle_live` in the gate): three Savante-style conversations take turns
  (A1 B1 C1 A2 …) through llama-server b11192 `-np 1` and through `bankml serve --native`. Every turn has the same
  text, counts and `cache_n`: 14 / 14 checks with the queue (four simultaneous requests, each answered as when asked
  alone; CHANGELOG 0.3.8). Without the cache, each turn found only the system prompt in the slot, prefill started
  elsewhere, and the answers differed.
- Unit test `interleaved_conversations_find_their_prefix_again`: the rules without a model — when an update is due
  (exactly half kept is not "under half"), a conversation taking its state back, no duplicate saves, an entry the
  slot holds whole replaced, the oldest evicted at the byte limit, and an entry larger than the limit not cached.

## Advantages and efficiency

- **The same answers as llama-server across turns.** Where prefill starts decides which kernels each row takes, so
  reproducing the cache is part of being token-identical, not only a speed-up.
- **A returning conversation skips its history.** It takes its saved state back and computes only its new tokens.
- **Copies only what is kept.** `save` calls the state closure only when the entry will be stored; nothing is copied
  when the slot already serves the prompt well (`wants_update` false).
- **Bounded.** The byte limit defaults to 8192 MiB but at most a quarter of the memory available at load
  (`cache_ram_limit` in `native.rs`), so the cache cannot push a small machine into swap or past a service's memory
  ceiling; the token limit evicts too.
- **Rust practice.** Generic over the state type `S`, so the rules are tested with `&str` states; no `unsafe`; std
  only (`VecDeque`).

## Limitations

- One slot. The cache serves interleaved conversations through it; it does not serve them in parallel.
- llama-server's prompt checkpoints (for recurrent and SWA models) are not modelled; the supported graphs have none.
- The cache lives in RAM only. A state that must survive a restart is saved with the slot API
  (`/slots/0?action=save`, [serve.md](serve.md)).

## See also

- [native.md](native.md) (the prompt cache rule, slot files), [serve.md](serve.md) (`/slots`, the single-slot
  contract)
- [../TODO.md](../TODO.md) — 0.4.0, the single-slot contract; more than one slot waits for continuous batching (O8)
- [../OLLAMA.md](../OLLAMA.md) — O2 and O8
