# `bankML/prompt_cache.rs` — llama-server's host prompt cache, for the native engine's one slot

## Summary

`prompt_cache.rs` is llama-server b11192's `server_prompt_cache` (`--cache-ram`, on by default at 8192 MiB). When a
request shares little of its prompt with the slot, the slot's state (its tokens and every layer's K and V) is copied
to RAM before it is overwritten. If a cached state would keep more of the new prompt, it moves back into the slot.
Interleaved conversations then each find their own prefix again. That changes `cache_n`, the point where prefill
starts, and with it the arithmetic. So it is needed for the answers to equal llama-server's, not only for speed.

Caller: `native.rs` (`Native::complete`), before the longest-common-prefix reuse. It was added in 0.3.8.

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
}
```

The rules, from `server_context::get_available_slot` and `server_prompt_cache::{alloc, load, update}`:

| step | rule |
|---|---|
| when | similarity (LCP / prompt length) ≤ 0.1 (`--slot-prompt-similarity`), an empty slot included; or above it, but the slot would keep under half its tokens |
| save | skip an empty slot or one an entry already holds whole; drop entries the slot holds whole; evict the oldest until the new entry fits the byte limit; an entry larger than the limit alone is not cached |
| load | among entries that keep at least a quarter of themselves, the one with both a larger kept fraction and a larger similarity than the slot's own moves into the slot |
| evict | the oldest while over the byte limit, then while over the token limit (n_ctx, raised to what the byte limit holds at the cache's average bytes per token) |

The byte limit is `BANKML_CACHE_RAM` in MiB, as `--cache-ram` (0 off, -1 no limit). Unset, it is llama-server's
8192 MiB, but at most a quarter of the memory available when the model loads.

## Oracle

`testing/session_oracle.py` (`session_oracle_live` in the gate): three Savante-style conversations take turns
(A1 B1 C1 A2 …) through llama-server b11192 `-np 1` and through `bankml serve --native`. Every turn has the same text,
counts and `cache_n`: 14 / 14 checks with the queue (four simultaneous requests, each answered as when asked
alone). Without the cache, the interleaved turns differed. The unit test covers the rules without a model.

## Limitations

- One slot. The cache serves interleaved conversations through it; it does not serve them in parallel.
- llama-server's prompt checkpoints (for recurrent and SWA models) are not modelled; the supported graphs have none.
