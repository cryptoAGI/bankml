# `bankML/sampler.rs` — llama-server's sampler chain, token for token with the same seed

## Summary

`sampler.rs` picks the next token from the logits as llama.cpp b11192's sampler chain does when llama-server builds
it for a chat request (`common/sampling.cpp`). The full chain is penalties → dry → top-n-σ → top-k → typical-p →
top-p → min-p → xtc → temperature → dist. bankml reproduces **all of it** (0.3.7), each step from
`src/llama-sampler.cpp` in its float order, with llama.cpp's `sorted` state tracked through the chain. With the same
seed it draws the same tokens as llama-server.

The penalties are new in 0.3.6 (O2): the repeat, frequency and presence penalties over the last `repeat_last_n`
tokens. They let the coach's `ollama_predict` (`repeat_penalty 1.3`) and mindXtrain's imprint gate run on bankml;
without a penalty, greedy `mindx-gen39` degenerates into `,,,,` (CHANGELOG 0.3.6).

Callers: the native engine (`native.rs`, `Native::complete`), and through it `bankml serve --native` (`/v1`, `/api/*`)
and the C API; `bankml generate --sample` (`main.rs`); the Ollama layer, which builds `Params` from Ollama's
defaults and options (`ollama.rs`). Under a grammar, `grammar.rs` supplies the check and the mask to
`sample_constrained`.

## Technical usage

```rust
pub struct Params {
    pub temp: f32, pub top_k: i32, pub top_p: f32, pub min_p: f32, pub min_keep: usize, pub seed: u32,
    pub penalty_last_n: i32, pub penalty_repeat: f32, pub penalty_freq: f32, pub penalty_present: f32,
}
impl Params { pub fn from_gguf(path: &std::path::Path) -> Result<Self, String> }
pub const DEFAULT_SEED: u32 = 0xFFFF_FFFF;

impl Sampler {
    pub fn new(p: Params) -> Result<Self, String>
    pub fn accept(&mut self, t: u32)
    pub fn sample(&mut self, logits: &[f32]) -> u32
    pub fn sample_constrained(&mut self, logits: &[f32], allows: impl Fn(u32) -> bool, mask: impl Fn(&mut [Cand])) -> (u32, bool)
}

pub struct Mt19937 { … }   // new(seed), next_u32(), uniform()
pub struct Cand { pub id: u32, pub logit: f32, pub p: f32 }   // llama_token_data
pub fn partial_sort(v: &mut [Cand], middle: usize)
```

- `Params::from_gguf` resolves as llama-server does without a request: the GGUF's `general.sampling.*` over llama.cpp's
  defaults (temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, `penalty_last_n` 64, penalties neutral), and
  `DEFAULT_SEED`, which means a seed from the clock and process id, as llama.cpp seeds from the random device.
- `Sampler::new` refuses what is not reproduced (top-k outside 1–128) and what llama-server refuses, with its
  message word for word (a negative `repeat_last_n`; a repeat penalty that is not finite and above 0; a non-finite
  frequency or presence penalty).
- `accept(t)` puts a token into the penalties' window. llama-server accepts **every prompt token** before the first
  draw, then each token drawn; callers do the same.
- `sample(logits)` runs the chain once. The RNG advances once per token.
- `sample_constrained` is `common_sampler_sample` with `grammar_first = false`: the chain on the raw logits; if
  `allows` the token, it stands. Otherwise the logits are taken afresh, `mask` sets rejected tokens to −∞, and the
  chain runs again: a second draw from the same generator. The `bool` says whether it redrew.

The chain, step by step:

| step | what it does |
|---|---|
| penalties | for a token seen `count` times in the window: logit divided by the repeat penalty if positive, multiplied if not; then `count · freq + present` taken off. Skipped when `last_n` is 0 or all three are neutral |
| DRY (0.3.7) | over the last `dry_penalty_last_n` tokens: the nearest restart sequence (a breaker) caps the repeat length; a reverse Z-algorithm finds each suffix's repeat; a token that would extend a repeat of at least `dry_allowed_length` loses `multiplier · base^(len − allowed)` (libm `pow`, the exponent clamped); single-token breakers are never penalised. Breakers are built from the vocabulary's pieces once per list and cached by the engine |
| top-n-σ (0.3.7) | over the finite logits: those below `max − n·σ` become −∞ (the squares in double, as C++'s `pow(float, 2)`) |
| top-k | libstdc++'s `std::partial_sort` (heap select, then heap sort), ported exactly, because tied logits are common on a 1-bit model and the order it leaves them in decides the draw |
| typical-p (0.3.7) | the softmax's entropy, each token's distance from it, the tokens sorted by that distance with libstdc++'s `std::sort` (ported: introsort leaves ties in its own order), kept while the running probability ≤ p; leaves the set unsorted |
| top-p | a float softmax over the kept tokens (in their current order), sorted afterwards if typical-p left them unsorted, a float running sum cut where it reaches p (respecting `min_keep`) |
| min-p | a cut at `max + logf(p)`: on an unsorted set the filter keeps order (and falls back to the sorted cut if fewer than `min_keep` pass) |
| XTC (0.3.7) | its own `mt19937` draws a float (`generate_canonical<float, 24>`); when it falls within the probability, the most probable tokens above the threshold are dropped, all but the last of them |
| temperature | `logit / temp`; at temp ≤ 0 every logit but the first maximum becomes −∞. Dynamic (0.3.7) when `dynatemp_range` > 0: the temperature follows the softmax's normalised entropy, `min + (max − min)·entropy^exponent` |
| dist | `expf(logit − max)` summed in double; one `uniform_real_distribution<double>` draw from `std::mt19937` (two 32-bit outputs, libstdc++'s `generate_canonical`) |

Request fields that reach `Params` (native engine, `native::sampling`): `temperature`, `top_k`, `top_p`, `min_p`,
`min_keep`, `seed`, `repeat_last_n`, `repeat_penalty`, `frequency_penalty`, `presence_penalty`. On `/api/*` they come
as Ollama `options`, and from a Modelfile's `PARAMETER`. `bankml generate --sample` takes `--temp`, `--top-k`,
`--top-p`, `--min-p` and `--seed` over the GGUF's defaults.

```rust
let mut s = Sampler::new(Params::from_gguf(model)?)?;
for &t in &prompt { s.accept(t); }        // the prompt fills the penalties' window
let next = s.sample(&w.logits(&rn)?);
s.accept(next);
```

## How it is verified

- Unit tests: `mt19937_reference_value` (the C++ standard's check: the 10,000th output of a 5489-seeded mt19937 is
  4,123,659,995) and `partial_sort_orders_the_top`.
- `oracle_sample_llama_server` (`forward.rs`, `#[ignore]`, in the gate): `testing/sample_oracle.py` has llama-server
  sample 40 continuations with fixed seeds over temperature 0–1.5, top-k 5–128, top-p and min-p, and keeps the
  parameters the server reports. Replayed through bankml's forward pass and sampler: **40 of 40, 1,175 tokens**.
  `oracle_llama_server_bonsai_1_7b` and `oracle_llama_server_llama_f16` do the same on the O4 models (40 of 40 each).
- `oracle_samplers` (`native.rs`, 0.3.7): `testing/penalty_oracle.py --kind sampler`, 23 variants × 4 prompts:
  mindx-gen39 **76 / 76** (3,576 tokens) and Bonsai-1.7B **76 / 76** (2,587 tokens), **16 / 16** refusals each with
  llama-server's message. `oracle_std_sort`: libstdc++'s own `std::sort` on 876 key arrays, **876 / 876** identical.
- `oracle_penalties`, `oracle_penalties_8b` (`native.rs`): `testing/penalty_oracle.py`, llama-server b11192 from an
  empty cache, greedy and seeded, 17 variants × 4 prompts made to repeat, `repeat_last_n` smaller than, equal to and
  larger than the prompt. mindx-gen39 **56 / 56** (2,478 tokens), Bonsai-1.7B **56 / 56** (1,895 tokens), Bonsai-8B
  **56 / 56** (1,568 tokens); **12 / 12** refusals each with llama-server's message (CHANGELOG 0.3.6).
- `penalty_oracle_live` (gate): the recorded cases through a running `serve --native`, via `/v1` and `/api/chat`:
  **85 / 85**, refusals as 400s.
- Under a grammar: `oracle_json_mode`, `oracle_json_schema*` check the redraw's use of the generator (152 of 860
  tokens redrawn on Bonsai-8B in JSON mode).

## Advantages and efficiency

- **Reproducible sampling.** A seed gives the same answer as llama-server, so seeded answers can be checked by an
  oracle like greedy ones, and a grammar's redraw consumes the generator exactly as llama-server's does.
- **Small after top-k.** Top-k runs a heap select over the vocabulary in place and truncates to at most 128
  candidates; top-p, min-p, temperature and the draw then work on that short, sorted list.
- **Penalties cost nothing when off.** They are skipped entirely when disabled, as llama.cpp disables them. The window
  is a ring (`VecDeque`) with a per-token count (`HashMap`), updated in constant time per accepted token.
- **The redraw only when needed.** Under a grammar the whole-vocabulary mask runs only when the first draw breaks
  the grammar (see [grammar.md](grammar.md) for its cost).
- **Rust practice.** No crates: mt19937 and libstdc++'s heap algorithms are written out. No `unsafe`. Refusals are
  `Err`s carrying llama-server's own text.
- **DRY's breakers cost once.** Building them scans every token's piece (151,669 on the Qwen3 models); the engine
  keeps the result per breaker list, so a conversation pays it once.
- **Next** (docs/TODO.md 0.4.0): `n_probs` and logprobs with bit-exact probabilities.

## Limitations

- Not reproduced, refused: mirostat, and a custom `samplers` order (`native::sampling`). DRY, XTC, top-n-σ and
  dynamic temperature are not Ollama options, so they come through `/v1` and the C API only.
- A DRY breaker whose split point falls inside a multi-byte character is refused (bankML tokenizes whole characters).
- Top-k 0 or above 128 is refused (llama.cpp sorts larger sets another way).
- mindXtrain's `no_repeat_ngram_size` is a transformers rule, not a llama.cpp sampler; it is not here (docs/OLLAMA.md).
- `DEFAULT_SEED` seeds from the clock and process id, so such a run is not reproducible, as in llama.cpp.
- `bankml generate --sample` has no penalty flags; penalties there come from the GGUF's defaults.

## See also

- [../oracles.md](../oracles.md) §1d (step eleven), §5c
- [../OLLAMA.md](../OLLAMA.md) — O2, the sampler chain
- [../TODO.md](../TODO.md) — 0.4.0
- [../usage.md](../usage.md) §13 — `bankml generate --sample`
- Sibling pages: [grammar.md](grammar.md), [forward.md](forward.md), [native.md](native.md),
  [serve.md](serve.md), [ollama.md](ollama.md)
