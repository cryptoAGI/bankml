# `bankML/convert.rs` — `bankml convert`: safetensors to GGUF F16, byte-identical to llama.cpp

## Summary

`convert.rs` turns a Hugging Face safetensors directory into a GGUF F16 file for the **Llama architecture as
SmolLM2-135M and mindX's `mindx-genN` use it**. It is written to be **byte-identical** to llama.cpp b11192's
`convert_hf_to_gguf.py DIR --outtype f16`, read from the tag's `conversion/{base,llama}.py` and `gguf-py`, in its
order of operations.

It exists because mindXtrain merges each generation's LoRA into safetensors (`ollama_push/merged`) and serves it
through Ollama. This converter and `bankml create` take that merged directory instead: one pinned GGUF, with the
persona as a verified layer ([create.md](create.md)). `bankml create … FROM <directory>` calls `convert()` and pins
the result; `bankml convert` runs it alone.

What would make llama.cpp write something this does not reproduce is refused, with the reason. Nothing is
approximated.

## Technical usage

```sh
bankml convert DIR -o mindx-gen39-F16.gguf --fork mindx-gen39-F16.gguf.FORK.json \
    --source "PYTHAI/mindXascension@4bd31b9d weights/gen39"
```

| flag | meaning |
|---|---|
| `-o OUT.gguf` / `--outfile` | required |
| `--outtype f16` | the only type accepted; another is refused (exit 2) |
| `--model-name NAME` | `general.name` instead of the heuristics' (llama.cpp's flag of the same meaning) |
| `--fork FORK.json` | also write the pin: the output's sha256 and size, and every input file's |
| `--source SRC` | where the directory came from, recorded in the pin (default: the directory's canonical path) |
| `--ignore-model-card` | convert although a `README.md` is present; the result then differs from llama.cpp's |

On success it prints `<sha256>  <out>` on stdout and a summary (name, tensors, metadata keys, weights, bytes) on
stderr. A refusal prints `bankml convert: refuse: …` and exits 2.

### API

```rust
pub fn convert(dir: &Path, out: &Path, opt: &Options) -> Result<Report, String>
pub struct Options { pub model_name: Option<String>, pub ignore_model_card: bool }
pub fn fork_json(r: &Report, source: &str) -> String
```

`Report` carries `out`, `sha256`, `bytes`, `tensors`, `kv`, `params`, `name`, and `inputs` (each input's name, size
and sha256).

### What it writes

- **Metadata, in llama.cpp's order:** `general.architecture`, `general.type`, the sampling defaults of
  `generation_config.json`, the name heuristics of `gguf-py/metadata.py`, `TextModel.set_gguf_parameters` with the
  defaults `AutoConfig(LlamaConfig).to_dict()` supplies (`head_dim`, `rope_theta` 10000, `rms_norm_eps` 1e-6, the kv
  heads), `file_type` 1, `vocab_size`, `rope.dimension_count`, `quantization_version` 2, then the tokenizer.
- **Tensors:** every safetensors part in name order; HF names mapped to GGUF names; the Q and K rows permuted from
  HF's rotate-half layout back to GGML's pairs; 1-D tensors and `*_norm.weight` in F32, every other matrix in F16 by
  round-to-nearest-even; BF16 read exactly; tied embeddings write no `output.weight`.
- **Tokenizer:** the `gpt2` byte-level BPE path: tokens and types, merges, the special token ids, `add_*_token`, the
  chat template (`tokenizer_config.json` or `chat_template.jinja`), and Llama's `add_bos_token = false` for
  49,152-token vocabularies.
- **Layout:** GGUF v3, little-endian, alignment 32. The file is written to `OUT.gguf.part`, hashed as it is written,
  then renamed.

### `general.name` comes from the directory

llama.cpp names the model from the directory's name, and so does bankML. The bytes, and so the pin, depend on it.
CHANGELOG 0.3.5: gen39 converted from a directory named `merged` is `bb41f62d…`; from `mindx-gen39` it is the pin
`6b64c748…`. A directory called `merged` gives `general.name` `Merged`, as llama.cpp's would. `--model-name` overrides.

### Tokenizers

llama.cpp names a BPE pre-tokenizer by hashing the token ids its Python tokenizer gives one fixed text (`chkhsh`).
bankML has no Python, so `KNOWN_TOKENIZERS` keys on the sha256 of the vocabulary and merges
(`vocab_merges_sha256`) plus the pre-tokenizer's shape, and records the `chkhsh` each entry gave. Today it has one
entry, `smollm` (SmolLM2-135M-Instruct and mindXascension gen39). gen39's tokenizer lost SmolLM2's `Digits`
pre-tokenizer when transformers 5.8 re-saved it; llama.cpp still names it `smollm`, and both shapes are accepted.

## How it is verified

- Unit tests: `name_heuristics_as_gguf_py`, `permute_matches_numpy_reshape_swapaxes`,
  `kv_set_keeps_the_first_position`, `regexes_as_gguf_py`.
- `oracle_convert_b11192` (`#[ignore]`, gate): `BANKML_CONVERT_DIR` against llama.cpp's GGUF in
  `BANKML_CONVERT_ORACLE`, sha256 for sha256. The gate runs it for each directory under `.models/convert/`.
- `oracle_name_heuristics` (gate): gguf-py's heuristics, written by `testing/convert_oracle.py --names`; 168 of 168
  Hub-style ids agree (CHANGELOG 0.3.5).
- `testing/convert_oracle.py`: `--record` runs llama.cpp's script, `--bankml` compares, `--compare` diffs two GGUFs
  key by key and tensor by tensor, `--chkhsh` measures a tokenizer for `KNOWN_TOKENIZERS`.
- `testing/cli.rs`: `convert_refuses_with_the_reason`.
- Recorded results (docs/OLLAMA.md): SmolLM2-135M-Instruct @ `12fd25f7` → `e9aba089…`, mindXascension gen39 @
  `4bd31b9d` → `6b64c748…`, the same as llama.cpp b11192. Run directly in 0.3.5 against `convert_hf_to_gguf.py`
  (torch 2.14.1+cpu, transformers 5.18.0): gen39 `6b64c748…`, SmolLM2 `ec30a679…`.

## Advantages and efficiency

- **No Python, no torch.** llama.cpp's converter is a Python script that needs torch (usage.md §5); `bankml convert`
  is part of the zero-dependency binary and gives the same bytes.
- **Pinned at birth.** `--fork` (and `create`) write a FORK.json with the output's sha256 and every input's, so the
  result is verified like any imported model.
- **Efficient.** The safetensors are memory-mapped read-only; the output goes through a 1 MiB buffered writer that
  hashes as it writes, so no second pass is needed for the pin. docs/OLLAMA.md records a conversion at 2.6 s.
- **Rust practice.** Every unsupported case is a `Result` error with the reason; the output is written to a `.part`
  file and renamed, so a failed run leaves no partial GGUF under the final name.
- **Next:** docs/TODO.md (0.6.0) lists Llama 3.x (rope factors, the `llama3` pre-tokenizer, its template) as still
  open and refused today; the converter likewise refuses rope scaling and unknown pre-tokenizers.

## Limitations

Refused, with the reason in the source:

- Any architecture other than `LlamaForCausalLM`: "use llama.cpp's convert_hf_to_gguf.py for the rest".
- `config.json` with `text_config`, `quantization_config`, `num_local_experts` or `draft_vocab_size`;
  `is_causal: false`.
- Rope scaling (a `rope_type` other than `default`); attention or MLP biases.
- A `README.md` model card, unless `--ignore-model-card`: llama.cpp would copy its licence, tags, languages, base
  models and datasets in.
- SentencePiece (`tokenizer.model`) and Llama-HF BPE vocabularies; a non-BPE `tokenizer.json`; a normalizer; most
  post-processors; non-special added tokens; named or listed chat templates; any tokenizer not in `KNOWN_TOKENIZERS`.
- Tensor dtypes other than BF16, F16 and F32; tensor names it cannot map.
- Output types other than F16.
- A conversion under `create` never overwrites an existing pin.

## See also

- [../usage.md, derived models and conversion](../usage.md#derived-models-and-conversion-bankml-create-bankml-convert-o5-035) ·
  [../OLLAMA.md, what O5's first cut built](../OLLAMA.md#what-o5s-first-cut-built-035) · [../oracles.md](../oracles.md) ·
  [../TODO.md](../TODO.md)
- [create.md](create.md) · [gguf.md](gguf.md) · [tokenizer.md](tokenizer.md) · [main.md](main.md)
