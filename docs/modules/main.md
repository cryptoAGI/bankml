# `bankML/main.rs` — the `bankml` command line

## Summary

`main.rs` is the `bankml` binary. It parses the arguments, calls into the library (`bankML/bankml.rs` and its
modules), and maps each outcome to an exit code. It holds little logic of its own: the guard, the pin and `verify`
form the gate; `serve` starts the gateway; `create` and `convert` make models; `tokenize`, `chat-template` and
`generate` expose the native pipeline one stage at a time.

Exit codes follow `gguf_guard.py`: **0** play, **2** refuse, **3** need more (a truncated file), **1** usage or I/O.
An unreadable FORK.json is an I/O error (1), not "unpinned" (2).

## Technical usage

The usage string, as the source states it, one line per subcommand:

| command | what it does |
|---|---|
| `bankml usage [PID …]` | memory, cores, and each process's resident memory and CPU % over 0.5 s (default: bankml itself) |
| `bankml guard FILE [--engine mainline\|prism] [--json]` | the header check: play, refuse (with reasons) or need_more |
| `bankml sha256 FILE` | the file's sha256 |
| `bankml pin FILE --fork FORK.json` | the sha256 against the fork's record |
| `bankml verify FILE --fork FORK.json [--engine mainline\|prism] [--json]` | the guard, then the pin |
| `bankml serve FILE --fork FORK.json [--upstream HOST:PORT \| --spawn LLAMA_SERVER] [--listen HOST:PORT] [--threads N] [--ctx N] [--spec-ngram] [--slot-dir DIR]` | the gateway in front of llama-server ([serve.md](serve.md)) |
| `bankml serve FILE --fork FORK.json --native [--listen HOST:PORT] [--upstream HOST:PORT] [--ctx N] [--registry [DIR]] [--keep-alive DUR]` | answers from bankML's own forward pass; OpenAI `/v1` and Ollama `/api`; with `--registry`, every model pinned in DIR by name, one resident at a time |
| `bankml tokenize MODEL.gguf [--no-special] < text` | token ids, as llama.cpp's `/tokenize` |
| `bankml chat-template MODEL.gguf < messages.json` | the prompt, as llama.cpp's `/apply-template` |
| `bankml generate MODEL.gguf [--max N] [--json] [--sample [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S]] < messages.json\|text` | bankML's forward pass: greedy, or llama-server's sampler chain; `--json` is JSON mode |
| `bankml create NAME -f Modelfile [--registry DIR] [--models DIR]` | a derived model written as `DIR/NAME.MODEL.json` over the base's pin ([create.md](create.md)) |
| `bankml convert SAFETENSORS_DIR -o OUT.gguf [--outtype f16] [--model-name NAME] [--fork FORK.json --source SRC] [--ignore-model-card]` | Llama safetensors to GGUF F16, byte-identical to llama.cpp b11192 ([convert.md](convert.md)) |
| `bankml gpu [--remote \| --verify]` | every video card found and which bankml will use; `--remote` adds Hugging Face's rented GPUs, listed only; `--verify` runs the bit-exact kernel oracle on each card |
| `bankml version` | `bankml <version>` (also `--version`, `-V`) |

`--slot-dir DIR` also works with `--native` (0.3.8: llama-server's slot save, restore and erase on the engine
address, [serve.md](serve.md)); the usage string lists it on both `serve` lines. Depth for each is in
[../usage.md §13](../usage.md#13-reference-commands-ports-environment).

### Defaults and environment read here

- `serve`: `--listen 127.0.0.1:18093`, `--upstream 127.0.0.1:18092`, `--threads 3`, `--ctx 4096`.
- `--registry` with no directory, and `create` without `--registry`: `$BANKML_FORKS`, else
  `~/.local/share/bankml/forks` (the importer's forks directory, as `sAGI/models.py`).
- `generate`: `--max` defaults to 256. With `--sample`, the model's GGUF defaults are the base, and the flags override
  them. `--json` opens the native engine with a 4096-token context and runs greedy unless `--sample` is given.
- `--engine prism` selects the guard's Prism rules; anything else is mainline.
- `serve --native` does not use `--threads` (that is llama-server's `-t` when spawned); the engine reads
  `BANKML_THREADS`, `BANKML_LLAMA_THREADS`, `BANKML_CACHE_TYPE` (`f16` or `q8_0`, 0.3.9), `BANKML_CACHE_RAM` (the host
  prompt cache in MiB, 0.3.8) and `BANKML_GPU`, `BANKML_GPU_SHARE`, `BANKML_GPU_LIMIT` from the environment
  ([forward.md](forward.md), [prompt_cache.md](prompt_cache.md), [gpu.md](gpu.md)).

### Exit codes by command

| command | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `guard` | play | cannot read | refuse | need more |
| `verify` | play | cannot read | refuse | guard needs more bytes |
| `pin` | pinned | no or unreadable `--fork` | refuse | — |
| `serve` | — | no or unreadable `--fork` | refused at start | — |
| `create`, `convert` | done | missing `-f` / `-o`, or the pin cannot be written | refuse | — |
| `generate`, `chat-template` | done | `generate`: unreadable GGUF defaults | an error | — |
| `gpu --verify` | every card verified | — | a card failed | — |

```sh
target/release/bankml verify .models/Bonsai-8B-Q1_0.gguf --fork FORK.json --json
echo 'Describe a cat.' | target/release/bankml generate .models/Bonsai-8B-Q1_0.gguf --json --max 64
```

## How it is verified

- `testing/cli.rs` runs the built binary end to end: `version_and_usage`, `guard_verdicts_and_exit_codes`,
  `hostile_headers_refuse_not_crash`, `pin_and_verify`, `serve_gates_and_signs_answers`,
  `serve_refuses_an_unverified_model_or_a_different_upstream_file`, `create_writes_a_layer_over_a_pin`,
  `convert_refuses_with_the_reason`.
- The commands' engines have their own oracles: the tokenizer and chat-template oracles, the greedy and sampled
  llama-server oracles for `generate`, and `gpu --verify` itself (docs/oracles.md).

## Advantages and efficiency

- **One static binary, no dependencies.** Cargo.toml has an empty `[dependencies]`; the release profile uses
  `lto = true`, `codegen-units = 1` and `panic = "abort"`, and picks AVX2 at run time, so no target-cpu flag is needed.
  The toolchain is pinned to Rust 1.99.0 in `rust-toolchain.toml`.
- **Scriptable gate.** Distinct exit codes and `--json` output let an installer or a supervisor act on play, refuse
  and truncated without parsing text.
- **Each stage checkable alone.** `tokenize`, `chat-template` and `generate` print what llama.cpp's `/tokenize`,
  `/apply-template` and server would, so a difference can be located to one stage.

## Limitations

- Argument parsing is positional and minimal: an unknown command prints the usage and exits 1.
- `generate --sample` has no flags for the penalties. Since 0.3.6 the sampler applies the GGUF's
  `general.sampling.penalty_last_n` and `penalty_repeat` when a model sets them (sampler.rs), and the prompt fills the
  penalties' window; none of the five native models sets them (CHANGELOG 0.3.6).
- `convert --outtype` accepts `f16` only.

## Design notes

- **Where each command comes from.** `tokenize` is phase P3 step one and `chat-template` P3 step two of
  [../BUILD_HISTORY.md](../BUILD_HISTORY.md); `generate` is P3's forward pass. `create` (the native `ollama create`)
  and `convert` are phase O5 of [../OLLAMA.md](../OLLAMA.md). `generate --json` (JSON mode through the native engine)
  arrived in 0.3.3.
- **What each stage is checked against.** `tokenize` is token-identical to llama.cpp b11192 on its oracle;
  `chat-template` is byte-identical to `/apply-template`; greedy `generate` is token-identical to llama-server b11192
  on its oracle (prompts under 64 tokens, contexts under 512 cells). `generate --json` applies the grammar, its
  prefill and the redraw exactly as llama-server answers `response_format: {"type": "json_object"}`, greedy being its
  temperature 0, and prints only the content. With `--sample`, the prompt fills the penalties' window first, as
  llama-server's does (O2).
- **`gpu`.** The listing merges every Vulkan device with the kernel's sysfs view and shows the selection. `--verify`
  is the on-card oracle: each selected card runs bankML's kernels and must give the CPU kernels' bits, or bankML does
  not use it.
- **`usage`** is bankML's equivalent of psutil.
- **Exit code 1 for an unreadable FORK.json.** Version 0.0.1 reported it as "unpinned" (2); the two are now distinct.

## See also

- [../usage.md §6](../usage.md#6-by-hand-build-verify-serve) · [§13 Reference](../usage.md#13-reference-commands-ports-environment) ·
  [../oracles.md](../oracles.md) · [../TODO.md](../TODO.md)
- [serve.md](serve.md) · [native.md](native.md) · [create.md](create.md) · [convert.md](convert.md) · [gguf.md](gguf.md) ·
  [sampler.md](sampler.md)
