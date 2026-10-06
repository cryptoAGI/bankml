# Installing and configuring bankML: the complete reference

This is the full reference for installing bankML 0.3.6 and the Savante UI, and for every setting they accept.
Settings from the next release (0.3.7, 0.3.8, and 0.3.9 in progress; not yet tagged) are marked with their version;
[CHANGELOG.md](../CHANGELOG.md) has the details. What each source module does is in [modules/](modules/README.md). For a
first install, read [usage.md §1–§3](usage.md#1-install); the README's *Install and use* gives the short version. This
page assumes you have read one of them. It covers every step, flag, port, path and environment variable, what each
one is for, and what it costs.

Everything here comes from the source: `install.sh`, `bankML/main.rs`, `bankML/serve.rs`, `bankML/native.rs`, the
kernels, `sAGI/*.py` and `testing/`. If a setting is not on this page, bankML does not read it.

- [1. Requirements in detail](#1-requirements-in-detail)
- [2. install.sh in full](#2-installsh-in-full)
- [3. Building by hand](#3-building-by-hand)
- [4. Models: where they live, how they are pinned](#4-models-where-they-live-how-they-are-pinned)
- [5. Running bankml serve, and every other command](#5-running-bankml-serve-and-every-other-command)
- [6. Environment variables: the complete list](#6-environment-variables-the-complete-list)
- [7. Performance and resource tuning](#7-performance-and-resource-tuning)
- [8. Running as a service](#8-running-as-a-service)
- [9. Security and exposure](#9-security-and-exposure)
- [10. Development and the release gate](#10-development-and-the-release-gate)
- [11. Troubleshooting](#11-troubleshooting)

---

## 1. Requirements in detail

### Platform and toolchain

| what | detail |
|---|---|
| OS and architecture | Linux on x86_64. bankML itself builds wherever Rust 1.99 does. The llama.cpp engine the installer downloads (`llama-b11192-bin-ubuntu-x64.tar.gz`) is Linux x86_64 only. On any other platform, build llama.cpp **b11192** yourself and set `BANKML_LLAMA_SERVER` to its `llama-server` |
| Rust | **1.99.0**, pinned in `rust-toolchain.toml` (channel `1.99.0`, component `clippy`), and `rust-version = "1.99"` in both crates. The C API (`capi/`) defines a C-variadic function (`bankml_log`), and Rust stabilized those in 1.99. With rustup installed, the first `cargo` run in the checkout fetches 1.99.0. The pin applies only inside the checkout, not to the rest of the machine |
| crates | none. Both packages in the workspace (`bankml` and `bankml-capi`) build offline, with no download |
| tools the installer checks | `git`, `curl`, `tar`, `ss` (iproute2), and `sha256sum` (or `shasum`) |
| Python | 3.10 or newer, for the Savante UI and the model importer |
| Gradio | `gradio>=3.37,<4`, plus `numpy`. The UI is written for Gradio 3, and its layout and scripts work around Gradio 3. Gradio 4 is not supported |

### CPU features: chosen at run time

The release build has no `target-cpu` flag. Each kernel checks the CPU when it runs, so one binary works on any
x86_64 machine:

| feature | what uses it | without it |
|---|---|---|
| **AVX2 + FMA + F16C**, all three (`q1_0::has_avx2`) | the fast 1-bit, ternary and F16 kernels, attention and prefill | the portable kernels: the same bits, slower (the AVX2 1-bit path is 15.2× the portable C port, [PERFORMANCE.md](PERFORMANCE.md)) |
| **SHA-NI + SSE4.1 + SSSE3** | SHA-256 of the model file at every `serve` start, model load and switch (5.5× the portable rounds; 2.9 s for the 1.16 GB 8B model) | the portable SHA-256, same digest; also chosen when `BANKML_NO_SHANI` is set |

`./install.sh check` reports AVX2. A machine without it is still correct, only slower.

### Memory and disk, per model

The models bankML's own forward pass plays (`serve --native`, `bankml generate`, the C API):

| model | file | weights (mapped) | notes |
|---|---|---|---|
| Bonsai-8B, 1-bit | `Bonsai-8B-Q1_0.gguf` | 1.16 GB | the default carrier |
| Ternary-Bonsai-8B | `Ternary-Bonsai-8B-Q2_0_g64.gguf` | 2.31 GB | better answers; llama.cpp runs it about 5× slower on a CPU, bankML's kernel does not |
| Bonsai-1.7B, 1-bit | `Bonsai-1.7B-Q1_0.gguf` | 0.25 GB | the oracles' test model |
| SmolLM2-135M-Instruct | `SmolLM2-135M-Instruct-F16.gguf` | 0.27 GB | converted, Llama architecture |
| mindx-gen39 | `mindx-gen39-F16.gguf` | 0.27 GB | mindX's generation 39, converted |

Other catalogue models (Q8_0, Q4_K_M) are served only through llama-server, behind `bankml serve` without `--native`.

What else takes memory:
- **The KV cache** is f16 and grows with the context. Qwen3-8B uses 147,456 bytes per token
  (`sAGI/models.py`), so a 2048-token context holds about 0.30 GB and 4096 tokens about 0.6 GB. The other models
  need less; the guard reports each one's `kv_f16_bytes_per_token`. In the native engine,
  `BANKML_CACHE_TYPE=q8_0` (0.3.9, in progress) keeps it in 53 % of those bytes, as llama.cpp's q8_0 cache does (§6).
- **llama-server** adds about 250 MB of compute buffers and runtime beside the weights and KV (`OVERHEAD` in
  `sAGI/models.py`, an order-of-magnitude measurement).
- **The whole stack** (8B 1-bit model, engine, UI) runs in about 2 GB of free RAM. Plan for 3 GB of free disk.

The importer refuses an import when either limit is crossed, and gives the numbers:
- **Disk:** the file's size plus a 1.5 GB margin must be free.
- **Memory:** weights larger than 70 % of the machine's RAM are refused, because they would page to swap.

### Optional components

| component | needed for | detail |
|---|---|---|
| a Vulkan GPU | the GPU worker (1-bit matrices only) | the loader `libvulkan.so.1` is opened at run time, so no link-time dependency. Integrated and discrete cards with a compute queue are used; software renderers (Mesa llvmpipe) are refused. A card is used only after it passes the on-card oracle (`bankml gpu --verify`) |
| `ffmpeg` | Savante's neural voice | Piper's output is pitched and equalised with ffmpeg; without it the eSpeak stand-in is used |
| Ollama with `bge-m3` | embeddings for the ragebar and PostgreSQL vectors | optional; without it, search is BM25 alone ([embedding.md](embedding.md)) |
| PostgreSQL with pgvector, `psql` | Agents → PostgreSQL | [usage.md §8d](usage.md#8d-postgresql-publish-an-agent-load-one-back) |
| Foundry's `anvil` and the iNFT4 build | the local iNFT devnet | [usage.md §8e](usage.md#8e-inft-mint-an-agent-load-one-from-a-token) |
| a C compiler | the C API's examples and its printf oracle | [CAPI.md](CAPI.md) |

### Python: the venv rule

The `python` step first tries the interpreter it was given (`BANKML_PYTHON`, else the one a previous run recorded,
else `python3`). If that interpreter imports both `gradio` and `numpy`, it is used as it is. Otherwise the step
creates `~/.local/share/bankml/venv` with `--system-site-packages` and installs `gradio>=3.37,<4` and `numpy` into it.
The system Python is never modified. The chosen interpreter is recorded as `INSTALL_PYTHON` in `install.env`, and
later runs use it. If `python3 -m venv` fails (on Debian and Ubuntu, `python3-venv` is missing), install that package,
or point `BANKML_PYTHON` at an interpreter that already has Gradio 3 and numpy.

---

## 2. install.sh in full

```
./install.sh [step …] [--skip-tests] [--no-start] [--view] [--voice] [-h|--help]
```

Nothing in it needs sudo. Unless steps are named, it runs `check build engine python canon model`, then `voice` if
`--voice` was given, then `start` unless `--no-start` was given. It stops at the first step that fails, and says why.

### The steps

| step | what it does | what it writes |
|---|---|---|
| `check` | Rust ≥ 1.99 (`cargo --version`), `python3` ≥ 3.10, `git`, `curl`, `tar`, `ss`, `sha256sum`/`shasum`; warns if the machine is not Linux/x86_64 and if AVX2 is missing; reports RAM (total, available) and free disk. Any missing item stops the run | nothing |
| `build` | `cargo build --release`, then `cargo test --release --quiet` (unit and end-to-end, offline) unless `--skip-tests`; prints `bankml version` | `target/release/bankml` |
| `engine` | uses an existing llama-server if one is set or found (see below); otherwise downloads llama.cpp b11192 (17 MB) and refuses it unless its sha256 is `34cf6fa5…881ec7`. A tarball that is already there and matches is reused; one that does not match is deleted and downloaded again | `~/.local/share/bankml/llama-b11192-bin-ubuntu-x64.tar.gz`, `~/.local/share/bankml/llama-b11192/`, `INSTALL_LLAMA_SERVER` in `install.env` |
| `python` | the venv rule above | `~/.local/share/bankml/venv/` if needed, `INSTALL_PYTHON` in `install.env` |
| `canon` | clones `github.com/cryptoAGI/savante` to the canon path if it is absent; an existing canon is left untouched (it is read-only) | the canon checkout |
| `model` | needs `build` and `engine` done. Runs `sAGI/models.py first-run`: imports Bonsai-8B if absent (checked against the publisher's sha256, guarded, pinned), then starts `bankml serve` on it with llama-server behind it. Then checks `http://127.0.0.1:18093/bankml` | `.models/Bonsai-8B-Q1_0.gguf`, its `FORK.json` in the forks directory, `logs/first-run.json`, `savante/carrier.log` |
| `voice` | downloads Piper 2023.11.14-2 and the `en_GB-cori-high` voice (`.onnx`, `.onnx.json`, `MODEL_CARD`); each file is skipped if already there | `$BANKML_PIPER` (default `~/.local/share/bankml/piper/`) |
| `start` | starts interact mode (`sAGI/savante.py --mode interact --port 7873`), and with `--view` also view mode (`sAGI/view.py --host 0.0.0.0 --port 7874`), each detached (`setsid nohup`), waiting up to 60 s for its port. A port that already listens is left as it is | `logs/interact.log`, `logs/view.log` |
| `stop` | for ports 7874, 7873, 18093 and 18092, finds the process listening there (by `ss`, never by matching command lines) and sends it SIGTERM | nothing |
| `status` | which of the four ports listen; `GET /bankml` (verified, model, sha256 prefix); `bankml version`; the engine and Python in use | `~/.local/share/bankml/.status.json` |
| `power` (0.3.7, opt-in, **sudo**) | lets a `rapl` group your user joins read the CPU package energy counter, so bankML can report watts and joules per token; explains the side channel (PLATYPUS, CVE-2020-8694) and asks first (`BANKML_POWER_YES=1` to agree non-interactively); `./install.sh power --remove` restores root-only | `/etc/udev/rules.d/60-bankml-rapl.rules` |

`start` does not start `bankml serve`; `model` does. Interact mode also starts the first-run carrier in the
background when `bankml serve` does not answer, unless `BANKML_FIRST_RUN=0`. So after a reboot `./install.sh start`
alone brings the whole stack back. Use `./install.sh model start` to wait for the carrier in the foreground.

### The options

| option | effect |
|---|---|
| `--skip-tests` | `build` skips `cargo test --release` |
| `--no-start` | the default run ends after `model` (or `voice`): everything installed, Savante not started. `bankml serve` is still started by `model` |
| `--view` | `start` also starts view mode on `0.0.0.0:7874`. It works with named steps too: `./install.sh start --view` |
| `--voice` | the default run includes `voice`. With named steps, name it instead: `./install.sh voice` |
| `-h`, `--help` | prints the header of `install.sh` and exits |

Any other argument stops the installer with exit code 2.

### What it reads

| variable | default | effect |
|---|---|---|
| `BANKML_DIR` | `~/bankml` | only on the `curl … \| bash` route: where bankml is cloned (or the existing clone used) before `install.sh` re-runs from there |
| `BANKML_DATA` | `~/.local/share/bankml` | the installer's data directory: `install.env`, `logs/`, the engine, the venv, Piper |
| `BANKML_LLAMA_SERVER` | — | the llama-server to use; skips the download |
| `INSTALL_LLAMA_SERVER` | from `install.env` | what a previous `engine` step chose; used when `BANKML_LLAMA_SERVER` is unset |
| `BANKML_PYTHON` | — | the Python for the UI and importer |
| `INSTALL_PYTHON` | from `install.env` | what a previous `python` step chose; used when `BANKML_PYTHON` is unset |
| `SAVANTE_CANON` | `~/cryptoAGI/savante`; `~/savante` if that is the only one present | the canon to clone into or use |
| `BANKML_PIPER` | `$BANKML_DATA/piper` | where `voice` installs Piper |

How the engine is found: `BANKML_LLAMA_SERVER`, else `INSTALL_LLAMA_SERVER`, else the first executable of
`$BANKML_DATA/llama-b11192/llama-server` and `~/sAGI/bonsai/llama-b11192/llama-server`; if none, `engine` downloads
it. The Python is `BANKML_PYTHON`, else `INSTALL_PYTHON`, else `python3`. (`check` tests the version of `python3` on
the `PATH`, whatever the UI will use.)

`LLAMA_TAG` (`b11192`) and `LLAMA_TGZ` (`llama-b11192-bin-ubuntu-x64.tar.gz`) are constants in the script, together
with the archive's sha256. They are not settings: bankML is verified against that one release.

The installer passes its choices on. The importer and both UIs run with `BANKML_LLAMA_SERVER` (the chosen engine)
and `BANKML_BIN` (`target/release/bankml`); the UIs also get `SAVANTE_CANON`. When you start `sAGI/savante.py` by
hand, set the same variables yourself (§11).

### What it leaves on disk

```
~/.local/share/bankml/              ($BANKML_DATA)
├── install.env                      INSTALL_LLAMA_SERVER=…, INSTALL_PYTHON=… (shell-quoted; sourced by later runs)
├── logs/                            first-run.json, interact.log, view.log
├── llama-b11192-bin-ubuntu-x64.tar.gz, llama-b11192/   the engine
├── venv/                            only if your Python lacked Gradio or numpy
├── piper/                           only after `voice`
├── .status.json                     the last `status` answer
├── forks/                           FORK.json pins ($BANKML_FORKS)
├── savante/                         the UI's state ($BANKML_UI_STATE): savante.history, savante.memory,
│                                    Savante.prompt, savante.aivatar, carrier.log, resources.json, slots/
└── agents/                          custom agents, one git repository each ($BANKML_AGENTS)
<checkout>/target/release/bankml     the binary
<checkout>/.models/                  imported models ($BANKML_MODELS)
~/cryptoAGI/savante/                 the canon ($SAVANTE_CANON), read-only
```

`forks/`, `savante/` and `agents/` are written by the importer and the UI, not by the installer. They sit under
`~/.local/share/bankml/` by their own defaults, so setting `BANKML_DATA` does not move them; set `BANKML_FORKS`,
`BANKML_UI_STATE` and `BANKML_AGENTS` as well.

### Re-running, and single steps

Every step checks before it acts, so a second run is quick and changes nothing that is already right. The tarball is
re-verified, not re-downloaded; the model is imported only if absent; a running UI is not restarted. Run any steps
alone, in any order: `./install.sh build`, `./install.sh engine python`, `./install.sh model start --view`.
To rebuild after a `git pull`, run `./install.sh build`, then `./install.sh stop` and `./install.sh model start`.

### The piped route

```sh
curl -fsSL https://raw.githubusercontent.com/cryptoAGI/bankml/main/install.sh | bash
curl -fsSL https://raw.githubusercontent.com/cryptoAGI/bankml/main/install.sh | BANKML_DIR=/opt/src/bankml bash
```

Outside a checkout, the script clones `github.com/cryptoAGI/bankml` into `$BANKML_DIR` (default `~/bankml`), or uses
the clone already there, and runs that copy's `install.sh` with the same arguments. Pass arguments through bash:
`curl … | bash -s -- --no-start`.

---

## 3. Building by hand

```sh
cargo build --release                       # target/release/bankml (the root package is the default member)
cargo test --release                        # unit tests and testing/cli.rs, offline
cargo build --release -p bankml-capi        # target/release/libbankml.so and libbankml.a
cargo test --release -p bankml-capi         # the C API's unit tests
cargo clippy --release --workspace --all-targets -- -D warnings
```

- **The workspace.** It has two members. The root package `bankml` builds the binary and the library
  (`bankML/bankml.rs`). `capi/` (`bankml-capi`) builds the C library `libbankml`, as `cdylib` and `staticlib`, over
  the root crate by path. A plain `cargo build` builds only the root package. Headers: `capi/include/bankml.h`.
  Linking, ownership and threading rules are in [CAPI.md](CAPI.md).
- **The release profile** uses `lto = true`, `codegen-units = 1` and `panic = "abort"`, for one small binary. Do not
  add `-C target-cpu=native`: the kernels already choose AVX2 at run time (§1), and a binary built for one CPU may
  not start on another.
- **`--locked`** is what the release gate uses (`cargo build --release --locked`). With no external crates,
  `Cargo.lock` lists only the two workspace packages.
- **Installing the binary elsewhere.** The binary is self-contained. Copy `target/release/bankml` where you need it
  (§8 keeps the previous version beside it for rollback). The UI and importer find it through `BANKML_BIN`.

---

## 4. Models: where they live, how they are pinned

### Directories

| what | default | set with |
|---|---|---|
| model files (`*.gguf`) | `<checkout>/.models` | `BANKML_MODELS` (`BANKML_REPO` moves the checkout the default is relative to) |
| pins (`*.FORK.json`) | `~/.local/share/bankml/forks` | `BANKML_FORKS` |
| derived models (`NAME.MODEL.json`) | the forks directory | `bankml create --registry DIR` |

A pin is `<file>.FORK.json`: a JSON object with provenance (`source_repo`, `source_url`, `source_revision`,
`licence_tag`, `imported_at_utc`, `pinned_from`) and `files`: `[{"path": "<file>.gguf", "bytes": …, "sha256": "…"}]`.
`bankml verify`, `bankml serve` and `bankml_open` pass a file only if its sha256 equals the `files` entry for its
name, and only after the guard says `play`.

### The importer: `sAGI/models.py`

```sh
python3 sAGI/models.py list                  # every GGUF in .models: pinned or UNPINNED, size, source, licence; the carrier
python3 sAGI/models.py catalog               # the curated catalogue, and whether each entry fits this machine
python3 sAGI/models.py search QUERY          # search ollama.com
python3 sAGI/models.py import ID             # a catalogue id
python3 sAGI/models.py import https://huggingface.co/OWNER/REPO[/blob|resolve/REV/FILE]
python3 sAGI/models.py import ollama:NAME[:TAG]
python3 sAGI/models.py use FILE              # make FILE the carrier (verified switch, rollback on failure)
python3 sAGI/models.py first-run             # what the installer's `model` step runs
```

With no subcommand, `list` runs. The Models tab in Savante does the same things.

**The rules an import must pass:**
- **Licence.** The licence must be OSI-approved or a public-domain dedication. The accepted SPDX tags are apache-2.0,
  mit, bsd-2-clause, bsd-3-clause, bsd, isc, mpl-2.0, gpl-2.0, gpl-3.0, lgpl-2.1, lgpl-3.0, agpl-3.0, unlicense,
  cc0-1.0, zlib, artistic-2.0, epl-2.0, ecl-2.0, upl-1.0 and 0bsd. Anything else is refused outright, which excludes
  Gemma and Llama.
- **A published sha256.** The download is hashed as it streams and kept only if it equals the sha256 the publisher
  lists: a catalogue entry is re-checked against its repository at the pinned revision; a Hugging Face file against
  the repository's LFS sha256 at the resolved revision; an Ollama model against its registry layer digest. With no
  published sha256 there is nothing to pin to, and the import is refused.
- **The guard.** `bankml guard --json` must say `play`. A refused file is deleted.
- **The fit.** The disk and memory limits of §1.

Then the importer writes the `FORK.json`.

**The sources:**
- **Catalogue ids:** `bonsai-8b` (the default), `ternary-bonsai-8b`, `bonsai-4b`, `bonsai-1.7b`, `qwen3-0.6b`,
  `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, `smollm2-1.7b`, `smollm3-3b`, `granite-3.3-2b`, `qwen2.5-coder-1.5b`,
  `qwen2.5-coder-7b` and `qwen3.8-27b` (16.5 GB, for a machine with 24 GB or more). `catalog` prints them with sizes
  and whether each fits.
- **Hugging Face:** a repository URL with several GGUF files asks you to pick one.
- **Ollama:** a model the local Ollama already holds is adopted with no download. The importer links to the blob in
  `BANKML_OLLAMA_MODELS` (default `/usr/share/ollama/.ollama/models`) after hashing it. Otherwise the model comes
  from `registry.ollama.ai`.

The only network calls are GETs to huggingface.co, ollama.com and registry.ollama.ai. Nothing is uploaded.

**Pinning a file that is already here.** `use FILE` pins an unpinned catalogue file first: it hashes the file and
compares the hash with the repository's published one. The two converted models, SmolLM2-135M-Instruct-F16 and
mindx-gen39-F16, are pinned against their recorded conversion. `pin_converted` hashes the GGUF and re-reads the
source safetensors' LFS sha256 at the recorded revision. There is no subcommand for this; call it as
[usage.md §5](usage.md#5-models) shows:

```sh
python3 -c "import sys; sys.path.insert(0, 'sAGI'); import models; print(models.adopt('SmolLM2-135M-Instruct-F16.gguf'))"
```

**Switching the carrier (`use`).** `use` refuses:
- while an answer is being written;
- a file that is not in `.models`;
- weights over 70 % of RAM;
- an embedding architecture (bert, nomic-bert, jina-bert-v2, xlm-roberta, modern-bert, neo-bert, t5encoder).

Otherwise it stops the carrier and starts `bankml serve` on the new file. The switch succeeds only when
`/bankml` answers verified with that file's sha256. If the new model fails, the previous one is restarted.

### The registry, and derived models

`bankml serve --native --registry [DIR]` serves every GGUF pinned in `DIR` by name: the file's stem in lower case,
with `:latest`, the file name, and (when unique) the name without its weight-type suffix accepted as aliases. A
pinned file is looked for in two places: beside the startup model, and in `DIR`. A pin whose file is missing is
listed in `/api/tags` with the reason. So a registry can be a directory of `FORK.json` files with the GGUFs symlinked
in (§8). Derived models (`bankml create`, `/api/create`) are layers over a pinned base, written as
`DIR/NAME.MODEL.json`, with no weights copied. Their use is described in
[usage.md §6a](usage.md#6a-ollamas-api) and [OLLAMA.md](OLLAMA.md).

---

## 5. Running bankml serve, and every other command

### Two engines, one gate

```
bankml serve FILE --fork FORK.json [--upstream HOST:PORT | --spawn LLAMA_SERVER] [--listen HOST:PORT] [--threads N] [--ctx N] [--spec-ngram] [--slot-dir DIR] [--engine mainline|prism]
bankml serve FILE --fork FORK.json --native [--listen HOST:PORT] [--upstream HOST:PORT] [--ctx N] [--registry [DIR]] [--keep-alive DUR] [--slot-dir DIR] [--engine mainline|prism]
```

The two modes start the same way. `serve` checks the guard and the sha256 pin, refuses if the file changed while it
was being hashed, then either:
- **llama.cpp mode** (the default) puts llama-server b11192 behind the gate, and checks through `/props` that
  llama-server serves the verified file;
- **native mode** (`--native`) answers from bankML's own forward pass, token-identical to llama-server on its
  oracle. It speaks the OpenAI API and Ollama's API, and it also answers llama-server's endpoints on the engine
  address, so Savante reaches it unchanged.

**`serve` binds `--listen` only after verification.** In llama.cpp mode it also waits until the upstream is healthy.
A client should wait for `GET /bankml` to answer, not for the port to open. `install.sh`, `sAGI/models.py` and the
oracles all do this.

### Every serve flag

| flag | default | mode | effect |
|---|---|---|---|
| `FILE` | required | both | the GGUF to verify and serve |
| `--fork FORK.json` | required | both | the pin. A missing flag prints the usage and exits 1; an unreadable file exits 1 |
| `--engine mainline\|prism` | `mainline` | both | which llama.cpp the guard judges for. `mainline` refuses PrismML-fork-only types (PQ2_0, PTQ1_0); `prism` accepts them |
| `--listen HOST:PORT` | `127.0.0.1:18093` | both | the gateway's address |
| `--upstream HOST:PORT` | `127.0.0.1:18092` | both | llama.cpp mode: the llama-server to check, or where `--spawn` starts one. Native mode: a **second listening address** that answers llama-server's endpoints (`/health`, `/props`, `/tokenize`, `/apply-template`, `/v1/chat/completions`). Native mode cannot start while something else holds this port |
| `--spawn LLAMA_SERVER` | — | llama.cpp | `serve` launches this llama-server on the verified path and on `--upstream`, waits up to 900 s for `/health`, and stops it when `serve` stops (on SIGTERM, SIGINT, SIGHUP). Refused if something already answers on `--upstream`. Without `--spawn`, an existing upstream must be healthy within 10 s |
| `--threads N` | `3` | llama.cpp | passed as `-t N` to the spawned llama-server. Native mode ignores it; use `BANKML_THREADS` (§7) |
| `--ctx N` | `4096` | both | the context: `-c N` for the spawned llama-server, `n_ctx` for the native engine. A prompt of `N` tokens or more is refused (0.3.8: with llama-server's `exceed_context_size_error` 400 body on `/v1`); an Ollama `num_ctx` above it is refused |
| `--spec-ngram` | off | llama.cpp | `--spec-type ngram-simple` in the spawned engine: n-gram speculative decoding, exact at temperature 0, opt-in because it measured within noise |
| `--slot-dir DIR` | — | both | llama.cpp mode: `--slot-save-path DIR` for the spawned engine. Native mode (0.3.8): `POST /slots/0?action=save\|restore\|erase` with `{"filename"}`, as llama-server's slot API; the directory is created at start. Either way, a restart restores the system prompt instead of computing it again. Without it, the slot actions answer 501 |
| `--native` | off | — | native mode |
| `--registry [DIR]` | off; bare: `$BANKML_FORKS`, else `~/.local/share/bankml/forks` | native | serve every model pinned in `DIR` by name, one resident at a time, each verified again when it loads; answer `/api/create`, `/api/copy`, `/api/delete` for derived models. A following argument that starts with `--` is not taken as `DIR` |
| `--keep-alive DUR` | `5m` | native | how long an `/api/*` request that names no `keep_alive` keeps the model resident: `"5m"`, `"1h30m"`, seconds, `0` (unload after the answer), negative (for good). OpenAI and llama-server endpoints keep the model resident |

A spawned llama-server is always started as:

```
llama-server -m <verified path> --host <upstream host> --port <upstream port> -t <threads> -c <ctx> -np 1 --jinja --reasoning off --no-webui [--spec-type ngram-simple] [--slot-save-path DIR]
```

When verification or startup fails, `serve` prints `bankml serve: refuse: <reason>` and exits 2.

### The gateway's fixed limits

These are constants in `serve.rs`, not settings:

| limit | value | over it |
|---|---|---|
| connections at once | 32 | `503 bankml serve: too many connections` |
| request line or header line | 16 KiB | 400 |
| header lines | 100 | 400 |
| request body | 16 MiB | `413 request too large` |
| chunked request bodies | not accepted | `400`: send `Content-Length` |
| read timeout per connection | 30 s | connection closed |
| write timeout per connection | 120 s | connection closed |
| upstream response body or chunk | 64 MiB | the request fails |
| `Host` | `127.0.0.1`, `localhost` or `[::1]`, any port | `403` |
| POST `Content-Type` | must begin `application/json` | `415` |

### Endpoints

| mode | endpoints |
|---|---|
| both | `GET /bankml` (what was verified, when, and what is upstream; in native mode also the resident model and every name), `GET /bankml/usage` (memory and CPU of `serve` and its engine, sampled at most once a second; 0.3.7 adds package watts, each GPU's busy %, VRAM and GTT, and the GPU limiter's state), `GET /bankml/metrics` (0.3.7: the native engine's own measurements of its last 256 answers) |
| llama.cpp | `GET /health`, `/v1/models`, `/props` (proxied); `POST /v1/chat/completions` (receipted) |
| native | `GET /health`, `/props`, `/v1/models`; `POST /tokenize`, `/apply-template`, `/v1/chat/completions`, `/slots/0?action=…` (0.3.8); `GET\|HEAD /`; Ollama's `/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/create`, `/api/copy`, `DELETE /api/delete`. `/api/embed`, `/api/pull` and `/api/push` answer 400 with the reason |

What each request may contain, and what is refused, is in [usage.md §6](usage.md#6-by-hand-build-verify-serve),
[§6a](usage.md#6a-ollamas-api) and [OLLAMA.md](OLLAMA.md).

### Ports

| port | service | bound to |
|---|---|---|
| 18092 | llama-server b11192, or native serve's engine address | loopback |
| 18093 | `bankml serve` | loopback |
| 7873 | Savante, interact mode (Gradio) | loopback only; anything else is refused |
| 7874 | view mode (standard library HTTP) | `0.0.0.0` by default |
| 7875 | the bankML console, `sAGI/console.py` (0.3.7) | loopback only; anything else is refused. `./install.sh stop` does not stop it |
| 11434 | Ollama, for embeddings (`BANKML_OLLAMA`) | loopback |
| 8545 | anvil, the local iNFT devnet, when started from the Agents tab | loopback |

The importer moves the carrier's two ports with `BANKML_SERVE_LISTEN` and `BANKML_UPSTREAM`, and the UI's view of it
with `BANKML_SERVE`. Keep the three consistent.

### Choosing the engine for the carrier

The carrier is what `sAGI/models.py` starts. On Savante's Admin tab the **engine** setting (under *advanced*, saved
in `savante/resources.json`) chooses how:
- `auto` (the default): native for the ternary (`Q2_0`) files, where bankML is about 8× llama-server with the same
  tokens; llama-server for everything else, since llama-server is still faster on the 1-bit files.
- `native`: always `bankml serve --native`.
- `llama.cpp`: always llama-server behind the gate.

The two carriers are started this way:
- **Native carrier:** `bankml serve FILE --fork F --native --upstream $BANKML_UPSTREAM --listen $BANKML_SERVE_LISTEN
  --ctx N --slot-dir ~/.local/share/bankml/savante/slots`, with `BANKML_THREADS` set to the chosen thread count and
  `BANKML_GPU_LIMIT` to the saved GPU limit (`BANKML_GPU=off` when it is 0).
- **llama.cpp carrier:** `bankml serve FILE --fork F --spawn $BANKML_LLAMA_SERVER --upstream … --listen … --threads N
  --ctx N [--spec-ngram] --slot-dir ~/.local/share/bankml/savante/slots`.

Either is given 1800 s to answer verified.

The same tab sets threads (1 to the number of cores) and a RAM budget. The budget becomes a context: whatever
remains after the weights and the 250 MB overhead, divided by the KV bytes per token, rounded down to 256, between
512 and 32768. **Apply** restarts the carrier with the new settings and restores the previous ones if it will not
start. Without a saved budget, the context is `BANKML_CTX` (2048) and the threads `BANKML_THREADS_SERVE` (3).

### Every other command

| command | options and defaults | exit codes |
|---|---|---|
| `bankml guard FILE` | `--engine mainline\|prism` (default `mainline`), `--json` | 0 play · 2 refuse · 3 need_more (truncated) · 1 cannot read |
| `bankml sha256 FILE` | — | 0 · 1 |
| `bankml pin FILE --fork F` | — | 0 pinned · 2 refuse · 1 cannot read the fork |
| `bankml verify FILE --fork F` | `--engine`, `--json` | 0 play · 2 refuse · 3 truncated · 1 I/O |
| `bankml tokenize MODEL.gguf < text` | `--no-special` (do not parse special tokens) | 0 · 1 |
| `bankml chat-template MODEL.gguf < messages.json` | — | 0 · 2 |
| `bankml generate MODEL.gguf < messages.json\|text` | `--max N` (256); `--json` (JSON mode); `--sample` with `--temp`, `--top-k`, `--top-p`, `--min-p`, `--seed` (each defaults to the model's GGUF value). Without `--sample`, greedy | 0 · 2 · 1 |
| `bankml create NAME -f Modelfile` | `-f`/`--file` required; `--registry DIR` (default `$BANKML_FORKS`, else `~/.local/share/bankml/forks`); `--models DIR` (where `FROM` a name also looks) | 0 · 2 refuse · 1 usage |
| `bankml convert DIR -o OUT.gguf` | `-o`/`--outfile` required; `--outtype f16` (the only one accepted); `--model-name NAME`; `--fork F` writes the pin, `--source SRC` names the source in it (default the directory's canonical path); `--ignore-model-card` | 0 · 2 refuse · 1 usage or write error |
| `bankml gpu` | `--remote` (also list the GPUs Hugging Face rents; listed, never started); `--verify` (run the on-card oracle on every selected card) | 0 · 2 a card failed `--verify` |
| `bankml usage [PID …]` | memory, cores, and each process's RSS and CPU % over 0.5 s (default: bankml itself) | 0 · 1 |
| `bankml version` | also `--version`, `-V` | 0 |

### The Python entry points

| command | options |
|---|---|
| `python3 sAGI/savante.py` | `--mode interact\|view` (default `interact`); `--host` (default `127.0.0.1`; interact accepts only `127.0.0.1` or `localhost`); `--port` (default 7873). `--mode view` hands over to `view.py`, on `0.0.0.0:7874` unless a host and port are given |
| `python3 sAGI/view.py` | `--host` (default `0.0.0.0`), `--port` (default 7874) |
| `python3 sAGI/console.py` | (0.3.7) `--host` (default `127.0.0.1`; loopback only), `--port` (default 7875); reaches `bankml serve` at `BANKML_SERVE_LISTEN` |
| `python3 sAGI/models.py` | `list`, `catalog`, `search Q`, `import ID\|URL\|ollama:NAME[:TAG]`, `use FILE`, `first-run` |
| `python3 sAGI/agents.py` | `adopt PERSONA [--slug SLUG]` (install an existing `.persona` as its own agent), `list`, `verify SLUG` |
| `python3 sAGI/speak.py` | renders every voice clip, then writes both exports; `--shard K/N` renders every N-th statement starting at K, for parallel renders, and skips the exports; `--prune` drops clips no current text uses |

---

## 6. Environment variables: the complete list

Every variable bankML or its tools read, where it is read, its default, and what it does. Where the installer
sets a variable for the processes it starts, that is noted.

### Engine and kernels (Rust: `bankml`, `libbankml`)

| variable | read by | default | effect |
|---|---|---|---|
| `BANKML_THREADS` | `bankML/par.rs` (`Pool::from_env`) | all available cores | the thread pool of the native engine: `serve --native`, `generate`, the C API. Every thread count gives the same bits. `sAGI/models.py` sets it for a native carrier. In the decode-budget benchmarks (`q1_0.rs`, `q2_0.rs`) it is instead a comma list of counts to measure, default `1,2,3,4` (Q1_0) and `1,3` (Q2_0) |
| `BANKML_LLAMA_THREADS` | `bankML/forward.rs` | `3` | the `-t` of the llama.cpp run being reproduced. In decode at 512 or more KV cells, llama.cpp's split attention kernel chunks by its thread count, so the bits depend on it. Keep it equal to the reference's `-t` for token identity. It does not change bankML's own thread count |
| `BANKML_NO_SHANI` | `bankML/sha256.rs` | unset | set to any value to hash with the portable SHA-256 instead of the CPU's SHA extensions |
| `BANKML_FORKS` | `bankML/main.rs` | `~/.local/share/bankml/forks` | the registry directory for `serve --registry` without a `DIR`, and for `bankml create` without `--registry` |
| `HOME` | `bankML/main.rs` | — | the base of the default forks directory |
| `BANKML_CACHE_TYPE` | `bankML/native.rs` (0.3.9) | `f16` | `q8_0` keeps the native engine's KV cache as llama.cpp's `--cache-type-k q8_0 --cache-type-v q8_0` does, Hadamard rotation included: 53 % of the f16 cache's memory, the same tokens as llama-server so configured |
| `BANKML_CACHE_RAM` | `bankML/native.rs` (0.3.8) | 8192 MiB, at most a quarter of the memory available at load | the host prompt cache's limit in MiB, as llama-server's `--cache-ram` (0 off, -1 no limit) |

### GPU (Rust)

| variable | read by | default | effect |
|---|---|---|---|
| `BANKML_GPU` | `bankML/gpu/mod.rs` | unset: every usable card, discrete first, largest memory first | `off`, `none` or `cpu` turn the GPU component off. A comma list (`0,2`) picks cards by backend index (as `bankml gpu` lists them). The worker uses the first selected card |
| `BANKML_GPU_SHARE` | `bankML/gpu/worker.rs` | calibrated at open: card rate ÷ combined rate (26–35 % on a Vega 3) | the fraction of each 1-bit matrix's rows the card computes, clamped to 0–0.95. A share under 0.02, set or calibrated, leaves the card unused |
| `BANKML_GPU_LIMIT` | `bankML/gpu/worker.rs` (0.3.7) | `0.8` | the share of the card's memory (its heap; on an integrated card, the RAM it could use) and of its time bankML may take; clamped 0.05–1. The console's GPU slider and `models.resources()["gpu_limit"]` set it (0 there means `BANKML_GPU=off`) |

### Installer (`install.sh`)

| variable | default | effect |
|---|---|---|
| `BANKML_DIR` | `~/bankml` | piped route only: where to clone and run |
| `BANKML_DATA` | `~/.local/share/bankml` | the installer's data directory |
| `BANKML_LLAMA_SERVER` | found or downloaded (§2) | the engine; passed on to the importer and UIs |
| `INSTALL_LLAMA_SERVER` | recorded in `install.env` | the previous choice of engine |
| `BANKML_PYTHON` | `INSTALL_PYTHON`, else `python3` | the UI's Python |
| `INSTALL_PYTHON` | recorded in `install.env` | the previous choice of Python |
| `SAVANTE_CANON` | `~/cryptoAGI/savante` (or `~/savante`) | the canon; passed on to the UIs |
| `BANKML_PIPER` | `$BANKML_DATA/piper` | where `voice` installs Piper (also read by `speak.py`) |
| `BANKML_POWER_YES` | unset | (0.3.7) `1` answers yes to the `power` step's question, for a non-interactive run |

### Model importer and carrier (`sAGI/models.py`)

| variable | default | effect |
|---|---|---|
| `BANKML_REPO` | the checkout (`sAGI/..`) | the base of `.models` and `target/release/bankml`, and the carrier's working directory |
| `BANKML_MODELS` | `$BANKML_REPO/.models` | where models are imported and looked for |
| `BANKML_FORKS` | `~/.local/share/bankml/forks` | where pins are written and read |
| `BANKML_LLAMA_SERVER` | `~/sAGI/bonsai/llama-b11192/llama-server` | the llama-server a llama.cpp carrier spawns. The installer sets it to its own choice for the processes it starts |
| `BANKML_BIN` | `$BANKML_REPO/target/release/bankml` | the bankml binary the importer, guard and switch call |
| `BANKML_OLLAMA_MODELS` | `/usr/share/ollama/.ollama/models` | the local Ollama store to adopt models from |
| `BANKML_SERVE_LISTEN` | `127.0.0.1:18093` | the carrier's `--listen`, and where the importer checks `/bankml` |
| `BANKML_UPSTREAM` | `127.0.0.1:18092` | the carrier's `--upstream` |
| `BANKML_CTX` | `2048` | the carrier's context when no RAM budget is saved |
| `BANKML_THREADS_SERVE` | `3` | the carrier's threads when none are saved (`--threads`, or `BANKML_THREADS` for a native carrier) |
| `BANKML_UI_STATE` | `~/.local/share/bankml/savante` | where `carrier.log`, `resources.json` and `slots/` are kept |

### Savante UI and its state (`sAGI/savante.py`, `view.py`, `agents.py`, `connectors.py`)

| variable | read by | default | effect |
|---|---|---|---|
| `SAVANTE_CANON` | `savante.py`, `speak.py` | `~/cryptoAGI/savante`; `~/savante` if only that exists | Savante's canon, read-only |
| `BANKML_SERVE` | `savante.py` | `http://127.0.0.1:18093` | where the UI reaches `bankml serve` |
| `BANKML_UI_STATE` | `savante.py`, `models.py` | `~/.local/share/bankml/savante` | `savante.history`, `savante.memory`, `Savante.prompt`, `savante.aivatar`, the carrier's log, resources and slots |
| `BANKML_AGENTS` | `agents.py` | `~/.local/share/bankml/agents` | custom agents, one directory and git repository each |
| `BANKML_AGENT` | `savante.py` | `mindx` | the agent the UI starts with; Savante when that agent is not installed. Empty: Savante |
| `BANKML_FIRST_RUN` | `savante.py` | `1` | `0` stops interact mode from starting the Bonsai-8B carrier by itself when `bankml serve` does not answer |
| `BANKML_REPO` | `savante.py`, `view.py` | the checkout | where the UI and view mode read `testing/` (the live log, the release records) |
| `BANKML_PG_DSN` | `connectors.py` | `dbname=bankml` | the libpq connection string for publishing and loading agents (local socket, your role) |
| `RAGE_PATH` | `savante.py` | `~/mindX/mindx/godel/mindxtrain/hf/space_ui` | where the ragebar finds mindX's `rage.py`; built-in BM25 otherwise |

### Voice (`sAGI/speak.py`)

| variable | default | effect |
|---|---|---|
| `BANKML_PIPER` | `~/.local/share/bankml/piper` | Piper and the `en_GB-cori-high` voice. The neural voice needs `piper/piper`, the `.onnx` and `ffmpeg` |
| `BANKML_VOICE_DIR` | `sAGI/voice/cache` | rendered clips |
| `BANKML_EXPORT_DIR` | `sAGI/voice/export` | the two complete exports (`Savante.opus`, `Savante-reading.opus`) |
| `BANKML_VOICE_ASYNC` | `1` | `0` stops the UI from rendering missing clips in the background |
| `BANKML_PRONUNCIATION` | `~/mindX/data/config/pronunciation.json` | the pronunciation table; a built-in table is the fallback |
| `BANKML_ESPEAK` | `~/DeltaVerse/vendor/espeak-ng/0.3.5-en` | the eSpeak NG stand-in voice's engine |
| `BANKML_ESPEAK_VOICES` | `~/DeltaVerse/vendor/espeak-ng/mindx-voices` | its voice files |

### Embeddings (`sAGI/embed.py`)

| variable | default | effect |
|---|---|---|
| `BANKML_OLLAMA` | `http://127.0.0.1:11434` | the Ollama server. Keep it local: history text is sent to it |
| `BANKML_EMBED_MODEL` | `bge-m3` | the embedding model; must give 1,024 dimensions |
| `BANKML_EMBED_KEEP_ALIVE` | `60s` | how long Ollama keeps it loaded after a call |
| `BANKML_EMBED_NEED_GB` | `1.3` | free memory required before loading it; below that, search is BM25 alone |

### Testing and oracles

| variable | read by | default | effect |
|---|---|---|---|
| `BANKML_GGML_LIB` | `q1_0.rs`, `q2_0.rs` tests, `release_gate.sh`, the `testing/*_oracle.py` recorders | — | the llama.cpp b11192 release directory (`libggml-base.so`, `libggml-cpu-haswell.so`, `libllama*.so`). Without it, the gate skips every oracle, A/B and budget |
| `LLAMA_SRC` | `grammar_oracle.py`, `content_oracle.py`, `schema_oracle.py`, `release_gate.sh` | — | a llama.cpp source checkout at tag b11192, for the oracles that compile against its headers |
| `BANKML_LLAMA_SRC` | `convert_oracle.py`, `release_gate.sh` | `upstream/llama.cpp` | the b11192 source tree whose `convert_hf_to_gguf.py` and `gguf-py` the conversion oracles run. In the gate it also enables `oracle_name_heuristics` |
| `BANKML_CONVERT_DIR` | `convert.rs` (`oracle_convert_b11192`) | — | the safetensors directory to convert (its name matters: llama.cpp names the model from it) |
| `BANKML_CONVERT_ORACLE` | same | — | llama.cpp's GGUF of that directory, to compare byte for byte |
| `BANKML_CONVERT_OUT` | same | `$TMPDIR/bankml-convert-oracle.gguf` | where bankML's conversion is written |
| `BANKML_CONVERT_KEEP` | same | unset | set to keep that file after the test |
| `BANKML_NAMES_ORACLE` | `convert.rs` (`oracle_name_heuristics`) | — | the JSONL that `convert_oracle.py --names` writes |
| `BANKML_PENALTY_STEMS` | `native.rs` (`oracle_penalties`) | `mindx-gen39-F16,Bonsai-1.7B-Q1_0` | which models' penalty records to replay |
| `BANKML_SAMPLER_STEMS` | `native.rs` (`oracle_samplers`, 0.3.7) | `mindx-gen39-F16,Bonsai-1.7B-Q1_0` | which models' sampler records to replay |
| `BANKML_ORACLE_MODEL` | `schema_oracle.py` | the Bonsai and O4 models in `.models` | one GGUF to read the chat template from, instead of every reproduced template |
| `BANKML_FORK` | `serve_oracle.py`, `json_oracle.py`, `json_schema_oracle.py` | `$BANKML_FORKS/Bonsai-8B-Q1_0.gguf.FORK.json` | the Bonsai-8B pin the live oracles serve with |
| `BANKML_FORKS` | the live oracles (`serve_`, `json_`, `json_schema_`, `persona_`, `penalty_oracle.py`) | `~/.local/share/bankml/forks` | where they find each model's pin |
| `BANKML` | `convert_oracle.py` | `target/release/bankml` | the binary `--bankml` runs |
| `BANKML_TEST_CARRIER` | `test_models.py` | unset | `1` adds a real carrier switch and rollback on spare ports (needs Bonsai-1.7B and llama-server); the gate sets it |
| `BANKML_INFT_ARTIFACT` | `test_chain.py` | `~/DeltaVerse/deploy/iNFT4/out/iNFT_7857.sol/iNFT_7857.json` | the compiled `iNFT_7857` the devnet test deploys |
| `BANKML_PIN_CPUS` | `testing/pinned.sh` | the last two CPUs this shell may use | the cores a benchmark is pinned to (`taskset`) |
| `BANKML_PIN_MEM` | `testing/pinned.sh` | `2500M` | the hard memory cap, with no swap (a user cgroup through `systemd-run`) |

The Python test suites set `BANKML_UI_STATE`, `BANKML_AGENTS`, `BANKML_MODELS`, `BANKML_FORKS`, `RAGE_PATH`,
`SAVANTE_CANON`, `BANKML_VOICE_DIR` and `BANKML_EXPORT_DIR` to temporary directories themselves, so they never touch
your state.

---

## 7. Performance and resource tuning

The figures below are from [PERFORMANCE.md](PERFORMANCE.md) unless stated. They were measured on an AMD Ryzen 3
3200U laptop (2 cores, 4 threads, 5.8 GB) and a 2-vCPU EPYC VPS.

### Threads: three knobs that do different things

- **`BANKML_THREADS`** sizes bankML's own pool: native serve, `generate`, the C API. The default is every core.
  Answers are bit-identical at any thread count, so this is purely a speed and sharing choice. On the laptop (2
  cores, 4 threads), one token's ternary matmuls took 0.45–0.47 s on 1 thread, 0.31 s on 2 and 0.23–0.25 s on 3;
  a fourth thread added nothing. The 1-bit kernel is at parity with ggml at every count. On a shared machine, give bankML fewer threads than cores, as the service in §8 does with 1.
- **`--threads`** is passed to a spawned llama-server (`-t`). It has no effect in native mode. The importer's
  carrier uses `BANKML_THREADS_SERVE` (3) or the Admin tab's choice for whichever engine it starts.
- **`BANKML_LLAMA_THREADS`** (default 3) is about exactness, not speed. It must equal the `-t` of the llama.cpp run
  whose tokens you compare against, because llama.cpp's split attention kernel (decode at 512 or more KV cells)
  chunks by thread count. Leave it at 3 unless your reference ran with another `-t`.

### Context size and KV memory

The KV cache is f16 and grows as the context fills. For the 8B Qwen3 models it costs about 0.15 MB per token: a
2048-token context is about 0.3 GB, 4096 about 0.6 GB. The importer's carrier defaults to 2048 (`BANKML_CTX`); `serve`
on its own defaults to 4096. The trade-off ([usage.md §8](usage.md#8-the-files-savante-keeps-history-memory-prompt)):
on a 2048-token context, Savante's persona prompt and `.memory` leave room for only a few exchanges. With 4096, the
history window can stay put for several turns and keep the prompt cache warm. Set the context through the Admin
tab's RAM budget, so it is planned from the model's actual KV size, or with `--ctx`.

### One resident model, and keep-alive

Native serve holds one model at a time, as Ollama does with `MAX_LOADED_MODELS=1`. A request for another model
drops the resident one, then verifies and loads the new one: the guard and the full sha256, about 2.9 s for the
1.16 GB model with SHA-NI. Memory is therefore bounded by the largest model you serve plus its KV cache, never two
models at once. `--keep-alive` (default `5m`) decides how long an idle model's memory map stays. A short keep-alive
gives memory back sooner but pays the verify and load again; `-1` keeps the model for good. OpenAI-style and
llama-server-style requests keep the model resident.

### Prompt cache and slots

Both engines reuse the longest common prefix of the previous prompt. The UI's history window moves in steps of six
exchanges for this reason, so each prompt extends the last one. Both serve one request at a time on one slot (the
spawned llama-server runs with `-np 1`). The native engine (0.3.8) also keeps llama-server's host prompt cache: when a
request shares little with the slot, the slot's state goes to RAM and a cached state that keeps more of the prompt
comes back, so conversations that take turns do not recompute their history. `BANKML_CACHE_RAM` bounds it (§6). With `--slot-dir` (the importer always sets it; for the native
carrier since 0.3.8), a restarted engine restores the system prompt's saved KV instead of computing it again. On the laptop that took
the first answer after a restart from 132 s to 15 s. Each saved slot is about 51 MB per model, context and system
prompt.

### The GPU share

The GPU worker helps only the 1-bit kernels. On the integrated Vega 3 it measured unchanged within noise (1.93–2.00
tokens/s with the card, 1.96–1.97 without): the card is worth about one CPU core there, and the per-matrix submit
cancels the gain. A discrete card gains directly. The card's buffers come out of system RAM on an integrated GPU.
On a memory-tight machine, `BANKML_GPU=off` frees that memory and changes no token, because the GPU path is bit-exact
to the CPU path by its own oracle. `BANKML_GPU_SHARE` overrides the calibration if you have measured a better split.
In the next release (0.3.7), `BANKML_GPU_LIMIT` (default 0.8) caps the card's memory and busy time, and each matrix
shape decides by measurement whether the card pays. On the Vega 3 (Bonsai-1.7B) that measured neutral, 10.36 tokens/s
off against 10.30–10.38 on, with card memory down from 66 MB to about 13 MB ([CHANGELOG.md](../CHANGELOG.md)).

### SHA-NI

SHA-NI is on whenever the CPU has it. It makes every start, load and switch faster: restarting the 8B carrier went
from 25 s to 7.0 s. Set `BANKML_NO_SHANI=1` only to compare against the portable path or to rule it out while
debugging.

### Fixed-resource benchmarking

```sh
BANKML_PIN_CPUS=2,3 BANKML_PIN_MEM=2500M testing/pinned.sh <command …>
```

`pinned.sh` pins the command to fixed cores with `taskset` and caps its memory, with no swap, in a user cgroup
(`systemd-run --user --scope`). It records load and free memory before and after. It says plainly when the cgroup
cap could not be enforced, in which case only the CPU is pinned. Compare runs only when they were made on the same
pins.

### Small machines: the memory pitfalls the docs record

- **The ternary file must stay in the page cache.** With the chat engine and a browser holding memory, the 2.31 GB
  ternary file was only partly resident, so every token re-read it from disk (4.5–5.2 s per token instead of about
  0.23 s of matmuls). Measuring the whole-token figure needs at least 3 GB free.
- **One Piper render at a time.** `speak.py` takes a machine-wide lock (`$BANKML_PIPER/.render.lock`), because
  memory, not speed, is the limit. On a small machine run `python3 sAGI/speak.py` alone, or set
  `BANKML_VOICE_ASYNC=0` so the UI does not render in the background.
- **bge-m3 needs about 1.2 GB to load** and holds 1.14 GB while loaded. Hence the 1.3 GB free-memory guard
  (`BANKML_EMBED_NEED_GB`) and the 60 s keep-alive. On a laptop with about 1 GB free while the chat model runs,
  search falls back to BM25 rather than push the machine into swap.
- **Weights over 70 % of RAM** are refused by the importer for the same reason.

---

## 8. Running as a service

This is the shape of the deployed native server on the mindX VPS (2 vCPU EPYC, 7.8 GB). It runs a registry of the
pinned models, on one core, with a hard memory cap. Replace the paths with yours.

### The registry directory

The server reads every `*.FORK.json` in the registry directory and looks for each pinned file beside the startup
model and in that directory. So a registry can hold the pins and **symlinks** to the GGUFs wherever they really
live:

```
/srv/bankml/registry/
├── Bonsai-8B-Q1_0.gguf                   -> /srv/bankml/models/Bonsai-8B-Q1_0.gguf
├── Bonsai-8B-Q1_0.gguf.FORK.json
├── Ternary-Bonsai-8B-Q2_0_g64.gguf       -> /srv/bankml/models/Ternary-Bonsai-8B-Q2_0_g64.gguf
├── Ternary-Bonsai-8B-Q2_0_g64.gguf.FORK.json
├── Bonsai-1.7B-Q1_0.gguf                 -> …
├── Bonsai-1.7B-Q1_0.gguf.FORK.json
├── mindx-gen39-F16.gguf                  -> …
├── mindx-gen39-F16.gguf.FORK.json
└── mindx-gen39.MODEL.json                (a derived model, if you create one: `bankml create … --registry` this dir)
```

Copy the `FORK.json` files from the machine that imported the models (`~/.local/share/bankml/forks/`), or import on
the server itself. The link is never trusted: each file is verified against its pin when it loads. Verification
hashes the canonical (resolved) path, and before every answer the gateway re-checks that file's device, inode, size
and modification time.

### The unit

```ini
# /etc/systemd/system/bankml.service
[Unit]
Description=bankml serve (native, registry)
After=network.target

[Service]
User=bankml
Group=bankml
Environment=BANKML_THREADS=1
Environment=BANKML_FORKS=/srv/bankml/registry
ExecStart=/opt/bankml/bankml serve /srv/bankml/registry/Bonsai-8B-Q1_0.gguf \
    --fork /srv/bankml/registry/Bonsai-8B-Q1_0.gguf.FORK.json \
    --native --registry /srv/bankml/registry --ctx 2048 --listen 127.0.0.1:18093
CPUQuota=100%
MemoryMax=3G
Nice=10
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Line by line:
- **`User=`, `Group=`**: an unprivileged account that can read the registry and the models. `serve` writes nothing
  in native mode, unless `/api/create`, `/api/copy` or `DELETE /api/delete` write derived models into the registry.
  Give that account write access to the registry only if you want those endpoints to work.
- **`Environment=BANKML_THREADS=1`**: one pool thread. Native mode ignores `--threads`; this variable is its thread
  count. Unset, the pool takes every core.
- **`Environment=BANKML_FORKS=…`**: the forks directory for a bare `--registry`, and for `bankml create` run as this
  user without `--registry`. Here `--registry` names the directory explicitly, so the variable keeps the two in
  agreement rather than changing what the server serves.
- **`ExecStart=`**: the startup model and its pin (the default model, served when a request names none), native
  mode, the registry, a 2048-token context and the loopback gateway address. `--upstream` is left at
  `127.0.0.1:18092`: native mode also listens there with llama-server's endpoints. If something else holds 18092 on
  this host, add `--upstream 127.0.0.1:<free port>`.
- **`CPUQuota=100%`**: at most one core's worth of CPU time, matching `BANKML_THREADS=1`. The VPS keeps its second
  core for everything else.
- **`MemoryMax=3G`**: a hard cap sized for the worst case. One model is resident at a time, so the largest one
  decides: the ternary model at 2.31 GB mapped, plus the f16 KV cache at `--ctx 2048` (about 0.3 GB for the 8B
  models), plus the process. If you raise `--ctx`, raise the cap with it.
- **`Nice=10`**: lower scheduling priority than interactive work on the same host.
- **`Restart=on-failure`**: `serve` exits 2 when it refuses to start (a file that fails its pin, a busy port).
  systemd restarts it after any failure, and the journal keeps the reason each time.

```sh
sudo systemctl daemon-reload && sudo systemctl enable --now bankml
journalctl -u bankml -f      # "bankml serve 0.3.6 --native: … verified (sha256 …); models …; listening on …"
```

### Verifying a deployed server

```sh
curl -s 127.0.0.1:18093/bankml          # verified, model, the resident model, every name in the registry
curl -s 127.0.0.1:18093/api/tags        # every pin: name, size, digest = its sha256, "bankml": {"native", "reason"}
curl -s 127.0.0.1:18093/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'   # look for bankml_receipt.model_sha256
curl -s 127.0.0.1:18093/bankml/usage    # what serve uses now (RSS, CPU %)
```

Run these on the server itself, or through an SSH tunnel (`ssh -L 18093:127.0.0.1:18093 host`). The gateway answers
loopback `Host` names only (§9). Compare `model_sha256` in the receipt with the pin in the `FORK.json`.

### Upgrades and rollback

Keep each release's binary beside the last one and point the unit at a symlink:

```sh
cp target/release/bankml /opt/bankml/bankml-0.3.6
ln -sfn bankml-0.3.6 /opt/bankml/bankml && sudo systemctl restart bankml
# rollback: ln -sfn bankml-0.3.5 /opt/bankml/bankml && sudo systemctl restart bankml
```

The binary has no dependencies to roll back with it. The pins and models do not change between versions.

---

## 9. Security and exposure

| service | bound to | why |
|---|---|---|
| `bankml serve` | loopback | it has no authentication. It requires a loopback `Host` (`127.0.0.1`, `localhost`, `[::1]`, else 403) and a JSON `Content-Type` on POST (else 415), so a web page can drive it neither by DNS rebinding nor by a simple cross-origin form POST. It caps connections, header sizes and body sizes, and times out slow clients |
| interact mode (7873) | loopback only | the installed Gradio 3.37 has published path-traversal CVEs (CVE-2023-51449, fixed in 4.11), and interact mode runs the Verifier, devnet mints and PostgreSQL publishing. `savante.py` refuses any `--host` other than `127.0.0.1` or `localhost` and exits with the reason. It also adds a trusted-host check, so a request whose `Host` is not loopback is refused |
| the bankML console (7875, 0.3.7) | loopback only | it can restart the engine with new CPU, RAM and GPU limits. `console.py` refuses any `--host` other than loopback |
| view mode (7874) | `0.0.0.0` by default | the page for the LAN, built on the Python standard library, not Gradio. It has a fixed set of GET routes (`/`, `/api/state`, `/api/result?name=` for a name `testing/results/` lists, `/savante.png`, `/favicon.ico`, `/knobs.js`, `/audio/*.ogg`, `/export/*.opus`), and every POST answers 405. Every response carries `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and a Content-Security-Policy: `default-src 'none'`, with images, scripts, media and connections from `'self'` only and inline styles and scripts allowed; file downloads get `default-src 'none'` alone. It shows commitments, never the content of `.history` or `.memory` |
| Ollama (embeddings) | keep it local | history text is sent to it |
| anvil devnet | loopback | started on `127.0.0.1:8545` |

**Never expose to a network:**
- `bankml serve` and its engine port 18092. A reverse proxy that forwards the public `Host` gets 403, and one that
  rewrites `Host` would publish an unauthenticated model server.
- Interact mode.
- The PostgreSQL DSN.
- The `install.env` and UI state directories.

Use view mode for others on the LAN, and an SSH tunnel for yourself.

**Receipts are not signatures.** A receipt binds an answer to its request and to the verified model's sha256, between
you and your own `bankml serve`. It proves nothing to a third party who does not trust your machine
([usage.md §9](usage.md#9-receipts-and-how-to-check-an-answer)).

The installer needs no sudo. Every download is checked before use: the llama.cpp archive against its published
sha256, models against their publisher's sha256 and the guard. On a public chain bankML never signs; the iNFT path
sends transactions only to a chain with id 31337 (anvil).

---

## 10. Development and the release gate

```sh
cargo test --release                                                    # offline, no models needed
BANKML_GGML_LIB=/path/to/llama-b11192 LLAMA_SRC=/path/to/llama.cpp-b11192 \
  BANKML_LLAMA_SRC=/path/to/llama.cpp-b11192 testing/release_gate.sh    # → testing/results/<version>.txt
```

- **`BANKML_GGML_LIB`** is the unpacked b11192 release, the same directory the installer's `engine` step makes
  (`~/.local/share/bankml/llama-b11192`). The oracles load ggml's own compiled kernels from it in-process, and compare
  bankML's results bit for bit.
- **`LLAMA_SRC`** is a llama.cpp checkout at tag b11192
  (`git clone --depth 1 --branch b11192 https://github.com/ggml-org/llama.cpp`). It enables the schema and content
  oracles, which compile against its headers.
- **`BANKML_LLAMA_SRC`** (default `upstream/llama.cpp`) is the source tree whose converter the conversion oracles
  run. It enables `oracle_name_heuristics`.
- **Without `BANKML_GGML_LIB`,** the gate runs only its offline part: build, tests, C API tests, clippy, licence
  headers, the Python suites, guard agreement and the printf oracle. It prints `BANKML_GGML_LIB unset: oracles, A/B
  and budgets skipped`.
- **Models.** Every oracle that needs a model reads it from `.models/`, and every live stage needs that model's pin
  in `BANKML_FORKS`. A missing model or record is reported as skipped, not passed.

**Running stages individually.** Each oracle is an ignored Rust test. The gate finds its full path and runs it alone:

```sh
cargo test --release -q -- --list --ignored | grep oracle_json_mode        # its full name
cargo test --release -- --ignored --exact <full::name> --nocapture --test-threads=1
testing/live.sh "json mode" cargo test --release -- --ignored --exact <full::name> --nocapture --test-threads=1
python3 -B testing/serve_oracle.py --bankml Bonsai-1.7B-Q1_0                # a live stage on its own
```

`testing/live.sh` appends a step's output to `testing/live.log`, which view mode shows as it runs. The live stages
start their own `serve --native` on spare loopback ports (`serve_oracle.py` uses 18195 and 18196), so they do not
collide with a running carrier on 18092 and 18093.

**Time and memory.** A full gate runs every oracle one after another (`--test-threads=1`), and takes a long time on a
laptop: plan to leave the machine to it. Memory is the usual limit:
- The live stages load real models. The 0.3.5 run was stopped by the development session's memory guard, and the
  0.3.6 run stalled at the same stage, because the Vega 3 GPU worker's buffers come out of system RAM. In both, the
  remaining stages ran with `BANKML_GPU=off` and the record marks where. The next release adds `BANKML_GPU_LIMIT`
  (0.3.7) for this.
- The whole-token budgets (`decode_budget_*`) measure the disk, not the kernel, unless the model stays in the page
  cache. That needs at least 3 GB free for the ternary model.
- Close other work and stop the carrier (`./install.sh stop`) before a gate run, and set `BANKML_GPU=off` on an
  integrated-GPU machine short of memory.

A speed counts only if every oracle passed on the same code. The full list of tests and their records is in
[testing/README.md](../testing/README.md) and [oracles.md](oracles.md).

---

## 11. Troubleshooting

### Installing

| symptom | cause | fix |
|---|---|---|
| `./install.sh: Permission denied` | the copy lost its executable bit (zip download, USB stick) | `bash install.sh`, or `chmod +x install.sh` |
| `unknown argument: …` (exit 2) | a typo, or an option `install.sh` does not have | `./install.sh --help` |
| `Rust X is older than 1.99` | an older toolchain without rustup | install rustup (it reads `rust-toolchain.toml`), or `rustup toolchain install 1.99.0` |
| `python3 -m venv failed` | `python3-venv` not installed | `sudo apt install python3-venv`, or `BANKML_PYTHON=/path/to/python` with Gradio 3 and numpy |
| `… sha256 … is not the published …: refused` | a corrupted or substituted llama.cpp download | run `./install.sh engine` again; the bad file was deleted |
| `no llama.cpp b11192 build for <OS>/<arch> here` | the release archive is Linux x86_64 only | build llama.cpp b11192 and set `BANKML_LLAMA_SERVER` |
| `bankml serve did not come up: see …/carrier.log` | the carrier failed to verify or start | read `~/.local/share/bankml/savante/carrier.log`; the last lines name the refusal |
| a UI step says `did not start: see …/logs/<name>.log` | Python, Gradio or the canon | read that log |

### Starting `bankml serve`

| symptom | cause | fix |
|---|---|---|
| the port does not open for a while after start | `serve` binds only after hashing the whole file and, with `--spawn`, after llama-server is healthy (up to 900 s for a cold load) | wait for `GET /bankml`, not for the port. Hashing the 1.16 GB model takes about 3 s with SHA-NI |
| `refuse: … != pinned` | the file is not the one the `FORK.json` pins | re-download it; `bankml sha256 FILE` |
| `… changed while it was being hashed: refused` | the file was being written or replaced during the start | let the copy finish, then start again |
| `<upstream> already answers: an engine is running there …` | `--spawn` with something already on `--upstream` | stop it, or drop `--spawn` and use `--upstream` to check it by its `/props` |
| `upstream … serves X, not the verified Y: refused` | the llama-server on that port runs another file | restart it on the verified file, or use `--spawn` |
| `upstream … not healthy after N s` | llama-server not up | check its log; a cold load can take a minute |
| `the engine exited (…) before it was healthy` | llama-server could not load the model (memory, a wrong build) | run the same llama-server by hand to see its error |
| `cannot listen on the engine address 127.0.0.1:18092 … (is llama-server running there?)` | native mode also binds `--upstream`, and something holds it | stop that process, or pass `--upstream 127.0.0.1:<free port>` |
| `cannot listen on 127.0.0.1:18093` | another `serve` is running | `./install.sh status`; `./install.sh stop` |
| `the carrier's ports are still held by [pids]; stop them by hand` | the importer can stop only this user's `bankml` and `llama-server` | stop those processes yourself |

### Requests

| symptom | cause | fix |
|---|---|---|
| `403 … loopback clients only` | the `Host` header is not `127.0.0.1`, `localhost` or `[::1]` (a LAN address, a proxy) | call it on loopback, through an SSH tunnel if remote |
| `415 POST bodies must be Content-Type: application/json` | `curl -d` alone sends a form type | add `-H 'Content-Type: application/json'` |
| `400 chunked requests are not accepted` | a client that streams the request body | send `Content-Length` |
| `413 request too large` | a body over 16 MiB | send less |
| `503 too many connections` | 32 connections already open | fewer concurrent clients; one model answers one request at a time anyway |
| `400 request (N tokens) exceeds the available context size (M tokens)` (0.3.8) | the prompt does not fit `--ctx` | trim the conversation, or restart with a larger `--ctx` if it fits in memory |
| `501 This server does not support slots action` | a `/slots` request to a server started without `--slot-dir` | restart with `--slot-dir DIR` |
| `404 model '…' not found: bankml serves pinned models only (…)` | a name that no pin has | use one of the names listed; add `--registry` to serve more than the startup model |
| `400 num_ctx N … is larger than the context this server holds` | a request or Modelfile asks for more than `--ctx` | restart with a larger `--ctx`, if it fits in memory |
| a 400 naming a refused option, sampler, `tools`, `images`, `think: true` or a schema | not something bankML reproduces exactly | the message says what; [usage.md §6a](usage.md#6a-ollamas-api) lists every refusal |
| the chat says *Cannot reach bankml serve* | the carrier is not running | `./install.sh status`, then `./install.sh model` |

### Savante, models and resources

| symptom | cause | fix |
|---|---|---|
| `refused: interact mode serves loopback only …` | `--host 0.0.0.0` (or any non-loopback host) on interact mode | use view mode for the LAN |
| `savante.py` started by hand cannot start the carrier | without the installer's environment, `BANKML_LLAMA_SERVER` defaults to `~/sAGI/bonsai/llama-b11192/llama-server` | export `BANKML_LLAMA_SERVER` to the engine in `install.env` (`INSTALL_LLAMA_SERVER`), or start with `./install.sh start` |
| an import says `licence … is not open source` | the model's licence is not on the OSI list | not importable, by design |
| `needs X GB + 1.5 GB margin; Y GB free on disk` | not enough disk | free space or set `BANKML_MODELS` to a larger disk |
| `X GB of weights on a Y GB machine would page to swap` | weights over 70 % of RAM | a smaller model |
| `… exists but is not the published file (sha256 differs); move it aside first` | a stale or partial file with that name | move it, import again |
| `… is an embedding model …; it cannot be the chat carrier` | bge-m3 and other encoders | use them through Ollama for embeddings ([embedding.md](embedding.md)) |
| **Apply** on the Admin tab says the budget cannot hold the model | the RAM budget is under weights + 250 MB + 512 tokens of KV | raise the budget; the message gives the minimum |
| search says BM25 alone | Ollama down, bge-m3 not pulled, or less than 1.3 GB free | `ollama pull bge-m3`; free memory, or lower `BANKML_EMBED_NEED_GB` if you know the risk |
| *refused: savante.persona does not verify* | the canon was edited or is incomplete | `git -C ~/cryptoAGI/savante status`; run the Verifier |
| the log says `bankml: GPU not used: … not verified` | the card failed the on-card oracle | expected behaviour: bankML stays on the CPU. `bankml gpu --verify` shows the row that differs |
| the machine swaps during a gate or a long session | the integrated GPU's buffers, a second model, Piper, bge-m3 | `BANKML_GPU=off`, one Piper render at a time, stop the carrier before a gate |
| view from another computer does not load | firewall, or view bound to 127.0.0.1 | `--host 0.0.0.0`; allow TCP 7874 |

The basic symptoms are also in [usage.md §12](usage.md#12-troubleshooting).
