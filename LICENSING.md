# Licensing

**Open source or go away — and do what you want with it, rights preserved.** bankml's own code is licensed
**`MIT OR Apache-2.0`**, at your option: [LICENSE-MIT](LICENSE-MIT) or [LICENSE-APACHE](LICENSE-APACHE). MIT is
"do anything, keep the copyright notice"; Apache-2.0 says the same with an explicit patent grant, for whoever wants
one. Either choice lets bankml be used by permissive and copyleft projects alike (GPLv2, GPLv3 and AGPL included).
Copyright stays with the authors: cryptoAGI — Professor Codephreak and Gregory L. Magnusson.

## Layers

| layer | where | licence (SPDX) | why |
|---|---|---|---|
| bankml's own code: runtime, kernels, gateway, UI, tests | everything today | `MIT OR Apache-2.0` | maximal freedom, rights preserved |
| client-side encryption and key handling: anything that makes, holds or checks a secret | `crypto/` (none yet; the first will be receipt signing) | `GPL-3.0-only` | transparency: no black-box modification of key handling can ship, because every build that contains it must publish its source |
| code derived from AGPL projects | `agpl/` (none yet) | `AGPL-3.0-only` (or `MIT AND AGPL-3.0-only` where a file mixes both) | the upstream licence stays intact |

Rules that keep the layers honest:

- **The core never imports `crypto/` or `agpl/`.** Neither exists yet. When one does, it **will be** an opt-in cargo
  feature (`crypto`, `agpl`), off by default, so the default binary stays `MIT OR Apache-2.0`; a build with either
  feature is, as a whole, under that feature's copyleft licence, and **will** say so in `bankml version` and in
  `GET /bankml`.
- **AGPL §13** (network use) is met the way bankon-vault meets it: the source is public, and a deployment runs a
  tagged commit whose source anyone can fetch.
- **Every source file carries an SPDX header**, and the release gate checks it (`testing/spdx_check.py`): a file in
  `crypto/` must be `GPL-3.0-only`, one in `agpl/` must name `AGPL-3.0-only`, everything else `MIT OR Apache-2.0`.
- **Code is taken only from projects whose licence allows it here**: MIT/BSD/Apache code keeps its notice (e.g.
  llama.cpp's, from upstream `ggml-org/llama.cpp`, never from a fork's copy); AGPL/GPL code only inside the layer
  above; nothing from projects that are not open source. Ideas from any project may be reimplemented from their
  documentation or papers (KoboldCpp was studied that way, docs only).

## Other material

| what | licence |
|---|---|
| `upstream/` (the AVX2 `Q2_0` kernel prepared for llama.cpp) | `MIT`, llama.cpp's licence, so it can be contributed as is |
| `sAGI/diagnostics.py` (its span model) | ported from [LlamaIndex](https://github.com/run-llama/llama_index)'s instrumentation (MIT, Copyright (c) Jerry Liu); the file is `MIT OR Apache-2.0`, with that notice kept |
| `sAGI/console/uif/` (`uif.js`, `uif.css`) | a build of [ultimate-input-field](https://github.com/Professor-Codephreak/ultimate-input-field) (MIT) and React (MIT), with the console's mount; source and build in `sAGI/console/uif/README.md` |
| `testing/gguf_guard.py` | vendored from [minaiml](https://github.com/minaiml) (same authors); `MIT OR Apache-2.0` here |
| `sAGI/voice/knobs/savante_knobs.js` | a build of DreamKnob and React (MIT); regenerate with `node sAGI/voice/knobs/build.mjs` |
| `sAGI/voice/cache/`, `sAGI/voice/export/` | Savante's voice, rendered by bankml with Piper and the `en_GB-cori-high` voice (trained on public-domain LibriVox recordings); offered under `MIT OR Apache-2.0` |
| models | not part of this repository; bankml imports only models with open-source licences and pins each by sha256 (`sAGI/models.py`) |
