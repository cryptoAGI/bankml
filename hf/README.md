# bankML on Hugging Face

The files of [PYTHAI/bankml](https://huggingface.co/spaces/PYTHAI/bankml), kept here as their source of record.

| file | in the Space | what it is |
|---|---|---|
| `space/index.html` | `index.html` | the page: the DELTAVERSE banner, who speaks (bankML or Savante) and who answers, the activity line, this computer's controls and diagnostics, the FAQ and Logs tabs |
| `space/bankml-chat.js` | `bankml-chat.js` | the engine of the ultimate input field, three ways: *bankML in this browser* (browseML, the default), *your own bankML* (`bankml serve` started with `./install.sh --space`), or *a Hugging Face provider* (labelled not bankML, no receipt); receipts checked in the page; the activity line and each answer's timing record |
| `space/browseml*.js`, `space/browseML/*.FORK.json` | the same | browseML: the engine in WebAssembly, its WASI, its worker and threads, the model's pin ([browseML/README.md](../browseML/README.md)) |
| `space/browseml.wasm`, `browseml-mt.wasm`, `browseml-mt-relaxed.wasm` | the same | built by `tools/browseml.sh`, not committed |
| `space/space-dashboard.js` | the same | this computer's controls (threads, context, answer length, lean or full prompt) and diagnostics (speed, memory, storage, what to change); T mode's `diag` |
| `space/uif-space.js`, `uif-space.css` | the same | the ultimate input field, built from ultimate-bankml-ui `space-mount` (`vite.space.config.ts`): response windows that fill the screen and shrink back, close all, tile, cascade, the Responses menu |
| `space/space-tabs.js` | the same | the highlights until the first question; the FAQ and Logs tabs |
| `sAGI/personas/bankml.context`, `savante.context` | the same | what each persona knows of its own repositories (`tools/context.py`) |
| `Dockerfile`, `start.sh` | `Dockerfile`, `hf/start.sh` | the live engine for when the Space has Docker hardware: `bankml serve --native` with the ternary Bonsai-8B and `sAGI/console.py --public` in front |

The Space is **static** because Hugging Face hosts Docker and Gradio Spaces on its free CPU only for paid plans (PRO for
a user, Team or Enterprise for an organisation; checked 2026-10-06). The Space's tree is this repository (`git
archive`), these files on top, and the README's front matter (`sdk: static`, `hf_oauth: true` with the
`inference-api` scope, and `custom_headers` — `cross-origin-opener-policy: same-origin`,
`cross-origin-embedder-policy: credentialless` — so browseML runs on threads).
