# bankML on Hugging Face

The files of [PYTHAI/bankml](https://huggingface.co/spaces/PYTHAI/bankml), kept here as their source of record.

| file | in the Space | what it is |
|---|---|---|
| `space/index.html` | `index.html` | the page: the DELTAVERSE banner, who speaks (bankML or Savante), who answers as four tabs over the conversation, the browseML consent card, the not-found card, the activity line, this computer's controls and diagnostics, the FAQ and Logs tabs; a meta CSP limits `connect-src` to the page, Hugging Face, jsdelivr (the Gradio client), the Savante persona on GitHub and 127.0.0.1/localhost |
| `space/bankml-chat.js` | `bankml-chat.js` | the engine of the ultimate input field, four tabs: *bankML · free CPU* (the default: Bonsai on Hugging Face's free CPU through the Space Gregory-L/mindXhfgradio, asked as PYTHAI/savante asks it — llama.cpp, not bankML's engine, no receipt, said on every answer; a `/bankml` endpoint with a receipt is used and checked when the Space offers one), *browseML* (bankML in this browser, after the visitor's click on the consent card), *your own bankML* (`bankml serve` started with `./install.sh --space`; when it does not answer, a card with browseML now, the install steps for this device, and Hugging Face), or *Hugging Face* (labelled not bankML, no receipt, your quota); receipts checked in the page; the activity line and each answer's timing record |
| `space/browseml-fetch.js` | the same | browseML's model download under the visitor's control: progress, MB/s, time left, pause (kept in the tab's memory), cancel (nothing kept); what browseML keeps, persistent storage, clearing it |
| `space/bankml-creds.js`, `bankml-creds-LICENSE.txt` | the same | **GPL-3.0-only**: the credential module — Hugging Face sign-in (OAuth, PKCE), the token kept in the tab by default and on the device only when chosen, the optional pasted token, the "your credentials" panel, and the only function that sends the token (to huggingface.co and router.huggingface.co only). A copy of ultimate-bankml-ui's `dist-creds/bankml-creds.js` (source `src/creds/`); the page's core reaches it only through `window.bankmlCreds` and works without it. `testing/creds_split.py` proves no core file carries token code |
| `space/browseml*.js`, `space/browseML/*.FORK.json` | the same | browseML: the engine in WebAssembly, its WASI, its worker and threads, the model's pin ([browseML/README.md](../browseML/README.md)) |
| `space/browseml.wasm`, `browseml-mt.wasm`, `browseml-mt-relaxed.wasm` | the same | built by `tools/browseml.sh`, not committed |
| `space/space-dashboard.js` | the same | this computer's controls (threads, context with its RAM estimate, answer length, lean or full prompt), disk and cache (what browseML keeps, persistent storage and the request to keep it, remove after this session, clear with an in-page confirmation) and diagnostics (speed, memory, storage, what to change); T mode's `diag` |
| `space/uif-space.js`, `uif-space.css` | the same | the ultimate input field (MIT), built from ultimate-bankml-ui `bankml-uif` (`vite.space.config.ts`, commit ac4d143): response windows that fill the screen and shrink back, close all, tile, cascade, the Responses menu; IME-safe Enter, voice input where the browser has it, `/` or Ctrl/Cmd+K to the field; and for this page `askSpace` (the free CPU), the device checks and install steps, and `loadCreds` |
| `space/space-tabs.js` | the same | the highlights until the first question; the FAQ and Logs tabs |
| `sAGI/personas/bankml.context`, `savante.context` | the same | what each persona knows of its own repositories (`tools/context.py`) |
| `Dockerfile`, `start.sh` | `Dockerfile`, `hf/start.sh` | the live engine for when the Space has Docker hardware: `bankml serve --native` with the ternary Bonsai-8B and `sAGI/console.py --public` in front |

The Space is **static** because Hugging Face hosts Docker and Gradio Spaces on its free CPU only for paid plans (PRO for
a user, Team or Enterprise for an organisation; checked 2026-10-06). The Space's tree is this repository (`git
archive`), these files on top, and the README's front matter (`sdk: static`, `hf_oauth: true` with the
`inference-api` scope, and `custom_headers` — `cross-origin-opener-policy: same-origin`,
`cross-origin-embedder-policy: credentialless` — so browseML runs on threads).

Licences: the page's core is `MIT OR Apache-2.0` (the input field's bundle MIT); `bankml-creds.js` is `GPL-3.0-only`, in
its own file the core never imports — bankML's rule for code that makes, holds or checks a secret
([LICENSING.md](../LICENSING.md)). The page's footer says the same.
