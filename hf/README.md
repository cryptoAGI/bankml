# bankML on Hugging Face

The files of [PYTHAI/bankml](https://huggingface.co/spaces/PYTHAI/bankml), kept here as their source of record.

| file | in the Space | what it is |
|---|---|---|
| `space/index.html` | `index.html` | the page: why bankML, the persona, the source, and **Ask bankML** |
| `space/bankml-chat.js` | `bankml-chat.js` | Ask bankML, two ways: *your own bankML* (the browser talks to your `bankml serve`, started with `./install.sh --space`; the receipt's sha256 is checked in the page) or *a Hugging Face provider* (sign in; labelled not bankML, no receipt) |
| `Dockerfile`, `start.sh` | `Dockerfile`, `hf/start.sh` | the live engine for when the Space has Docker hardware: `bankml serve --native` with the ternary Bonsai-8B and `sAGI/console.py --public` in front |

The Space is **static** because Hugging Face hosts Docker and Gradio Spaces on its free CPU only for paid plans (PRO for
a user, Team or Enterprise for an organisation; checked 2026-10-06). The Space's tree is this repository (`git
archive`), these files on top, and the README's front matter (`sdk: static`, `hf_oauth: true` with the
`inference-api` scope).
