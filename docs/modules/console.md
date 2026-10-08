# `sAGI/console.py` — the bankML console: bankML as itself, and what it measures

## Summary

The console (0.3.7; redesigned 2026-10-06) is bankML's own interface, deliberately unlike Savante's: a quiet, typographic
page, light or dark with the system, with six tabs over `bankml serve` — **Ask** (the landing: one question, one
streamed answer, its receipt checked in the browser), **Admin** (CPU, RAM and GPU sliders, the measured "now", and D3
charts), **Engine** (`GET /bankml/status` drawn by serve's own renderer: checks, CPU, memory, disk, GPU, the engine's log), **Receipts** (every answer's receipt, newest first, each re-checked and expandable to its JSON; the
commitments an iNFT of the session carries), **Logs** (the engine's log) and **Diagnostics** (0.4.0: measured checks of the engine — reachable, verified, metrics, memory, the engine log's last error — and every answer's trace, span by span, with durations and timed events; `sAGI/diagnostics.py`, after LlamaIndex's instrumentation). A **Savante | bankML** switch in its bar
and the same switch in Savante's header move between the two interfaces. A ☾/☀ button flips light and dark (kept per browser; `console/theme.js`), and `console/theme.css` gives every surface depth from its border, glass, a high-contrast question field and the input field's T mode in terminal green on black; the tab icon is the DeltaVerse $ (`sAGI/favicon.ico`, as Savante and view). It talks to bankML as itself — `sAGI/personas/bankml.persona` — and gives that
persona a SELF block measured at each question, so what bankML says about its own tokens, speed, load and power is
measurement, not invention.

**The first answer (0.4.3).** The persona's system prompt is sent first and unchanged; SELF, measured at each question,
rides as a second system message just before it, so the engine's prompt cache keeps the persona and the conversation
from turn to turn. When a newly started engine appears, the console restores the persona's saved slot or prefills it
once and saves it, so the first question does not pay for it; a question during that prefill cancels it. The
conversation is trimmed four exchanges at a time (`window`), so the cached prefix survives long conversations.

**Memory per response window (0.4.3, Ask → Advanced).** Each response window keeps a `.memory` named after its title
(`main` for the plain box), and a *collection* above them holds higher-level notes every window shares. Notes are
added by hand or gathered by reviewing `.history` (the console's or Savante's). Options: use each, their budgets,
recall from `.history` (0–4, off by default). Storage and limits: `sAGI/console_memory.py`.

**Ping and diagnostics (0.4.3).** In T mode, ◎ `ping` times three round trips to bankml serve (`/api/ping`), and ⚕
`diag` prints the machine now — CPU, memory, disk, GPU and the engine — as an accordion in the terminal log
(`/api/sysdiag`, `sAGI/sysdiag.py`), the sections that need a look open.

## Technical usage

```sh
python3 sAGI/console.py [--host 127.0.0.1] [--port 7875]   # loopback hosts only; bankml serve at $BANKML_SERVE_LISTEN (127.0.0.1:18093)
```

| route | what |
|---|---|
| `GET /`, `/app.js`, `/style.css`, `/vendor/d3.v7.min.js`, `/vendor/d3.LICENSE` | the page and its assets (D3 v7.9.0, ISC, vendored) |
| `GET /api/state` | `/bankml`, `/bankml/usage`, `/bankml/metrics`, the saved resources, the persona and its doctrine root, the SELF block |
| `POST /api/ask` `{"message", "history"?, "max_tokens"?, "temperature"?, "top_k"?, "top_p"?, "min_p"?, "repeat_penalty"?, "seed"?}` | NDJSON: `{"piece"}` lines, then `{"done", "error", "receipt", "metrics", "answer_sha256_ok"}`; the exchange (question, answer, receipt, the engine's metrics record, the SELF block it was asked with) is appended to `$BANKML_UI_STATE/console.jsonl` |
| `POST /api/resources` `{"threads", "ram_gb", "gpu_limit"}` | one verified restart of the native engine with rollback (`models.apply_resources`) |
| `GET /api/log` | the last 200 exchanges and the engine log's tail |
| `GET /api/infotags` | ERC-721 metadata: name, description, `attributes`, and the RFC 6962 root and CIDs |
| `GET /api/diagnostics` | the measured checks (`level` ok, warn or bad, and what was `seen`), the newest answers' trace trees (spans, durations, events) and the same as text |
| `GET /api/engine` | bankml serve's `GET /bankml/status`, passed through (502 with where it looked when serve does not answer) |

`/api/ask` sends the persona's system prompt with the SELF block appended, the last 12 user and assistant turns of
`history`, and the question to `/v1/chat/completions` (streamed, `max_tokens` 256 unless set). The SELF block is
rendered for the model as one sentence per measurement with its unit (`self_text`), "not measured" for a `null` — a
bare JSON key (`last_eval_tps`) was read as seconds in testing.

## How it is verified

`testing/test_console.py` (18 checks, in the gate, no engine needed): every asset served; the CSP; unknown paths 404; a foreign
`Host` 403; a cross-origin POST 403; a non-JSON POST 400; an empty question 400; SELF says "not measured" without an engine and names units
with one; the Infotags root equals `savante.merkle_root` over the log's exact lines. Live on Bonsai-1.7B: "I am bankML…",
its token totals and generation speed read correctly from SELF, and every receipt's sha256 matches the streamed text.

## Advantages and efficiency

- **No framework, no network.** Python's standard library and one vendored D3; CSP `default-src 'none'` with `'self'`.
- **Safe controls.** The sliders restart the engine through the importer's verified switch with rollback; the console
  refuses any host but loopback, because it can restart the engine.
- **Proof without disclosure.** The exchanges stay on disk; Infotags publishes their Merkle root and CID, as Savante's
  `.history` commitments do.
- **The persona's prompt is cached.** The persona comes first and the SELF block last, so after the first question the
  engine reuses the persona's prefix (first answer 37 s, then 16 s to the first token on Bonsai-1.7B).

## Limitations

- The persona's system prompt is about 550 tokens: the first question of a session pays for it on the CPU.
- It forwards only `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty` and `seed`; the rest of the 0.3.7
  sampler chain and logprobs are reachable through `/v1` directly, not from the page. One conversation at a time
  (bankML's single slot; requests sent together are answered in turn).
- The page has no DeltaVerse substrate background yet.

## See also

[metrics.md](metrics.md) · [gpu.md](gpu.md) · [sys.md](sys.md) · [../usage.md](../usage.md) §6c · [../install.md](../install.md)
