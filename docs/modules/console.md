# `sAGI/console.py` — the bankML console: bankML as itself, and what it measures

## Summary

The console (0.3.7) is a small page with four tabs over `bankml serve`: **Interaction** (ask; the streamed answer and its
receipt), **Admin** (CPU, RAM and GPU sliders and D3 charts of measured use), **Logging** (every exchange) and
**Infotags** (iNFT publication metadata). It talks to bankML as itself — `sAGI/personas/bankml.persona` — and gives that
persona a SELF block measured at each question, so what bankML says about its own tokens, speed, load and power is
measurement, not invention.

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
