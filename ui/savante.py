#!/usr/bin/env python3
"""bankml · Savante — the local chat UI, in Gradio, built from the Hugging Face template PYTHAI/savante.

What it takes from the template (the Space's `local:bonsai-8b` carrier): the system prompt is the canon
persona's own `system_prompt`, the history is the last 12 exchanges (each ≤ 4000 characters), answers are
labelled drafts, `<think>` blocks are stripped and Qwen3 carriers get ` /no_think`.

What it adds:
  · **iNFT compatibility.** Savante's canon (~/savante, or SAVANTE_CANON) is read and never written. At start
    every file the ledger `savante.commitments.json` commits to — the nine artefacts, the agent card, the image —
    is re-hashed (sha256) and compared; if the persona does not verify, the UI refuses to speak as Savante.
    `bind/savante_verify.py` (the full offline verifier: doctrine root, thot, mirror) runs from a button.
    Nothing here mints, and nothing is written into the canon — not even a Python cache.
  · **.prompt.** Which system prompt carries the conversation: the persona's `system_prompt` (canon, ledgered;
    the default), `sAGI.prompt` (canon facet, ledgered) or the Space's `Savante.prompt` (template, not ledgered).
    Each is shown with its sha256 and whether the ledger vouches for it.
  · **.history.** Every exchange is one JSONL line in `savante.history` under BANKML_UI_STATE
    (default ~/.local/share/bankml/savante/), outside the canon; the last session reloads on start.
  · **bankml.** Answers come through `bankml serve` — the guard and the sha256 pin in front of the model, and a
    receipt (model sha256, tokens, timings, sha256 of the answer) on every response.

  · **Two modes.** `--mode interact` (the operator: chat, .prompt, .history, the verifier) and `--mode view`
    (read-only, for anyone watching: the live testing log, every release's results, CI, the laptop's load, and
    Savante's office and ledger — no chat, no .history, nothing that runs a command).

  bankml serve .models/Bonsai-8B-Q1_0.gguf --fork FORK.json --upstream 127.0.0.1:18092 --listen 127.0.0.1:18093
  python3 ui/savante.py --mode interact               # http://127.0.0.1:7873
  python3 ui/savante.py --mode view --port 7874       # http://127.0.0.1:7874

stdlib + gradio (3.x or newer).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.dont_write_bytecode = True  # never leave a cache next to anything we import

CANON = Path(os.environ.get("SAVANTE_CANON", Path.home() / "savante")).expanduser()
SERVE = os.environ.get("BANKML_SERVE", "http://127.0.0.1:18093")
STATE = Path(os.environ.get("BANKML_UI_STATE", Path.home() / ".local" / "share" / "bankml" / "savante")).expanduser()
HISTORY = STATE / "savante.history"
REPO = Path(os.environ.get("BANKML_REPO", Path(__file__).resolve().parents[1])).expanduser()
LIVE = REPO / "testing" / "live.log"
CI_API = "https://api.github.com/repos/cryptoAGI/bankml/actions/runs?per_page=5"
SPACE_PROMPT_URL = "https://huggingface.co/spaces/PYTHAI/savante/resolve/main/Savante.prompt"
KEEP_EXCHANGES, KEEP_CHARS = 12, 4000  # the template's buildMessages limits


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── the canon, read-only, checked against its ledger ─────────────────────────────
class Canon:
    def __init__(self, root: Path):
        self.root = root
        self.ledger = self._json("savante.commitments.json") or {}
        self.persona = self._json("savante.persona") or {}
        self.card = self._json("savante.agentcard.json") or {}
        self.rows = self._check()

    def _json(self, rel):
        try:
            return json.loads((self.root / rel).read_bytes())
        except Exception:
            return None

    def _check(self):
        """(name, path, ok, detail) for every file the ledger commits to."""
        want = [(k, v.get("path"), v.get("sha256")) for k, v in (self.ledger.get("artifacts") or {}).items()]
        card = self.ledger.get("card") or {}
        v0 = (card.get("versions") or {}).get("v0") or {}
        want.append(("card v0", card.get("path"), v0.get("sha256")))
        img = self.ledger.get("image_candidate") or {}
        want.append(("image (named, not pinned)", img.get("path"), img.get("sha256")))
        bundle = self.ledger.get("bundle") or {}
        want.append(("bundle (thot)", bundle.get("path"), (bundle.get("file") or {}).get("sha256")))
        rows = []
        for name, rel, h in want:
            if not rel or not h:
                rows.append((name, rel or "?", False, "not in the ledger"))
                continue
            try:
                got = sha256((self.root / rel).read_bytes())
                rows.append((name, rel, got == h, "sha256 ✓" if got == h else f"sha256 {got[:12]}… ≠ ledger {h[:12]}…"))
            except OSError as e:
                rows.append((name, rel, False, f"unreadable: {e.strerror}"))
        return rows

    @property
    def persona_ok(self):
        return any(n == "identity" and ok for n, _, ok, _ in self.rows)

    def ledgered(self, rel: str) -> bool:
        return any(r == rel and ok for _, r, ok, _ in self.rows)

    def file(self, rel):
        try:
            return (self.root / rel).read_bytes()
        except OSError:
            return None


def space_prompt() -> bytes | None:
    """The template's Savante.prompt: cached under STATE, fetched from the Space once."""
    cache = STATE / "Savante.prompt"
    if cache.is_file():
        return cache.read_bytes()
    try:
        with urllib.request.urlopen(SPACE_PROMPT_URL, timeout=10) as r:
            b = r.read()
        STATE.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b)
        return b
    except Exception:
        return None


PROMPTS = ("persona · system_prompt (canon, ledgered)", "sAGI.prompt (canon facet, ledgered)", "Savante.prompt (Space template, not ledgered)")


def system_prompt(canon: Canon, which: str):
    """(text, provenance line) — or (None, why) when the choice cannot be used."""
    if which == PROMPTS[0]:
        if not canon.persona_ok:
            return None, "savante.persona does not verify against the ledger: refusing to speak as Savante"
        sp = canon.persona.get("system_prompt") or ""
        return sp, f"persona.system_prompt · persona sha256 {canon.ledger['artifacts']['identity']['sha256'][:12]}… (ledgered)"
    if which == PROMPTS[1]:
        b = canon.file("sAGI.prompt")
        if b is None or not canon.ledgered("sAGI.prompt"):
            return None, "sAGI.prompt missing or does not verify against the ledger"
        return b.decode("utf-8", "replace"), f"sAGI.prompt · sha256 {sha256(b)[:12]}… (ledgered)"
    b = space_prompt()
    if b is None:
        return None, f"Savante.prompt not cached and {SPACE_PROMPT_URL} unreachable"
    return b.decode("utf-8", "replace"), f"Savante.prompt · sha256 {sha256(b)[:12]}… (Space template; not in the ledger)"


# ── .history ─────────────────────────────────────────────────────────────────────
def history_append(rec: dict):
    STATE.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def history_load():
    """(session id, [[user, answer], …]) of the most recent session, or a new session."""
    try:
        lines = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, ValueError):
        lines = []
    if not lines:
        return uuid.uuid4().hex[:12], []
    sid = lines[-1].get("session")
    return sid, [[r.get("user", ""), r.get("shown", r.get("assistant", ""))] for r in lines if r.get("session") == sid]


# ── the carrier: bankml serve ─────────────────────────────────────────────────────
def serve_status():
    try:
        with urllib.request.urlopen(SERVE + "/bankml", timeout=3) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": f"bankml serve not reachable at {SERVE}: {e}"}


def build_messages(system, turns, question, qwen3):
    msgs = [{"role": "system", "content": system}]
    for u, a in turns[-KEEP_EXCHANGES:]:
        msgs += [{"role": "user", "content": u[:KEEP_CHARS]}, {"role": "assistant", "content": (a or "")[:KEEP_CHARS]}]
    msgs.append({"role": "user", "content": question + (" /no_think" if qwen3 else "")})
    return msgs


def stream(messages, max_tokens, temperature):
    """OpenAI SSE through bankml serve; yields (text, receipt-or-None)."""
    body = {"messages": messages, "stream": True, "max_tokens": int(max_tokens), "temperature": float(temperature),
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(SERVE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    acc, receipt = "", None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    ev = json.loads(line[5:])
                except ValueError:
                    continue
                if "bankml_receipt" in ev:
                    receipt = ev["bankml_receipt"]
                    continue
                for ch in ev.get("choices") or []:
                    acc += (ch.get("delta") or {}).get("content") or ""
                yield acc, None
    except urllib.error.HTTPError as e:
        yield f"bankml serve refused: {e.read().decode('utf-8', 'replace')[:500]}", {"error": True}
        return
    except urllib.error.URLError as e:
        yield (f"Cannot reach bankml serve at `{SERVE}` ({e.reason}).\n\nStart it:\n\n```sh\n"
               "bankml serve .models/Bonsai-8B-Q1_0.gguf --fork FORK.json --upstream http://127.0.0.1:18092\n```"), {"error": True}
        return
    yield acc, receipt or {}


def show_answer(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def receipt_line(rc: dict, text: str) -> str:
    if not rc or rc.get("error"):
        return ""
    bits = [f"{rc.get('prompt_tokens', 0)} prompt + {rc.get('completion_tokens', 0)} completion tokens"]
    if rc.get("wall_ms") is not None:
        bits.append(f"{rc['wall_ms'] / 1000:.1f} s")
    if rc.get("ttft_ms") is not None:
        bits.append(f"first token {rc['ttft_ms'] / 1000:.1f} s")
    ans = rc.get("response_sha256")
    match = ans and ans == sha256(text.encode())
    bits.append(f"answer sha256 {ans[:12]}…{' ✓' if match else ' (≠ received!)'}" if ans else "no answer hash")
    return (f"draft · not a finding — receipt: bankml {rc.get('bankml', '?')} · {rc.get('engine', '?')} · "
            f"model sha256 {str(rc.get('model_sha256', '?'))[:12]}… · " + " · ".join(bits))


# ── the page ─────────────────────────────────────────────────────────────────────
def office_md(canon: Canon) -> str:
    p, card = canon.persona, canon.card
    beliefs = [b.get("belief", "") for b in ((p.get("bdi") or {}).get("beliefs") or [])[:8] if isinstance(b, dict)]
    ok = sum(1 for r in canon.rows if r[2])
    status = (card.get("savante") or {}).get("status", "?")
    lines = [f"### {p.get('name', 'Savante')} — {p.get('kind', '')}",
             f"*{p.get('mantra', '')}*", "", f"**Oath.** {p.get('oath', '')}", "",
             f"**Card:** {card.get('type', '?')} · status `{status}` (DEFER on minting; no mint here) · "
             f"doctrine root `{(canon.ledger.get('doctrine_root') or {}).get('value', '?')[:18]}…` · "
             f"persona digest `{((canon.ledger.get('artifacts') or {}).get('identity') or {}).get('sha256', '?')[:18]}…` "
             "(on-chain slots unwritten)",
             f"**Ledger:** {ok}/{len(canon.rows)} files verify (sha256 against `savante.commitments.json`)", ""]
    lines += [f"- {b}" for b in beliefs]
    return "\n".join(lines)


def integrity_md(canon: Canon) -> str:
    rows = ["| ledger entry | path | check |", "|---|---|---|"]
    rows += [f"| {n} | `{p}` | {'✓' if ok else '✗'} {d} |" for n, p, ok, d in canon.rows]
    return "\n".join(rows)


# ── view mode: watching the testing ──────────────────────────────────────────────
def live_tail(n=60) -> str:
    try:
        lines = LIVE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"(no live log yet at {LIVE})"
    return "\n".join(lines[-n:])


def live_stage() -> str:
    try:
        lines = LIVE.read_text(encoding="utf-8", errors="replace").splitlines()
        age = time.time() - LIVE.stat().st_mtime
    except OSError:
        return "**idle** — no testing log yet"
    heads = [l for l in lines if l.startswith("# bankml")][-1:]
    step = [l for l in lines if l.startswith("## ")]
    running = age < 90 and not any("gate exit" in l or "gate done" in l for l in lines[-3:])
    state = "**running**" if running else f"**idle** (last write {int(age // 60)} min ago)"
    return f"{state} · {heads[0].lstrip('# ') if heads else ''} · current step: `{step[-1][3:] if step else '—'}`"


def results_list():
    d = REPO / "testing" / "results"
    return sorted((p.name for p in d.glob("*.txt")), key=lambda n: [int(x) if x.isdigit() else x for x in re.split(r"[.]", n)], reverse=True) if d.is_dir() else []


def results_read(name):
    if not name or "/" in name:
        return ""
    try:
        return (REPO / "testing" / "results" / name).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return str(e)


_CI = {"t": 0.0, "v": ""}


def ci_status() -> str:
    if time.time() - _CI["t"] < 60:
        return _CI["v"]
    try:
        req = urllib.request.Request(CI_API, headers={"Accept": "application/vnd.github+json", "User-Agent": "bankml-ui"})
        with urllib.request.urlopen(req, timeout=5) as r:
            runs = json.loads(r.read()).get("workflow_runs", [])
        rows = ["| CI run | result | when |", "|---|---|---|"]
        rows += [f"| {x.get('display_title', '')[:70]} | {x.get('conclusion') or x.get('status')} | {x.get('created_at', '')} |" for x in runs]
        _CI.update(t=time.time(), v="\n".join(rows))
    except Exception as e:
        _CI.update(t=time.time(), v=f"CI status unavailable: {e}")
    return _CI["v"]


def machine() -> str:
    try:
        la = open("/proc/loadavg").read().split()[:3]
        mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}
        cpu = open("/proc/cpuinfo").read().split("model name")[1].split(":")[1].split("\n")[0].strip()
        swap_used = (mem["SwapTotal"] - mem["SwapFree"]) / 1024
        warn = " · **swap full: measurements may be disturbed**" if mem["SwapTotal"] and mem["SwapFree"] < mem["SwapTotal"] * 0.05 else ""
        return (f"{cpu} · {os.cpu_count()} threads · load {' '.join(la)} · RAM available {mem['MemAvailable'] / 1048576:.1f} "
                f"of {mem['MemTotal'] / 1048576:.1f} GB · swap used {swap_used:.0f} MB{warn}")
    except Exception as e:
        return f"machine status unavailable: {e}"


def view_tabs(gr, canon: Canon):
    with gr.Tab("Testing (live)"):
        stage = gr.Markdown(live_stage())
        mach = gr.Markdown(machine())
        log = gr.Textbox(value=live_tail(), lines=28, max_lines=28, label=f"live — {LIVE}", interactive=False)
    with gr.Tab("Results"):
        names = results_list()
        pick = gr.Dropdown(names, value=names[0] if names else None, label="release record (testing/results/)")
        body = gr.Textbox(value=results_read(names[0]) if names else "", lines=30, max_lines=40, label="gate record", interactive=False)
        pick.change(results_read, pick, body)
        ci = gr.Markdown(ci_status())
    with gr.Tab("Office"):
        gr.Markdown(office_md(canon))
    with gr.Tab("Integrity"):
        gr.Markdown(integrity_md(canon))
    return stage, mach, log, ci


def build(canon: Canon, mode: str):
    import gradio as gr

    if mode == "view":
        with gr.Blocks(title="bankml · view") as demo:
            gr.Markdown("## bankml · view — watching the testing live\nRead-only. Every speed figure counts only if "
                        "every oracle passed on the same code. Refreshes every 2 s.")
            stage, mach, log, ci = view_tabs(gr, canon)
            demo.load(lambda: (live_stage(), machine(), live_tail(), ci_status()), None, [stage, mach, log, ci], every=2)
        return demo

    sid0, turns0 = history_load()
    avatar = CANON / ((canon.ledger.get("image_candidate") or {}).get("path") or "gfx/Savante3.png")
    with gr.Blocks(title="bankml · Savante") as demo:
        session = gr.State({"id": sid0})
        gr.Markdown("## bankml · Savante — verified low-bit inference on this computer\n"
                    "Every answer is a **draft**, carried by a local model behind bankml's guard and sha256 pin, "
                    "with a receipt. Savante's canon is read-only and checked against its ledger.")
        with gr.Tab("Interaction"):
            with gr.Row():
                with gr.Column(scale=3):
                    chat = gr.Chatbot(value=turns0, height=520, label="Savante (draft)")
                    msg = gr.Textbox(placeholder="Ask Savante — e.g. review: is this model ready to serve?", show_label=False)
                    with gr.Row():
                        send, stop, new = gr.Button("Send", variant="primary"), gr.Button("Stop"), gr.Button("New session")
                with gr.Column(scale=1):
                    if avatar.is_file():
                        gr.Image(value=str(avatar), label="Savante3.png — named, not pinned", height=220, interactive=False)
                    which = gr.Dropdown(list(PROMPTS), value=PROMPTS[0], label=".prompt")
                    prov = gr.Markdown(system_prompt(canon, PROMPTS[0])[1])
                    max_tokens = gr.Slider(16, 1024, value=256, step=16, label="max tokens")
                    temperature = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="temperature")
                    carrier = gr.JSON(value=serve_status(), label="bankml serve")
                    gr.Button("refresh carrier").click(lambda: serve_status(), None, carrier)
            gr.Markdown(f"`.history` → `{HISTORY}` (outside the canon) · session `{sid0}`")

            def add(m, h):
                return "", (h or []) + [[m, None]] if m.strip() else h

            def respond(h, which, max_tokens, temperature, sess):
                if not h or h[-1][1] is not None:
                    yield h
                    return
                system, why = system_prompt(canon, which)
                if system is None:
                    h[-1][1] = f"refused: {why}"
                    yield h
                    return
                st = serve_status()
                qwen3 = "qwen3" in json.dumps(st).lower() or "bonsai" in json.dumps(st).lower()
                msgs = build_messages(system, [t for t in h[:-1] if t[1] is not None], h[-1][0], qwen3)
                t0, text, rc = time.time(), "", {}
                for text, r in stream(msgs, max_tokens, temperature):
                    h[-1][1] = show_answer(text)
                    rc = r if r is not None else rc
                    yield h
                answer = show_answer(text)
                foot = receipt_line(rc, text)
                h[-1][1] = answer + (f"\n\n<sub>{foot}<br>{why}</sub>" if foot else "")
                history_append({"ts": round(t0, 3), "session": sess["id"], "user": h[-1][0], "assistant": answer,
                                "shown": h[-1][1], "prompt": which, "prompt_provenance": why, "receipt": rc})
                yield h

            ev = msg.submit(add, [msg, chat], [msg, chat]).then(respond, [chat, which, max_tokens, temperature, session], chat)
            ev2 = send.click(add, [msg, chat], [msg, chat]).then(respond, [chat, which, max_tokens, temperature, session], chat)
            stop.click(None, None, None, cancels=[ev, ev2])
            which.change(lambda w: system_prompt(canon, w)[1], which, prov)
            new.click(lambda: ([], {"id": uuid.uuid4().hex[:12]}), None, [chat, session])
        stage, mach, log, ci = view_tabs(gr, canon)
        demo.load(lambda: (live_stage(), machine(), live_tail(), ci_status()), None, [stage, mach, log, ci], every=2)
        with gr.Tab("Verifier"):
            out = gr.Textbox(label="bind/savante_verify.py (offline; exit 0 = APPROVE)", lines=14)

            def verify():
                v = CANON / "bind" / "savante_verify.py"
                if not v.is_file():
                    return f"{v} not found"
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                p = subprocess.run([sys.executable, "-B", str(v), str(CANON)], capture_output=True, text=True, env=env, timeout=300)
                return f"exit {p.returncode}\n" + (p.stdout + p.stderr)[-6000:]

            gr.Button("run the verifier").click(verify, None, out)
        with gr.Tab(".history"):
            hist = gr.Code(language="json", label=str(HISTORY))

            def tail():
                try:
                    return "\n".join(HISTORY.read_text(encoding="utf-8").splitlines()[-40:])
                except OSError:
                    return "(empty)"

            gr.Button("show the last 40 lines").click(tail, None, hist)
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=7873)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mode", choices=("interact", "view"), default="interact")
    a = ap.parse_args()
    canon = Canon(CANON)
    bad = [r for r in canon.rows if not r[2]]
    print(f"canon {CANON}: {len(canon.rows) - len(bad)}/{len(canon.rows)} ledger files verify" + (f"; FAILING: {bad}" if bad else ""))
    demo = build(canon, a.mode)
    demo.queue().launch(server_name=a.host, server_port=a.port, show_api=False)


if __name__ == "__main__":
    main()
