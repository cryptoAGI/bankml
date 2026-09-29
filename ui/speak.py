#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Savante's voice in the Savante UI — one voice, hers: pre-rendered statements for the aivatar card.

SAVANTE, AS OF 0.1.3 — A VOICE OF HER OWN, FROM OPEN PARTS (operator, 2026-09-28: "combine from Jaimla as template to
improve from Cori to create Savante as a unique voice that is Savante"; and "add confidence and a slower, more
thoughtful response style"):
  body       Piper `en_GB-cori-high` — trained on public-domain LibriVox recordings (open; Jenny/Jaimla's dataset is a
             custom licence, so it is the TEMPLATE, never the source) — rendered slower and steadier:
             length_scale 1.18 (thoughtful), noise_scale 0.50 and noise_w 0.60 (steady, confident), 0.8 s after
             each sentence
  template   Jaimla's measured f0 (182 Hz, docspeech_voices.json): each clip's own f0 is measured and moved onto it
             with rubberband, formants preserved — lower and grounded, not a slowed tape
  resonance  the house SAVANTE recipe: her own voice an octave below, 80–2600 Hz, heard only in echo (4 taps at 29 ms,
             decay 0.5), at 0.30 — derived from the same clip, so locked to her delivery exactly
  eq         SAVANTE's curve (+1.5 at 200, −2.5 at 500) with presence +6 dB at 3 kHz and +4 dB above 5 kHz
  measured   f0 181 Hz · spectral centroid ≈ 2270 Hz (Cori's own brightness, kept: forcing Jaimla's 2783 needs presence
             that turns sibilant) — written into the manifest per clip
The eSpeak stand-in below remains only as the fallback where Piper or the model is absent.

The voice is SAVANTE as the house defines her. Her full voice (mindX data/config/docspeech_voices.json, id
"savante") is layered piper — Jaimla's body, a male resonance an octave below in echo locked to her delivery,
breath at the edges — and renders on the house render host. On a computer without that host, the house's own
stand-in for her is the cast entry "savante" in the DeltaVerse voice index (vendor/espeak-ng/mindx-voices/
index.json): eSpeak NG `en-gb-x-rp+jaimla` at 168 wpm, "savante is FROM jaimla; her eSpeak stand-in carries it".
That stand-in is what listen.html plays for her in a browser, and it is what this module renders — here, on this
computer, with the same WASM build (vendor/espeak-ng/0.3.5-en) under Node. Every agent in this UI speaks with it.

Her name is said sav-ont: the house pronunciation table (mindX data/config/pronunciation.json, voaice-pronunciation/1)
is applied to what is spoken — case-insensitive, one pass, longest match first — never to what is shown.

Renders are cached under BANKML_UI_STATE/voice/savante/ by a key over (engine, voice, wpm, the variant file's hash,
the table version, the spoken text), encoded to Opus, with a manifest. Nothing is written into the canon.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HOME = Path.home()
ESPEAK = Path(os.environ.get("BANKML_ESPEAK", HOME / "DeltaVerse" / "vendor" / "espeak-ng" / "0.3.5-en"))
VOICES = Path(os.environ.get("BANKML_ESPEAK_VOICES", HOME / "DeltaVerse" / "vendor" / "espeak-ng" / "mindx-voices"))
PRON = Path(os.environ.get("BANKML_PRONUNCIATION", HOME / "mindX" / "data" / "config" / "pronunciation.json"))
PRERENDER = Path(__file__).resolve().parent / "voice" / "prerender.mjs"
# Rendered audio lives OUTSIDE every dot-directory (Gradio refuses to serve a path with one) and outside git.
PIPER = Path(os.environ.get("BANKML_PIPER", HOME / ".local" / "share" / "bankml" / "piper"))
NEURAL = {"id": "savante", "engine": "piper 2023.11.14-2 + ffmpeg (rubberband)", "model": "en_GB-cori-high",
          "model_licence": "public domain (LibriVox) — rhasspy/piper-voices MODEL_CARD", "template": "jaimla (docspeech_voices.json): f0 182 Hz",
          "length_scale": 1.18, "noise_scale": 0.5, "noise_w": 0.6, "sentence_silence": 0.8, "target_f0": 182.0,
          "resonance": {"octave": -1, "band": [80, 2600], "echo_ms": 29, "taps": 4, "decay": 0.5, "gain": 0.30},
          "eq": [[200, 200, 1.5], [500, 240, -2.5], [3050, 2600, 6.0]], "high_shelf": [5000, 4.0], "pad_s": 0.35}
VOICE_DIR = Path(os.environ.get("BANKML_VOICE_DIR", Path(__file__).resolve().parent / "voice" / "cache")).expanduser()
ENGINE = "espeak-ng-wasm 0.3.5 (echogarden, vendor/espeak-ng/0.3.5-en)"
_BUILTIN_TABLE = {"format": "voaice-pronunciation/1", "version": 2, "entries": [
    {"match": "PYTHAIML", "say": "Pith AI M L"}, {"match": "SAVANTE", "say": "Sav ont"}, {"match": "PYTHAI", "say": "Pith AI"}]}


def neural_available() -> bool:
    return (PIPER / "piper" / "piper").is_file() and (PIPER / f"{NEURAL['model']}.onnx").is_file() and bool(shutil.which("ffmpeg"))


def savante_voice() -> dict:
    """Savante's voice as used here: the neural one (Cori body, Jaimla template) when present, else the eSpeak stand-in."""
    if neural_available():
        return {**NEURAL, "voice": NEURAL["model"] + " → savante", "wpm": round(168 / NEURAL["length_scale"]), "standIn": False,
                "why": "a voice of her own: Cori's body (public domain) on Jaimla's pitch, slower and steadier, with SAVANTE's resonance",
                "source": "ui/speak.py NEURAL"}
    return _espeak_voice()


def _espeak_voice() -> dict:
    """The cast entry "savante" from the house voice index; the stand-in for her layered voice."""
    try:
        c = json.loads((VOICES / "index.json").read_text(encoding="utf-8"))["cast"]["savante"]
        return {"id": "savante", "voice": c["voice"], "wpm": int(c["rate"]), "standIn": bool(c.get("standIn")), "why": c.get("why", ""),
                "source": str(VOICES / "index.json") + " cast.savante"}
    except (OSError, KeyError, ValueError):
        return {"id": "savante", "voice": "en-gb-x-rp+jaimla", "wpm": 168, "standIn": True,
                "why": "savante is FROM jaimla; her eSpeak stand-in carries it", "source": "built-in copy of the cast entry"}


def table() -> dict:
    try:
        t = json.loads(PRON.read_text(encoding="utf-8"))
        if t.get("format") == "voaice-pronunciation/1" and isinstance(t.get("entries"), list):
            return t
    except (OSError, ValueError):
        pass
    return _BUILTIN_TABLE


# said, not shown: the token's split name, the runtime's name, and a web address read as "dot"
_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "16": "sixteen", "32": "thirty-two", "64": "sixty-four", "128": "one twenty-eight"}
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}


def _quant(m) -> str:
    """Q1_0 → "Q one zero", Q2_0_g64 → "Q two zero, g sixty-four", Q4_K_M → "Q four K M" (a ggml type, spelled)."""
    out = []
    for part in m.group(0).split("_"):
        mm = re.fullmatch(r"([A-Za-z]*)(\d*)", part)
        if mm and mm.group(2) and mm.group(1).lower() == "g":
            out.append(", g " + _NUM.get(mm.group(2), mm.group(2)))
        elif mm:
            out.append(" ".join(list(mm.group(1).upper()) + ([_NUM.get(mm.group(2), mm.group(2))] if mm.group(2) else [])))
        else:
            out.append(part)
    return " ".join(out).replace(" ,", ",")


def _section(m) -> str:
    return f"section {_NUM.get(str(_ROMAN.get(m.group(1), 0)), m.group(1))} point {_NUM.get(m.group(2), m.group(2))}"


# said, not shown; applied in order, before the underscore rule and the house table. Maths symbols are read only in a
# maths context (between numbers or single-letter variables), so a name such as "Professor / OVERLORD" or "DAIO ·
# savante_sagi" keeps its separator.
LOCAL_SAY = (
    (r"SCIEN·TIFIC", "Sci-en, Tiffic"), (r"(?i)\bbankml\.rs\b", "bank M L dot R S"), (r"(?i)\bbankml\b", "bank M L"),
    (r"\b([a-z0-9-]+)\.(pythai)\.(net)\b", r"\1 dot \2 dot \3"),
    # TECHNICAL.md's references: a parenthesis that only points somewhere is dropped; a section named in a sentence is read
    (r"\s*\((?:see\s+)?(?:§[^()]*|PERFORMANCE\.md[^()]*|`?[a-z0-9_]+\.rs`?(?:,[^()]*)?)\)", ""), (r"\bof\s+§\s*I\.2\b", "described earlier"),
    (r"§\s*([IVX]+)\.(\d+)", _section), (r"\b(?:[a-z0-9_]+)\.rs\b", lambda m: m.group(0)[:-3].replace("_", " ") + " dot R S"),
    (r"\b(?:PT|P|T|I)?Q\d(?:_[A-Za-z0-9]+)+\b", _quant), (r"(?i)\bq8_0\b", "Q eight zero"), (r"\*", ""),
    (r"\+ or −", "plus or minus"), (r"(?<=[\dA-Za-z)])\s\+\s(?=[\d(A-Za-z])", " plus "), (r"(?<=[\w)])\s=\s", " equals "),
    (r"(?<=\d)\s/\s(?=\d)", " over "), (r"log₂\s*3", "log base two of three"), (r"3⁵", "three to the fifth"),
    (r"Σ_\{w=\+1\}", "the sum over plus-one weights of"), (r"Σ_\{w=−1\}", "the sum over minus-one weights of"),
    (r"Σ_\{sᵢ=\+1\}", "the sum over plus signs of"), (r"Σ_\{sᵢ=−1\}", "the sum over minus signs of"),
    (r"Σ", "the sum of "), (r"≈", " about "), (r"≤", " at most "), (r"∈", " in "), (r"(?<=[\d\w)])\s?×\s?(?=[\d\w(])", " times "),
    (r"\{−1, 0, \+1, \+2\}", "minus one, zero, plus one, or plus two"),
    (r"\{\s*−1,\s*0,\s*\+1\s*\}", "minus one, zero, or plus one"), (r"\{\s*−1,\s*\+1\s*\}", "minus one or plus one"),
    (r"−(?=\s?\d)", " minus "), (r"\s−\s", " minus "), (r"(?<![\w])\+(?=\d)", "plus "), (r"ᵢ", " i"),
    (r"(?<=\b[a-z\d])\s·\s(?=[a-z\d]\b)", " times "), (r"\s{2,}", " "))
_RX_CACHE: dict = {}


def say(text: str, t: dict | None = None) -> str:
    """The respelling the synthesiser hears: one pass, case-insensitive, longest match first."""
    t = t or table()
    for m_, s_ in LOCAL_SAY:  # bankml's own words first, while Q1_0 and q1_0.rs still have their underscores
        text = re.sub(m_, s_, text)
    text = re.sub(r"(?<=\w)_(?=\w)", " ", text).strip()  # savante_sagi, APPROVE_WITH_CONDITIONS: said as words, shown as written
    ents = t["entries"]
    if not ents:
        return text
    key = (id(t), t.get("version"), len(ents))
    if key not in _RX_CACHE:  # the table compiled once, not per sentence
        es = sorted(ents, key=lambda e: -len(e["match"]))
        _RX_CACHE[key] = (re.compile("|".join(re.escape(e["match"]) for e in es), re.I), {e["match"].lower(): e["say"] for e in es})
    rx, by = _RX_CACHE[key]
    return rx.sub(lambda m: by[m.group(0).lower()], text)


def _salt(v: dict, t: dict) -> str:
    if v.get("model"):
        return json.dumps([v["engine"], v["model"], {k: v[k] for k in sorted(NEURAL) if k not in ("model_licence",)}, t.get("version")], sort_keys=True)
    return json.dumps([ENGINE, v["voice"], v["wpm"], _variant_sha(v["voice"]), t.get("version")])


def _f0(path: str) -> float | None:
    """Median f0 (autocorrelation over voiced frames), as the house measures a voice; numpy if present."""
    try:
        import wave
        import numpy as np
        w = wave.open(path)
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        f0 = []
        for i in range(0, len(x) - 2048, 512):
            fr = x[i:i + 2048] * np.hanning(2048)
            if np.sqrt((fr ** 2).mean()) < 300:
                continue
            ac = np.correlate(fr, fr, "full")[2047:]
            lo, hi = int(sr / 400), int(sr / 70)
            k = lo + int(np.argmax(ac[lo:hi]))
            if ac[k] > 0.35 * ac[0]:
                f0.append(sr / k)
        return float(np.median(f0)) if f0 else None
    except Exception:  # noqa: BLE001
        return None


def _render_neural(todo: list):
    """Piper (Cori, slower and steadier) → per-clip pitch onto Jaimla's f0 → SAVANTE's resonance and EQ → Opus."""
    import fcntl
    n = NEURAL
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(PIPER / ".render.lock", "w")  # beside Piper: machine-wide, whatever voice directory asked
    fcntl.flock(lock, fcntl.LOCK_EX)  # one Piper at a time on this machine, whoever asked (memory, not speed, is the limit)
    todo = [i for i in todo if not i["file"].is_file()]  # another renderer may have made them while we waited
    try:
        _render_neural_locked(todo, n)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _render_neural_locked(todo: list, n: dict):
    if not todo:
        return
    with tempfile.TemporaryDirectory(prefix="bankml-savante-") as tmp:
        lines = "".join(json.dumps({"text": i["say"], "output_file": f"{tmp}/{i['key']}.wav"}) + "\n" for i in todo)
        p = subprocess.run([str(PIPER / "piper" / "piper"), "--model", str(PIPER / f"{n['model']}.onnx"), "--json-input", "--quiet",
                            "--length_scale", str(n["length_scale"]), "--noise_scale", str(n["noise_scale"]), "--noise_w", str(n["noise_w"]),
                            "--sentence_silence", str(n["sentence_silence"])], input=lines, capture_output=True, text=True, timeout=7200)
        if p.returncode:
            raise RuntimeError("piper failed: " + p.stderr.strip()[-400:])
        r = n["resonance"]
        eq = ",".join(f"equalizer=f={f}:t=h:w={w}:g={g}" for f, w, g in n["eq"])
        taps = "|".join(str(r["echo_ms"] * (k + 1)) for k in range(r["taps"]))
        decays = "|".join(f"{r['decay'] ** (k + 1):.4f}" for k in range(r["taps"]))
        for i in todo:
            src = f"{tmp}/{i['key']}.wav"
            f0 = _f0(src)
            ratio = max(0.70, min(1.0, n["target_f0"] / f0)) if f0 else 0.80
            fc = (f"[0:a]rubberband=pitch={ratio:.4f}:formant=preserved:transients=smooth,asplit=2[b0][r];"
                  f"[b0]highpass=f=70,{eq},highshelf=f={n['high_shelf'][0]}:g={n['high_shelf'][1]}[b];"
                  f"[r]rubberband=pitch=0.5:formant=preserved,highpass=f={r['band'][0]},lowpass=f={r['band'][1]},"
                  f"aecho=0.0:1.0:{taps}:{decays},volume={r['gain']}[rv];"
                  f"[b][rv]amix=inputs=2:weights=1 1:normalize=0,alimiter=limit=0.89:level=false,apad=pad_dur={n['pad_s']}[o]")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex", fc, "-map", "[o]", "-ac", "1",
                            "-c:a", "libopus", "-b:a", "40k", str(i["file"])], check=True, timeout=300)
            dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(i["file"])],
                                 capture_output=True, text=True).stdout.strip()
            i["seconds"] = float(dur or 0)
            i["measured"] = {"body_f0": round(f0, 1) if f0 else None, "pitch_ratio": round(ratio, 4), "target_f0": n["target_f0"]}


def _variant_sha(voice: str) -> str:
    name = voice.split("+", 1)[1] if "+" in voice else ""
    p = VOICES / "voices" / "!v" / name
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if name and p.is_file() else "none"


def available() -> tuple:
    if neural_available():
        return True, "ok"
    for need, what in ((shutil.which("node"), "node"), (shutil.which("ffmpeg"), "ffmpeg"), ((ESPEAK / "espeak-ng.js").is_file(), "the espeak-ng WASM build"),
                       ((VOICES / "voices").is_dir(), "the house voice files"), (PRERENDER.is_file(), "ui/voice/prerender.mjs")):
        if not need:
            return False, f"needs {what}"
    return True, "ok"


def render(texts: list, state: Path | None = None) -> list:
    """[{text, say, key, file, seconds}] for each statement, rendered once and cached; [] with no engine."""
    ok, why = available()
    if not ok or not texts:
        return []
    v, t = savante_voice(), table()
    d = VOICE_DIR / "savante"
    d.mkdir(parents=True, exist_ok=True)
    salt = _salt(v, t)
    items = []
    for text in texts:
        s = say(text, t)
        key = hashlib.sha256((salt + "\x1f" + s).encode()).hexdigest()[:24]
        items.append({"text": text, "say": s, "key": key, "file": d / f"{key}.ogg"})
    todo = [i for i in items if not i["file"].is_file()]
    if todo and v.get("model"):
        _render_neural(todo)
    elif todo:
        with tempfile.TemporaryDirectory(prefix="bankml-voice-") as tmp:
            job = {"espeak": str(ESPEAK), "voices": str(VOICES), "voice": v["voice"], "wpm": v["wpm"], "out": tmp,
                   "items": [{"key": i["key"], "say": i["say"]} for i in todo]}
            p = subprocess.run(["node", str(PRERENDER)], input=json.dumps(job), capture_output=True, text=True, timeout=300)
            if p.returncode:
                raise RuntimeError("prerender failed: " + p.stderr.strip()[-400:])
            res = {r["key"]: r for r in json.loads(p.stdout)["items"]}
            for i in todo:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", res[i["key"]]["file"], "-c:a", "libopus", "-b:a", "32k",
                                "-ac", "1", str(i["file"])], check=True, timeout=60)
                i["seconds"] = res[i["key"]]["seconds"]
    import fcntl
    man_p = d / "manifest.json"
    lock = open(d / ".manifest.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)  # renders may run in parallel (python3 ui/speak.py --shard i/n)
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.is_file() else {}
    for i in items:
        if "seconds" in i:
            man[i["key"]] = {"text": i["text"], "said": i["say"], "seconds": round(i["seconds"], 3),
                             "opus_sha256": hashlib.sha256(i["file"].read_bytes()).hexdigest(), **({"measured": i["measured"]} if i.get("measured") else {})}
        i["seconds"] = man.get(i["key"], {}).get("seconds")
    man["_voice"] = {**v, "engine": v.get("engine") or ENGINE, "variant_sha": _variant_sha(v["voice"]), "pronunciation": {"path": str(PRON), "version": t.get("version")},
                     "note": "Savante's own voice (Cori body, Jaimla template, SAVANTE resonance), rendered on this computer" if v.get("model")
                     else "the house stand-in for SAVANTE's layered voice (docspeech_voices.json id savante), rendered on this computer"}
    _atomic(man_p, json.dumps(man, indent=1, ensure_ascii=False) + "\n")  # readers never see half a manifest
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return items


# ── the introduction a new participant listens to: chapters read from the canon, verbatim ─────────────────────
def speech(md: str) -> list:
    """Markdown → the sentences a listener hears: code blocks, tables, images and HTML dropped; links read as their
    text; emphasis and list markers removed; headings kept as short sentences of their own."""
    out = []
    md = re.sub(r"```.*?```", " ", md, flags=re.S)
    for para in re.split(r"\n\s*\n", md):
        lines = [l for l in para.splitlines() if l.strip() and not l.lstrip().startswith(("|", "<", "![", ">|"))]
        if not lines:
            continue
        t = " ".join(re.sub(r"^([-*+]|\d+\.)\s+", "", l.strip()) for l in lines)  # list markers at line starts only
        t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
        t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
        t = re.sub(r"`([^`]*)`", r"\1", t)
        t = re.sub(r"^#+\s*|\s#+\s", " ", t)
        t = re.sub(r"(?<![\w*])[*_]{1,3}(?=\S)([^*_]+?)(?<=\S)[*_]{1,3}(?![\w*])", r"\1", t)  # emphasis only; savante_sagi keeps its _
        t = re.sub(r"https?://\S+", "", t)
        t = re.sub(r"\s+", " ", t).strip(" >")
        if len(t) < 3:
            continue
        for sent in re.split(r"(?<=[.!?:;])\s+(?=[A-Z0-9\"'(])", t):
            if sent.strip():
                out.append(sent.strip())
    return out


def intro_chapters(canon: Path, persona: dict, card: dict) -> list:
    """[(title, [sentences])] — what Savante reads to a new participant, all of it from her canon."""
    ch = []
    who = [card.get("description") or "", persona.get("mantra") or ""]
    ch.append(("Who I am", [x for d in who for x in speech(d)]))
    ch.append(("My oath", speech(persona.get("oath") or "")))
    bel = [b.get("belief", "") for b in ((persona.get("bdi") or {}).get("beliefs") or []) if isinstance(b, dict)]
    ch.append(("What I believe", [x for b in bel for x in speech(b)]))
    ch.append(("How I work — my system prompt", speech(persona.get("system_prompt") or "")))
    for title, name in (("Why I exist", "explanation.md"), ("The manifesto", "MANIFESTO.md"), ("Savante, in full", "Savante.md")):
        try:
            ch.append((title, speech((canon / name).read_text(encoding="utf-8"))))
        except OSError:
            pass
        if name == "explanation.md":  # the one chapter that is bankml's, not her canon: SCIEN·TIFIC as the measure
            ch.append(("SCIEN·TIFIC: two to the 256, minus one", speech((READINGS / "scientific.md").read_text(encoding="utf-8"))))
    return [(t, s) for t, s in ch if s]


READINGS = Path(__file__).resolve().parent / "voice" / "readings"
REPO = Path(__file__).resolve().parents[1]
# the reading: bankml's thesis and the binary / ternary argument, from TECHNICAL.md by heading, verbatim
READING = (("The thesis", "## Thesis"), ("Binary and ternary weights", "### II.1"), ("The binary choice: Q1_0", "### III.2"),
           ("The ternary kernel: Q2_0_g64", "### III.5"), ("Can a binary computer perform a ternary operation?", "### III.8"))


def _section(md: str, head: str) -> str:
    """The text under the first heading that starts with `head`, up to the next heading of the same or higher level."""
    lv = len(head.split(" ", 1)[0])
    out, on = [], False
    for line in md.splitlines():
        h = re.match(r"(#+) ", line)
        if on and h and len(h.group(1)) <= lv:
            break
        if on:
            out.append(line)
        elif line.startswith(head):
            on = True
    return "\n".join(out)


def reading_chapters(repo: Path = REPO) -> list:
    """[(title, [sentences])]: Savante reads bankml's thesis and TECHNICAL.md's binary/ternary sections aloud."""
    try:
        md = (repo / "docs" / "TECHNICAL.md").read_text(encoding="utf-8")
    except OSError:
        return []
    return [(t, s) for t, h in READING if (s := speech(_section(md, h)))]


# ── export: a chapter set as one Ogg Opus file, complete or not at all ────────────────────────────────────────
EXPORT_DIR = Path(os.environ.get("BANKML_EXPORT_DIR", Path(__file__).resolve().parent / "voice" / "export")).expanduser()
GAP_S = 0.7  # silence between sentences; 2.0 between chapters


def _duration(f) -> float:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)], capture_output=True, text=True, timeout=60)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def _atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def export_state(name: str, chapters: list) -> dict:
    """{"file", "ready", "total", "current"}: current means the file on disk was built from exactly these clips."""
    items = [i for _, s in chapters for i in cached(s)]
    keys = [i["key"] if i else None for i in items]
    sig = hashlib.sha256("|".join(k or "-" for k in keys).encode()).hexdigest()[:16]
    f = EXPORT_DIR / f"{name}.opus"
    meta = EXPORT_DIR / f"{name}.json"
    try:
        cur = f.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("sig") == sig
    except (OSError, ValueError):  # absent or damaged: not current (it is rebuilt), never an error for the caller
        cur = False
    return {"file": f, "ready": sum(1 for k in keys if k), "total": len(keys), "current": cur, "sig": sig, "items": items}


def export(name: str, chapters: list, lead: list = ()) -> dict:
    """Concatenate the rendered clips of `lead` + every chapter into EXPORT_DIR/<name>.opus (Ogg Opus, 40 kbit/s,
    one chapter mark per chapter). Refuses unless every sentence is rendered: an export is complete."""
    chs = ([("Voice examples", list(lead))] if lead else []) + list(chapters)
    st = export_state(name, chs)
    if st["current"]:
        return st
    if st["ready"] < st["total"]:
        raise RuntimeError(f"{name}: {st['ready']}/{st['total']} sentences rendered; an export is complete or it is not made")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    q = lambda p: "file '" + str(p).replace("'", "'\\''") + "'"  # the concat demuxer's quoting
    with tempfile.TemporaryDirectory(prefix=".bankml-export-", dir=EXPORT_DIR) as tmp:  # same filesystem: the rename is atomic
        tmp = Path(tmp)
        for n, secs in (("gap", GAP_S), ("chap", 2.0)):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=mono", "-t", str(secs),
                            "-c:a", "libopus", "-b:a", "40k", str(tmp / f"{n}.ogg")], check=True, timeout=60)
        lines, meta, t, k = [], [";FFMETADATA1"], 0.0, 0
        for ci, (title, sents) in enumerate(chs):
            if ci:
                lines.append(q(tmp / "chap.ogg"))
                t += 2.0
            start = t
            for j, _ in enumerate(sents):
                i = st["items"][k]
                k += 1
                if j:
                    lines.append(q(tmp / "gap.ogg"))
                    t += GAP_S
                lines.append(q(i["file"]))
                t += i["seconds"] if i["seconds"] is not None else _duration(i["file"])  # a clip missing from the manifest is measured
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(start * 1000)}", f"END={int(t * 1000)}", "title=" + title.replace("=", "\\=")]
        (tmp / "list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (tmp / "meta.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
        out = tmp / "out.opus"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"), "-i", str(tmp / "meta.txt"),
                        "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1", "-c:a", "libopus", "-b:a", "40k", "-ac", "1",
                        "-metadata", f"title=Savante: {name}", "-metadata", "artist=Savante (bankml; Cori body, Jaimla template)",
                        "-f", "ogg", str(out)], check=True, timeout=1800)
        os.replace(out, st["file"])
    _atomic(EXPORT_DIR / f"{name}.json", json.dumps({"sig": st["sig"], "sentences": st["total"], "chapters": [c for c, _ in chs],
                                                        "seconds": round(t, 1), "sha256": hashlib.sha256(st["file"].read_bytes()).hexdigest()},
                                                       indent=1, ensure_ascii=False) + "\n")
    return export_state(name, chs)


EXPORTS = {"Savante": "the introduction: her voice examples, then every chapter", "Savante-reading": "the thesis, and binary and ternary, from TECHNICAL.md"}


def export_sets(canon: Path, persona: dict, card: dict) -> dict:
    """name -> (lead, chapters) for the two exports."""
    lead = [x for x in persona.get("voice_examples") or [] if isinstance(x, str) and x.strip()]
    return {"Savante": (lead, intro_chapters(canon, persona, card)), "Savante-reading": ([], reading_chapters())}


INTRO = {"state": "idle", "done": 0, "total": 0, "error": None}


def render_intro_async(canon: Path, persona: dict, card: dict):
    """Render the introduction in the background (once; cached thereafter)."""
    import threading
    if INTRO["state"] == "running" or os.environ.get("BANKML_VOICE_ASYNC", "1") == "0":
        return
    chapters = intro_chapters(canon, persona, card)
    INTRO.update(state="running", done=0, total=sum(len(s) for _, s in chapters + reading_chapters()), error=None)

    def run():
        try:
            for _, sents in chapters + reading_chapters():
                for k in range(0, len(sents), 25):
                    render(sents[k:k + 25])
                    INTRO["done"] += len(sents[k:k + 25])
            for name, (lead, chs) in export_sets(canon, persona, card).items():
                export(name, chs, lead)
            INTRO["state"] = "done"
        except Exception as e:  # noqa: BLE001
            INTRO.update(state="error", error=str(e))
    threading.Thread(target=run, daemon=True).start()


_ASYNC = set()


def render_async(texts: list):
    """render() in a background thread, once per set of texts at a time (the page never waits on a voice)."""
    import threading
    if os.environ.get("BANKML_VOICE_ASYNC", "1") == "0":
        return
    key = hashlib.sha256("\x1e".join(texts).encode()).hexdigest()
    if key in _ASYNC:
        return
    _ASYNC.add(key)

    def run():
        try:
            for k in range(0, len(texts), 10):
                render(texts[k:k + 10])
        finally:
            _ASYNC.discard(key)
    threading.Thread(target=run, daemon=True).start()


_MAN = {"mtime": None, "data": {}}


def _manifest() -> dict:
    """The voice manifest, re-read only when it changes on disk (view mode polls every few seconds); {} if damaged."""
    p = VOICE_DIR / "savante" / "manifest.json"
    try:
        mt = p.stat().st_mtime_ns
        if mt != _MAN["mtime"]:
            _MAN.update(mtime=mt, data=json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}
    return _MAN["data"]


def cached(texts: list) -> list:
    """render() without rendering: the items whose audio already exists (None where it does not)."""
    v, t = savante_voice(), table()
    salt = _salt(v, t)
    man = _manifest()
    out = []
    for text in texts:
        key = hashlib.sha256((salt + "\x1f" + say(text, t)).encode()).hexdigest()[:24]
        f = VOICE_DIR / "savante" / f"{key}.ogg"
        out.append({"text": text, "key": key, "file": f, "seconds": man.get(key, {}).get("seconds")} if f.is_file() else None)
    return out


if __name__ == "__main__":  # pre-render Savante's introduction and voice examples from the command line
    import sys
    c = Path(os.environ.get("SAVANTE_CANON", HOME / "savante"))
    per = json.loads((c / "savante.persona").read_text(encoding="utf-8"))
    crd = json.loads((c / "savante.agentcard.json").read_text(encoding="utf-8"))
    chs = intro_chapters(c, per, crd)
    # listening order: her voice examples first (short), then the introduction chapter by chapter, then the reading
    allt = [x for x in per.get("voice_examples") or [] if isinstance(x, str)] + [x for _, s in chs + reading_chapters() for x in s]
    if "--shard" in sys.argv:  # python3 ui/speak.py --shard 1/2 : every n-th statement, for parallel renders
        k, n = map(int, sys.argv[sys.argv.index("--shard") + 1].split("/"))
        allt = allt[k - 1::n]
    print(f"{len(chs)} chapters, {len(allt)} statements, {sum(len(x.split()) for x in allt)} words", flush=True)
    for k in range(0, len(allt), 10):  # small batches: each finished batch is playable at once
        render(allt[k:k + 10])
        print(f"  {min(k + 10, len(allt))}/{len(allt)}", flush=True)
    items = cached(allt)
    print(f"cached {sum(1 for i in items if i)} of {len(allt)} · {sum(i['seconds'] or 0 for i in items if i) / 60:.1f} min · {VOICE_DIR}")
    if "--shard" not in sys.argv:  # complete → one file per set: ui/voice/export/Savante.opus, Savante-reading.opus
        for name, (lead, cs) in export_sets(c, per, crd).items():
            try:
                e = export(name, cs, lead)
                print(f"export {e['file']} · {e['total']} sentences · {e['file'].stat().st_size / 1e6:.1f} MB")
            except RuntimeError as err:
                print("export skipped:", err)
    if "--prune" in sys.argv:  # drop clips no current text uses (older voices, edited canon): the committed cache stays lean
        keep = {i["key"] for i in cached(allt) if i}
        d = VOICE_DIR / "savante"
        gone = [f for f in d.glob("*.ogg") if f.stem not in keep]
        for f in gone:
            f.unlink()
        man_p = d / "manifest.json"
        man = json.loads(man_p.read_text(encoding="utf-8"))
        man_p.write_text(json.dumps({k: v for k, v in man.items() if k.startswith("_") or k in keep}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"pruned {len(gone)} clips no current text uses")
