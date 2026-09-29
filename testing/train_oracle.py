#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The oracle for bankml's Rust mindXtrain (bankML/train/): mindXtrain's own Python, run on every persona in mindX
and cryptoAGI plus edge cases (author stage: data/scripts.py), and on 3,000 randomized before/after/baseline sets
(score stage: eval/imprint.py, lexical path). Run it with mindXtrain's interpreter:
    ~/mindxtrain/.venv/bin/python testing/train_oracle.py [~/mindxtrain]
Writes .models/oracle-train/script.jsonl and imprint.jsonl."""
import json, random, sys, tempfile
from pathlib import Path

MXT = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "mindxtrain").expanduser()
sys.path.insert(0, str(MXT))
from mindxtrain.data.scripts import Exchange, build_script_rows, derive_training_params, persona_from_dict, write_script_jsonl  # noqa: E402
from mindxtrain.eval import imprint  # noqa: E402

out = Path(__file__).resolve().parents[1] / ".models" / "oracle-train"
out.mkdir(parents=True, exist_ok=True)
home = Path.home()

# ---- author stage: every persona we have, and edge cases
files = sorted((home / "mindX/mindx/godel/mindxtrain/personas").glob("*.json")) + sorted((home / "mindX/mindx/godel/mindxtrain/personas").glob("*.persona"))
for d in ("savante", "jaimla", "luvai", "codephreak", "simplecoder"):
    files += sorted((home / "cryptoAGI" / d).glob("*.persona")) + sorted((home / "cryptoAGI" / d).glob("*.json"))
edge = [
    ("edge: nameless", {"voice": "hi there", "examples": ["a", 3, True, {"x": 1}, None]}),
    ("edge: whitespace prompt", {"name": "W", "system_prompt": " \n\t\x1c "}),
    ("edge: control and quotes", {"name": "Q\"uote", "system": "line1\nline2\t\"q\" \\ \x01\x7f é 字 😀  ", "samples": ["\b\f\r"]}),
    ("edge: title and bio", {"title": "Doc", "bio": "  padded bio  ", "utterances": "one string"}),
    ("edge: falsy name", {"name": "", "persona": "fallback", "system_prompt": "", "description": "from description"}),
    ("edge: numbers as name", {"id": 42, "summary": "s"}),
]
cases = []
rnd = random.Random(20260929)
for f in files:
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(raw, dict):
        continue
    edge.append((f"{f.parent.name}/{f.name}", raw))
for label, raw in edge:
    ex = [{"user": e["user"], "assistant": e["assistant"]} for e in (raw.get("exchanges") or []) if isinstance(e, dict) and isinstance(e.get("user"), str) and isinstance(e.get("assistant"), str)]
    if not ex:
        ex = [{"user": "Who are you?", "assistant": "Myself."}, {"user": "What now?\n", "assistant": "Ship it, \"carefully\"."}]
    for seed in (True, False):
        p = persona_from_dict(raw)
        rows = build_script_rows(p, [Exchange(**e) for e in ex], seed_voice=seed)
        with tempfile.TemporaryDirectory() as t:
            text = write_script_jsonl(rows, Path(t) / "s.jsonl").read_text(encoding="utf-8")
        cases.append({"label": label, "persona_raw": raw, "persona": p.model_dump(), "exchanges": ex, "seed_voice": seed,
                      "jsonl": text, "params": derive_training_params(len(rows))})
(out / "script.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")

# ---- score stage: randomized utterance sets
words = ["I", "am", "Savante", "the", "office", "grades", "don't", "won't", "code", "is", "law", "keys", "ARE", "identity", "42", "2026",
         "İstanbul", "Kelvin", "café", "naïve", "hello", "world", "sovereignty", "verification", "beats", "permission", "—", "!!", "x'y", "a1b2"]


def utter():
    if rnd.random() < 0.05:
        return ""
    return " ".join(rnd.choice(words) for _ in range(rnd.randint(1, 14))) + rnd.choice(["", ".", "?", " …"])


imp = []
for i in range(3000):
    n = rnd.randint(0, 6)
    before = [utter() for _ in range(n)]
    after = [utter() for _ in range(n if rnd.random() < 0.9 else rnd.randint(0, 6))]
    baseline = [utter() for _ in range(rnd.randint(0, 5))]
    r = imprint.score_imprint(imprint.default_inquiries("Savante")[:n], before, after, baseline)
    imp.append({"before": before, "after": after, "baseline": baseline, "report": r.model_dump(exclude={"inquiries", "before", "after"})})
(out / "imprint.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in imp), encoding="utf-8")
print(f"{len(cases)} scripts ({len(edge)} personas and edge cases × seed_voice) and {len(imp)} imprint reports from mindXtrain {MXT} → {out}")
