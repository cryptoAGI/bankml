#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Record llama.cpp's own tokenization as the oracle for bankml's tokenizer (tokenizer.rs), as ggml_oracle.py
records its kernels. It asks a running llama-server (b11192, the Bonsai/Qwen3 vocabulary; since 0.3.4 also SmolLM2's,
written to .models/oracle-tokenizer-smollm) to tokenize a corpus — every
document in the repository, Savante's canon texts, the chat template's markers, and Unicode edge cases — once with
special tokens parsed and once without, and writes the cases for the Rust test to reproduce, token for token.
usage: python3 testing/tokenizer_oracle.py [HOST:PORT] [OUT]   (default 127.0.0.1:18092 and .models/oracle-tokenizer)"""
import json, os, sys, urllib.request
from pathlib import Path

up = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:18092"
out = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / ".models" / "oracle-tokenizer")
root = Path(__file__).resolve().parents[1]


def tok(text: str, special: bool) -> list:
    req = urllib.request.Request(f"http://{up}/tokenize", data=json.dumps({"content": text, "add_special": False, "parse_special": special}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["tokens"]


corpus = []
for p in sorted([*root.glob("*.md"), *root.glob("docs/*.md"), root / "testing/README.md", root / "upstream/README.md"]):
    t = p.read_text(encoding="utf-8")
    corpus += [t[i:i + 4000] for i in range(0, len(t), 4000)]  # whole documents, in chunks
canon = Path(os.environ.get("SAVANTE_CANON", next((p for p in (Path.home() / "cryptoAGI" / "savante", Path.home() / "savante") if p.exists()), Path.home() / "cryptoAGI" / "savante")))
for name in ("savante.persona", "explanation.md", "MANIFESTO.md", "Savante.md"):
    try:
        t = (canon / name).read_text(encoding="utf-8")
        corpus += [t[i:i + 4000] for i in range(0, len(t), 4000)]
    except OSError:
        pass
edge = [
    "", " ", "  ", "\n", "\r\n", "\n\n\n", " \n ", "\t\tx", "a  b", "a \n b", "x y", "　全角",
    "Hello, world! It's I'm we'll they've you'd she's IT'S DON'T", "'s 't 're 've 'm 'll 'd", "don’t (curly)",
    "123 4567 1,234.56 −7 ٣٤٥ ௧௨ Ⅻ ½", "naïve café Ångström ﬁ Ǆ ǅ", "é (combining) ä vs ä",
    "東京タワーは高い。中文，标点！한국어 텍스트", "العربية نص מימין לשמאל", "ไทยไม่มีช่องว่าง", "Ελληνικά Кириллица",
    "🙂🙃👍🏽👨‍👩‍👧 🇺🇸 ❤️", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝟙𝟚𝟛 𐍈", "tabs\tand\vverticals\fform", "ends with spaces   ", "   starts with spaces",
    "<|im_start|>system\nYou are Savante.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "<think>\n\n</think>\n\n", "<|endoftext|> <|im_start|><|im_end|>x", "a<|im_start|>b",
    "fn main() { println!(\"{}\", x[0]); } // code", "$ cargo test --release -- --ignored", "Σ c·x − Σ x ∈ {−1, 0, +1} ≈ 1.585",
    "URL https://github.com/cryptoAGI/bankml/blob/v0.2.0/README.md?x=1#y", "mixed123abc456 ABC123def", "\\n literal backslash-n",
    "\x00\x01 control", "\x04\x06 bytes \x13\x14\x16\x1d SmolLM2 has no token for", "\U00040000\U00050000 plane 4", "  \n\n  \n", "!!!???...", "---\n| a | b |\n|---|---|", "0x92fe83eb…ae137d0 sha256:284a335a",
]
corpus += edge
# a seeded fuzz set: random strings over many scripts, whitespace kinds, digits, marks, symbols and special markers
import random
rng = random.Random(20260929)
pools = [range(0x20, 0x7F), range(0xA0, 0x250), range(0x370, 0x530), range(0x590, 0x700), range(0x900, 0x980), range(0xE00, 0xE80),
         range(0x1100, 0x1200), range(0x2000, 0x2070), range(0x2150, 0x2190), range(0x2200, 0x2300), range(0x3000, 0x3100),
         range(0x4E00, 0x4F00), range(0xAC00, 0xAD00), range(0xFF00, 0xFFF0), range(0x1D400, 0x1D800), range(0x1F300, 0x1FA00),
         range(0x10330, 0x10350), [0x9, 0xA, 0xD, 0xB, 0xC, 0x20, 0x20, 0x20, 0x85, 0xA0, 0x2028, 0x3000, 0x27, 0x27, 0x301, 0x200D, 0xFE0F]]
specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>", "<tool_call>"]
for _ in range(2000):
    parts = []
    for _ in range(rng.randint(1, 24)):
        r = rng.random()
        if r < 0.04:
            parts.append(rng.choice(specials))
        elif r < 0.12:
            parts.append(rng.choice(["'s", "'LL", "'re", "'d", " the", " 42", "\r\n", "  ", "\n\n", "..."]))
        else:
            pool = rng.choice(pools)
            parts.append("".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 6))))
    corpus.append("".join(parts).replace("\\r\\n", "\r\n").replace("\\n", "\n"))
out.mkdir(parents=True, exist_ok=True)
n = 0
with open(out / "cases.jsonl", "w", encoding="utf-8") as f:
    for t in corpus:
        for sp in (True, False):
            f.write(json.dumps({"text": t, "parse_special": sp, "ids": tok(t, sp)}, ensure_ascii=False) + "\n")
            n += 1
print(f"{n} cases ({len(corpus)} texts × special on/off) from llama-server at {up} → {out / 'cases.jsonl'}")
