#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The grammar oracle (0.3.3): llama.cpp b11192's own grammar sampler, through libllama's public API
(`testing/grammar_oracle.cpp`), on the pinned model's vocabulary. For every grammar × input string × tokenization
(the tokenizer's own, and one token per byte, which splits UTF-8 characters across tokens), it records the mask over
the whole vocabulary (151,669 tokens) before each token and after the last — how many tokens are allowed and an
FNV-1a hash of which — and the token at which llama.cpp throws, if it does. `oracle_grammar_masks` replays every
case through bankML's grammar.rs and requires every mask and every rejection to be the same. It also records every
token's piece and the end-of-generation set (`pieces`), which bankML must reproduce for all 151,669 tokens.

The grammars: the one llama-server b11192 builds for `response_format: json_object` on the pinned template (read
from a running llama-server's `generation_settings.grammar`, with the generation prompt accepted first, as the
server does), every grammar in llama.cpp's `grammars/` directory, and a few written here for what those do not reach
(token terminals `<…>`/`!<…>`/`<[id]>`, `.`, `{m,n}` edges, comments, CRLF, astral ranges).

    LLAMA_SRC=<llama.cpp checkout at tag b11192> BANKML_GGML_LIB=<b11192 release dir> \\
        python3 testing/grammar_oracle.py [--server http://127.0.0.1:18301]
(git clone --depth 1 --branch b11192 https://github.com/ggml-org/llama.cpp; any llama-server of that release with
`--jinja --reasoning off` on a Bonsai/Qwen3 model gives the JSON grammar.) Writes .models/oracle-grammar/."""
import json, os, subprocess, sys, urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / ".models" / "oracle-grammar"
src = Path(os.environ.get("LLAMA_SRC", ""))
lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
model = root / ".models" / "Bonsai-8B-Q1_0.gguf"
server = sys.argv[sys.argv.index("--server") + 1] if "--server" in sys.argv else "http://127.0.0.1:18301"
if not (src / "include" / "llama.h").exists() or not (lib / "libllama.so").exists():
    sys.exit("set LLAMA_SRC (llama.cpp source at tag b11192) and BANKML_GGML_LIB (the b11192 release dir)")
out.mkdir(parents=True, exist_ok=True)
exe = out / "grammar_oracle"
subprocess.run(["g++", "-O2", "-std=c++17", f"-I{src}/include", f"-I{src}/ggml/include", str(root / "testing" / "grammar_oracle.cpp"),
                f"-L{lib}", "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{lib}", "-o", str(exe)], check=True)
p = subprocess.Popen([str(exe), str(model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)


def cmd(line, until=None):
    p.stdin.write(line + "\n")
    p.stdin.flush()
    lines = []
    while True:
        r = p.stdout.readline()
        if not r:
            sys.exit("the harness exited")
        r = r.rstrip("\n")
        if until is None:
            return r
        lines.append(r)
        if r == until:
            return lines


hx = lambda s: s.encode().hex() if isinstance(s, str) else s.hex()

# every token's piece and end flag
p.stdin.write("pieces\n")
p.stdin.flush()
pieces = [p.stdout.readline().rstrip("\n") for _ in range(151669)]
(out / "pieces-Bonsai-8B-Q1_0.txt").write_text("\n".join(pieces) + "\n")
byte_tok = {}
for line in pieces:
    i, h, _ = line.split(" ")
    b = bytes.fromhex(h)
    if len(b) == 1 and b[0] not in byte_tok:
        byte_tok[b[0]] = int(i)
assert len(byte_tok) == 256, len(byte_tok)


def tokenize(s):
    r = cmd("tokenize " + hx(s))
    return [int(x) for x in r.split()] if r else []


with urllib.request.urlopen(urllib.request.Request(server + "/v1/chat/completions", json.dumps(
        {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}, "max_tokens": 1, "temperature": 0}).encode(),
        {"Content-Type": "application/json"}), timeout=600) as r:
    v = json.load(r)["__verbose"]["generation_settings"]
chat_json, gen_prompt = v["grammar"], v["generation_prompt"]
grammars = {"chat-json-object": chat_json}
for f in sorted((src / "grammars").glob("*.gbnf")):
    grammars[f.stem] = f.read_text()
grammars.update({
    "tokens": 'root ::= <think> [a-z]+ !<|im_end|> tail | <[151644]> "y" .\ntail ::= "!" | <|im_end|>\n',
    "edges": '# comment\r\nroot ::= ( "a"{0,3} [^\\x00-\\x2F]? | . "z"{2} | "q"{1,} [\\U0001F600-\\U0001F64F] ) "!" # end\r\n',
    "utf8": 'root ::= ( [\\u00e0-\\u00ff] | [\\u4e00-\\u9fff] | [\\U0001F300-\\U0001FAFF] | "ü" )+ "."\n',
})

json_inputs = ['{"a": 1}', '{\n  "name": "Ada",\n  "age": 36\n}', '{"k": [1, 2.5, -3e10, true, null, {"x": "y"}]}',
               '{"esc": "q\\"uote \\\\ \\u00e9 \\n"}', '{"unicode": "héllo — 中文 😀"}', '{"deep": {"a": {"b": {"c": [[], [{}]]}}}}',
               '{}', '{ }', '{' + ' ' * 21 + '"a": 1}', '{\n\n\n"a": 1}', '{"a":1,}', "{'a': 1}", '{"n": 01}', '{"n": 12345678901234567}',
               '{"a": 1} trailing', 'Sure! Here is the JSON: {"a": 1}', '```json\n{"a": 1}\n```', '```json\n{"a": 1}', '``` {"a": 1}',
               '[1, 2, 3]', '"just a string"', '{"a": "unterminated', '{"tab": "a\tb"}', '{"ctrl": "\x01"}', '\n\n  {"lead": 1}  \n',
               '{"a": tru}', '{"a": 1}\n```\nmore', '{"num": -0.5e-3, "z": 0}', '{"big": 1.12345678901234567}']
inputs = {"chat-json-object": json_inputs, "json": json_inputs, "json_arr": ['[{"a": 1}, {"b": [2]}]', '[]', '{"a": 1}', '[1,]'],
          "arithmetic": ["1+2*3", "(4 - 2) / 7\n", "1++2", "  9"], "c": ["int main(int a) { return a; }", "void f() {}", "int x = ;"],
          "chess": ["1. e4 e5 2. Nf3 Nc6 ", "1. Ke9"], "english": ["Hello, world.", "Ünïcode."],
          "japanese": ["こんにちは 世界", "カタカナ、テスト", "hello"], "list": ["- one\n- two\n", "* x\n", "one\n"],
          "tokens": ["<think>abc!", "<think>abc<|im_end|>", "<think>ab<|im_end|><|im_end|>", "<|im_start|>yé", "<think>!"],
          "edges": ["aa!", "aaa!", "aaaa!", "ab!", "ézz!", "qqq\U0001F600!", "q\U0001F680!", "!"],
          "utf8": ["éè.", "中文.", "\U0001F600\U0001F937.", "ü.", "a.", "é\U0001F300一."]}
cases = []
for name, text in grammars.items():
    ok = cmd("grammar " + hx(text))
    assert ok == "ok", (name, ok)
    prefill = tokenize(gen_prompt) if name == "chat-json-object" else []
    for s in inputs.get(name, []):
        for how, toks in (("tokenizer", tokenize(s)), ("bytes", [byte_tok[b] for b in s.encode()])):
            cmd("grammar " + hx(text))
            if prefill:
                cmd("prefill " + " ".join(map(str, prefill)), until="done")
            res = cmd("run " + " ".join(map(str, toks)), until="done")
            # [allowed, hash as 16 hex digits]: a JSON number would lose the hash's low bits
            masks = [[int(r.split()[1]), "%016x" % int(r.split()[2])] for r in res if r.startswith("m ")]
            rejected = len(masks) - 1 if "x" in res else None
            cases.append({"grammar": name, "input": s, "how": how, "prefill": prefill, "tokens": toks, "masks": masks, "rejected_at": rejected})
    print(f"{name}: {sum(1 for c in cases if c['grammar'] == name)} runs, {sum(len(c['masks']) for c in cases if c['grammar'] == name)} masks")
p.stdin.close()
p.wait()
(out / "grammars.json").write_text(json.dumps(grammars, indent=1) + "\n")
(out / "masks-Bonsai-8B-Q1_0.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
print(f"{len(cases)} runs, {sum(len(c['masks']) for c in cases)} whole-vocabulary masks, {sum(c['rejected_at'] is not None for c in cases)} rejections "
      f"from llama.cpp b11192's grammar sampler → {out}")
