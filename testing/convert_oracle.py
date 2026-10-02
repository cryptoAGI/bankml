#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The oracle of `bankml convert` (O5): llama.cpp b11192's own converter on the same safetensors directory.

    convert_oracle.py --record DIR OUT.gguf      llama.cpp's `convert_hf_to_gguf.py DIR --outtype f16 --outfile OUT.gguf`
                                                 (needs the b11192 source tree in $BANKML_LLAMA_SRC and its Python
                                                 requirements: torch CPU, transformers, numpy, safetensors)
    convert_oracle.py --bankml DIR ORACLE.gguf   `bankml convert DIR` and the sha256 of both files: identical or not
    convert_oracle.py --compare A.gguf B.gguf    where two GGUFs differ: header, each key (order, type, value), each
                                                 tensor (name, order, type, shape, offset, sha256 of its data)
    convert_oracle.py --chkhsh DIR               llama.cpp's pre-tokenizer hash of DIR's tokenizer (its get_vocab_base_pre
                                                 rule, through transformers) and bankml's vocabulary+merges sha256: the
                                                 evidence a KNOWN_TOKENIZERS entry in convert.rs needs
    convert_oracle.py --names OUT.jsonl          gguf-py's name heuristics on a corpus of model ids, for the Rust test
                                                 oracle_name_heuristics (BANKML_NAMES_ORACLE=OUT.jsonl)

The directory's name matters: llama.cpp names the model from it (general.name, basename, finetune, size label), so
record and compare the same directory. Byte identity of the two files is the bar; --compare explains a miss.
"""
import ast, hashlib, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = Path(os.environ.get("BANKML_LLAMA_SRC", "upstream/llama.cpp"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gguf_mod():
    sys.path.insert(0, str(SRC / "gguf-py"))
    import gguf  # noqa: E402
    return gguf


def record(d, out):
    conv = SRC / "convert_hf_to_gguf.py"
    if not conv.is_file():
        sys.exit(f"no {conv}: set BANKML_LLAMA_SRC to a llama.cpp b11192 source tree")
    subprocess.run([sys.executable, str(conv), d, "--outtype", "f16", "--outfile", out], check=True)
    print(f"{sha256(out)}  {out}  (llama.cpp convert_hf_to_gguf.py --outtype f16)")


def bankml(d, oracle):
    out = str(Path(oracle).with_suffix(".bankml.gguf"))
    exe = os.environ.get("BANKML", str(ROOT / "target/release/bankml"))
    subprocess.run([exe, "convert", d, "-o", out], check=True)
    a, b = sha256(out), sha256(oracle)
    print(f"bankml    {a}\nllama.cpp {b}\n{'IDENTICAL' if a == b else 'DIFFERENT (run --compare)'}")
    if a == b:
        os.remove(out)
    sys.exit(0 if a == b else 1)


def compare(a, b):
    g = gguf_mod()
    ra, rb = g.GGUFReader(a), g.GGUFReader(b)
    def kv(r):
        out = []
        for f in r.fields.values():
            v = [bytes(f.parts[i]).hex() for i in f.data]
            out.append((f.name, [int(t) for t in f.types], hashlib.sha256("".join(v).encode()).hexdigest()[:16]))
        return out
    ka, kb = kv(ra), kv(rb)
    bad = 0
    for i in range(max(len(ka), len(kb))):
        x = ka[i] if i < len(ka) else None
        y = kb[i] if i < len(kb) else None
        if x != y:
            bad += 1
            print(f"key #{i}: {x}  !=  {y}")
    for i in range(max(len(ra.tensors), len(rb.tensors))):
        x = ra.tensors[i] if i < len(ra.tensors) else None
        y = rb.tensors[i] if i < len(rb.tensors) else None
        d = lambda t: None if t is None else (t.name, int(t.tensor_type), [int(s) for s in t.shape], int(t.data_offset), hashlib.sha256(bytes(t.data)).hexdigest()[:16])
        if d(x) != d(y):
            bad += 1
            print(f"tensor #{i}: {d(x)}  !=  {d(y)}")
    print(f"{len(ka)} / {len(kb)} keys, {len(ra.tensors)} / {len(rb.tensors)} tensors, {bad} differences")
    sys.exit(1 if bad else 0)


def chkhsh(d):
    src = (SRC / "conversion" / "base.py").read_text()
    chktxt = ast.literal_eval(re.search(r"\n        chktxt = (.*)\n", src).group(1))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(d)
    h = hashlib.sha256(str(tok.encode(chktxt)).encode()).hexdigest()
    names = re.findall(r'if chkhsh == "([0-9a-f]{64})":\n\s+# ref: (\S+)\n\s+res = "([^"]+)"', src)
    hit = [n for n in names if n[0] == h]
    t = json.load(open(Path(d) / "tokenizer.json"))
    m = t["model"]
    inv = {v: k for k, v in m["vocab"].items()}
    for a in t.get("added_tokens", []):
        inv[a["id"]] = a["content"]
    n = json.load(open(Path(d) / "config.json")).get("vocab_size", 32000)
    vm = hashlib.sha256()
    for i in range(n):
        vm.update((inv.get(i, f"[PAD{i}]")).encode() + b"\n")
    vm.update(b"\n")
    for x in m["merges"]:
        x = x if isinstance(x, str) else " ".join("".join(chr(ord(c) + 256) if c == " " else c for c in p) for p in x)
        vm.update((x + "\n").encode())
    print(json.dumps({"dir": d, "chkhsh": h, "llama.cpp pre": hit[0][2] if hit else None, "ref": hit[0][1] if hit else None,
                      "vocab_merges_sha256": vm.hexdigest(), "pre_tokenizer": t.get("pre_tokenizer")}, indent=1))


NAMES = """smollm2-135m-instruct mindx-gen39 Mixtral-8x7B-Instruct-v0.1 meta-llama/Meta-Llama-3-8B bigscience/bloom-7b1-petals
Qwen/Qwen2.5-0.5B-Instruct ./my-model ../x/model-7B HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM-135M
prism-ml/Bonsai-8B-gguf Qwen/Qwen3-8B Qwen/Qwen3-0.6B-Base microsoft/Phi-3-mini-4k-instruct google/gemma-2-2b-it
mistralai/Mistral-7B-Instruct-v0.3 deepseek-ai/DeepSeek-V2-Lite-Chat tiiuae/falcon-7b EleutherAI/pythia-1.4b-deduped
TheBloke/Llama-2-13B-chat-GPTQ stabilityai/stablelm-2-1_6b openai-community/gpt2-xl allenai/OLMo-7B-0724-Instruct-hf
ibm-granite/granite-3.0-2b-instruct unsloth/Llama-3.2-1B-Instruct mistralai/Mixtral-8x22B-v0.1 Qwen/Qwen1.5-MoE-A2.7B
model-32k-chat merged checkpoint gen39 mindx-gen40-v2 mindX-Gen41-135M-lora Llama-3-70B-Instruct-AWQ q4_k_m-model
some-model-F16 Model-bf16-v1.2.3 tiny-llama-1.1b-chat-v1.0 x-small-mini-medium iter12-model A3B-thing model-7b1""".split()


def names(out):
    g = gguf_mod()
    from gguf.metadata import Metadata
    with open(out, "w") as f:
        for i in NAMES:
            for total in (0, 134_515_008, 8_190_000_000, -2_000_000):
                parts = Metadata.get_model_id_components(i, total)
                f.write(json.dumps({"id": i, "total": total, "parts": list(parts), "title": Metadata.id_to_title(i),
                                    "size": g.size_label(abs(total), abs(total), 0, 0) if total else None}) + "\n")
    print(f"{len(NAMES) * 4} cases → {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "--record":
        record(a[1], a[2])
    elif len(a) == 3 and a[0] == "--bankml":
        bankml(a[1], a[2])
    elif len(a) == 3 and a[0] == "--compare":
        compare(a[1], a[2])
    elif len(a) == 2 and a[0] == "--chkhsh":
        chkhsh(a[1])
    elif len(a) == 2 and a[0] == "--names":
        names(a[1])
    else:
        sys.exit(__doc__)
