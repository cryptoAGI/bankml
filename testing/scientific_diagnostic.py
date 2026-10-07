#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""scientific.diagnostic — bankML against llama-server b11192, every number to 18 decimals.

One request, greedy, with `logprobs` and `top_logprobs`, sent to a fresh llama-server b11192 and to a fresh
`bankml serve --native` on the same model (both computing the whole prompt: `cache_prompt: false`). For every
generated token and each of its top candidates the record holds, from each engine:

  - the 32-bit float's bits (`f32`), which is what "identical" means for bankML;
  - its exact value to 18 decimals (`decimal`), rounded half-even from the float's exact binary value, and the same
    as an integer count of 10^-18 (`e18`, the 18-decimal fixed point of an ERC-20 amount such as SCIEN·TIFIC's);
  - the difference between the engines, computed exactly on those values (`delta`).

Sums (the answer's log-likelihood) are exact sums of exact values; the perplexity is computed with 60 significant
digits and then rounded to 18 decimals. Timings are measured with the nanosecond clock and written to 18 decimals:
the digits past the ninth are zero by construction, and the record says so (`resolution_s`), so no digit claims more
than was measured.

    BANKML_GGML_LIB=<b11192 release dir> python3 testing/scientific_diagnostic.py [MODEL_STEM] [max_tokens]
→ <state>/scientific.diagnostic (what the console's Diagnostics tab shows) and .models/scientific/<model>.scientific.diagnostic
Exit 0 when every token's id and every logprob's bits are equal."""
import datetime, decimal, json, os, struct, subprocess, sys, time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sAGI"))
from json_oracle import SAVANTE, wait  # noqa: E402
from logprobs_oracle import post  # noqa: E402
import models  # noqa: E402

root = Path(__file__).resolve().parents[1]
CTX = decimal.Context(prec=60, rounding=decimal.ROUND_HALF_EVEN)
Q18 = Decimal(1).scaleb(-18)
FORMAT = "bankml.scientific.diagnostic/1"


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x: float) -> str:
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", x))[0]


def d18(v: Decimal) -> str:
    """An exact value, rounded half-even to 18 decimals, printed in full (no exponent)."""
    return format(v.quantize(Q18, context=CTX), "f")


def e18(v: Decimal) -> str:
    return str(int((v * (10 ** 18)).to_integral_value(rounding=decimal.ROUND_HALF_EVEN, context=CTX)))


def fixed(x: float) -> dict:
    """A logprob as the engines' 32-bit float: its bits, and its exact value to 18 decimals."""
    exact = Decimal(f32(x))  # a binary float's decimal expansion is finite: this is its exact value
    return {"f32": bits(x), "decimal": d18(exact), "e18": e18(exact)}


def seconds(ns: int) -> str:
    return d18(Decimal(ns).scaleb(-9))


def engine(cmd, base, env=None):
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    wait(base, proc)
    return proc


def ask(base, body):
    t = time.perf_counter_ns()
    code, r = post(base, "/v1/chat/completions", body)
    ns = time.perf_counter_ns() - t
    if code != 200:
        raise SystemExit(f"{base}: HTTP {code}: {json.dumps(r)[:300]}")
    return r, ns


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "Bonsai-1.7B-Q1_0"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    lib = Path(os.environ.get("BANKML_GGML_LIB", ""))
    model = root / ".models" / f"{stem}.gguf"
    fork = Path(os.environ.get("BANKML_FORKS", Path.home() / ".local/share/bankml/forks")) / f"{stem}.gguf.FORK.json"
    body = {"messages": [{"role": "system", "content": SAVANTE}, {"role": "user", "content": "Explain in two sentences why the sky is blue."}],
            "max_tokens": max_tokens, "temperature": 0, "logprobs": True, "top_logprobs": 5, "cache_prompt": False}
    threads = "3"
    ref = engine([str(lib / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", "18407", "-t", threads, "-c", "2048",
                  "-np", "1", "--jinja", "--reasoning", "off", "--no-webui"], "http://127.0.0.1:18407")
    try:
        want, want_ns = ask("http://127.0.0.1:18407", body)
    finally:
        ref.terminate()
        ref.wait(timeout=60)
    bk = engine([str(root / "target" / "release" / "bankml"), "serve", str(model), "--fork", str(fork), "--native", "--listen", "127.0.0.1:18409",
                 "--upstream", "127.0.0.1:18410", "--ctx", "2048"], "http://127.0.0.1:18409",
                env={**os.environ, "BANKML_THREADS": threads, "BANKML_GPU": "off"})
    try:
        got, got_ns = ask("http://127.0.0.1:18409", body)
    finally:
        bk.terminate()
        bk.wait(timeout=60)
    receipt = got.get("bankml_receipt") or {}
    a = got["choices"][0]["logprobs"]["content"]
    b = want["choices"][0]["logprobs"]["content"]
    tokens, ids_equal, bits_equal, worst = [], 0, 0, Decimal(0)
    sum_a, sum_b = Decimal(0), Decimal(0)
    for i, (x, y) in enumerate(zip(a, b)):
        xa, yb = Decimal(f32(x["logprob"])), Decimal(f32(y["logprob"]))
        delta = CTX.subtract(xa, yb)
        sum_a, sum_b = CTX.add(sum_a, xa), CTX.add(sum_b, yb)
        worst = max(worst, abs(delta))
        same_id = (x["id"], x["token"]) == (y["id"], y["token"])
        same_bits = bits(x["logprob"]) == bits(y["logprob"])
        top = []
        for u, v in zip(x["top_logprobs"], y["top_logprobs"]):
            du = CTX.subtract(Decimal(f32(u["logprob"])), Decimal(f32(v["logprob"])))
            worst = max(worst, abs(du))
            same_bits = same_bits and bits(u["logprob"]) == bits(v["logprob"]) and u["id"] == v["id"]
            top.append({"id": u["id"], "token": u["token"], "bankml": fixed(u["logprob"]), "reference": fixed(v["logprob"]),
                        "reference_id": v["id"], "delta": d18(du)})
        ids_equal += same_id
        bits_equal += same_bits
        tokens.append({"i": i, "id": x["id"], "token": x["token"], "reference_id": y["id"], "bankml": fixed(x["logprob"]),
                       "reference": fixed(y["logprob"]), "delta": d18(delta), "bits_equal": same_bits, "top": top})
    n = len(tokens)
    complete = len(a) == len(b)
    identical = complete and ids_equal == n and bits_equal == n and got["choices"][0]["message"]["content"] == want["choices"][0]["message"]["content"]
    ppl = lambda s: d18(CTX.exp(CTX.divide(-s, Decimal(max(n, 1)))))
    timing = lambda r, ns: {"wall_s": seconds(ns), "prompt_tokens_per_s": d18(Decimal(repr(r["timings"]["prompt_per_second"]))),
                            "predicted_tokens_per_s": d18(Decimal(repr(r["timings"]["predicted_per_second"]))),
                            "prompt_ms": d18(Decimal(repr(r["timings"]["prompt_ms"]))), "predicted_ms": d18(Decimal(repr(r["timings"]["predicted_ms"])))}
    rec = {
        "diagnostic": "scientific", "format": FORMAT,
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "precision": {"decimals": 18, "unit": d18(Q18), "rounding": "half-even, from each 32-bit float's exact binary value",
                      "identity": "the f32 bits; the decimals are the same bits written out",
                      "time_resolution_s": seconds(1), "time_note": "measured with a nanosecond clock: digits past the ninth are zero by construction",
                      "rate_note": "engine-reported rates are doubles as each engine printed them, written to 18 decimals"},
        "bankml": receipt.get("bankml"), "reference": "llama-server, llama.cpp b11192",
        "model": {"name": stem, "sha256": receipt.get("model_sha256")}, "threads": int(threads),
        "request": body,
        "answer": {"bankml": got["choices"][0]["message"]["content"], "reference": want["choices"][0]["message"]["content"]},
        "summary": {"tokens": n, "tokens_reference": len(b), "ids_equal": ids_equal, "bits_equal": bits_equal, "max_abs_delta": d18(worst),
                    "log_likelihood": {"bankml": d18(sum_a), "reference": d18(sum_b), "delta": d18(CTX.subtract(sum_a, sum_b))},
                    "perplexity": {"bankml": ppl(sum_a), "reference": ppl(sum_b)}},
        "timing": {"bankml": timing(got, got_ns), "reference": timing(want, want_ns)},
        "verdict": "identical" if identical else "differs",
        "tokens": tokens,
    }
    text = json.dumps(rec, ensure_ascii=False, indent=1) + "\n"
    dest = root / ".models" / "scientific"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{stem}.scientific.diagnostic").write_text(text)
    state = models.LOG.parent
    state.mkdir(parents=True, exist_ok=True)
    (state / "scientific.diagnostic").write_text(text)
    s = rec["summary"]
    print(f"scientific.diagnostic ({stem}): {rec['verdict']} — {s['bits_equal']} of {s['tokens']} tokens bit-equal (top-5 included), "
          f"max |Δ| {s['max_abs_delta']}, log-likelihood {s['log_likelihood']['bankml']} vs {s['log_likelihood']['reference']}; "
          f"→ {state / 'scientific.diagnostic'}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
