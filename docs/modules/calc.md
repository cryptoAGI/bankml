# `sAGI/calc.py` — Savante's calculator: exact arithmetic

## Summary

A 1-bit model reads numbers as tokens and guesses arithmetic. The calculator computes it instead, exactly, and Savante
uses it two ways: a **Calculator** panel beside the conversation (a keypad, a terminal-green display, a tape of the last
eight results), and **inside every question**: arithmetic in the question is found, computed, and handed to the model
with the question as exact results it must use; the answer shows them under it (`🧮 … — exact, sAGI/calc.py`). A
message that starts with `=` is answered by the calculator alone: no model, no tokens.

## Technical usage

```sh
python3 sAGI/calc.py "2^64 - 1" "sqrt(2) * pi" "18% of 64.50"   # 18446744073709551615 · 4.44288293815837 · 11.61
```

| function | what |
|---|---|
| `evaluate(expr)` | the value (`int`, `Fraction` or `float`), or `CalcError` |
| `fmt(x)` | integers whole, terminating decimals exact (`0.375`), other ratios `p/q ≈ decimal`, floats to 15 significant digits |
| `find(text, limit=4)` | `[(expression as written, result)]` for the arithmetic in a question |
| `note(text)` | what the model is told: the exact results, appended to the question (empty when there is none) |

Accepted: numbers (`1,000`, `0.1`, `1e3`), `+ - * / // % **` and `^ × ÷ −`, parentheses and implicit `×` (`2(3+4)`,
`2π`), `√`, `n!`, `15% of 80`, the constants `pi e tau`, and `sqrt cbrt abs floor ceil round ln log log10 log2 exp sin
cos tan asin acos atan factorial gcd lcm min max`.

## Design notes

- **Never `eval`.** `ast.parse(mode="eval")` and a walker that accepts only the nodes above; any other name, attribute,
  call or literal is refused. Bounded: 240 characters, 120 nodes, exponents ±10,000, factorial ≤ 1,000, results ≤
  4,096 bits, so no question can make the UI compute for long.
- **Exact by default.** Integers stay integers and typed decimals are read as the decimal written (`0.1 + 0.2` is
  `0.3`), as `Fraction`s; perfect roots are exact (`sqrt(16/9)` is `4/3`, `8^(1/3)` is `2`). Only irrational functions
  answer in floats.
- **Found, not guessed at.** `find` takes a span only when it has an operator between operands and parses. A bare hyphen
  is a range or an identifier, not a subtraction (`3-5 days`, `sha-256`, `2026-10-08` are left alone); `4 - 2` with
  spaces is a subtraction.
- **The prompt stays the model's.** The calculator's line under an answer is for the reader: `savante.strip_calc`
  removes it before earlier turns are sent back, and `.history` records the results under `calculator`.

Tests: `python3 testing/test_calc.py`.
