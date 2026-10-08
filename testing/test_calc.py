#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Savante's calculator (sAGI/calc.py), offline: exact results, refusals, and the arithmetic found in questions.
run: python3 testing/test_calc.py"""
import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sAGI"))
import calc  # noqa: E402

fails = 0
def check(name, ok):
    global fails
    fails += not ok
    print(("ok   " if ok else "FAIL ") + name)

ev = lambda e: calc.fmt(calc.evaluate(e))
for expr, want in [("2^64 - 1", "18446744073709551615"), ("0.1 + 0.2", "0.3"), ("1/3", "1/3 ≈ 0.333333333333333"),
                   ("3/8", "0.375"), ("-7/4", "-1.75"), ("sqrt(16/9)", "4/3 ≈ 1.33333333333333"), ("8^(1/3)", "2"),
                   ("5!", "120"), ("2(3+4)", "14"), ("1,000 * 3.5", "3500"), ("17 % 5", "2"), ("15% of 80", "12"),
                   ("18% of 64.50", "11.61"), ("2 × 3 ÷ 4 − 1", "0.5"), ("√2", "1.4142135623731"), ("sqrt(2) * pi", "4.44288293815837"),
                   ("log(1000)", "3"), ("gcd(84, 36)", "12"), ("1e3 * 2", "2000"), ("2^-2", "0.25"), ("12*37 + 5 =", "449")]:
    try:
        got = ev(expr)
    except calc.CalcError as e:
        got = f"refused: {e}"
    check(f"{expr} = {want}", got == want)
check("exact type: 0.1 + 0.2 is a Fraction", calc.evaluate("0.1 + 0.2") == Fraction(3, 10))

for expr in ["__import__('os')", "().__class__", "open('x')", "a + 1", "1/0", "2**100000", "factorial(1001)", "9" * 300,
             "1+" * 80 + "1", "sqrt(-1)", "log(0)", "lambda: 1", "[1, 2]", "'a' * 3", "2 if 1 else 3", ""]:
    try:
        calc.evaluate(expr)
        check(f"refused: {expr[:30]!r}", False)
    except calc.CalcError:
        check(f"refused: {expr[:30]!r}", True)

for q, want in [("What is 12*37 + 5?", [("12*37 + 5", "449")]), ("how much is 1,250 × 0.08 in tax", [("1,250 × 0.08", "100")]),
                ("released 2026-10-08, takes 3-5 minutes", []), ("Is sha-256 safe? version 0.4.2", []), ("I have 3 apples", []),
                ("compute (3+4)^2 - 10 / 4", [("(3+4)^2 - 10 / 4", "46.5")]), ("what is 15% of 80", [("15% of 80", "12")]),
                ("x = 4 - 2", [("4 - 2", "2")])]:
    check(f"find: {q!r}", calc.find(q) == want)
check("note: empty without arithmetic", calc.note("hello") == "")
check("note: the exact result rides with the question", "12*37 + 5 = 449" in calc.note("What is 12*37 + 5?"))

print("all ok" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
