#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Savante's calculator: exact arithmetic for the answers a 1-bit model should not guess.

Never `eval`: the expression is parsed with `ast` and only numbers, + - * / // % ** (or ^), unary signs, the constants
pi, e, tau and a fixed list of functions are walked. Integers and decimals stay exact (`Fraction`: 0.1 + 0.2 is 0.3);
irrational functions answer in floats to 15 significant digits. Bounded: 240 characters, 120 nodes, exponents and
factorials capped, results of at most 4,096 bits. Details: docs/modules/calc.md

    python3 sAGI/calc.py "2^64 - 1"       → 18446744073709551615
    python3 sAGI/calc.py "sqrt(2) * pi"   → 4.44288293815837
"""
from __future__ import annotations

import ast
import math
import re
import sys
from fractions import Fraction

MAX_CHARS, MAX_NODES, MAX_BITS, MAX_EXP, MAX_FACT = 240, 120, 4096, 10_000, 1_000


class CalcError(ValueError):
    """An expression the calculator refuses: not arithmetic, too large, or undefined (division by zero)."""


Num = int | Fraction | float


def _real(x: Num) -> float:
    return float(x)


def _exact_root(x: Num, n: int) -> Num:
    """The exact n-th root of a perfect power (an int or a ratio of them), else a float."""
    if isinstance(x, (int, Fraction)) and x >= 0:
        q = Fraction(x)
        rn, rd = round(q.numerator ** (1 / n)), round(q.denominator ** (1 / n))
        for a in (rn - 1, rn, rn + 1):
            for b in (rd - 1, rd, rd + 1):
                if a >= 0 and b > 0 and a ** n == q.numerator and b ** n == q.denominator:
                    return _norm(Fraction(a, b))
    if isinstance(x, (int, Fraction)) and x < 0 and n % 2:
        r = _exact_root(-x, n)
        return -r
    if _real(x) < 0:
        raise CalcError("the square root of a negative number is not real")
    return _real(x) ** (1 / n)


def _int_arg(name: str, x: Num) -> int:
    if isinstance(x, float) and not x.is_integer():
        raise CalcError(f"{name} takes whole numbers")
    if isinstance(x, Fraction) and x.denominator != 1:
        raise CalcError(f"{name} takes whole numbers")
    return int(x)


def _factorial(x: Num) -> int:
    n = _int_arg("factorial", x)
    if n < 0 or n > MAX_FACT:
        raise CalcError(f"factorial takes 0 to {MAX_FACT}")
    return math.factorial(n)


def _log(x: Num, base: Num | None = None) -> float:
    if _real(x) <= 0 or (base is not None and (_real(base) <= 0 or _real(base) == 1)):
        raise CalcError("a logarithm needs a positive number (and a positive base other than 1)")
    return math.log(_real(x)) if base is None else math.log(_real(x), _real(base))


def _floatfn(f):
    def g(x: Num) -> float:
        try:
            return f(_real(x))
        except (ValueError, OverflowError) as e:
            raise CalcError(f"{f.__name__}: {e}") from None
    g.__name__ = f.__name__
    return g


FUNCS = {
    "sqrt": lambda x: _exact_root(x, 2), "cbrt": lambda x: _exact_root(x, 3),
    "abs": abs, "floor": lambda x: math.floor(x), "ceil": lambda x: math.ceil(x),
    "round": lambda x, d=0: _norm(round(Fraction(x), _int_arg("round", d))) if not isinstance(x, float) else round(x, _int_arg("round", d)),
    "ln": _log, "log": lambda x, b=10: _log(x, b), "log10": lambda x: _log(x, 10), "log2": lambda x: _log(x, 2),
    "exp": _floatfn(math.exp), "sin": _floatfn(math.sin), "cos": _floatfn(math.cos), "tan": _floatfn(math.tan),
    "asin": _floatfn(math.asin), "acos": _floatfn(math.acos), "atan": _floatfn(math.atan),
    "factorial": _factorial,
    "gcd": lambda a, b: math.gcd(_int_arg("gcd", a), _int_arg("gcd", b)),
    "lcm": lambda a, b: math.lcm(_int_arg("lcm", a), _int_arg("lcm", b)),
    "min": min, "max": max,
}
CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _norm(x: Num) -> Num:
    if isinstance(x, Fraction) and x.denominator == 1:
        return int(x)
    return x


def _check(x: Num) -> Num:
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            raise CalcError("the result is not a finite number")
        return x
    q = Fraction(x)
    if q.numerator.bit_length() > MAX_BITS or q.denominator.bit_length() > MAX_BITS:
        raise CalcError(f"the result is larger than {MAX_BITS} bits")
    return _norm(x)


def _pow(a: Num, b: Num) -> Num:
    if isinstance(b, (int, Fraction)) and Fraction(b).denominator == 1 and not isinstance(a, float):
        n = int(b)
        if abs(n) > MAX_EXP:
            raise CalcError(f"exponents are limited to ±{MAX_EXP}")
        q = Fraction(a)
        if q == 0 and n < 0:
            raise CalcError("division by zero")
        if max(abs(q.numerator), q.denominator).bit_length() * abs(n) > MAX_BITS + 64:
            raise CalcError(f"the result is larger than {MAX_BITS} bits")
        return q ** n
    if isinstance(b, Fraction) and b.denominator in (2, 3) and not isinstance(a, float):
        r = _exact_root(a, b.denominator)
        return _pow(r, b.numerator) if not isinstance(r, float) else r ** b.numerator
    try:
        r = _real(a) ** _real(b)
    except (OverflowError, ZeroDivisionError) as e:
        raise CalcError(str(e)) from None
    if isinstance(r, complex):
        raise CalcError("the result is not a real number")
    return r


def _eval(node: ast.AST) -> Num:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Constant) and type(node.value) is float:
        # a decimal as typed is taken exactly (0.1 is 1/10, not the nearest float); 1e300 stays a float
        r = repr(node.value)
        return _norm(Fraction(r)) if "e" not in r and "inf" not in r else node.value
    if isinstance(node, ast.Name):
        if node.id in CONSTS:
            return CONSTS[node.id]
        raise CalcError(f"unknown name: {node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left), _eval(node.right)
        op = node.op
        if isinstance(op, ast.Pow):
            return _check(_pow(a, b))
        if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
            raise CalcError("division by zero")
        exact = not isinstance(a, float) and not isinstance(b, float)
        if isinstance(op, ast.Add):
            return _check(a + b)
        if isinstance(op, ast.Sub):
            return _check(a - b)
        if isinstance(op, ast.Mult):
            return _check(a * b)
        if isinstance(op, ast.Div):
            return _check(Fraction(a) / Fraction(b) if exact else a / b)
        if isinstance(op, ast.FloorDiv):
            return _check(a // b)
        if isinstance(op, ast.Mod):
            return _check(a % b)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
        args = [_eval(a) for a in node.args]
        try:
            return _check(FUNCS[node.func.id](*args))
        except TypeError:
            raise CalcError(f"{node.func.id}: wrong number of arguments") from None
    raise CalcError("only numbers, + − × ÷ ^ %, parentheses, pi, e and the listed functions")


_DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d*|\.\d+)(?![\w.])")


def normalize(expr: str) -> str:
    """The forms people type, as Python's grammar spells them: × ÷ − ^ √ π n!, "15% of 80", thousands separators,
    implicit ×."""
    s = expr.strip().rstrip("=?").strip()
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-").replace("–", "-").replace("^", "**")
    s = s.replace("π", "pi").replace("τ", "tau")
    s = re.sub(r"%\s*of\b", "/100*", s)                               # 15% of 80
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)                       # 1,000,000
    s = re.sub(r"√\s*(\d+(?:\.\d+)?|\([^()]*\))", r"sqrt(\1)", s)    # √2, √(x)
    s = re.sub(r"(\d+|\([^()]*\))\s*!", r"factorial(\1)", s)         # 5!, (3+2)!
    s = re.sub(r"(?<=[\d)])\s*(?=\(|pi\b|tau\b|sqrt\()", "*", s)    # 2(3+4), 2pi, )(
    return s


def evaluate(expr: str) -> Num:
    """The exact value of `expr`, or `CalcError`."""
    if len(expr) > MAX_CHARS:
        raise CalcError(f"expressions are limited to {MAX_CHARS} characters")
    s = normalize(expr)
    if not s:
        raise CalcError("an empty expression")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        raise CalcError("not an arithmetic expression") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise CalcError(f"expressions are limited to {MAX_NODES} parts")
    return _check(_eval(tree))


def fmt(x: Num) -> str:
    """A value as Savante states it: integers whole, terminating decimals exact, other ratios as p/q ≈ decimal,
    floats to 15 significant digits."""
    if isinstance(x, float):
        return format(x, ".15g")
    if isinstance(x, int):
        return str(x)
    d = x.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    if d == 1:  # a terminating decimal: write it exactly
        k, places = x.denominator, 0
        while k % 10 == 0 or k % 2 == 0 or k % 5 == 0:
            k = k // 10 if k % 10 == 0 else (k // 2 if k % 2 == 0 else k // 5)
            places += 1
        digits = str(abs(x.numerator) * 10 ** places // x.denominator).rjust(places + 1, "0")
        return ("-" if x < 0 else "") + digits[:-places] + "." + digits[-places:]
    return f"{x.numerator}/{x.denominator} ≈ {format(float(x), '.15g')}"


_FN = "|".join(sorted(FUNCS, key=len, reverse=True))
_SPAN = re.compile(rf"(?:(?:{_FN})\s*\(|[√π(]|\d)(?:%\s*of\b|[\d\s.,+\-*/^×÷−%()π√!]|(?:{_FN}|pi|tau)\b|\be\b)*")
_OPS = set("+-*/^×÷−%√!") | {"("}
_DATE = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$")


def find(text: str, limit: int = 4) -> list[tuple[str, str]]:
    """The arithmetic in a question, computed: [(expression as written, result)], at most `limit`.

    A span counts only when it has an operator between operands and parses; dates, ranges and identifiers written
    with a bare hyphen (2026-10-08, 3-5, sha-256) are left alone."""
    out, seen = [], set()
    for m in _SPAN.finditer(text):
        span = m.group(0).strip().rstrip(".,;:")
        while span.count("(") < span.count(")") and span.endswith(")"):
            span = span[:-1].rstrip()
        if not any(c.isdigit() for c in span) and "pi" not in span and "π" not in span:
            continue
        ops = [c for c in span if c in _OPS]
        if not ops and not re.search(rf"(?:{_FN})\s*\(", span):
            continue
        if _DATE.match(span.replace(" ", "")):
            continue
        if set(ops) <= {"-", "−"} and not re.search(r"\s[-−]\s", span):
            continue  # a bare hyphen: a range or an identifier, not a subtraction
        if span in seen:
            continue
        try:
            v = evaluate(span)
        except CalcError:
            continue
        seen.add(span)
        out.append((span, fmt(v)))
        if len(out) >= limit:
            break
    return out


def note(text: str) -> str:
    """What the model is told about the arithmetic in a question (empty when there is none)."""
    rows = find(text)
    if not rows:
        return ""
    lines = "\n".join(f"{e} = {r}" for e, r in rows)
    return ("\n\n[calculator — exact results computed by sAGI/calc.py for the arithmetic above; use these values, "
            f"do not recompute them]\n{lines}")


if __name__ == "__main__":
    for a in sys.argv[1:] or ["2^64 - 1"]:
        try:
            print(fmt(evaluate(a)))
        except CalcError as e:
            print(f"refused: {e}", file=sys.stderr)
            sys.exit(1)
