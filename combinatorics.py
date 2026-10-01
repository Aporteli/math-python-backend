import re
import sympy as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

router = APIRouter(prefix="/api/combinatorics", tags=["combinatorics"])
T = standard_transformations + (implicit_multiplication_application, convert_xor)


class In(BaseModel):
    kind: str = "C"
    n: str = "10"
    k: str = "3"
    repetition: bool = False
    expression: str = ""


def _prep(s: str) -> str:
    s = s.strip().replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*")
    return re.sub(r"\s+", "", s)


def _parse(s: str):
    text = _prep(s)
    if not text:
        raise ValueError("missing value")
    expr = parse_expr(
        text,
        local_dict={"pi": sp.pi, "e": sp.E, "sqrt": sp.sqrt, "factorial": sp.factorial, "binom": sp.binomial},
        transformations=T,
    )
    if getattr(expr, "is_Float", False):
        expr = sp.nsimplify(expr, tolerance=1e-10)
    return expr


def _int_ok(expr, name: str, low: int = 0):
    if expr.free_symbols:
        return
    if not expr.is_integer or expr < low:
        raise ValueError(f"{name} must be an integer ≥ {low}")


def _num(expr):
    try:
        if expr.free_symbols or not getattr(expr, "is_finite", False):
            return None
        if expr.is_integer and abs(int(expr)) > 10**15:
            return None
        n = sp.N(expr)
        if getattr(n, "is_real", False) and getattr(n, "is_finite", False):
            return float(n)
    except Exception:
        return None
    return None


def _compute(kind: str, n, k, repetition: bool):
    if kind == "P":
        _int_ok(n, "n")
        return sp.factorial(n), rf"{sp.latex(n)}!"
    if kind not in ("A", "C", "binomial"):
        raise ValueError("unknown kind")
    _int_ok(n, "n")
    _int_ok(k, "k")
    if not (n.free_symbols or k.free_symbols) and not repetition and k > n:
        raise ValueError("k cannot exceed n")
    if kind == "A":
        if repetition:
            return n**k, rf"{sp.latex(n)}^{{{sp.latex(k)}}}"
        return sp.ff(n, k), rf"A_{{{sp.latex(n)}}}^{{{sp.latex(k)}}}"
    value = sp.binomial(n + k - 1, k) if repetition and kind == "C" else sp.binomial(n, k)
    formula = (
        rf"\binom{{{sp.latex(n + k - 1)}}}{{{sp.latex(k)}}}"
        if repetition and kind == "C"
        else rf"\binom{{{sp.latex(n)}}}{{{sp.latex(k)}}}"
    )
    return value, formula


@router.post("/analyze")
def analyze(d: In):
    try:
        expanded = None
        if d.expression.strip():
            expanded = sp.latex(sp.expand(_parse(d.expression)))
        need_count = d.kind != "binomial" or (d.n.strip() and d.k.strip())
        if d.kind == "binomial" and not need_count and expanded is None:
            raise ValueError("enter n and k or an expression")
        if d.kind == "P" or (d.kind != "binomial" or (d.n.strip() and d.k.strip())):
            n = _parse(d.n)
            k = _parse(d.k) if d.kind != "P" else sp.Integer(0)
            value, formula = _compute(d.kind, n, k, d.repetition)
            value = sp.simplify(value)
            return {
                "kind": d.kind,
                "formulaLatex": formula,
                "valueLatex": sp.latex(value),
                "numeric": _num(value),
                "expandedLatex": expanded,
            }
        return {
            "kind": d.kind,
            "formulaLatex": None,
            "valueLatex": None,
            "numeric": None,
            "expandedLatex": expanded,
        }
    except Exception as err:
        raise HTTPException(400, str(err))
