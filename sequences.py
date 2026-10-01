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

router = APIRouter(prefix="/api/sequences", tags=["sequences"])
T = standard_transformations + (implicit_multiplication_application, convert_xor)


class In(BaseModel):
    kind: str = "arithmetic"
    a: str = "2"
    d: str = "3"
    r: str = "1/2"
    n: str = "10"
    term: str = "1/n**2"
    variable: str = "n"


def _prep(s: str) -> str:
    s = s.strip().replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    return re.sub(r"\s+", "", s)


def _parse(s: str, extra=None):
    text = _prep(s)
    if not text:
        raise ValueError("missing value")
    local = {"pi": sp.pi, "e": sp.E, "sqrt": sp.sqrt, "oo": sp.oo, "inf": sp.oo}
    if extra:
        local.update(extra)
    expr = parse_expr(text, local_dict=local, transformations=T)
    if getattr(expr, "is_Float", False):
        expr = sp.nsimplify(expr, tolerance=1e-10)
    return expr


def _num(expr):
    try:
        if getattr(expr, "free_symbols", None) or not getattr(expr, "is_finite", False):
            return None
        n = sp.N(expr)
        if getattr(n, "is_real", False) and getattr(n, "is_finite", False):
            return float(n)
    except Exception:
        return None
    return None


def _pack(qid: str, expr):
    expr = sp.simplify(expr)
    return {"id": qid, "latex": sp.latex(expr), "numeric": _num(expr)}


def _closed_sum(term, var, start, end):
    try:
        value = sp.summation(term, (var, start, end))
    except Exception:
        value = sp.Sum(term, (var, start, end)).doit()
    return sp.simplify(value)


def _infinite(term, var):
    value = _closed_sum(term, var, 1, sp.oo)
    if value == sp.oo or value == -sp.oo or value.has(sp.oo, sp.zoo, sp.nan):
        return False, None
    if isinstance(value, sp.Sum) or value.has(sp.Sum):
        try:
            ratio = sp.limit(sp.Abs(term.subs(var, var + 1) / term), var, sp.oo)
        except Exception:
            return None, None
        if ratio.is_number and ratio < 1:
            return True, None
        if ratio.is_number and ratio > 1:
            return False, None
        return None, None
    if getattr(value, "is_finite", None) is False:
        return False, None
    return True, value


@router.post("/analyze")
def analyze(d: In):
    try:
        quantities = []
        converges = None
        if d.kind == "arithmetic":
            a, diff, n = _parse(d.a), _parse(d.d), _parse(d.n)
            quantities.append(_pack("nth", a + (n - 1) * diff))
            quantities.append(_pack("sum", n / 2 * (2 * a + (n - 1) * diff)))
        elif d.kind == "geometric":
            a, ratio, n = _parse(d.a), _parse(d.r), _parse(d.n)
            quantities.append(_pack("nth", a * ratio ** (n - 1)))
            if sp.simplify(ratio - 1) == 0:
                quantities.append(_pack("sum", n * a))
            else:
                quantities.append(_pack("sum", a * (1 - ratio**n) / (1 - ratio)))
            mag = sp.simplify(sp.Abs(ratio))
            if sp.simplify(a) == 0 or (mag.is_number and mag < 1):
                converges = True
                quantities.append(_pack("infinite", sp.simplify(a / (1 - ratio)) if sp.simplify(ratio - 1) != 0 else 0))
            elif mag.is_number and mag >= 1:
                converges = False
        elif d.kind == "series":
            var = sp.Symbol(d.variable or "n", integer=True, positive=True)
            term = _parse(d.term, {str(var): var})
            if d.n.strip():
                n = _parse(d.n, {str(var): var})
                quantities.append(_pack("nth", term.subs(var, n)))
                partial = _closed_sum(term, var, 1, n)
                if isinstance(partial, sp.Sum) and getattr(n, "is_integer", False) and n.is_number and 1 <= int(n) <= 80:
                    partial = sp.simplify(sum(term.subs(var, i) for i in range(1, int(n) + 1)))
                quantities.append(_pack("sum", partial))
            converges, total = _infinite(term, var)
            if converges and total is not None:
                quantities.append(_pack("infinite", total))
        else:
            raise ValueError("unknown kind")
        return {"kind": d.kind, "converges": converges, "quantities": quantities}
    except Exception as err:
        raise HTTPException(400, str(err))
