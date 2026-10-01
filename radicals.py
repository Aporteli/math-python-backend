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

router = APIRouter(prefix="/api/radicals", tags=["radicals"])
T = standard_transformations + (implicit_multiplication_application, convert_xor)


class In(BaseModel):
    expression: str = "sqrt(50)"
    variable: str = "x"


def _prep(s: str) -> str:
    s = s.strip().replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    s = s.replace("√", "sqrt")
    return re.sub(r"\s+", "", s)


def _parse(s: str, x: sp.Symbol):
    text = _prep(s)
    if not text:
        raise ValueError("missing expression")
    local = {
        "pi": sp.pi,
        "e": sp.E,
        "sqrt": sp.sqrt,
        "cbrt": sp.cbrt,
        "root": sp.root,
        str(x): x,
    }
    return parse_expr(text, local_dict=local, transformations=T)


def _num(expr):
    try:
        if expr.free_symbols or not getattr(expr, "is_finite", False):
            return None
        n = sp.N(expr)
        if getattr(n, "is_real", False) and getattr(n, "is_finite", False):
            return float(n)
    except Exception:
        return None
    return None


def _simplify(expr):
    expr = sp.powdenest(expr, force=True)
    expr = sp.sqrtdenest(expr)
    simplified = sp.simplify(expr)
    rationalized = sp.simplify(sp.radsimp(simplified))
    return simplified, rationalized


def _split(expr: str):
    s = _prep(expr)
    if "=" in s:
        left, right = s.split("=", 1)
        if left and right:
            return left, right
    return None, s


@router.post("/analyze")
def analyze(d: In):
    try:
        x = sp.Symbol(d.variable)
        left, rest = _split(d.expression)
        if left is None:
            expr = _parse(rest, x)
            simplified, rationalized = _simplify(expr)
            extra = sp.latex(rationalized) if rationalized != simplified else None
            return {
                "mode": "simplify",
                "inputLatex": sp.latex(expr),
                "simplifiedLatex": sp.latex(simplified),
                "rationalizedLatex": extra,
                "numeric": _num(simplified),
            }
        lhs, rhs = _parse(left, x), _parse(rest, x)
        eq = sp.Eq(lhs, rhs)
        if sp.simplify(lhs - rhs) == 0:
            return {"mode": "solve", "identity": True, "inputLatex": sp.latex(eq), "solutions": []}
        try:
            raw = sp.solve(eq, x)
        except Exception:
            raw = []
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        solutions = []
        seen = set()
        for sol in raw:
            if getattr(sol, "is_real", None) is False or (hasattr(sol, "has") and sol.has(sp.I)):
                continue
            try:
                diff = sp.simplify((lhs - rhs).subs(x, sol))
                if diff != 0 and abs(complex(sp.N(diff))) > 1e-6:
                    continue
            except Exception:
                pass
            sol = sp.simplify(sol)
            latex = sp.latex(sol)
            if latex in seen:
                continue
            seen.add(latex)
            solutions.append({"latex": latex, "numeric": _num(sol)})
            if len(solutions) >= 12:
                break
        return {
            "mode": "solve",
            "identity": False,
            "inputLatex": sp.latex(eq),
            "solutions": solutions,
        }
    except Exception as err:
        raise HTTPException(400, str(err))
