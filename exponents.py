import re
import sympy as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

router = APIRouter(prefix="/api/exponent", tags=["exponent"])
T = standard_transformations + (implicit_multiplication_application, convert_xor)
_OPS = ("<=", ">=", "<", ">", "=")


class In(BaseModel):
    expression: str = "2^(x + 1) = 16"
    variable: str = "x"


def _prep(s: str) -> str:
    s = s.strip().replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    return re.sub(r"\s+", "", s)


def _parse(s: str, x: sp.Symbol):
    local = {
        "e": sp.E,
        "E": sp.E,
        "pi": sp.pi,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "root": sp.root,
        "ln": sp.log,
        "log": sp.log,
        "abs": sp.Abs,
    }
    if str(x) in local:
        del local[str(x)]
    local[str(x)] = x
    return parse_expr(_prep(s), local_dict=local, transformations=T)


def _split(expr: str):
    s = _prep(expr)
    for op in _OPS:
        if op in s:
            left, right = s.split(op, 1)
            if left and right:
                return op, left, right
    return None, s, None


def _num(expr):
    try:
        n = sp.N(expr)
        if getattr(n, "is_real", False):
            return float(n)
    except Exception:
        return None
    return None


def _item(expr):
    expr = sp.simplify(expr)
    return {"latex": sp.latex(expr), "numeric": _num(expr)}


def _simplify_forms(e):
    nested = sp.powdenest(e, force=True)
    simplified = sp.simplify(sp.powsimp(nested, force=True))
    expanded = simplified
    if _safe_expand(e):
        expanded = sp.expand(sp.expand_power_exp(sp.expand_power_base(e, force=True)))
        expanded = sp.simplify(expanded)
    return simplified, expanded


def _safe_expand(e) -> bool:
    for power in e.atoms(sp.Pow):
        exp = power.exp
        if exp.is_number and exp.is_integer and abs(int(exp)) > 12 and power.base.free_symbols:
            return False
    return True


def _constant_latex(e):
    if e.free_symbols:
        return None
    exact = sp.simplify(e)
    numeric = sp.N(exact, 12)
    if exact == numeric or exact.is_number and not exact.is_Float:
        return sp.latex(exact) if exact != e else sp.latex(numeric)
    return sp.latex(exact)


def _keeps(lhs, rhs, x, sol) -> bool:
    try:
        if sol.has(sp.I) or sol.is_real is False:
            return False
        diff = sp.simplify((lhs - rhs).subs(x, sol))
        if diff == 0 or diff == sp.S.Zero:
            return True
        n = complex(sp.N(diff))
        return abs(n) < 1e-6
    except Exception:
        return True


def _from_solve(lhs, rhs, x):
    eq = sp.Eq(lhs, rhs)
    if sp.simplify(lhs - rhs) == 0:
        return {"mode": "solve", "identity": True, "inputLatex": sp.latex(eq), "solutions": []}
    try:
        raw = sp.solve(eq, x)
    except Exception:
        raw = []
    if isinstance(raw, dict):
        raw = [raw[x]] if x in raw else []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    real = []
    for sol in raw:
        if getattr(sol, "is_real", None) is False or (hasattr(sol, "has") and sol.has(sp.I)):
            continue
        if _keeps(lhs, rhs, x, sol):
            real.append(sol)
    if real:
        seen = []
        for sol in real:
            item = _item(sol)
            if item["latex"] not in {s["latex"] for s in seen}:
                seen.append(item)
            if len(seen) >= 20:
                break
        return {
            "mode": "solve",
            "identity": False,
            "inputLatex": sp.latex(eq),
            "solutions": seen,
            "solutionLatex": None,
        }
    try:
        solset = sp.solveset(eq, x, domain=sp.S.Reals)
    except Exception:
        solset = None
    if solset == sp.S.Reals:
        return {"mode": "solve", "identity": True, "inputLatex": sp.latex(eq), "solutions": []}
    if solset is None or solset == sp.S.EmptySet:
        return {
            "mode": "solve",
            "identity": False,
            "inputLatex": sp.latex(eq),
            "solutions": [],
            "solutionLatex": None,
        }
    if isinstance(solset, sp.FiniteSet):
        items = [_item(s) for s in list(solset)[:20] if _keeps(lhs, rhs, x, s)]
        return {
            "mode": "solve",
            "identity": False,
            "inputLatex": sp.latex(eq),
            "solutions": items,
            "solutionLatex": None,
        }
    return {
        "mode": "solve",
        "identity": False,
        "inputLatex": sp.latex(eq),
        "solutions": [],
        "solutionLatex": sp.latex(solset),
    }


def _relation(op, lhs, rhs):
    return {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op](lhs, rhs)


@router.post("/analyze")
def analyze(d: In):
    try:
        x = sp.Symbol(d.variable)
        op, left, right = _split(d.expression)
        if op is None:
            e = _parse(left, x)
            simplified, expanded = _simplify_forms(e)
            expanded_latex = sp.latex(expanded) if expanded != simplified else None
            return {
                "mode": "simplify",
                "inputLatex": sp.latex(e),
                "simplifiedLatex": sp.latex(simplified),
                "expandedLatex": expanded_latex,
                "numericLatex": _constant_latex(simplified),
            }
        lhs, rhs = _parse(left, x), _parse(right, x)
        if op == "=":
            return _from_solve(lhs, rhs, x)
        rel = _relation(op, lhs, rhs)
        try:
            sol = sp.reduce_inequalities(rel, [x])
        except Exception:
            sol = sp.solve_univariate_inequality(rel, x, relational=False)
        return {
            "mode": "solve",
            "identity": False,
            "inputLatex": sp.latex(rel),
            "solutions": [],
            "solutionLatex": sp.latex(sol),
        }
    except Exception as err:
        raise HTTPException(400, str(err))
