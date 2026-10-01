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

router = APIRouter(prefix="/api/rearrange", tags=["rearrange"])
T = standard_transformations + (implicit_multiplication_application, convert_xor)
_INVERSE = {sp.sin: sp.asin, sp.cos: sp.acos, sp.tan: sp.atan, sp.exp: sp.log, sp.asin: sp.sin, sp.acos: sp.cos, sp.atan: sp.tan}


class In(BaseModel):
    expression: str = "A = pi*r**2"
    variable: str = "r"


def _prep(s: str) -> str:
    s = s.strip().replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    return re.sub(r"\s+", "", s)


def _parse(s: str, var: sp.Symbol):
    text = _prep(s)
    if not text:
        raise ValueError("missing formula")
    local = {"sqrt": sp.sqrt, "cbrt": sp.cbrt, "root": sp.root, "log": sp.log, "ln": sp.log, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, str(var): var}
    if str(var) != "pi":
        local["pi"] = sp.pi
    if str(var) != "e":
        local["e"] = sp.E
    return parse_expr(text, local_dict=local, transformations=T)


def _eq(lhs, rhs) -> str:
    return sp.latex(sp.Eq(lhs, sp.simplify(rhs)))


def _split_args(expr, var):
    has, rest = [], []
    for arg in expr.args:
        (has if arg.has(var) else rest).append(arg)
    return has, rest


def _isolate(lhs, rhs, var, steps, depth=0):
    if depth > 16:
        return None
    if lhs == var:
        steps.append({"op": "result", "latex": _eq(var, rhs)})
        return [sp.simplify(rhs)]
    if not lhs.has(var) and rhs.has(var):
        lhs, rhs = rhs, lhs
        steps.append({"op": "swap", "latex": _eq(lhs, rhs)})
        return _isolate(lhs, rhs, var, steps, depth + 1)
    if lhs.has(var) and rhs.has(var):
        lhs = sp.together(sp.expand(lhs - rhs))
        rhs = sp.Integer(0)
        steps.append({"op": "collect", "latex": _eq(lhs, rhs)})
        return _isolate(lhs, rhs, var, steps, depth + 1)
    if lhs.func == sp.Add:
        has, rest = _split_args(lhs, var)
        if len(has) != 1:
            return None
        other = sp.Add(*rest) if rest else 0
        lhs, rhs = has[0], sp.simplify(rhs - other)
        steps.append({"op": "subtract", "latex": _eq(lhs, rhs)})
        return _isolate(lhs, rhs, var, steps, depth + 1)
    if lhs.func == sp.Mul:
        has, rest = _split_args(lhs, var)
        if len(has) != 1:
            return None
        other = sp.Mul(*rest) if rest else 1
        lhs, rhs = has[0], sp.simplify(rhs / other)
        steps.append({"op": "divide", "latex": _eq(lhs, rhs)})
        return _isolate(lhs, rhs, var, steps, depth + 1)
    if lhs.func == sp.Pow:
        base, exp = lhs.args
        if base.has(var) and not exp.has(var):
            root = sp.simplify(rhs ** (1 / exp))
            even = bool(exp.is_integer and exp.is_number and int(exp) % 2 == 0 and exp > 0)
            if even:
                steps.append({"op": "root", "latex": rf"{sp.latex(base)} = \pm {sp.latex(root)}"})
                pos = _isolate(base, root, var, steps, depth + 1)
                neg_steps = []
                neg = _isolate(base, sp.simplify(-root), var, neg_steps, depth + 1)
                if pos is None or neg is None:
                    return None
                sols = pos + [s for s in neg if sp.simplify(s - pos[0]) != 0]
                return sols
            lhs, rhs = base, root
            steps.append({"op": "root", "latex": _eq(lhs, rhs)})
            return _isolate(lhs, rhs, var, steps, depth + 1)
        if exp.has(var) and not base.has(var):
            lhs, rhs = exp, sp.simplify(sp.log(rhs) / sp.log(base))
            steps.append({"op": "log", "latex": _eq(lhs, rhs)})
            return _isolate(lhs, rhs, var, steps, depth + 1)
    if lhs.func in _INVERSE and lhs.args[0].has(var):
        inv = _INVERSE[lhs.func]
        lhs, rhs = lhs.args[0], sp.simplify(inv(rhs))
        steps.append({"op": "inverse", "latex": _eq(lhs, rhs)})
        return _isolate(lhs, rhs, var, steps, depth + 1)
    if isinstance(lhs, sp.log) and lhs.args[0].has(var):
        lhs, rhs = lhs.args[0], sp.simplify(sp.exp(rhs))
        steps.append({"op": "exp", "latex": _eq(lhs, rhs)})
        return _isolate(lhs, rhs, var, steps, depth + 1)
    return None


def _fallback(lhs, rhs, var):
    eq = sp.Eq(lhs, rhs)
    steps = [{"op": "start", "latex": sp.latex(eq)}]
    collected = sp.collect(sp.expand(lhs - rhs), var)
    if collected != sp.expand(lhs - rhs):
        steps.append({"op": "collect", "latex": _eq(collected, 0)})
    raw = sp.solve(eq, var)
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    sols = []
    for sol in raw:
        if getattr(sol, "is_real", None) is False and sol.has(sp.I):
            continue
        sol = sp.simplify(sol)
        sols.append(sol)
        steps.append({"op": "result", "latex": _eq(var, sol)})
        if len(sols) >= 8:
            break
    return steps, sols


@router.post("/analyze")
def analyze(d: In):
    try:
        if "=" not in d.expression:
            raise ValueError("formula needs an equals sign")
        var = sp.Symbol(_prep(d.variable) or "x")
        left, right = _prep(d.expression).split("=", 1)
        lhs, rhs = _parse(left, var), _parse(right, var)
        if not (lhs.has(var) or rhs.has(var)):
            raise ValueError("variable is not in the formula")
        steps = [{"op": "start", "latex": sp.latex(sp.Eq(lhs, rhs))}]
        sols = _isolate(lhs, rhs, var, steps)
        if not sols:
            steps, sols = _fallback(lhs, rhs, var)
        if not sols:
            raise ValueError("could not rearrange for this variable")
        seen = []
        for sol in sols:
            latex = sp.latex(sol)
            if latex not in {s["latex"] for s in seen}:
                seen.append({"latex": latex, "numeric": None})
        return {"variable": str(var), "steps": steps, "solutions": seen}
    except Exception as err:
        raise HTTPException(400, str(err))
