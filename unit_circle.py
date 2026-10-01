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

router = APIRouter(prefix="/api/unit-circle", tags=["unit-circle"])
T = standard_transformations + (implicit_multiplication_application, convert_xor)
_OPS = ("<=", ">=", "<", ">", "=")
_FUNCS = (
    ("sin", sp.sin),
    ("cos", sp.cos),
    ("tan", sp.tan),
    ("csc", sp.csc),
    ("sec", sp.sec),
    ("cot", sp.cot),
)
_TRIG = re.compile(
    r"\b(sin|cos|tan|csc|sec|cot|asin|acos|atan|arcsin|arccos|arctan)\b",
    re.I,
)


class In(BaseModel):
    expression: str = "30"
    unit: str = "deg"
    variable: str = "x"


def _prep(s: str) -> str:
    s = s.strip().replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    s = s.replace("°", "deg")
    return re.sub(r"\s+", "", s)


def _locals(x: sp.Symbol):
    local = {
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "csc": sp.csc,
        "sec": sp.sec,
        "cot": sp.cot,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "rad": sp.rad,
        "deg": lambda a: sp.rad(a),
    }
    if str(x) in local:
        del local[str(x)]
    local[str(x)] = x
    return local


def _parse(s: str, x: sp.Symbol):
    return parse_expr(s, local_dict=_locals(x), transformations=T)


def _clean(e):
    if getattr(e, "is_Float", False):
        return sp.nsimplify(e, tolerance=1e-10)
    return e


def _mark_degrees(s: str) -> str:
    s = re.sub(r"(\d+(?:\.\d+)?)deg\b", r"rad(\1)", s)
    s = re.sub(r"(\([^()]+\))deg\b", r"rad\1", s)
    return s


def _apply_unit(s: str, unit: str) -> str:
    s = _mark_degrees(s)
    if unit != "deg":
        return s

    def repl(m):
        fn, arg = m.group(1), m.group(2)
        if "pi" in arg or "rad(" in arg:
            return m.group(0)
        if re.fullmatch(r"[\d.+\-*/()]+", arg):
            return f"{fn}(rad({arg}))"
        return m.group(0)

    return re.sub(r"\b(sin|cos|tan|csc|sec|cot|asin|acos|atan)\(([^()]*)\)", repl, s)


def _split(expr: str):
    s = _prep(expr)
    for op in _OPS:
        if op in s:
            left, right = s.split(op, 1)
            if left and right:
                return op, left, right
    return None, s, None


def _is_angle(expr: str) -> bool:
    s = _prep(expr)
    if any(op in s for op in _OPS):
        return False
    return _TRIG.search(s) is None


def _num(expr):
    try:
        if expr is None or not getattr(expr, "is_finite", False):
            return None
        n = sp.N(expr)
        if getattr(n, "is_real", False) and getattr(n, "is_finite", False):
            return float(n)
    except Exception:
        return None
    return None


def _exact(expr):
    expr = sp.simplify(expr)
    if not getattr(expr, "is_finite", True) or expr.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        return None
    try:
        expr = sp.sqrtdenest(sp.trigsimp(expr))
        expr = sp.simplify(expr)
    except Exception:
        pass
    if not getattr(expr, "is_finite", True) or expr.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        return None
    return expr


def _value(expr):
    exact = _exact(expr)
    if exact is None:
        return {"latex": None, "numeric": None, "undefined": True}
    return {"latex": sp.latex(exact), "numeric": _num(exact), "undefined": False}


def _angle_pair(angle, unit: str, forced_deg: bool):
    angle = _clean(angle)
    if forced_deg or unit == "deg":
        deg, rad = sp.simplify(angle), sp.simplify(sp.rad(angle))
    else:
        deg, rad = sp.simplify(sp.deg(angle)), sp.simplify(angle)
    return deg, rad


def _item(sol):
    sol = sp.simplify(sol)
    deg = _exact(sp.deg(sol))
    return {
        "latex": sp.latex(sol),
        "numeric": _num(sol),
        "degreesLatex": sp.latex(deg) if deg is not None else None,
    }


def _finite(solset):
    if isinstance(solset, sp.FiniteSet):
        return list(solset)
    return []


@router.post("/analyze")
def analyze(d: In):
    try:
        x = sp.Symbol(d.variable)
        unit = "rad" if d.unit == "rad" else "deg"
        if _is_angle(d.expression):
            raw = _prep(d.expression)
            forced = "deg" in raw
            body = re.sub(r"deg$", "", raw) or "0"
            deg, rad = _angle_pair(_parse(body, x), unit, forced)
            values = []
            for name, fn in _FUNCS:
                item = _value(fn(rad))
                item["name"] = name
                values.append(item)
            return {
                "mode": "angle",
                "inputLatex": sp.latex(deg if unit == "deg" or forced else rad),
                "degreesLatex": sp.latex(deg),
                "radiansLatex": sp.latex(rad),
                "values": values,
            }

        op, left, right = _split(d.expression)
        if op is None:
            e = _parse(_apply_unit(_prep(d.expression), unit), x)
            simplified = sp.simplify(sp.trigsimp(e))
            expanded = sp.expand_trig(e)
            expanded_latex = sp.latex(expanded) if expanded != e and expanded != simplified else None
            degrees_latex = radians_latex = None
            ratio = sp.simplify(simplified / sp.pi) if not simplified.free_symbols else None
            if ratio is not None and getattr(ratio, "is_rational", False):
                degrees_latex = sp.latex(sp.simplify(sp.deg(simplified)))
                radians_latex = sp.latex(sp.simplify(simplified))
            return {
                "mode": "simplify",
                "inputLatex": sp.latex(e),
                "simplifiedLatex": sp.latex(simplified),
                "expandedLatex": expanded_latex,
                "degreesLatex": degrees_latex,
                "radiansLatex": radians_latex,
            }

        lhs = _parse(_apply_unit(left, unit), x)
        rhs = _parse(_apply_unit(right, unit), x)
        if op != "=":
            rel = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op](lhs, rhs)
            sol = sp.reduce_inequalities(rel, [x])
            return {
                "mode": "solve",
                "identity": False,
                "inputLatex": sp.latex(rel),
                "solutions": [],
                "solutionLatex": sp.latex(sol),
            }
        eq = sp.Eq(lhs, rhs)
        if sp.trigsimp(sp.simplify(lhs - rhs)) == 0:
            return {"mode": "solve", "identity": True, "inputLatex": sp.latex(eq), "solutions": [], "solutionLatex": None}
        period = sp.Interval(0, 2 * sp.pi, right_open=True)
        try:
            principal = sp.solveset(eq, x, domain=period)
        except Exception:
            principal = sp.S.EmptySet
        try:
            general = sp.solveset(eq, x, domain=sp.S.Reals)
        except Exception:
            general = principal
        items = []
        for sol in _finite(principal):
            if getattr(sol, "is_real", None) is False or (hasattr(sol, "has") and sol.has(sp.I)):
                continue
            item = _item(sol)
            if item["latex"] not in {s["latex"] for s in items}:
                items.append(item)
            if len(items) >= 12:
                break
        general_latex = None if isinstance(general, sp.FiniteSet) or general == principal else sp.latex(general)
        if not items and isinstance(general, sp.FiniteSet):
            items = [_item(sol) for sol in list(general)[:12]]
        return {
            "mode": "solve",
            "identity": False,
            "inputLatex": sp.latex(eq),
            "solutions": items,
            "solutionLatex": general_latex,
        }
    except Exception as err:
        raise HTTPException(400, str(err))
