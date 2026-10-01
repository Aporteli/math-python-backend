import re
import sympy as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

router = APIRouter(prefix="/api/geometry", tags=["geometry"])
T = standard_transformations + (implicit_multiplication_application, convert_xor)

FIELDS = {
    "rectangle": ("a", "b"),
    "square": ("a",),
    "circle": ("r",),
    "triangle": ("a", "b", "c"),
    "rightTriangle": ("a", "b"),
    "trapezoid": ("a", "b", "h"),
    "regularPolygon": ("n", "a"),
    "cube": ("a",),
    "rectangularPrism": ("a", "b", "c"),
    "triangularPrism": ("a", "b", "c", "l"),
    "cylinder": ("r", "h"),
    "squarePyramid": ("a", "h"),
    "cone": ("r", "h"),
    "sphere": ("r",),
}


class In(BaseModel):
    shape: str = "sphere"
    params: dict[str, str] = Field(default_factory=lambda: {"r": "3"})
    target: str = ""
    targetValue: str = ""
    variable: str = "x"


def _prep(s: str) -> str:
    s = s.strip().replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    return re.sub(r"\s+", "", s)


def _parse(s: str, x: sp.Symbol):
    local = {"pi": sp.pi, "e": sp.E, "sqrt": sp.sqrt, str(x): x}
    expr = parse_expr(_prep(s), local_dict=local, transformations=T)
    if getattr(expr, "is_Float", False):
        expr = sp.nsimplify(expr, tolerance=1e-10)
    return expr


def _positive(expr, label: str):
    if expr.free_symbols:
        return
    if expr.is_real is False or expr.is_infinite or expr <= 0:
        raise ValueError(f"{label} must be positive")


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


def _pack(qid: str, expr):
    expr = sp.simplify(expr)
    try:
        expr = sp.simplify(sp.sqrtdenest(expr))
    except Exception:
        pass
    return {"id": qid, "latex": sp.latex(expr), "numeric": _num(expr)}


def _heron(a, b, c):
    if not (a.free_symbols or b.free_symbols or c.free_symbols):
        if not (a + b > c and a + c > b and b + c > a):
            raise ValueError("triangle inequality failed")
    s = (a + b + c) / 2
    return sp.sqrt(sp.simplify(s * (s - a) * (s - b) * (s - c)))


def _formulas(shape: str, p: dict):
    if shape == "rectangle":
        a, b = p["a"], p["b"]
        return [("perimeter", 2 * (a + b)), ("area", a * b), ("diagonal", sp.sqrt(a**2 + b**2))]
    if shape == "square":
        a = p["a"]
        return [("perimeter", 4 * a), ("area", a**2), ("diagonal", a * sp.sqrt(2))]
    if shape == "circle":
        r = p["r"]
        return [("perimeter", 2 * sp.pi * r), ("area", sp.pi * r**2)]
    if shape == "triangle":
        a, b, c = p["a"], p["b"], p["c"]
        return [("perimeter", a + b + c), ("area", _heron(a, b, c))]
    if shape == "rightTriangle":
        a, b = p["a"], p["b"]
        hyp = sp.sqrt(a**2 + b**2)
        return [("hypotenuse", hyp), ("perimeter", a + b + hyp), ("area", a * b / 2)]
    if shape == "trapezoid":
        a, b, h = p["a"], p["b"], p["h"]
        leg = sp.sqrt(h**2 + ((a - b) / 2) ** 2)
        return [("area", (a + b) * h / 2), ("perimeter", a + b + 2 * leg)]
    if shape == "regularPolygon":
        n, a = p["n"], p["a"]
        if not n.free_symbols and (not n.is_integer or n < 3):
            raise ValueError("n must be an integer ≥ 3")
        return [("perimeter", n * a), ("area", n * a**2 / (4 * sp.tan(sp.pi / n)))]
    if shape == "cube":
        a = p["a"]
        return [("surface", 6 * a**2), ("volume", a**3), ("diagonal", a * sp.sqrt(3))]
    if shape == "rectangularPrism":
        a, b, c = p["a"], p["b"], p["c"]
        return [
            ("surface", 2 * (a * b + b * c + c * a)),
            ("volume", a * b * c),
            ("diagonal", sp.sqrt(a**2 + b**2 + c**2)),
        ]
    if shape == "triangularPrism":
        a, b, c, length = p["a"], p["b"], p["c"], p["l"]
        base = _heron(a, b, c)
        lateral = (a + b + c) * length
        return [("baseArea", base), ("lateral", lateral), ("surface", lateral + 2 * base), ("volume", base * length)]
    if shape == "cylinder":
        r, h = p["r"], p["h"]
        lateral = 2 * sp.pi * r * h
        return [("lateral", lateral), ("surface", lateral + 2 * sp.pi * r**2), ("volume", sp.pi * r**2 * h)]
    if shape == "squarePyramid":
        a, h = p["a"], p["h"]
        slant = sp.sqrt(h**2 + (a / 2) ** 2)
        lateral = 2 * a * slant
        return [("slant", slant), ("baseArea", a**2), ("lateral", lateral), ("surface", a**2 + lateral), ("volume", a**2 * h / 3)]
    if shape == "cone":
        r, h = p["r"], p["h"]
        slant = sp.sqrt(r**2 + h**2)
        lateral = sp.pi * r * slant
        return [("slant", slant), ("lateral", lateral), ("surface", sp.pi * r * (r + slant)), ("volume", sp.pi * r**2 * h / 3)]
    if shape == "sphere":
        r = p["r"]
        return [("surface", 4 * sp.pi * r**2), ("volume", 4 * sp.pi * r**3 / 3)]
    raise ValueError("unknown shape")


def _read(shape: str, raw: dict, x: sp.Symbol):
    if shape not in FIELDS:
        raise ValueError("unknown shape")
    parsed = {}
    for key in FIELDS[shape]:
        text = str(raw.get(key, "")).strip()
        if not text:
            raise ValueError(f"missing {key}")
        parsed[key] = _parse(text, x)
        if key != "n":
            _positive(parsed[key], key)
    return parsed


@router.post("/analyze")
def analyze(d: In):
    try:
        x = sp.Symbol(d.variable, positive=True)
        parsed = _read(d.shape, d.params, x)
        formulas = _formulas(d.shape, parsed)
        known = {qid for qid, _ in formulas}
        solutions = []
        if d.target.strip():
            if d.target not in known:
                raise ValueError("this shape has no such quantity")
            if not d.targetValue.strip():
                raise ValueError("missing target value")
            target_expr = _parse(d.targetValue, x)
            expr = next(e for qid, e in formulas if qid == d.target)
            try:
                raw = sp.solve(sp.Eq(expr, target_expr), x)
            except Exception:
                raw = []
            if not isinstance(raw, (list, tuple)):
                raw = [raw]
            seen = set()
            chosen = None
            for sol in raw:
                if getattr(sol, "is_real", None) is False or (hasattr(sol, "has") and sol.has(sp.I)):
                    continue
                if sol.is_number and sol.is_real and sol <= 0:
                    continue
                item_expr = sp.simplify(sol)
                latex = sp.latex(item_expr)
                if latex in seen:
                    continue
                seen.add(latex)
                if chosen is None:
                    chosen = item_expr
                solutions.append({"latex": latex, "numeric": _num(item_expr)})
                if len(solutions) >= 8:
                    break
            if chosen is not None:
                formulas = [(qid, e.subs(x, chosen)) for qid, e in formulas]
        return {
            "mode": "solve" if d.target.strip() else "compute",
            "shape": d.shape,
            "solutions": solutions,
            "quantities": [_pack(qid, expr) for qid, expr in formulas],
        }
    except Exception as err:
        raise HTTPException(400, str(err))
