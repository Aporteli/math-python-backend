import re
import sympy as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sympy.geometry import Line, Line3D, Point, Point3D
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

router = APIRouter(prefix="/api/vectors", tags=["vectors"])
T = standard_transformations + (implicit_multiplication_application, convert_xor)
_AXIS = ("x", "y", "z")


class In(BaseModel):
    mode: str = "points"
    a: list[str] = Field(default_factory=lambda: ["0", "0"])
    b: list[str] = Field(default_factory=lambda: ["3", "4"])


def _prep(s: str) -> str:
    s = s.strip().replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    return re.sub(r"\s+", "", s)


def _parse(s: str):
    if not _prep(s):
        raise ValueError("missing coordinate")
    expr = parse_expr(
        _prep(s),
        local_dict={"pi": sp.pi, "e": sp.E, "sqrt": sp.sqrt},
        transformations=T,
    )
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
    if isinstance(expr, str):
        return {"id": qid, "latex": expr, "numeric": None}
    expr = sp.simplify(expr)
    return {"id": qid, "latex": sp.latex(expr), "numeric": _num(expr)}


def _coords(items: list[str]):
    return sp.Matrix([_parse(s) for s in items])


def _tuple_latex(vec: sp.Matrix) -> str:
    return sp.latex(sp.Tuple(*[sp.simplify(vec[i]) for i in range(vec.rows)]))


def _parametric(origin: sp.Matrix, direction: sp.Matrix) -> str:
    t = sp.symbols("t")
    parts = []
    for i in range(origin.rows):
        rhs = sp.simplify(origin[i] + direction[i] * t)
        parts.append(sp.latex(sp.Eq(sp.Symbol(_AXIS[i]), rhs)))
    return ",\\ ".join(parts)


def _symmetric(origin: sp.Matrix, direction: sp.Matrix) -> str | None:
    fracs = []
    fixed = []
    for i in range(origin.rows):
        delta = sp.simplify(direction[i])
        start = sp.simplify(origin[i])
        if delta == 0:
            fixed.append(sp.latex(sp.Eq(sp.Symbol(_AXIS[i]), start)))
        else:
            fracs.append(rf"\dfrac{{{_AXIS[i]} - {sp.latex(start)}}}{{{sp.latex(delta)}}}")
    if not fracs:
        return None
    body = " = ".join(fracs) if len(fracs) > 1 else fracs[0]
    if fixed:
        body = body + ",\\ " + ",\\ ".join(fixed)
    return body


def _same(a: sp.Matrix, b: sp.Matrix) -> bool:
    return all(sp.simplify(a[i] - b[i]) == 0 for i in range(a.rows))


def _points(a: sp.Matrix, b: sp.Matrix):
    out = [
        _pack("distance", (b - a).norm()),
        _pack("midpoint", _tuple_latex((a + b) / 2)),
    ]
    if _same(a, b):
        return out
    direction = sp.simplify(b - a)
    out.append(_pack("direction", _tuple_latex(direction)))
    out.append(_pack("parametric", _parametric(a, direction)))
    if a.rows == 2:
        line = Line(Point(a[0], a[1]), Point(b[0], b[1]))
        out.append(_pack("line", sp.latex(sp.Eq(sp.simplify(line.equation()), 0))))
        dx = sp.simplify(b[0] - a[0])
        if dx != 0:
            out.append(_pack("slope", (b[1] - a[1]) / dx))
    else:
        line = Line3D(Point3D(a[0], a[1], a[2]), Point3D(b[0], b[1], b[2]))
        sym = _symmetric(a, direction)
        if sym:
            out.append(_pack("symmetric", sym))
        out.append(_pack("line", sp.latex(line.equation())))
    return out


def _vectors(a: sp.Matrix, b: sp.Matrix):
    dot = sp.simplify(a.dot(b))
    out = [
        _pack("dot", dot),
        _pack("magnitudeU", a.norm()),
        _pack("magnitudeV", b.norm()),
    ]
    if a.rows == 3:
        out.append(_pack("cross", _tuple_latex(a.cross(b))))
    mag_u, mag_v = a.norm(), b.norm()
    if sp.simplify(mag_u) != 0 and sp.simplify(mag_v) != 0:
        cos_th = sp.simplify(dot / (mag_u * mag_v))
        out.append(_pack("angle", sp.acos(cos_th)))
    if sp.simplify(b.dot(b)) != 0:
        proj = sp.simplify((dot / b.dot(b)) * b)
        out.append(_pack("projection", _tuple_latex(proj)))
    return out


@router.post("/analyze")
def analyze(d: In):
    try:
        if d.mode not in ("points", "vectors"):
            raise ValueError("unknown mode")
        if len(d.a) != len(d.b) or len(d.a) not in (2, 3):
            raise ValueError("enter 2 or 3 coordinates for both")
        a, b = _coords(d.a), _coords(d.b)
        quantities = _points(a, b) if d.mode == "points" else _vectors(a, b)
        return {"mode": d.mode, "dimension": len(d.a), "quantities": quantities}
    except Exception as err:
        raise HTTPException(400, str(err))
