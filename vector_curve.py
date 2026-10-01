import math
import re
import sympy as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

router = APIRouter(prefix="/api/vector-curve", tags=["vector-curve"])
T = standard_transformations + (implicit_multiplication_application, convert_xor)


class In(BaseModel):
    xExpr: str = "sqrt(t)"
    yExpr: str = "2 - t"
    variable: str = "t"
    tValue: str = "1"
    tMin: str = "0"
    tMax: str = "5"


def _prep(s: str) -> str:
    s = s.strip().replace("−", "-").replace("–", "-")
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/")
    return re.sub(r"\s+", "", s)


def _parse(s: str, t: sp.Symbol):
    local = {
        "pi": sp.pi, "e": sp.E, "E": sp.E,
        "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "ln": sp.log, "log": sp.log, "exp": sp.exp, "abs": sp.Abs,
        str(t): t,
    }
    return parse_expr(_prep(s), local_dict=local, transformations=T)


def _num(expr):
    try:
        if expr.free_symbols:
            return None
        n = sp.N(expr)
        if getattr(n, "is_real", False) and getattr(n, "is_finite", False):
            return float(n)
    except Exception:
        return None
    return None


def _tex(expr) -> str:
    try:
        return sp.latex(expr)
    except Exception:
        return str(expr)


def _vec_tex(xc, yc, var_tex: str, prime: str = "") -> str:
    """LaTeX: r'(t) = a i + (b) j"""
    def wrap(v):
        s = _tex(v)
        # only wrap if it needs grouping (has +/- at top level)
        if isinstance(v, sp.Add) and len(v.args) > 1:
            return f"\\left({s}\\right)"
        return s
    return (
        f"\\mathbf{{r}}{prime}({var_tex}) = {wrap(xc)}\\,\\mathbf{{i}} "
        f"+ {wrap(yc)}\\,\\mathbf{{j}}"
    )


def _sample_curve(x_expr, y_expr, t, t_min: float, t_max: float, n: int = 500):
    points = []
    try:
        xf = sp.lambdify(t, x_expr, modules=["math"])
        yf = sp.lambdify(t, y_expr, modules=["math"])
    except Exception:
        return points
    for i in range(n + 1):
        tv = t_min + (t_max - t_min) * i / n
        try:
            xv = xf(tv)
            yv = yf(tv)
            if isinstance(xv, complex) or isinstance(yv, complex):
                continue
            xv = float(xv); yv = float(yv)
            if math.isnan(xv) or math.isnan(yv) or math.isinf(xv) or math.isinf(yv):
                continue
            if abs(xv) > 1e6 or abs(yv) > 1e6:
                continue
            points.append([xv, yv, float(tv)])
        except Exception:
            continue
    return points


@router.post("/analyze")
def analyze(d: In):
    try:
        t = sp.Symbol(d.variable, real=True)
        x_expr = _parse(d.xExpr, t)
        y_expr = _parse(d.yExpr, t)
        t0 = _parse(d.tValue, t)
        t_min = _parse(d.tMin, t)
        t_max = _parse(d.tMax, t)

        t_min_f = float(t_min)
        t_max_f = float(t_max)
        if t_min_f >= t_max_f:
            raise ValueError("t_min უნდა იყოს ნაკლები t_max-ზე")

        # derivatives
        dx = sp.simplify(sp.diff(x_expr, t))
        dy = sp.simplify(sp.diff(y_expr, t))
        d2x = sp.simplify(sp.diff(x_expr, t, 2))
        d2y = sp.simplify(sp.diff(y_expr, t, 2))

        # values at t0
        x0 = sp.simplify(x_expr.subs(t, t0))
        y0 = sp.simplify(y_expr.subs(t, t0))
        dx0 = sp.simplify(dx.subs(t, t0))
        dy0 = sp.simplify(dy.subs(t, t0))
        d2x0 = sp.simplify(d2x.subs(t, t0))
        d2y0 = sp.simplify(d2y.subs(t, t0))

        # speed
        speed_sq = sp.simplify(dx0**2 + dy0**2)
        speed0 = sp.sqrt(speed_sq) if speed_sq.is_number else sp.simplify(sp.sqrt(speed_sq))

        # unit tangent
        if speed0 != 0 and speed_sq != 0:
            uTx = sp.simplify(dx0 / speed0)
            uTy = sp.simplify(dy0 / speed0)
        else:
            uTx, uTy = sp.Integer(0), sp.Integer(0)

        # unit normal (rotate unit tangent +90°: N = (-Ty, Tx))
        uNx = -uTy
        uNy = uTx

        # curvature κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numer = sp.Abs(dx0 * d2y0 - dy0 * d2x0)
        denom = speed_sq ** sp.Rational(3, 2)
        kappa = sp.simplify(numer / denom) if denom != 0 else sp.oo

        # tangent line (through point, along (dx0,dy0))
        if dx0 != 0:
            m = sp.simplify(dy0 / dx0)
            tan_line = (
                f"y - {_tex(y0)} = {_tex(m)}\\left(x - {_tex(x0)}\\right)"
            )
        else:
            tan_line = f"x = {_tex(x0)}"

        # normal line
        if dy0 != 0:
            nm = sp.simplify(-dx0 / dy0)
            norm_line = (
                f"y - {_tex(y0)} = {_tex(nm)}\\left(x - {_tex(x0)}\\right)"
            )
        else:
            norm_line = f"x = {_tex(x0)}"

        # curve samples
        curve_pts = _sample_curve(x_expr, y_expr, t, t_min_f, t_max_f, 500)

        # numerical arc length (Simpson-ish, using samples)
        arc_len = None
        if len(curve_pts) > 2:
            s = 0.0
            for i in range(1, len(curve_pts)):
                s += math.hypot(
                    curve_pts[i][0] - curve_pts[i - 1][0],
                    curve_pts[i][1] - curve_pts[i - 1][1],
                )
            arc_len = s

        # plot bounds
        if curve_pts:
            xs = [p[0] for p in curve_pts]
            ys = [p[1] for p in curve_pts]
            x_min_d, x_max_d = min(xs), max(xs)
            y_min_d, y_max_d = min(ys), max(ys)
        else:
            x_min_d, x_max_d, y_min_d, y_max_d = -5, 5, -5, 5

        var_tex = _tex(t)
        t0_tex = _tex(t0)

        return {
            "variable": d.variable,
            "tValueLatex": t0_tex,
            "tValueNumeric": _num(t0),
            "tMinLatex": _tex(t_min),
            "tMaxLatex": _tex(t_max),
            "tMinNumeric": _num(t_min),
            "tMaxNumeric": _num(t_max),
            "rLatex": _vec_tex(x_expr, y_expr, var_tex),
            "rpLatex": _vec_tex(dx, dy, var_tex, "'"),
            "r0Latex": _vec_tex(x0, y0, t0_tex),
            "rp0Latex": _vec_tex(dx0, dy0, t0_tex, "'"),
            "x0Numeric": _num(x0),
            "y0Numeric": _num(y0),
            "dx0Numeric": _num(dx0),
            "dy0Numeric": _num(dy0),
            "d2x0Numeric": _num(d2x0),
            "d2y0Numeric": _num(d2y0),
            "speedLatex": _tex(speed0),
            "speedNumeric": _num(speed0),
            "unitTangentLatex": _vec_tex(uTx, uTy, t0_tex).replace(
                "\\mathbf{r}", "\\mathbf{T}"
            ),
            "unitTangentNumeric": [_num(uTx), _num(uTy)],
            "unitNormalLatex": _vec_tex(uNx, uNy, t0_tex).replace(
                "\\mathbf{r}", "\\mathbf{N}"
            ),
            "unitNormalNumeric": [_num(uNx), _num(uNy)],
            "curvatureLatex": _tex(kappa),
            "curvatureNumeric": _num(kappa),
            "tangentLineLatex": tan_line,
            "normalLineLatex": norm_line,
            "arcLengthNumeric": arc_len,
            "curvePoints": curve_pts,
            "plotBounds": {
                "xMin": x_min_d, "xMax": x_max_d,
                "yMin": y_min_d, "yMax": y_max_d,
            },
        }
    except Exception as err:
        raise HTTPException(400, str(err))