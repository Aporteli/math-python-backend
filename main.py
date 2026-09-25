from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

app = FastAPI(title="Math Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://math-site-blond.vercel.app",
        "https://math-site.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
#  EXISTING: Parametric Curve Evaluation
# ═══════════════════════════════════════════════════════════════

class CurveInput(BaseModel):
    x_expr: str = "cos(t)"
    y_expr: str = "sin(t)"
    z_expr: str = "t / 2"
    t_min: float = -10.0
    t_max: float = 10.0
    num_points: int = 300


@app.get("/")
def read_root():
    return {"status": "Python Math Engine is running"}


@app.post("/api/eval-curve")
def eval_curve(data: CurveInput):
    try:
        t = sp.Symbol("t")
        x_sym = sp.sympify(data.x_expr)
        y_sym = sp.sympify(data.y_expr)
        z_sym = sp.sympify(data.z_expr)

        x_func = sp.lambdify(t, x_sym, modules=["numpy"])
        y_func = sp.lambdify(t, y_sym, modules=["numpy"])
        z_func = sp.lambdify(t, z_sym, modules=["numpy"])

        t_vals = np.linspace(data.t_min, data.t_max, data.num_points)
        x_vals = x_func(t_vals)
        y_vals = y_func(t_vals)
        z_vals = z_func(t_vals)

        if np.isscalar(x_vals):
            x_vals = np.full_like(t_vals, x_vals)
        if np.isscalar(y_vals):
            y_vals = np.full_like(t_vals, y_vals)
        if np.isscalar(z_vals):
            z_vals = np.full_like(t_vals, z_vals)

        x_vals = np.nan_to_num(x_vals, nan=0.0, posinf=100.0, neginf=-100.0)
        y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=100.0, neginf=-100.0)
        z_vals = np.nan_to_num(z_vals, nan=0.0, posinf=100.0, neginf=-100.0)

        max_r = max(
            float(np.max(np.abs(x_vals))),
            float(np.max(np.abs(y_vals))),
            3.0,
        )
        max_z = (
            float(np.max(np.abs(z_vals)))
            if np.max(np.abs(z_vals)) > 0
            else 3.0
        )

        grid_lines = []
        grid_range = np.linspace(-max_r, max_r, 9)
        for r in grid_range:
            grid_lines.append(
                {"x": [-max_r, max_r], "y": [float(r), float(r)], "z": [0.0, 0.0]}
            )
            grid_lines.append(
                {"x": [float(r), float(r)], "y": [-max_r, max_r], "z": [0.0, 0.0]}
            )

        return {
            "curve": {
                "x": x_vals.tolist(),
                "y": y_vals.tolist(),
                "z": z_vals.tolist(),
            },
            "grid_lines": grid_lines,
            "bounds": {"max_r": max_r, "max_z": max_z},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════════
#  NEW: Linear & Non-linear System Solver
# ═══════════════════════════════════════════════════════════════

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


class SolveInput(BaseModel):
    equations: list[str] = Field(..., min_length=2, max_length=3)
    variables: list[str] = Field(..., min_length=2, max_length=3)


def _parse_equation(eq: str, symbols: dict):
    """Parse '2x + 3y = 8' into Eq(2*x + 3*y, 8)."""
    if "=" in eq:
        lhs, rhs = eq.split("=", 1)
        return sp.Eq(
            parse_expr(lhs, local_dict=symbols, transformations=TRANSFORMS),
            parse_expr(rhs, local_dict=symbols, transformations=TRANSFORMS),
        )
    return sp.Eq(
        parse_expr(eq, local_dict=symbols, transformations=TRANSFORMS),
        0,
    )


def _build_substitution_steps(eqs, syms, variables):
    """Step-by-step explanation for linear 2x2 or 3x3 systems (substitution)."""
    steps = []
    # Step 1: state the system
    system_tex = " \\\\ ".join(
        [f"{sp.latex(e.lhs)} = {sp.latex(e.rhs)}" for e in eqs]
    )
    steps.append(
        {
            "title": "სისტემის ჩაწერა",
            "explanation": "მოცემულია განტოლებათა სისტემა.",
            "latex": f"\\begin{{cases}} {system_tex} \\end{{cases}}",
        }
    )

    # Step 2: solve each variable by isolating from the first equation
    A, b = sp.linear_eq_to_matrix(eqs, syms)
    det = A.det()
    steps.append(
        {
            "title": "მატრიცული სახე",
            "explanation": "გადავიყვანოთ სისტემა A·X = b სახეში.",
            "latex": (
                f"A = {sp.latex(A)},\\quad "
                f"b = {sp.latex(b)},\\quad "
                f"\\det(A) = {sp.latex(det)}"
            ),
        }
    )

    if det != 0:
        sol = A.LUsolve(b)
        for i, v in enumerate(variables):
            steps.append(
                {
                    "title": f"{v}-ის მნიშვნელობა",
                    "explanation": (
                        f"კრამერის წესით ან გაუსის მეთოდით ვიღებთ {v}-ს."
                    ),
                    "latex": f"{v} = {sp.latex(sol[i])}",
                }
            )
    return steps


def _build_elimination_steps(eqs, syms, variables):
    """Step-by-step for the elimination method."""
    steps = []
    system_tex = " \\\\ ".join(
        [f"{sp.latex(e.lhs)} = {sp.latex(e.rhs)}" for e in eqs]
    )
    steps.append(
        {
            "title": "სისტემის ჩაწერა",
            "explanation": "დავიწყოთ სისტემის ჩაწერით.",
            "latex": f"\\begin{{cases}} {system_tex} \\end{{cases}}",
        }
    )

    if len(eqs) == 2:
        e1, e2 = eqs[0], eqs[1]
        # coefficients of x in both
        c1 = e1.lhs.coeff(syms[0])
        c2 = e2.lhs.coeff(syms[0])
        if c1 != 0 and c2 != 0:
            steps.append(
                {
                    "title": "პირველი განტოლების გამრავლება",
                    "explanation": (
                        f"გავამრავლოთ პირველი განტოლება {sp.latex(c2)}-ზე, "
                        f"რომ x-ის კოეფიციენტები გაუტოლდეს."
                    ),
                    "latex": (
                        f"{sp.latex(c2)} \\cdot \\left({sp.latex(e1.lhs)}"
                        f"\\right) = {sp.latex(c2)} \\cdot {sp.latex(e1.rhs)}"
                    ),
                }
            )
            steps.append(
                {
                    "title": "მეორე განტოლების გამრავლება",
                    "explanation": (
                        f"გავამრავლოთ მეორე განტოლება {sp.latex(c1)}-ზე."
                    ),
                    "latex": (
                        f"{sp.latex(c1)} \\cdot \\left({sp.latex(e2.lhs)}"
                        f"\\right) = {sp.latex(c1)} \\cdot {sp.latex(e2.rhs)}"
                    ),
                }
            )
            steps.append(
                {
                    "title": "გამოკლება",
                    "explanation": "გამოვაკლოთ ერთმანეთს, რომ x გამოირიცხოს.",
                    "latex": (
                        f"{sp.latex(c2 * e1.lhs - c1 * e2.lhs)} = "
                        f"{sp.latex(c2 * e1.rhs - c1 * e2.rhs)}"
                    ),
                }
            )

    A, b = sp.linear_eq_to_matrix(eqs, syms)
    if A.det() != 0:
        sol = A.LUsolve(b)
        for i, v in enumerate(variables):
            steps.append(
                {
                    "title": f"საბოლოო: {v}",
                    "explanation": f"მივიღეთ {v}-ის მნიშვნელობა.",
                    "latex": f"{v} = {sp.latex(sol[i])}",
                }
            )
    return steps


def _build_matrix_steps(eqs, syms, variables):
    """Matrix method steps using Cramer / Gauss."""
    steps = []
    A, b = sp.linear_eq_to_matrix(eqs, syms)
    det = A.det()

    steps.append(
        {
            "title": "მატრიცის შედგენა",
            "explanation": "გამოვყოთ კოეფიციენტების მატრიცა A და თავისუფალი წევრები b.",
            "latex": f"A = {sp.latex(A)},\\quad b = {sp.latex(b)}",
        }
    )

    steps.append(
        {
            "title": "დეტერმინანტი",
            "explanation": "გამოვთვალოთ det(A). თუ ≠ 0, სისტემას აქვს ერთადერთი ამონახსნი.",
            "latex": f"\\det(A) = {sp.latex(det)}",
        }
    )

    if det == 0:
        steps.append(
            {
                "title": "განსაკუთრებული შემთხვევა",
                "explanation": "det(A) = 0. სისტემას ან არ აქვს ამონახსნი, ან უსასრულოდ ბევრი.",
                "latex": "\\det(A) = 0",
            }
        )
        return steps

    # Cramer's rule for each variable
    for i, v in enumerate(variables):
        Ai = A.copy()
        Ai[:, i] = b
        det_i = Ai.det()
        steps.append(
            {
                "title": f"კრამერი: {v}",
                "explanation": f"A-ს {i + 1}-ე სვეტი შევცვალოთ b-თი და გამოვთვალოთ დეტერმინანტი.",
                "latex": (
                    f"\\det(A_{{{v}}}) = {sp.latex(det_i)},\\quad "
                    f"{v} = \\frac{{{sp.latex(det_i)}}}{{{sp.latex(det)}}} = "
                    f"{sp.latex(det_i / det)}"
                ),
            }
        )
    return steps


def _build_nonlinear_steps(eqs, syms, variables, solutions):
    """Fallback for nonlinear systems — describe the result."""
    steps = []
    system_tex = " \\\\ ".join(
        [f"{sp.latex(e.lhs)} = {sp.latex(e.rhs)}" for e in eqs]
    )
    steps.append(
        {
            "title": "სისტემის ჩაწერა",
            "explanation": "მოცემულია არაწრფივი სისტემა.",
            "latex": f"\\begin{{cases}} {system_tex} \\end{{cases}}",
        }
    )

    steps.append(
        {
            "title": "ჩასმის მეთოდი",
            "explanation": (
                "გამოვსახოთ ერთი ცვლადი მეორის მეშვეობით და ჩავსვათ სხვა განტოლებაში. "
                "არაწრფივი სისტემისთვის შეიძლება რამდენიმე ამონახსნი არსებობდეს."
            ),
            "latex": "\\text{(იხილეთ ამონახსნები მარჯვნივ)}",
        }
    )

    for idx, sol in enumerate(solutions):
        sol_tex = ",\\quad ".join(
            [f"{v} = {sp.latex(sol[v])}" for v in variables if v in sol]
        )
        steps.append(
            {
                "title": f"ამონახსნი {idx + 1}",
                "explanation": "ერთ-ერთი შესაძლო კომბინაცია.",
                "latex": sol_tex,
            }
        )
    return steps


@app.post("/api/solve")
def solve_system(data: SolveInput):
    if len(data.equations) != len(data.variables):
        raise HTTPException(
            status_code=400,
            detail="განტოლებების და ცვლადების რაოდენობა არ ემთხვევა",
        )

    try:
        symbols = {v: sp.Symbol(v) for v in data.variables}
        eqs = [_parse_equation(e, symbols) for e in data.equations]
        syms = list(symbols.values())

        # General solve (works for linear & nonlinear)
        solutions = sp.solve(eqs, syms, dict=True)

        # Try matrix view (only for square linear systems)
        linear_info = None
        is_linear = False
        try:
            A, b = sp.linear_eq_to_matrix(eqs, syms)
            det = A.det()
            # Heuristic: if all equations are linear, this will not raise
            is_linear = True
            if det != 0:
                sol_vec = A.LUsolve(b)
                linear_info = {
                    "matrix_A": [
                        [str(c) for c in A.row(i)] for i in range(A.rows)
                    ],
                    "vector_b": [str(c) for c in b],
                    "determinant": str(det),
                    "solution": {
                        v: str(sol_vec[i]) for i, v in enumerate(data.variables)
                    },
                }
        except Exception:
            is_linear = False

        # Build step-by-step explanations
        methods = []
        if is_linear and linear_info is not None:
            methods.append(
                {
                    "method": "substitution",
                    "label": "ჩასმის მეთოდი",
                    "steps": _build_substitution_steps(eqs, syms, data.variables),
                }
            )
            methods.append(
                {
                    "method": "elimination",
                    "label": "შეკრების მეთოდი",
                    "steps": _build_elimination_steps(eqs, syms, data.variables),
                }
            )
            methods.append(
                {
                    "method": "matrix",
                    "label": "მატრიცული მეთოდი",
                    "steps": _build_matrix_steps(eqs, syms, data.variables),
                }
            )
        else:
            methods.append(
                {
                    "method": "nonlinear",
                    "label": "არაწრფივი ამოხსნა",
                    "steps": _build_nonlinear_steps(
                        eqs, syms, data.variables, solutions
                    ),
                }
            )

        return {
            "solutions": [
                {v: str(sol[v]) for v in data.variables if v in sol}
                for sol in solutions
            ],
            "is_linear": is_linear,
            "linear": linear_info,
            "methods": methods,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"პარსინგის შეცდომა: {e}")