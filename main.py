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

from polynomial import router as polynomial_router
from inequalities import router as inequality_router
from triangle import router as triangle_router
from logarithms import router as logarithm_router
from exponents import router as exponent_router
from unit_circle import router as unit_circle_router
from geometry import router as geometry_router
from vectors import router as vector_router
from combinatorics import router as combinatorics_router
from sequences import router as sequence_router
from radicals import router as radical_router
from rearrange import router as rearrange_router
from vector_curve import router as vector_curve_router

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

app.include_router(polynomial_router)
app.include_router(inequality_router)
app.include_router(triangle_router)
app.include_router(logarithm_router)
app.include_router(exponent_router)
app.include_router(unit_circle_router)
app.include_router(geometry_router)
app.include_router(vector_router)
app.include_router(combinatorics_router)
app.include_router(sequence_router)
app.include_router(radical_router)
app.include_router(rearrange_router)
app.include_router(vector_curve_router)

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
#  System Solver — real pedagogical step-by-step
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


# ───────────────────────────────────────────────────────────────
#  Formatting helpers
# ───────────────────────────────────────────────────────────────

def _step(title: str, explanation: str, latex: str) -> dict:
    return {"title": title, "explanation": explanation, "latex": latex}


def _sys_latex(eqs) -> str:
    body = " \\\\ ".join(
        f"{sp.latex(e.lhs)} &= {sp.latex(e.rhs)}" for e in eqs
    )
    return f"\\begin{{cases}} {body} \\end{{cases}}"


def _eq_latex(obj) -> str:
    """Safely render Eq, BooleanFalse, BooleanTrue, or any SymPy object."""
    if isinstance(obj, sp.Eq):
        return f"{sp.latex(obj.lhs)} = {sp.latex(obj.rhs)}"
    try:
        return sp.latex(obj)
    except Exception:
        return str(obj)


def _coeff(eq, var):
    """Coefficient of var on the left side of the equation."""
    return sp.expand(eq.lhs - eq.rhs).coeff(var)


def _const(eq, var):
    """The constant part when solving for var: everything that is NOT var."""
    expr = sp.expand(eq.lhs - eq.rhs)
    return expr - expr.coeff(var) * var


def _is_linear_in_all(eqs, syms) -> bool:
    for eq in eqs:
        for s in syms:
            if sp.degree(eq.lhs - eq.rhs, s) > 1:
                return False
    return True


def _classify_linear(eqs, syms):
    """Returns ('unique'|'infinite'|'inconsistent', rank_A, rank_aug)."""
    try:
        A, b = sp.linear_eq_to_matrix(eqs, syms)
        aug = A.row_join(b)
        rank_A = A.rank()
        rank_aug = aug.rank()
        n = len(syms)
        if rank_A < rank_aug:
            return "inconsistent", rank_A, rank_aug
        if rank_A < n:
            return "infinite", rank_A, rank_aug
        return "unique", rank_A, rank_aug
    except Exception:
        return "unique", 0, 0


# ───────────────────────────────────────────────────────────────
#  SUBSTITUTION METHOD (2x2 and 3x3)
# ───────────────────────────────────────────────────────────────

def _substitution_steps(eqs, syms, variables):
    steps = []
    steps.append(_step(
        "სისტემის ჩაწერა",
        "მოცემულია განტოლებათა სისტემა.",
        _sys_latex(eqs),
    ))

    current_eqs = list(eqs)
    current_syms = list(syms)
    chain = []

    while len(current_eqs) > 1 and current_syms:
        chosen_eq_idx = None
        chosen_var = None
        chosen_expr = None

        for ei, eq in enumerate(current_eqs):
            for v in current_syms:
                try:
                    sols = sp.solve(eq, v)
                except Exception:
                    sols = []
                if (
                    sols
                    and isinstance(sols, list)
                    and len(sols) > 0
                    and not isinstance(sols[0], (dict, sp.Eq))
                ):
                    chosen_eq_idx = ei
                    chosen_var = v
                    chosen_expr = sp.simplify(sols[0])
                    break
            if chosen_var is not None:
                break

        if chosen_var is None:
            return steps

        chosen_eq = current_eqs[chosen_eq_idx]

        steps.append(_step(
            f"{chosen_var}-ის გამოსახვა",
            f"გამოვსახოთ ${chosen_var}$ განტოლებიდან: {_eq_latex(chosen_eq)}.",
            f"{chosen_var} = {sp.latex(chosen_expr)}",
        ))

        chain.append((chosen_var, chosen_expr))

        remaining = [eq for i, eq in enumerate(current_eqs) if i != chosen_eq_idx]
        remaining_vars = [v for v in current_syms if v != chosen_var]

        new_eqs = []
        contradiction = False

        for i, eq in enumerate(remaining):
            subbed = sp.simplify(eq.subs(chosen_var, chosen_expr))

            if isinstance(subbed, sp.Eq):
                new_eqs.append(subbed)
                steps.append(_step(
                    f"ჩასმა ({i + 2}-ე განტოლება)",
                    f"ჩავსვათ ${chosen_var} = {sp.latex(chosen_expr)}$ "
                    f"{i + 2}-ე განტოლებაში.",
                    _eq_latex(subbed),
                ))
            else:
                contradiction = True
                steps.append(_step(
                    f"ჩასმა ({i + 2}-ე განტოლება)",
                    f"ჩავსვათ ${chosen_var} = {sp.latex(chosen_expr)}$ — "
                    f"მივიღეთ {sp.latex(subbed)}.",
                    sp.latex(subbed),
                ))

        if contradiction:
            steps.append(_step(
    "დასკვნა",
    "მივიღეთ წინააღმდეგობა — სისტემა არათავსებადია, ამონახსნი არ არსებობს.",
    "\\text{---}",
))
            return steps

        if not new_eqs:
            return steps

        current_eqs = new_eqs
        current_syms = remaining_vars

    if len(current_eqs) == 1 and len(current_syms) == 1:
        last_eq = current_eqs[0]
        last_var = current_syms[0]
        try:
            sols = sp.solve(last_eq, last_var)
        except Exception:
            sols = []

        if (
            not sols
            or not isinstance(sols, list)
            or len(sols) == 0
            or isinstance(sols[0], (dict, sp.Eq))
        ):
            steps.append(_step(
                "ბოლო განტოლება ვერ ამოიხსნა",
                "სისტემა შეიძლება არათანაბარი იყოს.",
                _eq_latex(last_eq),
            ))
            return steps

        z = sols[0]
        steps.append(_step(
            f"{last_var}-ის პოვნა",
            f"მივიღეთ ერთცვლადიანი განტოლება. ამოვხსნათ ${last_var}$-ის მიმართ.",
            f"{_eq_latex(last_eq)} \\;\\Rightarrow\\; {last_var} = {sp.latex(z)}",
        ))

        known = {last_var: z}
        for var, expr in reversed(chain):
            value = sp.simplify(expr.subs(known))
            known[var] = value
            steps.append(_step(
                f"უკუჩასმა: {var}",
                f"ჩავსვათ ნაპოვნი მნიშვნელობები და ვიპოვოთ ${var}$.",
                f"{var} = {sp.latex(expr)} \\;\\Rightarrow\\; {var} = {sp.latex(value)}",
            ))

    return steps


# ───────────────────────────────────────────────────────────────
#  ELIMINATION METHOD — Gaussian elimination with real row ops
# ───────────────────────────────────────────────────────────────

def _eliminate(eq_a, eq_b, var, symbols_map):
    """Return a new Eq where var is eliminated: coeff_b*(eq_a) - coeff_a*(eq_b)."""
    ca = _coeff(eq_a, var)
    cb = _coeff(eq_b, var)
    lhs = sp.expand(cb * eq_a.lhs - ca * eq_b.lhs)
    rhs = sp.expand(cb * eq_a.rhs - ca * eq_b.rhs)
    return sp.Eq(lhs, rhs), ca, cb


def _elimination_steps(eqs, syms, variables):
    steps = []
    steps.append(_step(
        "სისტემის ჩაწერა",
        "დავიწყოთ სისტემის ჩაწერით.",
        _sys_latex(eqs),
    ))

    work = list(eqs)

    # ── 2x2 ──
    if len(eqs) == 2:
        e1, e2 = work[0], work[1]
        x = syms[0]

        c1 = _coeff(e1, x)
        c2 = _coeff(e2, x)

        if c1 == 0 or c2 == 0:
            steps.append(_step(
                "x უკვე გამორიცხულია",
                "ერთ-ერთ განტოლებაში $x$ არ მონაწილეობს — გადავდივართ პირდაპირ ამოხსნაზე.",
                _sys_latex(eqs),
            ))
        else:
            steps.append(_step(
                "გამრავლება",
                f"გავამრავლოთ (1) განტოლება ${sp.latex(c2)}$-ზე და (2) განტოლება "
                f"${sp.latex(c1)}$-ზე, რომ $x$-ის კოეფიციენტები გაუტოლდეს.",
                (
                    "\\begin{aligned}"
                    f"{sp.latex(c2)} \\cdot \\left({sp.latex(e1.lhs)}\\right) &= "
                    f"{sp.latex(c2)} \\cdot \\left({sp.latex(e1.rhs)}\\right) \\\\"
                    f"{sp.latex(c1)} \\cdot \\left({sp.latex(e2.lhs)}\\right) &= "
                    f"{sp.latex(c1)} \\cdot \\left({sp.latex(e2.rhs)}\\right)"
                    "\\end{aligned}"
                ),
            ))

            new_eq, ca, cb = _eliminate(e1, e2, x, {})
            steps.append(_step(
                "გამოკლება (x გამოირიცხება)",
                f"გამოვაკლოთ ერთმანეთს: ${sp.latex(cb)} \\cdot (1) - "
                f"{sp.latex(ca)} \\cdot (2)$.",
                _eq_latex(new_eq),
            ))

            y = syms[1]
            try:
                y_sols = sp.solve(new_eq, y)
            except Exception:
                y_sols = []

            if y_sols:
                y_val = y_sols[0]
                steps.append(_step(
                    f"{y}-ის პოვნა",
                    f"ერთცვლადიანი განტოლებიდან ვპოულობთ ${y}$-ს.",
                    f"{y} = {sp.latex(y_val)}",
                ))

                back = sp.simplify(e1.subs(y, y_val))
                x_sols = sp.solve(back, x)
                if x_sols:
                    x_val = x_sols[0]
                    steps.append(_step(
                        f"უკუჩასმა: {x}",
                        f"ჩავსვათ ${y} = {sp.latex(y_val)}$ პირველ განტოლებაში.",
                        f"{_eq_latex(back)} \\;\\Rightarrow\\; {x} = {sp.latex(x_val)}",
                    ))
        return steps

    # ── 3x3 ──
    if len(eqs) == 3:
        e1, e2, e3 = work[0], work[1], work[2]
        x, y, z = syms

        c1 = _coeff(e1, x)
        c2 = _coeff(e2, x)
        c3 = _coeff(e3, x)

        if c1 == 0:
            for i in range(1, 3):
                if _coeff(work[i], x) != 0:
                    work[0], work[i] = work[i], work[0]
                    e1, e2, e3 = work[0], work[1], work[2]
                    c1 = _coeff(e1, x)
                    c2 = _coeff(e2, x)
                    c3 = _coeff(e3, x)
                    steps.append(_step(
                        "განტოლებების გაცვლა",
                        "პირველი განტოლების $x$-ის კოეფიციენტი 0-ია — გავცვალოთ განტოლებები.",
                        _sys_latex([e1, e2, e3]),
                    ))
                    break

        if c2 != 0 and c1 != 0:
            e2_new, _, _ = _eliminate(e1, e2, x, {})
            steps.append(_step(
                "x-ის გამორიცხვა (2)-დან",
                f"გამოვაკლოთ: ${sp.latex(c2)} \\cdot (1) - "
                f"{sp.latex(c1)} \\cdot (2)$.",
                _eq_latex(e2_new),
            ))
        else:
            e2_new = e2

        if c3 != 0 and c1 != 0:
            e3_new, _, _ = _eliminate(e1, e3, x, {})
            steps.append(_step(
                "x-ის გამორიცხვა (3)-დან",
                f"გამოვაკლოთ: ${sp.latex(c3)} \\cdot (1) - "
                f"{sp.latex(c1)} \\cdot (3)$.",
                _eq_latex(e3_new),
            ))
        else:
            e3_new = e3

        b2 = _coeff(e2_new, y)
        b3 = _coeff(e3_new, y)

        if b3 != 0 and b2 != 0:
            e3_final, _, _ = _eliminate(e2_new, e3_new, y, {})
            steps.append(_step(
                "y-ის გამორიცხვა (3)-დან",
                f"გამოვაკლოთ: ${sp.latex(b3)} \\cdot (2') - "
                f"{sp.latex(b2)} \\cdot (3')$.",
                _eq_latex(e3_final),
            ))
        else:
            e3_final = e3_new

        z_sols = sp.solve(e3_final, z) if z in e3_final.free_symbols else []
        if z_sols:
            z_val = sp.simplify(z_sols[0])
            steps.append(_step(
                f"{z}-ის პოვნა",
                f"მივიღეთ ერთცვლადიანი განტოლება ${z}$-ის მიმართ.",
                f"{_eq_latex(e3_final)} \\;\\Rightarrow\\; {z} = {sp.latex(z_val)}",
            ))

            e2_back = sp.simplify(e2_new.subs(z, z_val))
            y_sols = sp.solve(e2_back, y) if y in e2_back.free_symbols else []
            if y_sols:
                y_val = sp.simplify(y_sols[0])
                steps.append(_step(
                    f"უკუჩასმა: {y}",
                    f"ჩავსვათ ${z} = {sp.latex(z_val)}$ (2')-ში.",
                    f"{_eq_latex(e2_back)} \\;\\Rightarrow\\; {y} = {sp.latex(y_val)}",
                ))

                e1_back = sp.simplify(e1.subs({y: y_val, z: z_val}))
                x_sols = sp.solve(e1_back, x) if x in e1_back.free_symbols else []
                if x_sols:
                    x_val = sp.simplify(x_sols[0])
                    steps.append(_step(
                        f"უკუჩასმა: {x}",
                        f"ჩავსვათ ${y}$ და ${z}$ (1)-ში.",
                        f"{_eq_latex(e1_back)} \\;\\Rightarrow\\; {x} = {sp.latex(x_val)}",
                    ))

        return steps

    return steps


# ───────────────────────────────────────────────────────────────
#  MATRIX METHOD — Cramer's rule with explicit determinants
# ───────────────────────────────────────────────────────────────

def _matrix_steps(eqs, syms, variables):
    steps = []
    A, b = sp.linear_eq_to_matrix(eqs, syms)
    det = A.det()

    steps.append(_step(
        "მატრიცული სახე A·X = b",
        "გამოვყოთ კოეფიციენტების მატრიცა $A$ და თავისუფალი წევრების სვეტი $b$.",
        (
            f"A = {sp.latex(A)},\\quad "
            f"X = {sp.latex(sp.Matrix(syms))},\\quad "
            f"b = {sp.latex(b)}"
        ),
    ))

    steps.append(_step(
        "მთავარი დეტერმინანტი",
        f"გამოვთვალოთ $\\det(A)$. თუ $\\det(A) \\neq 0$, სისტემას აქვს ერთადერთი "
        f"ამონახსნი (კრამერის წესი).",
        f"\\det(A) = {sp.latex(det)}",
    ))

    if det == 0:
        steps.append(_step(
            "det(A) = 0 — განსაკუთრებული შემთხვევა",
            "სისტემას ან არ აქვს ამონახსნი, ან უსასრულოდ ბევრი. კრამერის წესი არ მუშაობს.",
            "\\det(A) = 0",
        ))
        return steps

    for i, v in enumerate(variables):
        Ai = A.copy()
        Ai[:, i] = b
        det_i = Ai.det()
        value = sp.simplify(det_i / det)

        steps.append(_step(
            f"კრამერი: {v}",
            f"$A$-ს {i + 1}-ე სვეტი შევცვალოთ $b$-თი, გამოვთვალოთ "
            f"$\\det(A_{{{v}}})$, შემდეგ გავყოთ $\\det(A)$-ზე.",
            (
                "\\begin{aligned}"
                f"A_{{{v}}} &= {sp.latex(Ai)},\\quad "
                f"\\det(A_{{{v}}}) = {sp.latex(det_i)} \\\\"
                f"{v} &= \\dfrac{{\\det(A_{{{v}}})}}{{\\det(A)}} = "
                f"\\dfrac{{{sp.latex(det_i)}}}{{{sp.latex(det)}}} = {sp.latex(value)}"
                "\\end{aligned}"
            ),
        ))

    return steps


# ───────────────────────────────────────────────────────────────
#  NONLINEAR — generic substitution
# ───────────────────────────────────────────────────────────────

def _nonlinear_steps(eqs, syms, variables, solutions):
    steps = []
    steps.append(_step(
        "სისტემის ჩაწერა",
        "მოცემულია არაწრფივი სისტემა.",
        _sys_latex(eqs),
    ))

    linear_eq = None
    linear_var = None
    linear_expr = None

    for eq in eqs:
        for v in syms:
            if sp.degree(eq.lhs - eq.rhs, v) == 1:
                sols = sp.solve(eq, v)
                if sols:
                    linear_eq = eq
                    linear_var = v
                    linear_expr = sp.simplify(sols[0])
                    break
        if linear_eq is not None:
            break

    if linear_eq is not None:
        steps.append(_step(
            f"წრფივი განტოლებიდან გამოსახვა: {linear_var}",
            f"ეს განტოლება წრფივია ${linear_var}$-ის მიმართ — გამოვსახოთ.",
            f"{linear_var} = {sp.latex(linear_expr)}",
        ))

        for eq in eqs:
            if eq is linear_eq:
                continue
            subbed = sp.simplify(eq.subs(linear_var, linear_expr))
            steps.append(_step(
                "ჩასმა",
                f"ჩავსვათ ${linear_var} = {sp.latex(linear_expr)}$ არაწრფივ განტოლებაში.",
                _eq_latex(subbed),
            ))
            other_vars = [v for v in syms if v != linear_var]
            if len(other_vars) == 1:
                sols = sp.solve(subbed, other_vars[0])
                if sols:
                    steps.append(_step(
                        f"{other_vars[0]}-ის ამოხსნა",
                        "ამოვხსნათ მიღებული ერთცვლადიანი განტოლება.",
                        (
                            f"{other_vars[0]} \\in \\left\\{{ "
                            + ", ".join(sp.latex(s) for s in sols)
                            + " \\right\\}"
                        ),
                    ))
    else:
        steps.append(_step(
    "არაწრფივი სისტემა",
    "წრფივი განტოლება ვერ მოიძებნა — SymPy ხსნის სისტემას პირდაპირ.",
    "\\text{---}",
))

    for idx, sol in enumerate(solutions):
        sol_tex = ",\\quad ".join(
            f"{v} = {sp.latex(sol[v])}" for v in variables if v in sol
        )
        steps.append(_step(
            f"ამონახსნი {idx + 1}",
            "ერთ-ერთი შესაძლო კომბინაცია.",
            sol_tex,
        ))

    return steps


# ───────────────────────────────────────────────────────────────
#  Main solver endpoint
# ───────────────────────────────────────────────────────────────

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

        try:
            raw_solutions = sp.solve(eqs, syms, dict=True)
        except Exception:
            raw_solutions = []

        if raw_solutions is True:
            solutions = []
        elif raw_solutions is False or raw_solutions is None:
            solutions = []
        elif isinstance(raw_solutions, list):
            solutions = [s for s in raw_solutions if isinstance(s, dict)]
        else:
            solutions = []

        linear_info = None
        is_linear = False
        status = "unique"
        rank_A = rank_aug = 0

        try:
            A, b = sp.linear_eq_to_matrix(eqs, syms)
            if A.shape[0] == A.shape[1]:
                is_linear = True
                det = A.det()
                status, rank_A, rank_aug = _classify_linear(eqs, syms)
                if det != 0:
                    sol_vec = A.LUsolve(b)
                    linear_info = {
                        "matrix_A": [
                            [str(c) for c in A.row(i)] for i in range(A.rows)
                        ],
                        "vector_b": [str(c) for c in b],
                        "determinant": str(det),
                        "solution": {
                            v: str(sol_vec[i])
                            for i, v in enumerate(data.variables)
                        },
                    }
        except Exception:
            is_linear = False

        methods = []

        if is_linear and status == "unique":
            for method_id, label, fn in [
                ("substitution", "ჩასმის მეთოდი", _substitution_steps),
                ("elimination", "შეკრების მეთოდი", _elimination_steps),
                ("matrix", "მატრიცული მეთოდი (კრამერი)", _matrix_steps),
            ]:
                try:
                    steps = fn(eqs, syms, data.variables)
                except Exception as e:
                    steps = [_step(
                        "მეთოდი ვერ დამუშავდა",
                        f"შიდა შეცდომა: {e}",
                        "\\text{---}",
                    )]
                methods.append({
                    "method": method_id,
                    "label": label,
                    "steps": steps,
                })

        elif is_linear and status == "inconsistent":
            methods.append({
                "method": "nonlinear",
                "label": "ანალიზი",
                "steps": [
                    _step(
    "სისტემა არათავსებადია",
    "მატრიცის დეტერმინანტი 0-ია, ხოლო გაფართოებული "
    "მატრიცის რანგი მეტია — სისტემას ამონახსნი არ აქვს.",
    (
        f"\\det(A) = 0,\\quad "
        f"\\operatorname{{rank}}(A) = {rank_A},\\quad "
        f"\\operatorname{{rank}}([A|b]) = {rank_aug} "
        f"\\Rightarrow \\text{{---}}"
    ),
),
                ],
            })

        elif is_linear and status == "infinite":
            methods.append({
                "method": "nonlinear",
                "label": "ანალიზი",
                "steps": [
                    _step(
                        "უსასრულოდ ბევრი ამონახსნი",
                        "დეტერმინანტი 0-ია და რანგები ტოლია — სისტემა "
                        "თავსებადია, მაგრამ ამონახსნები უსასრულოა "
                        "(ერთი ან მეტი თავისუფალი ცვლადი).",
                        (
                            f"\\det(A) = 0,\\quad "
                            f"\\operatorname{{rank}}(A) = "
                            f"\\operatorname{{rank}}([A|b]) = {rank_A} "
f"\\Rightarrow \\text{{---}}"
                        ),
                    ),
                ],
            })

        else:
            try:
                steps = _nonlinear_steps(eqs, syms, data.variables, solutions)
            except Exception as e:
                steps = [_step(
                    "მეთოდი ვერ დამუშავდა",
                    f"შიდა შეცდომა: {e}",
                    "\\text{---}",
                )]
            methods.append({
                "method": "nonlinear",
                "label": "არაწრფივი — ჩასმით",
                "steps": steps,
            })

        serialized_solutions = []
        for sol in solutions:
            entry = {}
            for v in data.variables:
                sym = symbols[v]
                if sym in sol:
                    entry[v] = str(sol[sym])
            if entry:
                serialized_solutions.append(entry)

        if not serialized_solutions and linear_info is not None:
            serialized_solutions.append(dict(linear_info["solution"]))

        return {
            "solutions": serialized_solutions,
            "is_linear": is_linear,
            "linear": linear_info,
            "methods": methods,
            "status": status,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"შეცდომა: {e}")