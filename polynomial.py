from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

router = APIRouter(prefix="/api/polynomial", tags=["polynomial"])

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


# ───────────────────────────────────────────────────────────────
#  Unicode → LaTeX conversion (SymPy emits π, ∞, ≤… which KaTeX
#  sometimes refuses to render inside mixed strings)
# ───────────────────────────────────────────────────────────────

_UNICODE_MAP = {
    # Greek lowercase
    "α": "\\alpha ", "β": "\\beta ", "γ": "\\gamma ", "δ": "\\delta ",
    "ε": "\\epsilon ", "ζ": "\\zeta ", "η": "\\eta ", "θ": "\\theta ",
    "ι": "\\iota ", "κ": "\\kappa ", "λ": "\\lambda ", "μ": "\\mu ",
    "ν": "\\nu ", "ξ": "\\xi ", "π": "\\pi ", "ρ": "\\rho ",
    "σ": "\\sigma ", "τ": "\\tau ", "υ": "\\upsilon ", "φ": "\\phi ",
    "χ": "\\chi ", "ψ": "\\psi ", "ω": "\\omega ",
    # Greek uppercase
    "Γ": "\\Gamma ", "Δ": "\\Delta ", "Θ": "\\Theta ", "Λ": "\\Lambda ",
    "Ξ": "\\Xi ", "Π": "\\Pi ", "Σ": "\\Sigma ", "Φ": "\\Phi ",
    "Ψ": "\\Psi ", "Ω": "\\Omega ",
    # Operators / relations
    "√": "\\sqrt ", "∞": "\\infty ", "≤": "\\le ", "≥": "\\ge ",
    "≠": "\\ne ", "≈": "\\approx ", "±": "\\pm ", "∓": "\\mp ",
    "×": "\\times ", "÷": "\\div ", "·": "\\cdot ", "⋅": "\\cdot ",
    "→": "\\to ", "←": "\\leftarrow ", "⇒": "\\Rightarrow ",
    "⇔": "\\Leftrightarrow ", "∑": "\\sum ", "∏": "\\prod ",
    "∫": "\\int ", "∂": "\\partial ", "∇": "\\nabla ",
    "∈": "\\in ", "∉": "\\notin ", "⊂": "\\subset ", "⊆": "\\subseteq ",
    "∪": "\\cup ", "∩": "\\cap ", "∅": "\\emptyset ",
    "∀": "\\forall ", "∃": "\\exists ", "¬": "\\neg ",
    "∧": "\\land ", "∨": "\\lor ",
    # Number systems
    "ℝ": "\\mathbb{R} ", "ℕ": "\\mathbb{N} ", "ℤ": "\\mathbb{Z} ",
    "ℚ": "\\mathbb{Q} ", "ℂ": "\\mathbb{C} ",
}


def _latex(expr) -> str:
    """SymPy LaTeX with Unicode math symbols converted to LaTeX commands."""
    s = sp.latex(expr)
    for uni, tex in _UNICODE_MAP.items():
        s = s.replace(uni, tex)
    return s


# ───────────────────────────────────────────────────────────────


class PolyInput(BaseModel):
    expression: str = "x^3 - 6x^2 + 11x - 6"
    variable: str = "x"
    divisor: str = ""


def _parse(expr_str: str, var):
    return parse_expr(
        expr_str,
        local_dict={str(var): var},
        transformations=TRANSFORMS,
    )


def _step(title: str, explanation: str, latex: str) -> dict:
    return {"title": title, "explanation": explanation, "latex": latex}


def _factor_steps(poly, var):
    steps = []
    expr = sp.expand(poly)
    steps.append(_step(
        "გაშლილი სახე",
        "მოცემული მრავალწევრი გაშლილ სახეში.",
        _latex(expr),
    ))

    factored = sp.factor(expr)
    if factored == expr:
        steps.append(_step(
            "დაშლა ვერ მოხერხდა",
            "მრავალწევრი ვერ დაიშალა მამრავლებად მთელ რიცხვებში.",
            _latex(expr),
        ))
    else:
        steps.append(_step(
            "მამრავლებად დაშლა",
            "მრავალწევრი დაიშალა მამრავლებად.",
            f"{_latex(expr)} = {_latex(factored)}",
        ))

    return steps


def _roots_steps(poly, var):
    steps = []
    expr = sp.expand(poly)
    deg = sp.degree(expr, var)

    steps.append(_step(
        "ხარისხი",
        "მრავალწევრის ხარისხი. ფუნდამენტური თეორემის მიხედვით "
        "მრავალწევრს აქვს ზუსტად ხარისხის ტოლი ფესვი (კომპლექსურების ჩათვლით).",
        f"\\deg = {deg}",
    ))

    steps.append(_step(
        "თავისუფალი წევრი",
        "ვიეტას თეორემის მიხედვით თავისუფალი წევრის გამყოფები შესაძლო ფესვებია.",
        f"a_0 = {_latex(expr.subs(var, 0))}",
    ))

    roots = sp.solve(expr, var)
    if not roots:
        steps.append(_step(
            "ფესვები",
            "ნამდვილი ფესვები ვერ მოიძებნა.",
            "\\text{---}",
        ))
        return steps

    # ── ფესვები ერთ ბლოკად, aligned გარემოში ──
    roots_tex = " \\\\ ".join(
        [f"{var} &= {_latex(r)}" for r in roots]
    )
    steps.append(_step(
        f"ფესვები ({len(roots)})",
        "ნაპოვნი ფესვები. კომპლექსური ფესვები წყვილებად მოდის (რთული კოეფიციენტების შემთხვევაში).",
        f"\\begin{{aligned}} {roots_tex} \\end{{aligned}}",
    ))

    return steps


def _long_division_steps(dividend, divisor, var):
    steps = []
    dividend = sp.expand(dividend)
    divisor = sp.expand(divisor)

    steps.append(_step(
        "საწყისი სახე",
        "გავყოთ მრავალწევრი მრავალწევრზე კუთხის მეთოდით.",
        f"\\dfrac{{{_latex(dividend)}}}{{{_latex(divisor)}}}",
    ))

    if divisor == 0:
        raise ValueError("გამყოფი ნულის ტოლია.")

    divisor_deg = sp.degree(divisor, var)
    if divisor_deg is None or divisor_deg < 0:
        raise ValueError("გამყოფი არ არის მრავალწევრი.")

    current = dividend
    quotient_terms = []
    step_num = 0
    max_steps = 25

    while step_num < max_steps:
        current_deg = sp.degree(current, var) if current != 0 else -1
        if current_deg < divisor_deg:
            break
        step_num += 1

        lt_cur = sp.LC(current, var) * var ** current_deg
        lt_div = sp.LC(divisor, var) * var ** divisor_deg
        term = sp.simplify(lt_cur / lt_div)
        quotient_terms.append(term)

        steps.append(_step(
            f"ნაბიჯი {step_num}: წამყვანი წევრების გაყოფა",
            f"ვყოფთ {_latex(lt_cur)}-ს {_latex(lt_div)}-ზე.",
            f"\\dfrac{{{_latex(lt_cur)}}}{{{_latex(lt_div)}}} = {_latex(term)}",
        ))

        product = sp.expand(term * divisor)
        new_current = sp.expand(current - product)
        steps.append(_step(
            f"ნაბიჯი {step_num}: გამრავლება და გამოკლება",
            f"გავამრავლოთ {_latex(term)} მთელ გამყოფზე და გამოვაკლოთ.",
            f"{_latex(current)} - ({_latex(product)}) = {_latex(new_current)}",
        ))

        current = new_current

    quotient = sp.expand(sum(quotient_terms))
    remainder = current

    steps.append(_step(
        "შედეგი",
        "მივიღეთ განაყოფი და ნაშთი.",
        (
            f"\\dfrac{{{_latex(dividend)}}}{{{_latex(divisor)}}} = "
            f"{_latex(quotient)} + "
            f"\\dfrac{{{_latex(remainder)}}}{{{_latex(divisor)}}}"
        ),
    ))

    return steps, quotient, remainder


@router.post("/analyze")
def analyze(data: PolyInput):
    try:
        var = sp.Symbol(data.variable)

        try:
            poly = _parse(data.expression, var)
        except Exception as e:
            raise HTTPException(400, f"ვერ დავამუშავე გამოსახულება: {e}")

        poly = sp.expand(poly)

        try:
            degree = int(sp.degree(poly, var))
        except Exception:
            degree = -1

        result = {
            "expanded_latex": _latex(poly),
            "factored_latex": _latex(sp.factor(poly)),
            "degree": degree,
            "leading_coeff": str(sp.LC(poly, var)) if degree >= 0 else "0",
            "roots": [],
            "operations": {},
        }

        try:
            roots = sp.solve(poly, var)
            result["roots"] = [_latex(r) for r in roots]
        except Exception:
            result["roots"] = []

        try:
            result["operations"]["factor"] = _factor_steps(poly, var)
        except Exception:
            result["operations"]["factor"] = []

        try:
            result["operations"]["roots"] = _roots_steps(poly, var)
        except Exception:
            result["operations"]["roots"] = []

        if data.divisor and data.divisor.strip():
            try:
                divisor = sp.expand(_parse(data.divisor, var))
                steps, q, r = _long_division_steps(poly, divisor, var)
                result["operations"]["division"] = steps
                result["division"] = {
                    "quotient_latex": _latex(q),
                    "remainder_latex": _latex(r),
                }
            except Exception as e:
                result["operations"]["division"] = [
                    _step("გაყოფა ვერ შესრულდა", str(e), "\\text{---}")
                ]

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"შეცდომა: {e}")