import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sympy as sp
from sympy import Symbol, oo, S, Interval, Union, FiniteSet, Abs
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

router = APIRouter(prefix="/api/inequality", tags=["inequality"])

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_UNICODE_MAP = {
    "π": "\\pi ", "∞": "\\infty ", "≤": "\\le ", "≥": "\\ge ",
    "≠": "\\ne ", "√": "\\sqrt ", "·": "\\cdot ", "×": "\\times ",
    "−": "-", "∪": "\\cup ",
}


# ═══════════════════════════════════════════════════════════════
#  LaTeX output (SymPy → LaTeX with Unicode cleanup)
# ═══════════════════════════════════════════════════════════════

def _latex(expr) -> str:
    try:
        combined = sp.together(expr)
        s = sp.latex(combined)
    except Exception:
        try:
            s = sp.latex(expr)
        except Exception:
            s = str(expr)
    for uni, tex in _UNICODE_MAP.items():
        s = s.replace(uni, tex)
    return s


# ═══════════════════════════════════════════════════════════════
#  LaTeX → Python (for parsing input)
# ═══════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    s = s.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("\\leq", "<=").replace("\\le", "<=")
    s = s.replace("\\geq", ">=").replace("\\ge", ">=")
    s = s.replace("\\lt", "<").replace("\\gt", ">")
    s = s.replace("\\neq", "!=")
    s = re.sub(r"\\,|\\;|\\:|\\ ", "", s)
    return s


def _latex_to_python(s: str) -> str:
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(
        r"\\(?:d)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s
    )
    s = re.sub(r"\\left\s*|\\right\s*", "", s)
    greek = {
        "\\pi": "pi", "\\theta": "theta", "\\alpha": "alpha",
        "\\beta": "beta", "\\gamma": "gamma", "\\lambda": "lambda",
        "\\mu": "mu", "\\sigma": "sigma",
    }
    for k, v in greek.items():
        s = re.sub(re.escape(k) + r"(?![a-zA-Z])", v, s)
    s = s.replace("{", "(").replace("}", ")")
    return s


def _replace_abs(s: str) -> str:
    while True:
        new = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
        if new == s:
            break
        s = new
    return s


# ═══════════════════════════════════════════════════════════════
#  LaTeX → Pretty Unicode (recursive, handles nested braces)
# ═══════════════════════════════════════════════════════════════

def _is_simple_token(x: str) -> bool:
    """Single atomic token that needs no parens: 5, -3, x, √5, √6."""
    x = x.strip()
    if not x:
        return False
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", x):
        return True
    if re.fullmatch(r"[+-]?[A-Za-z]", x):
        return True
    if re.fullmatch(r"[+-]?√[A-Za-z0-9]+", x):
        return True
    return False


def _wrap_if_needed(x: str) -> str:
    x = x.strip()
    if _is_simple_token(x):
        return x
    return f"({x})"


def _find_braced(s: str, open_idx: int) -> tuple:
    """s[open_idx] must be '{'. Returns (content, index_after_close).
    Handles nested braces correctly."""
    depth = 1
    i = open_idx + 1
    start = i
    while i < len(s):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
        i += 1
    return s[start:], len(s)


_SIMPLE_CMD_MAP = {
    'infty': '∞', 'pi': 'π', 'cdot': '·', 'times': '×',
    'div': '÷', 'le': '≤', 'leq': '≤', 'ge': '≥', 'geq': '≥',
    'ne': '≠', 'neq': '≠', 'pm': '±', 'cup': '∪',
    'left': '', 'right': '', ',': ' ', ';': ' ', ':': ' ',
}


def _parse_latex_recursive(s: str) -> str:
    """Proper LaTeX → Unicode with recursive nested-brace support."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '\\':
            j = i + 1
            while j < n and s[j].isalpha():
                j += 1
            cmd = s[i + 1:j]
            i = j

            if cmd == 'sqrt':
                while i < n and s[i] in ' \t':
                    i += 1
                if i < n and s[i] == '{':
                    content, i = _find_braced(s, i)
                    inner = _parse_latex_recursive(content)
                    if _is_simple_token(inner):
                        out.append(f"√{inner}")
                    else:
                        out.append(f"√({inner})")
                else:
                    out.append('√')

            elif cmd in ('frac', 'dfrac', 'tfrac'):
                while i < n and s[i] in ' \t':
                    i += 1
                num_raw = ''
                if i < n and s[i] == '{':
                    num_raw, i = _find_braced(s, i)
                while i < n and s[i] in ' \t':
                    i += 1
                den_raw = ''
                if i < n and s[i] == '{':
                    den_raw, i = _find_braced(s, i)
                num = _parse_latex_recursive(num_raw)
                den = _parse_latex_recursive(den_raw)
                out.append(f"{_wrap_if_needed(num)}/{_wrap_if_needed(den)}")

            else:
                out.append(_SIMPLE_CMD_MAP.get(cmd, cmd))

        elif c in '{}':
            i += 1
        else:
            out.append(c)
            i += 1

    return ''.join(out)


def _latex_to_pretty(s: str) -> str:
    return _parse_latex_recursive(s)


# ═══════════════════════════════════════════════════════════════
#  Parser for input inequality
# ═══════════════════════════════════════════════════════════════

OP_PATTERN = re.compile(r"(<=|>=|!=|<|>|=)")

OP_MAP = {
    "<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge,
}

OP_SYMBOLS = {"<": "<", "<=": "\\le ", ">": ">", ">=": "\\ge ", "=": "="}


def _parse_inequality(raw: str, var):
    s = _normalize(raw)
    s = _latex_to_python(s)
    s = _replace_abs(s)

    parts = OP_PATTERN.split(s)
    if len(parts) < 3:
        raise ValueError("ვერ ვიპოვე უტოლობის ნიშანი. გამოიყენეთ <, >, ≤, ≥")

    lhs_str, op, rhs_str = parts[0].strip(), parts[1].strip(), parts[2].strip()

    lhs = parse_expr(
        lhs_str, local_dict={str(var): var}, transformations=TRANSFORMS
    )
    rhs = parse_expr(
        rhs_str, local_dict={str(var): var}, transformations=TRANSFORMS
    )

    if op not in OP_MAP:
        raise ValueError(f"უცნობი ოპერატორი: {op}")

    ineq = OP_MAP[op](lhs, rhs)
    return ineq, op, lhs, rhs


# ═══════════════════════════════════════════════════════════════
#  Solvers
# ═══════════════════════════════════════════════════════════════

def _solve_abs_case_analysis(f_expr, op_str, var):
    """Solve multi-Abs inequalities by case analysis on sign intervals."""
    abs_atoms = list(f_expr.atoms(Abs))
    if len(abs_atoms) < 2:
        return None

    inners = [a.args[0] for a in abs_atoms]

    critical = set()
    for inner in inners:
        try:
            roots = sp.solve(sp.Eq(inner, 0), var)
            for r in roots:
                if r.is_real and r.is_finite:
                    critical.add(sp.nsimplify(r))
        except Exception:
            pass

    if not critical:
        return None

    critical_sorted = sorted(critical, key=lambda x: float(x))
    op_fn = OP_MAP[op_str]

    open_intervals = [(-sp.oo, critical_sorted[0])]
    for i in range(len(critical_sorted) - 1):
        open_intervals.append((critical_sorted[i], critical_sorted[i + 1]))
    open_intervals.append((critical_sorted[-1], sp.oo))

    total = S.EmptySet

    for lo, hi in open_intervals:
        if lo == -sp.oo and hi == sp.oo:
            sample = S.Zero
        elif lo == -sp.oo:
            sample = hi - S.One
        elif hi == sp.oo:
            sample = lo + S.One
        else:
            sample = (lo + hi) / 2

        replacements = {}
        valid = True
        for atom, inner in zip(abs_atoms, inners):
            try:
                val = sp.simplify(inner.subs(var, sample))
                if val > 0:
                    replacements[atom] = inner
                elif val < 0:
                    replacements[atom] = -inner
                else:
                    valid = False
                    break
            except Exception:
                valid = False
                break

        if not valid:
            continue

        f_no_abs = sp.expand(f_expr.subs(replacements))

        try:
            new_ineq = op_fn(f_no_abs, 0)
            local_sol = sp.solve_univariate_inequality(
                new_ineq, var, relational=False
            )
        except Exception:
            continue

        if lo == -sp.oo and hi == sp.oo:
            interval_set = S.Reals
        elif lo == -sp.oo:
            interval_set = Interval(-sp.oo, hi, True, True)
        elif hi == sp.oo:
            interval_set = Interval(lo, sp.oo, True, True)
        else:
            interval_set = Interval(lo, hi, True, True)

        try:
            piece = local_sol.intersect(interval_set)
            total = total.union(piece)
        except Exception:
            continue

    for cp in critical_sorted:
        try:
            val = sp.simplify(f_expr.subs(var, cp))
            if op_fn(val, 0) == S.true:
                total = total.union(FiniteSet(cp))
        except Exception:
            continue

    return total


def _solve(ineq, var, expr=None, op=None):
    """Standard solver with multi-Abs fallback."""
    try:
        return sp.solve_univariate_inequality(ineq, var, relational=False)
    except Exception as e:
        if expr is not None and op is not None:
            fallback = _solve_abs_case_analysis(expr, op, var)
            if fallback is not None:
                return fallback
        raise ValueError(f"ვერ ამოვხსენი უტოლობა: {e}")


# ═══════════════════════════════════════════════════════════════
#  Post-processing
# ═══════════════════════════════════════════════════════════════

def _is_real(x) -> bool:
    try:
        return bool(sp.im(x) == 0)
    except Exception:
        try:
            return bool(x.is_real)
        except Exception:
            return False


def _to_num(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _interval_dict(iv: Interval) -> dict:
    start = iv.start
    end = iv.end
    start_is_inf = start == -oo
    end_is_inf = end == oo
    return {
        "startOpen": bool(iv.left_open),
        "endOpen": bool(iv.right_open),
        "startLatex": "-\\infty" if start_is_inf else _latex(start),
        "endLatex": "+\\infty" if end_is_inf else _latex(end),
        "startNum": None if start_is_inf else _to_num(start),
        "endNum": None if end_is_inf else _to_num(end),
        "isPoint": False,
    }


def _set_to_intervals(solset) -> list:
    if solset == S.EmptySet:
        return []
    if solset == S.Reals:
        return [{
            "startOpen": True, "endOpen": True,
            "startLatex": "-\\infty", "endLatex": "+\\infty",
            "startNum": None, "endNum": None, "isPoint": False,
        }]
    if isinstance(solset, Interval):
        return [_interval_dict(solset)]
    if isinstance(solset, Union):
        result = []
        for arg in solset.args:
            result.extend(_set_to_intervals(arg))
        return sorted(result, key=lambda iv: iv["startNum"] if iv["startNum"] is not None else -1e18)
    if isinstance(solset, FiniteSet):
        return [{
            "startOpen": False, "endOpen": False,
            "startLatex": _latex(v), "endLatex": _latex(v),
            "startNum": _to_num(v), "endNum": _to_num(v),
            "isPoint": True,
        } for v in solset]
    return []


def _intervals_plain(ivs: list) -> str:
    if not ivs:
        return "∅"
    parts = []
    for iv in ivs:
        if iv.get("isPoint"):
            s = _latex_to_pretty(iv["startLatex"])
            parts.append("{" + s + "}")
            continue
        left = "(" if iv["startOpen"] else "["
        right = ")" if iv["endOpen"] else "]"
        start = _latex_to_pretty(iv["startLatex"])
        end = _latex_to_pretty(iv["endLatex"])
        parts.append(f"{left}{start}, {end}{right}")
    return " ∪ ".join(parts)


def _intervals_latex(ivs: list, var: str) -> str:
    if not ivs:
        return "\\varnothing"
    parts = []
    for iv in ivs:
        if iv.get("isPoint"):
            parts.append("\\{" + iv["startLatex"] + "\\}")
            continue
        left = "(" if iv["startOpen"] else "["
        right = ")" if iv["endOpen"] else "]"
        parts.append(f"{left}{iv['startLatex']}, {iv['endLatex']}{right}")
    return f"{var} \\in " + " \\cup ".join(parts)


def _critical_points(expr, var, ivs: list):
    points = set()
    try:
        for r in sp.solve(expr, var):
            if _is_real(r):
                points.add(r)
    except Exception:
        pass

    for iv in ivs:
        if iv.get("startNum") is not None:
            points.add(sp.nsimplify(iv["startNum"]))
        if iv.get("endNum") is not None:
            points.add(sp.nsimplify(iv["endNum"]))

    try:
        for a in expr.atoms(Abs):
            inner = a.args[0]
            for r in sp.solve(sp.Eq(inner, 0), var):
                if _is_real(r):
                    points.add(r)
    except Exception:
        pass

    return sorted(points, key=lambda v: float(v))


def _detect_type(raw: str, expr, var) -> str:
    if "|" in raw or expr.has(Abs):
        return "absolute"
    try:
        poly = sp.Poly(expr, var)
        d = poly.degree()
        if d == 1:
            return "linear"
        if d == 2:
            return "quadratic"
    except Exception:
        pass
    return "other"


def _step(title: str, explanation: str, latex: str) -> dict:
    return {"title": title, "explanation": explanation, "latex": latex}


def _ineq_latex(ineq, op: str) -> str:
    return f"{_latex(ineq.lhs)} {OP_SYMBOLS.get(op, '<')} {_latex(ineq.rhs)}"


def _build_steps(ineq, op, lhs, rhs, expr, ivs, var, kind):
    steps = []

    steps.append(_step(
        "საწყისი უტოლობა",
        "მოცემულია უტოლობა.",
        _ineq_latex(ineq, op),
    ))

    if rhs != 0:
        steps.append(_step(
            "ყველა წევრის გადატანა",
            "გადავიტანოთ ყველა წევრი მარცხენა მხარეს, რომ მივიღოთ f(x) ⋚ 0 სახე.",
            f"{_latex(lhs)} - \\left({_latex(rhs)}\\right) {OP_SYMBOLS.get(op, '<')} 0",
        ))
        steps.append(_step(
            "გამარტივება",
            "მივიღეთ ეკვივალენტური უტოლობა.",
            f"{_latex(expr)} {OP_SYMBOLS.get(op, '<')} 0",
        ))

    if kind == "absolute":
        n_abs = len(list(expr.atoms(Abs)))
        if n_abs >= 2:
            steps.append(_step(
                "მოდულების ჩამოშლა (შემთხვევების მეთოდი)",
                f"გვაქვს {n_abs} მოდული. ვყოფთ რიცხვით ღერძს ინტერვალებად "
                "თითოეული მოდულის ნულის მიხედვით. თითოეულ ინტერვალზე "
                "მოდულები იხსნება ფიქსირებული ნიშნით, ვხსნით ჩვეულებრივ უტოლობას "
                "და შედეგს ვაერთიანებთ.",
                "\\text{(იხ. ინტერვალები ქვემოთ)}",
            ))
        else:
            steps.append(_step(
                "მოდულის განმარტება",
                "მოდულიანი უტოლობა იშლება ორ შემთხვევად: "
                "|A| < B გულისხმობს −B < A < B; "
                "|A| > B გულისხმობს A < −B ან A > B.",
                "\\text{(იხ. ინტერვალები ქვემოთ)}",
            ))

    try:
        factored = sp.factor(expr)
        if factored != expr:
            steps.append(_step(
                "მამრავლებად დაშლა",
                "დავშალოთ მარცხენა მხარე მამრავლებად ინტერვალების მეთოდისთვის.",
                f"{_latex(expr)} = {_latex(factored)}",
            ))
    except Exception:
        pass

    points = _critical_points(expr, var, ivs)
    if points:
        steps.append(_step(
            "კრიტიკული წერტილები",
            "ვპოულობთ ნულებს — ისინი ყოფენ რიცხვით ღერძს ინტერვალებად.",
            ",\\quad ".join([f"{var} = {_latex(p)}" for p in points]),
        ))
        steps.append(_step(
            "ინტერვალების მეთოდი",
            "თითოეულ ინტერვალზე ვამოწმებთ ნიშანს და ვიღებთ პასუხს.",
            "\\text{(იხ. რიცხვითი ღერძი)}",
        ))

    steps.append(_step(
        "საბოლოო ამონახსნი",
        "მივიღეთ უტოლობის ამონახსნი ინტერვალების სახით.",
        _intervals_latex(ivs, var),
    ))

    return steps


# ═══════════════════════════════════════════════════════════════
#  Endpoint
# ═══════════════════════════════════════════════════════════════

class IneqInput(BaseModel):
    inequality: str = "x^2 - 5x + 6 < 0"
    variable: str = "x"


@router.post("/analyze")
def analyze(data: IneqInput):
    try:
        var = sp.Symbol(data.variable)
        ineq, op, lhs, rhs = _parse_inequality(data.inequality, var)
        expr = sp.expand(lhs - rhs)
        kind = _detect_type(data.inequality, expr, var)

        solset = _solve(ineq, var, expr=expr, op=op)
        ivs = _set_to_intervals(solset)
        points = _critical_points(expr, var, ivs)
        steps = _build_steps(ineq, op, lhs, rhs, expr, ivs, var, kind)

        return {
            "type": kind,
            "solutionLatex": _intervals_latex(ivs, var),
            "intervalNotation": _intervals_plain(ivs),
            "intervals": ivs,
            "criticalPoints": [_latex(p) for p in points],
            "criticalPointsNumeric": [
                _to_num(p) for p in points if _to_num(p) is not None
            ],
            "steps": steps,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"შეცდომა: {e}")