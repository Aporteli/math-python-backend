import math
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

router = APIRouter(prefix="/api/triangle", tags=["triangle"])

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_LOCAL_DICT = {
    "sqrt": sp.sqrt,
    "pi": sp.pi,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
}

EPS = 1e-9


def _preprocess(s: str) -> str:
    s = s.replace("√", "sqrt")
    s = s.replace("π", "pi")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("°", "")
    s = re.sub(r"\s+", "", s)
    return s


def _parse_num(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s).strip()
    if not txt:
        return None
    try:
        cleaned = _preprocess(txt)
        if not cleaned:
            return None
        val = parse_expr(
            cleaned, local_dict=_LOCAL_DICT, transformations=TRANSFORMS
        )
        return float(val.evalf())
    except Exception as e:
        raise ValueError(f"ვერ ვამუშავე რიცხვი: {txt} ({e})")


class TriangleInput(BaseModel):
    a: str = ""
    b: str = ""
    c: str = ""
    A: str = ""
    B: str = ""
    C: str = ""


# ═══════════════════════════════════════════════════════════════
#  Solvers
# ═══════════════════════════════════════════════════════════════

def _solve_sss(a, b, c):
    if a + b <= c + EPS or a + c <= b + EPS or b + c <= a + EPS:
        raise ValueError(
            "სამკუთხედის უტოლობა არ სრულდება: a+b>c, a+c>b, b+c>a"
        )
    cosA = (b * b + c * c - a * a) / (2 * b * c)
    cosB = (a * a + c * c - b * b) / (2 * a * c)
    cosA = max(-1.0, min(1.0, cosA))
    cosB = max(-1.0, min(1.0, cosB))
    A = math.degrees(math.acos(cosA))
    B = math.degrees(math.acos(cosB))
    C = 180.0 - A - B
    return {"a": a, "b": b, "c": c, "A": A, "B": B, "C": C}


def _solve_aas_asa(a, b, c, A, B, C):
    if a is not None:
        k = a / math.sin(math.radians(A))
    elif b is not None:
        k = b / math.sin(math.radians(B))
    else:
        k = c / math.sin(math.radians(C))
    return {
        "a": k * math.sin(math.radians(A)),
        "b": k * math.sin(math.radians(B)),
        "c": k * math.sin(math.radians(C)),
        "A": A, "B": B, "C": C,
    }


def _solve_sas(sides, angle_name, angle_val):
    adj = {"A": ["b", "c"], "B": ["a", "c"], "C": ["a", "b"]}[angle_name]
    s1 = sides[adj[0]]
    s2 = sides[adj[1]]
    opp_name = angle_name.lower()
    opp_val = math.sqrt(
        s1 * s1 + s2 * s2 - 2 * s1 * s2 * math.cos(math.radians(angle_val))
    )
    full = dict(sides)
    full[opp_name] = opp_val
    return _solve_sss(full["a"], full["b"], full["c"])


def _solve_ssa(sides, angle_name, angle_val):
    opp_name = angle_name.lower()
    if sides[opp_name] is None:
        raise ValueError("SSA: კუთხის მოპირდაპირე გვერდი უნდა იყოს ცნობილი")
    opp_val = sides[opp_name]
    other_name = next(
        k for k in ["a", "b", "c"] if k != opp_name and sides[k] is not None
    )
    other_val = sides[other_name]

    sin_ang = math.sin(math.radians(angle_val))
    if sin_ang <= EPS:
        raise ValueError("კუთხე უნდა იყოს 0°-სა და 180°-ს შორის")

    sin_other = other_val * sin_ang / opp_val
    if sin_other > 1 + 1e-6:
        raise ValueError(
            f"ამონახსნი არ არსებობს: sin = {sin_other:.4f} > 1"
        )
    sin_other = min(1.0, sin_other)

    other_ang_name = other_name.upper()
    third_side_name = next(
        k for k in ["a", "b", "c"] if k not in (opp_name, other_name)
    )
    third_ang_name = third_side_name.upper()

    base_asin = math.degrees(math.asin(sin_other))
    candidates = [base_asin, 180 - base_asin]

    solutions = []
    for other_ang_val in candidates:
        if other_ang_val <= EPS or other_ang_val >= 180 - EPS:
            continue
        third_ang_val = 180 - angle_val - other_ang_val
        if third_ang_val <= EPS:
            continue
        third_side_val = (
            opp_val * math.sin(math.radians(third_ang_val)) / sin_ang
        )
        sol = {k: None for k in ["a", "b", "c", "A", "B", "C"]}
        sol[opp_name] = opp_val
        sol[other_name] = other_val
        sol[third_side_name] = third_side_val
        sol[angle_name] = angle_val
        sol[other_ang_name] = other_ang_val
        sol[third_ang_name] = third_ang_val
        solutions.append(sol)

    unique = []
    for s in solutions:
        dup = False
        for u in unique:
            if all(
                abs(s[k] - u[k]) < 1e-6
                for k in ["a", "b", "c", "A", "B", "C"]
            ):
                dup = True
                break
        if not dup:
            unique.append(s)

    if not unique:
        raise ValueError("ამონახსნი არ არსებობს")
    return unique


def _dispatch(a, b, c, A, B, C):
    n_s = sum(1 for x in [a, b, c] if x is not None)
    n_a = sum(1 for x in [A, B, C] if x is not None)

    if n_s + n_a < 3:
        raise ValueError("საჭიროა მინიმუმ 3 პარამეტრი (გვერდი ან კუთხე)")

    for name, val in [("A", A), ("B", B), ("C", C)]:
        if val is not None and not (0 < val < 180):
            raise ValueError(
                f"კუთხე {name} = {val}° არასწორია (0 < {name} < 180)"
            )
    for name, val in [("a", a), ("b", b), ("c", c)]:
        if val is not None and val <= 0:
            raise ValueError(f"გვერდი {name} = {val} უნდა იყოს დადებითი")

    if n_a == 2 and n_s < 3:
        known = sum(x for x in [A, B, C] if x is not None)
        third = 180 - known
        if third <= EPS:
            raise ValueError("კუთხეების ჯამი ≥ 180°")
        if A is None:
            A = third
        elif B is None:
            B = third
        else:
            C = third
        n_a = 3

    if n_a == 3:
        total = A + B + C
        if abs(total - 180) > 1e-6:
            raise ValueError(f"კუთხეების ჯამი = {total:.6f}° ≠ 180°")

    if n_s == 3:
        result = _solve_sss(a, b, c)
        for name, val in [("A", A), ("B", B), ("C", C)]:
            if val is not None and abs(result[name] - val) > 1e-3:
                raise ValueError(
                    f"შეუთავსებელი მონაცემები: {name} = {val}° "
                    f"≠ {result[name]:.4f}° (SSS-დან)"
                )
        return [result]

    if n_s == 0:
        raise ValueError(
            "მხოლოდ კუთხეები არ არის საკმარისი — სამკუთხედი განუსაზღვრელია"
        )

    if n_s == 1 and n_a == 3:
        return [_solve_aas_asa(a, b, c, A, B, C)]

    if n_s == 2 and n_a == 1:
        sides = {"a": a, "b": b, "c": c}
        angles = {"A": A, "B": B, "C": C}
        ang_letter = next(k for k in ["A", "B", "C"] if angles[k] is not None)
        ang_val = angles[ang_letter]
        opp_letter = ang_letter.lower()
        if sides[opp_letter] is not None:
            return _solve_ssa(sides, ang_letter, ang_val)
        return [_solve_sas(sides, ang_letter, ang_val)]

    raise ValueError("ვერ ამოვიცანი სამკუთხედის ტიპი")


def _derive(tri):
    a, b, c = tri["a"], tri["b"], tri["c"]
    A, B, C = tri["A"], tri["B"], tri["C"]

    s = (a + b + c) / 2
    under = s * (s - a) * (s - b) * (s - c)
    area = math.sqrt(max(0.0, under))

    sinA = math.sin(math.radians(A))
    R = a / (2 * sinA) if sinA > EPS else None
    r = area / s if s > EPS else 0

    h_a = 2 * area / a
    h_b = 2 * area / b
    h_c = 2 * area / c

    m_a = 0.5 * math.sqrt(max(0.0, 2 * b * b + 2 * c * c - a * a))
    m_b = 0.5 * math.sqrt(max(0.0, 2 * a * a + 2 * c * c - b * b))
    m_c = 0.5 * math.sqrt(max(0.0, 2 * a * a + 2 * b * b - c * c))

    l_a = 2 * b * c * math.cos(math.radians(A / 2)) / (b + c)
    l_b = 2 * a * c * math.cos(math.radians(B / 2)) / (a + c)
    l_c = 2 * a * b * math.cos(math.radians(C / 2)) / (a + b)

    angles_sorted = sorted([A, B, C])
    if abs(angles_sorted[2] - 90) < 1e-6:
        tri_type = "right"
    elif angles_sorted[2] > 90:
        tri_type = "obtuse"
    else:
        tri_type = "acute"

    sides_sorted = sorted([a, b, c])
    if abs(sides_sorted[0] - sides_sorted[2]) < 1e-6 * sides_sorted[2]:
        shape = "equilateral"
    elif (
        abs(sides_sorted[0] - sides_sorted[1]) < 1e-6 * sides_sorted[2]
        or abs(sides_sorted[1] - sides_sorted[2]) < 1e-6 * sides_sorted[2]
    ):
        shape = "isosceles"
    else:
        shape = "scalene"

    return {
        "area": area,
        "perimeter": 2 * s,
        "semiperimeter": s,
        "R": R,
        "r": r,
        "heights": {"a": h_a, "b": h_b, "c": h_c},
        "medians": {"a": m_a, "b": m_b, "c": m_c},
        "bisectors": {"a": l_a, "b": l_b, "c": l_c},
        "tri_type": tri_type,
        "shape_type": shape,
    }


# ═══════════════════════════════════════════════════════════════
#  Step-by-step builders
# ═══════════════════════════════════════════════════════════════

def _fmt_num(x, digits=4):
    if x is None:
        return "?"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    return f"{xf:.{digits}f}".rstrip("0").rstrip(".")


def _step(title, explanation, latex):
    return {"title": title, "explanation": explanation, "latex": latex}


def _steps_sss(a, b, c, sol):
    cosA = (b * b + c * c - a * a) / (2 * b * c)
    cosB = (a * a + c * c - b * b) / (2 * a * c)
    s_per = (a + b + c) / 2
    return [
        _step(
            "მოცემული მონაცემები (SSS)",
            "ცნობილია სამი გვერდი — ვიპოვოთ კუთხეები კოსინუსების თეორემით.",
            f"a = {_fmt_num(a)},\\quad b = {_fmt_num(b)},\\quad c = {_fmt_num(c)}",
        ),
        _step(
            "კუთხე A — კოსინუსების თეორემა",
            "cos A = (b² + c² − a²) / (2bc)",
            "\\begin{aligned}"
            f"\\cos A &= \\dfrac{{b^2 + c^2 - a^2}}{{2bc}} \\\\"
            f"&= \\dfrac{{{_fmt_num(b*b)} + {_fmt_num(c*c)} - {_fmt_num(a*a)}}}"
            f"{{2 \\cdot {_fmt_num(b)} \\cdot {_fmt_num(c)}}} \\\\"
            f"&= {_fmt_num(cosA, 6)}"
            "\\end{aligned}",
        ),
        _step(
            "A კუთხის მნიშვნელობა",
            "ამოვიღოთ arccos — ვიღებთ A კუთხეს.",
            f"A = \\arccos\\!\\left({_fmt_num(cosA, 6)}\\right) \\approx {_fmt_num(sol['A'])}^\\circ",
        ),
        _step(
            "კუთხე B — კოსინუსების თეორემა",
            "cos B = (a² + c² − b²) / (2ac)",
            "\\begin{aligned}"
            f"\\cos B &= \\dfrac{{a^2 + c^2 - b^2}}{{2ac}} \\\\"
            f"&= \\dfrac{{{_fmt_num(a*a)} + {_fmt_num(c*c)} - {_fmt_num(b*b)}}}"
            f"{{2 \\cdot {_fmt_num(a)} \\cdot {_fmt_num(c)}}} \\\\"
            f"&= {_fmt_num(cosB, 6)}"
            "\\end{aligned}",
        ),
        _step(
            "B კუთხის მნიშვნელობა",
            "ამოვიღოთ arccos.",
            f"B = \\arccos\\!\\left({_fmt_num(cosB, 6)}\\right) \\approx {_fmt_num(sol['B'])}^\\circ",
        ),
        _step(
            "კუთხე C",
            "მესამე კუთხე: 180° − A − B.",
            f"C = 180^\\circ - A - B \\approx {_fmt_num(sol['C'])}^\\circ",
        ),
        _step(
            "ფართობი — ჰერონის ფორმულა",
            "S = √(s(s−a)(s−b)(s−c)), სადაც s = (a+b+c)/2",
            "\\begin{aligned}"
            f"s &= \\dfrac{{a+b+c}}{{2}} = {_fmt_num(s_per)} \\\\"
            f"S &= \\sqrt{{s(s-a)(s-b)(s-c)}} \\approx {_fmt_num(sol['area'])}"
            "\\end{aligned}",
        ),
    ]


def _steps_sas(sides, angle_name, angle_val, sol):
    adj = {"A": ["b", "c"], "B": ["a", "c"], "C": ["a", "b"]}[angle_name]
    s1_name, s2_name = adj
    s1 = sides[s1_name]
    s2 = sides[s2_name]
    opp_name = angle_name.lower()
    opp_val = sol[opp_name]
    under = s1 * s1 + s2 * s2 - 2 * s1 * s2 * math.cos(math.radians(angle_val))
    return [
        _step(
            "მოცემული მონაცემები (SAS)",
            f"ცნობილია ორი გვერდი და მათ შორის კუთხე {angle_name}.",
            f"{s1_name} = {_fmt_num(s1)},\\quad {s2_name} = {_fmt_num(s2)},"
            f"\\quad \\angle {angle_name} = {_fmt_num(angle_val)}^\\circ",
        ),
        _step(
            f"მესამე გვერდი {opp_name} — კოსინუსების თეორემა",
            f"{opp_name}² = {s1_name}² + {s2_name}² − 2·{s1_name}·{s2_name}·cos({angle_name})",
            "\\begin{aligned}"
            f"{opp_name}^2 &= {_fmt_num(s1)}^2 + {_fmt_num(s2)}^2"
            f" - 2 \\cdot {_fmt_num(s1)} \\cdot {_fmt_num(s2)}"
            f" \\cdot \\cos({_fmt_num(angle_val)}^\\circ) \\\\"
            f"&= {_fmt_num(under)} \\\\"
            f"{opp_name} &\\approx {_fmt_num(opp_val)}"
            "\\end{aligned}",
        ),
        _step(
            "დარჩენილი კუთხეები",
            "ახლა სამივე გვერდი ცნობილია — ვიპოვოთ კუთხეები კოსინუსების თეორემით.",
            f"B \\approx {_fmt_num(sol['B'])}^\\circ,\\quad C \\approx {_fmt_num(sol['C'])}^\\circ",
        ),
        _step(
            "ფართობი",
            "S = ½ · b · c · sin A",
            f"S = \\tfrac{{1}}{{2}} \\cdot {_fmt_num(sides['b'])} \\cdot {_fmt_num(sides['c'])}"
            f" \\cdot \\sin({_fmt_num(angle_val)}^\\circ) \\approx {_fmt_num(sol['area'])}",
        ),
    ]


def _steps_aas_asa(a, b, c, A, B, C, sol):
    known_sides = [(k, v) for k, v in [("a", a), ("b", b), ("c", c)] if v is not None]
    known_angles = [(k, v) for k, v in [("A", A), ("B", B), ("C", C)] if v is not None]

    given = []
    for k, v in known_sides:
        given.append(f"{k} = {_fmt_num(v)}")
    for k, v in known_angles:
        given.append(f"\\angle {k} = {_fmt_num(v)}^\\circ")

    steps = [
        _step(
            "მოცემული მონაცემები",
            "ცნობილია ორი კუთხე და ერთი გვერდი.",
            ",\\quad ".join(given),
        )
    ]

    if len(known_angles) == 2:
        known_set = set(k for k, _ in known_angles)
        missing = next(k for k in ["A", "B", "C"] if k not in known_set)
        total = sum(v for _, v in known_angles)
        steps.append(_step(
            f"მესამე კუთხე {missing}",
            "სამკუთხედის კუთხეების ჯამი 180°-ია.",
            f"\\angle {missing} = 180^\\circ - {_fmt_num(known_angles[0][1])}^\\circ"
            f" - {_fmt_num(known_angles[1][1])}^\\circ = {_fmt_num(180 - total)}^\\circ",
        ))

    steps.append(_step(
        "სინუსების თეორემა",
        "a / sin A = b / sin B = c / sin C",
        "\\dfrac{a}{\\sin A} = \\dfrac{b}{\\sin B} = \\dfrac{c}{\\sin C}",
    ))

    if known_sides:
        ref_name, ref_val = known_sides[0]
        ref_angle_val = sol[ref_name.upper()]
        for k in ["a", "b", "c"]:
            if k == ref_name:
                continue
            kA = k.upper()
            A_val = sol[kA]
            steps.append(_step(
                f"გვერდი {k}",
                f"სინუსების თეორემიდან: {k} = {ref_name} · sin({kA}) / sin({ref_name.upper()})",
                "\\begin{aligned}"
                f"{k} &= {_fmt_num(ref_val)}"
                f" \\cdot \\dfrac{{\\sin({_fmt_num(A_val)}^\\circ)}}"
                f"{{\\sin({_fmt_num(ref_angle_val)}^\\circ)}} \\\\"
                f"&\\approx {_fmt_num(sol[k])}"
                "\\end{aligned}",
            ))

    return steps


def _steps_ssa(sides, angle_name, angle_val, sol, which):
    opp_name = angle_name.lower()
    opp_val = sides[opp_name]
    other_name = next(
        k for k in ["a", "b", "c"] if k != opp_name and sides[k] is not None
    )
    other_val = sides[other_name]
    other_ang_name = other_name.upper()
    third_name = next(
        k for k in ["a", "b", "c"] if k not in (opp_name, other_name)
    )
    third_ang_name = third_name.upper()

    sin_ang = math.sin(math.radians(angle_val))
    sin_other = other_val * sin_ang / opp_val

    steps = [
        _step(
            "მოცემული მონაცემები (SSA)",
            "ცნობილია ორი გვერდი და ერთი კუთხე (არა მათ შორის).",
            f"{opp_name} = {_fmt_num(opp_val)},\\quad {other_name} = {_fmt_num(other_val)},"
            f"\\quad \\angle {angle_name} = {_fmt_num(angle_val)}^\\circ",
        ),
        _step(
            "სინუსების თეორემა",
            f"{opp_name} / sin({angle_name}) = {other_name} / sin({other_ang_name})",
            f"\\dfrac{{{opp_name}}}{{\\sin {angle_name}}} = "
            f"\\dfrac{{{other_name}}}{{\\sin {other_ang_name}}}",
        ),
        _step(
            f"{other_ang_name} — ვიპოვოთ sin",
            f"sin({other_ang_name}) = {other_name} · sin({angle_name}) / {opp_name}",
            "\\begin{aligned}"
            f"\\sin {other_ang_name} &= "
            f"\\dfrac{{{_fmt_num(other_val)} \\cdot \\sin {_fmt_num(angle_val)}^\\circ}}"
            f"{{{_fmt_num(opp_val)}}} \\\\"
            f"&= {_fmt_num(sin_other, 6)}"
            "\\end{aligned}",
        ),
    ]

    if which == 0:
        steps.append(_step(
            f"{other_ang_name} კუთხის მნიშვნელობა",
            "პირველი შემთხვევა — მახვილი კუთხე.",
            f"{other_ang_name} = \\arcsin\\!\\left({_fmt_num(sin_other, 6)}\\right)"
            f" \\approx {_fmt_num(sol[other_ang_name])}^\\circ",
        ))
    else:
        steps.append(_step(
            f"{other_ang_name} კუთხის მნიშვნელობა (ბლაგვი)",
            "მეორე შემთხვევა — ბლაგვი კუთხე (SSA-ის ბუნდოვანი შემთხვევა).",
            f"{other_ang_name} = 180^\\circ - \\arcsin\\!\\left({_fmt_num(sin_other, 6)}\\right)"
            f" \\approx {_fmt_num(sol[other_ang_name])}^\\circ",
        ))

    steps.append(_step(
        f"მესამე კუთხე {third_ang_name}",
        "კუთხეების ჯამი 180°.",
        f"{third_ang_name} = 180^\\circ - {angle_name} - {other_ang_name}"
        f" \\approx {_fmt_num(sol[third_ang_name])}^\\circ",
    ))

    steps.append(_step(
        f"მესამე გვერდი {third_name}",
        "ისევ სინუსების თეორემიდან.",
        f"{third_name} = {opp_name} \\cdot "
        f"\\dfrac{{\\sin {third_ang_name}}}{{\\sin {angle_name}}}"
        f" \\approx {_fmt_num(sol[third_name])}",
    ))

    return steps


def _detect_kind(a, b, c, A, B, C):
    n_s = sum(1 for x in [a, b, c] if x is not None)
    n_a = sum(1 for x in [A, B, C] if x is not None)
    if n_s == 3:
        return "SSS"
    if n_s == 2 and n_a >= 1:
        angles = {"A": A, "B": B, "C": C}
        ang_key = next(k for k in ["A", "B", "C"] if angles[k] is not None)
        sides = {"a": a, "b": b, "c": c}
        if sides[ang_key.lower()] is not None:
            return "SSA"
        return "SAS"
    if n_s == 1 and n_a >= 2:
        return "AAS_ASA"
    return "UNKNOWN"


# ═══════════════════════════════════════════════════════════════
#  Endpoint
# ═══════════════════════════════════════════════════════════════

@router.post("/solve")
def solve(data: TriangleInput):
    try:
        a = _parse_num(data.a)
        b = _parse_num(data.b)
        c = _parse_num(data.c)
        A = _parse_num(data.A)
        B = _parse_num(data.B)
        C = _parse_num(data.C)

        triangles = _dispatch(a, b, c, A, B, C)
        kind = _detect_kind(a, b, c, A, B, C)

        enriched = []
        for idx, t in enumerate(triangles):
            derived = _derive(t)
            full = {**t, **derived}

            try:
                if kind == "SSS":
                    steps = _steps_sss(a, b, c, full)
                elif kind == "SAS":
                    sides_in = {"a": a, "b": b, "c": c}
                    angles_in = {"A": A, "B": B, "C": C}
                    ang_key = next(
                        k for k in ["A", "B", "C"] if angles_in[k] is not None
                    )
                    steps = _steps_sas(sides_in, ang_key, angles_in[ang_key], full)
                elif kind == "AAS_ASA":
                    steps = _steps_aas_asa(a, b, c, A, B, C, full)
                elif kind == "SSA":
                    sides_in = {"a": a, "b": b, "c": c}
                    angles_in = {"A": A, "B": B, "C": C}
                    ang_key = next(
                        k for k in ["A", "B", "C"] if angles_in[k] is not None
                    )
                    steps = _steps_ssa(
                        sides_in, ang_key, angles_in[ang_key], full, idx
                    )
                else:
                    steps = []
            except Exception as e:
                steps = [_step("ნაბიჯები ვერ დაგენერირდა", str(e), "\\text{---}")]

            full["steps"] = steps
            full["kind"] = kind
            enriched.append(full)

        return {
            "solutions": enriched,
            "ambiguous": len(enriched) > 1,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"შეცდომა: {e}")