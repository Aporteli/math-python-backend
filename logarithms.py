import re, sympy as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

router = APIRouter(prefix='/api/logarithm', tags=['logarithm'])
T = standard_transformations + (implicit_multiplication_application, convert_xor)


def _parse(s, x):
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'log_\{([^{}]+)\}\(([^()]*)\)', r'log(\2,\1)', s)  # log_{2/3}(x) → log(x,2/3)
    s = re.sub(r'log_(\w+)\(([^()]*)\)', r'log(\2,\1)', s)          # log_2(x) → log(x,2)
    s = re.sub(r'\blog\(([^,()]+)\)', r'log10(\1)', s)              # log(x) → log10(x)
    return parse_expr(s, local_dict={str(x): x, 'log': sp.log, 'log10': lambda a: sp.log(a, 10),
                                     'ln': sp.log, 'e': sp.E, 'pi': sp.pi, 'sqrt': sp.sqrt}, transformations=T)


class In(BaseModel):
    expression: str = 'log_2(x-1)=3'
    variable: str = 'x'


@router.post('/analyze')
def analyze(d: In):
    try:
        x = sp.Symbol(d.variable)
        if '=' in d.expression:
            a, b = d.expression.split('=', 1)
            l, r = _parse(a, x), _parse(b, x)
            if sp.simplify(l - r) == 0:
                return {'mode': 'solve', 'identity': True, 'inputLatex': f'{a.strip()} = {b.strip()}'}
            sols = sp.solve(sp.Eq(l, r), x) or []
            if not isinstance(sols, list): sols = [sols]
            args = [t.args[0] for t in (l.atoms(sp.log) | r.atoms(sp.log)) if t.args[0].free_symbols]
            valid, rejected = [], []
            for s in sols:
                if not s.is_real: continue
                (valid if all(sp.N(arg.subs(x, s)) > 0 for arg in args) else rejected).append(s)
            return {'mode': 'solve', 'identity': False, 'inputLatex': f'{a.strip()} = {b.strip()}',
                    'domainLatex': ',\\; '.join(f'{sp.latex(t)} > 0' for t in args),
                    'solutions': [{'latex': sp.latex(s), 'numeric': float(sp.N(s))} for s in valid],
                    'rejected': [{'latex': sp.latex(s), 'numeric': float(sp.N(s))} for s in rejected]}
        e = _parse(d.expression, x)
        return {'mode': 'simplify', 'inputLatex': d.expression.strip(), 'simplifiedLatex': sp.latex(sp.simplify(e))}
    except Exception as err:
        raise HTTPException(400, str(err))