from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sympy as sp
import numpy as np

app = FastAPI(title="Math Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://math-site.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        t = sp.Symbol('t')
        
        # უსაფრთხო გარდაქმნა
        x_sym = sp.sympify(data.x_expr)
        y_sym = sp.sympify(data.y_expr)
        z_sym = sp.sympify(data.z_expr)

        x_func = sp.lambdify(t, x_sym, modules=['numpy'])
        y_func = sp.lambdify(t, y_sym, modules=['numpy'])
        z_func = sp.lambdify(t, z_sym, modules=['numpy'])

        t_vals = np.linspace(data.t_min, data.t_max, data.num_points)
        
        x_vals = x_func(t_vals)
        y_vals = y_func(t_vals)
        z_vals = z_func(t_vals)

        # მუდმივების შემოწმება
        if np.isscalar(x_vals): x_vals = np.full_like(t_vals, x_vals)
        if np.isscalar(y_vals): y_vals = np.full_like(t_vals, y_vals)
        if np.isscalar(z_vals): z_vals = np.full_like(t_vals, z_vals)

        # NaN/Inf მნიშვნელობების გასუფთავება
        x_vals = np.nan_to_num(x_vals, nan=0.0, posinf=100.0, neginf=-100.0)
        y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=100.0, neginf=-100.0)
        z_vals = np.nan_to_num(z_vals, nan=0.0, posinf=100.0, neginf=-100.0)

        max_r = max(float(np.max(np.abs(x_vals))), float(np.max(np.abs(y_vals))), 3.0)
        max_z = float(np.max(np.abs(z_vals))) if np.max(np.abs(z_vals)) > 0 else 3.0

        # XY ბადე
        grid_lines = []
        grid_range = np.linspace(-max_r, max_r, 9)
        for r in grid_range:
            grid_lines.append({"x": [-max_r, max_r], "y": [float(r), float(r)], "z": [0.0, 0.0]})
            grid_lines.append({"x": [float(r), float(r)], "y": [-max_r, max_r], "z": [0.0, 0.0]})

        return {
            "curve": {
                "x": x_vals.tolist(),
                "y": y_vals.tolist(),
                "z": z_vals.tolist()
            },
            "grid_lines": grid_lines,
            "bounds": {"max_r": max_r, "max_z": max_z}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))