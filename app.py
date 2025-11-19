"""
app.py - Zero of Functions Solver Web Application (Vercel Optimized)
No NumPy dependency - uses only standard library
"""

from flask import Flask, render_template, request, jsonify
import math
from typing import Callable, Dict, List

app = Flask(__name__)

class ZOFSolver:
    """Zero of Functions Solver with 6 numerical methods"""
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iteration_data = []
    
    def parse_function(self, func_str: str) -> Callable:
        """Convert string equation to callable function"""
        # Create safe namespace with math functions
        safe_dict = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
            "pi": math.pi, "e": math.e, "abs": abs,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "pow": pow
        }
        
        def f(x):
            safe_dict["x"] = x
            return eval(func_str, {"__builtins__": {}}, safe_dict)
        return f
    
    def parse_derivative(self, func_str: str) -> Callable:
        """Numerical derivative"""
        f = self.parse_function(func_str)
        def df(x, h=1e-7):
            return (f(x + h) - f(x - h)) / (2 * h)
        return df
    
    def bisection_method(self, f: Callable, a: float, b: float) -> Dict:
        """Bisection Method"""
        self.iteration_data = []
        
        try:
            fa, fb = f(a), f(b)
        except Exception as e:
            return {"error": f"Error evaluating function: {str(e)}"}
        
        if fa * fb >= 0:
            return {"error": "f(a) and f(b) must have opposite signs"}
        
        for i in range(self.max_iterations):
            c = (a + b) / 2
            fc = f(c)
            error = abs(b - a) / 2
            
            self.iteration_data.append({
                "iteration": i + 1,
                "a": round(a, 8),
                "b": round(b, 8),
                "c": round(c, 8),
                "f(c)": round(fc, 8),
                "error": round(error, 10)
            })
            
            if error < self.tolerance or abs(fc) < self.tolerance:
                return {
                    "root": c,
                    "iterations": i + 1,
                    "error": error,
                    "data": self.iteration_data
                }
            
            if f(a) * fc < 0:
                b = c
            else:
                a = c
        
        return {
            "root": c,
            "iterations": self.max_iterations,
            "error": error,
            "data": self.iteration_data,
            "warning": "Max iterations reached"
        }
    
    def regula_falsi_method(self, f: Callable, a: float, b: float) -> Dict:
        """Regula Falsi Method"""
        self.iteration_data = []
        
        try:
            fa, fb = f(a), f(b)
        except Exception as e:
            return {"error": f"Error evaluating function: {str(e)}"}
        
        if fa * fb >= 0:
            return {"error": "f(a) and f(b) must have opposite signs"}
        
        c_old = a
        for i in range(self.max_iterations):
            fa, fb = f(a), f(b)
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            error = abs(c - c_old) if i > 0 else abs(b - a)
            
            self.iteration_data.append({
                "iteration": i + 1,
                "a": round(a, 8),
                "b": round(b, 8),
                "c": round(c, 8),
                "f(c)": round(fc, 8),
                "error": round(error, 10)
            })
            
            if error < self.tolerance or abs(fc) < self.tolerance:
                return {
                    "root": c,
                    "iterations": i + 1,
                    "error": error,
                    "data": self.iteration_data
                }
            
            if fa * fc < 0:
                b = c
            else:
                a = c
            
            c_old = c
        
        return {
            "root": c,
            "iterations": self.max_iterations,
            "error": error,
            "data": self.iteration_data,
            "warning": "Max iterations reached"
        }
    
    def secant_method(self, f: Callable, x0: float, x1: float) -> Dict:
        """Secant Method"""
        self.iteration_data = []
        
        for i in range(self.max_iterations):
            try:
                f0, f1 = f(x0), f(x1)
            except Exception as e:
                return {"error": f"Error evaluating function: {str(e)}"}
            
            if abs(f1 - f0) < 1e-12:
                return {"error": "Division by zero - f(x1) ≈ f(x0)"}
            
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            
            try:
                f2 = f(x2)
            except Exception as e:
                return {"error": f"Error evaluating function at x2: {str(e)}"}
            
            error = abs(x2 - x1)
            
            self.iteration_data.append({
                "iteration": i + 1,
                "x0": round(x0, 8),
                "x1": round(x1, 8),
                "x2": round(x2, 8),
                "f(x2)": round(f2, 8),
                "error": round(error, 10)
            })
            
            if error < self.tolerance or abs(f2) < self.tolerance:
                return {
                    "root": x2,
                    "iterations": i + 1,
                    "error": error,
                    "data": self.iteration_data
                }
            
            x0, x1 = x1, x2
        
        return {
            "root": x2,
            "iterations": self.max_iterations,
            "error": error,
            "data": self.iteration_data,
            "warning": "Max iterations reached"
        }
    
    def newton_raphson_method(self, f: Callable, df: Callable, x0: float) -> Dict:
        """Newton-Raphson Method"""
        self.iteration_data = []
        x = x0
        
        for i in range(self.max_iterations):
            try:
                fx = f(x)
                dfx = df(x)
            except Exception as e:
                return {"error": f"Error evaluating function: {str(e)}"}
            
            if abs(dfx) < 1e-12:
                return {"error": "Derivative too close to zero"}
            
            x_new = x - fx / dfx
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i + 1,
                "x": round(x, 8),
                "f(x)": round(fx, 8),
                "f'(x)": round(dfx, 8),
                "x_new": round(x_new, 8),
                "error": round(error, 10)
            })
            
            if error < self.tolerance or abs(fx) < self.tolerance:
                return {
                    "root": x_new,
                    "iterations": i + 1,
                    "error": error,
                    "data": self.iteration_data
                }
            
            x = x_new
        
        return {
            "root": x,
            "iterations": self.max_iterations,
            "error": error,
            "data": self.iteration_data,
            "warning": "Max iterations reached"
        }
    
    def fixed_point_iteration(self, g: Callable, x0: float) -> Dict:
        """Fixed Point Iteration"""
        self.iteration_data = []
        x = x0
        
        for i in range(self.max_iterations):
            try:
                x_new = g(x)
            except Exception as e:
                return {"error": f"Error evaluating g(x): {str(e)}"}
            
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i + 1,
                "x": round(x, 8),
                "g(x)": round(x_new, 8),
                "error": round(error, 10)
            })
            
            if error < self.tolerance:
                return {
                    "root": x_new,
                    "iterations": i + 1,
                    "error": error,
                    "data": self.iteration_data
                }
            
            # Check for divergence
            if abs(x_new) > 1e10:
                return {"error": "Method diverging - try different initial guess or g(x)"}
            
            x = x_new
        
        return {
            "root": x,
            "iterations": self.max_iterations,
            "error": error,
            "data": self.iteration_data,
            "warning": "Max iterations reached"
        }
    
    def modified_secant_method(self, f: Callable, x0: float, delta: float = 0.01) -> Dict:
        """Modified Secant Method"""
        self.iteration_data = []
        x = x0
        
        for i in range(self.max_iterations):
            try:
                fx = f(x)
                fx_delta = f(x + delta * x) if x != 0 else f(x + delta)
            except Exception as e:
                return {"error": f"Error evaluating function: {str(e)}"}
            
            denominator = fx_delta - fx
            if abs(denominator) < 1e-12:
                return {"error": "Division by zero in modified secant"}
            
            x_new = x - (delta * x * fx) / denominator if x != 0 else x - (delta * fx) / denominator
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i + 1,
                "x": round(x, 8),
                "f(x)": round(fx, 8),
                "x_new": round(x_new, 8),
                "error": round(error, 10)
            })
            
            if error < self.tolerance or abs(fx) < self.tolerance:
                return {
                    "root": x_new,
                    "iterations": i + 1,
                    "error": error,
                    "data": self.iteration_data
                }
            
            x = x_new
        
        return {
            "root": x,
            "iterations": self.max_iterations,
            "error": error,
            "data": self.iteration_data,
            "warning": "Max iterations reached"
        }


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve():
    """Solve equation using selected method"""
    try:
        data = request.json
        method = data.get('method')
        function = data.get('function')
        tolerance = float(data.get('tolerance', 1e-6))
        max_iter = int(data.get('max_iterations', 100))
        
        # Validate inputs
        if not method or not function:
            return jsonify({"error": "Method and function are required"}), 400
        
        solver = ZOFSolver(tolerance=tolerance, max_iterations=max_iter)
        
        try:
            f = solver.parse_function(function)
        except Exception as e:
            return jsonify({"error": f"Invalid function syntax: {str(e)}"}), 400
        
        result = None
        
        if method == 'bisection':
            a = float(data.get('a'))
            b = float(data.get('b'))
            result = solver.bisection_method(f, a, b)
        
        elif method == 'regula_falsi':
            a = float(data.get('a'))
            b = float(data.get('b'))
            result = solver.regula_falsi_method(f, a, b)
        
        elif method == 'secant':
            x0 = float(data.get('x0'))
            x1 = float(data.get('x1'))
            result = solver.secant_method(f, x0, x1)
        
        elif method == 'newton_raphson':
            x0 = float(data.get('x0'))
            df = solver.parse_derivative(function)
            result = solver.newton_raphson_method(f, df, x0)
        
        elif method == 'fixed_point':
            g_function = data.get('g_function')
            if not g_function:
                return jsonify({"error": "g(x) function is required for Fixed Point method"}), 400
            g = solver.parse_function(g_function)
            x0 = float(data.get('x0'))
            result = solver.fixed_point_iteration(g, x0)
        
        elif method == 'modified_secant':
            x0 = float(data.get('x0'))
            delta = float(data.get('delta', 0.01))
            result = solver.modified_secant_method(f, x0, delta)
        
        else:
            return jsonify({"error": "Invalid method selected"}), 400
        
        if result and "error" not in result:
            result['root'] = round(result['root'], 10)
            result['error'] = round(result['error'], 10)
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# For Vercel deployment
app.debug = False

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
