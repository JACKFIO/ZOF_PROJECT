"""
ZOF_CLI.py - Zero of Functions Solver (Command Line Interface)
Implements 6 numerical methods for finding roots of nonlinear equations
"""

import math
import numpy as np
from typing import Callable, Tuple, List, Dict

class ZOFSolver:
    """Zero of Functions Solver with 6 numerical methods"""
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iteration_data = []
    
    def parse_function(self, func_str: str) -> Callable:
        """Convert string equation to callable function"""
        def f(x):
            return eval(func_str, {"x": x, "math": math, "np": np, 
                                   "sin": math.sin, "cos": math.cos, 
                                   "tan": math.tan, "exp": math.exp, 
                                   "log": math.log, "sqrt": math.sqrt})
        return f
    
    def parse_derivative(self, func_str: str) -> Callable:
        """Numerical derivative for Newton-Raphson"""
        f = self.parse_function(func_str)
        def df(x, h=1e-7):
            return (f(x + h) - f(x - h)) / (2 * h)
        return df
    
    def bisection_method(self, f: Callable, a: float, b: float) -> Dict:
        """Bisection Method"""
        self.iteration_data = []
        
        if f(a) * f(b) >= 0:
            return {"error": "f(a) and f(b) must have opposite signs"}
        
        for i in range(self.max_iterations):
            c = (a + b) / 2
            fc = f(c)
            error = abs(b - a) / 2
            
            self.iteration_data.append({
                "iteration": i + 1,
                "a": a,
                "b": b,
                "c": c,
                "f(c)": fc,
                "error": error
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
        """Regula Falsi (False Position) Method"""
        self.iteration_data = []
        
        if f(a) * f(b) >= 0:
            return {"error": "f(a) and f(b) must have opposite signs"}
        
        c_old = a
        for i in range(self.max_iterations):
            fa, fb = f(a), f(b)
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            
            error = abs(c - c_old) if i > 0 else abs(b - a)
            
            self.iteration_data.append({
                "iteration": i + 1,
                "a": a,
                "b": b,
                "c": c,
                "f(c)": fc,
                "error": error
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
            f0, f1 = f(x0), f(x1)
            
            if abs(f1 - f0) < 1e-12:
                return {"error": "Division by zero in secant method"}
            
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            f2 = f(x2)
            error = abs(x2 - x1)
            
            self.iteration_data.append({
                "iteration": i + 1,
                "x0": x0,
                "x1": x1,
                "x2": x2,
                "f(x2)": f2,
                "error": error
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
            fx = f(x)
            dfx = df(x)
            
            if abs(dfx) < 1e-12:
                return {"error": "Derivative too close to zero"}
            
            x_new = x - fx / dfx
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i + 1,
                "x": x,
                "f(x)": fx,
                "f'(x)": dfx,
                "x_new": x_new,
                "error": error
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
        """Fixed Point Iteration Method"""
        self.iteration_data = []
        x = x0
        
        for i in range(self.max_iterations):
            x_new = g(x)
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i + 1,
                "x": x,
                "g(x)": x_new,
                "error": error
            })
            
            if error < self.tolerance:
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
    
    def modified_secant_method(self, f: Callable, x0: float, delta: float = 0.01) -> Dict:
        """Modified Secant Method"""
        self.iteration_data = []
        x = x0
        
        for i in range(self.max_iterations):
            fx = f(x)
            fx_delta = f(x + delta * x) if x != 0 else f(x + delta)
            
            denominator = fx_delta - fx
            if abs(denominator) < 1e-12:
                return {"error": "Division by zero in modified secant"}
            
            x_new = x - (delta * x * fx) / denominator if x != 0 else x - (delta * fx) / denominator
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i + 1,
                "x": x,
                "f(x)": fx,
                "x_new": x_new,
                "error": error
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


def print_iteration_table(data: List[Dict]):
    """Print iteration data in table format"""
    if not data:
        return
    
    print("\n" + "="*80)
    print("ITERATION DETAILS")
    print("="*80)
    
    headers = list(data[0].keys())
    
    # Print headers
    header_str = " | ".join(f"{h:>12}" for h in headers)
    print(header_str)
    print("-" * len(header_str))
    
    # Print data rows
    for row in data:
        row_str = " | ".join(f"{row[h]:>12.6e}" if isinstance(row[h], float) else f"{row[h]:>12}" 
                            for h in headers)
        print(row_str)
    print("="*80)


def main():
    """Main CLI interface"""
    print("\n" + "="*80)
    print("ZERO OF FUNCTIONS (ZOF) SOLVER - CLI Application")
    print("="*80)
    
    methods = {
        "1": "Bisection Method",
        "2": "Regula Falsi Method",
        "3": "Secant Method",
        "4": "Newton-Raphson Method",
        "5": "Fixed Point Iteration",
        "6": "Modified Secant Method"
    }
    
    print("\nAvailable Methods:")
    for key, name in methods.items():
        print(f"{key}. {name}")
    
    choice = input("\nSelect method (1-6): ").strip()
    
    if choice not in methods:
        print("Invalid choice!")
        return
    
    print(f"\n--- {methods[choice]} ---")
    
    func_str = input("Enter function f(x) [e.g., x**3 - x - 2]: ")
    tolerance = float(input("Enter tolerance [default 1e-6]: ") or 1e-6)
    max_iter = int(input("Enter max iterations [default 100]: ") or 100)
    
    solver = ZOFSolver(tolerance=tolerance, max_iterations=max_iter)
    
    try:
        f = solver.parse_function(func_str)
        
        if choice == "1":  # Bisection
            a = float(input("Enter lower bound a: "))
            b = float(input("Enter upper bound b: "))
            result = solver.bisection_method(f, a, b)
        
        elif choice == "2":  # Regula Falsi
            a = float(input("Enter lower bound a: "))
            b = float(input("Enter upper bound b: "))
            result = solver.regula_falsi_method(f, a, b)
        
        elif choice == "3":  # Secant
            x0 = float(input("Enter first initial guess x0: "))
            x1 = float(input("Enter second initial guess x1: "))
            result = solver.secant_method(f, x0, x1)
        
        elif choice == "4":  # Newton-Raphson
            x0 = float(input("Enter initial guess x0: "))
            df = solver.parse_derivative(func_str)
            result = solver.newton_raphson_method(f, df, x0)
        
        elif choice == "5":  # Fixed Point
            g_str = input("Enter g(x) for x = g(x) [e.g., (x + 2)**(1/3)]: ")
            g = solver.parse_function(g_str)
            x0 = float(input("Enter initial guess x0: "))
            result = solver.fixed_point_iteration(g, x0)
        
        elif choice == "6":  # Modified Secant
            x0 = float(input("Enter initial guess x0: "))
            delta = float(input("Enter delta value [default 0.01]: ") or 0.01)
            result = solver.modified_secant_method(f, x0, delta)
        
        if "error" in result:
            print(f"\nError: {result['error']}")
            return
        
        # Print iteration table
        print_iteration_table(result["data"])
        
        # Print final results
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"Estimated Root: {result['root']:.10f}")
        print(f"Final Error: {result['error']:.10e}")
        print(f"Number of Iterations: {result['iterations']}")
        if "warning" in result:
            print(f"Warning: {result['warning']}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    while True:
        main()
        again = input("Solve another equation? (y/n): ").strip().lower()
        if again != 'y':
            print("\nThank you for using ZOF Solver!")
            break