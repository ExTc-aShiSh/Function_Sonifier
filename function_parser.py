"""
function_parser.py — Safe mathematical expression parsing and evaluation.

Uses SymPy to parse user-entered mathematical expressions and NumPy
for efficient numerical evaluation. Includes function analysis utilities
for detecting discontinuities, periodicity, and computing statistics.
"""

import numpy as np
from sympy import (
    sympify, Symbol, lambdify, oo, zoo, nan, pi,
    sin, cos, tan, exp, log, sqrt, Abs,
    Piecewise, S, periodicity, simplify
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from typing import Optional
from dataclasses import dataclass


# Symbol used across all expressions
x = Symbol('x')

# Allowed transformations for parsing
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Safe namespace for evaluation — prevents code injection
SAFE_NAMESPACE = {
    'sin': sin, 'cos': cos, 'tan': tan,
    'exp': exp, 'log': log, 'sqrt': sqrt,
    'abs': Abs, 'pi': pi, 'e': S.Exp1,
    'x': x,
}


@dataclass
class FunctionAnalysis:
    """Results of analyzing a mathematical function."""
    max_value: float
    min_value: float
    num_discontinuities: int
    approximate_period: Optional[float]


@dataclass
class ParseResult:
    """Result of parsing a mathematical expression."""
    success: bool
    error_message: str = ""
    y_values: Optional[np.ndarray] = None
    x_values: Optional[np.ndarray] = None
    expression_str: str = ""


def parse_function(expr_str: str) -> tuple[bool, object, str]:
    """
    Safely parse a mathematical expression string into a SymPy expression.

    Args:
        expr_str: The mathematical expression as a string (e.g., "sin(x)").

    Returns:
        Tuple of (success, sympy_expression, error_message).
    """
    try:
        # Strip whitespace
        expr_str = expr_str.strip()
        if not expr_str:
            return False, None, "Expression cannot be empty."

        # Parse with safe transformations
        expr = parse_expr(
            expr_str,
            local_dict={'x': x},
            transformations=TRANSFORMATIONS,
        )

        # Verify the expression only contains 'x' as a free symbol
        free_syms = expr.free_symbols
        if free_syms and free_syms != {x}:
            unknown = ', '.join(str(s) for s in free_syms if s != x)
            return False, None, f"Unknown variable(s): {unknown}. Use 'x' only."

        return True, expr, ""

    except Exception as e:
        return False, None, f"Parse error: {str(e)}"


def evaluate_function(
    expr_str: str,
    x_start: float,
    x_end: float,
    num_samples: int
) -> ParseResult:
    """
    Parse and evaluate a mathematical function over a specified range.

    Args:
        expr_str:    The function expression string.
        x_start:     Start of x range.
        x_end:       End of x range.
        num_samples: Number of evaluation points.

    Returns:
        ParseResult containing x_values, y_values, or error information.
    """
    success, expr, error = parse_function(expr_str)
    if not success:
        return ParseResult(success=False, error_message=error)

    try:
        # Create numerical function using lambdify
        f = lambdify(x, expr, modules=['numpy'])

        # Generate x values
        x_vals = np.linspace(x_start, x_end, num_samples)

        # Evaluate — catch runtime warnings
        with np.errstate(all='ignore'):
            y_vals = f(x_vals)

        # Handle scalar results (constant functions like "5")
        if np.isscalar(y_vals):
            y_vals = np.full_like(x_vals, float(y_vals))

        # Convert to float array
        y_vals = np.array(y_vals, dtype=np.float64)

        # Replace inf / nan with interpolated or clamped values
        y_vals = _sanitize_values(y_vals)

        return ParseResult(
            success=True,
            x_values=x_vals,
            y_values=y_vals,
            expression_str=str(expr),
        )

    except Exception as e:
        return ParseResult(success=False, error_message=f"Evaluation error: {str(e)}")


def _sanitize_values(y: np.ndarray) -> np.ndarray:
    """
    Replace inf and NaN values with interpolated neighbors to prevent
    audio clipping and graphing artifacts.

    Args:
        y: Array of y-values that may contain inf/NaN.

    Returns:
        Sanitized array with problematic values replaced.
    """
    # Create mask of invalid values
    invalid = ~np.isfinite(y)

    if not np.any(invalid):
        return y

    # If all values are invalid, return zeros
    if np.all(invalid):
        return np.zeros_like(y)

    # Interpolate invalid values from valid neighbors
    valid_indices = np.where(~invalid)[0]
    invalid_indices = np.where(invalid)[0]

    y[invalid_indices] = np.interp(
        invalid_indices,
        valid_indices,
        y[valid_indices]
    )

    return y


def analyze_function(
    expr_str: str,
    x_vals: np.ndarray,
    y_vals: np.ndarray
) -> Optional[FunctionAnalysis]:
    """
    Perform analysis on a function: compute stats, detect discontinuities,
    and estimate periodicity.

    Args:
        expr_str: The function expression string.
        x_vals:   Array of x values.
        y_vals:   Array of evaluated y values.

    Returns:
        FunctionAnalysis dataclass with computed properties.
    """
    try:
        # Basic statistics
        max_val = float(np.nanmax(y_vals))
        min_val = float(np.nanmin(y_vals))

        # Detect discontinuities — large jumps relative to the range
        dy = np.abs(np.diff(y_vals))
        y_range = max_val - min_val if max_val != min_val else 1.0
        threshold = y_range * 0.3  # 30% of range = likely discontinuity
        num_disc = int(np.sum(dy > threshold))

        # Attempt to find period using SymPy
        approx_period: Optional[float] = None
        try:
            success, expr, _ = parse_function(expr_str)
            if success and expr is not None:
                period = periodicity(expr, x)
                if period is not None and period != oo:
                    approx_period = float(period.evalf())
        except Exception:
            pass  # Period detection is best-effort

        return FunctionAnalysis(
            max_value=round(max_val, 6),
            min_value=round(min_val, 6),
            num_discontinuities=num_disc,
            approximate_period=round(approx_period, 6) if approx_period else None,
        )

    except Exception:
        return None
