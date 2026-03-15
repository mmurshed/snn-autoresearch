"""Numerical verification of surrogate gradient candidates.

Four hard checks — all must pass for a candidate to be valid:
  1. Non-negativity: g(x) >= 0 for all x in [-20, 20]
  2. Bounded integral: integral of |g(x)| over [-100, 100] < 1000
  3. Locality: g(x) → 0 for large |x| (|g(±30)|, |g(±50)| < 0.01)
  4. Numerical stability: no NaN/Inf over [-10, 10]
"""

from __future__ import annotations

import numpy as np

from .candidate import SurrogateCandidate


def verify(candidate: SurrogateCandidate) -> dict:
    """Verify a candidate passes all numerical checks.

    Returns dict with 'valid' bool and individual check results.
    """
    fn = _build_fn(candidate)
    if fn is None:
        return {"valid": False, "error": "failed to build function"}

    x_wide = np.linspace(-20, 20, 1000)
    try:
        y_wide = fn(x_wide)
    except Exception as e:
        return {"valid": False, "error": str(e)}

    # Check 1: Non-negativity
    non_negative = bool(np.min(y_wide) >= -1e-6)

    # Check 2: Bounded integral
    x_int = np.linspace(-100, 100, 2000)
    try:
        y_int = fn(x_int)
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        integral = float(_trapz(np.abs(y_int), x_int))
        bounded = integral < 1000
    except Exception:
        bounded = False

    # Check 3: Locality (tail decay)
    try:
        tail_points = np.array([-50.0, -30.0, 30.0, 50.0])
        tail_vals = fn(tail_points)
        local = bool(np.all(np.abs(tail_vals) < 0.01))
    except Exception:
        local = False

    # Check 4: Numerical stability
    x_dense = np.linspace(-10, 10, 200)
    try:
        y_dense = fn(x_dense)
        stable = bool(not (np.any(np.isnan(y_dense)) or np.any(np.isinf(y_dense))))
    except Exception:
        stable = False

    checks = {
        "non_negative": non_negative,
        "bounded_integral": bounded,
        "locality": local,
        "stable": stable,
    }
    checks["valid"] = all(checks.values())
    return checks


def _build_fn(candidate: SurrogateCandidate):
    """Build a numpy function from a candidate's python_expr."""
    namespace = {
        "__builtins__": {},
        "np": np,
        "abs": np.abs,
        "exp": np.exp,
        "tanh": np.tanh,
        "clip": np.clip,
        "pi": np.pi,
        **candidate.params,
    }
    try:
        fn = eval(f"lambda x: {candidate.python_expr}", namespace)
        # Smoke test
        fn(np.array([0.0]))
        return fn
    except Exception:
        return None
