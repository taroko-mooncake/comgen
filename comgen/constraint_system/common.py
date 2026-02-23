"""
Shared constraint-building primitives.

These small helper functions are used throughout the constraint_system
sub-package to create Z3 expressions for bounds checking, weighted sums,
and piece-wise linear functions (Abs and ReLU).
"""

from z3 import And, Sum, If


def check_bounds(exact, lb, ub):
    """Validate that exactly one style of bound specification is used.

    Raises :class:`AssertionError` if all three are ``None``, or if *exact*
    is combined with *lb* or *ub*.

    Args:
        exact: Exact equality value, or ``None``.
        lb: Lower bound, or ``None``.
        ub: Upper bound, or ``None``.
    """
    bounds = [exact, lb, ub]
    assert not all([b is None for b in bounds])
    assert not (exact is not None and lb is not None)
    assert not (exact is not None and ub is not None)


def apply_bounds(var, exact=None, *, lb=None, ub=None):
    """Return a Z3 constraint that bounds *var*.

    Supports three modes:

    * ``exact`` -- returns ``var == exact``
    * ``lb`` and/or ``ub`` -- returns the conjunction of ``var >= lb``
      and ``var <= ub`` (whichever are provided).

    Args:
        var: A Z3 arithmetic expression.
        exact: If given, the variable must equal this value.
        lb: Optional lower bound (inclusive).
        ub: Optional upper bound (inclusive).

    Returns:
        A Z3 ``BoolRef`` expression.
    """
    check_bounds(exact, lb, ub)
    
    if exact is not None:
        return var == exact
    
    constraints = []
    if lb is not None:
        constraints.append(var >= lb)
    if ub is not None:
        constraints.append(var <= ub)
    
    if len(constraints) > 1:
        return And(constraints)
    return constraints[0]


def zero_weighted_sum(vars, weights):
    """Shorthand for ``weighted_sum(vars, weights, exact=0)``.

    Args:
        vars: Sequence of Z3 variables.
        weights: Corresponding numeric weights.

    Returns:
        A Z3 constraint requiring the weighted sum to equal zero.
    """
    return weighted_sum(vars, weights, 0)


def weighted_sum(vars, weights, exact=None, *, lb=None, ub=None):
    """Return a Z3 constraint bounding the weighted sum of *vars*.

    Args:
        vars: Sequence of Z3 variables.
        weights: Numeric weights (same length as *vars*).
        exact: If given, the weighted sum must equal this value.
        lb: Optional lower bound on the sum (inclusive).
        ub: Optional upper bound on the sum (inclusive).

    Returns:
        A Z3 ``BoolRef`` expression.
    """
    assert len(vars) == len(weights)
    weighted_vars = [v * w for v, w in zip(vars, weights)]
    return apply_bounds(Sum(weighted_vars), exact, lb=lb, ub=ub)


def bound_weighted_average_value_ratio(weight_vars1, weight_vars2, values1, values2, exact=None, *, lb=None, ub=None):
    """Bound the ratio of two weighted averages.

    Computes ``(Σ w1·v1 / Σ w1) / (Σ w2·v2 / Σ w2)`` and constrains it.
    All weight variables are assumed to be non-negative.

    Args:
        weight_vars1: Z3 variables for the first group's weights.
        weight_vars2: Z3 variables for the second group's weights.
        values1: Numeric values for the first group.
        values2: Numeric values for the second group.
        exact: If given, the ratio must equal this value.
        lb: Optional lower bound on the ratio.
        ub: Optional upper bound on the ratio.

    Returns:
        A Z3 ``BoolRef`` expression.
    """
    ratio = weighted_sum(weight_vars1, values1) * Sum(weight_vars2) / weighted_sum(weight_vars2, values2) / Sum(weight_vars1)
    return apply_bounds(ratio, exact, lb=lb, ub=ub)


def Abs(x):
    """Z3-compatible absolute value: ``If(x >= 0, x, -x)``."""
    return If(x >= 0, x, -x)


def ReLU(x):
    """Z3-compatible ReLU activation: ``If(x >= 0, x, 0)``."""
    return If(x >= 0, x, 0)
