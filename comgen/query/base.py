"""
Base query class that wraps the Z3 SMT solver.

Provides the core solve loop: accumulate constraints, invoke the solver, and
iterate over satisfying models while tracking monitored (return) variables.
"""

from z3 import sat, unknown, Solver, Q
from fractions import Fraction
from z3.z3 import RatNumRef, BoolRef, IntNumRef


class Query:
    """Low-level wrapper around a Z3 :class:`Solver`.

    Subclasses populate :pyattr:`constraints` with Z3 Boolean expressions and
    :pyattr:`return_vars` with Z3 variables whose values should be extracted
    from every satisfying model.

    Attributes:
        constraints: List of Z3 Boolean expressions that define the problem.
        solutions: List of Z3 models found so far (one per ``get_next`` call).
        return_vars: Z3 variables to evaluate and return from each model.
    """

    def __init__(self):
        self.constraints = []
        self.solutions = []
        self.return_vars = []

    def frac_to_rational(self, val):
        """Convert a :class:`~fractions.Fraction` to a Z3 rational literal.

        Non-Fraction values are returned unchanged so that callers can pass
        ``int``, ``float``, or ``None`` without special-casing.

        Args:
            val: A :class:`~fractions.Fraction`, numeric value, or ``None``.

        Returns:
            A Z3 ``Q(numerator, denominator)`` if *val* is a Fraction,
            otherwise *val* unchanged.
        """
        if isinstance(val, Fraction):
            return Q(val.numerator, val.denominator)
        return val

    def get_monitored_vars(self, model):
        """Extract the values of all monitored variables from a Z3 model.

        Rational values with very large denominators (>100) are rounded to
        three decimal places for readability; zero-valued rationals are
        returned as plain ``0``.

        Args:
            model: A satisfying Z3 model.

        Returns:
            dict mapping variable name (``str``) to its value (``int``,
            ``float``, :class:`~fractions.Fraction`, or Z3 literal).
        """
        formatted_vars = {}
        for var in self.return_vars:
            name, val = str(var), model[var]
            if isinstance(val, RatNumRef):
                val = Fraction(val.numerator_as_long(), val.denominator_as_long())
                if val.numerator == 0:
                    val = 0
                elif val.denominator > 100:
                    val = round(float(val), 3)
            formatted_vars[name] = val
        return formatted_vars

    def get_next(self, timeout_ms=None):
        """Find the next satisfying assignment for the accumulated constraints.

        A fresh :class:`~z3.Solver` is created on every call so that
        additional constraints added between calls are always honoured.

        Args:
            timeout_ms: Optional solver timeout in milliseconds. ``None``
                means no timeout.

        Returns:
            A ``(model, monitored_vars)`` tuple on success, or
            ``(None, None)`` when the problem is unsatisfiable or the solver
            times out.
        """
        s = Solver()
        if timeout_ms is not None:
            s.set("timeout", timeout_ms)
        for con in self.constraints:
            s.add(con)
        result = s.check()
        if result == unknown:
            return None, None
        if result != sat:
            return None, None
        model = s.model()
        self.solutions.append(model)
        return model, self.get_monitored_vars(model)
