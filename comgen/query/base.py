from z3 import sat, unknown, Solver, Q
from fractions import Fraction
from z3.z3 import RatNumRef, BoolRef, IntNumRef

class Query:
    def __init__(self):
        self.constraints = []
        self.solutions = []
        self.return_vars = []

    def frac_to_rational(self, val):
        if isinstance(val, Fraction):
            return Q(val.numerator, val.denominator)
        return val

    def get_monitored_vars(self, model):
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
        s = Solver()
        if timeout_ms is not None:
            s.set("timeout", timeout_ms)
        for con in self.constraints:
            s.add(con)
        result = s.check()
        if result == unknown:
            return None, None  # timeout or resource limit
        if result != sat:
            return None, None
        model = s.model()
        self.solutions.append(model)
        return model, self.get_monitored_vars(model)
    