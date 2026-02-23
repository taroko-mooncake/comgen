"""
Synthesis constraint system.

Encodes the requirement that a target composition can be expressed as a
non-negative linear combination of given ingredient (starting-material)
compositions.  This ensures that the generated composition is, in
principle, synthesisable from the specified precursors.
"""

from comgen.constraint_system.common import zero_weighted_sum, weighted_sum
from z3 import Real, And


class Synthesis:
    """Constraint system linking a target composition to ingredient compositions.

    For each ingredient a non-negative real-valued "amount" variable is
    created.  The :meth:`fix_product` method then requires that the
    weighted sum of ingredient compositions (weighted by their amounts)
    equals the target element-quantity vector.

    Args:
        ingredient_compositions: Iterable of ingredient compositions, each
            a dict mapping element strings to fractional amounts.
        constraint_log: Mutable list to which Z3 constraints are appended.
        return_vars: Mutable list of Z3 variables to monitor.
    """

    def __init__(self, ingredient_compositions, constraint_log, return_vars):
        self.ingredient_compositions = ingredient_compositions
        self.name = f"Synthesis{id(self)}"
        self.cons = constraint_log
        self.return_vars = return_vars
        self._ingredient_quantity_variable_collection = {}
        self._setup()

    def _ingredient_quantity_vars(self, comp):
        """Return the Z3 variable for the amount of ingredient *comp*, or ``None``."""
        return self._ingredient_quantity_variable_collection.get(str(comp))
    
    def _new_ingredient_quantity_var(self, comp):
        """Create a non-negative Z3 Real variable for an ingredient's amount."""
        var = Real(f'{self.name}_{str(comp)}_ingredientquantity')
        self._ingredient_quantity_variable_collection[str(comp)] = var
        self.return_vars.append(var)
        return var

    def _setup(self):
        """Create ingredient-amount variables and constrain them to be >= 0."""
        for comp in self.ingredient_compositions:
            var = self._new_ingredient_quantity_var(comp)
            self.cons.append(var >= 0)

    def fix_product(self, element_quantities, return_constraint=False):
        """Require that a weighted combination of ingredients equals the target.

        For every element present in either the ingredients or the target,
        the following must hold::

            Σ_i  amount_i * ingredient_i[el]  ==  target[el]

        This is encoded as ``Σ (amount_i * weight_i) + (-1 * target_el) == 0``
        for each element.

        Args:
            element_quantities: Dict ``{element: Z3 Real or float}`` for the
                target composition.
            return_constraint: If ``True``, return the conjunction of
                per-element constraints instead of appending them.

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        elements = {str(el) for comp in self.ingredient_compositions for el in comp}
        elements.update(set(element_quantities.keys()))
        ing_vars = [self._ingredient_quantity_vars(comp) for comp in self.ingredient_compositions]
        
        cons = []
        for el in elements:
            vars = ing_vars + [element_quantities.get(el, 0)]
            weights = [comp.get(el, 0) for comp in self.ingredient_compositions]
            weights.append(-1)
            cons.append(zero_weighted_sum(vars, weights))

        if return_constraint:
            return And(cons)
        self.cons.append(And(cons))

    def bound_cost(self, ingredient_costs, lb, ub):
        """Bound the total cost of the ingredient mixture.

        Cost is the weighted sum of ingredient amounts where the weight
        for each ingredient is its unit cost.

        Args:
            ingredient_costs: Dict mapping each ingredient composition to
                its unit cost (numeric).
            lb: Lower bound on total cost (inclusive).
            ub: Upper bound on total cost (inclusive).
        """
        assert all([k in self.ingredient_compositions for k in ingredient_costs.keys()])
        vars = self._ingredient_quantity_vars()
        weights = [ingredient_costs[comp] for comp in self.ingredient_compositions]
        self.cons.append(weighted_sum(vars, weights, lb=lb, ub=ub))
