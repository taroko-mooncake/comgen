"""
Core composition constraint system.

Defines :class:`TargetComposition` (normalised element and species quantity
variables with charge-balance and electronegativity constraints) and
:class:`UnitCell` (integer atom counts per species that link back to the
normalised composition).
"""

import warnings
from fractions import Fraction

from z3 import And, Or, Not, Sum, Real, Q, Int, Implies
import pymatgen.core as pg

from comgen import SpeciesCollection, PolyAtomicSpecies
from comgen.constraint_system.common import (
    zero_weighted_sum,
    apply_bounds,
    check_bounds,
    bound_weighted_average_value_ratio,
)


class TargetComposition:
    """Z3 variable system representing a normalised composition.

    For each permitted species a real-valued quantity variable is created, and
    for each element (possibly shared across polyatomic species) a
    corresponding element-quantity variable is defined.  Base constraints
    ensure non-negativity, element--species consistency, and that element
    fractions sum to 1.

    Args:
        permitted_species: A :class:`~comgen.SpeciesCollection` of allowed
            ionic species.
        constraint_log: Mutable list to which Z3 constraints are appended.
        return_vars: Mutable list of Z3 variables whose values will be
            extracted from satisfying models.

    Raises:
        TypeError: If *permitted_species* is not a
            :class:`~comgen.SpeciesCollection`.
    """

    def __init__(self, permitted_species, constraint_log, return_vars):
        if not isinstance(permitted_species, SpeciesCollection): 
            raise TypeError("permitted_species argument must be a SpeciesCollection.")
        
        self.name = f'TargetComposition{str(id(self))}'
        
        self.permitted_species = permitted_species

        self.cons = constraint_log
        self.return_vars = return_vars

        self.element_quantity_variable_collection = {} 
        self.species_quantity_variable_collection = {}
        self._setup()

    def _new_element_quantity_var(self, el):
        """Create a Z3 Real variable for the normalised quantity of *el*."""
        el_id = str(el)
        var = Real(f'{self.name}_{el_id}_elementquantity')
        self.element_quantity_variable_collection[el_id] = var
        self.return_vars.append(var)

    def _new_species_quantity_var(self, sp):
        """Create a Z3 Real variable for the normalised quantity of *sp*."""
        sp_id = str(sp)
        var = Real(f'{self.name}_{sp_id}_speciesquantity')
        self.species_quantity_variable_collection[sp_id] = var
        self.return_vars.append(var)

    def _setup(self):
        """Initialise variables and add foundational constraints.

        1. Create a species-quantity variable for every permitted species and
           an element-quantity variable for every element.
        2. Constrain all quantities to be >= 0.
        3. For each element, require that the sum of contributing species
           quantities (weighted by multiplicity for polyatomic species)
           equals the element quantity.
        4. Require element quantities to sum to 1 (normalisation).
        """
        for sp in self.permitted_species:
            self._new_species_quantity_var(sp)
        for el in self.elements:
            self._new_element_quantity_var(el)
        
        for sp in self.permitted_species:
            self.cons.append(self.species_quantity_vars(sp) >= 0)
        for el in self.elements:
            self.cons.append(self.element_quantity_vars(el) >= 0)

        # species-to-element linkage: weighted sum of species = element quantity
        for el, sps in self.permitted_species.group_by_element_view().items():
            sps_vars, sps_weights = [], []
            for sp in sps:
                sps_vars.append(self.species_quantity_vars(sp))
                weight = 1
                if isinstance(sp, PolyAtomicSpecies):
                    weight = sp.multiplier(el)
                sps_weights.append(weight)
            
            self.cons.append(Sum([var*weight for var, weight in zip(sps_vars, sps_weights)]) == self.element_quantity_vars(el))

        # normalisation: all element fractions sum to 1
        vars = [self.element_quantity_vars(el) for el in self.elements]
        self.cons.append(Sum(*vars) == 1)

    @property
    def elements(self):
        """The set of elements covered by the permitted species."""
        return self.permitted_species.group_by_element_view().keys()

    def element_quantity_vars(self, el=None):
        """Return Z3 variable(s) for element quantities.

        Args:
            el: Element identifier (``str`` or pymatgen Element). If
                ``None``, returns the full ``{element: var}`` dict.

        Returns:
            A single Z3 ``Real`` variable, or a dict of all element
            variables.
        """
        if el is not None and not isinstance(el, str): el = str(el)

        if el:
            return self.element_quantity_variable_collection[el]
        return self.element_quantity_variable_collection
    
    def species_quantity_vars(self, sp=None):
        """Return Z3 variable(s) for species quantities.

        Args:
            sp: Species identifier (``str`` or pymatgen Species). If
                ``None``, returns the full ``{species: var}`` dict.

        Returns:
            A single Z3 ``Real`` variable, or a dict of all species
            variables.
        """
        if sp is not None and not isinstance(sp, str): sp = str(sp)

        if sp:
            return self.species_quantity_variable_collection[sp]
        return self.species_quantity_variable_collection    

    def balance_charges(self, return_constraint=False):
        """Add a charge-neutrality constraint: Σ (species_qty * charge) == 0.

        Args:
            return_constraint: If ``True``, return the constraint instead of
                appending it to the log.

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        sps_quants = [self.species_quantity_vars(str(sp)) for sp in self.permitted_species]
        sps_charges = [sp.oxi_state for sp in self.permitted_species]
        balance_charge_cons = zero_weighted_sum(sps_quants, sps_charges)

        if return_constraint:
            return balance_charge_cons 
        self.cons.append(balance_charge_cons)

    def restrict_charge_by_electronegativity(self, return_constraint=False):
        """Enforce electronegativity ordering on ion charges.

        No element may simultaneously contribute both positive and negative
        ions.  Furthermore, every positively charged ion must belong to an
        element with lower electronegativity than any negatively charged ion.

        Args:
            return_constraint: If ``True``, return the constraint instead of
                appending it.

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        elt_grouped_sps = self.permitted_species.group_by_element_view()
        eneg_cons = []
        
        pos, neg = {}, {}
        for elt, sps in elt_grouped_sps.items():
            sps = {sp for sp in sps if isinstance(sp, pg.Species)}
            pos[str(elt)] = Or([self.species_quantity_vars(str(sp)) > 0 for sp in sps if sp.oxi_state > 0])
            neg[str(elt)] = Or([self.species_quantity_vars(str(sp)) > 0 for sp in sps if sp.oxi_state < 0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for el_1 in elt_grouped_sps.keys():
                for el_2 in elt_grouped_sps.keys():
                    if el_1.X > el_2.X or el_1 == el_2:
                        eneg_cons.append(Not(And(pos[str(el_1)], neg[str(el_2)])))
                
        if return_constraint:
            return And(eneg_cons)
        self.cons.append(And(eneg_cons))

    def count_elements_from(self, elements: set, exact: int=None, return_constraint=False, *, lb: int=None, ub: int=None):
        """Constrain how many elements from the given set have non-zero quantity.

        Args:
            elements: Set of element symbols or pymatgen Elements to consider.
            exact: Require exactly this many to be present.
            return_constraint: Return rather than append the constraint.
            lb: Minimum count of elements present (inclusive).
            ub: Maximum count of elements present (inclusive).

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        count_elts_present = Sum([self.element_quantity_vars(str(elt)) > 0 for elt in elements])
        bound_elts_present_cons = apply_bounds(count_elts_present, exact, lb=lb, ub=ub)
        
        if return_constraint:
            return bound_elts_present_cons
        self.cons.append(bound_elts_present_cons)
    
    def count_elements(self, exact: int=None, return_constraint=False, *, lb: int=None, ub: int=None):
        """Constrain the total number of distinct elements in the composition.

        Equivalent to calling :meth:`count_elements_from` with all elements.

        Args:
            exact: Require exactly this many distinct elements.
            return_constraint: Return rather than append the constraint.
            lb: Minimum count (inclusive).
            ub: Maximum count (inclusive).

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        elements = {str(elt) for elt in self.elements}
        return self.count_elements_from(elements, exact, return_constraint, lb=lb, ub=ub)

    def fit_to_cell(self, unit_cell):
        """Link this composition to a :class:`UnitCell` so that quantities
        become integer multiples of a common formula unit.

        Args:
            unit_cell: A :class:`UnitCell` instance.
        """
        return unit_cell.fit_composition(self.species_quantity_vars())

    def bound_elements_quantity(self, elements: set, exact: float=None, return_constraint=False, *, lb: float=None, ub: float=None):
        """Bound the sum of normalised quantities for the given elements.

        Args:
            elements: Set of element symbols.
            exact: Fix the total to this value.
            return_constraint: Return rather than append the constraint.
            lb: Lower bound (inclusive).
            ub: Upper bound (inclusive).

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        total_elts_quantity = Sum([self.element_quantity_vars(str(elt)) for elt in elements])
        bound_elts_quantity_cons = apply_bounds(total_elts_quantity, exact, lb=lb, ub=ub)

        if return_constraint:
            return bound_elts_quantity_cons
        self.cons.append(bound_elts_quantity_cons)

    def bound_species_quantity(self, sps: set, exact: float=None, return_constraint=False, *, lb: float=None, ub: float=None):
        """Bound the sum of normalised quantities for the given species.

        Args:
            sps: Set of species identifiers (strings or pymatgen Species).
            exact: Fix the total to this value.
            return_constraint: Return rather than append the constraint.
            lb: Lower bound (inclusive).
            ub: Upper bound (inclusive).

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        total_sps_quantity = Sum([self.species_quantity_vars(str(sp)) for sp in sps])
        bound_sps_quantity_cons = apply_bounds(total_sps_quantity, exact, lb=lb, ub=ub)

        if return_constraint:
            return bound_sps_quantity_cons
        self.cons.append(bound_sps_quantity_cons)

    def exclude_composition(self, composition, precision=0.1, return_constraint=False):
        """Exclude a specific composition from the feasible region.

        A composition is considered "excluded" if *all* element quantities
        fall within ``±precision`` of the given values.  The added constraint
        is the negation of that conjunction.

        Args:
            composition: A dict ``{element: quantity}`` or pymatgen
                ``Composition``.
            precision: Tolerance around each element quantity.
            return_constraint: Return rather than append the constraint.

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        if isinstance(composition, pg.Composition):
            composition = dict(composition)
        
        cons = []
        for elt, quant in composition.items():
            lb, ub = quant - precision, quant + precision
            cons.append(apply_bounds(self.element_quantity_vars(elt), lb=lb, ub=ub))
        
        exclude_cons = Not(And(cons))
        if return_constraint:
            return exclude_cons
        self.cons.append(exclude_cons)

    def select_species_pair(self, pairs, return_constraint=False):
        """Require at least one species pair from the given list to be present.

        Both species in a pair must have non-zero quantity for the pair to
        be "selected".

        Args:
            pairs: Iterable of ``(species_str, species_str)`` tuples.
            return_constraint: Return rather than append the constraint.

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        sps_quants = self.species_quantity_vars()
        select_cons = []
        for sp1, sp2 in pairs:
            select_cons.append(And(sps_quants[sp1] > 0, sps_quants[sp2] > 0))
        
        if return_constraint:
            return Or(select_cons)
        self.cons.append(Or(select_cons))

    def exclude_species_pairs(self, pairs, return_constraint=False):
        """Forbid both species in each listed pair from co-occurring.

        For every listed pair, at least one of the two species must have
        zero quantity.

        Args:
            pairs: Iterable of ``(species_str, species_str)`` tuples.
            return_constraint: Return rather than append the constraint.

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        sps_quants = self.species_quantity_vars()
        exclude_cons = []
        for sp1, var1 in sps_quants.items():
            for sp2, var2 in sps_quants.items():
                if sp1 == sp2: continue
                if (sp1, sp2) in pairs:
                    exclude_cons.append(Or(var1 == 0, var2 == 0))

        if return_constraint:
            return And(exclude_cons)
        self.cons.append(And(exclude_cons))

    def bound_average_species_value_ratio(self, sps_1, sps_2, return_constraint=False, *, lb: float=None, ub: float=None):
        """Bound the ratio of weighted-average values for two species groups.

        The "value" of each species is the dict-value in *sps_1* / *sps_2*,
        weighted by the species quantity variables.

        Args:
            sps_1: Dict ``{species_str: value}`` for the first group.
            sps_2: Dict ``{species_str: value}`` for the second group.
            return_constraint: Return rather than append the constraint.
            lb: Lower bound on the ratio (inclusive).
            ub: Upper bound on the ratio (inclusive).

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        sps_quants = self.species_quantity_vars()
        vars_1 = [sps_quants[sp] for sp in sps_1.keys()]
        vars_2 = [sps_quants[sp] for sp in sps_2.keys()]
        vals_1 = sps_1.values()
        vals_2 = sps_2.values()

        ratio_cons = bound_weighted_average_value_ratio(vars_1, vars_2, vals_1, vals_2, lb=lb, ub=ub)
        if return_constraint:
            return ratio_cons
        self.cons.append(ratio_cons)

    def synthesise_from(self, synthesis, return_constraint=False):
        """Link this composition to a :class:`Synthesis` constraint system.

        Args:
            synthesis: A :class:`~comgen.constraint_system.Synthesis` instance.
            return_constraint: Return rather than append the constraint.

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.
        """
        return synthesis.fix_product(self.element_quantity_vars(), return_constraint)

    def bound_distance(self, other, calculator, return_constraint=False, *, ub=None, lb=None):
        """Bound the distance between this composition and another.

        The distance is computed by *calculator* (e.g. :class:`EMD`) over
        normalised element-quantity vectors.

        Args:
            other: A ``str``, pymatgen ``Composition``, or another
                :class:`TargetComposition`.
            calculator: A :class:`~comgen.constraint_system.Distance`
                subclass instance (e.g. :class:`EMD`).
            return_constraint: Return rather than append the constraint.
            ub: Upper bound on the distance.
            lb: Lower bound on the distance.

        Returns:
            Z3 constraint if *return_constraint* is ``True``, else ``None``.

        Raises:
            ValueError: If *other* is not a recognised type.
        """
        elt_vars = {self.name: self.element_quantity_vars()}
        
        if isinstance(other, str):
            other = pg.Composition(other)

        if isinstance(other, pg.Composition):
            other = other.fractional_composition
            elt_vars.update({str(other): dict(other)})
        elif isinstance(other, TargetComposition):
            elt_vars.update({str(other): other.element_quantity_vars()})
        else:
            raise ValueError(f'Expected TargetComposition or pymatgen Composition object. Received type {type(other)}.')
        
        bound_dist_cons = calculator.bound_distance(elt_vars, lb, ub)
        if return_constraint:
            return bound_dist_cons
        self.cons.append(bound_dist_cons)

    def property_predictor_category(self, model, n):
        """Encode an ONNX property-prediction model and constrain output class.

        The composition's element quantities (ordered by atomic number) are
        fed into the model, and the solver is required to make class *n*
        the argmax of the output layer.

        Args:
            model: An :class:`~comgen.constraint_system.ONNX` instance.
            n: Target output class index.
        """
        vars = self.element_quantity_vars()
        vars = [(pg.Element(elt).Z, var) for elt, var in vars.items()]
        vars.sort(key=lambda x: x[0])
        vars = [v for k, v in vars]

        model.setup(vars)
        model.select_class(n)

    def format_solution(self, model, as_frac=False):
        """Extract element quantities from a Z3 model into a Python dict.

        Zero-quantity elements are omitted.

        Args:
            model: A satisfying Z3 model.
            as_frac: If ``True``, values are :class:`~fractions.Fraction`;
                otherwise ``float`` rounded to 3 decimal places.

        Returns:
            dict mapping element ``str`` to quantity.
        """
        out = {elt: model[elt_var] 
               for elt, elt_var in self.element_quantity_vars().items() 
               if model[elt_var].numerator_as_long() != 0}
        
        if as_frac:
            return {elt: Fraction(quant.numerator_as_long(), quant.denominator_as_long())
                    for elt, quant in out.items()}
        
        return {elt: round(float(quant.numerator_as_long()) / float(quant.denominator_as_long()), 3) 
                for elt, quant in out.items()}

    def __str__(self):
        return self.name


class UnitCell:
    """Integer atom-count variables for a unit cell.

    Creates one integer count variable per species and a total-atom-count
    variable that can be bounded.  When linked to a
    :class:`TargetComposition` via :meth:`fit_composition`, the normalised
    species fractions are forced to correspond to integer stoichiometries.

    Args:
        permitted_species: A :class:`~comgen.SpeciesCollection` of allowed
            ionic species (must match the associated composition).
        constraint_log: Mutable list for Z3 constraints.
        return_vars: Mutable list of Z3 variables to monitor.
    """

    def __init__(self, permitted_species: SpeciesCollection, constraint_log, return_vars):
        self.name = f'UnitCell{str(id(self))}'
        self.cons = constraint_log
        self.return_vars = return_vars

        self.permitted_species = permitted_species

        self.species_count_variable_collection = {}
        self.num_atoms_variable = None

        self.num_atoms_lb, self.num_atoms_ub = None, None

        self._setup()

    def _new_species_count_var(self, sp):
        """Create a Z3 Int variable for the count of species *sp* in the cell."""
        sp_id = str(sp)
        var = Int(f'{self.name}_{sp_id}_speciescount')
        self.species_count_variable_collection[sp_id] = var
        self.return_vars.append(var)

    def _setup(self):
        """Create integer count variables for every species and for total atoms."""
        for sp in self.permitted_species:
            self._new_species_count_var(sp)
        self.num_atoms_variable = Int(f'{self.name}_numatoms')

    def species_count_vars(self, sp=None):
        """Return Z3 integer count variable(s) for species.

        Args:
            sp: Species identifier. If ``None``, returns the full dict.

        Returns:
            A single Z3 ``Int`` variable, or the entire ``{species: var}``
            dict.
        """
        if sp is not None and not isinstance(sp, str): sp = str(sp)

        if sp:
            return self.species_count_variable_collection[sp]
        return self.species_count_variable_collection

    def fit_composition(self, species_quantities):
        """Link normalised species fractions to integer atom counts.

        For every possible total atom count *n* in ``[lb, ub]``, adds
        implications of the form ``num_atoms == n  =>  count_sp == qty_sp * n``
        for each species.

        Args:
            species_quantities: Dict ``{species_str: Z3 Real or float}``
                mapping each species to its normalised quantity variable.

        Raises:
            AssertionError: If the species set does not match the cell's
                permitted species.
            ValueError: If atom-count bounds have not been set via
                :meth:`bound_total_atoms_count`.
        """
        assert self.permitted_species == set(species_quantities.keys()), 'Cell and composition must use the same species.'
        if self.num_atoms_ub is None:
            ValueError('Unit cell must have bounds on number of atoms before fitting a composition.')
        relate_quantities_cons = []
        for n in range(self.num_atoms_lb, self.num_atoms_ub+1):
            for sp, quant in species_quantities.items():
                cons = self.species_count_variable_collection[str(sp)] == quant * n
                relate_quantities_cons.append(Implies(self.num_atoms_variable == n, cons))
            self.cons.extend(relate_quantities_cons)

    def bound_total_atoms_count(self, lb, ub):
        """Set lower and upper bounds on the total number of atoms.

        Also records the bounds internally so that :meth:`fit_composition`
        knows the range of *n* to enumerate.

        Args:
            lb: Minimum total atoms (inclusive).
            ub: Maximum total atoms (inclusive).
        """
        self.num_atoms_lb = lb
        self.num_atoms_ub = ub
        total_atoms_constraint = apply_bounds(self.num_atoms_variable, lb=lb, ub=ub)
        self.cons.append(total_atoms_constraint)

    def bound_elements_count(self, elements, exact=None, *, lb=None, ub=None):
        """Bound the total integer atom count for a set of elements.

        Sums the species counts (weighted by multiplicity for polyatomic
        species) for every species belonging to the given elements.

        Args:
            elements: Iterable of element symbol strings.
            exact: Fix the count to this value.
            lb: Minimum count (inclusive).
            ub: Maximum count (inclusive).
        """
        grouped_sps = {str(el): sps for el, sps in self.permitted_species.group_by_element_view().items()}
        sps_counts = []
        for el in elements:
            elt_sps = grouped_sps[el]
            for sp in elt_sps:
                if isinstance(sp, PolyAtomicSpecies):
                    sps_counts.append(self.species_count_vars(sp)*sp.multiplier(el))
                else:
                    sps_counts.append(self.species_count_vars(sp))

        elt_count_constraint = apply_bounds(Sum(sps_counts), exact, lb=lb, ub=ub)
        self.cons.append(elt_count_constraint)
