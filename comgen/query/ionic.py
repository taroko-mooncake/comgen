"""
High-level query classes for generating ionic compositions.

:class:`SingleTarget` provides the main user-facing API for building a
constraint query over a single target composition, while
:class:`IonicComposition` extends it with automatic charge balance and
electronegativity ordering.
"""

from z3 import And, Or, sat, Solver
from comgen import SpeciesCollection
from comgen.constraint_system import TargetComposition, UnitCell, EMD, Synthesis, ONNX
from comgen.query import PETTIFOR_KEYS, element_to_pettifor, Query, get_radii
import pymatgen.core as pg
from typing import Optional


class SingleTarget(Query):
    """Query for a single target composition expressed as normalised element fractions.

    Wraps a :class:`~comgen.constraint_system.TargetComposition` and exposes
    convenience methods for adding element-count, distance, synthesis, and
    ONNX-classifier constraints before solving.

    Args:
        sps: A :class:`~comgen.SpeciesCollection` defining the allowed ionic
            species.
        precision: Minimum distance (in normalised fraction space) between
            successive solutions returned by :meth:`get_next`.  Defaults to
            ``0.1``.
    """

    def __init__(self, sps, precision=None):
        super().__init__()
        assert isinstance(sps, SpeciesCollection)
        self._sps = sps
        self.precision = precision or 0.1
        self._elmd_calculator = None
        self.total_atoms_lb = None
        self.total_atoms_ub = None
        self._setup()

    @property
    def precision(self):
        """float: Exclusion tolerance used when removing previous solutions."""
        return self._precision

    @precision.setter
    def precision(self, val):
        self._precision = val

    @property
    def species(self):
        """SpeciesCollection: The set of permitted ionic species."""
        return self._sps

    def _setup(self):
        """Initialise the underlying :class:`TargetComposition` constraint system."""
        self.new_comp = TargetComposition(self._sps, self.constraints, self.return_vars)
        self.unit_cell = None

    def _get_elmd_calc(self):
        """Lazily create and cache the EMD calculator (Pettifor-scale metric)."""
        if self._elmd_calculator is None:
            self._elmd_calculator = EMD(element_to_pettifor, PETTIFOR_KEYS, self.constraints, self.return_vars)
        return self._elmd_calculator

    def elmd_close_to_one(self, compositions, bounds):
        """Require the target to be close to *at least one* reference composition.

        Uses Earth Mover's Distance on the modified Pettifor scale.

        Args:
            compositions: Iterable of reference compositions (pymatgen
                ``Composition`` or string).
            bounds: Maximum EMD to each reference.  A single number applies
                the same bound to every reference; otherwise provide a list
                of the same length as *compositions*.
        """
        if isinstance(bounds, float) or isinstance(bounds, int): bounds = [bounds]*len(compositions)
        assert len(bounds) == len(compositions)

        distances = []
        for comp, dist in zip(compositions, bounds):
            distances.append(self.new_comp.bound_distance(comp, self._get_elmd_calc(), ub=dist, return_constraint=True))
        self.constraints.append(Or(distances))    

    def elmd_far_from_all(self, compositions, bounds):
        """Require the target to be far from *every* reference composition.

        Uses Earth Mover's Distance on the modified Pettifor scale.

        Args:
            compositions: Iterable of reference compositions.
            bounds: Minimum EMD from each reference.  A single number applies
                the same bound to every reference; otherwise provide a list
                of the same length as *compositions*.
        """
        if isinstance(bounds, float) or isinstance(bounds, int): bounds = [bounds]*len(compositions)
        assert len(bounds) == len(compositions)
        
        calc = self._get_elmd_calc()
        for comp, dist in zip(compositions, bounds):
            self.new_comp.bound_distance(comp, calc, lb=dist)

    def made_from(self, ingredients):
        """Constrain the target to be synthesisable from the given ingredients.

        The target must be expressible as a non-negative linear combination
        of the ingredient compositions.

        Args:
            ingredients: Iterable of ingredient compositions (dicts mapping
                element strings to fractional amounts).
        """
        synthesis = Synthesis(ingredients, self.constraints, self.return_vars)
        self.new_comp.synthesise_from(synthesis)

    def include_elements(self, elements, exact=None, *, lb=None, ub=None):
        """Bound how many elements from a given set appear in the composition.

        Args:
            elements: Set of element symbols to consider.
            exact: Require exactly this many elements from the set.
            lb: Minimum number of elements from the set (inclusive).
            ub: Maximum number of elements from the set (inclusive).
        """
        self.new_comp.count_elements_from(elements, exact, lb=lb, ub=ub)

    def include_elements_quantity(self, elements, exact=None, *, lb=None, ub=None):
        """Bound the total normalised quantity of a set of elements.

        Args:
            elements: Set of element symbols whose quantities are summed.
            exact: Fix the total to this value.
            lb: Lower bound on the total (inclusive).
            ub: Upper bound on the total (inclusive).
        """
        exact = self.frac_to_rational(exact)
        lb = self.frac_to_rational(lb)
        ub = self.frac_to_rational(ub)
        self.new_comp.bound_elements_quantity(elements, exact, lb=lb, ub=ub)

    def distinct_elements(self, exact=None, *, lb=None, ub=None):
        """Bound the total number of distinct elements in the composition.

        Args:
            exact: Require exactly this many distinct elements.
            lb: Minimum number of distinct elements (inclusive).
            ub: Maximum number of distinct elements (inclusive).
        """
        self.new_comp.count_elements(exact, lb=lb, ub=ub)

    def total_atoms(
            self,
            exact: Optional[int]=None,
            *,
            lb: Optional[int]=None,
            ub: Optional[int]=None):
        """Bound the total number of atoms in the unit cell.

        Internally creates a :class:`~comgen.constraint_system.UnitCell` (if
        one does not already exist) and links it to the composition so that
        species quantities become integer multiples of a common formula unit.

        Args:
            exact: Fix the atom count to this value.
            lb: Minimum atom count (defaults to 1).
            ub: Maximum atom count (required).

        Raises:
            ValueError: If neither *exact* nor *ub* is provided.
        """
        if exact: lb, ub = exact, exact
        if lb is None: lb = 1
        if ub is None: raise ValueError('Please provide an upper bound on the number of atoms.')

        if self.unit_cell is None:
            self.unit_cell = UnitCell(self._sps, self.constraints, self.return_vars)

        self.unit_cell.bound_total_atoms_count(lb, ub)

        self.new_comp.fit_to_cell(self.unit_cell)

    def include_elements_count(
            self,
            elements,
            exact: Optional[int]=None,
            *,
            lb: Optional[int]=None,
            ub: Optional[int]=None):
        """Bound the absolute atom count for a set of elements in the unit cell.

        Unlike :meth:`include_elements_quantity` (which works on normalised
        fractions), this constrains the *integer* number of atoms.

        Args:
            elements: Set of element symbols.
            exact: Fix the count to this value.
            lb: Minimum count (inclusive).
            ub: Maximum count (inclusive).
        """
        if self.unit_cell is None:
            self.unit_cell = UnitCell(self._sps, self.constraints, self.return_vars)

        self.unit_cell.bound_elements_count(elements, exact, lb=lb, ub=ub)

    def exclude(self, compositions):
        """Exclude one or more specific compositions from the search space.

        Args:
            compositions: Iterable of compositions (dicts or pymatgen
                ``Composition`` objects) to forbid.
        """
        for comp in compositions:
            self.new_comp.exclude_composition(comp)

    def category_prediction(self, onnx_model, category):
        """Require the composition to be classified into a given category by
        an ONNX neural-network model.

        Args:
            onnx_model: A loaded ``onnx.ModelProto`` object.
            category: Integer index of the target output class.
        """
        model = ONNX(onnx_model, self.constraints)
        self.new_comp.property_predictor_category(model, category)

    def get_next(self, as_frac=False, timeout_ms=None):
        """Solve for the next composition and return it.

        After a solution is found it is automatically excluded (within
        :pyattr:`precision`) so that subsequent calls yield distinct results.

        Args:
            as_frac: If ``True``, return element quantities as
                :class:`~fractions.Fraction` objects; otherwise as rounded
                floats.
            timeout_ms: Optional solver timeout in milliseconds.

        Returns:
            A ``(composition_dict, monitored_vars)`` tuple, or ``None`` when
            no further solutions exist.
        """
        model, return_vars = super().get_next(timeout_ms=timeout_ms)
        if model is None:
            return None
        elt_quants = self.new_comp.format_solution(model, as_frac)
        self.new_comp.exclude_composition(elt_quants, self.precision)
        return {elt: str(q) for elt, q in elt_quants.items()}, return_vars


class IonicComposition(SingleTarget):
    """Query for charge-balanced ionic compositions.

    Extends :class:`SingleTarget` with automatic charge-balance and
    electronegativity-ordering constraints, plus radius-ratio and
    radius-difference helpers for structure-aware filtering.

    Args:
        sps: A :class:`~comgen.SpeciesCollection` defining the allowed ionic
            species.
        precision: Exclusion tolerance (see :class:`SingleTarget`).
    """

    def _setup(self):
        """Add charge-balance and electronegativity constraints on top of
        the base :class:`SingleTarget` setup."""
        super()._setup()
        self.new_comp.balance_charges()
        self.new_comp.restrict_charge_by_electronegativity()

    def ion_pair_radius_ratio(self, sps1, sps2, cn1=None, cn2=None, *, lb=None, ub=None):
        """Require at least one selected ion pair to have a radius ratio within bounds.

        For every pair ``(sp1, sp2)`` drawn from *sps1* x *sps2* whose
        Shannon radius ratio falls inside ``[lb, ub]``, the solver is told
        that *at least one* such pair must be present in the composition.

        Args:
            sps1: First :class:`~comgen.SpeciesCollection` (numerator radii).
            sps2: Second :class:`~comgen.SpeciesCollection` (denominator radii).
            cn1: Coordination number for *sps1* (e.g. ``"VIII"``).
            cn2: Coordination number for *sps2* (e.g. ``"VI"``).
            lb: Lower bound on the radius ratio (inclusive).
            ub: Upper bound on the radius ratio (inclusive).
        """
        pairs = []
        for sp1, v1 in get_radii(sps1, cn1).items():
            assert isinstance(v1, (int, float))
            for sp2, v2 in get_radii(sps2, cn2).items():
                assert isinstance(v2, (int, float))
                if lb is None or v1 / v2 >= lb:
                    if ub is None or v1 / v2 <= ub:
                        pairs.append((sp1, sp2))

        self.new_comp.select_species_pair(pairs)

    def ion_pair_radius_difference(self, sps1, sps2=None, *, lb=None, ub=None):
        """Exclude ion pairs whose absolute radius difference is out of bounds.

        All pairs ``(sp1, sp2)`` from *sps1* x *sps2* with
        ``|r1 - r2|`` outside ``[lb, ub]`` are forbidden from co-occurring
        in the composition.

        Args:
            sps1: First :class:`~comgen.SpeciesCollection`.
            sps2: Second :class:`~comgen.SpeciesCollection`; defaults to
                *sps1* (self-comparison).
            lb: Minimum absolute radius difference (inclusive).
            ub: Maximum absolute radius difference (inclusive).
        """
        if sps2 == None: sps2 = sps1
        if isinstance(sps1, set): sps1 = SpeciesCollection(sps1)
        if isinstance(sps2, set): sps2 = SpeciesCollection(sps2)
        assert isinstance(sps1, SpeciesCollection)
        assert isinstance(sps2, SpeciesCollection)

        excluded_pairs = []
        for sp1, v1 in get_radii(sps1).items():
            assert isinstance(v1, (int, float))
            for sp2, v2 in get_radii(sps2).items():
                assert isinstance(v2, (int, float))
                if lb is not None and abs(v1 - v2) < lb:
                    excluded_pairs.append((sp1, sp2))
                elif ub is not None and abs(v1 - v2) > ub:
                    excluded_pairs.append((sp1, sp2))

        self.new_comp.exclude_species_pairs(excluded_pairs)

    def include_species_quantity(self, species, exact=None, *, lb=None, ub=None):
        """Bound the total normalised quantity of a set of species.

        Unlike :meth:`include_elements_quantity` which groups by element, this
        method works directly on individual species (element + oxidation state).

        Args:
            species: Set of species (strings or pymatgen ``Species``).
            exact: Fix the total species quantity to this value.
            lb: Lower bound on the total (inclusive).
            ub: Upper bound on the total (inclusive).
        """
        exact = self.frac_to_rational(exact)
        lb = self.frac_to_rational(lb)
        ub = self.frac_to_rational(ub)
        self.new_comp.bound_species_quantity(species, exact, lb=lb, ub=ub)
