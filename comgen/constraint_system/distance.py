"""
Earth Mover's Distance (EMD) constraint system.

Provides an abstract :class:`Distance` base class and a concrete
:class:`EMD` implementation that encodes the 1-D Wasserstein (EMD)
distance between two normalised element-quantity vectors as Z3
constraints.  The metric operates on an ordered scale (e.g. the
modified Pettifor scale) so that elements that are chemically similar
incur a smaller transport cost.
"""

from comgen.constraint_system.common import apply_bounds, Abs
from z3 import Sum, Real


class Distance:
    """Abstract base for pairwise distance constraint systems.

    Subclasses must implement :meth:`_setup_distance_calculation` (which
    creates Z3 variables and constraints that *define* the distance) and
    :meth:`_distance_var` (which returns the Z3 variable holding the
    computed distance between two objects).
    """

    def _setup_distance_calculation(self, object_vars):
        """Create Z3 variables and constraints defining the distance.

        Args:
            object_vars: Dict ``{object_id: {var_id: Z3 var or float}}``
                with exactly two entries.
        """
        raise NotImplementedError
    
    def _distance_var(self, id_1, id_2):
        """Return the Z3 variable representing the distance between two objects.

        Returns ``None`` if the distance has not been set up yet.
        """
        raise NotImplementedError
    
    def bound_distance(self, object_vars, lb=None, ub=None):
        """Add a bound on the distance between two objects.

        If the distance has not been set up for this pair, it is
        initialised automatically via :meth:`_setup_distance_calculation`.

        Args:
            object_vars: Dict ``{object_id: {var_id: Z3 var or float}}``
                with exactly two entries.
            lb: Lower bound on the distance (inclusive).
            ub: Upper bound on the distance (inclusive).

        Returns:
            A Z3 ``BoolRef`` constraint.
        """
        id_1, id_2 = object_vars.keys()
        if (var := self._distance_var(id_1, id_2)) is None:
            self._setup_distance_calculation(object_vars)
            var = self._distance_var(id_1, id_2)
        
        return apply_bounds(var, lb=lb, ub=ub)


class EMD(Distance):
    """Earth Mover's Distance on an ordered 1-D scale.

    The EMD between two normalised element vectors is computed by
    accumulating the signed difference of the two distributions in the
    order defined by ``ordered_metric_ids`` (e.g. the modified Pettifor
    scale) and summing the absolute values of these cumulative
    differences.

    Args:
        id_mapping_func: Callable that maps an element identifier (e.g.
            a pymatgen Element or string) to its position on the ordered
            scale.
        ordered_metric_ids: Tuple of all position IDs on the scale, in
            ascending order.
        constraint_log: Mutable list to which Z3 constraints are appended.
        return_vars: Mutable list of Z3 variables to monitor.
    """

    def __init__(self, id_mapping_func, ordered_metric_ids, constraint_log, return_vars):
        self.name = f'EMD{id(self)}'
        self._distance_var_collection = {}
        self._difference_var_collection = {}
        self.ordered_metric_ids = ordered_metric_ids
        self.id_mapping_func = id_mapping_func
        self.constraint_log = constraint_log
        self.return_vars = return_vars

    def _difference_var(self, id_1, id_2, var_id=None):
        """Look up the cumulative-difference variable for a pair at a given scale position.

        Handles both ``(id_1, id_2)`` and ``(id_2, id_1)`` key orderings.

        Args:
            id_1: First object identifier.
            id_2: Second object identifier.
            var_id: Scale position.  If ``None``, return the full dict for
                the pair.

        Returns:
            The Z3 variable, the full dict, or ``None`` if not yet created.
        """
        vars = self._difference_var_collection.get(str((id_1, id_2)))
        if vars is None: 
            vars = self._difference_var_collection.get(str((id_2, id_1)))
        if vars is not None:
            if var_id is None: return vars
            if vars.get(var_id) is not None: return vars[var_id]
        return None
    
    def _new_difference_var(self, id_1, id_2, var_id):
        """Create a Z3 Real variable for the cumulative difference at *var_id*."""
        var = Real(f'{self.name}_{str((id_1, id_2))}_{var_id}_EMDdiff')
        if not (vars := (self._difference_var_collection.get(str((id_1, id_2))) or \
            self._difference_var_collection.get(str((id_2, id_1))))):
            self._difference_var_collection[str((id_1, id_2))] = {}
            vars = self._difference_var_collection[str((id_1, id_2))]
        vars[var_id] = var
        return var
    
    def _distance_var(self, id_1, id_2):
        """Return the Z3 variable holding the total EMD for a pair, or ``None``."""
        var = self._distance_var_collection.get(str((id_1, id_2)))
        if var is None:
            var = self._distance_var_collection.get(str((id_2, id_1)))
        return var
    
    def _new_distance_var(self, id_1, id_2):
        """Create and register a Z3 Real variable for the total EMD between two objects."""
        var = Real(f'{self.name}_{str((id_1, id_2))}_EMDdistance')
        self._distance_var_collection[str((id_1, id_2))] = var
        self.return_vars.append(var)
        return var

    def _setup_distance_calculation(self, object_vars):
        """Build the Z3 encoding of the EMD between two element vectors.

        For each position *m* on the ordered scale the cumulative signed
        difference is::

            diff[m] = (obj1[m] - obj2[m]) + diff[m-1]

        The total EMD is then ``Σ |diff[m]|`` over all positions.

        Args:
            object_vars: Dict ``{object_id: {element_id: Z3 var or float}}``
                with exactly two entries.  Element IDs are mapped to scale
                positions via ``id_mapping_func``.
        """
        assert len(object_vars) == 2
        id_1, id_2 = object_vars.keys()
        object_vars[id_1] = {self.id_mapping_func(var_id): var for var_id, var in object_vars[id_1].items()}
        object_vars[id_2] = {self.id_mapping_func(var_id): var for var_id, var in object_vars[id_2].items()}

        assert all([k in self.ordered_metric_ids for obj in object_vars.values() for k in obj.keys()])

        ob_1, ob_2 = object_vars[id_1], object_vars[id_2]
        prev_m_id = None
        for m_id in self.ordered_metric_ids:
            if not (var := self._difference_var(id_1, id_2, m_id)):
                var = self._new_difference_var(id_1, id_2, m_id)

            prev_difference = 0 if prev_m_id is None else self._difference_var(id_1, id_2, prev_m_id)
            cons = (var == Sum(ob_1.get(m_id, 0), -1*ob_2.get(m_id, 0), prev_difference))
            self.constraint_log.append(cons)
            prev_m_id = m_id

        if not (var := self._distance_var(id_1, id_2)):
            var = self._new_distance_var(id_1, id_2)
        
        cons = var == Sum([Abs(self._difference_var(id_1, id_2, m_id)) for m_id in self.ordered_metric_ids])
        self.constraint_log.append(cons)
