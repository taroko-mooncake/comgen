"""
Species collection and polyatomic species utilities.

:class:`PolyAtomicSpecies` provides a pymatgen-compatible wrapper for
multi-element ions (e.g. CO3^2-, SiO4^4-) so they can be used alongside
:class:`pymatgen.core.Species` objects.

:class:`SpeciesCollection` is a managed set of ionic species (both
mono-atomic and polyatomic) with helper methods for filtering by charge,
element, and type, and a factory method for building collections from
periodic-table data.
"""

from collections import defaultdict
from .data import PossibleSpecies, mono_atomic_species, poly_atomic_species
import pymatgen.core as pg
import numpy as np


class PolyAtomicSpecies:
    """Representation of a polyatomic ion with a fixed oxidation state.

    Mimics key attributes of :class:`pymatgen.core.Species`
    (``oxi_state``, ``X``) so that polyatomic and mono-atomic species can
    be handled uniformly in :class:`SpeciesCollection`.

    Args:
        comp: Chemical formula string (e.g. ``"CO3"``) or a pymatgen
            :class:`~pymatgen.core.Composition`.
        oxi_state: Net charge of the polyatomic ion (e.g. ``-2``).
    """

    def __init__(self, comp, oxi_state):
        if isinstance(comp, str):
            comp = pg.Composition(comp)

        self._composition = comp
        self._oxi_state = oxi_state 
        self._electronegativity = np.nan

    @property
    def oxi_state(self):
        """int: Net charge of this polyatomic ion."""
        return self._oxi_state

    @property
    def composition(self):
        """pymatgen.Composition: The internal composition of the ion."""
        return self._composition

    @property
    def X(self):
        """float: Electronegativity (always ``NaN`` for polyatomic species).

        Polyatomic species are excluded from electronegativity ordering in
        :meth:`TargetComposition.restrict_charge_by_electronegativity`.
        """
        return self._electronegativity
    
    def multiplier(self, element: str) -> int:
        """Return the stoichiometric count of *element* within this ion.

        For example, ``PolyAtomicSpecies("CO3", -2).multiplier("O")``
        returns ``3``.

        Args:
            element: Element symbol string.

        Returns:
            Integer atom count of *element* in the ion (0 if absent).
        """
        return int(dict(self.composition).get(element, 0))

    @property
    def elements(self):
        """set: The set of pymatgen :class:`Element` objects in this ion."""
        return set(self.composition.elements)
    
    def __str__(self) -> str:
        charge_str = '' if self.oxi_state in [1, -1] else str(abs(self.oxi_state))
        charge_str += '-' if self.oxi_state < 0 else '+'

        return "("+self.composition.to_pretty_string()+")"+charge_str

    def __repr__(self) -> str:
        charge_str = '' if self.oxi_state in [1, -1] else str(abs(self.oxi_state))
        charge_str += '-' if self.oxi_state < 0 else '+'

        return "PolyAtomicSpecies ("+self.composition.to_pretty_string()+")"+charge_str  
        
    def __eq__(self, other) -> bool:
        return self.composition == other.composition and self.oxi_state == other.oxi_state  

    def __hash__(self) -> int:
        return hash((self.composition, self.oxi_state))


class SpeciesCollection:
    """Managed set of ionic species with grouping and filtering capabilities.

    Internally stores species as a flat ``set`` but lazily builds derived
    views (grouped by element, grouped by composition) for efficient
    look-ups.  Views are invalidated whenever the collection is mutated.

    Args:
        data: Initial set of :class:`pymatgen.core.Species` and/or
            :class:`PolyAtomicSpecies` objects.
    """

    def __init__(self, data: set):
        self._species = set()
        self.update(data) 

    def _set_empty_views(self):
        """Reset cached derived views (called on mutation)."""
        self._elements_view = defaultdict(set)
        self._compositions_view = defaultdict(set) 

    @staticmethod
    def _is_valid_collection_data(data):
        """Assert that every item in *data* is a Species or PolyAtomicSpecies."""
        for item in data:
            assert isinstance(item, pg.Species) or isinstance(item, PolyAtomicSpecies), print(data)

    def ungrouped_view(self) -> set:
        """Return the flat (ungrouped) set of species."""
        return self._species

    def group_by_element_view(self):
        """Return species grouped by their parent element.

        For mono-atomic species the key is the pymatgen :class:`Element`;
        for polyatomic species an entry is created for *every* constituent
        element.

        The returned dict is frozen (``default_factory = None``) so that
        accidental key-creation is prevented.

        Returns:
            dict mapping :class:`~pymatgen.core.Element` to a ``set`` of
            species containing that element.
        """
        if self._elements_view:
            return self._elements_view
        
        for sp in self._species:
            if isinstance(sp, PolyAtomicSpecies):
                for el in sp.elements:
                    self._elements_view[el].add(sp)
            else:
                self._elements_view[sp.element].add(sp)

        self._elements_view.default_factory = None
        
        return self._elements_view

    def update(self, species):
        """Add one or more species to the collection.

        Accepts a single :class:`Species`/:class:`PolyAtomicSpecies`, a
        set of them, or another :class:`SpeciesCollection`.  Cached views
        are invalidated.

        Args:
            species: Species to add.
        """
        if isinstance(species, PolyAtomicSpecies) or isinstance(species, pg.Species):
            species = {species}
        if isinstance(species, SpeciesCollection):
            species = species.ungrouped_view()
        self._is_valid_collection_data(species)
        self._set_empty_views()

        self._species.update(species)

    def difference(self, species):
        """Return a new collection with the given species removed.

        Args:
            species: Species to remove (same types accepted as
                :meth:`update`).

        Returns:
            A new :class:`SpeciesCollection` without the given species.
        """
        if isinstance(species, PolyAtomicSpecies) or isinstance(species, pg.Species):
            species = {species}
        if isinstance(species, SpeciesCollection):
            species = species.ungrouped_view()
        self._is_valid_collection_data(species)
        self._set_empty_views()

        return SpeciesCollection(self._species.difference(species))
  
    def having_charge(self, charges):
        """Return a new collection containing only species with the given charge(s).

        Args:
            charges: A single charge (``int``) or a list of charges.

        Returns:
            A filtered :class:`SpeciesCollection`.
        """
        filtered_collection = set()
        if isinstance(charges, int):
            charges = [charges]
        
        for item in self._species:
            if item.oxi_state in charges:
                filtered_collection.add(item)
        
        return SpeciesCollection(filtered_collection)

    def filter(self, elements: set):
        """Return a new collection restricted to species made from the given elements.

        Mono-atomic species are kept only if their element is in the set.
        Polyatomic species are kept only if *all* constituent elements are
        in the set.

        Args:
            elements: Set of element symbols (strings).

        Returns:
            A filtered :class:`SpeciesCollection`.
        """
        elements = {pg.Element(el) for el in elements}
        filtered_collection = self._species.copy()

        for item in self._species:
            if isinstance(item, pg.Species):
                if not item.element in elements:
                    filtered_collection.remove(item)
            if isinstance(item, PolyAtomicSpecies):
                if not item.elements.issubset(elements):
                    filtered_collection.remove(item)
        
        return SpeciesCollection(filtered_collection)

    def filter_mono_species(self):
        """Return a new collection containing only mono-atomic pymatgen Species.

        Polyatomic species are excluded.

        Returns:
            A filtered :class:`SpeciesCollection`.
        """
        filtered_collection = set()
        
        for item in self._species:
            if isinstance(item, pg.Species):
                filtered_collection.add(item)

        return SpeciesCollection(filtered_collection)

    @classmethod
    def for_elements(
        cls, 
        elements=None, 
        permitted=PossibleSpecies.SH_FIX_HALOGEN, 
        include_poly=False):
        """Build a collection of plausible ionic species for the given elements.

        Uses periodic-table data (Shannon radii) to determine which
        oxidation states are permitted for each element.

        Args:
            elements: Iterable of element symbols, or ``None`` for all
                elements up to Z = 103.
            permitted: A :class:`~comgen.util.data.PossibleSpecies` enum
                value controlling which oxidation states are included.
                Defaults to ``SH_FIX_HALOGEN`` (Shannon radii with fixed
                halogen charges).
            include_poly: If ``True``, also include common polyatomic ions
                whose constituent elements are a subset of *elements*.

        Returns:
            A new :class:`SpeciesCollection`.
        """
        if elements:
            elements = {pg.Element(el) for el in elements}

        mono_species = mono_atomic_species(elements, permitted)
        def _symbol(e):
            return e.symbol if hasattr(e, "symbol") else e
        species = {pg.Species(_symbol(el), chg) for (el, chg) in mono_species}
        if include_poly:
            poly_species = poly_atomic_species(elements)
            species.update({PolyAtomicSpecies(comp, chg) for (comp, chg) in poly_species})

        return cls(species)

    def __len__(self):
        return len(self._species)

    def __eq__(self, other):
        if isinstance(other, SpeciesCollection):
            return self._species == other._species
        if isinstance(other, set):
            for sp in self._species:
                if not (sp in other or str(sp) in other):
                    return False
            return True
        return False

    def __iter__(self):
        return iter(self._species)

    def __str__(self):
        return f'{__class__.__name__} {self.ungrouped_view()}'
