"""
comgen -- Compositional Generation via Constraint Solving.

Generates ionic compositions that satisfy user-defined constraints using the
Z3 SMT solver. Supports charge balance, element/species bounds, radius-ratio
rules, Earth Mover's Distance from known compositions, synthesis from
ingredients, and optional ONNX-based property filters.

Typical usage::

    from comgen import IonicComposition, SpeciesCollection

    sps = SpeciesCollection.for_elements({"Li", "La", "Zr", "O"})
    query = IonicComposition(sps)
    query.distinct_elements(4)
    result = query.get_next()
"""

from .util.species import SpeciesCollection, PolyAtomicSpecies
from .query.ionic import IonicComposition