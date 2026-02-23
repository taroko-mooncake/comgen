"""
Data loading and species enumeration utilities.

Loads periodic-table data and common polyatomic ion definitions from
bundled data files, and provides functions that enumerate plausible
mono-atomic and polyatomic species for a given set of elements.
"""

from enum import Enum
from pathlib import Path
import pymatgen.core as pg
import json
import csv
from ast import literal_eval

_DATA_DIR = Path(__file__).resolve().parent / "data_files"
_pt_data_file = "periodic_table.json"
_poly_data_file = "common_poly_ions.txt"

with open(_DATA_DIR / _pt_data_file, encoding="utf-8") as f:
    PT_DATA = json.load(f)
"""dict: Periodic-table data keyed by element symbol, loaded from ``periodic_table.json``."""

with open(_DATA_DIR / _poly_data_file, encoding="utf-8") as f:
    ions_reader = csv.reader(f, delimiter='\t')
    POLY_DATA = {name: literal_eval(charges) for (name, charges) in ions_reader}
"""dict: Common polyatomic ions mapping formula string to a list of possible charges."""

ELEMENTS = {pg.Element(el) for el in PT_DATA.keys() if pg.Element(el).Z <= 103}
"""set: All elements from the periodic-table data with atomic number <= 103."""


class PossibleSpecies(Enum):
    """Controls which oxidation states are considered when building species.

    Members:
        FIXED: A single hard-coded oxidation state per element.
        VARY_TRANSITION: Vary transition-metal oxidation states.
        SH_FIX_HALOGEN: Shannon-radii oxidation states, but fix halogens
            to common charges (Br-, Cl-, F-, I-, N^3-, S^2-).
        SHANNON: All oxidation states with known Shannon radii.
        ALL: Every oxidation state listed in pymatgen.
    """
    FIXED = 1
    VARY_TRANSITION = 2
    SH_FIX_HALOGEN = 3
    SHANNON = 4
    ALL = 5


def mono_atomic_species(
    elements=None, 
    permitted=PossibleSpecies.SHANNON) -> set:
    """Enumerate plausible (element, charge) pairs for mono-atomic species.

    Uses Shannon-radii data from the periodic table to determine which
    oxidation states are chemically reasonable for each element.

    Args:
        elements: Set of pymatgen :class:`Element` objects, or ``None``
            for all elements up to Z = 103.
        permitted: A :class:`PossibleSpecies` enum value.  Currently
            ``SHANNON`` and ``SH_FIX_HALOGEN`` are implemented.

    Returns:
        set of ``(element, charge)`` tuples.
    """
    if elements is None:
        elements = ELEMENTS

    if permitted == PossibleSpecies.SHANNON:
        el_radii = {el: PT_DATA[el.symbol].get('Shannon radii', {"0":{}}) for el in elements}
        species = {(el, int(ch)) for el, item in el_radii.items() for ch in item.keys()}

    if permitted == PossibleSpecies.SH_FIX_HALOGEN:
        el_radii = {el: PT_DATA[el.symbol].get('Shannon radii', {"0":{}}) for el in elements}
        species = set()
        for el, item in el_radii.items():
            if el.symbol in ['Br', 'Cl', 'F', 'I']:
                species.add((el, -1))
            elif el.symbol == 'N':
                species.add((el, -3))
            elif el.symbol == 'S':
                species.add(('S', -2))
            else:
                for ch in item.keys():
                    species.add((el, int(ch)))

    return species 


def poly_atomic_species(elements=None) -> set:
    """Enumerate plausible polyatomic species for the given elements.

    Returns all common polyatomic ions (from ``common_poly_ions.txt``)
    whose constituent elements are a subset of *elements*.

    Args:
        elements: Set of pymatgen :class:`Element` objects, or ``None``
            for all elements.

    Returns:
        set of ``(pymatgen.Composition, charge)`` tuples.
    """
    if elements is None:
        elements = ELEMENTS

    species = set()

    for c, chgs in POLY_DATA.items():
        comp = pg.Composition(c)
        if set(comp.elements).issubset(set(elements)):
            species.update({(comp, chg) for chg in chgs})

    return species
