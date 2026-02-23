"""ABX3 organic-inorganic perovskite candidate generator.

Generates potential ABX3 perovskite compounds where:
  A = methylammonium (MA+) / caesium (Cs+) / formamidinium (FA+) / guanidinium (GA+)
  B = Pb / Sn / Hg / Cd / Zn / Fe / Ni / Co / In / Bi / Ti
  X = Cl / Br / I

Partial substitution is allowed on all crystallographic sites (mixed occupancy).
Stability is characterised through an ONNX model of formation energy (3-class:
stable / metastable / unstable) when the model file and onnxruntime are available.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pymatgen.core as pg
from z3 import Int, Sum, And, RealVal

from comgen import SpeciesCollection, PolyAtomicSpecies
from comgen.query.ionic import IonicComposition
from comgen.constraint_system.composition import UnitCell

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODEL_PATH = _SCRIPT_DIR.parent / "low_formation_energy" / "ehull_1040_bn.onnx"

STABILITY_LABELS = {0: "stable", 1: "metastable", 2: "unstable"}

# ---------------------------------------------------------------------------
# Site definitions
# ---------------------------------------------------------------------------

A_SITE_ORGANIC = [
    ({"C": 1, "H": 6, "N": 1}, 1, "MA"),  # methylammonium  CH3NH3+
    ({"C": 1, "H": 5, "N": 2}, 1, "FA"),  # formamidinium   CH(NH2)2+
    ({"C": 1, "H": 6, "N": 3}, 1, "GA"),  # guanidinium     C(NH2)3+
]

A_SITE_INORGANIC = [("Cs", 1)]

B_SITE_METALS = [
    ("Pb", 2), ("Sn", 2), ("Hg", 2), ("Cd", 2), ("Zn", 2),
    ("Fe", 2), ("Ni", 2), ("Co", 2),
    ("In", 3), ("Bi", 3),
    ("Ti", 4),
]

X_SITE_HALIDES = [("Cl", -1), ("Br", -1), ("I", -1)]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poly(comp_dict: dict, charge: int) -> PolyAtomicSpecies:
    return PolyAtomicSpecies(pg.Composition(comp_dict), charge)


def _a_site_names() -> set:
    names = set()
    for comp_dict, charge, _ in A_SITE_ORGANIC:
        names.add(str(_poly(comp_dict, charge)))
    for elem, charge in A_SITE_INORGANIC:
        names.add(str(pg.Species(elem, charge)))
    return names


def _b_site_names() -> set:
    return {str(pg.Species(e, c)) for e, c in B_SITE_METALS}


def _x_site_names() -> set:
    return {str(pg.Species(e, c)) for e, c in X_SITE_HALIDES}


def _species_label(varname: str) -> str:
    """Extract readable species label from a UnitCell variable name."""
    after_prefix = varname.split("_", 1)[1] if "_" in varname else varname
    return after_prefix.rsplit("_speciescount", 1)[0]


@dataclass(frozen=True)
class Candidate:
    elements_frac: Dict[str, float]
    species_counts: Dict[str, int]
    stability_category: Optional[int] = None
    stability_label: Optional[str] = None

# ---------------------------------------------------------------------------
# Species space
# ---------------------------------------------------------------------------

def build_species_space() -> SpeciesCollection:
    """Build the full set of allowed ionic species across all ABX3 sites."""
    species: set = set()

    for comp_dict, charge, _ in A_SITE_ORGANIC:
        species.add(_poly(comp_dict, charge))
    for elem, charge in A_SITE_INORGANIC:
        species.add(pg.Species(elem, charge))

    for elem, charge in B_SITE_METALS:
        species.add(pg.Species(elem, charge))

    for elem, charge in X_SITE_HALIDES:
        species.add(pg.Species(elem, charge))

    return SpeciesCollection(species)

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def add_abx3_constraints(query: IonicComposition, species: SpeciesCollection) -> None:
    """Enforce ABX3 stoichiometry with partial substitution on every site.

    Sum(A-site counts) = k
    Sum(B-site counts) = k
    Sum(X-site counts) = 3k       (k = 1..4 formula units)

    Charge balance is already handled by IonicComposition.  Partial
    substitution is implicit: each individual species count >= 0 and only the
    site totals are fixed, so any mix of species on a site is permitted.
    """
    cell = UnitCell(species, query.constraints, query.return_vars)
    cell.bound_total_atoms_count(lb=5, ub=80)
    query.new_comp.fit_to_cell(cell)

    a_names = _a_site_names()
    b_names = _b_site_names()
    x_names = _x_site_names()

    k = Int("formula_units_k")
    query.constraints.append(And(k >= 1, k <= 4))

    a_counts = [cell.species_count_vars(sp) for sp in a_names]
    b_counts = [cell.species_count_vars(sp) for sp in b_names]
    x_counts = [cell.species_count_vars(sp) for sp in x_names]

    query.constraints.append(Sum(a_counts) == k)
    query.constraints.append(Sum(b_counts) == k)
    query.constraints.append(Sum(x_counts) == 3 * k)


def add_onnx_stability_constraint(
    query: IonicComposition,
    onnx_model,
    target_category: int = 0,
) -> None:
    """Embed the ONNX formation-energy classifier as a Z3 constraint.

    The solver will only emit compositions whose predicted category equals
    *target_category* (0 = stable, 1 = metastable, 2 = unstable).

    The element-fraction vector is padded to the model's expected input size
    (103 elements, Z = 1..103) with constant zeros for elements absent from
    the species collection.
    """
    from comgen.constraint_system.nn import ONNX as ONNXConstraint

    nn = ONNXConstraint(onnx_model, query.constraints)
    input_size = int(np.prod(nn.shapes[nn.inputNames[0]]))

    elt_vars = query.new_comp.element_quantity_vars()
    z_to_var = {pg.Element(elt).Z: var for elt, var in elt_vars.items()}

    padded = [z_to_var.get(z, RealVal(0)) for z in range(1, input_size + 1)]

    nn.setup(padded)
    nn.select_class(target_category)

# ---------------------------------------------------------------------------
# ONNX inference (post-processing)
# ---------------------------------------------------------------------------

_ort_session: Optional[object] = None


def _get_ort_session():
    """Lazily create (and cache) an onnxruntime InferenceSession."""
    global _ort_session
    if _ort_session is not None:
        return _ort_session
    if ort is None or not MODEL_PATH.exists():
        return None
    _ort_session = ort.InferenceSession(str(MODEL_PATH))
    return _ort_session


def predict_stability(elements_frac: Dict[str, float]) -> Optional[int]:
    """Run ONNX inference to predict the stability category of a composition."""
    session = _get_ort_session()
    if session is None:
        return None

    meta = session.get_inputs()[0]
    input_size = meta.shape[-1]

    vec = np.zeros(input_size, dtype=np.float32)
    for elt, frac in elements_frac.items():
        z = pg.Element(elt).Z
        if 1 <= z <= input_size:
            vec[z - 1] = frac

    outputs = session.run(None, {meta.name: vec.reshape(1, -1)})
    return int(np.argmax(outputs[0]))

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    n: int = 100,
    use_onnx_constraint: bool = False,
    target_category: int = 0,
) -> List[Candidate]:
    """Generate up to *n* ABX3 perovskite candidates.

    Parameters
    ----------
    n : int
        Maximum number of candidates to generate.
    use_onnx_constraint : bool
        If True **and** the ONNX model is available, add it as a Z3 hard
        constraint so only compositions predicted as *target_category* are
        emitted.  This is powerful but can be slow for deep networks.
    target_category : int
        Stability class to target when *use_onnx_constraint* is True
        (0 = stable).
    """
    species = build_species_space()
    q = IonicComposition(species)
    add_abx3_constraints(q, species)

    if use_onnx_constraint and onnx is not None and MODEL_PATH.exists():
        log.info("Loading ONNX model as Z3 constraint (target category=%d)", target_category)
        onnx_model = onnx.load(str(MODEL_PATH))
        add_onnx_stability_constraint(q, onnx_model, target_category)

    log.info("Generating up to %d ABX3 perovskite candidates ...", n)
    out: List[Candidate] = []

    for i in range(n):
        result = q.get_next(timeout_ms=60000)
        if result is None:
            log.info("Solver exhausted after %d candidates", i)
            break

        comp_dict, monitored = result
        elt_frac = {elt: float(v) for elt, v in comp_dict.items()}

        species_counts: Dict[str, int] = {}
        for name, val in monitored.items():
            if "_speciescount" not in name:
                continue
            try:
                iv = val.as_long() if hasattr(val, "as_long") else int(val)
            except (ValueError, TypeError, AttributeError):
                iv = 0
            if iv > 0:
                species_counts[_species_label(name)] = iv

        stab_cat = predict_stability(elt_frac)
        stab_label = STABILITY_LABELS.get(stab_cat) if stab_cat is not None else None

        out.append(Candidate(
            elements_frac=elt_frac,
            species_counts=species_counts,
            stability_category=stab_cat,
            stability_label=stab_label,
        ))

    out.sort(key=lambda c: (
        c.stability_category if c.stability_category is not None else 999,
    ))
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        log.warning(
            "ONNX model not found at %s — stability will not be characterised",
            MODEL_PATH,
        )
    if ort is None:
        log.warning(
            "onnxruntime not installed — stability inference unavailable "
            "(pip install onnxruntime)",
        )

    cands = generate_candidates(n=50)

    print(f"\nGenerated {len(cands)} ABX3 perovskite candidates")
    print("=" * 70)
    for i, c in enumerate(cands[:20], start=1):
        stab = f"  [{c.stability_label}]" if c.stability_label else ""
        print(f"\n#{i}{stab}")
        print(f"  composition : {c.elements_frac}")
        print(f"  species     : {c.species_counts}")
