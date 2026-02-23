"""ABX3 organic-inorganic perovskite candidate generator.

Generates potential ABX3 perovskite compounds where:
  A = methylammonium (MA+) / caesium (Cs+) / formamidinium (FA+) / guanidinium (GA+)
  B = Pb / Sn / Hg / Cd / Zn / Fe / Ni / Co / In / Bi / Ti
  X = Cl / Br / I

Partial substitution is allowed on all crystallographic sites (mixed occupancy).
Stability is characterised through an ONNX model of formation energy (3-class:
stable / metastable / unstable) when the model file and onnxruntime are available.

Known reference perovskites and starting materials can be loaded from CSV files
in the data/ directory to constrain the search to novel, synthesisable compositions.
"""

from __future__ import annotations

import logging
import sys
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

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
DATA_DIR = _SCRIPT_DIR.parent / "data"
REFERENCE_PEROVSKITES_CSV = DATA_DIR / "reference_perovskites.csv"
STARTING_MATERIALS_CSV = DATA_DIR / "perovskite_starting_materials.csv"

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
# CSV loaders
# ---------------------------------------------------------------------------

def load_reference_perovskites(
    path: Path = REFERENCE_PEROVSKITES_CSV,
) -> List[pg.Composition]:
    """Load known reference perovskite compositions from a CSV file.

    The CSV must have at least a ``composition`` column containing
    pymatgen-parseable elemental composition strings.

    Returns a list of ``pg.Composition`` objects suitable for ElMD
    distance constraints.
    """
    compositions: List[pg.Composition] = []
    with open(path, encoding="utf-8") as f:
        for row in DictReader(f):
            compositions.append(pg.Composition(row["composition"]))
    log.info("Loaded %d reference perovskites from %s", len(compositions), path.name)
    return compositions


def load_starting_materials(
    path: Path = STARTING_MATERIALS_CSV,
) -> List[pg.Composition]:
    """Load known starting-material compositions from a CSV file.

    The CSV must have at least a ``composition`` column containing
    pymatgen-parseable elemental formulas (e.g. ``Pb1I2``, ``C1H6N1I1``).

    Returns a list of ``pg.Composition`` objects suitable for the
    ``made_from`` synthesis constraint.
    """
    ingredients: List[pg.Composition] = []
    with open(path, encoding="utf-8") as f:
        for row in DictReader(f):
            ingredients.append(pg.Composition(row["composition"]))
    log.info("Loaded %d starting materials from %s", len(ingredients), path.name)
    return ingredients

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
# ElMD distance constraint
# ---------------------------------------------------------------------------

def add_elmd_constraint(
    query: IonicComposition,
    reference_compositions: list,
    distance: Union[float, int],
    *,
    mode: str = "close_to",
) -> None:
    """Add an ElMD (Earth Mover's Distance on the Pettifor scale) constraint.

    Parameters
    ----------
    reference_compositions : list
        Composition strings (e.g. ``"CsPbBr3"``) or pymatgen
        ``Composition`` objects to measure distance against.
    distance : float
        Distance threshold.
    mode : str
        ``"close_to"`` — candidate must be within *distance* of **at least
        one** reference  (ElMD <= *distance*).
        ``"far_from"`` — candidate must be at least *distance* away from
        **every** reference  (ElMD >= *distance*).
    """
    if mode == "close_to":
        query.elmd_close_to_one(reference_compositions, distance)
    elif mode == "far_from":
        query.elmd_far_from_all(reference_compositions, distance)
    else:
        raise ValueError(f"mode must be 'close_to' or 'far_from', got {mode!r}")


def add_synthesis_constraint(
    query: IonicComposition,
    ingredients: List[pg.Composition],
) -> None:
    """Constrain generated compositions to be synthesisable from *ingredients*.

    Uses a mass-balance constraint: the target composition must be expressible
    as a non-negative linear combination of the provided ingredient
    compositions (at the elemental level).
    """
    query.made_from(ingredients)

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

def _solve_batch(
    query: IonicComposition,
    n: int,
    timeout_ms: int = 60_000,
) -> List[Candidate]:
    """Run the solver up to *n* times, collecting candidates."""
    out: List[Candidate] = []
    for i in range(n):
        result = query.get_next(timeout_ms=timeout_ms)
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
    return out


def generate_candidates(
    n: int = 100,
    use_onnx_constraint: bool = False,
    target_category: int = 0,
    elmd_distance: Optional[float] = None,
    elmd_mode: str = "close_to",
    n_initial: int = 10,
    reference_compositions: Optional[list] = None,
    use_known_references: bool = False,
    min_distance: Optional[float] = None,
    use_starting_materials: bool = False,
    starting_materials_path: Optional[Path] = None,
    references_path: Optional[Path] = None,
) -> List[Candidate]:
    """Generate up to *n* ABX3 perovskite candidates.

    When *elmd_distance* is set the generation runs in two phases:

    1. **Discovery** — generate *n_initial* candidates without an ElMD
       constraint.  These become the reference set.
    2. **Filtered generation** — build a fresh query with an ElMD
       constraint (``mode="close_to"`` for ElMD <= *distance*, or
       ``mode="far_from"`` for ElMD >= *distance*) relative to the
       discovered references, then generate up to *n* candidates.

    If *reference_compositions* is provided explicitly the discovery
    phase is skipped and those compositions are used directly.

    When *use_known_references* is True, known perovskite compounds are
    loaded from the reference CSV and used as the distance reference set.
    Combined with *min_distance*, this constrains the solver to produce
    compositions that are at least *min_distance* (ElMD) away from every
    known reference — i.e. genuinely novel compositions.

    When *use_starting_materials* is True, the solver is further
    constrained so that every generated composition can be expressed as a
    non-negative linear combination of known precursor chemicals (e.g.
    MAI, PbI2, SnBr2, …).

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
    elmd_distance : float, optional
        ElMD distance threshold for the generic constraint.
    elmd_mode : str
        ``"close_to"`` (default) or ``"far_from"``.
    n_initial : int
        How many candidates to discover in the first phase (ignored when
        *reference_compositions* is supplied or *elmd_distance* is None).
    reference_compositions : list, optional
        Explicit reference compositions (strings or ``pg.Composition``).
        Skips the discovery phase when provided.
    use_known_references : bool
        If True, load known perovskites from ``examples/data/reference_perovskites.csv``
        and enforce a minimum ElMD distance from all of them.
    min_distance : float, optional
        Minimum ElMD distance from every known reference perovskite.
        Only used when *use_known_references* is True.  Defaults to 3
        when *use_known_references* is True and *min_distance* is None.
    use_starting_materials : bool
        If True, load precursor chemicals from
        ``examples/data/perovskite_starting_materials.csv`` and add a synthesis
        (mass-balance) constraint so that only compositions achievable
        from those ingredients are generated.
    starting_materials_path : Path, optional
        Override the default starting-materials CSV path.
    references_path : Path, optional
        Override the default reference-perovskites CSV path.
    """
    species = build_species_space()

    # ------------------------------------------------------------------
    # Load external data
    # ------------------------------------------------------------------
    known_refs: Optional[List[pg.Composition]] = None
    if use_known_references:
        ref_path = references_path or REFERENCE_PEROVSKITES_CSV
        known_refs = load_reference_perovskites(ref_path)
        if min_distance is None:
            min_distance = 3
            log.info("  min_distance not set; defaulting to %s", min_distance)

    ingredients: Optional[List[pg.Composition]] = None
    if use_starting_materials:
        mat_path = starting_materials_path or STARTING_MATERIALS_CSV
        ingredients = load_starting_materials(mat_path)

    # ------------------------------------------------------------------
    # Phase 1 — discover initial candidates to use as ElMD references
    #            (only when using the generic elmd_distance without an
    #             explicit reference set)
    # ------------------------------------------------------------------
    refs = reference_compositions
    if elmd_distance is not None and refs is None and not use_known_references:
        log.info("Phase 1: discovering %d initial candidates as ElMD references ...", n_initial)
        q_init = IonicComposition(species)
        add_abx3_constraints(q_init, species)
        initial = _solve_batch(q_init, n_initial)
        refs = [pg.Composition(c.elements_frac) for c in initial]
        log.info("  Discovered %d reference compounds", len(refs))

    # ------------------------------------------------------------------
    # Phase 2 — generate with all constraints
    # ------------------------------------------------------------------
    q = IonicComposition(species)
    add_abx3_constraints(q, species)

    if use_known_references and known_refs:
        log.info(
            "Adding novelty constraint: ElMD >= %s from %d known reference perovskites",
            min_distance, len(known_refs),
        )
        add_elmd_constraint(q, known_refs, min_distance, mode="far_from")

    if elmd_distance is not None and refs:
        op = ">=" if elmd_mode == "far_from" else "<="
        log.info("Adding ElMD %s constraint (distance %s %s) from %d references",
                 elmd_mode, op, elmd_distance, len(refs))
        add_elmd_constraint(q, refs, elmd_distance, mode=elmd_mode)

    if use_starting_materials and ingredients:
        log.info(
            "Adding synthesis constraint: composition must be achievable "
            "from %d known starting materials",
            len(ingredients),
        )
        add_synthesis_constraint(q, ingredients)

    if use_onnx_constraint and onnx is not None and MODEL_PATH.exists():
        log.info("Loading ONNX model as Z3 constraint (target category=%d)", target_category)
        onnx_model = onnx.load(str(MODEL_PATH))
        add_onnx_stability_constraint(q, onnx_model, target_category)

    log.info("Generating up to %d ABX3 perovskite candidates ...", n)
    out = _solve_batch(q, n)

    out.sort(key=lambda c: (
        c.stability_category if c.stability_category is not None else 999,
    ))
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_DIR = _SCRIPT_DIR.parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "perovskite_ABX3_candidates.txt"


def save_candidates(candidates: List[Candidate], path: Path) -> None:
    """Write candidates to a text file in the output folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# ABX3 perovskite candidates ({len(candidates)} total)\n")
        for i, c in enumerate(candidates, start=1):
            stab = f"  [{c.stability_label}]" if c.stability_label else ""
            f.write(f"\n#{i}{stab}\n")
            f.write(f"  composition : {c.elements_frac}\n")
            f.write(f"  species     : {c.species_counts}\n")
    log.info("Saved %d candidates to %s", len(candidates), path)


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

    cands = generate_candidates(
        n=50,
        use_known_references=True,
        min_distance=3,
        use_starting_materials=True,
    )

    print(f"\nGenerated {len(cands)} ABX3 perovskite candidates")
    print("=" * 70)
    for i, c in enumerate(cands[:20], start=1):
        stab = f"  [{c.stability_label}]" if c.stability_label else ""
        print(f"\n#{i}{stab}")
        print(f"  composition : {c.elements_frac}")
        print(f"  species     : {c.species_counts}")

    save_candidates(cands, OUTPUT_FILE)
