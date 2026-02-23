"""
Generate Li-ion conductor compositions synthesisable from known starting
materials, far from existing reference compositions (ElMD).

Output: examples/output/li_starting_materials.txt
"""

import logging
import random
import sys
import time
from csv import DictReader
from pathlib import Path

import pymatgen.core as pg

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from comgen import SpeciesCollection, IonicComposition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

examples_dir = _script_dir.parent
data_dir = examples_dir / "Li_ion_conductors" / "data"
output_dir = examples_dir / "output"
output_file = output_dir / "li_starting_materials.txt"
reps_file = data_dir / "LiIon_reps.csv"
ingredients_file = data_dir / "Li_starting_materials.csv"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

DISTANCE = 5
NUM_RESULTS = 20
TOTAL_ATOMS_LB = 10
TOTAL_ATOMS_UB = 15
SOLVER_TIMEOUT_MS = 60_000
MAX_REFERENCES = 10
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Element / species definitions
# ---------------------------------------------------------------------------

Li = {"Li"}
A = {"B", "Al", "Si", "P"}
B = {"Mg", "Zn", "Ta", "Sn", "Ge", "Ga"}
C = {"K", "Ca", "Sr", "Y", "Zr", "Ba", "La", "Gd"}
D = {"N", "O", "F"}
E = {"S", "Se", "Te", "Cl", "Br", "I"}

UNWANTED_SPECIES = {
    pg.Species("Ta", 4), pg.Species("Ta", 3),
    pg.Species("Se", 4), pg.Species("Se", 6),
    pg.Species("Te", 4), pg.Species("Te", 6),
    pg.Species("Ge", 4), pg.Species("P", 3),
}


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data files
    # ------------------------------------------------------------------
    log.info("Loading ingredients from %s", ingredients_file.name)
    ingredients, excluded = [], []
    with open(ingredients_file) as f:
        for row in DictReader(f):
            comp = pg.Composition(row["norm_composition"])
            excluded.append(comp)
            if row["starting_material"] == "Y":
                ingredients.append(comp)
    log.info(
        "  %d starting materials (Y), %d total compositions to exclude",
        len(ingredients), len(excluded),
    )

    log.info("Loading reference compositions from %s", reps_file.name)
    with open(reps_file) as f:
        all_comparisons = [row["composition"] for row in DictReader(f)]
    log.info("  %d reference compositions loaded", len(all_comparisons))

    if len(all_comparisons) > MAX_REFERENCES:
        rng = random.Random(RANDOM_SEED)
        comparisons = rng.sample(all_comparisons, MAX_REFERENCES)
        log.info(
            "  Sub-sampled to %d references (full set of %d creates too many "
            "constraints for the solver)",
            len(comparisons), len(all_comparisons),
        )
    else:
        comparisons = all_comparisons

    # ------------------------------------------------------------------
    # 2. Build species collection
    # ------------------------------------------------------------------
    elts = Li | A | B | C | D | E
    log.info("Building species for %d elements", len(elts))
    sps = SpeciesCollection.for_elements(elts)
    sps = sps.difference(UNWANTED_SPECIES)
    log.info("  %d species after filtering", len(sps))

    # ------------------------------------------------------------------
    # 3. Build query and add constraints
    # ------------------------------------------------------------------
    log.info("Building IonicComposition query")
    query = IonicComposition(sps)
    log.info("  Base constraints (charge balance + electronegativity): %d", len(query.constraints))

    query.include_elements_quantity(Li, lb=0.2)
    log.info("  After include_elements_quantity(Li>=0.2): %d", len(query.constraints))

    query.distinct_elements(ub=6)
    log.info("  After distinct_elements(<=6): %d", len(query.constraints))

    query.total_atoms(lb=TOTAL_ATOMS_LB, ub=TOTAL_ATOMS_UB)
    log.info("  After total_atoms(%d..%d): %d", TOTAL_ATOMS_LB, TOTAL_ATOMS_UB, len(query.constraints))

    query.made_from(ingredients)
    log.info("  After made_from(%d ingredients): %d", len(ingredients), len(query.constraints))

    query.exclude(excluded)
    log.info("  After exclude(%d compositions): %d", len(excluded), len(query.constraints))

    query.elmd_far_from_all(comparisons, DISTANCE)
    log.info(
        "  After elmd_far_from_all(%d refs, dist=%s): %d",
        len(comparisons), DISTANCE, len(query.constraints),
    )

    # ------------------------------------------------------------------
    # 4. Solve
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    log.info(
        "Solving for up to %d compositions (timeout %ds per solve) ...",
        NUM_RESULTS, SOLVER_TIMEOUT_MS // 1000,
    )
    t_total = time.perf_counter()

    for i in range(NUM_RESULTS):
        t0 = time.perf_counter()
        out = query.get_next(as_frac=True, timeout_ms=SOLVER_TIMEOUT_MS)
        elapsed = time.perf_counter() - t0

        if out is None:
            log.info("  No more solutions after %d result(s) (%.1fs)", len(results), elapsed)
            break

        comp, _ = out
        results.append(comp)
        log.info("  #%d: %s  (%.1fs)", len(results), comp, elapsed)

    total_elapsed = time.perf_counter() - t_total
    log.info("Finished: %d compositions in %.1fs", len(results), total_elapsed)

    # ------------------------------------------------------------------
    # 5. Write output
    # ------------------------------------------------------------------
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(f"# Li-ion starting-materials query ({len(results)} results)\n")
        for res in results:
            f_out.write(str(res) + "\n")
    log.info("Output written to %s", output_file)


if __name__ == "__main__":
    main()
