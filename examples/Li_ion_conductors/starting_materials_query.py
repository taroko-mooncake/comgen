"""Generate Li-ion compositions synthesisable from known starting materials.

Constrains compositions to be synthesisable from a set of precursors, excludes
already-known compositions, and requires distance >= DISTANCE from reference
compositions (ElMD). Combines synthesis, exclusion, and novelty constraints.
"""

import logging
import random
import time
from csv import DictReader
from pathlib import Path

import pymatgen.core as pg

from comgen import IonicComposition, SpeciesCollection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXAMPLES_DIR / "data"
OUTPUT_DIR = EXAMPLES_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "li_starting_materials.txt"
REPS_FILE = DATA_DIR / "LiIon_reps.csv"
INGREDIENTS_FILE = DATA_DIR / "Li_starting_materials.csv"

DISTANCE = 5
NUM_RESULTS = 20
TOTAL_ATOMS_LB = 10
TOTAL_ATOMS_UB = 15
SOLVER_TIMEOUT_MS = 60_000
MAX_REFERENCES = 10
RANDOM_SEED = 42

LI = {"Li"}
P_BLOCK = {"B", "Al", "Si", "P"}
POST_TRANSITION = {"Mg", "Zn", "Ta", "Sn", "Ge", "Ga"}
ALKALINE_AND_RARE_EARTH = {"K", "Ca", "Sr", "Y", "Zr", "Ba", "La", "Gd"}
LIGHT_ANIONS = {"N", "O", "F"}
HEAVY_ANIONS = {"S", "Se", "Te", "Cl", "Br", "I"}

UNWANTED_SPECIES = {
    pg.Species("Ta", 4), pg.Species("Ta", 3),
    pg.Species("Se", 4), pg.Species("Se", 6),
    pg.Species("Te", 4), pg.Species("Te", 6),
    pg.Species("Ge", 4), pg.Species("P", 3),
}


def main() -> None:
    log.info("Loading ingredients from %s", INGREDIENTS_FILE.name)
    ingredients, excluded = [], []
    with open(INGREDIENTS_FILE, encoding="utf-8") as f:
        for row in DictReader(f):
            comp = pg.Composition(row["norm_composition"])
            excluded.append(comp)
            if row["starting_material"] == "Y":
                ingredients.append(comp)
    log.info("  %d starting materials, %d compositions to exclude",
             len(ingredients), len(excluded))

    log.info("Loading reference compositions from %s", REPS_FILE.name)
    with open(REPS_FILE, encoding="utf-8") as f:
        all_comparisons = [row["composition"] for row in DictReader(f)]

    if len(all_comparisons) > MAX_REFERENCES:
        rng = random.Random(RANDOM_SEED)
        comparisons = rng.sample(all_comparisons, MAX_REFERENCES)
        log.info("  Sub-sampled %d -> %d references (reducing solver load)",
                 len(all_comparisons), len(comparisons))
    else:
        comparisons = all_comparisons

    elements = LI | P_BLOCK | POST_TRANSITION | ALKALINE_AND_RARE_EARTH | LIGHT_ANIONS | HEAVY_ANIONS
    species = SpeciesCollection.for_elements(elements)
    species = species.difference(UNWANTED_SPECIES)

    query = IonicComposition(species)
    query.include_elements_quantity(LI, lb=0.2)
    query.distinct_elements(ub=6)
    query.total_atoms(lb=TOTAL_ATOMS_LB, ub=TOTAL_ATOMS_UB)
    query.made_from(ingredients)
    query.exclude(excluded)
    query.elmd_far_from_all(comparisons, DISTANCE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    log.info("Solving for up to %d compositions ...", NUM_RESULTS)
    t_total = time.perf_counter()

    for _ in range(NUM_RESULTS):
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

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for res in results:
            f_out.write(str(res) + "\n")
    log.info("Output written to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
