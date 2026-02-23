"""Quick-start example: generate Li-ion compositions far from known references.

Run from the repository root:
    python examples/run_example.py

Output: examples/output/example_compositions.txt
Data:   examples/data/LiIon_reps.csv (reference compositions for EMD constraint).

Uses a narrower element set than Li_ion_conductors/distance_query.py for
faster solving — intended as a first run to verify the installation works.
"""

import logging
from csv import DictReader
from pathlib import Path

import pymatgen.core as pg

from comgen import IonicComposition, SpeciesCollection

EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLES_DIR / "data"
OUTPUT_DIR = EXAMPLES_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "example_compositions.txt"
REPS_FILE = DATA_DIR / "LiIon_reps.csv"

NUM_RESULTS = 10
EMD_MIN_DISTANCE = 5
TOTAL_ATOMS_LB = 10
TOTAL_ATOMS_UB = 15
DISTINCT_ELEMENTS_MAX = 6

LI = {"Li"}
P_BLOCK = {"B", "Al", "Si", "P"}
POST_TRANSITION = {"Mg", "Zn", "Sn", "Ga"}
ALKALINE_AND_RARE_EARTH = {"Ca", "Sr", "Y", "Zr", "Ba"}
LIGHT_ANIONS = {"N", "O", "F"}
HEAVY_ANIONS = {"S", "Cl", "Br", "I"}

UNWANTED_SPECIES = {
    pg.Species("Ta", 4), pg.Species("Ta", 3),
    pg.Species("Se", 4), pg.Species("Se", 6),
    pg.Species("Te", 4), pg.Species("Te", 6),
    pg.Species("Ge", 4), pg.Species("P", 3),
}


def load_reference_compositions(path: Path) -> list:
    """Load composition strings from a CSV with a 'composition' column."""
    with open(path, encoding="utf-8") as f:
        return [row["composition"] for row in DictReader(f)]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    if not REPS_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {REPS_FILE}. "
            "Run from repo root and ensure examples/data exists."
        )
    references = load_reference_compositions(REPS_FILE)
    log.info("Loaded %d reference compositions from %s", len(references), REPS_FILE.name)

    elements = LI | P_BLOCK | POST_TRANSITION | ALKALINE_AND_RARE_EARTH | LIGHT_ANIONS | HEAVY_ANIONS
    species = SpeciesCollection.for_elements(elements)
    species = species.difference(UNWANTED_SPECIES)

    query = IonicComposition(species)

    query.include_elements_quantity(LI, lb=0.2)
    query.include_elements_quantity(P_BLOCK, lb=0.05)
    query.include_elements_quantity(POST_TRANSITION, lb=0.05)
    query.include_elements_quantity(ALKALINE_AND_RARE_EARTH, lb=0.05)
    query.include_elements_quantity(LIGHT_ANIONS, lb=0.05)
    query.include_elements_quantity(HEAVY_ANIONS, lb=0.05)

    query.distinct_elements(ub=DISTINCT_ELEMENTS_MAX)
    query.elmd_far_from_all(references, EMD_MIN_DISTANCE)
    query.total_atoms(lb=TOTAL_ATOMS_LB, ub=TOTAL_ATOMS_UB)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    log.info("Solving for up to %d compositions ...", NUM_RESULTS)
    for _ in range(NUM_RESULTS):
        out = query.get_next(as_frac=True)
        if out is None:
            log.info("No more solutions after %d result(s)", len(results))
            break
        comp, _ = out
        results.append(comp)
        log.info("Found %d/%d: %s", len(results), NUM_RESULTS, comp)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(str(r) + "\n")
    log.info("Done — %d composition(s) written to %s", len(results), OUTPUT_FILE)


if __name__ == "__main__":
    main()
