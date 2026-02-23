"""Generate Li-ion compositions far from known representatives.

Loads representative compositions from LiIon_reps.csv and enumerates new
Li-ion compositions that are at least a given Earth Mover's Distance away
from all of them. Explores novel compositions distinct from the reference set.
"""

from csv import DictReader
from pathlib import Path

import pymatgen.core as pg

from comgen import IonicComposition, SpeciesCollection

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXAMPLES_DIR / "data"
OUTPUT_DIR = EXAMPLES_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "li_distance_query.txt"
REPS_FILE = DATA_DIR / "LiIon_reps.csv"

DISTANCE = 5
NUM_RESULTS = 20

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
    with open(REPS_FILE, encoding="utf-8") as f:
        comparisons = [row["composition"] for row in DictReader(f)]

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

    query.distinct_elements(ub=6)
    query.elmd_far_from_all(comparisons, DISTANCE)
    query.total_atoms(lb=10, ub=15)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for _ in range(NUM_RESULTS):
            out = query.get_next(as_frac=True)
            if out is None:
                break
            res, _ = out
            f_out.write(str(res) + "\n")


if __name__ == "__main__":
    main()
