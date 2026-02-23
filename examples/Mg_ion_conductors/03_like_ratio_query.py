"""Generate Mg-ion compositions across multiple formula sizes with radius-ratio constraints.

Uses the same radius-ratio constraints as like_ratio_query (Mg-cation
1.55-1.85, Mg-anion 0.45-0.55). Loops over total atom counts 10-13; for each
size n, constrains Mg in [3/n, 6/n], one cation at 1/n, anions at 6/n
(Li6PS5Cl-like stoichiometry). Appends all results to mg_like_ratio_3.txt.
"""

from fractions import Fraction
from pathlib import Path

from comgen import IonicComposition, SpeciesCollection

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = EXAMPLES_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "mg_like_ratio_3.txt"

NUM_RESULTS_PER_SIZE = 3
ATOM_COUNT_RANGE = range(10, 14)

MG_CATION_RATIO_LB = 1.55
MG_CATION_RATIO_UB = 1.85
MG_ANION_RATIO_LB = 0.45
MG_ANION_RATIO_UB = 0.55

MG = {"Mg"}
OTHER_ELEMENTS = {
    "S", "Se", "Te", "B", "Al", "Si", "P", "Zn",
    "Ta", "Sn", "Ge", "Ga", "K", "Ca", "Sr", "Y",
    "Zr", "Ba", "La", "Gd", "Mn",
    "N", "O", "F", "Cl", "Br", "I",
}


def query_for_size(total_atoms: int) -> IonicComposition:
    """Build a query for a given formula size with Li6PS5Cl-like stoichiometry."""
    other_species = SpeciesCollection.for_elements(OTHER_ELEMENTS)
    mg_species = SpeciesCollection.for_elements(MG).having_charge(2)

    cation_species = other_species.having_charge({1, 2, 3, 4, 5, 6, 7, 8, 9})
    anion_species = other_species.having_charge({-1, -2, -3, -4, -5, -6, -7, -8, -9})

    all_species = SpeciesCollection(other_species)
    all_species.update(mg_species)

    query = IonicComposition(all_species, precision=0.01)

    query.ion_pair_radius_ratio(mg_species, cation_species,
                                lb=MG_CATION_RATIO_LB, ub=MG_CATION_RATIO_UB)
    query.ion_pair_radius_ratio(mg_species, anion_species,
                                lb=MG_ANION_RATIO_LB, ub=MG_ANION_RATIO_UB)

    query.distinct_elements(lb=3, ub=6)
    query.total_atoms(total_atoms)

    n = total_atoms
    query.include_elements_quantity(MG, lb=Fraction(3, n), ub=Fraction(6, n))
    query.include_species_quantity(cation_species, Fraction(1, n))
    query.include_species_quantity(anion_species, Fraction(6, n))

    return query


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for total_atoms in ATOM_COUNT_RANGE:
            query = query_for_size(total_atoms)
            for _ in range(NUM_RESULTS_PER_SIZE):
                out = query.get_next(as_frac=True)
                if out is None:
                    break
                res, _ = out
                f_out.write(str(res) + "\n")


if __name__ == "__main__":
    main()
