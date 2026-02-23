"""Generate Mg-ion compositions with relaxed stoichiometry and radius-ratio constraints.

Same ion-pair radius-ratio ranges as like_ratio_query (Mg-cation 1.55-1.85,
Mg-anion 0.45-0.55), but with relaxed composition bounds: Mg between 3/13 and
6/13, cations >= 1/13, anion fraction unconstrained.
"""

from fractions import Fraction
from pathlib import Path

from comgen import IonicComposition, SpeciesCollection

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = EXAMPLES_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "mg_like_ratio_2.txt"

NUM_RESULTS = 5
TOTAL_ATOMS = 13

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


def main() -> None:
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
    query.total_atoms(TOTAL_ATOMS)

    query.include_elements_quantity(MG, lb=Fraction(3, 13), ub=Fraction(6, 13))
    query.include_species_quantity(cation_species, lb=Fraction(1, 13))

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
