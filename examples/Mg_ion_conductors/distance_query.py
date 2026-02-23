"""Generate Mg-ion compositions close to filtered Li-ion conductors.

Loads Li-ion compositions from LiIonDatabase.csv (filtered by conductivity
target >= 1e-3 S/cm and temperature 15-35 C), then enumerates Mg-ion
compositions within a given Earth Mover's Distance of at least one of them.
Explores Mg analogues of good Li-ion conductors.
"""

from csv import DictReader
from pathlib import Path

from comgen import IonicComposition, SpeciesCollection

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXAMPLES_DIR / "data"
OUTPUT_DIR = EXAMPLES_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "mg_distance_query.txt"
LI_CONDUCTORS_FILE = DATA_DIR / "LiIonDatabase.csv"

DISTANCE = 3
NUM_RESULTS = 5
MIN_CONDUCTIVITY = 1e-3
TEMP_LB = 15
TEMP_UB = 35

MG = {"Mg"}
OTHER_ELEMENTS = {
    "S", "Se", "Te", "B", "Al", "Si", "P", "Zn",
    "Ta", "Sn", "Ge", "Ga", "K", "Ca", "Sr", "Y",
    "Zr", "Ba", "La", "Gd",
    "N", "O", "F", "Cl", "Br", "I",
}


def load_good_li_conductors(path: Path) -> list:
    """Return compositions with conductivity >= MIN_CONDUCTIVITY at 15-35 C."""
    compositions = []
    with open(path, encoding="utf-8") as f:
        for row in DictReader(f):
            conductivity = float(row["target"])
            temperature = float(row["temperature"])
            if conductivity >= MIN_CONDUCTIVITY and TEMP_LB <= temperature <= TEMP_UB:
                compositions.append(row["composition"].strip('"'))
    return compositions


def main() -> None:
    elements = MG | OTHER_ELEMENTS
    species = SpeciesCollection.for_elements(elements)

    query = IonicComposition(species, precision=0.01)
    query.include_elements_quantity(MG, lb=0.1)
    query.distinct_elements(ub=6)
    query.total_atoms(lb=10, ub=20)

    li_compositions = load_good_li_conductors(LI_CONDUCTORS_FILE)
    query.elmd_close_to_one(li_compositions, DISTANCE)

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
