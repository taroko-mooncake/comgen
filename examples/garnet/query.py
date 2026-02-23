"""
Generate compositions which may have garnet-like structure.
Restrict the relative size of +2 and +3 cations.
Use SiO4 as the only anion.
"""

from pathlib import Path

from comgen import IonicComposition, SpeciesCollection, PolyAtomicSpecies


def get_permitted_ions():
    sps = SpeciesCollection.for_elements()
    sps = sps.having_charge([2, 3])
    sps.update(PolyAtomicSpecies("SiO4", -4))
    return sps


sps = get_permitted_ions()
query = IonicComposition(sps)

query.distinct_elements(4)
query.include_elements_quantity({"O"}, 12 / 20)
query.ion_pair_radius_ratio(
    sps.having_charge(2), sps.having_charge(3),
    cn1="VIII", cn2="VI", lb=1.5, ub=1.9,
)

out = query.get_next(as_frac=True)
if out is not None:
    comp, _ = out
    print(comp)
    examples_dir = Path(__file__).resolve().parent.parent
    output_dir = examples_dir / "output"
    output_file = output_dir / "garnet_query.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(comp) + "\n")
else:
    print("No solution found.")
