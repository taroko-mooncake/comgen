"""Generate Mg-ion compositions in Li6PS5Cl-like stoichiometry across multiple sizes.

Uses the same radius-ratio constraints as like_ratio_query. Loops over total
atom counts 10-13; for each size n, constrains Mg in [3/n, 6/n], one cation 1/n,
anions 6/n (Li6PS5Cl-like). Appends all results to mg_like_ratio_3.txt.
"""
from pathlib import Path
from fractions import Fraction

from comgen import SpeciesCollection, IonicComposition
import pymatgen.core as pg

examples_dir = Path(__file__).resolve().parent.parent
output_dir = examples_dir / "output"
output_file = output_dir / "mg_like_ratio_3.txt"
num_results = 3

def like_Li6PS5Cl(denom):
    Mg = {'Mg'}
    A = {'S', 'Se', 'Te', 'B', 'Al', 'Si', 'P', 'Zn', 
        'Ta', 'Sn', 'Ge', 'Ga', 'K', 'Ca', 'Sr', 'Y', 
        'Zr', 'Ba', 'La', 'Gd', 'Mn', 
        'N', 'O', 'F', 'Cl', 'Br', 'I'}

    sps = SpeciesCollection.for_elements(A)
    mg_sps = SpeciesCollection.for_elements(Mg)

    mg_sps = mg_sps.having_charge(2)
    sps1 = sps.having_charge({1,2,3,4,5,6,7,8,9})
    sps2 = sps.having_charge({-1,-2,-3,-4,-5,-6,-7,-8,-9})
    sps.update(mg_sps)

    query = IonicComposition(sps, precision=0.01)

    query.ion_pair_radius_ratio(mg_sps, sps1, lb=1.55, ub=1.85)
    query.ion_pair_radius_ratio(mg_sps, sps2, lb=0.45, ub=0.55)

    query.distinct_elements(lb=3, ub=6)

    query.total_atoms(denom)

    query.include_elements_quantity(Mg, lb=Fraction(3, denom), ub=Fraction(6, denom))

    query.include_species_quantity(sps1, Fraction(1, denom))
    query.include_species_quantity(sps2, Fraction(6, denom)) 

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f_out:
        for _ in range(num_results):
            out = query.get_next(as_frac=True)
            if out is None:
                break
            res, _ = out
            f_out.write(str(res) + '\n')

for n in range(10, 14):
    like_Li6PS5Cl(n)
