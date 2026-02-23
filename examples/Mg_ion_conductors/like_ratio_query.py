"""Generate Mg-ion compositions with Li-conductor-like ion-pair radius ratios.

Constrains Mg-cation and Mg-anion radius ratios to ranges typical of Li-ion
conductors (e.g. 1.55-1.85 and 0.45-0.55). Uses fixed 13-atom stoichiometry:
Mg 6/13, one cation 1/13, anions 6/13. Writes results to mg_like_ratio.txt.
"""
from pathlib import Path
from fractions import Fraction

from comgen import SpeciesCollection, IonicComposition
import pymatgen.core as pg

examples_dir = Path(__file__).resolve().parent.parent
output_dir = examples_dir / "output"
output_file = output_dir / "mg_like_ratio.txt"
num_results = 5

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

query.total_atoms(13)

query.include_elements_quantity(Mg, Fraction(6,13))
query.include_species_quantity(sps1, Fraction(1,13))
query.include_species_quantity(sps2, Fraction(6,13)) 

output_dir.mkdir(parents=True, exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f_out:
    for _ in range(num_results):
        out = query.get_next(as_frac=True)
        if out is None:
            break
        res, _ = out
        f_out.write(str(res) + '\n')
