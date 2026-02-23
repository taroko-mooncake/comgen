"""Generate Mg-ion compositions that are close to filtered Li-ion conductors.

Loads Li-ion compositions from LiIonDatabase.csv (filtered by conductivity
target >= 1e-3 S/cm and temperature 15-35 °C), then enumerates Mg-ion
compositions within a given distance of at least one of them. Explores
Mg analogues of good Li-ion conductors.
"""
from comgen import SpeciesCollection, IonicComposition
from csv import DictReader 
from pathlib import Path

examples_dir = Path(__file__).resolve().parent.parent
data_dir = examples_dir / "data"
output_dir = examples_dir / "output"
output_file = output_dir / "mg_distance_query.txt"
li_conductors_file = data_dir / "LiIonDatabase.csv"

distance = 3
num_results = 5

Mg = {'Mg'}
A = {'S', 'Se', 'Te', 'B', 'Al', 'Si', 'P', 'Zn', 
     'Ta', 'Sn', 'Ge', 'Ga', 'K', 'Ca', 'Sr', 'Y', 
     'Zr', 'Ba', 'La', 'Gd',
     'N', 'O', 'F', 'Cl', 'Br', 'I'}
elts = Mg|A
sps = SpeciesCollection.for_elements(elts)

query = IonicComposition(sps, precision=0.01)
query.include_elements_quantity(Mg, lb=0.1)
query.distinct_elements(ub=6)
query.total_atoms(lb=10, ub=20)
comps = []
with open(li_conductors_file) as f:
    for row in DictReader(f):
        if float(row['target']) >= float(1E-3):
            if float(row['temperature']) >= 15 and float(row['temperature']) <= 35:
                comps.append(row['composition'].strip('"'))
    query.elmd_close_to_one(comps, distance)

output_dir.mkdir(parents=True, exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f_out:
    for _ in range(num_results):
        out = query.get_next(as_frac=True)
        if out is None:
            break
        res, _ = out
        f_out.write(str(res) + '\n')
