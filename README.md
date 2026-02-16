# comgen

**comgen** generates ionic compositions that satisfy user-defined constraints. It uses the Z3 SMT solver to search the composition space and supports charge balance, element/species bounds, radius-ratio rules, distance from known compositions (Earth Mover’s Distance), synthesis-from-ingredients, and optional ONNX-based property filters.

---

## Setup

### Requirements

- Python 3.8+
- See `requirements.txt` and `setup.py` for dependency versions.

### Install

From the repository root:

```bash
pip install -e .
```

Or install dependencies only (if you run scripts with `python -m` from the project root):

```bash
pip install -r requirements.txt
```

Core dependencies:

- **pymatgen** – species, compositions, radii
- **z3-solver** – constraint solving
- **onnx** – optional, for ONNX-based category prediction in some examples
- **numpy** – used by the package

---

## Running programs

### Run from project root

Examples and tests assume the **comgen** package is on `PYTHONPATH`. Run from the repo root so that `import comgen` resolves:

```bash
cd /path/to/comgen
python examples/garnet/query.py
python examples/Li_ion_conductors/distance_query.py
# etc.
```

### Run an example that needs data

Some examples read CSV files from `examples/data/`. Run them from the repo root so relative paths work:

```bash
python examples/Li_ion_conductors/distance_query.py
python examples/Li_ion_conductors/starting_materials_query.py
python examples/Mg_ion_conductors/distance_query.py
```

### Run tests

```bash
python -m pytest tests/
# or
python tests/distance.py
```

---

## Generating outputs

### Basic usage

1. Define a **species collection** (allowed ions).
2. Build an **`IonicComposition`** query and add constraints.
3. Call **`get_next()`** repeatedly to get solutions (each new solution is excluded from the next call).

Example (garnet-like compositions: charge +2/+3 cations, SiO₄ anion, radius ratio and O fraction fixed):

```python
from comgen import IonicComposition, SpeciesCollection, PolyAtomicSpecies

def get_permitted_ions():
    sps = SpeciesCollection.for_elements()
    sps = sps.having_charge([2, 3])
    sps.update(PolyAtomicSpecies("SiO4", -4))
    return sps

sps = get_permitted_ions()
query = IonicComposition(sps)

query.distinct_elements(4)
query.include_elements_quantity({'O'}, 12/20)
query.ion_pair_radius_ratio(
    sps.having_charge(2), sps.having_charge(3),
    cn1='VIII', cn2='VI', lb=1.5, ub=1.9
)
print(query.get_next())
```

### Species collections

- **`SpeciesCollection.for_elements(elements=None)`**  
  Builds a collection of allowed species for the given set of elements (or all elements if `elements` is `None`). Uses pymatgen `Species` (e.g. oxidation states).
- **`sps.having_charge(charges)`**  
  Filter by charge(s), e.g. `[2, 3]`.
- **`sps.difference(...)`**  
  Remove specific species (e.g. certain oxidation states).
- **`PolyAtomicSpecies("SiO4", -4)`**  
  Add polyatomic ions; can be combined with `sps.update(...)`.

### Common constraints (query methods)

| Method | Purpose |
|--------|--------|
| `include_elements_quantity(elements, exact=None, lb=..., ub=...)` | Bound fractional quantity of given elements. |
| `include_elements(elements, exact=..., lb=..., ub=...)` | Bound count of elements (with unit cell). |
| `distinct_elements(exact=..., lb=..., ub=...)` | Limit number of distinct elements. |
| `total_atoms(lb=..., ub=...)` | Bound total atom count per formula/unit cell. |
| `ion_pair_radius_ratio(sps1, sps2, cn1=..., cn2=..., lb=..., ub=...)` | Require at least one ion pair with radius ratio in range (coordination numbers optional). |
| `ion_pair_radius_difference(sps1, sps2=..., lb=..., ub=...)` | Constrain absolute radius difference for selected pairs. |
| `elmd_far_from_all(compositions, distance)` | Earth Mover’s distance from each reference composition ≥ `distance`. |
| `elmd_close_to_one(compositions, bounds)` | Be close to at least one reference (EMD within bounds). |
| `made_from(ingredients)` | Composition must be synthesizable from given ingredient compositions. |
| `exclude(compositions)` | Exclude specific compositions. |
| `category_prediction(onnx_model, category)` | Apply an ONNX classifier; require given category. |

### Getting results

- **`get_next(as_frac=False)`**  
  Returns the next solution or `None` if no more. With `as_frac=True`, composition is returned as fractional amounts (e.g. for writing to a file). Typical use:

  ```python
  res, model = query.get_next(as_frac=True)
  if res is None:
      break
  # use res (dict of element/species -> quantity string)
  ```

- Repeated calls yield new compositions; previously returned ones are excluded automatically (within the chosen precision).

### Example: writing results to a file

```python
output_file = "results.txt"
num_results = 20
with open(output_file, 'w') as f_out:
    for i in range(num_results):
        res, _ = query.get_next(as_frac=True)
        if res is None:
            break
        f_out.write(str(res) + '\n')
```

### Data files for examples

- **`examples/data/LiIon_reps.csv`** – reference Li-ion compositions (column `composition`).
- **`examples/data/LiIonDatabase.csv`** – Li-ion database with `composition`, `target`, `temperature`, etc.
- **`examples/data/Li_starting_materials.csv`** – starting materials and normed compositions for synthesis constraints.

Examples that use these (e.g. `Li_ion_conductors/distance_query.py`, `starting_materials_query.py`, `Mg_ion_conductors/distance_query.py`) must be run from the repository root so that paths like `examples/data/...` resolve correctly.

### Optional: ONNX model (low formation energy example)

`examples/low_formation_energy/query.py` uses an ONNX model (`ehull_1040_bn.onnx`) to restrict to a given stability category. You need to provide that model file in the same directory (or adjust the path in the script).

---

## Summary

- **Setup:** `pip install -e .` (or `pip install -r requirements.txt`) in the repo root.
- **Run:** From repo root: `python examples/<example_folder>/<script>.py` or `python -m pytest tests/`.
- **Outputs:** Build an `IonicComposition(species)`, add constraints, then call `get_next(as_frac=True)` in a loop and write or process the returned composition dicts; use `examples/data/` when the script expects it.
