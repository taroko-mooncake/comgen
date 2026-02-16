# comgen

**comgen** generates ionic compositions that satisfy user-defined constraints. It uses the Z3 SMT solver to search the composition space and supports charge balance, element/species bounds, radius-ratio rules, distance from known compositions (Earth Mover’s Distance), synthesis-from-ingredients, and optional ONNX-based property filters.

---

## Setup

### Requirements

- Python 3.8+
- See `requirements.txt` and `pyproject.toml` for dependency versions.

### Install

From the repository root:

```bash
pip install -e .
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

For the ONNX-based example (`examples/low_formation_energy/query.py`), install the optional extra (avoids C++ build issues on some systems if you skip it):

```bash
pip install -e ".[onnx]"
# or
uv pip install -e ".[onnx]"
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

### Quick start

From the repository root, run the main example (generates Li-ion compositions and writes to `examples/output/example_compositions.txt`):

```bash
cd /path/to/comgen
pip install -e .
python examples/run_example.py
```

### Run from project root

Examples and tests assume the **comgen** package is on `PYTHONPATH`. Run from the repo root so that `import comgen` resolves:

```bash
cd /path/to/comgen
python examples/run_example.py   # recommended first run
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
from pathlib import Path
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "results.txt"
num_results = 20
with open(output_file, "w", encoding="utf-8") as f_out:
    for _ in range(num_results):
        out = query.get_next(as_frac=True)
        if out is None:
            break
        res, _ = out
        f_out.write(str(res) + "\n")
```

### Example data and queries

All example scripts that write results save to **`examples/output/`** (one file per script). Run examples from the repository root so paths to `examples/data/` and `examples/output/` resolve.

#### Data files (`examples/data/`)

| File | Description |
|------|--------------|
| **LiIon_reps.csv** | Reference Li-ion compositions (column `composition`). Used as a list of “known” compositions so queries can ask for new compositions *far from* (or similar to) these. |
| **LiIonDatabase.csv** | Li-ion conductor database with columns such as `composition`, `target`, `temperature`. Used to filter compositions by property (e.g. conductivity, temperature) and then constrain new compositions to be *close to* one of them (e.g. for Mg-ion analogues). |
| **Li_starting_materials.csv** | Starting materials and their `norm_composition`; a column indicates whether each row is a starting material. Used to constrain compositions to be *synthesizable from* a given set of precursors and to *exclude* existing compositions. |

#### Example queries

| Script | What it does | Output file |
|--------|----------------|-------------|
| **run_example.py** | Li-ion compositions *far from* all references in `LiIon_reps.csv` (EMD). Simplified element set and bounds for a fast first run. | `examples/output/example_compositions.txt` |
| **Li_ion_conductors/distance_query.py** | Li-ion compositions *far from* all references in `LiIon_reps.csv`. Full element set and explicit fraction bounds per group. | `examples/output/li_distance_query.txt` |
| **Li_ion_conductors/starting_materials_query.py** | Li-ion compositions *far from* `LiIon_reps.csv` and *synthesizable from* ingredients in `Li_starting_materials.csv`; also *excludes* all compositions in that file. | `examples/output/li_starting_materials.txt` |
| **Mg_ion_conductors/distance_query.py** | Mg-ion compositions *close to* at least one filtered Li-ion composition from `LiIonDatabase.csv` (target and temperature filters). Explores Mg analogues of good Li conductors. | `examples/output/mg_distance_query.txt` |
| **Mg_ion_conductors/like_ratio_query.py** | Mg-ion compositions with fixed 13-atom stoichiometry and radius-ratio constraints (Mg²⁺ vs cations and anions), inspired by Li₆PS₅Cl. No CSV. | `examples/output/mg_like_ratio.txt` |
| **Mg_ion_conductors/like_ratio_query_2.py** | Same radius-ratio idea with looser Mg and anion bounds. | `examples/output/mg_like_ratio_2.txt` |
| **Mg_ion_conductors/like_ratio_query_3.py** | Same radius-ratio idea over formula sizes 10–13 atoms; appends results for each size. | `examples/output/mg_like_ratio_3.txt` |
| **garnet/query.py** | Garnet-like compositions (+2/+3 cations, SiO₄ anion, radius ratio). Prints one result to stdout. | (stdout only) |
| **low_formation_energy/query.py** | Compositions classified as a given stability category by an ONNX model. Requires `.[onnx]` and the model file. Prints one result. | (stdout only) |

### Optional: ONNX (low formation energy example)

`examples/low_formation_energy/query.py` uses an ONNX model (`ehull_1040_bn.onnx`) to restrict to a given stability category. Install the optional dependency: `pip install -e ".[onnx]"` (or `uv pip install -e ".[onnx]"`). You also need the model file in the same directory as the script.

---

## Summary

- **Setup:** `pip install -e .` (or `pip install -r requirements.txt`) in the repo root.
- **Run:** From repo root: `python examples/<example_folder>/<script>.py` or `python -m pytest tests/`.
- **Outputs:** Example queries write to `examples/output/` (see table above). Build an `IonicComposition(species)`, add constraints, then call `get_next(as_frac=True)` in a loop; use `examples/data/` when the script expects CSV data.
