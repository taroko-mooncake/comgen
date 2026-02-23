# comgen

**comgen** generates ionic compositions that satisfy user-defined constraints. It uses the [Z3 SMT solver](https://github.com/Z3Prover/z3) to search the composition space and supports charge balance, element/species bounds, radius-ratio rules, distance from known compositions (Earth Mover's Distance), synthesis-from-ingredients, and optional ONNX-based property filters.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Examples](#examples)
  - [Li-Ion Conductors](#li-ion-conductors)
  - [Mg-Ion Conductors](#mg-ion-conductors)
  - [Garnet-Like Compositions](#garnet-like-compositions)
  - [ABX3 Perovskites](#abx3-perovskites)
  - [Low Formation Energy (ONNX)](#low-formation-energy-onnx)
- [Data Files](#data-files)
- [API Reference](#api-reference)
- [Tests](#tests)

---

## Project Structure

```
comgen/
├── comgen/                        # Main package
│   ├── __init__.py                # Exports: IonicComposition, SpeciesCollection, PolyAtomicSpecies
│   ├── query/                     # High-level query API
│   │   ├── base.py                # Base Query class (Z3 solver wrapper)
│   │   ├── ionic.py               # IonicComposition, SingleTarget
│   │   └── common.py              # Pettifor scale, ionic radii helpers
│   ├── constraint_system/         # Z3 constraint implementations
│   │   ├── composition.py         # TargetComposition, UnitCell
│   │   ├── distance.py            # Earth Mover's Distance (EMD)
│   │   ├── synthesis.py           # Synthesis-from-ingredients constraints
│   │   └── nn.py                  # ONNX neural network constraints
│   └── util/                      # Species definitions and data loading
│       ├── species.py             # SpeciesCollection, PolyAtomicSpecies
│       ├── data.py                # Data loading utilities
│       └── data_files/            # Bundled reference data (periodic table, Pettifor scale, etc.)
│
├── examples/                      # Runnable example scripts
│   ├── run_example.py             # Quick-start example
│   ├── Li_ion_conductors/         # Li-ion conductor discovery
│   ├── Mg_ion_conductors/         # Mg-ion conductor discovery
│   ├── garnet/                    # Garnet-like structure generation
│   ├── perovskite/                # ABX3 perovskite candidate generation
│   ├── low_formation_energy/      # ONNX-based stability filtering
│   ├── data/                      # CSV data for examples (Li/Mg conductors, perovskites)
│   └── output/                    # Generated results (one file per script)
│
├── tests/                         # Test suite
│   ├── distance.py                # EMD constraint tests
│   ├── nn.py                      # ONNX constraint tests
│   └── test_model.onnx            # Small ONNX model for testing
│
├── pyproject.toml                 # Package metadata and dependencies
├── requirements.txt               # Dependency list
└── main.py                        # Minimal entry point
```

---

## Setup

### Prerequisites

- **Python 3.8** or newer
- **pip** (or [uv](https://docs.astral.sh/uv/) as an alternative)

### 1. Clone the repository

```bash
git clone <repo-url>
cd comgen
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install the package

Install in editable mode so that `import comgen` works from anywhere:

```bash
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

This installs the core dependencies:

| Package | Purpose |
|---------|---------|
| **pymatgen** >= 2022.5.26 | Species, compositions, ionic radii |
| **z3-solver** >= 4.8.17.0 | Constraint solving (SMT) |
| **numpy** >= 1.21 | Numerical operations |

### 4. (Optional) Install ONNX support

The `low_formation_energy` and `perovskite` examples can use an ONNX model for stability prediction. Install the optional extra:

```bash
pip install -e ".[onnx]"
```

This adds **onnx** >= 1.15.0. For runtime inference in the perovskite example, also install:

```bash
pip install onnxruntime
```

### Alternative: install dependencies only

If you prefer not to install the package and instead run scripts from the repository root:

```bash
pip install -r requirements.txt
```

---

## Quick Start

From the repository root, run the main example. It generates Li-ion compositions that are far from known references (using Earth Mover's Distance) and writes results to `examples/output/example_compositions.txt`:

```bash
cd /path/to/comgen
pip install -e .
python3 examples/run_example.py
```

**What it does:**

1. Loads reference compositions from `examples/data/LiIon_reps.csv`
2. Builds a species collection for Li + p-block + post-transition + alkaline/rare-earth + anion elements
3. Adds constraints: Li fraction >= 20%, each group >= 5%, at most 6 distinct elements, 10-15 total atoms, EMD >= 5 from all references
4. Solves for 10 compositions and writes them to the output file

**Expected output** (fractional compositions):

```
{'Ca': '1/15', 'Al': '1/15', 'N': '4/15', 'Mg': '1/15', 'S': '1/15', 'Li': '7/15'}
{'Sr': '1/15', 'N': '2/5', 'Ga': '1/5', 'P': '1/15', 'Li': '1/5', 'Br': '1/15'}
...
```

---

## Examples

All example scripts should be run **from the repository root** so that relative paths to `examples/data/` and `examples/output/` resolve correctly. Each script writes its results to `examples/output/`.

### Li-Ion Conductors

#### Distance query

Generates Li-ion compositions that are far from all known representatives in `LiIon_reps.csv`. Uses the full element set with explicit fraction bounds per group.

```bash
python3 examples/Li_ion_conductors/distance_query.py
```

| Parameter | Value |
|-----------|-------|
| Data file | `examples/data/LiIon_reps.csv` |
| EMD distance | >= 5 from every reference |
| Total atoms | 10 - 15 |
| Distinct elements | <= 6 |
| Results | 20 compositions |
| Output | `examples/output/li_distance_query.txt` |

#### Starting materials query

Generates Li-ion compositions that are synthesisable from known precursors, excludes already-known compositions, and requires novelty (EMD >= 5 from references). Combines synthesis, exclusion, and distance constraints.

```bash
python3 examples/Li_ion_conductors/starting_materials_query.py
```

| Parameter | Value |
|-----------|-------|
| Data files | `examples/data/LiIon_reps.csv`, `examples/data/Li_starting_materials.csv` |
| Constraints | EMD >= 5, synthesisable from ingredients, known compositions excluded |
| Solver timeout | 60 s per solution |
| Reference sub-sampling | 10 references (random seed 42) to reduce solver load |
| Output | `examples/output/li_starting_materials.txt` |

### Mg-Ion Conductors

#### Distance query

Finds Mg-ion compositions that are close to at least one high-performance Li-ion conductor from `LiIonDatabase.csv` (filtered by conductivity >= 1e-3 S/cm at 15-35 C). Explores Mg analogues of good Li conductors.

```bash
python3 examples/Mg_ion_conductors/distance_query.py
```

| Parameter | Value |
|-----------|-------|
| Data file | `examples/data/LiIonDatabase.csv` |
| EMD distance | <= 3 from at least one filtered Li conductor |
| Total atoms | 10 - 20 |
| Mg fraction | >= 10% |
| Output | `examples/output/mg_distance_query.txt` |

#### Radius-ratio query (fixed stoichiometry)

Generates Mg-ion compositions with ion-pair radius ratios typical of Li6PS5Cl-like structures: Mg-cation ratio 1.55-1.85, Mg-anion ratio 0.45-0.55. Uses fixed 13-atom stoichiometry.

```bash
python3 examples/Mg_ion_conductors/like_ratio_query.py
```

| Parameter | Value |
|-----------|-------|
| Total atoms | 13 (fixed) |
| Mg fraction | 6/13 |
| Cation fraction | 1/13 |
| Anion fraction | 6/13 |
| Output | `examples/output/mg_like_ratio.txt` |

#### Radius-ratio query (relaxed bounds)

Same radius-ratio constraints as above but with relaxed composition bounds: Mg between 3/13 and 6/13, cations >= 1/13, anion fraction unconstrained.

```bash
python3 examples/Mg_ion_conductors/like_ratio_query_2.py
```

| Parameter | Value |
|-----------|-------|
| Mg fraction | 3/13 - 6/13 |
| Output | `examples/output/mg_like_ratio_2.txt` |

#### Radius-ratio query (multiple formula sizes)

Loops over total atom counts 10-13 and generates compositions for each size with Li6PS5Cl-like stoichiometry. Appends all results to a single file.

```bash
python3 examples/Mg_ion_conductors/like_ratio_query_3.py
```

| Parameter | Value |
|-----------|-------|
| Atom counts | 10, 11, 12, 13 |
| Results per size | 3 |
| Output | `examples/output/mg_like_ratio_3.txt` |

### Garnet-Like Compositions

Generates compositions that may have garnet-like structure: +2/+3 cations with SiO4 anion, constrained by radius ratio and fixed oxygen fraction.

```bash
python3 examples/garnet/query.py
```

| Parameter | Value |
|-----------|-------|
| Cation charges | +2 and +3 |
| Anion | SiO4^4- (polyatomic) |
| Distinct elements | 4 |
| O fraction | 12/20 = 60% |
| Radius ratio (VIII/VI) | 1.5 - 1.9 |
| Output | `examples/output/garnet_query.txt` |

**Expected output:**

```
{'Si': '3/20', 'O': '3/5', 'Mn': '3/20', 'As': '1/10'}
```

### ABX3 Perovskites

Generates ABX3 organic-inorganic perovskite candidates with partial substitution on all crystallographic sites:

- **A-site**: MA+ (methylammonium), FA+ (formamidinium), GA+ (guanidinium), Cs+
- **B-site**: Pb2+, Sn2+, Hg2+, Cd2+, Zn2+, Fe2+, Ni2+, Co2+, In3+, Bi3+, Ti4+
- **X-site**: Cl-, Br-, I-

The script enforces ABX3 stoichiometry (Sum(A) = k, Sum(B) = k, Sum(X) = 3k for k = 1..4 formula units) with charge balance and supports:

- **Novelty constraint**: ElMD >= 3 from known reference perovskites (`examples/data/reference_perovskites.csv`)
- **Synthesis constraint**: compositions must be achievable from known starting materials (`examples/data/perovskite_starting_materials.csv`)
- **ONNX stability filter** (optional): predicted stability category via a formation-energy classifier

```bash
python3 examples/perovskite/ABX3.py
```

| Parameter | Value |
|-----------|-------|
| Data files | `examples/data/reference_perovskites.csv`, `examples/data/perovskite_starting_materials.csv` |
| Formula units | 1 - 4 |
| Total atoms | 5 - 80 |
| Novelty | ElMD >= 3 from all known references |
| Synthesis | constrained to known starting materials |
| Max candidates | 50 |
| ONNX model (optional) | `examples/low_formation_energy/ehull_1040_bn.onnx` |
| Output | `examples/output/perovskite_ABX3_candidates.txt` |

**Expected output format:**

```
# ABX3 perovskite candidates (17 total)

#1
  composition : {'Ni': 0.019, 'C': 0.074, 'N': 0.185, ...}
  species     : {'Ni2+': 1, '(C1H6N1)+': 1, 'Cl-': 4, 'Br-': 7, ...}
```

> **Note:** If the ONNX model file is not present, stability will not be characterised but candidate generation still works.

### Low Formation Energy (ONNX)

Uses an ONNX neural network model (`ehull_1040_bn.onnx`) to restrict compositions to a given stability category. Runs in two phases: first without the ONNX constraint (fast sanity check), then with the ONNX constraint in a subprocess with a wall-clock timeout.

```bash
pip install -e ".[onnx]"
python3 examples/low_formation_energy/query.py
```

| Parameter | Value |
|-----------|-------|
| ONNX model | `examples/low_formation_energy/ehull_1040_bn.onnx` (not included; provide your own) |
| Total atoms | 5 - 10 |
| Distinct elements | <= 4 |
| Target category | 0 (stable) |
| Solver timeout | 600 s |
| Output | `examples/output/low_formation_energy_query.txt` |

> **Note:** You need the `ehull_1040_bn.onnx` model file in `examples/low_formation_energy/` to run this example. The ONNX constraint embeds the neural network directly into the Z3 solver, which can be slow for large models.

---

## Data Files

### `examples/data/` — Example data files

| File | Description | Used by |
|------|-------------|---------|
| `LiIon_reps.csv` | Reference Li-ion compositions (column: `composition`). Used so queries can find compositions *far from* these known compounds. | `run_example.py`, `distance_query.py`, `starting_materials_query.py` |
| `LiIonDatabase.csv` | Li-ion conductor database with columns `composition`, `target` (conductivity), `temperature`. Used to filter for high-performance conductors that serve as templates for Mg analogues. | `Mg_ion_conductors/distance_query.py` |
| `Li_starting_materials.csv` | Starting materials with `norm_composition` and `starting_material` flag. Used to constrain compositions to be synthesisable from known precursors and to exclude existing compositions. | `starting_materials_query.py` |
| `reference_perovskites.csv` | Known perovskite compositions (column: `composition`). Used as the reference set for ElMD novelty constraints so generated candidates are distinct from known compounds. | `perovskite/ABX3.py` |
| `perovskite_starting_materials.csv` | Precursor chemicals (column: `composition`, e.g. `Pb1I2`, `C1H6N1I1`). Used for the synthesis mass-balance constraint. | `perovskite/ABX3.py` |

---

## API Reference

### Core classes

```python
from comgen import IonicComposition, SpeciesCollection, PolyAtomicSpecies
```

#### `SpeciesCollection`

Manages the set of allowed ionic species for a query.

| Method | Description |
|--------|-------------|
| `SpeciesCollection.for_elements(elements=None)` | Build a collection for given elements (or all elements if `None`). Uses pymatgen `Species` with known oxidation states. |
| `.having_charge(charges)` | Filter to species with the given charge(s), e.g. `[2, 3]` or `{-1, -2}`. |
| `.difference(species_set)` | Remove specific species from the collection. |
| `.update(species)` | Add species (e.g. a `PolyAtomicSpecies`) to the collection. |

#### `PolyAtomicSpecies`

Represents polyatomic ions. Combine with `SpeciesCollection.update()`:

```python
sps.update(PolyAtomicSpecies("SiO4", -4))
```

#### `IonicComposition`

The main query class. Extends `SingleTarget` with automatic charge balance and electronegativity ordering.

```python
query = IonicComposition(species, precision=0.1)
```

### Constraint methods

| Method | Purpose |
|--------|---------|
| `include_elements_quantity(elements, exact=None, *, lb=None, ub=None)` | Bound the total normalised fraction of a set of elements. |
| `include_elements(elements, exact=None, *, lb=None, ub=None)` | Bound the count of elements from a set that appear in the composition. |
| `include_elements_count(elements, exact=None, *, lb=None, ub=None)` | Bound the absolute integer atom count for elements (requires unit cell). |
| `include_species_quantity(species, exact=None, *, lb=None, ub=None)` | Bound the total normalised fraction of a set of species (element + oxidation state). |
| `distinct_elements(exact=None, *, lb=None, ub=None)` | Limit the number of distinct elements. |
| `total_atoms(exact=None, *, lb=None, ub=None)` | Bound total atom count per unit cell. Creates a `UnitCell` internally. |
| `ion_pair_radius_ratio(sps1, sps2, cn1=None, cn2=None, *, lb=None, ub=None)` | Require at least one ion pair whose Shannon radius ratio is in `[lb, ub]`. Coordination numbers (e.g. `"VIII"`, `"VI"`) are optional. |
| `ion_pair_radius_difference(sps1, sps2=None, *, lb=None, ub=None)` | Exclude ion pairs whose absolute radius difference is outside `[lb, ub]`. |
| `elmd_far_from_all(compositions, distance)` | Earth Mover's Distance from every reference >= `distance`. |
| `elmd_close_to_one(compositions, bounds)` | EMD to at least one reference <= `bounds`. |
| `made_from(ingredients)` | Composition must be a non-negative linear combination of ingredient compositions. |
| `exclude(compositions)` | Exclude specific compositions from the search. |
| `category_prediction(onnx_model, category)` | Require the ONNX classifier to predict the given category. |

### Getting results

```python
result = query.get_next(as_frac=True, timeout_ms=60000)
if result is None:
    print("No more solutions")
else:
    composition, monitored_vars = result
    print(composition)
    # e.g. {'Li': '1/5', 'La': '1/10', 'Zr': '1/10', 'O': '3/5'}
```

- `as_frac=True` returns quantities as fraction strings (e.g. `'1/5'`); `False` returns rounded floats.
- `timeout_ms` sets a per-call solver timeout in milliseconds.
- Each call automatically excludes previous solutions (within `precision`) so repeated calls yield distinct compositions.

### Typical workflow

```python
from comgen import IonicComposition, SpeciesCollection, PolyAtomicSpecies

# 1. Define allowed species
sps = SpeciesCollection.for_elements({"Li", "La", "Zr", "O", "F"})

# 2. Build query (charge balance is automatic)
query = IonicComposition(sps)

# 3. Add constraints
query.distinct_elements(lb=3, ub=5)
query.include_elements_quantity({"Li"}, lb=0.1)
query.total_atoms(lb=8, ub=15)

# 4. Solve iteratively
for _ in range(10):
    out = query.get_next(as_frac=True)
    if out is None:
        break
    comp, _ = out
    print(comp)
```

### Writing results to a file

```python
from pathlib import Path

output_dir = Path("examples/output")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
    for _ in range(20):
        out = query.get_next(as_frac=True)
        if out is None:
            break
        comp, _ = out
        f.write(str(comp) + "\n")
```

---

## Tests

The test suite covers EMD distance calculations and ONNX neural network constraint embedding.

```bash
# Run all tests
python3 -m pytest tests/

# Run individual test scripts
python3 tests/distance.py
python3 tests/nn.py
```

The `tests/test_model.onnx` file is a small ONNX model used by `tests/nn.py` to verify forward pass, class selection, and output bounding constraints.

---

## Summary

| Step | Command |
|------|---------|
| Install | `pip install -e .` (or `pip install -r requirements.txt`) |
| Quick start | `python3 examples/run_example.py` |
| Run any example | `python3 examples/<folder>/<script>.py` (from repo root) |
| Run tests | `python3 -m pytest tests/` |
| ONNX support | `pip install -e ".[onnx]"` |

All example scripts write results to `examples/output/`. Build an `IonicComposition(species)`, add constraints, then call `get_next(as_frac=True)` in a loop. Use `examples/data/` for CSV reference data.
