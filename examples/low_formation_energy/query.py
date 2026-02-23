"""
Minimal example: query ONNX model for stable/metastable/unstable formation energy.
Model expects normed composition array as input.

Runs in two phases: first solve without the ONNX constraint (fast), then with
ONNX in a subprocess with a real wall-clock timeout so the run cannot hang.
"""

import logging
import multiprocessing
import sys
import time
from pathlib import Path

# Ensure repo root on path when run as script
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from comgen import IonicComposition, SpeciesCollection
import onnx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

examples_dir = _repo_root / "examples"
output_dir = examples_dir / "output"
output_file = output_dir / "low_formation_energy_query.txt"

# Tighter bounds so the solver has a chance
TOTAL_ATOMS_LB, TOTAL_ATOMS_UB = 5, 10
DISTINCT_ELEMENTS_UB = 4
ONNX_TIMEOUT_SEC = 600  # Wall-clock timeout: solver process is killed after this


def _run_phase2_in_subprocess():
    """Build query and run solver in child process. Returns (comp_dict, None) or (None, None)."""
    from comgen import IonicComposition, SpeciesCollection
    import onnx
    model_path = _script_dir / "ehull_1040_bn.onnx"
    onnx_model = onnx.load(str(model_path))
    sps = SpeciesCollection.for_elements()
    query = IonicComposition(sps)
    query.total_atoms(lb=TOTAL_ATOMS_LB, ub=TOTAL_ATOMS_UB)
    query.distinct_elements(ub=DISTINCT_ELEMENTS_UB)
    query.category_prediction(onnx_model, 0)
    out = query.get_next(as_frac=True)
    if out is not None:
        comp, _ = out
        return comp
    return None


def main():
    log.info("Loading ONNX model")
    model_path = _script_dir / "ehull_1040_bn.onnx"
    onnx_model = onnx.load(str(model_path))

    sps = SpeciesCollection.for_elements()

    # --- Phase 1: without ONNX ---
    log.info("Phase 1: Building query with total_atoms and distinct_elements only (no ONNX)")
    query1 = IonicComposition(sps)
    query1.total_atoms(lb=TOTAL_ATOMS_LB, ub=TOTAL_ATOMS_UB)
    query1.distinct_elements(ub=DISTINCT_ELEMENTS_UB)
    log.info("Phase 1: Constraints = %d", len(query1.constraints))
    t0 = time.perf_counter()
    out1 = query1.get_next(as_frac=True)
    elapsed1 = time.perf_counter() - t0
    if out1 is not None:
        comp1, _ = out1
        log.info("Phase 1: Solved in %.1fs -> %s", elapsed1, comp1)
    else:
        log.info("Phase 1: No solution in %.1fs (unexpected with these bounds)", elapsed1)

    # --- Phase 2: with ONNX in a subprocess (real timeout: process is killed) ---
    log.info("Phase 2: Building full query with ONNX category 0 (stable)")
    query2 = IonicComposition(sps)
    query2.total_atoms(lb=TOTAL_ATOMS_LB, ub=TOTAL_ATOMS_UB)
    query2.distinct_elements(ub=DISTINCT_ELEMENTS_UB)
    query2.category_prediction(onnx_model, 0)
    log.info("Phase 2: Constraints = %d (includes NN)", len(query2.constraints))
    log.info("Phase 2: Solving in subprocess (timeout=%ds, process will be killed if it hangs)...", ONNX_TIMEOUT_SEC)
    t0 = time.perf_counter()
    with multiprocessing.Pool(1) as pool:
        async_result = pool.apply_async(_run_phase2_in_subprocess)
        try:
            comp = async_result.get(timeout=ONNX_TIMEOUT_SEC)
        except multiprocessing.TimeoutError:
            comp = None
            log.warning("Phase 2: Timeout after %ds — killing solver process", ONNX_TIMEOUT_SEC)
    elapsed2 = time.perf_counter() - t0

    if comp is not None:
        log.info("Phase 2: Solved in %.1fs -> %s", elapsed2, comp)
        print(comp)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str(comp) + "\n")
        log.info("Wrote %s", output_file)
    else:
        log.info("Phase 2: No solution (or timed out) in %.1fs", elapsed2)
        print("No solution found.")


if __name__ == "__main__":
    main()
