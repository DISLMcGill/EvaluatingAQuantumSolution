"""
Run the production benchmark on a single test case.

Same harness path as the full ``run_unit_partition_experiment`` runner,
but scoped to one problem so a QPU run costs roughly two submissions
(one per hardware solver).  Useful for confirming the end-to-end
benchmark works before committing to a tier-1 sweep.

Solvers
-------
* ILP       -- CBC reference (local, no QPU)
* SQA_HW    -- S1 on real D-Wave hardware
* SQA_SF_HW -- S2 on real D-Wave hardware

S3 (SQA_DW_HW) is intentionally excluded -- same reason it's excluded
from the default hardware registry.  See
``solvers/quantum_hardware_solvers/SQA_DW_HW.py``.

QPU footprint
-------------
Two submissions, 100 reads each, 20 us anneal time per read.  Roughly
a few seconds of QPU access time total.

Usage
-----
    cd /path/to/QuantumClean
    python single_case_hw_bench.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from solvers.ILP import ILPSolver
from solvers.quantum_hardware_solvers.SQA_HW import SQAHardwareSolver
from solvers.quantum_hardware_solvers.SQA_SF_HW import SQASFHardwareSolver
from util.experiment_execution.run_experiment import (
    _QPU_RESULT_FIELDS,
    run_experiment,
)
from util.experiment_execution.run_unit_partition_experiment import (
    _with_selection_policy,
    result_dir_for,
)
from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    POLICY_LOWEST_ENERGY,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parent
TEST_CASE  = REPO_ROOT / "test_bank" / "unit_partition" / "tier1" / "n3_p4" / "t30" / "n-3_p-4_1.json"
# Output dir is resolved per-run from the selection_policy (see main()).

NUM_READS      = 100  # matches the default for the project-level runner
ANNEALING_TIME = 20   # microseconds


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-case hardware bench against the production registry.",
    )
    parser.add_argument(
        "--legacy-lowest-energy",
        action="store_true",
        help="Use the legacy 'lowest_energy' selection policy and write "
             "results to the original quantum_hardware_results/ tree.  "
             "Default is 'best_feasible', written to "
             "quantum_hardware_results_feasible/.",
    )
    args = parser.parse_args()

    selection_policy = (
        POLICY_LOWEST_ENERGY if args.legacy_lowest_energy
        else POLICY_BEST_FEASIBLE
    )
    output_dir = result_dir_for(hardware=True, selection_policy=selection_policy)

    if not TEST_CASE.exists():
        print(f"ERROR: test case not found: {TEST_CASE}", file=sys.stderr)
        return 1

    print(f"Test case  : {TEST_CASE.relative_to(REPO_ROOT)}")
    print(f"selection_policy: {selection_policy}")
    print(f"Output dir : {output_dir.relative_to(REPO_ROOT)}")
    print(f"num_reads={NUM_READS}, annealing_time={ANNEALING_TIME} us\n")

    # Mirror the production hardware registry from
    # util/experiment_execution/run_unit_partition_experiment.py
    # (ILP + S1_HW + S2_HW; S3_HW intentionally excluded).
    registry = [
        {"name": "ILP",       "class": ILPSolver,           "type": "ilp"},
        {"name": "SQA_HW",    "class": SQAHardwareSolver,   "type": "qpu"},
        {"name": "SQA_SF_HW", "class": SQASFHardwareSolver, "type": "qpu"},
    ]
    registry = _with_selection_policy(registry, selection_policy)

    output_path = run_experiment(
        test_case_paths=[TEST_CASE],
        solver_registry=registry,
        output_dir=output_dir,
        file_prefix="SingleCaseBench",
        num_reads=NUM_READS,
        annealing_time=ANNEALING_TIME,
        note=(
            f"single_case_hw_bench.py -- one-case bench against the "
            f"production registry, selection_policy={selection_policy}"
        ),
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Read the result back and emit a comparison table.  This is the
    # part a person would actually want to look at after the run -- it
    # answers "did S1 and S2 find the ILP-optimal cost? how much QPU
    # time? how big was the embedding?"  without needing to crack open
    # the JSON.
    # -----------------------------------------------------------------------
    payload = json.loads(Path(output_path).read_text())
    (case_key, entry), = payload["results"].items()
    solvers_data = entry["solvers"]

    ilp_cost = solvers_data["ILP"].get("cost")

    print(f"\nResult file : {output_path}")
    print(f"Case        : {case_key}\n")

    headers = [
        "solver", "valid", "cost", "abs_gap", "rel_gap",
        "wall_ms", "qpu_access_us", "phys_qubits", "max_chain",
        "chain_break", "k_viol", "cap_over", "err",
    ]
    rows = []
    for name in ("ILP", "SQA_HW", "SQA_SF_HW"):
        r = solvers_data[name]
        rows.append([
            name,
            "Y" if r.get("valid") else "N",
            r.get("cost"),
            r.get("optimality_gap_absolute"),
            r.get("optimality_gap_relative"),
            r.get("wall_time_ms"),
            r.get("qpu_access_time_us"),
            r.get("physical_qubits"),
            r.get("max_chain_length"),
            r.get("chain_break_fraction"),
            r.get("k_safety_violations"),
            r.get("capacity_overruns"),
            (r.get("error") or "")[:30],
        ])

    widths = [
        max(len(str(row[i])) for row in [headers] + rows)
        for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*[("" if v is None else v) for v in row]))

    # Identity / chip info for the two QPU runs.
    print("\nQPU identity:")
    for name in ("SQA_HW", "SQA_SF_HW"):
        r = solvers_data[name]
        print(f"  {name:10s} chip={r.get('chip_id')} problem_id={r.get('problem_id')}")

    # -----------------------------------------------------------------------
    # Fail loudly if any QPU run silently dropped metadata, or if the
    # schema is incomplete on either QPU row.
    # -----------------------------------------------------------------------
    rc = 0
    for name in ("SQA_HW", "SQA_SF_HW"):
        r = solvers_data[name]
        if r.get("error"):
            print(f"\n{name} returned an error: {r['error']}", file=sys.stderr)
            rc = max(rc, 2)
            continue
        missing = [f for f in _QPU_RESULT_FIELDS if f not in r]
        if missing:
            print(f"\n{name} result missing fields: {missing}", file=sys.stderr)
            rc = max(rc, 3)
        nulls = [
            f for f in (
                "chip_id", "problem_id", "physical_qubits",
                "embedding", "qpu_access_time_us", "qpu_timing",
            )
            if r.get(f) in (None, {}, [])
        ]
        if nulls:
            print(f"\n{name} left identity/timing fields null: {nulls}", file=sys.stderr)
            rc = max(rc, 4)
        if not r.get("solution"):
            print(f"\n{name} did not persist a 'solution' field", file=sys.stderr)
            rc = max(rc, 5)

    if rc == 0:
        print("\nOK -- all three solvers produced full, schema-complete result rows.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
