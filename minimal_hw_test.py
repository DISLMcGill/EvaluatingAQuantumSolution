"""
Minimal end-to-end smoke test against a real D-Wave QPU.

What this exercises
-------------------
1. D-Wave SDK connectivity (DWaveSampler picks up your configured token).
2. The smallest test case in the bank (n=3, p=4, tightness=0.3) runs
   through ``SQAHardwareSolver``.
3. The full experiment harness path -- the same ``run_experiment``
   function used by the project-level runners -- produces a results
   JSON file with the expanded QPU metadata schema.
4. The persisted JSON actually contains the new fields
   (chip_id, problem_id, embedding, full qpu_timing block, etc.) so we
   know the logging pipeline -- not just the solver class -- is wired
   end to end.

QPU footprint
-------------
One submission, 20 reads, 20 us anneal time per read.  Roughly 0.5 -- 1
seconds of QPU access time total -- the cheapest meaningful run we can
make.

Usage
-----
    cd /path/to/QuantumClean
    python minimal_hw_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from solvers.quantum_hardware_solvers.SQA_HW import SQAHardwareSolver
from util.experiment_execution.run_experiment import (
    _QPU_RESULT_FIELDS,
    run_experiment,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parent
TEST_CASE  = REPO_ROOT / "test_bank" / "unit_partition" / "tier1" / "n3_p4" / "t30" / "n-3_p-4_1.json"
OUTPUT_DIR = REPO_ROOT / "result_bank" / "quantum_hardware_results"

NUM_READS      = 20  # smallest sane number of anneals
ANNEALING_TIME = 20  # microseconds


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> int:
    if not TEST_CASE.exists():
        print(f"ERROR: test case not found: {TEST_CASE}", file=sys.stderr)
        return 1

    print(f"Test case  : {TEST_CASE.relative_to(REPO_ROOT)}")
    print(f"Output dir : {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print(f"num_reads={NUM_READS}, annealing_time={ANNEALING_TIME} us\n")

    registry = [
        {"name": "SQA_HW", "class": SQAHardwareSolver, "type": "qpu"},
    ]

    output_path = run_experiment(
        test_case_paths=[TEST_CASE],
        solver_registry=registry,
        output_dir=OUTPUT_DIR,
        file_prefix="MinimalSmoke",
        num_reads=NUM_READS,
        annealing_time=ANNEALING_TIME,
        note="minimal_hw_test.py -- smoke test of QPU logging path",
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Verify the logging path actually wrote the expanded QPU metadata.
    # We don't assert on *values* (chip_id varies, timing varies); we just
    # check that every field declared in _QPU_RESULT_FIELDS is present and
    # that the core identity / timing fields are not None.
    # -----------------------------------------------------------------------
    payload = json.loads(Path(output_path).read_text())
    assert len(payload["results"]) == 1, "Expected exactly one result entry."
    (case_key, entry), = payload["results"].items()
    result = entry["solvers"]["SQA_HW"]

    if result.get("error"):
        print(f"\nQPU call returned an error: {result['error']}", file=sys.stderr)
        print(f"(Result file still written to: {output_path})", file=sys.stderr)
        return 2

    missing = [f for f in _QPU_RESULT_FIELDS if f not in result]
    if missing:
        print(f"\nFAIL: result entry is missing expected fields: {missing}", file=sys.stderr)
        return 3

    # Identity fields a real QPU run must have populated.
    must_be_populated = [
        "chip_id",
        "problem_id",
        "physical_qubits",
        "logical_variables",
        "num_reads",
        "best_energy",
        "qpu_access_time_us",
        "qpu_anneal_time_per_sample_us",
        "qpu_timing",
        "embedding",
    ]
    nulls = [f for f in must_be_populated if result.get(f) in (None, {}, [])]
    if nulls:
        print(
            f"\nFAIL: real-QPU run left these fields null/empty: {nulls}",
            file=sys.stderr,
        )
        return 4

    # -----------------------------------------------------------------------
    # Print a short, human-readable summary of what landed in the file.
    # -----------------------------------------------------------------------
    print("\n--- Logged QPU metadata (from result JSON) ---")
    print(f"Result file       : {output_path}")
    print(f"Case key          : {case_key}")
    print(f"chip_id           : {result['chip_id']}")
    print(f"problem_id        : {result['problem_id']}")
    print(f"problem_label     : {result.get('problem_label')}")
    print(f"BQM variables     : {result['bqm_variables']}")
    print(f"BQM interactions  : {result['bqm_interactions']}")
    print(f"Logical vars      : {result['logical_variables']}")
    print(f"Physical qubits   : {result['physical_qubits']}")
    print(f"Max chain length  : {result['max_chain_length']}")
    print(f"Mean chain length : {result['mean_chain_length']}")
    print(f"Chain break frac. : {result['chain_break_fraction']}")
    print(f"Num reads         : {result['num_reads']}")
    print(f"Best energy       : {result['best_energy']}")
    print(f"Best occurrences  : {result['best_num_occurrences']}")
    print(f"Wall time (ms)    : {result['wall_time_ms']}")
    print(f"QPU access (us)   : {result['qpu_access_time_us']}")
    print(f"QPU anneal/samp   : {result['qpu_anneal_time_per_sample_us']}")
    print(f"Cost              : {result['cost']}")
    print(f"Valid             : {result['valid']}")
    print("Full qpu_timing   :")
    for k, v in result["qpu_timing"].items():
        print(f"  {k:32s} = {v}")

    print("\nOK -- QPU logging path verified end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
