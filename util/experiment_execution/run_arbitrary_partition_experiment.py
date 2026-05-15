"""
Run the arbitrary-partition benchmark experiment.

Discovers test cases from test_bank/arbitrary_partition/, registers the
ILP baseline plus the base SQA solver (which supports variable partition
sizes), and writes results to result_bank/.

The domain-wall and slack-free solvers (S2, S3) are excluded because
they require unit partition sizes.  Only S1 is available for both
simulated and hardware modes.

Usage:
    python -m util.experiment_execution.run_arbitrary_partition_experiment

Options can be adjusted in the __main__ block or by importing and
calling run_arbitrary_experiment() directly.
"""

from pathlib import Path

from solvers.ILP import ILPSolver
from solvers.simulated_solvers.SQA import SQASolver
from util.experiment_execution.run_experiment import (
    discover_test_cases,
    run_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_BANK    = PROJECT_ROOT / "test_bank" / "arbitrary_partition"
RESULT_DIR_SIM = PROJECT_ROOT / "result_bank" / "simulated_solver_results"
RESULT_DIR_HW  = PROJECT_ROOT / "result_bank" / "quantum_hardware_results"

SOLVER_REGISTRY_SIM = [
    {"name": "ILP", "class": ILPSolver, "type": "ilp"},
    {"name": "SQA", "class": SQASolver,  "type": "sqa"},
]


def _get_hw_registry():
    """Import and return the hardware solver registry.

    Only S1 supports arbitrary partition sizes.
    Deferred to avoid ImportError when dwave-system is not installed.
    """
    from solvers.quantum_hardware_solvers.SQA_HW import SQAHardwareSolver

    return [
        {"name": "SQA_HW", "class": SQAHardwareSolver, "type": "qpu"},
    ]


def run_arbitrary_experiment(
    tier=None,
    node_counts=None,
    partition_counts=None,
    max_cases=None,
    num_reads=1000,
    num_sweeps=1000,
    beta_range=None,
    hardware=False,
    annealing_time=20,
    chain_strength=None,
):
    """
    Run the arbitrary-partition experiment.

    Args:
        tier:             "tier1", "tier2", or None (both tiers).
        node_counts:      optional filter, e.g. [2, 3, 5].
        partition_counts: optional filter, e.g. [3, 8, 18].
        max_cases:        cap total number of test cases (useful for quick checks).
        num_reads:        num_reads for SQA and QPU solvers.
        num_sweeps:       SQA num_sweeps (simulated solvers only).
        beta_range:       SQA beta_range (simulated solvers only).
        hardware:         if True, run the S1 QPU hardware solver instead
                          of the simulated S1.  ILP is always included.
        annealing_time:   QPU anneal duration in microseconds (hardware only).
        chain_strength:   QPU chain strength (hardware only, None = default).

    Returns:
        Path to the results JSON file.
    """
    paths = discover_test_cases(
        TEST_BANK,
        tier=tier,
        node_counts=node_counts,
        partition_counts=partition_counts,
        max_cases=max_cases,
    )

    if not paths:
        print("No test cases found. Run populate_test_bank.py first.")
        return None

    if hardware:
        registry = [SOLVER_REGISTRY_SIM[0]] + _get_hw_registry()  # ILP + HW
        result_dir = RESULT_DIR_HW
        prefix = "ArbitraryExperiment_HW"
        note = "Arbitrary-partition benchmark (D-Wave QPU): variable partition sizes."
    else:
        registry = SOLVER_REGISTRY_SIM
        result_dir = RESULT_DIR_SIM
        prefix = "ArbitraryExperiment"
        note = "Arbitrary-partition benchmark: variable partition sizes."

    print(f"Found {len(paths)} arbitrary-partition test cases.")

    return run_experiment(
        test_case_paths=paths,
        solver_registry=registry,
        output_dir=result_dir,
        file_prefix=prefix,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=beta_range,
        annealing_time=annealing_time,
        chain_strength=chain_strength,
        note=note,
    )


if __name__ == "__main__":
    run_arbitrary_experiment(
        tier="tier1",
        num_reads=1000,
        num_sweeps=1000,
    )
