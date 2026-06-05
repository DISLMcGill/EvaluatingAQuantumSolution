"""
Run the arbitrary-partition benchmark experiment.

Discovers test cases from test_bank/arbitrary_partition/, registers the
ILP baseline plus the SQA solvers that handle variable partition sizes,
and writes results to result_bank/.

After the Phase-2 refactor, both S1 (binary slack) and S2 (calibrated
unbalanced penalty) support arbitrary partition sizes -- size_p shows up
in the storage encoding as the coefficient of A_{p,n} rather than as a
unit count.  Both are in the default registry.

S3 (SQA_DW) is deliberately excluded -- see
``run_unit_partition_experiment.py`` for the rationale -- and is
opt-in via ``SOLVER_REGISTRY_SIM_WITH_S3``.
"""

from pathlib import Path

from solvers.ILP import ILPSolver
from solvers.simulated_solvers.SQA import SQASolver
from solvers.simulated_solvers.SQA_SF import SQASlackFreeSolver
from solvers.simulated_solvers.SQA_DW import SQADomainWallSolver
from util.experiment_execution.run_experiment import (
    discover_test_cases,
    run_experiment,
)
from util.experiment_execution.run_unit_partition_experiment import (
    _with_selection_policy,
    result_dir_for,
)
from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    VALID_POLICIES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_BANK    = PROJECT_ROOT / "test_bank" / "arbitrary_partition"

SOLVER_REGISTRY_SIM = [
    {"name": "ILP",    "class": ILPSolver,             "type": "ilp"},
    {"name": "SQA",    "class": SQASolver,             "type": "sqa"},
    {"name": "SQA_SF", "class": SQASlackFreeSolver,    "type": "sqa"},
]

# Opt-in registry that includes S3.  See SQA_DW.py for the documented
# rationale; this exists so the negative finding remains reproducible.
SOLVER_REGISTRY_SIM_WITH_S3 = SOLVER_REGISTRY_SIM + [
    {"name": "SQA_DW", "class": SQADomainWallSolver, "type": "sqa"},
]


def _get_hw_registry(include_s3=False):
    """Hardware solver registry.

    Deferred to avoid ImportError when dwave-system is not installed.
    """
    from solvers.quantum_hardware_solvers.SQA_HW import SQAHardwareSolver
    from solvers.quantum_hardware_solvers.SQA_SF_HW import SQASFHardwareSolver

    registry = [
        {"name": "SQA_HW",    "class": SQAHardwareSolver,    "type": "qpu"},
        {"name": "SQA_SF_HW", "class": SQASFHardwareSolver,  "type": "qpu"},
    ]
    if include_s3:
        from solvers.quantum_hardware_solvers.SQA_DW_HW import SQADWHardwareSolver
        registry.append(
            {"name": "SQA_DW_HW", "class": SQADWHardwareSolver, "type": "qpu"},
        )
    return registry


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
    extra_registry=None,
    include_s3=False,
    selection_policy=POLICY_BEST_FEASIBLE,
    resume=False,
):
    """Run the arbitrary-partition experiment.

    See ``run_unit_partition_experiment.run_unit_experiment`` for argument
    semantics.  The two runners share a structure; the only difference is
    which test-bank subdirectory they read from.

    ``selection_policy`` works the same way here: 'best_feasible' (default)
    routes output to the parallel ``*_feasible/`` result-bank tree;
    'lowest_energy' restores legacy behaviour and the original tree.
    """
    if selection_policy not in VALID_POLICIES:
        raise ValueError(
            f"selection_policy must be one of {VALID_POLICIES}, "
            f"got {selection_policy!r}"
        )

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
        registry = [SOLVER_REGISTRY_SIM[0]] + _get_hw_registry(include_s3=include_s3)
        prefix = "ArbitraryExperiment_HW"
        note = "Arbitrary-partition benchmark (D-Wave QPU): variable partition sizes."
    else:
        registry = extra_registry if extra_registry is not None else SOLVER_REGISTRY_SIM
        prefix = "ArbitraryExperiment"
        note = "Arbitrary-partition benchmark: variable partition sizes."

    result_dir = result_dir_for(hardware=hardware, selection_policy=selection_policy)
    note = f"{note}  selection_policy={selection_policy}."
    registry = _with_selection_policy(registry, selection_policy)

    print(f"Found {len(paths)} arbitrary-partition test cases.")
    print(f"selection_policy={selection_policy} -> {result_dir.name}/")

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
        resume=resume,
    )


if __name__ == "__main__":
    # See run_unit_partition_experiment.py for the rationale behind
    # these sampler defaults.  Bump for tier 2 or the full grid.
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the simulated arbitrary-partition experiment.",
    )
    parser.add_argument("--tier", default="tier1",
                        help="tier1, tier2, or 'all' for both (default tier1).")
    parser.add_argument("--num-reads", type=int, default=200)
    parser.add_argument("--num-sweeps", type=int, default=500)
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue the most recent ArbitraryExperiment_<N>.json in "
             "the result bank instead of starting a fresh file: cases "
             "already recorded there are skipped.  Use this to pick a "
             "sweep back up after an interruption (sleep, Ctrl-C, crash).",
    )
    args = parser.parse_args()

    run_arbitrary_experiment(
        tier=None if args.tier == "all" else args.tier,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        resume=args.resume,
    )
