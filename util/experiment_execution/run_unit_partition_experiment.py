"""
Run the unit-partition benchmark experiment.

Discovers test cases from test_bank/unit_partition/, registers the
ILP baseline plus the production SQA solvers, and writes results to
result_bank/.

Production solvers (in the default registry):

    * **ILP**     -- CBC-backed mixed-integer reference.
    * **SQA**     -- S1: faithful Paper-1 QUBO with binary slack vars.
    * **SQA_SF**  -- S2: Paper-2 unbalanced penalty with calibrated lambdas.

S3 (SQA_DW, domain-wall + unbalanced) is **intentionally excluded** from
the default registry.  See ``solvers/simulated_solvers/SQA_DW.py`` for
the full explanation; the short version is that the domain-wall chain
must be linked back to the assignment variables ``A_{p,n}`` via an
``O(|N|^2)`` penalty, which negates the coupling-count advantage the
encoding offers in other problem classes.  The implementation is kept
in the repo as a documented negative result and to back the
``test_s3_*`` regression tests, but running it as part of the headline
benchmark would create misleading apples-to-oranges comparisons.

To run S3 deliberately (e.g., to confirm the negative finding):

    from util.experiment_execution.run_unit_partition_experiment import (
        run_unit_experiment, SOLVER_REGISTRY_SIM_WITH_S3,
    )
    run_unit_experiment(extra_registry=SOLVER_REGISTRY_SIM_WITH_S3)
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
from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    POLICY_LOWEST_ENERGY,
    VALID_POLICIES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_BANK    = PROJECT_ROOT / "test_bank" / "unit_partition"

# Result-bank directories are split by selection_policy so the historical
# lowest-energy result bank stays intact as the original baseline and
# new best_feasible runs land in a parallel tree.  See README.
RESULT_DIR_SIM            = PROJECT_ROOT / "result_bank" / "simulated_solver_results"
RESULT_DIR_HW             = PROJECT_ROOT / "result_bank" / "quantum_hardware_results"
RESULT_DIR_SIM_FEASIBLE   = PROJECT_ROOT / "result_bank" / "simulated_solver_results_feasible"
RESULT_DIR_HW_FEASIBLE    = PROJECT_ROOT / "result_bank" / "quantum_hardware_results_feasible"


def result_dir_for(hardware: bool, selection_policy: str) -> Path:
    """
    Map (hardware, selection_policy) -> the result_bank subdirectory
    that owns runs with that configuration.

    * lowest_energy  -> original directories (unchanged historical bank)
    * best_feasible  -> parallel ``*_feasible/`` directories

    Centralising the mapping here keeps every entrypoint (the runner,
    single_case_hw_bench.py, tier1_subset_sqa_hw.py) in agreement on
    where results land.
    """
    if selection_policy not in VALID_POLICIES:
        raise ValueError(
            f"selection_policy must be one of {VALID_POLICIES}, "
            f"got {selection_policy!r}"
        )
    if selection_policy == POLICY_LOWEST_ENERGY:
        return RESULT_DIR_HW if hardware else RESULT_DIR_SIM
    return RESULT_DIR_HW_FEASIBLE if hardware else RESULT_DIR_SIM_FEASIBLE

# Default registry -- ILP + S1 + S2 only.  S3 is excluded; see module
# docstring.
SOLVER_REGISTRY_SIM = [
    {"name": "ILP",    "class": ILPSolver,             "type": "ilp"},
    {"name": "SQA",    "class": SQASolver,             "type": "sqa"},
    {"name": "SQA_SF", "class": SQASlackFreeSolver,    "type": "sqa"},
]

# Opt-in registry for users who explicitly want to compare S3.  Kept
# separate from SOLVER_REGISTRY_SIM so accidental imports don't include
# it in headline benchmark numbers.
SOLVER_REGISTRY_SIM_WITH_S3 = SOLVER_REGISTRY_SIM + [
    {"name": "SQA_DW", "class": SQADomainWallSolver,  "type": "sqa"},
]


def _get_hw_registry(include_s3=False):
    """
    Import and return the hardware solver registry.

    Deferred to avoid ImportError when dwave-system is not installed.

    S3-on-hardware (SQA_DW_HW) is opt-in via ``include_s3``; on the
    sparse D-Wave topologies the linking constraint blows up chain
    lengths faster than for S1/S2 and the existing experiment has
    never been run on real hardware, so it is excluded by default.
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


def run_unit_experiment(
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
    """
    Run the unit-partition experiment.

    Args:
        tier:             "tier1", "tier2", or None (both tiers).
        node_counts:      optional filter, e.g. [2, 3, 5].
        partition_counts: optional filter, e.g. [3, 8, 18].
        max_cases:        cap total number of test cases.
        num_reads:        num_reads for SQA and QPU solvers.
        num_sweeps:       SQA num_sweeps (simulated solvers only).
        beta_range:       SQA beta_range (simulated solvers only).
        hardware:         if True, run QPU hardware solvers instead of
                          simulated solvers.  ILP is always included.
        annealing_time:   QPU anneal duration in microseconds (hardware only).
        chain_strength:   QPU chain strength (hardware only, None = default).
        extra_registry:   optional list of solver descriptors to *replace*
                          the default registry (e.g. SOLVER_REGISTRY_SIM_WITH_S3).
        include_s3:       opt in to S3 on hardware.  Ignored when ``hardware=False``
                          -- to add S3 to a simulated run, pass
                          ``extra_registry=SOLVER_REGISTRY_SIM_WITH_S3``.
        selection_policy: 'best_feasible' (default) -- return the
                          lowest-energy feasible sample, falling back
                          to the lowest-energy sample overall when no
                          read is feasible.  Routes output to the
                          ``*_feasible/`` result-bank directory.
                          'lowest_energy' -- legacy behaviour, routes
                          output to the original result-bank directory.

    Returns:
        Path to the results JSON file.
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
        prefix = "UnitExperiment_HW"
        note = (
            "Unit-partition benchmark (D-Wave QPU): test_bank cases drawn "
            "with size_p = 1 for every partition."
        )
    else:
        registry = extra_registry if extra_registry is not None else SOLVER_REGISTRY_SIM
        prefix = "UnitExperiment"
        note = (
            "Unit-partition benchmark: test_bank cases drawn with "
            "size_p = 1.  All solvers in the default registry also "
            "support arbitrary sizes -- see the arbitrary-partition runner."
        )
    result_dir = result_dir_for(hardware=hardware, selection_policy=selection_policy)
    note = f"{note}  selection_policy={selection_policy}."

    # Inject selection_policy into every solver's kwargs so the harness
    # constructs them with the requested policy.  ILP is excluded (it
    # has no selection step); we only attach the kwarg to entries whose
    # class accepts it.
    registry = _with_selection_policy(registry, selection_policy)

    print(f"Found {len(paths)} unit-partition test cases.")
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


def _with_selection_policy(registry, selection_policy):
    """
    Return a copy of ``registry`` where every non-ILP solver entry
    carries ``kwargs['selection_policy'] = selection_policy``.

    The harness reads ``solver_desc.get("kwargs", {})`` and forwards it
    to the solver constructor via ``**solver_kwargs``; this helper
    targets that exact field.  ILP doesn't sample, so the kwarg would
    be a constructor error; QPU and SQA solvers all accept it.
    Centralising the injection here keeps the entrypoints
    (run_unit_experiment, bench scripts) free of per-solver dispatch.
    """
    new_registry = []
    for entry in registry:
        entry = dict(entry)  # shallow copy so callers see no mutation
        if entry.get("type") in ("sqa", "qpu"):
            kwargs = dict(entry.get("kwargs") or {})
            kwargs.setdefault("selection_policy", selection_policy)
            entry["kwargs"] = kwargs
        new_registry.append(entry)
    return new_registry


if __name__ == "__main__":
    # Defaults tuned for the lean test bank.  Tier-1 instances are small
    # enough that num_reads=200, num_sweeps=500 reaches the QUBO ground
    # state reliably -- the old num_reads=1000, num_sweeps=1000 default
    # is 5-10x more work than needed at these sizes.  Bump them up for
    # tier 2 or for the full grid.
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the simulated unit-partition experiment.",
    )
    parser.add_argument("--tier", default="tier1",
                        help="tier1, tier2, or 'all' for both (default tier1).")
    parser.add_argument("--num-reads", type=int, default=200)
    parser.add_argument("--num-sweeps", type=int, default=500)
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue the most recent UnitExperiment_<N>.json in the "
             "result bank instead of starting a fresh file: cases already "
             "recorded there are skipped.  Use this to pick a sweep back "
             "up after an interruption (sleep, Ctrl-C, crash).",
    )
    args = parser.parse_args()

    run_unit_experiment(
        tier=None if args.tier == "all" else args.tier,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        resume=args.resume,
    )
