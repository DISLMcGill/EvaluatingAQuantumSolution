# QuantumDataAllocation

A benchmark suite comparing Quantum Annealing formulations of the
distributed-data-allocation optimisation problem against an ILP baseline.
Two QUBO encodings are exercised by default — a slack-variable encoding
(Trummer 2025) and a calibrated unbalanced-penalty encoding
(Montañez-Barrera et al. 2022) — and the result of each run is checked
against the exact optimum found by CBC.

The current code reflects the executed remediation of an earlier,
audited version of the project.

## The problem

Given a distributed storage system with `N` nodes (each with a fixed
capacity) and `P` data partitions (each with a size and a communication
cost), find an assignment of partitions to nodes that:

1. **k-safety**: stores each partition on exactly `k` nodes (replication
   for fault tolerance);
2. **capacity**: respects each node's storage capacity;
3. **minimum cost**: minimises `Σ r_{p,n} · c_p · (1 − A_{p,n})` —
   the total cost of remote data fetches.

The ILP solver finds the provably optimal solution.  The SQA solvers
encode the same problem as a QUBO and approximate the optimum via
simulated quantum annealing (Path Integral Monte Carlo on CPU) or — when
configured — real D-Wave hardware.

## QUBO formulations (current)

The repository implements three QUBO encodings; two are in the default
benchmark registry, one is opt-in.

| Label | File | Storage encoding | k-safety encoding | In default registry? |
|-------|------|------------------|--------------------|-----------------------|
| **S1** | `solvers/simulated_solvers/SQA.py` | Binary slack variables (Paper 1, faithful) | `(Σ A − k)²` | yes |
| **S2** | `solvers/simulated_solvers/SQA_SF.py` | Unbalanced penalty with **calibrated** `(λ₁, λ₂)` (Paper 2, faithful) | `(Σ A − k)²` | yes |
| **S3** | `solvers/simulated_solvers/SQA_DW.py` | Same as S2 | Chancellor domain-wall chain + linking | **no** (opt-in via `SOLVER_REGISTRY_SIM_WITH_S3`) |

### Why S1 and S2 are equivalent in expressiveness

After the Phase-1 fix, S1 supports **arbitrary** integer capacities — its
slack-chunk decomposition was generalised from the Mersenne-only binary
expansion to `[1, 2, 4, …, 2^J, residual]` so that the chunks sum to
exactly `C_n`.  After the Phase-2 fix, S2 supports arbitrary **partition
sizes** — the unbalanced penalty's coefficients now include `size_p`
explicitly.  Both solvers exercise the same problem on the same inputs;
they only differ in how the storage inequality is encoded.

### Why S3 is opt-in, not default

The original S3 was advertised as reducing k-safety couplings from
`O(N²)` to `O(N)`.  In practice the data-allocation problem requires
free `A_{p,n}` subset-selection variables (nodes are not
interchangeable), so the domain-wall chain has to be linked back to `A`
via `(Σ A − Σ W)² = 0`, which reintroduces `O(N²)` couplings.  The
fixed S3 in this repo is correct (oracle test passes on loose cases,
``pytest.skip`` on documented tight-case limits) and uses the same
calibrated unbalanced storage as S2 — but it strictly **does not** beat
S2 on this problem class.  Keeping it in the repo preserves the
falsifiable negative result; demoting it from the default registry
keeps headline numbers honest.  To run it deliberately, use
`SOLVER_REGISTRY_SIM_WITH_S3` (see `util/experiment_execution/README.md`).

## Project structure

```
QuantumClean/
├── README.md                            ← this file
├── pyproject.toml                       ← project metadata + pytest config
├── requirements.txt                     ← pinned dependencies
│
├── tier1_unit_sqa_hw.py                 ← entry point: unit sweep on QPU
├── tier1_unit_sqa_hybrid.py             ← entry point: unit sweep on Leap hybrid
├── tier1_arbitrary_sqa_hw.py            ← entry point: arbitrary sweep on QPU
├── tier1_arbitrary_sqa_hybrid.py        ← entry point: arbitrary sweep on Leap hybrid
├── single_case_hw_bench.py              ← entry point: single-case QPU benchmark
├── minimal_hw_test.py                   ← entry point: minimal QPU smoke test
├── hybrid_time_calibration.py           ← entry point: hybrid time_limit calibration
├── dwave_usage.py                       ← utility: report Leap QPU usage
│
├── solvers/                             ← all solver implementations
│   ├── README.md
│   ├── ILP.py                           ← exact classical baseline (PuLP/CBC)
│   ├── simulated_solvers/
│   │   ├── README.md
│   │   ├── SQA.py                       ← S1
│   │   ├── SQA_SF.py                    ← S2 (with calibrate_lambdas)
│   │   └── SQA_DW.py                    ← S3 (opt-in)
│   └── quantum_hardware_solvers/
│       ├── README.md
│       ├── __init__.py
│       ├── SQA_HW.py                    ← S1 on QPU
│       ├── SQA_SF_HW.py                 ← S2 on QPU
│       └── SQA_DW_HW.py                 ← S3 on QPU
│
├── util/
│   ├── solver_base.py
│   ├── brute_force.py                   ← oracle for tests; exhaustive enumeration
│   ├── calculate_solution_cost.py
│   ├── test_generation/
│   │   ├── README.md
│   │   ├── populate_test_bank.py
│   │   ├── generate_test_case.py        ← arbitrary partitions; tightness-stratified
│   │   ├── generate_unit_test_case.py   ← unit partitions; tightness-stratified
│   │   ├── generate_paired_test_cases.py
│   │   ├── generate_test_banks.py
│   │   ├── json_to_dict.py
│   │   ├── feasibility_probe.py         ← feasibility pre-check for stress cases
│   │   └── populate_hybrid_stress_bank.py ← builds test_bank/hybrid_stress/
│   └── experiment_execution/
│       ├── README.md
│       ├── run_experiment.py            ← Phase-5 harness
│       ├── run_unit_partition_experiment.py
│       ├── run_arbitrary_partition_experiment.py
│       └── hybrid_budget.py            ← time_limit budgeting for hybrid runs
│
├── tests/                               ← pytest suite; ExactSolver oracles
│   ├── README.md
│   ├── conftest.py                      ← shared problem fixtures
│   ├── test_oracle.py                   ← QUBO ground state vs brute-force optimum
│   ├── test_cost_and_validity.py        ← property tests for cost/validity
│   ├── test_generators.py               ← feasibility / round-trip
│   ├── test_harness.py                  ← Phase-5 result-field smoke test
│   └── test_json_roundtrip.py
│
├── test_bank/                           ← pre-generated problem instances
│   ├── unit_partition/{tier1,tier2}/n{N}_p{P}/t{30,70,100}/
│   ├── arbitrary_partition/{tier1,tier2}/n{N}_p{P}/t{30,70,90}/
│   └── hybrid_stress/{unit,arbitrary}_partition/n{N}_p{P}/t{…}/
│
├── result_bank/                         ← experiment outputs
│   ├── simulated_solver_results[_feasible]/  ← simulated-solver outputs (Unit/Arbitrary*.json)
│   ├── quantum_hardware_results[_feasible]/  ← QPU runner outputs
│   └── hybrid_results_feasible/              ← Leap hybrid runner outputs
│
└── result_analysis/                     ← Jupyter notebooks (not in CI)
    ├── unit_sweep_analysis.ipynb        ← schema-aware loader for UnitExperiment_*.json
    ├── arbitrary_sweep_analysis.ipynb   ← schema-aware loader for ArbitraryExperiment_*.json
    ├── <Experiment>_analysis.ipynb      ← per-run analyses (sim / hardware / hybrid-merged)
    ├── timing_analysis.ipynb            ← wall-time vs solver-time comparison
    ├── qpu_runtime_analysis.ipynb       ← QPU access-time breakdown
    └── plots/                           ← PNGs the notebooks write on `plt.savefig`
```

## Entry points (root scripts)

The runnable scripts live at the repo root and are the intended entry
points; everything under `solvers/`, `util/`, and `tests/` is library
code they call. Run them from the repo root (e.g. `python
tier1_unit_sqa_hw.py`). The four `tier1_*` sweeps form a grid of
**unit vs. arbitrary** partition bank × **hardware (QPU) vs. hybrid** solver.

| Script | What it runs |
|--------|--------------|
| `tier1_unit_sqa_hw.py` | Tier-1 **unit**-partition sweep (S1+S2) on real D-Wave **hardware**. `--full` = all 180 cases, `--no-pause` = unattended. |
| `tier1_unit_sqa_hybrid.py` | Same unit-partition sweep via the Leap **hybrid** BQM solver. |
| `tier1_arbitrary_sqa_hw.py` | Tier-1 **arbitrary**-partition sweep (S1+S2) on hardware. |
| `tier1_arbitrary_sqa_hybrid.py` | Same arbitrary-partition sweep via the hybrid solver. |
| `single_case_hw_bench.py` | Full benchmark on a single case (~2 QPU submissions) — sanity-check the pipeline before a full sweep. |
| `minimal_hw_test.py` | Minimal end-to-end QPU connectivity/smoke test on the smallest case. |
| `hybrid_time_calibration.py` | Finds the hybrid `time_limit` "knee" for the `hybrid_stress` bench. |
| `dwave_usage.py` | Reports D-Wave Leap QPU usage and timing across submitted jobs. |

## Getting started

### Install

```
pip install -r requirements.txt
# optional, only for D-Wave hardware:
pip install -e ".[hardware]"
```

### Run the tests

```
python -m pytest tests/ -v
```

The suite collects ~30 tests across `test_oracle.py`,
`test_cost_and_validity.py`, `test_generators.py`, `test_harness.py`,
and `test_json_roundtrip.py`.  Up to three of the S3 oracle
parametrizations (`test_s3_ground_state_is_near_optimum`) may
`pytest.skip` on tight-capacity fixtures where the redundant W+A
encoding has no feasible ground state under unbalanced penalization
alone -- this is the known structural limitation Paper 2
acknowledges, documented in `solvers/simulated_solvers/SQA_DW.py`.
Everything else should pass.

### Generate a test bank

```
python -m util.test_generation.populate_test_bank          # lean grid, ~630 cases
python -m util.test_generation.populate_test_bank --full   # full grid, ~5,340 cases
```

Test cases are stratified by **tightness** (`0.0` = loose capacity,
`1.0` = exact min capacity), and feasibility is verified by an ILP
probe inside the generator.  Capacities are *no longer* Mersenne-rounded
— the S1 chunk encoding now handles arbitrary integers.

The lean grid covers tier 1 (`n ∈ {3,5,9}`, `p ∈ {4,12,26,50}`) and
tier 2 (`n ∈ {5,9,15}`, `p ∈ {18,50,100}`) with 5 paired cases at each
of 3 tightness levels for both unit and arbitrary partitions
(`3×4×3×5×2 + 3×3×3×5×2 = 630` cases before feasibility rejection).
It is the default because paired comparisons between S1 and S2 reach
significance with far fewer instances than the original sweep used.
Use `--full` only when you need tight per-cell error bars or
fine-grained scaling curves; expect hours-to-days of downstream run
time.

### Run a benchmark

```bash
# Default registry: ILP + S1 + S2
python -m util.experiment_execution.run_unit_partition_experiment
python -m util.experiment_execution.run_arbitrary_partition_experiment
```

To opt in to S3 on the simulator:

```python
from util.experiment_execution.run_unit_partition_experiment import (
    run_unit_experiment, SOLVER_REGISTRY_SIM_WITH_S3,
)
run_unit_experiment(extra_registry=SOLVER_REGISTRY_SIM_WITH_S3)
```

To run on D-Wave hardware (requires `dwave-system` + LEAP token):

```python
from util.experiment_execution.run_unit_partition_experiment import run_unit_experiment

# Default hardware registry: ILP + S1 (HW) + S2 (HW)
run_unit_experiment(tier="tier1", hardware=True, num_reads=100, annealing_time=20)

# Add S3 to a hardware run via the dedicated flag (ignored on simulator runs)
run_unit_experiment(
    tier="tier1", hardware=True, include_s3=True,
    num_reads=100, annealing_time=20,
)
```

Results are written incrementally to `result_bank/` — interrupted runs
preserve all completed test cases.  Simulator runs land in
`result_bank/simulated_solver_results/` as
`UnitExperiment_N.json` / `ArbitraryExperiment_N.json` (`N` auto-
increments); hardware runs land in `result_bank/quantum_hardware_results/`
with the `_HW` prefix.

### Analyse results

The notebooks in `result_analysis/` consume the JSON files above
directly.  Each notebook auto-detects which solvers are present in the
file and picks up the **most recent** `UnitExperiment_*.json` or
`ArbitraryExperiment_*.json` by default — replace `RESULTS_FILE` in the
first cell to pin a specific run.

```
result_analysis/unit_sweep_analysis.ipynb        # generic loader, latest Unit*.json
result_analysis/arbitrary_sweep_analysis.ipynb   # generic loader, latest Arbitrary*.json
result_analysis/<Experiment>_analysis.ipynb      # per-run analyses (sim / hardware / hybrid-merged)
result_analysis/timing_analysis.ipynb            # wall-time vs solver-time comparison
result_analysis/qpu_runtime_analysis.ipynb       # QPU access-time breakdown
```

Each notebook produces an aggregate summary, validity / gap heatmaps
over the `(n_nodes, n_partitions)` grid, BQM size scaling curves, an
S1-vs-S2 head-to-head scatter, the `(lambda_1, lambda_2)` distribution
from the S2 calibrator, and a constraint-violation breakdown.  Saved
plots are written to `result_analysis/plots/`.  The "Empirical
Findings" markdown section at the bottom of each notebook records the
numerical conclusions drawn from the latest result file checked in.

## Result schema (Phase 5)

Every per-solver result entry contains:

| Field | All | SQA only | QPU only |
|-------|-----|----------|----------|
| `cost` | ✓ | | |
| `valid` | ✓ | | |
| `k_safety_violations` | ✓ | | |
| `capacity_overruns` | ✓ | | |
| `wall_time_ms` | ✓ | | |
| `optimality_gap_absolute` | ✓ | | |
| `optimality_gap_relative` | ✓ | | |
| `error` | ✓ | | |
| `bqm_variables` | | ✓ | ✓ |
| `bqm_interactions` | | ✓ | ✓ |
| `lambda_1`, `lambda_2` | | ✓ (S2/S3) | ✓ (S2/S3) |
| `physical_qubits` | | | ✓ |
| `chain_break_fraction` | | | ✓ |
| `qpu_anneal_time_per_sample_us` | | | ✓ |

`optimality_gap_relative` is `0.0` (not `null`) when both costs are `0`.
`null` is reserved for cases where the relative gap is genuinely
undefined (e.g. `ilp_cost == 0` but solver cost is non-zero).

Each top-level result entry also carries `n_nodes`, `n_partitions`,
`k_safety`, `source_file`, and the test-case metadata surfaced from
the input JSON under a `tc_` prefix (notably `tc_tightness`).  The
analysis notebooks read `tc_tightness` to stratify metrics by
storage-constraint tightness without re-opening the source test
cases.

## References

- Trummer, I. (2025). "Leveraging Quantum Computing for Optimal Data
  Allocation in Distributed Systems." Q-Data '25.
- Montañez-Barrera, J. A., Willsch, D., Maldonado-Romo, A., Michielsen,
  K. (2022). "Unbalanced penalization: A new approach to encode
  inequality constraints for quantum optimization algorithms."
  arXiv:2211.13914.
- Chancellor, N. (2019). "Domain wall encoding of discrete variables
  for quantum annealing and QAOA." arXiv:1903.05068.
