# QuantumDataAllocation

A benchmark suite for comparing Quantum Annealing formulations of the data allocation optimisation problem. The project encodes the same NP-hard placement problem — assigning replicated data partitions to capacity-constrained storage nodes to minimise remote communication cost — as three distinct QUBO (Quadratic Unconstrained Binary Optimisation) formulations, solves them with both simulated and real D-Wave quantum annealers, and measures solution quality against an exact ILP (Integer Linear Programming) baseline.


## The Problem

Given a distributed storage system with `N` nodes (each with a fixed capacity) and `P` data partitions (each with a size and a communication cost), find an assignment of partitions to nodes that:

1. **k-Safety:** stores each partition on exactly `k` nodes (replication for fault tolerance).
2. **Capacity:** does not exceed any node's storage capacity.
3. **Minimum cost:** minimises `sum(r_pn * c_p * (1 − A_pn))` — the total cost of remote data fetches across all partition-node pairs where the partition is not stored locally.

The ILP solver finds the provably optimal solution. The SQA solvers encode the constraints as penalty terms in a QUBO and approximate the optimum via quantum annealing, either simulated (Path Integral Monte Carlo on CPU) or on D-Wave hardware.


## QUBO Formulations

Three formulations are implemented, each encoding the same constraints differently:

**S1 — Standard (SQA):** Uses binary slack variables to convert the storage capacity inequality into an equality. Requires node capacities to be Mersenne numbers (2^k − 1). Supports arbitrary partition sizes. Produces the largest BQMs due to slack variables.

**S2 — Slack-Free (SQA_SF):** Eliminates all slack variables using an unbalanced penalty function (Montañez-Barrera et al., 2022) that penalises capacity overflow directly. No Mersenne capacity restriction, but requires all partition sizes to equal 1. Produces the smallest BQMs.

**S3 — Domain-Wall (SQA_DW):** Replaces the O(N²) quadratic k-safety penalty with an O(N) nearest-neighbour domain-wall chain encoding (Chancellor, 2019). Combines this with the slack-free storage encoding from S2. More variables than S2 but sparser coupling — expected to embed more efficiently on hardware. Requires unit partition sizes.


## Project Structure

```
QuantumClean/
├── README.md
├── backfill_sqa_sf.py                  # temporary backfill script (see below)
│
├── solvers/                            # all solver implementations
│   ├── README.md                       # overview of solvers directory
│   ├── ILP.py                          # exact classical baseline (PuLP/CBC)
│   ├── simulated_solvers/              # CPU-based SQA via PathIntegralAnnealingSampler
│   │   ├── README.md                   # detailed formulation documentation
│   │   ├── SQA.py                      # S1
│   │   ├── SQA_SF.py                   # S2
│   │   └── SQA_DW.py                   # S3
│   └── quantum_hardware_solvers/       # D-Wave QPU versions
│       ├── README.md                   # hardware parameters and QPU metadata docs
│       ├── __init__.py
│       ├── SQA_HW.py                   # S1 on QPU
│       ├── SQA_SF_HW.py               # S2 on QPU
│       └── SQA_DW_HW.py               # S3 on QPU
│
├── util/
│   ├── solver_base.py                  # abstract base class for all solvers
│   ├── calculate_solution_cost.py      # cost calculation and validation utilities
│   ├── test_generation/                # deterministic test case generation
│   │   ├── README.md
│   │   ├── populate_test_bank.py       # top-level orchestrator
│   │   ├── generate_test_case.py       # arbitrary-partition generator
│   │   ├── generate_unit_test_case.py  # unit-partition generator
│   │   ├── generate_paired_test_cases.py  # matched-pair generator
│   │   ├── generate_test_banks.py      # batch generation helper
│   │   └── json_to_dict.py            # test case JSON loader
│   └── experiment_execution/           # experiment harness
│       ├── README.md
│       ├── run_experiment.py           # core harness (ILP / SQA / QPU dispatch)
│       ├── run_unit_partition_experiment.py
│       └── run_arbitrary_partition_experiment.py
│
├── test_bank/                          # pre-generated problem instances
│   ├── unit_partition/
│   │   ├── tier1/                      # 400 cases: 5 node counts × 8 partition counts × 10
│   │   └── tier2/                      # 490 cases: 7 × 7 × 10
│   └── arbitrary_partition/
│       ├── tier1/                      # same grid as unit tier 1
│       └── tier2/                      # same grid as unit tier 2
│
├── result_bank/                        # experiment outputs
│   ├── simulated_solver_results/       # CPU-based SQA results
│   │   ├── UnitExperiment_1.json
│   │   └── ArbitraryExperiment_1.json
│   └── quantum_hardware_results/       # D-Wave QPU results
│
└── result_analysis/                    # Jupyter notebooks for visualisation
    ├── unit_sweep_analysis.ipynb       # unit-partition analysis (S1, S2, S3)
    └── arbitrary_sweep_analysis.ipynb  # arbitrary-partition analysis (S1 only)
```


## Getting Started

### Prerequisites

The simulated experiments require:

```
pip install dimod dwave-samplers pulp numpy pandas matplotlib
```

For D-Wave hardware experiments, additionally:

```
pip install dwave-system
dwave setup   # configure LEAP API token
```

### Generating Test Cases

```bash
python -m util.test_generation.populate_test_bank
```

This creates 1,780 deterministic, seeded test cases across both partition types and both tiers. Running it again produces byte-identical output.

### Running Experiments

Simulated (CPU):

```bash
# Unit-partition benchmark (ILP + S1 + S2 + S3)
python -m util.experiment_execution.run_unit_partition_experiment

# Arbitrary-partition benchmark (ILP + S1 only — S2/S3 require unit partitions)
python -m util.experiment_execution.run_arbitrary_partition_experiment
```

D-Wave hardware:

```python
from util.experiment_execution.run_unit_partition_experiment import run_unit_experiment

run_unit_experiment(
    tier="tier1",
    hardware=True,
    num_reads=100,
    annealing_time=20,
)
```

Results are written incrementally to `result_bank/` — interrupted runs preserve all completed test cases.

### Analysis

Open the Jupyter notebooks in `result_analysis/` to visualise results. The notebooks are configurable: edit the `SOLVERS` list at the top to include or exclude any subset of solvers, and all plots, heatmaps, and statistical comparisons adapt automatically.


## Test Bank

Test cases are organised into two tiers of increasing problem size:

| Tier | Nodes | Partitions | Cases per combo |
|------|-------|------------|-----------------|
| Tier 1 | 2, 3, 5, 7, 9 | 3, 4, 8, 12, 18, 26, 36, 50 | 10 |
| Tier 2 | 2, 3, 5, 7, 9, 12, 15 | 3, 8, 18, 36, 50, 75, 100 | 10 |

Each test case specifies node capacities (Mersenne numbers), partition sizes, a replication factor (k=2), per-(partition, node) request frequencies, and per-partition communication costs. Generation is deterministic via seeds derived from the problem parameters.


## Experiment Results

Each experiment run produces a JSON file recording per-solver statistics for every test case:

**All solvers:** cost, validity, solve time (ms), error status.

**SQA/QPU solvers:** BQM variable count, BQM interaction count, optimality gap vs ILP.

**QPU solvers additionally:** physical qubit count (post-embedding), chain break fraction, QPU access time (µs).


## Key Findings (Simulated)

The unit-partition analysis notebook documents several findings from the simulated experiments:

- S2 (slack-free) achieves the best overall solution quality, with the lowest optimality gaps and highest validity rates. Its smaller BQM (no slack variables) gives the annealer a more tractable energy landscape.
- S3 (domain-wall) shows an oscillating validity pattern at certain node counts (7, 9) caused by Mersenne capacity rounding creating alternating tight/loose fits as partition count grows. This is a confounding factor in the experimental design, not a formulation defect.
- S1 (standard) validity degrades with problem size due to the exponential growth of slack variables, which dilute the annealer's search effort.
- All three formulations show BQM variable/interaction counts that grow linearly with problem size, but at different rates — S2 is consistently the most compact.


## Known Confounding Factor

All node capacities are Mersenne numbers, a requirement of S1's binary slack encoding. This was applied uniformly across all formulations for fair comparison, but it creates a confounding factor: at certain (node, partition) combinations the Mersenne capacity exactly equals the total partition count, causing S2's unbalanced penalty to become identically zero and removing gradient signal for placement optimisation. This is documented in the analysis notebook conclusions and should be considered when interpreting results.


## References

- Montañez-Barrera, J. A., et al. (2022). "Unbalanced penalization: A new approach to encode inequality constraints for quantum optimization algorithms." arXiv:2211.13914.
- Chancellor, N. (2019). "Domain wall encoding of discrete variables for quantum annealing and QAOA." arXiv:1903.05068.
