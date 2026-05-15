# Solvers

This directory contains all solvers for the data allocation optimisation problem. The problem asks: given a set of storage nodes with limited capacity and a set of data partitions that must each be replicated on exactly `k` nodes, find the allocation that minimises total remote communication cost.

Every solver implements the `SolverBase` interface (defined in `util/solver_base.py`), which requires a `solve()` method and a `format_answer()` method, and stores results in `self.result` and timing in `self.time_taken`.


## Directory Structure

```
solvers/
├── README.md                          ← this file
├── ILP.py                             ← classical baseline
├── simulated_solvers/
│   ├── README.md                      ← detailed formulation docs
│   ├── SQA.py                         ← S1: binary slack encoding
│   ├── SQA_SF.py                      ← S2: slack-free (unbalanced penalty)
│   └── SQA_DW.py                      ← S3: domain-wall k-safety + slack-free
└── quantum_hardware_solvers/
    ├── README.md                      ← hardware parameters and QPU metadata docs
    ├── __init__.py
    ├── SQA_HW.py                      ← S1 on D-Wave QPU
    ├── SQA_SF_HW.py                   ← S2 on D-Wave QPU
    └── SQA_DW_HW.py                   ← S3 on D-Wave QPU
```


## Solver Summary

| Solver | Type | Variables | Partition Sizes | Capacity Restriction |
|--------|------|-----------|-----------------|---------------------|
| **ILP** | Classical (PuLP/CBC) | Assignment only | Any | Any |
| **S1** | Simulated QA | Assignment + slack | Any | Mersenne (2^k − 1) |
| **S2** | Simulated QA | Assignment only | Unit (all = 1) | Any |
| **S3** | Simulated QA | Assignment + domain-wall | Unit (all = 1) | Any |
| **S1 HW** | D-Wave QPU | Assignment + slack | Any | Mersenne (2^k − 1) |
| **S2 HW** | D-Wave QPU | Assignment only | Unit (all = 1) | Any |
| **S3 HW** | D-Wave QPU | Assignment + domain-wall | Unit (all = 1) | Any |


## ILP Baseline (ILP.py)

The ILP solver uses PuLP with the CBC (Coin-or Branch and Cut) backend to find the provably optimal solution via classical integer linear programming. It serves as the ground truth for computing optimality gaps of the SQA solvers. Unlike the SQA solvers, it handles capacity constraints natively as linear inequalities and does not require any QUBO encoding. It has no tunable anneal parameters — just call `solver.solve()`.


## Simulated Solvers

Three QUBO formulations solved via Path Integral Monte Carlo (D-Wave's `PathIntegralAnnealingSampler`). The key research question is how different encodings of the same constraints affect solution quality, BQM size, and coupling density.

- **S1** is the most general (supports arbitrary partition sizes) but produces the largest BQMs due to slack variables.
- **S2** eliminates slack variables entirely, producing the smallest BQMs, but requires unit partition sizes.
- **S3** replaces the O(N²) k-safety penalty with an O(N) domain-wall chain, which should embed more efficiently on sparse hardware topologies, at the cost of additional auxiliary variables.

For full details on the QUBO formulations, penalty functions, variable counts, and constraint encoding, see [`simulated_solvers/README.md`](simulated_solvers/README.md).


## Hardware Solvers

QPU versions of S1–S3. Each inherits `build_bqm()` from its simulated parent unchanged and only overrides `solve()` to use `EmbeddingComposite(DWaveSampler())`. The BQM formulation is identical — only the sampler changes. This means simulated and hardware results for the same formulation are directly comparable.

Hardware solvers additionally expose a `hardware_summary()` method returning QPU access time, physical qubit count, chain break fraction, and other embedding metadata.

For full details on hardware-specific parameters (`annealing_time`, `chain_strength`, `solver_name`), timing breakdown, chain strength tuning, and usage examples, see [`quantum_hardware_solvers/README.md`](quantum_hardware_solvers/README.md).
