# Simulated Solvers

This directory contains three Simulated Quantum Annealing (SQA) solvers for the data allocation optimisation problem. Each solver constructs a Binary Quadratic Model (BQM) encoding the same objective and constraints, but uses a different QUBO formulation. All three are solved locally on the CPU using D-Wave's `PathIntegralAnnealingSampler` (Path Integral Monte Carlo).

The solvers share a common interface defined by `SolverBase` (see `util/solver_base.py`) and expose two key methods: `build_bqm()`, which constructs the BQM without solving it, and `solve()`, which builds and samples the BQM.


## Files

| File | Class | Label |
|------|-------|-------|
| `SQA.py` | `SQASolver` | S1 |
| `SQA_SF.py` | `SQASlackFreeSolver` | S2 |
| `SQA_DW.py` | `SQADomainWallSolver` | S3 |


## Common Interface

All three solvers accept the same constructor arguments:

| Parameter | Type | Description |
|-----------|------|-------------|
| `nodes` | `dict[str, int]` | Node capacities, e.g. `{"n1": 7, "n2": 15}` |
| `partitions` | `dict[str, int]` | Partition sizes, e.g. `{"p1": 1, "p2": 1}` |
| `k_safety` | `int` | Replication factor — each partition is stored on exactly this many nodes |
| `requests` | `dict[tuple, int]` | Request frequencies keyed by `(partition_id, node_id)` |
| `comm_costs` | `dict[str, int]` | Per-partition communication cost |

After calling `solve()`, the solver stores its result in `self.result` (a `dimod.SampleView`) and the wall-clock time in `self.time_taken` (milliseconds).

### `solve()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_reads` | `int` | 1000 | Number of independent annealing runs (samples) |
| `num_sweeps` | `int` | 1000 | Number of Monte Carlo sweeps per read |
| `beta_range` | `tuple` or `None` | `None` | Inverse temperature schedule `(beta_start, beta_end)`. If `None`, the sampler's default heuristic is used |


## S1 — `SQASolver` (SQA.py)

The baseline QUBO formulation. Uses two types of binary variable:

- **Assignment variables** `A_{p}_{n}`: 1 if partition `p` is stored on node `n`, 0 otherwise.
- **Slack variables** `S_{n}_{i}`: binary encoding of unused capacity on node `n`, using power-of-two chunks (1, 2, 4, ..., 2^(k-1)).

### Constraint Encoding

**k-Safety (Q_R):** For each partition, `sum_n(A_{p,n}) = k` is enforced as a quadratic equality penalty with Lagrange multiplier `h`, where `h = sum(all r_{p,n} * c_p) + 1`. This ensures the penalty for any constraint violation exceeds the maximum possible objective improvement.

**Storage (Q_S):** For each node, the total assigned partition size must equal the sum of slack variable chunks: `sum_p(A_{p,n} * size_p) = sum_i(S_{n,i} * i)`. This converts the capacity inequality into an equality by absorbing unused capacity into the slack variables.

**Objective (Q_C):** Linear bias `−r_{p,n} * c_p` on each `A_{p,n}`. Minimising the BQM energy maximises local data placement, reducing remote communication cost.

### Capacity Restriction

Node capacities **must be Mersenne numbers** (2^k − 1: 1, 3, 7, 15, 31, 63, ...). The binary slack encoding uses chunks {1, 2, 4, ..., 2^(k-1)} which sum to exactly 2^k − 1. If the capacity is not Mersenne, the slack variables cannot represent the full range of unused capacity, and the storage constraint may not bind correctly.

### Variable Count

For `P` partitions and `N` nodes: `P × N` assignment variables, plus `sum_n(ceil(log2(C_n + 1)))` slack variables.


## S2 — `SQASlackFreeSolver` (SQA_SF.py)

Eliminates all slack variables by encoding the storage inequality directly as a quadratic penalty using only assignment variables. Based on the unbalanced penalty function from Montañez-Barrera et al. (2022).

### Partition Size Restriction

All partition sizes **must equal 1**. This simplifies the storage constraint to a pure cardinality constraint: `sum_p(A_{p,n}) <= C_n`.

### Constraint Encoding

**k-Safety (Q_R):** Same quadratic equality penalty as S1, but with a scaled Lagrange multiplier `h_k = h * max(C_n)` to ensure k-safety always dominates the storage penalty for feasible assignments.

**Storage (Q_S) — Unbalanced Penalty:** For each node with capacity `C`, the penalty function is:

```
P(x) = (x − C)(x − C + 1)
```

where `x = sum_p(A_{p,n})`. This function is zero at `x = C` and `x = C − 1`, and positive for any other value — penalising both overflow and significant underuse. For binary variables (`x^2 = x + 2 * cross_terms`), this expands to:

- Linear coefficient per `A_{p,n}`: `h_s * (2 − 2C)`
- Quadratic coefficient per pair `(A_{p,n}, A_{p',n})`: `2 * h_s`

where `h_s = h` (the base penalty weight).

**Objective (Q_C):** Identical to S1.

### Capacity Restriction

Capacities do **not** need to be Mersenne numbers. Any positive integer capacity is supported.

### Variable Count

`P × N` assignment variables only — no slack variables. This is the smallest BQM of the three formulations.


## S3 — `SQADomainWallSolver` (SQA_DW.py)

Combines the slack-free storage encoding from S2 with a domain-wall encoding for the k-safety constraint, based on Chancellor (2019).

### Partition Size Restriction

All partition sizes **must equal 1** (same as S2).

### Constraint Encoding

**k-Safety (Q_R) — Domain-Wall Encoding:** For each partition `p`, introduces `N` auxiliary wall variables `W_{p,1}, ..., W_{p,N}` forming a monotone chain where `W_{p,j} = 1` means "partition `p` has at least `j` copies". The chain is enforced by nearest-neighbour penalties:

- *Chain monotonicity:* penalise `W_{p,j+1} = 1, W_{p,j} = 0` (a "domain wall violation") with weight `h_dw`.
- *Count enforcement:* `W_{p,k}` must be 1 (penalty `h_dw * (1 − W_{p,k})`) and `W_{p,k+1}` must be 0 (penalty `h_dw * W_{p,k+1}`).
- *Linking:* `sum_n(A_{p,n}) = sum_j(W_{p,j})` is enforced as a quadratic equality constraint with weight `h_link`.

The chain constraint produces O(N) nearest-neighbour couplings per partition rather than the O(N^2) all-to-all couplings of the standard `(sum − k)^2` penalty. This is particularly relevant for hardware embedding, where sparse coupling patterns map more efficiently to the QPU's physical graph.

**Storage (Q_S):** Identical to S2 (unbalanced penalty, slack-free).

**Objective (Q_C):** Identical to S1 and S2.

### Penalty Scaling

All constraint penalties are scaled relative to the base weight `h`:

| Weight | Value | Purpose |
|--------|-------|---------|
| `h_dw` | `h * max(C_n)` | Domain-wall chain and count enforcement |
| `h_link` | `h * max(C_n)` | Wall-to-assignment linking equality |
| `h_s` | `h` | Storage penalty |

### Variable Count

`P × N` assignment variables plus `P × N` wall variables = `2 × P × N` total. More variables than S2 but with sparser coupling.


## References

- Montañez-Barrera, J. A., et al. (2022). "Unbalanced penalization: A new approach to encode inequality constraints for quantum optimization algorithms." arXiv:2211.13914.
- Chancellor, N. (2019). "Domain wall encoding of discrete variables for quantum annealing and QAOA." arXiv:1903.05068.
