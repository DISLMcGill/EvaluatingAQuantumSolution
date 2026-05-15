# Experiment Execution

This directory contains the harness for running benchmark experiments against the pre-generated test cases in `test_bank/`. The harness loads test cases from disk, runs every registered solver on each one, records a set of statistics about each solver's output, and writes the results incrementally to a JSON file in `result_bank/`.

The harness supports three solver types: classical ILP (baseline), simulated SQA (Path Integral Monte Carlo on CPU), and QPU (real D-Wave quantum hardware). Simulated and hardware results are written to separate directories so they can be analysed independently.


## Quick start

From the QuantumClean project root:

```bash
# Run the unit-partition benchmark (tier 1, simulated solvers)
python -m util.experiment_execution.run_unit_partition_experiment

# Run the arbitrary-partition benchmark (tier 1, ILP + simulated SQA only)
python -m util.experiment_execution.run_arbitrary_partition_experiment
```

To run on D-Wave hardware instead, call the wrapper functions with `hardware=True`:

```python
from util.experiment_execution.run_unit_partition_experiment import run_unit_experiment

run_unit_experiment(
    tier="tier1",
    hardware=True,
    num_reads=100,
    annealing_time=20,
)
```

Both commands require the test bank to be populated first. If `test_bank/` is empty, run `python -m util.test_generation.populate_test_bank` beforehand. Hardware mode additionally requires `dwave-system` installed and a LEAP API token configured (see `solvers/quantum_hardware_solvers/README.md`).


## File overview

| File | Purpose |
|---|---|
| `run_experiment.py` | Core harness. Provides `run_experiment()`, which takes a list of test case paths and a solver registry, runs everything, and writes the result JSON. Also provides `discover_test_cases()` for finding and filtering test cases on disk, and three internal runners: `_run_ilp()`, `_run_sqa()`, and `_run_qpu()`. |
| `run_unit_partition_experiment.py` | Thin wrapper for unit-partition benchmarks. Registers ILP + SQA + SQA_SF + SQA_DW (simulated) or ILP + SQA_HW + SQA_SF_HW + SQA_DW_HW (hardware). Discovers test cases from `test_bank/unit_partition/` and writes results as `UnitExperiment_N.json` or `UnitExperiment_HW_N.json`. |
| `run_arbitrary_partition_experiment.py` | Thin wrapper for arbitrary-partition benchmarks. Registers ILP + SQA (simulated) or ILP + SQA_HW (hardware). Only S1 supports arbitrary partition sizes, so S2/S3 are excluded. Writes results as `ArbitraryExperiment_N.json` or `ArbitraryExperiment_HW_N.json`. |

The two wrapper scripts are intentionally short. Each one wires up the right test-case directory, solver list, and output prefix, then delegates to the core harness. Hardware solver imports are deferred inside a `_get_hw_registry()` function so the files don't error out on machines without `dwave-system` installed.


## Solver types and dispatch

The `run_experiment()` harness dispatches to a type-specific runner based on each solver's `"type"` field:

| Type | Runner | Solver interface | Parameters passed |
|---|---|---|---|
| `"ilp"` | `_run_ilp()` | `solver.solve()` (no args) | None |
| `"sqa"` | `_run_sqa()` | `solver.solve(num_reads, num_sweeps, beta_range)` | `num_reads`, `num_sweeps`, `beta_range` |
| `"qpu"` | `_run_qpu()` | `solver.solve(num_reads, annealing_time, chain_strength)` | `num_reads`, `annealing_time`, `chain_strength` |

All three runners call `build_bqm()` before solving (for SQA and QPU types) to record BQM statistics, then evaluate the result with `calculate_solution_cost()` and `is_valid_solution()`. The QPU runner additionally calls `solver.hardware_summary()` to capture QPU-specific metadata.


## `run_experiment()` parameters

| Parameter | Type | Default | Used by |
|---|---|---|---|
| `test_case_paths` | `list[Path]` | — | All |
| `solver_registry` | `list[dict]` | — | All |
| `output_dir` | `str` or `Path` | — | All |
| `file_prefix` | `str` | `"Experiment"` | All |
| `num_reads` | `int` | 1000 | SQA, QPU |
| `num_sweeps` | `int` | 1000 | SQA only |
| `beta_range` | `tuple` or `None` | `None` | SQA only |
| `annealing_time` | `int` | 20 | QPU only |
| `chain_strength` | `float` or `None` | `None` | QPU only |
| `note` | `str` or `None` | `None` | All |


## Wrapper script parameters

Both `run_unit_experiment()` and `run_arbitrary_experiment()` accept all of the parameters above (except `test_case_paths`, `solver_registry`, `output_dir`, and `file_prefix`, which they set internally), plus:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tier` | `str` or `None` | `None` | `"tier1"`, `"tier2"`, or `None` (both tiers) |
| `node_counts` | `list[int]` or `None` | `None` | Filter by node count, e.g. `[3, 5]` |
| `partition_counts` | `list[int]` or `None` | `None` | Filter by partition count, e.g. `[8, 18]` |
| `max_cases` | `int` or `None` | `None` | Cap total test cases |
| `hardware` | `bool` | `False` | If `True`, use QPU hardware solvers instead of simulated |


## Recorded statistics

Every test case produces a per-solver result entry. The fields differ by solver type.

### ILP results

| Field | Description |
|---|---|
| `cost` | Total communication cost. `null` if the solver failed. |
| `valid` | Whether the solution satisfies all constraints. |
| `time_ms` | Wall-clock solve time in milliseconds. |
| `error` | Error message if the solver threw an exception, otherwise `null`. |

### SQA results (simulated)

All ILP fields, plus:

| Field | Description |
|---|---|
| `bqm_variables` | Number of variables in the BQM. |
| `bqm_interactions` | Number of quadratic interactions (couplings) in the BQM. |
| `optimality_gap` | `(sqa_cost - ilp_cost) / ilp_cost`. `null` if either solution is invalid or ILP cost is zero. |

### QPU results (hardware)

All SQA fields, plus:

| Field | Description |
|---|---|
| `physical_qubits` | Total physical qubits used after minor embedding (sum of all chain lengths). |
| `chain_break_fraction` | Mean fraction of samples containing at least one broken chain. 0.0 means no breaks. |
| `qpu_access_time_us` | Total time the job occupied the QPU, in microseconds. More scientifically meaningful than wall-clock time, which is dominated by network latency. |


### How each statistic is calculated

**Communication cost.** This is the objective function the solvers are minimising. For a solution with assignment variables A_pn (1 if partition p is stored on node n, 0 otherwise):

```
cost = sum over all (p, n):  r_pn * c_p * (1 - A_pn)
```

where r_pn is the request frequency for partition p at node n, and c_p is the communication cost for partition p. In plain terms: every time a partition is *not* stored locally on a node that requests it, the system pays the remote-fetch cost. The ILP finds the exact minimum; the SQA and QPU solvers approximate it.

**Validity.** A solution is valid if and only if it satisfies both constraints:

1. *k-Safety*: each partition is assigned to exactly k nodes (where k = k_safety, typically 2).
2. *Storage capacity*: the total size of partitions assigned to each node does not exceed that node's capacity.

The ILP enforces these as hard constraints. The SQA/QPU solvers encode them as penalty terms in the QUBO, so solutions can violate constraints if the penalty weights are insufficient or the annealer doesn't find a low-energy state.

**BQM variables.** The total number of binary variables in the QUBO formulation. This includes assignment variables (one per partition-node pair) plus any auxiliary variables (slack variables for S1, domain-wall chain variables for S3). Fewer variables generally means a smaller problem. For hardware, fewer logical variables also means fewer physical qubits after embedding.

**BQM interactions.** The number of quadratic couplings in the QUBO. Reflects the density of the problem graph. Denser graphs are harder to embed on sparse hardware topologies.

**Optimality gap.** Measures how far a solution is from the ILP optimum, as a fraction. A gap of 0.0 means the solver matched the ILP exactly; 0.15 means 15% higher cost. Only computed when both solutions are valid and the ILP cost is non-zero.

**Solve time.** Wall-clock time measured with `time.perf_counter()` around the solver's `.solve()` call. For ILP this is the CBC branch-and-bound solve. For simulated SQA this includes all `num_reads` annealing runs. For QPU this includes network latency, embedding, QPU queue wait, and readout — use `qpu_access_time_us` for the actual hardware time. Times are in milliseconds, rounded to one decimal place.

**Physical qubits (QPU only).** When a logical BQM is embedded onto D-Wave's hardware graph, each logical variable may require multiple physical qubits chained together. This field is the sum of all chain lengths across all variables. Comparing this across formulations shows which encoding maps most efficiently to hardware.

**Chain break fraction (QPU only).** Physical qubit chains can "break" during annealing — the qubits in a chain disagree on their value. A high chain break fraction (> 0.1) typically indicates the chain strength is too low. This metric is essential for diagnosing poor QPU solution quality.

**QPU access time (QPU only).** The actual time the job spent on the QPU, in microseconds. This excludes network round-trip time, queue wait, and embedding computation, making it the appropriate metric for comparing hardware execution speed. The full timing breakdown (programming time, sampling time, readout time) is available on the solver object via `solver.qpu_timing` after a run, but only `qpu_access_time_us` is written to the results JSON.


## Filtering and partial runs

Both wrapper scripts accept optional filters so you can run subsets of the test bank:

```python
from util.experiment_execution.run_unit_partition_experiment import run_unit_experiment

# Simulated: tier 1, only 3-node and 5-node problems
run_unit_experiment(tier="tier1", node_counts=[3, 5])

# Simulated: tier 2 cases with 50 or 100 partitions
run_unit_experiment(tier="tier2", partition_counts=[50, 100])

# Quick sanity check: first 10 test cases only
run_unit_experiment(max_cases=10)

# Adjust simulated SQA parameters
run_unit_experiment(num_reads=500, num_sweeps=500)

# Hardware: tier 1, small problems only, with custom anneal time
run_unit_experiment(
    tier="tier1",
    hardware=True,
    node_counts=[2, 3],
    num_reads=200,
    annealing_time=50,
)

# Hardware: explicit chain strength
run_unit_experiment(
    tier="tier1",
    hardware=True,
    num_reads=100,
    chain_strength=2.5,
)
```

The `discover_test_cases()` function handles the filtering. It parses directory names (e.g. `n5_p18`) to match against `node_counts` and `partition_counts`, and applies `max_cases` as a hard cap after filtering.


## Result output

Simulated and hardware results are written to separate directories with auto-incrementing filenames. Each run produces a new file — previous results are never overwritten.

```
result_bank/
    simulated_solver_results/
        UnitExperiment_1.json
        UnitExperiment_2.json
        ArbitraryExperiment_1.json
        ...
    quantum_hardware_results/
        UnitExperiment_HW_1.json
        ArbitraryExperiment_HW_1.json
        ...
```

### Simulated result JSON structure

```json
{
    "metadata": {
        "date": "2026-05-11",
        "time": "14:30:00",
        "total_cases": 400,
        "num_reads": 1000,
        "num_sweeps": 1000,
        "solvers": ["ILP", "SQA", "SQA_SF", "SQA_DW"],
        "note": "Unit-partition benchmark: all partition sizes = 1."
    },
    "results": {
        "n-3_p-8_1": {
            "source_file": "unit_partition/tier1/n3_p8/n-3_p-8_1.json",
            "n_nodes": 3,
            "n_partitions": 8,
            "k_safety": 2,
            "solvers": {
                "ILP": {
                    "cost": 42,
                    "valid": true,
                    "time_ms": 1.3,
                    "error": null
                },
                "SQA": {
                    "cost": 48,
                    "valid": true,
                    "time_ms": 350.2,
                    "bqm_variables": 38,
                    "bqm_interactions": 95,
                    "error": null,
                    "optimality_gap": 0.1429
                }
            }
        }
    }
}
```

### Hardware result JSON structure

Hardware results include the same fields as simulated results, plus QPU-specific metadata. The metadata block includes `annealing_time` instead of `num_sweeps`, and optionally `chain_strength` if one was specified.

```json
{
    "metadata": {
        "date": "2026-05-15",
        "time": "10:00:00",
        "total_cases": 400,
        "num_reads": 100,
        "annealing_time": 20,
        "solvers": ["ILP", "SQA_HW", "SQA_SF_HW", "SQA_DW_HW"],
        "note": "Unit-partition benchmark (D-Wave QPU): all partition sizes = 1."
    },
    "results": {
        "n-3_p-8_1": {
            "source_file": "unit_partition/tier1/n3_p8/n-3_p-8_1.json",
            "n_nodes": 3,
            "n_partitions": 8,
            "k_safety": 2,
            "solvers": {
                "ILP": {
                    "cost": 42,
                    "valid": true,
                    "time_ms": 1.3,
                    "error": null
                },
                "SQA_HW": {
                    "cost": 48,
                    "valid": true,
                    "time_ms": 4520.1,
                    "bqm_variables": 38,
                    "bqm_interactions": 95,
                    "physical_qubits": 142,
                    "chain_break_fraction": 0.023,
                    "qpu_access_time_us": 18540,
                    "error": null,
                    "optimality_gap": 0.1429
                }
            }
        }
    }
}
```

Each entry in `results` is keyed by the test case filename stem (e.g. `n-3_p-8_1`). The `source_file` field records the path relative to the test bank root, so you can trace any result back to its input.


## Incremental saves

The harness writes the full JSON to disk after every single test case. This means that if a long experiment is interrupted (crash, Ctrl-C, laptop running out of battery), all completed results are preserved in the output file. You can inspect partial results while an experiment is still running.


## Solver registry format

Each solver is registered as a dict with three keys:

```python
{"name": "SQA_DW", "class": SQADomainWallSolver, "type": "sqa"}
```

The `type` field determines how the harness invokes the solver:

- `"ilp"` — no BQM, calls `solver.solve()` with no arguments, result is a nested dict.
- `"sqa"` — has `build_bqm()`, calls `solver.solve(num_reads, num_sweeps, beta_range)`.
- `"qpu"` — has `build_bqm()`, calls `solver.solve(num_reads, annealing_time, chain_strength)`, captures hardware metadata via `solver.hardware_summary()`.

The `name` is used as the key in the result JSON and in the progress output.


## Adding a new solver

1. Create the solver class (must subclass `SolverBase`).
2. Import it in the relevant wrapper script.
3. Add it to the appropriate registry list with the correct `type`.
4. If it's a hardware solver, add the import inside `_get_hw_registry()` so the import is deferred and doesn't break environments without `dwave-system`.

The core harness doesn't need to change.


## Dependencies on other modules

The harness imports from two utility modules that live outside this directory:

- `util.calculate_solution_cost` — provides `calculate_solution_cost()` and `is_valid_solution()`, used to evaluate every solver's output.
- `util.test_generation.json_to_dict` — provides `json_to_test_case()`, used to load test case JSON files into the tuple format the solver constructors expect.
