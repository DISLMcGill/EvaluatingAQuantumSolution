# Quantum Hardware Solvers

This directory contains QPU versions of the three simulated SQA solvers. Each hardware solver inherits `build_bqm()` from its simulated counterpart unchanged — the QUBO formulation is identical — and overrides only `solve()` to submit the BQM to a real D-Wave quantum annealer via `EmbeddingComposite(DWaveSampler())`.

For details on how each BQM is constructed (variable types, penalty functions, constraint encoding), see [`simulated_solvers/README.md`](../simulated_solvers/README.md).


## Prerequisites

1. Install the D-Wave system client:
   ```
   pip install dwave-system
   ```

2. Configure your LEAP API token:
   ```
   dwave setup
   ```
   Alternatively, set the `DWAVE_API_TOKEN` environment variable.

3. Verify connectivity:
   ```
   dwave ping
   ```


## Files

| File | Class | Inherits From | Label |
|------|-------|---------------|-------|
| `SQA_HW.py` | `SQAHardwareSolver` | `SQASolver` | S1 Hardware |
| `SQA_SF_HW.py` | `SQASFHardwareSolver` | `SQASlackFreeSolver` | S2 Hardware |
| `SQA_DW_HW.py` | `SQADWHardwareSolver` | `SQADomainWallSolver` | S3 Hardware |
| `__init__.py` | — | — | Package init with convenience imports |


## Constructor Parameters

All three hardware solvers accept the same constructor arguments as their simulated parents, plus one additional parameter:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nodes` | `dict[str, int]` | — | Node capacities |
| `partitions` | `dict[str, int]` | — | Partition sizes |
| `k_safety` | `int` | — | Replication factor |
| `requests` | `dict[tuple, int]` | — | Request frequencies keyed by `(partition_id, node_id)` |
| `comm_costs` | `dict[str, int]` | — | Per-partition communication cost |
| `solver_name` | `str` or `None` | `None` | D-Wave QPU identifier (e.g. `"Advantage_system6.4"`). If `None`, the LEAP client's default QPU is used |

### Pinning a solver

Specifying `solver_name` ensures reproducibility by targeting a specific QPU. D-Wave periodically recalibrates machines and retires older systems, so results may vary across QPUs or calibration cycles. To list available solvers:

```python
from dwave.system import DWaveSampler
print(DWaveSampler().properties['chip_id'])
```

Or check the LEAP dashboard at https://cloud.dwavesys.com.


## `solve()` Parameters

The `solve()` method replaces the simulated sampler's parameters with QPU-appropriate ones:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_reads` | `int` | 100 | Number of annealing cycles (samples). Each read is a physical anneal on the QPU. Lower default than the simulated solvers (1000) because hardware reads are real quantum anneals, not Monte Carlo sweeps |
| `annealing_time` | `int` | 20 | Anneal duration in microseconds. Advantage systems support roughly 0.5–2000 µs. Longer anneals allow the system more time to find the ground state but consume more QPU time |
| `chain_strength` | `float` or `None` | `None` | Coupling strength for physical qubit chains (see below). If `None`, `EmbeddingComposite` applies its default heuristic (`uniform_torque_compensation`) |

### Parameters not carried over from simulated solvers

- **`num_sweeps`** — a simulation concept (Monte Carlo sweeps per read). Does not apply to hardware.
- **`beta_range`** — the inverse temperature schedule for Path Integral Monte Carlo. On hardware, the anneal schedule is controlled by `annealing_time` and the QPU's built-in anneal schedule.


## Chain Strength

When `EmbeddingComposite` maps logical variables to physical qubits, a single logical variable often requires multiple physical qubits coupled together in a "chain" via strong ferromagnetic interactions. The `chain_strength` parameter controls the magnitude of these couplings.

- **Too weak:** chains break during annealing, producing invalid logical states. Manifests as high `chain_break_fraction` in the results.
- **Too strong:** chain couplings dominate the problem's energy landscape, washing out the signal from the actual objective and constraint terms. The QPU effectively "sees" mostly chain couplings rather than your problem.

The default heuristic works reasonably well for many problems. For these solvers, where penalty weights `h` and `h_k` can be large, a good manual starting point is:

```python
chain_strength = 0.8 * max(abs(v) for v in bqm.quadratic.values())
```

If you observe high chain break fractions (> 0.1), increase chain strength. If validity rates are poor despite low chain breaks, the chain strength may be too high — try reducing it.


## `hardware_summary()`

Each solver provides a `hardware_summary()` method that returns a dict of QPU execution metadata after `solve()` has been called:

| Key | Type | Description |
|-----|------|-------------|
| `wall_time_ms` | `float` | Total wall-clock time including network latency, embedding, QPU access, and readout |
| `qpu_access_time_us` | `int` | Total time the job occupied the QPU, in microseconds |
| `qpu_anneal_time_per_sample_us` | `int` | Anneal duration per sample in microseconds (should match `annealing_time`) |
| `physical_qubits` | `int` | Total physical qubits used after embedding (sum of all chain lengths) |
| `logical_variables` | `int` | Number of logical variables in the BQM |
| `chain_break_fraction` | `float` | Mean fraction of samples containing at least one broken chain (0.0 = no breaks) |
| `num_reads` | `int` | Number of samples returned |

### Timing breakdown

Wall-clock time on hardware is dominated by network latency and QPU queue wait, not computation. For scientific comparison, use `qpu_access_time_us` (or the full `self.qpu_timing` dict which contains a detailed breakdown including programming time, sampling time, and readout time). The complete timing dict is accessible via `solver.qpu_timing` after calling `solve()`.


## Additional Instance Attributes

After calling `solve()`, each hardware solver populates:

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.result` | `dimod.SampleView` | Best sample (lowest energy). Same format as simulated solvers — compatible with `calculate_solution_cost()` and `is_valid_solution()` |
| `self.sampleset` | `dimod.SampleSet` | Full sample set with all reads, energies, and metadata |
| `self.embedding` | `dict` | The minor embedding used: maps logical variable names to lists of physical qubit indices |
| `self.qpu_timing` | `dict` | Full QPU timing breakdown from `sampleset.info['timing']` |
| `self.chain_break_fraction` | `float` | Mean chain break fraction across all samples |
| `self.physical_qubits` | `int` | Total physical qubits consumed |


## Usage Example

```python
from solvers.quantum_hardware_solvers import SQASFHardwareSolver
from util.test_generation.json_to_dict import json_to_test_case
from util.calculate_solution_cost import calculate_solution_cost, is_valid_solution

# Load a test case
nodes, partitions, k_safety, requests, comm_costs = json_to_test_case("test_bank/unit_partition/tier1/n-3_p-3_1.json")

# Solve on hardware
solver = SQASFHardwareSolver(nodes, partitions, k_safety, requests, comm_costs)
time_ms, result = solver.solve(num_reads=200, annealing_time=50)

# Evaluate
cost = calculate_solution_cost(nodes, partitions, k_safety, requests, comm_costs, result)
valid = is_valid_solution(nodes, partitions, k_safety, requests, comm_costs, result)

print(f"Cost: {cost}, Valid: {valid}")
print(solver.hardware_summary())
```


## Formulation Restrictions

The same restrictions that apply to the simulated solvers apply here:

| Solver | Capacity Restriction | Partition Size Restriction |
|--------|---------------------|--------------------------|
| S1 (`SQAHardwareSolver`) | Mersenne numbers only (2^k − 1) | Any |
| S2 (`SQASFHardwareSolver`) | Any positive integer | Must all equal 1 |
| S3 (`SQADWHardwareSolver`) | Any positive integer | Must all equal 1 |
