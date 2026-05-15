"""
Core experiment harness for QuantumClean.

Loads pre-generated test cases from disk, runs every registered solver on
each case, computes cost / validity / BQM statistics / optimality gap vs
ILP, and writes results incrementally to a JSON file.

This module provides the engine; the thin wrapper scripts
(run_unit_partition_experiment.py, run_arbitrary_partition_experiment.py)
supply the test-case paths and solver registrations.

Result format matches the QuantumPersonal SweepV2 / UnitSweep JSON
structure so downstream result_analysis can work uniformly across old and new
results.
"""

import json
import re
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

from util.calculate_solution_cost import (
    calculate_solution_cost,
    is_valid_solution,
)
from util.test_generation.json_to_dict import json_to_test_case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalars so json.dump doesn't choke on them."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def discover_test_cases(test_bank_dir, tier=None, node_counts=None,
                        partition_counts=None, max_cases=None):
    """
    Discover JSON test case files under *test_bank_dir*.

    Args:
        test_bank_dir:    root directory to search (e.g. test_bank/unit_partition)
        tier:             optional "tier1" or "tier2" filter
        node_counts:      optional list of node counts to include (e.g. [2, 3, 5])
        partition_counts: optional list of partition counts to include
        max_cases:        cap the total number of test cases returned

    Returns:
        Sorted list of Path objects.
    """
    test_bank_dir = Path(test_bank_dir)
    search_root = test_bank_dir / tier if tier else test_bank_dir
    paths = sorted(search_root.rglob("*.json"))

    if node_counts is not None or partition_counts is not None:
        filtered = []
        for p in paths:
            # Directory names are like n5_p18
            m = re.match(r'n(\d+)_p(\d+)', p.parent.name)
            if not m:
                continue
            n, pp = int(m.group(1)), int(m.group(2))
            if node_counts is not None and n not in node_counts:
                continue
            if partition_counts is not None and pp not in partition_counts:
                continue
            filtered.append(p)
        paths = filtered

    if max_cases is not None:
        paths = paths[:max_cases]

    return paths


def _flatten_ilp_result(ilp_result, partitions, nodes):
    """
    Convert ILP's nested {p: {n: 0|1}} result dict into the flat
    {'A_p1_n1': 0, ...} format that calculate_solution_cost expects.
    """
    if ilp_result is None:
        return None
    return {
        f'A_{p}_{n}': v
        for p, nd in ilp_result.items()
        for n, v in nd.items()
    }


def _run_ilp(solver_class, nodes, partitions, k_safety, requests, comm_costs):
    """Run ILP solver, return result dict in the standard format."""
    try:
        solver = solver_class(nodes, partitions, k_safety, requests, comm_costs)
        t_ms, raw_result = solver.solve()

        flat = _flatten_ilp_result(raw_result, partitions, nodes)
        cost = calculate_solution_cost(
            nodes, partitions, k_safety, requests, comm_costs, flat
        )
        valid = is_valid_solution(
            nodes, partitions, k_safety, requests, comm_costs, flat
        )

        return {
            "cost": cost,
            "valid": valid,
            "time_ms": round(t_ms, 1) if t_ms is not None else None,
            "error": None,
        }
    except Exception as e:
        return {
            "cost": None,
            "valid": False,
            "time_ms": None,
            "error": str(e),
        }


def _run_sqa(solver_class, nodes, partitions, k_safety, requests, comm_costs,
             num_reads, num_sweeps, beta_range):
    """Run an SQA-family solver, return result dict in the standard format."""
    try:
        solver = solver_class(nodes, partitions, k_safety, requests, comm_costs)
        bqm = solver.build_bqm()
        bqm_vars = len(bqm.variables)
        bqm_interactions = len(bqm.quadratic)

        solve_kwargs = dict(num_reads=num_reads, num_sweeps=num_sweeps)
        if beta_range is not None:
            solve_kwargs["beta_range"] = beta_range

        t_ms, result = solver.solve(**solve_kwargs)

        cost = calculate_solution_cost(
            nodes, partitions, k_safety, requests, comm_costs, result
        )
        valid = is_valid_solution(
            nodes, partitions, k_safety, requests, comm_costs, result
        )

        return {
            "cost": cost,
            "valid": valid,
            "time_ms": round(t_ms, 1),
            "bqm_variables": bqm_vars,
            "bqm_interactions": bqm_interactions,
            "error": None,
        }
    except Exception as e:
        return {
            "cost": None,
            "valid": False,
            "time_ms": None,
            "bqm_variables": None,
            "bqm_interactions": None,
            "error": str(e),
        }


def _run_qpu(solver_class, nodes, partitions, k_safety, requests, comm_costs,
             num_reads, annealing_time, chain_strength):
    """Run a QPU hardware solver, return result dict with hardware metadata."""
    try:
        solver = solver_class(nodes, partitions, k_safety, requests, comm_costs)
        bqm = solver.build_bqm()
        bqm_vars = len(bqm.variables)
        bqm_interactions = len(bqm.quadratic)

        solve_kwargs = dict(num_reads=num_reads, annealing_time=annealing_time)
        if chain_strength is not None:
            solve_kwargs["chain_strength"] = chain_strength

        t_ms, result = solver.solve(**solve_kwargs)

        cost = calculate_solution_cost(
            nodes, partitions, k_safety, requests, comm_costs, result
        )
        valid = is_valid_solution(
            nodes, partitions, k_safety, requests, comm_costs, result
        )

        hw = solver.hardware_summary()

        return {
            "cost": cost,
            "valid": valid,
            "time_ms": round(t_ms, 1),
            "bqm_variables": bqm_vars,
            "bqm_interactions": bqm_interactions,
            "physical_qubits": hw.get("physical_qubits"),
            "chain_break_fraction": hw.get("chain_break_fraction"),
            "qpu_access_time_us": hw.get("qpu_access_time_us"),
            "error": None,
        }
    except Exception as e:
        return {
            "cost": None,
            "valid": False,
            "time_ms": None,
            "bqm_variables": None,
            "bqm_interactions": None,
            "physical_qubits": None,
            "chain_break_fraction": None,
            "qpu_access_time_us": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_experiment(
    test_case_paths,
    solver_registry,
    output_dir,
    file_prefix="Experiment",
    num_reads=1000,
    num_sweeps=1000,
    beta_range=None,
    annealing_time=20,
    chain_strength=None,
    note=None,
):
    """
    Run a full experiment: every solver on every test case.

    Args:
        test_case_paths: list of Path objects pointing to JSON test cases.
        solver_registry: list of solver descriptors, each a dict:
            {
                "name":  str,               # e.g. "ILP", "SQA", "SQA_HW"
                "class": class,             # solver class
                "type":  "ilp"|"sqa"|"qpu", # determines how we invoke it
            }
        output_dir:      directory for the result JSON file.
        file_prefix:     prefix for the auto-numbered output filename.
        num_reads:       num_reads for SQA and QPU solvers (ignored for ILP).
        num_sweeps:      SQA num_sweeps (ignored for ILP and QPU solvers).
        beta_range:      SQA beta_range (ignored for ILP and QPU solvers).
        annealing_time:  QPU anneal duration in microseconds (ignored for
                         ILP and simulated SQA solvers).  Default 20.
        chain_strength:  QPU chain strength (ignored for ILP and simulated
                         SQA solvers).  None = sampler default heuristic.
        note:            optional string added to metadata.

    Returns:
        Path to the created results file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-increment file number
    existing = output_dir.glob(f"{file_prefix}_*.json")
    numbers = [
        int(m.group(1))
        for f in existing
        if (m := re.search(rf'{file_prefix}_(\d+)\.json', f.name))
    ]
    file_num = max(numbers, default=0) + 1
    output_path = output_dir / f"{file_prefix}_{file_num}.json"

    solver_names = [s["name"] for s in solver_registry]
    has_sqa = any(s["type"] == "sqa" for s in solver_registry)
    has_qpu = any(s["type"] == "qpu" for s in solver_registry)

    metadata = {
        "date":            date.today().isoformat(),
        "time":            datetime.now().strftime("%H:%M:%S"),
        "total_cases":     len(test_case_paths),
        "num_reads":       num_reads,
        "solvers":         solver_names,
    }
    if has_sqa:
        metadata["num_sweeps"] = num_sweeps
    if has_qpu:
        metadata["annealing_time"] = annealing_time
        if chain_strength is not None:
            metadata["chain_strength"] = chain_strength

    output = {"metadata": metadata, "results": {}}
    if note:
        output["metadata"]["note"] = note

    # Write initial metadata
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4, cls=_NumpyEncoder)

    total = len(test_case_paths)
    sweep_start = time.perf_counter()

    for idx, tc_path in enumerate(test_case_paths, 1):
        tc_path = Path(tc_path)
        key = tc_path.stem  # e.g. "n-3_p-8_1"

        nodes, partitions, k_safety, requests, comm_costs = json_to_test_case(
            str(tc_path)
        )

        entry = {
            "source_file": str(tc_path.relative_to(tc_path.parents[3])),
            "n_nodes": len(nodes),
            "n_partitions": len(partitions),
            "k_safety": k_safety,
            "solvers": {},
        }

        # --- Run each solver ---
        ilp_cost = None

        for solver_desc in solver_registry:
            name = solver_desc["name"]
            cls = solver_desc["class"]
            solver_type = solver_desc["type"]

            if solver_type == "ilp":
                result = _run_ilp(
                    cls, nodes, partitions, k_safety, requests, comm_costs
                )
                if result["valid"] and result["cost"] is not None:
                    ilp_cost = result["cost"]
            elif solver_type == "sqa":
                result = _run_sqa(
                    cls, nodes, partitions, k_safety, requests, comm_costs,
                    num_reads=num_reads,
                    num_sweeps=num_sweeps,
                    beta_range=beta_range,
                )
                # Compute optimality gap vs ILP
                if (ilp_cost is not None and ilp_cost > 0
                        and result["valid"]
                        and result["cost"] is not None):
                    result["optimality_gap"] = round(
                        (result["cost"] - ilp_cost) / ilp_cost, 4
                    )
                else:
                    result["optimality_gap"] = None
            elif solver_type == "qpu":
                result = _run_qpu(
                    cls, nodes, partitions, k_safety, requests, comm_costs,
                    num_reads=num_reads,
                    annealing_time=annealing_time,
                    chain_strength=chain_strength,
                )
                # Compute optimality gap vs ILP
                if (ilp_cost is not None and ilp_cost > 0
                        and result["valid"]
                        and result["cost"] is not None):
                    result["optimality_gap"] = round(
                        (result["cost"] - ilp_cost) / ilp_cost, 4
                    )
                else:
                    result["optimality_gap"] = None
            else:
                raise ValueError(f"Unknown solver type: {solver_type!r}")

            entry["solvers"][name] = result

        output["results"][key] = entry

        # Incremental save
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4, cls=_NumpyEncoder)

        # Progress
        elapsed = time.perf_counter() - sweep_start
        rate = idx / elapsed if elapsed > 0 else 0
        eta = (total - idx) / rate if rate > 0 else 0
        eta_min = int(eta // 60)
        eta_sec = int(eta % 60)

        solver_status = "  ".join(
            f"{s['name']}={'OK' if entry['solvers'][s['name']].get('valid') else 'X'}"
            for s in solver_registry
        )
        print(
            f"  [{idx}/{total}] {key}: {solver_status}"
            f"  [ETA {eta_min}m{eta_sec:02d}s]"
        )

    elapsed_total = time.perf_counter() - sweep_start
    print(f"\nCompleted {total} cases in {elapsed_total / 60:.1f} minutes.")
    print(f"Saved to: {output_path}")
    return output_path
