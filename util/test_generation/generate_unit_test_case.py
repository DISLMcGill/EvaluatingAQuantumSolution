"""
Generate random feasible test cases with **unit partition sizes** (all sizes = 1).

Same conventions as ``generate_test_case``: no Mersenne rounding, tightness
parameter controls storage-constraint binding, feasibility verified by ILP.
"""

import json
import random
from pathlib import Path

from util.test_generation.feasibility_probe import (
    constraints_feasible,
    full_optimize_feasible,
)


def _draw_capacity(min_cap, tightness, rng):
    if not 0.0 <= tightness <= 1.0:
        raise ValueError(f"tightness must be in [0, 1], got {tightness}")
    slack_factor = 1.0 + (1.0 - tightness)
    upper = max(min_cap, int(round(min_cap * slack_factor)))
    return rng.randint(min_cap, upper)


def generate_unit_test_case(
    n_nodes,
    n_partitions,
    k_safety=2,
    seed=None,
    req_range=(0, 10),
    cost_range=(1, 10),
    tightness=0.5,
    feasibility_retries=20,
    feasibility_only=False,
    probe_time_limit=None,
):
    """Generate a random feasible test case with all partition sizes = 1.

    ``feasibility_only`` switches the per-instance check from the legacy
    full ILP optimize to the scalable zero-objective probe; for unit
    partitions feasibility is a simple degree/capacity question, so the
    probe is cheap even at the hybrid-stress scale.  ``probe_time_limit``
    caps the CBC wall clock for that probe (seconds).
    """
    if k_safety > n_nodes:
        raise ValueError(f"k_safety ({k_safety}) cannot exceed n_nodes ({n_nodes})")

    base_rng = random.Random(seed)

    for attempt in range(feasibility_retries):
        rng = random.Random(base_rng.random())

        sizes = [1] * n_partitions
        min_cap = -(-k_safety * n_partitions // n_nodes)
        min_cap = max(min_cap, 1)

        caps = [_draw_capacity(min_cap, tightness, rng) for _ in range(n_nodes)]

        requests = {
            f"(p{pi}, n{ni})": rng.randint(*req_range)
            for pi in range(1, n_partitions + 1)
            for ni in range(1, n_nodes + 1)
        }
        comm_costs = {
            f"p{pi}": rng.randint(*cost_range)
            for pi in range(1, n_partitions + 1)
        }

        tc = {
            "nodes":      {f"n{ni}": cap for ni, cap in enumerate(caps, 1)},
            "partitions": {f"p{pi}": sz for pi, sz in enumerate(sizes, 1)},
            "k_safety":   k_safety,
            "requests":   requests,
            "comm_costs": comm_costs,
            "tightness":  round(tightness, 3),
        }

        if _is_feasible(tc, feasibility_only, probe_time_limit):
            return tc

    raise RuntimeError(
        f"Could not generate a feasible unit-partition instance "
        f"(n_nodes={n_nodes}, n_partitions={n_partitions}, k_safety={k_safety})"
    )


def _is_feasible(tc, feasibility_only=False, probe_time_limit=None):
    """Run a feasibility probe (legacy full optimize, or scalable probe)."""
    if feasibility_only:
        return constraints_feasible(tc, probe_time_limit=probe_time_limit)
    return full_optimize_feasible(tc)


def generate_unit_batch(
    n_nodes,
    n_partitions,
    count,
    output_dir,
    k_safety=2,
    base_seed=None,
    **kwargs,
):
    """Generate `count` unit-partition test cases and save as JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(1, count + 1):
        seed = None if base_seed is None else base_seed + i
        tc = generate_unit_test_case(n_nodes, n_partitions, k_safety=k_safety, seed=seed, **kwargs)
        fpath = output_dir / f"n-{n_nodes}_p-{n_partitions}_{i}.json"
        fpath.write_text(json.dumps(tc, indent=4))
        paths.append(fpath)
    return paths
