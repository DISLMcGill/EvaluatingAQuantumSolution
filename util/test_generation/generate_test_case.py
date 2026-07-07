"""
Generate random feasible test cases with arbitrary (non-unit) partition sizes.

After the S1 baseline fix (Phase 1), capacities are NOT rounded to
Mersenne numbers any more.  Real-world capacities are arbitrary integers
and the QUBO encoding now handles them exactly.

Test cases are also stratified by *storage-constraint tightness*:

    tightness = 1.0  -> capacity is the minimum required for feasibility;
                       every storage constraint binds at the optimum.
    tightness = 0.0  -> capacity is double the minimum; storage rarely binds.

This lets downstream experiments isolate the effect of the storage
encoding from other variables.
"""

import json
import random
from pathlib import Path

from util.test_generation.feasibility_probe import (
    constraints_feasible,
    full_optimize_feasible,
)


def _draw_capacity(min_cap, tightness, rng):
    """
    Return an integer capacity sampled to satisfy the requested tightness.

    tightness=1.0 -> capacity == min_cap exactly.
    tightness=0.0 -> capacity in [1.5*min_cap, 2.0*min_cap].
    """
    if not 0.0 <= tightness <= 1.0:
        raise ValueError(f"tightness must be in [0, 1], got {tightness}")
    slack_factor = 1.0 + (1.0 - tightness)        # 1.0..2.0
    upper = max(min_cap, int(round(min_cap * slack_factor)))
    return rng.randint(min_cap, upper)


def generate_test_case(
    n_nodes,
    n_partitions,
    k_safety=2,
    seed=None,
    size_range=(5, 20),
    req_range=(0, 10),
    cost_range=(1, 10),
    tightness=0.5,
    feasibility_retries=20,
    feasibility_only=False,
    probe_time_limit=None,
):
    """
    Generate a random, feasible test case with variable partition sizes.

    Args:
        n_nodes:               number of storage nodes
        n_partitions:          number of data partitions
        k_safety:              replication factor
        seed:                  random seed (None = non-deterministic)
        size_range:            (min, max) partition sizes (inclusive)
        req_range:             (min, max) per-(p, n) request frequencies
        cost_range:            (min, max) per-partition communication costs
        tightness:             0.0 (loose) .. 1.0 (tight) -- see module docstring
        feasibility_retries:   max attempts before raising RuntimeError
        feasibility_only:      use the scalable zero-objective feasibility
                               probe instead of a full ILP optimize.  Leave
                               False to preserve the legacy behaviour used by
                               the tier-1/2 banks; set True for large
                               instances where a full optimize is intractable.
        probe_time_limit:      CBC wall-clock cap (seconds) for the
                               feasibility-only probe.  Ignored when
                               feasibility_only is False.

    Returns:
        dict in standard test-case JSON format.  Capacities are kept as
        whatever the random draw produced -- *no Mersenne rounding*.

    Raises:
        ValueError: k_safety > n_nodes.
        RuntimeError: could not produce a feasible instance in the retry budget.
    """
    if k_safety > n_nodes:
        raise ValueError(f"k_safety ({k_safety}) cannot exceed n_nodes ({n_nodes})")

    base_rng = random.Random(seed)

    for attempt in range(feasibility_retries):
        rng = random.Random(base_rng.random())

        sizes = [rng.randint(*size_range) for _ in range(n_partitions)]
        max_size = max(sizes)
        total_size = sum(sizes)

        # Minimum per-node capacity so total capacity >= k * total_size
        # (necessary feasibility condition for a uniform draw).
        min_cap = -(-k_safety * total_size // n_nodes)
        min_cap = max(min_cap, max_size)   # each node must hold the largest partition

        node_caps = [_draw_capacity(min_cap, tightness, rng) for _ in range(n_nodes)]

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
            "nodes":      {f"n{ni}": cap for ni, cap in enumerate(node_caps, 1)},
            "partitions": {f"p{pi}": sz for pi, sz in enumerate(sizes, 1)},
            "k_safety":   k_safety,
            "requests":   requests,
            "comm_costs": comm_costs,
            "tightness":  round(tightness, 3),
        }

        # Verify the instance is actually feasible with an ILP probe.
        if _is_feasible(tc, feasibility_only, probe_time_limit):
            return tc

    raise RuntimeError(
        f"Could not generate a feasible instance after {feasibility_retries} "
        f"attempts (n_nodes={n_nodes}, n_partitions={n_partitions}, "
        f"k_safety={k_safety}, tightness={tightness})."
    )


def _is_feasible(tc, feasibility_only=False, probe_time_limit=None):
    """Run a feasibility probe on the given test case.

    feasibility_only=False -> legacy full ILP optimize probe.
    feasibility_only=True  -> scalable zero-objective probe (optionally
                              wall-clock capped) for large instances.
    """
    if feasibility_only:
        return constraints_feasible(tc, probe_time_limit=probe_time_limit)
    return full_optimize_feasible(tc)


def generate_batch(
    n_nodes,
    n_partitions,
    count,
    output_dir,
    k_safety=2,
    base_seed=None,
    **kwargs,
):
    """Generate `count` test cases and save them as JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(1, count + 1):
        seed = None if base_seed is None else base_seed + i
        tc = generate_test_case(n_nodes, n_partitions, k_safety=k_safety, seed=seed, **kwargs)
        fpath = output_dir / f"n-{n_nodes}_p-{n_partitions}_{i}.json"
        fpath.write_text(json.dumps(tc, indent=4))
        paths.append(fpath)

    return paths
