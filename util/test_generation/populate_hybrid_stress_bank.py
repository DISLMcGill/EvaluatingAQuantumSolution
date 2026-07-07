"""
Populate the ``hybrid_stress`` test bank.

A deliberately small but *hard* bench aimed at pushing the hybrid solvers
(SQA_H / SQA_SF_H) past the tier-1/2 regime.  Where tier 2 tops out at
n15 x p100 (~1,500 binary variables), this bench ramps diagonally from
that point up to n40 x p400 (~16,000 binaries), for both the
unit-partition and arbitrary-partition formulations.

Design choices
--------------
* **Diagonal ramp, 5 cells.**  Node count and partition count grow
  together so each cell is a distinct point on the scaling frontier
  rather than a dense grid:

      n15_p100   ~1.5k vars   (anchor: current tier-2 max)
      n20_p150   ~3.0k vars
      n25_p200   ~5.0k vars
      n30_p300   ~9.0k vars
      n40_p400  ~16.0k vars

* **Feasibility probe only -- no ILP optimum.**  At these sizes a full
  ILP optimize is intractable, so generation uses the scalable
  zero-objective feasibility probe (``feasibility_only=True``).  The
  downstream experiment is expected to compare the hybrid solvers to
  each other / best-found rather than to a proven optimum (run the
  hybrid scripts with ``--no-ilp``).

* **Two tightness levels per formulation (compact).**
    - unit:      t70, t100  (unit partitions stay feasible even when tight)
    - arbitrary: t30, t70   (arbitrary sizes are bin-packing-hard; very
                             high tightness is often infeasible at large p,
                             so we stop at t70)

* **3 instances per cell.**  Enough for a little within-cell variance
  without bloating Leap quota.

Totals: 5 cells x 2 tightness x 3 cases = 30 JSONs per formulation,
60 across unit + arbitrary.

Directory layout (note: no ``tierN`` level -- the bench *is* the root):

    test_bank/hybrid_stress/
        unit_partition/      n{N}_p{P}/t{70,100}/ n-{N}_p-{P}_{i}.json
        arbitrary_partition/ n{N}_p{P}/t{30,70}/  n-{N}_p-{P}_{i}.json
        manifest.json

The ``n{N}_p{P}`` / ``t{TT}`` naming matches the hybrid runners'
``_iter_tightness_dirs`` regex, so a runner pointed at
``test_bank/hybrid_stress/{unit,arbitrary}_partition`` discovers these
cells with no other change.

Usage:
    python -m util.test_generation.populate_hybrid_stress_bank
    python -m util.test_generation.populate_hybrid_stress_bank --probe-time-limit 120
"""

import argparse
import json
import time
from pathlib import Path

from util.test_generation.generate_test_case import generate_batch
from util.test_generation.generate_unit_test_case import generate_unit_batch

TEST_BANK = Path(__file__).resolve().parent.parent.parent / "test_bank"
BENCH_ROOT = TEST_BANK / "hybrid_stress"

# ---------- grid ----------
# (n_nodes, n_partitions) diagonal ramp.
CELLS = [
    (15, 100),
    (20, 150),
    (25, 200),
    (30, 300),
    (40, 400),
]
CASES_PER_CELL = 3
K_SAFETY = 2

UNIT_TIGHTNESS_LEVELS      = [0.7, 1.0]
ARBITRARY_TIGHTNESS_LEVELS = [0.3, 0.7]

# Base seed offsets keep unit and arbitrary streams distinct and make the
# whole bench reproducible.
UNIT_BASE_SEED = 7000
ARB_BASE_SEED  = 8000


def _seed(base_seed, n_nodes, n_parts, tightness):
    return base_seed + n_nodes * 1000 + n_parts + int(tightness * 1_000_000)


def populate(probe_time_limit):
    manifest = {
        "bench": "hybrid_stress",
        "purpose": "compact, large-scale bench to push hybrid solvers past "
                   "tier-1/2; feasibility-probe-only (no ILP optimum).",
        "k_safety": K_SAFETY,
        "cases_per_cell": CASES_PER_CELL,
        "cells": [{"n_nodes": n, "n_partitions": p, "approx_vars": n * p}
                  for n, p in CELLS],
        "unit_tightness_levels": UNIT_TIGHTNESS_LEVELS,
        "arbitrary_tightness_levels": ARBITRARY_TIGHTNESS_LEVELS,
        "feasibility_only": True,
        "probe_time_limit_s": probe_time_limit,
        "size_range_arbitrary": [5, 20],
        "req_range": [0, 10],
        "cost_range": [1, 10],
        "seeds": {"unit_base": UNIT_BASE_SEED, "arbitrary_base": ARB_BASE_SEED},
        "generated_files": {"unit_partition": [], "arbitrary_partition": []},
        "generation_seconds": {},
    }

    unit_root = BENCH_ROOT / "unit_partition"
    arb_root  = BENCH_ROOT / "arbitrary_partition"

    grand_total = 0
    t_start = time.time()

    for n_nodes, n_parts in CELLS:
        cell = f"n{n_nodes}_p{n_parts}"
        print(f"\n=== cell {cell}  (~{n_nodes * n_parts} vars) ===")

        # ---- unit-partition ----
        for tightness in UNIT_TIGHTNESS_LEVELS:
            out = unit_root / cell / f"t{int(tightness * 100)}"
            seed = _seed(UNIT_BASE_SEED, n_nodes, n_parts, tightness)
            t0 = time.time()
            paths = generate_unit_batch(
                n_nodes, n_parts, CASES_PER_CELL, out,
                k_safety=K_SAFETY, base_seed=seed, tightness=tightness,
                feasibility_only=True, probe_time_limit=probe_time_limit,
            )
            dt = time.time() - t0
            grand_total += len(paths)
            rels = [str(p.relative_to(TEST_BANK)) for p in paths]
            manifest["generated_files"]["unit_partition"].extend(rels)
            manifest["generation_seconds"][f"unit/{cell}/t{int(tightness*100)}"] = round(dt, 2)
            print(f"  [unit  t={tightness:.2f}] {len(paths)} cases "
                  f"in {dt:.1f}s -> {out.relative_to(TEST_BANK)}")

        # ---- arbitrary-partition ----
        for tightness in ARBITRARY_TIGHTNESS_LEVELS:
            out = arb_root / cell / f"t{int(tightness * 100)}"
            seed = _seed(ARB_BASE_SEED, n_nodes, n_parts, tightness)
            t0 = time.time()
            paths = generate_batch(
                n_nodes, n_parts, CASES_PER_CELL, out,
                k_safety=K_SAFETY, base_seed=seed, tightness=tightness,
                feasibility_only=True, probe_time_limit=probe_time_limit,
            )
            dt = time.time() - t0
            grand_total += len(paths)
            rels = [str(p.relative_to(TEST_BANK)) for p in paths]
            manifest["generated_files"]["arbitrary_partition"].extend(rels)
            manifest["generation_seconds"][f"arb/{cell}/t{int(tightness*100)}"] = round(dt, 2)
            print(f"  [arb   t={tightness:.2f}] {len(paths)} cases "
                  f"in {dt:.1f}s -> {out.relative_to(TEST_BANK)}")

    manifest["total_files"] = grand_total
    manifest["total_generation_seconds"] = round(time.time() - t_start, 2)

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    (BENCH_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=4))

    print(f"\n=== Done: {grand_total} files in "
          f"{manifest['total_generation_seconds']}s ===")
    print(f"Manifest: {(BENCH_ROOT / 'manifest.json').relative_to(TEST_BANK.parent)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-time-limit",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="CBC wall-clock cap (seconds) for the feasibility probe on "
             "each instance.  Default 60.",
    )
    args = parser.parse_args()
    populate(args.probe_time_limit)


if __name__ == "__main__":
    main()
