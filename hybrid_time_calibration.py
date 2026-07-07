"""
Calibrate the Leap-hybrid ``time_limit`` for the hybrid_stress bench.

At tier-1/2 scale the Leap service's per-problem *minimum* time limit
dominates, so ``time_limit`` is effectively a no-op there.  The
hybrid_stress bench (n15xp100 ~1.5k vars .. n40xp400 ~16k vars) is the
first regime large enough that (a) the service floor itself rises with
problem size and (b) giving the hybrid heuristic more time actually
changes the answer.  This script finds, per representative cell, the
point past which more time stops helping -- the *knee* -- so you can set
``--time-limit`` (or a size-dependent schedule) deliberately instead of
guessing.

There is no ILP optimum on this bench (a full optimize is intractable at
this scale), so calibration uses two self-contained signals rather than
an optimality gap:

  * best-feasible-cost convergence -- the lowest *feasible* cost found at
    each time limit; when it stops dropping, you have found the knee.
  * validity rate -- the fraction of repeats that returned a feasible
    sample at all.  At large scale the floor time may not reliably reach
    feasibility, which is itself a reason to raise the limit.

Quota model
-----------
A Leap hybrid submission bills roughly the ``time_limit`` you request
(with the service floor as a minimum).  Total spend therefore ~=
sum over (cell x time_limit x repeat) of the requested time_limit.  The
script prints this estimate up front and -- unless ``--yes`` -- waits for
confirmation before spending anything.  Querying ``min_time_limit`` does
NOT consume solver-access time (it is metadata), though it does need a
configured Leap client.

Typical use
-----------
    # 1. Free-ish: just learn the service floor per cell (no sampling).
    python hybrid_time_calibration.py --floor-only

    # 2. Plan the spend without submitting anything.
    python hybrid_time_calibration.py --dry-run

    # 3. Run the calibration sweep (S2 / unit, default 3 cells).
    python hybrid_time_calibration.py --yes

    # 4. Arbitrary formulation, custom grid, more repeats.
    python hybrid_time_calibration.py --formulation arbitrary \
        --multipliers 1,3,8 --repeats 5 --yes

Requirements:
    - dwave-system  (pip install dwave-system)
    - Valid LEAP API token (``dwave setup`` or DWAVE_API_TOKEN)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CELLS = ["n15_p100", "n25_p200", "n40_p400"]
DEFAULT_MULTIPLIERS = [1.0, 2.0, 5.0, 10.0]
CELL_RE = re.compile(r"^n(\d+)_p(\d+)$")
TIGHT_RE = re.compile(r"^t(\d+)$")

# Relative best-feasible-cost improvement (vs the previous, shorter time
# limit) below which extra time is judged "not worth it".  The knee is
# the first time limit whose improvement over its predecessor falls below
# this.
DEFAULT_KNEE_THRESHOLD = 0.01  # 1 %


# ---------------------------------------------------------------------------
# Pure helpers (no dwave / no token needed -- unit-testable)
# ---------------------------------------------------------------------------

def parse_cell(name: str) -> tuple[int, int]:
    m = CELL_RE.match(name)
    if not m:
        raise ValueError(f"bad cell name {name!r} (expected nN_pP)")
    return int(m.group(1)), int(m.group(2))


def _tightness_dirs(cell_dir: Path) -> list[Path]:
    """Tightness leaves under a cell dir, sorted by numeric tightness."""
    out = []
    for d in cell_dir.iterdir():
        if d.is_dir() and TIGHT_RE.match(d.name):
            out.append(d)
    return sorted(out, key=lambda d: int(TIGHT_RE.match(d.name).group(1)))


def select_calibration_cases(
    bench_root: Path,
    cells: list[str],
    instance: int,
    tightness: int | None,
) -> list[dict]:
    """
    Resolve one test-case file per requested cell.

    For each cell, pick a tightness leaf (the tightest available by
    default -- the storage constraint binds hardest there -- or the one
    named by ``tightness``), then the ``*_{instance}.json`` file in it.

    Returns a list of dicts: {cell, n, p, tightness, path}.
    """
    cases: list[dict] = []
    for cell in cells:
        n, p = parse_cell(cell)
        cell_dir = bench_root / cell
        if not cell_dir.is_dir():
            raise FileNotFoundError(f"cell dir not found: {cell_dir}")

        tdirs = _tightness_dirs(cell_dir)
        if not tdirs:
            raise FileNotFoundError(f"no tightness leaves under {cell_dir}")

        if tightness is not None:
            chosen = next(
                (d for d in tdirs
                 if int(TIGHT_RE.match(d.name).group(1)) == tightness),
                None,
            )
            if chosen is None:
                avail = [d.name for d in tdirs]
                raise FileNotFoundError(
                    f"tightness t{tightness} not found under {cell_dir} "
                    f"(have {avail})"
                )
        else:
            chosen = tdirs[-1]  # tightest

        matches = sorted(chosen.glob(f"*_{instance}.json"))
        if not matches:
            raise FileNotFoundError(
                f"no instance file *_{instance}.json in {chosen}"
            )
        cases.append({
            "cell": cell,
            "n": n,
            "p": p,
            "tightness": int(TIGHT_RE.match(chosen.name).group(1)),
            "path": matches[0],
        })
    return cases


def build_time_grid(
    floor: float,
    multipliers: list[float],
    abs_limits: list[float] | None,
) -> list[float]:
    """
    Time limits (seconds) to sweep for a single cell.

    With ``abs_limits`` set, use those verbatim but clamp anything below
    the service floor up to the floor (the service would do this anyway).
    Otherwise use ``multiplier * floor``.  The result is sorted and
    de-duplicated (rounded to 3 dp).
    """
    if abs_limits:
        raw = [max(floor, t) for t in abs_limits]
    else:
        raw = [m * floor for m in multipliers]
    seen = sorted({round(t, 3) for t in raw})
    return seen


def estimate_quota_seconds(per_cell_grids: dict[str, list[float]],
                           repeats: int) -> float:
    """Rough Leap solver-access spend: sum of all requested time limits."""
    return repeats * sum(sum(g) for g in per_cell_grids.values())


def aggregate(records: list[dict]) -> dict:
    """
    Collapse raw per-submission records into per-(cell, time_limit) stats.

    Each record: {cell, time_limit, repeat, cost, valid, charge_time_us,
                  error}.  Returns {cell: {time_limit: summary}} where
    summary has best_feasible_cost, mean_feasible_cost, validity_rate,
    mean_charge_s, n_repeats, n_errors.
    """
    by_cell: dict[str, dict[float, list[dict]]] = {}
    for r in records:
        by_cell.setdefault(r["cell"], {}).setdefault(r["time_limit"], []).append(r)

    out: dict = {}
    for cell, by_tl in by_cell.items():
        out[cell] = {}
        for tl, recs in sorted(by_tl.items()):
            feas_costs = [r["cost"] for r in recs
                          if r["valid"] and r["cost"] is not None]
            charges = [r["charge_time_us"] for r in recs
                       if r.get("charge_time_us") is not None]
            n_err = sum(1 for r in recs if r.get("error"))
            out[cell][tl] = {
                "best_feasible_cost": min(feas_costs) if feas_costs else None,
                "mean_feasible_cost": (round(statistics.mean(feas_costs), 2)
                                       if feas_costs else None),
                "validity_rate": round(
                    sum(1 for r in recs if r["valid"]) / len(recs), 3),
                "mean_charge_s": (round(statistics.mean(charges) / 1e6, 3)
                                  if charges else None),
                "n_repeats": len(recs),
                "n_errors": n_err,
            }
    return out


def find_knee(cell_summary: dict, threshold: float) -> float | None:
    """
    First time limit whose best-feasible-cost improvement over the
    previous (shorter) limit falls below ``threshold`` (relative).

    Returns None if cost never plateaus within the swept range (i.e. more
    time was still helping at the largest limit) or if no feasible cost
    was ever found.
    """
    items = sorted(cell_summary.items())  # (tl, summary) by tl asc
    prev_cost = None
    prev_tl = None
    for tl, s in items:
        cost = s["best_feasible_cost"]
        if cost is None:
            continue
        if prev_cost is not None and prev_cost > 0:
            improvement = (prev_cost - cost) / prev_cost
            if improvement < threshold:
                # Plateau reached at the *previous* (cheaper) limit.
                return prev_tl
        prev_cost = cost
        prev_tl = tl
    return None  # still improving at the top of the range, or no feasible cost


# ---------------------------------------------------------------------------
# D-Wave-dependent execution (imports kept local so the helpers above are
# importable / testable without dwave-system or a Leap token)
# ---------------------------------------------------------------------------

def _solver_class(which: str):
    if which == "s1":
        from solvers.hybrid_solvers.SQA_H import SQAHybridSolver
        return SQAHybridSolver, "SQA_H"
    from solvers.hybrid_solvers.SQA_SF_H import SQASFHybridSolver
    return SQASFHybridSolver, "SQA_SF_H"


def _load_inputs(path: Path):
    from util.test_generation.json_to_dict import json_to_test_case
    return json_to_test_case(str(path))


def _score(inputs, selected):
    from util.calculate_solution_cost import (
        calculate_solution_cost,
        is_valid_solution,
    )
    nodes, partitions, k_safety, requests, comm_costs = inputs
    flat = selected.sample if hasattr(selected, "sample") else selected
    cost = calculate_solution_cost(
        nodes, partitions, k_safety, requests, comm_costs, flat)
    valid = is_valid_solution(
        nodes, partitions, k_safety, requests, comm_costs, flat)
    return cost, valid


def _build_sampler(solver_name):
    from dwave.system import LeapHybridSampler
    return LeapHybridSampler(solver=solver_name) if solver_name \
        else LeapHybridSampler()


def run_calibration(args) -> int:
    solver_cls, solver_label = _solver_class(args.solver)
    formulation_dir = f"{args.formulation}_partition"
    bench_root = (
        (REPO_ROOT / args.bench_root).resolve() if args.bench_root
        else REPO_ROOT / "test_bank" / "hybrid_stress" / formulation_dir
    )
    if not bench_root.exists():
        print(f"ERROR: bench root not found: {bench_root}", file=sys.stderr)
        return 1

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    multipliers = [float(x) for x in args.multipliers.split(",") if x.strip()]
    abs_limits = ([float(x) for x in args.abs_time_limits.split(",")]
                  if args.abs_time_limits else None)

    cases = select_calibration_cases(
        bench_root, cells, args.instance, args.tightness)

    print(f"Formulation : {args.formulation}  (solver {solver_label})")
    print(f"Bench root  : {bench_root.relative_to(REPO_ROOT)}")
    print(f"Cells       : {[c['cell'] for c in cases]}")
    print(f"Instance    : *_{args.instance}.json"
          f"   tightness: {'tightest' if args.tightness is None else 't'+str(args.tightness)}")
    print(f"Repeats     : {args.repeats}")
    print(f"Selection   : {args.selection_policy}\n")

    # --- Build BQMs and gather per-cell metadata (no sampling yet) ---
    prepared = []
    for c in cases:
        inputs = _load_inputs(c["path"])
        solver = solver_cls(*inputs, selection_policy=args.selection_policy)
        bqm = solver.build_bqm()
        prepared.append({
            **c,
            "inputs": inputs,
            "bqm": bqm,
            "bqm_vars": len(bqm.variables),
            "bqm_interactions": len(bqm.quadratic),
        })
        print(f"  {c['cell']:10s} t{c['tightness']:<3d} "
              f"BQM: {len(bqm.variables):6d} vars  "
              f"{len(bqm.quadratic):8d} interactions")

    # --- Service floor per cell (needs a client, but no solver-access time) ---
    sampler = None
    if not args.dry_run:
        sampler = _build_sampler(args.solver_name)

    per_cell_grids: dict[str, list[float]] = {}
    print("\nService minimum time_limit per cell:")
    for pc in prepared:
        if sampler is not None:
            floor = float(sampler.min_time_limit(pc["bqm"]))
        else:
            floor = float(args.assumed_floor)  # dry-run estimate only
        pc["floor"] = floor
        grid = build_time_grid(floor, multipliers, abs_limits)
        per_cell_grids[pc["cell"]] = grid
        src = "queried" if sampler is not None else f"ASSUMED={floor}"
        print(f"  {pc['cell']:10s} floor={floor:7.3f}s ({src})  "
              f"grid={grid}")

    if args.floor_only:
        print("\n--floor-only: stopping before any sampling.")
        return 0

    est = estimate_quota_seconds(per_cell_grids, args.repeats)
    print(f"\nEstimated Leap solver-access spend: ~{est:.1f}s "
          f"({len(prepared)} cells x grid x {args.repeats} repeats).")

    if args.dry_run:
        print("--dry-run: no submissions made. Re-run without --dry-run "
              "(floors will be queried for real) to execute.")
        return 0

    if args.max_budget_seconds is not None and est > args.max_budget_seconds:
        print(f"\nERROR: estimated spend ~{est:.0f}s ({est/60:.1f} min) "
              f"exceeds --max-budget-seconds {args.max_budget_seconds:.0f}s "
              f"({args.max_budget_seconds/60:.1f} min). Aborting before any "
              f"submission.", file=sys.stderr)
        print("Reduce --cells / --multipliers / --repeats, or raise the cap.",
              file=sys.stderr)
        return 2

    if not args.yes:
        if not sys.stdin.isatty():
            print("Refusing to spend quota non-interactively without --yes.",
                  file=sys.stderr)
            return 2
        try:
            resp = input("Proceed and spend the above quota? [y/N] ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 0

    # --- The sweep ---
    records: list[dict] = []
    sweep_start = time.perf_counter()
    for pc in prepared:
        grid = per_cell_grids[pc["cell"]]
        for tl in grid:
            for rep in range(1, args.repeats + 1):
                rec = {"cell": pc["cell"], "n": pc["n"], "p": pc["p"],
                       "tightness": pc["tightness"], "time_limit": tl,
                       "repeat": rep, "cost": None, "valid": False,
                       "charge_time_us": None, "error": None}
                try:
                    solver = solver_cls(
                        *pc["inputs"], solver_name=args.solver_name,
                        selection_policy=args.selection_policy)
                    _, selected = solver.solve(time_limit=tl)
                    cost, valid = _score(pc["inputs"], selected)
                    timing = solver.hybrid_timing or {}
                    rec["cost"] = cost
                    rec["valid"] = bool(valid)
                    rec["charge_time_us"] = timing.get("charge_time")
                except Exception as e:  # noqa: BLE001 -- log and continue
                    rec["error"] = str(e)
                records.append(rec)
                tag = "OK" if rec["valid"] else ("ERR" if rec["error"] else "infeasible")
                print(f"  {pc['cell']:10s} tl={tl:7.3f}s rep{rep}/{args.repeats} "
                      f"-> cost={rec['cost']} [{tag}]")

    summary = aggregate(records)

    # --- Knee detection + report ---
    print("\n" + "=" * 64)
    print("CALIBRATION SUMMARY")
    print("=" * 64)
    knees = {}
    for pc in prepared:
        cell = pc["cell"]
        cs = summary.get(cell, {})
        knee = find_knee(cs, args.knee_threshold)
        knees[cell] = knee
        print(f"\n{cell}  (floor {pc['floor']:.3f}s, "
              f"{pc['bqm_vars']} vars):")
        print(f"  {'time_limit':>11s} {'best_feas':>10s} {'mean_feas':>10s} "
              f"{'valid%':>7s} {'charge_s':>9s}")
        for tl, s in sorted(cs.items()):
            bf = s["best_feasible_cost"]
            mf = s["mean_feasible_cost"]
            ch = s["mean_charge_s"]
            bf_s = "-" if bf is None else f"{bf:.1f}"
            mf_s = "-" if mf is None else f"{mf:.1f}"
            ch_s = "-" if ch is None else f"{ch:.3f}"
            print(f"  {tl:11.3f} {bf_s:>10s} {mf_s:>10s} "
                  f"{s['validity_rate']*100:6.0f}% {ch_s:>9s}")
        if knee is None:
            print(f"  -> no plateau within range: cost was still improving at "
                  f"the largest limit (consider extending --multipliers).")
        else:
            print(f"  -> knee at time_limit ~= {knee:.3f}s "
                  f"(improvement < {args.knee_threshold*100:.0f}% beyond this).")

    # --- Persist ---
    out_dir = (REPO_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"hybrid_time_calibration_{args.formulation}_{solver_label}_{stamp}.json"
    payload = {
        "metadata": {
            "script": "hybrid_time_calibration.py",
            "timestamp": stamp,
            "formulation": args.formulation,
            "solver": solver_label,
            "solver_name": args.solver_name,
            "selection_policy": args.selection_policy,
            "bench_root": str(bench_root.relative_to(REPO_ROOT)),
            "instance": args.instance,
            "tightness": args.tightness,
            "multipliers": multipliers,
            "abs_time_limits": abs_limits,
            "repeats": args.repeats,
            "knee_threshold": args.knee_threshold,
            "estimated_quota_seconds": round(est, 1),
            "actual_wall_seconds": round(time.perf_counter() - sweep_start, 1),
        },
        "cells": {pc["cell"]: {"n": pc["n"], "p": pc["p"],
                               "tightness": pc["tightness"],
                               "floor_seconds": pc["floor"],
                               "bqm_variables": pc["bqm_vars"],
                               "bqm_interactions": pc["bqm_interactions"],
                               "time_grid": per_cell_grids[pc["cell"]],
                               "knee_seconds": knees[pc["cell"]]}
                  for pc in prepared},
        "summary": {cell: {str(tl): s for tl, s in by_tl.items()}
                    for cell, by_tl in summary.items()},
        "raw_records": records,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nKnees: " + ", ".join(
        f"{c}={'none' if k is None else f'{k:.3f}s'}" for c, k in knees.items()))
    print(f"Saved: {out_path.relative_to(REPO_ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Calibrate Leap-hybrid time_limit on the hybrid_stress "
                    "bench by sweeping multiples of the service floor and "
                    "locating the cost-vs-time knee.")
    p.add_argument("--formulation", choices=["unit", "arbitrary"],
                   default="unit",
                   help="Which stress bench to calibrate against "
                        "(default: unit).")
    p.add_argument("--solver", choices=["s2", "s1"], default="s2",
                   help="s2 = SQA_SF_H (slack-free, the hybrid default); "
                        "s1 = SQA_H (slack-variable).  Default s2.")
    p.add_argument("--bench-root", default=None, metavar="PATH",
                   help="Override the bench root (default "
                        "test_bank/hybrid_stress/<formulation>_partition). "
                        "Relative to repo root.")
    p.add_argument("--cells", default=",".join(DEFAULT_CELLS),
                   help="Comma-separated cell names to calibrate "
                        f"(default {','.join(DEFAULT_CELLS)}).")
    p.add_argument("--instance", type=int, default=1,
                   help="Which instance index per cell (the *_{i}.json "
                        "file). Default 1.")
    p.add_argument("--tightness", type=int, default=None,
                   help="Tightness level (e.g. 70) to pick per cell. "
                        "Default: the tightest available for that cell.")
    p.add_argument("--multipliers", default=",".join(str(m) for m in DEFAULT_MULTIPLIERS),
                   help="Comma-separated multipliers of the service floor "
                        f"to sweep (default {','.join(str(m) for m in DEFAULT_MULTIPLIERS)}).")
    p.add_argument("--abs-time-limits", default=None,
                   help="Comma-separated absolute time limits in seconds, "
                        "overriding --multipliers (values below the floor "
                        "are clamped up).")
    p.add_argument("--repeats", type=int, default=3,
                   help="Submissions per (cell, time_limit) to capture "
                        "variance. Default 3.")
    p.add_argument("--selection-policy", default="best_feasible",
                   help="Sample-selection policy passed to the hybrid "
                        "solver. Default best_feasible.")
    p.add_argument("--solver-name", default=None,
                   help="Explicit Leap hybrid solver id (default: client's "
                        "default hybrid BQM solver).")
    p.add_argument("--knee-threshold", type=float,
                   default=DEFAULT_KNEE_THRESHOLD,
                   help="Relative best-cost improvement below which extra "
                        "time is deemed not worth it (default 0.01 = 1%%).")
    p.add_argument("--out-dir", default="result_bank/calibration",
                   help="Directory for the calibration result JSON "
                        "(default result_bank/calibration).")
    p.add_argument("--floor-only", action="store_true",
                   help="Query and print the service min_time_limit per "
                        "cell, then stop (no sampling, no quota).")
    p.add_argument("--dry-run", action="store_true",
                   help="Build BQMs and print the planned grid + estimated "
                        "quota without contacting Leap or sampling. Uses "
                        "--assumed-floor for the grid.")
    p.add_argument("--assumed-floor", type=float, default=3.0,
                   help="Floor (seconds) assumed for --dry-run grid/quota "
                        "estimates only. Default 3.0.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the interactive quota confirmation.")
    p.add_argument("--max-budget-seconds", type=float, default=None,
                   metavar="SECONDS",
                   help="Hard cap on estimated Leap solver-access spend. If "
                        "the (real-floor) estimate exceeds this, abort before "
                        "any submission. Unset = no cap.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run_calibration(args)


if __name__ == "__main__":
    sys.exit(main())
