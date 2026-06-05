"""
Tier-1 sweep with S1 + S2 *hybrid* encodings.

Hybrid analog of ``tier1_subset_sqa_hw.py``.  Same problem set and same
S1-vs-S2 comparison, but each BQM is submitted to a D-Wave Leap *hybrid*
BQM solver (``LeapHybridSampler``) rather than to a bare QPU.

What this runs
--------------
Tier 1 has 12 (n, p) combinations crossed with 3 tightness levels =
36 configurations, each with 5 instance JSONs (180 total).  This
script supports two case-selection modes:

  * Default (subset):       one instance per (n, p, tightness) leaf
                            (the ``_1.json`` file) -- 36 cases.
  * ``--full``:             every instance in every config -- 180
                            cases.  Within-config variance across the 5
                            instances gives the error bars on the
                            headline cost / validity / gap numbers.

Solvers in the registry:
  * ILP        -- local CBC reference, optimality-gap baseline.
  * SQA_H     (S1) -- slack-variable encoding on a Leap hybrid solver.
  * SQA_SF_H  (S2) -- slack-free unbalanced-penalty encoding, hybrid.

Both hybrid encodings run on every case so the S1-vs-S2 comparison is
recoverable at full scale with instance-level statistics.

Why no anneal / num_reads / chain_strength knobs
------------------------------------------------
The pure-QPU script tunes ``annealing_time``, ``num_reads`` and
``chain_strength`` because those govern a *physical* anneal and its
minor-embedding onto a sparse chip.  A Leap hybrid solver hides all of
that: it runs a classical heuristic alongside its own internal QPU
calls and chooses embedding, reads and anneal schedule itself.  The
only client-side knob is ``time_limit`` (seconds) -- how long the
hybrid heuristic is allowed to run.  Leaving it unset (the default
here) lets the service pick the minimum time limit appropriate for the
problem size, which for tier-1-scale instances is already its floor
(~3 s for the BQM solver).  Pass ``--time-limit`` to override.

Leap footprint
--------------
Two hybrid submissions per case (one for S1, one for S2).  Hybrid
solvers bill a minimum charge time per call (a few seconds) regardless
of problem size, so wall clock is dominated by that floor times the
number of submissions, not by problem difficulty:

  * subset:  36 cases * 2 submissions = 72 submissions.
  * --full: 180 cases * 2 submissions = 360 submissions.

Mind your Leap solver-access quota when running ``--full``.

Usage
-----
    cd /path/to/QuantumClean
    python tier1_subset_sqa_hybrid.py                          # subset (36 cases), interactive
    python tier1_subset_sqa_hybrid.py --no-pause               # subset, unattended
    python tier1_subset_sqa_hybrid.py --full --no-pause        # full tier-1 (180 cases)
    python tier1_subset_sqa_hybrid.py --time-limit 5 --no-pause  # force 5 s per submission

Requirements:
    - dwave-system  (pip install dwave-system)
    - Valid LEAP API token configured via ``dwave setup`` or DWAVE_API_TOKEN env var
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from solvers.ILP import ILPSolver
from solvers.hybrid_solvers.SQA_H import SQAHybridSolver
from solvers.hybrid_solvers.SQA_SF_H import SQASFHybridSolver
from util.experiment_execution.run_experiment import run_experiment


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parent
TIER1_DIR  = REPO_ROOT / "test_bank" / "unit_partition" / "tier1"
OUTPUT_DIR = REPO_ROOT / "result_bank" / "hybrid_results"

# A "tier-1 configuration" is a (n, p, tightness) triple, which
# corresponds 1:1 to a leaf directory under tier1/ (e.g. n3_p4/t30/).
# We take the lexicographically-first JSON in each leaf, which is the
# ``_1.json`` instance by the bank's naming convention.
INSTANCE_PER_CONFIG_GLOB = "*_1.json"


# ---------------------------------------------------------------------------
# Per-case interactive hook
# ---------------------------------------------------------------------------

def _print_hybrid_report(case_key, entry):
    """Per-case hybrid-time report -- runs in both interactive and unattended modes."""
    run_total_us = 0
    breakdown = []
    for solver_name, r in entry["solvers"].items():
        run_us = r.get("run_time_us")
        if run_us is None:
            continue
        run_total_us += run_us
        breakdown.append((solver_name, run_us, r.get("qpu_access_time_us"),
                          r.get("hybrid_timing", {})))

    print(f"\n  --- hybrid time report for {case_key} ---")
    if not breakdown:
        print("    (no hybrid solvers ran on this case)")
        return
    for name, run_us, qpu_us, timing in breakdown:
        charge_us = timing.get("charge_time") if timing else None
        print(f"    {name:10s} run_time={run_us} us"
              f"  (charge={charge_us}, qpu_access={qpu_us})")
    print(f"    TOTAL run_time across hybrid solvers: {run_total_us} us "
          f"= {run_total_us / 1000:.2f} ms")


def _make_per_case_handler(pause: bool):
    """
    Build the harness's ``on_case_complete`` callback.

    Always prints a per-case hybrid report.  If ``pause=True`` and stdin
    is a TTY, additionally blocks on user input before the next case --
    skipped after the last case and when stdin is not a TTY (would hang
    forever with no way to satisfy the prompt).  A bare ``Ctrl+C`` at
    the prompt propagates as ``KeyboardInterrupt``, which the harness
    lets escape; the result file is already saved through the
    just-completed case so aborting here is safe.
    """
    def handler(case_key, entry, idx, total):
        _print_hybrid_report(case_key, entry)
        if not pause:
            return
        if idx >= total:
            print()  # last case: don't pause
            return
        if not sys.stdin.isatty():
            return  # non-interactive: don't block
        try:
            input(f"  Press Enter to continue to case {idx + 1}/{total} "
                  f"(Ctrl+C to abort)... ")
        except EOFError:
            return
    return handler


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------

def _iter_tightness_dirs(tier_root: Path):
    """Yield every (n_p_dir, tightness_dir) under tier_root, skipping junk."""
    cfg_re   = re.compile(r"^n(\d+)_p(\d+)$")
    tight_re = re.compile(r"^t(\d+)$")
    for cfg_dir in sorted(tier_root.iterdir()):
        if not cfg_dir.is_dir() or not cfg_re.match(cfg_dir.name):
            continue  # skip macOS dup folders (e.g. "n3_p4 2") and stray files
        for tight_dir in sorted(cfg_dir.iterdir()):
            if not tight_dir.is_dir() or not tight_re.match(tight_dir.name):
                continue
            yield cfg_dir, tight_dir


def select_one_per_config(tier_root: Path) -> list[Path]:
    """
    Return one path per (n, p, tightness) configuration in ``tier_root``
    -- the lexicographically-first JSON in each leaf (``_1.json`` by
    bank convention).  36 cases for tier 1.
    """
    cases: list[Path] = []
    for _, tight_dir in _iter_tightness_dirs(tier_root):
        instances = sorted(tight_dir.glob(INSTANCE_PER_CONFIG_GLOB))
        if not instances:
            raise FileNotFoundError(
                f"No instance-1 JSON found in {tight_dir} -- expected "
                f"a file matching {INSTANCE_PER_CONFIG_GLOB}."
            )
        cases.append(instances[0])
    return cases


def select_all_instances(tier_root: Path) -> list[Path]:
    """
    Return every instance JSON in every (n, p, tightness) config.
    180 cases for tier 1 (12 configs * 3 tightnesses * 5 instances).
    """
    cases: list[Path] = []
    for _, tight_dir in _iter_tightness_dirs(tier_root):
        instances = sorted(tight_dir.glob("*.json"))
        if not instances:
            raise FileNotFoundError(
                f"No instance JSONs found in {tight_dir}."
            )
        cases.extend(instances)
    return cases


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tier-1 sweep on D-Wave Leap hybrid solvers: ILP "
                    "baseline + S1 (SQA_H) + S2 (SQA_SF_H).  Hybrid path "
                    "has no anneal/reads/chain knobs; the only tunable is "
                    "--time-limit (seconds).",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Run unattended -- do not block for user input between cases. "
             "The per-case hybrid-time report still prints.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full tier-1 suite (all 5 instances per "
             "(n, p, tightness) config = 180 cases) instead of the "
             "default one-instance-per-config subset (36 cases).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Per-submission hybrid solver run time, in seconds. If "
             "omitted, the Leap service picks the minimum time limit "
             "appropriate for each problem size (its floor at tier-1 "
             "scale). Increasing it gives the hybrid heuristic more "
             "time at higher quota cost.",
    )
    args = parser.parse_args()

    if not TIER1_DIR.exists():
        print(f"ERROR: tier-1 directory not found: {TIER1_DIR}", file=sys.stderr)
        return 1

    selector = select_all_instances if args.full else select_one_per_config
    cases = selector(TIER1_DIR)
    mode_label = "FULL tier-1 (all instances)" if args.full else "subset (one per config)"
    file_prefix = "Tier1Full_S1_S2_HYB" if args.full else "Tier1Final_S1_S2_HYB"

    tl_label = (
        f"{args.time_limit} s" if args.time_limit is not None
        else "service-default (per-problem minimum)"
    )

    print(f"Mode: {mode_label}")
    print(f"Selected {len(cases)} cases.")
    if not args.full:
        for c in cases:
            print(f"  {c.relative_to(REPO_ROOT)}")
    print(f"\ntime_limit : {tl_label}")
    print(f"Output dir : {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print(f"file_prefix: {file_prefix}")
    print(f"Interactive pause:       "
          f"{'OFF (--no-pause)' if args.no_pause else 'ON (default)'}\n")

    # S1 (SQA_H) and S2 (SQA_SF_H) both run on every case so the
    # S1-vs-S2 comparison is recoverable at full scale with
    # instance-level statistics.  ILP retained as the optimality-gap
    # baseline.
    registry = [
        {"name": "ILP",      "class": ILPSolver,          "type": "ilp"},
        {"name": "SQA_H",    "class": SQAHybridSolver,    "type": "hybrid"},
        {"name": "SQA_SF_H", "class": SQASFHybridSolver,  "type": "hybrid"},
    ]

    output_path = run_experiment(
        test_case_paths=cases,
        solver_registry=registry,
        output_dir=OUTPUT_DIR,
        file_prefix=file_prefix,
        time_limit=args.time_limit,
        on_case_complete=_make_per_case_handler(pause=not args.no_pause),
        note=(
            f"tier1_subset_sqa_hybrid.py [{mode_label}] -- "
            f"{len(cases)} cases on Leap hybrid solvers: ILP baseline + "
            f"S1 (SQA_H) + S2 (SQA_SF_H), both encodings for full-scale "
            f"comparison. time_limit={tl_label}."
        ),
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Quick post-run summary per hybrid solver: matched ILP cost / valid
    # but suboptimal / invalid / errored.  Detail lives in the result JSON.
    # -----------------------------------------------------------------------
    payload = json.loads(Path(output_path).read_text())
    hybrid_solver_names = [
        s["name"] for s in registry if s["type"] == "hybrid"
    ]
    total = len(payload["results"])
    print()
    for solver_name in hybrid_solver_names:
        matched = suboptimal = invalid = errored = 0
        for entry in payload["results"].values():
            r = entry["solvers"].get(solver_name)
            if r is None:
                continue
            if r.get("error"):
                errored += 1
                continue
            if not r.get("valid"):
                invalid += 1
                continue
            gap = r.get("optimality_gap_absolute")
            if gap == 0:
                matched += 1
            else:
                suboptimal += 1
        print(f"{solver_name} summary across {total} cases:")
        print(f"  matched ILP cost      : {matched}")
        print(f"  valid but suboptimal  : {suboptimal}")
        print(f"  invalid (constraints) : {invalid}")
        print(f"  errored               : {errored}")
        print()
    print(f"Result file: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
