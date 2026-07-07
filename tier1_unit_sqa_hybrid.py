"""
Tier-1 sweep with S1 + S2 *hybrid* encodings.

Hybrid analog of ``tier1_unit_sqa_hw.py``.  Same problem set and same
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
    python tier1_unit_sqa_hybrid.py                          # subset (36 cases), interactive
    python tier1_unit_sqa_hybrid.py --no-pause               # subset, unattended
    python tier1_unit_sqa_hybrid.py --full --no-pause        # full tier-1 (180 cases)
    python tier1_unit_sqa_hybrid.py --time-limit 5 --no-pause  # force 5 s per submission

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
from util.experiment_execution.hybrid_budget import (
    check_budget,
    estimate_experiment_seconds,
)
from util.experiment_execution.run_experiment import run_experiment
from util.experiment_execution.run_unit_partition_experiment import (
    _with_selection_policy,
)
from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    POLICY_LOWEST_ENERGY,
    VALID_POLICIES,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parent
TIER1_DIR  = REPO_ROOT / "test_bank" / "unit_partition" / "tier1"
# Hybrid results are split by selection_policy to mirror the QPU/sim
# banks: best_feasible runs land in hybrid_results_feasible/, the legacy
# lowest_energy runs in the original hybrid_results/ tree.
HYBRID_DIR          = REPO_ROOT / "result_bank" / "hybrid_results"
HYBRID_DIR_FEASIBLE = REPO_ROOT / "result_bank" / "hybrid_results_feasible"


def _hybrid_result_dir(selection_policy):
    if selection_policy == POLICY_LOWEST_ENERGY:
        return HYBRID_DIR
    return HYBRID_DIR_FEASIBLE

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


def _bench_label(bench_root_arg) -> str:
    """
    Result-file prefix label for the bench being run.

    Default (no --bench-root) keeps the historical 'Tier1' prefix.  For a
    custom root, label by the bench directory: if the root is a formulation
    folder (unit_partition / arbitrary_partition) use its parent name
    (e.g. hybrid_stress -> 'HybridStress'), so unit and arbitrary stress
    runs share the bench label and stay distinct from tier sweeps.
    """
    if not bench_root_arg:
        return "Tier1"
    p = Path(bench_root_arg)
    name = p.parent.name if p.name in ("unit_partition", "arbitrary_partition") else p.name
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", name).title().replace(" ", "")
    return cleaned or "Bench"


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
    parser.add_argument(
        "--legacy-lowest-energy",
        action="store_true",
        help="Use the legacy 'lowest_energy' selection policy (return the "
             "lowest-energy sample regardless of feasibility) and write to "
             "result_bank/hybrid_results/.  Default is 'best_feasible', "
             "matching the QPU/sim banks, which writes to "
             "result_bank/hybrid_results_feasible/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the most recent result file for this mode instead "
             "of starting a fresh one: cases already recorded are skipped "
             "and their hybrid submissions are not repeated.  Use this to "
             "pick a sweep back up after an interruption without "
             "re-spending Leap quota.",
    )
    parser.add_argument(
        "--with-s1",
        action="store_true",
        help="Also run S1 (SQA_H, slack-variable encoding) alongside S2.  "
             "Default is S2 only: on the hybrid path S1's slack variables "
             "no longer cost an embedding, so S1 and S2 tend to converge, "
             "and dropping S1 halves the Leap quota per case.",
    )
    parser.add_argument(
        "--s1-only",
        action="store_true",
        help="Run S1 (SQA_H, slack-variable encoding) only -- the inverse "
             "of the default, which runs S2 (SQA_SF_H, slack-free) only.  "
             "ILP stays as the optimality-gap baseline.  Mutually exclusive "
             "with --with-s1.",
    )
    parser.add_argument(
        "--bench-root",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory containing n{N}_p{P}/t{TT}/ leaves to run instead "
             "of the default unit-partition tier-1 bank.  Relative paths "
             "resolve against the repo root.  For the stress bench pass "
             "test_bank/hybrid_stress/unit_partition.",
    )
    parser.add_argument(
        "--no-ilp",
        action="store_true",
        help="Skip the ILP optimality-gap baseline.  Needed at "
             "hybrid-stress scale, where a full ILP optimize is "
             "intractable: solvers are then compared by cost/validity only "
             "and optimality_gap is reported as null.",
    )
    parser.add_argument(
        "--max-budget-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Hard cap on estimated Leap solver-access spend.  Before any "
             "submission the run is costed (one submission per case per "
             "hybrid solver, billed at --time-limit or, if unset, each "
             "BQM's service minimum) and aborted if the estimate exceeds "
             "this cap.  Unset = no cap.",
    )
    args = parser.parse_args()

    if args.s1_only and args.with_s1:
        parser.error("--s1-only and --with-s1 are mutually exclusive.")

    selection_policy = (
        POLICY_LOWEST_ENERGY if args.legacy_lowest_energy
        else POLICY_BEST_FEASIBLE
    )
    if selection_policy not in VALID_POLICIES:  # belt-and-braces
        raise RuntimeError(f"bad selection_policy: {selection_policy!r}")
    output_dir = _hybrid_result_dir(selection_policy)

    bench_root = (
        (REPO_ROOT / args.bench_root).resolve() if args.bench_root
        else TIER1_DIR
    )
    if not bench_root.exists():
        print(f"ERROR: bench root not found: {bench_root}", file=sys.stderr)
        return 1

    selector = select_all_instances if args.full else select_one_per_config
    cases = selector(bench_root)
    mode_label = "FULL (all instances)" if args.full else "subset (one per config)"
    solver_tag = "S1" if args.s1_only else ("S1_S2" if args.with_s1 else "S2")
    bench_label = _bench_label(args.bench_root)
    file_prefix = f"{bench_label}{'Full' if args.full else 'Final'}_{solver_tag}_HYB"

    tl_label = (
        f"{args.time_limit} s" if args.time_limit is not None
        else "service-default (per-problem minimum)"
    )

    print(f"Mode: {mode_label}")
    print(f"Bench root : {bench_root.relative_to(REPO_ROOT)}")
    print(f"ILP baseline: {'OFF (--no-ilp)' if args.no_ilp else 'ON'}")
    print(f"Selected {len(cases)} cases.")
    if not args.full:
        for c in cases:
            print(f"  {c.relative_to(REPO_ROOT)}")
    print(f"\ntime_limit : {tl_label}")
    print(f"selection_policy: {selection_policy}")
    print(f"Output dir : {output_dir.relative_to(REPO_ROOT)}")
    print(f"file_prefix: {file_prefix}")
    print(f"Interactive pause:       "
          f"{'OFF (--no-pause)' if args.no_pause else 'ON (default)'}\n")

    # Default to S2 (SQA_SF_H) only.  On the hybrid path S1's slack-
    # variable encoding no longer costs an embedding, so S1 and S2 tend
    # to converge; running S2 alone halves the Leap quota per case.  Pass
    # --with-s1 to run both for the S1-vs-S2 comparison.  ILP is always
    # the optimality-gap baseline.  selection_policy is injected into the
    # hybrid solvers' kwargs so their reported sample matches the QPU/sim
    # banks.
    registry = []
    if not args.no_ilp:
        registry.append({"name": "ILP", "class": ILPSolver, "type": "ilp"})
    if args.with_s1 or args.s1_only:
        registry.append(
            {"name": "SQA_H", "class": SQAHybridSolver, "type": "hybrid"}
        )
    if not args.s1_only:
        registry.append(
            {"name": "SQA_SF_H", "class": SQASFHybridSolver, "type": "hybrid"}
        )
    registry = _with_selection_policy(registry, selection_policy)
    hyb_label = ("S1 (SQA_H) only" if args.s1_only
                 else "S1 (SQA_H) + S2 (SQA_SF_H)" if args.with_s1
                 else "S2 (SQA_SF_H) only")

    # -----------------------------------------------------------------------
    # Budget pre-flight.  Cost the run before submitting anything: one
    # hybrid submission per case per hybrid solver, billed at --time-limit
    # (or each BQM's service minimum when unset).  Querying the floor uses
    # a Leap client but spends no solver-access time.  Always print the
    # estimate; abort if it exceeds --max-budget-seconds.
    # -----------------------------------------------------------------------
    hybrid_registry = [s for s in registry if s["type"] == "hybrid"]
    est_seconds, _ = estimate_experiment_seconds(
        cases, hybrid_registry, args.time_limit,
    )
    ok, msg = check_budget(est_seconds, args.max_budget_seconds)
    print(msg)
    if not ok:
        return 2
    print()

    output_path = run_experiment(
        test_case_paths=cases,
        solver_registry=registry,
        output_dir=output_dir,
        file_prefix=file_prefix,
        time_limit=args.time_limit,
        on_case_complete=_make_per_case_handler(pause=not args.no_pause),
        resume=args.resume,
        note=(
            f"tier1_unit_sqa_hybrid.py [{mode_label}] -- "
            f"{len(cases)} cases on Leap hybrid solvers: ILP baseline + "
            f"{hyb_label}. time_limit={tl_label}.  "
            f"selection_policy={selection_policy}."
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
