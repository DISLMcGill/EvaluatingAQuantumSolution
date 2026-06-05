"""
Tier-1 ARBITRARY-PARTITION sweep with S1 + S2 hardware encodings.

Sibling of ``tier1_subset_sqa_hw.py``.  Same principles, same harness,
same solver line-up -- the only structural differences are the test-bank
directory it reads from (``arbitrary_partition`` instead of
``unit_partition``) and the result-file prefix (so its output does not
collide with the unit-partition runs, which land in the same
result_bank/ tree).

In the arbitrary-partition bank every partition ``p`` carries its own
integer ``size_p`` (instead of the unit ``size_p = 1``).  That size
enters the storage-constraint encoding as the coefficient of the
assignment variable ``A_{p,n}`` rather than as a unit count, which
changes the magnitude and density of the BQM's quadratic terms.  Both S1
(slack-variable) and S2 (slack-free unbalanced-penalty) handle this after
the Phase-1/Phase-2 fixes; see the hardware solver module docstrings.

What this runs
--------------
Tier 1 has 12 (n, p) combinations crossed with 3 tightness levels =
36 configurations, each with 5 instance JSONs (180 total) -- the same
shape as the unit-partition tier-1 bank.  Two case-selection modes:

  * Default (subset):       one instance per (n, p, tightness) leaf
                            (the ``*_1.json`` file) -- 36 cases.
                            Used during hyperparameter tuning.
  * ``--full``:             every instance in every config -- 180
                            cases.  Used for final data collection;
                            within-config variance across the 5
                            instances gives the error bars.

Solvers in the registry:
  * ILP        -- local CBC reference, optimality-gap baseline.
  * SQA_HW    (S1) -- slack-variable encoding on D-Wave QPU.
  * SQA_SF_HW (S2) -- slack-free unbalanced-penalty encoding on QPU.

Both QPU encodings run on every case so the S1-vs-S2 comparison is
recoverable at full scale with instance-level statistics.

Hyperparameters -- STARTING POINTS FOR TUNING
---------------------------------------------
The annealing-time and num_reads schedules below are copied from the
unit-partition sweep (tier1_subset_sqa_hw.py) as a starting point.

Chain strength is now FLAT (prefactor=1.5) for both S1 and S2.  The
unit-partition sweep used a graduated S1 schedule to fight chain breaks
that grew with p, but that regime does not apply here: the first
arbitrary tier-1 HW run had cbf <= 0.07 everywhere, so chains were never
the bottleneck -- the zero-feasibility failure was a penalty-balance bug
in S1's BQM, since fixed (k-safety multiplier scaled by max(size_p)**2;
see solvers/simulated_solvers/SQA.py).  That fix also changed the BQM's
couplings, so any pre-fix cbf data is stale.  Torque compensation
auto-scales chain strength to the current BQM, so a flat prefactor is the
right starting point; re-graduate only if a fresh run shows cbf rising.

Expect to re-tune by running small subsets:

    python tier1_arbitrary_sqa_hw.py            # 36-case subset, interactive
    python tier1_arbitrary_sqa_hw.py --no-pause # 36-case subset, unattended

Watch chain_break_fraction (cbf) and feasibility_yield in the result
JSON.  If cbf climbs past ~0.05 on a cell, bump that cell's prefactor in
``schedule_chain_strength_s1`` (S1) or in ``schedule_chain_strength``
(S2) and re-run that subset.

Usage
-----
    cd /path/to/QuantumClean
    python tier1_arbitrary_sqa_hw.py                       # subset (36 cases), interactive
    python tier1_arbitrary_sqa_hw.py --no-pause            # subset (36 cases), unattended
    python tier1_arbitrary_sqa_hw.py --full --no-pause     # full tier-1 (180 cases), unattended
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dwave.embedding.chain_strength import uniform_torque_compensation

from solvers.ILP import ILPSolver
from solvers.quantum_hardware_solvers.SQA_HW import SQAHardwareSolver
from solvers.quantum_hardware_solvers.SQA_SF_HW import SQASFHardwareSolver
from util.experiment_execution.run_experiment import run_experiment
from util.experiment_execution.run_unit_partition_experiment import (
    _with_selection_policy,
    result_dir_for,
)
from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    POLICY_LOWEST_ENERGY,
    VALID_POLICIES,
)


# ---------------------------------------------------------------------------
# Chain-strength helper
# ---------------------------------------------------------------------------

class _TorqueWithPrefactor:
    """
    Callable wrapper around ``uniform_torque_compensation`` with a fixed
    prefactor, suitable to pass as ``chain_strength`` to
    ``EmbeddingComposite.sample``.

    Using ``uniform_torque_compensation`` (the SDK default) rather than
    an absolute number means the chain coupling is calibrated to the
    *BQM's own coupling magnitudes*.  This matters even more in the
    arbitrary-partition case: with non-unit ``size_p`` the storage
    constraint coefficients -- and hence the BQM couplings -- are larger
    than in the unit-partition bank, so an absolute chain strength tuned
    for unit partitions would be the wrong scale here.  Torque
    compensation tracks the scale automatically.

    ``repr()`` returns a readable identity so the result JSON can record
    which schedule was used without serialising a closure.
    """

    def __init__(self, prefactor: float):
        self.prefactor = float(prefactor)

    def __call__(self, bqm, embedding=None):
        return uniform_torque_compensation(bqm, embedding, prefactor=self.prefactor)

    def __repr__(self) -> str:
        return f"uniform_torque_compensation(prefactor={self.prefactor})"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parent
TIER1_DIR  = REPO_ROOT / "test_bank" / "arbitrary_partition" / "tier1"
# Output dir is resolved per-run from the chosen selection_policy via
# result_dir_for().  NOTE: result_dir_for() keys only on
# (hardware, selection_policy) -- it does NOT separate unit- from
# arbitrary-partition results.  Both land in the same
# result_bank/quantum_hardware_results[_feasible]/ tree, so the
# arbitrary runs are kept distinct purely by ``file_prefix`` below
# (ArbTier1Final_* / ArbTier1Full_* vs the unit Tier1Final_* /
# Tier1Full_*).

ANNEALING_TIME = 50  # microseconds.  Inherited from the unit-partition
                     # sweep as a starting point.  Longer anneals let the
                     # system follow the adiabatic path more closely;
                     # re-validate on arbitrary instances if convergence
                     # (best_num_occurrences) looks poor.


# Per-case ``num_reads`` schedule.  Inherited verbatim from the
# unit-partition sweep -- statistical coverage scales with problem size,
# which is unchanged here (tier 1 still tops out at n=9), so this is the
# most likely of the three schedules to transfer as-is.
def schedule_num_reads(n_nodes: int, n_partitions: int) -> int:
    """
    Per-case num_reads: 500 for small, 1000 for n>=5 or p>=26.

    Inherited from the unit-partition tier-1 sweep.  Coverage (not BQM
    scale) is what this knob controls, so it should transfer to the
    arbitrary-partition bank with little or no change.  Threshold still
    matches schedule_chain_strength_s1.
    """
    return 1000 if (n_nodes >= 5 or n_partitions >= 26) else 500


def schedule_chain_strength(n_nodes: int, n_partitions: int) -> _TorqueWithPrefactor:
    """
    Per-case chain_strength for S2: prefactor=1.5 flat across all sizes.

    Inherited starting point from the unit-partition sweep, where the
    slack-free BQM was small enough that torque compensation reached the
    chain-intact regime at prefactor 1.5 flat.  With non-unit size_p the
    S2 penalty coefficients grow, so this may need revisiting -- watch
    cbf and bump the flat prefactor (or switch to a size-graduated
    schedule like S1's) if chains break.

    S1 needs its own schedule (see ``schedule_chain_strength_s1``)
    because its slack-variable BQM is denser.
    """
    # Argument unused on purpose; kept for signature stability so the
    # harness's chain_strength_fn contract doesn't change.
    del n_nodes, n_partitions
    return _TorqueWithPrefactor(1.5)


def schedule_chain_strength_s1(n_nodes: int, n_partitions: int) -> _TorqueWithPrefactor:
    """
    Per-case chain_strength for S1 (SQA_HW): flat prefactor=1.5.

    This used to be a graduated 1.5 -> 3.0 schedule inherited verbatim
    from the unit-partition sweep, where S1's chains broke badly as p
    grew (cbf up to ~0.51 at n3_p50) and a rising prefactor was the
    response.  That schedule is wrong for the arbitrary-partition case
    for two independent reasons:

      1. Chain breaks were never the bottleneck here.  The first
         arbitrary tier-1 HW run had cbf <= 0.07 across every case
         (mostly ~0.01) yet zero feasible reads -- the failure was the
         storage/k-safety penalty imbalance, not broken chains.  Bumping
         the prefactor up the old schedule just over-provisioned chains
         against a problem that didn't exist.

      2. The S1 BQM has since changed.  The Phase-1 fix scales the
         k-safety multiplier by max(size_p)**2 to restore feasibility
         (see solvers/simulated_solvers/SQA.py), which changes the BQM's
         coupling magnitudes.  Any cbf-vs-prefactor data gathered before
         that fix is stale, so re-graduating off the old numbers would be
         fitting to a model that no longer exists.

    Flat 1.5 (matching S2) is therefore the right starting point: torque
    compensation auto-scales chain strength to the new BQM's couplings,
    so a single prefactor is robust across sizes.  Re-introduce
    graduation only if a fresh HW run on the fixed BQM actually shows cbf
    rising past ~0.05 on the larger cells -- and key it to the measured
    cbf, not to the inherited unit-partition curve.  The function is kept
    as a separate per-solver hook (rather than folding S1 into the
    shared schedule) precisely so that re-graduation stays a one-function
    change here.
    """
    del n_nodes, n_partitions  # flat schedule; args kept for hook signature
    return _TorqueWithPrefactor(1.5)


# ---------------------------------------------------------------------------
# Per-case interactive hook
# ---------------------------------------------------------------------------

def _print_qpu_report(case_key, entry):
    """Per-case QPU-time report -- runs in both interactive and unattended modes."""
    qpu_total_us = 0
    qpu_breakdown = []
    for solver_name, r in entry["solvers"].items():
        access_us = r.get("qpu_access_time_us")
        if access_us is None:
            continue
        qpu_total_us += access_us
        qpu_breakdown.append((solver_name, access_us, r.get("qpu_timing", {})))

    print(f"\n  --- QPU time report for {case_key} ---")
    if not qpu_breakdown:
        print("    (no QPU solvers ran on this case)")
        return
    for name, access_us, timing in qpu_breakdown:
        anneal_us = timing.get("qpu_anneal_time_per_sample") if timing else None
        readout_us = timing.get("qpu_readout_time_per_sample") if timing else None
        prog_us = timing.get("qpu_programming_time") if timing else None
        print(f"    {name:10s} access={access_us} us"
              f"  (programming={prog_us}, anneal/samp={anneal_us}, readout/samp={readout_us})")
    print(f"    TOTAL across QPU solvers: {qpu_total_us} us "
          f"= {qpu_total_us / 1000:.2f} ms")


def _make_per_case_handler(pause: bool):
    """
    Build the harness's ``on_case_complete`` callback.

    Always prints a per-case QPU report.  If ``pause=True`` and stdin
    is a TTY, additionally blocks on user input before the next case --
    skipped after the last case (no point pausing before exit) and
    skipped when stdin is not a TTY (would hang forever with no way to
    satisfy the prompt).  A bare ``Ctrl+C`` at the prompt propagates
    as ``KeyboardInterrupt``, which the harness lets escape; the result
    file is already saved through the just-completed case so aborting
    here is safe.
    """
    def handler(case_key, entry, idx, total):
        _print_qpu_report(case_key, entry)
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

# A "tier-1 configuration" is a (n, p, tightness) triple, which
# corresponds 1:1 to a leaf directory under tier1/ (e.g. n3_p4/t30/).
# We take the lexicographically-first JSON in each leaf, which is the
# ``_1.json`` instance by the bank's naming convention (the
# arbitrary-partition bank uses the same convention as the unit bank,
# e.g. n-3_p-4_1.json).
INSTANCE_PER_CONFIG_GLOB = "*_1.json"


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

    Used for parameter-tuning runs where 5x instances per config would
    waste QPU budget on what's effectively the same data point.
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

    Used for the final full-sweep data collection once the hyperparameter
    config is locked.  Within-config instance variance is what gives the
    error bars on the headline cost / validity numbers, so each instance
    is its own data point and matters for the writeup.
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
        description="Tier-1 ARBITRARY-partition sweep, S1 (SQA_HW) + S2 "
                    "(SQA_SF_HW) on D-Wave QPU.  Hyperparameters inherited "
                    "from the unit-partition sweep as a tuning starting "
                    "point: anneal=50us, reads=500/1000, chain_strength "
                    "chain_strength prefactor=1.5 flat (S1 and S2).",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Run unattended -- do not block for user input between cases. "
             "The per-case QPU-time report still prints.  Required for any "
             "multi-case sweep where a person in the loop for every prompt "
             "is not desirable.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full tier-1 suite (all 5 instances per "
             "(n, p, tightness) config = 180 cases) instead of the "
             "default one-instance-per-config subset (36 cases). "
             "Use for final data collection once hyperparameters are "
             "locked.",
    )
    parser.add_argument(
        "--legacy-lowest-energy",
        action="store_true",
        help="Use the legacy 'lowest_energy' selection policy: return "
             "the lowest-energy sample from each QPU submission, "
             "regardless of whether it satisfies the constraints.  "
             "Results land in the original "
             "result_bank/quantum_hardware_results/ tree.  Default is "
             "'best_feasible', which writes to the parallel "
             "quantum_hardware_results_feasible/ tree.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the most recent result file for this mode instead "
             "of starting a fresh one: cases already recorded are skipped "
             "and their QPU submissions are not repeated.  Use this to "
             "pick a sweep back up after an interruption (network drop, "
             "queue timeout, Ctrl-C) without re-spending QPU budget.",
    )
    args = parser.parse_args()
    selection_policy = (
        POLICY_LOWEST_ENERGY if args.legacy_lowest_energy
        else POLICY_BEST_FEASIBLE
    )
    if selection_policy not in VALID_POLICIES:  # belt-and-braces
        raise RuntimeError(f"bad selection_policy: {selection_policy!r}")
    output_dir = result_dir_for(hardware=True, selection_policy=selection_policy)

    if not TIER1_DIR.exists():
        print(f"ERROR: arbitrary-partition tier-1 directory not found: "
              f"{TIER1_DIR}", file=sys.stderr)
        return 1

    selector = select_all_instances if args.full else select_one_per_config
    cases = selector(TIER1_DIR)
    mode_label = "FULL tier-1 (all instances)" if args.full else "subset (one per config)"
    # Distinct prefix from the unit-partition script so the two sweeps'
    # result files do not collide in the shared result_bank tree.
    file_prefix = "ArbTier1Full_S1_S2" if args.full else "ArbTier1Final_S1_S2"

    print(f"Mode: {mode_label}  (arbitrary partition)")
    print(f"Selected {len(cases)} cases.")
    if not args.full:
        for c in cases:
            print(f"  {c.relative_to(REPO_ROOT)}")
    print(f"\nannealing_time={ANNEALING_TIME} us")
    print(f"num_reads schedule:      {schedule_num_reads.__doc__}")
    print(f"chain_strength (S2): "
          f"{schedule_chain_strength.__doc__.strip().splitlines()[0]}")
    print(f"chain_strength (S1): "
          f"{schedule_chain_strength_s1.__doc__.strip().splitlines()[0]}")
    print(f"selection_policy: {selection_policy}")
    print(f"Output dir : {output_dir.relative_to(REPO_ROOT)}")
    print(f"file_prefix: {file_prefix}")
    print(f"Interactive pause:       "
          f"{'OFF (--no-pause)' if args.no_pause else 'ON (default)'}\n")

    # ILP baseline + S1 + S2.  S1 carries a per-solver chain_strength_fn
    # override; S2 uses the case-level schedule (schedule_chain_strength).
    # The harness honours ``chain_strength_fn`` on a registry entry as an
    # override that applies to that solver only -- so the two encodings
    # get different prefactors at the same problem size without forking
    # the harness loop.
    registry = [
        {"name": "ILP",       "class": ILPSolver,           "type": "ilp"},
        {
            "name": "SQA_HW",
            "class": SQAHardwareSolver,
            "type": "qpu",
            "chain_strength_fn": schedule_chain_strength_s1,
        },
        {"name": "SQA_SF_HW", "class": SQASFHardwareSolver, "type": "qpu"},
    ]
    registry = _with_selection_policy(registry, selection_policy)

    output_path = run_experiment(
        test_case_paths=cases,
        solver_registry=registry,
        output_dir=output_dir,
        file_prefix=file_prefix,
        annealing_time=ANNEALING_TIME,
        num_reads_fn=schedule_num_reads,
        chain_strength_fn=schedule_chain_strength,
        on_case_complete=_make_per_case_handler(pause=not args.no_pause),
        resume=args.resume,
        note=(
            f"tier1_arbitrary_sqa_hw.py [{mode_label}] -- "
            f"{len(cases)} arbitrary-partition cases: ILP baseline + "
            f"S1 + S2 (both encodings for full-scale comparison); "
            f"num_reads 500/1000, annealing_time=50us, chain_strength "
            f"prefactor 1.5 flat (S1 and S2).  Hyperparameters "
            f"inherited from the unit-partition sweep as a tuning "
            f"starting point -- NOT yet validated on arbitrary partitions.  "
            f"selection_policy={selection_policy}."
        ),
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Quick post-run summary: how many QPU rows matched the ILP cost,
    # how many were valid but suboptimal, how many were invalid, and how
    # many errored.  Detail lives in the result JSON.
    # -----------------------------------------------------------------------
    payload = json.loads(Path(output_path).read_text())
    qpu_solver_names = [
        s["name"] for s in registry if s["type"] == "qpu"
    ]
    total = len(payload["results"])
    print()
    for solver_name in qpu_solver_names:
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
