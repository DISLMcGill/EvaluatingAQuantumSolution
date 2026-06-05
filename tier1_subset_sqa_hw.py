"""
Tier-1 sweep with S1 + S2 hardware encodings.

Final locked-in config after a five-run tuning sweep (see the
Tier1Subset_*_HW_*.json files for the tuning history).

What this runs
--------------
Tier 1 has 12 (n, p) combinations crossed with 3 tightness levels =
36 configurations, each with 5 instance JSONs (180 total).  This
script supports two case-selection modes:

  * Default (subset):       one instance per (n, p, tightness) leaf
                            (the ``_1.json`` file) -- 36 cases.
                            Used during hyperparameter tuning.
  * ``--full``:             every instance in every config -- 180
                            cases.  Used for final data collection;
                            within-config variance across the 5
                            instances gives the error bars on the
                            headline cost / validity / gap numbers.

Solvers in the registry:
  * ILP        -- local CBC reference, optimality-gap baseline.
  * SQA_HW    (S1) -- slack-variable encoding on D-Wave QPU.
  * SQA_SF_HW (S2) -- slack-free unbalanced-penalty encoding on QPU.

Both QPU encodings run on every case so the S1-vs-S2 comparison is
recoverable at full scale with instance-level statistics, not just
asserted from the small tuning sample.

Locked hyperparameters
----------------------
* annealing_time   = 50 us  (default 20 us was under-converging)
* num_reads        = 500 small, 1000 for n>=5 or p>=26
* chain_strength   = uniform_torque_compensation(prefactor=1.5) flat
                     -- prefactors > 1.5 measurably crowded the
                     objective term in the BQM's energy budget.

QPU footprint
-------------
Two QPU submissions per case (one for S1, one for S2).
Per-submission QPU access time is roughly 75-300 ms depending on
``num_reads`` and BQM size.

  * subset:  36 cases * 2 submissions = 72 submissions, ~10-30 s pure
             QPU time, tens of minutes wall clock.
  * --full: 180 cases * 2 submissions = 360 submissions, ~1 min pure
             QPU time, 1-3 hours wall clock dominated by queue/network.

Usage
-----
    cd /path/to/QuantumClean
    python tier1_subset_sqa_hw.py                       # subset (36 cases), interactive
    python tier1_subset_sqa_hw.py --no-pause            # subset (36 cases), unattended
    python tier1_subset_sqa_hw.py --full --no-pause     # full tier-1 (180 cases), unattended
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
    *BQM's own coupling magnitudes*.  Absolute values like ``2.0`` are
    meaningless when the BQM's couplings are in the tens or hundreds --
    that's the trap the first revised run fell into.

    ``prefactor=1.0`` matches roughly the SDK default behaviour (the
    SDK's own default uses ``prefactor=1.414``); higher values
    strengthen chains relative to the BQM.  ``repr()`` returns a
    readable identity so the result JSON can record which schedule was
    used without serialising a closure.
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
TIER1_DIR  = REPO_ROOT / "test_bank" / "unit_partition" / "tier1"
# Output dir is resolved per-run from the chosen selection_policy so
# best_feasible runs land in result_bank/quantum_hardware_results_feasible/
# while --legacy-lowest-energy runs continue to write the original
# result_bank/quantum_hardware_results/ tree.  See result_dir_for().

ANNEALING_TIME = 50  # microseconds.  Default is 20 us; we bump to 50
                     # after Tier1Subset_S1_S2_HW_1.json showed S2 with
                     # intact chains still under-sampling the ground
                     # state (best_num_occurrences=1 across all cases).
                     # Longer anneals let the system follow the adiabatic
                     # path more closely -- the only knob left once
                     # encoding and chain quality are sorted.

# Per-case ``num_reads`` schedule.  Rationale: anneal time stays at the
# default 20 us across the sweep (D-Wave's own guidance is to leave it
# there unless there's direct evidence longer anneals help on this
# specific problem class); statistical coverage is what we vary with
# problem size.  Tier 1 only reaches n=9, so a two-tier schedule is
# sufficient.  The chosen numbers are conservative compared with the
# 500/500 simulated default.
def schedule_num_reads(n_nodes: int, n_partitions: int) -> int:
    """
    Per-case num_reads: 500 for small, 1000 for n>=5 or p>=26.

    Rationale (Tier1Subset_S1_S2_HW_1.json): S2 with intact chains
    still had best_num_occurrences=1 across all cases, meaning even
    where the ground state was found it was a one-off in 250-500
    reads.  Doubling the schedule pushes 99%-confidence TTS coverage
    further out; combined with the bump in annealing_time it directly
    addresses the convergence problem now that encoding and chain
    quality are no longer the bottleneck.  Threshold still matches
    schedule_chain_strength.
    """
    return 1000 if (n_nodes >= 5 or n_partitions >= 26) else 500


def schedule_chain_strength(n_nodes: int, n_partitions: int) -> _TorqueWithPrefactor:
    """
    Per-case chain_strength for S2: prefactor=1.5 flat across all sizes.

    Tuning history on the n=3 tier-1 instances:

      run            small / large prefactor    cbf range     headline
      -------------  -----------------------    -----------   ----------------
      r1 S2          1.5 / 2.5                  0 - 0.001     n3_p50 invalid
      r2 S2 longer   1.5 / 2.5                  0 - 0.0013    n3_p50 invalid
      r3 S2 cs15_20  1.5 / 2.0                  0 - 0.0011    n3_p50 VALID,
                                                              cost drops on
                                                              p12/26/50
      Tier1Final_1   1.5 (flat)                 0.0003-0.003  S2 always valid
                                                              (best_feasible),
                                                              cbf well below
                                                              concern; gap
                                                              from ILP large
                                                              on p>=26 due to
                                                              lambda calib

    For S2 the 1.5-flat schedule is well-validated -- the slack-free
    BQM is small enough that torque compensation gets us into the chain-
    intact regime without further tuning.  S1 needs its own schedule
    (see ``schedule_chain_strength_s1``) because its slack-variable BQM
    is 1.5-2x denser and the chain-break fraction explodes at large p.

    NOTE on history: Tier1Subset_SQAHW_2.json tried *absolute*
    chain_strength values of 2.0/3.0, which were catastrophically
    weaker than the BQM's coupling scale (penalty terms in the tens to
    hundreds).  Chain breaks went from cbf=0/0.02/0.06/0.29 on run 1
    to 0.50/0.87/0.94/0.99 on run 2 -- a self-inflicted regression.
    Use prefactors, not absolutes, when you don't know the BQM scale
    in advance.
    """
    # Argument unused on purpose; kept for signature stability so the
    # harness's chain_strength_fn contract doesn't change.
    del n_nodes, n_partitions
    return _TorqueWithPrefactor(1.5)


def schedule_chain_strength_s1(n_nodes: int, n_partitions: int) -> _TorqueWithPrefactor:
    """
    Per-case chain_strength for S1 (SQA_HW): graduated prefactor schedule.

    S1's BQM is denser than S2's because of the slack variables (one
    chunk per node per binary digit of the largest capacity), which
    expands both ``|V(BQM)|`` and ``|E(BQM)|`` and crowds the same
    torque budget across more couplings.  At prefactor=1.5 -- which is
    fine for S2 -- S1's chains start breaking badly as p grows:

      Tier1Final_S1_S2_1.json (this chip, Advantage_system4):
        n=3, p=4    cbf=0.0      OK   (matches ILP)
        n=3, p=12   cbf=0.0253   feasibility_yield=0.0  -- 0/500 reads feasible
        n=3, p=26   cbf=0.0752   feasibility_yield=0.0  -- 0/500 reads feasible
        n=3, p=50   cbf=0.5128   feasibility_yield=0.0  -- catastrophic, 50% of
                                                          samples have broken
                                                          chains

    The n=3, p=50 row is the smoking gun: at cbf=0.51 the logical
    state is destroyed before the energy landscape can be sampled, so
    no read ever satisfies the constraints.  The new feasibility
    filter then has nothing to choose and falls back to the
    lowest-energy infeasible sample -- which is what
    feasibility_fallback=True in the result JSON now reports.

    Schedule design: monotone in (n_nodes, n_partitions), tied to
    where cbf rose past ~0.05 (the conventional concern threshold) in
    the observed data, and extrapolated conservatively to (n=5,7,9)
    cells we haven't measured yet.  The increments are 0.5 per step;
    smaller steps would not move cbf appreciably given how flat the
    torque-vs-prefactor curve is on Advantage_system4 for dense BQMs.

      (n,  p_max)      ->  prefactor
      (3,    4)        ->  1.5    matches S2 baseline; cbf was 0 in r1
      (3,   12)        ->  2.0    cbf=0.025 at 1.5 -- bump small
      (3,   26)        ->  2.5    cbf=0.075 at 1.5 -- past the 0.05 line
      (3,   50)        ->  3.0    cbf=0.51 at 1.5  -- catastrophic, needs a lot
      (n>=5, any)      ->  one tier higher than the n=3 row at the
                          same p, since larger n adds another factor of
                          |N| to the storage-constraint quadratic terms

    If the first run with this schedule shows cbf still >0.10 on any
    case, bump that cell another 0.5 and re-run.  If cbf is now
    <0.001 on cells where S2 sits at <0.01, we may have over-
    corrected; drop the largest cell by 0.5.
    """
    # Decide the prefactor by the "harder of the two" dimension.  Bins
    # are kept coarse so the schedule is easy to reason about; finer
    # bins should only follow more data.
    if n_partitions >= 50 or n_nodes >= 9:
        prefactor = 3.0
    elif n_partitions >= 26 or n_nodes >= 7:
        prefactor = 2.5
    elif n_partitions >= 12 or n_nodes >= 5:
        prefactor = 2.0
    else:
        prefactor = 1.5
    return _TorqueWithPrefactor(prefactor)


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
# ``_1.json`` instance by the bank's naming convention.
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
        description="Tier-1 sweep at 1 instance per configuration, "
                    "S2 (SQA_SF_HW) only.  Final locked-in config: "
                    "anneal=50us, reads=500/1000, chain_strength prefactor=1.5.",
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
        print(f"ERROR: tier-1 directory not found: {TIER1_DIR}", file=sys.stderr)
        return 1

    selector = select_all_instances if args.full else select_one_per_config
    cases = selector(TIER1_DIR)
    mode_label = "FULL tier-1 (all instances)" if args.full else "subset (one per config)"
    file_prefix = "Tier1Full_S1_S2" if args.full else "Tier1Final_S1_S2"

    print(f"Mode: {mode_label}")
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

    # S1 (SQA_HW) re-added alongside S2 (SQA_SF_HW) for the final
    # data collection: the 4-case tuning comparison in
    # Tier1Subset_S1_S2_HW_1.json already showed S2 strictly dominates,
    # but running both at full tier-1 scale gives instance-level
    # statistics on that claim instead of asserting it from a small
    # sample.  ILP retained as the optimality-gap baseline.
    # S1 carries a per-solver chain_strength_fn override; S2 uses the
    # case-level schedule (schedule_chain_strength, prefactor 1.5 flat).
    # The harness honours ``chain_strength_fn`` on a registry entry as
    # an override that applies to that solver only -- so the two
    # encodings get different prefactors at the same problem size
    # without forking the harness loop.
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
            f"tier1_subset_sqa_hw.py [{mode_label}] -- "
            f"{len(cases)} cases, final locked config: ILP baseline + "
            f"S1 + S2 (both encodings for full-scale comparison); "
            f"num_reads 500/1000, annealing_time=50us, chain_strength "
            f"prefactor 1.5 flat.  selection_policy={selection_policy}.  "
            f"Tuning history in the prior Tier1Subset_*_HW_*.json files."
        ),
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Quick post-run summary: how many SQA_HW rows matched the ILP cost,
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
