"""
Pre-flight Leap solver-access cost estimate for hybrid experiments.

A Leap hybrid submission bills roughly the ``time_limit`` it runs for,
which -- when ``time_limit`` is left unset -- is the service's
*minimum* time limit for that BQM (``LeapHybridSampler.min_time_limit``).
That minimum rises steeply with problem size, so at the hybrid_stress
scale a sweep that looks cheap on paper can cost tens of minutes.

``estimate_experiment_seconds`` works out the spend *before* any
submission so callers can refuse to start a run that would blow a budget.
Querying ``min_time_limit`` consumes no solver-access time (it is derived
from solver properties), though it does construct a Leap client, so the
floor path needs a configured token.  The fixed-``time_limit`` path needs
neither a client nor a token.
"""

from __future__ import annotations

from pathlib import Path


def estimate_experiment_seconds(case_paths, hybrid_registry, time_limit,
                                solver_name=None):
    """
    Estimate total Leap solver-access seconds for a hybrid experiment.

    Args:
        case_paths:      iterable of test-case JSON paths to be run.
        hybrid_registry: the ``type == "hybrid"`` entries of the solver
                         registry (each a dict with ``class`` and optional
                         ``kwargs``).  One submission is billed per case
                         per hybrid solver.
        time_limit:      the per-submission ``time_limit`` (seconds) that
                         will be passed to the solvers, or ``None`` to use
                         each BQM's service minimum.
        solver_name:     optional explicit Leap hybrid solver id used when
                         querying the floor; ``None`` uses the client
                         default.

    Returns:
        (total_seconds, breakdown) where breakdown is a list of
        ``(case_path, solver_name, seconds)`` tuples.
    """
    from util.test_generation.json_to_dict import json_to_test_case

    sampler = None  # lazily constructed only if we need to query floors
    total = 0.0
    breakdown = []

    for path in case_paths:
        inputs = json_to_test_case(str(path))
        for desc in hybrid_registry:
            cls = desc["class"]
            kwargs = dict(desc.get("kwargs") or {})

            if time_limit is not None:
                secs = float(time_limit)
            else:
                if sampler is None:
                    from dwave.system import LeapHybridSampler
                    sampler = (LeapHybridSampler(solver=solver_name)
                               if solver_name else LeapHybridSampler())
                solver = cls(*inputs, **kwargs)
                bqm = solver.build_bqm()
                secs = float(sampler.min_time_limit(bqm))

            total += secs
            breakdown.append((Path(path), desc["name"], secs))

    return total, breakdown


def check_budget(total_seconds, max_budget_seconds):
    """
    Return (ok, message).  ``ok`` is False when a budget is set and the
    estimate exceeds it.  ``max_budget_seconds=None`` always passes.
    """
    if max_budget_seconds is None:
        return True, (f"Estimated Leap solver-access spend: "
                      f"~{total_seconds:.0f}s ({total_seconds / 60:.1f} min). "
                      f"No --max-budget-seconds set.")
    if total_seconds > max_budget_seconds:
        return False, (
            f"Estimated spend ~{total_seconds:.0f}s "
            f"({total_seconds / 60:.1f} min) exceeds --max-budget-seconds "
            f"{max_budget_seconds:.0f}s ({max_budget_seconds / 60:.1f} min). "
            f"Aborting before any submission. Reduce the case set / "
            f"--time-limit, or raise the budget.")
    return True, (
        f"Estimated spend ~{total_seconds:.0f}s "
        f"({total_seconds / 60:.1f} min) is within --max-budget-seconds "
        f"{max_budget_seconds:.0f}s ({max_budget_seconds / 60:.1f} min).")
