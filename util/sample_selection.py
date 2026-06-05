"""
Sample-selection policies for SQA / QPU sampler output.

Both the simulated and hardware solvers historically returned
``sampleset.first`` -- i.e. the lowest-energy sample, regardless of
whether it satisfies the problem's k-safety and capacity constraints.
Because constraint violations are encoded as energy penalties in the
QUBO, the lowest-energy sample is usually feasible, but not always:
when the penalty weights are mis-calibrated or the anneal happens to
land on a near-feasible local minimum with a slightly lower energy
than every truly-feasible sample, the reported result is infeasible
and downstream cost / validity reporting reflects that.

This module adds a second selection policy that walks samples in
ascending energy order and returns the first one that satisfies
``is_valid_solution``.  When no read is feasible it falls back to
``sampleset.first`` so the harness still produces a result entry and
the (now infeasible) sample plus a ``feasibility_fallback=True`` flag
are recorded for analysis.

The two policies are exposed as the strings ``'best_feasible'`` and
``'lowest_energy'``; both solver families (simulated and hardware)
accept ``selection_policy=<policy>`` at construction time and call
``select_sample`` after every ``sampler.sample(...)`` call.

The selector is deliberately solver-agnostic: it only needs the
problem definition (nodes / partitions / k_safety / requests /
comm_costs) and a dimod SampleSet.  It is therefore safe to call
from both the simulated path (``solvers/simulated_solvers``) and the
hardware path (``solvers/quantum_hardware_solvers``) without
introducing a cross-package dependency.
"""

from __future__ import annotations

from util.calculate_solution_cost import is_valid_solution


# Public string constants for the two policies.  Solvers that accept
# ``selection_policy=...`` should validate against this set.
POLICY_BEST_FEASIBLE = 'best_feasible'
POLICY_LOWEST_ENERGY = 'lowest_energy'
VALID_POLICIES = (POLICY_BEST_FEASIBLE, POLICY_LOWEST_ENERGY)


def _empty_diagnostics(policy):
    """Diagnostics shape returned when the sampleset is empty / None."""
    return {
        'selection_policy': policy,
        'selected_rank_by_energy': None,
        'selected_energy': None,
        'selected_num_occurrences': None,
        'num_feasible_reads': None,
        'num_distinct_feasible': None,
        'feasibility_yield': None,
        'feasibility_fallback': None,
    }


def _count_feasible(sampleset, nodes, partitions, k_safety,
                    requests, comm_costs):
    """
    Walk every row in the sampleset, return:
      * total number of feasible *reads* (weighted by num_occurrences),
      * number of distinct feasible *samples* (rows).

    This is a single O(num_distinct_samples) pass and reports the
    headline diagnostic for whether the lambda calibration is doing
    its job: if very few reads are feasible, the penalty weights are
    probably too soft.
    """
    n_feasible_reads = 0
    n_distinct_feasible = 0
    for record in sampleset.data(['sample', 'num_occurrences'],
                                  sorted_by=None):
        if is_valid_solution(nodes, partitions, k_safety,
                             requests, comm_costs, record.sample):
            n_distinct_feasible += 1
            n_feasible_reads += int(record.num_occurrences)
    return n_feasible_reads, n_distinct_feasible


def select_sample(sampleset, nodes, partitions, k_safety,
                  requests, comm_costs,
                  policy=POLICY_BEST_FEASIBLE):
    """
    Select one sample from ``sampleset`` according to ``policy`` and
    return ``(selected, diagnostics)``.

    Parameters
    ----------
    sampleset : dimod.SampleSet
        The full set of reads returned by ``sampler.sample(bqm, ...)``.
    nodes, partitions, k_safety, requests, comm_costs
        Problem definition (matches ``SolverBase.__init__`` args), used
        to evaluate feasibility via ``is_valid_solution``.
    policy : str
        One of ``'best_feasible'`` (default) or ``'lowest_energy'``.
        ``'best_feasible'``: walk samples in ascending energy order and
        return the first one that satisfies every constraint; fall back
        to the lowest-energy sample if none is feasible.
        ``'lowest_energy'``: legacy behaviour -- return the lowest-energy
        sample without checking feasibility.

    Returns
    -------
    selected : dimod Sample (the namedtuple yielded by ``sampleset.data``)
        Has ``.sample`` (dict), ``.energy``, and ``.num_occurrences``;
        ``sampleset.first`` returns the same shape and downstream code
        already expects it.  ``None`` if the sampleset is empty.
    diagnostics : dict
        Schema:
            selection_policy        : str  (the policy actually used)
            selected_rank_by_energy : int  (0 = lowest-energy sample)
            selected_energy         : float
            selected_num_occurrences: int
            num_feasible_reads      : int (sum of num_occurrences over feasible rows)
            num_distinct_feasible   : int (count of feasible rows)
            feasibility_yield       : float in [0, 1] (feasible reads / total reads)
            feasibility_fallback    : bool (True if policy=='best_feasible'
                                            but no feasible sample existed,
                                            so we fell back to lowest energy)
    """
    if policy not in VALID_POLICIES:
        raise ValueError(
            f"unknown selection_policy {policy!r}; expected one of {VALID_POLICIES}"
        )

    if sampleset is None or len(sampleset) == 0:
        return None, _empty_diagnostics(policy)

    # Feasibility yield is independent of policy and worth reporting on
    # every run -- it answers "what fraction of reads were feasible?",
    # which is the headline diagnostic for whether the lambdas are
    # tuned correctly.  We compute it once up front.
    n_feasible_reads, n_distinct_feasible = _count_feasible(
        sampleset, nodes, partitions, k_safety, requests, comm_costs,
    )
    total_reads = int(sum(int(o) for o in sampleset.record.num_occurrences))
    feasibility_yield = (
        round(n_feasible_reads / total_reads, 4) if total_reads > 0 else None
    )

    diagnostics = {
        'selection_policy': policy,
        'selected_rank_by_energy': None,
        'selected_energy': None,
        'selected_num_occurrences': None,
        'num_feasible_reads': n_feasible_reads,
        'num_distinct_feasible': n_distinct_feasible,
        'feasibility_yield': feasibility_yield,
        'feasibility_fallback': False,
    }

    if policy == POLICY_LOWEST_ENERGY:
        selected = sampleset.first
        diagnostics['selected_rank_by_energy'] = 0
        diagnostics['selected_energy'] = float(selected.energy)
        diagnostics['selected_num_occurrences'] = int(selected.num_occurrences)
        return selected, diagnostics

    # policy == POLICY_BEST_FEASIBLE
    #
    # Walk rows in ascending-energy order; first feasible row wins.
    # ``sampleset.data(sorted_by='energy')`` yields one Sample-namedtuple
    # per distinct row -- ``sampleset.first`` is exactly the head of this
    # iterator, so when no row is feasible we re-fetch it for the
    # fallback path.
    for rank, record in enumerate(
        sampleset.data(['sample', 'energy', 'num_occurrences'],
                       sorted_by='energy')
    ):
        if is_valid_solution(nodes, partitions, k_safety,
                             requests, comm_costs, record.sample):
            diagnostics['selected_rank_by_energy'] = rank
            diagnostics['selected_energy'] = float(record.energy)
            diagnostics['selected_num_occurrences'] = int(record.num_occurrences)
            return record, diagnostics

    # No feasible sample -- fall back to the lowest-energy sample so
    # the harness still has something to score, and surface the fact
    # in the diagnostics so downstream analysis can see what happened.
    selected = sampleset.first
    diagnostics['selected_rank_by_energy'] = 0
    diagnostics['selected_energy'] = float(selected.energy)
    diagnostics['selected_num_occurrences'] = int(selected.num_occurrences)
    diagnostics['feasibility_fallback'] = True
    return selected, diagnostics
