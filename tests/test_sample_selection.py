"""
Unit tests for util.sample_selection.

These tests construct synthetic ``dimod.SampleSet`` objects by hand so
we can control which samples are feasible and what their energies are,
then verify the selector's policy semantics independently of the SQA
or QPU pipeline.

Problem fixture: 2 nodes, 1 unit partition, k_safety = 1.  This makes
the feasibility check easy to reason about:

  * A_p1_n1 + A_p1_n2 == 1  (k-safety)
  * each node's load <= 1   (capacity, trivially satisfied)

So the four possible samples are:
  (0, 0) -- k=0 infeasible
  (1, 0) -- feasible
  (0, 1) -- feasible
  (1, 1) -- k=2 infeasible
"""

import dimod
import pytest

from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    POLICY_LOWEST_ENERGY,
    VALID_POLICIES,
    select_sample,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_problem():
    """Minimal-but-non-trivial problem: 2 nodes, 1 partition, k=1."""
    return {
        'nodes': {'n1': 1, 'n2': 1},
        'partitions': {'p1': 1},
        'k_safety': 1,
        'requests': {('p1', 'n1'): 1, ('p1', 'n2'): 1},
        'comm_costs': {'p1': 1},
    }


def _make_sampleset(rows):
    """
    rows: list of (sample_dict, energy, num_occurrences).
    Returns a dimod.SampleSet built so the rows appear in the order given.
    """
    samples = [r[0] for r in rows]
    energies = [r[1] for r in rows]
    occs = [r[2] for r in rows]
    return dimod.SampleSet.from_samples(
        samples,
        vartype=dimod.BINARY,
        energy=energies,
        num_occurrences=occs,
    )


# ---------------------------------------------------------------------------
# Policy: best_feasible
# ---------------------------------------------------------------------------

def test_picks_lowest_energy_feasible_over_lower_energy_infeasible(tiny_problem):
    """
    Lowest-energy row is infeasible (A_p1_n1 = A_p1_n2 = 1, k=2).
    Selector under best_feasible must skip it and return the next-lowest
    energy row, which IS feasible.
    """
    ss = _make_sampleset([
        ({'A_p1_n1': 1, 'A_p1_n2': 1}, -10.0, 4),   # rank 0, infeasible
        ({'A_p1_n1': 1, 'A_p1_n2': 0},  -5.0, 3),   # rank 1, feasible
        ({'A_p1_n1': 0, 'A_p1_n2': 1},  -3.0, 2),   # rank 2, feasible
        ({'A_p1_n1': 0, 'A_p1_n2': 0},   0.0, 1),   # rank 3, infeasible
    ])

    selected, diag = select_sample(
        ss, **tiny_problem, policy=POLICY_BEST_FEASIBLE,
    )

    assert selected is not None
    assert selected.sample == {'A_p1_n1': 1, 'A_p1_n2': 0}
    assert selected.energy == -5.0
    assert diag['selection_policy'] == POLICY_BEST_FEASIBLE
    assert diag['selected_rank_by_energy'] == 1
    assert diag['selected_energy'] == -5.0
    assert diag['selected_num_occurrences'] == 3
    assert diag['feasibility_fallback'] is False
    assert diag['num_distinct_feasible'] == 2
    assert diag['num_feasible_reads'] == 3 + 2     # weighted by occurrences
    assert diag['feasibility_yield'] == round(5 / 10, 4)


def test_fallback_to_lowest_energy_when_all_infeasible(tiny_problem):
    """
    No row is feasible -- selector must fall back to the lowest-energy
    sample and flag ``feasibility_fallback``.
    """
    ss = _make_sampleset([
        ({'A_p1_n1': 1, 'A_p1_n2': 1}, -10.0, 4),  # k=2, infeasible
        ({'A_p1_n1': 0, 'A_p1_n2': 0},   0.0, 1),  # k=0, infeasible
    ])

    selected, diag = select_sample(
        ss, **tiny_problem, policy=POLICY_BEST_FEASIBLE,
    )

    assert selected is not None
    assert selected.sample == {'A_p1_n1': 1, 'A_p1_n2': 1}
    assert selected.energy == -10.0
    assert diag['feasibility_fallback'] is True
    assert diag['selected_rank_by_energy'] == 0
    assert diag['selected_energy'] == -10.0
    assert diag['num_distinct_feasible'] == 0
    assert diag['num_feasible_reads'] == 0
    assert diag['feasibility_yield'] == 0.0


def test_num_feasible_reads_counts_occurrences_not_rows(tiny_problem):
    """
    ``num_feasible_reads`` must sum num_occurrences over feasible rows,
    not count rows.  This is the diagnostic that matters when comparing
    feasibility yield across runs with different num_reads totals.
    """
    ss = _make_sampleset([
        ({'A_p1_n1': 1, 'A_p1_n2': 0}, -5.0, 7),   # feasible, 7 reads
        ({'A_p1_n1': 1, 'A_p1_n2': 1}, -1.0, 90),  # infeasible, 90 reads
        ({'A_p1_n1': 0, 'A_p1_n2': 1},  2.0, 3),   # feasible, 3 reads
    ])

    _, diag = select_sample(
        ss, **tiny_problem, policy=POLICY_BEST_FEASIBLE,
    )

    assert diag['num_distinct_feasible'] == 2
    assert diag['num_feasible_reads'] == 10        # 7 + 3
    assert diag['feasibility_yield'] == 0.1        # 10 / 100


def test_lowest_energy_row_already_feasible_returns_it(tiny_problem):
    """
    When the lowest-energy sample IS already feasible, best_feasible
    must return it (rank 0, no fallback).  This is the common case and
    should match the legacy 'lowest_energy' behaviour exactly.
    """
    ss = _make_sampleset([
        ({'A_p1_n1': 1, 'A_p1_n2': 0}, -5.0, 6),   # feasible
        ({'A_p1_n1': 1, 'A_p1_n2': 1}, -1.0, 2),   # infeasible
        ({'A_p1_n1': 0, 'A_p1_n2': 1},  2.0, 2),   # feasible
    ])

    selected, diag = select_sample(
        ss, **tiny_problem, policy=POLICY_BEST_FEASIBLE,
    )

    assert selected.sample == {'A_p1_n1': 1, 'A_p1_n2': 0}
    assert diag['selected_rank_by_energy'] == 0
    assert diag['feasibility_fallback'] is False


# ---------------------------------------------------------------------------
# Policy: lowest_energy (legacy)
# ---------------------------------------------------------------------------

def test_lowest_energy_policy_ignores_feasibility(tiny_problem):
    """
    Under 'lowest_energy' the selector must return ``sampleset.first``
    regardless of whether it is feasible, matching the historical
    behaviour for fair comparison against the existing result bank.
    """
    ss = _make_sampleset([
        ({'A_p1_n1': 1, 'A_p1_n2': 1}, -10.0, 4),  # infeasible
        ({'A_p1_n1': 1, 'A_p1_n2': 0},  -5.0, 3),  # feasible
    ])

    selected, diag = select_sample(
        ss, **tiny_problem, policy=POLICY_LOWEST_ENERGY,
    )

    assert selected.sample == {'A_p1_n1': 1, 'A_p1_n2': 1}
    assert selected.energy == -10.0
    assert diag['selection_policy'] == POLICY_LOWEST_ENERGY
    assert diag['selected_rank_by_energy'] == 0
    assert diag['feasibility_fallback'] is False   # not a fallback; chosen on purpose
    # Feasibility yield is still reported under either policy.
    assert diag['num_distinct_feasible'] == 1
    assert diag['num_feasible_reads'] == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_unknown_policy_raises(tiny_problem):
    ss = _make_sampleset([
        ({'A_p1_n1': 1, 'A_p1_n2': 0}, -5.0, 1),
    ])
    with pytest.raises(ValueError):
        select_sample(ss, **tiny_problem, policy='made_up_policy')


def test_empty_sampleset_returns_none(tiny_problem):
    empty = dimod.SampleSet.from_samples(
        [], vartype=dimod.BINARY, energy=[],
    )
    selected, diag = select_sample(
        empty, **tiny_problem, policy=POLICY_BEST_FEASIBLE,
    )
    assert selected is None
    # Empty diagnostics: every numeric field is None, policy is preserved.
    assert diag['selection_policy'] == POLICY_BEST_FEASIBLE
    assert diag['selected_rank_by_energy'] is None
    assert diag['num_feasible_reads'] is None
    assert diag['feasibility_yield'] is None


def test_valid_policies_constant_is_complete():
    """Belt-and-braces: make sure both policies are exported."""
    assert POLICY_BEST_FEASIBLE in VALID_POLICIES
    assert POLICY_LOWEST_ENERGY in VALID_POLICIES
    assert len(VALID_POLICIES) == 2


# ---------------------------------------------------------------------------
# Returned sample shape -- downstream compatibility
# ---------------------------------------------------------------------------

def test_selected_sample_has_dimod_first_shape(tiny_problem):
    """
    Downstream code (run_experiment._to_flat, calculate_solution_cost)
    accesses ``.sample`` on the returned object.  The selector returns
    a Sample namedtuple from sampleset.data(), which must expose the
    same attributes as ``sampleset.first``.
    """
    ss = _make_sampleset([
        ({'A_p1_n1': 1, 'A_p1_n2': 1}, -10.0, 4),  # infeasible
        ({'A_p1_n1': 1, 'A_p1_n2': 0},  -5.0, 3),  # feasible
    ])
    selected, _ = select_sample(
        ss, **tiny_problem, policy=POLICY_BEST_FEASIBLE,
    )
    assert hasattr(selected, 'sample')
    assert hasattr(selected, 'energy')
    assert hasattr(selected, 'num_occurrences')
    # `.sample` is the same flat dict shape as `sampleset.first.sample`.
    assert selected.sample == {'A_p1_n1': 1, 'A_p1_n2': 0}
