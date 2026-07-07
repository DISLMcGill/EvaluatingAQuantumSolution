"""
Shared QPU metadata extraction for the hardware solvers.

The three hardware solvers (SQA_HW, SQA_SF_HW, SQA_DW_HW) all need to
record the same set of QPU-side facts after a submission completes:

  * the full ``sampleset.info['timing']`` block (qpu_access_time,
    qpu_anneal_time_per_sample, qpu_readout_time_per_sample,
    qpu_programming_time, etc.) -- without this, "QPU time" reported
    upstream is ambiguous and not comparable across runs;
  * the chip / solver identifier (e.g. ``Advantage_system6.4``) -- two
    runs on two different QPUs are not the same experiment;
  * the Leap problem ID -- needed to look the submission up in the Leap
    dashboard after the fact;
  * the embedding (chain dict) and derived statistics -- the embedding
    is the single biggest determinant of solution quality on sparse
    topologies and the harness was previously discarding it.

The functions in this module are the single point of truth for what
QPU metadata gets persisted; the three solver classes should call
``extract_hardware_summary`` rather than reimplement it.
"""

from __future__ import annotations

import time as _time


# Wall-clock cap (seconds) for the client-side minor-embedding search.
# EmbeddingComposite delegates embedding to minorminer.find_embedding,
# whose own default timeout is 1000 s (~16.7 min).  On a problem that has
# no embedding on the target topology -- e.g. the n9_p50 cell, whose BQM
# is a dense ~450-variable graph that does not fit Advantage's Pegasus
# connectivity -- that full 1000 s is spent, per solver and per case,
# before the search gives up and raises "no embedding found".  Capping
# the search at 5 minutes bounds that wasted time; the failure is still
# recorded as a normal solver error by the harness, just 3x sooner.
# Embeddable cases in this benchmark are found in well under a second, so
# the cap never bites a case that would otherwise have embedded.
DEFAULT_EMBED_TIMEOUT = 300.0  # 5 minutes


# Substrings that mark a QPU submission failure as *transient* -- a
# network/queue hiccup worth retrying rather than a real problem with the
# BQM or account.  Matched case-insensitively against ``str(exc)`` so we
# don't have to import (and pin) D-Wave's exception hierarchy, which is
# not importable in environments without dwave-system.  'API request
# timed out' (seen in the tier-1 arbitrary run) is the motivating case.
_TRANSIENT_MARKERS = (
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "connection",
    "connectionerror",
    "502",
    "503",
    "504",
    "too many requests",
    "429",
)


def _is_transient(exc) -> bool:
    """True if ``exc`` looks like a retryable transient QPU/API error."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def submit_with_retry(
    sampler,
    bqm,
    sample_kwargs,
    *,
    max_attempts: int = 4,
    base_delay: float = 2.0,
    sleep=_time.sleep,
    on_retry=None,
):
    """
    Call ``sampler.sample(bqm, **sample_kwargs)`` with bounded
    exponential backoff on *transient* failures.

    A single transient error (e.g. ``API request timed out``) on one
    case would otherwise leave a permanent hole in a multi-case sweep --
    the result file just never gets that solver's entry.  Retrying a few
    times with backoff turns the common transient hiccup into a brief
    pause instead of lost data.

    Non-transient errors (a malformed BQM, an auth failure, an
    embedding error) are re-raised immediately -- retrying those would
    only burn time and queue budget.  After ``max_attempts`` transient
    failures the last exception is re-raised so the harness still records
    the case as errored rather than hanging forever.

    Args:
        sampler:        a dimod-style sampler with ``.sample(bqm, **kw)``.
        bqm:            the BinaryQuadraticModel to submit.
        sample_kwargs:  kwargs forwarded to ``sampler.sample``.
        max_attempts:   total tries, including the first (>= 1).
        base_delay:     seconds before the first retry; doubles each time.
        sleep:          injectable sleep fn (tests pass a no-op).
        on_retry:       optional callable ``(attempt, exc, delay)`` for
                        logging/observability; exceptions from it are
                        ignored.

    Returns:
        The SampleSet from the first successful ``sample`` call.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return sampler.sample(bqm, **sample_kwargs)
        except Exception as exc:  # noqa: BLE001 -- re-raised below unless transient
            last_exc = exc
            if attempt >= max_attempts or not _is_transient(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            if on_retry is not None:
                try:
                    on_retry(attempt, exc, delay)
                except Exception:
                    pass
            sleep(delay)
    # Defensive: loop either returns or raises; this satisfies linters.
    raise last_exc


# All QPU timing fields we know about.  Anything not present in a given
# sampleset will simply be reported as None; this list is the union of
# what current D-Wave QPUs return.
_QPU_TIMING_FIELDS = (
    'qpu_sampling_time',
    'qpu_anneal_time_per_sample',
    'qpu_readout_time_per_sample',
    'qpu_access_time',
    'qpu_access_overhead_time',
    'qpu_programming_time',
    'qpu_delay_time_per_sample',
    'total_post_processing_time',
    'post_processing_overhead_time',
)


def chain_break_fraction(sampleset):
    """Mean per-sample chain-break fraction, or None if unavailable."""
    if not hasattr(sampleset, 'record'):
        return None
    if 'chain_break_fraction' not in sampleset.record.dtype.names:
        return None
    fractions = sampleset.record['chain_break_fraction']
    if len(fractions) == 0:
        return None
    return round(float(fractions.mean()), 4)


def _embedding_stats(embedding):
    """Return (physical_qubits, max_chain_len, mean_chain_len)."""
    if not embedding:
        return None, None, None
    chain_lengths = [len(chain) for chain in embedding.values()]
    if not chain_lengths:
        return 0, 0, 0.0
    return (
        sum(chain_lengths),
        max(chain_lengths),
        round(sum(chain_lengths) / len(chain_lengths), 2),
    )


def _best_sample_stats(sampleset):
    """Return (best_energy, num_occurrences_of_best)."""
    if sampleset is None or len(sampleset) == 0:
        return None, None
    try:
        first = sampleset.first
        best_energy = float(first.energy)
        best_occ = int(first.num_occurrences)
        return best_energy, best_occ
    except Exception:
        return None, None


def extract_hardware_summary(
    sampleset,
    embedding,
    wall_time_ms,
    chip_id=None,
    selection_diagnostics=None,
):
    """
    Build the canonical metadata dict for one QPU submission.

    Parameters
    ----------
    sampleset : dimod.SampleSet
        The sampleset returned by ``sampler.sample(bqm, ...)``.
    embedding : dict or None
        The minor embedding used (chain dict), as recorded from
        ``sampleset.info['embedding_context']['embedding']``.
    wall_time_ms : float
        End-to-end wall-clock time around the sample() call (ms).
    chip_id : str or None
        The QPU identifier (e.g. ``Advantage_system6.4``).  The solver
        captures this from ``sampler.child.properties['chip_id']``
        before the sampler goes out of scope; we accept it as an arg
        here because the sampler is no longer in scope at summary time.
    selection_diagnostics : dict or None
        Diagnostics dict returned by
        ``util.sample_selection.select_sample`` describing which sample
        was actually returned (policy, energy rank, feasibility-fallback
        flag, feasibility yield).  Promoted to top-level keys for
        downstream analysis.  ``None`` if the solver did not run a
        selection step (e.g. failed before sampling).

    Returns
    -------
    dict
        Single flat dict containing every field worth persisting.
        Missing values are reported as None rather than dropped.
        ``best_energy`` / ``best_num_occurrences`` always describe the
        *lowest-energy* sample (so historical schemas stay comparable);
        ``selected_*`` describe the sample that was actually returned
        to the harness.
    """
    timing = sampleset.info.get('timing', {}) if sampleset is not None else {}
    timing_payload = {f: timing.get(f) for f in _QPU_TIMING_FIELDS}

    phys, max_chain, mean_chain = _embedding_stats(embedding)
    best_energy, best_occ = _best_sample_stats(sampleset)

    problem_id = sampleset.info.get('problem_id') if sampleset is not None else None
    problem_label = sampleset.info.get('problem_label') if sampleset is not None else None

    logical_vars = (
        len(sampleset.variables) if sampleset is not None and sampleset.variables else None
    )
    num_reads = len(sampleset) if sampleset is not None else None

    sel = selection_diagnostics or {}

    return {
        'wall_time_ms': (
            round(wall_time_ms, 1) if wall_time_ms is not None and wall_time_ms >= 0 else None
        ),

        # Identity / traceability
        'chip_id': chip_id,
        'problem_id': problem_id,
        'problem_label': problem_label,

        # Embedding
        'physical_qubits': phys,
        'logical_variables': logical_vars,
        'max_chain_length': max_chain,
        'mean_chain_length': mean_chain,
        'chain_break_fraction': chain_break_fraction(sampleset),
        'embedding': embedding if embedding else None,

        # Sampling -- lowest-energy stats (unchanged semantics)
        'num_reads': num_reads,
        'best_energy': best_energy,
        'best_num_occurrences': best_occ,

        # Sample selection -- describes which sample was actually
        # returned to the harness (may not be the lowest-energy one).
        'selection_policy': sel.get('selection_policy'),
        'selected_rank_by_energy': sel.get('selected_rank_by_energy'),
        'selected_energy': sel.get('selected_energy'),
        'selected_num_occurrences': sel.get('selected_num_occurrences'),
        'num_feasible_reads': sel.get('num_feasible_reads'),
        'num_distinct_feasible': sel.get('num_distinct_feasible'),
        'feasibility_yield': sel.get('feasibility_yield'),
        'feasibility_fallback': sel.get('feasibility_fallback'),

        # QPU timing (full breakdown).  We promote the two most-used
        # fields to the top level for backwards compatibility with the
        # existing harness and downstream analysis; the full dict is
        # included alongside.
        'qpu_access_time_us': timing.get('qpu_access_time'),
        'qpu_anneal_time_per_sample_us': timing.get('qpu_anneal_time_per_sample'),
        'qpu_timing': timing_payload,
    }
