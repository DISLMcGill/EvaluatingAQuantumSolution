"""
Shared Leap-hybrid metadata extraction for the hybrid solvers.

The three hybrid solvers (SQA_H, SQA_SF_H, SQA_DW_H) all submit their
BQM to one of D-Wave's Leap *hybrid* solvers (e.g.
``hybrid_binary_quadratic_model_version2``) rather than to a bare QPU.
A hybrid solver runs a classical heuristic in tandem with one or more
QPU calls and hides the minor-embedding step entirely, so the set of
facts worth persisting differs from the pure-QPU path:

  * there is **no** client-side embedding, chain dict, chain-break
    fraction or physical-qubit count -- the hybrid service performs (and
    discards) its own internal embedding, and none of it is returned to
    the client.  Reporting those fields would be meaningless, so this
    module deliberately omits them rather than emitting misleading
    zeros;
  * the timing block returned in ``sampleset.info`` is different.
    Hybrid solvers report ``run_time`` (total solver wall-clock, us),
    ``charge_time`` (the billed time, us) and ``qpu_access_time`` (the
    portion actually spent on the QPU, us).  These are the analog of
    the QPU ``timing`` dict and are what "hybrid solver time" must mean
    to be comparable across runs;
  * the solver identifier is the hybrid solver *name* rather than a
    ``chip_id`` -- two runs against two different hybrid solver
    versions are not the same experiment.

The functions here are the single point of truth for what hybrid
metadata gets persisted; the three solver classes should call
``extract_hybrid_summary`` rather than reimplement it.  The shape of
the returned dict is intentionally aligned with
``_hw_common.extract_hardware_summary`` (same key names where the
concept survives, ``None`` where it does not) so downstream analysis
can consume hardware and hybrid runs through one code path.
"""

from __future__ import annotations


# All hybrid-solver timing fields we know about.  Anything not present
# in a given sampleset is reported as None; this list is the union of
# what current D-Wave Leap hybrid solvers return.
_HYBRID_TIMING_FIELDS = (
    'run_time',
    'charge_time',
    'qpu_access_time',
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


def extract_hybrid_summary(
    sampleset,
    wall_time_ms,
    solver_name=None,
):
    """
    Build the canonical metadata dict for one hybrid submission.

    Parameters
    ----------
    sampleset : dimod.SampleSet
        The sampleset returned by ``sampler.sample(bqm, ...)``.
    wall_time_ms : float
        End-to-end wall-clock time around the sample() call (ms).
        This is the *client-side* wall time and includes network
        latency and queueing; ``run_time`` / ``charge_time`` from the
        timing block are the server-side figures.
    solver_name : str or None
        The hybrid solver identifier (e.g.
        ``hybrid_binary_quadratic_model_version2``).  The solver
        captures this from ``sampler.solver.id`` before the sampler
        goes out of scope; we accept it as an arg here because the
        sampler is no longer in scope at summary time.

    Returns
    -------
    dict
        Single flat dict containing every field worth persisting.
        Missing values are reported as None rather than dropped.
        Embedding / chain fields are intentionally absent: hybrid
        solvers do not expose them.
    """
    timing = sampleset.info.get('timing', {}) if sampleset is not None else {}
    # Some dwave-system versions surface the hybrid timing fields at the
    # top level of ``info`` rather than nested under ``timing``.  Merge
    # both so the payload is populated regardless of client version.
    info = sampleset.info if sampleset is not None else {}
    timing_payload = {
        f: timing.get(f, info.get(f)) for f in _HYBRID_TIMING_FIELDS
    }

    best_energy, best_occ = _best_sample_stats(sampleset)

    problem_id = sampleset.info.get('problem_id') if sampleset is not None else None
    problem_label = sampleset.info.get('problem_label') if sampleset is not None else None

    logical_vars = (
        len(sampleset.variables) if sampleset is not None and sampleset.variables else None
    )
    num_samples = len(sampleset) if sampleset is not None else None

    return {
        'wall_time_ms': (
            round(wall_time_ms, 1) if wall_time_ms is not None and wall_time_ms >= 0 else None
        ),

        # Identity / traceability
        'solver_name': solver_name,
        'problem_id': problem_id,
        'problem_label': problem_label,

        # Problem size (the hybrid analog of the embedding block --
        # logical variables only; no physical-qubit / chain concept).
        'logical_variables': logical_vars,

        # Sampling
        'num_samples': num_samples,
        'best_energy': best_energy,
        'best_num_occurrences': best_occ,

        # Hybrid timing (full breakdown).  We promote the two most-used
        # fields to the top level for parity with the hardware summary;
        # the full dict is included alongside.
        'run_time_us': timing_payload.get('run_time'),
        'qpu_access_time_us': timing_payload.get('qpu_access_time'),
        'hybrid_timing': timing_payload,
    }
