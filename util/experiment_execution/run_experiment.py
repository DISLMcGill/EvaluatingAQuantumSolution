"""
Core experiment harness for QuantumClean (revised in Phase 5).

Loads pre-generated test cases from disk, runs every registered solver on
each case, computes cost / validity / BQM stats / optimality gap, and writes
results incrementally to a JSON file.

Phase 5 changes vs. the old harness:

* ``optimality_gap`` is reported as both absolute (``cost_optimal - ilp_cost``)
  and relative (with sensible behaviour when ``ilp_cost == 0``: relative is
  0.0 iff the solver also returned 0, ``None`` otherwise).
* Result entries include ``k_safety_violations`` and ``capacity_overruns``
  so "invalid" results carry signal instead of a single boolean.
* ``time_ms`` is split into ``wall_time_ms`` (always present) and
  ``qpu_anneal_time_per_sample_us`` / ``ilp_branch_nodes`` (where
  applicable).  Plotting these side-by-side is the *caller's* responsibility.
* ``_NumpyEncoder`` is used on every JSON write.
"""

import json
import os
import re
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

from util.calculate_solution_cost import (
    calculate_solution_cost,
    is_valid_solution,
)
from util.test_generation.json_to_dict import (
    json_to_test_case,
    load_test_case_metadata,
)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _write_json(path, payload):
    """
    Atomically write ``payload`` as JSON to ``path``.

    The harness rewrites the result file after *every* case, so a process
    killed mid-write (laptop sleeping/closing, Ctrl-C, OOM) could
    otherwise leave a truncated, unparseable file -- which would also
    defeat ``--resume``, since resume reloads exactly this file.  Writing
    to a temp file in the same directory and then ``os.replace``-ing it
    into place makes each write all-or-nothing: the result file is always
    either the previous complete state or the new complete state.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=4, cls=_NumpyEncoder)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Constraint diagnostics
# ---------------------------------------------------------------------------

def _violations(nodes, partitions, k_safety, sample_dict):
    """Return (k_safety_violations, capacity_overruns) for a flat A_p_n dict."""
    if sample_dict is None:
        return None, None
    k_viol = 0
    for p in partitions:
        cnt = sum(int(sample_dict.get(f"A_{p}_{n}", 0)) for n in nodes)
        if cnt != k_safety:
            k_viol += 1
    overruns = 0
    for n, cap in nodes.items():
        load = sum(
            int(sample_dict.get(f"A_{p}_{n}", 0)) * partitions[p]
            for p in partitions
        )
        if load > cap:
            overruns += 1
    return k_viol, overruns


def _to_flat(sample_obj):
    if sample_obj is None:
        return None
    return sample_obj.sample if hasattr(sample_obj, "sample") else sample_obj


def _jsonable_solution(flat):
    """
    Convert a flat sample dict into a plain ``{str: int}`` mapping that
    is safe to embed directly in the result JSON.

    The dimod-returned dicts use numpy integer types for values;
    ``_NumpyEncoder`` handles those at write time, but we coerce here so
    that downstream tools reading the result file (e.g. notebook
    inspection, ``json.loads`` round-trips) don't have to think about
    numpy at all.  Keys are already plain strings (``A_p_n``, ``S_n_k``,
    etc.) by the time we get here.

    Auxiliary variables (slack vars for S1, domain-wall ``W`` vars for
    S3) are preserved alongside the ``A_*`` assignment so that manual
    inspection of an invalid sample can see exactly what the sampler
    emitted, not just the projection onto the assignment.
    """
    if flat is None:
        return None
    return {str(k): int(v) for k, v in flat.items()}


def _gap(cost, ilp_cost):
    """
    Return (absolute_gap, relative_gap).

    * absolute is always defined when both are defined.
    * relative is 0.0 if both are 0, (cost - ilp)/ilp if ilp > 0,
      else None (we don't divide by 0 silently).
    """
    if cost is None or ilp_cost is None:
        return None, None
    abs_gap = cost - ilp_cost
    if ilp_cost > 0:
        rel = round(abs_gap / ilp_cost, 4)
    elif cost == 0:
        rel = 0.0
    else:
        rel = None
    return abs_gap, rel


# ---------------------------------------------------------------------------
# Test-case discovery (unchanged from the previous version)
# ---------------------------------------------------------------------------

def discover_test_cases(test_bank_dir, tier=None, node_counts=None,
                        partition_counts=None, max_cases=None):
    test_bank_dir = Path(test_bank_dir)
    search_root = test_bank_dir / tier if tier else test_bank_dir
    paths = sorted(search_root.rglob("*.json"))

    if node_counts is not None or partition_counts is not None:
        filtered = []
        for p in paths:
            # The (n, p) config lives in an ``n<N>_p<P>`` directory, but
            # how far up the tree that sits depends on the bank layout:
            # flat banks have it as the file's parent, while the
            # tightness-stratified banks nest it one level higher
            # (``.../n3_p4/t70/inst.json``).  Walk up the ancestry and
            # match the first ``n<N>_p<P>`` component rather than assuming
            # it is always ``p.parent`` -- the old code did the latter and
            # silently filtered *every* case out of the tightness banks,
            # so node/partition-scoped discovery returned nothing.
            m = None
            for ancestor in p.parents:
                m = re.fullmatch(r"n(\d+)_p(\d+)", ancestor.name)
                if m:
                    break
            if not m:
                continue
            n, pp = int(m.group(1)), int(m.group(2))
            if node_counts is not None and n not in node_counts:
                continue
            if partition_counts is not None and pp not in partition_counts:
                continue
            filtered.append(p)
        paths = filtered

    if max_cases is not None:
        paths = paths[:max_cases]

    return paths


# ---------------------------------------------------------------------------
# Per-solver execution
# ---------------------------------------------------------------------------

def _run_ilp(solver_class, nodes, partitions, k_safety, requests, comm_costs):
    try:
        solver = solver_class(nodes, partitions, k_safety, requests, comm_costs)
        t_ms, raw_result = solver.solve()
        flat = None
        if raw_result is not None:
            flat = {
                f"A_{p}_{n}": v
                for p, nd in raw_result.items() for n, v in nd.items()
            }
        cost = calculate_solution_cost(
            nodes, partitions, k_safety, requests, comm_costs, flat
        )
        valid = is_valid_solution(
            nodes, partitions, k_safety, requests, comm_costs, flat
        )
        k_viol, overruns = _violations(nodes, partitions, k_safety, flat)
        return {
            "cost": cost,
            "valid": valid,
            "k_safety_violations": k_viol,
            "capacity_overruns": overruns,
            "wall_time_ms": round(t_ms, 1) if t_ms is not None else None,
            "solution": _jsonable_solution(flat),
            "error": None,
        }
    except Exception as e:
        return {
            "cost": None, "valid": False,
            "k_safety_violations": None, "capacity_overruns": None,
            "wall_time_ms": None,
            "solution": None,
            "error": str(e),
        }


def _run_sqa(solver_class, nodes, partitions, k_safety, requests, comm_costs,
             num_reads, num_sweeps, beta_range, solver_kwargs=None):
    try:
        solver_kwargs = solver_kwargs or {}
        solver = solver_class(nodes, partitions, k_safety, requests, comm_costs,
                              **solver_kwargs)
        bqm = solver.build_bqm()
        bqm_vars = len(bqm.variables)
        bqm_interactions = len(bqm.quadratic)

        kw = dict(num_reads=num_reads, num_sweeps=num_sweeps)
        if beta_range is not None:
            kw["beta_range"] = beta_range

        t_ms, result = solver.solve(**kw)
        flat = _to_flat(result)
        cost = calculate_solution_cost(
            nodes, partitions, k_safety, requests, comm_costs, flat
        )
        valid = is_valid_solution(
            nodes, partitions, k_safety, requests, comm_costs, flat
        )
        k_viol, overruns = _violations(nodes, partitions, k_safety, flat)
        sel = getattr(solver, "selection_diagnostics", None) or {}
        return {
            "cost": cost,
            "valid": valid,
            "k_safety_violations": k_viol,
            "capacity_overruns": overruns,
            "wall_time_ms": round(t_ms, 1),
            "bqm_variables": bqm_vars,
            "bqm_interactions": bqm_interactions,
            "lambda_1": getattr(solver, "lambda_1", None),
            "lambda_2": getattr(solver, "lambda_2", None),
            "solution": _jsonable_solution(flat),
            "selection_policy": sel.get("selection_policy"),
            "selected_rank_by_energy": sel.get("selected_rank_by_energy"),
            "selected_energy": sel.get("selected_energy"),
            "selected_num_occurrences": sel.get("selected_num_occurrences"),
            "num_feasible_reads": sel.get("num_feasible_reads"),
            "num_distinct_feasible": sel.get("num_distinct_feasible"),
            "feasibility_yield": sel.get("feasibility_yield"),
            "feasibility_fallback": sel.get("feasibility_fallback"),
            "error": None,
        }
    except Exception as e:
        return {
            "cost": None, "valid": False,
            "k_safety_violations": None, "capacity_overruns": None,
            "wall_time_ms": None,
            "bqm_variables": None, "bqm_interactions": None,
            "lambda_1": None, "lambda_2": None,
            "solution": None,
            "selection_policy": None,
            "selected_rank_by_energy": None,
            "selected_energy": None,
            "selected_num_occurrences": None,
            "num_feasible_reads": None,
            "num_distinct_feasible": None,
            "feasibility_yield": None,
            "feasibility_fallback": None,
            "error": str(e),
        }


#: QPU metadata keys persisted on every result entry.  Listed
#: explicitly (rather than ``hw.copy()``) so the schema of the result
#: JSON is discoverable from this file and stays under harness control
#: even if hardware_summary() adds a field upstream.
_QPU_RESULT_FIELDS = (
    "chip_id",
    "problem_id",
    "problem_label",
    "physical_qubits",
    "logical_variables",
    "max_chain_length",
    "mean_chain_length",
    "chain_break_fraction",
    "embedding",
    "num_reads",
    "best_energy",
    "best_num_occurrences",
    # Sample-selection diagnostics (see util.sample_selection).  The
    # solver's selection_policy decides which sample becomes
    # entry["solution"]; these fields say which one was picked and why.
    "selection_policy",
    "selected_rank_by_energy",
    "selected_energy",
    "selected_num_occurrences",
    "num_feasible_reads",
    "num_distinct_feasible",
    "feasibility_yield",
    "feasibility_fallback",
    "qpu_access_time_us",
    "qpu_anneal_time_per_sample_us",
    "qpu_timing",
)


def _empty_qpu_fields():
    return {k: None for k in _QPU_RESULT_FIELDS}


def _run_qpu(solver_class, nodes, partitions, k_safety, requests, comm_costs,
             num_reads, annealing_time, chain_strength, solver_kwargs=None):
    try:
        solver_kwargs = solver_kwargs or {}
        solver = solver_class(nodes, partitions, k_safety, requests, comm_costs,
                              **solver_kwargs)
        bqm = solver.build_bqm()
        bqm_vars = len(bqm.variables)
        bqm_interactions = len(bqm.quadratic)

        kw = dict(num_reads=num_reads, annealing_time=annealing_time)
        if chain_strength is not None:
            kw["chain_strength"] = chain_strength

        t_ms, result = solver.solve(**kw)
        flat = _to_flat(result)
        cost = calculate_solution_cost(
            nodes, partitions, k_safety, requests, comm_costs, flat
        )
        valid = is_valid_solution(
            nodes, partitions, k_safety, requests, comm_costs, flat
        )
        k_viol, overruns = _violations(nodes, partitions, k_safety, flat)
        hw = solver.hardware_summary()

        entry = {
            "cost": cost, "valid": valid,
            "k_safety_violations": k_viol, "capacity_overruns": overruns,
            "wall_time_ms": round(t_ms, 1),
            "bqm_variables": bqm_vars,
            "bqm_interactions": bqm_interactions,
            "lambda_1": getattr(solver, "lambda_1", None),
            "lambda_2": getattr(solver, "lambda_2", None),
            "solution": _jsonable_solution(flat),
            "error": None,
        }
        # Pull every QPU metadata field through ``_QPU_RESULT_FIELDS``.
        # ``hardware_summary()`` may legitimately return None for any
        # of these (e.g. timing fields unavailable on a given QPU);
        # we record None rather than dropping the key, so the result
        # schema is stable across runs and QPUs.
        for field in _QPU_RESULT_FIELDS:
            entry[field] = hw.get(field)
        return entry
    except Exception as e:
        entry = {
            "cost": None, "valid": False,
            "k_safety_violations": None, "capacity_overruns": None,
            "wall_time_ms": None,
            "bqm_variables": None, "bqm_interactions": None,
            "lambda_1": None, "lambda_2": None,
            "solution": None,
            "error": str(e),
        }
        entry.update(_empty_qpu_fields())
        return entry


#: Hybrid-solver metadata keys persisted on every result entry.  The
#: Leap hybrid path has no embedding, chains, or physical qubits, so
#: this list deliberately omits the QPU-only fields (chip_id,
#: embedding, chain_break_fraction, ...) and adds the hybrid timing
#: breakdown (run_time / qpu_access_time) instead.  Listed explicitly
#: for the same reason as ``_QPU_RESULT_FIELDS``: the result-JSON
#: schema stays discoverable from this file and under harness control.
_HYBRID_RESULT_FIELDS = (
    "solver_name",
    "problem_id",
    "problem_label",
    "logical_variables",
    "num_samples",
    "best_energy",
    "best_num_occurrences",
    "run_time_us",
    "qpu_access_time_us",
    "hybrid_timing",
)


def _empty_hybrid_fields():
    return {k: None for k in _HYBRID_RESULT_FIELDS}


def _run_hybrid(solver_class, nodes, partitions, k_safety, requests, comm_costs,
                time_limit, solver_kwargs=None):
    """
    Run one Leap-hybrid solver on one case.

    Mirrors ``_run_qpu`` but for the hybrid path: the solver's
    ``solve()`` takes a single ``time_limit`` (seconds) instead of
    ``num_reads`` / ``annealing_time`` / ``chain_strength``, and metadata
    comes from ``hybrid_summary()`` rather than ``hardware_summary()``.
    ``time_limit=None`` lets the service pick its minimum for the
    problem size.
    """
    try:
        solver_kwargs = solver_kwargs or {}
        solver = solver_class(nodes, partitions, k_safety, requests, comm_costs,
                              **solver_kwargs)
        bqm = solver.build_bqm()
        bqm_vars = len(bqm.variables)
        bqm_interactions = len(bqm.quadratic)

        kw = {}
        if time_limit is not None:
            kw["time_limit"] = time_limit

        t_ms, result = solver.solve(**kw)
        flat = _to_flat(result)
        cost = calculate_solution_cost(
            nodes, partitions, k_safety, requests, comm_costs, flat
        )
        valid = is_valid_solution(
            nodes, partitions, k_safety, requests, comm_costs, flat
        )
        k_viol, overruns = _violations(nodes, partitions, k_safety, flat)
        hyb = solver.hybrid_summary()

        entry = {
            "cost": cost, "valid": valid,
            "k_safety_violations": k_viol, "capacity_overruns": overruns,
            "wall_time_ms": round(t_ms, 1),
            "bqm_variables": bqm_vars,
            "bqm_interactions": bqm_interactions,
            "lambda_1": getattr(solver, "lambda_1", None),
            "lambda_2": getattr(solver, "lambda_2", None),
            "solution": _jsonable_solution(flat),
            "error": None,
        }
        # Pull every hybrid metadata field through ``_HYBRID_RESULT_FIELDS``.
        # ``hybrid_summary()`` may legitimately return None for any of
        # these (e.g. a timing field a given solver version omits); we
        # record None rather than dropping the key so the result schema
        # is stable across runs and solver versions.
        for field in _HYBRID_RESULT_FIELDS:
            entry[field] = hyb.get(field)
        return entry
    except Exception as e:
        entry = {
            "cost": None, "valid": False,
            "k_safety_violations": None, "capacity_overruns": None,
            "wall_time_ms": None,
            "bqm_variables": None, "bqm_interactions": None,
            "lambda_1": None, "lambda_2": None,
            "solution": None,
            "error": str(e),
        }
        entry.update(_empty_hybrid_fields())
        return entry


# ---------------------------------------------------------------------------
# Test-case loading (with retry)
# ---------------------------------------------------------------------------

def _load_case_with_retry(tc_path, max_attempts=5, base_delay=1.0,
                          sleep=time.sleep, verbose=False):
    """
    Load ``(inputs, metadata)`` for one test case, retrying transient
    read failures with exponential backoff.

    Reading a test-case JSON can stall on cloud-synced or network
    volumes -- an iCloud-offloaded file on the macOS Desktop is the
    motivating case: the OS blocks while re-materialising the file and
    eventually surfaces ``TimeoutError``/``OSError`` (errno 60,
    ETIMEDOUT).  A truncated download can also yield a JSON parse error.
    Either way a single hiccup on case N used to abort the entire sweep
    after N-1 successful cases.  Backing off and retrying lets the stall
    settle (or the download finish) so the sweep continues; a genuinely
    unreadable file still raises after the attempt budget, at which point
    ``--resume`` can pick up the rest once the file is available.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            inputs = json_to_test_case(str(tc_path))
            metadata = load_test_case_metadata(str(tc_path))
            return inputs, metadata
        except (OSError, ValueError) as exc:  # ValueError covers JSONDecodeError
            last_exc = exc
            if attempt >= max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            if verbose:
                print(f"  read of {Path(tc_path).name} failed "
                      f"({type(exc).__name__}: {exc}); "
                      f"retry {attempt}/{max_attempts - 1} in {delay:.0f}s ...")
            sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_experiment(
    test_case_paths,
    solver_registry,
    output_dir,
    file_prefix="Experiment",
    num_reads=1000,
    num_sweeps=1000,
    beta_range=None,
    annealing_time=20,
    chain_strength=None,
    time_limit=None,
    note=None,
    verbose=True,
    num_reads_fn=None,
    chain_strength_fn=None,
    time_limit_fn=None,
    on_case_complete=None,
    resume=False,
):
    """
    Run every solver on every test case.

    ``num_reads_fn`` (optional): a callable ``(n_nodes, n_partitions) -> int``
    that returns the per-case ``num_reads`` to use for SQA / QPU solvers.
    When provided, the scalar ``num_reads`` argument is ignored for the
    sample call and the function's return value is used instead.  The
    per-case value is recorded on the result entry as
    ``case_num_reads`` so the schedule is recoverable from the output
    file.  Useful for scaling reads with problem size without running
    multiple experiments.

    ``chain_strength_fn`` (optional): same idea as ``num_reads_fn``, but
    for the QPU ``chain_strength`` parameter.  Signature
    ``(n_nodes, n_partitions) -> float``.  When provided, overrides the
    scalar ``chain_strength`` per case; the per-case value lands on the
    result entry as ``case_chain_strength``.  Useful when chain breaks
    scale with problem size and the default
    ``uniform_torque_compensation`` heuristic is too gentle on dense
    BQMs.

    ``on_case_complete`` (optional): a callable
    ``(case_key, entry, idx, total) -> None`` invoked after each case's
    result has been persisted to disk.  Used to drive interactive
    workflows (per-case reports, pause-for-input, abort prompts) without
    forking the harness loop.  The callback may raise to abort the
    sweep -- ``KeyboardInterrupt`` propagates naturally and the result
    file remains valid up to the last completed case.

    ``resume`` (optional): when True, continue the most recent
    ``{file_prefix}_<N>.json`` in ``output_dir`` instead of starting a
    fresh file.  Cases whose key is already present in that file are
    skipped (their work is already on disk), so an interrupted sweep --
    laptop slept, Ctrl-C, crash -- picks up where it left off rather than
    re-running everything.  Because the result file is written
    atomically after each case, the reloaded file is always parseable and
    contains only fully-completed cases.  If no matching file exists,
    ``resume`` falls back to starting a fresh one.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = output_dir.glob(f"{file_prefix}_*.json")
    numbers = [
        int(m.group(1))
        for f in existing
        if (m := re.search(rf"{file_prefix}_(\d+)\.json", f.name))
    ]

    output = None
    if resume and numbers:
        # Reattach to the highest-numbered existing file and reload its
        # completed cases.  Atomic writes guarantee it parses.
        resume_path = output_dir / f"{file_prefix}_{max(numbers)}.json"
        try:
            output = json.loads(resume_path.read_text())
            assert isinstance(output, dict) and "results" in output
            output_path = resume_path
            done = len(output["results"])
            if verbose:
                print(f"Resuming {resume_path.name}: "
                      f"{done}/{len(test_case_paths)} cases already done.")
        except (OSError, ValueError, AssertionError) as e:
            if verbose:
                print(f"Could not resume {resume_path.name} ({e}); "
                      f"starting a fresh file.")
            output = None

    if output is None:
        file_num = max(numbers, default=0) + 1
        output_path = output_dir / f"{file_prefix}_{file_num}.json"

    solver_names = [s["name"] for s in solver_registry]
    has_sqa = any(s["type"] == "sqa" for s in solver_registry)
    has_qpu = any(s["type"] == "qpu" for s in solver_registry)
    has_hybrid = any(s["type"] == "hybrid" for s in solver_registry)

    metadata = {
        "date": date.today().isoformat(),
        "time": datetime.now().strftime("%H:%M:%S"),
        "total_cases": len(test_case_paths),
        # Top-level num_reads is the scalar fallback / default.  When a
        # schedule is in use, the authoritative per-case value lives on
        # each entry as ``case_num_reads`` and the schedule itself is
        # recorded under ``num_reads_schedule``.
        "num_reads": num_reads if num_reads_fn is None else "variable",
        "solvers": solver_names,
        "harness_version": "phase5",
    }
    if num_reads_fn is not None:
        metadata["num_reads_schedule"] = (
            getattr(num_reads_fn, "__name__", "anonymous"),
            (getattr(num_reads_fn, "__doc__", None) or "").strip(),
        )
    if has_sqa:
        metadata["num_sweeps"] = num_sweeps
    if has_qpu:
        metadata["annealing_time"] = annealing_time
        # chain_strength: scalar, schedule, or omitted (use SDK default).
        if chain_strength_fn is not None:
            metadata["chain_strength"] = "variable"
            metadata["chain_strength_schedule"] = (
                getattr(chain_strength_fn, "__name__", "anonymous"),
                (getattr(chain_strength_fn, "__doc__", None) or "").strip(),
            )
        elif chain_strength is not None:
            metadata["chain_strength"] = chain_strength
    if has_hybrid:
        # time_limit: scalar, schedule, or omitted (let the hybrid
        # service choose its per-problem minimum).
        if time_limit_fn is not None:
            metadata["time_limit"] = "variable"
            metadata["time_limit_schedule"] = (
                getattr(time_limit_fn, "__name__", "anonymous"),
                (getattr(time_limit_fn, "__doc__", None) or "").strip(),
            )
        elif time_limit is not None:
            metadata["time_limit"] = time_limit
        else:
            metadata["time_limit"] = "service-default"

    # On resume, ``output`` already holds the reloaded file (metadata +
    # completed results); only initialise a fresh file when not resuming.
    if output is None:
        output = {"metadata": metadata, "results": {}}
        if note:
            output["metadata"]["note"] = note
        _write_json(output_path, output)

    total = len(test_case_paths)
    sweep_start = time.perf_counter()

    for idx, tc_path in enumerate(test_case_paths, 1):
        tc_path = Path(tc_path)
        # Case key: the file stem is NOT unique across the bank because
        # the tightness leaves (t30/t70/t90) under each (n, p) config all
        # hold an identically named instance file (e.g. n-3_p-4_1.json).
        # Keying on the stem alone let later tightness levels overwrite
        # earlier ones in ``results``, silently collapsing the tightness
        # axis and discarding the QPU work already spent on the
        # overwritten cases.  Fold the tightness leaf into the key when
        # the parent dir looks like one (``t\d+``); a numeric backstop
        # guarantees uniqueness so no case is ever dropped, whatever the
        # directory layout.  Stratification by (n, p, tightness) still
        # uses the n_nodes / n_partitions / tc_tightness fields on each
        # entry, so this only affects the dict key, not the analysis path.
        key = tc_path.stem
        if re.fullmatch(r"t\d+", tc_path.parent.name):
            key = f"{key}__{tc_path.parent.name}"

        # --resume: this case's result is already persisted; skip it.
        # Checked before the collision-dedup below so a completed key is
        # recognised as done rather than sidestepped into a new ``__N``
        # key (which would re-run it and duplicate the entry).
        if resume and key in output["results"]:
            if verbose:
                print(f"  [{idx}/{total}] {key}: already done, skipping")
            continue

        if key in output["results"]:
            dedup = 2
            while f"{key}__{dedup}" in output["results"]:
                dedup += 1
            key = f"{key}__{dedup}"

        (nodes, partitions, k_safety, requests, comm_costs), tc_metadata = (
            _load_case_with_retry(tc_path, verbose=verbose)
        )

        # Resolve the per-case num_reads.  If a schedule was provided,
        # it gets the (n_nodes, n_partitions) of this case and decides;
        # otherwise we fall back to the scalar.  Either way the value
        # is recorded on the entry so the schedule is recoverable from
        # the result JSON without re-running the schedule fn.
        case_num_reads = (
            int(num_reads_fn(len(nodes), len(partitions)))
            if num_reads_fn is not None
            else num_reads
        )
        # Same resolution rule for chain_strength, with one wrinkle: the
        # SDK accepts either a number (absolute Ising-unit coupling) or
        # a callable ``(bqm, embedding) -> float`` (typically
        # ``uniform_torque_compensation`` with a prefactor).  The latter
        # is the right way to scale chain strength relative to a BQM
        # whose coupling magnitudes you don't know in advance.  We pass
        # whichever the schedule produced through to the QPU solver, but
        # for the result file we coerce: numbers go in as floats,
        # callables as their repr (so the value is JSON-serialisable
        # and downstream analysis can still identify the schedule used).
        case_chain_strength_raw = (
            chain_strength_fn(len(nodes), len(partitions))
            if chain_strength_fn is not None
            else chain_strength
        )
        if case_chain_strength_raw is None:
            case_chain_strength_for_log = None
        elif isinstance(case_chain_strength_raw, (int, float)):
            case_chain_strength_for_log = float(case_chain_strength_raw)
        else:
            case_chain_strength_for_log = repr(case_chain_strength_raw)

        # Resolve the per-case time_limit (hybrid solvers).  Same rule
        # as num_reads: a schedule, if provided, decides from the case
        # size; otherwise the scalar (which may be None = let the
        # service pick its minimum) is used.  Recorded on the entry so
        # the schedule is recoverable from the result JSON.
        case_time_limit = (
            time_limit_fn(len(nodes), len(partitions))
            if time_limit_fn is not None
            else time_limit
        )

        entry = {
            "source_file": str(tc_path),
            "n_nodes": len(nodes),
            "n_partitions": len(partitions),
            "k_safety": k_safety,
            "case_num_reads": case_num_reads,
            "case_chain_strength": case_chain_strength_for_log,
            "case_time_limit": case_time_limit,
            # Phase-5 + tightness extension: surface test-case metadata
            # (notably ``tightness``) so downstream analysis can stratify
            # results without re-reading the source JSON.  Each metadata
            # field becomes a top-level key on the entry, prefixed with
            # ``tc_`` to avoid colliding with solver-result keys.
            **{f"tc_{k}": v for k, v in tc_metadata.items()},
            "solvers": {},
        }

        ilp_cost = None

        for solver_desc in solver_registry:
            name = solver_desc["name"]
            cls = solver_desc["class"]
            solver_type = solver_desc["type"]
            solver_kwargs = solver_desc.get("kwargs", {})

            # Per-solver chain_strength override: registry entries may
            # carry a ``chain_strength_fn`` of their own that takes
            # ``(n_nodes, n_partitions)`` and overrides the case-level
            # value for *that solver only*.  This is the lever for
            # cases where S1 and S2 have very different BQM densities
            # (S1's slack variables blow up |E(BQM)|) and therefore
            # need different chain strengths to keep cbf in check.
            # When no override is set, the case-level value (computed
            # from the harness's ``chain_strength_fn``) is used,
            # preserving the previous shared-schedule behaviour.
            per_solver_cs_fn = solver_desc.get("chain_strength_fn")
            if per_solver_cs_fn is not None:
                solver_cs_raw = per_solver_cs_fn(len(nodes), len(partitions))
            else:
                solver_cs_raw = case_chain_strength_raw

            if solver_type == "ilp":
                result = _run_ilp(cls, nodes, partitions, k_safety, requests, comm_costs)
                if result["valid"] and result["cost"] is not None:
                    ilp_cost = result["cost"]
            elif solver_type == "sqa":
                result = _run_sqa(
                    cls, nodes, partitions, k_safety, requests, comm_costs,
                    case_num_reads, num_sweeps, beta_range, solver_kwargs,
                )
            elif solver_type == "qpu":
                result = _run_qpu(
                    cls, nodes, partitions, k_safety, requests, comm_costs,
                    case_num_reads, annealing_time, solver_cs_raw,
                    solver_kwargs,
                )
                # Surface the actual per-solver chain_strength used so
                # the schedule is recoverable from the result JSON.
                # Numbers go in as floats; callables as their repr.
                if solver_cs_raw is None:
                    result["chain_strength_used"] = None
                elif isinstance(solver_cs_raw, (int, float)):
                    result["chain_strength_used"] = float(solver_cs_raw)
                else:
                    result["chain_strength_used"] = repr(solver_cs_raw)
            elif solver_type == "hybrid":
                result = _run_hybrid(
                    cls, nodes, partitions, k_safety, requests, comm_costs,
                    case_time_limit, solver_kwargs,
                )
            else:
                raise ValueError(f"Unknown solver type: {solver_type!r}")

            abs_gap, rel_gap = _gap(result.get("cost"), ilp_cost)
            result["optimality_gap_absolute"] = abs_gap
            result["optimality_gap_relative"] = rel_gap
            entry["solvers"][name] = result

        output["results"][key] = entry
        _write_json(output_path, output)

        if verbose:
            elapsed = time.perf_counter() - sweep_start
            rate = idx / elapsed if elapsed > 0 else 0
            eta_total = (total - idx) / rate if rate > 0 else 0
            eta_min = int(eta_total // 60)
            eta_sec = int(eta_total % 60)
            status_parts = []
            for s in solver_registry:
                r = entry["solvers"][s["name"]]
                tag = "OK" if r.get("valid") else "X"
                gap = r.get("optimality_gap_absolute")
                if gap is not None and gap != 0:
                    tag = f"{tag}(+{gap})" if r.get("valid") else tag
                status_parts.append(f"{s['name']}={tag}")
            print(
                f"  [{idx}/{total}] {key}: {'  '.join(status_parts)}"
                f"  [ETA {eta_min}m{eta_sec:02d}s]"
            )

        # Per-case hook fires after the result is on disk and the
        # verbose status line has printed.  Raising from here aborts
        # the sweep without corrupting the result file -- everything
        # up to and including this case is already persisted.
        if on_case_complete is not None:
            on_case_complete(key, entry, idx, total)

    elapsed_total = time.perf_counter() - sweep_start
    if verbose:
        print(f"\nCompleted {total} cases in {elapsed_total / 60:.1f} minutes.")
        print(f"Saved to: {output_path}")
    return output_path
