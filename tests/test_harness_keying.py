"""
Regression test for the experiment-harness case-key collision.

Instance files in the test bank are named identically across tightness
leaves (e.g. ``n-3_p-4_1.json`` exists under ``t30/``, ``t70/`` and
``t90/``).  The harness used to key results on the bare file *stem*, so
the three tightness levels of a given (n, p) collapsed onto one key and
silently overwrote each other -- a 36-case sweep produced 12 entries and
the tightness axis was unrecoverable.

The fix folds the ``t\\d+`` tightness-leaf directory into the key.  This
test pins that behaviour: two instance files with the *same stem* under
different tightness parents must both survive as distinct result entries,
each carrying its own ``tc_tightness``.

Uses an SQA-only registry so the test needs neither a QPU nor the ILP
(pulp) dependency -- the keying logic under test is solver-agnostic.
"""

import json
from pathlib import Path

from solvers.simulated_solvers.SQA import SQASolver
from util.experiment_execution.run_experiment import run_experiment


def _write_instance(path: Path, tightness: float):
    """Write a tiny 2-node / 2-partition instance with the given tightness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    case = {
        "nodes": {"n1": 8, "n2": 8},
        "partitions": {"p1": 2, "p2": 3},
        "k_safety": 1,
        "requests": {
            "(p1, n1)": 4, "(p1, n2)": 1,
            "(p2, n1)": 2, "(p2, n2)": 5,
        },
        "comm_costs": {"p1": 3, "p2": 2},
        "tightness": tightness,
    }
    path.write_text(json.dumps(case))


def test_identical_stems_under_tightness_leaves_do_not_collide(tmp_path):
    # Same stem ("inst_1") under two different t<NN> tightness leaves --
    # exactly the bank layout that triggered the collapse.
    p_t30 = tmp_path / "n2_p2" / "t30" / "inst_1.json"
    p_t90 = tmp_path / "n2_p2" / "t90" / "inst_1.json"
    _write_instance(p_t30, tightness=0.3)
    _write_instance(p_t90, tightness=0.9)

    registry = [{"name": "SQA", "class": SQASolver, "type": "sqa"}]

    out = run_experiment(
        test_case_paths=[p_t30, p_t90],
        solver_registry=registry,
        output_dir=tmp_path / "out",
        file_prefix="KeyCollision",
        num_reads=20,
        num_sweeps=20,
        verbose=False,
    )
    results = json.loads(Path(out).read_text())["results"]

    # Both cases must survive -- the bug produced exactly one entry here.
    assert len(results) == 2, (
        f"expected 2 distinct result entries, got {len(results)}: "
        f"{sorted(results)} -- identical stems under different tightness "
        f"leaves collided."
    )

    # Keys must disambiguate by the tightness leaf.
    assert any(k.endswith("__t30") for k in results), sorted(results)
    assert any(k.endswith("__t90") for k in results), sorted(results)

    # Both tightness values propagated through to the entries.
    tightnesses = sorted(entry["tc_tightness"] for entry in results.values())
    assert tightnesses == [0.3, 0.9], tightnesses


def test_unique_stems_keep_bare_keys(tmp_path):
    """
    The disambiguation must not disturb the common case: when stems are
    already unique (or the parent isn't a tightness leaf), keys stay as
    the bare stem so existing banks and the smoke test are unaffected.
    """
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_instance(a, tightness=0.3)
    _write_instance(b, tightness=0.9)

    registry = [{"name": "SQA", "class": SQASolver, "type": "sqa"}]
    out = run_experiment(
        test_case_paths=[a, b],
        solver_registry=registry,
        output_dir=tmp_path / "out",
        file_prefix="BareKeys",
        num_reads=20,
        num_sweeps=20,
        verbose=False,
    )
    results = json.loads(Path(out).read_text())["results"]
    assert set(results) == {"a", "b"}, sorted(results)
