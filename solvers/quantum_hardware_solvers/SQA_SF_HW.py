"""
S2 Hardware -- QPU version of the Slack-Free SQA solver.

Reuses the BQM construction from SQASlackFreeSolver (calibrated
unbalanced penalisation, no slack variables) and submits it to a real
D-Wave QPU.

Headline advantage over S1 on hardware: fewer logical variables (no
slack variables), so the embedding is smaller and fewer physical qubits
are consumed.  After Phase-2, the partition-size restriction that used
to apply to S2 is gone -- the unbalanced penalty's coefficients now
include ``size_p`` explicitly, so arbitrary integer partition sizes are
supported.

Lambda calibration is the same as the simulated S2 path: pass explicit
``lambda_1`` / ``lambda_2``, or pass neither and let the constructor
auto-calibrate via ``dimod.ExactSolver`` (small instances) or the
heuristic fallback (large instances).  Calibration runs locally and
does *not* require a QPU call.

Usage:
    solver = SQASFHardwareSolver(nodes, partitions, k_safety, requests, comm_costs)
    time_ms, result = solver.solve(num_reads=100, annealing_time=20)
"""

import time

from dwave.system import DWaveSampler, EmbeddingComposite
from solvers.quantum_hardware_solvers._hw_common import (
    DEFAULT_EMBED_TIMEOUT,
    chain_break_fraction as _chain_break_fraction,
    extract_hardware_summary,
    submit_with_retry,
)
from solvers.simulated_solvers.SQA_SF import SQASlackFreeSolver
from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    VALID_POLICIES,
    select_sample,
)


class SQASFHardwareSolver(SQASlackFreeSolver):
    """S2 (slack-free) on real D-Wave hardware."""

    def __init__(self, nodes, partitions, k_safety, requests, comm_costs,
                 solver_name=None, selection_policy=POLICY_BEST_FEASIBLE):
        """
        Args:
            nodes, partitions, k_safety, requests, comm_costs:
                Standard problem definition (see SolverBase).  Arbitrary
                integer partition sizes are supported -- ``size_p`` enters
                the unbalanced penalty's coefficients explicitly (see the
                module docstring and ``build_bqm``); it is not restricted
                to 1.
            solver_name:
                Optional D-Wave solver identifier, e.g. 'Advantage_system6.4'.
                If None, the client's default QPU is used.
            selection_policy:
                'best_feasible' (default) or 'lowest_energy' -- see
                ``util.sample_selection``.
        """
        super().__init__(nodes, partitions, k_safety, requests, comm_costs)
        self.solver_name = solver_name
        if selection_policy not in VALID_POLICIES:
            raise ValueError(
                f"selection_policy must be one of {VALID_POLICIES}, "
                f"got {selection_policy!r}"
            )
        self.selection_policy = selection_policy

        # Populated after solve()
        self.embedding = None
        self.qpu_timing = None
        self.chain_break_fraction = None
        self.physical_qubits = None
        self.sampleset = None
        self.chip_id = None
        self.problem_id = None
        self.selection_diagnostics = None

    # build_bqm() is inherited unchanged from SQASlackFreeSolver.

    def solve(self, num_reads=100, annealing_time=20, chain_strength=None,
              embed_timeout=DEFAULT_EMBED_TIMEOUT):
        """
        Build the BQM and submit it to the D-Wave QPU.

        Args:
            num_reads:      Number of annealing cycles (samples).
            annealing_time: Anneal duration in microseconds (default 20).
            chain_strength: Coupling strength for physical qubit chains.
                            If None, the default heuristic is used.
            embed_timeout:  Wall-clock cap (seconds) on the client-side
                            minor-embedding search.  Defaults to 5 minutes
                            (DEFAULT_EMBED_TIMEOUT) so an unembeddable BQM
                            fails fast instead of burning minorminer's
                            1000 s default.  Pass None for the SDK default.

        Returns:
            (time_ms, result): wall-clock time in ms and the sample
            picked by ``self.selection_policy`` (see __init__).
            QPU-specific timing is stored in self.qpu_timing.
        """
        bqm = self.build_bqm()

        # --- Sampler setup ---
        qpu_kwargs = {}
        if self.solver_name is not None:
            qpu_kwargs['solver'] = self.solver_name

        qpu = DWaveSampler(**qpu_kwargs)
        sampler = EmbeddingComposite(qpu)

        # Capture chip_id eagerly: the underlying sampler may be
        # garbage-collected before hardware_summary() is called.
        try:
            self.chip_id = qpu.properties.get('chip_id')
        except Exception:
            self.chip_id = None

        sample_kwargs = dict(
            num_reads=num_reads,
            annealing_time=annealing_time,
            # EmbeddingComposite defaults this to False in current
            # versions of dwave-system; without it,
            # sampleset.info['embedding_context'] is empty and we lose
            # the chain dict + physical-qubit count.
            return_embedding=True,
        )
        if chain_strength is not None:
            sample_kwargs['chain_strength'] = chain_strength
        # Cap the minor-embedding search so an unembeddable BQM fails in
        # ~embed_timeout instead of minorminer's 1000 s default.  Passed
        # through EmbeddingComposite to minorminer.find_embedding.
        if embed_timeout is not None:
            sample_kwargs['embedding_parameters'] = {'timeout': embed_timeout}

        # --- Submit to QPU (bounded retry/backoff on transient errors) ---
        start = time.perf_counter()
        sampleset = submit_with_retry(sampler, bqm, sample_kwargs)
        end = time.perf_counter()

        wall_time_ms = (end - start) * 1000

        # --- Store rich metadata ---
        self.sampleset = sampleset
        self.qpu_timing = sampleset.info.get('timing', {})
        self.embedding = sampleset.info.get('embedding_context', {}).get('embedding', {})
        self.physical_qubits = (
            sum(len(chain) for chain in self.embedding.values())
            if self.embedding else None
        )
        self.chain_break_fraction = _chain_break_fraction(sampleset)
        self.problem_id = sampleset.info.get('problem_id')

        # --- Pick the reported result per the configured policy ---
        selected, sel_diag = select_sample(
            sampleset,
            self.nodes, self.partitions, self.k_safety,
            self.requests, self.comm_costs,
            policy=self.selection_policy,
        )
        self.selection_diagnostics = sel_diag

        self.time_taken = wall_time_ms
        self.result = selected

        return wall_time_ms, selected

    def hardware_summary(self):
        """Return a dict summarising QPU execution metadata."""
        return extract_hardware_summary(
            sampleset=self.sampleset,
            embedding=self.embedding,
            wall_time_ms=self.time_taken,
            chip_id=self.chip_id,
            selection_diagnostics=self.selection_diagnostics,
        )
