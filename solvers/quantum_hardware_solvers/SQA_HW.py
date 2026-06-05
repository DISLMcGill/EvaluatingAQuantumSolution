"""
S1 Hardware — QPU version of the standard SQA solver.

Reuses the BQM construction from SQASolver (binary slack variables for
storage constraints, quadratic penalty for k-safety) and submits it to
a real D-Wave quantum annealer via EmbeddingComposite.

The BQM is identical to the simulated version; only the sampler changes.

Usage:
    solver = SQAHardwareSolver(nodes, partitions, k_safety, requests, comm_costs)
    time_ms, result = solver.solve(num_reads=100, annealing_time=20)

Requirements:
    - dwave-system  (pip install dwave-system)
    - Valid LEAP API token configured via ``dwave setup`` or DWAVE_API_TOKEN env var

After the Phase-1 fix, S1 supports arbitrary integer capacities -- the
slack-chunk decomposition was generalised from the Mersenne-only binary
expansion to ``[1, 2, 4, ..., 2^J, residual]`` so that the chunks sum
to exactly ``C_n``.  The hardware solver inherits this fix unchanged.
"""

import time

from dwave.system import DWaveSampler, EmbeddingComposite
from solvers.quantum_hardware_solvers._hw_common import (
    chain_break_fraction as _chain_break_fraction,
    extract_hardware_summary,
    submit_with_retry,
)
from solvers.simulated_solvers.SQA import SQASolver
from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    VALID_POLICIES,
    select_sample,
)


class SQAHardwareSolver(SQASolver):
    """S1 on real D-Wave hardware."""

    def __init__(self, nodes, partitions, k_safety, requests, comm_costs,
                 solver_name=None, selection_policy=POLICY_BEST_FEASIBLE):
        """
        Args:
            nodes, partitions, k_safety, requests, comm_costs:
                Standard problem definition (see SolverBase).
            solver_name:
                Optional D-Wave solver identifier, e.g. 'Advantage_system6.4'.
                If None, the client's default QPU is used.
            selection_policy:
                'best_feasible' (default) -- return the lowest-energy
                feasible sample, falling back to the lowest-energy
                sample overall when no read is feasible.
                'lowest_energy' -- legacy behaviour: return the
                lowest-energy sample regardless of feasibility.
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

    # build_bqm() is inherited unchanged from SQASolver.

    def solve(self, num_reads=100, annealing_time=20, chain_strength=None):
        """
        Build the BQM and submit it to the D-Wave QPU.

        Args:
            num_reads:      Number of annealing cycles (samples). Hardware
                            default is typically 1-10000.  Lower than the
                            simulated default because each read is a real
                            anneal, not a Monte Carlo sweep.
            annealing_time: Anneal duration in microseconds (default 20).
                            Range depends on the QPU; Advantage supports
                            roughly 0.5-2000 us.
            chain_strength: Coupling strength for physical qubit chains.
                            If None, EmbeddingComposite uses its default
                            heuristic (uniform_torque_compensation).

        Returns:
            (time_ms, result): wall-clock time in ms and the selected
            sample.  Which sample is "selected" depends on
            ``self.selection_policy``:
              * 'best_feasible' (default): lowest-energy sample that
                satisfies every k-safety and capacity constraint, or
                ``sampleset.first`` if no read is feasible (in which
                case ``self.selection_diagnostics['feasibility_fallback']``
                is True).
              * 'lowest_energy': ``sampleset.first`` unconditionally.
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
