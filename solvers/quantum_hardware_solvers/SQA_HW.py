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
    - Valid LEAP API token configured via `dwave setup` or DWAVE_API_TOKEN env var
    - Capacities must be Mersenne numbers (same restriction as simulated S1)
"""

import time

from dwave.system import DWaveSampler, EmbeddingComposite
from solvers.simulated_solvers.SQA import SQASolver


class SQAHardwareSolver(SQASolver):
    """S1 on real D-Wave hardware."""

    def __init__(self, nodes, partitions, k_safety, requests, comm_costs,
                 solver_name=None):
        """
        Args:
            nodes, partitions, k_safety, requests, comm_costs:
                Standard problem definition (see SolverBase).
            solver_name:
                Optional D-Wave solver identifier, e.g. 'Advantage_system6.4'.
                If None, the client's default QPU is used.
        """
        super().__init__(nodes, partitions, k_safety, requests, comm_costs)
        self.solver_name = solver_name

        # Populated after solve()
        self.embedding = None
        self.qpu_timing = None
        self.chain_break_fraction = None
        self.physical_qubits = None
        self.sampleset = None

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
            (time_ms, result): wall-clock time in ms and the best sample
            (sampleset.first).  QPU-specific timing is stored in
            self.qpu_timing.
        """
        bqm = self.build_bqm()

        # --- Sampler setup ---
        qpu_kwargs = {}
        if self.solver_name is not None:
            qpu_kwargs['solver'] = self.solver_name

        sampler = EmbeddingComposite(DWaveSampler(**qpu_kwargs))

        sample_kwargs = dict(
            num_reads=num_reads,
            annealing_time=annealing_time,
        )
        if chain_strength is not None:
            sample_kwargs['chain_strength'] = chain_strength

        # --- Submit to QPU ---
        start = time.perf_counter()
        sampleset = sampler.sample(bqm, **sample_kwargs)
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

        self.time_taken = wall_time_ms
        self.result = sampleset.first

        return wall_time_ms, sampleset.first

    def hardware_summary(self):
        """Return a dict summarising QPU execution metadata."""
        qpu_access_us = self.qpu_timing.get('qpu_access_time', None)
        qpu_anneal_us = self.qpu_timing.get('qpu_anneal_time_per_sample', None)
        return {
            'wall_time_ms': round(self.time_taken, 1) if self.time_taken >= 0 else None,
            'qpu_access_time_us': qpu_access_us,
            'qpu_anneal_time_per_sample_us': qpu_anneal_us,
            'physical_qubits': self.physical_qubits,
            'logical_variables': len(self.sampleset.variables) if self.sampleset else None,
            'chain_break_fraction': self.chain_break_fraction,
            'num_reads': len(self.sampleset) if self.sampleset else None,
        }


def _chain_break_fraction(sampleset):
    """Compute the fraction of samples that contain at least one chain break."""
    if not hasattr(sampleset, 'record') or 'chain_break_fraction' not in sampleset.record.dtype.names:
        return None
    fractions = sampleset.record['chain_break_fraction']
    if len(fractions) == 0:
        return None
    return round(float(fractions.mean()), 4)
