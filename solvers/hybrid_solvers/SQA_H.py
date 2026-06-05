"""
S1 Hybrid — Leap-hybrid version of the standard SQA solver.

Reuses the BQM construction from SQASolver (binary slack variables for
storage constraints, quadratic penalty for k-safety) and submits it to
a D-Wave Leap *hybrid* BQM solver via ``LeapHybridSampler`` rather than
to a bare QPU.

The BQM is identical to the simulated and pure-QPU versions; only the
sampler changes.  Unlike the pure-QPU path there is no minor-embedding
step on the client side: the hybrid service embeds (and discards its
embedding) internally, so this solver does *not* expose chain strength,
chain-break fraction, or a physical-qubit count.  Instead of
``num_reads`` / ``annealing_time`` the hybrid solver is driven by a
single ``time_limit`` (seconds).

Because the hybrid service handles its own decomposition, the practical
problem-size ceiling is far higher than the pure-QPU path: the slack
variables that make S1 expensive on hardware are no longer gated by an
embedding into a sparse topology.

Usage:
    solver = SQAHybridSolver(nodes, partitions, k_safety, requests, comm_costs)
    time_ms, result = solver.solve()                 # service-chosen time limit
    time_ms, result = solver.solve(time_limit=5)     # explicit 5 s limit

Requirements:
    - dwave-system  (pip install dwave-system)
    - Valid LEAP API token configured via ``dwave setup`` or DWAVE_API_TOKEN env var
"""

import time

from dwave.system import LeapHybridSampler
from solvers.hybrid_solvers._hybrid_common import extract_hybrid_summary
from solvers.simulated_solvers.SQA import SQASolver


class SQAHybridSolver(SQASolver):
    """S1 on a D-Wave Leap hybrid BQM solver."""

    def __init__(self, nodes, partitions, k_safety, requests, comm_costs,
                 solver_name=None):
        """
        Args:
            nodes, partitions, k_safety, requests, comm_costs:
                Standard problem definition (see SolverBase).
            solver_name:
                Optional Leap hybrid solver identifier, e.g.
                'hybrid_binary_quadratic_model_version2'.  If None, the
                client's default hybrid BQM solver is used.
        """
        super().__init__(nodes, partitions, k_safety, requests, comm_costs)
        self.solver_name = solver_name

        # Populated after solve()
        self.hybrid_timing = None
        self.sampleset = None
        self.solver_id = None
        self.problem_id = None

    # build_bqm() is inherited unchanged from SQASolver.

    def solve(self, time_limit=None):
        """
        Build the BQM and submit it to a Leap hybrid BQM solver.

        Args:
            time_limit: Solver run time in **seconds**.  If None, the
                        service picks the minimum time limit appropriate
                        for the problem size (see
                        ``LeapHybridSampler.min_time_limit``).  Larger
                        values give the hybrid heuristic more time and
                        generally improve solution quality at higher
                        Leap-quota cost.

        Returns:
            (time_ms, result): wall-clock time in ms and the best sample
            (sampleset.first).  Hybrid-side timing is stored in
            self.hybrid_timing.
        """
        bqm = self.build_bqm()

        # --- Sampler setup ---
        sampler_kwargs = {}
        if self.solver_name is not None:
            sampler_kwargs['solver'] = self.solver_name

        sampler = LeapHybridSampler(**sampler_kwargs)

        # Capture the solver id eagerly: the sampler may be
        # garbage-collected before hybrid_summary() is called.
        try:
            self.solver_id = sampler.solver.id
        except Exception:
            self.solver_id = self.solver_name

        sample_kwargs = {}
        if time_limit is not None:
            sample_kwargs['time_limit'] = time_limit

        # --- Submit to the hybrid solver ---
        start = time.perf_counter()
        sampleset = sampler.sample(bqm, **sample_kwargs)
        end = time.perf_counter()

        wall_time_ms = (end - start) * 1000

        # --- Store rich metadata ---
        self.sampleset = sampleset
        self.hybrid_timing = sampleset.info.get('timing', {})
        self.problem_id = sampleset.info.get('problem_id')

        self.time_taken = wall_time_ms
        self.result = sampleset.first

        return wall_time_ms, sampleset.first

    def hybrid_summary(self):
        """Return a dict summarising hybrid-solver execution metadata."""
        return extract_hybrid_summary(
            sampleset=self.sampleset,
            wall_time_ms=self.time_taken,
            solver_name=self.solver_id,
        )
