"""
S3 Hybrid -- Leap-hybrid version of the Domain-Wall + Unbalanced-Penalty
solver.

Reuses the BQM construction from SQADomainWallSolver (Chancellor-style
domain-wall k-safety chain + calibrated unbalanced storage) and submits
it to a D-Wave Leap *hybrid* BQM solver via ``LeapHybridSampler``.

S3 is **opt-in** even on the hybrid path -- it is excluded from the
default solver registry and requires ``include_s3=True`` to be included.
The reason mirrors the simulated and pure-QPU sides: the W-to-A linking
constraint reintroduces O(|N|^2) couplings, so the advertised coupling
reduction does not materialise on the data-allocation problem.

The original motivation for the domain-wall encoding was to reduce
chain length and coupling on a sparse QPU topology.  A hybrid solver
hides embedding entirely, so that motivation is even weaker here than
on hardware: the domain-wall machinery adds variables and couplings to
the BQM handed to the hybrid heuristic without any compensating
embedding benefit.  The file is kept for reproducibility / parity with
the other solver families, not advocacy.

Like the other hybrid solvers, this one is driven by a single
``time_limit`` (seconds) and exposes no chain or physical-qubit
metadata.

Usage (opt in):
    solver = SQADWHybridSolver(nodes, partitions, k_safety, requests, comm_costs)
    time_ms, result = solver.solve(time_limit=5)
"""

import time

from dwave.system import LeapHybridSampler
from solvers.hybrid_solvers._hybrid_common import extract_hybrid_summary
from solvers.simulated_solvers.SQA_DW import SQADomainWallSolver


class SQADWHybridSolver(SQADomainWallSolver):
    """S3 (domain-wall + slack-free) on a D-Wave Leap hybrid BQM solver."""

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

    # build_bqm() is inherited unchanged from SQADomainWallSolver.

    def solve(self, time_limit=None):
        """
        Build the BQM and submit it to a Leap hybrid BQM solver.

        Args:
            time_limit: Solver run time in **seconds**.  If None, the
                        service picks the minimum time limit appropriate
                        for the problem size.

        Returns:
            (time_ms, result): wall-clock time in ms and the best sample.
            Hybrid-side timing is stored in self.hybrid_timing.
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
