"""
S3 Hardware -- QPU version of the Domain-Wall + Unbalanced-Penalty solver.

Reuses the BQM construction from SQADomainWallSolver (Chancellor-style
domain-wall k-safety chain + calibrated unbalanced storage) and submits
it to a real D-Wave QPU.

S3 is **opt-in** even on hardware -- it is excluded from the default
``_get_hw_registry()`` and requires ``include_s3=True`` to be included.
The reason is the same one that excludes it on the simulated side:
the W-to-A linking constraint reintroduces O(|N|^2) couplings, so the
advertised coupling reduction does not materialise on the data-
allocation problem.  On sparse hardware topologies this typically
makes embeddings strictly worse than S2's.

The original docstring framed S3 as a key hypothesis to test on
hardware.  After the audit and the Phase-3 rewrite, the hypothesis is
falsified at the encoding level on this problem class; the file is
kept for reproducibility, not advocacy.

Usage (opt in):
    from util.experiment_execution.run_unit_partition_experiment import (
        run_unit_experiment,
    )
    run_unit_experiment(hardware=True, include_s3=True)
"""

import time

from dwave.system import DWaveSampler, EmbeddingComposite
from solvers.quantum_hardware_solvers._hw_common import (
    chain_break_fraction as _chain_break_fraction,
    extract_hardware_summary,
)
from solvers.simulated_solvers.SQA_DW import SQADomainWallSolver
from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    VALID_POLICIES,
    select_sample,
)


class SQADWHardwareSolver(SQADomainWallSolver):
    """S3 (domain-wall + slack-free) on real D-Wave hardware."""

    def __init__(self, nodes, partitions, k_safety, requests, comm_costs,
                 solver_name=None, selection_policy=POLICY_BEST_FEASIBLE):
        """
        Args:
            nodes, partitions, k_safety, requests, comm_costs:
                Standard problem definition (see SolverBase).
                All partition sizes must be 1.
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

    # build_bqm() is inherited unchanged from SQADomainWallSolver.

    def solve(self, num_reads=100, annealing_time=20, chain_strength=None):
        """
        Build the BQM and submit it to the D-Wave QPU.

        Args:
            num_reads:      Number of annealing cycles (samples).
            annealing_time: Anneal duration in microseconds (default 20).
            chain_strength: Coupling strength for physical qubit chains.
                            If None, the default heuristic is used.

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
