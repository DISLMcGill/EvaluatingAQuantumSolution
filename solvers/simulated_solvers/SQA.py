"""
S1 — SQA baseline solver.

Faithful implementation of Paper 1 (Trummer 2025): QUBO formulation with
binary slack variables S_{i,n} encoding the storage inequality
    sum_p A_{p,n} * size_p  <=  C_n
as an exact equality
    sum_p A_{p,n} * size_p  ==  sum_i i * S_{i,n}.

To encode every value in 0..C_n exactly, we use a binary expansion of
C_n: chunks {1, 2, 4, ..., 2^J} where 2^(J+1) - 1 <= C_n, plus a residual
chunk of value (C_n - (2^(J+1) - 1)) if C_n is not Mersenne. The chunk
values sum to exactly C_n, so the slack variables can represent any
storage usage in [0, C_n] -- no over-loose constraint, no Mersenne
requirement.
"""

import time

import dimod
import pandas as pd
from dwave.samplers import PathIntegralAnnealingSampler

from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    VALID_POLICIES,
    select_sample,
)
from util.solver_base import SolverBase


def _binary_chunks(capacity):
    """
    Return chunk values whose sum equals ``capacity``.

    Uses the standard binary expansion (1, 2, 4, ...) up to the largest
    power of two whose cumulative sum does not exceed ``capacity``, then
    appends a single residual chunk for the remainder. The resulting
    chunks let a sum of selected chunks represent every integer in
    [0, capacity] exactly.

    Examples:
        capacity=7  -> [1, 2, 4]           (Mersenne; no residual)
        capacity=10 -> [1, 2, 4, 3]        (3 residual after 1+2+4=7)
        capacity=1  -> [1]
        capacity=0  -> []
    """
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative, got {capacity}")
    chunks = []
    cumulative = 0
    j = 0
    while cumulative + (1 << j) <= capacity:
        chunks.append(1 << j)
        cumulative += 1 << j
        j += 1
    residual = capacity - cumulative
    if residual > 0:
        chunks.append(residual)
    assert sum(chunks) == capacity, (
        f"chunk decomposition broken: {chunks} sums to {sum(chunks)}, "
        f"expected {capacity}"
    )
    return chunks


class SQASolver(SolverBase):
    def __init__(self, nodes, partitions, k_safety, requests, comm_costs,
                 selection_policy=POLICY_BEST_FEASIBLE):
        SolverBase.__init__(self, nodes, partitions, k_safety, requests, comm_costs)
        if selection_policy not in VALID_POLICIES:
            raise ValueError(
                f"selection_policy must be one of {VALID_POLICIES}, "
                f"got {selection_policy!r}"
            )
        self.selection_policy = selection_policy
        self.selection_diagnostics = None

    def build_bqm(self):
        """Build and return the BinaryQuadraticModel without solving."""
        bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

        # 1. Define Variables
        # A_pn: partition p assigned to node n
        A = {(p, n): f"A_{p}_{n}" for p in self.partitions for n in self.nodes}

        # S_{n,c}: binary slack variables; each S has a chunk value c.
        #   Variable name uses the chunk *index* (0, 1, 2, ...) so that
        #   two chunks with the same value (impossible here, but defensive)
        #   would not collide.
        S = {}                          # (n, chunk_index) -> var_name
        chunks_by_node = {}             # n -> [chunk_values]
        for n, capacity in self.nodes.items():
            chunks = _binary_chunks(int(capacity))
            chunks_by_node[n] = chunks
            for idx, val in enumerate(chunks):
                S[(n, idx)] = f"S_{n}_{idx}"

        # 2. Penalty weight (Paper 1, Eq. 9): h > sum(r_pn * c_p) suffices.
        h = sum(
            self.requests[p, n] * self.comm_costs[p]
            for p in self.partitions for n in self.nodes
        ) + 1

        # k-safety multiplier: scale h by max(size_p)**2.
        #
        # The storage equality below is squared into the BQM, so its
        # quadratic penalty terms on the assignment variables scale as
        # ``size_p * size_p'`` -- i.e. up to ``smax**2`` where
        # ``smax = max_p size_p``.  With Paper 1's single ``h`` shared by
        # both constraint families this is fine for unit partitions
        # (smax = 1) but, for arbitrary ``size_p``, the storage penalty
        # outweighs the O(h) k-safety penalty by a factor of smax**2.
        # The sampler then satisfies the storage constraint and ignores
        # k-safety, so *zero* reads come out jointly feasible (observed on
        # both the QPU and the PathIntegral sampler: feasibility_yield ->
        # 0 as size_p and |P| grow).  Scaling the k-safety multiplier by
        # smax**2 restores the balance and lifts feasibility yield from ~0
        # back to ~0.5 on the arbitrary tier-1 cases, while leaving the
        # unit case byte-identical (smax = 1 -> h_k = h).  This mirrors
        # S2's ``h_k = h * max_C`` rationale ("k-safety must dominate"),
        # but keyed to the quantity that actually inflates S1's storage
        # term -- the partition size, not the node capacity.  The storage
        # multiplier itself stays at the proven-sufficient value ``h``, so
        # the feasible ground state is preserved.
        smax = max((int(s) for s in self.partitions.values()), default=1)
        h_k = h * smax * smax

        # 3. Q_R: k-safety constraints
        for p in self.partitions:
            k_safety_expr = [(A[p, n], 1) for n in self.nodes]
            bqm.add_linear_equality_constraint(
                k_safety_expr,
                constant=-self.k_safety,
                lagrange_multiplier=h_k,
            )

        # 4. Q_S: storage constraint as equality
        #    sum_p A_pn * size_p  ==  sum_i chunk_i * S_{n,i}
        for n, _capacity in self.nodes.items():
            storage_expr = []
            for p in self.partitions:
                storage_expr.append((A[p, n], int(self.partitions[p])))
            for idx, val in enumerate(chunks_by_node[n]):
                storage_expr.append((S[(n, idx)], -int(val)))
            bqm.add_linear_equality_constraint(
                storage_expr, constant=0, lagrange_multiplier=h
            )

        # 5. Q_C: processing costs (Paper 1, Eq. 5; constants dropped)
        for p in self.partitions:
            for n in self.nodes:
                bqm.add_variable(
                    A[p, n], -self.requests[p, n] * self.comm_costs[p]
                )

        return bqm

    def solve(self, num_reads=1000, num_sweeps=1000, beta_range=None):
        bqm = self.build_bqm()
        sampler = PathIntegralAnnealingSampler()
        sample_kwargs = dict(num_reads=num_reads, num_sweeps=num_sweeps)
        if beta_range is not None:
            sample_kwargs["beta_range"] = beta_range

        start = time.perf_counter()
        sampleset = sampler.sample(bqm, **sample_kwargs)
        end = time.perf_counter()

        time_taken = (end - start) * 1000
        selected, sel_diag = select_sample(
            sampleset,
            self.nodes, self.partitions, self.k_safety,
            self.requests, self.comm_costs,
            policy=self.selection_policy,
        )
        self.time_taken = time_taken
        self.result = selected
        self.selection_diagnostics = sel_diag
        return time_taken, selected

    def format_answer(self, result=None):
        sample_obj = result if result is not None else self.result
        if sample_obj is None:
            print("No valid solution found.")
            return

        best_sample = sample_obj.sample
        allocation_data = []
        for p in self.partitions:
            row = {"Partition": p}
            for n in self.nodes:
                row[n] = best_sample[f"A_{p}_{n}"]
            allocation_data.append(row)

        matrix_df = pd.DataFrame(allocation_data).set_index("Partition")
        print("--- Data Allocation Matrix ---")
        print(matrix_df)
