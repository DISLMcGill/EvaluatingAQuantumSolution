"""
S2 — SQA Slack-Free Unit-Partition Solver.

Eliminates all S_in slack variables by encoding the storage inequality
constraint directly via an unbalanced penalty function.

With unit partition sizes the storage constraint becomes a simple
cardinality constraint:  sum(A_pn over p) <= capacity_n  for each node n.

Instead of converting this to an equality with slack variables we penalise
violations quadratically:

    Q_S(n) = h * max(0,  sum(A_pn) - C_n )^2

For binary variables, (sum A_pn)^2  =  sum_p A_pn  +  2 * sum_{p<p'} A_pn * A_p'n
(since A_pn^2 = A_pn for binary).  Combining with the linear reward term
-2*C_n * sum A_pn gives a penalty that is zero when sum <= C_n and positive
when sum > C_n, using only A_pn variables — no auxiliary slack variables.

**Capacity does NOT need to be a Mersenne number.**  Any positive integer
capacity is supported.

Reference:
    Montañez-Barrera et al. (2022). "Unbalanced penalization: A new approach
    to encode inequality constraints for quantum optimization algorithms."
    arXiv:2211.13914.
"""

import time

import dimod
import pandas as pd
from dwave.samplers import PathIntegralAnnealingSampler
from util.solver_base import SolverBase


class SQASlackFreeSolver(SolverBase):
    def __init__(self, nodes, partitions, k_safety, requests, comm_costs):
        for p, size in partitions.items():
            if size != 1:
                raise ValueError(
                    f"SQASlackFreeSolver requires all partition sizes = 1, "
                    f"but partition {p} has size {size}"
                )
        SolverBase.__init__(self, nodes, partitions, k_safety, requests, comm_costs)

    def build_bqm(self):
        """Build and return the BQM without solving."""
        bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

        # 1. Assignment variables A_pn  (the ONLY variables — no slack)
        A = {(p, n): f'A_{p}_{n}' for p in self.partitions for n in self.nodes}

        partition_list = list(self.partitions.keys())
        node_list = list(self.nodes.keys())

        # 2. Penalty weights
        h = sum(
            self.requests[p, n] * self.comm_costs[p]
            for p in self.partitions for n in self.nodes
        ) + 1

        # Scale k-safety penalty by max capacity so it always dominates
        # the under-capacity storage penalty for feasible assignments.
        h_k = h * max(self.nodes.values())

        # 3. Q_R: k-safety constraints  —  (sum_n A_pn - k)^2 == 0
        for p in self.partitions:
            k_safety_expr = [(A[p, n], 1) for n in self.nodes]
            bqm.add_linear_equality_constraint(
                k_safety_expr, constant=-self.k_safety, lagrange_multiplier=h_k
            )

        # 4. Q_S: storage constraints  (SLACK-FREE)
        #
        #    For each node n with capacity C_n we want:
        #        sum_p A_pn  <=  C_n
        #
        #    Unbalanced penalty:
        #        P(x) = (x - C)(x - C + 1)
        #
        #    Properties:
        #      P(C)   = 0      (at capacity — no penalty)
        #      P(C-1) = 0      (one below capacity — no penalty)
        #      P(C+1) = 2      (over capacity — penalised)
        #      P(C+2) = 6      (further over — heavily penalised)
        #      P(C-2) = 2      (far under — mild penalty, prevents waste)
        #
        #    Expansion for binary variables (x^2 = x + 2*cross):
        #      P = x^2 + (1-2C)x + C^2 - C
        #        = x + 2*cross + (1-2C)*x + const
        #        = (2-2C)*x + 2*cross + const
        #
        #    Linear per A_pn:     h_s * (2 - 2*C)
        #    Quadratic per pair:  2 * h_s

        h_s = h  # storage penalty weight (can be tuned separately)

        for n in node_list:
            capacity = self.nodes[n]

            # Quadratic terms: A_pn * A_p'n for all p < p'
            for i in range(len(partition_list)):
                for j in range(i + 1, len(partition_list)):
                    p1 = partition_list[i]
                    p2 = partition_list[j]
                    bqm.add_interaction(A[p1, n], A[p2, n], 2 * h_s)

            # Linear terms: unbalanced coefficient (2-2C) instead of (1-2C)
            for p in partition_list:
                bqm.add_variable(A[p, n], h_s * (2 - 2 * capacity))

        # 5. Q_C: processing costs
        for p in self.partitions:
            for n in self.nodes:
                bqm.add_variable(A[p, n], -self.requests[p, n] * self.comm_costs[p])

        return bqm

    def solve(self, num_reads=1000, num_sweeps=1000, beta_range=None):
        bqm = self.build_bqm()

        sampler = PathIntegralAnnealingSampler()

        sample_kwargs = dict(num_reads=num_reads, num_sweeps=num_sweeps)
        if beta_range is not None:
            sample_kwargs['beta_range'] = beta_range

        start = time.perf_counter()
        sampleset = sampler.sample(bqm, **sample_kwargs)
        end = time.perf_counter()

        time_taken = (end - start) * 1000
        self.time_taken = time_taken
        self.result = sampleset.first
        return time_taken, sampleset.first

    def format_answer(self, result=None):
        sample_obj = result if result is not None else self.result
        if sample_obj is None:
            print("No valid solution found.")
            return

        best_sample = sample_obj.sample
        allocation_data = []
        for p in self.partitions:
            row = {'Partition': p}
            for n in self.nodes:
                row[n] = best_sample[f'A_{p}_{n}']
            allocation_data.append(row)

        matrix_df = pd.DataFrame(allocation_data).set_index('Partition')
        print("--- Data Allocation Matrix (S2 Slack-Free) ---")
        print(matrix_df)
