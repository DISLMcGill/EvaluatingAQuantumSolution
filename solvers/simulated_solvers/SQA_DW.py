"""
S3 — SQA Domain-Wall + Unbalanced-Penalty Solver.

This solver combines:

  * **Domain-wall k-safety encoding** (Chancellor 2019, arXiv:1903.05068).
    For each partition p, introduce |N| binary "wall" variables
    ``W_{p,1}, ..., W_{p,|N|}`` constrained to a monotone chain
    ``W_{p,1} >= W_{p,2} >= ... >= W_{p,|N|}`` and forced to have exactly
    k ones (i.e. ``W_{p,k} = 1``, ``W_{p,k+1} = 0``).  The chain
    monotonicity is encoded with O(|N|) nearest-neighbour quadratic terms.

  * **Calibrated unbalanced-penalty storage** (Paper 2, see SQA_SF.py).
    Identical to the S2 storage encoding, with ``(lambda_1, lambda_2)``
    tuned per instance.

Honest caveat -- read this before using S3
-------------------------------------------
On the data-allocation problem, the domain-wall chain only replaces the
``(sum_n A_{p,n} - k)^2`` penalty; the per-node assignment variables
``A_{p,n}`` are still required (the nodes are not interchangeable: they
have distinct storage capacities and per-(p,n) request frequencies).
Linking the chain variables ``W_{p,*}`` back to the assignment variables
``A_{p,*}`` via ``(sum_n A_{p,n} - sum_j W_{p,j})^2 = 0`` reintroduces
``O(|N|^2)`` couplings.

The net effect is therefore *not* a coupling-count reduction over S1/S2 --
this solver exists to make that fact reproducible, not to claim a
quantum-hardware speedup.  If you want the smallest BQM, use S2.
"""

import time
from typing import Optional

import dimod
import pandas as pd
from dwave.samplers import PathIntegralAnnealingSampler

from solvers.simulated_solvers.SQA_SF import calibrate_lambdas
from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    VALID_POLICIES,
    select_sample,
)
from util.solver_base import SolverBase


class SQADomainWallSolver(SolverBase):
    """S3 - domain-wall k-safety + calibrated unbalanced-penalty storage."""

    def __init__(
        self,
        nodes,
        partitions,
        k_safety,
        requests,
        comm_costs,
        lambda_1: Optional[float] = None,
        lambda_2: Optional[float] = None,
        selection_policy: str = POLICY_BEST_FEASIBLE,
    ):
        # S3 does NOT require unit partition sizes: the unbalanced
        # storage penalty handles arbitrary sizes (same as S2).
        SolverBase.__init__(self, nodes, partitions, k_safety, requests, comm_costs)

        if selection_policy not in VALID_POLICIES:
            raise ValueError(
                f"selection_policy must be one of {VALID_POLICIES}, "
                f"got {selection_policy!r}"
            )
        self.selection_policy = selection_policy
        self.selection_diagnostics = None

        if (lambda_1 is None) ^ (lambda_2 is None):
            raise ValueError(
                "Pass both lambda_1 and lambda_2, or neither (for auto-calibration)."
            )

        if lambda_1 is None:
            # S3 calibration is more expensive than S2 because the BQM
            # has |P| * |N| extra W variables, so ExactSolver scales
            # as 2^(2 * |P| * |N|).  Limit exact calibration to small
            # instances and use a heuristic for everything else.
            n_a_vars = len(partitions) * len(nodes)
            n_total_vars = n_a_vars * 2   # rough estimate including W

            target = None
            if n_a_vars <= 9:   # ExactSolver over <= ~2^18 states is OK
                from util.brute_force import brute_force_solve
                target, _ = brute_force_solve(
                    nodes, partitions, k_safety, requests, comm_costs
                )

                def _builder(l1, l2):
                    stash = (self.lambda_1, self.lambda_2) if hasattr(self, "lambda_1") else None
                    self.lambda_1 = l1
                    self.lambda_2 = l2
                    bqm = self.build_bqm()
                    if stash is not None:
                        self.lambda_1, self.lambda_2 = stash
                    return bqm

                self.lambda_1, self.lambda_2 = 1.0, 1.0
                self.lambda_1, self.lambda_2 = calibrate_lambdas(
                    nodes, partitions, k_safety, requests, comm_costs,
                    target_cost=target,
                    max_vars_for_exact=n_total_vars + 1,
                    bqm_builder=_builder,
                )
            else:
                # Heuristic fallback for larger instances -- S3 is not
                # recommended for these sizes anyway (see module docstring).
                self.lambda_1, self.lambda_2 = calibrate_lambdas(
                    nodes, partitions, k_safety, requests, comm_costs,
                    target_cost=None,
                    max_vars_for_exact=0,   # force heuristic branch
                )
        else:
            if lambda_1 <= 0 or lambda_2 <= 0:
                raise ValueError("lambda_1 and lambda_2 must be positive")
            self.lambda_1 = float(lambda_1)
            self.lambda_2 = float(lambda_2)

    def build_bqm(self):
        bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

        partition_list = list(self.partitions.keys())
        node_list = list(self.nodes.keys())
        num_nodes = len(node_list)

        A = {(p, n): f"A_{p}_{n}" for p in self.partitions for n in self.nodes}
        W = {
            (p, j): f"W_{p}_{j}"
            for p in partition_list
            for j in range(1, num_nodes + 1)
        }

        # ---- penalty weights ----
        h = sum(
            self.requests[p, n] * self.comm_costs[p]
            for p in self.partitions for n in self.nodes
        ) + 1
        max_C = max(int(c) for c in self.nodes.values())
        h_chain = h * max_C        # domain-wall + linking penalty weight
        l1 = self.lambda_1
        l2 = self.lambda_2

        # ---- Q_R: domain-wall k-safety encoding ----

        # (a) Chain monotonicity: penalise W_{j+1}=1 while W_j=0
        #     penalty = h_chain * (W_{j+1} - W_j * W_{j+1})
        for p in partition_list:
            for j in range(1, num_nodes):
                bqm.add_variable(W[p, j + 1], h_chain)
                bqm.add_interaction(W[p, j], W[p, j + 1], -h_chain)

        # (b) Fix count to exactly k: penalise W_{p,k}=0 and W_{p,k+1}=1
        k = self.k_safety
        for p in partition_list:
            if 1 <= k <= num_nodes:
                bqm.add_variable(W[p, k], -h_chain)
            if k + 1 <= num_nodes:
                bqm.add_variable(W[p, k + 1], h_chain)

        # (c) Link chain to assignment: (sum_n A_{p,n} - sum_j W_{p,j})^2 == 0
        #
        # NOTE: this reintroduces O(|N|^2) couplings -- see the module
        # docstring.  We keep it because dropping it disconnects A from W
        # and would let the solver pick any subset.
        for p in partition_list:
            expr = [(A[p, n], 1) for n in node_list]
            expr += [(W[p, j], -1) for j in range(1, num_nodes + 1)]
            bqm.add_linear_equality_constraint(
                expr, constant=0, lagrange_multiplier=h_chain
            )

        # ---- Q_S: unbalanced-penalty storage (same as S2) ----
        for n, capacity in self.nodes.items():
            C = int(capacity)
            for i, p in enumerate(partition_list):
                size_p = int(self.partitions[p])
                lin = l1 * size_p + l2 * (size_p * size_p - 2 * C * size_p)
                bqm.add_variable(A[p, n], lin)
                for j in range(i + 1, len(partition_list)):
                    p2 = partition_list[j]
                    size_p2 = int(self.partitions[p2])
                    bqm.add_interaction(
                        A[p, n], A[p2, n],
                        2 * l2 * size_p * size_p2,
                    )

        # ---- Q_C: processing costs ----
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
        print("--- Data Allocation Matrix (S3 Domain-Wall + Unbalanced) ---")
        print(matrix_df)
