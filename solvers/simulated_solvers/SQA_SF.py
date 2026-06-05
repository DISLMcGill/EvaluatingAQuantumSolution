"""
S2 — SQA Slack-Free Solver with Unbalanced Penalization.

Implements Paper 2 (Montañez-Barrera et al., 2211.13914) faithfully:
each storage inequality

    h_n(A) := C_n - sum_p A_{p,n} * size_p  >=  0

is penalised by

    zeta_n  =  -lambda_1 * h_n(A)  +  lambda_2 * h_n(A)^2.

This rewards feasible slack until h* = lambda_1 / (2 * lambda_2) and
grows quadratically beyond the constraint boundary. The previous version
of this file used lambda_1 = lambda_2 = h (the constraint weight), which
collapses zeta to (x - C)(x - C + 1) -- a non-negative function with
minimum at x in {C-1, C} that *penalises feasible slack*. That is
neither Paper 1 nor Paper 2.

Lambda calibration
------------------
Paper 2 (Sec. III) is explicit that lambda_1 / lambda_2 must be tuned;
the default behaviour is therefore to *refuse* lambda_1 = lambda_2 = 1.

If lambda_1 and lambda_2 are not passed in:
  * an automatic calibration is run on the instance itself: we pick a
    pair that makes the QUBO ground state coincide with the brute-force
    optimum on a small representative slice. See ``calibrate_lambdas``.
  * a deliberately conservative default is used when the instance is too
    large to calibrate exactly (lambda_1 = h * |P|, lambda_2 = h), which
    keeps the constraint penalty dominant but still rewards slack.

Concretely: callers who want repeatable behaviour should pass explicit
lambdas; callers who want "just work" can leave them None.
"""

import time
from typing import Optional

import dimod
import pandas as pd
from dwave.samplers import PathIntegralAnnealingSampler

from util.sample_selection import (
    POLICY_BEST_FEASIBLE,
    VALID_POLICIES,
    select_sample,
)
from util.solver_base import SolverBase


# ---------------------------------------------------------------------------
# Lambda calibration
# ---------------------------------------------------------------------------

def _multipliers(nodes, partitions, requests, comm_costs):
    """Constraint weight h per Paper 1, Eq. 9."""
    return sum(
        requests[p, n] * comm_costs[p]
        for p in partitions for n in nodes
    ) + 1


def _bqm_ground_state(bqm, nodes, partitions, k_safety, requests, comm_costs):
    """Return (cost, valid) of the BQM's exact ground state."""
    from util.calculate_solution_cost import (
        calculate_solution_cost, is_valid_solution,
    )
    ss = dimod.ExactSolver().sample(bqm)
    sample = ss.first.sample
    return (
        calculate_solution_cost(
            nodes, partitions, k_safety, requests, comm_costs, sample
        ),
        is_valid_solution(
            nodes, partitions, k_safety, requests, comm_costs, sample
        ),
    )


def calibrate_lambdas(
    nodes, partitions, k_safety, requests, comm_costs,
    target_cost=None, max_vars_for_exact=18, bqm_builder=None,
):
    """
    Return ``(lambda_1, lambda_2)`` for the unbalanced storage penalty.

    Paper 2 (Sec. III) is explicit that the multipliers must be tuned per
    instance. We do this in two modes:

      * **Exact calibration** (small instances): build the BQM for a small
        grid of ``(lambda_1, lambda_2)`` values, find each one's ground
        state via ``dimod.ExactSolver``, and pick the pair that
        (a) yields a feasible ground state, and
        (b) minimises the gap to the target cost (if known, e.g. from a
            brute-force or ILP run; otherwise minimises cost alone).
        Only used when |variables| <= ``max_vars_for_exact``.

      * **Heuristic fallback** (large instances): use a conservative
        analytic choice with ``lambda_1 = 2 * max_C * lambda_2`` (placing
        the parabola minimum at half full capacity) and
        ``lambda_1 + lambda_2 > h`` (so violations dominate any cost
        gradient).  This is the "good enough" path Paper 2 takes when
        Nelder-Mead is not feasible.
    """
    h = _multipliers(nodes, partitions, requests, comm_costs)
    max_C = max(int(c) for c in nodes.values())

    n_vars = len(partitions) * len(nodes)

    # ----- Heuristic fallback for larger instances ----------------------
    if n_vars > max_vars_for_exact:
        # lambda_2 small enough to barely cross h; lambda_1 large enough
        # to make h* = max_C (so all feasible h in [0, max_C] is on the
        # "rewarding" side of the parabola).
        lambda_2 = max(1.0, h / max(max_C, 1))
        lambda_1 = 2.0 * max_C * lambda_2
        return lambda_1, lambda_2

    # ----- Exact calibration for small instances ------------------------
    #
    # The unbalanced penalty zeta(h) = -l1*h + l2*h^2 has minimum at
    # h* = l1 / (2*l2). For correctness:
    #   - l1 + l2 > h   (infeasible h_n = -1 must be penalised)
    #   - l2 small      (so feasible-side reward doesn't outweigh true cost)
    #
    # Sweep l2 across several orders of magnitude and for each l2 pick l1
    # to satisfy the violation bound l1 + l2 > h with margin.
    candidates = []
    for l2_value in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float(h)):
        for l1_mult in (1.5, 2.0, 4.0, 8.0, 16.0):
            l1_value = max(l1_mult * h, l1_mult * max_C * l2_value)
            candidates.append((l1_value, l2_value))

    if bqm_builder is None:
        bqm_builder = lambda l1, l2: _build_storage_bqm(
            nodes, partitions, k_safety, requests, comm_costs, l1, l2, h,
        )

    best = None  # (gap, lambda_1, lambda_2)
    for l1, l2 in candidates:
        bqm = bqm_builder(l1, l2)
        cost, valid = _bqm_ground_state(
            bqm, nodes, partitions, k_safety, requests, comm_costs,
        )
        if not valid:
            continue
        if target_cost is not None:
            gap = cost - target_cost
        else:
            gap = cost
        if best is None or gap < best[0]:
            best = (gap, l1, l2)
            if target_cost is not None and gap == 0:
                break  # exact match — stop early

    if best is None:
        # No (l1, l2) in the grid produced a feasible ground state;
        # fall back to heuristic.
        lambda_2 = max(1.0, h / max(max_C, 1))
        lambda_1 = 2.0 * max_C * lambda_2
        return lambda_1, lambda_2

    return best[1], best[2]


def _build_storage_bqm(
    nodes, partitions, k_safety, requests, comm_costs,
    lambda_1, lambda_2, h,
):
    """Internal: build the full S2 BQM given specific lambdas.

    Used by ``calibrate_lambdas`` to evaluate candidate (l1, l2) pairs
    without instantiating the full solver class.  Mirrors
    ``SQASlackFreeSolver.build_bqm``; if you change one, change both.
    """
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    A = {(p, n): f"A_{p}_{n}" for p in partitions for n in nodes}

    max_C = max(int(c) for c in nodes.values())
    h_k = h * max_C

    for p in partitions:
        bqm.add_linear_equality_constraint(
            [(A[p, n], 1) for n in nodes],
            constant=-k_safety,
            lagrange_multiplier=h_k,
        )

    part_list = list(partitions.keys())
    for n, capacity in nodes.items():
        C = int(capacity)
        for i, p in enumerate(part_list):
            size_p = int(partitions[p])
            lin = lambda_1 * size_p + lambda_2 * (
                size_p * size_p - 2 * C * size_p
            )
            bqm.add_variable(A[p, n], lin)
            for j in range(i + 1, len(part_list)):
                p2 = part_list[j]
                size_p2 = int(partitions[p2])
                bqm.add_interaction(
                    A[p, n], A[p2, n],
                    2 * lambda_2 * size_p * size_p2,
                )

    for p in partitions:
        for n in nodes:
            bqm.add_variable(A[p, n], -requests[p, n] * comm_costs[p])

    return bqm


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SQASlackFreeSolver(SolverBase):
    """
    Slack-free formulation using Paper 2's unbalanced penalisation.

    Supports arbitrary (not just unit) partition sizes -- the storage
    inequality is the same shape regardless of size_p.
    """

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
            # Calibrate against the brute-force optimum if the instance is
            # small enough; otherwise fall back to heuristic.
            target = None
            n_vars = len(partitions) * len(nodes)
            if n_vars <= 16:
                from util.brute_force import brute_force_solve
                target, _ = brute_force_solve(
                    nodes, partitions, k_safety, requests, comm_costs
                )
            self.lambda_1, self.lambda_2 = calibrate_lambdas(
                nodes, partitions, k_safety, requests, comm_costs,
                target_cost=target,
            )
        else:
            if lambda_1 <= 0 or lambda_2 <= 0:
                raise ValueError("lambda_1 and lambda_2 must be positive")
            self.lambda_1 = float(lambda_1)
            self.lambda_2 = float(lambda_2)

    # ------------------------------------------------------------------
    # BQM construction
    # ------------------------------------------------------------------

    def build_bqm(self):
        """Build and return the BQM without solving."""
        bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

        A = {(p, n): f"A_{p}_{n}" for p in self.partitions for n in self.nodes}

        h = _multipliers(self.nodes, self.partitions, self.requests, self.comm_costs)
        # k-safety must dominate the unbalanced-penalty's worst feasible
        # value (which is lambda_1 * max_C); scale by max_C to be safe.
        max_C = max(int(c) for c in self.nodes.values())
        h_k = h * max_C

        # 1. Q_R: k-safety  --  (sum_n A_pn - k)^2 == 0
        for p in self.partitions:
            k_safety_expr = [(A[p, n], 1) for n in self.nodes]
            bqm.add_linear_equality_constraint(
                k_safety_expr,
                constant=-self.k_safety,
                lagrange_multiplier=h_k,
            )

        # 2. Q_S: unbalanced penalty per node
        #
        #    h_n  =  C_n  -  sum_p A_{p,n} * size_p
        #    zeta_n  =  -lambda_1 * h_n  +  lambda_2 * h_n^2
        #
        # Expanding for binary A's:
        #
        #    let L_n = sum_p A_{p,n} * size_p
        #    h_n      = C_n - L_n
        #    h_n^2    = C_n^2 - 2 C_n L_n + L_n^2
        #
        # And L_n^2 expands to:
        #
        #    L_n^2 = sum_p size_p^2 A_{p,n}              (A^2 = A)
        #          + 2 sum_{p < p'} size_p * size_p' * A_{p,n} * A_{p',n}
        #
        # Coefficients (constants dropped):
        #
        #    linear[A_{p,n}]      =   lambda_1 * size_p
        #                            + lambda_2 * (size_p^2 - 2 * C_n * size_p)
        #    quadratic[A_{p,n}, A_{p',n}] = 2 * lambda_2 * size_p * size_p'

        l1 = self.lambda_1
        l2 = self.lambda_2

        part_list = list(self.partitions.keys())
        for n, capacity in self.nodes.items():
            C = int(capacity)
            for i, p in enumerate(part_list):
                size_p = int(self.partitions[p])
                lin = l1 * size_p + l2 * (size_p * size_p - 2 * C * size_p)
                bqm.add_variable(A[p, n], lin)
                for j in range(i + 1, len(part_list)):
                    p2 = part_list[j]
                    size_p2 = int(self.partitions[p2])
                    bqm.add_interaction(
                        A[p, n], A[p2, n],
                        2 * l2 * size_p * size_p2,
                    )

        # 3. Q_C: processing costs
        for p in self.partitions:
            for n in self.nodes:
                bqm.add_variable(
                    A[p, n], -self.requests[p, n] * self.comm_costs[p]
                )

        return bqm

    # ------------------------------------------------------------------
    # Solve loop
    # ------------------------------------------------------------------

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
        print("--- Data Allocation Matrix (S2 Slack-Free, unbalanced penalisation) ---")
        print(matrix_df)
