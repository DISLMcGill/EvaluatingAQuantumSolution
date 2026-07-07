"""
Feasibility probes for test-case generation.

Two probes are provided:

* ``full_optimize_feasible`` -- the legacy probe: run the full ILP
  (objective + constraints) to proven optimality and call the instance
  feasible iff the solver returns an optimal allocation.  Correct, but
  it pays for a full branch-and-bound *optimization* just to answer a
  yes/no feasibility question.  That is fine at tier-1/2 scale and
  intractable at the hybrid-stress scale (n40 x p400 ~ 16k binaries).

* ``constraints_feasible`` -- a scalable probe: build the SAME feasible
  region (k-safety + capacity) with a **constant (zero) objective** and
  an optional CBC wall-clock cap.  With nothing to minimize, CBC stops
  the instant it finds any feasible point, so this answers feasibility
  orders of magnitude faster on large instances.  The objective terms
  (requests / comm_costs) do not affect feasibility, so they are not
  needed here.

Both return a plain ``bool``.  The generators choose between them via the
``feasibility_only`` flag.
"""

import pulp

from solvers.ILP import ILPSolver


def full_optimize_feasible(tc):
    """Legacy probe: full ILP optimize; feasible iff an optimum is found."""
    requests = {
        tuple(k[1:-1].split(", ")): v
        for k, v in tc["requests"].items()
    }
    solver = ILPSolver(
        tc["nodes"], tc["partitions"], tc["k_safety"],
        requests, tc["comm_costs"],
    )
    _, result = solver.solve()
    return result is not None


def constraints_feasible(tc, probe_time_limit=None):
    """
    Scalable probe: is the (k-safety + capacity) region non-empty?

    Builds a zero-objective ILP over binary assignment variables A[p, n]:

        sum_n A[p, n] == k_safety        for every partition p
        sum_p size_p * A[p, n] <= cap_n  for every node n

    Args:
        tc:               test-case dict (only ``nodes``, ``partitions``
                          and ``k_safety`` are read).
        probe_time_limit: optional CBC wall-clock cap in seconds.  ``None``
                          lets CBC run unbounded (still fast with a
                          constant objective).

    Returns:
        True iff CBC proves a feasible point exists.  A capped solve that
        neither finds a point nor proves infeasibility returns False --
        i.e. we only ever accept an instance we have *confirmed* feasible,
        never one we merely failed to disprove.
    """
    nodes = tc["nodes"]
    partitions = tc["partitions"]
    k_safety = tc["k_safety"]

    prob = pulp.LpProblem("feasibility_probe", pulp.LpMinimize)
    A = pulp.LpVariable.dicts(
        "A",
        ((p, n) for p in partitions for n in nodes),
        cat="Binary",
    )

    # Constant objective: nothing to optimize, so CBC returns on first
    # feasible incumbent.
    prob += 0

    for p in partitions:
        prob += pulp.lpSum(A[p, n] for n in nodes) == k_safety, f"safety_{p}"

    for n, cap in nodes.items():
        prob += (
            pulp.lpSum(partitions[p] * A[p, n] for p in partitions) <= cap,
            f"capacity_{n}",
        )

    cmd = pulp.PULP_CBC_CMD(msg=0, timeLimit=probe_time_limit)
    prob.solve(cmd)

    # With a zero objective, "Optimal" == "a feasible point was found".
    return pulp.LpStatus[prob.status] == "Optimal"
