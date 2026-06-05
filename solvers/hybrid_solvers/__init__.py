"""
Hybrid solvers — submit BQMs to a D-Wave Leap hybrid solver.

Each solver inherits its BQM construction from the corresponding
simulated solver and only overrides the solve() method to use
LeapHybridSampler() instead of PathIntegralAnnealingSampler() (simulated)
or EmbeddingComposite(DWaveSampler()) (pure QPU).

Unlike the pure-QPU hardware solvers, the hybrid path performs minor
embedding internally and never returns it, so there is no chain /
physical-qubit metadata; runs are driven by a single ``time_limit``
(seconds) rather than ``num_reads`` / ``annealing_time``.

Prerequisites:
    pip install dwave-system
    dwave setup          # configure your LEAP API token

Solvers:
    SQAHybridSolver     (S1) — binary slack variables for storage
    SQASFHybridSolver   (S2) — slack-free, unbalanced penalty
    SQADWHybridSolver   (S3) — domain-wall k-safety + slack-free
"""

from solvers.hybrid_solvers.SQA_H import SQAHybridSolver
from solvers.hybrid_solvers.SQA_SF_H import SQASFHybridSolver
from solvers.hybrid_solvers.SQA_DW_H import SQADWHybridSolver

__all__ = [
    'SQAHybridSolver',
    'SQASFHybridSolver',
    'SQADWHybridSolver',
]
