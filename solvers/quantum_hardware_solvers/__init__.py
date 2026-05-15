"""
Quantum hardware solvers — submit BQMs to a real D-Wave QPU.

Each solver inherits its BQM construction from the corresponding
simulated solver and only overrides the solve() method to use
EmbeddingComposite(DWaveSampler()) instead of
PathIntegralAnnealingSampler().

Prerequisites:
    pip install dwave-system
    dwave setup          # configure your LEAP API token

Solvers:
    SQAHardwareSolver     (S1) — binary slack variables for storage
    SQASFHardwareSolver   (S2) — slack-free, unbalanced penalty
    SQADWHardwareSolver   (S3) — domain-wall k-safety + slack-free
"""

from solvers.quantum_hardware_solvers.SQA_HW import SQAHardwareSolver
from solvers.quantum_hardware_solvers.SQA_SF_HW import SQASFHardwareSolver
from solvers.quantum_hardware_solvers.SQA_DW_HW import SQADWHardwareSolver

__all__ = [
    'SQAHardwareSolver',
    'SQASFHardwareSolver',
    'SQADWHardwareSolver',
]
