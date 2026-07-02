"""
statistics — Quantitative diff metrics.

Public API
----------
compute_statistics  Aggregate all metrics into a DiffStatistics dataclass.
DiffStatistics      Frozen dataclass holding computed stats.
"""

from statistics.stats import DiffStatistics, compute_statistics

__all__: list[str] = [
    "compute_statistics",
    "DiffStatistics",
]
