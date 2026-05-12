"""Occupancy estimator interface.

The headline pipeline (`scripts/run_simulation.py`) and the Streamlit demo
(`demo/app.py`) implement the constants-only class-based footprint model
inline. This module exposes only the abstract :class:`OccupancyEstimator`
used as a type hint by :mod:`src.control.algorithm`.

Position-aware variants (homography union-of-disks, BEV mask) are
identified in §5.5 of the thesis as future work and are not implemented
in this prototype.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.detection.detector import Detection


class OccupancyEstimator(ABC):
    """Returns the cabin occupancy ratio in [0, 1]."""

    @abstractmethod
    def estimate(self, detections: Sequence[Detection]) -> float: ...
