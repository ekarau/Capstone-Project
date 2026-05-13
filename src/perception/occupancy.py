"""Class-based footprint occupancy estimator.

The thesis (§3.5 *Class-Based Footprint Estimation*) operationalises
cabin occupancy as

.. math::

    \\rho \\;=\\; \\min\\!\\left(
        \\frac{\\sum_{c}\\, n_c\\, \\bar{a}_c}{A_{\\text{cabin}}},\\;
        1
    \\right)

where :math:`n_c` is the per-class detection count and :math:`\\bar{a}_c`
is the standardised footprint area for class :math:`c` (ISO 8100-32:2020
§6.4, EN 81-20:2020 §5.4.2.1.1, EN 1888-1:2018, IATA Resolution 753 —
see ``configs/default.yaml``).

This module exposes the :class:`OccupancyEstimator` interface and the
:class:`ClassFootprintOccupancy` concrete implementation used by both
the simulation script and the Streamlit demo. The estimator is
position-agnostic: two passengers standing shoulder-to-shoulder are
still counted as :math:`2 \\cdot \\bar{a}_{\\text{person}}`. Position-aware
variants (homography union-of-disks, BEV mask) are listed in thesis
§5.5 as future work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.detection.detector import Detection


class OccupancyEstimator(ABC):
    """Returns the cabin occupancy ratio in [0, 1]."""

    @abstractmethod
    def estimate(self, detections: Sequence[Detection]) -> float: ...


@dataclass(frozen=True)
class OccupancyBreakdown:
    """Full decomposition of an occupancy estimate.

    Exposed so the Streamlit demo can display per-class contributions
    without re-implementing the formula. ``counts`` and ``breakdown_m2``
    only contain classes with positive footprint and at least one
    detection.
    """

    counts: Mapping[str, int]
    breakdown_m2: Mapping[str, float]
    occupied_m2: float
    cabin_m2: float
    ratio: float


class ClassFootprintOccupancy(OccupancyEstimator):
    """Thesis §3.5 footprint model — single source of truth for ρ.

    Parameters
    ----------
    footprints_m2:
        Mapping ``class_name → standardised footprint (m²)``. Classes
        absent from the mapping (or mapped to a non-positive value) are
        treated as having zero footprint and silently ignored.
    cabin_m2:
        Cabin floor area :math:`A_{\\text{cabin}}` in m². A non-positive
        value forces :math:`\\rho = 0` (defensive default for malformed
        configs).
    """

    def __init__(
        self,
        footprints_m2: Mapping[str, float],
        cabin_m2: float,
    ) -> None:
        self.footprints_m2: dict[str, float] = dict(footprints_m2)
        self.cabin_m2: float = float(cabin_m2)

    def estimate(self, detections: Sequence[Detection]) -> float:
        counts = Counter(d.class_name for d in detections)
        return self.estimate_from_counts(counts)

    def estimate_from_counts(self, counts: Mapping[str, int]) -> float:
        return self.compute(counts).ratio

    def compute(self, counts: Mapping[str, int]) -> OccupancyBreakdown:
        breakdown: dict[str, float] = {}
        kept_counts: dict[str, int] = {}
        occupied = 0.0
        for cls, n in counts.items():
            a = self.footprints_m2.get(cls, 0.0)
            if a <= 0 or n <= 0:
                continue
            kept_counts[cls] = int(n)
            breakdown[cls] = float(n) * a
            occupied += float(n) * a
        ratio = min(occupied / self.cabin_m2, 1.0) if self.cabin_m2 > 0 else 0.0
        return OccupancyBreakdown(
            counts=kept_counts,
            breakdown_m2=breakdown,
            occupied_m2=occupied,
            cabin_m2=self.cabin_m2,
            ratio=ratio,
        )
