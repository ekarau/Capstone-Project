"""Occupancy estimators — three strategies of increasing fidelity.

Given a list of detections :math:`D = \\{d_i\\}_{i=1}^{N}` with class
:math:`c_i \\in \\mathcal{C}`, each estimator returns the cabin occupancy
ratio :math:`\\rho \\in [0, 1]`.

1. **CoefficientOccupancy** — fast, camera-naive baseline.

   .. math::

       \\rho \\;=\\; \\frac{1}{A_\\text{cabin}} \\sum_{i=1}^{N}
            \\alpha_{c_i}\\, a^\\text{px}_i \\,/\\, \\beta,

   where :math:`a^\\text{px}_i` is the bbox area in pixels,
   :math:`\\alpha_{c_i}` is a per-class weight, and
   :math:`\\beta = (W_\\text{frame}\\, H_\\text{frame}) / A_\\text{cabin}` is
   the px-to-m² scale. Sensitive to camera angle.

2. **FootprintOccupancy** — physically meaningful, requires homography
   :math:`H \\in \\mathbb{R}^{3\\times3}` mapping image pixels to floor
   metres. Each detection is projected via its bbox bottom-midpoint and
   replaced by a disk of class-specific radius :math:`r_{c_i}`:

   .. math::

       \\rho \\;=\\; \\frac{1}{A_\\text{cabin}}\\,
           \\text{Area}\\!\\left(
               \\bigcup_{i=1}^{N} \\mathrm{Disk}\\!\\big(H\\,p_i,\\; r_{c_i}\\big)
           \\right).

   Uses :mod:`shapely` for the analytic union.

3. **BEVMaskOccupancy** — rasterized version of the above on a
   :math:`P \\times P` birds-eye-view mask. The same mask is reused for
   visualization, guaranteeing that the displayed BEV and the reported
   ratio agree exactly:

   .. math::

       \\rho \\;=\\; \\frac{\\#\\{(u,v) : M(u,v) = 1\\}}{P^2}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from src.detection.detector import Detection
from src.perception.bev import BirdsEyeView


class OccupancyEstimator(ABC):
    """Returns occupancy ratio in [0, 1]."""

    @abstractmethod
    def estimate(self, detections: Sequence[Detection]) -> float: ...


class CoefficientOccupancy(OccupancyEstimator):
    """Baseline: bbox area × class coefficient / cabin area (camera-naive)."""

    def __init__(
        self,
        total_floor_area_m2: float,
        area_coefficients: dict[str, float],
        frame_width_px: int,
        frame_height_px: int,
    ) -> None:
        self.total_floor_area_m2 = total_floor_area_m2
        self.area_coefficients = area_coefficients
        self.frame_area_px = frame_width_px * frame_height_px
        self.px_per_m2 = self.frame_area_px / total_floor_area_m2

    def estimate(self, detections: Sequence[Detection]) -> float:
        total_m2 = 0.0
        for det in detections:
            coef = self.area_coefficients.get(det.class_name, 0.0)
            if coef <= 0:
                continue
            total_m2 += (det.area_px * coef) / self.px_per_m2
        return min(total_m2 / self.total_floor_area_m2, 1.0)


class FootprintOccupancy(OccupancyEstimator):
    """Analytic union of per-class footprint disks on the floor plane."""

    def __init__(
        self,
        total_floor_area_m2: float,
        homography_matrix: np.ndarray,
        footprint_radius_m: dict[str, float],
    ) -> None:
        self.total_floor_area_m2 = total_floor_area_m2
        self.H = np.asarray(homography_matrix, dtype=np.float64)
        self.footprint_radius_m = footprint_radius_m

    def estimate(self, detections: Sequence[Detection]) -> float:
        from shapely.geometry import Point
        from shapely.ops import unary_union

        from src.perception.homography import project_to_floor

        circles = []
        for det in detections:
            radius = self.footprint_radius_m.get(det.class_name, 0.0)
            if radius <= 0:
                continue
            wx, wy = project_to_floor(det.bottom_center_px, self.H)
            circles.append(Point(wx, wy).buffer(radius))
        if not circles:
            return 0.0
        return min(unary_union(circles).area / self.total_floor_area_m2, 1.0)


class BEVMaskOccupancy(OccupancyEstimator):
    """Rasterized BEV mask occupancy — visual & numeric stay in sync."""

    def __init__(
        self,
        bev: BirdsEyeView,
        footprint_radius_m: dict[str, float],
    ) -> None:
        self.bev = bev
        self.footprint_radius_m = footprint_radius_m

    def estimate(self, detections: Sequence[Detection]) -> float:
        mask = self.bev.render_occupancy_mask(detections, self.footprint_radius_m)
        return self.bev.occupancy_ratio_from_mask(mask)


def build_occupancy_estimator(
    config: dict,
    homography_matrix: np.ndarray | None = None,
) -> OccupancyEstimator:
    """Build the estimator named by config['occupancy']['method']."""
    elevator = config["elevator"]
    occupancy = config["occupancy"]
    method = occupancy["method"]

    if method == "coefficient":
        camera = config["camera"]
        return CoefficientOccupancy(
            total_floor_area_m2=elevator["total_floor_area_m2"],
            area_coefficients=occupancy["area_coefficients"],
            frame_width_px=camera["frame_width_px"],
            frame_height_px=camera["frame_height_px"],
        )

    if method == "footprint":
        if homography_matrix is None:
            raise ValueError("footprint method requires homography_matrix.")
        return FootprintOccupancy(
            total_floor_area_m2=elevator["total_floor_area_m2"],
            homography_matrix=homography_matrix,
            footprint_radius_m=occupancy["footprint_radius_m"],
        )

    if method == "bev_mask":
        if homography_matrix is None:
            raise ValueError("bev_mask method requires homography_matrix.")
        bev = BirdsEyeView(
            homography=homography_matrix,
            cabin_width_m=elevator["width_m"],
            cabin_depth_m=elevator["depth_m"],
            bev_size_px=config["camera"].get("bev_resolution_px", 400),
        )
        return BEVMaskOccupancy(bev=bev, footprint_radius_m=occupancy["footprint_radius_m"])

    raise ValueError(f"Unknown occupancy.method: {method!r}")
