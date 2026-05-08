"""Occupancy estimators — three strategies.

  CoefficientOccupancy : bbox_area_px × class_coef. Camera-angle sensitive,
                         used as a baseline.
  FootprintOccupancy   : Project bottom-mid of bbox to floor plane,
                         union of per-class footprint disks. Physically
                         meaningful but uses analytic shapely union.
  BEVMaskOccupancy     : Same projection, but rasterizes onto a
                         birds-eye-view binary mask. The mask is also used
                         for visualization, ensuring numeric == visual.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

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
