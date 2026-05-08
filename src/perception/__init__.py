"""Perception: homography, birds-eye view, occupancy estimation."""

from src.perception.bev import BirdsEyeView
from src.perception.homography import (
    cabin_corner_world_points,
    compute_homography,
    load_homography,
    project_to_floor,
    save_homography,
    synthetic_homography,
)
from src.perception.occupancy import (
    BEVMaskOccupancy,
    CoefficientOccupancy,
    FootprintOccupancy,
    OccupancyEstimator,
    build_occupancy_estimator,
)

__all__ = [
    "BEVMaskOccupancy",
    "BirdsEyeView",
    "CoefficientOccupancy",
    "FootprintOccupancy",
    "OccupancyEstimator",
    "build_occupancy_estimator",
    "cabin_corner_world_points",
    "compute_homography",
    "load_homography",
    "project_to_floor",
    "save_homography",
    "synthetic_homography",
]
