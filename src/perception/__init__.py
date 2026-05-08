"""Perception: homography, birds-eye view, occupancy estimation."""

from src.perception.homography import (
    compute_homography,
    project_to_floor,
    save_homography,
    load_homography,
    cabin_corner_world_points,
    synthetic_homography,
)
from src.perception.bev import BirdsEyeView
from src.perception.occupancy import (
    OccupancyEstimator,
    CoefficientOccupancy,
    FootprintOccupancy,
    BEVMaskOccupancy,
    build_occupancy_estimator,
)

__all__ = [
    "compute_homography",
    "project_to_floor",
    "save_homography",
    "load_homography",
    "cabin_corner_world_points",
    "synthetic_homography",
    "BirdsEyeView",
    "OccupancyEstimator",
    "CoefficientOccupancy",
    "FootprintOccupancy",
    "BEVMaskOccupancy",
    "build_occupancy_estimator",
]
