"""4-point homography utilities for image -> floor-plane projection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def compute_homography(
    image_points_px: np.ndarray,
    world_points_m: np.ndarray,
) -> np.ndarray:
    """Compute 3x3 homography from 4 image points (px) to 4 world points (m)."""
    import cv2

    if image_points_px.shape != (4, 2) or world_points_m.shape != (4, 2):
        raise ValueError("Inputs must both be (4, 2) arrays.")
    H, _ = cv2.findHomography(
        image_points_px.astype(np.float32),
        world_points_m.astype(np.float32),
    )
    if H is None:
        raise RuntimeError("Homography failed; check point selection.")
    return H


def project_to_floor(point_px: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """Project a single image-pixel point onto the floor plane (meters)."""
    x, y = point_px
    pt = np.array([x, y, 1.0], dtype=np.float64)
    out = H @ pt
    if out[2] == 0:
        raise ValueError("Singular projection (w=0).")
    out = out / out[2]
    return float(out[0]), float(out[1])


def save_homography(H: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"homography": np.asarray(H).tolist()}, f)


def load_homography(path: str | Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array(data["homography"], dtype=np.float64)


def cabin_corner_world_points(width_m: float, depth_m: float) -> np.ndarray:
    """4 cabin corners in floor-plane meters (top-left clockwise)."""
    return np.array(
        [
            [0.0, 0.0],
            [width_m, 0.0],
            [width_m, depth_m],
            [0.0, depth_m],
        ],
        dtype=np.float64,
    )


def synthetic_homography(
    frame_width_px: int,
    frame_height_px: int,
    cabin_width_m: float,
    cabin_depth_m: float,
    margin_ratio: float = 0.10,
) -> np.ndarray:
    """A reasonable default homography when no real calibration exists.

    Assumes a corner-mounted CCTV camera looking diagonally down. Use
    `compute_homography` with real point clicks for accurate work.
    """
    w, h = frame_width_px, frame_height_px
    margin_x = w * margin_ratio
    margin_y = h * margin_ratio

    image_pts = np.array([
        [margin_x, margin_y],                    # far-left (back-left corner)
        [w - margin_x, margin_y],                # far-right
        [w - margin_x * 0.3, h - margin_y],      # near-right (perspective)
        [margin_x * 0.3, h - margin_y],          # near-left
    ], dtype=np.float64)

    world_pts = cabin_corner_world_points(cabin_width_m, cabin_depth_m)
    return compute_homography(image_pts, world_pts)
