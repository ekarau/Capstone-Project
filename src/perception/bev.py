"""Bird's-Eye View (BEV) renderer.

Given a homography mapping image-pixels -> floor meters, this module:

  1. Warps the camera image into a top-down `bev_size x bev_size` map of
     the cabin floor.
  2. Rasterizes per-detection footprints onto a separate occupancy mask
     (also bev_size x bev_size). The mask is what `BEVMaskOccupancy` uses
     to compute the floor occupancy ratio (occupied_pixels / total_pixels).

The same `BirdsEyeView` instance is reused for visualization and for the
occupancy mask, ensuring perfect alignment between the two views.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.detection.detector import Detection
from src.perception.homography import project_to_floor


@dataclass
class BirdsEyeView:
    """Maps image pixels onto a top-down rendering of the cabin floor.

    Coordinate convention:
        World (meters):  x in [0, width_m],  y in [0, depth_m]
        BEV (pixels):    x' = x * (bev_size / width_m)
                         y' = y * (bev_size / depth_m)
    BEV is rendered as a square (bev_size x bev_size) — the world rectangle
    is stretched isotropically; for accurate area we use the *world-space*
    occupancy ratio, not the BEV pixel ratio.
    """

    homography: np.ndarray
    cabin_width_m: float
    cabin_depth_m: float
    bev_size_px: int = 400

    @property
    def total_floor_area_m2(self) -> float:
        return self.cabin_width_m * self.cabin_depth_m

    def world_to_bev(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Convert world (m) -> BEV pixel coords."""
        u = round(x_m * self.bev_size_px / self.cabin_width_m)
        v = round(y_m * self.bev_size_px / self.cabin_depth_m)
        return u, v

    def warp_image(self, image: np.ndarray) -> np.ndarray:
        """Warp the camera frame onto a top-down BEV image."""
        import cv2

        # Build a homography from image px -> BEV px:
        #   image_to_world = self.homography
        #   world_to_bev   = scale matrix
        sx = self.bev_size_px / self.cabin_width_m
        sy = self.bev_size_px / self.cabin_depth_m
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        H_img_to_bev = S @ self.homography
        return cv2.warpPerspective(image, H_img_to_bev, (self.bev_size_px, self.bev_size_px))

    def project_detection_world(self, det: Detection) -> tuple[float, float]:
        """Bottom-mid of bbox -> floor-plane meters."""
        return project_to_floor(det.bottom_center_px, self.homography)

    def render_occupancy_mask(
        self,
        detections: Sequence[Detection],
        footprint_radius_m: dict[str, float],
    ) -> np.ndarray:
        """Rasterize per-class footprint circles onto a BEV mask (uint8 0/255)."""
        import cv2

        mask = np.zeros((self.bev_size_px, self.bev_size_px), dtype=np.uint8)
        for det in detections:
            radius_m = footprint_radius_m.get(det.class_name, 0.0)
            if radius_m <= 0:
                continue
            world_x, world_y = self.project_detection_world(det)
            # Skip detections that fall outside the cabin (mis-projection)
            if not (0.0 <= world_x <= self.cabin_width_m and 0.0 <= world_y <= self.cabin_depth_m):
                # Clamp inside — better than dropping (still flags occupancy nearby)
                world_x = float(np.clip(world_x, 0.0, self.cabin_width_m))
                world_y = float(np.clip(world_y, 0.0, self.cabin_depth_m))
            cx, cy = self.world_to_bev(world_x, world_y)
            # Use the smaller of the two scales so the circle stays a circle
            # in BEV pixel space *and* under-approximates rather than over-.
            scale = min(
                self.bev_size_px / self.cabin_width_m, self.bev_size_px / self.cabin_depth_m
            )
            r_px = max(1, round(radius_m * scale))
            cv2.circle(mask, (cx, cy), r_px, 255, thickness=-1)
        return mask

    def occupancy_ratio_from_mask(self, mask: np.ndarray) -> float:
        """Total floor occupancy ratio from a BEV mask (in [0, 1])."""
        if mask.size == 0:
            return 0.0
        occupied = int((mask > 0).sum())
        return min(occupied / float(mask.size), 1.0)

    def visualize(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        footprint_radius_m: dict[str, float],
        occupancy_ratio: float | None = None,
    ) -> np.ndarray:
        """Side-by-side: original frame (with bboxes) | BEV warp + footprints."""
        import cv2

        annotated = image.copy()
        class_colors = {
            "person": (0, 255, 0),
            "stroller": (255, 200, 0),
            "luggage": (0, 165, 255),
            "box": (255, 0, 255),
        }
        for det in detections:
            color = class_colors.get(det.class_name, (200, 200, 200))
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        bev = self.warp_image(image)
        mask = self.render_occupancy_mask(detections, footprint_radius_m)
        # Overlay footprint mask onto BEV in semi-transparent red
        overlay = bev.copy()
        red_layer = np.zeros_like(bev)
        red_layer[..., 2] = 255  # red in BGR
        overlay = np.where(mask[..., None] > 0, cv2.addWeighted(bev, 0.4, red_layer, 0.6, 0), bev)
        # Draw cabin border + grid on BEV
        cv2.rectangle(
            overlay, (0, 0), (self.bev_size_px - 1, self.bev_size_px - 1), (255, 255, 255), 2
        )
        # Show occupancy text
        if occupancy_ratio is None:
            occupancy_ratio = self.occupancy_ratio_from_mask(mask)
        cv2.putText(
            overlay,
            f"Occupancy: {occupancy_ratio * 100:.1f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Resize annotated to match BEV height for side-by-side
        h_target = self.bev_size_px
        scale = h_target / annotated.shape[0]
        ann_resized = cv2.resize(annotated, (int(annotated.shape[1] * scale), h_target))
        combined = np.hstack([ann_resized, overlay])
        return combined
