"""Interactive 4-point camera calibration GUI.

Click 4 cabin floor corners in this order:
    1. Far-left  (back-left corner of the floor)
    2. Far-right (back-right corner)
    3. Near-right (front-right corner)
    4. Near-left  (front-left corner)

Then press 's' to save homography to configs/homography.yaml or 'q' to quit.

Usage:
    python -m tools.calibrate_camera --image sample_cabin.jpg \
        --cabin-width 1.4 --cabin-depth 1.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.homography import (
    cabin_corner_world_points,
    compute_homography,
    save_homography,
)


CORNER_LABELS = ["1: BACK-LEFT", "2: BACK-RIGHT", "3: FRONT-RIGHT", "4: FRONT-LEFT"]


class Picker:
    def __init__(self, image: np.ndarray, window: str) -> None:
        self.image = image
        self.window = window
        self.points: list[tuple[int, int]] = []
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def render(self) -> np.ndarray:
        img = self.image.copy()
        colors = [(0, 255, 0), (0, 200, 255), (255, 100, 100), (255, 0, 255)]
        for i, p in enumerate(self.points):
            cv2.circle(img, p, 6, colors[i], -1)
            cv2.putText(img, str(i + 1), (p[0] + 8, p[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
        if len(self.points) == 4:
            cv2.polylines(img, [np.array(self.points)], True, (255, 255, 255), 1)
        # Hint
        next_idx = len(self.points)
        hint = "DONE — press 's' to save, 'q' to quit, RIGHT-CLICK to undo"
        if next_idx < 4:
            hint = f"Click corner {CORNER_LABELS[next_idx]}  (right-click to undo)"
        cv2.putText(img, hint, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, hint, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--cabin-width", type=float, default=1.4)
    ap.add_argument("--cabin-depth", type=float, default=1.6)
    ap.add_argument("--out", default="configs/homography.yaml")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot open {args.image}")

    win = "Calibrate (click 4 floor corners)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    picker = Picker(img, win)

    while True:
        cv2.imshow(win, picker.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s") and len(picker.points) == 4:
            image_pts = np.array(picker.points, dtype=np.float64)
            world_pts = cabin_corner_world_points(args.cabin_width, args.cabin_depth)
            H = compute_homography(image_pts, world_pts)
            save_homography(H, args.out)
            print(f"Saved homography to {args.out}")
            print(H)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
