"""End-to-end demo: detect → BEV → occupancy → control decision.

Usage:
    python -m scripts.demo --image path/to/cabin.jpg \
        --weights models/weights/best.pt \
        [--cabin-width 1.4 --cabin-depth 1.6 --max-weight 630] \
        [--weight 250]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.control.algorithm import ElevatorController
from src.detection.detector import YOLOv8Detector
from src.perception.bev import BirdsEyeView
from src.perception.homography import load_homography, synthetic_homography
from src.perception.occupancy import build_occupancy_estimator
from src.utils.config_loader import ElevatorConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="models/weights/best.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--weight", type=float, default=200.0, help="Current load (kg)")
    ap.add_argument("--cabin-width", type=float, default=None)
    ap.add_argument("--cabin-depth", type=float, default=None)
    ap.add_argument("--max-weight", type=float, default=None)
    ap.add_argument("--out", default="results/demo_output.jpg")
    args = ap.parse_args()

    # Load + override config
    cfg = ElevatorConfig.from_yaml(args.config)
    cfg = cfg.with_cabin(
        width_m=args.cabin_width,
        depth_m=args.cabin_depth,
        max_weight_kg=args.max_weight,
    )

    # Image
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    h, w = img.shape[:2]
    cfg = cfg.with_overrides(camera={"frame_width_px": w, "frame_height_px": h})

    # Homography
    homography_path = Path(cfg.camera.homography_path)
    if homography_path.exists():
        H = load_homography(homography_path)
        logger.info(f"Loaded homography: {homography_path}")
    else:
        H = synthetic_homography(
            frame_width_px=w,
            frame_height_px=h,
            cabin_width_m=cfg.elevator.width_m,
            cabin_depth_m=cfg.elevator.depth_m,
        )
        logger.warning("Using synthetic homography — calibrate with tools/calibrate_camera.py")

    # Detector + occupancy
    detector = YOLOv8Detector(
        weights_path=args.weights,
        conf_threshold=cfg.thresholds.confidence_min,
        iou_threshold=cfg.thresholds.iou_min,
        target_classes=list(cfg.classes),
    )
    occupancy = build_occupancy_estimator(cfg.to_dict(), homography_matrix=H)

    # Decision
    controller = ElevatorController(
        detector=detector,
        occupancy_estimator=occupancy,
        max_weight_kg=cfg.elevator.max_weight_kg,
        weight_bypass_ratio=cfg.thresholds.weight_bypass_ratio,
        area_bypass_ratio=cfg.thresholds.area_bypass_ratio,
    )
    result = controller.decide(args.weight, img)

    print(f"\n=== DECISION: {result.decision.value.upper()} ===")
    print(f"  weight       : {result.weight_kg:.1f} kg ({result.weight_ratio*100:.1f}%)")
    if result.occupancy_ratio is not None:
        print(f"  occupancy    : {result.occupancy_ratio*100:.1f}%")
    print(f"  detections   : {result.num_detections}")

    # Visualization (only meaningful if BEV occupancy was used)
    bev = BirdsEyeView(
        homography=H,
        cabin_width_m=cfg.elevator.width_m,
        cabin_depth_m=cfg.elevator.depth_m,
        bev_size_px=cfg.camera.get("bev_resolution_px", 400),
    )
    detections = detector.detect(img)
    viz = bev.visualize(
        img,
        detections,
        cfg.occupancy.footprint_radius_m.to_dict(),
        occupancy_ratio=result.occupancy_ratio,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), viz)
    print(f"\nVisualization saved: {out_path}")


if __name__ == "__main__":
    main()
