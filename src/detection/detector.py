"""YOLOv8 inference wrapper — lazy-loads ultralytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.logger import get_logger


@dataclass(frozen=True)
class Detection:
    """One detected object — bbox in pixel coords (x1, y1, x2, y2)."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def width_px(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height_px(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area_px(self) -> int:
        return self.width_px * self.height_px

    @property
    def bottom_center_px(self) -> tuple[float, float]:
        """Bottom-mid point of bbox — used as floor projection anchor."""
        return ((self.bbox[0] + self.bbox[2]) / 2.0, float(self.bbox[3]))

    @property
    def center_px(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


class YOLOv8Detector:
    """Thin inference wrapper around an Ultralytics YOLOv8 model."""

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        target_classes: Sequence[str] | None = None,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = set(target_classes) if target_classes else None
        self.device = device
        self._logger = get_logger(__name__)
        self.model = YOLO(str(self.weights_path))
        self._logger.info(f"YOLOv8 loaded: {self.weights_path}")

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run inference on a BGR/RGB numpy image. Returns list of Detection."""
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        return self._parse(results)

    def _parse(self, results) -> list[Detection]:
        detections: list[Detection] = []
        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls_name = names[cls_id]
                if self.target_classes is not None and cls_name not in self.target_classes:
                    continue
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )
        return detections
