"""Two-stage load-and-area elevator control (Andrei & Ruokokoski, 2022).

For each incoming hall call the controller emits one of three decisions
:math:`\\delta \\in \\{\\text{accept},\\text{bypass}_W,\\text{bypass}_A\\}`
based on the current cabin weight :math:`W` and visual occupancy
:math:`\\rho`:

.. math::

    \\delta \\;=\\;
    \\begin{cases}
        \\text{bypass}_W & \\text{if } W \\;\\ge\\; \\tau_W \\cdot W_\\text{rated},\\\\
        \\text{bypass}_A & \\text{else if } \\rho \\;\\ge\\; \\tau_A,\\\\
        \\text{accept}   & \\text{otherwise.}
    \\end{cases}

Defaults: :math:`\\tau_W = 0.80`, :math:`\\tau_A = 0.90`. The weight gate
runs first so the (cheap) load-cell reading short-circuits the (expensive)
detector inference whenever it already returns a definitive answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.detection.detector import YOLOv8Detector
from src.perception.occupancy import OccupancyEstimator
from src.utils.logger import get_logger


class ControlDecision(str, Enum):
    BYPASS_BY_WEIGHT = "bypass_by_weight"
    BYPASS_BY_AREA = "bypass_by_area"
    ACCEPT = "accept"


@dataclass(frozen=True)
class ControlResult:
    decision: ControlDecision
    weight_kg: float
    weight_ratio: float
    occupancy_ratio: float | None
    num_detections: int


class ElevatorController:
    def __init__(
        self,
        detector: YOLOv8Detector,
        occupancy_estimator: OccupancyEstimator,
        max_weight_kg: float,
        weight_bypass_ratio: float,
        area_bypass_ratio: float,
    ) -> None:
        self.detector = detector
        self.occupancy_estimator = occupancy_estimator
        self.max_weight_kg = max_weight_kg
        self.weight_bypass_ratio = weight_bypass_ratio
        self.area_bypass_ratio = area_bypass_ratio
        self._logger = get_logger(__name__)

    def decide(self, weight_kg: float, image: np.ndarray) -> ControlResult:
        weight_ratio = weight_kg / self.max_weight_kg

        # Stage 1: weight check
        if weight_ratio >= self.weight_bypass_ratio:
            self._logger.debug(
                f"Bypass (weight): {weight_kg:.1f} kg / {self.max_weight_kg} kg"
            )
            return ControlResult(
                decision=ControlDecision.BYPASS_BY_WEIGHT,
                weight_kg=weight_kg,
                weight_ratio=weight_ratio,
                occupancy_ratio=None,
                num_detections=0,
            )

        # Stage 2: vision-based area check
        detections = self.detector.detect(image)
        occupancy = self.occupancy_estimator.estimate(detections)

        if occupancy >= self.area_bypass_ratio:
            decision = ControlDecision.BYPASS_BY_AREA
            self._logger.debug(f"Bypass (area): occ={occupancy:.3f}")
        else:
            decision = ControlDecision.ACCEPT
            self._logger.debug(f"Accept: occ={occupancy:.3f}")

        return ControlResult(
            decision=decision,
            weight_kg=weight_kg,
            weight_ratio=weight_ratio,
            occupancy_ratio=occupancy,
            num_detections=len(detections),
        )
