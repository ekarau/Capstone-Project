"""Smoke tests for the two-stage hall-call controller.

The controller depends on a YOLOv8 detector and an occupancy estimator;
we substitute lightweight fakes so the tests are model- and image-free.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.control.algorithm import ControlDecision, ControlResult, ElevatorController


class _FakeDetector:
    """Returns a fixed list of detections regardless of input image."""

    def __init__(self, detections: list | None = None) -> None:
        self._detections = detections or []

    def detect(self, image: np.ndarray) -> list:
        return list(self._detections)


class _FakeOccupancy:
    """Returns a constant occupancy ratio."""

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def estimate(self, detections) -> float:
        return self.ratio


@pytest.fixture
def dummy_image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_controller(
    occupancy: float, weight_thr: float = 0.80, area_thr: float = 0.90
) -> ElevatorController:
    return ElevatorController(
        detector=_FakeDetector(),
        occupancy_estimator=_FakeOccupancy(occupancy),
        max_weight_kg=630.0,
        weight_bypass_ratio=weight_thr,
        area_bypass_ratio=area_thr,
    )


def test_weight_bypass_takes_priority(dummy_image: np.ndarray) -> None:
    # 600 kg / 630 kg = 0.95 ≥ 0.80 → weight bypass; occupancy ignored.
    ctrl = _make_controller(occupancy=0.10)
    result = ctrl.decide(weight_kg=600.0, image=dummy_image)
    assert result.decision == ControlDecision.BYPASS_BY_WEIGHT
    assert result.occupancy_ratio is None  # vision not consulted


def test_area_bypass_when_weight_below_threshold(dummy_image: np.ndarray) -> None:
    ctrl = _make_controller(occupancy=0.95)
    result = ctrl.decide(weight_kg=200.0, image=dummy_image)
    assert result.decision == ControlDecision.BYPASS_BY_AREA
    assert result.occupancy_ratio == 0.95


def test_accept_when_neither_threshold_reached(dummy_image: np.ndarray) -> None:
    ctrl = _make_controller(occupancy=0.30)
    result = ctrl.decide(weight_kg=200.0, image=dummy_image)
    assert result.decision == ControlDecision.ACCEPT
    assert result.weight_ratio == pytest.approx(200.0 / 630.0)


def test_at_weight_threshold_triggers_bypass(dummy_image: np.ndarray) -> None:
    ctrl = _make_controller(occupancy=0.10, weight_thr=0.80)
    result = ctrl.decide(weight_kg=0.80 * 630.0, image=dummy_image)
    assert result.decision == ControlDecision.BYPASS_BY_WEIGHT


def test_control_result_is_immutable() -> None:
    r = ControlResult(
        decision=ControlDecision.ACCEPT,
        weight_kg=100.0,
        weight_ratio=0.16,
        occupancy_ratio=0.25,
        num_detections=2,
    )
    try:
        r.decision = ControlDecision.BYPASS_BY_WEIGHT  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ControlResult must be frozen.")
