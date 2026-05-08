"""Smoke tests for the lightweight Detection dataclass."""

from __future__ import annotations

from src.detection.detector import Detection


def test_detection_geometric_properties() -> None:
    d = Detection(class_id=0, class_name="person", confidence=0.87, bbox=(10, 20, 110, 220))
    assert d.width_px == 100
    assert d.height_px == 200
    assert d.area_px == 20_000
    assert d.bottom_center_px == (60.0, 220.0)
    assert d.center_px == (60.0, 120.0)


def test_detection_immutable() -> None:
    d = Detection(class_id=1, class_name="stroller", confidence=0.91, bbox=(0, 0, 50, 50))
    try:
        d.confidence = 0.5  # type: ignore[misc]
    except Exception:
        return  # frozen dataclass behaves as expected
    raise AssertionError("Detection must be frozen / immutable.")
