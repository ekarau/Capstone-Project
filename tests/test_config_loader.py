"""Smoke tests for the YAML config loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.utils.config_loader import ElevatorConfig, load_config


@pytest.fixture
def base_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "base.yaml"
    p.write_text(
        textwrap.dedent(
            """
            elevator:
              width_m: 1.4
              depth_m: 1.6
              max_weight_kg: 630
            thresholds:
              area_bypass_ratio: 0.90
            """
        ).strip()
    )
    return p


def test_load_config_returns_dict(base_yaml: Path) -> None:
    cfg = load_config(base_yaml)
    assert cfg["elevator"]["width_m"] == 1.4
    assert cfg["thresholds"]["area_bypass_ratio"] == 0.90


def test_load_config_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_no_paths_raises() -> None:
    with pytest.raises(ValueError):
        load_config()


def test_elevator_config_dot_access(base_yaml: Path) -> None:
    cfg = ElevatorConfig.from_yaml(base_yaml)
    assert cfg.elevator.width_m == 1.4
    assert cfg.thresholds.area_bypass_ratio == 0.90


def test_elevator_config_auto_recomputes_floor_area(base_yaml: Path) -> None:
    cfg = ElevatorConfig.from_yaml(base_yaml)
    assert cfg.elevator.total_floor_area_m2 == pytest.approx(1.4 * 1.6)


def test_with_cabin_returns_new_config(base_yaml: Path) -> None:
    cfg = ElevatorConfig.from_yaml(base_yaml)
    cfg2 = cfg.with_cabin(width_m=1.7, depth_m=1.8)
    assert cfg.elevator.width_m == 1.4  # original unchanged
    assert cfg2.elevator.width_m == 1.7
    assert cfg2.elevator.total_floor_area_m2 == pytest.approx(1.7 * 1.8)


def test_unknown_attribute_raises(base_yaml: Path) -> None:
    cfg = ElevatorConfig.from_yaml(base_yaml)
    with pytest.raises(AttributeError):
        _ = cfg.nonexistent_section
