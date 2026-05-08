"""YAML config loader with override / CLI hooks.

`ElevatorConfig` is a thin wrapper around a dict that exposes structured
sections (elevator, thresholds, model, ...) and supports merging multiple
YAML files. Cabin geometry parameters are also overridable at runtime.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (returns new dict)."""
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(*paths: str | Path) -> dict:
    """Load and merge multiple YAML config files (later overrides earlier)."""
    if not paths:
        raise ValueError("En az bir config yolu verin.")
    merged: dict = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Config bulunamadı: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)
    return merged


class ElevatorConfig:
    """Convenience wrapper that exposes config sections as attributes.

    Usage:
        cfg = ElevatorConfig.from_yaml("configs/default.yaml")
        cfg.elevator.width_m            # 1.4
        cfg.with_cabin(width_m=1.7)     # returns a NEW config with override
    """

    def __init__(self, data: dict) -> None:
        self._data = data
        # Auto-recompute total_floor_area_m2 if missing/inconsistent
        elev = self._data.get("elevator", {})
        if "width_m" in elev and "depth_m" in elev:
            elev["total_floor_area_m2"] = float(elev["width_m"]) * float(elev["depth_m"])

    @classmethod
    def from_yaml(cls, *paths: str | Path) -> "ElevatorConfig":
        return cls(load_config(*paths))

    @classmethod
    def from_dict(cls, data: dict) -> "ElevatorConfig":
        return cls(deepcopy(data))

    def to_dict(self) -> dict:
        return deepcopy(self._data)

    # Section accessors --------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return _DotDict(value)
            return value
        raise AttributeError(f"Config'te '{name}' yok.")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # Overrides ----------------------------------------------------------------
    def with_overrides(self, **section_overrides: dict) -> "ElevatorConfig":
        """Return a NEW config with given section overrides merged in.

        Example:
            cfg2 = cfg.with_overrides(elevator={"width_m": 1.7, "depth_m": 1.8})
        """
        return ElevatorConfig(_deep_merge(self._data, section_overrides))

    def with_cabin(
        self,
        *,
        width_m: float | None = None,
        depth_m: float | None = None,
        height_m: float | None = None,
        max_weight_kg: float | None = None,
    ) -> "ElevatorConfig":
        """Convenience override for cabin geometry."""
        update: dict = {}
        if width_m is not None:
            update["width_m"] = float(width_m)
        if depth_m is not None:
            update["depth_m"] = float(depth_m)
        if height_m is not None:
            update["height_m"] = float(height_m)
        if max_weight_kg is not None:
            update["max_weight_kg"] = float(max_weight_kg)
        return self.with_overrides(elevator=update)


class _DotDict:
    """Lightweight read-only dict wrapper that supports attribute access."""

    __slots__ = ("_d",)

    def __init__(self, d: dict) -> None:
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str) -> Any:
        d = object.__getattribute__(self, "_d")
        if name not in d:
            raise AttributeError(name)
        v = d[name]
        return _DotDict(v) if isinstance(v, dict) else v

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()

    def to_dict(self) -> dict:
        return deepcopy(self._d)
