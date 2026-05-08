"""Baseline vs Smart comparison.

Given a synthetic stream of "hall calls" with known weight & cabin image,
runs both control strategies and reports unnecessary-stops, AWT, and
energy consumption deltas.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from src.control.algorithm import ControlDecision, ElevatorController
from src.energy.consumption import EnergyParams, StartProfile, estimate_session_energy


@dataclass
class HallCall:
    floor: int
    weight_kg: float
    cabin_image: np.ndarray
    expected_passengers_at_floor: int = 1


@dataclass
class StrategyResult:
    accepted: int = 0
    bypassed: int = 0
    bypass_by_weight: int = 0
    bypass_by_area: int = 0
    avg_wait_time_s: float = 0.0
    energy: object | None = None  # SessionResult


@dataclass
class ComparisonResult:
    baseline: StrategyResult = field(default_factory=StrategyResult)
    smart: StrategyResult = field(default_factory=StrategyResult)

    def savings_pct(self) -> dict:
        if self.baseline.energy is None or self.smart.energy is None:
            return {}
        b = self.baseline.energy.total_j
        s = self.smart.energy.total_j
        return {
            "energy_saved_pct": (b - s) / b * 100 if b > 0 else 0.0,
            "stops_saved_pct": (
                (self.baseline.accepted - self.smart.accepted) / max(1, self.baseline.accepted) * 100
            ),
        }


def run_strategy(
    controller: ElevatorController,
    calls: list[HallCall],
    energy_params: EnergyParams,
    seed: int = 42,
) -> StrategyResult:
    rng = random.Random(seed)
    accepted_profiles: list[StartProfile] = []
    bypassed_profiles: list[StartProfile] = []
    res = StrategyResult()

    for call in calls:
        result = controller.decide(call.weight_kg, call.cabin_image)
        # Each call corresponds to ~1 floor traversal
        profile = StartProfile(
            load_kg=call.weight_kg,
            floors_traveled=1,
            direction_up=rng.random() < 0.5,
        )
        if result.decision == ControlDecision.ACCEPT:
            res.accepted += 1
            accepted_profiles.append(profile)
        else:
            res.bypassed += 1
            bypassed_profiles.append(profile)
            if result.decision == ControlDecision.BYPASS_BY_WEIGHT:
                res.bypass_by_weight += 1
            elif result.decision == ControlDecision.BYPASS_BY_AREA:
                res.bypass_by_area += 1

    res.energy = estimate_session_energy(
        accepted_profiles,
        bypassed_profiles,
        stationary_seconds=0.0,
        p=energy_params,
    )

    # Toy AWT model: each accepted stop adds ~stop_time_s to others' wait
    res.avg_wait_time_s = res.accepted * energy_params.stop_time_s / max(1, len(calls))
    return res
