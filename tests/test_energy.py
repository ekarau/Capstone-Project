"""Smoke tests for the Tukia (2018) energy model."""

from __future__ import annotations

import pytest
from src.energy.consumption import (
    EnergyParams,
    StartProfile,
    estimate_running_energy,
    estimate_session_energy,
    estimate_stationary_energy,
    estimate_stop_energy,
)


@pytest.fixture
def params() -> EnergyParams:
    return EnergyParams()  # all defaults


def test_running_energy_loaded_car_going_up_costs_energy(params: EnergyParams) -> None:
    e = estimate_running_energy(load_kg=400.0, distance_m=3.0, direction_up=True, p=params)
    assert e > 0  # heavier than counterweight, going up = positive energy


def test_running_energy_empty_car_going_up_regenerates(params: EnergyParams) -> None:
    e = estimate_running_energy(load_kg=0.0, distance_m=3.0, direction_up=True, p=params)
    # counterweight (K * rated) > empty car alone, so net force pulls car up
    # → motor regenerates → negative energy with regen=True
    assert e < 0


def test_running_energy_no_regen_returns_zero(params: EnergyParams) -> None:
    params.regenerative = False
    e = estimate_running_energy(load_kg=0.0, distance_m=3.0, direction_up=True, p=params)
    assert e == 0.0


def test_stop_energy_components_sum_to_total(params: EnergyParams) -> None:
    profile = StartProfile(load_kg=200.0, floors_traveled=3, direction_up=True)
    result = estimate_stop_energy(profile, params)
    components = (
        result["running_j"] + result["aux_motion_j"] + result["doors_j"] + result["stop_idle_j"]
    )
    assert components == pytest.approx(result["total_j"])


def test_stationary_three_tier_schedule(params: EnergyParams) -> None:
    # 1 minute of idle = idle power × 60s
    e_short = estimate_stationary_energy(60.0, params)
    assert e_short == pytest.approx(params.power_idle_w * 60.0)

    # 10 minutes = 5 min idle + 5 min standby_5min
    e_med = estimate_stationary_energy(600.0, params)
    expected = params.power_idle_w * 300.0 + params.power_standby_5min_w * 300.0
    assert e_med == pytest.approx(expected)


def test_session_bypass_saves_energy(params: EnergyParams) -> None:
    starts = [StartProfile(load_kg=300.0, floors_traveled=2) for _ in range(5)]
    accepted_all = estimate_session_energy(starts, [], 0.0, params)
    accepted_3 = estimate_session_energy(starts[:3], starts[3:], 0.0, params)
    assert accepted_3.total_j < accepted_all.total_j
    assert accepted_3.bypassed == 2
