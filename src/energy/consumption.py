"""Elevator power and energy estimator.

This module implements a simplified physical energy model. Total session
energy decomposes as

.. math::

    E_\\text{tot} \\;=\\; E_\\text{stationary} \\;+\\; E_\\text{running}.

For each *start* (a floor-to-floor traversal of distance :math:`h`):

.. math::

    E_\\text{potential} &= \\big(m_\\text{load} - K \\cdot m_\\text{nominal}\\big)\\,g\\,h \\\\
    E_\\text{running}   &= \\begin{cases}
        E_\\text{potential} / \\eta & \\text{if } E_\\text{potential} \\ge 0\\\\
        E_\\text{potential} \\cdot \\eta & \\text{otherwise (regenerative)}
    \\end{cases} \\\\
    E_\\text{start} &= E_\\text{running}
        + (P_\\text{idle} + P_\\text{control})\\,t_\\text{start}

Stationary stretches follow a three-tier idle / standby schedule:

* :math:`t \\le 5\\,\\text{min}`  : :math:`P_\\text{idle}`
* :math:`5\\text{–}30\\,\\text{min}`: :math:`P_\\text{standby,5}`
* :math:`t > 30\\,\\text{min}`     : :math:`P_\\text{standby,30}`

The high-level entry point is :func:`estimate_session_energy`, which a
control simulator calls after each scheduled stop to compare "smart
bypass" against "naive accept".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GRAVITY_MPS2 = 9.81


@dataclass
class EnergyParams:
    """All physical and power parameters required by the energy model."""

    # Cabin / load
    empty_car_mass_kg: float = 800.0
    rated_load_kg: float = 630.0
    counterweight_K: float = 0.45
    motor_efficiency: float = 0.85
    regenerative: bool = True

    # Motion
    rated_speed_mps: float = 1.0
    acceleration_mps2: float = 0.8
    floor_height_m: float = 3.0

    # Auxiliary power components (W)
    power_idle_w: float = 50.0
    power_control_w: float = 30.0
    power_standby_5min_w: float = 30.0
    power_standby_30min_w: float = 15.0
    power_doors_w: float = 100.0

    # Time constants (s)
    door_open_close_time_s: float = 3.0
    stop_time_s: float = 4.0

    @classmethod
    def from_config(cls, cfg: dict) -> EnergyParams:
        elev = cfg["elevator"]
        en = cfg["energy"]
        return cls(
            empty_car_mass_kg=elev["empty_car_mass_kg"],
            rated_load_kg=elev["max_weight_kg"],
            counterweight_K=elev["counterweight_K"],
            motor_efficiency=elev["motor_efficiency"],
            regenerative=en.get("regenerative", True),
            rated_speed_mps=elev["rated_speed_mps"],
            acceleration_mps2=elev["acceleration_mps2"],
            floor_height_m=elev["floor_height_m"],
            power_idle_w=en["power_idle_w"],
            power_control_w=en["power_control_w"],
            power_standby_5min_w=en["power_standby_5min_w"],
            power_standby_30min_w=en["power_standby_30min_w"],
            power_doors_w=en["power_doors_w"],
            door_open_close_time_s=en["door_open_close_time_s"],
            stop_time_s=en["stop_time_s"],
        )


@dataclass
class StartProfile:
    """One start = traversal between two floors.

    Attributes:
        load_kg: passenger + cargo mass on the car side.
        floors_traveled: number of floors covered (>=1).
        direction_up: True if going up.
    """

    load_kg: float
    floors_traveled: int
    direction_up: bool = True


# ─────────────────────────────────────────────────────────────────────
def _start_time_s(distance_m: float, p: EnergyParams) -> float:
    """Trapezoidal velocity profile time:  t = v/a + d/v  if d > v^2/a."""
    v = p.rated_speed_mps
    a = p.acceleration_mps2
    full_accel_distance = v * v / a
    if distance_m >= full_accel_distance:
        return distance_m / v + v / a
    # Triangular profile (never reaches v)
    return 2 * math.sqrt(distance_m / a)


def estimate_running_energy(
    load_kg: float, distance_m: float, direction_up: bool, p: EnergyParams
) -> float:
    """Running energy of one start in joules (E_potential / eta or, when
    regenerative, E_potential * eta)."""
    sign = 1.0 if direction_up else -1.0
    net_mass = (p.empty_car_mass_kg + load_kg) - (
        p.empty_car_mass_kg + p.counterweight_K * p.rated_load_kg
    )
    e_potential = net_mass * GRAVITY_MPS2 * (sign * distance_m)
    if e_potential >= 0:
        return e_potential / p.motor_efficiency
    return e_potential * p.motor_efficiency if p.regenerative else 0.0


def estimate_stop_energy(start: StartProfile, p: EnergyParams) -> dict:
    """Total energy consumed for one start + the stop/door cycle that ends it.

    Returns a dict of energy components in Joules + total in kJ.
    """
    distance_m = start.floors_traveled * p.floor_height_m
    e_run = estimate_running_energy(start.load_kg, distance_m, start.direction_up, p)
    t_start = _start_time_s(distance_m, p)
    e_aux_motion = (p.power_idle_w + p.power_control_w) * t_start
    e_doors = p.power_doors_w * p.door_open_close_time_s * 2  # open + close
    e_stop_idle = (p.power_idle_w + p.power_control_w) * p.stop_time_s
    e_total = e_run + e_aux_motion + e_doors + e_stop_idle
    return {
        "running_j": e_run,
        "aux_motion_j": e_aux_motion,
        "doors_j": e_doors,
        "stop_idle_j": e_stop_idle,
        "total_j": e_total,
        "total_kj": e_total / 1000.0,
        "total_wh": e_total / 3600.0,
        "duration_s": t_start + p.stop_time_s + 2 * p.door_open_close_time_s,
    }


def estimate_stationary_energy(idle_seconds: float, p: EnergyParams) -> float:
    """ISO 25745-2 three-tier stationary energy in Joules."""
    if idle_seconds <= 0:
        return 0.0
    t1 = min(idle_seconds, 300.0)  # 0–5 min
    t2 = max(0.0, min(idle_seconds, 1800.0) - 300.0)  # 5–30 min
    t3 = max(0.0, idle_seconds - 1800.0)  # >30 min
    return p.power_idle_w * t1 + p.power_standby_5min_w * t2 + p.power_standby_30min_w * t3


@dataclass
class SessionResult:
    starts: int
    bypassed: int
    total_running_j: float
    total_aux_j: float
    total_doors_j: float
    total_stationary_j: float
    total_j: float

    @property
    def total_kwh(self) -> float:
        return self.total_j / 3_600_000.0


def estimate_session_energy(
    accepted_starts: list[StartProfile],
    bypassed_starts: list[StartProfile],
    stationary_seconds: float,
    p: EnergyParams,
) -> SessionResult:
    """Roll up energy across one simulation session.

    Bypassed starts contribute zero stop-energy (this is the savings).
    `stationary_seconds` is total time the cabin sat idle between motions.
    """
    run, aux, doors = 0.0, 0.0, 0.0
    for s in accepted_starts:
        ec = estimate_stop_energy(s, p)
        run += ec["running_j"]
        aux += ec["aux_motion_j"] + ec["stop_idle_j"]
        doors += ec["doors_j"]
    e_stat = estimate_stationary_energy(stationary_seconds, p)
    return SessionResult(
        starts=len(accepted_starts),
        bypassed=len(bypassed_starts),
        total_running_j=run,
        total_aux_j=aux,
        total_doors_j=doors,
        total_stationary_j=e_stat,
        total_j=run + aux + doors + e_stat,
    )
