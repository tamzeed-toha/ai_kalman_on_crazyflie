"""
Per-tick safety checks: battery, multiranger obstacle-stop, and a position-based geofence.

check_safety() is called every control tick from the main flight loop and composes checks in
priority order -- each stage can override the previous stage's command. Battery is checked
first since it overrides everything; obstacle-stop and geofence then independently clamp
velocity components.

The geofence is a backstop, not a guarantee: stateEstimate.x/y is dead-reckoned by the same
EKF this whole project studies, and there is no independent ground truth available to verify
it against (no Lighthouse/mocap on this setup). Trajectories are designed to be net-zero-
displacement specifically because of this (see trajectories.py).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from trajectories import VelCmd

SafetyResult = Tuple[VelCmd, str, bool, bool]  # (effective_cmd, safety_flag, should_land, should_abort_rep)


def _obstacle_stop(cmd: VelCmd, logger, cfg) -> Tuple[VelCmd, str, bool]:
    vx, vy = cmd.vx, cmd.vy
    flag = ""
    should_land = False

    front_m = logger.get_value("range_front_m")
    back_m = logger.get_value("range_back_m")
    left_m = logger.get_value("range_left_m")
    right_m = logger.get_value("range_right_m")
    up_m = logger.get_value("range_up_m")

    if front_m is not None and front_m < cfg.OBSTACLE_STOP_M and vx > 0:
        vx = 0.0
        flag = "obstacle_front_stop"
    if back_m is not None and back_m < cfg.OBSTACLE_STOP_M and vx < 0:
        vx = 0.0
        flag = "obstacle_back_stop"
    if left_m is not None and left_m < cfg.OBSTACLE_STOP_M and vy > 0:
        vy = 0.0
        flag = "obstacle_left_stop"
    if right_m is not None and right_m < cfg.OBSTACLE_STOP_M and vy < 0:
        vy = 0.0
        flag = "obstacle_right_stop"
    if up_m is not None and 0.0 < up_m < cfg.UP_STOP_M:
        vx, vy = 0.0, 0.0
        flag = "obstacle_up_land"
        should_land = True

    return replace(cmd, vx=vx, vy=vy), flag, should_land


def _geofence_component(pos, vel, bound_min, bound_max, margin):
    """Soft-clamp velocity moving further outside [bound_min, bound_max] near the boundary."""
    if pos is None:
        return vel
    soft_min, soft_max = bound_min + margin, bound_max - margin
    if pos < soft_min and vel < 0:
        return 0.0
    if pos > soft_max and vel > 0:
        return 0.0
    return vel


def _axis_breach_severity(pos, bound_min, bound_max) -> float:
    """0.0 inside bounds, 1.0 exactly at a bound, >1.0 beyond it."""
    if pos is None:
        return 0.0
    center = (bound_min + bound_max) / 2.0
    half_range = (bound_max - bound_min) / 2.0
    if half_range <= 0:
        return 0.0
    return abs(pos - center) / half_range


def _geofence(cmd: VelCmd, logger, cfg) -> Tuple[VelCmd, str, bool, bool]:
    x_m = logger.get_value("x_m")
    y_m = logger.get_value("y_m")
    z_m = logger.get_value("z_m")

    vx = _geofence_component(x_m, cmd.vx, *cfg.GEOFENCE_X_M, cfg.GEOFENCE_SOFT_MARGIN_M)
    vy = _geofence_component(y_m, cmd.vy, *cfg.GEOFENCE_Y_M, cfg.GEOFENCE_SOFT_MARGIN_M)

    severity_x = _axis_breach_severity(x_m, *cfg.GEOFENCE_X_M)
    severity_y = _axis_breach_severity(y_m, *cfg.GEOFENCE_Y_M)
    severity_z = _axis_breach_severity(z_m, *cfg.GEOFENCE_Z_M)
    max_severity = max(severity_x, severity_y, severity_z)

    flag = ""
    should_land = False
    should_abort = False
    if max_severity >= cfg.GEOFENCE_ABORT_FACTOR:
        flag = "geofence_hard_land"
        should_land = True
        should_abort = True
        vx, vy = 0.0, 0.0
    elif max_severity > 1.0:
        flag = "geofence_soft_abort"
        should_abort = True
        vx, vy = 0.0, 0.0

    return replace(cmd, vx=vx, vy=vy), flag, should_land, should_abort


def check_safety(raw_cmd: VelCmd, logger, cfg) -> SafetyResult:
    battery_v = logger.get_value("battery_v")
    if battery_v is not None and battery_v < cfg.BATTERY_HARD_ABORT_V:
        return replace(raw_cmd, vx=0.0, vy=0.0, yaw_rate=0.0), "battery_hard_abort", True, True

    cmd, obstacle_flag, obstacle_land = _obstacle_stop(raw_cmd, logger, cfg)
    if obstacle_land:
        return cmd, obstacle_flag, True, True

    cmd, geofence_flag, geofence_land, geofence_abort = _geofence(cmd, logger, cfg)

    flag = geofence_flag or obstacle_flag
    return cmd, flag, geofence_land, geofence_abort


def battery_ok_to_start_rep(logger, cfg) -> bool:
    """Checked once before starting a new trajectory rep (not every tick)."""
    battery_v = logger.get_value("battery_v")
    return battery_v is None or battery_v >= cfg.BATTERY_SOFT_LAND_V
