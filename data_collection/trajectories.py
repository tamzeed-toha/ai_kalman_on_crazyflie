"""
Trajectory/motif library for Phase 3 data collection.

Key finding driving this design (elaborate_plan.md Sec. 2, from Cellini et al. 2025):
altitude `z` is only observable from optical flow + accelerometer during *horizontal
acceleration* (speeding up/slowing down) -- not during hover or constant-velocity flight.
`accel_decel_pulse` is therefore the highest-value motif in this library; `vertical_bob` is
included specifically as a negative control (the paper predicts near-zero z-observability
from vertical motion alone via this sensor set).

All motifs are built net-zero-displacement (a mirrored/computed return leg) since velocity-
setpoint control has no absolute position feedback and many trajectories are chained in one
flight with no independent ground truth to correct accumulated drift against (see safety.py's
geofence, which is a backstop, not a guarantee).

Every generator has the signature:
    generator(params: dict, height_m: float) -> (duration_s: float, vel_fn: Callable[[float], VelCmd])
vel_fn(t) returns the commanded VelCmd for elapsed time t (seconds) since the rep started;
callers should treat t >= duration_s as an error (the flight loop caps at min(duration_s,
spec.max_duration_s), so vel_fn is never actually called past its own duration_s).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import config

DEFAULT_HEIGHT_M = config.DEFAULT_HEIGHT_M
DEFAULT_MAX_REP_DURATION_S = config.DEFAULT_MAX_REP_DURATION_S


@dataclass(frozen=True)
class VelCmd:
    vx: float
    vy: float
    yaw_rate: float
    height: float
    phase: str


@dataclass(frozen=True)
class TrajectorySpec:
    id: str
    motif_type: str
    params: dict
    height_m: float = DEFAULT_HEIGHT_M
    reps_required: int = 2
    max_duration_s: float = DEFAULT_MAX_REP_DURATION_S
    description: str = ""


def _wrap_to_pi(angle_rad: float) -> float:
    return ((angle_rad + math.pi) % (2.0 * math.pi)) - math.pi


# ---------------------------------------------------------------------------
# Motif generators
# ---------------------------------------------------------------------------

def hover_profile(params: dict, height_m: float) -> Tuple[float, Callable[[float], VelCmd]]:
    duration_s = params["duration_s"]

    def vel_fn(t: float) -> VelCmd:
        return VelCmd(0.0, 0.0, 0.0, height_m, "hold")

    return duration_s, vel_fn


def _trapezoid_speed(t: float, ramp_time_s: float, cruise_s: float, peak_speed_mps: float) -> float:
    """Speed at time t within one [0, 2*ramp_time_s + cruise_s] trapezoidal leg."""
    leg_s = 2.0 * ramp_time_s + cruise_s
    if t < 0.0 or t > leg_s:
        return 0.0
    if t < ramp_time_s:
        return peak_speed_mps * (t / ramp_time_s) if ramp_time_s > 0 else peak_speed_mps
    if t < ramp_time_s + cruise_s:
        return peak_speed_mps
    t2 = t - (ramp_time_s + cruise_s)
    return peak_speed_mps * (1.0 - t2 / ramp_time_s) if ramp_time_s > 0 else 0.0


def accel_decel_pulse_profile(params: dict, height_m: float) -> Tuple[float, Callable[[float], VelCmd]]:
    """
    The key motif: horizontal accel/decel pulse, then a mirrored negative-direction pulse to
    cancel net displacement. This is the motif that makes z observable per the paper's finding.
    """
    accel_mps2 = params["accel_mps2"]
    peak_speed_mps = params["peak_speed_mps"]
    cruise_s = params.get("cruise_s", 0.0)
    axis = params.get("axis", "x")
    pause_s = params.get("pause_s", 0.5)

    ramp_time_s = peak_speed_mps / accel_mps2
    leg_s = 2.0 * ramp_time_s + cruise_s
    duration_s = 2.0 * leg_s + 2.0 * pause_s

    def vel_fn(t: float) -> VelCmd:
        if t < leg_s:
            speed = _trapezoid_speed(t, ramp_time_s, cruise_s, peak_speed_mps)
            phase = "outbound"
        elif t < leg_s + pause_s:
            speed = 0.0
            phase = "pause_mid"
        elif t < 2.0 * leg_s + pause_s:
            t2 = t - (leg_s + pause_s)
            speed = -_trapezoid_speed(t2, ramp_time_s, cruise_s, peak_speed_mps)
            phase = "return"
        else:
            speed = 0.0
            phase = "pause_end"

        vx = speed if axis == "x" else 0.0
        vy = speed if axis == "y" else 0.0
        return VelCmd(vx, vy, 0.0, height_m, phase)

    return duration_s, vel_fn


def vertical_bob_profile(params: dict, height_m: float) -> Tuple[float, Callable[[float], VelCmd]]:
    """
    Negative-control motif: vertical oscillation only, zero horizontal velocity. The paper
    predicts near-zero z-observability from this motion via flow+accel alone -- included to
    verify that prediction empirically on real data, not because it's expected to help.
    """
    amplitude_m = params["amplitude_m"]
    period_s = params["period_s"]
    n_cycles = params.get("n_cycles", 3)
    base_height_m = params.get("base_height_m", height_m)
    duration_s = n_cycles * period_s

    def vel_fn(t: float) -> VelCmd:
        h = base_height_m + amplitude_m * math.sin(2.0 * math.pi * t / period_s)
        return VelCmd(0.0, 0.0, 0.0, h, "bob")

    return duration_s, vel_fn


def offset_turn_profile(params: dict, height_m: float) -> Tuple[float, Callable[[float], VelCmd]]:
    """
    Straight -> coordinated yaw+forward turn -> straight, then a computed return: turn to face
    home, translate home, turn to restore original heading. Unlike accel_decel_pulse, a simple
    sign-mirror doesn't undo a turn (rotation isn't its own inverse under negation), so the
    return leg is derived by forward-Euler integrating the outbound world-frame path at
    spec-build time and solving for the turn/translate/turn return that closes it.
    """
    forward_speed_mps = params["forward_speed_mps"]
    yaw_rate_deg_s = params["yaw_rate_deg_s"]
    straight_s = params.get("straight_s", 1.0)
    turn_s = params.get("turn_s", 1.0)
    pause_s = params.get("pause_s", 0.5)

    outbound = [
        (straight_s, forward_speed_mps, 0.0, "outbound_straight_1"),
        (turn_s, forward_speed_mps, yaw_rate_deg_s, "outbound_turn"),
        (straight_s, forward_speed_mps, 0.0, "outbound_straight_2"),
    ]

    x, y, heading_rad = 0.0, 0.0, 0.0
    dt_integrate = 0.02
    for seg_dur, vx, yaw_rate_dps, _phase in outbound:
        n_steps = max(1, int(seg_dur / dt_integrate))
        step = seg_dur / n_steps
        yaw_rate_rad_s = math.radians(yaw_rate_dps)
        for _ in range(n_steps):
            x += vx * math.cos(heading_rad) * step
            y += vx * math.sin(heading_rad) * step
            heading_rad += yaw_rate_rad_s * step

    dist_home_m = math.hypot(x, y)
    bearing_home_rad = math.atan2(-y, -x) if dist_home_m > 1e-6 else heading_rad
    turn_to_home_rad = _wrap_to_pi(bearing_home_rad - heading_rad)
    turn_to_home_s = abs(math.degrees(turn_to_home_rad)) / yaw_rate_deg_s if yaw_rate_deg_s else 0.0
    turn_to_home_sign = 1.0 if turn_to_home_rad >= 0 else -1.0

    return_translate_s = dist_home_m / forward_speed_mps if forward_speed_mps else 0.0

    heading_after_turn_home = heading_rad + turn_to_home_rad
    turn_restore_rad = _wrap_to_pi(0.0 - heading_after_turn_home)
    turn_restore_s = abs(math.degrees(turn_restore_rad)) / yaw_rate_deg_s if yaw_rate_deg_s else 0.0
    turn_restore_sign = 1.0 if turn_restore_rad >= 0 else -1.0

    segments = outbound + [
        (pause_s, 0.0, 0.0, "pause_mid"),
        (turn_to_home_s, 0.0, turn_to_home_sign * yaw_rate_deg_s, "turn_to_home"),
        (return_translate_s, forward_speed_mps, 0.0, "return_straight"),
        (turn_restore_s, 0.0, turn_restore_sign * yaw_rate_deg_s, "restore_heading"),
        (pause_s, 0.0, 0.0, "pause_end"),
    ]

    boundaries: List[Tuple[float, float, float, float, str]] = []
    cum = 0.0
    for seg_dur, vx, yaw_rate_dps, phase in segments:
        boundaries.append((cum, cum + seg_dur, vx, yaw_rate_dps, phase))
        cum += seg_dur
    duration_s = cum

    def vel_fn(t: float) -> VelCmd:
        for start, end, vx, yaw_rate_dps, phase in boundaries:
            if start <= t < end:
                return VelCmd(vx, 0.0, yaw_rate_dps, height_m, phase)
        return VelCmd(0.0, 0.0, 0.0, height_m, "pause_end")

    return duration_s, vel_fn


def _integrate_residual_displacement(speed_fn: Callable[[float], float], duration_s: float, dt: float = 0.02) -> float:
    """Trapezoidal-rule integral of speed_fn over [0, duration_s]."""
    n_steps = max(1, int(duration_s / dt))
    step = duration_s / n_steps
    total = 0.0
    prev = speed_fn(0.0)
    for i in range(1, n_steps + 1):
        cur = speed_fn(min(i * step, duration_s))
        total += 0.5 * (prev + cur) * step
        prev = cur
    return total


def sum_of_sines_profile(params: dict, height_m: float) -> Tuple[float, Callable[[float], VelCmd]]:
    """
    Random-frequency sum-of-sines horizontal velocity profile, mirroring the paper's simulated-
    data generation approach for richer frequency-content coverage than discrete pulses (useful
    for Phase 4 sim-to-real transfer). The residual displacement of a random sum-of-sines is not
    exactly zero, so an explicit closing correction leg is appended, sized by numerically
    integrating the sines' net displacement at spec-build time.
    """
    n_components = params.get("n_components", 3)
    freq_range_hz = params["freq_range_hz"]
    amp_mps = params["amp_mps"]
    duration_s = params["duration_s"]
    axis = params.get("axis", "x")
    seed = params.get("seed", 0)
    pause_s = params.get("pause_s", 0.5)

    rng = random.Random(seed)
    per_component_amp = amp_mps / n_components
    freqs = [rng.uniform(*freq_range_hz) for _ in range(n_components)]
    phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(n_components)]

    def sines_speed(t: float) -> float:
        return sum(
            per_component_amp * math.sin(2.0 * math.pi * f * t + phi)
            for f, phi in zip(freqs, phases)
        )

    residual_m = _integrate_residual_displacement(sines_speed, duration_s)
    correction_speed_mps = max(per_component_amp, 0.05)
    correction_s = abs(residual_m) / correction_speed_mps if correction_speed_mps else 0.0
    correction_sign = -1.0 if residual_m >= 0 else 1.0

    total_duration_s = duration_s + correction_s + pause_s

    def vel_fn(t: float) -> VelCmd:
        if t < duration_s:
            speed = sines_speed(t)
            phase = "sines"
        elif t < duration_s + correction_s:
            speed = correction_sign * correction_speed_mps
            phase = "return_correction"
        else:
            speed = 0.0
            phase = "pause_end"

        vx = speed if axis == "x" else 0.0
        vy = speed if axis == "y" else 0.0
        return VelCmd(vx, vy, 0.0, height_m, phase)

    return total_duration_s, vel_fn


def mixed_free_profile(params: dict, height_m: float) -> Tuple[float, Callable[[float], VelCmd]]:
    """
    Concatenates several curated sub-motifs (each already net-zero-displacement) with a pause
    between, so the concatenation is net-zero overall without needing its own correction leg.
    """
    segments_params = params["segments"]
    pause_s = params.get("pause_s", 0.5)

    boundaries: List[Tuple[float, float, Callable[[float], VelCmd]]] = []
    cum = 0.0
    for seg in segments_params:
        seg_duration_s, seg_vel_fn = MOTIF_GENERATORS[seg["motif_type"]](seg["params"], height_m)
        boundaries.append((cum, cum + seg_duration_s, seg_vel_fn))
        cum += seg_duration_s
        boundaries.append((cum, cum + pause_s, lambda _t, h=height_m: VelCmd(0.0, 0.0, 0.0, h, "pause_between")))
        cum += pause_s
    duration_s = cum

    def vel_fn(t: float) -> VelCmd:
        for start, end, seg_vel_fn in boundaries:
            if start <= t < end:
                return seg_vel_fn(t - start)
        return VelCmd(0.0, 0.0, 0.0, height_m, "pause_end")

    return duration_s, vel_fn


MOTIF_GENERATORS: Dict[str, Callable[[dict, float], Tuple[float, Callable[[float], VelCmd]]]] = {
    "hover": hover_profile,
    "accel_decel_pulse": accel_decel_pulse_profile,
    "vertical_bob": vertical_bob_profile,
    "offset_turn": offset_turn_profile,
    "sum_of_sines": sum_of_sines_profile,
    "mixed_free": mixed_free_profile,
}


# ---------------------------------------------------------------------------
# Library assembly
# ---------------------------------------------------------------------------
# Sized against a 15-flight / ~5-min-per-flight battery budget (~65-70 min usable flight
# time after takeoff/landing overhead): ~108 total rep-instances below take roughly 20-25
# min to fly at these parameters, leaving generous margin for aborted/re-flown reps. Bump
# reps_required for accel_decel_pulse (the highest-value motif) first if more margin is used.

def build_trajectory_library(height_m: float = DEFAULT_HEIGHT_M) -> List[TrajectorySpec]:
    specs: List[TrajectorySpec] = []

    for duration_s in (3.0, 5.0, 8.0):
        specs.append(TrajectorySpec(
            id=f"hover_d{duration_s:.0f}s",
            motif_type="hover",
            params={"duration_s": duration_s},
            height_m=height_m,
            reps_required=2,
            description="Baseline: zero horizontal velocity, low-observability control condition.",
        ))

    for axis in ("x", "y"):
        accels = (0.2, 0.4, 0.6, 0.8) if axis == "x" else (0.3, 0.6)
        cruises = (0.0, 0.5, 1.0) if axis == "x" else (0.0, 0.5)
        for accel_mps2 in accels:
            for cruise_s in cruises:
                peak_speed_mps = min(0.5, accel_mps2 * 0.75)
                specs.append(TrajectorySpec(
                    id=f"accel_decel_a{accel_mps2:.2f}_c{cruise_s:.2f}_{axis}",
                    motif_type="accel_decel_pulse",
                    params={
                        "accel_mps2": accel_mps2, "peak_speed_mps": peak_speed_mps,
                        "cruise_s": cruise_s, "axis": axis,
                    },
                    height_m=height_m,
                    reps_required=4,
                    description="Key motif: horizontal accel/decel pulse (z observable per paper's finding).",
                ))

    for amplitude_m in (0.1, 0.2):
        for period_s in (2.0, 3.0):
            specs.append(TrajectorySpec(
                id=f"vertical_bob_amp{amplitude_m:.2f}_T{period_s:.1f}",
                motif_type="vertical_bob",
                params={"amplitude_m": amplitude_m, "period_s": period_s, "n_cycles": 3},
                height_m=height_m,
                reps_required=2,
                description="Negative control: vertical-only motion, predicted near-zero z observability.",
            ))

    for forward_speed_mps in (0.2, 0.3):
        for yaw_rate_deg_s in (30.0, 60.0):
            specs.append(TrajectorySpec(
                id=f"offset_turn_v{forward_speed_mps:.2f}_yr{yaw_rate_deg_s:.0f}",
                motif_type="offset_turn",
                params={
                    "forward_speed_mps": forward_speed_mps, "yaw_rate_deg_s": yaw_rate_deg_s,
                    "straight_s": 1.0, "turn_s": 1.5,
                },
                height_m=height_m,
                reps_required=2,
                max_duration_s=25.0,
                description="Coordinated yaw+forward turn, decoupling heading from velocity direction.",
            ))

    for freq_range_hz in ((0.1, 0.3), (0.3, 0.6)):
        for amp_mps in (0.2, 0.4):
            for axis in ("x", "y"):
                specs.append(TrajectorySpec(
                    id=f"sines_{axis}_f{freq_range_hz[0]:.2f}-{freq_range_hz[1]:.2f}_a{amp_mps:.2f}",
                    motif_type="sum_of_sines",
                    params={
                        "n_components": 3, "freq_range_hz": freq_range_hz, "amp_mps": amp_mps,
                        "duration_s": 8.0, "axis": axis, "seed": hash((freq_range_hz, amp_mps, axis)) % (2**31),
                    },
                    height_m=height_m,
                    reps_required=2,
                    description="Continuous random-frequency acceleration profile for frequency coverage.",
                ))

    mixed_variants = [
        [
            {"motif_type": "accel_decel_pulse", "params": {"accel_mps2": 0.4, "peak_speed_mps": 0.3, "cruise_s": 0.5, "axis": "x"}},
            {"motif_type": "offset_turn", "params": {"forward_speed_mps": 0.2, "yaw_rate_deg_s": 45.0, "straight_s": 1.0, "turn_s": 1.0}},
        ],
        [
            {"motif_type": "vertical_bob", "params": {"amplitude_m": 0.15, "period_s": 2.5, "n_cycles": 2}},
            {"motif_type": "accel_decel_pulse", "params": {"accel_mps2": 0.6, "peak_speed_mps": 0.4, "cruise_s": 0.0, "axis": "y"}},
        ],
        [
            {"motif_type": "sum_of_sines", "params": {"n_components": 2, "freq_range_hz": (0.2, 0.4), "amp_mps": 0.3, "duration_s": 6.0, "axis": "x", "seed": 7}},
            {"motif_type": "offset_turn", "params": {"forward_speed_mps": 0.25, "yaw_rate_deg_s": 30.0, "straight_s": 0.8, "turn_s": 1.2}},
        ],
    ]
    for idx, segments in enumerate(mixed_variants):
        specs.append(TrajectorySpec(
            id=f"mixed_free_{idx}",
            motif_type="mixed_free",
            params={"segments": segments},
            height_m=height_m,
            reps_required=2,
            max_duration_s=30.0,
            description="Curated concatenation of motifs, closer to unstructured real flight.",
        ))

    return specs
