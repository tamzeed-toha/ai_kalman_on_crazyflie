"""
Trajectory/motif library for Phase 3 data collection.

Key finding driving this design (elaborate_plan.md Sec. 2, from Cellini et al. 2025):
altitude `z` is only observable from optical flow + accelerometer during *horizontal
acceleration* (speeding up/slowing down) -- not during hover or constant-velocity flight.
`accel_decel_pulse` is therefore the highest-value motif in this library; `vertical_bob` is
included specifically as a negative control (the paper predicts near-zero z-observability
from vertical motion alone via this sensor set).

Altitude sweep (added after the first 146-rep collection campaign showed 87% of real flights
sitting within 1cm of the single fixed height that was used everywhere,
DEFAULT_HEIGHT_M=0.4 -- see collect_data.py's call into build_trajectory_library()): an ANN
altitude estimator trained on data with almost no z variation can satisfy its loss by
predicting the training mean z, without ever learning a real flow-to-altitude relationship.
The paper's own sensor-set observability result (Table 1) says z is only observable given
horizontal acceleration, not given altitude variation alone -- so the fix isn't "fly at more
heights in general", it's specifically "fly the accel/decel-carrying motifs across a spread of
heights" (see ALTITUDE_SWEEP_M / ALTITUDE_SWEEP_SECONDARY_M below and their use in
build_trajectory_library()). `hover`, `vertical_bob`, and `offset_turn` are deliberately left at
the single default height: none of them produce the horizontal acceleration this sensor set
needs to render z observable in the first place, so sweeping their height would add flight time
without adding estimator training signal.

`accel_decel_with_climb` goes one step further: it combines the accel/decel horizontal profile
with a concurrent, slow, continuous altitude ramp within a *single* rep, so both the numerator
(v_x) and denominator (z) of r_x = v_x/z vary together in one continuous trajectory. Cellini et
al. never tested this combination -- their own altitude case study only paired accel/decel with
a *fixed* altitude across repeated trials -- so treat this motif as a plausible engineering
extension worth trying, not a validated finding from the paper.

All motifs are built net-zero-displacement (a mirrored/computed return leg) since velocity-
setpoint control has no absolute position feedback and many trajectories are chained in one
flight with no independent ground truth to correct accumulated drift against (see safety.py's
geofence, which is a backstop, not a guarantee). Altitude is the one exception:
`accel_decel_with_climb` ends at a different height than it started (that's the point), so its
TrajectorySpec.height_m is set to the *end* height, not the start -- collect_data.py uses
spec.height_m for the interstitial hover immediately after the rep, and the drone is physically
at the end height by then, not the start height.

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

# Altitude sweep for the accel/decel-carrying motifs (the ones the paper's Table 1 says can
# actually render z observable via this sensor set -- see module docstring). Spans ~0.3-1.2m,
# safely inside config.GEOFENCE_Z_M=(0.15, 1.3): at z=1.2m the geofence breach severity is
# ~0.83 (1.0 = exactly at a hard bound, see safety.py:_axis_breach_severity), and at z=0.3m it's
# ~0.74 -- both comfortably under the 1.0/1.2 soft/hard abort thresholds with margin to spare.
ALTITUDE_SWEEP_M: Tuple[float, ...] = (0.3, 0.6, 0.9, 1.2)

# A reduced sweep for the secondary accel-carrying motifs (sum_of_sines, zigzag) -- these get
# altitude variety too since they also involve horizontal acceleration, but only 2 points
# instead of 4 to keep the battery-time rebalance in build_trajectory_library() affordable.
ALTITUDE_SWEEP_SECONDARY_M: Tuple[float, ...] = (0.3, 0.9)


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


def accel_decel_with_climb_profile(params: dict, height_m: float) -> Tuple[float, Callable[[float], VelCmd]]:
    """
    accel_decel_pulse's horizontal profile (same trapezoidal outbound/pause/return legs, reused
    verbatim via _trapezoid_speed) plus a concurrent, slow, continuous altitude ramp from
    climb_start_m to climb_end_m spanning the *entire* rep duration (not synced to any one
    phase). That keeps the vertical rate constant and low relative to the horizontal accel/decel
    event, so climbing doesn't dominate or confound the maneuver -- see module docstring for why
    this combination (continuous z change concurrent with horizontal acceleration) is untested by
    Cellini et al. and included here as an engineering extension, not a validated finding.

    Net-zero-displacement horizontally like every other motif, but *not* net-zero vertically by
    design -- climb_end_m != climb_start_m is the point. Callers should set the owning
    TrajectorySpec.height_m to climb_end_m (see build_trajectory_library()), since that's where
    the drone actually is when collect_data.py issues the post-rep interstitial hover.
    """
    accel_mps2 = params["accel_mps2"]
    peak_speed_mps = params["peak_speed_mps"]
    cruise_s = params.get("cruise_s", 0.0)
    axis = params.get("axis", "x")
    pause_s = params.get("pause_s", 0.5)
    climb_start_m = params.get("climb_start_m", height_m)
    climb_end_m = params.get("climb_end_m", height_m)

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
        climb_frac = min(1.0, t / duration_s) if duration_s > 0 else 1.0
        h = climb_start_m + (climb_end_m - climb_start_m) * climb_frac
        return VelCmd(vx, vy, 0.0, h, phase)

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


def _leg_peak_and_cruise(distance_m: float, target_speed_mps: float, accel_mps2: float) -> Tuple[float, float]:
    """
    Speed/cruise for a straight leg covering exactly distance_m: reaches target_speed_mps with
    a cruise segment if there's room for a full trapezoid, otherwise falls back to a triangular
    (no-cruise) profile whose achieved peak is below target_speed_mps -- covering exactly
    distance_m either way, never overshooting into more room than the leg is allotted.
    """
    ramp_time_s = target_speed_mps / accel_mps2
    min_dist_for_target = target_speed_mps * ramp_time_s
    if distance_m >= min_dist_for_target:
        return target_speed_mps, distance_m / target_speed_mps - ramp_time_s
    return math.sqrt(distance_m * accel_mps2), 0.0


def zigzag_profile(params: dict, height_m: float) -> Tuple[float, Callable[[float], VelCmd]]:
    """
    A direct, task-representative motif (as opposed to the systematic magnitude/duration sweeps
    above): several straight legs alternating +/-zigzag_angle_deg off a fixed heading (no yaw --
    body frame stays aligned with the initial heading throughout, so this is a "coverage/search
    pattern" style zigzag, not a banked turn), each a trapezoidal (or triangular, if the leg is
    too short to reach target_speed_mps at the given accel) speed profile, then a single computed
    straight leg back to the start point.

    leg_distance_m/zigzag_angle_deg/n_legs jointly determine the pattern's physical footprint --
    tuned in build_trajectory_library() to stay safely inside the geofence's soft-clamp margin at
    the higher speeds this motif targets (see config.py's GEOFENCE_SOFT_MARGIN_M/OBSTACLE_STOP_M,
    both sized for this motif's top speed).
    """
    leg_distance_m = params.get("leg_distance_m", 0.4)
    leg_speed_mps = params.get("leg_speed_mps", 1.0)
    leg_accel_mps2 = params.get("leg_accel_mps2", 1.5)
    n_legs = params.get("n_legs", 3)
    zigzag_angle_deg = params.get("zigzag_angle_deg", 35.0)
    corner_pause_s = params.get("corner_pause_s", 0.3)

    leg_peak, leg_cruise_s = _leg_peak_and_cruise(leg_distance_m, leg_speed_mps, leg_accel_mps2)
    leg_ramp_time_s = leg_peak / leg_accel_mps2
    leg_duration_s = 2.0 * leg_ramp_time_s + leg_cruise_s
    angle_rad = math.radians(zigzag_angle_deg)

    legs: List[Tuple[float, float, float]] = []  # (start_t, end_t, angle_sign)
    x, y = 0.0, 0.0
    cum = 0.0
    for i in range(n_legs):
        sign = 1.0 if i % 2 == 0 else -1.0
        legs.append((cum, cum + leg_duration_s, sign))
        cum += leg_duration_s
        x += leg_distance_m * math.cos(angle_rad)
        y += sign * leg_distance_m * math.sin(angle_rad)
        if i < n_legs - 1:
            cum += corner_pause_s
    outbound_end_t = cum
    cum += corner_pause_s
    return_start_t = cum

    dist_home_m = math.hypot(x, y)
    heading_home_rad = math.atan2(-y, -x) if dist_home_m > 1e-9 else 0.0
    return_peak, return_cruise_s = _leg_peak_and_cruise(dist_home_m, leg_speed_mps, leg_accel_mps2)
    return_ramp_time_s = return_peak / leg_accel_mps2 if return_peak > 0 else 0.0
    return_duration_s = 2.0 * return_ramp_time_s + return_cruise_s
    cum += return_duration_s
    duration_s = cum

    def vel_fn(t: float) -> VelCmd:
        for idx, (start, end, sign) in enumerate(legs):
            if start <= t < end:
                speed = _trapezoid_speed(t - start, leg_ramp_time_s, leg_cruise_s, leg_peak)
                vx = speed * math.cos(angle_rad)
                vy = sign * speed * math.sin(angle_rad)
                return VelCmd(vx, vy, 0.0, height_m, f"leg_{idx}")
        if outbound_end_t <= t < return_start_t:
            return VelCmd(0.0, 0.0, 0.0, height_m, "corner_pause")
        if return_start_t <= t < duration_s:
            speed = _trapezoid_speed(t - return_start_t, return_ramp_time_s, return_cruise_s, return_peak)
            vx = speed * math.cos(heading_home_rad)
            vy = speed * math.sin(heading_home_rad)
            return VelCmd(vx, vy, 0.0, height_m, "return")
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
    "accel_decel_with_climb": accel_decel_with_climb_profile,
    "vertical_bob": vertical_bob_profile,
    "offset_turn": offset_turn_profile,
    "sum_of_sines": sum_of_sines_profile,
    "mixed_free": mixed_free_profile,
    "zigzag": zigzag_profile,
}


# ---------------------------------------------------------------------------
# Library assembly
# ---------------------------------------------------------------------------
# Sized against a 15-flight / ~5-min-per-flight battery budget (~65-70 min usable flight
# time after takeoff/landing overhead): ~144 total rep-instances below take ~16 min of active
# flight time at these parameters (verify with a quick integration script after any edit here --
# see chat history / adjacent comments for the technique), leaving generous margin for
# aborted/re-flown reps.
#
# Rebalanced for the altitude sweep (see module docstring): crossing accel_decel_pulse with
# ALTITUDE_SWEEP_M (4 heights) would have quadrupled its rep-instances at the old
# accel_mps2/cruise_s breadth and blown the budget, so that breadth was trimmed (low-speed: 4
# accels x 3 cruises -> 2x2 for x-axis, 2x2 -> 1x2 for y-axis; fast: 2x2 -> 2x1 for both axes) and
# reps_required dropped from 4 to 2 per (combo, height) pair -- height variety now supplies part
# of the diversity the extra reps used to. sum_of_sines/zigzag got the same treatment with the
# smaller 2-point ALTITUDE_SWEEP_SECONDARY_M. hover/vertical_bob/offset_turn/mixed_free are
# unchanged from the original library (no height variation -- see module docstring for why).
# Bump reps_required for accel_decel_pulse/zigzag/accel_decel_with_climb (the highest-value
# motifs) first if more margin is used.

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

    # Low-speed accel/decel sweep, now crossed with ALTITUDE_SWEEP_M (see module docstring): this
    # is the motif the paper's Table 1 says can actually render z observable, so it's the one
    # that most needs height variety to stop an ANN altitude estimator from learning "predict the
    # training mean z" (the failure mode that motivated this sweep -- see collect_data.py). The
    # accel_mps2/cruise_s combo breadth below is intentionally trimmed relative to the pre-sweep
    # version (was 4 accels x 3 cruises for x, 2x2 for y) to pay for the 4x altitude multiplier
    # within the same battery budget -- see the accounting comment above build_trajectory_library.
    for axis in ("x", "y"):
        accels = (0.3, 0.6) if axis == "x" else (0.4,)
        cruises = (0.0, 1.0) if axis == "x" else (0.0, 0.5)
        for accel_mps2 in accels:
            for cruise_s in cruises:
                peak_speed_mps = min(0.5, accel_mps2 * 0.75)
                for h in ALTITUDE_SWEEP_M:
                    specs.append(TrajectorySpec(
                        id=f"accel_decel_a{accel_mps2:.2f}_c{cruise_s:.2f}_{axis}_h{h:.2f}",
                        motif_type="accel_decel_pulse",
                        params={
                            "accel_mps2": accel_mps2, "peak_speed_mps": peak_speed_mps,
                            "cruise_s": cruise_s, "axis": axis,
                        },
                        height_m=h,
                        reps_required=2,
                        description=(
                            "Key motif: horizontal accel/decel pulse (z observable per paper's "
                            f"finding), flown at height={h:.2f}m for altitude-sweep coverage."
                        ),
                    ))

    # Higher-speed/higher-accel additions, appended rather than blended into the sweep above so
    # the low-speed combos above stay easy to reason about independently. Deliberately short/
    # no-cruise (brief pulses, not sustained cruise) to reach ~0.75-1.0 m/s while keeping
    # footprint safely inside the geofence's soft-clamp margin -- see the footprint math in the
    # PR/chat history (or recompute via trajectories.MOTIF_GENERATORS + a quick integration
    # script) before widening these further. Motivated by real deployment flight (e.g. ~1 m/s
    # zig-zag maneuvering) being well outside the original 0.5 m/s / 0.8 m/s^2 envelope -- a
    # data-driven filter shouldn't be asked to operate outside the range of conditions it was
    # trained on. Also crossed with ALTITUDE_SWEEP_M for the same z-observability reason as the
    # low-speed sweep above; the single no-cruise value below (cruise_s breadth was trimmed from
    # 2 to 1) is what pays for the altitude multiplier here.
    for axis, accels, cap_mps in (("x", (1.0, 1.5), 1.0), ("y", (0.9, 1.2), 1.0)):
        for accel_mps2 in accels:
            cruise_s = 0.0
            peak_speed_mps = min(cap_mps, accel_mps2 * 0.75)
            for h in ALTITUDE_SWEEP_M:
                specs.append(TrajectorySpec(
                    id=f"accel_decel_fast_a{accel_mps2:.2f}_c{cruise_s:.2f}_{axis}_h{h:.2f}",
                    motif_type="accel_decel_pulse",
                    params={
                        "accel_mps2": accel_mps2, "peak_speed_mps": peak_speed_mps,
                        "cruise_s": cruise_s, "axis": axis,
                    },
                    height_m=h,
                    reps_required=2,
                    description=(
                        "Higher-speed accel/decel pulse, bracketing faster deployment flight, "
                        f"flown at height={h:.2f}m for altitude-sweep coverage."
                    ),
                ))

    # accel_decel_with_climb: an engineering extension beyond the paper (see module docstring) --
    # concurrent horizontal accel/decel + slow continuous altitude ramp in one rep, so both v_x
    # and z vary together within a single continuous trajectory rather than only across separate
    # reps. climb_start_m/climb_end_m span the same 0.3-1.2m range as ALTITUDE_SWEEP_M; "up" and
    # "down" variants cover both climb directions so the collected dz/dt sign isn't confounded
    # with anything else in the maneuver. TrajectorySpec.height_m is set to the *end* height
    # (see accel_decel_with_climb_profile's docstring for why).
    for axis in ("x", "y"):
        for direction, climb_start_m, climb_end_m in (("up", 0.3, 0.9), ("down", 0.9, 0.3)):
            specs.append(TrajectorySpec(
                id=f"accel_decel_climb_{direction}_{axis}",
                motif_type="accel_decel_with_climb",
                params={
                    "accel_mps2": 0.5, "peak_speed_mps": 0.375, "cruise_s": 0.5, "axis": axis,
                    "climb_start_m": climb_start_m, "climb_end_m": climb_end_m,
                },
                height_m=climb_end_m,
                reps_required=3,
                description=(
                    "Accel/decel pulse concurrent with a slow continuous altitude ramp "
                    f"({climb_start_m:.1f}m -> {climb_end_m:.1f}m) -- untested-by-the-paper "
                    "extension, see module docstring."
                ),
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

    # sum_of_sines carries horizontal acceleration too (it's a continuous random-frequency accel
    # profile, not constant-velocity), so it also gets altitude variety -- via
    # ALTITUDE_SWEEP_SECONDARY_M (2 points, not the primary sweep's 4) since it's a lower-value
    # motif than accel_decel_pulse per the module docstring. freq_range_hz breadth is trimmed
    # from 2 ranges to 1 to pay for the height multiplier within budget.
    for freq_range_hz in ((0.2, 0.5),):
        for amp_mps in (0.2, 0.4):
            for axis in ("x", "y"):
                for h in ALTITUDE_SWEEP_SECONDARY_M:
                    specs.append(TrajectorySpec(
                        id=f"sines_{axis}_f{freq_range_hz[0]:.2f}-{freq_range_hz[1]:.2f}_a{amp_mps:.2f}_h{h:.2f}",
                        motif_type="sum_of_sines",
                        params={
                            "n_components": 3, "freq_range_hz": freq_range_hz, "amp_mps": amp_mps,
                            "duration_s": 8.0, "axis": axis,
                            "seed": hash((freq_range_hz, amp_mps, axis, h)) % (2**31),
                        },
                        height_m=h,
                        reps_required=2,
                        description=(
                            "Continuous random-frequency acceleration profile for frequency "
                            f"coverage, flown at height={h:.2f}m for altitude-sweep coverage."
                        ),
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

    # Task-representative motif: a zig-zag coverage pattern approximating real deployment flight
    # (e.g. ~1 m/s cruise with cornering accel/decel), as a direct example rather than only a
    # systematic parameter sweep. leg_distance_m/n_legs/zigzag_angle_deg are tuned to keep the
    # pattern's footprint safely inside the geofence soft-clamp margin at this speed -- do not
    # widen them without rechecking the footprint (integrate vel_fn's vx/vy over time; see the
    # accel_decel_pulse fast-additions comment above for why this matters). Each corner is an
    # accel/decel event, so this also gets altitude variety via ALTITUDE_SWEEP_SECONDARY_M (same
    # reduced 2-point sweep as sum_of_sines, to fit the battery budget); reps_required dropped
    # from 3 to 2 per (variant, height) pair since height itself now supplies some of the
    # diversity the extra rep used to.
    zigzag_variants = [
        {"leg_distance_m": 0.4, "leg_speed_mps": 1.0, "leg_accel_mps2": 1.5, "n_legs": 3, "zigzag_angle_deg": 35.0},
        {"leg_distance_m": 0.35, "leg_speed_mps": 0.9, "leg_accel_mps2": 1.5, "n_legs": 3, "zigzag_angle_deg": 30.0},
    ]
    for idx, params in enumerate(zigzag_variants):
        for h in ALTITUDE_SWEEP_SECONDARY_M:
            specs.append(TrajectorySpec(
                id=f"zigzag_{idx}_h{h:.2f}",
                motif_type="zigzag",
                params=params,
                height_m=h,
                reps_required=2,
                max_duration_s=20.0,
                description=(
                    "Task-representative zig-zag coverage pattern at near-deployment speed, "
                    f"flown at height={h:.2f}m for altitude-sweep coverage."
                ),
            ))

    return specs
