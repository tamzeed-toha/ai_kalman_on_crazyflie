"""
Shared configuration for Phase 3 data-collection scripts.

Edit this file before flying. Thresholds marked "PLACEHOLDER - CALIBRATE" have not been
verified against the actual Crazyflie 2.1 Brushless hardware and must be tuned on real
hardware before unattended/full-library runs (see elaborate_plan.md risk register and
data_collection/README.md).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Radio URI
# ---------------------------------------------------------------------------
URI = os.environ.get("CFLIB_URI", "radio://0/80/2M/E7E7E7E7E7")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "progress_state.json"


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def session_csv_path(session_id: str) -> Path:
    return DATA_DIR / f"session_{session_id}.csv"


# ---------------------------------------------------------------------------
# Flight / control parameters
# ---------------------------------------------------------------------------
DEFAULT_HEIGHT_M = 0.4

# Rate at which send_hover_setpoint is re-issued during a trajectory rep.
CONTROL_RATE_HZ = 50.0

# Rate at which the background thread writes a CSV row from the latest logged values.
# Matches the fastest LogConfig blocks (imu/flow, 100 Hz) so those samples are not
# silently dropped by a slower "latest value" write cadence.
CSV_WRITE_RATE_HZ = 100.0

# Pause between trajectory reps, hovering at height, to let transients settle and to give
# a clean boundary between reps in the CSV/labels.
INTERSTITIAL_HOVER_S = 1.5

# Hard backstop against a runaway session regardless of queue/battery logic. Matches the
# ~5 min/flight battery-life assumption this project is scoped around (15 flights total).
MAX_SESSION_TIME_S = 300.0

# Hard cap on any single trajectory rep, independent of its own computed duration.
DEFAULT_MAX_REP_DURATION_S = 20.0

# ---------------------------------------------------------------------------
# Battery safety thresholds -- PLACEHOLDER, CALIBRATE ON REAL HARDWARE (see README)
# ---------------------------------------------------------------------------
# Checked once before starting each new trajectory rep: refuse to start a new rep below
# this voltage so a multi-second maneuver is never begun without headroom to finish it.
BATTERY_SOFT_LAND_V = 3.3

# Checked every control tick during a rep: land immediately regardless of what the
# trajectory is doing.
BATTERY_HARD_ABORT_V = 3.0

# ---------------------------------------------------------------------------
# Multiranger obstacle-stop thresholds
# ---------------------------------------------------------------------------
# Sized for stopping distance, not just "mirrors bisccits' 0.35m default": trajectories.py's
# top commanded speed is now ~1.0 m/s (accel_decel_pulse "fast" entries, zigzag legs). Stopping
# distance ~= v^2/(2*decel); at 1.0 m/s with a ~1.5 m/s^2 braking deceleration that's ~0.33m, so
# 0.35m would leave almost no margin. Raised to 0.45m. Vertical speed is unchanged by that
# widening (only vertical_bob commands height directly, and it's still slow), so UP_STOP_M is
# untouched.
OBSTACLE_STOP_M = 0.45
UP_STOP_M = 0.20

# ---------------------------------------------------------------------------
# Geofence (position bounds relative to the takeoff origin), "medium room" defaults
# ---------------------------------------------------------------------------
GEOFENCE_X_M = (-1.5, 1.5)
GEOFENCE_Y_M = (-1.5, 1.5)
# trajectories.py's ALTITUDE_SWEEP_M flies accel-carrying motifs across ~0.3-1.2m; at the
# extremes that's a geofence breach severity of ~0.74/~0.83 (1.0 = exactly at a bound, see
# safety.py:_axis_breach_severity) -- comfortably inside this bound with margin, no change needed.
GEOFENCE_Z_M = (0.15, 1.3)

# Soft-clamp begins this far inside each bound; hard abort-and-return triggers this
# factor beyond the bound. Raised from 0.2 to 0.4m alongside OBSTACLE_STOP_M above, for the same
# stopping-distance reason -- momentum at ~1.0 m/s needs more room to bleed off before the hard
# bound than a 0.2m margin gives. trajectories.py's accel_decel_pulse "fast" and zigzag entries
# were deliberately sized (see comments there) to stay inside the resulting +-1.1m soft interior.
GEOFENCE_SOFT_MARGIN_M = 0.4
GEOFENCE_ABORT_FACTOR = 1.2

# ---------------------------------------------------------------------------
# LogConfig blocks
# ---------------------------------------------------------------------------
# Verified against cflib source (cflib/crazyflie/log.py): LogConfig.MAX_LEN = 26 bytes of
# variable payload per block, Log.MAX_BLOCKS = 16, Log.MAX_VARIABLES = 128. period_ms is
# stored as int(period_ms / 10), i.e. 10 ms granularity (100 Hz max per block).
#
# Byte totals below (float/uint32/int32 = 4B, uint16/int16 = 2B, uint8/int8 = 1B):
#   imu:       6 x float                          = 24 B
#   flow:      1 x uint16 + 2 x int16 + 1 x uint8  =  7 B
#   state_pv:  6 x float                           = 24 B
#   state_att: 3 x float                           = 12 B
#   kalman_z:  4 x float                           = 16 B
#   power:     1 float + 1 uint8 + 1 int8 + 4 uint16 = 14 B
#   ranger:    5 x uint16                          = 10 B
#   baro:      3 x float                           = 12 B
# All well under the 26 B/block limit; 8 blocks / 30 variables total, well under the
# 16-block / 128-variable ceiling.
LOG_BLOCKS = [
    dict(
        name="imu",
        period_ms=10,  # 100 Hz -- core IMU input to the ANN
        vars=[
            ("acc.x", "float"), ("acc.y", "float"), ("acc.z", "float"),
            ("gyro.x", "float"), ("gyro.y", "float"), ("gyro.z", "float"),
        ],
    ),
    dict(
        name="flow",
        period_ms=10,  # 100 Hz -- raw optical flow (ANN input) + z-ranger (ground-truth label)
        vars=[
            ("range.zrange", "uint16_t"),
            ("motion.deltaX", "int16_t"), ("motion.deltaY", "int16_t"),
            ("motion.squal", "uint8_t"),
        ],
    ),
    dict(
        name="state_pv",
        period_ms=20,  # 50 Hz -- EKF position/velocity baseline; also feeds the geofence
        vars=[
            ("stateEstimate.x", "float"), ("stateEstimate.y", "float"), ("stateEstimate.z", "float"),
            ("stateEstimate.vx", "float"), ("stateEstimate.vy", "float"), ("stateEstimate.vz", "float"),
        ],
    ),
    dict(
        name="state_att",
        period_ms=20,  # 50 Hz -- attitude context
        vars=[
            ("stateEstimate.roll", "float"), ("stateEstimate.pitch", "float"), ("stateEstimate.yaw", "float"),
        ],
    ),
    dict(
        name="kalman_z",
        period_ms=50,  # 20 Hz -- EKF's internal z-channel diagnostics
        vars=[
            ("kalman.stateZ", "float"), ("kalman.statePZ", "float"),
            ("kalman.varZ", "float"), ("kalman.varPZ", "float"),
        ],
    ),
    dict(
        name="power",
        period_ms=50,  # 20 Hz -- battery/motor telemetry, feeds battery abort logic
        vars=[
            ("pm.vbat", "float"), ("pm.batteryLevel", "uint8_t"), ("pm.state", "int8_t"),
            ("motor.m1", "uint16_t"), ("motor.m2", "uint16_t"),
            ("motor.m3", "uint16_t"), ("motor.m4", "uint16_t"),
        ],
    ),
    dict(
        name="ranger",
        period_ms=50,  # 20 Hz -- multiranger, feeds obstacle-stop logic
        vars=[
            ("range.front", "uint16_t"), ("range.back", "uint16_t"),
            ("range.left", "uint16_t"), ("range.right", "uint16_t"), ("range.up", "uint16_t"),
        ],
    ),
    dict(
        name="baro",
        period_ms=100,  # 10 Hz -- optional cross-check; first block to drop if bandwidth is tight
        vars=[
            ("baro.asl", "float"), ("baro.pressure", "float"), ("baro.temp", "float"),
        ],
    ),
]
