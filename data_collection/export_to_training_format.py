"""
Convert data_collection/'s real flight CSVs into train.py's expected input format: one CSV per
completed trajectory rep (time, meas_r_x, meas_r_y, meas_v_x_dot, meas_v_y_dot, state_z) plus a
manifest.csv -- so train.py can point at real data via its own existing
`--simulated-trajectories-dir` flag, with zero changes needed to train.py itself.

Targets the model train.py ACTUALLY consumes today: model/drone_simulator.py's body_level-frame
measurement equations (r_x = v_x/z, r_y = v_y/z; see its h()). realtime/sensor_conversion.py
targets the same model now too (previously targeted a different, 2D model -- since reconciled;
see realtime/README.md). See crazyflie_data_adaptation_brief.md Sec. 1-3 for the still-open
accel-semantics question (point 2 below).

1. FLOW CONVERSION -- fixed, empirically validated against real data, not a placeholder anymore.
   Two things had to be right, not just gain:
     a. AXIS SWAP + SIGN: crazyflie-firmware's flowdeck_v1v2.c computes
        `accpx = -currentMotion.deltaY` (forward), `accpy = -currentMotion.deltaX` (lateral)
        before using flow at all -- "flip motion information to comply with sensor mounting"
        per its own comment. The raw LOGGED flow_delta_x_px/flow_delta_y_px are PRE-swap
        sensor-native axes, not aircraft forward/lateral. Using them directly (as this script
        used to) gives R^2~0.002 (pure noise) when regressed against stateEstimate.vx/zrange_m;
        applying the swap gives R^2~0.63.
     b. GAIN: FLOW_GAIN = FLOW_RESOLUTION * FLOW_THETAPIX / FLOW_NPIX, derived from
        crazyflie-firmware's src/modules/src/kalman_core/mm_flow.c constants, not guessed.
   Rotation-rate compensation (gyro_y for forward flow, gyro_x for lateral, matching
   mm_flow.c's omegay_b/omegax_b) and the R[2][2]=cos(roll)*cos(pitch) tilt projection are
   included for fidelity to the firmware model, though empirically they didn't clearly improve
   R^2 over the translation-only version on this dataset (plausibly attitude/gyro noise, not
   evidence the compensation is wrong -- worth re-checking with more/cleaner data). If
   FLOW_NPIX/FLOW_THETAPIX/FLOW_RESOLUTION/FLOW_GAIN change in realtime/config.py, mirror the
   change here -- these two files must stay in sync (see this module's earlier note).
2. meas_v_x_dot/meas_v_y_dot are computed by tilt-compensating raw body-frame acc.x/acc.y using
   logged roll/pitch (removing gravity's projection onto the tilted body axes) to estimate
   level-frame horizontal acceleration -- this correctly removes tilt, but whether the model's
   v_x_dot/v_y_dot is meant to be this exact quantity (vs. some other kinematic/specific-force
   convention) is NOT independently verified. Flag for Phase 4 review alongside the brief's
   Sec. 1-2 accel-semantics question.
3. state_z uses zrange_m directly: the z-ranger IS the sensor this whole project aims to
   replace, not independent ground truth (no Lighthouse/mocap on this setup). This is a known,
   accepted Phase 3/4 limitation (elaborate_plan.md), not a data-collection gap -- real data can
   fine-tune/validate a sim-pretrained model, not train one from scratch, by design.
4. Only rows from reps recorded as "completed" in progress_state.json are exported (cross-
   referenced by session_id) -- aborted/interrupted reps are excluded so partial maneuvers don't
   get treated as valid full trajectories.

Usage:
    python export_to_training_format.py
        Reads data/session_*.csv + data/progress_state.json, writes
        real_trajectories/*.csv + real_trajectories/manifest.csv.
    python ../train.py --simulated-trajectories-dir real_trajectories --models-dir ../models_real
        (train.py's own existing flags -- no code changes to train.py needed)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config

REAL_TRAJECTORIES_DIR = config.BASE_DIR.parent / "real_trajectories"

# Derived from crazyflie-firmware's src/modules/src/kalman_core/mm_flow.c -- see module
# docstring point 1. MUST match realtime/config.py's FLOW_NPIX/FLOW_THETAPIX/FLOW_RESOLUTION/
# FLOW_GAIN exactly (training-time and inference-time preprocessing have to agree).
FLOW_NPIX = 35.0
FLOW_THETAPIX = 0.71674
FLOW_RESOLUTION = 0.1
FLOW_GAIN = FLOW_RESOLUTION * FLOW_THETAPIX / FLOW_NPIX  # ~0.002048

# Reject physically-impossible flow-derived values rather than feeding them into training data --
# see realtime/config.py's FLOW_SANITY_CLAMP for the matching real-time-side safety net.
FLOW_SANITY_CLAMP = 5.0

MIN_FLOW_SQUAL = 10  # skip rows with very low flow-tracking confidence


def _tilt_compensate(ax, ay, az, roll_rad, pitch_rad):
    """Rotate body-frame accel (m/s^2) into a yaw-following 'level' frame: undoes roll (about
    body x) then pitch (about body y), matching drone_simulator.py's body_level convention
    (yaw-relative axes, no yaw rotation applied). See module docstring point 2."""
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    ay1 = ay * cr - az * sr
    az1 = ay * sr + az * cr
    ax_level = ax * cp + az1 * sp
    ay_level = ay1
    return ax_level, ay_level


def _flow_to_body_frame(delta_x_raw, delta_y_raw, dt, gyro_x_rad_s, gyro_y_rad_s, roll_rad, pitch_rad):
    """Raw motion.deltaX/deltaY (sensor-native axes) -> [meas_r_x, meas_r_y] = [v_x/z, v_y/z]
    (aircraft forward/lateral), matching crazyflie-firmware's mm_flow.c. See module docstring
    point 1 -- MUST match realtime/sensor_conversion.py's raw_flow_to_body_frame_optic_flow."""
    dpixel_forward = -delta_y_raw
    dpixel_lateral = -delta_x_raw
    r22 = np.cos(roll_rad) * np.cos(pitch_rad)

    r_x = (dpixel_forward * FLOW_GAIN / dt + gyro_y_rad_s) / r22
    r_y = (dpixel_lateral * FLOW_GAIN / dt + gyro_x_rad_s) / r22

    r_x = np.clip(r_x, -FLOW_SANITY_CLAMP, FLOW_SANITY_CLAMP)
    r_y = np.clip(r_y, -FLOW_SANITY_CLAMP, FLOW_SANITY_CLAMP)
    return r_x, r_y


def _completed_rep_sessions(state: dict) -> dict:
    """(traj_id, rep_index) -> session_id, for every rep_history entry with status=='completed'."""
    lookup = {}
    for traj_id, entry in state["completed"].items():
        for h in entry["rep_history"]:
            if h["status"] == "completed":
                lookup[(traj_id, h["rep_index"])] = h["session_id"]
    return lookup


def _trajectory_segments(df: pd.DataFrame):
    """Contiguous (motif_id, rep_index) runs within flight_phase=='trajectory' rows, in order.
    Yields (motif_id, rep_index, sub_dataframe). A retried-and-then-completed rep within the same
    session produces two separate runs with identical labels; callers wanting "the last attempt"
    should take the last yielded match."""
    traj_df = df[df["flight_phase"] == "trajectory"]
    if traj_df.empty:
        return
    keys = list(zip(traj_df["motif_id"], traj_df["rep_index"]))
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            motif_id, rep_index = keys[start]
            yield motif_id, rep_index, traj_df.iloc[start:i]
            start = i


def _build_rep_dataframe(seg: pd.DataFrame) -> pd.DataFrame:
    seg = seg.reset_index(drop=True)
    pc_time = seg["pc_time_s"].to_numpy()
    dt = pd.Series(pc_time).diff().fillna(1.0 / config.CSV_WRITE_RATE_HZ).to_numpy().copy()
    dt[dt <= 0] = 1.0 / config.CSV_WRITE_RATE_HZ  # guard against clock jitter/duplicate timestamps

    roll_rad = seg["roll_deg"].to_numpy() * (math.pi / 180.0)
    pitch_rad = seg["pitch_deg"].to_numpy() * (math.pi / 180.0)
    gyro_x_rad_s = seg["gyro_x_dps"].to_numpy() * (math.pi / 180.0)
    gyro_y_rad_s = seg["gyro_y_dps"].to_numpy() * (math.pi / 180.0)
    acc_x = seg["acc_x_g"].to_numpy() * 9.81
    acc_y = seg["acc_y_g"].to_numpy() * 9.81
    acc_z = seg["acc_z_g"].to_numpy() * 9.81

    meas_v_x_dot = []
    meas_v_y_dot = []
    for ax, ay, az, r, p in zip(acc_x, acc_y, acc_z, roll_rad, pitch_rad):
        vx_dot, vy_dot = _tilt_compensate(ax, ay, az, r, p)
        meas_v_x_dot.append(vx_dot)
        meas_v_y_dot.append(vy_dot)

    meas_r_x, meas_r_y = _flow_to_body_frame(
        seg["flow_delta_x_px"].to_numpy(), seg["flow_delta_y_px"].to_numpy(), dt,
        gyro_x_rad_s, gyro_y_rad_s, roll_rad, pitch_rad,
    )

    out = pd.DataFrame({
        "time": pc_time - pc_time[0],
        "meas_r_x": meas_r_x,
        "meas_r_y": meas_r_y,
        "meas_v_x_dot": meas_v_x_dot,
        "meas_v_y_dot": meas_v_y_dot,
        "state_z": seg["zrange_m"].to_numpy(),
        "flow_squal": seg["flow_squal"].to_numpy(),
    })
    out = out[out["flow_squal"] >= MIN_FLOW_SQUAL].drop(columns=["flow_squal"]).reset_index(drop=True)
    return out


def main() -> None:
    state_path = config.STATE_FILE
    if not state_path.exists():
        raise FileNotFoundError(f"{state_path} not found -- run collect_data.py first.")
    state = json.loads(state_path.read_text())
    completed = _completed_rep_sessions(state)
    print(f"{len(completed)} completed reps recorded in {state_path}")

    sessions_by_id = {}
    for csv_path in sorted(config.DATA_DIR.glob("session_*.csv")):
        session_id = csv_path.stem.replace("session_", "")
        sessions_by_id[session_id] = csv_path

    REAL_TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    skipped = 0

    # Group target (traj_id, rep_index) pairs by which session they came from, so each session
    # CSV is only parsed once regardless of how many completed reps it contains.
    by_session = {}
    for (traj_id, rep_index), session_id in completed.items():
        by_session.setdefault(session_id, []).append((traj_id, rep_index))

    for session_id, targets in by_session.items():
        csv_path = sessions_by_id.get(session_id)
        if csv_path is None:
            print(f"WARNING: session {session_id} referenced in progress_state.json but "
                  f"{config.DATA_DIR}/session_{session_id}.csv is missing -- skipping "
                  f"{len(targets)} rep(s).")
            skipped += len(targets)
            continue

        df = pd.read_csv(csv_path, low_memory=False)
        segments_by_key = {}
        for motif_id, rep_index, seg in _trajectory_segments(df):
            segments_by_key.setdefault((motif_id, rep_index), []).append(seg)

        for traj_id, rep_index in targets:
            matches = segments_by_key.get((traj_id, rep_index))
            if not matches:
                print(f"WARNING: no matching trajectory segment for {traj_id} rep {rep_index} "
                      f"in session {session_id} -- skipping.")
                skipped += 1
                continue
            seg = matches[-1]  # last attempt in this session is the one that completed

            rep_df = _build_rep_dataframe(seg)
            if len(rep_df) < 2:
                print(f"WARNING: {traj_id} rep {rep_index} (session {session_id}) has "
                      f"<2 usable rows after flow-quality filtering -- skipping.")
                skipped += 1
                continue

            motif_type = seg["motif_type"].iloc[0]
            filename = f"real_{traj_id}_rep{rep_index}_{session_id}.csv"
            rep_df.to_csv(REAL_TRAJECTORIES_DIR / filename, index=False)
            manifest_rows.append({
                "filename": filename,
                "params_json": json.dumps({
                    "motif": motif_type,
                    "motif_id": traj_id,
                    "rep_index": rep_index,
                    "session_id": session_id,
                }),
            })

    manifest_path = REAL_TRAJECTORIES_DIR / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"wrote {len(manifest_rows)} trajectory CSVs + manifest to {REAL_TRAJECTORIES_DIR}")
    if skipped:
        print(f"skipped {skipped} rep(s) -- see warnings above.")


if __name__ == "__main__":
    main()
