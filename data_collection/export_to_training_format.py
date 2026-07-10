"""
Convert data_collection/'s real flight CSVs into train.py's expected input format: one CSV per
completed trajectory rep (time, meas_r_x, meas_r_y, meas_v_x_dot, meas_v_y_dot, state_z) plus a
manifest.csv -- so train.py can point at real data via its own existing
`--simulated-trajectories-dir` flag, with zero changes needed to train.py itself.

Targets the model train.py ACTUALLY consumes today: model/drone_simulator.py's body_level-frame
measurement equations (r_x = v_x/z, r_y = v_y/z; see its h()). NOT references/planar_drone.py --
realtime/sensor_conversion.py's conversions target THAT different (2D, pitch-only) model, which
is a separate, real inconsistency in this repo worth reconciling before the realtime pipeline and
this training pipeline are consistent. See crazyflie_data_adaptation_brief.md Sec. 1-3.

PROVISIONAL -- read before trusting output for anything beyond a real-data pipeline smoke test:

1. FLOW_GAIN_PLACEHOLDER (pixel -> rad conversion for the Flow deck v2 PMW3901) is the same
   unresolved placeholder flagged in realtime/sensor_conversion.py's raw_flow_to_optic_flow --
   not a calibrated value. Recalibrate once Phase 1 produces a real constant, then re-run this
   script (cheap) to regenerate real_trajectories/.
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

import pandas as pd

import config

REAL_TRAJECTORIES_DIR = config.BASE_DIR.parent / "real_trajectories"

# Placeholder -- see module docstring point 1. Units: radians of optical angle per pixel.
FLOW_GAIN_PLACEHOLDER = 0.0075

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
    acc_x = seg["acc_x_g"].to_numpy() * 9.81
    acc_y = seg["acc_y_g"].to_numpy() * 9.81
    acc_z = seg["acc_z_g"].to_numpy() * 9.81

    meas_v_x_dot = []
    meas_v_y_dot = []
    for ax, ay, az, r, p in zip(acc_x, acc_y, acc_z, roll_rad, pitch_rad):
        vx_dot, vy_dot = _tilt_compensate(ax, ay, az, r, p)
        meas_v_x_dot.append(vx_dot)
        meas_v_y_dot.append(vy_dot)

    meas_r_x = seg["flow_delta_x_px"].to_numpy() * FLOW_GAIN_PLACEHOLDER / dt
    meas_r_y = seg["flow_delta_y_px"].to_numpy() * FLOW_GAIN_PLACEHOLDER / dt

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
