"""
Cross-run progress state: tracks which trajectory reps have been flown so a script re-run
after a battery swap resumes with the next incomplete rep instead of starting over.

Written atomically (temp file + os.replace) after every single rep, so at most one rep is
ever left in an ambiguous state if power is lost mid-rep -- that rep is simply re-flown on
the next run, rather than the whole trajectory or whole session being redone.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trajectories import TrajectorySpec

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _library_version(specs: List[TrajectorySpec]) -> str:
    ids = sorted(spec.id for spec in specs)
    return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()[:12]


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def new_state(specs: List[TrajectorySpec]) -> dict:
    now = _now_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_library_version": _library_version(specs),
        "created_at": now,
        "updated_at": now,
        "queue_order": [spec.id for spec in specs],
        "completed": {
            spec.id: {"reps_required": spec.reps_required, "reps_completed": 0, "rep_history": []}
            for spec in specs
        },
        "sessions": [],
    }


def load_or_init_state(state_path: Path, specs: List[TrajectorySpec]) -> dict:
    if not state_path.exists():
        state = new_state(specs)
        _atomic_write_json(state_path, state)
        return state

    with state_path.open() as f:
        state = json.load(f)

    current_version = _library_version(specs)
    if state.get("trajectory_library_version") != current_version:
        print(
            "Warning: trajectory library has changed since progress_state.json was last "
            "written. Appending new trajectory IDs; existing history/order is preserved."
        )
        specs_by_id = {spec.id: spec for spec in specs}
        existing_ids = set(state["queue_order"])
        for spec in specs:
            if spec.id not in existing_ids:
                state["queue_order"].append(spec.id)
                state["completed"][spec.id] = {
                    "reps_required": spec.reps_required, "reps_completed": 0, "rep_history": [],
                }
        # Keep reps_required in sync in case a spec's own rep count changed.
        for traj_id, entry in state["completed"].items():
            if traj_id in specs_by_id:
                entry["reps_required"] = specs_by_id[traj_id].reps_required
        state["trajectory_library_version"] = current_version
        _atomic_write_json(state_path, state)

    return state


def get_next_incomplete(state: dict) -> Tuple[Optional[str], Optional[int]]:
    for traj_id in state["queue_order"]:
        entry = state["completed"].get(traj_id)
        if entry is None:
            continue
        if entry["reps_completed"] < entry["reps_required"]:
            return traj_id, entry["reps_completed"]
    return None, None


def mark_rep_result(
    state_path: Path,
    state: dict,
    traj_id: str,
    rep_index: int,
    status: str,
    session_id: str,
    reason: Optional[str] = None,
) -> None:
    entry = state["completed"][traj_id]
    entry["rep_history"].append({
        "rep_index": rep_index,
        "status": status,
        "session_id": session_id,
        "reason": reason,
        "completed_at": _now_iso(),
    })
    if status == "completed":
        entry["reps_completed"] += 1
    state["updated_at"] = _now_iso()
    _atomic_write_json(state_path, state)


def start_session(state_path: Path, state: dict, session_id: str, csv_file: str) -> None:
    state["sessions"].append({
        "session_id": session_id,
        "started_at": _now_iso(),
        "ended_at": None,
        "csv_file": csv_file,
        "end_reason": None,
    })
    state["updated_at"] = _now_iso()
    _atomic_write_json(state_path, state)


def end_session(state_path: Path, state: dict, session_id: str, end_reason: str) -> None:
    for session in reversed(state["sessions"]):
        if session["session_id"] == session_id:
            session["ended_at"] = _now_iso()
            session["end_reason"] = end_reason
            break
    state["updated_at"] = _now_iso()
    _atomic_write_json(state_path, state)


def progress_summary(state: dict) -> str:
    total_required = sum(e["reps_required"] for e in state["completed"].values())
    total_completed = sum(e["reps_completed"] for e in state["completed"].values())
    return f"{total_completed}/{total_required} reps completed across {len(state['queue_order'])} trajectories"
