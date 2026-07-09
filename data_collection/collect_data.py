"""
Phase 3 data-collection flight script.

Connects to a Crazyflie 2.1 Brushless, arms it, takes off, then flies through the queued
trajectory library (trajectories.py) one rep at a time, logging IMU/optical-flow/z-range/EKF
state/battery/multiranger/baro data throughout (logging_io.py). Progress is persisted to
data/progress_state.json (state_store.py) after every rep, so re-running this script after a
battery swap resumes with the next incomplete rep rather than starting over.

Nothing is withheld from the onboard EKF here -- the z-ranger stays on and logged as the
reference signal throughout. Disabling its contribution to state estimation is future work
(elaborate_plan.md Phase 8), not part of this script.

Run from inside this directory (flat imports, matching examples/bisccits' convention):
    source ~/.venv/bin/activate
    cd data_collection
    python bench_test_logging.py   # run this FIRST, before ever arming (see README.md)
    python collect_data.py
"""

from __future__ import annotations

import json
import signal
import time

from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

import cf_link
import config
import logging_io
import safety
import state_store
import trajectories


def main() -> None:
    cf_link.initialize_cflib_drivers()

    library = trajectories.build_trajectory_library(height_m=config.DEFAULT_HEIGHT_M)
    specs_by_id = {spec.id: spec for spec in library}
    state = state_store.load_or_init_state(config.STATE_FILE, library)
    print(f"Resuming: {state_store.progress_summary(state)}")

    session_id = config.new_session_id()
    csv_path = config.session_csv_path(session_id)
    print(f"Connecting to {config.URI}")
    print(f"Session {session_id}: logging to {csv_path}")

    logger = logging_io.FlightSessionLogger(csv_path, session_id, config.LOG_BLOCKS)
    state_store.start_session(config.STATE_FILE, state, session_id, str(csv_path))

    land_requested = {"flag": False}

    def handle_sigint(signum, frame) -> None:
        print("\nSIGINT received: will land at the next safe point.")
        land_requested["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    current_rep = {"traj_id": None, "rep_index": None}
    end_reason = "unknown"

    try:
        with SyncCrazyflie(config.URI) as scf:
            cf = scf.cf
            logger.start_log_configs(cf)
            logger.start_continuous_logging(rate_hz=config.CSV_WRITE_RATE_HZ)
            time.sleep(1.0)  # let the first log packets arrive before arming
            print("Log established.")

            logger.set_flight_phase("arming")
            cf_link.arm_cf(cf)

            session_start = time.time()

            with MotionCommander(scf, default_height=config.DEFAULT_HEIGHT_M) as mc:
                logger.set_flight_phase("takeoff")
                logger.mark_event("takeoff")
                try:
                    mc.take_off(height=config.DEFAULT_HEIGHT_M)
                except Exception:
                    pass
                time.sleep(1.0)  # let takeoff settle before starting trajectories

                while True:
                    if land_requested["flag"]:
                        end_reason = "interrupted"
                        break
                    if time.time() - session_start > config.MAX_SESSION_TIME_S:
                        end_reason = "max_session_time"
                        break
                    if not safety.battery_ok_to_start_rep(logger, config):
                        end_reason = "battery_low"
                        break

                    traj_id, rep_index = state_store.get_next_incomplete(state)
                    if traj_id is None:
                        end_reason = "all_trajectories_complete"
                        break

                    spec = specs_by_id[traj_id]
                    current_rep["traj_id"] = traj_id
                    current_rep["rep_index"] = rep_index

                    duration_s, vel_fn = trajectories.MOTIF_GENERATORS[spec.motif_type](spec.params, spec.height_m)
                    effective_duration_s = min(duration_s, spec.max_duration_s)

                    print(
                        f"Flying {traj_id} rep {rep_index + 1}/{spec.reps_required} "
                        f"(duration {effective_duration_s:.1f}s)"
                    )
                    logger.mark_event("motif_start")

                    t0 = time.time()
                    dt = 1.0 / config.CONTROL_RATE_HZ
                    rep_status = "completed"
                    rep_reason = None

                    while True:
                        t = time.time() - t0
                        if t >= effective_duration_s:
                            break
                        if land_requested["flag"]:
                            rep_status = "aborted"
                            rep_reason = "interrupted"
                            break

                        raw_cmd = vel_fn(t)
                        cmd, safety_flag, should_land, should_abort = safety.check_safety(raw_cmd, logger, config)

                        logger.set_motif_state(
                            motif_id=spec.id, motif_type=spec.motif_type, motif_phase=cmd.phase,
                            rep_index=rep_index, motif_param_json=json.dumps(spec.params),
                            flight_phase="trajectory",
                        )
                        logger.set_command(
                            vx=cmd.vx, vy=cmd.vy, yaw_rate=cmd.yaw_rate, height=cmd.height,
                            vx_raw=raw_cmd.vx, vy_raw=raw_cmd.vy,
                            yaw_rate_raw=raw_cmd.yaw_rate, height_raw=raw_cmd.height,
                            safety_override_active=(cmd.vx != raw_cmd.vx or cmd.vy != raw_cmd.vy),
                        )
                        logger.set_safety_state(safety_flag=safety_flag, should_land=should_land)
                        if safety_flag:
                            logger.mark_event(safety_flag)

                        cf.commander.send_hover_setpoint(cmd.vx, cmd.vy, cmd.yaw_rate, cmd.height)

                        if should_land:
                            land_requested["flag"] = True
                            rep_status = "aborted"
                            rep_reason = safety_flag
                            break
                        if should_abort:
                            rep_status = "aborted"
                            rep_reason = safety_flag
                            break

                        time.sleep(dt)

                    logger.mark_event("motif_end")
                    state_store.mark_rep_result(
                        config.STATE_FILE, state, traj_id, rep_index, rep_status, session_id, reason=rep_reason,
                    )
                    current_rep["traj_id"] = None
                    current_rep["rep_index"] = None

                    if land_requested["flag"]:
                        end_reason = "interrupted" if end_reason == "unknown" else end_reason
                        break

                    logger.set_flight_phase("interstitial_hover")
                    cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, spec.height_m)
                    time.sleep(config.INTERSTITIAL_HOVER_S)

                print(f"Landing (reason: {end_reason})")
                logger.set_flight_phase("landing")
                logger.mark_event("land")
                cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, 0.0)
                time.sleep(0.2)
                mc.land()

            cf_link.send_stop_setpoint(cf)
            cf_link.disarm_cf(cf)
            logger.set_flight_phase("disarmed")

    except KeyboardInterrupt:
        end_reason = "interrupted"
        print("\nInterrupted during setup/connection.")

    finally:
        if current_rep["traj_id"] is not None:
            state_store.mark_rep_result(
                config.STATE_FILE, state, current_rep["traj_id"], current_rep["rep_index"],
                "aborted", session_id, reason="interrupted",
            )
        state_store.end_session(config.STATE_FILE, state, session_id, end_reason)
        logger.stop_log_configs()
        logger.close()
        print(f"Saved session log: {csv_path}")
        print(state_store.progress_summary(state))


if __name__ == "__main__":
    main()
