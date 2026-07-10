"""
Minimal smoke-test flight: connect, log, arm, take off, hover, a short forward/back nudge,
land, disarm.

This is deliberately simpler than collect_data.py -- it does NOT use trajectories.py,
state_store.py, or safety.py's geofence/obstacle logic. Its only job is to verify the basic
connect/arm/takeoff/log/land/disarm pipeline works on real hardware, using the same logging
setup (config.LOG_BLOCKS, FlightSessionLogger) the full data-collection script depends on --
so a clean run here means collect_data.py's logging is trustworthy, before adding the
trajectory-queue/safety-check complexity on top.

Run bench_test_logging.py FIRST (no arming) to check log throughput; only fly this once that
looks clean. See README.md for the full recommended order of operations.

Usage:
    source ~/.venv/bin/activate
    cd data_collection
    python simple_flight_test.py
"""

from __future__ import annotations

import signal
import time

from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

import cf_link
import config
import logging_io

FLIGHT_HEIGHT_M = config.DEFAULT_HEIGHT_M
HOVER_S = 5.0
NUDGE_DISTANCE_M = 0.3
NUDGE_SPEED_M_S = 0.15


def main() -> None:
    cf_link.initialize_cflib_drivers()

    session_id = config.new_session_id()
    csv_path = config.session_csv_path(session_id)
    print(f"Connecting to {config.URI}")
    print(f"Logging to {csv_path}")

    logger = logging_io.FlightSessionLogger(csv_path, session_id, config.LOG_BLOCKS)

    land_requested = {"flag": False}

    def handle_sigint(signum, frame) -> None:
        print("\nSIGINT received: will land as soon as possible.")
        land_requested["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        with SyncCrazyflie(config.URI) as scf:
            cf = scf.cf
            logger.start_log_configs(cf)
            logger.start_continuous_logging(rate_hz=config.CSV_WRITE_RATE_HZ)
            time.sleep(1.0)  # let the first log packets arrive
            print("Log established.")

            battery_v = logger.get_value("battery_v")
            print(f"Battery: {battery_v} V")

            logger.set_flight_phase("arming")
            cf_link.arm_cf(cf)

            with MotionCommander(scf, default_height=FLIGHT_HEIGHT_M) as mc:
                logger.set_flight_phase("takeoff")
                logger.mark_event("takeoff")
                print(f"Taking off to {FLIGHT_HEIGHT_M} m")
                try:
                    mc.take_off(height=FLIGHT_HEIGHT_M)
                except Exception:
                    pass

                logger.set_flight_phase("hover")
                logger.mark_event("hover_start")
                print(f"Hovering for {HOVER_S:.0f}s")
                for _ in range(int(HOVER_S * 10)):
                    if land_requested["flag"]:
                        break
                    time.sleep(0.1)

                if not land_requested["flag"]:
                    logger.set_flight_phase("nudge_forward")
                    logger.mark_event("nudge_forward")
                    print(f"Nudging forward {NUDGE_DISTANCE_M} m")
                    mc.forward(NUDGE_DISTANCE_M, velocity=NUDGE_SPEED_M_S)

                if not land_requested["flag"]:
                    logger.set_flight_phase("nudge_back")
                    logger.mark_event("nudge_back")
                    print(f"Nudging back {NUDGE_DISTANCE_M} m")
                    mc.back(NUDGE_DISTANCE_M, velocity=NUDGE_SPEED_M_S)

                logger.set_flight_phase("landing")
                logger.mark_event("land")
                print("Landing")
                mc.land()

            cf_link.send_stop_setpoint(cf)
            cf_link.disarm_cf(cf)
            logger.set_flight_phase("disarmed")
            print("Disarmed.")

    except KeyboardInterrupt:
        print("\nInterrupted during setup/connection.")

    finally:
        logger.stop_log_configs()
        logger.close()
        print(f"Saved log: {csv_path}")
        print("Open the CSV and check: did x/y/z_m, acc_*_g, zrange_m, battery_v all update "
              "over the flight, and do the 'event'/'flight_phase' columns show the expected "
              "takeoff/hover/nudge/land sequence?")


if __name__ == "__main__":
    main()
