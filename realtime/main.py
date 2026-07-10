"""Entry point for running the real-time AI-KF pipeline against a live Crazyflie.

Assumes a trained ANN artifact already exists on disk (see references/keras_ann_utility.py's
save_model_complete / Phase 4's "v1_real") and that references/extended_kalman_filter.py and
references/planar_drone.py are importable. See README.md for prerequisites and known gaps
before flying.
"""

import argparse
import logging
import sys
import time

from realtime.ann_estimator import AltitudeANNEstimator
from realtime.config import DEFAULT_MODEL_NAME, DEFAULT_URI
from realtime.crazyflie_link import CrazyflieLink
from realtime.filter_wrapper import AIKFFilter
from realtime.fusion_loop import FusionLoop
from realtime.safety_monitor import SafetyMonitor


def parse_args():
    parser = argparse.ArgumentParser(description="Run the real-time AI-KF altitude estimator.")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Crazyflie radio URI")
    parser.add_argument("--model-dir", required=True, help="Directory containing the trained ANN artifact")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                         help="Base filename passed to save_model_complete/load_model_complete")
    parser.add_argument("--initial-z", type=float, default=1.0,
                         help="Initial altitude guess (m) -- the AI-KF success criterion is "
                              "converging from a poor/arbitrary guess here, so this can "
                              "deliberately be wrong")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    ann = AltitudeANNEstimator(model_dir=args.model_dir, model_name=args.model_name)

    # state = [theta, theta_dot, x, x_dot, z, z_dot, k]
    x0 = [0.0, 0.0, 0.0, 0.0, args.initial_z, 0.0, 1.0]
    aikf_filter = AIKFFilter(x0=x0)

    safety = SafetyMonitor()
    link = CrazyflieLink(args.uri)
    fusion = FusionLoop(link=link, ann_estimator=ann, aikf_filter=aikf_filter, safety_monitor=safety)

    link.connect()
    link.add_imu_flow_callback(fusion.on_imu_flow)
    link.add_state_att_callback(fusion.on_state_att)
    link.add_state_pv_callback(lambda ts, data: None)
    link.add_diagnostic_callbacks(
        kalman_z_cb=safety.on_kalman_z, power_cb=safety.on_power,
        ranger_cb=safety.on_ranger, baro_cb=safety.on_baro,
    )

    fusion.start()
    logging.info("Fusion loop running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
            safe, reason = safety.is_safe()
            if not safe:
                logging.warning("SafetyMonitor: unsafe (%s)", reason)
    except KeyboardInterrupt:
        pass
    finally:
        fusion.stop()
        link.disconnect()


if __name__ == "__main__":
    sys.exit(main())
