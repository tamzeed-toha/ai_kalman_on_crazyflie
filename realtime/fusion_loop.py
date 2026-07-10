"""Ties the log callbacks, ANN, and filter together.

Threading model (see plan §2): the imu_flow log callback (running on cflib's own receiver
thread) does only cheap work -- buffer append + signal a worker -- and returns immediately. A
separate fusion-worker thread does the ANN inference, filter tick, and CRTP push-back, so slow
compute never blocks/delays radio I/O. The queue has room for exactly one pending tick: if the
worker falls behind, it always processes the newest window, never a stale backlog.
"""

import queue
import threading
import warnings

import numpy as np

from realtime.ann_estimator import compute_r_aug_z
from realtime.buffers import RollingWindow, ThetaTracker
from realtime.config import DT, FLOW_GAIN_PLACEHOLDER, G_MPS2, R_AUG_Z_COLD_START
from realtime.sensor_conversion import body_accel_to_planar_frame, raw_flow_to_optic_flow


class FusionLoop:
    def __init__(self, link, ann_estimator, aikf_filter, safety_monitor,
                 control_input_provider=None):
        self.link = link
        self.ann = ann_estimator
        self.filter = aikf_filter
        self.safety = safety_monitor
        self.control_input_provider = control_input_provider
        self._warned_no_control_provider = False

        self.window = RollingWindow()
        self.theta_tracker = ThetaTracker()

        self._queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = None

    # --- log callbacks (run on cflib's thread; keep these cheap) ------------------------------

    def on_imu_flow(self, timestamp, data):
        theta = self.theta_tracker.get_theta()
        theta_dot = np.deg2rad(data["gyro.y"])
        self.theta_tracker.integrate(theta_dot, DT)

        acc_x_body = data["acc.x"] * G_MPS2
        acc_z_body = data["acc.z"] * G_MPS2
        accel_x, accel_z = body_accel_to_planar_frame(acc_x_body, acc_z_body, theta)

        optic_flow = raw_flow_to_optic_flow(data["motion.deltaX"], DT, FLOW_GAIN_PLACEHOLDER)

        self.window.push({"optic_flow": optic_flow, "accel_x": accel_x, "accel_z": accel_z})
        self._latest_theta = theta
        self._latest_theta_dot = theta_dot
        self._latest_optic_flow = optic_flow
        self._latest_accel = (accel_x, accel_z)

        try:
            self._queue.put_nowait(True)
        except queue.Full:
            pass  # a tick is already pending; the worker will use the newest window anyway

    def on_state_att(self, timestamp, data):
        self.theta_tracker.reset_from_attitude(np.deg2rad(data["stateEstimate.pitch"]))

    # --- worker thread ------------------------------------------------------------------------

    def start(self):
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)

    def _get_control_input(self):
        if self.control_input_provider is None:
            if not self._warned_no_control_provider:
                warnings.warn(
                    "No control_input_provider supplied -- using u=[0, 0]. The filter's "
                    "process model will not feel commanded thrust/pitch torque. Fine for "
                    "wiring/replay testing; do not fly with this.", stacklevel=2,
                )
                self._warned_no_control_provider = True
            return np.zeros(2)
        return self.control_input_provider()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            safe, reason = self.safety.is_safe()
            if not safe:
                warnings.warn(f"SafetyMonitor reports unsafe ({reason}); holding last estimate.")
                continue

            if self.window.is_full:
                z_pred = self.ann.predict(self.window.get_vector())
                r_aug_z = compute_r_aug_z(self.window.accel_x_window())
            else:
                # Window not warmed up yet (first WINDOW_SIZE ticks after startup): still run
                # the base 5-measurement update, but with the ANN slot effectively disabled
                # (huge R, and z_pred pinned to the current estimate so its residual is ~0),
                # matching the reference notebook's zero-padded/huge-R warmup convention.
                z_pred = self.filter.z_estimate
                r_aug_z = R_AUG_Z_COLD_START

            base_measurements = [
                self._latest_optic_flow, self._latest_theta, self._latest_theta_dot,
                self._latest_accel[0], self._latest_accel[1],
            ]
            u = self._get_control_input()

            self.filter.tick(base_measurements, z_pred, r_aug_z, u)
            self.link.push_measurement(self.filter.z_estimate, self.filter.z_variance)
