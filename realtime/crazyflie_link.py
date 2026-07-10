"""cflib connection: log-block setup (per references/rates.tsv) and the external-measurement
push-back path.

KNOWN GAP -- verify before flight (see elaborate_plan.md Phase 0/5): the external-measurement
push in push_measurement() uses cflib's extpos.send_extpos(x, y, z), the standard mechanism used
for mocap/Lighthouse position sources. Two things about it are NOT yet confirmed against
crazyflie-firmware's kalman_core.c:
  1. Whether its measurement variance is configurable per-call, or only via a fixed parameter
     (e.g. locSrv.extPosStdDev) applied uniformly to all three axes -- if the latter, our
     per-tick time-varying R_aug_z cannot be carried through this exact path as-is.
  2. Whether passing through the last known onboard x/y (rather than 0, 0) avoids perturbing the
     horizontal state estimate the way we intend.
This needs a firmware-source check before being trusted in flight; it is not blocking for
building/bench-testing the rest of the pipeline against replayed logs.
"""

import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from realtime.config import (
    ATTITUDE_RATE_HZ,
    BARO_RATE_HZ,
    FUSION_RATE_HZ,
    KALMAN_DIAG_RATE_HZ,
    POWER_RATE_HZ,
    RANGER_RATE_HZ,
)


class CrazyflieLink:
    def __init__(self, uri):
        cflib.crtp.init_drivers()
        self.uri = uri
        self.scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache"))
        self._lock = threading.Lock()
        self._last_xy = (0.0, 0.0)

    def connect(self):
        self.scf.open_link()
        return self

    def disconnect(self):
        self.scf.close_link()

    @property
    def cf(self):
        return self.scf.cf

    # --- Log block setup -------------------------------------------------------------------

    def add_imu_flow_callback(self, callback):
        """Merged imu+flow block at 100Hz (both natively 100Hz per rates.tsv) so optic_flow and
        accel arrive sample-synchronized in one packet -- see plan §1."""
        lg = LogConfig(name="imu_flow", period_in_ms=int(1000 / FUSION_RATE_HZ))
        for var, fmt in [
            ("acc.x", "float"), ("acc.y", "float"), ("acc.z", "float"),
            ("gyro.x", "float"), ("gyro.y", "float"), ("gyro.z", "float"),
            ("range.zrange", "uint16_t"),
            ("motion.deltaX", "int16_t"), ("motion.deltaY", "int16_t"),
        ]:
            lg.add_variable(var, fmt)
        # NOTE: motion.squal is intentionally omitted -- if this combined block exceeds the CRTP
        # log payload limit, drop it first (flow-quality diagnostic, not an ANN input or label).
        self.cf.log.add_config(lg)
        lg.data_received_cb.add_callback(lambda ts, data, logconf: callback(ts, data))
        lg.start()
        return lg

    def add_state_att_callback(self, callback):
        """50Hz onboard attitude estimate -- used to reset the gyro-integrated theta tracker."""
        lg = LogConfig(name="state_att", period_in_ms=int(1000 / ATTITUDE_RATE_HZ))
        for var in ("stateEstimate.roll", "stateEstimate.pitch", "stateEstimate.yaw"):
            lg.add_variable(var, "float")
        self.cf.log.add_config(lg)
        lg.data_received_cb.add_callback(lambda ts, data, logconf: callback(ts, data))
        lg.start()
        return lg

    def add_state_pv_callback(self, callback):
        """50Hz onboard position/velocity estimate -- only used to pass through x/y so our z-only
        external measurement doesn't corrupt the horizontal state (see module docstring)."""
        lg = LogConfig(name="state_pv", period_in_ms=int(1000 / ATTITUDE_RATE_HZ))
        for var in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
                    "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"):
            lg.add_variable(var, "float")
        self.cf.log.add_config(lg)

        def _cb(ts, data, logconf):
            with self._lock:
                self._last_xy = (data["stateEstimate.x"], data["stateEstimate.y"])
            callback(ts, data)

        lg.data_received_cb.add_callback(_cb)
        lg.start()
        return lg

    def add_diagnostic_callbacks(self, kalman_z_cb, power_cb, ranger_cb, baro_cb):
        """20Hz/20Hz/20Hz/10Hz side-channel blocks -- not fusion inputs, consumed only by
        safety_monitor.py. Independent of the 100Hz fusion tick, per plan §1."""
        specs = [
            ("kalman_z", KALMAN_DIAG_RATE_HZ,
             [("kalman.stateZ", "float"), ("kalman.statePZ", "float"),
              ("kalman.varZ", "float"), ("kalman.varPZ", "float")], kalman_z_cb),
            ("power", POWER_RATE_HZ,
             [("pm.vbat", "float"), ("pm.batteryLevel", "uint8_t"), ("pm.state", "int8_t"),
              ("motor.m1", "uint16_t"), ("motor.m2", "uint16_t"),
              ("motor.m3", "uint16_t"), ("motor.m4", "uint16_t")], power_cb),
            ("ranger", RANGER_RATE_HZ,
             [("range.front", "uint16_t"), ("range.back", "uint16_t"),
              ("range.left", "uint16_t"), ("range.right", "uint16_t"),
              ("range.up", "uint16_t")], ranger_cb),
            ("baro", BARO_RATE_HZ,
             [("baro.asl", "float"), ("baro.pressure", "float"), ("baro.temp", "float")], baro_cb),
        ]
        configs = []
        for name, rate_hz, variables, cb in specs:
            lg = LogConfig(name=name, period_in_ms=int(1000 / rate_hz))
            for var, fmt in variables:
                lg.add_variable(var, fmt)
            self.cf.log.add_config(lg)
            lg.data_received_cb.add_callback(lambda ts, data, logconf, cb=cb: cb(ts, data))
            lg.start()
            configs.append(lg)
        return configs

    # --- External measurement push-back ------------------------------------------------------

    def push_measurement(self, z_est, variance):
        """Push the fused altitude estimate back to the Crazyflie as an external position
        measurement, holding the last known onboard x/y so only z is affected.

        See the KNOWN GAP note in this module's docstring -- confirm against kalman_core.c
        whether/how per-call variance is actually honored before relying on this in flight.
        """
        with self._lock:
            x, y = self._last_xy
        self.cf.extpos.send_extpos(x, y, z_est)
