"""
Flight-session logging: multi-block LogConfig setup + background CSV-writer thread.

Adapted from examples/bisccits/common_cf.py's FlightDataLogger, but:
- Uses the explicit named/rated LogConfig blocks from config.LOG_BLOCKS instead of generic
  fixed-size chunking, since here the grouping/rate per sensor type is deliberate.
- Adds an error_cb callback per block (examples/EAG/crazyflie_pi_logger.py's pattern) so a
  block that fails to start is not silently missing from the CSV.
- Adds persistent motif/trajectory-label fields (set_motif_state) alongside bisccits' style
  one-shot `event` field (mark_event) and command fields (set_command).
- Defaults CSV_WRITE_RATE_HZ to 100 Hz (matching the fastest LogConfig blocks) rather than
  bisccits' 20 Hz, so the highest-value IMU/flow samples are not dropped by a slower
  "latest value" write cadence. See config.py / README.md for the throughput risk this
  implies and the bench-test step that should be run before relying on it.
"""

from __future__ import annotations

import csv
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cflib.crazyflie.log import LogConfig

CSV_FIELDS = [
    "pc_time_s", "pc_time_iso", "cf_time_ms", "session_id",
    "motif_id", "motif_type", "motif_phase", "rep_index", "motif_param_json", "flight_phase",
    "cmd_vx_m_s", "cmd_vy_m_s", "cmd_yaw_rate_deg_s", "cmd_height_m",
    "cmd_vx_raw_m_s", "cmd_vy_raw_m_s", "cmd_yaw_rate_raw_deg_s", "cmd_height_raw_m",
    "safety_override_active",
    "acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "flow_delta_x_px", "flow_delta_y_px", "flow_squal", "zrange_mm", "zrange_m",
    "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "roll_deg", "pitch_deg", "yaw_deg",
    "kalman_state_z_m", "kalman_state_pz", "kalman_var_z", "kalman_var_pz",
    "battery_v", "battery_level_pct", "pm_state",
    "motor_m1", "motor_m2", "motor_m3", "motor_m4",
    "range_front_m", "range_back_m", "range_left_m", "range_right_m", "range_up_m",
    "baro_asl_m", "baro_pressure_hpa", "baro_temp_c",
    "safety_flag", "should_land", "event",
]

# Maps a Crazyflie log variable name to its CSV column. range.zrange is intentionally
# stored raw (zrange_mm); zrange_m is derived from it at write time, not logged directly.
VAR_TO_COLUMN = {
    "acc.x": "acc_x_g", "acc.y": "acc_y_g", "acc.z": "acc_z_g",
    "gyro.x": "gyro_x_dps", "gyro.y": "gyro_y_dps", "gyro.z": "gyro_z_dps",
    "range.zrange": "zrange_mm",
    "motion.deltaX": "flow_delta_x_px", "motion.deltaY": "flow_delta_y_px", "motion.squal": "flow_squal",
    "stateEstimate.x": "x_m", "stateEstimate.y": "y_m", "stateEstimate.z": "z_m",
    "stateEstimate.vx": "vx_m_s", "stateEstimate.vy": "vy_m_s", "stateEstimate.vz": "vz_m_s",
    "stateEstimate.roll": "roll_deg", "stateEstimate.pitch": "pitch_deg", "stateEstimate.yaw": "yaw_deg",
    "kalman.stateZ": "kalman_state_z_m", "kalman.statePZ": "kalman_state_pz",
    "kalman.varZ": "kalman_var_z", "kalman.varPZ": "kalman_var_pz",
    "pm.vbat": "battery_v", "pm.batteryLevel": "battery_level_pct", "pm.state": "pm_state",
    "motor.m1": "motor_m1", "motor.m2": "motor_m2", "motor.m3": "motor_m3", "motor.m4": "motor_m4",
    "range.front": "range_front_m", "range.back": "range_back_m",
    "range.left": "range_left_m", "range.right": "range_right_m", "range.up": "range_up_m",
    "baro.asl": "baro_asl_m", "baro.pressure": "baro_pressure_hpa", "baro.temp": "baro_temp_c",
}

# mm -> m for the multiranger deck. range.zrange is NOT scaled here (kept raw in zrange_mm).
VAR_SCALE = {
    "range.front": 0.001, "range.back": 0.001, "range.left": 0.001, "range.right": 0.001, "range.up": 0.001,
}


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


class FlightSessionLogger:
    """
    Continuously writes one flight session's data to a CSV file.

    Log callbacks only update a lock-protected "latest values" dict; a background thread
    samples that dict at a fixed rate and writes/flushes a CSV row. This decouples disk I/O
    from the real-time control loop issuing send_hover_setpoint.
    """

    def __init__(self, filename: Path, session_id: str, log_blocks: List[dict]):
        self.filename = Path(filename)
        self.session_id = session_id
        self.log_blocks = log_blocks

        self.file = self.filename.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        self.writer.writeheader()
        self.file.flush()

        self.latest: Dict[str, Any] = {field: None for field in CSV_FIELDS}
        self.latest["session_id"] = session_id
        self.latest["event"] = ""
        self.latest["motif_type"] = ""
        self.latest["motif_phase"] = ""
        self.latest["flight_phase"] = "ground_idle"
        self.latest["safety_override_active"] = False
        self.latest["should_land"] = False

        self._event = ""
        self._lock = threading.Lock()
        self._stop_writer = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._log_configs: List[LogConfig] = []

    # -- Crazyflie LogConfig setup -----------------------------------------------------

    def _log_error_callback(self, logconf: LogConfig, msg: str) -> None:
        print(f"Warning: log block '{logconf.name}' error: {msg}")

    def _update_from_log_callback(self, timestamp_ms: int, data: Dict[str, Any], logconf: LogConfig) -> None:
        with self._lock:
            self.latest["cf_time_ms"] = timestamp_ms
            for cf_name, value in data.items():
                csv_name = VAR_TO_COLUMN.get(cf_name)
                if csv_name is None:
                    continue
                v = safe_float(value)
                if v is not None:
                    v *= VAR_SCALE.get(cf_name, 1.0)
                self.latest[csv_name] = v

    def start_log_configs(self, cf) -> None:
        for block in self.log_blocks:
            lg = LogConfig(name=block["name"], period_in_ms=block["period_ms"])
            for cf_name, cf_type in block["vars"]:
                lg.add_variable(cf_name, cf_type)
            lg.data_received_cb.add_callback(self._update_from_log_callback)
            lg.error_cb.add_callback(self._log_error_callback)
            cf.log.add_config(lg)
            lg.start()
            self._log_configs.append(lg)

    def stop_log_configs(self) -> None:
        for log_config in self._log_configs:
            try:
                log_config.stop()
            except Exception:
                pass
        self._log_configs = []

    # -- Flight-loop -> logger updates --------------------------------------------------

    def set_motif_state(
        self,
        *,
        motif_id: str,
        motif_type: str,
        motif_phase: str,
        rep_index: int,
        motif_param_json: str,
        flight_phase: str,
    ) -> None:
        with self._lock:
            self.latest["motif_id"] = motif_id
            self.latest["motif_type"] = motif_type
            self.latest["motif_phase"] = motif_phase
            self.latest["rep_index"] = rep_index
            self.latest["motif_param_json"] = motif_param_json
            self.latest["flight_phase"] = flight_phase

    def set_command(
        self,
        *,
        vx: float, vy: float, yaw_rate: float, height: float,
        vx_raw: float, vy_raw: float, yaw_rate_raw: float, height_raw: float,
        safety_override_active: bool,
    ) -> None:
        with self._lock:
            self.latest["cmd_vx_m_s"] = vx
            self.latest["cmd_vy_m_s"] = vy
            self.latest["cmd_yaw_rate_deg_s"] = yaw_rate
            self.latest["cmd_height_m"] = height
            self.latest["cmd_vx_raw_m_s"] = vx_raw
            self.latest["cmd_vy_raw_m_s"] = vy_raw
            self.latest["cmd_yaw_rate_raw_deg_s"] = yaw_rate_raw
            self.latest["cmd_height_raw_m"] = height_raw
            self.latest["safety_override_active"] = safety_override_active

    def set_safety_state(self, *, safety_flag: str, should_land: bool) -> None:
        with self._lock:
            self.latest["safety_flag"] = safety_flag
            self.latest["should_land"] = should_land

    def set_flight_phase(self, flight_phase: str) -> None:
        with self._lock:
            self.latest["flight_phase"] = flight_phase

    def mark_event(self, event: str) -> None:
        if not event:
            return
        with self._lock:
            self._event = event

    def get_value(self, column_name: str) -> Any:
        with self._lock:
            return self.latest.get(column_name)

    # -- Background CSV writer -----------------------------------------------------------

    def _write_one_row(self) -> None:
        with self._lock:
            row = dict(self.latest)
            event = self._event
            self._event = ""

        zrange_mm = row.get("zrange_mm")
        row["zrange_m"] = (zrange_mm / 1000.0) if zrange_mm is not None else None

        now = time.time()
        row["pc_time_s"] = now
        row["pc_time_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        row["event"] = event
        self.writer.writerow(row)
        self.file.flush()

    def start_continuous_logging(self, rate_hz: float) -> None:
        if self._writer_thread is not None:
            return
        period_s = 1.0 / rate_hz

        def writer_loop() -> None:
            while not self._stop_writer.is_set():
                self._write_one_row()
                time.sleep(period_s)

        self._stop_writer.clear()
        self._writer_thread = threading.Thread(target=writer_loop, daemon=True)
        self._writer_thread.start()

    def stop_continuous_logging(self) -> None:
        if self._writer_thread is None:
            return
        self._stop_writer.set()
        self._writer_thread.join(timeout=1.0)
        self._writer_thread = None
        self._write_one_row()

    def close(self) -> None:
        self.stop_continuous_logging()
        self.file.flush()
        self.file.close()
