"""Consumes the low-rate diagnostic log blocks (kalman_z, power, ranger, baro) independently of
the 100Hz fusion loop -- these are safety/monitoring side-channels, not AI-KF inputs (see
references/rates.tsv purposes column and plan §1).

This is intentionally simple threshold logic, not a control system -- wire is_safe() into
whatever abort/land behavior the flight test harness (Phase 6/7) implements.
"""


class SafetyMonitor:
    def __init__(self, min_battery_level=15, min_ranger_mm=150):
        self.min_battery_level = min_battery_level
        self.min_ranger_mm = min_ranger_mm

        self.battery_level = None
        self.ranger = {}
        self.baro_asl = None
        self.kalman_var_z = None

    def on_power(self, timestamp, data):
        self.battery_level = data["pm.batteryLevel"]

    def on_ranger(self, timestamp, data):
        self.ranger = {
            "front": data["range.front"], "back": data["range.back"],
            "left": data["range.left"], "right": data["range.right"],
            "up": data["range.up"],
        }

    def on_baro(self, timestamp, data):
        self.baro_asl = data["baro.asl"]

    def on_kalman_z(self, timestamp, data):
        self.kalman_var_z = data["kalman.varZ"]

    def is_safe(self):
        if self.battery_level is not None and self.battery_level < self.min_battery_level:
            return False, "battery_low"
        for side, dist_mm in self.ranger.items():
            if dist_mm and dist_mm < self.min_ranger_mm:
                return False, f"obstacle_{side}"
        return True, None
