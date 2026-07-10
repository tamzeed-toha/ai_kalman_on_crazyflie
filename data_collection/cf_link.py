"""
Low-level Crazyflie link helpers: driver init and brushless arm/disarm.

Ported from examples/bisccits/common_cf.py so this project (a separate git repo) does not
depend on a cross-repo relative import.
"""

from __future__ import annotations

import time


def initialize_cflib_drivers() -> None:
    """Initialize Crazyradio/Crazyflie low-level drivers."""
    import cflib.crtp

    time.sleep(0.2)
    cflib.crtp.init_drivers()
    time.sleep(0.2)


def arm_cf(cf) -> None:
    """
    Explicitly request arming before sending motion commands.

    The Crazyflie 2.1 Brushless will not spin its motors without this -- unlike brushed
    CF2.1, which will accept setpoints while disarmed.
    """
    try:
        cf.platform.send_arming_request(True)
        time.sleep(0.2)
    except Exception as exc:
        print(f"Warning: arming request failed: {exc}")


def disarm_cf(cf) -> None:
    """Explicitly request disarming after flight is complete."""
    try:
        cf.platform.send_arming_request(False)
    except Exception:
        pass


def send_stop_setpoint(cf) -> None:
    """Tell the Crazyflie commander that setpoint streaming is finished."""
    try:
        cf.commander.send_stop_setpoint()
    except Exception:
        pass
