"""Raw Crazyflie log fields -> the model-frame values planar_drone.H('h_camera_imu') expects.

These conversions depend on physical calibration (Flow deck v2 optical gain, IMU axis/frame
convention vs. the planar model's world-aligned x/z) that belongs to Phase 1 model calibration,
not to this real-time plumbing. The functions below are the extension points where that
calibration plugs in -- replace the placeholders once Phase 1 produces real constants; don't
trust the defaults for flight.
"""


def raw_flow_to_optic_flow(delta_x_px, dt, flow_gain):
    """Convert Flow deck v2 pixel displacement (motion.deltaX) to the model's optic_flow = x_dot/z.

    flow_gain is the deck's rad/px (or equivalent) calibration constant -- TODO: pull the real
    value from the Flow deck v2 datasheet / Phase 1 calibration instead of using a placeholder.
    """
    return (delta_x_px * flow_gain) / dt


def body_accel_to_planar_frame(acc_x_body, acc_z_body, theta):
    """Rotate body-frame accelerometer readings into the planar model's world-aligned accel_x/z.

    planar_drone.H.h_camera_imu defines accel_x/accel_z in the world/motion-plane frame
    (derived from tilted collective thrust), while acc.x/acc.z from the IMU log block are in
    the body frame. TODO: verify/replace this rotation against the firmware's actual IMU axis
    convention and gravity-compensation behavior (Phase 1) -- this is a placeholder identity-ish
    rotation, not a validated conversion.
    """
    import numpy as np

    accel_x = acc_x_body * np.cos(theta) - acc_z_body * np.sin(theta)
    accel_z = acc_x_body * np.sin(theta) + acc_z_body * np.cos(theta)
    return accel_x, accel_z


def motor_commands_to_control_input(motor_pwms):
    """Map logged motor PWM commands (power block: motor.m1..m4) to the planar model's [j1, j2].

    j1/j2 are pitch-torque and collective-thrust control inputs in planar_drone.F -- computing
    them from raw per-motor PWM requires the drone's calibrated thrust/torque model (Phase 1),
    which does not exist in this repo yet. This placeholder returns zeros; do not fly with it.
    """
    raise NotImplementedError(
        "Supply a calibrated motor_pwms -> (j1, j2) mapping from Phase 1 before running the "
        "live pipeline. This conversion has no safe placeholder."
    )
