"""Raw Crazyflie log fields -> the model-frame values realtime/drone_model_np.h_camera_imu
expects (meas_r_x, meas_r_y, meas_v_x_dot, meas_v_y_dot).

These conversions depend on physical calibration (Flow deck v2 optical gain, IMU axis/frame
convention vs. the model's body_level world-aligned x/y) that belongs to Phase 1 model
calibration, not to this real-time plumbing. The functions below are the extension points where
that calibration plugs in -- replace the placeholders once Phase 1 produces real constants;
don't trust the defaults for flight.

The tilt-compensation and flow-axis-swap math here is the SAME logic used by
data_collection/export_to_training_format.py to build training data from real flight logs
(deliberately -- training-time and inference-time preprocessing must match, or the ANN sees a
systematically different input distribution live than it was trained on). If one changes, change
the other.
"""

import math


def raw_flow_to_body_frame_optic_flow(
    delta_x_raw, delta_y_raw, dt, gyro_x_rad_s, gyro_y_rad_s, roll_rad, pitch_rad,
    flow_gain, clamp,
):
    """Convert raw Flow deck v2 pixel counts (motion.deltaX, motion.deltaY -- SENSOR-NATIVE
    axes) into the model's body_level [meas_r_x, meas_r_y] = [v_x/z, v_y/z] (aircraft
    forward/lateral), matching crazyflie-firmware's own flow measurement model
    (src/modules/src/kalman_core/mm_flow.c) as closely as this dependency-light module can.

    TWO THINGS MUST BOTH BE RIGHT, NOT JUST GAIN -- confirmed by empirically regressing real
    logged flow against stateEstimate.vx//zrange_m (see
    data_collection/export_to_training_format.py's validation / chat history):

    1. AXIS SWAP + SIGN: crazyflie-firmware's flowdeck_v1v2.c explicitly computes
       `accpx = -currentMotion.deltaY` (forward) and `accpy = -currentMotion.deltaX` (lateral)
       before pushing into the EKF -- "flip motion information to comply with sensor mounting"
       per its own comment. The raw LOGGED motion.deltaX/deltaY are PRE-swap sensor axes, not
       aircraft forward/lateral. Using them directly (as this function's predecessor did) gives
       R^2~0.002 (pure noise) against real vx/z; applying the swap gives R^2~0.63.
    2. GAIN: flow_gain = FLOW_RESOLUTION * THETAPIX / NPIX (config.py), derived from
       crazyflie-firmware's mm_flow.c constants -- not a guess, but the empirical slope was
       ~0.6-0.7 rather than exactly 1.0 (likely regression dilution from sensor noise / vx not
       being independent ground truth, not necessarily a residual error worth chasing further
       without better ground truth).

    Rotation-rate compensation (gyro_y for forward flow, gyro_x for lateral -- matching
    mm_flow.c's omegay_b/omegax_b cross terms) and the R[2][2]=cos(roll)*cos(pitch) tilt
    projection are included for fidelity to the firmware model, though empirically on the
    dataset used for validation they didn't clearly improve R^2 over the translation-only
    version -- plausibly attitude/gyro noise in this specific dataset, not evidence the
    compensation is wrong. Worth re-checking once more/cleaner data is available.

    clamp: reject physically-impossible outputs (see config.FLOW_SANITY_CLAMP) rather than
    feeding them to the EKF -- a wrong gain/axis/mounting assumption should fail loud (clipped,
    visibly saturated in logs) not diverge the filter silently.
    """
    dpixel_forward = -delta_y_raw
    dpixel_lateral = -delta_x_raw
    r22 = math.cos(roll_rad) * math.cos(pitch_rad)

    r_x = (dpixel_forward * flow_gain / dt + gyro_y_rad_s) / r22
    r_y = (dpixel_lateral * flow_gain / dt + gyro_x_rad_s) / r22

    r_x = max(-clamp, min(clamp, r_x))
    r_y = max(-clamp, min(clamp, r_y))
    return r_x, r_y


def tilt_compensate_horizontal_accel(acc_x_body, acc_y_body, acc_z_body, roll_rad, pitch_rad):
    """Rotate body-frame accelerometer readings (m/s^2) into a yaw-following 'level' frame:
    undoes roll (about body x) then pitch (about body y), removing gravity's projection onto the
    tilted body axes so the result approximates the model's body_level v_x_dot/v_y_dot (meas_v_x_dot/
    meas_v_y_dot). Matches drone_model_np's body_level convention (yaw-relative axes, no yaw
    rotation applied) -- NOT references/planar_drone.py's 2D pitch-only frame this used to target.

    TODO(Phase 1/4): whether real acc.x/acc.y (raw specific force) is exactly the quantity the
    model's v_x_dot/v_y_dot is meant to represent (vs. some other kinematic convention) is not
    independently verified -- see crazyflie_data_adaptation_brief.md Sec. 1-2. This at minimum
    correctly removes tilt, which is necessary regardless of how that question resolves.
    """
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    ay1 = acc_y_body * cr - acc_z_body * sr
    az1 = acc_y_body * sr + acc_z_body * cr
    ax_level = acc_x_body * cp + az1 * sp
    ay_level = ay1
    return ax_level, ay_level


def motor_commands_to_control_input(motor_pwms):
    """Map logged motor PWM commands (power block: motor.m1..m4) to the model's control input
    [u_x, u_y, u_psi, u_z]. Computing this from raw per-motor PWM requires the drone's calibrated
    thrust/torque model (Phase 1), which does not exist in this repo yet. No safe placeholder --
    do not fly with a fabricated value.

    For SCRIPTED test flights (a known, self-generated trajectory) prefer supplying a
    control_input_provider that returns the trajectory's own analytical commanded acceleration/
    yaw-rate directly (see e.g. the square+zigzag test script) instead of reconstructing it from
    motor PWMs -- that's exact, not estimated, for anything the script itself is commanding.
    """
    raise NotImplementedError(
        "Supply a calibrated motor_pwms -> (u_x, u_y, u_psi, u_z) mapping from Phase 1 before "
        "running the live pipeline against arbitrary/piloted flight. This conversion has no "
        "safe placeholder. For scripted test flights, supply a control_input_provider based on "
        "the commanded trajectory instead."
    )
