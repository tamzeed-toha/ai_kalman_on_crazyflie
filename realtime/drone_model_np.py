"""
Dependency-light numpy mirror of model/drone_simulator.py's DroneModel.f()/h() (body_level
frame, moving_frame='on') -- the model train.py actually trains against, NOT
references/planar_drone.py (the 2D model realtime/ was originally, incorrectly, built around).

Duplicated rather than imported because model/drone_simulator.py unconditionally imports casadi,
pybounds, sympy, and matplotlib at module level (needed for DroneSimulator's MPC/plotting, not
for DroneModel.f()/h() themselves, which are pure numpy) -- pulling those into a latency-
sensitive real-time control-loop dependency footprint isn't worth it just to reuse ~30 lines of
math. See CLAUDE.md / crazyflie_data_adaptation_brief.md for the broader model-choice context.

Per instruction, model/drone_simulator.py itself is NOT modified -- if its f()/h() equations
change, this file must be updated to match by hand; there is no automated check that they stay
in sync.

Reduced from DroneModel's full 12-state model to 7 states by fixing the states this project
doesn't need or can't observe with the current sensor set:
  - No ambient wind (indoor flight, matches utils/trajectory_generator.py's `w = zeros_like(t)`)
    -> drop w, zeta, fix wind=0 in the equations.
  - No online motor-calibration estimation -> drop k_x, k_y, k_psi, fix all three to 1.0 (matches
    utils/trajectory_generator.py's simulated setpoints, which also just use k_x=k_y=k_psi=1.0).
Tracking un-observed states with no measurement constraining them just accumulates uncontrolled
covariance growth for no benefit, so they're fixed constants here rather than filter states.

State order: [x, y, z, v_x, v_y, v_z, psi]  (world position/velocity + yaw; v_x/v_y are
body_level-frame, i.e. forward/lateral, not world-frame -- matches DroneModel's body_level
convention).
Input order: [u_x, u_y, u_psi, u_z]  (commanded body-level accel_x/accel_y, yaw rate, vertical
accel -- dropping DroneModel's u_w/u_zeta wind-rate inputs along with the wind states).
"""

import numpy as np

STATE_NAMES = ["x", "y", "z", "v_x", "v_y", "v_z", "psi"]
INPUT_NAMES = ["u_x", "u_y", "u_psi", "u_z"]

# Matches train.py's INPUT_COLUMNS order exactly -- the EKF's base measurement vector must be
# built in this same order every tick (see fusion_loop.py).
MEASUREMENT_NAMES = ["meas_r_x", "meas_r_y", "meas_v_x_dot", "meas_v_y_dot"]


def f(X, U):
    """Continuous-time dynamics, body_level frame, moving_frame='on' (mirrors DroneModel.f with
    w=zeta=0, k_x=k_y=k_psi=1)."""
    x, y, z, v_x, v_y, v_z, psi = X
    u_x, u_y, u_psi, u_z = U

    psi_dot = u_psi
    v_z_dot = u_z
    z_dot = v_z

    v_x_dot = u_x + psi_dot * v_y
    v_y_dot = u_y - psi_dot * v_x

    x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
    y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)

    return np.array([x_dot, y_dot, z_dot, v_x_dot, v_y_dot, v_z_dot, psi_dot])


def h_camera_imu(X, U):
    """Measurement model: [meas_r_x, meas_r_y, meas_v_x_dot, meas_v_y_dot] -- exactly the
    features train.py's ANN is trained on (see export_to_training_format.py). r_x = v_x/z,
    r_y = v_y/z per DroneModel's body_level convention (v_para=v_x, v_perp=v_y, no psi rotation)."""
    x, y, z, v_x, v_y, v_z, psi = X

    x_dot_vec = f(X, U)
    v_x_dot, v_y_dot = x_dot_vec[3], x_dot_vec[4]

    r_x = v_x / z
    r_y = v_y / z
    return np.array([r_x, r_y, v_x_dot, v_y_dot])
