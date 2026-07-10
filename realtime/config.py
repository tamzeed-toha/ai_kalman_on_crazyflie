"""Constants for the real-time AI-KF pipeline.

Rates below come from references/rates.tsv (the planned Crazyflie log-block layout).
"""

DEFAULT_URI = "radio://0/80/2M"

# --- Fusion loop rate ---
# Driven by the merged imu+flow log block (both natively 100Hz per rates.tsv).
FUSION_RATE_HZ = 100
DT = 1.0 / FUSION_RATE_HZ

# state_att (theta/theta_dot ground truth refresh) is only 50Hz -- theta is dead-reckoned
# with gyro (theta_dot) between refreshes. See buffers.ThetaTracker.
ATTITUDE_RATE_HZ = 50

# Diagnostic-only blocks (not fusion inputs) -- see rates.tsv purposes column.
KALMAN_DIAG_RATE_HZ = 20
POWER_RATE_HZ = 20
RANGER_RATE_HZ = 20
BARO_RATE_HZ = 10

# --- ANN window ---
# Must exactly match whatever collect_offset_rows(...) call was used to build the training set.
ANN_CHANNELS = ["optic_flow", "accel_x", "accel_z"]
WINDOW_SIZE = 10
ANN_OFFSETS = list(range(0, -WINDOW_SIZE, -1))  # [0, -1, -2, ..., -9], causal only

# R applied to the ANN's pseudo-measurement while the window isn't full yet (i.e. "ignore it").
R_AUG_Z_COLD_START = 1e8
# Floor to avoid dividing by ~0 when accel_x is near zero within a window (low observability).
R_AUG_Z_ACCEL_FLOOR = 1e-3

# --- EKF measurement set (planar_drone.H('h_camera_imu')) ---
# ['optic_flow', 'theta', 'theta_dot', 'accel_x', 'accel_z']
R_BASE_DIAG = [1e-2, 1e-3, 1e-3, 1e-1, 1e-1]  # placeholder -- tune against real sensor noise

# --- EKF process noise (planar_drone.F, state = [theta, theta_dot, x, x_dot, z, z_dot, k]) ---
Q_DIAG = [1e-4, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-6]  # placeholder -- tune in Phase 5 offline replay

# --- Trained model artifact ---
DEFAULT_MODEL_NAME = "v1_real"

# --- Raw-sensor-to-model-frame calibration placeholders (see sensor_conversion.py) ---
# TODO(Phase 1): replace with the real Flow deck v2 optical gain constant.
FLOW_GAIN_PLACEHOLDER = 1.0
# Bitcraze IMU log values are conventionally in units of g; multiply by this to get m/s^2.
G_MPS2 = 9.81
