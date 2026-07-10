"""Constants for the real-time AI-KF pipeline.

Rates below come from references/rates.tsv (the planned Crazyflie log-block layout).
"""

DEFAULT_URI = "radio://0/80/2M"

# --- Fusion loop rate ---
# Driven by the merged imu+flow log block (both natively 100Hz per rates.tsv).
FUSION_RATE_HZ = 100
DT = 1.0 / FUSION_RATE_HZ

# state_att (roll/pitch ground truth refresh) is only 50Hz -- roll/pitch are dead-reckoned with
# gyro between refreshes. See buffers.AttitudeTracker.
ATTITUDE_RATE_HZ = 50

# Diagnostic-only blocks (not fusion inputs) -- see rates.tsv purposes column.
KALMAN_DIAG_RATE_HZ = 20
POWER_RATE_HZ = 20
RANGER_RATE_HZ = 20
BARO_RATE_HZ = 10

# --- ANN window ---
# NOT hardcoded here anymore -- ANN_CHANNELS/window_size are derived from whatever model was
# actually loaded (AltitudeANNEstimator reads config['input_columns']/config['window_steps']
# from the trained model's own .config.json), so "pull the most recent trained model" stays true
# even if a retrain changes the window length or feature set. See ann_estimator.py.

# R applied to the ANN's pseudo-measurement while the window isn't full yet (i.e. "ignore it").
R_AUG_Z_COLD_START = 1e8
# Floor to avoid dividing by ~0 when horizontal accel is near zero within a window (low observability).
R_AUG_Z_ACCEL_FLOOR = 1e-3

# --- EKF measurement set (realtime/drone_model_np.h_camera_imu) ---
# ['meas_r_x', 'meas_r_y', 'meas_v_x_dot', 'meas_v_y_dot'] -- matches train.py's INPUT_COLUMNS.
R_BASE_DIAG = [1e-2, 1e-2, 1e-1, 1e-1]  # placeholder -- tune against real sensor noise

# --- EKF process noise (realtime/drone_model_np.f, state = [x, y, z, v_x, v_y, v_z, psi]) ---
Q_DIAG = [1e-3, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-4]  # placeholder -- tune in Phase 5 offline replay

# --- Trained model artifact ---
# None -> AltitudeANNEstimator auto-discovers the most recently modified '*.config.json' in
# --model-dir (see ann_estimator.find_latest_model). Set explicitly to pin a specific model.
DEFAULT_MODEL_NAME = None

# --- Raw-sensor-to-model-frame calibration (see sensor_conversion.py) ---
# Derived from crazyflie-firmware's src/modules/src/kalman_core/mm_flow.c -- the firmware's own
# flow measurement model -- not a guess. The OLD FLOW_GAIN_PLACEHOLDER=1.0 was ~500x too large
# AND (separately, more importantly) built on the wrong axis mapping: the firmware explicitly
# swaps and negates axes before use (src/deck/drivers/src/flowdeck_v1v2.c:
# dpixelx=-currentMotion.deltaY, dpixely=-currentMotion.deltaX) -- raw logged motion.deltaX/
# motion.deltaY are sensor-native axes, NOT aircraft forward/lateral. sensor_conversion.py
# applies this swap; get it wrong and every flow-derived measurement is on the wrong axis
# regardless of gain. Empirically confirmed against real flight data (see
# data_collection/export_to_training_format.py's validation): correlating the corrected r_x
# against stateEstimate.vx/zrange_m gives R^2~0.63 (vs R^2~0.002 for the old placeholder/mapping).
# Slope was ~0.6-0.7, not exactly 1.0 -- likely regression dilution from sensor noise and
# stateEstimate.vx not being independent ground truth, not necessarily a residual gain error.
FLOW_NPIX = 35.0
FLOW_THETAPIX = 0.71674  # 2*sin(21 deg); firmware's comment: "42-degree angle of aperture"
FLOW_RESOLUTION = 0.1    # raw PMW3901 registers report 10x actual motion pixels (firmware comment)
FLOW_GAIN = FLOW_RESOLUTION * FLOW_THETAPIX / FLOW_NPIX  # ~0.002048

# Reject/clip flow-derived measurements outside this range before they reach the EKF -- r_x/r_y
# should physically be O(0.01-3) for this vehicle's speed/height envelope; anything wildly beyond
# that is almost certainly a bad conversion (wrong gain/axis/units), not real motion, and feeding
# it to the filter causes runaway divergence (confirmed: a single-pixel bad-mapping sample drove
# z from 0.4m to >100m in seconds). This is a safety net, not a substitute for correct calibration.
FLOW_SANITY_CLAMP = 5.0

# Bitcraze IMU log values are conventionally in units of g; multiply by this to get m/s^2.
G_MPS2 = 9.81
