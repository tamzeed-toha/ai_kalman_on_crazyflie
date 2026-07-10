"""Rolling window buffer for the ANN's causal input, and a gyro dead-reckoning attitude tracker.

channels/window_size are NOT hardcoded here -- they must come from whatever model was actually
loaded (AltitudeANNEstimator derives them from the trained model's own config.json), so the live
window always matches training-time feature construction, even as that changes across retrains.
"""

from collections import deque

import numpy as np


class RollingWindow:
    """Holds the last `window_size` samples of `channels` and flattens them in the exact column
    order train.py/utils/ann_utility.build_windowed_dataset produces: row k's flat vector is
    inputs[k-window+1 : k+1] flattened in (row-major) time-then-channel order, i.e. oldest sample
    first, channels in `channels` order within each sample.

    This ordering must match training-time feature construction exactly, or the ANN sees a
    differently-shaped/ordered input live than it was trained on.
    """

    def __init__(self, channels, window_size):
        self.channels = list(channels)
        self.window_size = window_size
        self._samples = deque(maxlen=window_size)

    def push(self, sample):
        """sample: dict mapping each channel name to its latest value."""
        self._samples.append({ch: float(sample[ch]) for ch in self.channels})

    @property
    def is_full(self):
        return len(self._samples) == self.window_size

    def get_vector(self):
        """Return the flat (window_size * n_channels,) feature vector, oldest-sample-first,
        channel-minor -- matching build_windowed_dataset's row-major window flatten
        (inputs[i:i+window].reshape(-1)). Only valid once is_full."""
        if not self.is_full:
            raise RuntimeError("RollingWindow.get_vector() called before window is full")

        vec = []
        for sample in self._samples:  # oldest (index 0) to newest (index -1)
            for ch in self.channels:
                vec.append(sample[ch])
        return np.array(vec, dtype=np.float32)

    def horizontal_accel_magnitude_window(self):
        """sqrt(meas_v_x_dot^2 + meas_v_y_dot^2) for every buffered sample, oldest-to-newest --
        used for R_aug_z (z is only observable under horizontal acceleration in EITHER axis, not
        just forward, unlike the 2D model this used to mirror)."""
        return np.array([
            np.hypot(s["meas_v_x_dot"], s["meas_v_y_dot"]) for s in self._samples
        ], dtype=np.float64)


class AttitudeTracker:
    """Maintains roll/pitch estimates at the 100Hz fusion tick rate, even though the firmware's
    own attitude estimate (state_att) only refreshes at 50Hz: dead-reckons with gyro.x/gyro.y
    (available every tick) between state_att refreshes, and resets to the firmware's estimate
    whenever a fresh one arrives. Needed to tilt-compensate the raw body-frame accelerometer into
    a level (gravity-consistent) frame every 100Hz tick, not just every 50Hz attitude refresh.
    """

    def __init__(self, initial_roll=0.0, initial_pitch=0.0):
        self.roll = initial_roll
        self.pitch = initial_pitch

    def integrate(self, roll_rate, pitch_rate, dt):
        self.roll += roll_rate * dt
        self.pitch += pitch_rate * dt

    def reset_from_attitude(self, roll_measured, pitch_measured):
        self.roll = roll_measured
        self.pitch = pitch_measured

    def get_attitude(self):
        return self.roll, self.pitch
