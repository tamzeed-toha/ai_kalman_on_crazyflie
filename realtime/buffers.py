"""Rolling window buffer for the ANN's causal input, and a gyro dead-reckoning theta tracker."""

from collections import deque

import numpy as np

from realtime.config import ANN_CHANNELS, WINDOW_SIZE


class RollingWindow:
    """Holds the last WINDOW_SIZE samples of ANN_CHANNELS and flattens them in the exact
    column order collect_offset_rows(states=ANN_CHANNELS, state_offsets=[0,-1,...,-9]) produces:
    for offset in [0, -1, ..., -(W-1)]: for channel in ANN_CHANNELS: value.

    This ordering must match training-time feature construction exactly, or the ANN sees a
    differently-shaped input live than it was trained on.
    """

    def __init__(self, channels=ANN_CHANNELS, window_size=WINDOW_SIZE):
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
        """Return the flat (window_size * n_channels,) feature vector, offset-major then
        channel-minor, matching collect_offset_rows's column order. Only valid once is_full."""
        if not self.is_full:
            raise RuntimeError("RollingWindow.get_vector() called before window is full")

        vec = []
        # offset 0 is the newest sample (deque[-1]); offset -(k) is deque[-(k+1)]
        for k in range(self.window_size):
            sample = self._samples[-(k + 1)]
            for ch in self.channels:
                vec.append(sample[ch])
        return np.array(vec, dtype=np.float32)

    def accel_x_window(self):
        """Raw accel_x values currently buffered, oldest-to-newest -- used for R_aug_z."""
        return np.array([s["accel_x"] for s in self._samples], dtype=np.float64)


class ThetaTracker:
    """Maintains a theta (pitch) estimate at the 100Hz fusion tick rate, even though the
    firmware's own attitude estimate (state_att) only refreshes at 50Hz: dead-reckons with
    theta_dot (gyro, available every tick) between state_att refreshes, and resets to the
    firmware's estimate whenever a fresh one arrives. See elaborate_plan.md discussion of the
    imu/flow-vs-state_att rate mismatch.
    """

    def __init__(self, initial_theta=0.0):
        self.theta = initial_theta

    def integrate(self, theta_dot, dt):
        self.theta += theta_dot * dt

    def reset_from_attitude(self, theta_measured):
        self.theta = theta_measured

    def get_theta(self):
        return self.theta
