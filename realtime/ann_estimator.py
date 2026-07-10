"""Loads an already-trained altitude ANN (see references/keras_ann_utility.py and Phase 4) and
wraps it for low-latency per-tick inference.

This module assumes a trained model already exists on disk (produced by
keras_ann_utility.save_model_complete, e.g. Phase 4's "v1_real") -- it does not train anything.
"""

import os

import numpy as np
import tensorflow as tf

from references.keras_ann_utility import create_fast_inference_model, load_model_complete
from realtime.config import ANN_CHANNELS, R_AUG_Z_ACCEL_FLOOR, WINDOW_SIZE


class AltitudeANNEstimator:
    def __init__(self, model_dir, model_name, channels=ANN_CHANNELS, window_size=WINDOW_SIZE):
        model_path = os.path.join(model_dir, model_name)
        self.model, self.config = load_model_complete(model_path)
        self.fast_predict = create_fast_inference_model(self.model)

        expected_input_dim = window_size * len(channels)
        actual_input_dim = self.config["input_architecture"][0]["core_input_dim"]
        if actual_input_dim != expected_input_dim:
            raise ValueError(
                f"Loaded model expects input_dim={actual_input_dim}, but the configured "
                f"window_size={window_size} x channels={channels} produces "
                f"{expected_input_dim}. The live window builder and the trained model must "
                f"agree -- check WINDOW_SIZE/ANN_CHANNELS in realtime/config.py against how "
                f"this model was trained."
            )

        # Warm up the XLA-compiled path now (first call is slow) so the first live tick isn't.
        self.fast_predict(tf.constant(np.zeros((1, expected_input_dim), dtype=np.float32)))

    def predict(self, window_vector):
        """window_vector: flat (window_size * n_channels,) array from RollingWindow.get_vector()."""
        x = tf.constant(np.atleast_2d(window_vector).astype(np.float32))
        z_pred = self.fast_predict(x)
        return float(np.asarray(z_pred).reshape(-1)[0])


def compute_r_aug_z(accel_x_window):
    """Time-varying measurement covariance for the ANN's z_pred pseudo-measurement, following
    the reference notebook's R_aug_z ~= 1 / min(|accel_x|) over the current window: small
    (trustworthy) when the window contains a strong horizontal-acceleration burst, large
    (distrusted) during hover/constant-velocity segments where z isn't observable at all.
    """
    min_abs_accel_x = np.min(np.abs(accel_x_window))
    return 1.0 / max(min_abs_accel_x, R_AUG_Z_ACCEL_FLOOR)
