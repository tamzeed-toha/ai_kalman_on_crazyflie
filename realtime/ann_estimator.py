"""Loads an already-trained altitude ANN (see utils/ann_utility.py and train.py) and wraps it
for low-latency per-tick inference.

This module assumes a trained model already exists on disk (produced by
utils/ann_utility.save_model_complete via train.py) -- it does not train anything. Loads via
utils.ann_utility (the schema train.py actually writes: flat config['input_dim'], not
references/keras_ann_utility.py's nested config['input_architecture'] -- that mismatch meant a
model trained by train.py could not be loaded here at all before this fix).

window_size and channels are derived from the loaded model's own config (config['window_steps'],
config['input_columns']), NOT from hardcoded constants -- so "pull the most recent trained
model" stays true even if a future retrain changes the window length or feature set.
"""

import glob
import os

import numpy as np
import tensorflow as tf

from utils.ann_utility import load_model_complete


def find_latest_model(models_dir):
    """Return the base model name (without '.config.json'/'.weights.h5') of the most recently
    modified model artifact in models_dir -- "pull the most recent trained stuff"."""
    config_paths = glob.glob(os.path.join(models_dir, "*.config.json"))
    if not config_paths:
        raise FileNotFoundError(f"No '*.config.json' model artifacts found in {models_dir}")
    latest = max(config_paths, key=os.path.getmtime)
    return os.path.basename(latest)[: -len(".config.json")]


class AltitudeANNEstimator:
    def __init__(self, model_dir, model_name=None):
        if model_name is None:
            model_name = find_latest_model(model_dir)
        model_path = os.path.join(model_dir, model_name)
        self.model, self.config = load_model_complete(model_path)
        self.model_name = model_name

        self.channels = self.config["input_columns"]
        self.window_size = self.config["window_steps"]
        self.expected_input_dim = self.window_size * len(self.channels)
        if self.config["input_dim"] != self.expected_input_dim:
            raise ValueError(
                f"Loaded model '{model_name}' config is internally inconsistent: "
                f"input_dim={self.config['input_dim']} but window_steps={self.window_size} x "
                f"len(input_columns)={len(self.channels)} = {self.expected_input_dim}."
            )

        self._fast_predict = tf.function(
            lambda x: self.model(x, training=False),
            input_signature=[tf.TensorSpec(shape=[None, self.expected_input_dim], dtype=tf.float32)],
        )
        # Warm up the traced path now (first call is slow) so the first live tick isn't.
        self._fast_predict(tf.constant(np.zeros((1, self.expected_input_dim), dtype=np.float32)))

    def predict(self, window_vector):
        """window_vector: flat (window_size * n_channels,) array from RollingWindow.get_vector()."""
        x = tf.constant(np.atleast_2d(window_vector).astype(np.float32))
        z_pred = self._fast_predict(x)
        return float(np.asarray(z_pred).reshape(-1)[0])


def compute_r_aug_z(horizontal_accel_magnitude_window, floor):
    """Time-varying measurement covariance for the ANN's z_pred pseudo-measurement, following
    the reference notebook's R_aug_z ~= 1 / min(|accel|) over the current window: small
    (trustworthy) when the window contains a strong horizontal-acceleration burst in EITHER axis,
    large (distrusted) during hover/constant-velocity segments where z isn't observable at all.
    """
    min_accel = np.min(horizontal_accel_magnitude_window)
    return 1.0 / max(min_accel, floor)
