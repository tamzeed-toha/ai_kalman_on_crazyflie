"""ANN utilities for training the AI-KF's data-driven state estimator (H_i).

Adapted from references/keras_ann_utility.py for this project's altitude estimator:
    - build_model() keeps the reference's small-Sequential-MLP approach, but takes a flat
      hidden_units list instead of the reference's input_architecture/core_architecture dict
      pair, and adds an optional GaussianNoise input layer (the paper's altitude/wind estimators
      both train with input noise so the network is robust to real sensor noise).
    - save_model_complete()/load_model_complete() keep the reference's weights+JSON-config
      round-trip, but write '<name>.weights.h5' rather than '<name>_weights.h5' -- Keras 3
      (installed here) requires save_weights()'s filename to literally end in '.weights.h5'.
    - build_windowed_dataset() is new: a single-purpose version of the reference's
      collect_offset_rows(), specialized to "one window of input columns -> one output column at
      the current time step", which is exactly the H_i estimator's input/output shape.
"""

import json
from pathlib import Path

import numpy as np
from tensorflow import keras


def build_windowed_dataset(df, input_cols, output_col, window):
    """Build (X, y) arrays for training H_i from one trajectory's dataframe.

    X[k] is the flattened window of `input_cols` over the `window` most recent rows ending at
    (and including) row k; y[k] is `output_col` at row k. Equivalent to the reference utility's
    collect_offset_rows(df, states=input_cols, state_offsets=range(-(window-1), 1),
    outputs=[output_col], output_offsets=[0]), specialized for speed/clarity to this one shape.
    """
    n_rows = len(df)
    if n_rows < window:
        raise ValueError(f'trajectory has {n_rows} rows, shorter than window={window}')

    inputs = df[input_cols].values  # (n_rows, n_channels)
    output = df[output_col].values  # (n_rows,)

    n_samples = n_rows - window + 1
    n_channels = len(input_cols)

    X = np.zeros((n_samples, window, n_channels))
    for i in range(n_samples):
        X[i] = inputs[i:i + window]
    y = output[window - 1:]

    return X.reshape(n_samples, -1), y


def build_model(input_dim, hidden_units=(64, 64, 64), activation='relu',
                output_activation='linear', input_noise_std=0.0):
    """Small feed-forward regressor, matching the reference paper's altitude-estimator
    architecture (3 hidden layers x 64 neurons, ReLU, linear output).

    input_noise_std > 0 adds a Gaussian-noise layer right after the input (active only during
    training, inactive at inference/evaluation), matching the paper's approach of training the
    estimator to be robust to sensor noise.
    """
    model = keras.models.Sequential()

    if input_noise_std > 0:
        model.add(keras.layers.GaussianNoise(input_noise_std, input_shape=(input_dim,)))
        model.add(keras.layers.Dense(hidden_units[0], activation=activation))
    else:
        model.add(keras.layers.Dense(hidden_units[0], input_dim=input_dim, activation=activation))

    for units in hidden_units[1:]:
        model.add(keras.layers.Dense(units, activation=activation))

    model.add(keras.layers.Dense(1, activation=output_activation))
    return model


def save_model_complete(model, filepath, config):
    """Save model weights ('<filepath>.weights.h5') and a JSON config ('<filepath>.config.json')
    alongside it. `config` should be whatever build_model() was called with, plus any extra
    bookkeeping (input/output column names, window size, which trajectories were used, ...)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    weights_path = filepath.parent / f'{filepath.name}.weights.h5'
    model.save_weights(weights_path)

    config_path = filepath.parent / f'{filepath.name}.config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'saved weights to: {weights_path}')
    print(f'saved config to: {config_path}')


def load_model_complete(filepath):
    """Rebuild the model from a saved config + weights. Returns (model, config)."""
    filepath = Path(filepath)

    config_path = filepath.parent / f'{filepath.name}.config.json'
    with open(config_path) as f:
        config = json.load(f)

    model = build_model(
        input_dim=config['input_dim'],
        hidden_units=tuple(config['hidden_units']),
        activation=config.get('activation', 'relu'),
        output_activation=config.get('output_activation', 'linear'),
        input_noise_std=config.get('input_noise_std', 0.0),
    )

    weights_path = filepath.parent / f'{filepath.name}.weights.h5'
    model.load_weights(weights_path)

    print(f'loaded weights from: {weights_path}')
    return model, config
