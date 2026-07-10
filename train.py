"""Train the AI-KF's ANN altitude estimator (H_i) on simulated trajectory data.

Scope: this trains ONLY the data-driven state estimator piece of the Augmented Information
Kalman Filter described in references/summaries/bens_epic_paper.md -- a small feed-forward
network that maps a sliding window of "impaired" sensor measurements (optical flow + accelerometer,
no direct height sensor) to an altitude estimate. It does NOT implement the AI-KF's Kalman-filter
fusion step (that's elaborate_plan.md Phase 5); references/A_planar_drone_AI_UKF.ipynb has a
working template for that step (on a different, external model) to adapt later.

Data: reads every trajectory listed in <directory>/manifest.csv, where <directory> is either
converted_real_trajectories/ (real flight data, converted by utils/real_data_conversion.py --
the default) or simulated_trajectories/ (produced by utils/trajectory_generator.py), selected via
--directory. For each trajectory it builds windowed (X, y) training samples using the "impaired"
sensor columns -- optical flow (meas_r_x, meas_r_y) and accelerometer (meas_v_x_dot, meas_v_y_dot)
-- as input, and altitude (state_z) as the regression target. This mirrors the paper's altitude
estimator, extended to 2D (the paper's own case only used forward flow + forward acceleration;
our model also has a lateral/y axis).

The train/test split is done BY TRAJECTORY, not by row, so overlapping sliding-window samples
from the same flight never span the split (a naive row-level split would leak information between
train and test, since adjacent windows overlap almost entirely).

With very few trajectories or only one motif represented (as when first testing this pipeline),
there isn't enough data or diversity for a meaningful split or a genuinely useful estimator --
the script detects this, adapts (skips the test split if needed), and prints a clear warning that
the run is a pipeline smoke test, not a real trained estimator.

Usage:
    python3 train.py
        Train on everything in converted_real_trajectories/ (the default) with default settings.
    python3 train.py --directory simulated_trajectories
        Train on simulated_trajectories/ instead.
    python3 train.py --window-s 2.0 --epochs 200 --test-fraction 0.2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so `utils.` resolves regardless of cwd

from utils.ann_utility import build_windowed_dataset, build_model, save_model_complete

CONVERTED_REAL_TRAJECTORIES_DIRNAME = 'converted_real_trajectories'
SIMULATED_TRAJECTORIES_DIRNAME = 'simulated_trajectories'
DIRECTORY_CHOICES = [CONVERTED_REAL_TRAJECTORIES_DIRNAME, SIMULATED_TRAJECTORIES_DIRNAME]
DEFAULT_DIRECTORY = CONVERTED_REAL_TRAJECTORIES_DIRNAME

MODELS_DIR = REPO_ROOT / 'models'

# "Impaired sensor" set: optical flow (flow deck) + accelerometer, no direct height sensor --
# matching the grand-prize scenario and the sensor set used in crazyfly_simulation.ipynb.
INPUT_COLUMNS = ['meas_r_x', 'meas_r_y', 'meas_v_x_dot', 'meas_v_y_dot']
OUTPUT_COLUMN = 'state_z'

DEFAULT_WINDOW_S = 2.0       # matches the reference paper's altitude-estimator window length
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_NOISE_STD = 0.01     # Gaussian input noise during training, paper-style
DEFAULT_HIDDEN_UNITS = (64, 64, 64)
MIN_MOTIFS_FOR_MEANINGFUL_TRAINING = 3


def load_manifest(trajectories_dir):
    manifest_path = trajectories_dir / 'manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'{manifest_path} not found -- run utils/trajectory_generator.py (for '
            f'simulated_trajectories) or utils/real_data_conversion.py (for '
            f'converted_real_trajectories) first.')

    manifest = pd.read_csv(manifest_path)
    manifest['motif'] = manifest['params_json'].apply(lambda s: json.loads(s)['motif'])
    return manifest


def split_trajectories(filenames, test_fraction, seed):
    """Split a list of trajectory filenames into (train_files, test_files), BY TRAJECTORY (not
    by row), so overlapping sliding-window samples from one flight never span the split."""
    filenames = list(filenames)
    if len(filenames) < 2:
        print(f'WARNING: only {len(filenames)} trajectory(ies) available -- too few for any '
              f'train/test split. Training on everything with no held-out evaluation; treat '
              f'this as a pipeline smoke test, not a real trained estimator.')
        return filenames, []

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(filenames))
    n_test = max(1, round(test_fraction * len(shuffled)))
    n_test = min(n_test, len(shuffled) - 1)  # always keep at least 1 trajectory for training
    return shuffled[n_test:], shuffled[:n_test]


def build_dataset(filenames, trajectories_dir, window_steps):
    X_parts, y_parts = [], []
    for filename in filenames:
        df = pd.read_csv(trajectories_dir / filename)
        missing = [c for c in INPUT_COLUMNS + [OUTPUT_COLUMN] if c not in df.columns]
        if missing:
            print(f'skipping {filename}: missing columns {missing}')
            continue
        X, y = build_windowed_dataset(df, INPUT_COLUMNS, OUTPUT_COLUMN, window_steps)
        X_parts.append(X)
        y_parts.append(y)

    if not X_parts:
        raise RuntimeError('no usable trajectories found (check INPUT_COLUMNS/OUTPUT_COLUMN)')

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def infer_dt(trajectories_dir, filename):
    df = pd.read_csv(trajectories_dir / filename)
    return float(np.median(np.diff(df['time'].values)))


def main(trajectories_dir=REPO_ROOT / DEFAULT_DIRECTORY, models_dir=MODELS_DIR,
         window_s=DEFAULT_WINDOW_S, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
         test_fraction=DEFAULT_TEST_FRACTION, noise_std=DEFAULT_NOISE_STD,
         hidden_units=DEFAULT_HIDDEN_UNITS, seed=0):
    manifest = load_manifest(trajectories_dir)
    all_filenames = manifest['filename'].tolist()
    motif_counts = manifest['motif'].value_counts().to_dict()
    print(f'{len(all_filenames)} trajectories available: {motif_counts}')

    if manifest['motif'].nunique() < MIN_MOTIFS_FOR_MEANINGFUL_TRAINING:
        print(f'WARNING: only {manifest["motif"].nunique()} distinct motif(s) present '
              f'({sorted(manifest["motif"].unique())}). The whole point of this project\'s '
              f'sensor set is that altitude is only observable during specific motifs (e.g. '
              f'accel/decel pulses) -- with this little motif diversity the estimator cannot '
              f'yet learn that distinction. Treat this run as a pipeline smoke test.')

    dt = infer_dt(trajectories_dir, all_filenames[0])
    window_steps = max(1, round(window_s / dt))
    print(f'dt={dt:.4f}s -> window={window_steps} steps ({window_steps * dt:.2f}s)')

    train_files, test_files = split_trajectories(all_filenames, test_fraction, seed)
    print(f'train trajectories: {len(train_files)}  test trajectories: {len(test_files)}')

    X_train, y_train = build_dataset(train_files, trajectories_dir, window_steps)
    print(f'X_train: {X_train.shape}  y_train: {y_train.shape}')

    if test_files:
        X_test, y_test = build_dataset(test_files, trajectories_dir, window_steps)
        print(f'X_test: {X_test.shape}  y_test: {y_test.shape}')
    else:
        X_test, y_test = None, None

    model = build_model(input_dim=X_train.shape[1], hidden_units=hidden_units,
                         input_noise_std=noise_std)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    validation_data = (X_test, y_test) if X_test is not None else None
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                         validation_data=validation_data, verbose=2)

    config = {
        'input_dim': X_train.shape[1],
        'hidden_units': list(hidden_units),
        'activation': 'relu',
        'output_activation': 'linear',
        'input_noise_std': noise_std,
        'input_columns': INPUT_COLUMNS,
        'output_column': OUTPUT_COLUMN,
        'window_steps': window_steps,
        'window_s': window_s,
        'dt': dt,
        'train_files': train_files,
        'test_files': test_files,
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'altitude_estimator'
    save_model_complete(model, model_path, config)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=150)
    ax.plot(history.history['loss'], label='train loss')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='val loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    training_curve_path = models_dir / 'training_curve.png'
    fig.savefig(training_curve_path)
    print(f'saved training curve to: {training_curve_path}')

    if X_test is not None:
        y_pred = model.predict(X_test, verbose=0).flatten()
        rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        print(f'test RMSE: {rmse:.4f} m')

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
        ax.scatter(y_test, y_pred, s=4, alpha=0.3)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlabel('true z [m]')
        ax.set_ylabel('predicted z [m]')
        ax.set_title(f'test set: RMSE={rmse:.4f} m')
        fig.tight_layout()
        predictions_path = models_dir / 'test_predictions.png'
        fig.savefig(predictions_path)
        print(f'saved test predictions plot to: {predictions_path}')
    else:
        print('no held-out test set (too few trajectories) -- skipping test evaluation plot.')

    print('done.')


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--directory', type=str, choices=DIRECTORY_CHOICES, default=DEFAULT_DIRECTORY,
                         help=f'which trajectory folder (under the repo root) to train on '
                              f'(default: {DEFAULT_DIRECTORY})')
    parser.add_argument('--models-dir', type=str, default=str(MODELS_DIR))
    parser.add_argument('--window-s', type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--test-fraction', type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument('--noise-std', type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(trajectories_dir=REPO_ROOT / args.directory,
         models_dir=Path(args.models_dir), window_s=args.window_s, epochs=args.epochs,
         batch_size=args.batch_size, test_fraction=args.test_fraction, noise_std=args.noise_std,
         seed=args.seed)
