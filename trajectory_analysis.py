"""Combined per-trajectory analysis: flight path and the trained ANN altitude estimator's raw
estimate -- merged into one PNG per trajectory under trajectory_analysis/.

*** BOUNDS observability analysis is temporarily DISABLED (commented out) as of the current
version, to keep this fast while iterating on trajectories/estimation. Search this file for
"OBSERVABILITY DISABLED" to find every commented block to restore, and see git history /
conversation log for the original 3x2-panel version (x-y path colored by observability, z error
variance over time) if you need the full diff back. The short version of how to re-enable: ***
    1. Uncomment the pybounds/model imports + patch_pybounds_simulator_time_conversion() call
       near the top of this file.
    2. Uncomment compute_observability()'s body (remove its early `return None`).
    3. In build_figure(), uncomment the compute_observability() call and restore a grid with
       room for the two observability panels (search "OBSERVABILITY DISABLED").

Panels (see build_figure()), currently a 1x2 grid:
    (0,0) x-y planar position: commanded (open-loop kinematic integration of the reconstructed
          setpoint -- no MPC tracking) vs actual (MPC-tracked), if reconstructable (needs a
          known seed + motif, i.e. simulated_trajectories only, and state_x/state_y columns).
    (0,1) The AI-KF's data-driven estimator (H_i): true z vs the trained ANN's raw estimate over
          time, shaded by a high-horizontal-acceleration proxy (since real observability is
          currently disabled) -- the same "the raw estimate is only accurate when observable"
          story as the paper's Figure 3g/4c.

Scope note: this plots the trained ANN state estimator's raw output (H_i in the paper's
notation), not a full running Kalman-filter fusion -- the actual AI-KF fusion step
(observability-weighted covariance feeding a Kalman filter) is not yet implemented anywhere in
this repo (see CLAUDE.md's "Known cross-file inconsistency" / Phase 5 status).

Usage:
    python3 trajectory_analysis.py --trajectory 0
        Analyze manifest index 0 in simulated_trajectories/ (the default directory).
    python3 trajectory_analysis.py --trajectory all
        Fast pass (flight path + ANN estimate only) across every trajectory.
    python3 trajectory_analysis.py --trajectory 3 --directory converted_real_trajectories
        Analyze real trajectory #3.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so `utils.`/`model.`/`train` resolve regardless of cwd

# OBSERVABILITY DISABLED: these imports are only needed by compute_observability() and the
# (1,0)/(1,1) panels that used to plot it. Uncomment to re-enable (see module docstring).
# from utils.pybounds_compat import patch_pybounds_simulator_time_conversion
# patch_pybounds_simulator_time_conversion()  # must run before any DroneSimulator is constructed
# from pybounds import SlidingEmpiricalObservabilityMatrix, SlidingFisherObservability, colorline
# from model.drone_simulator import DroneSimulator

from utils.trajectory_generator import MOTIFS, Z_RANGE
from utils.ann_utility import build_windowed_dataset, load_model_complete
from train import (SIMULATED_TRAJECTORIES_DIRNAME, CONVERTED_REAL_TRAJECTORIES_DIRNAME,
                   DIRECTORY_CHOICES, DEFAULT_DIRECTORY, INPUT_COLUMNS, OUTPUT_COLUMN)

OUTPUT_DIR = REPO_ROOT / 'trajectory_analysis'
MODELS_DIR = REPO_ROOT / 'models'
DEFAULT_MODEL_NAME = 'altitude_estimator'

DEFAULT_WINDOW_S = 2.0  # observability window, matches the paper's altitude estimator & train.py
MPC_HORIZON = 10

IMPAIRED_SENSORS = ['r_x', 'r_y', 'v_x_dot', 'v_y_dot']
FULL_SENSORS = IMPAIRED_SENSORS + ['z']
MEASUREMENT_NOISE_STDS = {'r_x': 0.05, 'r_y': 0.05, 'v_x_dot': 0.2, 'v_y_dot': 0.2, 'z': 0.02}

FULL_STATE_CANARY_COLUMN = 'state_x'  # presence implies the full 12-state schema BOUNDS needs

# Rough empirical calibration (seconds per empirical-observability simulate step) from timed
# runs recorded in CLAUDE.md/conversation history (~0.0010-0.0014 s/step) -- used only to print
# a heads-up estimate before the slow BOUNDS step, not for anything load-bearing. Will vary by
# machine.
EMPIRICAL_SIMULATE_STEP_SECONDS = 0.0012
HEARTBEAT_INTERVAL_S = 30


def _format_duration(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f'{hours}h{minutes:02d}m'
    if minutes:
        return f'{minutes}m{secs:02d}s'
    return f'{secs}s'


class _Heartbeat:
    """Prints a periodic 'still working' message from a background thread, so a long blocking
    call (like SlidingEmpiricalObservabilityMatrix, which has no progress callback of its own)
    doesn't look hung. Use as a context manager."""

    def __init__(self, message, interval_s=HEARTBEAT_INTERVAL_S):
        self.message = message
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start_time = None

    def _run(self):
        while not self._stop_event.wait(self.interval_s):
            print(f'  ... {self.message} ({_format_duration(time.time() - self._start_time)} '
                 f'elapsed, still running)')

    def __enter__(self):
        self._start_time = time.time()
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop_event.set()
        self._thread.join(timeout=1.0)


class _NullContext:
    """No-op context manager, used in place of _Heartbeat when verbose=False."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest(directory):
    manifest_path = directory / 'manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(f'{manifest_path} not found.')
    manifest = pd.read_csv(manifest_path)
    manifest['motif'] = manifest['params_json'].apply(lambda s: json.loads(s)['motif'])
    return manifest


def has_full_state_schema(df):
    return FULL_STATE_CANARY_COLUMN in df.columns


def infer_dt(df):
    return float(np.median(np.diff(df['time'].values)))


def _strip_trajectory_extension(filename):
    """Strip the trailing .csv.gz/.csv extension. NOT `Path(filename).stem.split('.')[0]` --
    real trajectory filenames embed decimal parameter values (e.g. 'a0.60', 'v0.20'), and a
    naive split on '.' truncates at the FIRST one, silently collapsing many distinct
    trajectories (different accel/speed/etc.) onto the same output filename."""
    name = str(filename)
    if name.endswith('.csv.gz'):
        return name[:-len('.csv.gz')]
    if name.endswith('.csv'):
        return name[:-len('.csv')]
    return Path(name).stem


# ---------------------------------------------------------------------------
# Setpoint reconstruction (simulated_trajectories only -- replays the exact seeded RNG draws
# trajectory_generator.py used, to recover the commanded profile BEFORE MPC tracking).
# ---------------------------------------------------------------------------

def reconstruct_setpoint(motif, seed, t):
    if seed is None or (isinstance(seed, float) and np.isnan(seed)):
        return None
    if motif not in MOTIFS:
        return None

    rng = np.random.default_rng(int(seed))
    v_x_set, v_y_set, psi_set, _params = MOTIFS[motif](t, rng)
    z_set = rng.uniform(*Z_RANGE)
    return {'v_x': v_x_set, 'v_y': v_y_set, 'psi': psi_set, 'z': z_set * np.ones_like(t)}


def reconstruct_setpoint_xy(setpoint, t):
    """Kinematically (open-loop, no MPC tracking) integrate the commanded v_x/v_y/psi into an
    x-y position path, for comparison against the actual MPC-tracked path. Uses the same
    body-level-frame kinematics as model/drone_simulator.py's DroneModel.f() (x_dot/y_dot don't
    depend on moving_frame on/off, only v_x_dot/v_y_dot do, so this is valid regardless)."""
    v_x, v_y, psi = setpoint['v_x'], setpoint['v_y'], setpoint['psi']
    x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
    y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
    dt_arr = np.diff(t)
    x = np.concatenate([[0.0], np.cumsum(x_dot[:-1] * dt_arr)])
    y = np.concatenate([[0.0], np.cumsum(y_dot[:-1] * dt_arr)])
    return x, y


# ---------------------------------------------------------------------------
# Generic "high horizontal acceleration" mask -- works on any trajectory (only needs meas_*
# columns), used for panel (0,1)'s shading and as a fallback for panel (2,1) when full BOUNDS
# observability isn't available. Same approach as crazyfly_simulation_3d.ipynb.
# ---------------------------------------------------------------------------

def mask_to_segments(t, mask):
    segments = []
    in_run = False
    start_i = 0
    for i, flag in enumerate(mask):
        if flag and not in_run:
            start_i = i
            in_run = True
        elif not flag and in_run:
            segments.append((t[start_i], t[i - 1]))
            in_run = False
    if in_run:
        segments.append((t[start_i], t[-1]))
    return segments


def high_accel_segments(df, percentile=75):
    accel_mag = np.sqrt(df['meas_v_x_dot'].values ** 2 + df['meas_v_y_dot'].values ** 2)
    mask = accel_mag > np.percentile(accel_mag, percentile)
    return mask_to_segments(df['time'].values, mask)


def high_observability_segments(time_grid, ev_time, ev_values, percentile=25):
    """Windows where the (resampled-onto-time_grid) minimum error variance is in the lowest
    `percentile` -- i.e. the most observable regions."""
    ev_resampled = np.interp(time_grid, ev_time, ev_values)
    mask = ev_resampled < np.percentile(ev_resampled, percentile)
    return mask_to_segments(time_grid, mask)


def _shade_segments(ax, segments, color='tab:orange', alpha=0.15, label=None):
    for i, (t0, t1) in enumerate(segments):
        ax.axvspan(t0, t1, color=color, alpha=alpha, label=label if i == 0 else None)


# ---------------------------------------------------------------------------
# Observability (BOUNDS) -- same pipeline as crazyfly_simulation_3d.ipynb, factored out for
# batch/CLI use.
#
# OBSERVABILITY DISABLED: body commented out for now (see module docstring for why / how to
# restore). Uncomment everything below the `return None` and remove that line to re-enable.
# ---------------------------------------------------------------------------

def compute_observability(df, dt, window_s, verbose=True):
    return None

    # if not has_full_state_schema(df):
    #     return None
    #
    # simulator = DroneSimulator(frame='body_level', moving_frame='on', dt=dt, mpc_horizon=MPC_HORIZON)
    #
    # t_sim = df['time'].values
    # x_sim = {name: df[f'state_{name}'].values for name in simulator.state_names}
    # u_sim = {name: df[f'input_{name}'].values for name in simulator.input_names}
    #
    # window_steps = max(2, int(round(window_s / dt)))
    # measurement_noise_vars = {k: v ** 2 for k, v in MEASUREMENT_NOISE_STDS.items()}
    #
    # if verbose:
    #     n_windows = max(1, len(df) - window_steps + 1)
    #     n_states = len(simulator.state_names)
    #     est_steps = n_windows * window_steps * (2 * n_states + 1)
    #     est_seconds = est_steps * EMPIRICAL_SIMULATE_STEP_SECONDS
    #     print(f'  computing BOUNDS observability: {n_windows} windows x {window_steps} steps x '
    #          f'{2 * n_states + 1} sims/window (~{est_steps:,} total simulate-steps) -- rough '
    #          f'estimate ~{_format_duration(est_seconds)}, varies by machine')
    #
    # start_time = time.time()
    # with _Heartbeat('computing observability') if verbose else _NullContext():
    #     SEOM = SlidingEmpiricalObservabilityMatrix(simulator, t_sim, x_sim, u_sim, w=window_steps, eps=1e-4)
    # if verbose:
    #     print(f'  observability matrix built in {_format_duration(time.time() - start_time)}, '
    #          f'computing Fisher information for each sensor set...')
    #
    # o_states = ['z', 'v_x', 'v_y', 'psi']
    # o_time_steps = np.arange(0, window_steps, step=1)
    #
    # ev_by_sensor_set = {}
    # for label, sensors in (('impaired', IMPAIRED_SENSORS), ('full', FULL_SENSORS)):
    #     R = {k: measurement_noise_vars[k] for k in sensors}
    #     SFO = SlidingFisherObservability(SEOM.O_df_sliding, time=SEOM.t_sim, lam=1e-8, R=R,
    #                                      states=o_states, sensors=sensors, time_steps=o_time_steps, w=None)
    #     EV = SFO.get_minimum_error_variance()
    #     try:
    #         EV = EV.bfill().ffill()
    #     except AttributeError:
    #         EV = EV.fillna(method='bfill').fillna(method='ffill')
    #     ev_by_sensor_set[label] = EV
    #
    # if verbose:
    #     print(f'  observability done in {_format_duration(time.time() - start_time)} total')
    #
    # return {'t_sim': t_sim, 'x_sim': x_sim, 'window_steps': window_steps,
    #         'ev_impaired': ev_by_sensor_set['impaired'], 'ev_full': ev_by_sensor_set['full']}


# ---------------------------------------------------------------------------
# ANN altitude estimator (H_i) inference
# ---------------------------------------------------------------------------

def run_ann_estimate(df, models_dir=MODELS_DIR, model_name=DEFAULT_MODEL_NAME, verbose=True):
    model_path = models_dir / model_name
    config_path = model_path.parent / f'{model_path.name}.config.json'
    if not config_path.exists():
        if verbose:
            print(f'  no trained model at {model_path} -- skipping AI-KF estimate panel')
        return None

    missing = [c for c in INPUT_COLUMNS + [OUTPUT_COLUMN] if c not in df.columns]
    if missing:
        if verbose:
            print(f'  trajectory missing columns {missing} -- skipping AI-KF estimate panel')
        return None

    if verbose:
        print(f'  running ANN estimator (H_i) inference from {model_path}...')
    start_time = time.time()

    model, config = load_model_complete(model_path)
    window_steps = config['window_steps']
    if len(df) < window_steps:
        if verbose:
            print(f'  trajectory has {len(df)} rows, shorter than the model\'s '
                 f'{window_steps}-step window -- skipping AI-KF estimate panel')
        return None

    X, y_true = build_windowed_dataset(df, INPUT_COLUMNS, OUTPUT_COLUMN, window_steps)
    y_pred = model.predict(X, verbose=0).flatten()
    time_aligned = df['time'].values[window_steps - 1:]

    if verbose:
        print(f'  ANN inference done in {_format_duration(time.time() - start_time)} '
             f'({len(y_pred)} windows)')

    return {'time': time_aligned, 'y_true': y_true, 'y_pred': y_pred,
            'trained_on': config.get('train_files', [])}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _placeholder(ax, message):
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes,
            fontsize=8, wrap=True, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def build_figure(filepath, metadata, seed, output_path, window_s, models_dir, model_name,
                 skip_observability=False, verbose=True):
    if verbose:
        print(f'  loading {filepath}...')
    df = pd.read_csv(filepath)
    dt = infer_dt(df)
    t = df['time'].values
    motif = metadata.get('motif', '?')
    if verbose:
        print(f'  {len(df)} rows, dt={dt:.4f}s, motif={motif}')

    setpoint = reconstruct_setpoint(motif, seed, t)
    if verbose:
        print(f'  commanded setpoint {"reconstructed" if setpoint is not None else "not available"}')
    accel_segments = high_accel_segments(df)

    # OBSERVABILITY DISABLED: compute_observability() currently always returns None (see its
    # definition above). Kept as a real call (rather than deleted) so re-enabling is a one-line
    # change there instead of also having to restore a call site.
    observability = compute_observability(df, dt, window_s, verbose=verbose)

    ann_result = run_ann_estimate(df, models_dir, model_name, verbose=verbose)

    if verbose:
        print('  building figure...')

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle(f'{filepath.name}  (motif={motif}, dt={dt:.4f}s)', fontsize=10)

    # OBSERVABILITY DISABLED: the x-y-colored-by-observability panel and the z-error-variance-
    # over-time panel (paper Figure 2c / crazyfly_simulation_3d.ipynb style) used to go here, in
    # a 3x2 grid. See module docstring for how to restore them.

    # (0,0) x-y planar position: commanded (open-loop kinematic integration of the setpoint) vs
    # actual (MPC-tracked)
    ax0 = ax[0]
    if setpoint is not None and 'state_x' in df.columns and 'state_y' in df.columns:
        x_actual = df['state_x'].values
        y_actual = df['state_y'].values
        x_cmd, y_cmd = reconstruct_setpoint_xy(setpoint, t)
        ax0.plot(x_actual, y_actual, color='firebrick', label='actual', linewidth=1.5)
        ax0.plot(x_cmd, y_cmd, color='gray', linestyle='--', label='commanded (open-loop)')
        ax0.set_xlabel('x [m]')
        ax0.set_ylabel('y [m]')
        ax0.set_aspect('equal')
        ax0.legend(fontsize=6)
        ax0.set_title('x-y position: commanded vs actual', fontsize=9)
    else:
        _placeholder(ax0, 'commanded/actual x-y position not available\n(no seed / unknown '
                          'motif, or missing state_x/state_y columns)')
        ax0.set_title('x-y position: commanded vs actual', fontsize=9)

    # (0,1) AI-KF data-driven estimator (H_i): true z vs ANN raw estimate
    ax1 = ax[1]
    if ann_result is not None:
        ax1.plot(ann_result['time'], ann_result['y_true'], color='black', label='true z', linewidth=1.5)
        ax1.plot(ann_result['time'], ann_result['y_pred'], color='tab:orange', label='ANN estimate (H_i)',
                linewidth=1.2, alpha=0.9)

        if observability is not None:
            obs_segments = high_observability_segments(
                ann_result['time'], observability['ev_impaired']['time'].values,
                observability['ev_impaired']['z'].values)
            _shade_segments(ax1, obs_segments, color='tab:green', label='high observability')
        else:
            _shade_segments(ax1, high_accel_segments(df), color='tab:green',
                          label='high horiz. accel (proxy)')

        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('z [m]')
        ax1.legend(fontsize=6)
        ax1.set_title('AI-KF estimator (H_i): true vs estimated z', fontsize=9)
    else:
        _placeholder(ax1, f'no trained model found at\n{models_dir / model_name}\n'
                          'or trajectory missing required columns')
        ax1.set_title('AI-KF estimator (H_i): true vs estimated z', fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    if verbose:
        print(f'  saved: {output_path}')
    return output_path


# ---------------------------------------------------------------------------
# CLI / batch driver
# ---------------------------------------------------------------------------


def analyze_one(directory, row, output_dir, window_s, models_dir, model_name, overwrite,
                skip_observability, position=None):
    filename = row['filename']
    filepath = directory / filename
    output_path = output_dir / f'{_strip_trajectory_extension(filename)}_analysis.png'
    motif = json.loads(row['params_json']).get('motif', '?')

    banner = f'[{position}] ' if position else ''
    print(f'{banner}=== analyzing: {filename}  (index={row["index"]}, motif={motif}) ===')

    if output_path.exists() and not overwrite:
        print(f'  skip (output exists): {output_path.name}')
        return output_path

    metadata = json.loads(row['params_json'])
    seed = row.get('seed', None)
    build_figure(filepath, metadata, seed, output_path, window_s, models_dir, model_name,
                skip_observability=skip_observability)
    return output_path


def main(trajectory_arg, directory_name, output_dir, window_s, skip_observability,
        models_dir, model_name, overwrite):
    directory = REPO_ROOT / directory_name
    manifest = load_manifest(directory)

    if str(trajectory_arg).lower() == 'all':
        rows = [row for _, row in manifest.iterrows()]
    else:
        index = int(trajectory_arg)
        matches = manifest[manifest['index'] == index]
        if matches.empty:
            raise ValueError(f'no trajectory with index={index} in {directory}/manifest.csv '
                             f'(valid range: {manifest["index"].min()}-{manifest["index"].max()})')
        rows = [matches.iloc[0]]

    start_time = time.time()
    for i, row in enumerate(rows):
        analyze_one(directory, row, output_dir, window_s, models_dir, model_name, overwrite,
                   skip_observability, position=f'{i + 1}/{len(rows)}')
        elapsed = time.time() - start_time
        avg = elapsed / (i + 1)
        eta = avg * (len(rows) - i - 1)
        print(f'[{i + 1}/{len(rows)}] done. elapsed {_format_duration(elapsed)}  '
             f'ETA {_format_duration(eta)}\n')

    print(f'done. {len(rows)} analysis figure(s) in: {output_dir}')


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--trajectory', type=str, default='0',
                        help='manifest index (int) or "all" (default: 0)')
    parser.add_argument('--directory', type=str, choices=DIRECTORY_CHOICES,
                        default=SIMULATED_TRAJECTORIES_DIRNAME,
                        help=f'trajectory folder under the repo root (default: '
                             f'{SIMULATED_TRAJECTORIES_DIRNAME} -- needed for observability)')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--window-s', type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument('--skip-observability', action='store_true',
                        help='skip the slow BOUNDS analysis; flight-path + ANN-estimate panels only')
    parser.add_argument('--models-dir', type=str, default=str(MODELS_DIR))
    parser.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--overwrite', action='store_true',
                        help='regenerate even if the output PNG already exists')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(trajectory_arg=args.trajectory, directory_name=args.directory,
        output_dir=Path(args.output_dir), window_s=args.window_s,
        skip_observability=args.skip_observability, models_dir=Path(args.models_dir),
        model_name=args.model_name, overwrite=args.overwrite)
