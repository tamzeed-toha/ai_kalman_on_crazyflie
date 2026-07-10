"""Generate a library of simulated Crazyflie flight trajectories, spanning a variety of
active-sensing motifs, for training the ANN state/observability estimators described in
elaborate_plan.md Phase 2.

Each trajectory is produced by defining a body-level velocity/heading setpoint profile for one
motif, handing it to model/drone_simulator.py's DroneSimulator, and letting pybounds/do_mpc's MPC
controller fly the setpoint. The resulting state/input/measurement time-series is saved to
simulated_trajectories/ as one compressed CSV per trajectory, indexed by a top-level manifest.csv.

Motifs implemented (see MOTIFS below):
    - sinusoidal   : single-frequency forward-speed oscillation, frequency/amplitude randomized
    - straight     : constant-velocity straight flight (the "unobservable" null case)
    - accel_decel  : a horizontal acceleration/deceleration pulse (the motif the reference paper,
                     references/summaries/bens_epic_paper.md, found necessary for altitude
                     observability from optic flow + accelerometer alone)
    - turn         : a heading-change ramp, with or without concurrent forward speed
    - circle       : constant forward speed + constant turn rate -> circular flight path
    - casting      : side-to-side lateral sweeping (insect-inspired "casting" search behavior)
    - random       : a sum-of-sines random signal (same method the paper's own Methods section
                     used to generate its bulk altitude-estimator training set)

Trajectory length and sample rate:
    - Length = 11.0 s, matching the altitude-estimator trajectory length used in Cellini et al.
      2025 ("Each trajectory was 11 s (111 discrete points)").
    - dt = 0.01 s (100 Hz), matching the `imu` and `flow` blocks in references/rates.tsv -- the
      two data streams explicitly marked as the ANN's inputs -- so simulated trajectories are
      sampled at the same rate real onboard logs will be.

Output format: each trajectory is a plain numeric CSV (time + state_* + input_* + meas_* columns,
gzip-compressed), with all motif/parameter metadata kept in manifest.csv instead of duplicated in
every file. references/generate_training_data_utility.py's load_trajectory_data() expects
per-trajectory HDF5 (pandas .to_hdf); CSV was used here instead since the `tables` package
(pytables, required for to_hdf/read_hdf) isn't installed in this project's .venv. If `tables` is
installed later, swapping `to_csv(..., compression='gzip')` for `to_hdf(...)` is a one-line change.

Usage:
    python3 utils/trajectory_generator.py
        Generates the full default set (1000 trajectories) into simulated_trajectories/.

    python3 utils/trajectory_generator.py --n-trajectories 10 --length 2.0
        Generates a small, fast batch for testing.

Already-generated files are skipped on re-run, so an interrupted batch can be safely resumed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so `utils.` / `model.` resolve regardless of cwd

from utils.pybounds_compat import patch_pybounds_simulator_time_conversion

patch_pybounds_simulator_time_conversion()  # must run before any DroneSimulator is constructed

from model.drone_simulator import DroneSimulator


# Trajectory length from the reference paper's altitude-estimator dataset (see module docstring).
TRAJECTORY_LENGTH_S = 11.0

# Sample rate from references/rates.tsv (imu & flow blocks, both marked as ANN inputs, 100 Hz).
DT = 0.01

N_TRAJECTORIES = 1000
OUTPUT_DIR = REPO_ROOT / 'simulated_trajectories'

# Indoor altitude-hold envelope for a Crazyflie test flight [m].
Z_RANGE = (0.3, 2.0)

MPC_HORIZON = 10
BASE_SEED = 0


# ---------------------------------------------------------------------------
# Motif definitions
#
# Each motif function takes the trajectory time vector `t` (seconds) and a seeded
# numpy Generator `rng`, and returns (v_x, v_y, psi, params) where v_x/v_y/psi are arrays the
# same length as `t` and `params` is a dict of the randomized values used (saved to the manifest).
# ---------------------------------------------------------------------------

def sinusoidal_motion(t, rng):
    """Single-frequency forward-speed oscillation."""
    freq_hz = rng.uniform(0.05, 0.5)
    amplitude_mps = rng.uniform(0.1, 1.0)
    phase_rad = rng.uniform(-np.pi, np.pi)

    v_x = amplitude_mps * np.sin(2 * np.pi * freq_hz * t + phase_rad)
    v_y = np.zeros_like(t)
    psi = np.zeros_like(t)

    params = {'freq_hz': freq_hz, 'amplitude_mps': amplitude_mps, 'phase_rad': phase_rad}
    return v_x, v_y, psi, params


def straight_flight(t, rng):
    """Constant-velocity straight flight -- the 'unobservable' null case for altitude."""
    speed_mps = rng.uniform(0.1, 1.0)

    v_x = speed_mps * np.ones_like(t)
    v_y = np.zeros_like(t)
    psi = np.zeros_like(t)

    params = {'speed_mps': speed_mps}
    return v_x, v_y, psi, params


def accel_decel_pulse(t, rng):
    """A horizontal acceleration/deceleration pulse: ramp up, hold, ramp down."""
    T = t[-1]
    peak_speed_mps = rng.uniform(0.3, 1.5)
    t0 = rng.uniform(0.1 * T, 0.3 * T)
    accel_dur = rng.uniform(0.1 * T, 0.25 * T)
    hold_dur = rng.uniform(0.0, 0.2 * T)
    decel_dur = rng.uniform(0.1 * T, 0.25 * T)

    t1 = t0 + accel_dur
    t2 = t1 + hold_dur
    t3 = min(t2 + decel_dur, T)

    v_x = np.interp(t, [0.0, t0, t1, t2, t3, T], [0.0, 0.0, peak_speed_mps, peak_speed_mps, 0.0, 0.0])
    v_y = np.zeros_like(t)
    psi = np.zeros_like(t)

    params = {'peak_speed_mps': peak_speed_mps, 't0_s': t0, 'accel_dur_s': accel_dur,
              'hold_dur_s': hold_dur, 'decel_dur_s': decel_dur}
    return v_x, v_y, psi, params


def turn_motif(t, rng):
    """A heading-change ramp, with a randomly chosen concurrent forward speed (which may be
    zero, matching the reference paper's distinction between an 'offset turn' at zero forward
    speed and 'turning while moving')."""
    T = t[-1]
    base_speed_mps = 0.0 if rng.uniform() < 0.5 else rng.uniform(0.1, 0.5)
    turn_angle_rad = rng.uniform(-np.pi, np.pi)
    t0 = rng.uniform(0.2 * T, 0.4 * T)
    dur = rng.uniform(0.15 * T, 0.35 * T)
    t1 = min(t0 + dur, T)

    psi = np.interp(t, [0.0, t0, t1, T], [0.0, 0.0, turn_angle_rad, turn_angle_rad])
    v_x = base_speed_mps * np.ones_like(t)
    v_y = np.zeros_like(t)

    params = {'base_speed_mps': base_speed_mps, 'turn_angle_rad': turn_angle_rad,
              't0_s': t0, 'turn_dur_s': dur}
    return v_x, v_y, psi, params


def circle_motif(t, rng):
    """Constant forward speed + constant turn rate -> a circular flight path."""
    speed_mps = rng.uniform(0.2, 1.0)
    radius_m = rng.uniform(0.3, 2.0)
    direction = rng.choice([-1.0, 1.0])
    turn_rate_rad_s = direction * speed_mps / radius_m

    v_x = speed_mps * np.ones_like(t)
    v_y = np.zeros_like(t)
    psi = turn_rate_rad_s * t

    params = {'speed_mps': speed_mps, 'radius_m': radius_m,
              'turn_rate_rad_s': turn_rate_rad_s, 'direction': float(direction)}
    return v_x, v_y, psi, params


def casting_motif(t, rng):
    """Side-to-side lateral sweeping, inspired by insect 'casting' search behavior: a lateral
    (v_y) oscillation, optionally combined with slow forward progress."""
    period_s = rng.uniform(2.0, 6.0)
    lateral_amplitude_mps = rng.uniform(0.2, 1.0)
    forward_speed_mps = rng.uniform(0.0, 0.3)

    v_y = lateral_amplitude_mps * np.sin(2 * np.pi * t / period_s)
    v_x = forward_speed_mps * np.ones_like(t)
    psi = np.zeros_like(t)

    params = {'period_s': period_s, 'lateral_amplitude_mps': lateral_amplitude_mps,
              'forward_speed_mps': forward_speed_mps}
    return v_x, v_y, psi, params


def random_motion(t, rng, n_components=4):
    """A sum-of-sines random signal, following the random-trajectory-generation method in
    Cellini et al. 2025's Methods (varying forward velocity via a sum-of-sines signal with
    several randomized frequency/amplitude/phase components)."""

    def sum_of_sines(amp_range, freq_range=(0.1, 0.9)):
        signal = np.zeros_like(t)
        for _ in range(n_components):
            freq_hz = rng.uniform(*freq_range)
            amp = rng.uniform(*amp_range) / n_components
            phase_rad = rng.uniform(-np.pi / 2, np.pi / 2)
            signal = signal + amp * np.sin(2 * np.pi * freq_hz * t + phase_rad)
        return signal

    v_x = sum_of_sines((0.1, 1.0)) + rng.uniform(-0.2, 0.2)
    v_y = sum_of_sines((0.05, 0.5))
    psi = sum_of_sines((0.2, 1.0), freq_range=(0.05, 0.3))

    params = {'n_components': n_components}
    return v_x, v_y, psi, params


MOTIFS = {
    'sinusoidal': sinusoidal_motion,
    'straight': straight_flight,
    'accel_decel': accel_decel_pulse,
    'turn': turn_motif,
    'circle': circle_motif,
    'casting': casting_motif,
    'random': random_motion,
}


# ---------------------------------------------------------------------------
# Generation / saving
# ---------------------------------------------------------------------------

def distribute_counts(total, n_bins):
    """Split `total` as evenly as possible across `n_bins` bins, summing exactly to `total`."""
    base = total // n_bins
    remainder = total % n_bins
    return [base + 1 if i < remainder else base for i in range(n_bins)]


def _format_duration(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f'{hours}h{minutes:02d}m'
    if minutes:
        return f'{minutes}m{secs:02d}s'
    return f'{secs}s'


def generate_single_trajectory(motif_name, t, rng, z_range=Z_RANGE, dt=DT, mpc_horizon=MPC_HORIZON):
    """Simulate one trajectory for the given motif and return (dataframe, metadata)."""
    v_x, v_y, psi, motif_params = MOTIFS[motif_name](t, rng)

    z_level_m = rng.uniform(*z_range)
    z = z_level_m * np.ones_like(t)
    w = np.zeros_like(t)      # no ambient wind -- indoor flight
    zeta = np.zeros_like(t)

    simulator = DroneSimulator(frame='body_level', moving_frame='on', dt=dt, mpc_horizon=mpc_horizon)
    simulator.update_setpoint(z=z, v_x=v_x, v_y=v_y, psi=psi, w=w, zeta=zeta)
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=None, mpc=True, return_full_output=True)

    columns = {'time': np.asarray(t_sim).reshape(-1)}
    for name, values in x_sim.items():
        columns[f'state_{name}'] = np.asarray(values).reshape(-1)
    for name, values in u_sim.items():
        columns[f'input_{name}'] = np.asarray(values).reshape(-1)

    # y_sim also echoes every state & input; only keep the derived measurements that aren't
    # already saved above (v_x_dot, v_y_dot, r_x, r_y, r, g, beta, a_x, a_y, a, gamma, w_x, w_y).
    already_saved = set(x_sim.keys()) | set(u_sim.keys())
    for name, values in y_sim.items():
        if name not in already_saved:
            columns[f'meas_{name}'] = np.asarray(values).reshape(-1)

    df = pd.DataFrame(columns)

    metadata = {
        'motif': motif_name,
        'z_level_m': z_level_m,
        'dt': dt,
        'length_s': float(t[-1]),
        'mpc_horizon': mpc_horizon,
        **motif_params,
    }
    return df, metadata


def save_trajectory(df, filepath):
    """Write atomically: if interrupted mid-write, only the .tmp file is left behind, never a
    truncated/corrupt file at `filepath` -- so resume logic can trust filepath.exists()."""
    tmp_path = filepath.parent / (filepath.name + '.tmp')
    df.to_csv(tmp_path, index=False, compression='gzip')
    tmp_path.replace(filepath)


def _is_complete_trajectory_file(filepath):
    """A trajectory file only counts as 'already done' if it's readable and has at least one
    row -- guards against a previous run being killed mid-write and leaving a truncated file
    that would otherwise be silently treated as finished forever."""
    if not filepath.exists():
        return False
    try:
        return len(pd.read_csv(filepath, nrows=1)) >= 1
    except Exception:
        return False


def _append_manifest_row(manifest_path, row):
    """Append one row to the manifest immediately, so an interrupted run keeps metadata for
    every trajectory that did finish, instead of losing it all until the batch completes."""
    row_df = pd.DataFrame([row])
    write_header = not manifest_path.exists()
    row_df.to_csv(manifest_path, mode='a', header=write_header, index=False)


def generate_dataset(n_trajectories=N_TRAJECTORIES, output_dir=OUTPUT_DIR, dt=DT,
                      length_s=TRAJECTORY_LENGTH_S, seed=BASE_SEED, motif_names=None):
    """Generate `n_trajectories` total, split evenly across `motif_names` (default: all motifs),
    saving each to `output_dir` and recording every trajectory in output_dir/manifest.csv as it
    completes. Skips trajectories whose output file already exists *and* is a valid, non-empty
    file, so an interrupted run can be resumed without losing progress or keeping corrupt files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if motif_names is None:
        motif_names = list(MOTIFS.keys())

    counts = distribute_counts(n_trajectories, len(motif_names))
    n_steps = int(round(length_s / dt)) + 1
    t = np.arange(n_steps) * dt

    manifest_path = output_dir / 'manifest.csv'
    global_index = 0

    start_time = time.time()
    n_generated = 0  # excludes skipped (already-existing) trajectories

    for motif_name, count in zip(motif_names, counts):
        for local_index in range(count):
            filename = f'{motif_name}_{local_index:04d}.csv.gz'
            filepath = output_dir / filename

            if _is_complete_trajectory_file(filepath):
                print(f'[{global_index + 1}/{n_trajectories}] skip (exists): {filename}')
                global_index += 1
                continue

            seed_i = seed + global_index
            rng = np.random.default_rng(seed_i)

            df, metadata = generate_single_trajectory(motif_name, t, rng, dt=dt)
            save_trajectory(df, filepath)

            row = {'filename': filename, 'index': global_index, 'seed': seed_i,
                   'params_json': json.dumps(metadata)}
            _append_manifest_row(manifest_path, row)

            n_generated += 1
            elapsed = time.time() - start_time
            avg_per_traj = elapsed / n_generated
            n_remaining = n_trajectories - (global_index + 1)
            eta = avg_per_traj * n_remaining

            print(f'[{global_index + 1}/{n_trajectories}] generated: {filename} '
                  f'| elapsed {_format_duration(elapsed)} '
                  f'| avg {avg_per_traj:.1f}s/traj '
                  f'| ETA {_format_duration(eta)}')

            global_index += 1

    total_elapsed = time.time() - start_time
    print(f'Done. Generated {n_generated} new trajectories in {_format_duration(total_elapsed)}. '
          f'Trajectories saved under: {output_dir}')


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n-trajectories', type=int, default=N_TRAJECTORIES)
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--dt', type=float, default=DT)
    parser.add_argument('--length', type=float, default=TRAJECTORY_LENGTH_S)
    parser.add_argument('--seed', type=int, default=BASE_SEED)
    parser.add_argument('--motifs', type=str, default=None,
                         help='comma-separated subset of motif names (default: all)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    motif_names = args.motifs.split(',') if args.motifs else None

    generate_dataset(n_trajectories=args.n_trajectories, output_dir=args.output_dir, dt=args.dt,
                      length_s=args.length, seed=args.seed, motif_names=motif_names)
