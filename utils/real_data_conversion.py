"""Convert real Crazyflie flight-log trajectories (real_trajectories/) into the same
folder/manifest convention utils/trajectory_generator.py produces for simulated trajectories, so
train.py can train on real and simulated data seamlessly (same file layout, same manifest schema).

Scope: this is a pure FORMAT/FOLDER conversion, not a physics/schema fix. The files in
real_trajectories/ (produced upstream by data_collection/export_to_training_format.py) already
use column names matching train.py's exact requirements (time, meas_r_x, meas_r_y,
meas_v_x_dot, meas_v_y_dot, state_z) -- this module does not re-derive, rescale, or correct any
of those values.

IMPORTANT CAVEAT -- read crazyflie_data_adaptation_brief.md before trusting anything trained on
the converted output. That brief flags several unresolved physics questions about how the real
IMU/flow readings were turned into meas_v_x_dot/meas_v_y_dot/meas_r_x/meas_r_y -- most notably,
whether "meas_v_x_dot"/"meas_v_y_dot" represent kinematic acceleration (what the simulated model
in model/drone_simulator.py assumes) or raw specific-force accelerometer readings (what a real
MEMS IMU actually measures), and whether the flow-deck pixel-to-physical-flow conversion used the
firmware's actual calibration. Converting the file format does not resolve any of that -- this
module intentionally only reorganizes files, it does not touch the numbers inside them.

What this module DOES do, for every trajectory listed in real_trajectories/manifest.csv:
    1. Validates it has the columns train.py needs
       (time, meas_r_x, meas_r_y, meas_v_x_dot, meas_v_y_dot, state_z) -- skips (with a message)
       if not.
    2. Copies it to converted_real_trajectories/<name>.csv.gz (gzip, matching
       simulated_trajectories/'s file convention).
    3. Writes converted_real_trajectories/manifest.csv in the same schema as
       simulated_trajectories/manifest.csv (filename, index, seed, params_json). Each row's
       params_json is the original real-trajectory metadata (motif, motif_id, rep_index,
       session_id, ...) enriched with source='real', dt, length_s, n_rows. seed is left null --
       these are real flights, not seeded simulations.

Writes are atomic (temp file + rename) and the manifest is appended incrementally per file, and
the "already converted" check re-validates the existing file (not just its presence) -- the same
interrupt-safety fixes applied to utils/trajectory_generator.py after an earlier Ctrl+C left a
corrupt file and a stale manifest, applied here from the start.

Usage:
    python3 utils/real_data_conversion.py
        Convert everything in real_trajectories/ into converted_real_trajectories/.
    python3 utils/real_data_conversion.py --real-trajectories-dir other_dir --output-dir other_out
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_TRAJECTORIES_DIR = REPO_ROOT / 'real_trajectories'
CONVERTED_TRAJECTORIES_DIR = REPO_ROOT / 'converted_real_trajectories'

# Must match train.py's INPUT_COLUMNS + OUTPUT_COLUMN (+ time).
REQUIRED_COLUMNS = ['time', 'meas_r_x', 'meas_r_y', 'meas_v_x_dot', 'meas_v_y_dot', 'state_z']


def load_real_manifest(real_trajectories_dir):
    manifest_path = real_trajectories_dir / 'manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(f'{manifest_path} not found.')
    return pd.read_csv(manifest_path)


def _is_complete_trajectory_file(filepath):
    """A converted file only counts as 'already done' if it's readable and has at least one
    row -- guards against a previous run being killed mid-write and leaving a truncated file
    that would otherwise be silently treated as finished forever."""
    if not filepath.exists():
        return False
    try:
        return len(pd.read_csv(filepath, nrows=1)) >= 1
    except Exception:
        return False


def save_converted(df, filepath):
    """Write atomically: if interrupted mid-write, only the .tmp file is left behind, never a
    truncated/corrupt file at `filepath`."""
    tmp_path = filepath.parent / (filepath.name + '.tmp')
    df.to_csv(tmp_path, index=False, compression='gzip')
    tmp_path.replace(filepath)


def append_manifest_row(manifest_path, row):
    """Append one row to the manifest immediately, so an interrupted run keeps metadata for
    every trajectory that did finish, instead of losing it all until the batch completes."""
    row_df = pd.DataFrame([row])
    write_header = not manifest_path.exists()
    row_df.to_csv(manifest_path, mode='a', header=write_header, index=False)


def convert_one(filename, source_params, real_trajectories_dir, output_dir, index):
    src_path = real_trajectories_dir / filename
    if not src_path.exists():
        print(f'[{index}] skip (missing source file): {filename}')
        return None

    dst_name = f'{Path(filename).stem}.csv.gz'
    dst_path = output_dir / dst_name

    if _is_complete_trajectory_file(dst_path):
        print(f'[{index}] skip (already converted): {dst_name}')
        return None

    df = pd.read_csv(src_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f'[{index}] skip (missing columns {missing}): {filename}')
        return None

    save_converted(df, dst_path)

    dt = float(np.median(np.diff(df['time'].values))) if len(df) > 1 else None
    params = {
        **source_params,
        'source': 'real',
        'source_file': filename,
        'dt': dt,
        'length_s': float(df['time'].iloc[-1]) if len(df) else 0.0,
        'n_rows': len(df),
    }

    print(f'[{index}] converted: {filename} -> {dst_name}')
    return dst_name, params


def main(real_trajectories_dir=REAL_TRAJECTORIES_DIR, output_dir=CONVERTED_TRAJECTORIES_DIR):
    real_trajectories_dir = Path(real_trajectories_dir)
    output_dir = Path(output_dir)

    real_manifest = load_real_manifest(real_trajectories_dir)
    print(f'{len(real_manifest)} real trajectories listed in {real_trajectories_dir / "manifest.csv"}')

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.csv'

    n_converted = 0
    for i, row in real_manifest.iterrows():
        filename = row['filename']
        source_params = json.loads(row['params_json'])

        result = convert_one(filename, source_params, real_trajectories_dir, output_dir, i)
        if result is None:
            continue
        dst_name, params = result

        manifest_row = {'filename': dst_name, 'index': i, 'seed': None,
                        'params_json': json.dumps(params)}
        append_manifest_row(manifest_path, manifest_row)
        n_converted += 1

    print(f'done. converted {n_converted} new trajectory(ies) into: {output_dir}')


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--real-trajectories-dir', type=str, default=str(REAL_TRAJECTORIES_DIR))
    parser.add_argument('--output-dir', type=str, default=str(CONVERTED_TRAJECTORIES_DIR))
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(real_trajectories_dir=args.real_trajectories_dir, output_dir=args.output_dir)
