"""Visualize a single simulated trajectory produced by utils/trajectory_generator.py.

Loads one `simulated_trajectories/{motif}_{index}.csv.gz` file (time + state_* + input_* +
meas_* columns) and plots everything in it: the x-y flight path (colored by time), and grids of
every state, input, and measurement time-series. Column groups are discovered generically from
the `state_`/`input_`/`meas_` prefixes trajectory_generator.py writes, so this works for any
trajectory file following that convention, not just a hardcoded set of crazyflie variables.

Usage (as a script):
    python3 utils/trajectory_visualizer.py simulated_trajectories/accel_decel_0000.csv.gz
    python3 utils/trajectory_visualizer.py simulated_trajectories/accel_decel_0000.csv.gz --save-dir figures/

Usage (as a class):
    from utils.trajectory_visualizer import TrajectoryVisualizer
    viz = TrajectoryVisualizer('simulated_trajectories/accel_decel_0000.csv.gz')
    viz.summary()
    viz.plot_all()
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pybounds import colorline as _pybounds_colorline
except ImportError:  # pybounds is optional here -- fall back to a plain scatter
    _pybounds_colorline = None


class TrajectoryVisualizer:
    """Loads one trajectory CSV(.gz) file and plots its full state/input/measurement content."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.df = pd.read_csv(self.filepath)
        self.metadata = self._load_metadata()

        self.time = self.df['time'].values
        self.state_cols = [c for c in self.df.columns if c.startswith('state_')]
        self.input_cols = [c for c in self.df.columns if c.startswith('input_')]
        self.meas_cols = [c for c in self.df.columns if c.startswith('meas_')]

    def _load_metadata(self):
        """Look up this trajectory's motif/params from a sibling manifest.csv, if present."""
        manifest_path = self.filepath.parent / 'manifest.csv'
        if not manifest_path.exists():
            return None

        manifest = pd.read_csv(manifest_path)
        match = manifest[manifest['filename'] == self.filepath.name]
        if match.empty:
            return None

        return json.loads(match.iloc[0]['params_json'])

    def summary(self):
        """Print a quick description of the loaded trajectory."""
        print(f'file: {self.filepath}')
        print(f'rows: {len(self.df)}  duration: {self.time[-1] - self.time[0]:.2f}s')
        print(f'states ({len(self.state_cols)}): {[c.removeprefix("state_") for c in self.state_cols]}')
        print(f'inputs ({len(self.input_cols)}): {[c.removeprefix("input_") for c in self.input_cols]}')
        print(f'measurements ({len(self.meas_cols)}): {[c.removeprefix("meas_") for c in self.meas_cols]}')
        if self.metadata is not None:
            print(f'metadata: {self.metadata}')
        else:
            print('metadata: none found (no sibling manifest.csv, or filename not listed in it)')

    def _title_suffix(self):
        if self.metadata is None:
            return self.filepath.name
        motif = self.metadata.get('motif', '?')
        return f'{self.filepath.name}  (motif={motif})'

    def plot_trajectory_xy(self, ax=None, cmap='viridis'):
        """Plot the x-y flight path, colored by time. No-op (with a message) if the trajectory
        doesn't have state_x/state_y columns."""
        if 'state_x' not in self.df.columns or 'state_y' not in self.df.columns:
            print('plot_trajectory_xy: no state_x/state_y columns in this trajectory, skipping.')
            return None

        x = self.df['state_x'].values
        y = self.df['state_y'].values

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
        else:
            fig = ax.figure

        if _pybounds_colorline is not None:
            norm = plt.Normalize(self.time.min(), self.time.max())
            _pybounds_colorline(x, y, self.time, ax=ax, cmap=cmap, norm=norm)
            ax.set_xlim(x.min() - 0.05 * (np.ptp(x) + 1e-9), x.max() + 0.05 * (np.ptp(x) + 1e-9))
            ax.set_ylim(y.min() - 0.05 * (np.ptp(y) + 1e-9), y.max() + 0.05 * (np.ptp(y) + 1e-9))
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            cbar.set_label('time (s)', fontsize=8)
        else:
            sc = ax.scatter(x, y, c=self.time, cmap=cmap, s=4)
            fig.colorbar(sc, ax=ax, label='time (s)')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal')
        ax.set_title(f'x-y trajectory\n{self._title_suffix()}', fontsize=9)
        fig.tight_layout()
        return fig

    def _plot_grid(self, cols, prefix, ncols=3, figsize_per_axis=(3.2, 2.2)):
        """Plot each column in `cols` against time, arranged in a grid."""
        if not cols:
            print(f'no columns with prefix "{prefix}" found, skipping.')
            return None

        n = len(cols)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, dpi=150,
                                  figsize=(figsize_per_axis[0] * ncols, figsize_per_axis[1] * nrows),
                                  squeeze=False)
        axes_flat = axes.flatten()

        for i, col in enumerate(cols):
            ax = axes_flat[i]
            ax.plot(self.time, self.df[col].values, linewidth=1.2)
            ax.set_title(col.removeprefix(prefix), fontsize=8)
            ax.tick_params(axis='both', labelsize=6)

        for ax in axes_flat[n:]:
            ax.axis('off')

        for ax in axes[-1, :]:
            ax.set_xlabel('time (s)', fontsize=7)

        fig.suptitle(f'{prefix.rstrip("_")} variables\n{self._title_suffix()}', fontsize=9)
        fig.tight_layout()
        return fig

    def plot_states(self, ncols=3):
        return self._plot_grid(self.state_cols, 'state_', ncols=ncols)

    def plot_inputs(self, ncols=3):
        return self._plot_grid(self.input_cols, 'input_', ncols=ncols)

    def plot_measurements(self, ncols=3):
        return self._plot_grid(self.meas_cols, 'meas_', ncols=ncols)

    def plot_all(self, show=True, save_dir=None):
        """Generate every plot (x-y trajectory, states, inputs, measurements). Returns the list
        of created figures (some entries may be None if a column group/x-y wasn't available)."""
        figures = {
            'trajectory_xy': self.plot_trajectory_xy(),
            'states': self.plot_states(),
            'inputs': self.plot_inputs(),
            'measurements': self.plot_measurements(),
        }

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = self.filepath.name.split('.')[0]
            for name, fig in figures.items():
                if fig is not None:
                    out_path = save_dir / f'{stem}_{name}.png'
                    fig.savefig(out_path, bbox_inches='tight')
                    print(f'saved: {out_path}')

        if show:
            plt.show()

        return [fig for fig in figures.values() if fig is not None]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('filepath', type=str, help='path to a trajectory .csv.gz file')
    parser.add_argument('--save-dir', type=str, default=None,
                         help='if set, save each figure as a PNG in this directory')
    parser.add_argument('--no-show', action='store_true', help='do not open interactive plot windows')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    viz = TrajectoryVisualizer(args.filepath)
    viz.summary()
    viz.plot_all(show=not args.no_show, save_dir=args.save_dir)
