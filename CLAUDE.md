# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is transitioning from planning into **Phase 1 (model & simulation
replication)** — see elaborate_plan.md §4. There is no build system, test suite, or dependency
manifest (no `requirements.txt`/`pyproject.toml`/venv committed) yet, but early modeling code has
started under `model/`. Read [plan.md](plan.md) and [elaborate_plan.md](elaborate_plan.md) to
understand the full roadmap before proposing further implementation, and check the reference
notebooks below first — they are close-to-literal worked examples of what Phase 1 (and later
Phase 5) needs to reproduce.

## Project goal

Real-time AI-Kalman state estimation running live on a Crazyflie drone, holding/estimating
altitude (`z`) accurately during flight **without a direct vertical height sensor** (the onboard
ToF/z-ranger withheld from the estimator), by fusing a neural-network-based estimator with a
Kalman filter using observability-informed, time-varying measurement trust. This is a direct
application of the **Augmented Information Kalman Filter (AI-KF)** method from
[references/summaries/bens_epic_paper.md](references/summaries/bens_epic_paper.md)
(Cellini et al. 2025, arXiv:2511.08766) to Crazyflie hardware.

Key insight carried over from that paper: altitude `z` is only observable from optic-flow +
accelerometer sensing when there is **non-zero horizontal acceleration** (speed-up/slow-down
bursts) — not from hovering or constant-velocity/vertical motion. This drives the whole
data-collection and flight-test design (see elaborate_plan.md §2–3 for the full
paper-to-hardware mapping and phased roadmap).

Planned architecture (not yet implemented): a Raspberry Pi 5 acts as a ground/edge station, not
an onboard companion computer — the Crazyflie streams IMU + optical flow over Crazyradio PA to
the RPi 5 (`cflib`), which runs the AI-KF pipeline and pushes the fused altitude estimate back as
an external measurement via the same CRTP path the firmware uses for mocap/Lighthouse/Loco
position sources. The onboard ToF sensor stays logged throughout as ground truth but is excluded
from the estimator's inputs. See elaborate_plan.md §3–4 for the full phased roadmap (Phase 0
environment setup through Phase 8 live flight demo) and the risk register (§5) and open
decisions (§6) before starting new work — check which phase is current before assuming later
work is unblocked.

## Working with the reference library

`references/` holds structured summaries of papers relevant to this project, built with the
`digest-paper` skill (`.claude/skills/digest-paper/SKILL.md`).

- `references/INDEX.md` — one paragraph per paper, links to the full summary. Read this first to
  see what's already been digested.
- `references/summaries/*.md` — full structured summary per paper (Key Finding, Methods, Main
  Results, Limitations, Relevance to This Project, etc.), one file per paper, named by slug
  (e.g. `bens_epic_paper.md`).
- To add a new paper to the library, use the `digest-paper` skill rather than summarizing ad hoc
  — it enforces a consistent template and appends to INDEX.md without disturbing existing
  entries. It reads this CLAUDE.md to fill in each summary's "Relevance to This Project" section,
  so keep the Project goal section above current.
- Source PDFs (e.g. `references/Bens_epic_paper.pdf`) live in `references/` alongside the
  notebooks below, not the repo root.

### Worked-example notebooks (`references/A_*.ipynb`, `references/B_*.ipynb`)

Two Jupyter notebooks pulled from the paper's companion codebase
(`florisvb/Nonlinear_and_Data_Driven_Estimation` on GitHub) — these are the most concrete
templates we have for Phase 1/Phase 5 and should be adapted rather than re-derived from scratch:

- **`B_empirical_nonlinear_observability_pybounds.ipynb`** — runs `pybounds` end-to-end on a
  *planar* drone model (states: `theta, theta_dot, x, x_dot, z, z_dot, k`): MPC-simulates a
  trajectory, then computes sliding-window Fisher information / minimum error variance per state
  for several candidate sensor sets (GPS-like `h_a`, camera+theta+k `h_b`, camera+IMU `h_c`). The
  `h_camera_imu` measurement set (`optic_flow, theta, theta_dot, accel_x, accel_z`) is the one
  that matches Crazyflie's Flow deck v2 + onboard IMU — this notebook *is* essentially the Phase 1
  deliverable already, just with placeholder (non-Crazyflie) physical parameters. This is the
  template for reproducing elaborate_plan.md's Phase 1 motif/observability sweep.
- **`A_planar_drone_AI_UKF.ipynb`** — takes the same planar drone model and measurement set,
  runs a standard UKF (showing it diverge from a bad initial altitude guess), then builds the
  AI-UKF: trains/loads a small Keras ANN altitude estimator over a sliding window of
  `optic_flow, accel_x, accel_z`, derives a time-varying `R_aug_z ≈ 1/min(|accel_x|)` over each
  window (encoding "z is only observable under horizontal acceleration" as a filter covariance),
  augments the measurement function and R matrix, and reruns the UKF to show it now converges.
  This is the template for Phase 5 (AI-KF integration) and the ANN part of Phase 2.

Both notebooks depend on `casadi`, `do_mpc`, `pybounds` (installed via
`pip install git+https://github.com/vanbreugel-lab/pybounds`), and `tensorflow`/Keras (notebook A
only) — none of which are installed or pinned anywhere in this repo yet. Both also import several
helper modules (`planar_drone`, `plot_utility`, `generate_training_data_utility`,
`keras_ann_utility`, `extended_kalman_filter`, `unscented_kalman_filter`) from a local `../Utility`
directory that does not exist in this repo; their fallback path fetches each file individually
from `raw.githubusercontent.com/florisvb/Nonlinear_and_Data_Driven_Estimation/main/Utility/` at
runtime. Decide whether to vendor these into the repo (for offline/reproducible runs) before
relying on the notebooks running unattended.

## In-progress modeling code (`model/`)

Early Phase 1 code, not yet wired to any notebook or test:

- `model/drone_model.py` — a standalone `Drone` class with a *wind-focused* kinematic model
  (states `z, v_x, v_y, psi, w_x, w_y, w_x_dot, w_y_dot`; note `z_dot` is hardcoded to `0.0`, so
  altitude is not yet dynamic here) and a hand-rolled RK4 discretizer. Self-contained — no
  `pybounds`/`casadi` dependency.
- `model/drone_simulator.py` — a fuller `DroneModel`/`DroneSimulator(pybounds.Simulator)` pair
  (states include `x, y, z, v_x, v_y, v_z, psi, w, zeta` plus motor-calibration params `k_x, k_y,
  k_psi`; supports `global` or `body_level` frames) using `casadi`/`do_mpc` for MPC trajectory
  generation, mirroring the paper's full 3D wind+altitude quadcopter case study rather than the
  simpler 2D altitude submodel in the notebooks above. Imports `from utils import figure_functions
  as ff` (see below) for its `plot_trajectory` method only — everything else in the class doesn't
  depend on it.
- `crazyfly_simulation.ipynb` (repo root) — currently empty; presumably the intended home for
  Crazyflie-specific Phase 1 simulation work.
- `utils/figure_functions.py` — vendored plotting helpers (`plot_trajectory`, `pi_axis`,
  `circplot`, heatmap/error-variance plotting, a `LatexStates` symbol dict) that `drone_simulator.py`
  partly depends on (`ff.plot_trajectory`, which calls `fpl.colorline_with_heading`). Gaps:
  - Unconditionally does `import figurefirst as fifi` — a real PyPI package (`pip install
    figurefirst`, currently at 0.0.6) but not yet installed in any environment/manifest here, so
    `import figure_functions` fails until it is.
  - `plot_trajectory_error_variance` (one function, not used by `drone_simulator.py`) references
    `Colormaps` and `colorline` without importing them (likely meant to be a `pybounds` colormap
    helper / `pybounds.colorline`, per notebook B's usage) — `NameError` if ever called. It also
    calls `util.get_indices(...)` and `util.log_interpolator(...)`, neither of which exist in
    `utils/util.py` (which only has `wrapToPi`, `wrapTo2Pi`, `smart_unwrap`, `range_of_vals`) —
    another `AttributeError` if this function is ever called, independent of the `Colormaps`/
    `colorline` gap.
  - Line 182 calls `scipy.ndimage.zoom` but the file only does `import scipy` — needs
    `import scipy.ndimage` added explicitly, or it's an `AttributeError` at runtime (only reached
    if `interpolation` is truthy).
- `utils/fly_plot_lib.py` — vendored legacy plotting library (`colorline`, `colorline_with_heading`,
  `histogram`, `boxplot`, `scatter`, etc.) providing the `fpl.colorline_with_heading` that
  `figure_functions.plot_trajectory` needs. **The file's contents are accidentally duplicated
  end-to-end** (2701 lines; the same ~1300-line module, including its header comment, appears
  twice back-to-back, at lines 1–1339 and 1340–2701). This doesn't currently break anything —
  Python just re-defines each function with the second copy's (identical, except
  `colorline_with_heading` which only exists in the second copy) — but it's worth deduplicating
  next time this file is touched, or a future partial edit could silently apply to only one copy.
  `scatter_line`/`scatter_box` also do function-local `import flystat.resampling`, an extra
  dependency only needed if those two functions are actually called.

- `utils/util.py` — small, self-contained angle-wrapping helpers (`wrapToPi`, `wrapTo2Pi`,
  `smart_unwrap`, `range_of_vals`); only depends on `numpy`/`copy`, no further gaps.

With `fly_plot_lib.py` and `util.py` now both present, the only remaining blocker to `import
utils.figure_functions` (and thus `drone_simulator.py.plot_trajectory`) succeeding is installing
the `figurefirst` package — there's still no dependency manifest/venv anywhere in this repo, so
that (and `pybounds`/`casadi`/`do_mpc`/`tensorflow` from the notebooks) all still need to be
installed before anything here can actually run.

These two models (`drone_model.py`'s wind-focused kinematics vs. `drone_simulator.py`'s fuller
wind+altitude MPC model) do not obviously correspond to the same case study as the
`planar_drone.py` model the reference notebooks use (θ-pitch, forward-accel, optic-flow altitude
submodel — Eq. 6–10 of the paper). Confirm with the user which model is meant to be the Phase 1
target for Crazyflie before extending either one further, rather than assuming.

## Other local tooling

- `.claude/skills/make-cheatsheet/` — generates a validated quick-reference doc for a Python
  package (e.g. `cflib`, `pybounds`) once those dependencies are actually in use; it inspects the
  installed package via `dir()`/`inspect` rather than relying on memory, and validates the result
  with `validate_cheatsheet.py`. Not yet exercised in this repo since no dependencies are
  installed.
