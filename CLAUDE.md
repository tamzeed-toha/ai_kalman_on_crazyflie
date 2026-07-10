# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Phases 1–3 of elaborate_plan.md's roadmap have working code (simulation model, simulated-data
generation + ANN training, real flight data collection); Phase 4 (sim-to-real transfer) and
Phase 5 (AI-KF fusion) are partially scaffolded but have a known, unresolved model inconsistency
(see "Known cross-file inconsistency" below) that must be fixed before their output can be
trusted. There is still no dependency manifest at the repo root (no `requirements.txt`/
`pyproject.toml`) — only `realtime/requirements.txt`. Convention is a shared virtualenv activated
via `source ~/.venv/bin/activate` (referenced throughout `data_collection/` and
`utils/trajectory_generator.py`), not committed to the repo. There is no test suite or CI.

Read [plan.md](plan.md) and [elaborate_plan.md](elaborate_plan.md) for the full phased roadmap
(risk register in §5, open decisions in §6) before proposing new work, and
[crazyflie_data_adaptation_brief.md](crazyflie_data_adaptation_brief.md) before touching the
model/filter or wiring real log data into it — it's a standing checklist of concrete model/data
mismatches (accelerometer semantics, 2D-vs-3D model choice, flow pixel conversion, sim-vs-real
sensor units) that Phase 4/5 work must resolve, several of which are *not yet resolved* in the
code as it stands today.

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

Architecture (per elaborate_plan.md §3): a Raspberry Pi 5 acts as a ground/edge station, not an
onboard companion computer — the Crazyflie streams IMU + optical flow over Crazyradio PA to the
RPi 5 (`cflib`), which runs the AI-KF pipeline and pushes the fused altitude estimate back as an
external measurement via the same CRTP path the firmware uses for mocap/Lighthouse/Loco position
sources. The onboard ToF sensor stays logged throughout as ground truth but is excluded from the
estimator's inputs. Check elaborate_plan.md's phase list before assuming later work is unblocked.

## End-to-end data/model pipeline

Two parallel data sources feed the same training script, and both ultimately need to agree with
whatever model `realtime/` runs online — they currently don't (see below).

**Simulated path (Phase 2):**
`model/drone_simulator.py` (`DroneModel`/`DroneSimulator(pybounds.Simulator)`, casadi/do_mpc MPC)
→ `utils/trajectory_generator.py` flies each of ~7 motifs (sinusoidal, straight, accel_decel,
turn, circle, casting, random sum-of-sines) through the simulator → writes one gzipped CSV per
trajectory plus `simulated_trajectories/manifest.csv` → `train.py` builds sliding-window
(optic-flow + accel → altitude) samples via `utils/ann_utility.py` and trains/saves a Keras MLP
into `models/` (`<name>.config.json` + `<name>.weights.h5`; only the JSON is committed — weights
are gitignored, regenerate by rerunning training).

**Real path (Phase 3):** `data_collection/collect_data.py` flies a queued trajectory library
(`data_collection/trajectories.py`) on real hardware, logging IMU/flow/ToF/EKF-state/motor/power
at up to 100 Hz per `data_collection/config.py`'s `LOG_BLOCKS`, resumable via
`data_collection/state_store.py` + `data/progress_state.json` (see
[data_collection/README.md](data_collection/README.md) for the full flight checklist — read it
before running anything in that directory, it has hardware/safety prerequisites).
`data_collection/export_to_training_format.py` then converts completed reps into the *same*
per-trajectory-CSV-plus-manifest format `train.py` expects, so `train.py
--simulated-trajectories-dir real_trajectories --models-dir ../models_real` fine-tunes on real
data with zero changes to `train.py` itself.

**Realtime path (Phase 5 skeleton, not flight-tested):** `realtime/main.py` loads a trained
model artifact and runs `realtime/fusion_loop.py`, which wires `realtime/crazyflie_link.py`
(cflib log/command streaming) → `realtime/sensor_conversion.py` (raw log units → model inputs) →
`realtime/ann_estimator.py` (loads the Keras model) → `realtime/filter_wrapper.py` (EKF fusion)
→ pushes the fused estimate back via `crazyflie_link.push_measurement`. See
[realtime/README.md](realtime/README.md) for prerequisites, the full tick-by-tick data flow, and
placeholders that need calibration before trusting it near a real flight (flow pixel→physical
gain, accel rotation, firmware log payload size, `R`/`Q` magnitudes).

### Known cross-file inconsistency (read before extending either pipeline)

`train.py`/`export_to_training_format.py` target `model/drone_simulator.py`'s measurement
equations (`r_x = v_x/z`, body-level frame). `realtime/sensor_conversion.py`'s conversions
instead target `references/planar_drone.py`'s different (2D, pitch-only) model/measurement
convention. These are two different models with different measurement functions — the training
pipeline and the realtime pipeline are not currently consistent with each other. This is flagged
in detail, with a suggested resolution order, in
[crazyflie_data_adaptation_brief.md](crazyflie_data_adaptation_brief.md) §1–3 — resolve which
model is authoritative there before adding to either pipeline. Separately, `model/drone_simulator.py`
itself may correspond to the paper's *wind-estimation* case study rather than the *altitude* one
the notebooks below use — this is still an open decision, not a resolved fact; don't assume it's
the correct Phase 1 target without confirming. (The repo previously also had a `model/drone_model.py`
wind-focused model; it has since been removed — `drone_simulator.py` is now the only model file.)

## Commands

No build/lint/test tooling exists yet. The runnable entry points, in pipeline order:

```bash
# Phase 2: generate simulated training trajectories (defaults to 1000; already-generated files
# are skipped, so an interrupted batch resumes safely)
python3 utils/trajectory_generator.py --n-trajectories 10 --length 2.0   # small/fast smoke test
python3 utils/trajectory_generator.py                                    # full default set

# Inspect one simulated trajectory
python3 utils/trajectory_visualizer.py simulated_trajectories/accel_decel_0000.csv.gz

# Phase 2: train the ANN altitude estimator on everything in simulated_trajectories/
python3 train.py
python3 train.py --window-s 2.0 --epochs 200 --test-fraction 0.2

# Phase 3: real flight data collection (run from data_collection/; see its README for the
# hardware checklist and required order of operations — bench test, then smoke test, before
# the full unattended queue)
cd data_collection
python bench_test_logging.py     # verify log throughput at current config.LOG_BLOCKS rates
python simple_flight_test.py     # minimal supervised smoke flight
python collect_data.py           # full trajectory queue, resumable across battery swaps
python export_to_training_format.py   # convert collected CSVs into train.py's input format

# Phase 5: run the (not yet flight-validated) realtime fusion loop
python -m realtime.main --uri radio://0/80/2M --model-dir /path/to/models --model-name v1_real
```

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
(`florisvb/Nonlinear_and_Data_Driven_Estimation` on GitHub) — these are the templates the
in-repo pipeline above was adapted from, and remain the reference for the still-unbuilt AI-KF
fusion step:

- **`B_empirical_nonlinear_observability_pybounds.ipynb`** — runs `pybounds` end-to-end on a
  *planar* drone model (states: `theta, theta_dot, x, x_dot, z, z_dot, k`): MPC-simulates a
  trajectory, then computes sliding-window Fisher information / minimum error variance per state
  for several candidate sensor sets (GPS-like `h_a`, camera+theta+k `h_b`, camera+IMU `h_c`). The
  `h_camera_imu` measurement set (`optic_flow, theta, theta_dot, accel_x, accel_z`) is the one
  that matches Crazyflie's Flow deck v2 + onboard IMU. This is the template for the Phase 1
  motif/observability sweep — see `crazyflie_data_adaptation_brief.md` §1 for why its `accel_x/z`
  convention (kinematic acceleration) doesn't match a real accelerometer's specific-force output
  without an explicit conversion.
- **`A_planar_drone_AI_UKF.ipynb`** — takes the same planar drone model and measurement set,
  runs a standard UKF (showing it diverge from a bad initial altitude guess), then builds the
  AI-UKF: trains/loads a small Keras ANN altitude estimator over a sliding window of
  `optic_flow, accel_x, accel_z`, derives a time-varying `R_aug_z ≈ 1/min(|accel_x|)` over each
  window (encoding "z is only observable under horizontal acceleration" as a filter covariance),
  augments the measurement function and R matrix, and reruns the UKF to show it now converges.
  This is the template for Phase 5's still-unbuilt AI-KF fusion step (`realtime/filter_wrapper.py`
  currently does plain EKF fusion, not yet the observability-weighted AI-KF augmentation).

Both notebooks depend on `casadi`, `do_mpc`, `pybounds` (installed via
`pip install git+https://github.com/vanbreugel-lab/pybounds`), and `tensorflow`/Keras — the same
stack `model/drone_simulator.py`, `utils/trajectory_generator.py`, and `train.py` depend on.
`utils/pybounds_compat.py` patches a `pybounds`/numpy incompatibility at runtime
(`patch_pybounds_simulator_time_conversion()` — must be called before any `Simulator` is
constructed); check there first if `DroneSimulator.simulate(mpc=True)` throws a
`TypeError: only 0-dimensional arrays can be converted to Python scalars`.

## Vendored plotting utilities (`utils/`)

- `utils/figure_functions.py` — plotting helpers (`plot_trajectory`, `pi_axis`, `circplot`,
  heatmap/error-variance plotting) that `model/drone_simulator.py`'s `plot_trajectory` method
  depends on via `fpl.colorline_with_heading`. Requires `figurefirst` (`pip install figurefirst`),
  not yet in any dependency manifest. `plot_trajectory_error_variance` (unused elsewhere in this
  repo) references `Colormaps`/`colorline` without importing them and calls
  `util.get_indices`/`util.log_interpolator`, neither of which exist in `utils/util.py` — don't
  call it without fixing those first.
- `utils/fly_plot_lib.py` — vendored legacy plotting library providing `fpl.colorline_with_heading`.
  Its contents are accidentally duplicated end-to-end (~1300 lines appearing twice back-to-back);
  harmless today since Python just redefines each function with the (identical) second copy, but
  worth deduplicating next time this file is touched, and a future partial edit could otherwise
  silently apply to only one copy.
- `utils/util.py` — small, self-contained angle-wrapping helpers (`wrapToPi`, `wrapTo2Pi`,
  `smart_unwrap`, `range_of_vals`); only depends on `numpy`/`copy`.

## Other local tooling

- `.claude/skills/make-cheatsheet/` — generates a validated quick-reference doc for a Python
  package (e.g. `cflib`, `pybounds`) once those dependencies are actually in use; it inspects the
  installed package via `dir()`/`inspect` rather than relying on memory, and validates the result
  with `validate_cheatsheet.py`.
- `.claude/skills/digest-paper/` — see "Working with the reference library" above.
