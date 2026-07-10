# realtime/ — running the AI-KF live on a Crazyflie

This package runs the AI-KF altitude estimator online: it streams IMU + optical flow from the
Crazyflie to this machine over Crazyradio PA (`cflib`), runs a trained ANN + Kalman filter on a
rolling window of that data, and pushes the fused altitude estimate back to the Crazyflie as an
external measurement. It assumes you already have a **trained ANN artifact** (Phase 2/4/`train.py`)
and the existing EKF code in `references/extended_kalman_filter.py` — it does not train anything
itself.

**Model note:** this package targets `model/drone_simulator.py`'s body_level model (states
`[x, y, z, v_x, v_y, v_z, psi]`, measurements `[meas_r_x, meas_r_y, meas_v_x_dot, meas_v_y_dot]`)
via `realtime/drone_model_np.py` — a dependency-light numpy mirror of it, duplicated rather than
imported because `model/drone_simulator.py` unconditionally pulls in casadi/pybounds/sympy/
matplotlib at module level for its MPC/plotting code, which this latency-sensitive package
doesn't need. **This package used to target `references/planar_drone.py` (a different, 2D
model) — that was a real mismatch with what `train.py` actually trains, since a model trained by
`train.py` couldn't be loaded or fed correctly here. Fixed; if you're reading old notes/PRs that
mention `planar_drone`/`theta`/`accel_z`/`ANN_CHANNELS=["optic_flow","accel_x","accel_z"]` in
this package, they predate the fix.**

## Prerequisites

- `pip install -r requirements.txt` (cflib, tensorflow, numpy, pandas)
- A Crazyflie 2.1 with Flow deck v2, and a Crazyradio PA connected to this machine
- A trained ANN artifact from `utils/ann_utility.save_model_complete(...)` (written by
  `train.py`): two files, `<model_name>.config.json` and `<model_name>.weights.h5`, in some
  directory
- Run everything from the repo root (so `references.*`, `utils.*`, and `realtime.*` are all importable)

## Running it

```bash
python -m realtime.main --uri radio://0/80/2M --model-dir /path/to/models
```

- Omit `--model-name` to auto-discover the most recently modified model in `--model-dir` (see
  `ann_estimator.find_latest_model`) -- this is what makes "retrain, then run this script" work
  without manually updating a pinned model name each time.
- `--initial-z` lets you set a deliberately poor starting altitude guess — the project's success
  criterion is the AI-KF converging from a bad guess faster than the stock EKF, so this is useful
  for testing that specifically.
- Ctrl+C stops the fusion loop and disconnects cleanly.

## What happens when you run it

1. `cflib` streams a merged IMU+optical-flow log block from the Crazyflie at 100Hz.
2. Each new sample is tilt-compensated (roll/pitch, dead-reckoned between 50Hz attitude refreshes
   -- see `buffers.AttitudeTracker`) and converted to `[meas_r_x, meas_r_y, meas_v_x_dot,
   meas_v_y_dot]`, pushed into a rolling window, and handed off to a separate worker thread (so a
   slow ANN/filter tick never blocks/delays the radio link).
3. Once the window is full (size determined by the loaded model, not hardcoded), the trained ANN
   produces an altitude estimate from it; a time-varying trust weight is derived from how much
   horizontal acceleration (either axis) was in that window (per the paper: `z` is only
   observable during accel/decel bursts, not hover).
4. That gets fed into an EKF alongside the raw flow/accel measurements.
5. The fused `(z, variance)` is sent back to the Crazyflie as an external position measurement,
   the same mechanism used for mocap/Lighthouse.

## Known gaps — verify before flying

These are placeholders/assumptions this code makes explicit rather than hiding; check them
before trusting this near real flight (see `elaborate_plan.md` Phase 0/5/6):

- **`sensor_conversion.py`'s flow conversion — FIXED, empirically validated, not a placeholder
  anymore.** It used to be `FLOW_GAIN_PLACEHOLDER=1.0` fed straight from raw `motion.deltaX/Y`,
  which was wrong in two independent ways: (1) ~500x too large in magnitude, and (2) built on
  the wrong axis mapping -- `crazyflie-firmware`'s `flowdeck_v1v2.c` explicitly swaps and negates
  axes before use (`accpx=-deltaY`, `accpy=-deltaX`) with the comment "flip motion information to
  comply with sensor mounting"; the raw logged `motion.deltaX/deltaY` are sensor-native axes, not
  aircraft forward/lateral. Both are now fixed (gain derived from `mm_flow.c`'s constants, axis
  swap applied) and verified by regressing corrected `meas_r_x` against real logged
  `stateEstimate.vx`/`zrange_m`: R²≈0.63 (vs ≈0.002 for the old placeholder+mapping) -- see
  `data_collection/export_to_training_format.py`'s docstring for the full validation. A sanity
  clamp (`FLOW_SANITY_CLAMP`) also rejects physically-impossible flow values as a safety net.
  The tilt-compensation rotation (roll/pitch de-tilt) for accel is mathematically standard, but
  whether the model's `v_x_dot`/`v_y_dot` is meant to be exactly this quantity (vs. some other
  kinematic/specific-force convention) is not independently verified — see
  `crazyflie_data_adaptation_brief.md` Sec. 1-2. `motor_commands_to_control_input` has no safe
  default at all for arbitrary/piloted flight — for scripted test flights, supply a
  `control_input_provider` that returns the trajectory's own known commanded
  acceleration/yaw-rate instead (see e.g. the square+zigzag test script), which is exact rather
  than estimated.
- **`crazyflie_link.py`**: `push_measurement` uses `cf.extpos.send_extpos(x, y, z)`. Whether the
  firmware honors a per-tick time-varying variance for this path, or only a fixed configured
  variance across all three axes, isn't confirmed against `crazyflie-firmware`'s `kalman_core.c`
  yet — check this before assuming the AI-KF's trust-weighting is actually reaching the firmware.
- **Log block size**: the merged `imu_flow` block is ~31 bytes; confirm this fits the CRTP log
  payload limit. If not, drop `motion.squal` first (already omitted) then reconsider rate.
- **`config.py`**: `R_BASE_DIAG` and `Q_DIAG` are placeholder magnitudes, not tuned values —
  tune these by replaying logged/simulated data through the pipeline before trusting live output.
- **State reduction**: `drone_model_np.py` fixes wind (`w`, `zeta`) at 0 and motor calibration
  (`k_x`, `k_y`, `k_psi`) at 1.0 rather than tracking them as EKF states, since nothing in the
  current sensor set constrains them (indoor flight, no wind sensor, no calibration procedure).
  Revisit if either assumption stops holding (e.g. outdoor flight, or a real calibration step
  becomes available).

## Testing without hardware

Before connecting to a real Crazyflie, feed `FusionLoop.on_imu_flow` / `on_state_att` directly
from a replayed log or simulated trajectory (bypassing `CrazyflieLink`) to validate the
ANN→filter wiring end-to-end, and to benchmark per-tick latency on the target machine — see
`elaborate_plan.md` Phase 6 for why both compute latency and radio round-trip latency need to be
measured separately. A minimal no-hardware wiring smoke test (load a real trained model, push
synthetic samples through the window, tick the filter) is worth running after any change here
before ever touching a real Crazyflie with it.
