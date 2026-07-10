# realtime/ — running the AI-KF live on a Crazyflie

This package runs the AI-KF altitude estimator online: it streams IMU + optical flow from the
Crazyflie to this machine over Crazyradio PA (`cflib`), runs a trained ANN + Kalman filter on a
rolling window of that data, and pushes the fused altitude estimate back to the Crazyflie as an
external measurement. It assumes you already have a **trained ANN artifact** (Phase 2/4) and the
existing filter/model code in `references/` — it does not train anything itself.

## Prerequisites

- `pip install -r requirements.txt` (cflib, tensorflow, numpy, pandas)
- A Crazyflie 2.1 with Flow deck v2, and a Crazyradio PA connected to this machine
- A trained ANN artifact from `keras_ann_utility.save_model_complete(...)`: two files,
  `<model_name>_config.json` and `<model_name>_weights.h5`, in some directory
- Run everything from the repo root (so `references.*` and `realtime.*` are both importable)

## Running it

```bash
python -m realtime.main --uri radio://0/80/2M --model-dir /path/to/models --model-name v1_real
```

- `--initial-z` lets you set a deliberately poor starting altitude guess — the project's success
  criterion is the AI-KF converging from a bad guess faster than the stock EKF, so this is useful
  for testing that specifically.
- Ctrl+C stops the fusion loop and disconnects cleanly.

## What happens when you run it

1. `cflib` streams a merged IMU+optical-flow log block from the Crazyflie at 100Hz.
2. Each new sample is pushed into a rolling window and handed off to a separate worker thread
   (so a slow ANN/filter tick never blocks/delays the radio link).
3. Once the window has 10 samples, the trained ANN produces an altitude estimate from it; a
   time-varying trust weight is derived from how much horizontal acceleration was in that window
   (per the paper: `z` is only observable during accel/decel bursts, not hover).
4. That gets fed into an EKF alongside the raw optic-flow/attitude/accel measurements.
5. The fused `(z, variance)` is sent back to the Crazyflie as an external position measurement,
   the same mechanism used for mocap/Lighthouse.

## Known gaps — verify before flying

These are placeholders/assumptions this code makes explicit rather than hiding; check them
before trusting this near real flight (see `elaborate_plan.md` Phase 0/5/6):

- **`sensor_conversion.py`**: the Flow-deck pixel→optic-flow gain and the body→planar-frame
  accelerometer rotation are placeholders, not calibrated constants. `motor_commands_to_control_input`
  has no safe default at all — you must supply a real one (see `FusionLoop(control_input_provider=...)`)
  or the filter runs with a zero-thrust process model (fine for wiring tests, not for flight).
- **`crazyflie_link.py`**: `push_measurement` uses `cf.extpos.send_extpos(x, y, z)`. Whether the
  firmware honors a per-tick time-varying variance for this path, or only a fixed configured
  variance across all three axes, isn't confirmed against `crazyflie-firmware`'s `kalman_core.c`
  yet — check this before assuming the AI-KF's trust-weighting is actually reaching the firmware.
- **Log block size**: the merged `imu_flow` block is ~31 bytes; confirm this fits the CRTP log
  payload limit. If not, drop `motion.squal` first (see comment in `crazyflie_link.py`).
- **`config.py`**: `R_BASE_DIAG` and `Q_DIAG` are placeholder magnitudes, not tuned values —
  tune these by replaying logged/simulated data through the pipeline before trusting live output.

## Testing without hardware

Before connecting to a real Crazyflie, feed `FusionLoop.on_imu_flow` / `on_state_att` directly
from a replayed log or simulated trajectory (bypassing `CrazyflieLink`) to validate the
ANN→filter wiring end-to-end, and to benchmark per-tick latency on the target machine — see
`elaborate_plan.md` Phase 6 for why both compute latency and radio round-trip latency need to be
measured separately.
