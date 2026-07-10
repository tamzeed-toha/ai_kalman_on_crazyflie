# Phase 3 Data Collection

Flies a Crazyflie 2.1 Brushless through a queued library of structured trajectories
(`trajectories.py`) to collect training data for the AI-Kalman altitude estimator described in
[../elaborate_plan.md](../elaborate_plan.md). Nothing is withheld from the onboard EKF during
collection -- the z-ranger stays on throughout and is logged as the reference/label signal.
Disabling it is future work (Phase 8), not part of this script.

Scoped for **at most 15 flights, ~5 minutes of battery life each** (~75 min total). The default
trajectory library (48 specs, 146 rep-instances) takes ~16 minutes of active flight time at its
default parameters, leaving generous margin for aborted/re-flown reps or added reps. Per
elaborate_plan.md's own design, this real flight data is meant to fine-tune and validate a model
pretrained on Phase 2's simulated data -- not to train an ANN from scratch.

Speed envelope: most motifs stay under 0.5 m/s, but `accel_decel_pulse`'s "fast" entries and the
`zigzag` motif reach up to **1.0 m/s / 1.5 m/s^2** -- deliberately widened to bracket a
representative deployment scenario (e.g. ~1 m/s zig-zag maneuvering) rather than leaving the
data-driven filter to extrapolate outside its training range. Every spec's footprint is verified
(via numeric integration of its velocity profile) to stay inside the geofence's soft-clamp
interior before being added -- rerun that check after editing any motif parameters; see the
comments above `build_trajectory_library()`'s accel_decel_pulse/zigzag sections in
`trajectories.py`.

## Hardware checklist (confirm before flying)

- [ ] Crazyflie 2.1 Brushless -- **must be armed** via `send_arming_request(True)` before it will
  spin its motors (`cf_link.arm_cf`); it will not respond to setpoints while disarmed, unlike
  brushed CF2.1.
- [ ] Flow deck v2 mounted (PMW3901 optical flow + VL53L1 z-range/ToF).
- [ ] Multiranger deck mounted (front/back/left/right/up).
- [ ] No Lighthouse/mocap on this setup -- there is **no independent ground truth**. The
  z-ranger itself is the only reference signal for now; the geofence (`safety.py`) is a backstop
  against `stateEstimate` drift, not a guarantee, since that estimate depends on the same
  flow+IMU+ranger fusion this whole project studies.
- [ ] `CFLIB_URI` environment variable set if your radio address differs from the default in
  `config.py` (`radio://0/80/2M/E7E7E7E7E7`).
- [ ] Safe flight space matching (or updating) `config.py`'s geofence defaults
  (`GEOFENCE_X_M`/`Y_M` = ±1.5m, `GEOFENCE_Z_M` = 0.15-1.3m -- "medium room" ~3-4m).

## Before the first real flight

1. `source ~/.venv/bin/activate` (per project convention; `cflib` should already be installed
   in this shared venv).
2. **Run the bench test first, every time you change `config.LOG_BLOCKS` or rates:**
   ```
   cd data_collection
   python bench_test_logging.py
   ```
   This connects and starts all 8 log blocks (no arming, no flight) for 30s and reports measured
   vs. requested packet rate per block, plus any `error_cb` firings. ~370 combined log
   packets/sec across 8 blocks (2 at 100 Hz) concurrent with 50 Hz commander setpoints during
   real flight is untested on this radio link -- if this bench test shows dropped packets or
   errors, reduce rates in `config.py` before flying: drop the `baro` block first, then halve
   `imu`/`flow` from 100 -> 50 Hz.
3. **First real flight -- run the minimal smoke test:**
   ```
   python simple_flight_test.py
   ```
   Connects, arms, takes off to `config.DEFAULT_HEIGHT_M`, hovers 5s, nudges forward/back 0.3m,
   lands, disarms -- using the same `FlightSessionLogger`/`config.LOG_BLOCKS` logging setup
   `collect_data.py` depends on, but none of the trajectory-queue/safety-check machinery. Fly
   this supervised, props-on, in your confirmed-safe space. Check the resulting
   `data/session_<id>.csv` afterward: do `x/y/z_m`, `acc_*_g`, `zrange_m`, `battery_v` all
   update sensibly over the flight, and do `event`/`flight_phase` show the expected
   takeoff/hover/nudge/land sequence? Don't move on until this looks right.
4. Once the smoke test looks clean, temporarily edit `trajectories.build_trajectory_library()`
   to only build the `hover` specs and one low-`accel_mps2` `accel_decel_pulse` variant, fly
   `collect_data.py` supervised, and check the resulting CSV and `data/progress_state.json`
   look right before committing to the full unattended queue.
5. The `accel_decel_fast_*` and `zigzag_*` entries are new/untested on real hardware (added to
   reach ~1.0 m/s, vs. ~0.5 m/s for everything else) -- fly a couple of those specifically,
   supervised, before trusting the full unattended queue at these speeds. Watch for the
   obstacle-stop/geofence soft-clamp engaging sooner than expected (thresholds were raised to
   `OBSTACLE_STOP_M=0.45m`/`GEOFENCE_SOFT_MARGIN_M=0.4m` for the higher speed, but that's a
   calculation, not a flight-tested value yet).

## Running / resuming

```
python collect_data.py
```

Each run: connects, arms, takes off, then repeatedly pulls the next incomplete trajectory rep
from `data/progress_state.json` and flies it, until the queue is empty, the battery drops below
`config.BATTERY_SOFT_LAND_V` (checked before starting each new rep) or
`config.BATTERY_HARD_ABORT_V` (checked every tick), `config.MAX_SESSION_TIME_S` is reached, or
Ctrl-C is pressed -- then lands and disarms.

**After a battery swap**, just re-run `python collect_data.py` again -- it reads
`data/progress_state.json` and resumes with the next incomplete rep. Progress is written
atomically after every single rep, so at most one rep is ever left ambiguous by a mid-rep power
loss (it gets re-flown automatically).

Each run's CSV lives at `data/session_<timestamp>.csv` (schema documented at the top of
`logging_io.py`); the `motif_id`/`motif_type`/`motif_phase`/`rep_index` columns identify which
trajectory is executing at each row, `event` carries one-shot markers (`takeoff`, `motif_start`,
`obstacle_front_stop`, `land`, etc.).

## Known placeholders needing real-hardware calibration

See `elaborate_plan.md`-style risk callouts inline in `config.py`/`safety.py`/`trajectories.py`:

1. `BATTERY_SOFT_LAND_V`/`BATTERY_HARD_ABORT_V` are placeholders -- CF2.1 Brushless voltage-sag
   differs from brushed CF2.1. Calibrate with one supervised flight-to-depletion.
2. Log+command throughput (~370 pkt/s) -- see bench test above.
3. `kalman.stateZ`/`statePZ` and `varZ`/`varPZ` semantics should be confirmed against
   `crazyflie-firmware`'s `kalman_core.h` `STATE_*` enum before relying on them in analysis.
4. `sum_of_sines`/`mixed_free` return-leg math (numeric-integration-based closing correction)
   needs empirical verification the drone actually ends up near its start point.
5. Default speeds/accelerations were extrapolated from `examples/bisccits`' brushed-CF2.1
   defaults -- verify the low end of each motif's parameter sweep flies stably on brushless
   before running the full unattended queue.
6. `OBSTACLE_STOP_M=0.45m`/`GEOFENCE_SOFT_MARGIN_M=0.4m` were sized from a stopping-distance
   calculation (`v^2/(2*decel)` at ~1.0 m/s / ~1.5 m/s^2 braking), not flight-tested -- confirm
   the drone actually stops within that margin at top speed before relying on it as a safety
   backstop.
