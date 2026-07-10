# Brief: Adapting the AI-KF Model to Real Crazyflie Log Data

**Audience:** a Claude Code agent (or engineer) picking up the task of adapting the paper's
model/filter (`references/planar_drone.py`, `references/A_planar_drone_AI_UKF.ipynb`,
`references/B_empirical_nonlinear_observability_pybounds.ipynb`) to actual Crazyflie hardware
and the real flight-log data now being collected by `data_collection/`. Read
[elaborate_plan.md](elaborate_plan.md) and [CLAUDE.md](CLAUDE.md) first for full project
context (this brief assumes that background and focuses specifically on the model/data
mismatches that need resolving). This document does not implement anything -- it's a checklist
of concrete considerations, several backed by exact code evidence below, to work through before
or during that adaptation.

---

## 1. The single most important, non-obvious mismatch: what "accel_x"/"accel_z" mean

`references/planar_drone.py`'s `H.h_camera_imu` (lines 235-264) defines its IMU measurements as:

```python
accel_x = -k*np.sin(theta)*j2 / m
accel_z = -g + k*np.cos(theta)*j2 / m
```

`accel_z` here is `thrust_z/m - g` -- i.e. it **equals `z_ddot`, the kinematic/coordinate
acceleration in the world frame** (consistent with the state's own dynamics: `f.f()` line 84
shows `z_dot`'s derivative is exactly `-g/m + ...thrust term`, same structure). This is **not**
what a real accelerometer physically reports. A MEMS accelerometer measures *specific force*
(reaction to non-gravitational forces only): a stationary Crazyflie sitting on a table reads
`acc.z ~= +1.0 g` (not 0), and a hovering Crazyflie's `acc.z` also reads `~1.0 g` (thrust/m/g,
not thrust/m/g - 1). The paper's own simulated "IMU measurement" is therefore the *kinematic*
acceleration, deliberately or as a modeling simplification -- not the raw specific-force signal
`acc.x/y/z` in the Crazyflie logs actually contain.

**This must be resolved deliberately, not silently defaulted:**
- **Option A** -- convert real `acc.x/y/z` (raw specific force, in units of *g*, body frame) into
  an estimate of world-frame kinematic acceleration by rotating into world frame using
  `stateEstimate.roll/pitch/yaw` and subtracting gravity's projection, before feeding the model.
  This is the "match the paper's equations as written" option.
- **Option B** -- re-derive `h_camera_imu`'s acceleration terms to model *specific force* instead
  (drop the `-g` term, or add it back in depending on sign convention actually used in Crazyflie's
  own firmware -- verify against `crazyflie-firmware/src/modules/src/estimator/estimator_kalman.c`,
  which already does exactly this rotation+gravity-subtraction for its own EKF propagation step,
  so it's a good source of the *correct*, already-implemented conversion to copy/adapt rather than
  re-derive from scratch).
- Whichever is chosen, re-run Phase 1's `pybounds`/BOUNDS observability analysis with the
  corrected measurement model -- the paper's Fig. 4-style observability result was derived under
  their specific convention, and Option B in particular changes the measurement function's
  Jacobian, which changes the observability math (not just a units nit).

## 2. Filter-input vs. filter-measurement: pick one architecture deliberately

`planar_drone.py`'s `h_camera_imu` treats `accel_x`/`accel_z` as **measurements** (outputs of `h()`,
fed into an EKF/UKF's correction step) rather than **control inputs** (fed into `f()`'s propagation
step). This is a valid Kalman filter architecture, but it's the opposite of how Crazyflie's own
onboard EKF works: `estimator_kalman.c` uses gyro+accel as **process-model inputs** driving strapdown
propagation (standard practice for a "gyro/accel as the fast prediction step" INS-style EKF), with
flow/ToF/etc. as the (slower) correction-step measurements. Confirm which architecture the AI-KF
should use *before* wiring real Crazyflie data in -- if reusing the paper's `h_camera_imu` literally,
IMU data becomes a measurement (fine, just different from Crazyflie's own filter); if instead
following Crazyflie's own convention (so the augmented filter composes more naturally with the
firmware's existing `kalman_core`, per elaborate_plan.md's Phase 5 "reuse existing external-
measurement path" default), IMU data becomes a propagation input and the measurement function
needs to be re-derived to contain only `optic_flow` (+ whatever else is actually measured
independently of the IMU). This decision materially changes the shape of both `f()` and `h()`.

Also note: `planar_drone.py`'s control inputs `j1`/`j2` are idealized torque/thrust *commands* to
an assumed direct low-level controller. On the real Crazyflie, you never get to command `j1`/`j2`
directly -- `cf.commander.send_hover_setpoint()` (used by `data_collection/`) issues
velocity/height/yaw-rate setpoints, and the firmware's own attitude/thrust PID stack turns those
into motor PWMs (logged as `motor.m1-m4` in the collected CSVs, but only as an *output*, not a
clean `j1`/`j2` you can invert). If the adapted model needs a control input at all (see above),
IMU-measured `acc`/`gyro` is the practical stand-in available in real data, not a reconstructed
`j1`/`j2`.

## 3. 2D planar model vs. Crazyflie's actual 3D dynamics

`planar_drone.py` is explicitly 2D (states `theta, theta_dot, x, x_dot, z, z_dot, k` -- a single
pitch angle and a single horizontal axis `x`). The Crazyflie has roll **and** pitch, and `x`
**and** `y`. Two options, and this decision is currently unresolved in this repo (already flagged
in `CLAUDE.md`, still open):
- **Restrict to single-axis maneuvers**: only use trajectories/log rows where motion is
  (approximately) confined to one horizontal axis with the corresponding attitude angle (e.g. a
  pure-pitch, x-only `accel_decel_pulse` rep with `axis="x"` and near-zero roll/`y`-motion) so the
  2D model applies directly. `data_collection/trajectories.py`'s `accel_decel_pulse` and
  `sum_of_sines` motifs already have an `axis` parameter (`"x"` or `"y"`) for exactly this -- filter
  the collected CSVs by `motif_param_json`'s `axis` field and by low roll/low-vy to get clean 2D-
  consistent segments.
- **Generalize the model to 3D**: `model/drone_simulator.py`'s `DroneModel`/`DroneSimulator`
  already attempts a fuller 3D case (states `x, y, z, v_x, v_y, v_z, psi, w, zeta` + calibration
  params `k_x, k_y, k_psi`), but per `CLAUDE.md`, **it's unclear this corresponds to the same
  altitude-observability case study as `planar_drone.py`** -- it looks derived from the paper's
  separate *wind-estimation* case study, not the *altitude* one. `model/drone_model.py` is a third,
  different, wind-focused model with **`z_dot` hardcoded to `0.0`** (altitude isn't even dynamic in
  it) -- almost certainly not usable for altitude estimation without rewriting its `z` dynamics.
  **Do not silently extend one of these three inconsistent models further; explicitly decide (and
  document in `CLAUDE.md`) which one -- or a new 3D altitude-specific model -- is the actual Phase 1
  target before building on it.**

## 4. Optical flow: raw pixel counts vs. the model's physical `optic_flow` variable

`planar_drone.py`'s `optic_flow = x_dot / z` (rad/s-equivalent physical flow rate). The Crazyflie
logs give `motion.deltaX`/`motion.deltaY` (raw PMW3901 pixel counts accumulated over each 10ms log
block, per `data_collection/config.py`'s `flow` block) plus `motion.squal` (a flow-quality metric,
0-255ish, useful for filtering low-confidence samples e.g. over low-texture/low-light surfaces).
Converting raw pixel counts to the model's physical flow rate requires the PMW3901's angular
resolution (counts-per-radian, itself height-dependent for a fixed-focus lens) -- **do not guess
this conversion factor from memory; read it out of
`crazyflie-firmware/src/deck/drivers/src/flowdeck_v1v2.c` and/or how `estimator_kalman.c` consumes
the flow measurement** (search for `flowdeck`/`MOTION_` in that source tree), since the firmware
already implements the authoritative conversion and it should be matched rather than re-derived
independently.

## 5. Physical parameters are placeholders, not Crazyflie's actual values

`planar_drone.py`'s globals (top of file): `m = 1` kg, `L = 0.5` m, `Iyy = 0.02` kg*m^2, `g = 9.81`
m/s^2. These are generic toy-quadrotor values from the paper's companion repo. Crazyflie's actual
mass is ~27-37g depending on decks (elaborate_plan.md Sec. 4, Phase 1) -- **two to three orders of
magnitude lighter** than `m=1`. `L` (arm length) and `Iyy` (pitch moment of inertia) need
Crazyflie-specific values too (Bitcraze publishes some of these; otherwise estimate from CAD/specs
or system ID). Any observability/Cramer-Rao-bound result computed with the placeholder parameters
does not carry over to the real vehicle -- Phase 1's re-run with corrected parameters is required,
not optional polish.

## 6. Real Crazyflie log data: rates, units, and honesty about "ground truth"

The actual CSV schema now being produced by `data_collection/logging_io.py` (see that file's
`CSV_FIELDS` for the full list) is heterogeneous, unlike a clean simulated dataset:

- **Units differ from the model's assumed SI units** -- `acc_x/y/z_g` are in units of standard
  gravity (multiply by `9.81` for m/s^2, *after* deciding Sec. 1's specific-force-vs-kinematic
  question), `gyro_x/y/z_dps` are in deg/s (convert to rad/s), `zrange_mm` is millimeters
  (`zrange_m` is precomputed for convenience). Do not assume m/s^2 or rad/s without checking the
  column name suffix.
- **Sampling rates are not uniform across columns.** The CSV is written at one fixed rate
  (`config.CSV_WRITE_RATE_HZ = 100`), but the underlying LogConfig blocks update at 100/50/20/10 Hz
  (`config.LOG_BLOCKS`) -- so `kalman_state_z_m`, `battery_v`, `range_front_m`, `baro_asl_m`, etc.
  will show the **same repeated value across several consecutive 100Hz rows** until their own
  block's next update arrives. Do not naively finite-difference or treat every row as an
  independent new sample for those slower columns; either resample/interpolate deliberately per
  column, or restrict per-column analysis to that column's own native rate.
- **There is no independent ground truth**, and this cuts deeper than "we don't have mocap":
  `stateEstimate.x/y/z/vx/vy/vz` (and `kalman_state_z_m`/`kalman_var_z` etc.) are themselves outputs
  of the onboard EKF that *already fuses* the z-ranger this project wants to replace. Using
  `stateEstimate.*` as a training label is circular for anything derived from that fusion, not just
  for `z` -- treat `zrange_mm`/`zrange_m` as the only semi-independent reference signal available
  (and even that is a direct sensor reading being validated against, not ground truth for the
  *fused* state).
- **`kalman_state_z_m`/`kalman_state_pz` and `kalman_var_z`/`kalman_var_pz` naming needs
  verification** against `crazyflie-firmware`'s `kalman_core.h` `STATE_*` enum before use -- based on
  Crazyflie convention, `Z` is likely the world-frame z-position state and `PZ` the body-frame
  z-velocity state (an inherited naming quirk, not "position" vs. "filtered position"), but confirm
  before building analysis that assumes otherwise; mislabeling this would corrupt any "how much is
  the ranger correcting the z estimate" diagnostic.
- **Real data volume is small and highly autocorrelated within a rep.** ~15 flights x ~5 min,
  ~38 trajectory specs x 2-4 reps each (`data_collection/trajectories.py`). Per elaborate_plan.md's
  own phasing, this is meant to fine-tune/validate a model pretrained on Phase 2's *simulated*
  data, not train an ANN from scratch. When splitting real data for train/val, **split by
  `session_id` or by whole trajectory rep (`motif_id` + `rep_index`), never by individual row** --
  rows within one ~10s maneuver are strongly correlated (they're one continuous signal, not
  independent draws), so a random row-level split leaks and will produce an optimistic validation
  score that doesn't reflect real generalization.
- **Motif metadata is available for stratification.** `motif_type`, `motif_phase`,
  `motif_param_json`, `rep_index` columns let you slice by exact trajectory parameters (e.g. hold
  out all `accel_decel_pulse` reps with `axis="y"` as a generalization test, or compare error
  across `motif_phase` segments like `outbound` vs. `return` vs. `pause_mid`).
- **Real sensor noise (vibration, bias, quantization) differs from whatever noise model Phase 2's
  simulator assumes.** If Phase 2 hasn't already empirically characterized this, a quick pass is
  cheap: use the collected `hover` reps (near-static, known near-zero true horizontal velocity) to
  estimate real `acc`/`gyro`/flow noise statistics and compare against the simulator's assumed
  noise -- flagged as a sim-to-real gap risk in elaborate_plan.md's risk register, and this is the
  fastest way to quantify it.

## 7. Quick reference: exact CSV columns available

See `data_collection/logging_io.py`'s `CSV_FIELDS`/`VAR_TO_COLUMN` for the authoritative list (do
not hand-copy this into other docs where it can drift out of sync -- read that file directly).
Grouped summary: identity/timing (`pc_time_s`, `cf_time_ms`, `session_id`); motif/trajectory label
(`motif_id`, `motif_type`, `motif_phase`, `rep_index`, `motif_param_json`, `flight_phase`);
commanded setpoints, both post-safety-override and raw (`cmd_*` / `cmd_*_raw`); IMU
(`acc_x/y/z_g`, `gyro_x/y/z_dps`); flow deck (`flow_delta_x/y_px`, `flow_squal`, `zrange_mm`,
`zrange_m`); EKF baseline (`x/y/z_m`, `vx/vy/vz_m_s`, `roll/pitch/yaw_deg`); EKF z-channel
diagnostics (`kalman_state_z_m`, `kalman_state_pz`, `kalman_var_z`, `kalman_var_pz`); power/motors
(`battery_v`, `battery_level_pct`, `pm_state`, `motor_m1-m4`); multiranger (`range_front/back/
left/right/up_m`); baro (`baro_asl_m`, `baro_pressure_hpa`, `baro_temp_c`); safety/events
(`safety_flag`, `should_land`, `event`).

## 8. Suggested order of operations

1. Resolve Sec. 1 (accel semantics) and Sec. 2 (filter architecture) first -- everything else
   depends on these two decisions.
2. Resolve Sec. 3 (which model file / 2D-restricted vs. 3D-generalized) and update `CLAUDE.md` with
   the decision so it doesn't stay an open question indefinitely.
3. Re-derive/re-run Phase 1's observability analysis with corrected physical parameters (Sec. 5)
   and the corrected measurement model (Sec. 1/2) before trusting any "which motifs matter" guidance
   derived from the placeholder-parameter version.
4. Only then wire in real `data_collection/` CSVs, applying the unit conversions and per-column
   rate-awareness from Sec. 6, with the flow pixel-to-physical conversion from Sec. 4 sourced from
   firmware, not guessed.
5. Do train/val splitting by session/rep, not row, from the start -- retrofitting this after an
   ANN has already been tuned on a leaky split invalidates whatever tuning happened in between.
