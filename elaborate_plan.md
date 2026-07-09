# Elaborate Plan: AI-Kalman Altitude Estimation on Crazyflie

This formalizes [plan.md](plan.md) into a phased, technically grounded roadmap. It leans heavily
on the methodology in [Cellini et al. 2025 — "Discovering and exploiting active sensing motifs
for estimation"](references/summaries/bens_epic_paper.md) (BOUNDS + AI-KF), whose altitude
case study (their Eq. 6–10, Figures 4–5) is almost a direct analog of what we want to do on
Crazyflie.

---

## 1. Goal

**Grand prize goal (from plan.md):** real-time AI-Kalman state estimation running live on a
Crazyflie, holding/estimating **z (height)** accurately during flight **without a direct vertical
height sensor** (i.e., the onboard time-of-flight/z-ranger is unavailable or ignored), by fusing
a neural-network-based estimator with a Kalman filter using observability-informed, time-varying
measurement trust — exactly the Augmented Information Kalman Filter (AI-KF) pattern from the
paper.

**Success is demonstrated when:** with the ToF height measurement withheld, the AI-KF-fused z
estimate (a) converges from an arbitrary/poor initial guess faster and more reliably than the
stock EKF running on flow+accel alone, and (b) tracks true altitude (validated against the
withheld ToF signal, logged but not fed to the estimator) within an acceptable error band during
representative flight maneuvers.

---

## 2. Key insight carried over from the paper

The paper's altitude sub-problem (Eqs. 6–10, Figure 4) is *already* the Crazyflie problem:

- **States:** altitude `z`, vertical velocity `vz`, forward velocity `vx`
- **Measurement:** forward optic flow `rx = -vx/z` (ambiguous: velocity and height are entangled)
- **Inputs:** measured vertical and forward acceleration
- **Finding:** `z` is only observable when there is **non-zero horizontal acceleration**
  (speeding up/slowing down), not from vertical motion. Constant-velocity or hovering flight
  gives *zero* information about `z` from flow+accel alone.

This directly maps onto Crazyflie hardware:

| Paper symbol | Crazyflie equivalent |
|---|---|
| forward optic flow `rx` | Flow deck v2 (PMW3901) optical flow, converted to flow angle/magnitude |
| forward/vertical acceleration | Onboard IMU (BMI088/BMI160) accelerometer |
| altitude `z` (state, unmeasured) | To be estimated — normally supplied by the flow deck's VL53L1 ToF sensor, which we treat as "impaired"/withheld |
| ANN altitude estimator | Same architecture: small feed-forward net, ~2s window of flow + accel |
| AI-UKF | Augment Crazyflie's onboard EKF (or an offboard replica) with the ANN's z estimate, weighted by an observability-derived, time-varying covariance |

**Design consequence:** the active-sensing motif we need to fly is **horizontal acceleration
pulses** (brief speed-up/slow-down bursts), *not* vertical bobbing. This is the single most
important, slightly counter-intuitive takeaway to build the whole data-collection and flight-test
plan around.

---

## 3. System architecture decision

**Decision: Raspberry Pi 5 as a ground/edge station, not an onboard companion computer.**

- Compute is not the constraint (a 3×64-neuron ANN + a low-dimensional UKF run in microseconds
  on an RPi 5). Payload is: the Crazyflie 2.1 payload budget is a few grams, so nothing beyond
  its existing decks (Flow deck v2, and its own IMU) will fly onboard for this project.
- **Architecture:**
  1. Crazyflie streams IMU + optical flow logs over Crazyradio PA to the RPi 5 at the highest
     sustainable log rate (target ~100 Hz per stream, `cflib` logging framework).
  2. The RPi 5 runs the real-time AI-KF pipeline: rolling-window ANN state estimator → ANN (or
     heuristic) observability estimator → time-varying-covariance fusion.
  3. The fused z-estimate (with its covariance) is pushed back to the Crazyflie as an **external
     measurement** over the same radio link, using the same CRTP mechanism the firmware already
     supports for external position sources (mocap/Lighthouse/Loco use this path). This lets us
     reuse the onboard `kalman_core` estimator as the "base KF" instead of reimplementing a UKF
     from scratch — closely mirroring how the paper augments a UKF's measurement vector and `R`
     matrix.
  4. The onboard ToF sensor stays enabled and logged throughout **all data collection**, purely
     as a ground-truth label. It is excluded from the estimator's input set for anything labeled
     "impaired-sensor" — first in analysis/replay, later in true live flight (physically disabled
     or its measurement path dropped) for the grand-prize demo.
- **Fallback / future upgrade path (not required initially):** if radio latency/jitter proves too
  high for a tight control loop, revisit onboard inference on a Bitcraze AI-deck (GAP8). Flag
  this as a contingency, not a phase-1 dependency.

**Open engineering question to resolve early (Phase 5):** whether we inject the fused estimate
into the *firmware's* existing external-measurement queue (likely least invasive, reuses proven
code path) versus running a fully custom off-board UKF and only using the Crazyflie for control
setpoints (more control over the AI-KF internals, but reimplements what the firmware already
does well). Recommendation: start with firmware injection; only fork the estimator if the
injection path can't carry a custom time-varying covariance cleanly.

---

## 4. Phased roadmap

### Phase 0 — Environment & tooling setup
- Install and smoke-test: `cflib` (Python Crazyflie link), `pybounds` (from the paper's repo),
  a build of `crazyflie-firmware` (for reference/inspection of `kalman_core.c` and the external
  measurement CRTP path), and an ML stack (PyTorch or TensorFlow/Keras, matching the paper's
  Keras implementation is fine).
- Set up the RPi 5 as the radio host: Crazyradio PA over USB, verify basic connect/log/command
  round-trip latency with `cflib` examples.
- Confirm which Crazyflie hardware revision and decks are in hand (2.1 vs 2.1+, Flow deck v2
  confirmed via your ground-truth answer). Record exact firmware version being flown.
- **Deliverable:** a checked-in `hardware.md` or README note capturing exact hardware/firmware
  versions, plus a working "hello world" log-and-command round trip through the RPi 5.

### Phase 1 — Model & simulation replication
- Reimplement the paper's 2D kinematic altitude model (Eq. 6–10) with Crazyflie-specific
  parameters (mass ~27–37 g depending on decks, drag terms re-estimated or left as free/augmented
  states as the paper does).
- Run `pybounds`/BOUNDS on this model to reproduce the paper's Figure 4-style result for our own
  parameters: confirm `z` is observable under horizontal accel/decel and confirm rough magnitude
  of the effect (how much acceleration is "enough").
- Sweep motif types (hover, constant velocity, accel/decel pulses of varying magnitude, offset
  turns) to build our own version of the paper's Table 1, specific to whatever sensor set we
  actually have (flow deck + IMU only — no airflow/wind sensor, so this is the "optic flow,
  accel" row of their table).
- **Deliverable:** a small simulation notebook/script + a written observability summary telling
  us exactly which flight motifs to fly in Phase 3.

### Phase 2 — Simulated data generation & ANN prototyping
- Generate a large simulated trajectory dataset (order of thousands, following the paper's
  approach of randomizing initial velocity, acceleration profiles via sum-of-sines, altitude)
  spanning the motif envelope identified in Phase 1.
- Train the ANN altitude estimator (rolling window of flow + accel → z) and, optionally, an ANN
  observability estimator, exactly as in the paper's Methods (feed-forward, 3×64 hidden, circular
  handling not needed here since `z` isn't circular — plain MSE loss is fine).
- Validate against the Cramér–Rao bound computed in Phase 1; confirm the sim-only model behaves
  the way the paper's did before ever touching hardware.
- **Deliverable:** a trained "v0" ANN estimator + observability model, sim-only, with a
  reproducible training script and a validation report (error vs. observability bins, cf. paper
  Fig. 3d–f).

### Phase 3 — Real flight data collection
- Fly structured sessions covering: hover baseline, horizontal accel/decel pulses (varying
  magnitude and duration), offset turns, and free/mixed trajectories, per the motif set from
  Phase 1.
- Log IMU + optical flow **and** the onboard ToF (ground truth) simultaneously at the highest
  sustainable rate for every flight. Nothing is withheld at collection time — impairment is
  applied later, in software, when selecting what the estimator sees.
- Use full battery flight time per session and repeat across many battery cycles to build a
  real-data set large enough to fine-tune/retrain the Phase 2 ANN (the paper used 40k simulated
  trajectories for its wind estimator and 2k for altitude — real flight data will be far scarcer,
  so plan for many short repeated sessions rather than one long flight).
- Standardize a log format (e.g., one file per flight session with timestamp, imu, flow, tof,
  motif label) so Phase 4 training and Phase 7's regression suite can consume it uniformly.
- **Deliverable:** a structured, versioned real-flight dataset with motif labels and ToF ground
  truth, and a repeatable data-collection script/checklist.

### Phase 4 — Model transfer to real data
- Fine-tune (or retrain) the ANN estimator and observability estimator on real flight data;
  quantify the sim-to-real gap (does real sensor noise/drag/vibration change which motifs are
  "observable enough" versus the Phase 1 simulation predicted?).
- Re-validate against the withheld ToF ground truth as the accuracy benchmark.
- **Deliverable:** a "v1" real-data-tuned estimator, plus a written comparison of sim-predicted
  vs. real-observed observability/error, so we know if Phase 1's model needs correction.

### Phase 5 — AI-KF integration architecture
- Implement the observability-weighted fusion: convert the ANN's error/observability estimate
  into a time-varying measurement covariance, exactly mirroring the paper's Eq. 9–10 relationship
  (small `Ř` when the estimator is trustworthy, huge `Ř` when it isn't) — likely re-derived
  empirically for our own ANN rather than reusing the paper's exact exponential form verbatim.
- Resolve the injection-path question from Section 3 (firmware external-measurement queue vs.
  fully custom off-board UKF). Implement the double-counting correction (paper's "relevance
  ratio"/throttling step) so the AI-KF gracefully degrades to the stock EKF once converged and
  in agreement.
- **Deliverable:** an end-to-end pipeline, testable on logged (replayed) data first: raw
  logs in → fused z-estimate out, with the covariance-weighting logic implemented and unit
  tested against Phase 4's dataset.

### Phase 6 — Real-time closed-loop bench testing
- Characterize round-trip latency/jitter of the RPi 5 ↔ Crazyradio PA ↔ Crazyflie loop under
  load (streaming logs out, pushing fused measurements back in) — confirm it's fast enough
  (target ~30–100 Hz) before trusting it near real flight.
- Bench-test (props off / tethered) the full closed loop: does the firmware accept and use the
  injected measurement as expected? Does the fused estimate behave sanely under synthetic
  disturbances?
- **Deliverable:** latency/throughput measurements and a bench-test report; go/no-go for free
  flight.

### Phase 7 — Structured testing framework
- This is plan.md's "structured testing using Claude" item, formalized: build a reusable harness
  around `cflib` that can (a) fly a specified motif/trajectory script unattended, (b) capture and
  timestamp all relevant logs, (c) automatically run the AI-KF pipeline against those logs, and
  (d) produce a comparison report against baselines.
- Baselines to always compare against: stock EKF with full sensors (ToF enabled — the "easy"
  case), stock EKF with ToF withheld ("impaired," no AI help), and AI-KF with ToF withheld (our
  contribution).
- Metrics: altitude error variance/RMSE vs. withheld-ToF ground truth, convergence time from
  poor initialization, and stability during low- vs. high-observability segments (identified via
  the observability estimator itself).
- **Deliverable:** an automated regression suite that can be re-run after any model or firmware
  change and immediately shows whether performance improved or regressed.

### Phase 8 — Live flight demonstration ("grand prize")
- With the ToF measurement excluded from the estimator (ideally physically disabled/covered for
  the final demo, not just software-filtered, to make the claim airtight), fly a commanded
  altitude-hold/trajectory-tracking demo using only the AI-KF fused estimate.
- Run repeated trials across a few conditions (calm hover with periodic accel pulses to keep `z`
  observable, larger horizontal maneuvers, an induced poor-initialization case) and report
  tracking error and stability against the Phase 7 baselines.
- **Deliverable:** a demo flight (video + logged data) plus a written report quantifying the
  AI-KF's advantage over the impaired stock EKF, and how close it comes to the full-sensor
  baseline.

---

## 5. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Radio latency/dropout under sustained bidirectional streaming | Breaks real-time fusion loop | Characterize early (Phase 6) before committing to firmware-injection design; keep a "graceful hold last estimate" fallback |
| Sim-to-real gap in drag/noise parameters | ANN trained in Phase 2 underperforms on real data | Phase 4 exists specifically to catch and correct this before integration |
| Real flight dataset is much smaller than the paper's simulated one | ANN overfits / poor generalization | Favor many short structured sessions over few long ones; consider data augmentation or semi-supervised use of Phase 2's simulator to pretrain |
| Firmware modification complexity if external-measurement injection proves insufficient | Schedule risk | Treat "reuse existing external-measurement path" as the default; only fork firmware if forced, and treat that as its own sub-phase |
| Safety during "impaired sensor" free flight | Crash risk if AI-KF diverges | Bench-test thoroughly (Phase 6) before free flight; keep ToF enabled-but-ignored as an internal safety monitor until confidence is high, only physically disabling it for the final Phase 8 demo |
| Battery life limits data volume per session | Slower data collection in Phase 3 | Plan session count around it explicitly rather than assuming one long flight suffices |

---

## 6. Open decisions still needing a call

- **Firmware modification tolerance:** are we comfortable forking/patching `crazyflie-firmware`
  if the existing external-measurement CRTP path can't cleanly carry a custom time-varying
  covariance, or should the plan constrain itself to unmodified stock firmware plus the external
  measurement API as-is?
- **Timeline/deadline:** is there a target date (e.g., a competition, thesis defense, semester
  end) driving how many phases can run in parallel vs. sequentially?
- **Safety envelope:** is there a netted/tracking space available for free-flight testing once we
  get to Phases 6–8, and is there a mocap system available at all (even if not used as the primary
  ground truth), which could serve as an independent safety check during early impaired-sensor
  flights?

---

## 7. Reference

Methodology source: [references/summaries/bens_epic_paper.md](references/summaries/bens_epic_paper.md)
(Cellini, Boyacıoğlu, Lopez, van Breugel — BOUNDS + Augmented Information Kalman Filter,
arXiv:2511.08766).
