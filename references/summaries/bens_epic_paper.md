# Discovering and exploiting active sensing motifs for estimation

**Authors:** Cellini, B.; Boyacıoğlu, B.; Lopez, A.P.; van Breugel, F.
**Year:** 2025 (arXiv preprint; v2 posted 2026-03-06)
**Journal / Venue:** arXiv preprint, arXiv:2511.08766v2 [eess.SY] (not yet peer-reviewed)
**DOI / URL:** https://arxiv.org/abs/2511.08766

---

## Key Finding
BOUNDS, a new empirical observability method based on Fisher information and the Cramér–Rao bound, can quantify how observable each individual state variable of a nonlinear system is along a trajectory, revealing that different state variables require distinct movement motifs and sensor sets to become estimable; building on this, the Augmented Information Kalman Filter (AI-KF) uses BOUNDS-derived observability estimates to fuse neural-network state estimates into a Kalman filter and outperforms standard Kalman filtering — especially under poor initialization and sparse/intermittent observability — in both simulation and real quadcopter flight data.

## Background & Motivation
Autonomous agents (organisms and machines) must estimate task-relevant states from sensory cues that are often nonlinearly entangled (e.g., optic flow conflates velocity and distance). Strategic movement ("active sensing") can decouple these variables, but no existing tool could quantify observability of individual state variables in partially observable nonlinear systems using real trajectory data with unknown control inputs — a prerequisite for systematically designing or discovering active sensing motifs. Separately, classical Kalman filters do not account for time-varying observability and can become unstable or fail to recover from a poor initial state guess when observability is weak, limiting their ability to exploit sporadic bouts of active sensing.

## Methods
- **BOUNDS pipeline (computational/empirical):** given a state trajectory (simulated or measured) and a forward-simulable dynamics/measurement model, use model predictive control (MPC) to reconstruct the control inputs that produce the trajectory; perturb each initial state variable by ±ε and record the resulting change in simulated measurements to build an empirical observability (Jacobian) matrix O; compute the Fisher information matrix F = OᵀR⁻¹O and a regularized ("Chernoff") inverse F⁻¹, whose diagonal gives the minimum error variance (inverse observability) per state variable; repeat in sliding time windows to get an observability time series.
- **Case study system:** a 3D kinematic quadcopter/flying-agent model (attitude, body-frame velocity, altitude, ambient wind speed/direction) with candidate sensors (heading, apparent airflow magnitude/direction, acceleration magnitude/direction, ventral optic flow magnitude/direction); evaluated across 4 movement motifs (accel/decel, heading turn, offset turn, small upwind turn) × 5 sensor subsets. Robustness checked with a more complex full dynamic (force/torque) quadcopter model.
- **Observability-informed state estimation:** feed-forward ANN state estimators (Hi) trained on sliding windows of measurements; a companion ANN observability estimator (Gi); an adaptive low-pass "observability filter" whose update rate is gated by estimated observability. Demonstrated for wind-direction estimation using N = 40,000 simulated training trajectories, split into observability-sorted bins.
- **Augmented Information Kalman Filter (AI-KF):** augments a Kalman filter's (here, Unscented KF) measurement vector with an ANN state estimate, and augments the measurement noise covariance with a time-varying term derived from the observability estimate (or an empirically fit error-covariance function), plus a "relevance ratio" correction to prevent double-counting information once the filter converges. Demonstrated for altitude + forward-velocity estimation from optic flow and acceleration, in simulation and validated against real outdoor flight data from a DJI Matrice 300 RTK quadcopter (~50 s trajectory, ventral camera optic flow, onboard accelerometer, GPS ground truth).

## Main Results
- With angular sensors only (heading, course, apparent airflow angle), wind direction is observable only during heading changes, not straight flight; larger turns help more, but small turns through the upwind direction suffice.
- Adding airflow magnitude makes offset turns sufficient for wind direction and enables ground-speed estimation; further adding optic-flow magnitude makes acceleration/deceleration motifs viable for both and renders altitude observable across all tested motifs; a direct ground-speed sensor makes wind direction and ground speed observable even during straight flight.
- An optic-flow + acceleration sensor set (no airflow sensor) supports ground speed and altitude estimation during acceleration/offset-turn motifs but never renders wind direction observable — confirming that observability is state- and sensor-set-specific, not a general property of a motif.
- Training the wind-direction ANN estimator on a smaller but more observable subset of data (top 40–70% by observability) outperformed training on randomly sampled data of the same size; ANN error correlated with, but sometimes beat, the theoretical Cramér–Rao bound (attributed to learned heuristics).
- The observability-gated adaptive filter tracked time-varying wind direction accurately when turns occurred frequently enough, despite the raw ANN estimate being accurate only during turns.
- In simulation, the AI-UKF converged quickly to true altitude/forward velocity across a wide range of poor initial guesses, whereas the standard UKF often failed to converge or was thrown off by biased sensor noise; the AI-UKF's advantage was largest at moderate (low but nonzero) acceleration levels.
- On real outdoor quadcopter data, the AI-UKF converged to correct altitude/velocity within seconds across multiple initial conditions, while the standard UKF failed to converge within the full ~50 s trajectory for some initializations.

## Evidence Strength
Primarily a methods/computational paper. The core observability claims are derived analytically/empirically from Fisher information on models (a toy system plus kinematic and full-dynamic quadcopter models), not statistical inference, so classical significance testing doesn't apply; robustness was checked by repeating the analysis with an alternative dynamics model, yielding consistent conclusions. The ANN estimator comparisons are backed by large simulated datasets (40,000 trajectories for wind direction, 2,000 for altitude) with train/test splits. The real-world validation of the AI-KF, however, is a single ~50 s flight on one quadcopter under relatively benign outdoor conditions — a proof-of-concept demonstration rather than a replicated, multi-trial empirical validation.

## Limitations & Caveats
- Observability results are contingent on the chosen dynamics/measurement model and noise assumptions; the empirical perturbation approach requires continuous, well-defined measurement functions (careful handling of angle wrapping and magnitude positivity), or it yields misleading Jacobians.
- The method estimates observability (based on future/enclosing measurement windows), not constructability/posterior estimation error using only past measurements, though the authors argue the two are similar for short windows and small process noise (assumed ≈0 throughout).
- The AI-KF's double-counting correction (the "relevance ratio" / throttling function) is an ad hoc fix, not a fully principled statistical solution, and is only demonstrated for a single augmented state variable at a time.
- Real-world validation is limited to one quadcopter platform and one short flight segment over flat terrain with moderate wind; generalization to other vehicles, noisier/higher-dimensional systems, or biological organisms is proposed but not empirically tested here.
- An earlier version of this work was posted to bioRxiv but "was never submitted for publication" per the authors' own footnote, and the current version is an arXiv preprint that has not undergone peer review.

## Open Questions
- Do biological neural circuits implement something like observability-weighted integration (as in the AI-KF) when combining sporadic active-sensing bouts into continuous estimates? Proposed as testable via neural recordings during naturalistic behavior.
- Would closed-loop experiments that block or disrupt heading turns in flies confirm the prediction that wind-direction estimation specifically degrades without heading changes?
- How should the framework generalize to multiple simultaneously augmented state estimators, higher-dimensional systems, or fully data-driven (non-parametric) dynamics models?
- How should hyperparameters (regularization λ, window size ω, and the AI-KF's ρmin/ρmax/c/ε) be chosen systematically in new applications, beyond the heuristics given?
- How well does the AI-KF generalize beyond a single quadcopter/trajectory to broader GPS-denied navigation scenarios, which motivate but are not fully tested by this work?

## Relevance to This Project
No CLAUDE.md was found in the project directory, so I could not confirm this against documented project goals. However, based on the project directory name ("ai_kalman_on_crazyflie"), this paper is likely highly relevant: it introduces the Augmented Information Kalman Filter (AI-KF), a concrete method for fusing neural-network-based state estimates into an (Unscented) Kalman Filter via observability-derived, time-varying measurement noise covariances, demonstrated specifically on a quadcopter estimating altitude, ground speed, and wind direction from limited sensors in GPS-denied conditions — closely matching what a Crazyflie-based Kalman filtering project would likely need. Consider adding a CLAUDE.md describing the project's goals so future summaries can draw a more precise connection.

## Keywords
active sensing, observability, fisher information, cramér-rao bound, kalman filter, neural network state estimation, quadcopter, wind estimation, gps-denied navigation, nonlinear systems
