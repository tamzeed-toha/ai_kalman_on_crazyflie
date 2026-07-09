# Reference Index

Summaries of papers relevant to this project.
Each entry links to a full structured summary in `references/summaries/`.

---

### Cellini2025 — Discovering and exploiting active sensing motifs

Introduces BOUNDS, an empirical Fisher-information/Cramér-Rao-bound method for quantifying per-state-variable observability along nonlinear trajectories, and the Augmented Information Kalman Filter (AI-KF), which fuses neural-network state estimates into a Kalman filter using observability-derived, time-varying noise covariances. Demonstrated on a simulated flying-agent/quadcopter model (wind direction, altitude, ground speed estimation from limited sensors) and validated on real GPS-denied quadcopter flight data, where the AI-KF converges faster and more robustly than a standard Unscented Kalman Filter, especially under poor initialization. Included because it directly targets Kalman-filter state estimation for a flying quadcopter with limited sensing — closely aligned with this project's likely focus (per the "ai_kalman_on_crazyflie" project name).
→ `references/summaries/bens_epic_paper.md`
