"""Online (tick-ready) AI-KF: planar_drone's model + EKF, with the ANN's z_pred spliced in as a
6th augmented pseudo-measurement, mirroring references/A_planar_drone_AI_UKF.ipynb's h_aug/R_aug
pattern but using EKF.forward_update (already tick-ready) instead of the batch-only UKF.

references/unscented_kalman_filter.py would need a predict/update refactor before it could run
here; per the project plan, start with EKF and only switch if an offline UKF-vs-EKF bake-off
shows EKF doesn't reliably converge from a poor initial z guess.
"""

import numpy as np

from references.extended_kalman_filter import EKF
from references.planar_drone import F, H
from realtime.config import DT, Q_DIAG, R_BASE_DIAG


class AIKFFilter:
    def __init__(self, x0, dt=DT, q_diag=Q_DIAG, r_base_diag=R_BASE_DIAG, p0_diag=None):
        self.f = F(k=None).f  # 7-state: [theta, theta_dot, x, x_dot, z, z_dot, k]
        self._h_base = H("h_camera_imu", k=None).h  # ['optic_flow','theta','theta_dot','accel_x','accel_z']

        n = len(x0)
        p0_diag = p0_diag if p0_diag is not None else [1.0] * n
        Q = np.diag(q_diag)
        R_base = np.diag(r_base_diag)

        # Augmented R: base 5x5 plus one slot for the ANN's z_pred pseudo-measurement.
        self.R_aug = np.eye(R_base.shape[0] + 1)
        self.R_aug[:5, :5] = R_base

        self.ekf = EKF(
            f=self.f,
            h=self._h_aug,
            x0=np.array(x0, dtype=np.float64),
            u0=np.zeros(2),
            P0=np.diag(p0_diag),
            Q=Q,
            R=self.R_aug,
            dynamics_type="continuous",
            discretization_timestep=dt,
        )

    def _h_aug(self, x_vec, u_vec):
        base = self._h_base(x_vec, u_vec)
        return np.append(base, x_vec[4])  # z state is the augmented "measurement"'s expected value

    def tick(self, base_measurements, z_pred, r_aug_z, u):
        """base_measurements: [optic_flow, theta, theta_dot, accel_x, accel_z]
        z_pred: ANN's altitude estimate for the current window
        r_aug_z: time-varying variance for z_pred (see ann_estimator.compute_r_aug_z)
        u: [j1, j2] control input at this tick
        """
        y_aug = np.append(np.asarray(base_measurements, dtype=np.float64), z_pred)
        R = self.R_aug.copy()
        R[-1, -1] = r_aug_z
        self.ekf.forward_update(y_aug, np.asarray(u, dtype=np.float64), R=R)
        return self.ekf.x.copy(), self.ekf.P.copy()

    @property
    def z_estimate(self):
        return self.ekf.x[4]

    @property
    def z_variance(self):
        return self.ekf.P[4, 4]
