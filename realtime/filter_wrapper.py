"""Online (tick-ready) AI-KF: drone_model_np's model + EKF, with the ANN's z_pred spliced in as
a 5th augmented pseudo-measurement (4 base + 1 augmented), mirroring
references/A_planar_drone_AI_UKF.ipynb's h_aug/R_aug pattern but using EKF.forward_update
(already tick-ready) instead of the batch-only UKF, and against drone_model_np's 7-state
body_level model (matching train.py's actual training model) rather than planar_drone.py's 2D
model this used to target.

references/unscented_kalman_filter.py would need a predict/update refactor before it could run
here; per the project plan, start with EKF and only switch if an offline UKF-vs-EKF bake-off
shows EKF doesn't reliably converge from a poor initial z guess.
"""

import numpy as np

from realtime.drone_model_np import STATE_NAMES, f, h_camera_imu
from references.extended_kalman_filter import EKF
from realtime.config import DT, Q_DIAG, R_BASE_DIAG

Z_STATE_INDEX = STATE_NAMES.index("z")


class AIKFFilter:
    def __init__(self, x0, dt=DT, q_diag=Q_DIAG, r_base_diag=R_BASE_DIAG, p0_diag=None):
        self.f = f
        self._h_base = h_camera_imu  # ['meas_r_x', 'meas_r_y', 'meas_v_x_dot', 'meas_v_y_dot']

        n = len(x0)
        p0_diag = p0_diag if p0_diag is not None else [1.0] * n
        Q = np.diag(q_diag)
        R_base = np.diag(r_base_diag)

        # Augmented R: base 4x4 plus one slot for the ANN's z_pred pseudo-measurement.
        self.R_aug = np.eye(R_base.shape[0] + 1)
        self.R_aug[:4, :4] = R_base

        self.ekf = EKF(
            f=self.f,
            h=self._h_aug,
            x0=np.array(x0, dtype=np.float64),
            u0=np.zeros(4),  # [u_x, u_y, u_psi, u_z]
            P0=np.diag(p0_diag),
            Q=Q,
            R=self.R_aug,
            dynamics_type="continuous",
            discretization_timestep=dt,
        )

    def _h_aug(self, x_vec, u_vec):
        base = self._h_base(x_vec, u_vec)
        return np.append(base, x_vec[Z_STATE_INDEX])  # z state is the augmented measurement's expected value

    def tick(self, base_measurements, z_pred, r_aug_z, u):
        """base_measurements: [meas_r_x, meas_r_y, meas_v_x_dot, meas_v_y_dot]
        z_pred: ANN's altitude estimate for the current window
        r_aug_z: time-varying variance for z_pred (see ann_estimator.compute_r_aug_z)
        u: [u_x, u_y, u_psi, u_z] control input at this tick
        """
        y_aug = np.append(np.asarray(base_measurements, dtype=np.float64), z_pred)
        R = self.R_aug.copy()
        R[-1, -1] = r_aug_z
        self.ekf.forward_update(y_aug, np.asarray(u, dtype=np.float64), R=R)
        return self.ekf.x.copy(), self.ekf.P.copy()

    @property
    def z_estimate(self):
        return self.ekf.x[Z_STATE_INDEX]

    @property
    def z_variance(self):
        return self.ekf.P[Z_STATE_INDEX, Z_STATE_INDEX]
