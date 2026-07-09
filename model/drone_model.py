import numpy as np
import scipy
import pandas as pd

class Drone:
    def __init__(self, measurement_names, dt=0.1):
        # Set state names in cartesian coordinates
        self.state_names = ('z', 'v_x', 'v_y', 'psi', 'w_x', 'w_y', 'w_x_dot', 'w_y_dot')
        self.dt = dt
        # Set measurement names & current measurement Z
        self.measurement_names = tuple(measurement_names)
        self.p = len(self.measurement_names)  # number of measurements
        self.Z = pd.DataFrame(np.zeros((1, len(self.measurement_names))), columns=self.measurement_names)
        # Set input names
        self.input_names = ('u_x', 'u_y', 'u_psi')  # number of inputs
        self.m = len(self.input_names)
        self.n = len(self.state_names)  # number of states

    def f_continuous(self, X, U, return_state_names=False):
        """ Continuous-time system dynamics: dx/dt = f(X, U) """
        if return_state_names:
            return ['z', 'v_x', 'v_y', 'psi', 'w_x', 'w_y', 'w_x_dot', 'w_y_dot']

        # Inputs
        u_x, u_y, u_psi = U

        # States
        z, v_x, v_y, psi, w_x, w_y, w_x_dot, w_y_dot = X

        # Altitude dynamics
        z_dot = 0.0

        # Heading
        psi_dot = u_psi

        # Acceleration dynamics (depends on model)
        v_x_dot = u_x + psi_dot * v_y
        v_y_dot = u_y - psi_dot * v_x

        # Wind dynamics
        w_x_dot = w_x_dot
        w_y_dot = w_y_dot

        # Wind accelerations
        w_x_ddot = 0.0
        w_y_ddot = 0.0

        # Combine into state derivative vector
        dxdt = np.array([z_dot,
                        v_x_dot,
                        v_y_dot,
                        psi_dot,
                        w_x_dot,
                        w_y_dot,
                        w_x_ddot,
                        w_y_ddot
                        ])

        return dxdt
    
    def h(self, X, U):
        """ Discrete-time measurement function.
        """

        # Inputs
        u_x, u_y, u_psi = U

        # States
        z, v_x, v_y, psi, w_x, w_y, w_x_dot, w_y_dot = X

        psi_dot = u_psi

        # Acceleration
        v_x_dot = u_x + psi_dot * v_y
        v_y_dot = u_y - psi_dot * v_x

        # Compute wind speed magnitude & direction
        w = np.sqrt(w_x ** 2 + w_y ** 2)
        zeta = np.arctan2(w_y, w_x)

        # Potential measurements
        g = np.sqrt(v_x ** 2 + v_y ** 2)  # ground speed
        beta = np.arctan2(v_y, v_x)  # ground speed angle
        a_x = v_x - w * np.cos(psi - zeta)  # apparent airflow in x direction
        a_y = v_y + w * np.sin(psi - zeta)  # apparent airflow in y direction
        a = np.sqrt(a_x ** 2 + a_y ** 2)  # apparent airflow magnitude
        gamma = np.arctan2(a_y, a_x)  # apparent airflow angle
        r = g / z  # optic flow magnitude
        r_x = v_x / z
        r_y = v_y / z
        alpha = np.arctan2(v_y_dot, v_x_dot)  # acceleration angle
        v_dot = np.sqrt(v_y_dot ** 2 + v_x_dot ** 2)

        # Create dict of potential measurements
        measurements = {'psi': psi,
                        'gamma': gamma,
                        'beta': beta,
                        'zeta': zeta,
                        'g': g,
                        'a': a,
                        'r': r,
                        'w': w,
                        'z': z,
                        'u_x': u_x,
                        'u_y': u_y,
                        'alpha': alpha,
                        'psi_dot': psi_dot,
                        'r_x': r_x,
                        'r_y': r_y,
                        'v_x_dot': v_x_dot,
                        'v_y_dot': v_y_dot,
                        'v_dot': v_dot,
                        'v_x': v_x,
                        'v_y': v_y,
                        'a_x': a_x,
                        'a_y': a_y,
                        'w_x': w_x,
                        'w_y': w_y}

        # Set current measurement based on measurement names
        measurements_df = pd.DataFrame(measurements, index=[0])
        self.Z = measurements_df.loc[:, self.measurement_names]
        Z = np.atleast_1d(self.Z.values.squeeze())

        # Return measurement
        return Z
    
    def rk4_discretize(self, f, x, u, dt):
        # Step 1: Compute k1, the first estimate of the state change (function evaluation at time t)
        k1 = f(x, u)  # k1 is the rate of change at the current state

        # Step 2: Compute k2, estimate of state change at time t + dt/2, based on k1
        # Perturb x by half the step size (dt/2) in the direction of k1
        k2 = f(x + 0.5 * dt * k1, u)  # k2 is the rate of change at t + dt/2

        # Step 3: Compute k3, another estimate of state change at time t + dt/2, based on k2
        # Perturb x by half the step size (dt/2) in the direction of k2
        k3 = f(x + 0.5 * dt * k2, u)  # k3 is the rate of change at t + dt/2 (but using k2)

        # Step 4: Compute k4, estimate of state change at time t + dt, based on k3
        # Perturb x by the full time step (dt) in the direction of k3
        k4 = f(x + dt * k3, u)  # k4 is the rate of change at t + dt

        # Step 5: Compute the weighted sum of the estimates (k1, k2, k3, k4) to update x
        # The final estimate is a weighted average of all k's
        x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next
    
    def f_discrete(self, x, u, *args):
        return self.rk4_discretize(self.f_continuous, x, u, self.dt)