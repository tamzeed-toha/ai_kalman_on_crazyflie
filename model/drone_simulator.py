import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from casadi import sin, cos, atan2
from utils import figure_functions as ff
from pybounds import Simulator


class DroneModel:
    def __init__(self, frame='body_level', moving_frame='on'):
        self.frame = frame
        self.moving_frame = moving_frame

        # State names
        self.state_names = ['x',  # x position in inertial frame [m]
                            'y',  # y position in inertial frame [m]
                            'z',  # elevation
                            'v_x',  # x velocity in global frame (parallel to heading) [m/s]
                            'v_y',  # y velocity in global frame (perpendicular to heading) [m/s]
                            'v_z',  # z velocity in global frame [m/s]
                            'psi',  # yaw in body-level frame (vehicle-1) [rad]
                            'w',  # wind speed in XY-plane [ms]
                            'zeta',  # wind direction in XY-plane[rad]
                            'k_x',  # x-velocity input motor calibration parameter
                            'k_y',  # y-velocity input motor calibration parameter
                            'k_psi',  # yaw angular velocity input motor calibration parameter
                            ]

        # Polar state names
        replacement_polar_states = {'v_x': 'g', 'v_y': 'beta'}
        self.state_names_polar = self.state_names.copy()
        self.state_names_polar = [replacement_polar_states.get(x, x) for x in self.state_names_polar]

        # if self.frame == 'body_level':
        #     replacement_body_level_states = {'v_x': 'v_para', 'v_y': 'v_perp'}
        #     self.state_names = [replacement_body_level_states.get(x, x) for x in self.state_names]

        # Input names
        self.input_names = ['u_x',  # acceleration in parallel direction
                            'u_y',  # acceleration in perpendicular direction
                            'u_psi',  # yaw angular velocity [rad/s]
                            'u_z',  # change elevation
                            'u_w',  # change wind speed
                            'u_zeta',  # change wind direction
                            ]

        # Measurement names
        self.measurement_names = ['v_x_dot', 'v_y_dot',
                                  'r_x', 'r_y', 'r', 'g', 'beta',
                                  'a_x', 'a_y', 'a', 'gamma',
                                  'w_x', 'w_y']

        self.measurement_names = self.state_names + self.input_names + self.measurement_names

    def f(self, X, U):
        """ Dynamic model.
        """

        # States
        x, y, z, v_x, v_y, v_z, psi, w, zeta, k_x, k_y, k_psi = X

        # Inputs
        u_x, u_y, u_psi, u_z, u_w, u_zeta = U

        # Heading
        psi_dot = u_psi * k_psi

        # z-velocity
        v_z_dot = u_z
        z_dot = v_z

        # xy-velocity
        if self.frame == 'global':
            # Velocity in global frame
            v_x_dot = (u_x * k_x) * np.cos(psi) - (u_y * k_y) * np.sin(psi)
            v_y_dot = (u_x * k_x) * np.sin(psi) + (u_y * k_y) * np.cos(psi)

            # Position in global frame
            x_dot = v_x
            y_dot = v_y

        elif self.frame == 'body_level':
            # Velocity in body-level frame
            if self.moving_frame == 'off':
                v_x_dot = u_x * k_x
                v_y_dot = u_y * k_y

            elif self.moving_frame == 'on':
                v_x_dot = u_x * k_x + 1.0 * psi_dot * v_y
                v_y_dot = u_y * k_y - 1.0 * psi_dot * v_x

            else:
                raise ValueError('Moving frame must be on or off')

            # Position in global frame
            x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
            y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)

        else:
            raise ValueError('Frame must be global or body_level.')

        # Wind in global frame
        w_dot = u_w
        zeta_dot = u_zeta

        # Motor calibration parameters
        k_x_dot = k_x * 0.0
        k_y_dot = k_y * 0.0
        k_psi_dot = k_psi * 0.0

        # Package and return xdot
        x_dot = [x_dot, y_dot, z_dot,
                 v_x_dot, v_y_dot, v_z_dot,
                 psi_dot,
                 w_dot, zeta_dot,
                 k_x_dot, k_y_dot, k_psi_dot
                 ]

        return x_dot

    def h(self, X, U):
        """ Measurement model.
        """

        # States
        x, y, z, v_x, v_y, v_z, psi, w, zeta, k_x, k_y, k_psi = X

        # Inputs
        u_x, u_y, u_psi, u_z, u_w, u_zeta = U

        # Dynamics
        (x_dot, y_dot, z_dot, v_x_dot, v_y_dot, v_z_dot, psi_dot, w_dot, zeta_dot,
         k_x_dot, k_y_dot, k_psi_dot) = self.f(X, U)

        # Body-level velocity
        if self.frame == 'global':
            v_para = v_x * np.cos(psi) + v_y * np.sin(psi)
            v_perp = -v_x * np.sin(psi) + v_y * np.cos(psi)

        elif self.frame == 'body_level':
            v_para = v_x
            v_perp = v_y

        else:
            raise ValueError('Frame must be global or body_level.')

        # Ground speed & course direction in body-level frame
        g = np.sqrt(v_para ** 2 + v_perp ** 2)
        r_x = v_para / z
        r_y = v_perp / z
        # r_x = v_x / z
        # r_y = v_y / z
        r = g / z
        beta = np.arctan2(v_perp, v_para)

        # Apparent airflow
        a_x = v_para - w * np.cos(psi - zeta)
        a_y = v_perp + w * np.sin(psi - zeta)
        a = np.sqrt(a_x ** 2 + a_y ** 2)
        gamma = np.arctan2(a_y, a_x)

        # # Acceleration
        # v_dot = np.sqrt(v_x_dot ** 2 + v_y_dot ** 2)
        # alpha = np.arctan2(v_y, v_x)

        # Wind
        w_x = w * np.cos(zeta)
        w_y = w * np.sin(zeta)

        # Unwrap angles
        if np.array(psi).ndim > 0:
            if np.array(psi).shape[0] > 1:
                X[6] = np.unwrap(psi)
                beta = np.unwrap(beta)
                gamma = np.unwrap(gamma)

        Y = (list(X) + list(U) +
             [v_x_dot, v_y_dot, r_x, r_y, r, g, beta, a_x, a_y, a, gamma, w_x, w_y])

        return Y

    # Coordinate transformation function
    def z_function(self, X):
        # Old states as sympy variables
        x, y, z, v_x, v_y, v_z, psi, w, zeta, k_x, k_y, k_psi = X

        # Body-level velocity
        if self.frame == 'global':
            v_para = v_x * sp.cos(psi) + v_y * sp.sin(psi)
            v_perp = -v_x * sp.sin(psi) + v_y * sp.cos(psi)

        elif self.frame == 'body_level':
            v_para = v_x
            v_perp = v_y

        else:
            raise ValueError('Frame must be global or body_level.')

        # Expressions for new states in terms of old states
        g = (v_para ** 2 + v_perp ** 2) ** (1 / 2)  # ground speed magnitude
        beta = sp.atan(v_perp / v_para)  # ground speed angle

        # Define new state vector
        z = [x, y, z, g, beta, v_z, psi, w, zeta, k_x, k_y, k_psi]
        return sp.Matrix(z)


class DroneSimulator(Simulator):
    def __init__(self, frame=None, moving_frame=None, dt=0.1,
                 mpc_horizon=10, r_u=1e-4, control_mode='velocity_body_level'):
        """ Set up simulation.
        """
        self.dynamics = DroneModel(frame=frame, moving_frame=moving_frame)
        super().__init__(self.dynamics.f, self.dynamics.h, dt=dt, mpc_horizon=mpc_horizon,
                         state_names=self.dynamics.state_names,
                         input_names=self.dynamics.input_names,
                         measurement_names=self.dynamics.measurement_names)

        # Define cost function
        # Wind cost
        wind_cost = ((self.model.x['w'] - self.model.tvp['w_set']) ** 2 +
                     (self.model.x['zeta'] - self.model.tvp['zeta_set']) ** 2)

        # z cost
        z_cost = (self.model.x['z'] - self.model.tvp['z_set']) ** 2

        # Heading cost
        psi_cost = (self.model.x['psi'] - self.model.tvp['psi_set']) ** 2

        # xy velocity or position cost
        self.control_mode = control_mode
        if self.control_mode == 'velocity_body_level':
            if self.dynamics.frame == 'global':
                # Body-level velocity
                v_para = (self.model.x['v_x'] * cos(self.model.x['psi']) +
                          self.model.x['v_y'] * sin(self.model.x['psi']))
                v_perp = (-self.model.x['v_x'] * sin(self.model.x['psi']) +
                          self.model.x['v_y'] * cos( self.model.x['psi']))

                v_cost = ((v_para - self.model.tvp['v_x_set']) ** 2 +
                          (v_perp - self.model.tvp['v_y_set']) ** 2)

            elif self.dynamics.frame == 'body_level':
                v_cost = ((self.model.x['v_x'] - self.model.tvp['v_x_set']) ** 2 +
                          (self.model.x['v_y'] - self.model.tvp['v_y_set']) ** 2)

            else:
                raise ValueError('Frame must be body_level or body_level.')

        elif self.control_mode == 'position_global':
            v_cost = ((self.model.x['x'] - self.model.tvp['x_set']) ** 2 +
                      (self.model.x['y'] - self.model.tvp['y_set']) ** 2)

        else:
            raise Exception('Control mode not available')

        # Total cost
        total_cost = wind_cost + z_cost + psi_cost + v_cost

        # Set cost function
        self.mpc.set_objective(mterm=total_cost, lterm=total_cost)
        self.mpc.set_rterm(u_x=r_u * 1e-2, u_y=r_u * 1e-2, u_psi=r_u, u_z=r_u * 1e-2, u_w=r_u, u_zeta=r_u * 1e1)

        # Place limit on states
        self.mpc.bounds['lower', '_x', 'z'] = 0

    def update_setpoint(self, x=None, y=None, z=None, v_x=None, v_y=None, psi=None, w=None, zeta=None):
        """ Set the set-point variables.
        """

        # Set time
        T = self.dt * (len(w) - 1)
        tsim = np.arange(0, T + self.dt / 2, step=self.dt)

        # Set control setpoints
        if self.control_mode == 'velocity_body_level':  # control the body-level x & y velocities
            if (v_x is None) or (v_y is None):  # must set velocities
                raise Exception('x or y velocity not set')
            else:  # x & y don't matter, set to 0
                x = 0.0 * np.ones_like(tsim)
                y = 0.0 * np.ones_like(tsim)

        elif self.control_mode == 'position_global':  # control the global position
            if (x is None) or (y is None):  # must set positions
                raise Exception('x or y position not set')
            else:  # v_x & v_y don't matter, set to 0
                pass
                v_x = 0.0 * np.ones_like(tsim)
                v_y = 0.0 * np.ones_like(tsim)
        else:
            raise Exception('Control mode not available')

        if self.dynamics.frame == 'global':
            # Global velocity
            g = np.sqrt(v_x**2 + v_y**2)
            v_x_global = g * np.cos(psi)
            v_y_global = g * np.sin(psi)
            v_x[0] = v_x_global[0]
            v_y[0] = v_y_global[0]

        # Define the set-points to follow
        setpoint = {'x': x,
                    'y': y,
                    'z': z,
                    'v_x': v_x,
                    'v_y': v_y,
                    'v_z': 0.0*np.ones_like(tsim),
                    'psi': psi,
                    'w': w,
                    'zeta': zeta,
                    'k_x': 1.0*np.ones_like(tsim),
                    'k_y': 1.0*np.ones_like(tsim),
                    'k_psi': 1.0*np.ones_like(tsim),
                    }

        # if self.dynamics.frame == 'body_level':
        #     setpoint['v_para'] = setpoint.pop('v_x')
        #     setpoint['v_perp'] = setpoint.pop('v_y')

        # Update the simulator set-point
        self.update_dict(setpoint, name='setpoint')

    def plot_trajectory(self, start_index=0, nskip=0, size_radius=0.1, dpi=200):
        """ Plot the trajectory.
        """

        fig, ax = plt.subplots(1, 1, figsize=(3 * 1, 3 * 1), dpi=dpi)

        x = self.y['x'][start_index:]
        y = self.y['y'][start_index:]
        heading = self.y['psi'][start_index:]
        time = self.time[start_index:]

        ff.plot_trajectory(x, y, heading,
                           color=time,
                           ax=ax,
                           size_radius=size_radius,
                           nskip=nskip)

        ax.set_axis_off()
