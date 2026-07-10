"""
Square + zigzag test flight: a trajectory deliberately more complex than anything in the
systematic training-data library (data_collection/trajectories.py) -- a closed 4-leg square
with an accel/cruise/decel run down each side and a pure in-place 90-degree turn at each corner,
then a zigzag phase over the same area -- while running the real-time AI-KF
(realtime/) to estimate altitude live, with a live dual plot:
  (1) commanded/reference/AI-KF-estimated height vs. time
  (2) top-down (x/y) flight path colored by horizontal acceleration magnitude

Both phases keep horizontal acceleration flowing almost continuously (straight-leg
accel/cruise/decel, zigzag corners) rather than long constant-velocity or hover stretches, since
per the paper z is only observable during horizontal acceleration -- so this trajectory is one
the AI-KF has a real chance of tracking throughout, not by accident.

Always loads the MOST RECENTLY MODIFIED trained model in --models-dir (see
realtime.ann_estimator.find_latest_model) -- retrain via train.py before running this if you
want fresh weights; no code change needed here for that to take effect.

SPACE REQUIREMENT: verified by numeric integration of the actual commanded velocity profile
(see the trajectory-footprint check in chat history / rerun the same technique after any
parameter change here) -- default parameters reach ~1.13m max excursion from the takeoff point
(the square's diagonal, side_length_m=0.8) and close back to within a few mm of the start point.
This script does NOT implement geofence safety (only realtime.safety_monitor's battery/obstacle
checks, no position bound), so use a clear space of at least ~3m x 3m and supervise closely,
unlike the automated data_collection/ campaign which does have a geofence.

Run from the repo root, after retraining if you want fresh weights:
    source ~/.venv/bin/activate
    python train.py --simulated-trajectories-dir real_trajectories --models-dir models_real
    python test_flight_square_zigzag.py --models-dir models_real
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DATA_COLLECTION_DIR = REPO_ROOT / "data_collection"
for _p in (REPO_ROOT, DATA_COLLECTION_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cf_link  # data_collection -- flat import, arm_cf/disarm_cf/send_stop_setpoint/init_drivers
from trajectories import VelCmd, _leg_peak_and_cruise, _trapezoid_speed, zigzag_profile  # data_collection

from realtime.ann_estimator import AltitudeANNEstimator
from realtime.config import DEFAULT_URI
from realtime.config import DT as FUSION_DT
from realtime.crazyflie_link import CrazyflieLink
from realtime.filter_wrapper import AIKFFilter
from realtime.fusion_loop import FusionLoop
from realtime.safety_monitor import SafetyMonitor

DEFAULT_HEIGHT_M = 0.4
CONTROL_RATE_HZ = 50.0
TAKEOFF_S = 1.5
PLOT_REFRESH_EVERY_N_TICKS = 5  # ~10Hz plot refresh at a 50Hz control loop


# ---------------------------------------------------------------------------
# Trajectory: closed square (accel/cruise/decel legs + in-place corner turns) + zigzag
# ---------------------------------------------------------------------------

def square_profile(params, height_m):
    """Closed n-sided polygon (default square): each leg is a trapezoidal accel/cruise/decel
    straight run (body-forward only, VelCmd.vx), with a pure in-place turn (zero translation) at
    each corner -- so the path closes exactly regardless of leg length, unlike motifs that turn
    while translating (no residual-displacement correction needed)."""
    side_length_m = params.get("side_length_m", 1.0)
    cruise_speed_mps = params.get("cruise_speed_mps", 0.8)
    accel_mps2 = params.get("accel_mps2", 1.2)
    yaw_rate_deg_s = params.get("yaw_rate_deg_s", 90.0)
    corner_pause_s = params.get("corner_pause_s", 0.3)
    n_sides = params.get("n_sides", 4)
    turn_deg = 360.0 / n_sides

    leg_peak, leg_cruise_s = _leg_peak_and_cruise(side_length_m, cruise_speed_mps, accel_mps2)
    leg_ramp_time_s = leg_peak / accel_mps2
    leg_duration_s = 2.0 * leg_ramp_time_s + leg_cruise_s
    turn_duration_s = turn_deg / yaw_rate_deg_s

    segments = []
    for _ in range(n_sides):
        segments += [(leg_duration_s, "leg"), (corner_pause_s, "pause"),
                     (turn_duration_s, "turn"), (corner_pause_s, "pause")]

    boundaries = []
    cum = 0.0
    for seg_dur, kind in segments:
        boundaries.append((cum, cum + seg_dur, kind))
        cum += seg_dur
    duration_s = cum

    def vel_fn(t):
        for start, end, kind in boundaries:
            if start <= t < end:
                if kind == "leg":
                    speed = _trapezoid_speed(t - start, leg_ramp_time_s, leg_cruise_s, leg_peak)
                    return VelCmd(speed, 0.0, 0.0, height_m, "square_leg")
                if kind == "turn":
                    return VelCmd(0.0, 0.0, yaw_rate_deg_s, height_m, "square_corner_turn")
                return VelCmd(0.0, 0.0, 0.0, height_m, "square_corner_pause")
        return VelCmd(0.0, 0.0, 0.0, height_m, "pause_end")

    return duration_s, vel_fn


def build_test_trajectory(height_m):
    square_params = dict(side_length_m=0.8, cruise_speed_mps=0.8, accel_mps2=1.2, yaw_rate_deg_s=90.0)
    zigzag_params = dict(leg_distance_m=0.4, leg_speed_mps=1.0, leg_accel_mps2=1.5, n_legs=3, zigzag_angle_deg=35.0)
    pause_s = 1.0

    square_duration_s, square_vel_fn = square_profile(square_params, height_m)
    zigzag_duration_s, zigzag_vel_fn = zigzag_profile(zigzag_params, height_m)

    boundaries = [
        (0.0, square_duration_s, square_vel_fn),
        (square_duration_s, square_duration_s + pause_s,
         lambda t: VelCmd(0.0, 0.0, 0.0, height_m, "pause_between_phases")),
        (square_duration_s + pause_s, square_duration_s + pause_s + zigzag_duration_s, zigzag_vel_fn),
    ]
    total_duration_s = square_duration_s + pause_s + zigzag_duration_s

    def vel_fn(t):
        for start, end, fn in boundaries:
            if start <= t < end:
                return fn(t - start)
        return VelCmd(0.0, 0.0, 0.0, height_m, "pause_end")

    return total_duration_s, vel_fn


# ---------------------------------------------------------------------------
# Control-input-from-known-trajectory: exact, since this script commands the trajectory itself
# ---------------------------------------------------------------------------

class TrajectoryControlInputProvider:
    """Supplies the AI-KF's [u_x, u_y, u_psi, u_z] control input by finite-differencing this
    script's OWN commanded velocity profile at the current elapsed trajectory time -- exact for
    a known scripted trajectory, unlike realtime.sensor_conversion.motor_commands_to_control_input
    (which has no safe default for arbitrary/piloted flight). Valid here specifically because
    every leg in this trajectory has yaw_rate=0 while translating (or zero velocity while
    turning), so the model's psi_dot*v_y/v_x cross-terms are always zero and u_x=dvx/dt,
    u_y=dvy/dt, u_psi=commanded yaw rate directly, u_z=0 (height is held constant throughout)."""

    def __init__(self, vel_fn, get_elapsed_s, dt=FUSION_DT):
        self.vel_fn = vel_fn
        self.get_elapsed_s = get_elapsed_s
        self.dt = dt

    def __call__(self):
        t = self.get_elapsed_s()
        cmd_now = self.vel_fn(t)
        cmd_prev = self.vel_fn(max(0.0, t - self.dt))
        u_x = (cmd_now.vx - cmd_prev.vx) / self.dt
        u_y = (cmd_now.vy - cmd_prev.vy) / self.dt
        u_psi = np.deg2rad(cmd_now.yaw_rate)
        u_z = 0.0
        return np.array([u_x, u_y, u_psi, u_z])


# ---------------------------------------------------------------------------
# Live recording + dual plot
# ---------------------------------------------------------------------------

class LiveRecorder:
    """Polled from the main flight loop (not from FusionLoop's worker thread) -- simple attribute
    reads/writes are fine under the GIL for this monitoring purpose, no lock needed."""

    def __init__(self):
        self.t0 = time.time()
        self.time_s = []
        self.zrange_m = []
        self.z_estimate_m = []
        self.xy = []
        self.accel_mag = []
        self.latest_zrange_m = None
        self.latest_xy = (0.0, 0.0)
        self.latest_accel_mag = 0.0

    def on_imu_flow_extra(self, timestamp, data):
        self.latest_zrange_m = data["range.zrange"] / 1000.0

    def on_state_pv_extra(self, timestamp, data):
        self.latest_xy = (data["stateEstimate.x"], data["stateEstimate.y"])

    def sample(self, fusion):
        t = time.time() - self.t0
        r_x, r_y, v_x_dot, v_y_dot = getattr(fusion, "_latest_measurements", (0.0, 0.0, 0.0, 0.0))
        self.latest_accel_mag = float(np.hypot(v_x_dot, v_y_dot))
        self.time_s.append(t)
        self.zrange_m.append(self.latest_zrange_m)
        self.z_estimate_m.append(fusion.filter.z_estimate)
        self.xy.append(self.latest_xy)
        self.accel_mag.append(self.latest_accel_mag)


class LivePlot:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax_height, self.ax_xy) = plt.subplots(1, 2, figsize=(11, 5))

        self.ax_height.set_xlabel("time [s]")
        self.ax_height.set_ylabel("height [m]")
        self.ax_height.set_title("Height: reference (z-range) vs. AI-KF estimate")
        (self.line_zrange,) = self.ax_height.plot([], [], label="z-range (reference)", color="tab:gray")
        (self.line_estimate,) = self.ax_height.plot([], [], label="AI-KF estimate", color="tab:red")
        self.ax_height.legend(loc="upper right")

        self.ax_xy.set_xlabel("x [m]")
        self.ax_xy.set_ylabel("y [m]")
        self.ax_xy.set_title("Top-down trajectory, colored by |horizontal accel|")
        self.ax_xy.set_aspect("equal", adjustable="datalim")
        self.scatter_xy = self.ax_xy.scatter([], [], c=[], cmap="viridis", s=8)
        self.colorbar = self.fig.colorbar(self.scatter_xy, ax=self.ax_xy, label="accel magnitude [m/s^2]")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, recorder: LiveRecorder):
        self.line_zrange.set_data(recorder.time_s, recorder.zrange_m)
        self.line_estimate.set_data(recorder.time_s, recorder.z_estimate_m)
        self.ax_height.relim()
        self.ax_height.autoscale_view()

        if recorder.xy:
            xy = np.array(recorder.xy)
            self.scatter_xy.set_offsets(xy)
            self.scatter_xy.set_array(np.array(recorder.accel_mag))
            if len(recorder.accel_mag) > 1:
                self.scatter_xy.set_clim(min(recorder.accel_mag), max(recorder.accel_mag) + 1e-6)
            self.ax_xy.relim()
            self.ax_xy.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, path):
        self.fig.savefig(path, dpi=150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models_real"))
    parser.add_argument("--model-name", default=None,
                         help="Pin a specific model; default auto-discovers the most recent one")
    parser.add_argument("--initial-z", type=float, default=DEFAULT_HEIGHT_M,
                         help="Initial altitude guess fed to the AI-KF (m)")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT_M, help="Flight height (m)")
    parser.add_argument("--save-plot", default=str(REPO_ROOT / "test_flight_square_zigzag.png"))
    return parser.parse_args()


def main():
    args = parse_args()

    cf_link.initialize_cflib_drivers()

    print(f"Loading most recent model in {args.models_dir} ...")
    ann = AltitudeANNEstimator(model_dir=args.models_dir, model_name=args.model_name)
    print(f"Loaded '{ann.model_name}' (window_size={ann.window_size}, channels={ann.channels})")

    x0 = [0.0, 0.0, args.initial_z, 0.0, 0.0, 0.0, 0.0]  # [x, y, z, v_x, v_y, v_z, psi]
    aikf_filter = AIKFFilter(x0=x0)
    safety = SafetyMonitor()

    total_duration_s, vel_fn = build_test_trajectory(args.height)
    print(f"Trajectory built: {total_duration_s:.1f}s total (square + zigzag)")

    recorder = LiveRecorder()
    trajectory_state = {"start_time": None}

    def get_elapsed_s():
        if trajectory_state["start_time"] is None:
            return 0.0
        return time.time() - trajectory_state["start_time"]

    control_input_provider = TrajectoryControlInputProvider(vel_fn, get_elapsed_s)

    link = CrazyflieLink(args.uri)
    fusion = FusionLoop(link=link, ann_estimator=ann, aikf_filter=aikf_filter,
                         safety_monitor=safety, control_input_provider=control_input_provider)

    land_requested = {"flag": False}

    def handle_sigint(signum, frame):
        print("\nSIGINT received: will land at the next safe point.")
        land_requested["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    plot = LivePlot()

    try:
        link.connect()
        imu_flow_lg = link.add_imu_flow_callback(fusion.on_imu_flow)
        imu_flow_lg.data_received_cb.add_callback(lambda ts, data, logconf: recorder.on_imu_flow_extra(ts, data))
        link.add_state_att_callback(fusion.on_state_att)
        state_pv_lg = link.add_state_pv_callback(lambda ts, data: None)
        state_pv_lg.data_received_cb.add_callback(lambda ts, data, logconf: recorder.on_state_pv_extra(ts, data))
        link.add_diagnostic_callbacks(
            kalman_z_cb=safety.on_kalman_z, power_cb=safety.on_power,
            ranger_cb=safety.on_ranger, baro_cb=safety.on_baro,
        )
        time.sleep(1.0)  # let the first log packets arrive before arming
        print("Log established.")

        cf_link.arm_cf(link.cf)
        fusion.start()

        dt = 1.0 / CONTROL_RATE_HZ
        print(f"Taking off to {args.height} m")
        t0 = time.time()
        while time.time() - t0 < TAKEOFF_S:
            frac = (time.time() - t0) / TAKEOFF_S
            link.cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, args.height * frac)
            time.sleep(dt)

        print("Flying square + zigzag trajectory")
        trajectory_state["start_time"] = time.time()
        tick = 0
        while not land_requested["flag"]:
            t = get_elapsed_s()
            if t >= total_duration_s:
                break

            safe, reason = safety.is_safe()
            if not safe:
                print(f"SafetyMonitor: unsafe ({reason}) -- landing.")
                break

            cmd = vel_fn(t)
            link.cf.commander.send_hover_setpoint(cmd.vx, cmd.vy, cmd.yaw_rate, cmd.height)

            recorder.sample(fusion)
            tick += 1
            if tick % PLOT_REFRESH_EVERY_N_TICKS == 0:
                plot.update(recorder)

            time.sleep(dt)

        print("Landing")
        h = args.height
        land_t0 = time.time()
        land_duration_s = 1.5
        while time.time() - land_t0 < land_duration_s:
            frac = 1.0 - (time.time() - land_t0) / land_duration_s
            link.cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, h * max(frac, 0.0))
            time.sleep(dt)

        cf_link.send_stop_setpoint(link.cf)
        cf_link.disarm_cf(link.cf)

    finally:
        fusion.stop()
        try:
            link.disconnect()
        except Exception:
            pass
        plot.update(recorder)
        plot.save(args.save_plot)
        print(f"Saved final plot: {args.save_plot}")
        print(f"Recorded {len(recorder.time_s)} samples over {get_elapsed_s():.1f}s")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
