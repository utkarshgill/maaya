"""
Core system engine: scheduler, state machine, control loops.
"""

import math
import numpy as np
from sim import Controller
from sim.math import Quaternion
from .utils import wrap_angle, load_config, GRAVITY

class Scheduler:
    def __init__(self, config):
        self.config = config
        self.tasks = []

    def add_task(self, func, interval):
        """
        Register a task function to be called every 'interval' seconds.
        """
        self.tasks.append((func, interval))
        # TODO: implement task scheduling

    def run(self):
        """
        Start the cooperative task loop.
        """
        # TODO: implement cooperative loop
        while True:
            for func, interval in self.tasks:
                func()
            # TODO: add timing control

class StateMachine:
    def __init__(self):
        self.state = "DISARMED"

    def arm(self):
        self.state = "ARMED"

    def disarm(self):
        self.state = "DISARMED"

    def failsafe(self):
        self.state = "FAILSAFE"


class PIDController:
    """
    Generic PID controller.
    Accepts pre-computed error, does not handle setpoints or wrapping.
    """
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        """Clear integral and derivative state."""
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error, dt):
        """
        Compute PID output given an error and timestep dt.
        """
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        return output


def mixer(roll, pitch, yaw, thrust):
    """
    Simple X-configuration mixer for a quadcopter.
    Returns motor thrusts [m1, m2, m3, m4].
    """
    # Load mixer coefficients from config if available
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    m_cfg = cfg.get("mixer", {})
    thrust_c = m_cfg.get("thrust_coeff", 1.0)
    roll_c = m_cfg.get("roll_coeff", 1.0)
    pitch_c = m_cfg.get("pitch_coeff", 1.0)
    yaw_c = m_cfg.get("yaw_coeff", 1.0)
    # Apply coefficients
    t = thrust_c * thrust
    r = roll_c * roll
    p = pitch_c * pitch
    y = yaw_c * yaw
    m1 = t + r + p - y
    m2 = t - r + p + y
    m3 = t - r - p - y
    m4 = t + r - p + y
    return [m1, m2, m3, m4]


class StabilityController(Controller):
    """Renamed DroneController, handles core stabilization PID loops."""
    def __init__(self, config=None):
        # Load configuration for PID gains if provided
        if config is None:
            try:
                config = load_config()
            except Exception:
                config = {}
        pid_cfg = config.get("pid", {})
        # Extract per-loop gains with defaults
        x_cfg = pid_cfg.get("x", {})
        y_cfg = pid_cfg.get("y", {})
        z_cfg = pid_cfg.get("z", {})
        roll_cfg = pid_cfg.get("roll", {})
        pitch_cfg = pid_cfg.get("pitch", {})
        yaw_cfg = pid_cfg.get("yaw", {})
        # Instantiate PID controllers with config-driven gains
        self.x_pid = PIDController(kp=x_cfg.get("kp", 0.2), ki=x_cfg.get("ki", 0.0), kd=x_cfg.get("kd", 0.3))
        self.y_pid = PIDController(kp=y_cfg.get("kp", 0.2), ki=y_cfg.get("ki", 0.0), kd=y_cfg.get("kd", 0.3))
        self.z_pid = PIDController(kp=z_cfg.get("kp", 1.5), ki=z_cfg.get("ki", 0.2), kd=z_cfg.get("kd", 3.0))
        self.roll_pid = PIDController(kp=roll_cfg.get("kp", 2.0), ki=roll_cfg.get("ki", 0.0), kd=roll_cfg.get("kd", 0.3))
        self.pitch_pid = PIDController(kp=pitch_cfg.get("kp", 2.0), ki=pitch_cfg.get("ki", 0.0), kd=pitch_cfg.get("kd", 0.3))
        self.yaw_pid = PIDController(kp=yaw_cfg.get("kp", 1.0), ki=yaw_cfg.get("ki", 0.0), kd=yaw_cfg.get("kd", 0.1))

        # Reset integral and previous error state for all PIDs
        for pid in (self.x_pid, self.y_pid, self.z_pid, self.roll_pid, self.pitch_pid, self.yaw_pid):
            pid.reset()

        # Initialize setpoints for each loop
        self.x_setpoint = 0.0
        self.y_setpoint = 0.0
        self.z_setpoint = 1.0  # Default hover altitude
        self.roll_setpoint = 0.0
        self.pitch_setpoint = 0.0
        self.yaw_setpoint = 0.0

    def update(self, body, dt):
        # Position errors
        x_error = self.x_setpoint - body.position.v[0]
        y_error = self.y_setpoint - body.position.v[1]
        z_error = self.z_setpoint - body.position.v[2]

        # Attitude error via quaternion (avoiding Euler singularities)
        q_current = body.orientation
        q_desired = Quaternion.from_euler(self.roll_setpoint, self.pitch_setpoint, self.yaw_setpoint)
        q_error = q_desired * q_current.conjugate()
        q_error.normalize()
        # Small-angle approximation: rotation vector ~ 2 * [x, y, z]
        err_vec = 2 * q_error.q[1:]
        roll_error, pitch_error, yaw_error = err_vec.tolist()

        # Update PIDs with pre-computed errors
        pitch_cmd_from_x = np.clip(self.x_pid.update(x_error, dt), -0.3, 0.3)
        roll_cmd_from_y = np.clip(-self.y_pid.update(y_error, dt), -0.3, 0.3)  # Note negation for y-axis control
        # Add gravity baseline to thrust
        thrust_cmd_from_z = GRAVITY + self.z_pid.update(z_error, dt)
        thrust_cmd = float(np.clip(thrust_cmd_from_z, 0.0, 20.0))

        # Update inner-loop attitude PIDs
        roll_final_cmd = float(np.clip(self.roll_pid.update(roll_error, dt), -0.5, 0.5))
        pitch_final_cmd = float(np.clip(self.pitch_pid.update(pitch_error, dt), -0.5, 0.5))
        yaw_final_cmd = float(np.clip(self.yaw_pid.update(yaw_error, dt), -0.3, 0.3))
        
        return [thrust_cmd, roll_final_cmd, pitch_final_cmd, yaw_final_cmd]

    def set_xyz_target(self, x_target, y_target, z_target):
        """Dynamically update horizontal XYZ set-points via unified API."""
        self.set_target(x=x_target, y=y_target, z=z_target)

    def set_attitude_target(self, roll_target, pitch_target, yaw_target):
        """Dynamically update attitude set-points via unified API."""
        self.set_target(roll=roll_target, pitch=pitch_target, yaw=yaw_target)

    def set_target(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None):
        """Unified API for setting position and attitude setpoints."""
        if x is not None:
            self.x_setpoint = x
        if y is not None:
            self.y_setpoint = y
        if z is not None:
            self.z_setpoint = z
        if roll is not None:
            self.roll_setpoint = roll
        if pitch is not None:
            self.pitch_setpoint = pitch
        if yaw is not None:
            self.yaw_setpoint = yaw 