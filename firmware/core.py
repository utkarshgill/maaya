"""
Core system engine: scheduler, state machine, control loops.
"""

import math
import numpy as np
from maaya import Controller

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
    Generic PID controller with optional angle wrapping.
    """
    def __init__(self, kp, ki, kd, setpoint=0.0, wrap=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.wrap = wrap
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        """Clear integral and derivative state."""
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, measurement, dt):
        """
        Compute PID output given a measurement and timestep dt.
        """
        error = self.setpoint - measurement
        if self.wrap:
            # wrap error into [-pi, pi]
            error = (error + math.pi) % (2 * math.pi) - math.pi
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
    m1 = thrust + roll + pitch - yaw
    m2 = thrust - roll + pitch + yaw
    m3 = thrust - roll - pitch - yaw
    m4 = thrust + roll - pitch + yaw
    return [m1, m2, m3, m4]


class StabilityController(Controller):
    """Renamed DroneController, handles core stabilization PID loops."""
    def __init__(self):
        self.x_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=0.0)
        self.y_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=0.0)
        self.z_pid = PIDController(kp=1.5, ki=0.2, kd=3.0, setpoint=1.0)
        self.roll_pid = PIDController(kp=2.0, ki=0.0, kd=0.3)
        self.pitch_pid = PIDController(kp=2.0, ki=0.0, kd=0.3)
        self.yaw_pid = PIDController(kp=1.0, ki=0.0, kd=0.1, wrap=True)

        # Reset integral and previous error state for all PIDs
        self.x_pid.reset()
        self.y_pid.reset()
        self.z_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()

    def update(self, body, dt):
        pitch_set = np.clip(self.x_pid.update(body.position.v[0], dt), -0.3, 0.3)
        roll_set = np.clip(-self.y_pid.update(body.position.v[1], dt), -0.3, 0.3)
        thrust = 9.8 + self.z_pid.update(body.position.v[2], dt)
        thrust = float(np.clip(thrust, 0.0, 20.0))
        self.roll_pid.setpoint = roll_set
        self.pitch_pid.setpoint = pitch_set
        roll_cmd = float(np.clip(self.roll_pid.update(body.orientation.to_euler()[0], dt), -0.5, 0.5))
        pitch_cmd = float(np.clip(self.pitch_pid.update(body.orientation.to_euler()[1], dt), -0.5, 0.5))
        yaw_cmd = float(np.clip(self.yaw_pid.update(body.orientation.to_euler()[2], dt), -0.3, 0.3))
        return [thrust, roll_cmd, pitch_cmd, yaw_cmd]

    def set_xyz_target(self, x_target, y_target, z_target):
        """Dynamically update horizontal XYZ set-points."""
        self.x_pid.setpoint = x_target
        self.y_pid.setpoint = y_target
        self.z_pid.setpoint = z_target

    def set_attitude_target(self, roll_target, pitch_target, yaw_target):
        """Dynamically update attitude set-points."""
        self.roll_pid.setpoint = roll_target
        self.pitch_pid.setpoint = pitch_target
        self.yaw_pid.setpoint = yaw_target 