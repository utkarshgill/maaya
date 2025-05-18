"""
Core system engine: scheduler, state machine, control loops.
"""

import math
import numpy as np
from maaya import Controller
from .utils import wrap_angle

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
    m1 = thrust + roll + pitch - yaw
    m2 = thrust - roll + pitch + yaw
    m3 = thrust - roll - pitch - yaw
    m4 = thrust + roll - pitch + yaw
    return [m1, m2, m3, m4]


class StabilityController(Controller):
    """Renamed DroneController, handles core stabilization PID loops."""
    def __init__(self):
        self.x_pid = PIDController(kp=0.2, ki=0.0, kd=0.3)
        self.y_pid = PIDController(kp=0.2, ki=0.0, kd=0.3)
        self.z_pid = PIDController(kp=1.5, ki=0.2, kd=3.0)
        self.roll_pid = PIDController(kp=2.0, ki=0.0, kd=0.3)
        self.pitch_pid = PIDController(kp=2.0, ki=0.0, kd=0.3)
        self.yaw_pid = PIDController(kp=1.0, ki=0.0, kd=0.1)

        # Reset integral and previous error state for all PIDs
        self.x_pid.reset()
        self.y_pid.reset()
        self.z_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()

        # Store setpoints for each PID internally
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

        # Attitude errors (yaw is wrapped)
        roll_error = self.roll_setpoint - body.orientation.to_euler()[0]
        pitch_error = self.pitch_setpoint - body.orientation.to_euler()[1]
        yaw_raw_error = self.yaw_setpoint - body.orientation.to_euler()[2]
        yaw_error = wrap_angle(yaw_raw_error) # Use the new wrap_angle utility

        # Update PIDs with pre-computed errors
        pitch_cmd_from_x = np.clip(self.x_pid.update(x_error, dt), -0.3, 0.3)
        roll_cmd_from_y = np.clip(-self.y_pid.update(y_error, dt), -0.3, 0.3) # Note negation for y-axis control
        thrust_cmd_from_z = 0 + self.z_pid.update(z_error, dt)
        thrust_cmd = float(np.clip(thrust_cmd_from_z, 0.0, 20.0))

        # Update attitude setpoints based on position controller outputs
        # (This effectively makes roll/pitch PIDs act as inner-loop rate controllers if setpoints are dynamic)
        # self.roll_pid.setpoint = roll_cmd_from_y
        # self.pitch_pid.setpoint = pitch_cmd_from_x
        # Re-evaluate if the above lines are still needed or if roll/pitch_setpoint are now exclusively driven by set_attitude_target
        # For now, we assume setpoints are updated externally via set_attitude_target for direct PS5 control.
        
        # Update attitude PIDs (yaw error is already wrapped)
        roll_final_cmd = float(np.clip(self.roll_pid.update(roll_error, dt), -0.5, 0.5))
        pitch_final_cmd = float(np.clip(self.pitch_pid.update(pitch_error, dt), -0.5, 0.5))
        yaw_final_cmd = float(np.clip(self.yaw_pid.update(yaw_error, dt), -0.3, 0.3))
        
        return [thrust_cmd, roll_final_cmd, pitch_final_cmd, yaw_final_cmd]

    def set_xyz_target(self, x_target, y_target, z_target):
        """Dynamically update horizontal XYZ set-points."""
        self.x_setpoint = x_target
        self.y_setpoint = y_target
        self.z_setpoint = z_target

    def set_attitude_target(self, roll_target, pitch_target, yaw_target):
        """Dynamically update attitude set-points."""
        self.roll_setpoint = roll_target
        self.pitch_setpoint = pitch_target
        self.yaw_setpoint = yaw_target 