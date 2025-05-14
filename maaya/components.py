# components.py
# Sensors, actuators, mixers, and controllers

import numpy as np
from .math import Vector3D

# Sensors
class Sensor:
    """Base class for sensors."""
    def read(self, objects, dt):
        raise NotImplementedError

class IMUSensor(Sensor):
    """IMU sensor model: returns noisy acceleration and angular velocity."""
    def __init__(self, accel_noise_std=0.0, gyro_noise_std=0.0,
                 accel_bias_rw_std=0.0, gyro_bias_rw_std=0.0):
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std

        self.accel_bias_rw_std = accel_bias_rw_std
        self.gyro_bias_rw_std = gyro_bias_rw_std

        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)

    def read(self, objects, dt):
        for obj in objects:
            true_accel = obj.acceleration.v
            true_gyro = obj.angular_velocity.v

            if self.accel_bias_rw_std > 0.0:
                self._accel_bias += np.random.randn(3) * self.accel_bias_rw_std * np.sqrt(dt)
            if self.gyro_bias_rw_std > 0.0:
                self._gyro_bias += np.random.randn(3) * self.gyro_bias_rw_std * np.sqrt(dt)

            noisy_accel = true_accel + self._accel_bias + np.random.randn(3) * self.accel_noise_std
            noisy_gyro = true_gyro + self._gyro_bias + np.random.randn(3) * self.gyro_noise_std

            obj.sensor_data = {
                'accel': noisy_accel,
                'gyro': noisy_gyro,
                'accel_bias': self._accel_bias.copy(),
                'gyro_bias': self._gyro_bias.copy(),
            }

# Actuators
class Actuator:
    """Base class for actuators."""
    def apply_to(self, obj, dt):
        raise NotImplementedError

class SimpleThrustActuator(Actuator):
    """Convert control command into force along body z-axis."""
    def apply_to(self, obj, dt):
        thrust = getattr(obj, 'control_command', 0.0)
        force_body = Vector3D(0, 0, thrust)
        world_force = obj.orientation.rotate(force_body)
        obj.apply_force(world_force)

class QuadrotorActuator(Actuator):
    """Actuator mapping 4-element control_command [T, R, P, Y] to force and torque."""
    def __init__(self, thrust_noise_std=0.0, torque_noise_std=0.0):
        self.thrust_noise_std = thrust_noise_std
        self.torque_noise_std = torque_noise_std

    def apply_to(self, obj, dt):
        cmd = getattr(obj, 'control_command', None)
        if cmd is None:
            return
        if isinstance(cmd, (list, tuple)) and len(cmd) == 4:
            T, R, P, Y = cmd
        else:
            T = float(cmd)
            R = P = Y = 0.0
        force_body = Vector3D(0, 0, T)
        world_force = obj.orientation.rotate(force_body)
        obj.apply_force(world_force)
        torque_body = Vector3D(R, P, Y)
        obj.apply_torque(torque_body, dt)
        if self.thrust_noise_std > 0.0:
            noise_f = Vector3D(*(np.random.randn(3) * self.thrust_noise_std))
            obj.apply_force(noise_f)
        if self.torque_noise_std > 0.0:
            noise_tau = Vector3D(*(np.random.randn(3) * self.torque_noise_std))
            obj.apply_torque(noise_tau, dt)

class Mixer(Actuator):
    """Convert total thrust+torques into individual motor thrusts (plus config)."""
    def __init__(self, arm_length, kT=1.0, kQ=0.02):
        self.L = arm_length
        self.kT = kT
        self.kQ = kQ

    def apply_to(self, obj, dt):
        cmd = getattr(obj, 'control_command', None)
        if cmd is None or len(cmd) != 4:
            return
        T_tot, tau_x, tau_y, tau_z = cmd
        L = self.L
        kQ = self.kQ
        inv = 1/4
        f0 = inv * (T_tot - 2*tau_y/L + tau_z/kQ)
        f1 = inv * (T_tot + 2*tau_x/L - tau_z/kQ)
        f2 = inv * (T_tot + 2*tau_y/L + tau_z/kQ)
        f3 = inv * (T_tot - 2*tau_x/L - tau_z/kQ)
        thrusts = np.clip(np.array([f0, f1, f2, f3]), 0.0, None)
        obj.motor_thrusts = thrusts

class GenericMixer(Actuator):
    """Generic mixer for arbitrary multirotor geometry."""
    def __init__(self, motor_positions, spins, kT=1.0, kQ=0.02):
        if len(motor_positions) != len(spins):
            raise ValueError("motor_positions and spins must have same length")
        self.motor_positions = motor_positions
        self.spins = spins
        self.kT = kT
        self.kQ = kQ
        rows = []
        rows.append([kT]*len(spins))
        rows.append([pos.v[1]*kT for pos in motor_positions])
        rows.append([-pos.v[0]*kT for pos in motor_positions])
        rows.append([s*kQ for s in spins])
        self._A = np.array(rows)
        self._A_inv = np.linalg.pinv(self._A)

    def apply_to(self, obj, dt):
        cmd = getattr(obj, 'control_command', None)
        if cmd is None or len(cmd) != 4:
            return
        thrusts = np.dot(self._A_inv, np.array(cmd))
        thrusts = np.clip(thrusts, 0.0, None)
        obj.motor_thrusts = thrusts

class Motor(Actuator):
    """Individual rotor model with first-order lag and noise."""
    def __init__(self, idx, r_body, spin, kQ=0.02, thrust_noise_std=0.0, torque_noise_std=0.0, tau=0.02, max_thrust=None):
        self.idx = idx
        self.r_body = r_body
        self.spin = spin
        self.kQ = kQ
        self.thrust_noise_std = thrust_noise_std
        self.torque_noise_std = torque_noise_std
        self.tau = tau
        self._thrust_state = 0.0
        self.max_thrust = max_thrust

    def apply_to(self, obj, dt):
        thrusts = getattr(obj, 'motor_thrusts', None)
        if thrusts is None or self.idx >= len(thrusts):
            return
        desired = thrusts[self.idx]
        if self.max_thrust is not None:
            desired = min(desired, self.max_thrust)
        self._thrust_state += (desired - self._thrust_state)*(dt/self.tau)
        thrust = self._thrust_state
        if self.max_thrust is not None and thrust > self.max_thrust:
            thrust = self.max_thrust
        if self.thrust_noise_std>0.0:
            thrust += np.random.randn()*self.thrust_noise_std
            thrust = max(0.0, thrust)
        force_body = Vector3D(0,0,thrust)
        reaction = Vector3D(0,0,self.spin*self.kQ*thrust)
        arm_torque = self.r_body.cross(force_body)
        total_torque = arm_torque + reaction
        if self.torque_noise_std>0.0:
            noise = Vector3D(*(np.random.randn(3)*self.torque_noise_std))
            total_torque += noise
        world_force = obj.orientation.rotate(force_body)
        obj.apply_force(world_force)
        obj.apply_torque(total_torque, dt)

# Controllers
class Controller:
    """Base class for controllers."""
    def update(self, body, dt):
        raise NotImplementedError

class PIDController(Controller):
    """Simple PID controller for altitude by default."""
    def __init__(self, kp, ki, kd, setpoint=0.0, dt=0.01, measurement_fn=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.previous_error = 0.0
        self.integral = 0.0
        self.measurement_fn = (measurement_fn if measurement_fn is not None else lambda body: body.position.v[2])

    def update(self, body, dt):
        error = self.setpoint - self.measurement_fn(body)
        self.integral += error * dt
        derivative = (error - self.previous_error)/dt if dt>0.0 else 0.0
        self.previous_error = error
        return self.kp*error + self.ki*self.integral + self.kd*derivative 