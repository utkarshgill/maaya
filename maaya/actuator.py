from .math import Vector3D
import numpy as np  # for actuator noise

class Actuator:
    """Base class for actuators."""
    def apply_to(self, obj, dt):
        """Apply actuation for given object; to be implemented by subclasses. dt is timestep."""
        raise NotImplementedError

class SimpleThrustActuator(Actuator):
    """Convert control command into force along body z-axis."""
    def apply_to(self, obj, dt):  # dt unused for simple thrust
        """Apply vertical thrust based on scalar control_command."""
        thrust = getattr(obj, 'control_command', 0.0)
        # Apply thrust in body frame +Z
        force_body = Vector3D(0, 0, thrust)
        # Rotate to world frame and apply
        world_force = obj.orientation.rotate(force_body)
        obj.apply_force(world_force)

class QuadrotorActuator(Actuator):
    """Actuator mapping 4-element control_command [T, R, P, Y] to force and torque, with optional noise."""
    def __init__(self, thrust_noise_std=0.0, torque_noise_std=0.0):
        """
        Args:
            thrust_noise_std: standard deviation of thrust noise (N)
            torque_noise_std: standard deviation of torque noise (N·m)
        """
        self.thrust_noise_std = thrust_noise_std
        self.torque_noise_std = torque_noise_std
    def apply_to(self, obj, dt):
        cmd = getattr(obj, 'control_command', None)
        if cmd is None:
            return
        # Unpack thrust and torques
        if isinstance(cmd, (list, tuple)) and len(cmd) == 4:
            T, R, P, Y = cmd
        else:
            # fallback to scalar thrust only
            T = float(cmd)
            R = P = Y = 0.0
        # Apply thrust in body +Z
        force_body = Vector3D(0, 0, T)
        world_force = obj.orientation.rotate(force_body)
        obj.apply_force(world_force)
        # Apply torques in body frame
        torque_body = Vector3D(R, P, Y)
        obj.apply_torque(torque_body, dt)
        # Add actuator noise
        if self.thrust_noise_std > 0.0:
            noise_f = Vector3D(* (np.random.randn(3) * self.thrust_noise_std))
            obj.apply_force(noise_f)
        if self.torque_noise_std > 0.0:
            noise_tau = Vector3D(* (np.random.randn(3) * self.torque_noise_std))
            obj.apply_torque(noise_tau, dt) 