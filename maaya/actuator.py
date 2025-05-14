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

class Mixer(Actuator):
    """Convert high‐level total thrust + body torques into individual motor thrust commands.

    Assumes a plus configuration with arms along +X, +Y, −X, −Y axes.
    control_command must be a 4‐vector [T, τx, τy, τz].
    The resulting per‐motor thrusts are stored in obj.motor_thrusts (length-4 array).
    No forces/torques are applied in this class – that is handled by Motor actuators.
    """

    def __init__(self, arm_length, kT=1.0, kQ=0.02):
        self.L = arm_length
        self.kT = kT  # thrust coefficient (N per unit command)
        self.kQ = kQ  # reaction torque coefficient (N·m per N thrust)

    def apply_to(self, obj, dt):
        cmd = getattr(obj, 'control_command', None)
        if cmd is None or len(cmd) != 4:
            return  # nothing to mix

        T_tot, tau_x, tau_y, tau_z = cmd

        # Invert simple mixing matrix for plus config with spin signs [1,-1,1,-1]
        L = self.L
        kQ = self.kQ

        # The matrix A maps motor thrusts f0..f3 to [T, τx, τy, τz]
        # A = [[ 1,  1,  1,  1],
        #      [ 0,  L,  0, -L],
        #      [-L, 0,   L,  0],
        #      [ kQ, -kQ, kQ, -kQ]]
        # We hard-code the analytic inverse for speed & readability
        # Define helpers
        inv = 1 / 4
        f0 = inv * (T_tot - 2 * tau_y / L +  tau_z / kQ)
        f1 = inv * (T_tot + 2 * tau_x / L -  tau_z / kQ)
        f2 = inv * (T_tot + 2 * tau_y / L +  tau_z / kQ)
        f3 = inv * (T_tot - 2 * tau_x / L -  tau_z / kQ)

        # Ensure non-negative thrusts
        thrusts = np.clip(np.array([f0, f1, f2, f3]), 0.0, None)
        obj.motor_thrusts = thrusts  # publish for Motor actuators

class Motor(Actuator):
    """Individual rotor model with its own noise parameters.

    Parameters
    ----------
    idx : int
        Index of this motor (0..3) – used to pick thrust from obj.motor_thrusts.
    r_body : Vector3D
        Position vector from vehicle COM to motor in body frame (m).
    spin : int
        +1 or −1 for spin direction (sign of reaction torque).
    kQ : float, optional
        Reaction-torque coefficient (N·m per N thrust).
    thrust_noise_std : float, optional
    torque_noise_std : float, optional
    """

    def __init__(self, idx, r_body, spin, kQ=0.02, thrust_noise_std=0.0, torque_noise_std=0.0):
        self.idx = idx
        self.r_body = r_body
        self.spin = spin
        self.kQ = kQ
        self.thrust_noise_std = thrust_noise_std
        self.torque_noise_std = torque_noise_std

    def apply_to(self, obj, dt):
        thrusts = getattr(obj, 'motor_thrusts', None)
        if thrusts is None or self.idx >= len(thrusts):
            return  # mixer hasn't provided commands yet

        thrust = thrusts[self.idx]

        # Add thrust noise
        if self.thrust_noise_std > 0.0:
            thrust += np.random.randn() * self.thrust_noise_std
            thrust = max(0.0, thrust)

        # Force along +Z body axis
        force_body = Vector3D(0, 0, thrust)

        # Reaction torque about body Z
        reaction_torque_body = Vector3D(0, 0, self.spin * self.kQ * thrust)

        # Torque from force applied off-axis: τ = r × F
        arm_torque_body = self.r_body.cross(force_body)

        total_torque_body = arm_torque_body + reaction_torque_body

        # Add torque noise
        if self.torque_noise_std > 0.0:
            noise_tau = Vector3D(* (np.random.randn(3) * self.torque_noise_std))
            total_torque_body += noise_tau

        # Apply to object (forces need to be in world frame)
        world_force = obj.orientation.rotate(force_body)
        obj.apply_force(world_force)
        obj.apply_torque(total_torque_body, dt) 