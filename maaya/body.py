import numpy as np
from .math import Vector3D, Quaternion
from .physics import EulerIntegrator

class Body:
    def __init__(self, position=None, velocity=None, acceleration=None, mass=1.0,
                 orientation=None, angular_velocity=None, inertia=np.eye(3)):
        self.position = position if position is not None else Vector3D()
        self.velocity = velocity if velocity is not None else Vector3D()
        self.acceleration = acceleration if acceleration is not None else Vector3D()
        self.mass = mass
        self.orientation = orientation if orientation is not None else Quaternion()
        # Angular velocity is stored as a pure 3-vector (rad/s)
        self.angular_velocity = angular_velocity if angular_velocity is not None else Vector3D()
        self.inertia = inertia  # Placeholder for moment of inertia as a 3x3 matrix

    def apply_torque(self, torque, dt=0.01):
        """Update angular velocity given a torque vector.

        Parameters
        ----------
        torque : Vector3D
            Torque expressed in the body frame (N·m).
        dt : float, optional
            Timestep size in seconds. Defaults to 0.01.
        """
        # α = I⁻¹ τ
        angular_acceleration = np.linalg.inv(self.inertia).dot(torque.v)
        # Integrate to update angular velocity (simple Euler)
        self.angular_velocity += Vector3D(*angular_acceleration) * dt

    def update(self, dt):
        EulerIntegrator().step(self, dt)

    def apply_force(self, force):
        # F = m * a, therefore a = F / m
        self.acceleration += Vector3D(*(force.v / self.mass))

    def __repr__(self):
        return f"Body(position={self.position}, velocity={self.velocity}, acceleration={self.acceleration}, mass={self.mass})" 