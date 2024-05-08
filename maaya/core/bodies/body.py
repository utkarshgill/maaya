import numpy as np
from ..math import Vector3D, Quaternion

class Body:
    def __init__(self, position=None, velocity=None, acceleration=None, mass=1.0,
                 orientation=None, angular_velocity=None, inertia = np.eye(3)):
        self.position = position if position is not None else Vector3D()
        self.velocity = velocity if velocity is not None else Vector3D()
        self.acceleration = acceleration if acceleration is not None else Vector3D()
        self.mass = mass
        self.orientation = orientation if orientation is not None else Quaternion()
        self.angular_velocity = angular_velocity if angular_velocity is not None else Quaternion()
        self.inertia = inertia  # Placeholder for moment of inertia as a 3x3 matrix

    def apply_torque(self, torque, dt=0.01):
        # Adjusting torque application to include inertia
        angular_acceleration = np.linalg.inv(self.inertia).dot(torque.v)  # Inertia matrix must be invertible
        angular_acceleration_quaternion = Quaternion(0, *angular_acceleration)
        self.angular_velocity += angular_acceleration_quaternion * dt
        self.angular_velocity.normalize()  # Optional, depends on your handling of angular_velocity

    def update(self, dt):
        # Update orientation using a more accurate quaternion integration approach
        orientation_delta = self.angular_velocity * self.orientation * 0.5 * dt
        self.orientation += orientation_delta
        self.orientation.normalize()

        # Update linear motion
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        self.acceleration = Vector3D()  # Reset acceleration if needed

        # Normalize the orientation quaternion to prevent drift
        self.orientation.normalize()

    def apply_force(self, force):
        # F = m * a, therefore a = F / m
        self.acceleration += Vector3D(*(force.v / self.mass))

    def __repr__(self):
        return f"Body(position={self.position}, velocity={self.velocity}, acceleration={self.acceleration}, mass={self.mass})"
