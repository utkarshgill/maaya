import numpy as np
from .quaternion import Quaternion

class Vector3D:
    def __init__(self, x=0, y=0, z=0):
        self.v = np.array([x, y, z], dtype=float)

    def __add__(self, other):
        return Vector3D(*(self.v + other.v))

    def __sub__(self, other):
        return Vector3D(*(self.v - other.v))

    def __mul__(self, scalar):
        return Vector3D(*(self.v * scalar))

    def dot(self, other):
        """Return scalar dot‐product between two vectors."""
        return float(np.dot(self.v, other.v))

    def cross(self, other):
        return Vector3D(*np.cross(self.v, other.v))

    def magnitude(self):
        return np.linalg.norm(self.v)
    
    def apply_rotation(self, quaternion):
        # Rotates this vector by the given quaternion
        q_vector = Quaternion(0, *self.v)
        q_rotated = quaternion * q_vector * quaternion.conjugate()
        self.v = q_rotated.q[1:]  # update vector with rotated coordinates

    def __repr__(self):
        return f"Vector3D({self.v[0]}, {self.v[1]}, {self.v[2]})"
