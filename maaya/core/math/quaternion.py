import numpy as np

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.q = np.array([w, x, y, z], dtype=float)

    def __add__(self, other):
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.q
            w2, x2, y2, z2 = other.q
            return Quaternion(w1 + w2, x1 + x2, y1 + y2, z1 + z2)
        else:
            raise TypeError("Addition is only defined for Quaternion objects.")

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.q
            w2, x2, y2, z2 = other.q
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
            z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            w, x, y, z = self.q
            return Quaternion(w * other, x * other, y * other, z * other)
        else:
            raise TypeError("Multiplication is only defined for Quaternion objects and scalars.")


    def conjugate(self):
        w, x, y, z = self.q
        return Quaternion(w, -x, -y, -z)

    def normalize(self):
        norm = np.linalg.norm(self.q)
        self.q /= norm

    def as_rotation_matrix(self):
        w, x, y, z = self.q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], dtype=float)

    def __repr__(self):
        return f"Quaternion({self.q[0]}, {self.q[1]}, {self.q[2]}, {self.q[3]})"
   