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

    def __rmul__(self, other):
        return self.__mul__(other)

    def to_euler(self):
        w, x, y, z = self.q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
        else:
            pitch = np.arctan2(sinp, np.sqrt(1 - sinp * sinp))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


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
    
    def rotate(self, vector):
        """Rotate a Vector3D by this quaternion and return a new Vector3D."""
        from .vector3d import Vector3D  # Local import to avoid circular dependency

        # Convert vector into a quaternion with zero scalar part
        v_quat = Quaternion(0, vector.v[0], vector.v[1], vector.v[2])
        # Apply rotation
        rotated_quat = self * v_quat * self.conjugate()
        # Return rotated vector
        return Vector3D(rotated_quat.q[1], rotated_quat.q[2], rotated_quat.q[3])
    
    @staticmethod
    def from_axis_angle(axis, angle):
        axis = axis / np.linalg.norm(axis)
        sin_a = np.sin(angle / 2)
        cos_a = np.cos(angle / 2)
        
        w = cos_a
        x = axis[0] * sin_a
        y = axis[1] * sin_a
        z = axis[2] * sin_a
        
        return Quaternion(w, x, y, z)

    @staticmethod
    def from_euler(roll, pitch, yaw):
    # Correcting the order of application to ZYX (yaw, pitch, roll) for proper aerospace sequence
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr

        return Quaternion(w, x, y, z)


    def __repr__(self):
        return f"Quaternion({self.q[0]}, {self.q[1]}, {self.q[2]}, {self.q[3]})"
   