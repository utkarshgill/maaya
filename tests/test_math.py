import unittest
import sys
import os

# Ensure project root is in path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
from common.math import Vector3D, Quaternion
from scipy.spatial.transform import Rotation as SciRot

class TestMath(unittest.TestCase):
    def test_quaternion_from_euler_matches_scipy(self):
        for angles in [
            (0.1, 0.2, 0.3),
            (1.0, 0.0, 0.0),
            (0.0, 1.57, 0.0)
        ]:
            with self.subTest(angles=angles):
                roll, pitch, yaw = angles
                q = Quaternion.from_euler(roll, pitch, yaw)
                mat = q.as_rotation_matrix()
                r = SciRot.from_euler('xyz', angles, degrees=False)
                np.testing.assert_allclose(mat, r.as_matrix(), atol=1e-6)

    def test_quaternion_rotate_vector_matches_scipy(self):
        axis = np.array([0, 0, 1])
        angle = np.pi / 4
        q = Quaternion.from_axis_angle(axis, angle)
        v = Vector3D(1, 0, 0)
        v_rot = q.rotate(v)
        expected = SciRot.from_rotvec(axis * angle).apply(v.v)
        np.testing.assert_allclose(v_rot.v, expected, atol=1e-6)

if __name__ == '__main__':
    unittest.main() 