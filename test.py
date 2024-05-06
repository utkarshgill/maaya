import unittest
import numpy as np
from run import Vector3D, Quaternion, PhysicsObject

# Assuming the Vector3D class is already defined above

class TestVector3D(unittest.TestCase):
    def test_addition(self):
        vec1 = Vector3D(1, 2, 3)
        vec2 = Vector3D(4, 5, 6)
        result = vec1 + vec2
        self.assertTrue(np.allclose(result.v, [5, 7, 9]))

    def test_subtraction(self):
        vec1 = Vector3D(10, 20, 30)
        vec2 = Vector3D(1, 2, 3)
        result = vec1 - vec2
        self.assertTrue(np.allclose(result.v, [9, 18, 27]))

    def test_scalar_multiplication(self):
        vec = Vector3D(1, 2, 3)
        result = vec * 5
        self.assertTrue(np.allclose(result.v, [5, 10, 15]))

    def test_dot_product(self):
        vec1 = Vector3D(1, 2, 3)
        vec2 = Vector3D(4, 5, 6)
        result = vec1.dot(vec2)
        self.assertEqual(result, 32)

    def test_cross_product(self):
        vec1 = Vector3D(1, 2, 3)
        vec2 = Vector3D(4, 5, 6)
        result = vec1.cross(vec2)
        self.assertTrue(np.allclose(result.v, [-3, 6, -3]))

    def test_magnitude(self):
        vec = Vector3D(1, 2, 2)
        result = vec.magnitude()
        self.assertAlmostEqual(result, 3)

    def test_rotation(self):
        vec = Vector3D(1, 0, 0)
        quaternion = Quaternion(np.cos(np.pi/4), 0, np.sin(np.pi/4), 0)
        vec.apply_rotation(quaternion)
        self.assertTrue(np.allclose(vec.v, [0, 1, 0]))

    def test_repr(self):
        vec = Vector3D(1, 2, 3)
        self.assertEqual(repr(vec), "Vector3D(1.0, 2.0, 3.0)")

if __name__ == '__main__':
    unittest.main()
