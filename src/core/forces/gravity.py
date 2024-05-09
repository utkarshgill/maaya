from ..math import Vector3D

class GravitationalForce:
    def __init__(self, g=9.8):
        self.g = g

    def apply_to(self, obj):
        gravitational_force = Vector3D(0, 0, -self.g * obj.mass)
        obj.apply_force(gravitational_force)