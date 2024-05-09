import numpy as np
from .math import Vector3D
from .forces import GravitationalForce

class NoiseGenerator:
    def __init__(self, intensity=0.1):
        self.intensity = intensity

    def apply_to(self, obj):
        force_noise = Vector3D(*np.random.normal(0, self.intensity, size=3))
        torque_noise = Vector3D(*np.random.normal(0, self.intensity, size=3))
        obj.apply_force(force_noise)
        obj.apply_torque(torque_noise)


class World:
    def __init__(self, gravity=GravitationalForce(), noise=NoiseGenerator()):
        self.objects = []
        self.time = 0
        self.gravity = gravity
        self.noise = noise

    def add_object(self, obj):
        self.objects.append(obj)

    def update(self, dt):
        for obj in self.objects:
            self.gravity.apply_to(obj)  # apply gravity force
            self.noise.apply_to(obj)
            obj.update(dt)

    def simulate(self, frames, render=False):
        for _ in range(frames):
            self.update(0.01)

