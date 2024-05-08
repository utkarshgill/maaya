from .math import Vector3D

class World:
    def __init__(self, g=9.8):
        self.objects = []
        self.time = 0
        self.g = g  # gravitational acceleration

    def add_object(self, obj):
        self.objects.append(obj)

    def update(self, dt):

        for obj in self.objects:
            print(obj.position.v[2])
            obj.apply_force(Vector3D(0, 0, -self.g * obj.mass))  # apply gravity force
            obj.update(dt)

    def simulate(self, frames, render=False):
        for _ in range(frames):
            self.update(0.01)
