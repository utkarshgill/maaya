import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time

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
        return Vector3D(*np.dot(self.v, other.v))

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

    def to_rotation_matrix(self):
        w, x, y, z = self.q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], dtype=float)

    def __repr__(self):
        return f"Quaternion({self.q[0]}, {self.q[1]}, {self.q[2]}, {self.q[3]})"

        
class PhysicsObject:
    def __init__(self, position=None, velocity=None, acceleration=None, mass=1.0,
                 orientation=None, angular_velocity=None):
        self.position = position if position is not None else Vector3D()
        self.velocity = velocity if velocity is not None else Vector3D()
        self.acceleration = acceleration if acceleration is not None else Vector3D()
        self.mass = mass
        self.orientation = orientation if orientation is not None else Quaternion()
        self.angular_velocity = angular_velocity if angular_velocity is not None else Quaternion()

    def apply_torque(self, torque, dt):
        # Assuming torque is also a Quaternion, where w=0 and (x, y, z) represent the torque vector
        # This is a simple placeholder; real torque application requires more complex physics
        angular_acceleration = Quaternion(0, *(torque.v / self.mass))
        self.angular_velocity += angular_acceleration * dt
        self.orientation += self.angular_velocity * dt
        self.orientation.normalize()  # Keep the quaternion normalized

    def update(self, dt):
        # Update linear motion
        self.velocity += Vector3D(*(self.acceleration.v * dt))
        self.position += Vector3D(*(self.velocity.v * dt))
        self.acceleration = Vector3D()

        # Update angular motion
        self.orientation += self.angular_velocity * dt
        self.orientation.normalize() 

    def apply_force(self, force):
        # F = m * a, therefore a = F / m
        self.acceleration += Vector3D(*(force.v / self.mass))

    def __repr__(self):
        return f"PhysicsObject(position={self.position}, velocity={self.velocity}, acceleration={self.acceleration}, mass={self.mass})"

class World:
    def __init__(self, g=9.8):
        self.objects = []
        self.time = 0
        self.g = g  # gravitational acceleration

    def add_object(self, obj):
        self.objects.append(obj)

    def update(self, dt):
        for obj in self.objects:
            obj.apply_force(Vector3D(0, 0, -self.g * obj.mass))  # apply gravity force
            obj.update(dt)
            pos = obj.position.v[2]

            # manual impl of floor, fix after rigid body
            # print(pos)
            if pos <= 0:
                obj.position.v[2] = 0  # Reset height to floor level
                obj.velocity.v[2] *= -1

    def simulate(self, frames, render=False):
        if render:
            renderer = Renderer(self)
            renderer.run(1000) 
        else:
            for _ in range(frames):
                self.update(0.01)

class Renderer:
    def __init__(self, world):
        self.world = world
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([0, 20])
        self.scatters = []
        # self.ax.grid(False) # Remove gridlines
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        for _ in self.world.objects:
            scatter, = self.ax.plot([], [], [], 'o', markersize=8)
            self.scatters.append(scatter)

    def init_func(self):
        for scatter in self.scatters:
            scatter.set_data([], [])
            scatter.set_3d_properties([])
        return self.scatters

    def update_func(self, frame):
        self.world.update(0.01)  # update physics
        for i, obj in enumerate(self.world.objects):
            x, y, z = obj.position.v
            self.scatters[i].set_data([x], [y])
            self.scatters[i].set_3d_properties([z])
        return self.scatters

    def run(self, frames):
        anim = FuncAnimation(self.fig, self.update_func, frames=frames, init_func=self.init_func,
                             interval=10, blit=True)
        plt.show()

# Example usage:
world = World(g=10)
quad = PhysicsObject(position=Vector3D(0, 0, 10.0), mass=1.0)
world.add_object(quad) 
world.simulate(1000, render=True)