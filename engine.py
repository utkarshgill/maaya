import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d.art3d import Line3DCollection
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

    def as_rotation_matrix(self):
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

    def simulate(self, frames, render=False):
        if render:
            renderer = Renderer(self)
            renderer.run(frames) 
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
        self.quadcopter_lines = []
        for obj in self.world.objects:
            # Define lines for a quadcopter X model with front half red and back half black
            lines = [[(-1, -1, 0), (0, 0, 0)], [(0, 0, 0), (1, 1, 0)],
                     [(1, -1, 0), (0, 0, 0)], [(0, 0, 0), (-1, 1, 0)]]
            colors = ['k', 'r', 'k', 'r']  # Alternating colors for the arms
            line_collection = Line3DCollection(lines, colors=colors, linewidths=2)
            self.quadcopter_lines.append(self.ax.add_collection3d(line_collection))

    def update_func(self, frame):
        self.world.update(0.01)  # update physics
        for i, obj in enumerate(self.world.objects):
            position = obj.position.v
            orientation = obj.orientation.as_rotation_matrix()
            # Update positions of line segments based on the object's position and orientation
            lines = np.array([[(-1, -1, 0), (0, 0, 0)], [(0, 0, 0), (1, 1, 0)],
                              [(1, -1, 0), (0, 0, 0)], [(0, 0, 0), (-1, 1, 0)]])
            lines = np.dot(lines.reshape(-1, 3), orientation).reshape(-1, 2, 3)  # Rotate lines
            lines += position  # Translate lines
            self.quadcopter_lines[i].set_segments(lines)
        return self.quadcopter_lines

    def run(self, frames):
        anim = FuncAnimation(self.fig, self.update_func, frames=frames, init_func=lambda: self.quadcopter_lines,
                             interval=10, blit=False)
        plt.show()

class QuadCopter(PhysicsObject):
    def __init__(self, position=None, velocity=None, acceleration=None, mass=1.0,
                 orientation=None, angular_velocity=None, ctrl=None):
        L = 0.3  # Length of each arm from center to tip
        num_arms = 4

        # Calculate the mass of each arm assuming equal distribution
        m_arm = mass / num_arms

        # Calculate the moments of inertia
        # Inertia about the Z-axis, assuming arms act like rods rotating around their center
        I_z = num_arms * (1/12) * m_arm * (L**2)

        # Inertia about the X and Y axes
        # Considering arms are at 45 degrees, projecting to L*cos(45 degrees) for each axis
        L_projected = L * np.cos(np.pi / 4)  # cos(45 degrees) = sqrt(2)/2
        I_x = num_arms * (1/3) * m_arm * (L_projected**2)
        I_y = I_x  # Symmetry in the configuration

        # Construct the inertia matrix
        inertia = np.array([
            [I_x, 0, 0],
            [0, I_y, 0],
            [0, 0, I_z]
        ])
        self.ctrl = ctrl

        super().__init__(position, velocity, acceleration, mass, orientation, angular_velocity, inertia)
        
    def update(self, dt):

        T = self.ctrl.update(self.position.v[2])
        self.command([T, 0, 0, 0])
        super().update(dt)

    def command(self, c):
        T, R, Y, P = c
        self.apply_torque(Vector3D(-P, -R, Y))
        self.apply_force(Vector3D(0, 0, T))

    def __repr__(self):
        return (f"QuadCopter(position={self.position}, velocity={self.velocity}, "
                f"acceleration={self.acceleration}, mass={self.mass}, "
                f"orientation={self.orientation}, motor_speeds={self.motor_speeds})")

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.previous_error = 0
        self.integral = 0
        
    def update(self, current_value):
        error = self.setpoint - current_value

        
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.previous_error = error
        # print(error, output)
        return output

frames = 1000
world = World(g=9.81)
z_ctrl = PIDController(10.0, 5.0, 5.0, setpoint=10.0, dt=0.01)
quad = QuadCopter(position=Vector3D(0, 0, 10.0), mass=1.0, ctrl = z_ctrl)

world.add_object(quad) 

world.simulate(frames, render=True)