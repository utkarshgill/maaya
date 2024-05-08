import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from maaya.engine import Vector3D, PhysicsObject, World

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
z_ctrl = PIDController(10.0, 10.0, 5.0, setpoint=10.0, dt=0.01)
quad = QuadCopter(position=Vector3D(0, 0, 10.0), mass=1.0, ctrl=z_ctrl)

world.add_object(quad) 

r = Renderer(world)

r.run(frames)