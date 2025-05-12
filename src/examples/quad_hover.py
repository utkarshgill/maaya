import numpy as np
import sys
# Ensure the directory above 'src' is in the Python path
sys.path.insert(0, '/Users/engelbart/Desktop/stuff')  

from maaya import Vector3D, Body, World, Renderer, GravitationalForce

class QuadCopter(Body):
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
        # Rotate thrust vector into world frame using orientation
        thrust_world = self.orientation.rotate(Vector3D(0, 0, T))
        self.apply_force(thrust_world)

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
world = World(gravity=GravitationalForce(9.81))
z_ctrl = PIDController(10.0, 10.0, 5.0, setpoint=10.0, dt=0.01)
quad = QuadCopter(position=Vector3D(0, 0, 10.0), mass=1.0, ctrl=z_ctrl)

world.add_object(quad) 

r = Renderer(world)

r.run(frames)