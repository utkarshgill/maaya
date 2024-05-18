import numpy as np
import sys
sys.path.insert(0, '/Users/engelbart/Desktop/stuff')
from maaya import Vector3D, Quaternion, Body, World, Renderer, NoiseGenerator, GravitationalForce

class QuadCopter(Body):
    def __init__(self, position=None, velocity=None, acceleration=None, mass=1.0,
                 orientation=None, angular_velocity=None, ctrl=None, board=None):
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

        inertia = np.array([
            [I_x, 0, 0],
            [0, I_y, 0],
            [0, 0, I_z]
        ])
       
        self.ctrl = ctrl
        self.board = board

        super().__init__(position, velocity, acceleration, mass, orientation, angular_velocity, inertia)
        

    # PID stability loop

    def update(self, dt):
        # Assuming self.position.v[2] is altitude, and self.orientation provides roll, pitch, and yaw directly
        
        self.command([1, 0, 0, 0])
        super().update(dt)


  

    def command(self, c):
        T, R, P, Y = c

        # write logic to apply thrust downward realtive to the orientation of the quad
        x, y, z = self.orientation.rotate(Vector3D(0, 0, T))
        thrust_vector = Vector3D(x, y, z)
        self.apply_torque(Vector3D(R, P, Y))
        self.apply_force(thrust_vector)

    def __repr__(self):
        return (f"QuadCopter(position={self.position}, velocity={self.velocity}, "
                f"acceleration={self.acceleration}, mass={self.mass}, "
                f"orientation={self.orientation}, motor_speeds={self.motor_speeds})")

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, dt=0.01):
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
    
class DroneController:
    def __init__(self):
        # PID controllers for roll, pitch, yaw, and altitude
        self.altitude_pid = PIDController(kp=1.2, ki=0.01, kd=0.5, setpoint=10.0)  # Altitude set to 10 meters
        self.roll_pid = PIDController(kp=1.0, ki=0.0, kd=0.1)
        self.pitch_pid = PIDController(kp=1.0, ki=0.0, kd=0.1)
        self.yaw_pid = PIDController(kp=1.0, ki=0.0, kd=0.1)

    def update(self, sensors):
        # Sensors should provide current altitude, roll, pitch, and yaw
        current_altitude, current_roll, current_pitch, current_yaw = sensors

        # Update each PID controller
        altitude_output = self.altitude_pid.update(current_altitude)
        roll_output = self.roll_pid.update(current_roll)
        pitch_output = self.pitch_pid.update(current_pitch)
        yaw_output = self.yaw_pid.update(current_yaw)

        # Convert PID outputs to drone's command format (T, R, Y, P)
        return [altitude_output, -roll_output,-pitch_output, yaw_output ]


frames = 1000
world = World(noise=NoiseGenerator(intensity=0))
# z_ctrl = PIDController(10.0, 10.0, 5.0, setpoint=10.0, dt=0.01)
ctrl = DroneController()

quad = QuadCopter(position=Vector3D(0, 0, 10.0), orientation=Quaternion(0, 0, 1, 0), mass=1.0)

world.add_object(quad) 

r = Renderer(world)
r.run(frames)


