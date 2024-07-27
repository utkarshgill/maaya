import gymnasium as gym
from gymnasium import spaces
import numpy as np

import matplotlib.pyplot as plt

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
        return np.dot(self.v, other.v)

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

    def to_euler(self):
        w, x, y, z = self.q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
        else:
            pitch = np.arctan2(sinp, np.sqrt(1 - sinp * sinp))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


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
    
    def rotate(self, vector):
        """ Rotate a vector by the quaternion """
        # Convert vector into a quaternion with zero scalar part
        v_quat = Quaternion(0, vector.v[0], vector.v[1], vector.v[2])
        # The rotated quaternion
        rotated_quat = self * v_quat * self.conjugate() 
        # Convert quaternion back to vector
        return Vector3D(rotated_quat.q[1], rotated_quat.q[2], rotated_quat.q[3])
    
    @staticmethod
    def from_axis_angle(axis, angle):
        axis = axis / np.linalg.norm(axis)
        sin_a = np.sin(angle / 2)
        cos_a = np.cos(angle / 2)
        
        w = cos_a
        x = axis[0] * sin_a
        y = axis[1] * sin_a
        z = axis[2] * sin_a
        
        return Quaternion(w, x, y, z)

    @staticmethod
    def from_euler(roll, pitch, yaw):
    # Correcting the order of application to ZYX (yaw, pitch, roll) for proper aerospace sequence
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr

        return Quaternion(w, x, y, z)


    def __repr__(self):
        return f"Quaternion({self.q[0]}, {self.q[1]}, {self.q[2]}, {self.q[3]})"
  
class Body:
    def __init__(self, position=None, velocity=None, acceleration=None, mass=1.0,
                 orientation=None, angular_velocity=None, inertia=None):
        self.position = position if position is not None else Vector3D()
        self.velocity = velocity if velocity is not None else Vector3D()
        self.acceleration = acceleration if acceleration is not None else Vector3D()
        self.mass = mass
        self.orientation = orientation if orientation is not None else Quaternion()
        self.angular_velocity = angular_velocity if angular_velocity is not None else Quaternion()
        self.inertia = np.eye(3)  # Placeholder for moment of inertia as a 3x3 matrix
        self.on_ground = False

    def apply_torque(self, torque, dt):
        # Adjusting torque application to include inertia
        angular_acceleration = np.linalg.inv(self.inertia).dot(torque.v)  # Inertia matrix must be invertible
        angular_acceleration_quaternion = Quaternion(0, *angular_acceleration)
        self.angular_velocity += angular_acceleration_quaternion * dt
        self.angular_velocity.normalize() 

    def update(self, dt, solid_ground=False):
        # Update orientation
        orientation_delta = self.angular_velocity * self.orientation * 0.5 * dt
        self.orientation += orientation_delta
        self.orientation.normalize()

        # Update linear motion
        self.velocity += self.acceleration * dt
        new_position = self.position + self.velocity * dt

        if solid_ground and new_position.v[2] <= 0:
            # Collision with ground
            new_position.v[2] = 0
            self.velocity.v[2] = 0
            self.on_ground = True
        else:
            self.on_ground = False

        self.position = new_position
        self.acceleration = Vector3D()  # Reset acceleration if needed

    def apply_force(self, force):
        # F = m * a, therefore a = F / m
        self.acceleration += Vector3D(*(force.v / self.mass))

    def __repr__(self):
        return f"Body(position={self.position}, velocity={self.velocity}, acceleration={self.acceleration}, mass={self.mass})"

class GravitationalForce:
    def __init__(self, g=10.0):
        self.g = g

    def apply_to(self, obj):
        gravitational_force = Vector3D(0, 0, -self.g * obj.mass)
        obj.apply_force(gravitational_force)
        
class NoiseGenerator:
    def __init__(self, intensity=0.1):
        self.intensity = intensity

    def apply_to(self, obj, dt):
        force_noise = Vector3D(*np.random.normal(0, self.intensity, size=3))
        torque_noise = Vector3D(*np.random.normal(0, self.intensity, size=3))
        obj.apply_force(force_noise)
        obj.apply_torque(torque_noise, dt)

class World:
    def __init__(self, gravity=None, noise=None, solid_ground=False):
        self.objects = []
        self.time = 0
        self.gravity = gravity
        self.noise = noise
        self.solid_ground = solid_ground

    def add_object(self, obj):
        self.objects.append(obj)

    def update(self, dt):
        for obj in self.objects:
            if self.gravity is not None:
                self.gravity.apply_to(obj)
            if self.noise is not None:
                self.noise.apply_to(obj, dt)
            obj.update(dt, self.solid_ground)

class DroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, gravity=10.0, noise_intensity=0.1, solid_ground=True):
        super(DroneEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # Define observation space with appropriate bounds
        obs_low = np.array([-np.inf, -np.inf, -np.inf,  # position
                            -np.inf, -np.inf, -np.inf,  # velocity
                            -1, -1, -1, -1,             # orientation (quaternion)
                            -np.inf, -np.inf, -np.inf]) # angular velocity
        obs_high = np.array([np.inf, np.inf, np.inf,
                             np.inf, np.inf, np.inf,
                             1, 1, 1, 1,
                             np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Create the world with configurable gravity and noise
        self.gravity = gravity
        self.noise_intensity = noise_intensity
        self.world = World(gravity=GravitationalForce(g=self.gravity),
                           noise=NoiseGenerator(intensity=self.noise_intensity),
                           solid_ground=solid_ground)
        self.drone = None
        
        self.fig = None
        self.ax = None
        self.trajectory = []

        self.drone_size = 0.3 * np.sqrt(2)  # Drone size is half the distance between the two motors / √2 = (0.6/2) / √2 

        self.solid_ground = solid_ground
        

    def _calc_reward(self):
        # Define target position and orientation
        target_position = Vector3D(0, 0, 1)  # Hovering at 1 meter above origin
        target_orientation = Quaternion(1, 0, 0, 0)  # Upright orientation

        # Calculate errors
        position_error = (self.drone.position - target_position).magnitude()
        orientation_error = 1 - abs(np.dot(self.drone.orientation.q, target_orientation.q))
        velocity_magnitude = self.drone.velocity.magnitude()
        angular_velocity_magnitude = np.linalg.norm(self.drone.angular_velocity.q[1:])

        # Position reward
        if position_error < 0.1:
            position_reward = 10
        elif position_error < 0.5:
            position_reward = 5
        elif position_error < 1.0:
            position_reward = 2
        else:
            position_reward = 1 / (1 + position_error)  # Inverse reward, approaches 1 as error decreases

        # Orientation reward
        if orientation_error < 0.01:
            orientation_reward = 10
        elif orientation_error < 0.1:
            orientation_reward = 5
        elif orientation_error < 0.3:
            orientation_reward = 2
        else:
            orientation_reward = 1 / (1 + orientation_error)

        # Velocity penalty
        velocity_penalty = -velocity_magnitude

        # Angular velocity penalty
        angular_velocity_penalty = -angular_velocity_magnitude

        # Boundary penalties
        boundary_penalty = 0
        env_limit = 5  # Assuming the environment is a 10x10x10 cube centered at the origin
        for coord in self.drone.position.v:
            if abs(coord) > env_limit:
                boundary_penalty -= 50  # Large penalty for each axis that's out of bounds

        # Ground collision penalty
        ground_penalty = 0
        if self.drone.on_ground:
            ground_penalty = -50  # Penalty for touching the ground

        # Combine rewards and penalties
        reward = (
            position_reward +
            2 * orientation_reward +  # Weigh orientation more
            0.5 * velocity_penalty +
            0.5 * angular_velocity_penalty +
            boundary_penalty +
            ground_penalty
        )

        # Bonus for being very close to target state
        if position_error < 0.1 and orientation_error < 0.01 and velocity_magnitude < 0.1 and angular_velocity_magnitude < 0.1:
            reward += 20  # Big bonus for achieving stable hover

        return reward

    def step(self, action, dt=0.01):

        # Apply the action
        torque = Vector3D(action[0], action[1], action[2])
        force = Vector3D(0, 0, action[3])
        local_force = self.drone.orientation.rotate(force) 
        self.drone.apply_force(local_force)
        self.drone.apply_torque(torque, dt=dt)
        
        # Update the world
        self.world.update(dt=dt)
        
        # Get the new state
        observation = self._get_obs()
        
        # Calculate reward
        reward = self._calc_reward()
        
        # Check if episode is done (example: drone too far from origin)
        terminated = np.linalg.norm(self.drone.position.v) > 10
        
        # In this environment, we don't use truncated, but we need to return it
        truncated = False
        
        # Optional info dict
        info = {}
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset drone to initial state
        self.drone = Body(position=Vector3D(0, 0, 0.0), mass=1.0)
        self.world.objects = [self.drone]
        self.trajectory = [self.drone.position.v]

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info
    
    def render(self):
        if self.render_mode == "human":
            if self.fig is None:
                self.fig = plt.figure(figsize=(10, 8))
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.ax.set_xlabel('X (right)')
                self.ax.set_ylabel('Y (depth)')
                self.ax.set_zlabel('Z (up)')
                self.ax.set_title('Drone Trajectory')

            self.trajectory.append(self.drone.position.v)
            
            # self.ax.view_init(elev=20, azim=-45) NOTE: this sets the viewing angle

            self.ax.clear()
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.set_zlim(-5, 5)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('Drone Trajectory')
            
            trajectory = np.array(self.trajectory)
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', alpha=0.5)

            # Add ground plane visualization
            if self.solid_ground:
                x = np.linspace(-5, 5, 2)
                y = np.linspace(-5, 5, 2)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                self.ax.plot_surface(X, Y, Z, alpha=0.5, color='gray')
            
            # Plot drone as a quadcopter
            self._plot_drone()
            
            self.ax.legend()
            
            plt.draw()
            plt.pause(0.0001)
        
        elif self.render_mode == "rgb_array":
            # Here you would return an RGB array representing the image of the environment.
            # This is just a placeholder and should be replaced with actual rendering code.
            return np.zeros((300, 300, 3), dtype=np.uint8)

    def _plot_drone(self):
        pos = self.drone.position.v
        orientation = self.drone.orientation.as_rotation_matrix()

        # Define the drone's arms in X configuration
        # Front-right, front-left, back-left, back-right
        arm_coords = np.array([
            [self.drone_size, -self.drone_size, 0],
            [self.drone_size, self.drone_size, 0],
            [-self.drone_size, self.drone_size, 0],
            [-self.drone_size, -self.drone_size, 0]
        ])

        # Rotate and translate the arms
        rotated_arms = np.dot(orientation, arm_coords.T).T + pos

        # Plot the arms
        self.ax.plot([pos[0], rotated_arms[0, 0]], [pos[1], rotated_arms[0, 1]], [pos[2], rotated_arms[0, 2]], 'r-', linewidth=2)
        self.ax.plot([pos[0], rotated_arms[1, 0]], [pos[1], rotated_arms[1, 1]], [pos[2], rotated_arms[1, 2]], 'r-', linewidth=2)
        self.ax.plot([pos[0], rotated_arms[2, 0]], [pos[1], rotated_arms[2, 1]], [pos[2], rotated_arms[2, 2]], 'r-', linewidth=2)
        self.ax.plot([pos[0], rotated_arms[3, 0]], [pos[1], rotated_arms[3, 1]], [pos[2], rotated_arms[3, 2]], 'r-', linewidth=2)

        # Plot the rotors
        for rotor_pos in rotated_arms:
            self.ax.scatter(rotor_pos[0], rotor_pos[1], rotor_pos[2], color='blue', s=30)

        # Plot the drone's center
        self.ax.scatter(pos[0], pos[1], pos[2], color='green', s=100, label='Drone Center')

        # Plot orientation arrow (pointing in the positive X direction)
        arrow_length = self.drone_size
        arrow_end = pos + orientation[:, 0] * arrow_length
        self.ax.quiver(pos[0], pos[1], pos[2], 
                       arrow_end[0] - pos[0], arrow_end[1] - pos[1], arrow_end[2] - pos[2], 
                       color='g', linewidth=2, arrow_length_ratio=0.2, label='Orientation')

    def _get_obs(self):
        # Construct observation from drone's state
        obs = np.concatenate([
            self.drone.position.v,
            self.drone.velocity.v,
            self.drone.orientation.q,
            self.drone.angular_velocity.q[1:]  # Exclude w component
        ])
        return obs.astype(np.float32)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

from gymnasium.envs.registration import register

register(
    id='DroneEnv-v0',
    entry_point='hover_env:DroneEnv',
    max_episode_steps=5000,
    kwargs={'gravity': 10.0, 'noise_intensity': 0.1, 'solid_ground': True}
)