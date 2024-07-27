import gymnasium as gym
import hover_env
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# Create the environment
env = gym.make('DroneEnv-v0', gravity=9.81, noise_intensity=1.0, render_mode='human')

# Set target state
target_position = np.array([0.0, 0.0, 4.0])  # x, y, z
target_orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw

# Control parameters
position_gains = {'p': 5.0, 'i': 0.1, 'd': 10.0}
attitude_gains = {'p': 10.0, 'i': 0.1, 'd': 5.0}

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def update(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# Initialize PID controllers
dt = 0.01
position_controllers = [PIDController(position_gains['p'], position_gains['i'], position_gains['d'], dt) for _ in range(3)]
attitude_controllers = [PIDController(attitude_gains['p'], attitude_gains['i'], attitude_gains['d'], dt) for _ in range(3)]

def quaternion_to_euler(q):
    r = R.from_quat(q)
    return r.as_euler('xyz', degrees=False)

observation, info = env.reset()
print(f"Initial position: {observation[:3]}")

for step in range(1000):
    current_position = observation[:3]
    current_velocity = observation[3:6]
    current_quaternion = observation[6:10]
    current_angular_velocity = observation[10:]

    current_orientation = quaternion_to_euler(current_quaternion)

    # Position control (outer loop)
    position_error = target_position - current_position
    desired_acceleration = np.array([position_controllers[i].update(position_error[i]) for i in range(3)])

    # Calculate desired orientation
    thrust_magnitude = np.linalg.norm(desired_acceleration)
    desired_orientation = np.array([
        np.arctan2(desired_acceleration[1], desired_acceleration[2]),
        np.arctan2(-desired_acceleration[0], np.sqrt(desired_acceleration[1]**2 + desired_acceleration[2]**2)),
        target_orientation[2]  # Maintain target yaw
    ])

    # Attitude control (inner loop)
    orientation_error = desired_orientation - current_orientation
    attitude_control = np.array([attitude_controllers[i].update(orientation_error[i]) for i in range(3)])

    # Combine controls
    action = np.concatenate([attitude_control, [thrust_magnitude]])
    
    # Clip actions to avoid extreme values
    action = np.clip(action, -1, 1)

    observation, reward, terminated, truncated, info = env.step(action)

    if step % 50 == 0 or terminated or truncated:
        print(f"Step {step}: Position: {current_position}, Orientation: {current_orientation}")

    # Stopping condition
    if np.all(np.abs(position_error) < 0.01) and np.all(np.abs(current_velocity) < 0.01) and np.all(np.abs(orientation_error) < 0.01) and np.all(np.abs(current_angular_velocity) < 0.01):
        print(f"Stabilized at step {step}")
        break

env.close()