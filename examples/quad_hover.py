# Interactive target entry needs threading
import threading, sys

import numpy as np
# Ensure project root and src are on PYTHONPATH regardless of where script is launched
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent  # project root
_SRC = _ROOT / 'src'
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from maaya import Vector3D, Body, World, Renderer, GravitationalForce
from maaya.sensor import IMUSensor
from maaya.controller import Controller
from maaya.actuator import QuadrotorActuator

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.previous_error = 0
        self.integral = 0
        
    def update(self, current_value, setpoint):
        error = setpoint - current_value
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
class DroneController:
    def __init__(self):
        # PID controllers for roll, pitch, yaw, and altitude
        self.x_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=2.0)
        self.y_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=2.0)
        self.altitude_pid = PIDController(kp=1.5, ki=0.2, kd=3.0, setpoint=10.0)
        # Inner‐loop attitude gains tuned for quick but damped response
        self.roll_pid  = PIDController(kp=2.0, ki=0.0, kd=0.3)
        self.pitch_pid = PIDController(kp=2.0, ki=0.0, kd=0.3)
        self.yaw_pid = PIDController(kp=1.0, ki=0.0, kd=0.1)

    def update(self, x, y, z, roll, pitch, yaw):
        # Outer‐loop PIDs with explicit setpoints
        pitch_set = np.clip(self.x_pid.update(x, self.x_pid.setpoint), -0.3, 0.3)
        roll_set = np.clip(-self.y_pid.update(y, self.y_pid.setpoint), -0.3, 0.3)
        # Gravity compensation: hover thrust baseline ≈ m·g (here m=1 kg).
        gravity_ff = 9.8  # N
        thrust = gravity_ff + self.altitude_pid.update(z, self.altitude_pid.setpoint)
        thrust     = float(np.clip(thrust, 0.0, 20.0))
        # Inner‐loop: drive attitude PIDs toward those setpoints
        roll_cmd = self.roll_pid.update(roll, roll_set)
        pitch_cmd = self.pitch_pid.update(pitch, pitch_set)
        yaw_cmd = self.yaw_pid.update(yaw, self.yaw_pid.setpoint)

        # Constrain generated torques to reasonable bounds to prevent instability
        roll_cmd  = float(np.clip(roll_cmd,  -0.5, 0.5))
        pitch_cmd = float(np.clip(pitch_cmd, -0.5, 0.5))
        yaw_cmd   = float(np.clip(yaw_cmd,   -0.3, 0.3))
        return [thrust, roll_cmd, pitch_cmd, yaw_cmd]

    def set_xy_target(self, x_target, y_target):
        """Dynamically update horizontal position set‐points."""
        self.x_pid.setpoint = x_target
        self.y_pid.setpoint = y_target

class DroneControllerAdapter(Controller):
    """Wrap existing DroneController into the new Controller API."""
    def __init__(self, drone_ctrl):
        self.drone_ctrl = drone_ctrl

    def update(self, objects, dt):
        for obj in objects:
            x, y, z = obj.position.v
            roll, pitch, yaw = obj.orientation.to_euler()
            cmd = self.drone_ctrl.update(x, y, z, roll, pitch, yaw)
            obj.control_command = cmd

frames = 1000
world = World(gravity=GravitationalForce())
# z_ctrl = PIDController(10.0, 10.0, 5.0, setpoint=10.0, dt=0.01)
ctrl = DroneController()

# Build rigid-body quadcopter with computed inertia matrix
L = 0.3
num_arms = 4
m_arm = 1.0 / num_arms
I_z = num_arms * (1/12) * m_arm * (L**2)
L_proj = L * np.cos(np.pi / 4)
I_x = num_arms * (1/3) * m_arm * (L_proj**2)
I_y = I_x
inertia = np.array([[I_x, 0, 0], [0, I_y, 0], [0, 0, I_z]])
quad = Body(position=Vector3D(0, 0, 10.0), mass=1.0, inertia=inertia)

world.add_object(quad)

# Register modular components: sensor, controller, actuator
world.add_sensor(IMUSensor(accel_noise_std=0.02, gyro_noise_std=0.005))
world.add_controller(DroneControllerAdapter(ctrl))
world.add_actuator(QuadrotorActuator())

# ---------------------------------------------------------------------------
# Spawn a background thread that listens for "x y" pairs typed in the terminal
# and updates the controller's horizontal targets on the fly.

def _stdin_reader(ctrl_ref):
    print("Type new x y z set-points (e.g. '2.0 3.5 10.0') and press <Enter>. Ctrl-D to stop input.")
    for line in sys.stdin:
        parts = line.strip().split()
        if len(parts) != 3:
            print("Please enter exactly three numbers: x y z.")
            continue
        try:
            x_t, y_t, z_t = map(float, parts)
        except ValueError:
            print("Invalid input. Expect numeric values.")
            continue
        ctrl_ref.set_xy_target(x_t, y_t)
        # Update altitude setpoint
        ctrl_ref.altitude_pid.setpoint = z_t
        print(f"→ Updated targets: x={x_t:.2f}, y={y_t:.2f}, z={z_t:.2f}")

# Start reader BEFORE launching the blocking Matplotlib event loop
threading.Thread(target=_stdin_reader, args=(ctrl,), daemon=True).start()

r = Renderer(world)
r.run(frames)


