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
from maaya.controller import Controller, PIDController
from maaya.actuator import GenericMixer, Motor

class DroneController(Controller):
    def __init__(self):
        self.x_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=2.0)
        self.y_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=2.0)
        self.z_pid = PIDController(kp=1.5, ki=0.2, kd=3.0, setpoint=10.0)
        # Inner‐loop attitude gains tuned for quick but damped response
        self.roll_pid  = PIDController(kp=2.0, ki=0.0, kd=0.3)
        self.pitch_pid = PIDController(kp=2.0, ki=0.0, kd=0.3)
        self.yaw_pid = PIDController(kp=1.0, ki=0.0, kd=0.1)

    def update(self, bodies, dt):
        # Full-state update for each body
        for body in bodies:
            x, y, z = body.position.v
            roll, pitch, yaw = body.orientation.to_euler()
            # Outer-loop PIDs with explicit setpoints
            pitch_set = np.clip(self.x_pid.update(x, self.x_pid.setpoint, dt), -0.3, 0.3)
            roll_set = np.clip(-self.y_pid.update(y, self.y_pid.setpoint, dt), -0.3, 0.3)
            # Gravity compensation: hover thrust baseline ≈ m·g (here m=1 kg).
            gravity_ff = 9.8  # N
            thrust = gravity_ff + self.z_pid.update(z, self.z_pid.setpoint, dt)
            thrust = float(np.clip(thrust, 0.0, 20.0))
            # Inner-loop: drive attitude PIDs toward those setpoints
            roll_cmd = self.roll_pid.update(roll, roll_set, dt)
            pitch_cmd = self.pitch_pid.update(pitch, pitch_set, dt)
            yaw_cmd = self.yaw_pid.update(yaw, self.yaw_pid.setpoint, dt)
            # Constrain generated torques to reasonable bounds
            roll_cmd = float(np.clip(roll_cmd, -0.5, 0.5))
            pitch_cmd = float(np.clip(pitch_cmd, -0.5, 0.5))
            yaw_cmd = float(np.clip(yaw_cmd, -0.3, 0.3))
            body.control_command = [thrust, roll_cmd, pitch_cmd, yaw_cmd]

    def set_xyz_target(self, x_target, y_target, z_target):
        """Dynamically update horizontal position set‐points."""
        self.x_pid.setpoint = x_target
        self.y_pid.setpoint = y_target
        self.z_pid.setpoint = z_target

frames = 1000
world = World(gravity=GravitationalForce())
ctrl = DroneController()
L = 0.3
# Extend Body to create a Quadcopter class
class Quadcopter(Body):
    def __init__(self, position=None, mass=1.0, arm_length=0.3):
        num_arms = 4
        m_arm = mass / num_arms
        L = arm_length
        I_z = num_arms * (1/12) * m_arm * (L**2)
        L_proj = L * np.cos(np.pi / 4)
        I_x = num_arms * (1/3) * m_arm * (L_proj**2)
        I_y = I_x
        inertia = np.array([[I_x, 0, 0], [0, I_y, 0], [0, 0, I_z]])
        super().__init__(position=position if position is not None else Vector3D(0, 0, 10.0),
                         mass=mass, inertia=inertia)
        self.arm_length = arm_length

# Instantiate the quadcopter
quad = Quadcopter(position=Vector3D(0, 0, 10.0), mass=1.0, arm_length=0.3)

# Register the quadcopter body with the world
world.add_body(quad)

# Register modular components: sensor, controller, actuator
quad.add_sensor(IMUSensor(accel_noise_std=0.02, gyro_noise_std=0.005))
quad.add_controller(ctrl)

# Register mixer and four motors with individual noise parameters
motor_positions = [Vector3D( L, 0, 0), Vector3D(0,  L, 0), Vector3D(-L, 0, 0), Vector3D(0, -L, 0)]
spins = [1, -1, 1, -1]
quad.add_actuator(GenericMixer(motor_positions, spins, kT=1.0, kQ=0.02))

# Motor positions in body frame (plus configuration)
motor_configs = [
    (0, Vector3D( L, 0, 0),  1),  # +X, spin CW (+1)
    (1, Vector3D(0,  L, 0), -1),  # +Y, spin CCW (-1)
    (2, Vector3D(-L, 0, 0),  1),  # -X
    (3, Vector3D(0, -L, 0), -1)   # -Y
]

# Different noise specifications for each motor
motor_noise_specs = [
    {'thrust_noise_std': 0.05, 'torque_noise_std': 0.05},
    {'thrust_noise_std': 0.05, 'torque_noise_std': 0.05},
    {'thrust_noise_std': 0.05, 'torque_noise_std': 0.05},
    {'thrust_noise_std': 0.05, 'torque_noise_std': 0.05},
]

for (idx, r_vec, spin), noise in zip(motor_configs, motor_noise_specs):
    quad.add_actuator(Motor(idx, r_body=r_vec, spin=spin,
                              thrust_noise_std=noise['thrust_noise_std'],
                              torque_noise_std=noise['torque_noise_std']))

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
        ctrl_ref.set_xyz_target(x_t, y_t, z_t)
        print(f"→ Updated targets: x={x_t:.2f}, y={y_t:.2f}, z={z_t:.2f}")

# Start reader BEFORE launching the blocking Matplotlib event loop
if __name__ == "__main__":
    threading.Thread(target=_stdin_reader, args=(ctrl,), daemon=True).start()

    r = Renderer(world)
    r.run(frames)


