import sys, os
import threading  # for background stdin reader
from pathlib import Path

# Ensure project src and root are on PYTHONPATH
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / 'src'
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from maaya.math import Vector3D
from maaya.body import Body
from maaya.simulator import Simulator
from maaya.physics import GravitationalForce, RungeKuttaIntegrator
from maaya.ground import GroundCollision
from maaya.sensor import IMUSensor
from maaya.controller import Controller, PIDController
from maaya.actuator import GenericMixer, Motor
from maaya.render import Renderer

# -------------------------------------------------------------------------
# Re-use DroneController and Quadcopter definitions from quad_hover.py
# -------------------------------------------------------------------------
class DroneController(Controller):
    def __init__(self):
        self.x_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=1.0,
                                   measurement_fn=lambda b: b.position.v[0])
        self.y_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=1.0,
                                   measurement_fn=lambda b: b.position.v[1])
        self.z_pid = PIDController(kp=1.5, ki=0.2, kd=3.0, setpoint=10.0,
                                   measurement_fn=lambda b: b.position.v[2])
        self.roll_pid  = PIDController(kp=2.0, ki=0.0, kd=0.3,
                                      measurement_fn=lambda b: b.orientation.to_euler()[0])
        self.pitch_pid = PIDController(kp=2.0, ki=0.0, kd=0.3,
                                      measurement_fn=lambda b: b.orientation.to_euler()[1])
        self.yaw_pid   = PIDController(kp=1.0, ki=0.0, kd=0.1,
                                      measurement_fn=lambda b: b.orientation.to_euler()[2])

    def update(self, body, dt):
        x, y, z = body.position.v
        roll, pitch, yaw = body.orientation.to_euler()
        pitch_set = np.clip(self.x_pid.update(body, dt), -0.3, 0.3)
        roll_set  = np.clip(-self.y_pid.update(body, dt), -0.3, 0.3)
        thrust = 9.8 + self.z_pid.update(body, dt)
        thrust = float(np.clip(thrust, 0.0, 20.0))
        self.roll_pid.setpoint  = roll_set
        self.pitch_pid.setpoint = pitch_set
        roll_cmd  = float(np.clip(self.roll_pid.update(body, dt), -0.5, 0.5))
        pitch_cmd = float(np.clip(self.pitch_pid.update(body, dt), -0.5, 0.5))
        yaw_cmd   = float(np.clip(self.yaw_pid.update(body, dt), -0.3, 0.3))
        return [thrust, roll_cmd, pitch_cmd, yaw_cmd]

    def set_xyz_target(self, x_target, y_target, z_target):
        """Dynamically update horizontal XYZ set-points."""
        self.x_pid.setpoint = x_target
        self.y_pid.setpoint = y_target
        self.z_pid.setpoint = z_target

class Quadcopter(Body):
    def __init__(self, position=None, mass=1.0, arm_length=0.3):
        import numpy as _np
        num_arms = 4
        m_arm = mass / num_arms
        L = arm_length
        I_z = num_arms * (1/12) * m_arm * (L**2)
        L_proj = L * _np.cos(_np.pi / 4)
        I_x = num_arms * (1/3) * m_arm * (L_proj**2)
        I_y = I_x
        inertia = _np.array([[I_x,0,0],[0,I_y,0],[0,0,I_z]])
        super().__init__(position=position or Vector3D(0,0,10.0),
                         mass=mass, inertia=inertia)
        self.arm_length = arm_length

# -------------------------------------------------------------------------
class QuadHoverEnv(gym.Env):
    """Gymnasium environment for a quadrotor hovering demo using Maaya."""
    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, render_mode='human', dt=0.01, frame_skip=1):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.frame_skip = frame_skip

        # Build controller
        self.ctrl = DroneController()
        # Build quad and integrator
        self.quad = Quadcopter()
        self.quad.integrator = RungeKuttaIntegrator()
        # Attach sensor directly to the quadcopter body
        self.quad.add_sensor(IMUSensor(accel_noise_std=0.02, gyro_noise_std=0.005))

        # Build world + simulator
        self.sim = Simulator(
            body=self.quad,
            controllers=[self.ctrl],
            actuators=[],
            forces=[GravitationalForce(), GroundCollision(ground_level=0.0, restitution=0.5)],
            dt=self.dt,
        )

        # Attach mixers & motors
        L = self.quad.arm_length
        motor_positions = [Vector3D(L,0,0), Vector3D(0,L,0), Vector3D(-L,0,0), Vector3D(0,-L,0)]
        spins = [1, -1, 1, -1]
        self.quad.add_actuator(GenericMixer(motor_positions, spins, kT=1.0, kQ=0.02))
        for idx, (r, s) in enumerate(zip(motor_positions, spins)):
            self.quad.add_actuator(Motor(idx, r_body=r, spin=s,
                                          thrust_noise_std=0.05, torque_noise_std=0.05))

        # Define spaces
        spec = self.sim.state_spec
        dim = (1 + spec['position']['shape'][0] + spec['velocity']['shape'][0]
               + spec['orientation']['shape'][0] + spec['angular_velocity']['shape'][0])
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(0,), dtype=np.float32)

        self.renderer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Re-build quad and simulator while preserving controller instance
        # Create a fresh quad and integrator
        self.quad = Quadcopter()
        self.quad.integrator = RungeKuttaIntegrator()
        # Create new Simulator using existing controller
        self.sim = Simulator(
            body=self.quad,
            controllers=[self.ctrl],
            actuators=[],
            forces=[GravitationalForce(), GroundCollision(ground_level=0.0, restitution=0.5)],
            dt=self.dt,
        )
        # Re-attach mixers & motors
        L = self.quad.arm_length
        motor_positions = [Vector3D(L,0,0), Vector3D(0,L,0), Vector3D(-L,0,0), Vector3D(0,-L,0)]
        spins = [1, -1, 1, -1]
        self.quad.add_actuator(GenericMixer(motor_positions, spins, kT=1.0, kQ=0.02))
        for idx, (r, s) in enumerate(zip(motor_positions, spins)):
            self.quad.add_actuator(Motor(idx, r_body=r, spin=s,
                                          thrust_noise_std=0.05, torque_noise_std=0.05))
        # Return initial observation
        obs, _ = self.sim.get_state()
        return obs, {}

    def step(self, action):
        # Advance the simulation; control and physics happen inside sim.step()
        for _ in range(self.frame_skip):
            self.sim.step()
        obs, _ = self.sim.get_state()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def render(self, mode=None):
        if self.render_mode is None:
            return None
        if self.renderer is None:
            self.renderer = Renderer(self.sim.world)
        # Only draw current state; physics was already stepped in step()
        self.renderer.draw()

    def close(self):
        if self.renderer:
            import matplotlib.pyplot as plt
            plt.close(self.renderer.fig)

# -------------------------------------------------------------------------
def _stdin_reader(ctrl_ref):
    print("Type new x y z set-points (e.g. '1.0 1.0 10.0') and press <Enter>. Ctrl-D to stop input.")
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

# -------------------------------------------------------------------------
if __name__ == '__main__':
    # Toggle rendering via environment variable: set RENDER=1 to enable, 0 to disable
    RENDER = int(os.getenv('RENDER', '1'))
    render_mode = 'human' if RENDER else None
    # Toggle debug printing via env var: set DEBUG=1 to enable
    DEBUG = int(os.getenv('DEBUG', '0'))
    env = QuadHoverEnv(render_mode=render_mode)
    # Spawn background reader for dynamic target updates
    threading.Thread(target=_stdin_reader, args=(env.ctrl,), daemon=True).start()
    obs, info = env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(None)
        # Print the current simulation state at each step if in debug mode
        if DEBUG:
            print(f"[t={obs['time']:.2f}] state = {obs}")
        env.render()
    env.close() 