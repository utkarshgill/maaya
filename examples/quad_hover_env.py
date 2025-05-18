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
try:
    import hid  # provided by `pip install hidapi`
except ImportError:
    hid = None

from maaya import (
    Vector3D, Body, Simulator, World, MultiForce,
    GravitationalForce, RungeKuttaIntegrator, GroundCollision,
    IMUSensor, Controller, PIDController,
    GenericMixer, Motor, Renderer
)

# PS5 DualSense input constants
VENDOR_ID = 0x054C
PRODUCT_ID = 0x0CE6
REPORT_ID = 0x01

# -------------------------------------------------------------------------
# Re-use DroneController and Quadcopter definitions from quad_hover.py
# -------------------------------------------------------------------------
class DroneController(Controller):
    def __init__(self):
        self.x_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=0.0,
                                   measurement_fn=lambda b: b.position.v[0])
        self.y_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=0.0,
                                   measurement_fn=lambda b: b.position.v[1])
        self.z_pid = PIDController(kp=1.5, ki=0.2, kd=3.0, setpoint=1.0,
                                   measurement_fn=lambda b: b.position.v[2])
        self.roll_pid  = PIDController(kp=2.0, ki=0.0, kd=0.3,
                                      measurement_fn=lambda b: b.orientation.to_euler()[0])
        self.pitch_pid = PIDController(kp=2.0, ki=0.0, kd=0.3,
                                      measurement_fn=lambda b: b.orientation.to_euler()[1])
        self.yaw_pid   = PIDController(kp=1.0, ki=0.0, kd=0.1,
                                      measurement_fn=lambda b: b.orientation.to_euler()[2],
                                      wrap=True)

    def update(self, body, dt):
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

class PS5AttitudeController(Controller):
    """
    Reads PS5 DualSense sticks and drives DroneController's built-in PID loops for altitude and attitude.
    Maintains stable hover and attitude hold, with sticks modifying target thrust and target orientation.
    """
    def __init__(self, drone_ctrl, thrust_gain=5.0, max_tilt_rad=0.5, yaw_rate_gain=np.pi/2):
        self.ctrl = drone_ctrl
        self.thrust_gain = thrust_gain
        self.max_tilt = max_tilt_rad
        self.yaw_rate_gain = yaw_rate_gain
        if hid:
            try:
                self.h = hid.device()
                self.h.open(VENDOR_ID, PRODUCT_ID)
                try:
                    self.h.set_nonblocking(True)
                except AttributeError:
                    pass
            except (OSError, IOError) as e:
                print(f"Warning: could not open PS5 controller HID device: {e}")
                self.h = None
        else:
            self.h = None
        # Initialize keyboard fallback state
        self.key_state = {}

    def update(self, body, dt):
        # If HID unavailable, skip
        if not self.h:
            return self._keyboard_control(body, dt)
        data = self.h.read(64)
        if not data or data[0] != REPORT_ID:
            return None
        # Unpack sticks
        lx, ly, rx, ry = data[1], data[2], data[3], data[4]
        # Deadzone
        def deadzone(val, dz=0.1):
            return val if abs(val) > dz else 0.0

        norm_lx = deadzone((lx - 127) / 127.0)
        norm_ly = deadzone((127 - ly) / 127.0)
        norm_rx = deadzone((rx - 127) / 127.0)
        norm_ry = deadzone((127 - ry) / 127.0)

        # --- Altitude setpoint change (LY throttle) ---
        z_sp = self.ctrl.z_pid.setpoint + norm_ly * self.thrust_gain * dt
        self.ctrl.z_pid.setpoint = float(np.clip(z_sp, 0.0, 20.0))  # Lower max altitude

        # --- Attitude setpoints directly to inner PIDs ---
        max_tilt = 0.3  # radians (~17 deg)
        self.ctrl.roll_pid.setpoint = np.clip(norm_rx * max_tilt, -max_tilt, max_tilt)
        self.ctrl.pitch_pid.setpoint = np.clip(norm_ry * max_tilt, -max_tilt, max_tilt)

        # Yaw: integrate yaw rate (negative sign to correct direction)
        yaw_sp = self.ctrl.yaw_pid.setpoint - norm_lx * (np.pi/3) * dt  # slower yaw
        # Wrap yaw setpoint continuously into [-pi, pi] for seamless rotation
        self.ctrl.yaw_pid.setpoint = ((yaw_sp + np.pi) % (2 * np.pi)) - np.pi

        # Compute individual PID outputs
        thrust_cmd = np.clip(9.8 + self.ctrl.z_pid.update(body, dt), 0.0, 15.0)
        roll_cmd = np.clip(self.ctrl.roll_pid.update(body, dt), -0.3, 0.3)
        pitch_cmd = np.clip(self.ctrl.pitch_pid.update(body, dt), -0.3, 0.3)
        yaw_cmd = np.clip(self.ctrl.yaw_pid.update(body, dt), -0.3, 0.3)

        return [float(thrust_cmd), float(roll_cmd), float(pitch_cmd), float(yaw_cmd)]

    def _keyboard_control(self, body, dt):
        """Keyboard fallback: w/s for thrust, a/d for yaw, arrow keys for pitch and roll."""
        # Altitude control (w/s)
        if self.key_state.get('w', False):
            z_sp = self.ctrl.z_pid.setpoint + self.thrust_gain * dt
            self.ctrl.z_pid.setpoint = float(np.clip(z_sp, 0.0, 20.0))
        if self.key_state.get('s', False):
            z_sp = self.ctrl.z_pid.setpoint - self.thrust_gain * dt
            self.ctrl.z_pid.setpoint = float(np.clip(z_sp, 0.0, 20.0))
        # Attitude control (arrows)
        max_tilt = self.max_tilt
        # Roll (left/right)
        if self.key_state.get('left', False):
            self.ctrl.roll_pid.setpoint = -max_tilt
        elif self.key_state.get('right', False):
            self.ctrl.roll_pid.setpoint = max_tilt
        else:
            self.ctrl.roll_pid.setpoint = 0.0
        # Pitch (up/down)
        if self.key_state.get('up', False):
            self.ctrl.pitch_pid.setpoint = max_tilt
        elif self.key_state.get('down', False):
            self.ctrl.pitch_pid.setpoint = -max_tilt
        else:
            self.ctrl.pitch_pid.setpoint = 0.0
        # Yaw control (a/d)
        yaw_rate = self.yaw_rate_gain * dt
        if self.key_state.get('a', False):
            yaw_sp = self.ctrl.yaw_pid.setpoint + yaw_rate
            self.ctrl.yaw_pid.setpoint = ((yaw_sp + np.pi) % (2 * np.pi)) - np.pi
        if self.key_state.get('d', False):
            yaw_sp = self.ctrl.yaw_pid.setpoint - yaw_rate
            self.ctrl.yaw_pid.setpoint = ((yaw_sp + np.pi) % (2 * np.pi)) - np.pi
        # Compute PID outputs
        thrust_cmd = float(np.clip(9.8 + self.ctrl.z_pid.update(body, dt), 0.0, 15.0))
        roll_cmd = float(np.clip(self.ctrl.roll_pid.update(body, dt), -max_tilt, max_tilt))
        pitch_cmd = float(np.clip(self.ctrl.pitch_pid.update(body, dt), -max_tilt, max_tilt))
        yaw_cmd = float(np.clip(self.ctrl.yaw_pid.update(body, dt), -0.3, 0.3))
        return [thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd]

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
        super().__init__(position=position or Vector3D(0,0,0.1),
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

        # Build DroneController and PS5-driven attitude controller
        self.ctrl = DroneController()
        self.att_ctrl = PS5AttitudeController(self.ctrl)
        self._build_sim()

        # Define spaces
        spec = self.sim.state_spec
        dim = (1 + spec['position']['shape'][0] + spec['velocity']['shape'][0]
               + spec['orientation']['shape'][0] + spec['angular_velocity']['shape'][0])
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(0,), dtype=np.float32)

        self.renderer = None
        # Track whether Matplotlib events are attached
        self._key_handlers_attached = False

    def _build_sim(self):
        # 1. Create and configure the body (Quadcopter)
        self.quad = Quadcopter()
        self.quad.integrator = RungeKuttaIntegrator()
        self.quad.add_sensor(IMUSensor(accel_noise_std=0.0, gyro_noise_std=0.0))
        
        # Add the PS5AttitudeController (which wraps DroneController) to the quad.
        # self.att_ctrl is initialized in QuadHoverEnv.__init__ before _build_sim is called.
        self.quad.add_controller(self.att_ctrl)

        # Add actuators to the quad
        L = self.quad.arm_length
        motor_positions = [Vector3D(L, 0, 0), Vector3D(0, L, 0), Vector3D(-L, 0, 0), Vector3D(0, -L, 0)]
        spins = [1, -1, 1, -1]
        self.quad.add_actuator(GenericMixer(motor_positions, spins, kT=1.0, kQ=0.02))
        for idx, (r, s) in enumerate(zip(motor_positions, spins)):
            self.quad.add_actuator(Motor(idx, r_body=r, spin=s,
                                          thrust_noise_std=0.0, torque_noise_std=0.0))

        # 2. Define forces for the world
        world_forces = MultiForce([
            GravitationalForce(), 
            GroundCollision(ground_level=0.0, restitution=0.5)
        ])

        # 3. Create the World
        # self.dt is available from QuadHoverEnv's constructor.
        sim_world = World(gravity=world_forces, dt=self.dt)

        # 4. Add the configured body to the world
        sim_world.add_body(self.quad)

        # 5. Create the Simulator with the configured world
        self.sim = Simulator(world=sim_world)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._build_sim()
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
        # Initialize renderer once
        if self.renderer is None:
            self.renderer = Renderer(self.sim.world)
        # Matplotlib GUI: attach key handlers once
        if hasattr(self.renderer, 'fig'):
            if not self._key_handlers_attached:
                fig = self.renderer.fig
                fig.canvas.mpl_connect('key_press_event', self._on_key_press)
                fig.canvas.mpl_connect('key_release_event', self._on_key_release)
                self._key_handlers_attached = True
        # PyBullet GUI: poll keyboard events
        elif hasattr(self.renderer, 'p'):
            events = self.renderer.p.getKeyboardEvents()
            for code, state in events.items():
                # Determine key down/up
                if state & self.renderer.p.KEY_IS_DOWN or state & self.renderer.p.KEY_WAS_TRIGGERED:
                    down = True
                elif state & self.renderer.p.KEY_WAS_RELEASED:
                    down = False
                else:
                    continue
                # Map codes to our key_state
                if code in (ord('w'), ord('W')):
                    # Track thrust key state
                    self.att_ctrl.key_state['w'] = down
                    # Disable wireframe toggle only on initial press
                    if state & self.renderer.p.KEY_WAS_TRIGGERED:
                        self.renderer.p.configureDebugVisualizer(self.renderer.p.COV_ENABLE_WIREFRAME, 0)
                if code in (ord('s'), ord('S')):
                    self.att_ctrl.key_state['s'] = down
                if code in (ord('a'), ord('A')):
                    self.att_ctrl.key_state['a'] = down
                if code in (ord('d'), ord('D')):
                    self.att_ctrl.key_state['d'] = down
                if code == self.renderer.p.B3G_LEFT_ARROW:
                    self.att_ctrl.key_state['left'] = down
                if code == self.renderer.p.B3G_RIGHT_ARROW:
                    self.att_ctrl.key_state['right'] = down
                if code == self.renderer.p.B3G_UP_ARROW:
                    self.att_ctrl.key_state['up'] = down
                if code == self.renderer.p.B3G_DOWN_ARROW:
                    self.att_ctrl.key_state['down'] = down
        # Draw current state (PyBullet draw advances simulation internally)
        self.renderer.draw()

    def close(self):
        if not self.renderer:
            return
        # Close Matplotlib figure if present
        if hasattr(self.renderer, 'fig'):
            import matplotlib.pyplot as plt
            plt.close(self.renderer.fig)
        # Disconnect PyBullet client if present
        elif hasattr(self.renderer, 'p') and hasattr(self.renderer, 'client'):
            try:
                self.renderer.p.disconnect(self.renderer.client)
            except Exception:
                pass

    def _on_key_press(self, event):
        """Handle key press events for keyboard fallback control."""
        key = event.key
        self.att_ctrl.key_state[key] = True

    def _on_key_release(self, event):
        """Handle key release events for keyboard fallback control."""
        key = event.key
        self.att_ctrl.key_state[key] = False

# -------------------------------------------------------------------------
def _stdin_reader(ctrl_ref):
    print("Type new x y z set-points (e.g. '1.0 1.0 10.0')")
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
    # PS5AttitudeController handles stick input directly, no stdin needed
    # threading.Thread(target=_stdin_reader, args=(env.ctrl,), daemon=True).start()
    obs, info = env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(None)
        # Print the current simulation state at each step if in debug mode
        if DEBUG:
            print(f"[t={obs['time']:.2f}] state = {obs}")
        env.render()
    env.close() 