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

from sim import (
    Vector3D, Quaternion, Body, Simulator, World, MultiForce,
    GravitationalForce, RungeKuttaIntegrator, GroundCollision,
    IMUSensor, # Controller and PIDController are now in firmware
    GenericMixer, Motor, Renderer, Actuator
)
from firmware.targets import get_target # Import for the new controller

# Grasp actuator for picking up and carrying a box
class GraspActuator(Actuator):
    """Attaches a box rigidly to the quad at a fixed offset."""
    def __init__(self, box, offset=Vector3D(0, 0, -0.1)):
        self.box = box
        self.offset = offset

    def apply_to(self, obj, dt):
        # place box at relative offset in body frame for rigid attachment
        world_offset = obj.orientation.rotate(self.offset)
        self.box.position = obj.position + world_offset
        self.box.velocity = obj.velocity
        # match quad's orientation and angular velocity (rigid attachment)
        self.box.orientation = obj.orientation
        self.box.angular_velocity = obj.angular_velocity
        # disable gravity for carried box
        self.box.mass = self.box.mass  # keep mass
        # no actuators or controllers on box while carried

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

class QuadHoverEnv(gym.Env):
    """Gymnasium environment for a quadrotor hovering demo using miniflight."""
    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, render_mode='human', dt=0.01, frame_skip=1, config='X'):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.frame_skip = frame_skip
        self.config = config
        self.carrying_box = None
        self.pickup_radius = 0.75  # slightly increased grasping distance for easier pickup

        # Build controller using the firmware target system
        target_cfg = get_target("sim_dualsense")
        self.att_ctrl = target_cfg["make_controller"]()
        # self.ctrl is no longer needed separately if att_ctrl has all methods
        # or self.ctrl = self.att_ctrl.stability_ctrl for direct access if needed

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
        self.quad = Quadcopter(position=Vector3D(0, 0, 0.1))
        self.quad.integrator = RungeKuttaIntegrator()
        self.quad.add_sensor(IMUSensor(accel_noise_std=0.0, gyro_noise_std=0.0))
        self.quad.urdf_filename = "quadrotor.urdf" # Explicitly assign URDF for the quad
        
        # Add the PS5AttitudeController (which wraps DroneController) to the quad.
        # self.att_ctrl is initialized in QuadHoverEnv.__init__ before _build_sim is called.
        self.quad.add_controller(self.att_ctrl)

        # Add actuators to the quad (configurable X or plus)
        L = self.quad.arm_length
        if self.config.upper() == 'X':
            diag = L / np.sqrt(2)
            motor_positions = [
                Vector3D(diag, diag, 0),  # front-right
                Vector3D(-diag, diag, 0),  # front-left
                Vector3D(-diag, -diag, 0),  # back-left
                Vector3D(diag, -diag, 0),  # back-right
            ]
        else:
            motor_positions = [
                Vector3D(L, 0, 0),  # front
                Vector3D(0, L, 0),  # left
                Vector3D(-L, 0, 0),  # back
                Vector3D(0, -L, 0),  # right
            ]
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

        # Spawn pickable boxes in the world
        self.boxes = []
        BOX_HALF_HEIGHT = 0.25  # For cube.urdf which is 0.5x0.5x0.5
        # Keep boxes in loading area (at x=6.0 and 10.0 for wider separation)
        for x_coord in [6.0, 10.0]:
            pos = Vector3D(x_coord, 0, BOX_HALF_HEIGHT)
            box = Body(position=pos, mass=0.5, inertia=np.eye(3)*0.01)
            # mark URDF for rendering
            box.urdf_filename = 'cube.urdf'
            # store half-height so ground collision places COM correctly
            box.half_height = BOX_HALF_HEIGHT
            self.boxes.append(box)
            sim_world.add_body(box)

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
            self.renderer = Renderer(self.sim.world, config=self.config)
        # Matplotlib GUI: attach key handlers once
        if hasattr(self.renderer, 'fig'):
            if not self._key_handlers_attached:
                fig = self.renderer.fig
                fig.canvas.mpl_connect('key_press_event', self._on_key_press)
                fig.canvas.mpl_connect('key_release_event', self._on_key_release)
                self._key_handlers_attached = True
        # PyBullet GUI: poll keyboard events
        elif hasattr(self.renderer, 'p'):
            # Reset key states each frame to avoid sticky keys on quick presses
            self.att_ctrl.key_state.clear()
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
                    # Thrust up
                    self.att_ctrl.key_state['w'] = down
                elif code in (ord('s'), ord('S')):
                    # Thrust down
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
                # Pickup/drop with X or Spacebar
                if (code in (ord('x'), ord('X')) or code == ord(' ')) and down:
                    self._handle_pick_drop()
                    continue
        # DualSense Cross button mapping (pickup/drop)
        if hasattr(self.att_ctrl, 'cross_pressed'):
            # Trigger pickup on cross button down event
            if self.att_ctrl.cross_pressed and not getattr(self.att_ctrl, 'cross_handled', False):
                self._handle_pick_drop()
                self.att_ctrl.cross_handled = True
            # Reset handled flag when button is released
            elif not self.att_ctrl.cross_pressed:
                self.att_ctrl.cross_handled = False
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

    def _handle_pick_drop(self):
        """Pick up or drop a box when pressing X."""
        if self.carrying_box is None:
            # try to pick up any nearby box
            for box in self.boxes:
                if (self.quad.position - box.position).magnitude() < self.pickup_radius:
                    # compute offset so box hangs below quad without overlap
                    # offset_z = -(box half-height + small margin)
                    margin = 0.1
                    offset_z = -(box.half_height + margin)
                    actuator = GraspActuator(box, offset=Vector3D(0, 0, offset_z))
                    self.quad.add_actuator(actuator)
                    self.carrying_box = (box, actuator)
                    print("Picked up box")
                    break
        else:
            # drop current box
            box, actuator = self.carrying_box
            if actuator in self.quad.actuators:
                self.quad.actuators.remove(actuator)
            # detach box orientation so it no longer rotates with the quad after drop
            box.angular_velocity = Vector3D(0, 0, 0)
            self.carrying_box = None
            print("Dropped box")

def _stdin_reader(ctrl_ref):
    print("\n\n--- Background command listener started ---")
    print("Enter 'x,y,z' to set target position (e.g. '0,0,1.5')")
    print("Enter 'q' to quit.")
    while True:
        try:
            cmd = input("> ")
            if cmd.lower() == 'q':
                print("Exiting stdin reader.")
                os._exit(0) # Force exit if main thread is stuck
                break
            coords = [float(c.strip()) for c in cmd.split(',')]
            if len(coords) == 3:
                ctrl_ref.set_xyz_target(coords[0], coords[1], coords[2])
                print(f"Target set to: {coords}")
            else:
                print("Invalid input. Use 'x,y,z' format or 'q'.")
        except ValueError:
            print("Invalid coordinate format.")
        except Exception as e:
            print(f"Error in stdin_reader: {e}")

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