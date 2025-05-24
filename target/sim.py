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
from gymnasium import spaces, Env
import numpy as np
from sim import (
    Vector3D, Quaternion, Body, World,
    GravitationalForce, RungeKuttaIntegrator, GroundCollision,
    IMUSensor, Motor, Renderer, Actuator
)
from firmware.control import StabilityController, GenericMixer
from firmware.hal import HAL
from firmware.hil import Keyboard, DualSense
from sim.engine import Quadcopter, GraspActuator
from common.scheduler import Scheduler  # scheduler for discrete stepping

class Simulator(Env):
    """Gymnasium environment for a quadrotor hovering demo using miniflight."""
    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, render_mode='human', dt=0.01, frame_skip=1, config='X', hil=None):
        super().__init__()
        # External HIL object provided by Board (for pick/drop)
        self.hil = hil
        self.render_mode = render_mode
        self.dt = dt
        self.frame_skip = frame_skip
        self.config = config
        self.carrying_box = None
        self.pickup_radius = 0.75  # slightly increased grasping distance for easier pickup

        # Instantiate stability controller (shared with Board interface)
        self.stability_ctrl = StabilityController()

        self._build_sim()
        # Build a private scheduler for discrete stepping (actuate→integrate→sense)
        self._sched = Scheduler(time_fn=lambda: self.world.time)
        # 1) Actuate based on last control_command
        self._sched.add_task(lambda: self.world._actuate(self.dt), period=self.dt)
        # 2) Integrate dynamics and record state
        def _integrate_and_record():
            self.world._integrate(self.dt)
            self.world.time += self.dt
            self.world.current_state, self.world.current_flat = self.world.get_state()
        self._sched.add_task(_integrate_and_record, period=self.dt)
        # 3) Sense for next control cycle
        self._sched.add_task(lambda: self.world._sense(self.dt), period=self.dt)

        # Define spaces
        spec = self.world.state_spec
        dim = (1 + spec['position']['shape'][0] + spec['velocity']['shape'][0]
               + spec['orientation']['shape'][0] + spec['angular_velocity']['shape'][0])
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        # Action: [motor1_thrust, motor2_thrust, motor3_thrust, motor4_thrust, pick_flag]
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)

        self.renderer = None
        # Flag to track keyboard pickup/drop handling
        self._kb_handled = False

    def _build_sim(self):
        # 1. Create and configure the body (Quadcopter)
        self.quad = Quadcopter(position=Vector3D(0, 0, 0.1))
        self.quad.integrator = RungeKuttaIntegrator()
        self.quad.add_sensor(IMUSensor(accel_noise_std=0.0, gyro_noise_std=0.0))
        self.quad.urdf_filename = "quadrotor.urdf" # Explicitly assign URDF for the quad
        
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
        for idx, (r, s) in enumerate(zip(motor_positions, spins)):
            self.quad.add_actuator(Motor(idx, r_body=r, spin=s,
                                          thrust_noise_std=0.0, torque_noise_std=0.0))

        # 2. Define forces for the world
        world_forces = [
            GravitationalForce(),
            GroundCollision(ground_level=0.0, restitution=0.5)
        ]

        # 3. Create the World
        # self.dt is available from QuadHoverEnv's constructor.
        sim_world = World(forces=world_forces, dt=self.dt)

        # 4. Add the configured body to the world
        sim_world.add_body(self.quad)

        # Spawn pickable boxes in the world
        self.boxes = []
        BOX_HALF_HEIGHT = 0.25  # For cube.urdf which is 0.5x0.5x0.5
        for x_coord in [6.0, 10.0]:
            pos = Vector3D(x_coord, 0, BOX_HALF_HEIGHT)
            box = Body(position=pos, mass=0.5, inertia=np.eye(3)*0.01)
            box.urdf_filename = 'cube.urdf'
            box.half_height = BOX_HALF_HEIGHT
            self.boxes.append(box)
            sim_world.add_body(box)

        # 5. Use World directly (no Simulator needed)
        self.world = sim_world

    def reset(self, *, seed=None, options=None):
        """
        Reset the simulator to initial state and return initial observation.
        """
        # Rebuild simulation to initial conditions
        self._build_sim()
        # Ensure state caches are populated
        self.world.update()
        obs, _ = self.get_state()
        return obs, {}

    def get_state(self):
        """Return observation (flat numpy array) and full state dict."""
        # Ensure state is up-to-date. If still None (e.g. before first update), call world.update().
        if self.world.current_flat is None:
            self.world.update()
        return self.world.current_flat, self.world.current_state

    def step(self, action):
        """Advance simulation using the externally provided *action* vector via the internal scheduler."""
        # Parse pick/drop flag from action
        pick_flag = float(action[4]) if len(action) > 4 else 0.0
        # Inject per-motor thrusts directly from firmware
        self.quad.motor_thrusts = list(action[:4])
        # Handle pick/drop if requested
        if pick_flag:
            self._handle_pick_drop()
        # Advance physics frames: each sched.step() runs sense→actuate→integrate for dt
        for _ in range(self.frame_skip):
            self._sched.step()
        obs, _ = self.get_state()
        reward = 0.0
        done = False
        info = {}
        # Automatic rendering for human mode
        if self.render_mode:
            # lazy-init renderer
            if self.renderer is None:
                self.renderer = Renderer(self.world, config=self.config)
            # HIL keyboard/pick-drop inside renderer
            if self.hil and hasattr(self.hil, 'handle_pybullet'):
                self.hil.handle_pybullet(self.renderer)
            # draw the scene
            self.renderer.draw()
        return obs, reward, done, info

    def close(self):
        if not self.renderer:
            return
        # Disconnect PyBullet client if present
        if hasattr(self.renderer, 'p') and hasattr(self.renderer, 'client'):
            try:
                self.renderer.p.disconnect(self.renderer.client)
            except Exception:
                pass

    def _on_key_press(self, event):
        """Handle key press events for keyboard fallback control."""
        key = event.key
        self.stability_ctrl.key_state[key] = True

    def _on_key_release(self, event):
        """Handle key release events for keyboard fallback control."""
        key = event.key
        self.stability_ctrl.key_state[key] = False

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

    def run(self, realtime=True):
        """Run the simulation and rendering in real time using multi-rate scheduling."""
        # Reset environment
        self.reset()
        # Initialize renderer once
        if self.render_mode and self.renderer is None:
            self.renderer = Renderer(self.world, config=self.config)

        # Inline automatic render callback (handles HIL pick/drop and drawing)
        def _auto_render():
            if self.hil and hasattr(self.hil, 'handle_pybullet'):
                self.hil.handle_pybullet(self.renderer)
            self.renderer.draw()

        # Run real-time loop via World with automatic rendering
        self.world.run(render_fn=_auto_render if self.render_mode else None,
                       render_fps=self.metadata.get('render_fps', 50))

# -------------------------------------------------------------------------
# Board moved in from firmware/board.py
class Board(HAL):
    """Firmware-side abstraction that runs control algorithms at its own rate.

    Board.update(obs) -> actions
    obs is expected to be a flat numpy array matching World.get_state() spec.
    """
    def __init__(self, dt: float = 0.01, controller: StabilityController = None,
                 config: str = 'X', arm_length: float = 0.3, kT: float = 1.0, kQ: float = 0.02):
        super().__init__(config=None)
        self.dt = dt
        self._sim_time = 0.0
        self._latest_obs = None
        self._body = None
        self._latest_action = None
        self.controller = controller if controller is not None else StabilityController()
        # On-board mixer
        self.config = config
        L = arm_length
        if self.config.upper() == 'X':
            diag = L / np.sqrt(2)
            motor_positions = [
                Vector3D(diag, diag, 0),
                Vector3D(-diag, diag, 0),
                Vector3D(-diag, -diag, 0),
                Vector3D(diag, -diag, 0),
            ]
        else:
            motor_positions = [
                Vector3D(L, 0, 0),
                Vector3D(0, L, 0),
                Vector3D(-L, 0, 0),
                Vector3D(0, -L, 0),
            ]
        spins = [1, -1, 1, -1]
        self.mixer = GenericMixer(motor_positions, spins, kT, kQ)
        # HIL interfaces
        self.keyboard = Keyboard()
        self.dualsense = DualSense()
        self.hil = self.dualsense if getattr(self.dualsense, 'h', None) else self.keyboard
        # Scheduler
        self._sched = Scheduler(time_fn=lambda: self._sim_time)
        self._sched.add_task(self._read_task, period=self.dt)
        self._sched.add_task(self._control_task, period=self.dt)
        self._pick_handled = False

    def update(self, obs):
        self._latest_obs = obs
        try:
            self._sim_time = float(obs[0])
        except Exception:
            self._sim_time += self.dt
        self._sched.step()
        if self._latest_action is None and self._body is not None:
            self._latest_action = self.controller.update(self._body, self.dt)
        return self._latest_action

    def _read_task(self):
        if self._latest_obs is None:
            return
        obs = self._latest_obs
        px, py, pz = obs[1:4]
        vx, vy, vz = obs[4:7]
        qw, qx, qy, qz = obs[7:11]
        wx, wy, wz = obs[11:14]
        self._body = Body(
            position=Vector3D(px, py, pz),
            velocity=Vector3D(vx, vy, vz),
            orientation=Quaternion(qw, qx, qy, qz),
            angular_velocity=Vector3D(wx, wy, wz),
            mass=1.0
        )

    def _control_task(self):
        if self._body is None:
            return
        if self.hil:
            self.hil.update(self.controller, self.dt)
        base_cmds = self.controller.update(self._body, self.dt)
        thrusts = list(self.mixer.mix(base_cmds))
        # pick/drop
        pick_flag = 0.0
        kb_down = False
        if isinstance(self.hil, Keyboard):
            kb_down = bool(self.hil.key_state.get('x')) or bool(self.hil.key_state.get(' '))
        if hasattr(self.hil, 'cross_pressed'):
            kb_down = kb_down or bool(self.hil.cross_pressed)
        if kb_down and not self._pick_handled:
            pick_flag = 1.0
            self._pick_handled = True
        elif not kb_down:
            self._pick_handled = False
        self._latest_action = thrusts + [pick_flag]

    def write(self, commands):
        return commands
# -------------------------------------------------------------------------

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
    """Minimal manual loop: Board scheduler driving control and Simulator scheduler driving physics."""
    # Board is defined above in this file
    # Initialize firmware Board (contains HIL) first
    board = Board(dt=0.01)
    # Initialize environment with Board's HIL for pick/drop and keyboard
    env = Simulator(render_mode='human', dt=board.dt, hil=board.hil)

    # Reset to get initial observation
    obs, _ = env.reset()
    done = False
    try:
        while not done:
            # 1) Firmware control computes action via its internal scheduler
            action = board.update(obs)
            # 2) Simulator steps its scheduler and returns next observation
            obs, reward, done, info = env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()