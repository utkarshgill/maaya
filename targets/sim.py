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
from firmware.hil import Keyboard
from firmware.control import StabilityController, GenericMixer
from sim.engine import Quadcopter, GraspActuator
from common.scheduler import Scheduler  # scheduler for discrete stepping

class Simulator(Env):
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

        # Instantiate stability controller (shared with Board interface)
        self.stability_ctrl = StabilityController()
        # HIL (keyboard/dualsense) is provided by Board for pick/drop, assign env.hil externally
        self.hil = None

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
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

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
        self.quad.add_actuator(GenericMixer(motor_positions, spins, kT=1.0, kQ=0.02))
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
        # Inject control command into the quad
        self.quad.control_command = action
        # Advance physics frames: each sched.step() runs sense→actuate→integrate for dt
        for _ in range(self.frame_skip):
            self._sched.step()
        obs, _ = self.get_state()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def render(self, mode=None):
        if self.render_mode is None:
            return None
        # Initialize renderer once
        if self.renderer is None:
            self.renderer = Renderer(self.world, config=self.config)
        # Delegate to PyBullet HIL for pick/drop input
        if self.hil and hasattr(self.hil, 'handle_pybullet') and hasattr(self.renderer, 'p'):
            self.hil.handle_pybullet(self.renderer)
        # Handle keyboard pickup/drop (Keyboard HIL only)
        if isinstance(self.hil, Keyboard):
            kb_down = self.hil.key_state.get('x') or self.hil.key_state.get(' ')
            if kb_down and not self._kb_handled:
                self._handle_pick_drop()
                self._kb_handled = True
            elif not kb_down:
                self._kb_handled = False
        # Handle PS5 cross pickup/drop (DualSense HIL only)
        if hasattr(self.hil, 'cross_pressed') and hasattr(self.hil, 'cross_handled'):
            if self.hil.cross_pressed and not self.hil.cross_handled:
                self._handle_pick_drop()
                self.hil.cross_handled = True
            if not self.hil.cross_pressed and self.hil.cross_handled:
                self.hil.cross_handled = False
        # Draw current state (PyBullet draw advances simulation internally)
        self.renderer.draw()

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
        """
        Run the simulation and rendering in real time using multi-rate scheduling.
        """
        # Reset environment
        self.reset()
        # Initialize renderer
        if self.render_mode and self.renderer is None:
            self.renderer = Renderer(self.world, config=self.config)

        # Run real-time loop via World
        self.world.run(render_fn=self.render if self.render_mode else None,
                       render_fps=self.metadata.get('render_fps', 50))

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
    from firmware.board import Board

    # Initialize environment and firmware Board
    env = Simulator(render_mode='human')
    board = Board(dt=env.dt, controller=env.stability_ctrl)
    # Link Board HIL to environment for pick/drop handling
    env.hil = board.hil

    # Reset to get initial observation
    obs, _ = env.reset()
    done = False
    try:
        while not done:
            # 1) Firmware control computes action via its internal scheduler
            action = board.update(obs)
            # 2) Simulator steps its scheduler and returns next observation
            obs, reward, done, info = env.step(action)
            # 3) Render current state
            env.render()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()