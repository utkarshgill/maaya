# engine.py
# Core simulation engine: messaging, scheduling, world, and high-level simulator

import numpy as np
from common.math import Vector3D

# --- World ---
class World:
    """Simulation world orchestrating sense→control→actuate→integrate."""
    DEFAULT_DIVISORS = {
        'sense': 1,
        'control': 1,
        'actuate': 1,
        'integrate': 1,
    }

    def __init__(self, gravity=None, dt: float = 0.01, **divisors):
        self.bodies = []
        self.time: float = 0.0
        self.dt: float = dt
        self.gravity = gravity

        # Per-phase divisors and step counter
        self.divisors = {**World.DEFAULT_DIVISORS, **divisors}
        self.step_count = 0

    def add_body(self, body):
        self.bodies.append(body)

    def update(self, dt: float | None = None):
        if dt is not None:
            self.dt = dt
        s = self.step_count
        if s % self.divisors['sense'] == 0:
            self._sense(self.dt)
        if s % self.divisors['control'] == 0:
            self._control(self.dt)
        if s % self.divisors['actuate'] == 0:
            self._actuate(self.dt)
        if s % self.divisors['integrate'] == 0:
            self._integrate(self.dt)
            self.time += self.dt
        self.step_count += 1

    step = update

    def simulate(self, frames: int):
        for _ in range(frames):
            self.update()

    def _sense(self, dt: float):
        for body in self.bodies:
            for sensor in getattr(body, 'sensors', []):
                sensor.read([body], dt)
            body.acceleration = Vector3D()  # Reset acceleration after sensors read

    def _control(self, dt: float):
        for body in self.bodies:
            for controller in getattr(body, 'controllers', []):
                cmd = controller.update(body, dt)
                if cmd is not None:
                    body.control_command = cmd

    def _actuate(self, dt: float):
        for body in self.bodies:
            if self.gravity is not None:
                self.gravity.apply_to(body)
            for actuator in getattr(body, 'actuators', []):
                actuator.apply_to(body, dt)

    def _integrate(self, dt: float):
        for body in self.bodies:
            body.update(dt)

# --- High-level Simulator ---
class MultiForce:
    """Aggregate multiple force models into a single interface."""
    def __init__(self, forces):
        self.forces = forces

    def apply_to(self, body):
        for f in self.forces:
            f.apply_to(body)

class Simulator:
    """High-level API wrapping World for easy simulation and logging."""
    def __init__(self, world: World):
        if not isinstance(world, World):
            raise TypeError("Simulator must be initialized with a World instance.")
        self.world = world

        if not self.world.bodies:
            raise ValueError("The provided World must contain at least one body for the Simulator.")
        
        self.body = self.world.bodies[0] 

        self.logs = []
        self.step_idx = 0
        self.divisors = self.world.divisors

    def run(self, duration):
        steps = int(duration / self.world.dt)
        for _ in range(steps):
            self.step()

    def simulate(self, steps):
        for _ in range(steps):
            self.step()

    def _record_state(self):
        self.logs.append({
            'time': self.world.time,
            'position': self.body.position.v.copy(),
            'velocity': self.body.velocity.v.copy(),
            'orientation': self.body.orientation.q.copy(),
            'angular_velocity': self.body.angular_velocity.v.copy(),
        })

    def get_logs(self):
        return self.logs

    def step(self):
        dt = self.world.dt
        idx = self.step_idx
        divs = self.divisors
        if idx % divs.get('sense', 1) == 0:
            self.world._sense(dt)
        if idx % divs.get('control', 1) == 0:
            self.world._control(dt)
        if idx % divs.get('actuate', 1) == 0:
            self.world._actuate(dt)
        if idx % divs.get('integrate', 1) == 0:
            self.world._integrate(dt)
        self.step_idx += 1
        self._record_state()

    @property
    def state_spec(self):
        """Specification of the state vector: shapes and dtypes for each component."""
        return {
            'time': {'shape': (), 'dtype': float},
            'position': {'shape': (3,), 'dtype': float},
            'velocity': {'shape': (3,), 'dtype': float},
            'orientation': {'shape': (4,), 'dtype': float},
            'angular_velocity': {'shape': (3,), 'dtype': float},
        }

    def get_state(self):
        """Return the current state as a dict and as a flat numpy array."""
        s = {
            'time': self.world.time,
            'position': self.body.position.v.copy(),
            'velocity': self.body.velocity.v.copy(),
            'orientation': self.body.orientation.q.copy(),
            'angular_velocity': self.body.angular_velocity.v.copy(),
        }
        flat = np.concatenate([
            [s['time']],
            s['position'],
            s['velocity'],
            s['orientation'],
            s['angular_velocity'],
        ]).astype(np.float32)
        return s, flat 