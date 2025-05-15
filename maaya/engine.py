# engine.py
# Core simulation engine: messaging, scheduling, world, and high-level simulator

from collections import defaultdict
from typing import Callable, List, Tuple
import numpy as np
from .math import Vector3D

# --- Messaging bus ---
class Bus:
    """A minimal topic-based message bus."""
    def __init__(self):
        self._subscribers = defaultdict(list)

    def subscribe(self, topic: str, fn):
        self._subscribers[topic].append(fn)

    def publish(self, topic: str, data):
        for fn in self._subscribers.get(topic, []):
            try:
                fn(data)
            except Exception:
                pass

# --- Scheduler ---
class Scheduler:
    """Ultra-lightweight deterministic task scheduler."""
    def __init__(self):
        self._tasks: List[Tuple[int, Callable[[], None]]] = []
        self._step_idx: int = 0

    def add(self, fn: Callable[[], None], *, every: int = 1) -> None:
        if every < 1:
            raise ValueError("`every` must be >= 1")
        self._tasks.append((every, fn))

    def step(self) -> None:
        for every, fn in self._tasks:
            if self._step_idx % every == 0:
                fn()
        self._step_idx += 1

    def current_step(self) -> int:
        return self._step_idx

    def reset(self) -> None:
        self._step_idx = 0

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
        self.bus = Bus()

        self.divisors = {**World.DEFAULT_DIVISORS, **divisors}
        self.scheduler = Scheduler()
        self._register_tasks()

        self.bus.subscribe("sense", lambda data: self._sense(data["dt"]))
        self.bus.subscribe("control", lambda data: self._control(data["dt"]))
        self.bus.subscribe("actuate", lambda data: self._actuate(data["dt"]))
        self.bus.subscribe("integrate", lambda data: self._integrate(data["dt"]))

    def add_body(self, body):
        self.bodies.append(body)

    def update(self, dt: float | None = None):
        if dt is not None:
            self.dt = dt
        self.scheduler.step()

    step = update

    def simulate(self, frames: int):
        for _ in range(frames):
            self.update()

    def _register_tasks(self):
        self.scheduler.add(lambda: self.bus.publish("sense", {"dt": self.dt}), every=self.divisors['sense'])
        self.scheduler.add(lambda: self.bus.publish("control", {"dt": self.dt}), every=self.divisors['control'])
        self.scheduler.add(lambda: self.bus.publish("actuate", {"dt": self.dt}), every=self.divisors['actuate'])
        self.scheduler.add(lambda: self.bus.publish("integrate", {"dt": self.dt}), every=self.divisors['integrate'])

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
        self.time += dt

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
        self.bus = Bus()
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
            sensor_data = getattr(self.body, 'sensor_data', None)
            self.bus.publish('sensor', {
                'body': self.body,
                'sensor_data': sensor_data,
                'time': self.world.time,
            })
        if idx % divs.get('control', 1) == 0:
            self.world._control(dt)
            cmd = getattr(self.body, 'control_command', None)
            self.bus.publish('control', {
                'body': self.body,
                'control_command': cmd,
                'time': self.world.time,
            })
        if idx % divs.get('actuate', 1) == 0:
            self.world._actuate(dt)
            self.bus.publish('actuate', {
                'body': self.body,
                'time': self.world.time,
            })
        if idx % divs.get('integrate', 1) == 0:
            self.world._integrate(dt)
            self.bus.publish('integrate', {
                'body': self.body,
                'time': self.world.time,
            })
        self.step_idx += 1
        self._record_state()

    @property
    def state_spec(self):
        return {
            'time': {'shape': (), 'dtype': float},
            'position': {'shape': (3,), 'dtype': float},
            'velocity': {'shape': (3,), 'dtype': float},
            'orientation': {'shape': (4,), 'dtype': float},
            'angular_velocity': {'shape': (3,), 'dtype': float},
        }

    def get_state(self):
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
            s['angular_velocity']
        ]).astype(np.float32)
        return s, flat 