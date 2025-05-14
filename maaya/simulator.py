"""maaya/simulator.py

High-level Simulator API that orchestrates World, Body, sensors, controllers, actuators, and forces."""

from .world import World
from .bus import Bus


class MultiForce:
    """Aggregate multiple force models into a single gravity-like interface."""
    def __init__(self, forces):
        self.forces = forces

    def apply_to(self, body):
        for f in self.forces:
            f.apply_to(body)


class Simulator:
    """High-level API wrapping World for easy simulation and logging."""

    def __init__(
        self,
        body,
        sensors=None,
        controllers=None,
        actuators=None,
        forces=None,
        dt=0.01,
        rates=None,
    ):
        """
        Args:
            body: a Body instance to simulate.
            sensors: list of Sensor instances to attach to the body.
            controllers: list of Controller instances to attach to the body.
            actuators: list of Actuator instances to attach to the body.
            forces: list of external force models (e.g. GravitationalForce).
            dt: base-rate timestep in seconds.
            rates: dict of sub-rate divisors for 'sense','control','actuate','integrate'.
        """
        divisors = rates or {}
        # Combine forces into a single gravity interface
        gravity = None
        if forces:
            gravity = forces[0] if len(forces) == 1 else MultiForce(forces)

        # Initialize World
        self.world = World(gravity=gravity, dt=dt, **divisors)
        self.body = body
        self.world.add_body(body)

        # Attach sensors, controllers, actuators
        for sensor in sensors or []:
            body.add_sensor(sensor)
        for ctrl in controllers or []:
            body.add_controller(ctrl)
        for act in actuators or []:
            body.add_actuator(act)

        # Prepare log storage
        self.logs = []
        # Initialize message bus
        self.bus = Bus()
        # Scheduler step index and divisors for pub/sub
        self.step_idx = 0
        self.divisors = self.world.divisors

    def run(self, duration):
        """Run the simulation for the given duration (seconds)."""
        steps = int(duration / self.world.dt)
        for _ in range(steps):
            self.step()

    def simulate(self, steps):
        """Run the simulation for a fixed number of base-rate steps."""
        for _ in range(steps):
            self.step()

    def _record_state(self):
        """Internal: snapshot current body state to logs."""
        self.logs.append({
            'time': self.world.time,
            'position': self.body.position.v.copy(),
            'velocity': self.body.velocity.v.copy(),
            'orientation': self.body.orientation.q.copy(),
            'angular_velocity': self.body.angular_velocity.v.copy(),
        })

    def get_logs(self):
        """Return the recorded state history as a list of dictionaries."""
        return self.logs

    def step(self):
        """Perform one simulation base-rate step with message bus events."""
        dt = self.world.dt
        idx = self.step_idx
        divs = self.divisors
        # Sense stage
        if idx % divs.get('sense', 1) == 0:
            self.world._sense(dt)
            sensor_data = getattr(self.body, 'sensor_data', None)
            self.bus.publish('sensor', {
                'body': self.body,
                'sensor_data': sensor_data,
                'time': self.world.time,
            })
        # Control stage
        if idx % divs.get('control', 1) == 0:
            self.world._control(dt)
            cmd = getattr(self.body, 'control_command', None)
            self.bus.publish('control', {
                'body': self.body,
                'control_command': cmd,
                'time': self.world.time,
            })
        # Actuate stage
        if idx % divs.get('actuate', 1) == 0:
            self.world._actuate(dt)
            self.bus.publish('actuate', {
                'body': self.body,
                'time': self.world.time,
            })
        # Integrate stage
        if idx % divs.get('integrate', 1) == 0:
            self.world._integrate(dt)
            self.bus.publish('integrate', {
                'body': self.body,
                'time': self.world.time,
            })
        # Advance step counter and record state
        self.step_idx += 1
        self._record_state() 