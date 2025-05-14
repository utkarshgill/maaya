"""Simulation world orchestrating sense → control → actuate → integrate."""
class World:
    def __init__(self, gravity=None, dt=0.01):
        """Simulation world holding objects, physics components, and timestep."""
        self.bodies = []
        self.time = 0.0
        self.dt = dt
        self.gravity = gravity

    def add_body(self, body):
        """Add a Body instance to be simulated."""
        self.bodies.append(body)

    def update(self, dt=None):
        """
        Perform one simulation cycle: sense → control → actuate → integrate.
        Sensors and controllers are run per-body.
        dt: timestep for this update (defaults to world.dt).
        """
        dt = self.dt if dt is None else dt
        # Sense, Control, Actuate per body
        for body in self.bodies:
            # 1. Sense
            for sensor in getattr(body, 'sensors', []):
                sensor.read([body], dt)
            # 2. Control
            for controller in getattr(body, 'controllers', []):
                controller.update([body], dt)
            # 3. Environment & Actuation
            if self.gravity is not None:
                self.gravity.apply_to(body)
            for actuator in getattr(body, 'actuators', []):
                actuator.apply_to(body, dt)
        # 4. Integrate
        for body in self.bodies:
            body.update(dt)
        self.time += dt

    def simulate(self, frames):
        """
        Run the simulation for a number of steps using the world timestep.
        """
        for _ in range(frames):
            self.update()

