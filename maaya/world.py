from .math import Vector3D

class World:
    def __init__(self, gravity=None, dt=0.01):
        """Simulation world holding objects, physics components, and timestep."""
        self.objects = []
        self.time = 0.0
        self.dt = dt
        self.gravity = gravity
        self.sensors = []
        self.controllers = []
        self.actuators = []

    def add_object(self, obj):
        self.objects.append(obj)

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def add_controller(self, controller):
        self.controllers.append(controller)

    def add_actuator(self, actuator):
        self.actuators.append(actuator)

    def update(self, dt=None):
        """
        Perform one simulation cycle: sense → control → actuate → integrate.
        dt: timestep for this update (defaults to world.dt).
        """
        dt = self.dt if dt is None else dt
        # 1. Sense
        for sensor in self.sensors:
            sensor.read(self.objects, dt)
        # 2. Control
        for controller in self.controllers:
            controller.update(self.objects, dt)
        # 3. Actuate (environment and actuators)
        for obj in self.objects:
            if self.gravity is not None:
                self.gravity.apply_to(obj)
            for actuator in self.actuators:
                actuator.apply_to(obj, dt)
        # 4. Integrate
        for obj in self.objects:
            obj.update(dt)
        self.time += dt

    def simulate(self, frames):
        """
        Run the simulation for a number of steps using the world timestep.
        """
        for _ in range(frames):
            self.update()

