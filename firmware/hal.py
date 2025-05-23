"""
Hardware Abstraction Layer (HAL)
Provides a unified interface for sensors, actuators, and controllers.
"""
from common.interface import Sensor, Actuator, Controller

class HAL:
    """
    Core HAL class that orchestrates sensor reads, controller updates, and actuator writes.
    """
    def __init__(self, config=None, controller=None):
        """
        Initialize HAL; config and controller are optional (e.g. for Simulator subclassing).
        """
        self.config = config
        self.controller = controller
        self.sensors: list[Sensor] = []
        self.actuators: list[Actuator] = []

    def add_sensor(self, sensor: Sensor):
        self.sensors.append(sensor)

    def add_actuator(self, actuator: Actuator):
        self.actuators.append(actuator)

    def run_cycle(self, dt: float):
        """
        Perform one control cycle: read sensors, compute commands, write actuators.
        """
        # Read all sensors
        sensor_data = {}
        for sensor in self.sensors:
            data = sensor.read()
            sensor_data.update(data if isinstance(data, dict) else {sensor.__class__.__name__: data})

        # Compute control commands
        commands = self.controller.update(sensor_data, dt)

        # Write to actuators
        if commands is not None:
            for actuator, cmd in zip(self.actuators, commands):
                actuator.write(cmd) 