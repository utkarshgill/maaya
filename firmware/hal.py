"""
Hardware Abstraction Layer (HAL)
Provides a unified interface for sensors, actuators, and controllers.
"""

class HAL:
    """
    Hardware Abstraction Layer (HAL) base class.
    Provides abstract read/write interfaces to communicate with a controller.
    """
    def __init__(self, config=None):
        """Initialize HAL with optional configuration."""
        self.config = config

    def read(self, data):
        """Read raw input data (e.g., observations) and parse into structured form.
        Must be implemented by subclasses."""
        raise NotImplementedError("HAL.read() must be implemented by subclass")

    def write(self, commands):
        """Write actuator commands and return output action.
        Must be implemented by subclasses."""
        raise NotImplementedError("HAL.write() must be implemented by subclass") 