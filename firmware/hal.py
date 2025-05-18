"""
Hardware Abstraction Layer stubs.
"""

class HAL:
    def __init__(self, config):
        self.config = config
        # Initialize hardware interfaces here

    def gpio(self, pin, mode):
        """Stub for GPIO interface."""
        raise NotImplementedError("GPIO interface not implemented")

    def i2c_read(self, address, num_bytes):
        """Stub for I2C read."""
        raise NotImplementedError("I2C read not implemented")

    def i2c_write(self, address, data):
        """Stub for I2C write."""
        raise NotImplementedError("I2C write not implemented")

    def spi_transfer(self, data):
        """Stub for SPI transfer."""
        raise NotImplementedError("SPI transfer not implemented")

    def uart_send(self, data):
        """Stub for UART send."""
        raise NotImplementedError("UART send not implemented")

    def uart_receive(self, num_bytes):
        """Stub for UART receive."""
        raise NotImplementedError("UART receive not implemented") 