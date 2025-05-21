"""
Default target configuration.
"""

from . import register_target

@register_target("default")
def default_target():
    """
    Default board target settings.
    """
    return {
        "pin_led": 13,
        "i2c_bus": 1,
    } 