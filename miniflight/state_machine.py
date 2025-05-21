"""
State machine module: handles arming/disarming/failsafe states.
"""
class StateMachine:
    def __init__(self):
        self.state = "DISARMED"

    def arm(self):
        self.state = "ARMED"

    def disarm(self):
        self.state = "DISARMED"

    def failsafe(self):
        self.state = "FAILSAFE" 