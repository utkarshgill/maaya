"""
Ground collision model: clamps position at ground_level and reflects velocity.
"""

class GroundCollision:
    """Simple ground collision with restitution coefficient."""

    def __init__(self, ground_level: float = 0.0, restitution: float = 0.5):
        """
        Args:
            ground_level: z-coordinate of the ground plane.
            restitution: coefficient of restitution (0–1) for bounce.
        """
        self.ground_level = ground_level
        self.restitution = restitution

    def apply_to(self, body):
        """Clamp body position and reflect its vertical velocity on collision."""
        # Assume position and velocity are Vector3D instances
        z = body.position.v[2]
        if z < self.ground_level:
            # Clamp position to ground
            body.position.v[2] = self.ground_level
            vz = body.velocity.v[2]
            # Reflect only if moving downward
            if vz < 0:
                body.velocity.v[2] = -vz * self.restitution 