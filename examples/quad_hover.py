from ..engine import Vector3D, World, PhysicsObject

# Example usage:
world = World(g=10)
quad = PhysicsObject(position=Vector3D(0, 0, 10.0), mass=1.0)
world.add_object(quad) 
world.simulate(1000, render=True)