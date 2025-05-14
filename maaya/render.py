import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class Renderer:
    def __init__(self, world):
        self.world = world
        # Enable interactive mode so plotting does not block
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        self.ax.set_zlim([0, 20])
        
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')

        self.quadcopter_lines = []

        for body in self.world.bodies:
            # Define lines for a quadcopter X model with front half red and back half black
            lines = [[(-1, -1, 0), (0, 0, 0)], [(0, 0, 0), (1, 1, 0)],
                     [(1, -1, 0), (0, 0, 0)], [(0, 0, 0), (-1, 1, 0)]]
            colors = ['r', 'k', 'k', 'r']  # Alternating colors for the arms
            line_collection = Line3DCollection(lines, colors=colors, linewidths=2)
            self.quadcopter_lines.append(self.ax.add_collection3d(line_collection))

    def update_func(self, frame):
        # update physics with consistent timestep from world
        self.world.update()
        for i, body in enumerate(self.world.bodies):
            position = body.position.v
            orientation = body.orientation.as_rotation_matrix()
            
            # Define the initial lines of the quadcopter in the local frame
            lines = np.array([[[-1, -1, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 0]],
                              [[1, -1, 0], [0, 0, 0]], [[0, 0, 0], [-1, 1, 0]]])
            
            # Rotate lines according to the orientation matrix
            rotated_lines = []
            for line in lines:
                rotated_line = np.dot(line, orientation.T)
                rotated_lines.append(rotated_line)
            
            rotated_lines = np.array(rotated_lines)
            
            # Translate lines to the position of the quadcopter
            rotated_lines += position
            
            # Update the segments of the Line3DCollection
            self.quadcopter_lines[i].set_segments(rotated_lines)
        
        return self.quadcopter_lines
    
    def run(self, frames):
        # Non-blocking rendering: advance the simulation and update plot
        for _ in range(frames):
            self.update_func(None)
        # Draw and process GUI events
        plt.draw()
        plt.pause(0.001)

    def draw(self):
        """Draw the current simulation state without advancing physics."""
        import numpy as _np
        for i, body in enumerate(self.world.bodies):
            position = body.position.v
            orientation = body.orientation.as_rotation_matrix()
            # Define the initial lines of the quadcopter in the local frame
            lines = _np.array([[[-1, -1, 0], [0, 0, 0]],
                               [[0, 0, 0], [1, 1, 0]],
                               [[1, -1, 0], [0, 0, 0]],
                               [[0, 0, 0], [-1, 1, 0]]])
            rotated_lines = []
            for line in lines:
                # Rotate then translate
                rl = _np.dot(line, orientation.T) + position
                rotated_lines.append(rl)
            # Update the segments of the Line3DCollection
            self.quadcopter_lines[i].set_segments(rotated_lines)
        # Redraw
        plt.draw()
        plt.pause(0.001)
