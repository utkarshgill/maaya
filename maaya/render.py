import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class MatplotlibRenderer:
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

# Add PyBullet based renderer and make it default
try:
    import pybullet as p
except ImportError:
    PyBulletRenderer = None
else:
    import pybullet_data

    class PyBulletRenderer:
        def __init__(self, world, gui=True):
            import os
            import urllib.request
            import pybullet_data
            import numpy as np

            self.world = world
            self.p = p
            self.client = p.connect(p.GUI if gui else p.DIRECT)

            # Camera smoothing parameters
            self.smoothed_camera_yaw_deg = None
            self.camera_smoothing_factor = 0.15
            self.smoothed_camera_target_pos = None

            # Arm visualization parameters
            self.arm_vis_length = 0.25 # Slightly increased for visibility
            self.arm_colors = [[1, 0, 0, 1], [1, 0, 0, 1], [0.1, 0.1, 0.1, 1], [0.1, 0.1, 0.1, 1]] # RGBA: FR, FL, BR, BL (using dark grey for black)
            L = self.arm_vis_length
            self.arm_local_endpoints = [
                np.array([L, -L, 0]), # Front-Right 
                np.array([L, L, 0]),   # Front-Left
                np.array([-L, -L, 0]),# Back-Right
                np.array([-L, L, 0])  # Back-Left
            ]
            self.arm_debug_line_ids = [None] * len(self.arm_local_endpoints) # Initialize IDs for reusable lines

            # Set up and load standard PyBullet assets (like plane.urdf) FIRST
            pybullet_data_dir = pybullet_data.getDataPath()
            print(f"DEBUG: Using pybullet_data path: {pybullet_data_dir}")
            p.setAdditionalSearchPath(pybullet_data_dir)

            p.resetSimulation() # Reset simulation state
            
            # Set gravity
            g_world = 0
            if hasattr(self.world, 'gravity') and self.world.gravity:
                try:
                    g_world = self.world.gravity.g # Assumes gravity object has 'g' attribute
                except AttributeError:
                    print("DEBUG: world.gravity found, but no 'g' attribute. Using g=0.")
                    pass 
            p.setGravity(0, 0, -g_world)

            # Load ground plane
            try:
                self.plane_id = p.loadURDF("plane.urdf")
                print("DEBUG: plane.urdf loaded successfully.")
            except p.error as e:
                print(f"ERROR: Failed to load plane.urdf. PyBullet error: {e}")
                print(f"Checked pybullet_data path: {pybullet_data_dir}")
                print("Please ensure 'pybullet_data' is correctly installed and contains 'plane.urdf'.")
                raise # Re-raise the error to stop execution if plane can't be loaded

            # Prepare local URDF directory for quadrotor model
            module_dir = os.path.dirname(__file__) # Should be maaya/render/
            custom_urdf_assets_dir = os.path.join(module_dir, 'data', 'Quadrotor')
            os.makedirs(custom_urdf_assets_dir, exist_ok=True)
            
            quad_urdf_filename = "quadrotor.urdf"
            quad_urdf_local_path = os.path.join(custom_urdf_assets_dir, quad_urdf_filename)
            quad_mesh_filename = "quadrotor_base.obj" # Mesh referenced in quadrotor.urdf
            quad_mesh_local_path = os.path.join(custom_urdf_assets_dir, quad_mesh_filename)

            # Download URDF if not present
            if not os.path.exists(quad_urdf_local_path):
                print(f"INFO: {quad_urdf_filename} not found locally. Downloading to {custom_urdf_assets_dir}...")
                quad_urdf_url = f'https://raw.githubusercontent.com/bulletphysics/bullet3/master/data/Quadrotor/{quad_urdf_filename}'
                try:
                    urllib.request.urlretrieve(quad_urdf_url, quad_urdf_local_path)
                    print(f"INFO: {quad_urdf_filename} downloaded successfully.")
                except Exception as e_download_urdf:
                    print(f"ERROR: Failed to download {quad_urdf_filename}: {e_download_urdf}")
                    # Decide how to proceed if URDF download fails
            
            # Download mesh if not present (assuming URDF download was successful or file existed)
            if not os.path.exists(quad_mesh_local_path):
                print(f"INFO: {quad_mesh_filename} not found locally. Downloading to {custom_urdf_assets_dir}...")
                quad_mesh_url = f'https://raw.githubusercontent.com/bulletphysics/bullet3/master/data/Quadrotor/{quad_mesh_filename}'
                try:
                    urllib.request.urlretrieve(quad_mesh_url, quad_mesh_local_path)
                    print(f"INFO: {quad_mesh_filename} downloaded successfully.")
                except Exception as e_download_mesh:
                    print(f"ERROR: Failed to download {quad_mesh_filename}: {e_download_mesh}")
                    # Mesh is critical for visualization

            # Add the custom URDF assets directory to PyBullet's search path
            # This allows loadURDF("quadrotor.urdf") to find the URDF,
            # and for the URDF to find its relative mesh files (quadrotor_base.obj)
            p.setAdditionalSearchPath(custom_urdf_assets_dir)

            # Load quadrotor model and keep track of IDs
            self.robot_ids = []
            for body_idx, body_sim in enumerate(self.world.bodies):
                initial_pos = body_sim.position.v.tolist()
                # Ensure z is slightly above ground if starting at 0,0,0 for visibility
                if initial_pos == [0.0, 0.0, 0.0]:
                    print("DEBUG: Initial position is 0,0,0. Adjusting to 0,0,0.1 for visibility.")
                    initial_pos = [0.0, 0.0, 0.1]

                initial_orn_quat_wxyz = body_sim.orientation.q.tolist() 
                initial_orn_quat_xyzw = [initial_orn_quat_wxyz[1], initial_orn_quat_wxyz[2], initial_orn_quat_wxyz[3], initial_orn_quat_wxyz[0]]
                
                try:
                    # For visualization synced from an external engine, useFixedBase=True is appropriate
                    # as PyBullet is not meant to simulate this body, only display it.
                    is_fixed_base = True 
                    robot_id = p.loadURDF(
                        quad_urdf_filename, 
                        basePosition=initial_pos,
                        baseOrientation=initial_orn_quat_xyzw,
                        useFixedBase=is_fixed_base 
                    )
                    self.robot_ids.append(robot_id)
                    print(f"DEBUG: Loaded {quad_urdf_filename} for body {body_idx} with ID {robot_id} at {initial_pos}.")

                    # If this is the first robot, set camera to look at it
                    if body_idx == 0:
                        p.resetDebugVisualizerCamera(
                            cameraDistance=1.5, # Closer view
                            cameraYaw=30,       # Adjusted yaw
                            cameraPitch=-30,    # Adjusted pitch
                            cameraTargetPosition=initial_pos
                        )
                        print(f"DEBUG: Camera reset to view robot {robot_id} at {initial_pos}")

                except p.error as e_load_quad:
                    print(f"ERROR: Failed to load {quad_urdf_filename} for body {body_idx}. PyBullet error: {e_load_quad}")
                    print(f"Attempted to load from search paths including: {custom_urdf_assets_dir}")
                    print(f"Ensure {quad_urdf_filename} and its mesh {quad_mesh_filename} are in this directory.")
                    # Continue without this body or raise error

        def draw(self):
            target_body_for_camera = None
            if self.world.bodies and self.robot_ids:
                target_body_for_camera = self.world.bodies[0]

            for idx, body in enumerate(self.world.bodies):
                pos_list = body.position.v.tolist()
                pos_np = body.position.v 
                orn_q_wxyz = body.orientation.q 
                orn_q_xyzw_list = [orn_q_wxyz[1], orn_q_wxyz[2], orn_q_wxyz[3], orn_q_wxyz[0]]
                
                self.p.resetBasePositionAndOrientation(
                    self.robot_ids[idx], pos_list, orn_q_xyzw_list
                )

                if body == target_body_for_camera: 
                    orientation_matrix = body.orientation.as_rotation_matrix()
                    for i, endpoint_local_np in enumerate(self.arm_local_endpoints):
                        p0_world_np = pos_np
                        p1_world_np = pos_np + orientation_matrix.dot(endpoint_local_np)
                        color_rgb = self.arm_colors[i][:3]
                        line_width = 3.0

                        if self.arm_debug_line_ids[i] is None:
                            self.arm_debug_line_ids[i] = self.p.addUserDebugLine(
                                p0_world_np.tolist(), 
                                p1_world_np.tolist(), 
                                lineColorRGB=color_rgb,
                                lineWidth=line_width,
                                lifeTime=0 # Persistent, will be updated
                            )
                        else:
                            self.p.addUserDebugLine(
                                p0_world_np.tolist(), 
                                p1_world_np.tolist(), 
                                lineColorRGB=color_rgb,
                                lineWidth=line_width,
                                replaceItemUniqueId=self.arm_debug_line_ids[i]
                            )

            if target_body_for_camera:
                target_pos_list = target_body_for_camera.position.v.tolist()
                _roll_rad, _pitch_rad, raw_yaw_rad = target_body_for_camera.orientation.to_euler()
                raw_yaw_deg = np.degrees(raw_yaw_rad)

                # Smooth the yaw for the camera
                if self.smoothed_camera_yaw_deg is None:
                    self.smoothed_camera_yaw_deg = raw_yaw_deg
                else:
                    # Normalize difference to handle wrap-around from -180 to 180
                    diff = raw_yaw_deg - self.smoothed_camera_yaw_deg
                    diff = (diff + 180) % 360 - 180 
                    self.smoothed_camera_yaw_deg += self.camera_smoothing_factor * diff
                    # Keep smoothed yaw in [-180, 180]
                    self.smoothed_camera_yaw_deg = (self.smoothed_camera_yaw_deg + 180) % 360 - 180

                TPP_CAMERA_DISTANCE = 3.0  # Adjusted distance
                TPP_CAMERA_PITCH = -20.0 # Adjusted pitch
                
                # Offset camera yaw so that +X points forward in view
                CAMERA_YAW_OFFSET = -90.0
                effective_camera_yaw = ((self.smoothed_camera_yaw_deg + CAMERA_YAW_OFFSET + 180) % 360) - 180

                # Smooth the camera target position (low-pass filter)
                if self.smoothed_camera_target_pos is None:
                    self.smoothed_camera_target_pos = np.array(target_pos_list)
                else:
                    diff_pos = np.array(target_pos_list) - self.smoothed_camera_target_pos
                    self.smoothed_camera_target_pos += self.camera_smoothing_factor * diff_pos
                smoothed_target_pos_list = self.smoothed_camera_target_pos.tolist()

                self.p.resetDebugVisualizerCamera(
                    cameraDistance=TPP_CAMERA_DISTANCE,
                    cameraYaw=effective_camera_yaw,
                    cameraPitch=TPP_CAMERA_PITCH,
                    cameraTargetPosition=smoothed_target_pos_list
                )
            
            self.p.stepSimulation()

        def run(self, frames):
            # Advance physics and render
            for _ in range(frames):
                self.world.update()
                self.draw()

# Default renderer alias
Renderer = PyBulletRenderer
