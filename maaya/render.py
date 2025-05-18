import numpy as np

# Only PyBulletRenderer is supported now; MatplotlibRenderer removed

# Add PyBullet based renderer and make it default
try:
    import pybullet as p
except ImportError:
    PyBulletRenderer = None
else:
    import pybullet_data

    class PyBulletRenderer:
        def __init__(self, world, config='X', gui=True):
            import os
            import urllib.request
            import pybullet_data
            import numpy as np

            self.world = world
            self.p = p
            self.client = p.connect(p.GUI if gui else p.DIRECT)
            # Disable PyBullet's built-in keyboard shortcuts (wireframe, shadows, etc.)
            if hasattr(self.p, 'COV_ENABLE_KEYBOARD_SHORTCUTS'):
                self.p.configureDebugVisualizer(self.p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
            # Precompute yaw offset: apply 45° rotation for X-config
            if config.upper() == 'X':
                self._x_offset_quat = self.p.getQuaternionFromEuler([0, 0, np.pi/4])
            else:
                self._x_offset_quat = [0, 0, 0, 1]

            # Camera smoothing parameters
            self.smoothed_camera_yaw_deg = None
            self.camera_smoothing_factor = 0.15
            self.smoothed_camera_target_pos = None

            # Arm visualization parameters
            self.arm_vis_length = 0.25  # Slightly increased for visibility
            self.arm_colors = [[1, 0, 0, 1], [1, 0, 0, 1], [0.1, 0.1, 0.1, 1], [0.1, 0.1, 0.1, 1]]  # RGBA: FR, FL, BR, BL
            L = self.arm_vis_length
            # Select local endpoints based on configuration
            if config.upper() == 'X':
                self.arm_local_endpoints = [
                    np.array([L, -L, 0]),  # front-right
                    np.array([L, L, 0]),   # front-left
                    np.array([-L, -L, 0]), # back-right
                    np.array([-L, L, 0])   # back-left
                ]
            else:
                self.arm_local_endpoints = [
                    np.array([0, L, 0]),   # front
                    np.array([L, 0, 0]),    # right
                    np.array([0, -L, 0]),   # back
                    np.array([-L, 0, 0])    # left
                ]
            self.arm_debug_line_ids = [None] * len(self.arm_local_endpoints)  # Initialize IDs for reusable lines

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
                # Create pastel rectangular takeoff, loading, and delivery pads
                pad_half_extents = [3.0, 2.0, 0.01]  # 6m x 4m rectangular pads
                pads = {
                    'takeoff': ([0.0, 0.0, pad_half_extents[2]], [0.6, 0.9, 0.6, 1]),   # pastel green
                    'loading': ([8.0, 0.0, pad_half_extents[2]], [1.0, 1.0, 0.7, 1]),    # pastel yellow
                    'delivery': ([-8.0, 0.0, pad_half_extents[2]], [0.7, 0.8, 1.0, 1])   # pastel blue
                }
                for name, (center, color) in pads.items():
                    vis_shape = self.p.createVisualShape(
                        self.p.GEOM_BOX,
                        halfExtents=pad_half_extents,
                        rgbaColor=color
                    )
                    self.p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=vis_shape,
                        basePosition=center
                    )
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
            # remember quad URDF for identifying bodies
            self._quad_urdf_filename = quad_urdf_filename
            # corrected path for downloading quad URDF
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

            # Add the custom URDF assets directory to PyBullet's search path (for quadrotor)
            p.setAdditionalSearchPath(custom_urdf_assets_dir)
            # Also ensure pybullet_data directory is in search paths for built-in URDFs (e.g., cube.urdf)
            p.setAdditionalSearchPath(pybullet_data_dir)

            # Load quadrotor and box URDFs, track IDs
            self.robot_ids = []
            for body_idx, body_sim in enumerate(self.world.bodies):
                # Base position (lift if at origin)
                initial_pos = body_sim.position.v.tolist()
                if initial_pos == [0.0, 0.0, 0.0]:
                    initial_pos[2] = 0.1
                # Convert orientation to XYZW for PyBullet
                q_wxyz = body_sim.orientation.q.tolist()
                orn_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]

                urdf_to_load_on_pybullet_path = None
                pybullet_load_options = {'useFixedBase': True}
                rotated_orn = orn_xyzw # Default orientation

                if hasattr(body_sim, 'urdf_filename'):
                    assigned_urdf = body_sim.urdf_filename

                    if assigned_urdf == self._quad_urdf_filename: # Quadrotor
                        _, rotated_orn = self.p.multiplyTransforms(
                            [0, 0, 0], orn_xyzw,
                            [0, 0, 0], self._x_offset_quat
                        )
                        # Load from the downloaded absolute path
                        urdf_to_load_on_pybullet_path = quad_urdf_local_path
                    elif assigned_urdf == 'cube.urdf': # Cube
                        urdf_to_load_on_pybullet_path = 'cube.urdf'
                        pybullet_load_options['globalScaling'] = 0.5 # Ensure 0.5x0.5x0.5 size
                    else:
                        print(f"WARNING: Unknown URDF specified: {assigned_urdf} for body {body_idx}")
                        self.robot_ids.append(None)
                        continue
                else:
                    print(f"INFO: Body {body_idx} (type: {type(body_sim).__name__}) has no 'urdf_filename', not loading URDF.")
                    self.robot_ids.append(None)
                    continue
                
                if urdf_to_load_on_pybullet_path:
                    try:
                        robot_id = self.p.loadURDF(
                            urdf_to_load_on_pybullet_path,
                            basePosition=initial_pos,
                            baseOrientation=rotated_orn,
                            **pybullet_load_options
                        )
                        self.robot_ids.append(robot_id)
                        print(f"DEBUG: Loaded {urdf_to_load_on_pybullet_path} for body {body_idx} with ID {robot_id} at {initial_pos}.")

                        # If this is a cube, color it brown like cardboard
                        if assigned_urdf == 'cube.urdf':
                            self.p.changeVisualShape(robot_id, -1, rgbaColor=[0.65, 0.50, 0.39, 1])

                        # If this is the first robot (assumed to be the quad), set camera to look at it
                        if assigned_urdf == self._quad_urdf_filename and body_idx == 0: # More specific check for quad
                            p.resetDebugVisualizerCamera(
                                cameraDistance=1.5, 
                                cameraYaw=30,       
                                cameraPitch=-30,    
                                cameraTargetPosition=initial_pos
                            )
                            print(f"DEBUG: Camera reset to view robot {robot_id} at {initial_pos}")

                    except p.error as e:
                        print(f"WARNING: Could not load URDF '{urdf_to_load_on_pybullet_path}' for body {body_idx}: {e}")
                        self.robot_ids.append(None)
                        continue
                else: # Should not be reached if logic above is correct
                    self.robot_ids.append(None)

        def draw(self):
            target_body_for_camera = None
            # Identify the primary quadrotor for camera tracking (assuming it's the first body with quadrotor URDF)
            # This also assumes that self.world.bodies[0] is the quad if it exists.
            # A more robust way might be to explicitly tag the main quadrotor.
            if self.world.bodies and hasattr(self.world.bodies[0], 'urdf_filename') and self.world.bodies[0].urdf_filename == self._quad_urdf_filename:
                 target_body_for_camera = self.world.bodies[0]


            for idx, body in enumerate(self.world.bodies):
                pos_list = body.position.v.tolist()
                pos_np = body.position.v 
                orn_q_wxyz = body.orientation.q 
                orn_q_xyzw_list = [orn_q_wxyz[1], orn_q_wxyz[2], orn_q_wxyz[3], orn_q_wxyz[0]]
                
                is_quad = hasattr(body, 'urdf_filename') and body.urdf_filename == self._quad_urdf_filename

                if is_quad:
                    _, rotated_draw_orn = self.p.multiplyTransforms(
                        [0, 0, 0], orn_q_xyzw_list,
                        [0, 0, 0], self._x_offset_quat
                    )
                else:
                    rotated_draw_orn = orn_q_xyzw_list
                
                robot_id = self.robot_ids[idx]
                if robot_id is not None:
                    self.p.resetBasePositionAndOrientation(
                        robot_id, pos_list, rotated_draw_orn
                    )

                # Draw arms only for the quadcopter
                if is_quad and body == target_body_for_camera: 
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

        def run(self, frames):
            # Advance physics and render
            for _ in range(frames):
                self.world.update()
                self.draw()

# Default renderer alias
Renderer = PyBulletRenderer
