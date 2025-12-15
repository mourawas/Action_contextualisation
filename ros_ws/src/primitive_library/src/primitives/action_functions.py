import typing as tp
import numpy as np
from scipy.spatial.transform import Rotation
import contextlib

import rospy

from primitives.js_lds_oa import JS_LDS_OA
from llm_common import utils as llmu
from llm_common import helpers as llmh
from llm_simulator.srv import objPos, objMesh


static_elements = []
js_lds = None

approach_side_dst = 0.05
approach_ee_offset_side = 0.14
approach_ee_offset_side_additional = 0.02
approach_ee_offset_top = 0.18
approach_top_dst = 0.10
approach_top_dst_offset = 0.15
table_altitude = 0.00

BEAM_LENGTH = 0.3  # meters
BEAM_HALF_LENGTH = BEAM_LENGTH / 2  # meters
BEAM_WIDTH = 0.07 # meters
TABLE_HEIGHT = 1.01  # meters



def robot_action(func):
    def wrapper_robot_action(*args, **kwargs):

        # Controller reset
        global js_lds
        if js_lds is None:
            reset_controller()

        # Function execution
        func(js_lds, *args, **kwargs)

    return wrapper_robot_action


def reset_controller():
    global js_lds
    with contextlib.redirect_stdout(None):
        js_lds = JS_LDS_OA(llmu.IIWA_URDF_FOLDER,
                        llmu.IIWA_URDF_PATH,
                        llmu.ALLEGRO_URDF_FOLDER,
                        llmu.ALLEGRO_URDF_PATH)
        js_lds.send_robot_zero_torque()


def get_shelf_mesh() -> np.ndarray:
    # Original shelf mesh
    # <body name="shelf" pos="-0.1 -0.55 0.45">
    x_min = -0.4
    x_max = 0.2
    y_min = -0.65
    y_max = -0.35
    z_min = 0.45  # Only model the top part of the shelf
    z_max = 0.9

    # # New shelf mesh
    # <body name="shelf" pos="-0.1 0.75 0.45">
    # <geom name="shelf" class="collision" type="box" size="0.3 0.2 .45" pos="0 0 0" mass="10" rgba="0.5 0.5 0.5 1"/>
    # x_min = -0.4
    # x_max = 0.2
    # y_min = 0.95
    # y_max = 0.55
    # z_min = 0.45  # Only model the top part of the shelf
    # z_max = 0.9

    total_nb_pts = 200

    # Generate top obstascle
    nb_pts_per_side = int(np.round(np.sqrt(total_nb_pts)))
    x_range = np.linspace(x_min, x_max, nb_pts_per_side)
    y_range = np.linspace(y_min, y_max, nb_pts_per_side)
    z_range = np.array([z_max])
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    xx = np.expand_dims(xx.flatten(), axis=1)
    yy = np.expand_dims(yy.flatten(), axis=1)
    zz = np.expand_dims(zz.flatten(), axis=1)
    top_mesh = np.concatenate((xx, yy, zz), axis=1)

    # Generate side pannel 1
    x_range = np.array([x_min])
    y_range = np.linspace(y_min, y_max, nb_pts_per_side)
    z_range = np.linspace(z_min, z_max, nb_pts_per_side)
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    xx = np.expand_dims(xx.flatten(), axis=1)
    yy = np.expand_dims(yy.flatten(), axis=1)
    zz = np.expand_dims(zz.flatten(), axis=1)
    side_mesh_1 = np.concatenate((xx, yy, zz), axis=1)

    # Generate side pannel 2
    x_range = np.array([x_max])
    y_range = np.linspace(y_min, y_max, nb_pts_per_side)
    z_range = np.linspace(z_min, z_max, nb_pts_per_side)
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    xx = np.expand_dims(xx.flatten(), axis=1)
    yy = np.expand_dims(yy.flatten(), axis=1)
    zz = np.expand_dims(zz.flatten(), axis=1)
    side_mesh_2 = np.concatenate((xx, yy, zz), axis=1)

    # Generate front pannel
    x_range = np.linspace(x_min, x_max, nb_pts_per_side)
    y_range = np.array([y_max])
    z_range = np.linspace(z_min, z_max, nb_pts_per_side)
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    xx = np.expand_dims(xx.flatten(), axis=1)
    yy = np.expand_dims(yy.flatten(), axis=1)
    zz = np.expand_dims(zz.flatten(), axis=1)
    front_mesh = np.concatenate((xx, yy, zz), axis=1)

    return np.concatenate((front_mesh, side_mesh_1, side_mesh_2, top_mesh), axis=0)


def get_table_mesh() -> np.ndarray:
    nb_table_points = 1000
    nb_legs_points = 50
    table_x_min = 0.35
    table_x_max = 1.05
    table_y_min = -0.5
    table_y_max = 0.7

    nb_pts_per_side = int(np.round(np.sqrt(nb_table_points)))
    x_range = np.linspace(table_x_min, table_x_max, nb_pts_per_side)
    y_range = np.linspace(table_y_min, table_y_max, nb_pts_per_side)
    z_range = np.array([0.99])

    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

    xx = np.expand_dims(xx.flatten(), axis=1)
    yy = np.expand_dims(yy.flatten(), axis=1)
    zz = np.expand_dims(zz.flatten(), axis=1)

    mesh = np.concatenate((xx, yy, zz), axis=1)

    return mesh


def get_meshes(obj_names: tp.List[str], detailed_meshes: bool = True, use_robot_frame: bool = True):
    rospy.wait_for_service('objMesh')
    rospy.wait_for_service('objPos')

    obj_frame_service = rospy.ServiceProxy('objPos', objPos, persistent=True)
    mesh_service = rospy.ServiceProxy('objMesh', objMesh, persistent=True)

    iiwa_pos = obj_frame_service('kuka_base').object_position
    iiwa_base_pos = llmh.mujoco_pos_quat_to_se3(iiwa_pos[:3], iiwa_pos[3:])

    meshes = None
    radius = None
    names = None
    for name in obj_names:

        if (name in llmu.ROUND_OBJECTS and not detailed_meshes) and False:
            new_mesh, new_radius = get_mesh_for_round_object(name)
        else:

            if name == "table":
                new_mesh = get_table_mesh()
                if detailed_meshes:
                    new_radius = np.ones([new_mesh.shape[0]]) * 0.006  # from trigo
                else:
                    new_radius = np.ones([new_mesh.shape[0]]) * 0.01
            elif name == "shelf":
                new_mesh = get_shelf_mesh()
                if detailed_meshes:
                    new_radius = np.ones([new_mesh.shape[0]]) * 0.007
                else:
                    new_radius = np.ones([new_mesh.shape[0]]) * 0.01
            else:
                mesh_details = mesh_service(name)
                new_mesh = np.reshape(mesh_details.object_vertices, (-1, 3))
                new_radius = np.asarray(mesh_details.object_radii)
                if detailed_meshes:
                    new_radius -= 0.001
                else:
                    new_radius += 0.005

        if use_robot_frame:
            new_mesh = (np.linalg.inv(iiwa_base_pos) @ np.concatenate((new_mesh, np.ones((new_mesh.shape[0], 1))), axis=1).T).T[:, :3]
        new_names = [name] * new_mesh.shape[0]
        if meshes is None:
            meshes = new_mesh
            radius = new_radius
            names = new_names
        else:
            meshes = np.concatenate([meshes, new_mesh], axis=0)
            radius = np.concatenate([radius, new_radius], axis=0)
            names += new_names

    return (meshes, radius, names)


def get_mesh_for_round_object(obj_name: str) -> tp.Tuple[np.ndarray, np.ndarray]:
    rospy.wait_for_service('objMesh')
    rospy.wait_for_service('objComPos')
    com_pos_service = rospy.ServiceProxy('objComPos', objPos, persistent=True)
    mesh_service = rospy.ServiceProxy('objMesh', objMesh, persistent=True)

    obj_com_pos = np.asarray(com_pos_service(obj_name).object_position)
    obj_geom = mesh_service(obj_name)

    obj_mesh = np.reshape(obj_geom.object_vertices, (-1, 3))
    obj_radii = np.asarray(obj_geom.object_radii)
    radius = np.max(np.linalg.norm(obj_mesh - obj_com_pos, axis=1) + obj_radii)

    return (np.array([obj_com_pos]), np.array([radius]))


def update_static_obstacles() -> None:
    global static_elements
    if len(static_elements) == 0:
        static_elements = np.asarray(get_meshes(llmu.STATIC_ELEMENTS))


@robot_action
def approach(js_lds, # object_to_grasp: str,
             object_to_grasp: tp.Union[str, list], # Modified to accept direct position
             speed: float,
             obstacle_clearance: tp.Optional[float],
             grasp: str,
             # orientation used to be here
             placement_angle: float = None,
             disregard_object_to_grasp: bool = False,
             detailed_obstacles: bool = False,
             apply_offsets: bool = True,
             obstacle_ik: bool = False,
             vertical_clearance_offset: float = 0.,
             force_altitude: tp.Optional[float] = None,
             mock_run: bool = False,
             disregard_table: bool = False,
             drop_side_offset: bool = False,
             disregard_object: tp.Optional[str] = None,
             vertical: bool = False,
             orientation: float = 0.) -> None:

    # Default grasp to top
    if grasp == '':
        grasp = 'top'
    
    # NEW: Check if object_to_grasp is a position list
    if isinstance(object_to_grasp, list):
        # Direct position provided - skip service calls
        rospy.wait_for_service('objPos')
        obj_frame_service = rospy.ServiceProxy('objPos', objPos, persistent=True)
        iiwa_pos = obj_frame_service('kuka_base').object_position
        iiwa_base_pos = llmh.mujoco_pos_quat_to_se3(iiwa_pos[:3], iiwa_pos[3:])
        
        obj_com_pos = np.ones((4, 1))
        obj_com_pos[:3, 0] = object_to_grasp  # Use provided position
        
        # For positions, use dummy mesh data
        obj_mesh = np.array([object_to_grasp])
        obj_radii = np.array([0.01])  # Small default radius
        
    else:
        # FROM EXISTING CODE: Get position from services
        rospy.wait_for_service('objComPos')
        rospy.wait_for_service('objPos')
        rospy.wait_for_service('objMesh')
        obj_frame_service = rospy.ServiceProxy('objPos', objPos, persistent=True)
        com_pos_service = rospy.ServiceProxy('objComPos', objPos, persistent=True)
        mesh_service = rospy.ServiceProxy('objMesh', objMesh, persistent=True)
        
        iiwa_pos = obj_frame_service('kuka_base').object_position
        iiwa_base_pos = llmh.mujoco_pos_quat_to_se3(iiwa_pos[:3], iiwa_pos[3:])
        obj_com_pos = np.ones((4, 1))
        obj_com_pos[:3] = np.expand_dims(com_pos_service(object_to_grasp).object_position, axis=1)
        
        # Get mesh for objects
        if object_to_grasp in ['shelf', 'table']:
            (obj_mesh, obj_radii, _) = get_meshes([object_to_grasp], detailed_meshes=True, use_robot_frame=False)
        else:
            obj_mesh = np.reshape(mesh_service(object_to_grasp).object_vertices, (-1, 3))
            obj_radii = np.asarray(mesh_service(object_to_grasp).object_radii)
    

    # # Waiting for services
    # rospy.wait_for_service('objComPos')
    # rospy.wait_for_service('objPos')
    # rospy.wait_for_service('objMesh')
    # obj_frame_service = rospy.ServiceProxy('objPos', objPos, persistent=True)
    # com_pos_service = rospy.ServiceProxy('objComPos', objPos, persistent=True)
    # mesh_service = rospy.ServiceProxy('objMesh', objMesh, persistent=True)

    # # Getting object positions
    # iiwa_pos = obj_frame_service('kuka_base').object_position
    # iiwa_base_pos = llmh.mujoco_pos_quat_to_se3(iiwa_pos[:3], iiwa_pos[3:])
    # obj_com_pos = np.ones((4, 1))
    # obj_com_pos[:3] = np.expand_dims(com_pos_service(object_to_grasp).object_position, axis=1)

    # # Here we are doing something cheeky because I am too tired to figure out why the proper way doesn't work. TODO: Do this right
    # if object_to_grasp in ['shelf', 'table']:
    #     (obj_mesh, obj_radii, _) = get_meshes([object_to_grasp], detailed_meshes=True, use_robot_frame=False)
    # else:
    #     obj_mesh = np.reshape(mesh_service(object_to_grasp).object_vertices, (-1, 3))
    #     obj_radii = np.asarray(mesh_service(object_to_grasp).object_radii)

    base_obj_vec = np.squeeze(obj_com_pos[:3]) - iiwa_base_pos[:3, 3]
    obj_goal = obj_com_pos

    # Find direction of approach
    approach_direction = np.copy(base_obj_vec)
    approach_direction[2] = 0
    approach_direction = approach_direction / np.linalg.norm(approach_direction)
    approach_direction_perp = np.array([-approach_direction[1], approach_direction[0], 0])

    # Handle offset for approach
    grasp_orientation = grasp
    if grasp == "side":
        # Compute object radius
        obj_radii_perp = (obj_mesh.T - obj_com_pos[:3]).T @ np.expand_dims(approach_direction_perp,axis=1)
        obj_radius_perp = -np.min(obj_radii_perp - obj_radii)
        obj_radii_parl = (obj_mesh.T - obj_com_pos[:3]).T @ np.expand_dims(approach_direction,axis=1)
        obj_radius_parl = -np.min(obj_radii_parl - obj_radii)

        # Adjust goal position to object
        # best : +0.1, +0.01
        if js_lds.grasping :
            # tangential offset
            obj_goal[:3] -= np.expand_dims(approach_direction_perp * (obj_radius_perp +0.1), axis=1)
            # radial offset
            obj_goal[:3] -= np.expand_dims(approach_direction * (obj_radius_parl + approach_ee_offset_side + 0.01), axis=1)
        else:
            # more to the left, closer 
            obj_goal[:3] -= np.expand_dims(approach_direction_perp * (obj_radius_perp +0.05), axis=1)

            obj_goal[:3] -= np.expand_dims(approach_direction * (obj_radius_parl + approach_ee_offset_side +0.01), axis=1)

        obj_goal[2] += vertical_clearance_offset

        if apply_offsets:
            obj_goal[:3] -= np.expand_dims(approach_direction_perp * (approach_side_dst+obstacle_clearance), axis=1)
            obj_goal[:3] -= np.expand_dims(approach_direction * approach_ee_offset_side_additional, axis=1)

        obj_yaw = np.degrees(np.arctan2(base_obj_vec[1], base_obj_vec[0]))
        if placement_angle is not None:
            target_yaw = np.clip(placement_angle, -75, 75)
        else:
            target_yaw = obj_yaw

        if vertical:
            obj_goal[2] += BEAM_HALF_LENGTH  # Offset by half beam length

        # Adjust altitude if we are over the table
        if obj_goal[2] < table_altitude:
            obj_goal[2] = table_altitude

        if orientation == 0:
            goal_rot = Rotation.from_euler('xyz', [90, 0, 90 + target_yaw], degrees=True).as_quat()

    elif grasp == "top":

        obj_goal[2] = np.max(obj_mesh[:, 2])
        obj_goal[2] += approach_top_dst
        obj_goal[2] += vertical_clearance_offset

        if vertical:
            obj_goal[2] += BEAM_HALF_LENGTH  # Offset by half beam length

        if apply_offsets:
            obj_goal[2] += obstacle_clearance
            obj_goal[2] += approach_top_dst_offset

        if drop_side_offset:
            obj_goal[:3] -= np.expand_dims(approach_direction * 0.09, axis=1)
        else:
            # + goes to the right of the robot
            obj_goal[:3] -= np.expand_dims(approach_direction_perp * (-0.01), axis=1) 
            obj_goal[:3] -= np.expand_dims(approach_direction * approach_ee_offset_top, axis=1)

        obj_yaw = np.degrees(np.arctan2(base_obj_vec[1], base_obj_vec[0]))
        if placement_angle is not None:
            target_yaw = np.clip(placement_angle, -75, 75)
        else:
            target_yaw = obj_yaw

        if orientation == 0:
            if vertical:
                # Vertical: rotate beam to stand upright along Z-axis
                # [90, 0, target_yaw] rotates the beam from horizontal (X-axis) to vertical (Z-axis)
                goal_rot = Rotation.from_euler('xyz', [90, 0, target_yaw], degrees=True).as_quat()
            else:
                # Horizontal: existing rotation for flat placement
                goal_rot = Rotation.from_euler('xyz', [0, 90, target_yaw], degrees=True).as_quat()

        if force_altitude is not None:
            obj_goal[2] = force_altitude

    else:
        raise ValueError(f"Unknown grasp: {grasp}")

    if orientation != 0:
        if placement_angle is not None:
            target_yaw = np.clip(placement_angle, -75, 75)
        else:
            target_yaw = obj_yaw
        goal_rot = np.array([target_yaw])

        # when vertical, palm always faces towards +y (left of robot from robot pov)
        if vertical:
            # For vertical with orientation tracking, need to specify the full rotation
            goal_rot = Rotation.from_euler('xyz', [100, 0, target_yaw], degrees=True).as_quat()
        else:
            # Original behavior: just yaw angle
            goal_rot = np.array([target_yaw])

    # Compute goal position in IIWA frame
    print(obj_goal)
    obj_pos_in_iiwa = np.squeeze(np.linalg.inv(iiwa_base_pos) @ obj_goal)[:3]
    if goal_rot is not None:
        obj_pos_in_iiwa = np.concatenate([obj_pos_in_iiwa, goal_rot])

    # Initializing controller
    js_lds.joint_speed_scale = speed
    js_lds.orientation_factor = orientation

    if obstacle_clearance is None:
        js_lds.reset_collosion_proximity()
    else:
        js_lds.collision_proximity = obstacle_clearance

    # Set obstacles
    # obstacles = ["apple", "eaten_apple",
    #              "paper_ball_1", "paper_ball_2",
    #              # "paper_ball_3",
    #              "champagne_1", "champagne_2",
    #              "table", "sink", "shelf", "trash_bin"]
    # obstacles = ["eaten_apple", "champagne_1", "sink", "shelf", "trash_bin", "table"]
    obstacles = llmu.OBSTACLES.copy()
    if disregard_table:
        obstacles.remove("table")

    if disregard_object_to_grasp or js_lds.grasping:
        js_lds._obstacle_to_approach = js_lds.obj_grasped
    elif disregard_object is not None:
        js_lds._obstacle_to_approach = disregard_object 
    else:
        js_lds._obstacle_to_approach = ""

    (meshes, radii, names) = get_meshes(obstacles, detailed_meshes=detailed_obstacles)
    js_lds.set_obstacles(meshes, radii, names)
    js_lds._obstacle_ik = obstacle_ik
    js_lds.cartesian_goal = obj_pos_in_iiwa
    print(f"APPROACH DEBUG: _failed_ik={js_lds._failed_ik}, _in_collision={js_lds._in_collision}, mock_run={mock_run}")
    print(f"APPROACH DEBUG: Will call run_controller? {not js_lds._failed_ik and not mock_run}")
    if not js_lds._failed_ik and not mock_run:
        js_lds.run_controller()
    else:
        print(f"APPROACH DEBUG: Skipping run_controller!")


@robot_action
def pick(js_lds, object_to_grasp: str,
         speed: float,
         obstacle_clearance: tp.Optional[float] = None,
         grasp_orientation: tp.Optional[str] = None,
         mock_run: bool = False) -> None:

    

    # Fine-tuned approach
    approach(object_to_grasp, speed, obstacle_clearance,
                grasp_orientation,
                disregard_object_to_grasp=False,
                detailed_obstacles=True,
                apply_offsets=False,
                obstacle_ik=True,
                vertical_clearance_offset=0.01,
                mock_run=mock_run,
                disregard_table=False, # Was initially False
                disregard_object=object_to_grasp)
    if mock_run:
        return

    after_approach_pos = js_lds.hand_position[:3, 3]
    print(f"Position after approach: {after_approach_pos}")
    print(f"In collision: {js_lds._in_collision}")
    print(f"Obstacle collided: {js_lds._obstacle_collided}")
    print(f"Failed IK: {js_lds._failed_ik}")
    print(f"Timeout: {js_lds.timeout}")

    if not js_lds._in_collision or js_lds._obstacle_collided == object_to_grasp:
        js_lds._in_collision = False
        js_lds._obstacle_collided = ''
        # Perform grasping
        js_lds.grasping = True
        js_lds.obj_grasped = object_to_grasp
        # obstacles = ["apple", "eaten_apple",
        #             "paper_ball_1", "paper_ball_2",
        #             #"paper_ball_3",
        #             "champagne_1", "champagne_2", "sink", "shelf",
        #             "trash_bin"]
        # obstacles = ["eaten_apple", "champagne_1", "sink", "shelf", "trash_bin"]
        obstacles = llmu.OBSTACLES.copy()

        js_lds._obstacle_to_approach = js_lds.obj_grasped

        (meshes, radii, names) = get_meshes(obstacles, detailed_meshes=True)
        js_lds.set_obstacles(meshes, radii, names)
        js_lds._obstacle_ik = True
        try:
            js_lds.run_controller()
        except ValueError as e:
            if not (js_lds._in_collision and js_lds._obstacle_collided == object_to_grasp):
                raise e
        print(f"Position after grasp: {js_lds.hand_position[:3, 3]}")

    # Flyoff straight up to avoid some collisions
    if not js_lds._in_collision or js_lds._obstacle_collided == object_to_grasp:
        js_lds._in_collision = False
        js_lds._obstacle_collided = ''
        # Fly off as high as possible up to 0.4
        current_hand_pos = js_lds.hand_position[:3, 3]
        
        # Get current orientation as quaternion
        current_hand_matrix = js_lds.hand_position
        current_rotation = Rotation.from_matrix(current_hand_matrix[:3, :3])
        current_quat = current_rotation.as_quat()  # [qx, qy, qz, qw]

        print(f"Hand position before flyoff: {current_hand_pos}")

        for fly_off_offset in [0.2]:
            hand_pos_goal = np.copy(current_hand_pos)
            hand_pos_goal[2] += fly_off_offset
            
            # Create full 7-element goal with position and orientation
            full_goal = np.concatenate([hand_pos_goal, current_quat])
            
            js_lds.orientation_factor = .9
            print(f"Hand position flyoff goal: {hand_pos_goal}")
            js_lds.cartesian_goal = full_goal  # Pass 7-element goal
            
            if not js_lds._failed_ik and not js_lds._in_collision:
                print(f"Flyoff of pick {fly_off_offset}")
                js_lds.run_controller()
            else:
                print(f"Flyoff failed for pick")
                break

# These action functions expect object names that get resolved to positions
# Place and drop pass their first argument to approach function
# So modify approach function
# remove orientation
@robot_action
def place(js_lds, object_to_grasp: tp.Union[str, list],
          speed: float, obstacle_clearance: float, 
          placement_angle: float = 0.,
          vertical: bool = False) -> None:

    # Parse relative placement (e.g., "beam_2_end1")
    if isinstance(object_to_grasp, str) and '_end' in object_to_grasp:
        import re
        match = re.match(r'(beam_\d+)_end([12])', object_to_grasp)
        if match:
            beam_name = match.group(1)
            end_num = int(match.group(2))
            
            # Get beam position and quaternion
            rospy.wait_for_service('objPos')
            obj_frame_service = rospy.ServiceProxy('objPos', objPos, persistent=True)
            beam_data = obj_frame_service(beam_name).object_position
            beam_pos = np.array(beam_data[:3])
            beam_quat = np.array(beam_data[3:7])  # [w, x, y, z]
            
            # Convert quaternion to rotation matrix
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat([beam_quat[1], beam_quat[2], beam_quat[3], beam_quat[0]])  # scipy uses [x,y,z,w]
            rot_matrix = rot.as_matrix()
            
            # Beam's local x-axis in world frame (along length)
            beam_x_axis = rot_matrix[:, 0]
            
            # Calculate end position
            if end_num == 1:
                end_pos = beam_pos - ((BEAM_LENGTH / 2) - 0.05) * beam_x_axis
            else:  # end_num == 2
                end_pos = beam_pos + ((BEAM_LENGTH / 2) - 0.05) * beam_x_axis
            
            # Set z to table height + beam width + offset
            end_pos[2] = TABLE_HEIGHT + BEAM_WIDTH + 0.01 
            
            # Convert to list for placement
            object_to_grasp = end_pos.tolist()
            print(f"Placing at {beam_name}_end{end_num}: {object_to_grasp}")

    for vertical_offset in [0.2]:
        print(f"Trying place with vertical offset: {vertical_offset}")
        if js_lds._in_collision:
            print("Place entered break")
            break

        place_grasp = "side" if vertical else "top"
        
        print("First place approach")
        approach(object_to_grasp,
                speed, grasp=place_grasp,
                placement_angle=placement_angle,
                detailed_obstacles=True,
                disregard_object_to_grasp=True,
                vertical_clearance_offset=vertical_offset,
                disregard_table = False,
                apply_offsets=False,
                obstacle_clearance=obstacle_clearance,
                drop_side_offset=True,
                vertical=vertical)
        print(f"After first approach: _failed_ik={js_lds._failed_ik}, _in_collision={js_lds._in_collision}")
        if not js_lds._failed_ik:
            break

    # call to correct based on beam
    # just need to make vertical = None possible
    # and give object to grasp the current x y position (?) or not
    if not js_lds._in_collision:
        print("Second place approach")
        approach(object_to_grasp,
                    speed, grasp=place_grasp,
                    placement_angle=placement_angle,
                    detailed_obstacles=True,
                    disregard_object_to_grasp=True,
                    vertical_clearance_offset=0.03,
                    disregard_table = False,
                    apply_offsets=False,
                    obstacle_clearance=obstacle_clearance,
                    drop_side_offset=True,
                    vertical=vertical)
        print(f"After second approach: _failed_ik={js_lds._failed_ik}, _in_collision={js_lds._in_collision}")

    if not js_lds._in_collision:
        # Drop the object
        js_lds.let_go = True
        js_lds.grasping = False
        js_lds.run_controller()
        #js_lds.grasping = False
        js_lds.let_go = False

    if not js_lds._in_collision:
        # Flyoff
        # Try to enable obstacle avoidance here
        print("Flyoff of place")
        hand_pos_goal = js_lds.hand_position[:3, 3]

        if vertical:
            hand_pos_goal[2] += 0.2     # z
            hand_pos_goal[1] -= 0.05   # y
        else:
            hand_pos_goal[2] += 0.3

        # js_lds.orientation = 1.

        js_lds.collision_proximity = 0.05 # to change obstacle clearance
        js_lds._obstacle_ik = True # to enable obstacle avoidance

        js_lds.cartesian_goal = hand_pos_goal
        
        print(f"Before running place flyoff: _failed_ik={js_lds._failed_ik}")
        if not js_lds._failed_ik:

            js_lds.run_controller()
            js_lds.obj_grasped = ""


@robot_action
def drop(js_lds, object_to_grasp: str,
         speed: float = 1.,
         obstacle_clearance: float = 0.05,
         orientation: float = 0,
         placement_angle: float = None) -> None:
    print(orientation)
    approach(object_to_grasp,
             speed, grasp="top",
             orientation=orientation,
             placement_angle=placement_angle,
             detailed_obstacles=True,
             disregard_object_to_grasp=True,
             disregard_table=True,
             apply_offsets=False,
             force_altitude=1.5,
             obstacle_clearance=obstacle_clearance,
             drop_side_offset=False)
    if not js_lds._in_collision and not js_lds._failed_ik:
        # Drop the object
        js_lds.let_go = True
        js_lds.grasping = False
        js_lds.run_controller()
        js_lds.let_go = False
        js_lds.obj_grasped = ""


@robot_action
def throw(js_lds, object_to_grasp: str,
          speed: float = 1.,
          obstacle_clearance: float = 0.05) -> None:

    approach(object_to_grasp,
             speed, grasp="top",
             orientation=0.,
             disregard_object_to_grasp=True,
             force_altitude=1.3,
             obstacle_clearance=obstacle_clearance,
             drop_side_offset=True)

    if not js_lds._in_collision:
        # Drop the object
        js_lds.let_go = True
        js_lds.grasping = False
        js_lds.run_controller()
        js_lds.let_go = False
        js_lds.obj_grasped = ""
