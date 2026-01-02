import numpy as np
import typing as tp
import re
from scipy.spatial import distance

import rospy

from llm_common import helpers as llmh
from llm_simulator.srv import objPos, objMesh
from primitives.action_functions import pick, get_meshes, get_table_mesh, get_shelf_mesh

# Constants from action_functions.py
BEAM_LENGTH = 0.3  # meters
BEAM_WIDTH = 0.07  # meters
TABLE_HEIGHT = 1.01  # meters

def can_grasp(object_to_grasp: str, grasp_side: str) -> bool:
    from primitives.action_functions import js_lds
    GRASP_DISTANCE = js_lds.collision_proximity + 0.5
    # GRASP_DISTANCE = js_lds.collision_proximity + 0.15

    # Check if the object is in the vicinity of the hand
    object_in_vicinity = True

    dst_to_obj = js_lds.hand_distance_to_obj(object_to_grasp)

    if dst_to_obj > GRASP_DISTANCE:
        object_in_vicinity = False

    # Check if the object is in the workspace by performing a mock pick
    pick(object_to_grasp, 1., 0.0, grasp_side, mock_run=True)
    object_in_workspace = not js_lds._failed_ik

    # Check holding
    (mesh, radii, name) = get_meshes([object_to_grasp], detailed_meshes=True)
    js_lds.set_obstacles(mesh, radii, name)
    is_holding = js_lds.is_holding()

    return (object_in_vicinity and object_in_workspace) or is_holding

# Modify to handle table positions 
def can_reach(object_to_reach: str, grasp_side: str) -> bool:
    from primitives.action_functions import js_lds
    from primitives.execute_task_plan import TaskPlanExecutor
    
    # Check if it's a table position
    tpe = TaskPlanExecutor()
    if object_to_reach in tpe.table_positions:
        # For table positions, pass the position list to pick
        position = tpe.table_positions[object_to_reach]
        pick(position, 1., 0.0, grasp_side, mock_run=True)
    else:
        # For objects, use the object name
        pick(object_to_reach, 1., 0.0, grasp_side, mock_run=True)
    
    object_reachable = not js_lds._failed_ik
    return object_reachable


def collision_free() -> str:
    from primitives.action_functions import js_lds
    obstacle_collided = ''

    if js_lds._in_collision:
        if js_lds.obj_grasped != js_lds._obstacle_collided:
            obstacle_collided = js_lds._obstacle_collided

    return obstacle_collided


def timeout() -> bool:
    from primitives.action_functions import js_lds
    return not js_lds.timeout


def check_motion_health() -> bool:
    from primitives.action_functions import js_lds
    motion_health = js_lds.compute_motion_health(reset=False)
    return motion_health > 0


def get_motion_health() -> float:
    from primitives.action_functions import js_lds
    motion_health = js_lds.compute_motion_health()
    return motion_health


def holding() -> bool:
    from primitives.action_functions import js_lds
    is_holding = False
    if js_lds.obj_grasped != "":
        (mesh, radii, name) = get_meshes([js_lds.obj_grasped], detailed_meshes=True)
        js_lds.set_obstacles(mesh, radii, name)
        is_holding = js_lds.is_holding()

    return is_holding

# Modify to handle positions directly
def at_location(object: str, location: tp.Union[str, list]) -> bool:
    from primitives.action_functions import js_lds
    from primitives.execute_task_plan import TaskPlanExecutor
    from scipy.spatial.transform import Rotation

    if location == "robot":
        if object == js_lds.obj_grasped:
            return holding()
        else:
            return False
        
    # Check if location is a coordinate list [x, y, z]
    if isinstance(location, list):
        target_pos = np.array(location)
        obj_mesh, obj_radii, _ = get_meshes([object], detailed_meshes=True)
        
        # Only check x-y distance (horizontal plane)
        xy_distances = np.linalg.norm(obj_mesh[:, :2] - target_pos[:2], axis=1)
        min_dst = np.min(xy_distances - obj_radii)
        
        return min_dst < 0.02  # bigger tolerances are more fun

    # NEW: Parse beam end locations (e.g., "beam_2_end1", "beam_2_end2")
    if isinstance(location, str) and '_end' in location:
        match = re.match(r'(beam_\d+)_end([12])', location)
        if match:
            beam_name = match.group(1)
            end_num = int(match.group(2))
            
            # Get beam position and quaternion from vision service
            rospy.wait_for_service('objPos')
            obj_frame_service = rospy.ServiceProxy('objPos', objPos, persistent=True)
            beam_data = obj_frame_service(beam_name).object_position
            beam_pos = np.array(beam_data[:3])
            beam_quat = np.array(beam_data[3:7])
            
            # Convert quaternion to rotation matrix
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat([beam_quat[1], beam_quat[2], beam_quat[3], beam_quat[0]])
            rot_matrix = rot.as_matrix()
            
            # Beam's local x-axis in world frame (along length)
            beam_x_axis = rot_matrix[:, 0]
            
            # Calculate end position
            if end_num == 1:
                target_pos = beam_pos - ((BEAM_LENGTH / 2) - 0.05) * beam_x_axis
            else:  # end_num == 2
                target_pos = beam_pos + ((BEAM_LENGTH / 2) - 0.05) * beam_x_axis
            
            # Get object mesh and check distance
            obj_mesh, obj_radii, _ = get_meshes([object], detailed_meshes=True)
            
            # Only check x-y distance (horizontal plane) - SAME AS COORDINATE LISTS
            xy_distances = np.linalg.norm(obj_mesh[:, :2] - target_pos[:2], axis=1)
            min_dst = np.min(xy_distances - obj_radii)
            
            return min_dst < 0.02  # bigger tolerances are more fun

    # Check if location is a table position
    tpe = TaskPlanExecutor()
    if location in tpe.table_positions:
        # Get the table position coordinates
        table_pos = tpe.table_positions[location]
        
        # Get object position
        obj_mesh, obj_radii, _ = get_meshes([object], detailed_meshes=True)
        
        # Calculate distance to table position
        distances = np.linalg.norm(obj_mesh - np.array(table_pos), axis=1)
        min_dst = np.min(distances - obj_radii)
        
        return min_dst < 0.02  # bigger tolerances are more fun
    
    else: # from original work, unused
        # Existing code for object-to-object distance
        obj_mesh, obj_radii, _ = get_meshes([object], detailed_meshes=True)
        location_mesh, location_radii, _ = get_meshes([location], detailed_meshes=True)
        distances = distance.cdist(obj_mesh, location_mesh)
        
        distances = (distances.T - obj_radii).T
        distances = distances - location_radii
        
        min_dst = np.min(distances)
        return min_dst < 0.75
    
def beam_contact(beam1: str, beam2: str, tolerance: float = 0.08) -> bool:
    # Check if two beams are touching within tolerance.
    beam1_mesh, beam1_radii, _ = get_meshes([beam1], detailed_meshes=True)
    beam2_mesh, beam2_radii, _ = get_meshes([beam2], detailed_meshes=True)
    
    distances = distance.cdist(beam1_mesh, beam2_mesh)
    distances = (distances.T - beam1_radii).T
    distances = distances - beam2_radii
    
    min_dst = np.min(distances)
    return min_dst < tolerance  # 8CM IS ACTUALLY 1CM BECAUSE OF THE SPHERES

def beam_angle(beam1: str, beam2: str, target_angle: float = 90.0, 
               tolerance: float = 5.0) -> bool:
    from scipy.spatial.transform import Rotation
    
    # Get beam orientations from vision service
    rospy.wait_for_service('objPos')
    obj_frame_service = rospy.ServiceProxy('objPos', objPos, persistent=True)
    
    beam1_data = obj_frame_service(beam1).object_position
    beam2_data = obj_frame_service(beam2).object_position
    
    # Extract quaternions [w, x, y, z]
    beam1_quat = np.array(beam1_data[3:7])
    beam2_quat = np.array(beam2_data[3:7])
    
    # Convert to rotation matrices
    # Note: scipy uses [x, y, z, w] format
    rot1 = Rotation.from_quat([beam1_quat[1], beam1_quat[2], 
                               beam1_quat[3], beam1_quat[0]])
    rot2 = Rotation.from_quat([beam2_quat[1], beam2_quat[2], 
                               beam2_quat[3], beam2_quat[0]])
    
    # Get beam length directions (local x-axis in world frame)
    axis1 = rot1.as_matrix()[:, 0]
    axis2 = rot2.as_matrix()[:, 0]
    
    # Calculate angle between axes using dot product
    dot_product = np.clip(np.dot(axis1, axis2), -1.0, 1.0)
    
    # Use abs() to map angles to [0°, 90°] range
    # This treats parallel beams the same regardless of direction
    angle_rad = np.arccos(np.abs(dot_product))
    angle_deg = np.degrees(angle_rad)
    
    # Check if within tolerance
    return np.abs(angle_deg - target_angle) < tolerance

def beam_parallel(beam1: str, beam2: str, tolerance: float = 5.0) -> bool:

    return beam_angle(beam1, beam2, target_angle=0.0, tolerance=tolerance)
