"""Small files containing names and paths for the simulation.
"""
import os
from pathlib import Path

# Topics
IIWA_TOPIC = '/iiwa'
JOINT_SETTER_TOPIC = IIWA_TOPIC + "/joints"
IIWA_TORQUE_CMD_TOPIC = IIWA_TOPIC + "/TorqueController/command"
IIWA_JOINT_STATES_TOPIC = IIWA_TOPIC + "/joint_states"
ALLEGRO_TOPIC = '/allegro_hand_right'
ALLEGRO_JOINT_STATES_TOPIC = ALLEGRO_TOPIC + "/joint_states"
ALLEGRO_TORQUE_CMD_TOPIC = ALLEGRO_TOPIC + "/torque_cmd"
SIM_RESET_TOPIC = "/sim_reset"

WS_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

# Simulator related paths
LLM_SIM_PATH = WS_PATH / "llm_simulator/"
LLM_CACHE_FOLDER = LLM_SIM_PATH / "src/.cache/"
OBJECT_COLLIDER_FILE = LLM_CACHE_FOLDER / "colliders.pkl"
MUJOCO_DESCRIPTION_FOLDER = LLM_SIM_PATH / "src/description/"
MUJOCO_KINEMATICS_FOLDER = LLM_SIM_PATH / "src/kinematics/"
MUJOCO_MODEL_PATH = MUJOCO_DESCRIPTION_FOLDER / "iiwa7_allegro_llm.xml"

# Robots urdf paths
IIWA_URDF_FOLDER = WS_PATH / "primitive_library/descriptions/iiwa_yitting/meshes"
IIWA_URDF_PATH = IIWA_URDF_FOLDER / "model.urdf"
ALLEGRO_URDF_FOLDER = WS_PATH / "primitive_library/descriptions/allegro"
ALLEGRO_URDF_PATH = ALLEGRO_URDF_FOLDER / "allegro_hand_description/allegro_hand_description_left.urdf"

# TODO: This is not super effective. Think of a better way # HERE: Add new object names here
SIM_OBJECT_LIST = ["champagne_2","eaten_apple"]

# These objects can be approximated by a sphere (during approach)
ROUND_OBJECTS = ["eaten_apple"]

STATIC_ELEMENTS = ['table', 'sink', 'trash_bin', 'shelf'] # Static objects, name same as before
OBSTACLES = SIM_OBJECT_LIST + STATIC_ELEMENTS

# TODO: Get that programatically as well?
IIWA_JOINT_NAMES = ['iiwa_joint_1',
                    'iiwa_joint_2',
                    'iiwa_joint_3',
                    'iiwa_joint_4',
                    'iiwa_joint_5',
                    'iiwa_joint_6',
                    'iiwa_joint_7',]

# TODO: Check order
ALLEGRO_JOINT_NAMES = ['finger_0/joint_0',
                       'finger_0/joint_1',
                       'finger_0/joint_2',
                       'finger_0/joint_3',
                       'finger_1/joint_0',
                       'finger_1/joint_1',
                       'finger_1/joint_2',
                       'finger_1/joint_3',
                       'finger_2/joint_0',
                       'finger_2/joint_1',
                       'finger_2/joint_2',
                       'finger_2/joint_3',
                       'finger_3/joint_0',
                       'finger_3/joint_1',
                       'finger_3/joint_2',
                       'finger_3/joint_3',]
