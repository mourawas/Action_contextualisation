import numpy as np
from math import pi
from scipy.spatial.transform import Rotation

import rospy

from primitives.js_lds import JS_LDS
from primitives.js_lds_oriented import JS_LDS_ORIENTED
from llm_common import utils as llmu



js_lds = JS_LDS(llmu.IIWA_URDF_FOLDER, llmu.IIWA_URDF_PATH, llmu.ALLEGRO_URDF_FOLDER, llmu.ALLEGRO_URDF_PATH)
js_lds_oriented = JS_LDS_ORIENTED(llmu.IIWA_URDF_FOLDER, llmu.IIWA_URDF_PATH, llmu.ALLEGRO_URDF_FOLDER, llmu.ALLEGRO_URDF_PATH)

cart_pos_dict = \
    {'large red trash can':         {'approach':  [-0.7, 0.00, 0.7, 0, 90, 180],
                                     'top':       [-0.7, 0.00,  0.45, 0, 90, 180],
                                     'hack':      [0., 0.6, 0.7, 0., 90, 180],
                                     'side':      []},
     'crumpled paper ball 1':       {'approach':  [0.47, -0.38, 0.9, 0, 90, 0],
                                     'top':       [0.47, -0.38, 0.62, 0, 90, 0],
                                     'side':      []},
     'empty glass 1':               {'approach':  [0.6,  0.21, 0.9, 0, 90, 0],
                                     'top':       [0.6,  0.21, 0.80, 0, 90, 0],
                                     'side':      []},
     'empty glass 2':               {'approach':  [0.6,  -0.21, 0.9, 0, 90, 0],
                                     'top':       [0.6,  -0.21, 0.80, 0, 90, 0],
                                     'side':      []},
     'glass with yellowish liquid': {'approach':  [0.6,  0.00, 0.9, 0, 90, 0],
                                     'top':       [0.6,  0.00, 0.80, 0, 90, 0],
                                     'side':      []}}


def approach(object: str, speed: float, side: str, grasping: bool = False, orientation: float = 0, move_yaw = False) -> None:

    if orientation == 0:
        controller = js_lds
        goal_xyz = cart_pos_dict[object][side][:3]
        goal_rot = Rotation.from_euler('xyz', cart_pos_dict[object][side][3:], degrees=True).as_quat().tolist()
        goal = goal_xyz + goal_rot
    else:
        controller = js_lds_oriented
        controller.orientation_factor = orientation
        if move_yaw:
            goal = cart_pos_dict[object][side][:3] + [cart_pos_dict[object][side][-1]]
        else:
            goal = cart_pos_dict[object][side][:3]

    controller.grasping = grasping
    controller.cartesian_goal = goal
    controller.joint_speed_scale = speed
    controller.run_controller()


def place_in_trashcan(speed: float, orientation: float = 0, hack=False) -> None:
    if orientation != 0:
        move_yaw = True
    else:
        move_yaw = False

    if hack:
        approach('large red trash can', speed, 'hack', orientation=orientation, grasping=True, move_yaw=move_yaw)
    approach('large red trash can', speed, 'approach', orientation=orientation, grasping=True, move_yaw=move_yaw)   # Place part 1
    approach('large red trash can', speed, 'top', orientation=orientation, grasping=True, move_yaw=move_yaw)        # Place part 2
    approach('large red trash can', speed, 'top', orientation=orientation, grasping=False, move_yaw=move_yaw)       # Place part 3

if __name__ == "__main__":
    print("Starting main")
    rospy.init_node("js_lds")
    print("Node initialized")

    approach('empty glass 2', .8, 'approach')
    approach('empty glass 2', .8, 'top', orientation=0)
    approach('empty glass 2', .8, 'top', grasping=True, orientation=0)  # Grasping
    approach('empty glass 2', 0.8, 'approach', grasping=True, orientation=0)  # Grasping
    place_in_trashcan(0.8, orientation=0.)

    approach('empty glass 1', 0.8, 'approach')                    # Approach
    approach('empty glass 1', 0.8, 'top', orientation=0)                         # Pick part 1
    approach('empty glass 1', 0.8, 'top', orientation=0, grasping=True)          # Pick part 2
    approach('empty glass 1', 0.8, 'approach', grasping=True, orientation=0.)          # Pick part 2
    place_in_trashcan(0.8, orientation=0.)

    approach('glass with yellowish liquid', 0.8, 'approach')
    approach('glass with yellowish liquid', 0.8, 'top', orientation=1)
    approach('glass with yellowish liquid', 0.8, 'top', orientation=1, grasping=True)  # Grasping
    approach('glass with yellowish liquid', 0.8, 'approach', grasping=True, orientation=0.8, move_yaw=False)  # Grasping
    place_in_trashcan(0.5, orientation=0.9, hack=False)



