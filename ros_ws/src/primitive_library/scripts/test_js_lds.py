import numpy as np
from math import pi
from scipy.spatial.transform import Rotation

import rospy

from primitives.js_lds import JS_LDS
from llm_common import utils as llmu


def main() -> None:
    print("Starting main")
    rospy.init_node("js_lds")
    print("Node initialized")

    js_lds = JS_LDS(llmu.IIWA_URDF_FOLDER, llmu.IIWA_URDF_PATH, llmu.ALLEGRO_URDF_FOLDER, llmu.ALLEGRO_URDF_PATH)
    print("LDS initialized")

    rot = Rotation.from_euler('xyz', [0, 90, 180], degrees=True)
    rot_quat = rot.as_quat()
    xyz_pos = [-0.7, -0.05, 0.45]
    js_lds.cartesian_goal = np.array(xyz_pos + rot_quat.tolist())
    js_lds.joint_speed_scale = .5
    js_lds.run_controller()
    print("Went to trash")

    rot = Rotation.from_euler('xyz', [0, 90, 0], degrees=True)
    rot_quat = rot.as_quat()
    xyz_pos = [0.69914, 0.00031, 0.6525]
    js_lds.cartesian_goal = np.array(xyz_pos + rot_quat.tolist())
    js_lds.joint_speed_scale = .5
    js_lds.grasping = False
    print("Cartesian goal set")
    js_lds.run_controller()
    print("Controller 1 ran")

if __name__ == "__main__":
    main()
