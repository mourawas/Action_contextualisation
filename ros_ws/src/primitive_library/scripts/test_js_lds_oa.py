import numpy as np

import rospy

from primitives.js_lds_oa import JS_LDS_OA
from llm_common import utils as llmu


def main() -> None:

    obstacle_list = np.asarray([[-0.60, -0.05, 0.30],
                                [ 0.50,  0.00, 0.40]])

    print("Starting main")
    rospy.init_node("js_lds_oriented")
    print("Node initialized")

    js_lds = JS_LDS_OA(llmu.IIWA_URDF_FOLDER, llmu.IIWA_URDF_PATH, llmu.ALLEGRO_URDF_FOLDER, llmu.ALLEGRO_URDF_PATH)
    print("LDS initialized")
    js_lds.cartesian_goal = np.array([-0.7, -0.05, 0.45, 180])
    js_lds.joint_speed_scale = 1.
    js_lds.orientation_factor = 0.8
    js_lds.set_obstacles(obstacle_list)
    print("Cartesian goal set, going to trash")
    js_lds.run_controller()
    print("Trash reached")

    print("Going back to work bench")
    js_lds.cartesian_goal = np.array([0.7, -0.05, 0.8, 0])
    js_lds.run_controller()
    rospy.signal_shutdown("Controller test finished")


if __name__ == "__main__":
    main()