import numpy as np

import rospy

from primitives.js_lds_oriented import JS_LDS_ORIENTED
from llm_common import utils as llmu


def main() -> None:
    print("Starting main")
    rospy.init_node("js_lds_oriented")
    print("Node initialized")

    js_lds = JS_LDS_ORIENTED(llmu.IIWA_URDF_FOLDER, llmu.IIWA_URDF_PATH, llmu.ALLEGRO_URDF_FOLDER, llmu.ALLEGRO_URDF_PATH)
    print("LDS initialized")
    js_lds.cartesian_goal = np.array([-0.7, -0.05, 0.45, 180])
    js_lds.joint_speed_scale = 0.5
    js_lds.orientation_factor = 1.0
    print("Cartesian goal set")
    js_lds.run_controller()
    print("Controller ran")
    rospy.signal_shutdown("Controller test finished")


if __name__ == "__main__":
    main()