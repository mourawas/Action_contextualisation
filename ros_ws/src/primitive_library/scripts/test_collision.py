import rospy
import time
from primitives.action_functions import approach, pick, place
from primitives.predicates import holding, collision_free

def main() -> None:

    print("Starting main")
    rospy.init_node("test_collision")
    print("Node initialized")


    # next beam
    object_to_grasp = "beam_3"

    print("Approach")
    approach(object_to_grasp=object_to_grasp,
             speed=0.8,
             grasp="side", 
             obstacle_clearance=0.05)

    print("Pick")
    pick(object_to_grasp=object_to_grasp,
         speed=0.9, obstacle_clearance=0.001, grasp_orientation="side")
    time.sleep(4)

    print("Place")
    place(object_to_grasp="beam_1_end1",
          speed=0.4, obstacle_clearance=0.02, placement_angle=None, vertical=True)

    rospy.signal_shutdown("Collision test finished")


if __name__ == "__main__":
    main()