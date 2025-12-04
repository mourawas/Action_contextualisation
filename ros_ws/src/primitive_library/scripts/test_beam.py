import rospy
import time
from primitives.action_functions import approach, pick, place, drop
from primitives.predicates import holding, collision_free

def main() -> None:

    print("Starting main")
    rospy.init_node("test_beam")
    print("Node initialized")
    
    # object_to_grasp = "beam_1"

    # print("Approach")
    # approach(object_to_grasp=object_to_grasp,
    #          speed=0.7,
    #          grasp="top", 
    #          obstacle_clearance=0.005)
    # print(collision_free())
    
    # print("Pick")
    # pick(object_to_grasp=object_to_grasp,
    #      speed=0.8, obstacle_clearance=0.001, grasp_orientation="top")
    # time.sleep(4)
    # print(collision_free())
    # holding()
    
    # print("Place")
    # place(object_to_grasp=[0.5, 0.3, 1.08], # +y goes to the left of the robot, +x goes away from the robot starts at 0.4
    #       speed=0.4, obstacle_clearance=0.02, orientation=0.0, placement_angle=None, vertical=True)
    # print(collision_free())

    # next beam
    object_to_grasp = "beam_1"

    print("Approach")
    approach(object_to_grasp=object_to_grasp,
             speed=0.8,
             grasp="top", 
             obstacle_clearance=0.05)
    print(collision_free())

    print("Pick")
    pick(object_to_grasp=object_to_grasp,
         speed=0.8, obstacle_clearance=0.001, grasp_orientation="top")
    time.sleep(4)
    print(collision_free())
    holding()

    print("Place")
    place(object_to_grasp="beam_2_end2",
          speed=0.4, obstacle_clearance=0.02, orientation=0.0, placement_angle=None, vertical=True)
    print(collision_free())

    # try the priginal way of grasping
    rospy.signal_shutdown("Beam test finished")


if __name__ == "__main__":
    main()