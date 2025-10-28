import rospy
import time
from primitives.action_functions import approach, pick, place, drop
from primitives.predicates import holding, collision_free

def main() -> None:
    object_to_grasp = "beam_1"
    print("Starting main")
    rospy.init_node("test_beam")
    print("Node initialized")
    
    print("Approach")
    approach(object_to_grasp=object_to_grasp,
             speed=1.,
             grasp="top", 
             obstacle_clearance=0.005)
    print(collision_free())
    
    print("Pick")
    pick(object_to_grasp=object_to_grasp,
         speed=.5, obstacle_clearance=0.001, grasp_orientation="top")
    time.sleep(4)
    print(collision_free())
    holding()
    
    print("Place")
    place(object_to_grasp=[0.6, 0.2, 1.04], # +y goes to the left of the robot, +x goes away from the robot starts at 0.4
          speed=0.9, obstacle_clearance=0.02, orientation=0.0, placement_angle=-75.0)
    print(collision_free())

    
    rospy.signal_shutdown("Beam test finished")


if __name__ == "__main__":
    main()