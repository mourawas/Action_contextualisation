import rospy
import time
from primitives.action_functions import approach, pick, place, drop
from primitives.predicates import holding, collision_free

def main() -> None:

    print("Starting main")
    rospy.init_node("test_beam")
    print("Node initialized")

    # this kinda works
    # print("Approach")
    # approach(object_to_grasp="beam_3",
    #          speed=0.8,
    #          grasp="side", 
    #          obstacle_clearance=0.05)
    
    # print("Pick")
    # pick(object_to_grasp="beam_3",
    #      speed=0.8, obstacle_clearance=0.001, grasp_orientation="side")
    # time.sleep(4)
    # holding()
    
    # print("Place")
    # place(object_to_grasp=[0.63, -0.15, 1.07],
    #       speed=0.4, obstacle_clearance=0.02, placement_angle=0.0, vertical=True)


    # print("Approach")
    # approach(object_to_grasp="beam_2",
    #          speed=0.8,
    #          grasp="top", 
    #          obstacle_clearance=0.05)

    # print("Pick")
    # pick(object_to_grasp="beam_2",
    #      speed=0.8, obstacle_clearance=0.001, grasp_orientation="top")
    # time.sleep(4)
    # holding()

    # print("Place")
    # place(object_to_grasp=[0.62, 0.07, 1.07],
    #       speed=0.4, obstacle_clearance=0.005, placement_angle=0.0, vertical=True)

    #other way around

    print("Approach")
    approach(object_to_grasp="beam_2",
             speed=0.8,
             grasp="top", 
             obstacle_clearance=0.05)

    print("Pick")
    pick(object_to_grasp="beam_2",
         speed=0.8, obstacle_clearance=0.001, grasp_orientation="top")
    time.sleep(4)
    holding()

    print("Place")
    place(object_to_grasp=[0.62, 0.08, 1.07],
          speed=0.4, obstacle_clearance=0.005, placement_angle=0.0, vertical=True)


    print("Approach")
    approach(object_to_grasp="beam_3",
             speed=0.8,
             grasp="side", 
             obstacle_clearance=0.05)
    
    print("Pick")
    pick(object_to_grasp="beam_3",
         speed=0.8, obstacle_clearance=0.001, grasp_orientation="side")
    time.sleep(4)
    holding()
    
    print("Place")
    place(object_to_grasp=[0.63, -0.15, 1.07],
          speed=0.4, obstacle_clearance=0.02, placement_angle=0.0, vertical=True)

    rospy.signal_shutdown("Beam test finished")


if __name__ == "__main__":
    main()