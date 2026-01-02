import rospy
import time
from primitives import action_functions
from primitives.action_functions import approach, pick, place, drop

def main() -> None:

    print("Starting main")
    rospy.init_node("test_logging")
    print("Node initialized")

    # 'left_side': [0.6, 0.5, 1.01],
    # 'center': [0.6, 0.1, 1.01],
    # 'right_side': [0.6, -0.3, 1.01]
    

    print("Approach 1")
    approach(object_to_grasp=[0.6, 0.5, 1.05],
             speed=0.7,
             grasp="top", 
             obstacle_clearance=0.05)

    action_functions.js_lds.start_logging("approach_test_no_obstacle")

    print("Approach 2")
    approach(object_to_grasp=[0.6, -0.3, 1.05],
             speed=0.7,
             grasp="top", 
             obstacle_clearance=0.05)

    filepath = action_functions.js_lds.save_log()
    print(f"Log saved: {filepath}")
    
    print("\n" + "="*60)
    print("DONE - Log saved to:")
    print(filepath)
    print("="*60)

    rospy.signal_shutdown("logging test finished")


if __name__ == "__main__":
    main()