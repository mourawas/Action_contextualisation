from typing import Dict
from typing import Tuple

def can_grasp(object_to_grasp: str) -> bool:
    # if True:
    #     print("object_to_grasp")
    #     print(object_to_grasp)
    #     print(object_to_grasp == "plant_in_wicker_pot_1")
    #     if object_to_grasp == "plant_in_wicker_pot_1":
    #         return False
    # else:
    #     return True #True if robot is close-enough to the "object_to_grasp" to securley grasp it
    # # return True #True if robot is close-enough to the "object_to_grasp" to securley grasp it

    return True

def holding() -> bool:
    return False

def at_location(object: str, location: str) -> bool: # Returns True if the "object" is at the "location"
    return True

def collision_free() -> str: # Checks if the robot encountered an unforeseen collision while executing the preceding action. In case of collision, returns the name of the colliding object, else ''.
    return ''

def timeout() -> bool: # Returns True if the preceding action was executed within a span of 60 seconds
    return True

def check_motion_health() ->  bool: # Returns True if the robot's motion during the preceding action was safe for its hardware
    return True

def get_score() -> float: # Computes an intrinsic score of the motion on robot. Returns (True, score) if the preceding action received a score above an intrinsic threshold, and (False, score) otherwise
    return 0.5