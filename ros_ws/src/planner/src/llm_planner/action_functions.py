#### Operators
def drop(location: str) -> None: #Releases the grasped object mid-air dropping it over the "location"
    print("drop " + location)

def throw(location: str) -> None: #Throw the grasped object towards the "location"
    print("throw " + location)

def approach(object_to_grasp: str, speed: float, grasp: str) -> None:  #Instructs the robot to move the its gripper, not holding any object, to the vicinity of "object_to_grasp".
    print("approach " + object_to_grasp)  #Implements logic to move the robot's gripper to the vicinity of "object_to_grasp"

def pick(object_to_grasp: str, orientation: float, speed: float) -> None: # Instructs the robot to pick up the "object_to_grasp", if it is close enough
    print("pick " + object_to_grasp) #If fixed_orientation == False, pick the "object_to_grasp" as quickly as possible, else pick steadily while maintaining the original orientation of the "object_to_grasp".
    return object_to_grasp

def place(location: str, orientation: float, speed: float) -> None: #Positions the "grasped_object" at the "location" and release the grasp. In case of collision with external object, returns the name of the object. Else, returns an empty string.
    print("place " + location) #Place the "grasped_object" on a flat "surface" gently.
