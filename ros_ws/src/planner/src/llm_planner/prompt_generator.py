# Import task_plan and evaluation_plan from 'param_server'
# from .param_server import task_plan, evaluation_plan

# Sample import for action functions and checking functions libraries
from .action_functions import *
from .checking_functions import *

from copy import deepcopy

import typing as tp



def image_description_gen():
    """Generates the prompt to generate the image description

    Returns:
        str: Prompt
    """
    prompt0 = "Generate a description of the scene and objects in it. Output as a Python string (named 'scene_description')"

    return prompt0


def label_gen():
    """Generates the prompt to generate labels for the objects

    Returns:
        str: Prompt
    """
    prompt0_5 = """
    Identify all the objects and generate relevant labels for them. Objects also includes potential the locations. When there are multiple objects of the same type, differentiate the labels by their respective counts. Output a list named 'object_labels', where each list element is a python string
    """

    return prompt0_5


def modify_labels(labels: tp.List[str]):
    """Adds the robot label to labels

    Args:
        labels (dict[str, str]): Existing labels and their descriptions
    """
    mod_labels = deepcopy(labels)
    mod_labels.append("robot_1")
    return mod_labels

# everything is defined and called from experiment file
def task_plan_gen(task: str, prompt0: str, objects: tp.List[str], locations: tp.List[str]):
    """Prepares the prompt to fetch the task plan

    Args:
        task (str): The task for the robot
        prompt0 (str): Image desription
        objects (list[str]): Object labels inside the scene
        locations (list[str]): Location labels inside the scene

    Returns:
        str: The composed prompt
    """

    # object_labels = modify_labels(objects) # Adding the robot_1 label to the list
    object_labels = deepcopy(objects) # Adding the robot_1 label to the list

    str_0 = prompt0
    str_1_1 = f"""The list of recognised objects is:

    objects = {object_labels}

    """
    str_1_2 = f"""The list of recognised locations is:

    locations = {locations}
    """
    # For grasp, added to never use None
    str_2 = f"""

    There is a robot, labeled 'robot', that can only manipulate ONE beam at a time. It is here to do beam assembly tasks. The robot accepts commands in the form 'action functions' written in python. These action functions, which can be imported from the 'action_functions' library are:

    approach(object_to_grasp: str, speed: float, obstacle_clearance: float, grasp: str) -> None:  # Moves the robot close to "object_to_grasp" so that the beam is in the robot's reach

    pick(object_to_grasp: str, speed: float, obstacle_clearance: float, grasp: str) -> None: # Instructs the robot to pick up the "object_to_grasp", if it is close enough

    place(location: str, speed: float, obstacle_clearance: float, placement_angle: float = 0., vertical: bool = False) -> None: # Positions the "grasped_object" on/at/in the "location" and release the grasp. It is not advised to use the approach function directly before this one.

    The "speed" argument for 'approach', 'pick', 'drop' and 'place' functions, assumes value in [0,1] and regulates how fast the robot moves. The closer the the value is to 1 the faster the robot moves. moving with higher speed is faster but might result in a jerky and less precise motion.

    The "grasp" argument for 'approach' and 'pick' is mandatory and assumes one of the two values {"top", "side"}, never use None, where "top" instructs the robot to approach or pick the beam from the top and selecting "side" instructs the robot to approach or pick the beam from the side. For the same object, the same grasp type should be used for both 'approach' and 'pick' functions. "top" should be used for beams that are laying down, "side" should be used for beams that are standing up.

    The "obstacle_clearance" for 'approach', 'pick' and 'place' functions defines how close the robot can get to a beam (including the one it is trying to grasp in the pick action) before starting to avoid it. The distance is in meter. Small values allow the robot to get closer to obstacles and usually give a better chance of reaching the beam, picking it and holding it. Typically values are between 0.005 and 0.05 although values out of this range are possible.

    The "vertical" argument for the 'place' function is a boolean that indicates whether the robot should place the beam standing up in a vertical orientation (True) or laying down in a horizontal orientation (False). If vertical=True, the robot will orient the beam such that its main axis is aligned with the vertical axis of the world frame during placement. If vertical=False, the robot will orient the beam such that its main axis is aligned with the horizontal plane of the world frame during placement.

    The "placement_angle" argument for the 'place' function defines the yaw angle in degrees at which the robot should place the beam down. The yaw describes an orientation within the horizontal plane, and is clipped between -75 and 75 degrees, with zero being straight ahead, and positive values are counter-clockwise when viewed from above. If vertical=True, this argument should be ignored.

    L (with 2 beams) or U-shape (with 3 beams) assembly tasks can be requested. To do so, one beam should be laying flat on the table and serve as a base for the other beam(s) to be placed vertically (standing up) on top of the base beam, on its ends. The base beam could already be laying down without needing any action to it, or it could be placed down by the robot as part of the task.
    In the "place" function:
    - If you want to place down the base beam: use location='left_side', 'center', or 'right_side'
    - For beam assembly/stacking: use location='beam_X_end1' or 'beam_X_end2' where X is the beam number.
    Example: place('beam_1_end1', ...) places the held beam at end 1 of beam_1

    end2 is the left end, end1 is the right end. Pick and place beams from left to right for efficient motion (left beam to end2, right beam to end1).

    The actions described in these functions are the only motions known to the robot. The task for the robot is: "{task}". First explain how you are going to solve the task, why each step is executed and how it makes sense to a human that you do it this way.

    Then, using the actions functions, 'objects' and 'locations', define a task plan as a Python list of tuples (named 'task_plan'), for the robot to follow. The action_functions from the task_plan will be directly run on the robot. Each element of the list is a tuple of the form (action number, action function name string, (arguments)). For each action, use object and task specific arguments.

    The first index of the plan should be 0.

    Do not make any assumptions. For the task plan, output a single code block. With each action include as the comment the reasoning behind the action and its parameters.

    Assume the necessary modules containing action functions have already been loaded. Write only a single assignment statement creating the full 'task_plan'

    Do not abbreviate anything in the code you are writing. No matter how long it is, write it in full.
    """

    prompt1 = str_0 + str_1_1 + str_1_2 + str_2

    return prompt1


def eval_plan_gen():
    """Generates the prompt to fetch the evaluation plan
    """
    #Added the "for functions that take no arguments" part
    prompt2 = """
    The robot may not be able to execute an action function, or encounter object collision during execution. Thus, it is required to check for completion of each action function after they have been performed.

    For this, we define some 'checking functions' written in python. These checking functions, which can be imported from the 'checking_functions' library are:

    can_grasp(object_to_grasp: str, grasp: str) -> bool: # Returns True if robot is close-enough to the "object_to_grasp" to securely grasp it with the determined grasp "side" or "top"

    holding() -> bool # Returns True if the robot is holding an object

    at_location(object: str, location: str) -> bool: # Returns True if the "object" is at the "location"

    collision_free() -> str: # If the robot encounters a collision while executing the preceding action, returns the object label string. Otherwise, ''.

    timeout() -> Bool: # Returns True if the preceding action was executed in a timely fashion

    check_motion_health() ->  bool: # Returns True if the robot's motion during the preceding action was safe for its hardware

    can_reach(goal: str, grasp: str) -> bool: # Returns True if it is feasible for the robot to reach the "goal" object or location from the current state from the side determined by the grasp argument "side" or "top". Objects that are out of the workspace will always return False.

    beam_contact(beam1: str, beam2: str, tolerance: float = 0.08) -> bool: # Returns True if two beams are touching within the given tolerance (in meters). Default tolerance is 5cm.

    beam_angle(beam1: str, beam2: str, target_angle: float = 90.0, tolerance: float = 5.0) -> bool: # Returns True if the angle between two beams matches the target angle within tolerance (in degrees). Use target_angle=90.0 for perpendicular beams such as in an L or U shape. Default tolerance is ±5°.

    beam_parallel(beam1: str, beam2: str, tolerance: float = 10.0) -> bool: # Returns True if two beams are parallel within tolerance (in degrees). Equivalent to beam_angle with target_angle=0.0. Use for the two vertical beams in a U-shape assembly. Default tolerance is ±5°.

    The grasp argument is the same as the one in the "approach" and "pick" functions. It assumes one of the two values {"top", "side"}

    Using the checking_functions, locations and objects, define an evaluation plan (named 'evaluation_plan') to verify the succesful execution of each action. Additionally, for each action verify without fail:
    - collision free
    - timely motion
    - motion health

    Important guidelines for our construction tasks:
    - Multiple checks of the same type can be included in a single action's evaluation (e.g., check contact with multiple beams)
    - Structural checks (beam_contact, beam_angle, beam_parallel) should typically only be verified AFTER the complete structure is assembled
    - Intermediate actions should verify basic predicates: can_grasp, holding, collision_free, timeout, check_motion_health
    - The final action should verify both basic predicates AND all structural requirements (contact, angles, parallelism)

    Each tuple is meant to be checked after the action with the corresponding number.

    Output this plan as a Python list of tuples, where each tuple is of the form (action_num, [('predicate_name', (args,)), ...], (expected_results,)). Do not assume any other object or location, beyond those in object_labels.
    
    For functions that take no arguments (collision_free, timeout, check_motion_health, holding), use empty tuples () as the value. For functions that take arguments, use a tuple containing the arguments.

    Each tuple is meant to be checked after the acton with the corresponding number.

    Generate the entire plan. No reasoning, direct output.
    """

    return prompt2


def object_type():
    """Segregates the object_labels into locations and objects"""
    prompt2_0_5 = """
    Segregate the labels present in object_labels into two lists - 'objects' and 'locations'. The 'objects' list should contain the labels of objects which the robot interacts with. The 'locations' list should contains labels of locations
    """

    return prompt2_0_5


def start_state_gen():
    """Generates the prompt to fetch the start state description
    """
    prompt2_1 = """
    Using the relevant checking functions, 'objects' and 'locations',  generate an abstract description of the location of objects for the initial scene.

    Output this description, by defining a Python list of tuples (named 'initial_description'), where each list element is a tuple of the form (check_function, (arguments), expected output).
    """
    return prompt2_1


def goal_state_gen():
    """Generates the prompt to fetch the goal state description
    """
    prompt2_2 = """
    Using the relevant checking functions, 'objects' and 'locations', generate an abstract description of the location of objects at the end of succesful task completion.

    Output this description, by defining a Python list of tuples (named 'final_description'), where each list element is a tuple of the form (check_function, (arguments), expected output).
    """
    return prompt2_2


def replan_correction_gen(history_log, task_plan, parameter_history):
    """For use on the real robot to handle intermediate failure"""

    failure_index = history_log[-1][0]
    failure_score = history_log[-1][2]
    failure_reason = history_log[-1][-1]
    action_name, action_args = task_plan[int(failure_index)][1], task_plan[int(failure_index)][2]

    str_0 = f"""
    The robot failed during the task_plan's execution. The details are:
    failure index: {failure_index}
    failed action: {action_name}
    current_arguments: {action_args}
    motion score: {failure_score}
    failure reason: {failure_reason}
    """

    # List of parameters of where each list element is of the form ((arguments), score)
    past_failed_parameters = []
    past_succesful_parameters = []

    str_past_failures = ""
    if len(parameter_history['failure'].keys()):
        str_past_failures = "Past failures:\n"
        for action in parameter_history['failure'].keys():
            for object_name in parameter_history['failure'][action].keys():
                failed_parameters = parameter_history['failure'][action][object_name]
                for f_param in failed_parameters:
                    f_arg = list(f_param[0])
                    f_score = f_param[1]
                    str_past_failures += f"FAILURE: {action}: {object_name}{tuple(f_arg)} | score = {f_score}\n"

    str_past_successes = ""
    if len(parameter_history['success'].keys()):
        str_past_successes = "Past successes:\n"
        for action in parameter_history['success'].keys():
            for object_name in parameter_history['success'][action].keys():
                success_parameters = parameter_history['success'][action][object_name]
                for s_param in success_parameters:
                    s_arg = list(s_param[0])
                    s_score = s_param[1]
                    str_past_successes += f"SUCCESS: {action}: {object_name}{tuple(s_arg)} | score = {s_score}\n"


    str_3 = """

    Perform replanning by either using alternative action functions or altering the object interaction sequence.
    Output the code as you did at first by assigning the full task plan to the variable 'task_plan' in a single statement.
    Take into account the parameters I have asked you to change during our conversation if any.
    In the new plan, make sure it is within plus or minus 1 action of the original task plan.
    """

    return str_0 + str_past_failures + str_past_successes +  str_3


def replan_after_success_gen(parameter_history, execution_time, average_score):
    """For use on the real robot to handle intermediate failure"""

    str_0 = f"""
        The action plan was successfull with a total time of {execution_time} and an average score of {average_score}.
        Based on the this action plan and the information followng suggest a new action plan that increases the score.
        """

    # List of parameters of where each list element is of the form ((arguments), score)
    past_failed_parameters = []
    past_succesful_parameters = []

    str_past_failures = ""
    if len(parameter_history['failure'].keys()):
        str_past_failures = "Past failures:\n"
        for action in parameter_history['failure'].keys():
            for object_name in parameter_history['failure'][action].keys():
                failed_parameters = parameter_history['failure'][action][object_name]
                for f_param in failed_parameters:
                    f_arg = list(f_param[0])
                    f_score = f_param[1]
                    str_past_failures += f"FAILURE: {action}: {object_name}{tuple(f_arg)} | score = {f_score}\n"

    str_past_successes = ""
    if len(parameter_history['success'].keys()):
        str_past_successes = "Past successes:\n"
        for action in parameter_history['success'].keys():
            for object_name in parameter_history['success'][action].keys():
                success_parameters = parameter_history['success'][action][object_name]
                for s_param in success_parameters:
                    s_arg = list(s_param[0])
                    s_score = s_param[1]
                    str_past_successes += f"SUCCESS: {action}: {object_name}{tuple(s_arg)} | score = {s_score}\n"


    str_3 = """

    Perform replanning by either using alternative action functions or altering the object interaction sequence.
    Output the code as you did at first by assigning the full task plan to the variable 'task_plan' in a single statement.
    Take into account the parameters I have asked you to change during our conversation if any.
    In the new plan, make sure it is within plus or minus 1 action of the original task plan.
    Retain the format of task_plan.

    """

    return str_0 + str_past_failures + str_past_successes +  str_3


def retune_after_success_gen(parameter_history, execution_time, average_score):

    # List of parameters of where each list element is of the form ((arguments), score)
    past_failed_parameters = []
    past_succesful_parameters = []

    str_0 = f"""
        The action plan was successful with a total time of {execution_time} and an average score of {average_score}.
        Please suggest the retuning of a single action of your choice to impove the score.
        """

    str_past_failures = ""
    if len(parameter_history['failure'].keys()):
        str_past_failures = "Past failures:\n"
        for action_name in parameter_history['failure'].keys():
            for object_name in parameter_history['failure'][action_name].keys():
                failed_parameters = parameter_history['failure'][action_name][object_name]
                for f_param in failed_parameters:
                    f_arg = list(f_param[0])
                    f_score = f_param[1]
                    str_past_failures += f"        FAILURE: {action_name}: {object_name} {tuple(f_arg)} | score = {f_score}\n"

    str_past_successes = ""
    if len(parameter_history['success'].keys()):
        str_past_successes = "Past successes of this action:\n"
        for action_name in parameter_history['success'].keys():
            for object_name in parameter_history['success'][action_name].keys():
                success_parameters = parameter_history['success'][action_name][object_name]
                for s_param in success_parameters:
                    s_arg = list(s_param[0])
                    s_score = s_param[1]
                    str_past_successes += f"        SUCCESS: {action_name}: {object_name} {tuple(s_arg)} | score = {s_score}\n"

    str_3 = f"""

    The score indicates the suitability of a combination. It is a combination of the minimum distance to objects during a movement and how far the robot is from singularities.

    A higher score is better. First explain how the changes your are making will improve the chances of success of the task.

    Then alter the arguments of on single action the index of your choice in task_plan, to increase the score and the total time of execution.

    The plan should still be viable.

    Do not use other actions. Make in-place change in task_plan, while retaining the format of task_plan.

    """

    return str_0 + str_past_failures + str_past_successes + str_3


def retune_gen(history_log, parameter_history, task_plan):
    """Generates the prompt to perform parameter tuning for the failed action"""

    failure_index = history_log[-1][0]
    failure_score = history_log[-1][2]
    failure_reason = history_log[-1][-1]
    action_name, action_args = task_plan[int(failure_index)][1], task_plan[int(failure_index)][2]
    object_label = action_args[0]

    # List of parameters of where each list element is of the form ((arguments), score)
    past_failed_parameters = []
    past_succesful_parameters = []
    str_0 = f"""
    The robot failed during the task_plan's execution. The details are:
    failure index: {failure_index}
    failed action: {action_name}
    current_arguments: {action_args}
    motion score: {failure_score}
    failure reason: {failure_reason}
    """

    str_past_failures = ""
    if action_name in parameter_history['failure'].keys():
        str_past_failures = "Past failures of this action:\n"
        for object_name in parameter_history['failure'][action_name].keys():
            failed_parameters = parameter_history['failure'][action_name][object_name]
            for f_param in failed_parameters:
                f_arg = list(f_param[0])
                f_score = f_param[1]
                str_past_failures += f"        FAILURE: {action_name}: {object_name} {tuple(f_arg)} | score = {f_score}\n"

    str_past_successes = ""
    if action_name in parameter_history['success'].keys():
        str_past_successes = "Past successes of this action:\n"
        for object_name in parameter_history['success'][action_name].keys():
            success_parameters = parameter_history['success'][action_name][object_name]
            for s_param in success_parameters:
                s_arg = list(s_param[0])
                s_score = s_param[1]
                str_past_successes += f"        SUCCESS: {action_name}: {object_name} {tuple(s_arg)} | score = {s_score}\n"

    #Added the "make significant parameter adjustments" part
    str_3 = f"""

    The score indicates the suitability of a combination. A higher score is better. First explain how the changes your are making will improve the chances of success of the task.

    Then alter the arguments of the failed action at index {failure_index} in task_plan, to overcome the failure. Output should be of the for task_plan[i] = (index, action, new parameters), similar to initial task_plan's format

    Make significant parameter adjustments in order to avoid the previous failure mode.

    Do not use other action. Make in-place change in task_plan, while retaining the format of task_plan. 

    """

    return str_0 + str_past_failures + str_past_successes + str_3

