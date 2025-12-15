from llm_planner.chatbots import GptChatBot
import llm_planner.prompt_generator as pg
from llm_planner.utils import create_experiment_log, get_log_folder
from llm_planner.problem_interpreter import ProblemInterpreter
import llm_planner.helper_functions as helper
import pickle
from primitives.execute_task_plan import TaskPlanExecutor
import rospy
from copy import deepcopy
import typing as tp
import os
import numpy as np


# used only when warm_start=True, to SKIP the llm
action_plan_ask = """
    The image shows an indoor scene with a white table at the center, and a few identical rectangular beams either laying down flat on the table, or standing up vertically. The list of recognised objects is:

    objects = ['white table', 'beam_1', 'beam_2', 'beam_3']

    beam_1 is laying flat horizontally on the table. beam_2 to the left of beam_1 and is laying flat horizontally on the table. beam_3 is to the right of beam_1 and is standing up vertically on the table. 

    The list of recognised locations, which are designated areas on the table, is:

    locations = ['left_side', 'center', 'right_side']

    There is a robot, labeled 'robot', that can only manipulate ONE beam at a time. It is here to do beam assembly tasks. The robot accepts commands in the form 'action functions' written in python. These action functions, which can be imported from the 'action_functions' library are:

    approach(object_to_grasp: str, speed: float, obstacle_clearance: float, grasp: str) -> None:  # Moves the robot close to "object_to_grasp" so that the beam is in the robot's reach

    pick(object_to_grasp: str, speed: float, obstacle_clearance: float, grasp: str) -> None: # Instructs the robot to pick up the "object_to_grasp", if it is close enough

    place(location: str, speed: float, obstacle_clearance: float, placement_angle: float = 0., vertical: bool = False) -> None: # Positions the "grasped_object" on/at/in the "location" and release the grasp. It is not advised to use the approach function directly before this one.

    The "speed" argument for 'approach', 'pick', 'drop' and 'place' functions, assumes value in [0,1] and regulates how fast the robot moves. The closer the the value is to 1 the faster the robot moves. moving with higher speed is faster but might result in a jerky and less precise motion.

    The "grasp" argument for 'approach' and 'pick' is mandatory and assumes one of the two values {"top", "side"}, never use None, where "top" instructs the robot to approach or pick the beam from the top and selecting "side" instructs the robot to approach or pick the beam from the side. For the same object, you must use the same grasp for both 'approach' and 'pick' functions. "top" should be used for beams that are laying down, "side" should be used for beams that are standing up.

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

# used only when warm_start=True, to SKIP the llm -> must be for our exact task
# I did this wrong. it's supposed to roleplay as the llm
action_plan_answer = """
To efficiently assemble the beams in a U shape, we should follow a plan that minimizes the amount of robot movement and takes in account the spacial positions the beams in order to order the placements correctly.

1. **Notice the base beam**: Identify which beam will serve as the base for the assembly. This beam could already be placed down on the table, or should be placed first on the table, either at the left_side, center, or right_side location.
2. **Pick and place the first beam**: Choose between the remaining beams which to pick up first. It should then be placed vertically on one of the base beam's ends.
3. **Pick and place the second beam**: Finally, pick up the last beam and place it vertically on the other end of the base beam.

```python
task_plan = [
        (0, 'approach', ('beam_1', 0.8, 0.05, 'top')),
        (1, 'pick', ('beam_1', 0.8, 0.001, 'top')),
        (2, 'place', ('beam_2_end2', 0.0, 0.4, 0.02, 0.0, True)),
        (3, 'approach', ('beam_3', 0.8, 0.05, 'side')),
        (4, 'pick', ('beam_3', 0.8, 0.001, 'top')),
        (5, 'place', ('beam_2_end1', 0.0, 0.4, 0.02, 0.0, True)),
    ]
```

This task plan assumes that:
- The 'pick' function allows the robot to hold the object until the 'place' function is executed.
- The 'place' function can place objects either vertically or horizontally based on the 'vertical' argument, and place the beams with yaws (set to 0 degrees when placing vertically).
- After executing a 'pick', the robot does not have to execute an 'approach' again before 'place' if the object is already held.

The task plan aims for both efficiency in motion and proper handling of the objects.

"""

# used only when warm_start=True, to SKIP the llm -> must be for our exact task
eval_plan_ask = """

    The robot may not be able to execute an action function, or encounter object collision during execution. Thus, it is required to check for completion of each action function after they have been performed.

    For this, we define some 'checking functions' written in python. These checking functions, which can be imported from the 'checking_functions' library are:

    can_grasp(object_to_grasp: str, grasp: str) -> bool: # Returns True if robot is close-enough to the "object_to_grasp" to securely grasp it with the determined grasp "side" or "top"

    holding() -> bool # Returns True if the robot is holding an object

    at_location(object: str, location: str) -> bool: # Returns True if the "object" is at the "location"

    collision_free() -> str: # If the robot encounters a collision while executing the preceding action, returns the object label string. Otherwise, ''.

    timeout() -> Bool: # Returns True if the preceding action was executed in a timely fashion

    check_motion_health() ->  bool: # Returns True if the robot's motion during the preceding action was safe for its hardware

    can_reach(goal: str, grasp: str) -> bool: # Returns True if it is feasible for the robot to reach the "goal" object or location from the current state from the side determined by the grasp argument "side" or "top". Objects that are out of the workspace will always return False.

    beam_contact(beam1: str, beam2: str, tolerance: float = 0.05) -> bool: # Returns True if two beams are touching within the given tolerance (in meters). Default tolerance is 5cm.

    beam_angle(beam1: str, beam2: str, target_angle: float = 90.0, tolerance: float = 5.0) -> bool: # Returns True if the angle between two beams matches the target angle within tolerance (in degrees). Use target_angle=90.0 for perpendicular beams such as in an L or U shape. Default tolerance is ±5°.

    beam_parallel(beam1: str, beam2: str, tolerance: float = 5.0) -> bool: # Returns True if two beams are parallel within tolerance (in degrees). Equivalent to beam_angle with target_angle=0.0. Use for the two vertical beams in a U-shape assembly. Default tolerance is ±5°.

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

# eval plan answer as tuple instead of dict
eval_plan_answer = """
```python
# Example evaluation plan for a U-shape beam assembly task
evaluation_plan = [
    # Check after approaching beam_1
    (0, [('can_grasp', ('beam_1', 'top')), ('collision_free', ()), ('timeout', ()), ('check_motion_health', ())], 
        (True, '', True, True)),

    # Check after picking beam_1
    (1, [('holding', ()), ('collision_free', ()), ('timeout', ()), ('check_motion_health', ())], 
        (True, '', True, True)),

    # Check after placing beam_1 at beam_2_end2
    (2, [('at_location', ('beam_1', 'beam_2_end2')), ('collision_free', ()), ('timeout', ()), ('check_motion_health', ())], 
        (True, '', True, True)),

    # Check after approaching beam_3
    (3, [('can_grasp', ('beam_3', 'top')), ('collision_free', ()), ('timeout', ()), ('check_motion_health', ())], 
        (True, '', True, True)),

    # Check after picking beam_3
    (4, [('holding', ()), ('collision_free', ()), ('timeout', ()), ('check_motion_health', ())], 
        (True, '', True, True)),

    # Final assembly check after placing beam_3 at beam_2_end1
    (5, [('at_location', ('beam_1', 'beam_2_end2')),
         ('at_location', ('beam_3', 'beam_2_end1')),
         ('beam_contact', ('beam_1', 'beam_2', 0.05)),
         ('beam_contact', ('beam_3', 'beam_2', 0.05)),
         ('beam_angle', ('beam_1', 'beam_2', 90.0, 5.0)),
         ('beam_angle', ('beam_3', 'beam_2', 90.0, 5.0)),
         ('beam_parallel', ('beam_1', 'beam_3', 5.0)),
         ('collision_free', ()),
         ('timeout', ()),
         ('check_motion_health', ())],
        (True, True, True, True, True, True, True, '', True, True)),
]

```

"""


warm_start_history = [action_plan_ask, action_plan_answer, eval_plan_ask, eval_plan_answer]


scene_description = (
    "The image shows an indoor scene with a white table at the center. "
    "On the table, there are a few identical rectangular beams. They are either laying flat or standing upright. "
    "beam_1 is laying flat horizontally on the table. beam_2 higher on the table than beam_1 and is laying flat horizontally on the table. beam_3 is on the right side of beam_1, and is standing up vertically on the table"
)


# Labels that are considered objects which a robot may interact with
objects = [
    "white table",
    "beam_1",
    "beam_2",
    "beam_3"
    # "crumpled paper ball 1",
    # "crumpled paper ball 2",
    #"crumpled paper ball 3",
    #"crumpled paper ball 4",
    #"crumpled paper ball 5",
    # "whole apple",
    # "half-eaten apple",
    # "empty glass 1",
    # #"empty glass 2",
    # "glass with yellowish liquid",
    # "large red trash can",
    # "discarded plastic wrapper 1",
    # "discarded plastic wrapper 2",
    # "discarded plastic wrapper 3",
]

# Labels that are considered locations
locations = [
    "left_side",
    "center",
    "right_side",
]

task = "Make a U shape structure using the beams on the table"

iter = 1 # do not touch
retuned_iter = 1 # do not touch
retune_count = 3
replan_count = 7

log_folder = ""


def extract_plan(full_answer: str) -> tp.Tuple[bool, str, tp.List]:

    error_message = ""
    plan = []

    raw_code = ProblemInterpreter.extract_code_from_md(full_answer, ['python', 'python3', 'python2', ''])

    # Check number of code blocks
    if len(raw_code) == 0:
        error_message = "I expected to find python code in your answer but didn't get any. Please regenerate the plan"
    elif len(raw_code) > 1:
        error_message = "I expected to find only one python code in your answer but found more. Please regenerate your answer with a single python code block"

    # Check if the code can run without error
    if len(error_message) == 0:
        try:
            exec(raw_code[0])
        except Exception as e:
            error_message = f"I tried running your code it output the following error {str(e)} could you please correct it"

    # Check if the code has a correct assignment
    if len(error_message) == 0:
        operands = raw_code[0].split("=")
        if len(operands) == 1:
            error_message = "I expected to find an assignment in your code but didn't get any. Please regenerate your answer with a single assignment"
        elif len(operands) > 2:
            error_message = "I expected ot find a single assignment in your code but found multiple. Please regenerate your answer with a single assignment"

    # Assign plan
    if len(error_message) == 0:
        try:
            plan = eval(operands[1])
        except Exception as e:
            error_message = f"I tried running your code it output the following error {str(e)} could you please correct it"

    # Check that plan starts at 0, is in order and has no gaps
    if len(error_message) == 0:
        plan_indices = [task[0] for task in plan]
        if plan_indices[0] != 0:
            error_message = "The first index of the plan does not start at 0. Please regenerate your plan"
        elif not all(plan_indices[i] < plan_indices[i + 1] for i in range(len(plan_indices) - 1)):
            error_message = "The plan is not in order. Please regenerate your plan"
        elif any(plan_indices[i + 1] - plan_indices[i] != 1 for i in range(len(plan_indices) - 1)):
            error_message = "The plan indices are not consecutive. Please regenerate your plan"

    return len(error_message) == 0, error_message, plan

def check_eval_plan_structure(plan):
    """Check if evaluation plan uses lists of tuples instead of dicts"""
    for i, item in enumerate(plan):
        # Check tuple has 3 elements
        if len(item) != 3:
            return False, f"Evaluation plan item {i} must have exactly 3 elements (action_num, predicates, expected_results)"
        
        # Check second element is not a dict
        if isinstance(item[1], dict):
            return False, "ERROR: evaluation_plan uses dictionaries for predicates. Must use lists of tuples instead. Format: (action_num, [('predicate_name', (args,)), ('predicate_name2', (args,))], (expected1, expected2)). Please regenerate your evaluation_plan using lists of tuples, NOT dictionaries."
        
        # Check second element is a list
        if not isinstance(item[1], list):
            return False, f"Evaluation plan item {i}: second element must be a list of tuples, not {type(item[1]).__name__}"
        
        # Check each element in the list is a tuple
        for j, pred in enumerate(item[1]):
            if not isinstance(pred, tuple):
                return False, f"Evaluation plan item {i}, predicate {j}: must be a tuple (predicate_name, args), not {type(pred).__name__}"
            if len(pred) != 2:
                return False, f"Evaluation plan item {i}, predicate {j}: must be (predicate_name, args) with 2 elements"
    
    return True, ""

def extract_retune_action(full_retune_answer: str,
                          previous_task_plan: tp.List) -> tp.Tuple[bool, str, tp.List]:

    error_message = ""
    task_plan = deepcopy(previous_task_plan)

    raw_code = ProblemInterpreter.extract_code_from_md(full_retune_answer, ['python', 'python3', 'python2', ''])

    # Check number of code blocks
    if len(raw_code) == 0:
       error_message = "I expected to find python code in your answer but didn't get any. Please regenerate the plane"
    elif len(raw_code) > 1:
        error_message = "I expected to find only one python code in your answer but found more. Please regenerate your answer with a single python code block"

    # Check if the code can run without error
    if len(error_message) == 0:
        try:
            exec(raw_code[0])
        except Exception as e:
            error_message = f"I tried running your code it output the following error {str(e)} could you please correct it"

    # Check if task plan was modfied
    if len(error_message) == 0:
        task_plan_changed = False
        for task, old_task in zip(task_plan, previous_task_plan):
            if not (task == old_task):
                task_plan_changed = True
                break

        if not task_plan_changed:
            error_message = "After running your code the plan 'task_plan' = You are expected to provide on or more python lines that have the effect of updating the task plan according to the instructions you were give. Pleas regenerate your answer"

    # Check that plan starts at 0, is in order and has no gaps
    if len(error_message) == 0:
        plan_indices = [task[0] for task in task_plan]

        for index in plan_indices:
            if type(index) != int:
                error_message = "After running your code, the plan contains non-integer indices. Please regenerate your answer"
                break

        if len(error_message) == 0:
            if plan_indices[0] != 0:
                error_message = "After running your code, the first index of the plan does not start at 0. Please regenerate your answer"
            elif not all(plan_indices[i] < plan_indices[i + 1] for i in range(len(plan_indices) - 1)):
                error_message = "After running your code, the plan is not in order. Please regenerate your answer"
            elif any(plan_indices[i + 1] - plan_indices[i] != 1 for i in range(len(plan_indices) - 1)):
                error_message = "After running your code, the plan indices are not consecutive. Please regenerate your answer"

    # Reset task plan if error
    if len(error_message) != 0:
        task_plan = deepcopy(previous_task_plan)

    return len(error_message) == 0, error_message, task_plan


def ask_for_task_plan(llm_bot: GptChatBot,
                      task: str,
                      scene_description: str,
                      objects: tp.List[str],
                      locations=tp.List[str]) -> tp.List:

    task_plan_prompt = pg.task_plan_gen(task=task,
                                         prompt0=scene_description,
                                         objects=objects,
                                         locations=locations)
    print("Asking: ", task_plan_prompt)
    task_plan_answer = llm_bot.ask(task_plan_prompt, show_output=True)

    for _ in range(3):
        plan_is_correct, error_message, task_plan = extract_plan(task_plan_answer)

        if plan_is_correct:
            break
        else:
            print("Asking: ", error_message)
            task_plan_answer = llm_bot.ask(error_message, show_output=True)

    if not plan_is_correct:
        raise Exception(f"Unable to generate valid task plan from LLM. The latest error message was {error_message}")

    with open(os.path.join(log_folder, 'task_plan' + str(iter) + '.pkl'), 'wb') as file:
        pickle.dump(f"task_plan = {str(task_plan)}", file)

    return task_plan


def ask_for_evaluation_plan(llm_bot: GptChatBot):

    eval_plan_prompt = pg.eval_plan_gen()
    print("Asking: ", eval_plan_prompt)
    eval_plan_answer = llm_bot.ask(eval_plan_prompt, show_output=True)

    for _ in range(5):
        plan_is_correct, error_message, eval_plan = extract_plan(eval_plan_answer)

        if plan_is_correct:
            # Additional check for evaluation plan structure
            struct_ok, struct_error = check_eval_plan_structure(eval_plan)
            if not struct_ok:
                plan_is_correct = False
                error_message = struct_error
                print("Asking: ", error_message)
                eval_plan_answer = llm_bot.ask(error_message, show_output=True)
            else:
                break
        else:
            print("Asking: ", error_message)
            eval_plan_answer = llm_bot.ask(error_message, show_output=True)

    plan_is_correct, error_message, eval_plan = extract_plan(eval_plan_answer)
    if plan_is_correct:
        # Final structure check
        struct_ok, struct_error = check_eval_plan_structure(eval_plan)
        if not struct_ok:
            raise Exception(f"Unable to generate valid evaluation plan from LLM. The latest error message was {struct_error}")
    elif not plan_is_correct:
        raise Exception(f"Unable to generate valid evaluation plan from LLM. The latest error message was {error_message}")

    with open(os.path.join(log_folder, 'evaluation_plan' + str(iter) + '.pkl'), 'wb') as file:
        pickle.dump(f"task_plan = {str(eval_plan)}", file)

    return eval_plan


def ask_for_action_plan_retune(llm_bot: GptChatBot,
                               history_log,
                               parameter_history,
                               previous_task_plan: tp.List,
                               retune_idx: int) -> tp.List:

    retune_prompt = pg.retune_gen(history_log, parameter_history, previous_task_plan)
    print(f"Asking: {retune_prompt}")
    retune_answer = llm_bot.ask(retune_prompt, show_output=True)

    for _ in range(5):
        plan_is_correct, error_message, retuned_plan = extract_retune_action(retune_answer, previous_task_plan)

        if plan_is_correct:
            break
        else:
            print("Asking: ", error_message)
            retune_answer = llm_bot.ask(error_message, show_output=True)
    plan_is_correct, error_message, retuned_plan = extract_retune_action(retune_answer, previous_task_plan)

    if not plan_is_correct:
        raise Exception(f"Unable to generate valid retune plan from LLM. The latest error message was {error_message}")
    global retuned_iter
    with open(os.path.join(log_folder, 'retuned_plan' + str(iter) + "_" + str(retuned_iter) + '.pkl'), 'wb') as file:
        pickle.dump(f"task_plan = {str(retuned_plan)}", file)


    retuned_iter += 1

    return retuned_plan


def ask_for_action_replanned(llm_bot: GptChatBot, history_log, previous_task_plan, parameter_history):

    new_plan_prompt = pg.replan_correction_gen(history_log, previous_task_plan, parameter_history)
    print("Asking: ", new_plan_prompt)
    new_plan_answer = llm_bot.ask(new_plan_prompt, show_output=True)

    for _ in range(3):
        plan_is_correct, error_message, task_plan = extract_plan(new_plan_answer)

        if plan_is_correct:
            # Check that the has the expected number of indices
            if np.absolute(len(task_plan) - len(previous_task_plan)) > 5:
                error_message = "We expected the new plan to be withing 5 actions of the old plan. After running your code it is not. Please regenerate your answer"
            else:
                break
        new_plan_answer = llm_bot.ask(error_message, show_output=True)

    plan_is_correct, error_message, task_plan = extract_plan(new_plan_answer)
    if not plan_is_correct:
        raise Exception(f"Unable to generate valid task plan from LLM. The latest error message was {error_message}")

    global iter
    iter += 1
    global retuned_iter
    retuned_iter = 1

    with open(os.path.join(log_folder, 'task_plan' + str(iter) + '.pkl'), 'wb') as file:
        pickle.dump(f"task_plan = {str(task_plan)}", file)

    return task_plan


def execute_and_log_plans(task_plan: tp.List, evaluation_plan: tp.List, tpu: TaskPlanExecutor):
    # Log of what happened
    task_success, evaluation_log, failure_id = tpu.execute_task_plan(task_plan, evaluation_plan)
    print("evaluation_log")
    print(evaluation_log)

    # A detailed log
    history_log = helper.performance_logger(evaluation_log, evaluation_plan)
    print("history_log")
    print(history_log)

    return task_success, history_log, failure_id


def plan_and_retune(parameter_history, domain_history, warm_start=False):
    global log_folder
    log_folder = create_experiment_log()
    chat_log_file = os.path.join(log_folder, 'chat_log.pckl')
    task_plan_meta_data_file = os.path.join(log_folder, 'task_plan_log.pkl')
    llm_bot = GptChatBot(auto_save_file_name=chat_log_file)
    tpu = TaskPlanExecutor(metadata_file=task_plan_meta_data_file)

    if warm_start:
        llm_bot.set_history(warm_start_history)
        llm_bot.print_history()
        _, _, task_plan = extract_plan(action_plan_answer)  # We know this to be correct
        _, _, evaluation_plan = extract_plan(eval_plan_answer)  # We know this to be correct
    else:
        task_plan = ask_for_task_plan(llm_bot, task, scene_description, objects, locations)
        evaluation_plan = ask_for_evaluation_plan(llm_bot)

    # TODO: Check task plan content
    # TODO: Check evaluation plan content
    # TODO: Check task-evaluation congruence

    # Test plan
    print("=============== EXECUTING INITIAL PLAN =================")
    task_success, history_log, failure_id = execute_and_log_plans(task_plan, evaluation_plan, tpu)

    # Update explored space
    parameter_history = helper.explored_parameter_space(task_plan, history_log, parameter_history)

    for i_replan in range(replan_count):

        if task_success:
            print("=============== TASK SUCCEEDED =================")
            break

        else:

            # Retuning plan
            i_retune = 0
            while i_retune < retune_count:
                # Successful plan: finish
                if task_success:
                    print("=============== TASK SUCCEEDED =================")
                    break

                # Retune plan
                else:
                    task_plan = ask_for_action_plan_retune(llm_bot, history_log, parameter_history, task_plan, i_retune)

                print(f"=============== EXECUTING RETUNED PLAN {i_retune} =================")
                # Test plan
                task_success, history_log, new_failure_id = execute_and_log_plans(task_plan, evaluation_plan, tpu)

                if new_failure_id <= failure_id:
                    i_retune += 1
                else:
                    failure_id = new_failure_id
                    # i_retune += 1

                # Update explored space
                parameter_history = helper.explored_parameter_space(task_plan, history_log, parameter_history)

            # Replan if needed
            if not task_success:
                print("=============== TASK RETUNING FAILED =================")

                task_plan = ask_for_action_replanned(llm_bot, history_log, task_plan, parameter_history)
                evaluation_plan = ask_for_evaluation_plan(llm_bot)
                print(f"=============== EXECUTING REPLANNED PLAN {i_replan} =================")
                task_success, history_log, failure_id = execute_and_log_plans(task_plan, evaluation_plan, tpu)

    # Save parameter_history
    with open(os.path.join(log_folder, 'parameter_history.pkl'), 'wb') as file:
        pickle.dump(parameter_history, file)

    # Logging the domain information
    domain_history = helper.explored_domain(task_plan=task_plan,
                                            evaluation_plan=evaluation_plan,
                                            success_state=task_success,
                                            domain_history=domain_history)

    if not task_success:
        print("=====TOTAL_FAILURE============")
    return task_success, parameter_history, domain_history


if __name__ == "__main__":
    rospy.init_node("llm_planner")
    parameter_history = parameter_history = {'success': {}, 'failure': {}}
    domain_history = None
    success, parameter_history, domain_history =  plan_and_retune(parameter_history, domain_history, warm_start=False)
    # success, parameter_history, domain_history =  plan_and_retune(parameter_history, domain_history, warm_start=True) # Originally warm start is True
