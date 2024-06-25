import rospy
from primitives.execute_task_plan import TaskPlanExecutor

# Assigning the full task plan as a Python list of tuples. Each tuple represents an action with parameters.

task_plan = task_plan = [
    # Replanning: Approach the crumpled paper ball from the top this time with moderate speed and adjusted obstacle clearance
    (0, 'approach', ('crumpled paper ball 1', 0.5, 0.075, 'top')),
    # Pick up the crumpled paper ball from the top
    (1, 'pick', ('crumpled paper ball 1', 0.5, 0.01, 'top')),
    # Drop the crumpled paper ball into the trash can without concern for orientation
    (2, 'drop', ('large red trash can', 1., 0.05))
]



evaluation_plan = [
    (0, {'can_grasp': ('crumpled paper ball 1', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
    (1, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True)),
    (2, {'at_location': ('crumpled paper ball 1', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()}, (True, '', True, True))
]






def main():
    rospy.init_node("test_task_plan")
    tpu = TaskPlanExecutor()
    tpu.execute_task_plan(task_plan, evaluation_plan)


if __name__ == "__main__":
    main()