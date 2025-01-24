import rospy
from primitives.execute_task_plan import TaskPlanExecutor

# Assigning the full task plan as a Python list of tuples. Each tuple represents an action with parameters.

task_plan = [
    # Approach and pick the first crumpled paper ball, then drop it in the large red trash can
    (0, 'approach', ('crumpled paper ball 1', 0.5, 0.05, 'top')),
    (1, 'pick', ('crumpled paper ball 1', 0.3, 0.02, 'top')),
    (2, 'drop', ('large red trash can', 0.5, 0.05)),
    
    # Approach and pick the second crumpled paper ball, then drop it in the trash can
    (3, 'approach', ('crumpled paper ball 2', 0.5, 0.05, 'top')),
    (4, 'pick', ('crumpled paper ball 2', 0.3, 0.01, 'top')),
    (5, 'drop', ('large red trash can', 0.5, 0.05)),
    
    # Pick up the whole apple requiring top grasp and place it on the storage shelf
    (6, 'approach', ('whole apple', 0.5, 0.05, 'top')),
    (7, 'pick', ('whole apple', 0.3, 0.01, 'top')),
    (8, 'place', ('storage shelf', 0.5, 0.5, 0.05)),
    
    # Pick up the empty glass requiring side grasp and place it in the sink
    (9, 'approach', ('empty glass 1', 0.5, 0.05, 'side')),
    (10, 'pick', ('empty glass 1', 0.3, 0.01, 'side')),
    (11, 'place', ('large white sink', 0.5, 0.5, 0.05)),
    
    # Again attempt to pick the glass with yellowish liquid by adjusting the parameters
    (12, 'approach', ('glass with yellowish liquid', 0.5, 0.05, 'side')), # Decrease obstacle clearance
    (13, 'pick', ('glass with yellowish liquid', 0.3, 0.02, 'side')), # Increase chance of avoiding collision
    (14, 'place', ('large white sink', 0.5, 0.5, 0.05)), # Place the glass into the sink if picked
    
    # Tackle the half-eaten apple last, with suitable parameters for top grasp
    (15, 'approach', ('half-eaten apple', 0.5, 0.05, 'top')),
    (16, 'pick', ('half-eaten apple', 0.3, 0.02, 'top')),
    (17, 'drop', ('large red trash can', 0.5, 0.05))
]

evaluation_plan = [
    # Checks after approaching the first crumpled paper ball
    (0, {'can_grasp': ('crumpled paper ball 1', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after picking the first crumpled paper ball 
    (1, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after dropping the first crumpled paper ball into the trash can
    (2, {'at_location': ('crumpled paper ball 1', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    
    # Checks after approaching the second crumpled paper ball
    (3, {'can_grasp': ('crumpled paper ball 2', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after picking the second crumpled paper ball 
    (4, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after dropping the second crumpled paper ball into the trash can
    (5, {'at_location': ('crumpled paper ball 2', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    
    # Checks after approaching the whole apple
    (6, {'can_grasp': ('whole apple', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after picking the whole apple
    (7, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after placing the whole apple on the storage shelf
    (8, {'at_location': ('whole apple', 'storage shelf'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    
    # Checks after approaching the empty glass
    (9, {'can_grasp': ('empty glass 1', 'side'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after picking the empty glass
    (10, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after placing the empty glass in the sink
    (11, {'at_location': ('empty glass 1', 'large white sink'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    
    # Checks after approaching the glass with yellowish liquid
    (12, {'can_grasp': ('glass with yellowish liquid', 'side'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after picking the glass with yellowish liquid
    (13, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after placing the glass with yellowish liquid in the sink
    (14, {'at_location': ('glass with yellowish liquid', 'large white sink'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    
    # Checks after approaching the half-eaten apple
    (15, {'can_grasp': ('half-eaten apple', 'top'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after picking the half-eaten apple
    (16, {'holding': (), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True)),
    # Checks after dropping the half-eaten apple into the trash can
    (17, {'at_location': ('half-eaten apple', 'large red trash can'), 'collision_free': (), 'timeout': (), 'check_motion_health': ()},
        (True, '', True, True))
]






def main():
    rospy.init_node("test_task_plan")
    tpu = TaskPlanExecutor()
    tpu.execute_task_plan(task_plan, evaluation_plan)


if __name__ == "__main__":
    main()