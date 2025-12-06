import rospy
from primitives.execute_task_plan import TaskPlanExecutor
from llm_planner.utils import create_experiment_log
import os

def main() -> None:
    print("Starting beam predicate test")
    rospy.init_node("test_beam_predicates")
    print("Node initialized")
    
    # Define task plan for L-shape assembly
    task_plan = [
        (0, 'approach', ('beam_1', 0.8, 0.05, 'top')),
        (1, 'pick', ('beam_1', 0.8, 0.001, 'top')),
        (2, 'place', ('beam_2_end2', 0.0, 0.4, 0.02, 0.0, True)),
        (3, 'approach', ('beam_3', 0.8, 0.05, 'side')),
        (4, 'pick', ('beam_3', 0.8, 0.001, 'top')),
        (5, 'place', ('beam_2_end1', 0.0, 0.4, 0.02, 0.0, True)),
    ]
    
    # Define evaluation plan
    evaluation_plan = [
        (0, {'can_grasp': ('beam_1', 'top'), 'collision_free': (), 'timeout': ()}, 
            (True, '', True)),
        (1, {'holding': (), 'collision_free': (), 'timeout': ()}, 
            (True, '', True)),
        (2, {'at_location': ('beam_1', 'beam_2_end2'), 'collision_free': (), 'timeout': ()}, 
            (True, '', True)),
        (3, {'can_grasp': ('beam_3', 'top'), 'collision_free': (), 'timeout': ()}, 
            (True, '', True)),
        (4, {'holding': (), 'collision_free': (), 'timeout': ()}, 
            (True, '', True)),
        (5, {'at_location': ('beam_1', 'beam_2_end2'),
             'at_location': ('beam_3', 'beam_2_end1'),
             'beam_contact': ('beam_1', 'beam_2', 0.05),
             'beam_contact': ('beam_3', 'beam_2', 0.05),
             'beam_angle': ('beam_1', 'beam_2', 90.0, 5.0),
             'beam_angle': ('beam_3', 'beam_2', 90.0, 5.0),
             'beam_parallel': ('beam_1', 'beam_3', 5.0),
             'collision_free': (), 
             'timeout': ()}, 
            (True, True, True, '', True)),
    ]
    
    print("\n" + "="*60)
    print("EXECUTING PLAN:")
    print("="*60)
    
    # Create executor and run
    tpe = TaskPlanExecutor()
    success, history_log, failure_id = tpe.execute_task_plan(task_plan, evaluation_plan)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Task success: {success}")
    print(f"Failure ID: {failure_id}")
    
    if success:
        print("\n✓ L-SHAPE ASSEMBLY SUCCESSFUL!")
    else:
        print(f"\n✗ L-shape assembly failed at action {failure_id}")
    
    rospy.signal_shutdown("Beam predicate test finished")


if __name__ == "__main__":
    main()