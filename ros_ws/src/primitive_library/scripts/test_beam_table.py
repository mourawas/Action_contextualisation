import rospy
from primitives.execute_task_plan import TaskPlanExecutor
from llm_planner.utils import create_experiment_log
import os

# test table positions here
# only with beam 1

def main() -> None:
    print("Starting beam table test")
    rospy.init_node("test_beam_table")
    print("Node initialized")
    
    # Define task plan for U-shape assembly
    task_plan = [
        (0, 'approach', ('beam_1', 0.8, 0.05, 'top')),
        (1, 'pick', ('beam_1', 0.8, 0.001, 'top')),
        (2, 'place', ('left_side', 0.0, 0.4, 0.02, 0.0, False))
    ]
    
    # Define evaluation plan
    evaluation_plan = [
    (0, [('can_grasp', ('beam_1', 'top')), 
         ('collision_free', ()), 
         ('timeout', ())], 
        (True, '', True)),
    (1, [('holding', ()), 
         ('collision_free', ()), 
         ('timeout', ())], 
        (True, '', True)),
    (2, [('at_location', ('beam_1', 'left_side')),
         ('collision_free', ()), 
         ('timeout', ())], 
        (True, '', True)) #expected results
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
        print("\n✓ TABLE PLACING SUCCESSFUL")
    else:
        print(f"\n✗ Table placing failed at action {failure_id}")
    
    rospy.signal_shutdown("Beam table test finished")


if __name__ == "__main__":
    main()