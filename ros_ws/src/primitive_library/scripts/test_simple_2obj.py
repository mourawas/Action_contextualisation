import rospy
from primitives.execute_task_plan import TaskPlanExecutor
from llm_planner.utils import create_experiment_log
import os

# SIMPLE 2-OBJECT PICK AND PLACE TASK
# task: Move champagne glass to sink, then move apple to trash

task_plan = [
    # obj 1: Move empty glass to sink
    (0, 'approach', ['empty glass 1', 1, 0.05, 'top']), #speed, precision, grasp_type
    (1, 'pick', ['empty glass 1', 1, 0.02, 'top']),
    (2, 'place', ['large white sink', 0.5, 1, 0.03]), #place(location, orientation, speed, obstacle_clearance)

    # obj 2: Move apple to trash
    (3, 'approach', ['whole apple', 1, 0.06, 'top']),
    (4, 'pick', ['whole apple', 1, 0.03, 'top']),
    (5, 'drop', ['large red trash can', 1, 0.05])
]

evaluation_plan = [
    (0, {'can_grasp': ('empty glass 1', 'top'), 'collision_free': (), 'timeout': ()}, (True, '', True)),
    (1, {'holding': (), 'collision_free': (), 'timeout': ()}, (True, True, True)),
    (2, {'at_location': ('empty glass 1', 'large white sink'), 'collision_free': ()}, (True, True)),
    (3, {'can_grasp': ('whole apple', 'top'), 'collision_free': (), 'timeout': ()}, (True, '', True)),
    (4, {'holding': (), 'collision_free': (), 'timeout': ()}, (True, True, True)),
    (5, {'at_location': ('whole apple', 'large red trash can'), 'collision_free': ()}, (True, True))
]


def main():
    rospy.init_node("test_simple_2_object")
    log_folder = create_experiment_log('simple_2_object_test')
    task_plan_meta_data_file = os.path.join(log_folder, 'task_plan_log.pkl')

    print("=== Starting 2-Object Pick & Place Test ===")
    print("Task: Glass to sink, Apple to trash")

    tpu = TaskPlanExecutor(metadata_file=task_plan_meta_data_file)
    tpu.execute_task_plan(task_plan, evaluation_plan)
    
    print("=== Task Complete ===")

if __name__ == "__main__":
    main()