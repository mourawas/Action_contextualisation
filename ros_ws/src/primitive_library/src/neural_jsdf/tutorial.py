"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
from JSDF import iiwa_JSDF
import numpy as np

if __name__ == "__main__":
    # initialize our joint space distance function
    jsdf = iiwa_JSDF(grid_size=[110] * 3)

    # sample random robot configuration
    rand_q = jsdf.sample_random_robot_config()
    print(rand_q)
    rand_q = [0.25909432, 0.47715488, -0.7506828, -1.7942765, 0.22864909, 1.78214122, 2.73635163]
    # rand_q = [-0.61853585,  0.95173387, -2.66000573,  0.9582079,   2.14836581,  1.37605599, -0.68319343]

    # set the robot into sampled configuration
    jsdf.set_robot_joint_positions(rand_q)

    # show predicted implicit surface using Voxel Grid
    # jsdf.show_voxel_grid(with_robot=False)

    # show implicit distance function
    # jsdf.show_implicit_surface(with_robot=True, value=-0.0005, material=mat)
    # jsdf.show_hierarchical_distance_value(q=rand_q)

    # show the robot
    # jsdf.show_robot()

    # predict the minimum distance of single point
    position = [0, 0, 1]
    jsdf.show_robot_with_points(position)
    print("Minimum distance to the robot is {}".format(jsdf.calculate_signed_distance(position, min_dist=True)))
    print("Minimum distance to each link are {}".format(jsdf.calculate_signed_distance(position, min_dist=False)))
    print("Calculate gradient wrt current joint configuration + input position {}".format(
        jsdf.calculate_gradient(position).shape))

    # multiple points prediction
    positions = [[0, 0, 1], [0.5, 0, 1], [0, 0.5, 1]]
    jsdf.show_robot_with_points(positions)
    print("Batch minimum distance to the robot are {}".format(
        jsdf.calculate_signed_distance(positions, min_dist=True)))
    print("Batch minimum distance to each link are {}".format(
        jsdf.calculate_signed_distance(positions, min_dist=False)))
