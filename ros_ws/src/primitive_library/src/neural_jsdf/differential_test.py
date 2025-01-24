"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import numpy as np
import torch
import matplotlib.pylab as plt
from utils import load_object_pcd
import open3d as o3d
from JSDF import opt_JSDF, opt_arm_hand_JSDF
from scipy.optimize import minimize
import time

if __name__ == "__main__":
    jsdf = opt_arm_hand_JSDF()
    position = [-0.4, 0.1, 0.45]
    obj = load_object_pcd('objects/plum.ply', every_k=10, position=position)
    obj_points = np.asarray(obj.points)
    jsdf.arm_hand_robot.set_system_config([0.1] * 10, hand_synergy=True)
    # jsdf.arm_hand_robot.set_system_config([0.57981086, -1.53783544, -1.88542572, 1.62025416, 2.96705973,
    #                                        -1.92032181, 0.0544818, 0.1, 0.1, 0.1], hand_synergy=True)

    robot_mesh = jsdf.arm_hand_robot.get_system_combined_mesh()
    robot_mesh = robot_mesh.as_open3d
    robot_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([obj, robot_mesh])

    upper_bounds = jsdf.arm_hand_robot.arm.joint_upper_bound
    lower_bounds = jsdf.arm_hand_robot.arm.joint_lower_bound
    bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(7)]
    hand_bounds = [(-0.196, 1.61), (-0.174, 1.709), (-0.227, 1.618)]
    bounds = bounds + hand_bounds

    links = [4, 8]


    def objective_func(robot_q):
        return jsdf.calculate_squared_dist_hand_link(robot_q, link_index=links, position=obj_points)


    def gradient_func(robot_q):
        gradient = jsdf.calculate_gradient_dist_hand_link(robot_q, link_index=links, position=obj_points)

        return gradient.cpu().detach().numpy()


    start_time = time.time()
    res = minimize(
        objective_func,
        method="SLSQP",
        x0=np.array([0.1] * 10),
        jac=gradient_func,
        bounds=bounds,
        tol=0.00001
    )
    end_time = time.time()

    print(res)
    print(end_time - start_time)
    print(res.x)
    jsdf.arm_hand_robot.set_system_config(list(res.x))
    robot_mesh = jsdf.arm_hand_robot.get_system_combined_mesh()
    robot_mesh = robot_mesh.as_open3d
    robot_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([obj, robot_mesh])
