"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import numpy as np

from JSDF import iiwa_JSDF
import open3d as o3d


# import open3d.visualization as vis


def show_mesh_file(mesh, file_name="test.jpg"):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera.json")
    camera_params.extrinsic = parameters.extrinsic
    ctr.convert_from_pinhole_camera_parameters(camera_params)

    render = vis.get_render_option()
    render.background_color = [1, 1, 1]
    render.light_on = True
    # render.show_coordinate_frame = True

    vis.run()
    vis.capture_screen_image(file_name, do_render=True)
    vis.destroy_window()


def save_gt_image(mesh, file_name):
    o3d.visualization.gui.Application.instance.initialize()
    o3d_vis = o3d.visualization.O3DVisualizer("03DVisualizer", 1200, 1200)
    o3d_vis.show_ground = False
    o3d_vis.ground_plane = o3d.visualization.rendering.Scene.GroundPlane.XY
    o3d_vis.show_skybox(False)

    bg_color = np.array([[1, 1, 1, 1]], dtype=np.float32).T
    o3d_vis.set_background(bg_color, None)

    # o3d.visualization.gui.Application.instance.add_window(vis)
    # vis.set_background(np.array([1, 1, 1, 1]))

    mat_robot = o3d.visualization.rendering.MaterialRecord()
    mat_robot.shader = 'defaultLit'
    mat_robot.base_color = [0.7, 0, 0, 1.0]
    o3d_vis.add_geometry({'name': 'gt', 'geometry': mesh, 'material': mat_robot})

    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2.json")
    o3d_vis.setup_camera(parameters.intrinsic, parameters.extrinsic)

    app = o3d.visualization.gui.Application.instance
    app.add_window(o3d_vis)

    app.run_one_tick()
    o3d_vis.export_current_image(file_name)
    app.run_one_tick()
    o3d_vis.close()
    # app.quit()


def save_RDF_image(meshes, file_name):
    o3d.visualization.gui.Application.instance.initialize()
    o3d_vis = o3d.visualization.O3DVisualizer("03DVisualizer", 1200, 1200)
    o3d_vis.show_ground = False
    o3d_vis.ground_plane = o3d.visualization.rendering.Scene.GroundPlane.XY
    o3d_vis.show_skybox(False)
    # o3d_vis.scene_shader = 'NORMALS'
    bg_color = np.array([[1, 1, 1, 1]], dtype=np.float32).T
    o3d_vis.set_background(bg_color, None)

    # o3d.visualization.gui.Application.instance.add_window(vis)
    # vis.set_background(np.array([1, 1, 1, 1]))

    for mesh in meshes:
        o3d_vis.add_geometry(mesh)

    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2.json")
    o3d_vis.setup_camera(parameters.intrinsic, parameters.extrinsic)
    # o3d_vis.enable_raw_mode(True)
    app = o3d.visualization.gui.Application.instance

    app.add_window(o3d_vis)
    # app.run()

    app.run_one_tick()
    o3d_vis.export_current_image(file_name)
    app.run_one_tick()
    o3d_vis.close()
    app.quit()
    time.sleep(1)


if __name__ == "__main__":
    import time

    jsdf = iiwa_JSDF(grid_size=[90] * 3)

    # sample random robot configuration

    rand_q = [0, 0.6, 0., -1.8, 0., 0.6, 0.]
    # rand_q = [0.2, 0.7, 0.1, -1.7, 0.1, 0.5, 0.1]
    # rand_q = [0.4, 0.8, 0.2, -1.6, 0.2, 0.4, 0.2]
    # rand_q = [0.6, 0.9, 0.3, -1.5, 0.3, 0.3, 0.3]
    # rand_q = [0.8, 1., 0.4, -1.4, 0.4, 0.2, 0.4]
    # rand_q = [1.0, 1., 0.4, -1.4, 0.4, 0.2, 0.4]

    # rand_q = [2.15936294, -1.65379741, -1.28514611, -0.00955869,  1.81417312, -0.10412193, -2.99875333]
    # rand_q = [-1.2363863 , 0.69413376 , 0.52264738 , 1.86259791,  1.8230936, 0.76618751, 0.00639464]
    # rand_q = [-1.88203963  ,1.37810102 ,-0.95431457 , 1.15501165,  1.24451764,  1.70181668, 0.56685588]
    # rand_q = [-0.94401467, 0.84032803, -1.54750914, 1.63287046, -1.87048655, 1.58634666, 0.07308435]
    # rand_q = [0.29427962, -0.92049783, 1.4322318, -0.96705457, -1.25757282, -0.98027085, -2.71405221]
    # rand_q = [ 1.34263949, -0.90293251, -2.35626988, -1.95700772, 1.28003971, -1.75097503, 0.27926656]
    print(jsdf.sample_random_robot_config())

    angles = np.linspace(0, 1.57, 120)

    # for i in range(0, 120):
    i = 2
    # rand_q[0] = angles[i]
    #
    # # set the robot into sampled configuration
    jsdf.set_robot_joint_positions(rand_q)
    # # jsdf.show_robot()
    #         robot_gt = jsdf.robot.get_combined_mesh(convex=False, bounding_box=False).as_open3d
    #         robot_gt.compute_vertex_normals()
    #         save_gt_image(robot_gt, file_name='animation/gt/{}.jpg'.format(i))
    # time.sleep(1)
    meshes = jsdf.show_hierarchical_distance_value(q=rand_q, alpha_level=[1, 0.5], value=[-0.004, -0.1],
                                                   color=[0.1, 0.8, 0., 0])
    save_RDF_image(meshes, file_name='paper/mk/{}.jpg'.format(i))
    # robot_gt = jsdf.robot.get_combined_mesh(convex=False, bounding_box=False).as_open3d
    # robot_gt.compute_vertex_normals()
    # save_gt_image(robot_gt, file_name='paper/gt/{}.jpg'.format(i))

    # i = 16
    # rand_q[0] = angles[i]
    # meshes = jsdf.show_hierarchical_distance_value(q=rand_q, alpha_level=[1, 0.5], value=[-0.006, -0.05])
    # save_RDF_image(meshes, file_name='animation/pred/a128_{}.jpg'.format(200))
    # time.sleep(0.5)
    #
    # save_gt_image(robot_gt, file_name='animation/gt/{}.jpg'.format(i))
    # meshes = jsdf.show_hierarchical_distance_value(q=rand_q)
    # save_RDF_image(meshes, "test_1.jpg")
    # for i in range(240):
    #
    #     rand_q[0] = angles[i]
    #
    # # set the robot into sampled configuration
    #     jsdf.set_robot_joint_positions(rand_q)
    # # jsdf.show_robot()
    #     robot_gt = jsdf.robot.get_combined_mesh(convex=False, bounding_box=False).as_open3d
    #     robot_gt.compute_vertex_normals()
    #     save_gt_image(robot_gt, file_name='animation/gt/{}.jpg'.format(i))

    # jsdf.show_hierarchical_distance_value(q=rand_q)
