"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import copy
import os
import sys

sys.path.append('../')
import torch
import numpy as np
from neural_jsdf.utils import robot_kinematic
import open3d as o3d
import trimesh
from skimage.measure import marching_cubes
import open3d.visualization as vis
from torch.autograd.functional import jacobian


class iiwa_JSDF:
    def __init__(self,
                 grid_size=None,
                 lower_bound=None,
                 upper_bound=None,
                 use_GPU=False,
                 model_pkl='SOTA_model/64_multi_res.pkl',
                 # model_pkl='SOTA_model/128_res_multi_15.pkl',
                 # model_pkl='SOTA_model/64_res_multi_18.pkl',
                 # model_pkl='SOTA_model/64_multi_head.pkl',
                 # model_pkl='SOTA_model/mk_mlp_3.pkl',
                 enable_gradient=True,
                 generate_grid=False) -> None:

        self.enable_gradient = enable_gradient
        self.model = None

        # marching cube
        if lower_bound is None:
            lower_bound = [-0.3, -0.2, 0]
        if upper_bound is None:
            upper_bound = [1.2, 0.2, 1.2]
        if grid_size is None:
            grid_size = [150, 40, 120]

        assert len(lower_bound) == 3, "Bound for x, y, z three dimension"
        assert len(upper_bound) == 3, "Bound for x, y, z three dimension"
        assert len(grid_size) == 3, "Grid size should be in type int"

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.grid_size = grid_size
        self.spacing = tuple((np.array(self.upper_bound) - np.array(self.lower_bound)) / np.array(grid_size))

        # initialize model
        print("Loading neural network models ...")
        self.device = "cuda:0" if (torch.cuda.is_available() and use_GPU) else "cpu"
        self.model = torch.load(self._get_file_path(model_pkl)).to(self.device)
        self.set_requires_grad(self.enable_gradient)

        print("Initializing Robot Model ...")
        self.robot = robot_kinematic()

        if generate_grid:
            self.test_grid = self.generate_grid()
        else:
            self.test_grid = None

    def _get_file_path(self, filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    def set_requires_grad(self, requires_grad=True):
        assert self.model is not None, "Model initialization failed."
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def generate_grid(self) -> np.ndarray:
        print("Generating test grid ...")
        axis_x = np.linspace(self.lower_bound[0], self.upper_bound[0], self.grid_size[0])
        axis_y = np.linspace(self.lower_bound[1], self.upper_bound[1], self.grid_size[1])
        axis_z = np.linspace(self.lower_bound[2], self.upper_bound[2], self.grid_size[2])

        grid = []
        for x in np.nditer(axis_x):
            for y in np.nditer(axis_y):
                for z in np.nditer(axis_z):
                    grid.append([x, y, z])
        print("Done.")

        return np.asarray(grid)

    def calculate_signed_distance(self, position: list, min_dist=True) -> np.ndarray:
        """ get implicit function values for batch input positions
        :param min_dist: return minimum distance to the whole robot, corresponding to the maximum signed distance
        :param position: (n, 3) or (3, )
        :return: pred values (n, )
        """
        assert self.model is not None
        position = np.array(position)
        if position.shape == (3,):
            position = np.expand_dims(position, axis=0)

        data_len = position.shape[0]

        q = torch.from_numpy(np.array([self.robot.robot_q]))
        nq = q.repeat(data_len, 1)
        qp_tensor = torch.concat((nq, torch.from_numpy(position)), dim=1).type(torch.float32).to(self.device)

        pred = self.model_inference_with_gradient(qp_tensor)
        pred = pred.cpu().detach().numpy()

        if not min_dist:
            return pred
        else:
            max_index = pred.argmax(axis=1)
            pred_min_dist = pred[np.arange(data_len), max_index]
            return pred_min_dist

    def model_inference_with_gradient(self, qp) -> torch.tensor:
        """
        Just data in&out
        :param qp: torch tensor of shape (n, 10)
        :return: pred torch tensor of shape (n, 8)
        """
        assert qp.shape[1] == 10, "The input should be 7 (arm config) + 3 (x y z)"
        # with torch.no_grad():
        pred = self.model(qp)
        return pred

    def calculate_gradient(self, position: list):
        """ get gradient wrt input of the implicit function for batch input joint configuration and positions
        :param position: (n, 3) or (3, )
        :return: pred values (n, 10)
        """
        assert self.model is not None
        position = np.array(position)
        if position.shape == (3,):
            position = np.expand_dims(position, axis=0)

        data_len = position.shape[0]

        q = torch.from_numpy(np.array([self.robot.robot_q])).to(self.device)
        nq = q.repeat(data_len, 1).to(self.device)

        qp_tensor = torch.concat((nq, torch.from_numpy(position)), dim=1).type(torch.float32)
        qp_tensor.requires_grad = True

        gradient = jacobian(self.model, qp_tensor)
        print(gradient.shape)
        gradient = torch.sum(gradient, dim=2)

        return gradient

    def np2o3d_pcd(self, points, color=None) -> o3d.geometry.PointCloud:
        if color is None:
            color = [0., 0, 0.6]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)

        return pcd

    def get_rendering_material(self,
                               # shader='defaultLitSSR',
                               shader='defaultLitTransparency',
                               base_color=None,
                               base_roughness=1.0,
                               base_reflectance=1.0,
                               base_clearcoat=1.0,
                               thickness=1.0,
                               transmission=0.0,
                               absorption_distance=10,
                               absorption_color=None):
        if absorption_color is None:
            absorption_color = [0.5, 0.5, 0.5]
        if base_color is None:
            base_color = [0.467, 0.467, 0.467, 0.3]

        mat = o3d.visualization.rendering.MaterialRecord()
        # mat_box.shader = 'defaultLitTransparency'
        mat.shader = shader
        mat.base_color = base_color
        mat.base_roughness = base_roughness
        mat.base_reflectance = base_reflectance
        mat.base_clearcoat = base_clearcoat
        mat.thickness = thickness
        mat.transmission = transmission
        mat.absorption_distance = absorption_distance
        mat.absorption_color = absorption_color

        return mat

    def show_hierarchical_distance_value(self, q=None, with_robot=False, value=None, color=None, alpha_level=None):

        if alpha_level is None:
            alpha_level = [1, 0.6, 0.4]

        if q is None:
            q = self.robot.sample_random_robot_config()
            self.robot.set_robot_joints(q)
        else:
            self.robot.set_robot_joints(q)

        if color is None:
            color = [0, 0, 0.6, 0]
        if value is None:
            value = [-0.006, -0.04, -0.1]
        assert len(value) > 1, "This is for multiple distance visualization"

        mat_materials = []
        pred_meshes = []

        for i, distance in enumerate(value):
            pred_meshes.append(self.get_implicit_surface(q, value=distance))
            print(color)
            color[3] = alpha_level[i]
            mat = self.get_rendering_material(base_color=color)
            if i == 0:
                mat.shader = 'defaultLit'
                mat.base_clearcoat = 0
            if i > 0:
                mat.base_roughness = 0.
                mat.base_reflectance = 0.
                mat.thickness = 0.
                mat.base_clearcoat = 0.5
            mat_materials.append(mat)

        robots = []
        for i, mesh in enumerate(pred_meshes):
            robots.append({'name': 'pred_{}'.format(i), 'geometry': mesh, 'material': mat_materials[i]})

        if with_robot:
            robot_mesh = self.robot.get_combined_mesh(convex=False, bounding_box=False)
            robot_mesh = robot_mesh.as_open3d
            robot_mesh.compute_vertex_normals()
            robot_mesh.translate([1, 0, 0])
            mat_robot = vis.rendering.MaterialRecord()
            mat_robot.shader = 'defaultLit'
            mat_robot.base_color = [0.3, 0, 0, 1.0]
            robots.append({'name': 'gt_robot', 'geometry': robot_mesh, 'material': mat_robot})

        return robots

    def get_implicit_surface(self, q=None, value=0.):
        if self.test_grid is None:
            self.test_grid = self.generate_grid()

        if q is None:
            q = self.robot.sample_random_robot_config()
            self.robot.set_robot_joints(q)
        else:
            self.robot.set_robot_joints(q)
        # test_grid = copy.deepcopy(self.test_grid)
        print("Model inference ...")
        raw_pred = self.calculate_signed_distance(self.test_grid)
        print("Done.")
        raw_pred = np.reshape(raw_pred, self.grid_size)

        print(raw_pred.shape)
        vertices, faces, _, _ = marching_cubes(raw_pred,
                                               gradient_direction='ascent',
                                               level=value,
                                               spacing=self.spacing)

        pred_mesh = o3d.geometry.TriangleMesh()
        pred_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        pred_mesh.triangles = o3d.utility.Vector3iVector(faces)
        pred_mesh.paint_uniform_color([0.2, 0.4196, 0.7])
        pred_mesh.compute_vertex_normals()
        pred_mesh.translate(self.lower_bound)

        return pred_mesh

    # def show_

    def show_implicit_surface(self, q=None, with_robot=True, value=0, material=None) -> None:
        pred_mesh = self.get_implicit_surface(q, value)

        if with_robot:
            robot_mesh = self.robot.get_combined_mesh(convex=False, bounding_box=False)
            robot_mesh = robot_mesh.as_open3d
            robot_mesh.compute_vertex_normals()
            robot_mesh.translate([1, 0, 0])
            if material is None:
                o3d.visualization.draw_geometries([robot_mesh, pred_mesh])
            else:
                mat_robot = vis.rendering.MaterialRecord()
                mat_robot.shader = 'defaultLit'
                mat_robot.base_color = [0.3, 0, 0, 1.0]
                vis.draw([{'name': 'sphere', 'geometry': robot_mesh, 'material': mat_robot},
                          {'name': 'box', 'geometry': pred_mesh, 'material': material}])
        else:
            if material is None:
                o3d.visualization.draw_geometries([pred_mesh])

            else:
                vis.draw([{'name': 'box', 'geometry': pred_mesh, 'material': material}])

    def show_voxel_grid(self, q=None, with_robot=True, threshold=-0.0005, voxel_size=0.01) -> None:
        if self.test_grid is None:
            self.test_grid = self.generate_grid()

        if q is None:
            q = self.robot.sample_random_robot_config()
            self.robot.set_robot_joints(q)
        else:
            self.robot.set_robot_joints(q)

        test_grid = copy.deepcopy(self.test_grid)

        print("Model inference ...")
        raw_pred = self.calculate_signed_distance(self.test_grid)
        print("Done.")

        pred_grid = test_grid[np.where(raw_pred >= threshold)[0]]
        pred_pcd = self.np2o3d_pcd(pred_grid)
        o3d.io.write_point_cloud("arm_pcd.ply", pred_pcd)
        pred_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pred_pcd, voxel_size=voxel_size)
        if with_robot:
            robot_mesh = self.robot.get_combined_mesh(convex=False, bounding_box=False)
            robot_mesh = robot_mesh.as_open3d
            robot_mesh.translate([1, 0, 0])
            o3d.visualization.draw_geometries([robot_mesh, pred_voxels])
        else:
            o3d.visualization.draw_geometries([pred_voxels])

    def set_robot_joint_positions(self, q) -> None:
        assert len(q) == 7
        self.robot.set_robot_joints(q)

    def show_robot(self, convex=False, bounding_box=False) -> None:
        self.robot.show_robot_meshes(convex, bounding_box)

    def sample_random_robot_config(self) -> np.ndarray:
        return self.robot.sample_random_robot_config()

    def show_robot_with_points(self, points, point_radius=0.02) -> None:
        points = np.array(points)
        if points.shape == (3,):
            points = np.expand_dims(points, axis=0)

        robot_mesh = self.robot.get_combined_mesh(convex=False, bounding_box=False)
        scene = trimesh.Scene()
        for point in points:
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=point_radius,
                                                color=np.random.uniform(size=4))
            sphere.apply_translation(point)
            scene.add_geometry(sphere)

        scene.add_geometry(robot_mesh)
        scene.show()


class opt_JSDF(iiwa_JSDF):
    def __init__(self):
        super().__init__()

    def calculate_squared_signed_distance(self, position: list, link_index=None):
        assert len(link_index) <= 8

        dist = self.calculate_signed_distance(position, min_dist=False)
        dist = dist.squeeze()
        if link_index is not None:
            dist = dist[np.array(link_index)]

        return np.power(dist, 2) * 100

    def calculate_gradient(self, position: list, link_index=None):
        """ get gradient wrt input of the implicit function for batch input joint configuration and positions
        :param link_index:
        :param position: (n, 3) or (3, )
        :return: pred values (n, 10)
        """
        assert self.model is not None
        position = np.array(position)
        if position.shape == (3,):
            position = np.expand_dims(position, axis=0)

        data_len = position.shape[0]

        q = torch.from_numpy(np.array([self.robot.robot_q])).to(self.device)
        nq = q.repeat(data_len, 1).to(self.device)

        qp_tensor = torch.concat((nq, torch.from_numpy(position)), dim=1).type(torch.float32)
        qp_tensor.requires_grad = True

        pred = self.model(qp_tensor).squeeze()
        pred = pred[np.array(link_index)]

        pred = torch.pow(pred, 2) * 100
        pred.sum().backward()
        gradient = qp_tensor.grad.cpu().numpy()

        self.model.zero_grad()

        return gradient.squeeze()[:7]


if __name__ == "__main__":
    # inputs = torch.rand(2, 10)
    # inputs.requires_grad = True

    jsdf = iiwa_JSDF()
    jsdf.show_voxel_grid(q=[0.15174446, 0.6, -0.48907439, -1.9, 1.95646813, 0, 3.14], with_robot=False)


    # gradient = jacobian(jsdf.model, inputs)
    # print(gradient.shape)
    # gradient = torch.sum(gradient, dim=2)
    #
    # # gradient = torch.sum(gradient, dim=1)
    # gradient = jsdf.calculate_gradient(np.random.rand(2, 3))
    # print(gradient.shape)
    # print(gradient)
    #
    # pred = jsdf.model(inputs)
    # pred.sum().backward()
    # print(inputs.grad.cpu().numpy())
