import torch
import numpy as np
from .NN_model.nn_model_eval import NN_hand_obj
# from NN_model.nn_model_eval import NN_hand_obj

from neural_jsdf.utils import hand_kinematic
import copy
import open3d as o3d


class hand_JSDF(NN_hand_obj):
    def __init__(self,
                 grid_size=None,
                 lower_bound=None,
                 upper_bound=None,
                 use_GPU=False,
                 generate_grid=False):
        super().__init__(use_cuda=use_GPU, right=False, g=[0, 1, 2, 3, 4])
        if grid_size is None:
            grid_size = [150, 150, 75]
        if lower_bound is None:
            lower_bound = [-0.2, -0.2, -0.]
        if upper_bound is None:
            upper_bound = [0.2, 0.2, 0.2]
        self.device = "cuda:0" if (torch.cuda.is_available() and use_GPU) else "cpu"

        assert len(lower_bound) == 3, "Bound for x, y, z three dimension"
        assert len(upper_bound) == 3, "Bound for x, y, z three dimension"
        assert len(grid_size) == 3, "Grid size should be in type int"

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.grid_size = grid_size
        self.spacing = tuple((np.array(self.upper_bound) - np.array(self.lower_bound)) / np.array(grid_size))
        self.robot = hand_kinematic()
        if generate_grid:
            self.test_grid = self.generate_grid()
        else:
            self.test_grid = None

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

    def model_inference_with_gradient(self, qp):
        assert qp.shape[1] == 19
        # print(position.shape)
        for i in range(16):  # normalize the joint positions
            qp[:, i] = (qp[:, i] - self.hand_bound[0][i]) / (self.hand_bound[1][i] - self.hand_bound[0][i]) * 2 - 1

        # position = torch.from_numpy(position).type(torch.float32)
        q = qp[:, :16]
        x = qp[:, 16:]

        outputs = []
        for g in self.g:
            # x_obj_gpu = x.detach().clone()
            x_obj_gpu = x.clone()
            if g in [2, 3]:
                x_obj_gpu = (torch.matmul(self.T[g - 2][:3, :3], x_obj_gpu.T) + self.T[g - 2][:3, 3:4]).T
                k = 1
            else:
                k = g
            for i in range(3):  # normalize the object positions
                x_obj_gpu[:, i] = (x_obj_gpu[:, i] - self.obj_bounds[k][0, i]) / (
                        self.obj_bounds[k][1, i] - self.obj_bounds[k][0, i]) * 2 - 1
            if g == 0:
                inputs = x_obj_gpu  # (n, 3), for the palm
            else:
                inputs = torch.cat([q[:, (g - 1) * 4: g * 4], x_obj_gpu], 1)  # (n, 4+3), for each finger

            # with torch.no_grad():
            output = self.nets[k](inputs)
            output = recover_dis(output, self.dis_scales[k])

            # keep the same with the arm model, outside neg inside pos
            output = output * -1

            # output = output.cpu().numpy()
            outputs.append(output)

        return torch.hstack(outputs)


    def calculate_signed_distance(self, position: list) -> np.ndarray:
        """ get implicit function values for batch input positions
        :param min_dist: return minimum distance to the whole robot, corresponding to the maximum signed distance
        :param position: (n, 19) or (19, )
        :return: pred values (n, )
        """
        assert self.nets is not None
        position = np.array(position)
        if position.shape == (3,):
            position = np.expand_dims(position, axis=0)

        data_len = position.shape[0]

        q = torch.from_numpy(np.array([self.robot.robot_q]))
        nq = q.repeat(data_len, 1)

        position = torch.concat((nq, torch.from_numpy(position)), dim=1).type(torch.float32).to(self.device)
        outputs = self.model_inference_with_gradient(position)
        outputs = outputs.cpu().detach().numpy()

        return outputs

    def np2o3d_pcd(self, points, color=None) -> o3d.geometry.PointCloud:
        if color is None:
            color = [0.6, 0, 0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)

        return pcd

    def show_voxel_grid(self, q=None, with_robot=True, threshold=-0.001, voxel_size=0.003) -> None:
        if self.test_grid is None:
            self.test_grid = self.generate_grid()

        if q is None:
            q = self.robot.sample_random_robot_config()
            # syn_q = self.robot.sample_synergy_finger_config()
            # self.robot.set_finger_synergy_control(syn_q)
            self.robot.set_hand_joints(q)
        else:
            self.robot.set_hand_joints(q)

        test_grid = copy.deepcopy(self.test_grid)

        print("Model inference ...")
        raw_pred = self.calculate_signed_distance(self.test_grid)
        print("Done.")

        pred_grid = test_grid[np.where(raw_pred >= threshold)[0]]
        pred_pcd = self.np2o3d_pcd(pred_grid)
        o3d.io.write_point_cloud("hand_pcd.ply", pred_pcd)

        pred_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pred_pcd, voxel_size=voxel_size)
        if with_robot:
            robot_mesh = self.robot.get_combined_mesh(convex=False, bounding_box=False)
            robot_mesh = robot_mesh.as_open3d
            robot_mesh.compute_vertex_normals()
            robot_mesh.translate([0.1, 0, 0])
            o3d.visualization.draw_geometries([robot_mesh, pred_voxels])
        else:
            o3d.visualization.draw_geometries([pred_voxels])


def recover_dis(output, dis_scale):
    # if mode == 'cpu':
    #     output = np.copy(output)
    # else:  # 'gpu'
    #     pass
    for i in range(dis_scale.shape[1]):
        neg = output[:, i] < 0
        output[neg, i] = - output[neg, i] * dis_scale[0, i]
        pos = output[:, i] > 0
        output[pos, i] = output[pos, i] * dis_scale[1, i]
    return output



if __name__ == "__main__":
    hand_model = hand_JSDF()
    rand_q = np.random.rand(10000, 19)
    # rand_x = torch.rand(1, 3)
    q = [0, 0.44646047, 1.0629986, 0.10425064,
         0, 0.44646047, 1.0629986, 0.10425064,
         0, 0.44646047, 1.0629986, 0.10425064,
         0.3255768, 1.00260788, 0.5091602, 0.83114794]
    hand_model.show_voxel_grid(q, with_robot=False)
    print(hand_model.robot.robot_q)
