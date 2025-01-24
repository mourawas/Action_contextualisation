"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import numpy as np

from .arm import iiwa_JSDF
from .hand import hand_JSDF
from neural_jsdf.utils import KUKA_with_hand
import torch
import open3d as o3d
import copy
from torch.autograd import Variable


class arm_hand_JSDF:
    def __init__(self,
                 grid_upper_bound=None,
                 grid_lower_bound=None,
                 grid_size=None,
                 generate_grid=True,
                 use_GPU=True):

        if grid_lower_bound is None:
            grid_upper_bound = np.array([1.2, 1.2, 1.1])
            grid_upper_bound = [1.1, 0.3, 1.2]
        if grid_lower_bound is None:
            grid_lower_bound = np.array([-0.2, -0.2, -0.2])
            grid_lower_bound = [-0.3, -0.3, 0]

        if grid_size is None:
            # grid_size = np.array([120] * 3)
            grid_size = [140, 60, 110]

        assert (len(grid_lower_bound) == 3) and (len(grid_upper_bound) == 3) and (len(grid_size) == 3), "Length " \
                                                                                                        "should be 3"
        use_GPU = torch.cuda.is_available() and use_GPU

        self.device = "cuda:0" if use_GPU else "cpu"
        self.dtype = torch.float32

        self.grid_size = grid_size
        self.grid_upper_bound = grid_upper_bound
        self.grid_lower_bound = grid_lower_bound
        self.arm_hand_robot = KUKA_with_hand()
        self.arm_hand_robot.arm_chain.to(dtype=self.dtype, device=self.device)

        self.arm_jsdf = iiwa_JSDF(generate_grid=False, use_GPU=use_GPU)
        self.hand_jsdf = hand_JSDF(generate_grid=False, use_GPU=use_GPU)
        self.spacing = tuple((np.array(self.grid_upper_bound) - np.array(self.grid_lower_bound)) / np.array(grid_size))
        self.test_grid = None
        if generate_grid:
            self.test_grid = self.generate_grid()

    def generate_grid(self) -> np.ndarray:
        print("Generating test grid ...")
        axis_x = np.linspace(self.grid_lower_bound[0], self.grid_upper_bound[0], self.grid_size[0])
        axis_y = np.linspace(self.grid_lower_bound[1], self.grid_upper_bound[1], self.grid_size[1])
        axis_z = np.linspace(self.grid_lower_bound[2], self.grid_upper_bound[2], self.grid_size[2])

        grid = []
        for x in np.nditer(axis_x):
            for y in np.nditer(axis_y):
                for z in np.nditer(axis_z):
                    grid.append([x, y, z])
        print("Done.")

        return np.asarray(grid)

    def calculate_signed_distance_raw(self, batch_jp, position):
        arm_q_tensor = torch.from_numpy(batch_jp[:, :7]).type(torch.float32).to(self.device)
        hand_q_tensor = torch.from_numpy(batch_jp[:, 7:]).type(torch.float32).to(self.device)

        assert position.shape == (3,)
        batch_len = len(batch_jp)
        p_tensor = torch.from_numpy(np.repeat(np.expand_dims(position, axis=0),
                                              batch_jp.shape[0],
                                              axis=0)).type(torch.float32).to(self.device)

        arm_q_p_tensor = torch.hstack((arm_q_tensor, p_tensor))
        with torch.no_grad():
            arm_pred = self.arm_jsdf.model(arm_q_p_tensor)

        tg_batch = self.arm_hand_robot.arm_chain.forward_kinematics(arm_q_tensor, end_only=True).get_matrix().type(
            torch.float32)

        mat_ones = torch.ones(batch_len, 1).to(self.device)
        p_tensor_mat = torch.cat((p_tensor, mat_ones), 1)
        p_tensor_hand = torch.linalg.solve(tg_batch, p_tensor_mat)
        hand_q_p_tensor = torch.hstack((hand_q_tensor,
                                        p_tensor_hand[:, :3])).to(self.device)
        with torch.no_grad():
            hand_pred = self.hand_jsdf.model_inference_with_gradient(hand_q_p_tensor)

        # switch back to - inside and + outside
        pred = torch.hstack((arm_pred, hand_pred)) * -1

        pred_min, _ = torch.min(pred, dim=1)

        return pred_min

    def calculate_signed_distance_raw_input(self, batch_input):
        assert batch_input.shape[1] == 26
        arm_q_tensor = torch.from_numpy(batch_input[:, :7]).type(torch.float32).to(self.device)
        hand_q_tensor = torch.from_numpy(batch_input[:, 7: -3]).type(torch.float32).to(self.device)
        p_tensor = torch.from_numpy(batch_input[:, -3:]).type(torch.float32).to(self.device)

        arm_q_p_tensor = torch.hstack((arm_q_tensor, p_tensor))
        with torch.no_grad():
            arm_pred = self.arm_jsdf.model(arm_q_p_tensor)

        tg_batch = self.arm_hand_robot.arm_chain.forward_kinematics(arm_q_tensor, end_only=True).get_matrix().type(
            torch.float32)

        mat_ones = torch.ones(len(batch_input), 1).to(self.device)
        p_tensor_mat = torch.cat((p_tensor, mat_ones), 1)
        p_tensor_hand = torch.linalg.solve(tg_batch, p_tensor_mat)
        hand_q_p_tensor = torch.hstack((hand_q_tensor,
                                        p_tensor_hand[:, :3])).to(self.device)
        with torch.no_grad():
            hand_pred = self.hand_jsdf.model_inference_with_gradient(hand_q_p_tensor)

        # switch back to - inside and + outside
        pred = torch.hstack((arm_pred, hand_pred)) * -1

        pred_min = torch.min(pred)

        return pred_min

    def calculate_signed_distance(self, position: list) -> np.ndarray:
        """ get implicit function values for batch input positions
        :param min_dist: return minimum distance to the whole robot, corresponding to the maximum signed distance
        :param position: (n, 3) or (3, )
        :return: pred values (n, )
        """
        q = self.arm_hand_robot.robot_q
        arm_q = q[:7]
        hand_q = q[7:]

        self.arm_jsdf.robot.set_robot_joints(arm_q)
        arm_signed_dist = self.arm_jsdf.calculate_signed_distance(position, min_dist=False)
        # print(arm_signed_dist.shape)

        position = np.array(position)
        if position.shape == (3,):
            position = np.expand_dims(position, axis=0)
        self.hand_jsdf.robot.set_hand_joints(hand_q)
        trans_mat = self.arm_hand_robot.link72base_matrix
        position = np.insert(position, 3, 1, axis=1)

        trans_position = trans_mat @ position.T  # matmul
        # print(trans_position.T)

        hand_signed_dist = self.hand_jsdf.calculate_signed_distance(trans_position.T[:, :3])
        signed_dist = np.hstack([arm_signed_dist, hand_signed_dist])

        return signed_dist

    def whole_arm_inference_with_gradient(self, position, return_grad=True):
        # if link_index is not None:
        #     assert len(link_index) <= 3, "Too many links might lead to optimization failure."

        position = np.array(position)
        if position.shape == (3,):
            position = np.expand_dims(position, axis=0)
        data_len = position.shape[0]

        q = self.arm_hand_robot.robot_q
        arm_q = q[:7]
        hand_q = q[7:]

        p_tensor = torch.from_numpy(position).to(self.device)
        p_tensor.requires_grad = True

        arm_q_tensor = torch.from_numpy(np.array([arm_q]))

        n_arm_q_tensor = arm_q_tensor.repeat(data_len, 1).to(self.device)
        n_arm_q_tensor.requires_grad = True

        hand_q_tensor = torch.from_numpy(np.array([hand_q]))
        n_hand_q_tensor = hand_q_tensor.repeat(data_len, 1).to(self.device)
        n_hand_q_tensor.requires_grad = True

        arm_tensor = torch.concat((n_arm_q_tensor, p_tensor), dim=1).type(torch.float32).to(self.device)

        arm_pred = self.arm_jsdf.model_inference_with_gradient(arm_tensor)
        ee_pose = self.arm_hand_robot.arm_chain.forward_kinematics(n_arm_q_tensor.float(), end_only=True).get_matrix()

        mat_ones = torch.ones(data_len, 1).to(self.device)
        p_tensor_mat = torch.cat((p_tensor, mat_ones), 1)

        # p_tensor_hand = torch.linalg.solve(ee_pose, p_tensor_mat)
        # p_tensor_hand = ee_pose_inv @ p_tensor_mat.T
        ee_pose_inv = torch.linalg.inv(ee_pose)
        p_tensor_hand = ee_pose_inv @ p_tensor_mat.reshape(data_len, 4, 1).float().to(self.device)

        hand_tensor = torch.concat((n_hand_q_tensor, p_tensor_hand.squeeze(axis=2)[:, :3]), dim=1).type(torch.float32).to(
            self.device)
        # hand_tensor = torch.concat((n_hand_q_tensor, p_tensor_hand[:, :3]), dim=1).type(torch.float32).to(self.device)

        hand_pred = self.hand_jsdf.model_inference_with_gradient(hand_tensor)

        output = torch.hstack([arm_pred, hand_pred])
        if not return_grad:
            return output
        else:
            output_sum = torch.sum(output)
            output_sum.backward()
            # return torch.hstack([arm_pred, hand_pred]), torch.hstack([arm_q_tensor.grad, hand_q_tensor.grad, p_tensor.grad])
            return torch.hstack([arm_pred, hand_pred]), torch.hstack([n_arm_q_tensor.grad, n_hand_q_tensor.grad])

    def show_voxel_grid(self, q=None, use_synergy=1, with_robot=True, threshold=-0.0005, voxel_size=0.011) -> None:
        if self.test_grid is None:
            self.test_grid = self.generate_grid()

        if q is None:
            q = self.arm_hand_robot.sample_random_robot_config(use_synergy)
            # print(q)
            self.arm_hand_robot.set_system_config(q, syn_type=use_synergy)
        else:
            self.arm_hand_robot.set_system_config(q, syn_type=use_synergy)

        test_grid = copy.deepcopy(self.test_grid)

        print("Model inference ...")
        raw_pred = self.calculate_signed_distance(self.test_grid)

        print("Done.")

        pred_grid = test_grid[np.where(raw_pred >= threshold)[0]]
        pred_pcd = self.np2o3d_pcd(pred_grid)
        pred_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pred_pcd, voxel_size=voxel_size)
        if with_robot:
            robot_mesh = self.arm_hand_robot.get_system_combined_mesh()
            robot_mesh = robot_mesh.as_open3d
            robot_mesh.translate([1, 0, 0])
            o3d.visualization.draw_geometries([robot_mesh, pred_voxels])
        else:
            o3d.visualization.draw_geometries([pred_voxels])

    def np2o3d_pcd(self, points, color=None) -> o3d.geometry.PointCloud:
        if color is None:
            color = [0., 0.5 , 0.]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)

        return pcd

    def system_zero_grad(self):
        self.arm_jsdf.model.zero_grad()
        for net in self.hand_jsdf.nets:
            if type(net) is not list:
                net.zero_grad()


class opt_arm_hand_JSDF(arm_hand_JSDF):
    def __init__(self):
        super().__init__()

    def inference(self, arm_q_tensor, hand_q_syn, position, link_index):
        data_len = position.shape[0]

        hand_q_syn_4 = torch.hstack([torch.tensor(0), hand_q_syn[:3]])

        hand_q_tensor = torch.hstack([hand_q_syn_4,
                                      hand_q_syn_4,
                                      hand_q_syn_4,
                                      hand_q_syn[3:]])

        p_tensor = torch.from_numpy(position)
        p_tensor.requires_grad = True

        n_arm_q_tensor = arm_q_tensor.repeat(data_len, 1)
        n_hand_q_tensor = hand_q_tensor.repeat(data_len, 1)

        arm_tensor = torch.concat((n_arm_q_tensor, p_tensor), dim=1).type(torch.float32).to(self.device)

        arm_pred = self.arm_jsdf.model_inference_with_gradient(arm_tensor)

        ee_pose = self.arm_hand_robot.arm_chain.forward_kinematics(n_arm_q_tensor.type(torch.float32),
                                                                   end_only=True).get_matrix()

        mat_ones = torch.ones(data_len, 1).type(torch.float32)
        p_tensor_mat = torch.cat((p_tensor, mat_ones), 1).type(torch.float32)

        p_tensor_hand = torch.linalg.solve(ee_pose, p_tensor_mat)
        # p_tensor_hand = ee_pose_inv @ p_tensor_mat.T

        hand_tensor = torch.concat((n_hand_q_tensor, p_tensor_hand[:, :3]), dim=1).type(torch.float32)

        hand_pred = self.hand_jsdf.model_inference_with_gradient(hand_tensor)

        sys_output = torch.hstack([arm_pred, hand_pred])
        contact_output = sys_output[:, link_index]
        contact_output_min, _ = torch.min(contact_output, dim=0)

        lambda_1 = 1
        lambda_2 = 0.2
        obj_value = (lambda_1*torch.max(torch.nn.functional.relu(contact_output_min)) +
                     lambda_2*torch.max(torch.nn.functional.relu(-contact_output_min)))
        # hand_pred_abs = torch.abs(hand_pred)
        # arm_pred_abs = torch.abs(arm_pred)
        #
        # output_abs = torch.hstack([arm_pred_abs, hand_pred_abs])
        #
        # output_abs_selected = output_abs[:, link_index]
        # output_abs_min, _ = torch.min(output_abs_selected, dim=0)
        # obj_value = torch.sum(output_abs_min)

        return obj_value

    def calculate_squared_dist_hand_link(self, q, link_index, position):
        assert len(link_index) >= 1, "At least one link for touch"
        assert len(q) == 14, "7 DoF of Arm + 3 DoF of hand Synergy"
        q = np.array(q)
        position = np.array(position)
        if position.shape == (3,):
            position = np.expand_dims(position, axis=0)

        q_tensor = torch.from_numpy(q)
        arm_q_tensor = q_tensor[:7]
        arm_q_tensor.requires_grad = True
        hand_q_syn = q_tensor[7:]
        hand_q_syn.requires_grad = True

        output_sqr = self.inference(arm_q_tensor, hand_q_syn, position, link_index)

        self.system_zero_grad()

        return output_sqr.cpu().detach().numpy()

    def calculate_gradient_dist_hand_link(self, q, link_index, position):
        assert len(link_index) >= 1, "At least one link for touch"
        assert len(q) == 14, "7 DoF of Arm + 3 DoF of hand Synergy + 4 Dof of thumb"
        q = np.array(q)
        position = np.array(position)
        if position.shape == (3,):
            position = np.expand_dims(position, axis=0)
        data_len = position.shape[0]

        q_tensor = torch.from_numpy(q)
        arm_q_tensor = q_tensor[:7]
        arm_q_tensor.requires_grad = True
        hand_q_syn = q_tensor[7:]
        hand_q_syn.requires_grad = True

        output_sqr = self.inference(arm_q_tensor, hand_q_syn, position, link_index)

        output_sqr.backward()

        grad = torch.hstack([arm_q_tensor.grad, hand_q_syn.grad])

        self.system_zero_grad()
        return grad

    def whole_arm_constraint_inference(self, position, q, link_index, return_grad=True):
        assert len(q) == 14, "7 DoF of Arm + 3 DoF of hand Synergy + 4 Dof of thumb"
        q = np.array(q)
        position = np.array(position)
        if position.shape == (3,):
            position = np.expand_dims(position, axis=0)
        data_len = position.shape[0]

        q_tensor = torch.from_numpy(q)
        arm_q_tensor = q_tensor[:7]
        arm_q_tensor.requires_grad = True
        hand_q_syn = q_tensor[7:]
        hand_q_syn.requires_grad = True

        hand_q_syn_4 = torch.hstack([torch.tensor(0), hand_q_syn[:3]])

        hand_q_tensor = torch.hstack([hand_q_syn_4,
                                      hand_q_syn_4,
                                      hand_q_syn_4,
                                      hand_q_syn[3:]])

        p_tensor = torch.from_numpy(position)
        p_tensor.requires_grad = True

        n_arm_q_tensor = arm_q_tensor.repeat(data_len, 1)
        n_hand_q_tensor = hand_q_tensor.repeat(data_len, 1)

        arm_tensor = torch.concat((n_arm_q_tensor, p_tensor), dim=1).type(torch.float32).to(self.device)

        arm_pred = self.arm_jsdf.model_inference_with_gradient(arm_tensor)

        ee_pose = self.arm_hand_robot.arm_chain.forward_kinematics(n_arm_q_tensor.type(torch.float32),
                                                                   end_only=True).get_matrix()

        mat_ones = torch.ones(data_len, 1).type(torch.float32)
        p_tensor_mat = torch.cat((p_tensor, mat_ones), 1).type(torch.float32)

        p_tensor_hand = torch.linalg.solve(ee_pose, p_tensor_mat)

        hand_tensor = torch.concat((n_hand_q_tensor, p_tensor_hand[:, :3]), dim=1).type(torch.float32)

        hand_pred = self.hand_jsdf.model_inference_with_gradient(hand_tensor)

        output = torch.hstack([arm_pred, hand_pred]) * -1

        # links that not in use
        output = output[:, link_index]
        output_min = torch.min(output)

        if not return_grad:
            self.system_zero_grad()
            return output_min
        else:
            output_min.backward()
            grad = torch.hstack([arm_q_tensor.grad, hand_q_syn.grad])
            self.system_zero_grad()
            return output_min, grad


if __name__ == "__main__":
    test = arm_hand_JSDF(use_GPU=True)
    a = np.array([[0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, -0.1],
                  [0, 0, 0.54, 0.4, 0.2, 0.1, 0.3,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, -0.15],
                  [0, 0, 0.54, 0.4, 0.2, 0.1, 0.3,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0.05]
                  ])
    dist = test.calculate_signed_distance_raw_input(a)
    print(dist)
    # test = opt_arm_hand_JSDF()
    # test.calculate_squared_dist_hand_link([[0, 0.1, 0.2], [0.2, 0.1, 0.3]])
    # position = [[0, 0.1, 0.2], [0.1, 0.3, 0.3]]
    # test.calculate_squared_dist_hand_link(position)
    # print(test.whole_arm_inference_with_gradient(position, return_grad=True))
    # dist = test.calculate_squared_dist_hand_link([0.] * 14,
    #                                              link_index=[3, -1],
    #                                              position=position)
    # # print(dist)
    # print(test.calculate_gradient_dist_hand_link([0.] * 14,
    #                                              link_index=[3, -1],
    #                                              position=position))
    test = arm_hand_JSDF()
    q = [0.15174446, 0.6, -0.48907439, -1.9, 1.95646813, 0, 3.14,
         0.44646047, 1.0629986, 0.10425064,
         0.3255768, 1.00260788, 0.5091602, 0.83114794]
    test.arm_hand_robot.set_system_config(q, syn_type=1)
    test.show_voxel_grid(q=q, with_robot=False, use_synergy=1)
    # print(test.calculate_signed_distance(position=[[0, 0, 1], [0, 0, 3]]).shape)
