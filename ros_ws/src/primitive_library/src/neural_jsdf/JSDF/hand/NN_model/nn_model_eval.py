import time
from typing import List, Any

import torch.nn as nn
import torch
import numpy as np
from .nn_model import Net, Net_5
import os
import sys


class NN_hand_obj:
    def __init__(self, path_prefix_suffix=None, use_cuda=True, g=[0, 1, 2, 3, 4], add_sin_cos=1,
                 x_obj_normalized=False, right=True):
        self.nets: list = [[]] * 5
        self.obj_bounds: list = [[]] * 5
        self.hand_bound = []
        self.dis_scales: list = [[]] * 5
        self.use_cuda = use_cuda
        self.add_sin_cos = add_sin_cos

        self.g = g
        if (2 in g or 3 in g) and 1 not in g:
            g = [1] + g

        # self.g_index_q = [[]] * 5
        self.x_obj_normalized = x_obj_normalized

        for i in g:
            if i in [2, 3]:
                if path_prefix_suffix is None:
                    if right:
                        T12 = np.loadtxt(self._get_file_path("models/T12"))
                    else:
                        T12 = np.loadtxt(self._get_file_path("models/T12_left"))
                else:
                    if right:
                        T12 = np.loadtxt(path_prefix_suffix[0][:-7] + "T12")
                    else:
                        T12 = np.loadtxt(path_prefix_suffix[0][:-7] + "T12_left")
                T1 = T12[:4, :]
                T2 = T12[4:8, :]
                self.T = [np.linalg.inv(T1), np.linalg.inv(T2)]
                # self.T = [T1, T2]
                if use_cuda:
                    self.T = [torch.Tensor(mat).to('cuda:0') for mat in self.T]
                else:
                    self.T = [torch.Tensor(mat) for mat in self.T]
                # this matrix will transfer the points in middle/ring fingers to the index finger.
            else:  # [0, 1, 4]
                if path_prefix_suffix is None:
                    path_check_point = 'models/single_' + str(i) + '01.pt'
                    # path_check_point = 'models/single_' + str(i) + '01.pt'

                else:
                    assert len(path_prefix_suffix) == 2
                    path_check_point = path_prefix_suffix[0] + str(i) + path_prefix_suffix[1]
                if not right:
                    path_check_point = path_check_point[:-3] + '_left.pt'

                path_check_point = self._get_file_path(path_check_point)
                print('Hand-obj collision: load NN model from', path_check_point, 'use_cuda=', use_cuda)
                if use_cuda:
                    checkpoint = torch.load(path_check_point)
                else:
                    device = torch.device('cpu')
                    checkpoint = torch.load(path_check_point, map_location=device)  # add this to load the model to cpu.
                dim_in, dim_out = checkpoint['net_dims']
                layer_nums = checkpoint['layer_nums']
                net = Net(dim_in, dim_out, layer_nums=layer_nums)
                net.load_state_dict(checkpoint['model_state_dict'])
                if use_cuda:
                    net.to('cuda:0')
                self.nets[i] = net
                self.obj_bounds[i] = checkpoint['obj_bounds']
                self.hand_bound = checkpoint['hand_bounds']
                self.dis_scales[i] = checkpoint['dis_scale']

    def _get_file_path(self, filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    def eval_multiple_full(self, q, x, gradient=False):
        """
        :q, (n, 16),  the joint positions
        :x, (m, 3),pointcloud of the object
        :Return: distances to 22 links and gradients
                    (n, m, 22),                (n, m, 22, 19)
        """

        n = q.shape[0]
        m = x.shape[0]
        for i in range(16):  # normalize the joint positions
            q[:, i] = (q[:, i] - self.hand_bound[0][i]) / (self.hand_bound[1][i] - self.hand_bound[0][i]) * 2 - 1
        q, x = q.repeat_interleave(m, 0), x.repeat(n, 1)

        outputs = []
        if gradient:
            grads = []
        else:
            grads = None
        for g in self.g:
            x_obj_gpu = x.detach().clone()
            # x_obj_gpu = x
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

            if gradient:
                inputs.requires_grad = True
                output = self.nets[k](inputs)
                # min_index = torch.argmin(output, dim=1)
                if g:
                    d_qx_all = []
                    for i in range(output.shape[1]):
                        torch.sum(output[:, i]).backward(retain_graph=True)
                        d_qx = inputs.grad[:, :4].cpu().numpy()  # derivative with q, (mn, 4)
                        d_qx_all.append(d_qx)

                    d_qx_all = np.dstack(d_qx_all)   # (mn, 16, 5)
                    grads.append(d_qx_all)

                output = recover_dis(output, self.dis_scales[k], mode='gpu')
                output = output.cpu().detach().numpy()

                outputs.append(output)


            else:
                with torch.no_grad():
                    output = self.nets[k](inputs)
                    output = recover_dis(output, self.dis_scales[k], mode='gpu')

                output = output.cpu().numpy()
                outputs.append(output)

        output_final = np.hstack(outputs).reshape(n, m, 22)
        if gradient:
            grads_final = np.vstack(grads).reshape(n, m, 16, 5)
            # return output_5, grads[i[0]], i[0]
            return output_final, grads_final
        else:
            return output_final, None

    def eval_multiple(self, q: torch.Tensor, x: torch.Tensor, real_dis=False, only_min_dis=True, gradient=False,
                      only_one_mesh_dis=None, dx=False):
        """
            only_min_dis : only return the min distance between x and all meshes
        """
        # assert q.device.type == 'cuda'
        if x.device.type == 'cpu' and self.use_cuda:
            x = x.to('cuda:0')
        if q.device.type == 'cpu' and self.use_cuda:
            q = q.to('cuda:0')
        xn = x.shape[0]
        qn = q.shape[0]

        q, x = q.repeat_interleave(xn, 0), x.repeat(qn, 1)
        # q, x = q.repeat(xn, 1), x.repeat_interleave(qn, 0)
        # print(q.shape, x.shape)

        # outputs = [[]] * 5
        if q.shape[1] == 4 and len(self.g) == 1 and self.g[0] != 0:
            # only one finger
            lb = self.hand_bound[0, (self.g[0] - 1) * 4:self.g[0] * 4]
            ub = self.hand_bound[1, (self.g[0] - 1) * 4:self.g[0] * 4]
            for i in range(4):  # joint angles for only one finger
                q[:, i] = (q[:, i] - lb[i]) / (ub[i] - lb[i]) * 2 - 1

            g = self.g[0]
            x_obj_gpu = x.detach().clone()
            if g in [2, 3]:
                x_obj_gpu = (torch.matmul(self.T[g - 2][:3, :3], x_obj_gpu.T) + self.T[g - 2][:3, 3:4]).T
                k = 1
            else:
                k = g
            for i in range(3):  # normalize the object position
                x_obj_gpu[:, i] = (x_obj_gpu[:, i] - self.obj_bounds[k][0, i]) / (
                        self.obj_bounds[k][1, i] - self.obj_bounds[k][0, i]) * 2 - 1

            inputs = torch.cat([q, x_obj_gpu], 1)
            if gradient:
                # pass
                inputs.requires_grad = True
                output = self.nets[k](inputs)

                min_index = torch.argmin(output, dim=1)
                if only_one_mesh_dis is not None:
                    min_index[:] = only_one_mesh_dis
                    # use the sum of distance to calculate gradient
                torch.sum(output[range(0, output.shape[0]), min_index]).backward()
                # torch.sum(output).backward()
                # d_qx_all = inputs.grad
                d_qx = inputs.grad
                if dx:
                    dq = d_qx.cpu().numpy()
                else:
                    dq = d_qx[:, :4].cpu().numpy()
                if real_dis:
                    output = recover_dis(output, self.dis_scales[k], mode='gpu')
                if only_min_dis:
                    output, _ = torch.min(output, 1)
                output = output.cpu().detach().numpy()
                # torch.cuda.empty_cache()  # use this to empty GPU memory

                if xn == 1:
                    return output, dq
                else:  # here have to enable only_min_dis
                    output_2D = output.reshape(-1, xn)
                    min_index2 = np.argmin(output_2D, axis=1)
                    # min_index2 = [np.argmin(output[i * xn: (i + 1) * xn]) + i * xn for i in
                    #               range(int(output.shape[0] / xn))]
                    dq2 = dq.reshape(qn, xn, 4)
                    ax_0 = np.arange(dq2.shape[0])[:, None]
                    ax_2 = np.arange(dq2.shape[2])[None, :]
                    # return  distances (qn, )                               gradient     (qn,4)
                    return output_2D[ax_0, min_index2.reshape(-1, 1)].flatten(), dq2[
                        ax_0, min_index2.reshape(-1, 1), ax_2]

            else:
                with torch.no_grad():
                    output = self.nets[k](inputs)
                    if real_dis:
                        output = recover_dis(output, self.dis_scales[k], mode='gpu')
                    if only_min_dis:
                        output, _ = torch.min(output, 1)
                output = output.cpu().numpy()
                if xn == 1:
                    return output
                else:
                    output_2D = output.reshape(-1, xn)
                    output = np.min(output_2D, axis=1)
                    # min_index2 = output
                    # min_index2 = [np.argmin(output[i * xn: (i + 1) * xn]) + i * xn for i in
                    #               range(int(output.shape[0] / xn))]
                    # return output[min_index2]
                    return output
        else:
            # 16 dimension, eval 3 models
            for i in range(16):
                q[:, i] = (q[:, i] - self.hand_bound[0][i]) / (self.hand_bound[1][i] - self.hand_bound[0][i]) * 2 - 1

            outputs = []
            if gradient:
                grads = []
            for g in self.g:
                x_obj_gpu = x.detach().clone()
                if g in [2, 3]:
                    x_obj_gpu = (torch.matmul(self.T[g - 2][:3, :3], x_obj_gpu.T) + self.T[g - 2][:3, 3:4]).T
                    k = 1
                else:
                    k = g
                for i in range(3):  # normalize the object position
                    x_obj_gpu[:, i] = (x_obj_gpu[:, i] - self.obj_bounds[k][0, i]) / (
                            self.obj_bounds[k][1, i] - self.obj_bounds[k][0, i]) * 2 - 1
                if g == 0:
                    inputs = x_obj_gpu
                else:
                    inputs = torch.cat([q[:, (g - 1) * 4: g * 4], x_obj_gpu], 1)

                if gradient:
                    inputs.requires_grad = True
                    output = self.nets[k](inputs)
                    min_index = torch.argmin(output, dim=1)
                    torch.sum(output[range(0, output.shape[0]), min_index]).backward()
                    d_q = []
                    if g > 0:
                        d_qx = inputs.grad
                        d_q = d_qx[:, :4].cpu().numpy()  # derivate with q
                    if real_dis:
                        output = recover_dis(output, self.dis_scales[k], mode='gpu')
                    if only_min_dis:
                        output, _ = torch.min(output, 1)
                    output = output.cpu().detach().numpy()
                    if xn != 1:
                        output_2D = output.reshape(-1, xn)
                        output = np.min(output_2D, axis=1)
                        if g > 0:
                            d_q = d_q[np.argmin(output_2D, axis=1), :]
                    outputs.append(output)
                    grads.append(d_q)


                else:
                    with torch.no_grad():
                        output = self.nets[k](inputs)
                        if real_dis:
                            output = recover_dis(output, self.dis_scales[k], mode='gpu')
                        if only_min_dis:
                            output, _ = torch.min(output, 1)
                    output = output.cpu().numpy()
                    if xn == 1:
                        pass
                    else:
                        output_2D = output.reshape(-1, xn)  # (qn ,xn)
                        output = np.min(output_2D, axis=1)  # (qn, )

                    outputs.append(output)

            if gradient:
                # i = np.argmin(np.vstack(outputs), axis=0)  # (5, qn) here qn=1 for robot point
                output_5 = np.min(np.vstack(outputs), axis=0)
                # grads[0] = np.zeros(output_5.shape)
                # grads_16 = []
                # for i_min, j in enumerate(i):
                #     if i_min == 0:
                # 
                #     tmp = np.concatenate([grads[]])
                #     grads_16.append()

                grads_16 = np.concatenate([grads[1], grads[2], grads[3], grads[4]], axis=1)
                # return output_5, grads[i[0]], i[0]
                return output_5, grads_16
            else:
                output_5 = np.min(np.vstack(outputs), axis=0)  # here for multiple q and multiple x, return (qn, )
                return output_5

    def eval(self, q, x, gradient=False, real_dis=False, only_min_dis=True):
        # input single q and multiple(nx) x for obstacles, return (nx,), for rviz
        # output min distances and gradient
        inputs = self.data_preprocess(q, x)
        outputs = [[]] * 5

        if gradient:
            grad = [[]] * 5
            for g in self.g:
                if g in [2, 3]:
                    k = 1  # use the index NN model
                else:
                    k = g
                # t1 = time.time()
                inputs[g].requires_grad = True
                output = self.nets[k](inputs[g])
                torch.min(output).backward()
                row = torch.div(torch.argmin(output), output.shape[1], rounding_mode='trunc')
                # output.backward()
                d_qx = inputs[g].grad
                d_qx = d_qx[row, :].cpu().numpy()
                # t2 = time.time() - t1

                output_min_row = output[row:row + 1, :]
                if real_dis:
                    output_min_row = recover_dis(output_min_row, self.dis_scales[k], mode='gpu')
                if only_min_dis:
                    outputs[g] = torch.min(output_min_row).cpu().detach().numpy()
                else:
                    outputs[g] = output_min_row.cpu().detach().numpy()[0, :]
                grad[g] = d_qx
            return outputs, grad

            # this one is more efficient than calculate the min(output) and then input only one row to NN.
            #### 2
            # t3 = time.time()
            # with torch.no_grad():
            #     output = self.nets[k](inputs[g])
            # row = torch.argmin(output) // output.shape[1]
            # input_min = inputs[g][row:row+1, :]
            # input_min.requires_grad = True
            # output_min = self.nets[k](input_min)
            # torch.min(output_min).backward()
            # b = input_min.grad
            # t4 = time.time() - t3

        else:
            for g in self.g:
                if g in [2, 3]:
                    k = 1  # use the index NN model
                else:
                    k = g
                with torch.no_grad():
                    output = self.nets[k](inputs[g])
                    if real_dis:
                        output = recover_dis(output, self.dis_scales[k], mode='gpu')
                    if only_min_dis:
                        output, _ = torch.min(output, 1)
                    else:
                        output, _ = torch.min(output, 1)
                outputs[g] = output.cpu().numpy()
            # outputs = [outputs[g] for g in self.g]
            return outputs

    def data_preprocess(self, q, x) -> list:
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
        assert q.shape[0] == 1

        if type(q) == np.ndarray:
            q = torch.Tensor(q)

        if self.x_obj_normalized:
            assert type(x) == list
            assert len(x) == 5
        else:
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            if type(x) == np.ndarray:
                x = torch.Tensor(x)
            if self.use_cuda and x.device.type == 'cpu':
                x = x.to('cuda:0')

        if self.use_cuda and q.device.type == 'cpu':
            q = q.to('cuda:0')

        # check the dimension of q and x, and unify them
        q_normal = q.detach().clone()
        lb = self.hand_bound[0, :]
        ub = self.hand_bound[1, :]
        for i in range(16):  # joint angles # todo
            q_normal[:, i] = (q_normal[:, i] - lb[i]) / (ub[i] - lb[i]) * 2 - 1

        if self.add_sin_cos == 3:
            q_all = torch.cat((q_normal, torch.sin(q), torch.cos(q)), 1)
        elif self.add_sin_cos == 2:
            q_all = torch.cat((torch.sin(q), torch.cos(q)), 1)
        elif self.add_sin_cos == 1:
            q_all = q_normal.detach().clone()
        else:
            raise NotImplementedError

        inputs = [[]] * 5
        # first transfer points to index, then normalize positions
        for g in self.g:
            if self.x_obj_normalized:
                x_obj_gpu = x[g]
            else:
                x_obj_gpu = x.detach().clone()
                if g in [2, 3]:
                    x_obj_gpu = (torch.matmul(self.T[g - 2][:3, :3], x_obj_gpu.T) + self.T[g - 2][:3, 3:4]).T
                    k = 1
                else:
                    k = g
                for i in range(3):  # normalize the object position
                    x_obj_gpu[:, i] = (x_obj_gpu[:, i] - self.obj_bounds[k][0, i]) / (
                            self.obj_bounds[k][1, i] - self.obj_bounds[k][0, i]) * 2 - 1
            if g == 0:
                input_data = x_obj_gpu
            else:
                if len(self.g) == 1:
                    finger = np.arange(0, 4)
                else:
                    finger = np.arange((g - 1) * 4, g * 4)
                if self.add_sin_cos > 1:
                    for j in range(1, self.add_sin_cos):
                        tmp = np.arange((g - 1) * 4 + j * 16, g * 4 + j * 16)
                        finger = np.concatenate([finger, tmp])
                q_i = q_all[:, finger].repeat(x_obj_gpu.shape[0], 1)
                input_data = torch.cat([q_i, x_obj_gpu], 1)
            inputs[g] = input_data

        return inputs


class NN_model:
    def __init__(self, net, opt=None, path_check_point=None, use_cuda=True):
        # pass
        # load model
        self.net = net
        if path_check_point is None:
            path_check_point = 'NN_model/models/model_03.pt'
        checkpoint = torch.load(path_check_point)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        # self.max_dis = checkpoint['max_dis']
        self.obj_bound = checkpoint['obj_bounds']
        self.hand_bounds = checkpoint['hand_bounds']
        self.dis_scale = checkpoint['dis_scale']
        print('Load data from', path_check_point)
        # opt.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        # filepath = 'NN_model/models/linear4_error2.7'
        # print(filepath)
        # self.net.load_state_dict(torch.load(filepath))
        self.net.eval()
        if use_cuda:
            self.net.to('cuda:0')
        if path_check_point[:2] == '..':
            path = path_check_point[:-11]
        else:
            path = 'NN_model/models/'
        # self.max_dis = np.loadtxt(path + 'max_dis.txt')

        # self.hand_bounds = np.loadtxt(path + 'hand_joint_bound.txt')

    def eval_single(self, qx, g=0, use_cuda=True):
        if len(qx.shape) == 1:
            q = qx.reshape(1, -1)
        x = qx[:, 16:19]
        obj_lb = self.obj_bound[0, :]
        obj_ub = self.obj_bound[1, :]
        for i in range(3):  # obj position to [-1, 1]
            x[:, i] = (x[:, i] - obj_lb[i]) / (obj_ub[i] - obj_lb[i]) * 2 - 1

        q = qx[:, :16]
        lb = self.hand_bounds[0, :]
        ub = self.hand_bounds[1, :]
        for i in range(16):  # joint angles
            q[:, i] = (q[:, i] - lb[i]) / (ub[i] - lb[i]) * 2 - 1
        if g == 0:
            input_data = x
        elif g == 1:
            input_data = np.concatenate([q[:, [0, 1, 2, 3]], x], axis=1)
        elif g == 4:
            input_data = np.concatenate([q[:, [12, 13, 14, 15]], x], axis=1)
        else:
            raise NotImplementedError

        device = torch.device("cuda:0" if use_cuda else "cpu")
        input_data = torch.Tensor(input_data)
        if use_cuda:
            input_data = input_data.to(device)
        self.net.to(device)

        with torch.no_grad():
            outputs = self.net(input_data)

        if use_cuda:
            outputs = outputs.cpu().numpy()
        else:
            outputs = outputs.numpy()
        return outputs

    def eval(self, q, use_cuda=True, add_sin_cos=3):
        # q : numpy array, (n, 16)
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
        if q.shape[1] == 19:
            obj = True
            obj_bounds = self.obj_bound

            x = q[:, 16:19]
            q = q[:, :16]

            obj_lb = obj_bounds[0, :]
            obj_ub = obj_bounds[1, :]
            for i in range(3):  # obj position to [-1, 1]
                x[:, i] = (x[:, i] - obj_lb[i]) / (obj_ub[i] - obj_lb[i]) * 2 - 1
        else:
            obj = False
        # transfer q to [-1,1]
        lb = self.hand_bounds[0, :]
        ub = self.hand_bounds[1, :]

        if add_sin_cos == 3:
            q_all = np.concatenate([q, np.sin(q), np.cos(q)], axis=1)
        elif add_sin_cos == 2:
            q_all = np.concatenate([np.sin(q), np.cos(q)], axis=1)
        elif add_sin_cos == 1:
            q_all = q
        else:
            raise NotImplementedError

        for i in range(16):  # joint angles
            q_all[:, i] = (q_all[:, i] - lb[i]) / (ub[i] - lb[i]) * 2 - 1
        if obj:
            q_all = np.concatenate([q_all, x], axis=1)

        q_all = torch.Tensor(q_all)

        device = torch.device("cuda:0" if use_cuda else "cpu")
        if use_cuda:
            q_all = q_all.to(device)
        self.net.to(device)
        # self.net.to(device)

        # print(q_all.device)
        with torch.no_grad():
            outputs = self.net(q_all)

        if use_cuda:
            outputs = outputs.cpu().numpy()
        else:
            outputs = outputs.numpy()
        return outputs

    def eval_rviz(self, q, q_normal, x, use_cuda=True, add_sin_cos=1, g=0):

        # assert q.shape == (16,)
        # q = q.repeat(x.shape[0],1)
        # qx = torch.cat((q, x), 1)
        if add_sin_cos == 3:
            q_all = torch.cat((q_normal, torch.sin(q), torch.cos(q)), 1)
        elif add_sin_cos == 2:
            q_all = torch.cat((torch.sin(q), torch.cos(q)), 1)
        elif add_sin_cos == 1:
            q_all = q_normal
        else:
            raise NotImplementedError

        if g == 0:
            input_data = x
        else:
            finger = np.arange((g - 1) * 4, g * 4)
            if add_sin_cos > 1:
                for j in range(1, add_sin_cos):
                    tmp = np.arange((g - 1) * 4 + j * 16, g * 4 + j * 16)
                    finger = np.concatenate([finger, tmp])
            q_i = q_all[:, finger].repeat(x.shape[0], 1)
            input_data = torch.cat([q_i, x], 1)

        with torch.no_grad():
            output = self.net(input_data)
            output, _ = torch.min(output, 1)
        output = output.cpu().numpy()
        return output


# def recover_dis_grad(output, grad,  dis_scale, mode='cpu'):
#     if mode == 'cpu':
#         output = np.copy(output)
#     else:  # 'gpu'
#         pass
#     for i in range(dis_scale.shape[1]):
#         neg = output[:, i] < 0
#         output[neg, i] = - output[neg, i] * dis_scale[0, i]
#         grad[]
#         pos = output[:, i] > 0
#         output[pos, i] = output[pos, i] * dis_scale[1, i]
#     return output

def recover_dis(output, dis_scale, mode='cpu'):
    if mode == 'cpu':
        output = np.copy(output)
    else:  # 'gpu'
        pass
    for i in range(dis_scale.shape[1]):
        neg = output[:, i] < 0
        output[neg, i] = - output[neg, i] * dis_scale[0, i]
        pos = output[:, i] > 0
        output[pos, i] = output[pos, i] * dis_scale[1, i]
    return output


def validate_results_new(result, result_nn, dis_scale, real_dis=False, collision_detect=0):
    if not real_dis:
        result_nn = recover_dis(result_nn, dis_scale)
    error = result_nn - result
    print('     1. dis error mean', np.mean(np.sqrt(error * error), axis=0))
    print('     1. dis error std', np.std(np.sqrt(error * error), axis=0))
    print('     1. dis error mean overall', np.mean(np.sqrt(error * error)))
    print('     1. dis error std overall', np.std(np.sqrt(error * error)))
    dis_bool = result <= 0
    dis_bool_p = result_nn <= collision_detect
    c1 = np.equal(dis_bool, dis_bool_p)
    c1 = np.sum(c1) / c1.shape[0] / c1.shape[1]
    print('     2. collision accuracy for all dis', c1)

    # only compare the minimal distance
    dis_bool = np.min(result, axis=1) <= 0
    dis_bool_p = np.min(result_nn, axis=1) <= collision_detect
    c = np.equal(dis_bool, dis_bool_p)
    c2 = np.sum(c) / len(c)
    print('     3. collision accuracy for min dis', c2)

    true_neg = dis_bool_p[dis_bool]

    dis_bool = np.min(result, axis=1) > 0
    dis_bool_p = np.min(result_nn, axis=1) > collision_detect
    true_pos = dis_bool_p[dis_bool]
    assert np.sum(np.logical_and(dis_bool, dis_bool_p)) == np.sum(true_pos)
    assert len(true_pos) == np.sum(dis_bool)

    FN = 1 - np.sum(true_pos) / len(true_pos)
    FP = 1 - np.sum(true_neg) / len(true_neg)
    print('False positive:', FP)
    print('False negative:', FN)


def validate_nn_results(result, result_nn, max_dis=None):
    if max_dis is not None:
        result_nn[result_nn <= 0] = 0
        result[result <= 0] = 0
        error = result_nn - result
        # # remove the distance while both pred and real distances are bigger than 0.5
        error[np.logical_and(result > 0.5, result_nn > 0.5)] = 0
        error = error * max_dis
        print('     1. dis error mean', np.sqrt(np.mean(error * error, axis=0)))
        # print('     1. dis error std', np.std(error, axis=0))

    dis_bool = result <= 0
    dis_bool_p = result_nn <= 0  # 0.001/max_dis
    c = np.equal(dis_bool, dis_bool_p)
    print('     2. collision accuracy for 10 dis', np.sum(c) / c.shape[0] / c.shape[1])

    # only compare the minimal distance
    dis_bool = np.min(result, axis=1) <= 0
    # result_nn[result_nn <= 0.001 * max_dis] = 0
    dis_bool_p = np.min(result_nn, axis=1) <= 0
    c = np.equal(dis_bool, dis_bool_p)
    print('     3. collision accuracy for min dis', np.sum(c) / len(c))

    true_neg = dis_bool_p[dis_bool]

    dis_bool = np.min(result, axis=1) > 0
    dis_bool_p = np.min(result_nn, axis=1) > 0
    true_pos = dis_bool_p[dis_bool]

    print('True positive:', np.sum(true_pos) / len(true_pos))
    print('True negative:', np.sum(true_neg) / len(true_neg))
