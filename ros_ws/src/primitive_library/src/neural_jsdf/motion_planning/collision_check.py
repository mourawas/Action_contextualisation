"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import time
from JSDF import arm_hand_JSDF
from time import time
import numpy as np
import random

from .kdtree import KDTree


class SimpleTree:

    def __init__(self, dim):
        self._parents_map = {}
        self._kd = KDTree(dim)

    def __len__(self):
        return len(self._kd)

    def insert_new_node(self, point, parent=None):
        node_id = self._kd.insert(point)
        self._parents_map[node_id] = parent

        return node_id

    def get_parent(self, child_id):
        return self._parents_map[child_id]

    def get_point(self, node_id):
        return self._kd.get_node(node_id).point

    def get_nearest_node(self, point):
        return self._kd.find_nearest_point(point)

    def construct_path_to_root(self, leaf_node_id):
        path = []
        node_id = leaf_node_id
        while node_id is not None:
            path.append(self.get_point(node_id))
            node_id = self.get_parent(node_id)

        return path

    def get_num_nodes(self):
        return len(self._parents_map)


class robot_RRT_connect:
    def __init__(self, sample_num=100, min_dist=0.02, env_points=None, use_GPU=True):
        assert env_points is not None, "Please define env obstacles"

        self.min_dist = min_dist
        self.sample_num = sample_num
        self.env_points = np.array(env_points)

        self.jsdf = arm_hand_JSDF(use_GPU=use_GPU)
        self._joint_limits_low = np.asarray(self.jsdf.arm_jsdf.robot.joint_lower_bound)
        self._joint_limits_high = np.asarray(self.jsdf.arm_jsdf.robot.joint_upper_bound)
        self.num_dof = len(self._joint_limits_low)
        self.len_p = len(self.env_points)

        self._q_step_size = 0.04  # Default：0.015
        self._max_n_nodes = int(1e5)
        self._smoothed_nodes = 1

    def sample_valid_joints(self):
        q = np.random.random() * (self.joint_limits_high - self.joint_limits_low) + self.joint_limits_low
        return q

    def _is_seg_valid(self, q0, q1):
        qs = np.linspace(q0, q1, round(np.linalg.norm(q1 - q0) / self._q_step_size))
        # res = self.collision_check(qs)
        for q in qs:
            res = self.collision_check(q)
            if res:
                return False
        return True
        # print("qs", qs)
        # res = self.batch_collision_check(qs)
        # return res

    def getDistance(self, p):
        # print(p)
        dist = 0
        prev = p[0]
        for q in p[1:]:
            dist += np.linalg.norm(q - prev)
            prev = q
        return dist

    def collision_check(self, q):
        hand_q = np.zeros((16,))
        hand_q[-4] = 0.3
        q = np.hstack((q, hand_q))

        batch_q = np.repeat([q], self.len_p, axis=0)

        batch_qp = np.concatenate((batch_q, self.env_points), axis=1)

        # print("input shape", batch_qp.shape)

        signed_dist = self.jsdf.calculate_signed_distance_raw_input(batch_qp).detach().cpu().numpy()
        # print(signed_dist)
        # signed_dist = 0.05
        if signed_dist < self.min_dist:
            return True
        else:
            return False

    def smoothPath(self, path):
        for num_smoothed in range(self._smoothed_nodes):
            tree = SimpleTree(len(path[0]))
            i = random.randint(0, len(path) - 2)
            j = random.randint(i + 1, len(path) - 1)
            q_reach, q_reach_id = self.constrained_extend(tree, path[i], path[j], None)
            if not (q_reach == path[j]).all():
                continue
            # print(i, j, q_reach_id)
            temp_path = tree.construct_path_to_root(q_reach_id)
            # print(temp_path[::-1])
            # print(path[i:j+1])
            if self.getDistance(temp_path) < self.getDistance(path[i:j + 1]):
                path = path[:i + 1] + temp_path[::-1] + path[j + 1:]
        return path

    def constrained_extend(self, tree, q_near, q_target, q_near_id):
        '''
        TODO: Implement extend for RRT Connect
        - Only perform self.project_to_constraint if constraint is not None
        - Use self._is_seg_valid, self._q_step_size
        '''
        qs = q_near
        qs_id = q_near_id
        while True:
            if (q_target == qs).all():
                return qs, qs_id
            qs_old = qs
            qs_old_id = qs_id

            qs = qs + min(self._q_step_size, np.linalg.norm(q_target - qs)) * (q_target - qs) / np.linalg.norm(
                q_target - qs)

            if not self.collision_check(qs) and self._is_seg_valid(qs_old, qs):
                qs_id = tree.insert_new_node(qs, qs_id)
            else:
                return qs_old, qs_old_id

    def plan(self, q_start, q_target):
        tree_0 = SimpleTree(len(q_start))
        tree_0.insert_new_node(q_start)

        tree_1 = SimpleTree(len(q_target))
        tree_1.insert_new_node(q_target)

        q_start_is_tree_0 = True
        reached_target = False

        s = time()
        for n_nodes_sampled in range(self._max_n_nodes):
            if n_nodes_sampled > 0 and n_nodes_sampled % 20 == 0:
                print('RRTC: Sampled {} nodes'.format(n_nodes_sampled))
            q_rand = self.sample_valid_joints()
            # print(q_rand.shape)
            node_id_near_0 = tree_0.get_nearest_node(q_rand)[0]
            q_near_0 = tree_0.get_point(node_id_near_0)
            qa_reach, qa_reach_id = self.constrained_extend(tree_0, q_near_0, q_rand, node_id_near_0)

            node_id_near_1 = tree_1.get_nearest_node(qa_reach)[0]
            q_near_1 = tree_1.get_point(node_id_near_1)
            qb_reach, qb_reach_id = self.constrained_extend(tree_1, q_near_1, qa_reach, node_id_near_1)

            if np.linalg.norm(qa_reach - qb_reach) < self._q_step_size and self._is_seg_valid(qa_reach, qb_reach):
                reached_target = True
                break

            q_start_is_tree_0 = not q_start_is_tree_0
            tree_0, tree_1 = tree_1, tree_0

        print('RRTC: {} nodes extended in {:.2f}s'.format(len(tree_0) + len(tree_1), time() - s))

        # if not q_start_is_tree_0:
        #     tree_0, tree_1 = tree_1, tree_0

        if reached_target:
            tree_0_backward_path = tree_0.construct_path_to_root(qa_reach_id)
            tree_1_forward_path = tree_1.construct_path_to_root(qb_reach_id)

            # q0 = tree_0_backward_path[0]
            # q1 = tree_1_forward_path[0]
            # tree_01_connect_path = np.linspace(q0, q1, int(np.linalg.norm(q1 - q0) / self._q_step_size))[1:].tolist()
            if not q_start_is_tree_0:
                path = tree_1_forward_path[::-1] + tree_0_backward_path
            else:
                path = tree_0_backward_path[::-1] + tree_1_forward_path
            print('RRTC: Found a path! Path length is {}.'.format(len(path)))
        else:
            path = []
            print('RRTC: Was not able to find a path!')
        print('RRTC: Start path smooth')
        # path = self.smoothPath(path)
        print('RRTC: Path length after path smooth is {}.'.format(len(path)))

        return path

    @property
    def joint_limits_low(self):
        return self._joint_limits_low

    @property
    def joint_limits_high(self):
        return self._joint_limits_high


class Robot_explore:
    def __init__(self, initial_state=None, goal_state=None, sample_num=10000, tol_res=0.001, min_dist=0.03,
                 use_GPU=True,
                 max_iter_num=1000):
        self.min_dist = min_dist
        self.initial_state = np.array(initial_state)
        self.goal_state = np.array(goal_state)
        self.current_state = self.initial_state
        self.state_sequence = []
        self.tol_res = tol_res
        self.current_res = np.inf
        self.jsdf = arm_hand_JSDF(use_GPU=use_GPU)
        self.joint_limits_low = np.asarray(self.jsdf.arm_jsdf.robot.joint_lower_bound)
        self.joint_limits_high = np.asarray(self.jsdf.arm_jsdf.robot.joint_upper_bound)
        self.max_iter_num = max_iter_num
        self.sample_num = sample_num

    def sample_nearby_states(self, sample_num=10000):
        temp_state = np.asarray(self.current_state)
        temp_upper_bound = temp_state + 0.1
        temp_lower_bound = temp_state - 0.1

        for i, bound in enumerate(temp_upper_bound):
            if bound > self.joint_limits_high[i]:
                temp_upper_bound[i] = self.joint_limits_high[i]

        for i, bound in enumerate(temp_lower_bound):
            if bound < self.joint_limits_low[i]:
                temp_lower_bound[i] = self.joint_limits_low[i]

        sampled_states = []
        for i in range(sample_num):
            sampled_states.append(np.random.uniform(temp_lower_bound,
                                                    temp_upper_bound))

        return sampled_states

    def collision_check(self, batch_states, check_points):
        batch_states = np.array(batch_states)
        check_points = np.array(check_points)
        if batch_states.shape == (7,):
            batch_states = np.expand_dims(batch_states, axis=0)
        batch_len = len(batch_states)

        hand_states = np.zeros((batch_len, 16))
        batch_states = np.hstack((batch_states, hand_states))

        if check_points.shape == (3,):
            check_points = np.expand_dims(check_points, axis=0)

        free_state_indices = []
        for point in check_points:
            signed_dist = self.jsdf.calculate_signed_distance_raw(batch_states, point).detach().cpu().numpy()
            free_state_indices.append(np.argwhere(signed_dist > self.min_dist).squeeze())

        free_state = free_state_indices[0]
        for state in free_state_indices:
            free_state = np.intersect1d(free_state,
                                        state)

        collision_free_states = batch_states[free_state, :7]

        return collision_free_states

    def distance_check(self, batch_states):
        distances = []
        for state in batch_states:
            res = self.goal_state - state
            dist = np.sum(np.power(res, 2))
            distances.append(dist)

        distances = np.array(distances)
        min_dist = np.min(distances)
        min_index = np.argmin(distances)

        return batch_states[min_index], min_dist

    def exploration(self, env_points: np.ndarray):
        print("Initial state is ", self.initial_state)
        print("Target state is ", self.goal_state)

        iter_num = 0
        while self.current_res > self.tol_res:
            sample_batches = self.sample_nearby_states(sample_num=self.sample_num)
            collision_free_states = self.collision_check(sample_batches, env_points)
            next_state, cur_dist = self.distance_check(collision_free_states)
            self.state_sequence.append(next_state)
            self.current_res = cur_dist
            self.current_state = next_state
            iter_num += 1
            print(iter_num)
            if iter_num > self.max_iter_num:
                break

        print("Exploration finished!")


if __name__ == "__main__":
    rrt = Robot_explore(initial_state=[0.] * 7, goal_state=[1.] * 7)
    # rand_jp = rrt.sample_nearby_states()
    # # rand_p = np.random.rand(100, 3)
    rand_p = np.array([[0, 0, 1.3], [0, 0, 1.2], [0, 0, 1.05]])

    start_t = time.time()
    rrt.exploration(rand_p)
    print(time.time() - start_t)
    print(rrt.current_state)
    print(rrt.state_sequence)
