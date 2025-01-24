"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import os
import sys
import math

sys.path.append("../")
from urdfpy import URDF
import trimesh
from collections import OrderedDict
import copy
from omegaconf import OmegaConf
import numpy as np
import open3d as o3d
import pytorch_kinematics as pk


class robot_kinematic:
    def __init__(self, urdf_path=None):
        self.conf = OmegaConf.load(self._get_file_path('config/robot_info.yaml'))

        if urdf_path is None:
            urdf_path = self._get_file_path(self.conf.robot_urdf_path)

        self.robot = URDF.load(urdf_path)
        self._robot_joints = {}

        self._joint_upper_bound = np.array(self.conf.joint_upper_bound)
        self._joint_lower_bound = np.array(self.conf.joint_lower_bound)

        self.joint_names = []
        self.link_names = []

        self.num_joints = None
        self.num_links = None

        self.robot_links_mesh = OrderedDict()
        self.robot_links_convex_mesh = OrderedDict()
        self.init_robot_info()

    def _get_file_path(self, filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    def init_robot_info(self):
        for joint in self.robot.joints:
            print('{} connects {} to {}'.format(joint.name, joint.parent, joint.child))
            self._robot_joints[joint.name] = 0.
            if joint.parent not in self.link_names:
                self.link_names.append(joint.parent)
            if joint.child not in self.link_names:
                self.link_names.append(joint.child)

        self.joint_names = list(self._robot_joints.keys())
        self.num_joints = len(self.joint_names)
        self.num_links = len(self.link_names)

        # meshes = self.robot.collision_trimesh_fk()
        meshes = self.robot.visual_trimesh_fk()
        for i in range(self.num_links):
            link_name = self.link_names[i]
            self.robot_links_mesh[link_name] = list(meshes.keys())[i].copy()
            self.robot_links_convex_mesh[link_name] = trimesh.convex.convex_hull(self.robot_links_mesh[link_name])

    def show_robot_meshes(self, convex=True, bounding_box=True):
        combined_meshes = self.get_combined_mesh(convex, bounding_box)
        combined_meshes.show()

    def get_combined_mesh(self, convex=False, bounding_box=False):
        if bool(self.robot_links_convex_mesh) is False:
            raise ValueError('Please init the robot first!')
        convex_meshes = []
        fk = self.robot.link_fk(cfg=self.robot_joints)

        if convex:
            robot_meshes = self.robot_links_convex_mesh
        else:
            robot_meshes = self.robot_links_mesh

        for i in range(self.num_links):
            name = self.link_names[i]

            # use deepcopy for not messing up the original mesh
            mesh = robot_meshes[name].copy()
            mesh = mesh.apply_transform(fk[self.robot.links[i]])
            if bounding_box:
                mesh = mesh.bounding_box_oriented
                # print(trimesh.bounds.corners(mesh.bounds))
            convex_meshes.append(mesh)
        combined_meshes = trimesh.util.concatenate(convex_meshes)
        return combined_meshes

    def get_link_mesh_old(self, link_name, joint_positions=None):
        assert link_name in self.link_names
        if joint_positions is None:
            fk = self.robot.link_fk(cfg=self.robot_joints)
        else:
            robot_config = copy.deepcopy(self.robot_joints)
            for i, name in self.link_names:
                robot_config[name] = joint_positions[i]
            fk = self.robot.link_fk(cfg=self.robot_joints)

        robot_meshes = self.robot_links_mesh
        mesh = robot_meshes[link_name].copy()
        link_index = self.link_names.index(link_name)
        mesh = mesh.apply_transform(fk[self.robot.links[link_index]])
        return mesh

    def get_link_mesh(self, link_name, joint_positions=None):
        assert link_name in self.link_names
        if joint_positions is None:
            fk = self.robot.visual_trimesh_fk(cfg=self.robot_joints, links=[link_name])
        else:
            robot_config = copy.deepcopy(self.robot_joints)
            for i, name in self.link_names:
                robot_config[name] = joint_positions[i]
            fk = self.robot.visual_trimesh_fk(cfg=self.robot_joints, links=[link_name])

        mesh = list(fk.keys())[0]
        mesh = mesh.copy()
        mesh = mesh.apply_transform(list(fk.values())[0])

        return mesh

    def self_collision_detected(self):
        collision_detector = trimesh.collision.CollisionManager()
        fk = self.robot.link_fk(cfg=self.robot_joints)
        robot_meshes = self.robot_links_mesh

        for i in range(self.num_links):
            name = self.link_names[i]
            link_mesh = robot_meshes[name].copy()
            collision_detector.add_object(name=name,
                                          mesh=link_mesh,
                                          transform=fk[self.robot.links[i]])

        _, collision_set = collision_detector.in_collision_internal(return_names=True)
        # print(collision_set)
        collided_pair = collision_set - self.link_connection_set
        if collided_pair:
            return True
        else:
            return False

    def set_robot_joints(self, positions):
        assert len(positions) == 7
        # for i in range(self.num_joints):
        #     assert (self.joint_lower_bound[i] < positions[i]) & (positions[i] < self.joint_upper_bound[i])

        for i in range(self.num_joints):
            self._robot_joints[self.joint_names[i]] = positions[i]

    @staticmethod
    def trimesh2pcd(mesh, num_points, even_sample=False):
        if even_sample:
            sampled_points, _ = trimesh.sample.sample_surface_even(mesh=mesh,
                                                                   count=num_points)
        else:
            sampled_points, _ = trimesh.sample.sample_surface(mesh=mesh,
                                                              count=num_points)

        # print(sampled_points.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(sampled_points))
        pcd.remove_duplicated_points()
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        pcd.normalize_normals()
        pcd.orient_normals_towards_camera_location(pcd.get_center())

        return pcd

    def sample_random_robot_config(self):
        rand_jp = []

        for i in range(self.num_joints):
            rand_jp.append(np.random.uniform(self.joint_lower_bound[i],
                                             self.joint_upper_bound[i]))

        return np.asarray(rand_jp)

    @property
    def robot_joints(self) -> dict:
        return self._robot_joints

    @property
    def robot_q(self) -> list:
        q = []
        for name in self.joint_names:
            q.append(self.robot_joints[name])
        return q

    @property
    def joint_upper_bound(self):
        return self._joint_upper_bound

    @property
    def joint_lower_bound(self):
        return self._joint_lower_bound

    @property
    def link_connection_set(self):
        return {('lbr_iiwa_link_2', 'lbr_iiwa_link_3'),
                ('lbr_iiwa_link_4', 'lbr_iiwa_link_5'),
                ('lbr_iiwa_link_5', 'lbr_iiwa_link_6'),
                ('lbr_iiwa_link_3', 'lbr_iiwa_link_4'),
                ('lbr_iiwa_link_6', 'lbr_iiwa_link_7'),
                ('lbr_iiwa_link_0', 'lbr_iiwa_link_1'),
                ('lbr_iiwa_link_1', 'lbr_iiwa_link_2')}


class hand_kinematic:
    def __init__(self, urdf_path=None):
        self.conf = OmegaConf.load(self._get_file_path('config/hand_info.yaml'))

        if urdf_path is None:
            urdf_path = self._get_file_path(self.conf.robot_urdf_path)

        self.robot = URDF.load(urdf_path)
        self._robot_joints = {}

        self._joint_upper_bound = np.array(self.conf.joint_upper_bound)
        self._joint_lower_bound = np.array(self.conf.joint_lower_bound)

        self.joint_names = list(self.conf.JOINT_NAMES)
        self.link_names = []
        self.tip_joint_index = [4, 8, 12, 16]

        self.num_joints = None
        self.num_links = None
        self.num_controllable_joins = None

        self.robot_links_mesh = OrderedDict()
        self.robot_links_convex_mesh = OrderedDict()
        self.init_robot_info()

    def _get_file_path(self, filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    def init_robot_info(self):
        for joint in self.robot.joints:
            print('{} connects {} to {}'.format(joint.name, joint.parent, joint.child))
            self._robot_joints[joint.name] = 0.
            if joint.parent not in self.link_names:
                self.link_names.append(joint.parent)
            if joint.child not in self.link_names:
                self.link_names.append(joint.child)

        # self.joint_names = list(self._robot_joints.keys())
        self.num_joints = len(self.joint_names)
        self.num_links = len(self.link_names)
        self.num_controllable_joins = len(self.joint_names) - 4

        print(self.joint_names)
        print(self._robot_joints)

    def show_robot_meshes(self, convex=True, bounding_box=True):
        combined_meshes = self.get_combined_mesh(convex, bounding_box)
        combined_meshes.show()

    def get_combined_mesh(self, convex=False, bounding_box=False):

        convex_meshes = []
        fk = self.robot.visual_trimesh_fk(cfg=self.robot_joints)

        for tm in fk:
            pose = fk[tm]
            mesh = copy.deepcopy(tm)
            if convex:
                mesh = trimesh.convex.convex_hull(mesh)
            if bounding_box:
                mesh = mesh.bounding_box_oriented
            mesh.apply_transform(pose)
            convex_meshes.append(mesh)
        combined_meshes = trimesh.util.concatenate(convex_meshes)
        return combined_meshes

    def get_link_mesh(self, link_name, joint_positions=None):
        assert link_name in self.link_names
        if joint_positions is None:
            fk = self.robot.visual_trimesh_fk(cfg=self.robot_joints, links=[link_name])
        else:
            robot_config = copy.deepcopy(self.robot_joints)
            for i, name in self.link_names:
                robot_config[name] = joint_positions[i]
            fk = self.robot.visual_trimesh_fk(cfg=self.robot_joints, links=[link_name])

        mesh = list(fk.keys())[0]
        mesh = copy.deepcopy(mesh)
        mesh = mesh.apply_transform(list(fk.values())[0])

        return mesh

    def sample_random_robot_config(self, with_tip=False):
        rand_jp = []

        for i in range(self.num_controllable_joins):
            rand_jp.append(np.random.uniform(self.joint_lower_bound[i],
                                             self.joint_upper_bound[i]))

        if with_tip:
            return np.insert(np.array(rand_jp), self.tip_joint_index, 0.)
        else:
            return rand_jp

    def sample_synergy_finger_config(self):
        syn_fingers = np.random.rand(3, )
        thumber_finger = np.random.uniform(low=self._joint_lower_bound[12:],
                                           high=self._joint_upper_bound[12:],
                                           size=(4,))

        rand_jp = np.hstack([syn_fingers, thumber_finger])
        return rand_jp

    def set_finger_synergy_control(self, syn_q, syn_type=1):
        if syn_type == 1:
            assert len(syn_q) == 7
            syn_q = list(syn_q)
            self.set_hand_joints(np.hstack([[0] + syn_q[:3],
                                             [0] + syn_q[:3],
                                             [0] + syn_q[:3],
                                             syn_q[3:]]))

        elif syn_type == 2:
            assert len(syn_q) == 3
            syn_q = list(syn_q)
            self.set_hand_joints(np.hstack([[0] + syn_q,
                                             [0] + syn_q,
                                             [0] + syn_q,
                                             [0.3, 0, 0, 0]]))

        else:
            raise ValueError

    def set_hand_joints(self, positions):
        # print("inside set robot joint", positions)
        assert len(positions) == 16
        mid_value = copy.deepcopy(positions[0:4])
        positions[0:4] = positions[8:12]
        positions[8:12] = mid_value
        positions = np.insert(np.array(positions), self.tip_joint_index, 0.)
        # print(positions)
        for i in range(self.num_joints):
            self._robot_joints[self.joint_names[i]] = positions[i]

    @property
    def robot_joints(self) -> dict:
        return self._robot_joints

    @property
    def robot_q(self) -> np.ndarray:
        q = []
        for name in self.joint_names:
            q.append(self.robot_joints[name])

        q = np.array(q)
        q = np.delete(q, [4, 9, 14, 19])
        # q[0: 4], q[8: 12] = q[8: 12], q[0: 4]

        return q

    @property
    def joint_upper_bound(self):
        return self._joint_upper_bound

    @property
    def joint_lower_bound(self):
        return self._joint_lower_bound


class KUKA_with_hand:
    def __init__(self, ee_offset=None):
        if ee_offset is None:
            ee_offset = [0., 0., 0.14]
        self.hand = hand_kinematic()
        self.arm = robot_kinematic()
        self.arm_chain = pk.build_serial_chain_from_urdf(
            open(self._get_file_path(self.arm.conf.robot_urdf_path)).read(),
            "lbr_iiwa_link_7")
        print(self.arm_chain)
        self.ee_offset = ee_offset
        self._ee_matrix = None
        self._link7_matrix = None

    def _get_file_path(self, filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    def sample_random_robot_config(self, syn_type=1):
        """
        sample random hand-arm config
        :param syn_type: syn_type is 0, 1, 2
        :return: arm hand config
        """
        if syn_type:
            arm_q = self.arm.sample_random_robot_config()
            hand_q = self.hand.sample_synergy_finger_config()
        else:
            arm_q = self.arm.sample_random_robot_config()
            hand_q = self.hand.sample_random_robot_config()

        return np.hstack([arm_q, hand_q])

    def set_system_config(self, q, syn_type=1):
        """

        :param q: 23, 14, 10 for syn_type = 0, 1, 2
        :param syn_type:
        :return:
        """
        if syn_type == 1:
            assert len(q) == 14, "Input configuration's length should be 7 + 3(synergy) + 4(thumb)"
        elif syn_type == 2:
            assert len(q) == 10,  "Input configuration's length should be 7 + 3(synergy) + no thumb"
        elif syn_type == 0:
            assert len(q) == 23, "16(full-hand) + 7(full-arm) combination"


        arm_config = q[:7]
        hand_config = q[7:]

        # set arm config
        self.arm.set_robot_joints(arm_config)

        # set hand config
        if syn_type:
            self.hand.set_finger_synergy_control(hand_config, syn_type)
        else:
            self.hand.set_hand_joints(hand_config)

    def get_system_combined_mesh(self, q=None):
        if q is not None:
            self.set_system_config(q)
        arm_mesh = self.arm.get_combined_mesh(convex=False, bounding_box=False)
        # arm_mesh.show()

        hand_mesh = self.hand.get_combined_mesh(convex=False, bounding_box=False)
        ee_pose = self.arm_chain.forward_kinematics(self.arm.robot_q,
                                                    end_only=True).get_matrix().cpu().numpy().squeeze()
        self._link7_matrix = ee_pose

        ee_offset = np.eye(4)
        ee_offset[:3, 3] = np.array(self.ee_offset)
        self._ee_matrix = np.matmul(ee_pose, ee_offset)

        hand_mesh.apply_transform(self.ee_matrix)

        combined_meshes = trimesh.util.concatenate([arm_mesh, hand_mesh])

        return combined_meshes

    @property
    def ee_matrix(self):
        ee_offset = np.eye(4)
        ee_offset[:3, 3] = np.array(self.ee_offset)
        self._ee_matrix = np.matmul(self.link7_matrix, ee_offset)
        return self._ee_matrix

    @property
    def link7_matrix(self):
        self._link7_matrix = self.arm_chain.forward_kinematics(self.arm.robot_q,
                                                               end_only=True).get_matrix().cpu().numpy().squeeze()

        # ee_offset = np.eye(4)
        # ee_offset[:3, 3] = np.array([0, 0, 0.05])
        # self._link7_matrix = np.matmul(self._link7_matrix, ee_offset)
        return self._link7_matrix

    @property
    def link72base_matrix(self):
        return np.linalg.inv(self.link7_matrix)

    @property
    def robot_q(self):
        arm_q = self.arm.robot_q
        hand_q = self.hand.robot_q
        return np.hstack([arm_q, hand_q])


def load_object_pcd(obj_ply, every_k=100, position=None, orientation=None) -> o3d.geometry.PointCloud:
    if orientation is None:
        orientation = [0, 0, 0, 1]
    if position is None:
        position = [0.3, 0, 0.5]
    obj_pcd = o3d.io.read_point_cloud(obj_ply)
    obj_pcd = obj_pcd.uniform_down_sample(every_k)
    obj_pcd.translate(position)
    # obj_pcd.rotate()

    return obj_pcd


def test_hand():
    robot = hand_kinematic()
    print(robot.robot_q)

    rand_q = robot.sample_synergy_finger_config()
    robot.finger_synergy_control(rand_q)
    robot.show_robot_meshes(convex=False, bounding_box=False)
    print(robot.robot_q)
    robot.robot.show(robot.robot_joints)


def test_arm():
    robot = robot_kinematic()
    rand_q = robot.sample_random_robot_config()
    robot.set_robot_joints(rand_q)
    robot.show_robot_meshes(convex=False, bounding_box=False)
    print(robot.robot_q)
    robot.robot.show(robot.robot_joints)

    scene = trimesh.Scene()

    link1 = robot.get_link_mesh("lbr_iiwa_link_0")
    link2 = robot.get_link_mesh("lbr_iiwa_link_1")
    link3 = robot.get_link_mesh("lbr_iiwa_link_2")
    link4 = robot.get_link_mesh("lbr_iiwa_link_3")
    link5 = robot.get_link_mesh("lbr_iiwa_link_4")
    link6 = robot.get_link_mesh("lbr_iiwa_link_5")

    scene.add_geometry(link1)
    scene.add_geometry(link2)
    scene.add_geometry(link3)
    scene.add_geometry(link4)
    scene.add_geometry(link5)
    scene.add_geometry(link6)
    combined = robot.get_combined_mesh()
    scene.add_geometry(combined)
    scene.show()


if __name__ == "__main__":
    arm_hand_sys = KUKA_with_hand()
    q = arm_hand_sys.sample_random_robot_config(syn_type=1)
    print(q)
    q = [0.15174446, 0.6, -0.48907439, -1.9, 1.95646813, 0, 3.14,
         0.44646047, 1.0629986, 0.10425064,
         0.3255768, 1.00260788, 0.5091602, 0.83114794]

    arm_hand_sys.set_system_config(q, syn_type=1)
    mesh = arm_hand_sys.get_system_combined_mesh()
    mesh.show()
