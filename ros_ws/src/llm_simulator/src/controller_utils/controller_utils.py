"""
Control tools for the iiwa and allegro hand.

basic functions:
 - read all states from MuJoCo simulator and store them as properties (see the end part of this file)
 - send torque commands to the robotic system
 - run one step simulation in MuJoCo
 - reset/reposition all robots/objects

Controllers:
 - impedance controller in Cartesian space for iiwa
 - impedance controller in joint space for allegro hand
 - A coupled DS to reach an attractor for both iiwa and hand


Notes:
 - All quatersions are in (w x y z) order
 - Fingers of hand are in ('index', 'middle', 'ring', 'thumb') order
"""
import numpy as np
import mujoco
import tools.rotations as rot
import kinematics.allegro_hand_sym as allegro
import trimesh
from sklearn.cluster import k_means
from pathlib import Path
import pickle

from llm_common import helpers as llmh
from llm_common import utils as llmu


class Robot:
    def __init__(self, m: mujoco._structs.MjModel, d: mujoco._structs.MjModel, view, obj_names=[], auto_sync=True,
                 q0=None, description_folder='description', kinematics_folder='kinematics'):
        self.m = m
        self.d = d
        self.view = view
        self.auto_sync = auto_sync
        self.fingers = ['index', 'middle', 'ring', 'thumb']
        if len(obj_names):
            self.obj_names = obj_names  # Note that the order of objects must be the same as the .xml
            self.obj_id = {i: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, i) for i in self.obj_names}

        if q0 is None:
            # self.q0 = np.array( [-0.32032486, 0.02707055, -0.22881525, -1.309, 1.38608943, 0.5596685, -1.34659665 + np.pi])
            self.q0 = np.array( [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.modify_joint(self.q0)  # set the initial joint positions
        self.step()
        self.view.sync()
        self.viewer_setup()

        # hand kinematics
        self.hand = allegro.Robot(right_hand=False, description_folder=description_folder, kinematics_folder=kinematics_folder)
        self.fingertip_sites = ['index_site', 'middle_site', 'ring_site', 'thumb_site']  # These site points are the fingertip (center of semisphere) positions
        self._fitted_mesh_dict = {}

    def step(self):
        mujoco.mj_step(self.m, self.d)  # run one-step dynamics simulation

    def viewer_setup(self):
        """
        setup camera angles and distances
        These data is generated from a notebook, change the view direction you wish and print the view.cam to get desired ones
        :return:
        """
        self.view.cam.trackbodyid = 0  # id of the body to track ()
        # self.viewer.cam.distance = self.sim.model.stat.extent * 0.05  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.view.cam.distance = 0.6993678113883466  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.view.cam.lookat[0] = 0.55856114  # x,y,z offset from the object (works if trackbodyid=-1)
        self.view.cam.lookat[1] = 0.00967048
        self.view.cam.lookat[2] = 1.20266637
        self.view.cam.elevation = -21.105028822285007  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.view.cam.azimuth = 94.61867426942274  # camera rotation around the camera's vertical axis

    def send_torque(self, torque):
        """
        control joints by torque and send to mujoco, then run a step.
        input the joint control torque
        Here I imply that the joints with id from 0 to n are the actuators.
        so, please put robot xml before the object xml.
        todo, use joint_name2id to avoid this problem
        :param torque:  (n, ) numpy array
        :return:
        """

        self.d.ctrl[:len(torque)] = torque   # apply the control torque
        self.step()
        if self.auto_sync:
            self.view.sync()

    def send_torque_h(self, torque_h):
        """
        control joints of the hand by torque and send to mujoco, then run a step.
        input the joint control torque for the hand
        Here I imply that the joints with id from 0 to n are the actuators.
        so, please put robot xml before the object xml.
        todo, use joint_name2id to avoid this problem
        :param torque:  (n_h, ) numpy array
        :return:
        """
        new_ctrl = np.copy(self.d.ctrl)
        new_ctrl[7:23] = torque_h

        self.d.ctrl = new_ctrl   # apply the control torque
        self.step()
        if self.auto_sync:
            self.view.sync()

    def set_zero_speed(self):
        self.d.qvel[:23] = np.zeros((23,))

    def modify_joint(self, joints: np.ndarray) -> None:
        """
        :param joints: (7,) or (16,) or (23,), modify joints for iiwa or/and allegro hand
        :return:
        """
        assert len(joints) in [7, 16, 23]
        if len(joints) == 7:
            self.d.qpos[:7] = joints
        elif len(joints) == 16:
            self.d.qpos[7:23] = joints
        else:
            self.d.qpos[:23] = joints

    @property
    def q(self):
        """
        iiwa joint angles
        :return: (7, ), numpy array
        """
        return self.d.qpos[:7]  # noting that the order of joints is based on the order in *.xml file

    @property
    def qh(self):
        """
        hand angles: index - middle - ring - thumb
        :return:  (16, )
        """
        return self.d.qpos[7:23]

    @property
    def q_all(self):
        """
        iiwa joint angles and allegro hand angles
        :return: (23, )
        """
        return self.d.qpos[:23]

    @property
    def dq(self):
        """
        iiwa joint velocities
        :return: (7, )
        """
        return self.d.qvel[:7]

    @property
    def dqh(self):
        """
        hand angular velocities: index - middle - ring - thumb
        :return:  (16, )
        """
        return self.d.qvel[7:23]

    @property
    def dq_all(self):
        """
        iiwa and allegro joint velocities
        :return: (23, )
        """
        return self.d.qvel[:23]

    @property
    def ddq(self):
        """
        iiwa joint acc
        :return: (7, )
        """
        return self.d.qacc[:7]

    @property
    def ddqh(self):
        """
        hand angular acc: index - middle - ring - thumb
        :return:  (16, )
        """
        return self.d.qacc[7:23]

    @property
    def ddq_all(self):
        """
        iiwa and allegro joint acc
        :return: (23, )
        """
        return self.d.qacc[:23]

    @property
    def torque(self):
        return self.d.actuator_force[:7]

    @property
    def torque_h(self):
        return self.d.actuator_force[7:23]

    @property
    def torque_all(self):
        return self.d.actuator_force[:23]

    @property
    def xh(self):
        """
        hand fingertip poses: index - middle - ring - thumb
        All quaternions are in [w, x, y, z] ordery<
        :return: (4, 7)
        """
        poses = []
        for i in self.fingers:
            site_name = i + '_site'
            xpos = self.d.site(site_name).xpos
            xquat = rot.mat2quat(self.d.site(site_name).xmat.reshape(3, 3))
            poses.append(np.concatenate([xpos, xquat]))
        return np.vstack(poses)

    @property
    def x(self):
        """
        Cartesian position and orientation (quat) of the end-effector frame, from site
        return: (7, )
        """
        xpos = self.d.site('ee_site').xpos
        xquat = rot.mat2quat(self.d.site('ee_site').xmat.reshape(3, 3))
        return np.concatenate([xpos, xquat])

    @property
    def x_obj(self):
        """
        :return: [(7,),...] objects poses by list,
         // computed by mj_fwdPosition/mj_kinematics
        https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=xipos#mjdata
        """  # print(joint_id)
        poses = []
        for i in self.obj_names:
            poses.append(np.concatenate([self.d.body(i).xpos, self.d.body(i).xquat]))
        return poses

    @property
    def x_obj_dict(self):
        """
        :return: [(7,),...] objects poses by list,
         // computed by mj_fwdPosition/mj_kinematics
        https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=xipos#mjdata
        """  # print(joint_id)
        poses = {}
        for i in self.obj_names:
            poses[i] = (np.concatenate([self.d.body(i).xpos, self.d.body(i).xquat]))
        return poses

    def obj_pos(self, obj_name: str) -> np.ndarray:
        quat = self.d.body(obj_name).xquat
        return np.concatenate([self.d.body(obj_name).xpos, quat])

    def obj_com_pos(self, obj_name: str) -> np.ndarray:
        return self.d.body(obj_name).subtree_com

    @property
    def dx(self):
        """
            Cartesian velocities of the end-effector frame
            Compute site end-effector Jacobian
        :return: (6, )
        """
        dx = self.J @ self.dq
        return dx.flatten()

    @property
    def M(self):
        """
        get inertia matrix for iiwa in joint space
        :return:
        """
        M = np.zeros([self.m.nv, self.m.nv])
        mujoco.mj_fullM(self.m, M, self.d.qM)

        return M[:7, :7]

    @property
    def M_hand(self):
        """
        get inertia matrix for iiwa in joint space
        :return:
        """
        M = np.zeros([self.m.nv, self.m.nv])
        mujoco.mj_fullM(self.m, M, self.d.qM)
        return M[7:23, 7:23]

    @property
    def M_all(self):
        """
        get inertia matrix for iiwa in joint space
        :return:
        """
        M = np.zeros([self.m.nv, self.m.nv])
        mujoco.mj_fullM(self.m, M, self.d.qM)
        return M[:23, :23]

    @property
    def C(self):
        """
        for iiwa, C(qpos,qvel), Coriolis, computed by mj_fwdVelocity/mj_rne (without acceleration)
        :return: (7, )
        """
        return self.d.qfrc_bias[:7]

    @property
    def C_(self):
        """
        for all, C(qpos,qvel), Coriolis, computed by mj_fwdVelocity/mj_rne (without acceleration)
        :return: (nv, )
        """
        return self.d.qfrc_bias[:23]

    @property
    def C_allegro(self):
        """
        for all, C(qpos,qvel), Coriolis, computed by mj_fwdVelocity/mj_rne (without acceleration)
        :return: (nv, )
        """
        return self.d.qfrc_bias[7:23]

    def get_object_vertices(self, object_name: str):
        saved_object_vert = llmu.OBJECT_COLLIDER_FILE
        if not self._fitted_mesh_dict:
            if saved_object_vert.is_file():
                with open(str(saved_object_vert), 'rb') as f:
                    print('Loading meshes from disk')
                    self._fitted_mesh_dict = pickle.load(f)

        if object_name not in self._fitted_mesh_dict.keys():
            print(f"Fitting mesh for {object_name} it might take a few seconds")
            nb_geom = self.m.body(object_name).geomnum[0]
            geom_addr_0 = self.m.body(object_name).geomadr[0]
            obj_geom_addr = [adr for adr in range(geom_addr_0, geom_addr_0 + nb_geom)]
            geoms = [self.m.geom(i) if i != -1 else None for i in obj_geom_addr]
            data_ids = [geom.dataid if geom is not None else -1 for geom in geoms]

            centers = np.array([])
            radii = np.array([])

            for data_id in data_ids:
                if data_id == -1:
                    continue

                id = data_id[0]
                first_vertex = self.m.mesh_vertadr[id]
                nb_vertices = self.m.mesh_vertnum[id]

                new_vertices = self.m.mesh_vert[first_vertex:first_vertex + nb_vertices]
                mesh_pos = self.m.mesh_pos[id]
                mesh_quat = self.m.mesh_quat[id]
                vertices_extended = np.concatenate((new_vertices, np.ones((new_vertices.shape[0], 1))), axis=1)

                # Get vertices transform
                vertices_tf = llmh.mujoco_pos_quat_to_se3(mesh_pos, mesh_quat)

                # Transform vertices in world frame
                vertices_extended_tf = (vertices_tf @ vertices_extended.T).T

                # Make a trimesh object
                new_vertices = vertices_extended_tf[:, :3]
                first_face = self.m.mesh_faceadr[id]
                nb_faces = self.m.mesh_facenum[id]
                new_faces = self.m.mesh_face[first_face:first_face + nb_faces]
                msh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

                if not msh.is_empty:
                    # Sample and use k-means to find sphrical clusters
                    sampled_points = trimesh.sample.volume_mesh(msh, 250000)
                    if object_name == 'sink':
                        new_centers, new_labels, _ = k_means(sampled_points, 300, n_init=10)
                    else:
                        new_centers, new_labels, _ = k_means(sampled_points, 300, n_init=10)

                    clusters = []
                    cluster_labels = []
                    for label in set(new_labels):
                        clusters.append(sampled_points[new_labels == label, :])
                        cluster_labels.append(label)

                    for label, cluster in zip(cluster_labels, clusters):
                        fitted_center, new_radius, _ = trimesh.nsphere.fit_nsphere(cluster, new_centers[label])

                        if len(centers) == 0:
                            centers = np.expand_dims(fitted_center, 0)
                        else:
                            centers = np.concatenate((centers, np.expand_dims(fitted_center, 0)), axis=0)

                        if len(radii) == 0:
                            radii = np.array([new_radius])
                        else:
                            radii = np.concatenate((radii, np.array([new_radius])))

            if len(centers):
                self._fitted_mesh_dict[object_name] = (centers, radii)

            # Save new dictionnary
            with open(str(saved_object_vert), 'wb') as f:
                    pickle.dump(self._fitted_mesh_dict, f)
        else:
            centers, radii = self._fitted_mesh_dict[object_name]

        obj_tf = llmh.mujoco_pos_quat_to_se3(self.d.body(object_name).xpos, self.d.body(object_name).xquat)
        centers_extended = np.concatenate((centers, np.ones((centers.shape[0], 1))), axis=1)
        centers = (obj_tf @ centers_extended.T).T[:, :3]

        return centers, radii





