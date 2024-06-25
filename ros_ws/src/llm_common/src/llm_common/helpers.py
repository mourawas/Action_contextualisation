import numpy as np
from scipy.spatial.transform import Rotation


def pos_mat_to_se3(pos: np.ndarray, mat: np.ndarray) -> np.ndarray:

    pos = np.asarray(pos)
    mat = np.asarray(mat).reshape((3, 3))

    se3 = np.identity(4)
    se3[:3, :3] = mat
    se3[:3, 3] = pos

    return se3


def mujoco_pos_quat_to_se3(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:

    pos = np.asarray(pos)
    quat = np.asarray(quat)

    # Changing quaternion convention
    quat = np.array([quat[1], quat[2], quat[3], quat[0]])

    se3 = np.identity(4)
    se3[:3, :3] = Rotation.from_quat(quat).as_matrix()
    se3[:3, 3] = pos

    return se3


def se3_to_quaternion(se3_pose: np.ndarray) -> np.ndarray:
    r = Rotation.from_matrix(se3_pose)
    return r.as_quat()


def quaternion_to_se3(quaternion: np.ndarray) -> np.ndarray:
    r = Rotation.from_quat(quaternion)
    return r.as_matrix()


def cartesian_to_se3(catersian_pose: np.ndarray) -> np.ndarray:
    se3_mat = np.zeros((4, 4))
    se3_mat[3, 3] = 1
    se3_mat[:3, :3] = quaternion_to_se3(catersian_pose[3:])
    se3_mat[:3, 3] = catersian_pose[:3]
    return se3_mat


def se3_to_cartesian(se3_pose: np.ndarray) -> np.ndarray:
    trans_vec = se3_pose[:3, 3]
    quat_vec = se3_to_quaternion(se3_pose)
    return np.concatenate((trans_vec, quat_vec))

def se3_to_mujoco_cartesian(se3_pose: np.ndarray) -> np.ndarray:
    trans_vec = se3_pose[:3, 3]
    quat_vec = se3_to_quaternion(se3_pose[:3, :3])
    quat_vec = np.array([quat_vec[3], quat_vec[0], quat_vec[1], quat_vec[2]])
    return np.concatenate((trans_vec, quat_vec))

