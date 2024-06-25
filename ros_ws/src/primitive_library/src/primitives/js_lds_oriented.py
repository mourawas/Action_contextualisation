import numpy as np
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation
import typing as tp


from llm_common import helpers as llmh

from .js_lds import JS_LDS


class JS_LDS_ORIENTED(JS_LDS):

    def _compute_desired_qd(self, q: np.ndarray, desired_qd: np.ndarray) -> tp.Tuple[np.ndarray, bool]:
        desired_qd = self._project_in_null_rot_space(q, desired_qd)
        return desired_qd, False

    @property
    def orientation_factor(self) -> float:
        return self._orientation_factor

    @orientation_factor.setter
    def orientation_factor(self, factor: float) -> None:
        if 0 <= factor <= 1:
            self._orientation_factor = factor
        else:
            raise ValueError("Orientation factor must be between 0 and 1")

    @property
    def cartesian_goal(self) -> np.ndarray:
        return self._cartesian_goal

    @cartesian_goal.setter
    def cartesian_goal(self, goal: np.ndarray) -> None:

        # Get the current rotation at the end effector
        self._new_iiwa_js_event.wait()  # Wait until we are sure we have an accurate joint position
        q_iiwa = self.x_iiwa

        if len(goal) == 4 or len(goal) == 3:

            # Find current ee position
            goal_se3 = self.fkine(q_iiwa, end=self._robot_ee_link).A
            goal_rot = Rotation.from_matrix(goal_se3[:3, :3])

            # Get goal as matrix
            if len(goal) == 4:
                xy_rot_proj = goal_rot.as_matrix()[:, :2].transpose()
                xy_rot_proj[:, -1] = 0
                (inverse_yaw_rot, _) = Rotation.align_vectors(np.array([[1, 0, 0], [0, 1, 0]]) * np.linalg.norm(xy_rot_proj), xy_rot_proj)
                goal_rot_no_yaw = inverse_yaw_rot * goal_rot
                goal_rot = Rotation.from_rotvec(np.array([0, 0, goal[-1]]), degrees=True) * goal_rot_no_yaw

            goal_se3[:3, :3] = goal_rot.as_matrix()
            goal_se3[:3, 3] = goal[:3]

        elif len(goal) == 7:
            goal_se3 = llmh.cartesian_to_se3(goal)

        else:
            raise ValueError("Cartesian goal is expected in 3 or 4dimensions (x, y, z) and optionally yaw in degrees")

        # Run IK
        (iiwa_joint_goal, success) = self.compute_ik_iiwa(goal_se3, q_iiwa, verbose=True)

        # Report on the solution
        if success:
            self._failed_ik = False
            self._cartesian_goal = goal
            if self.let_go:
                q_allegro = self.qlim[0][self._n_rbt:]
                q_allegro[12] = self.qlim[1][self._n_rbt + 12]
            else:
                q_allegro = self.qz[self._n_rbt:]

            self._joint_goal = np.concatenate((iiwa_joint_goal, q_allegro))
        else:
            self._failed_ik = True

    def _project_in_null_rot_space(self, q: np.ndarray, desired_qd: np.ndarray) -> np.ndarray:

        # Get jacobian rotation null space
        jacob = self.jacobe(q, end=self._robot_ee_link)
        # breakpoint()
        jacob_rot = jacob[(3, 5), :]
        jacob_rot_ns = null_space(jacob_rot)

        # Project speed in null space
        desired_qd_ns = jacob_rot_ns @ (jacob_rot_ns.transpose() @ desired_qd[:self._n_rbt])

        # Compute residual speed in non-null space
        desired_qd_residual = desired_qd.copy()
        desired_qd_residual[:self._n_rbt] = desired_qd[:self._n_rbt] - desired_qd_ns

        # Concatenate full speed of null space and hand
        full_desired_qd_ns = desired_qd.copy()
        full_desired_qd_ns[:self._n_rbt] = desired_qd_ns

        # Apply orientation factor
        if self.orientation_factor is not None:
            desired_qd = full_desired_qd_ns + desired_qd_residual * (1-self.orientation_factor)
        else:
            desired_qd = full_desired_qd_ns

        return desired_qd


