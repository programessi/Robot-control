import mujoco
import numpy as np
from numpy.linalg import norm

def rotmat_to_rotvec(R):
    """把 3x3 旋转矩阵转成旋转向量 (axis * angle)."""
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz]) / (2.0 * np.sin(angle) + 1e-12)
    return axis * angle

class InverseKinematicsSolver:
    """
    6D IK: 支持位置 + 完整姿态 (target_rot is 3x3 rotation matrix).
    输出 q_target (7,) 和 qd (7,).
    不会直接写回 data.qpos。
    """
    def __init__(self, model, damping=0.01, max_iter=80):
        self.model = model
        self.damping = damping
        self.max_iter = max_iter
        self.ee_site = "link_tcp"
        self.j7_body = "link7"
        self.nq = 7

    def solve(self, data, target_pos, target_rot=None, weight_ori=1.0, dt=1.0):
        """
        target_pos: (3,)
        target_rot: None or 3x3 rotation matrix (desired orientation of joint7 frame)
        weight_ori: orientation weight
        dt: step scale for delta_q (numeric stability)
        """
        q = data.qpos[:self.nq].copy()
        site_id = self.model.site(self.ee_site).id
        j7_id = self.model.body(self.j7_body).id

        for it in range(self.max_iter):
            # use temporary data to evaluate forward kinematics for candidate q
            tmp = mujoco.MjData(self.model)
            tmp.qpos[:] = data.qpos[:]           # preserve full-state reference
            tmp.qpos[:self.nq] = q
            mujoco.mj_forward(self.model, tmp)

            # position error
            curr_pos = tmp.site(site_id).xpos.copy()
            err_pos = target_pos - curr_pos

            # orientation error -> rotation vector (3,)
            if target_rot is not None:
                # current joint7 rotation matrix
                R7 = tmp.body(j7_id).xmat.reshape(3,3).copy()
                R_err = target_rot @ R7.T
                err_ori = rotmat_to_rotvec(R_err)   # axis*angle
            else:
                err_ori = np.zeros(3)

            # stacked error (6,)
            err = np.concatenate([err_pos, weight_ori * err_ori])

            # stopping
            if norm(err) < 1e-6:
                break

            # jacobians for site (linear and angular)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, tmp, jacp, jacr, site_id)

            Jp = jacp[:, :self.nq]
            Jr = jacr[:, :self.nq]
            J = np.vstack([Jp, weight_ori * Jr])   # 6 x 7

            # damped least squares: use J J^T + lambda^2 I to invert stably
            JJt = J @ J.T
            inv = np.linalg.inv(JJt + (self.damping**2) * np.eye(JJt.shape[0]))
            Jinv = J.T @ inv   # 7 x 6

            delta_q = Jinv @ err
            q = q + dt * delta_q

            q = np.clip(q,
                        self.model.jnt_range[:self.nq, 0],
                        self.model.jnt_range[:self.nq, 1])

        # qd: approximate or zero
        qd = np.zeros(self.nq)
        return q.copy(), qd.copy()
