#z轴有约束
import mujoco
import numpy as np

class InverseKinematicsSolver:
    """
    逆运动学求解器：包含位置 + joint7 轴向方向约束
    """
    def __init__(self, model, damping=0.001, max_iter=100):
        self.model = model
        self.damping = damping
        self.max_iter = max_iter
        self.ee_site_name = "link_tcp"      # 末端位姿参考 site
        self.joint7_body_name = "link7"     # joint7 所属 body 名
        self.nq = 7                         # 只解前7个关节

    def solve(self, data, target_pos, target_axis_world=None,
              weight_ori=1.0, target_vel=None, dt=0.002):
        """
        参数:
            target_pos: (3,) world 目标位置
            target_axis_world: (3,) world 中希望 joint7 轴线对齐的方向
            weight_ori: 姿态约束的权重
            dt: 数值步长
        返回:
            (q, qd)
        """
        site_id = self.model.site(self.ee_site_name).id
        joint7_body_id = self.model.body(self.joint7_body_name).id

        q = data.qpos[:self.nq].copy()

        for _ in range(self.max_iter):
            mujoco.mj_forward(self.model, data)

            # --- 位置误差 ---
            current_pos = data.site(site_id).xpos.copy()
            pos_err = target_pos - current_pos  # (3,)

            # --- 雅可比 Jp Jr ---
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data, jacp, jacr, site_id)
            Jp = jacp[:, :self.nq]
            Jr = jacr[:, :self.nq]

            # --- joint7 轴向约束 ---
            if target_axis_world is not None:
                R7 = data.body(joint7_body_id).xmat.reshape(3, 3)
                v_cur = R7 @ np.array([0., 0., 1.])
                v_cur /= np.linalg.norm(v_cur) + 1e-9
                v_des = target_axis_world / (np.linalg.norm(target_axis_world) + 1e-9)

                # small angle: e = v_cur × v_des
                e_ori = np.cross(v_cur, v_des)

                err = np.concatenate((pos_err, weight_ori * e_ori))     # (6,)
                J = np.vstack((Jp, weight_ori * Jr))                    # (6×7)
            else:
                err = pos_err
                J = Jp

            # stop if small
            if np.linalg.norm(pos_err) < 1e-4:
                break

            # damped least-squares
            JTJ = J.T @ J
            reg = (self.damping ** 2) * np.eye(self.nq)
            Jinv = np.linalg.pinv(JTJ + reg) @ J.T

            delta_q = Jinv @ err
            q += delta_q * dt

            # 限位
            q = np.clip(q, self.model.jnt_range[:self.nq, 0], self.model.jnt_range[:self.nq, 1])

            # 写回 data 以便下一轮 forward
            data.qpos[:self.nq] = q
            data.qvel[:self.nq] = 0

        # qd（不是必须）
        if target_vel is not None:
            if target_axis_world is not None:
                vel_stack = np.concatenate((target_vel, np.zeros(3)))
                qd = Jinv @ vel_stack
            else:
                qd = Jinv @ target_vel
        else:
            qd = np.zeros(self.nq)

        return q.copy(), qd.copy()
