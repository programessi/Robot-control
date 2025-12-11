import mujoco
import numpy as np

class InverseKinematicsSolver:
    """
    Fully 7-DoF IK with 4×7 velocity constraints:
      - 3 position constraints
      - 1 steering-axis angular velocity constraint
    """

    def __init__(self, model, damping=0.05, max_iter=80):
        self.model = model
        self.damping = damping
        self.max_iter = max_iter
        self.ee_site = model.site("link_tcp").id

    def solve(self, data, target_pos, v_pos, axis_world, omega, dt=0.002):
        """
        Returns:
            q_target (7,)
            qd_target (7,)
        """

        # -------------------------
        # 1) iterative position IK
        # -------------------------
        q = data.qpos[:7].copy()

        for _ in range(self.max_iter):
            mujoco.mj_forward(self.model, data)
            cur = data.site_xpos[self.ee_site].copy()
            err = target_pos - cur

            if np.linalg.norm(err) < 1e-4:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ee_site)
            Jp = jacp[:, :7]

            # DLS inverse
            JJ = Jp.T @ Jp + (self.damping**2) * np.eye(7)
            dq = np.linalg.solve(JJ, Jp.T @ err)
            q += dq * dt

            # clip to joint limits
            try:
                jmin = self.model.jnt_range[:7, 0]
                jmax = self.model.jnt_range[:7, 1]
                q = np.clip(q, jmin, jmax)
            except:
                pass

            data.qpos[:7] = q

        # -------------------------
        # 2) velocity-level 4×7 IK
        # -------------------------
        mujoco.mj_forward(self.model, data)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ee_site)

        Jp = jacp[:, :7]             # 3×7
        Jr = jacr[:, :7]             # 3×7

        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-9)

        # scalar projection of Jr onto axis
        Js = (axis_world.reshape(1, 3) @ Jr)  # (1×7)

        # full 4×7 Jacobian
        J_full = np.vstack([Jp, Js])

        # target velocity 4×1
        v_full = np.hstack([v_pos, [omega]]).reshape(4, 1)

        lam = self.damping
        JJt = J_full @ J_full.T
        inv_term = np.linalg.inv(JJt + (lam**2) * np.eye(4))

        qdot = (J_full.T @ (inv_term @ v_full)).reshape(7)

        return q.copy(), qdot.copy()
