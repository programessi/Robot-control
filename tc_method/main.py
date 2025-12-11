import sys, os
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

from ik_solver import InverseKinematicsSolver

# -------------------------
# Simulation parameters
# -------------------------
DT = 0.002
T_TOTAL = 10.0
N = int(T_TOTAL / DT)

# -------------------------
# Load model
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "..", "scene.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
model.opt.timestep = DT
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# IK solver
solver = InverseKinematicsSolver(model)

ee_id = model.site("link_tcp").id
base_id = model.body("link_base").id

# PD gains
Kp = np.array([1200, 1100, 900, 800, 600, 400, 300])
Kd = 0.1 * Kp

# -------------------------
# Circle trajectory (base local)
# -------------------------
R = 0.2
FREQ = 0.2
OMEGA = 2 * np.pi * FREQ
CENTER_LOCAL = np.array([0.45, 0.0, 0.45])

def circ_local(t):
    th = OMEGA * t
    pos = CENTER_LOCAL + np.array([0, R * np.cos(th), R * np.sin(th)])
    vel = np.array([0, -R * OMEGA * np.sin(th), R * OMEGA * np.cos(th)])
    return pos, vel

# -------------------------
# Logging arrays
# -------------------------
pos_tar = np.zeros((N, 3))
pos_act = np.zeros((N, 3))
joint = np.zeros((N, 7))
qd_cmd = np.zeros((N, 7))

# -------------------------
# Start simulation
# -------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:

    for i in range(N):
        t = i * DT

        mujoco.mj_forward(model, data)

        # ---- 1) target in base local frame ----
        p_loc, v_loc = circ_local(t)

        base_pos = data.xpos[base_id].copy()
        base_rot = data.xmat[base_id].reshape(3, 3).copy()

        p_w = base_pos + base_rot @ p_loc
        v_w = base_rot @ v_loc

        # ---- 2) steering axis: use Jr[:,6] ----
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
        axis = jacr[:, 6]
        if np.linalg.norm(axis) < 1e-6:
            axis = np.array([1, 0, 0])

        # ---- 3) IK solve ----
        q_des, qd_des = solver.solve(
            data,
            target_pos=p_w,
            v_pos=v_w,
            axis_world=axis,
            omega=OMEGA
        )

        # ---- 4) PD control ----
        pos_err = q_des - data.qpos[:7]
        vel_err = qd_des - data.qvel[:7]
        tau = Kp * pos_err + Kd * vel_err
        tau = np.clip(tau, -50, 50)

        data.ctrl[:7] = tau
        mujoco.mj_step(model, data)
        viewer.sync()

        # ---- 5) log ----
        pos_tar[i] = p_w
        pos_act[i] = data.site_xpos[ee_id].copy()
        joint[i] = data.qpos[:7]
        qd_cmd[i] = qd_des

# -------------------------
# Plot results
# -------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(pos_tar[:,0], pos_tar[:,1],'r--')
plt.plot(pos_act[:,0], pos_act[:,1],'b')
plt.title("XY view")

plt.subplot(1,2,2)
plt.plot(pos_tar[:,1], pos_tar[:,2],'r--')
plt.plot(pos_act[:,1], pos_act[:,2],'b')
plt.title("YZ view")
plt.show()
