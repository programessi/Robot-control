# 2025.12.11
# 圆心坐标修改为相对于基座的
# 画圆平面改成竖直平面，解耦机械臂末端控制
# 出现的问题：机械臂可以到达该平面，但是姿态不对
import sys
import os
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from IK.ik_solver import InverseKinematicsSolver

rcParams['axes.unicode_minus'] = False


# ========================= 仿真参数 =========================
SIM_DURATION = 10.0
TIMESTEP = 0.002

# ========================= 加载 MuJoCo 模型 =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, '..', 'scene.xml')

model = mujoco.MjModel.from_xml_path(xml_path)
model.opt.timestep = TIMESTEP
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)


# ========================= 初始化 IK 求解器 =========================
solver = InverseKinematicsSolver(model=model, damping=0.001, max_iter=100)


# ========================= 圆轨迹（相对于 base 的 local 坐标） =========================
def circular_trajectory_local(t, radius=0.2, freq=0.5,
                              center_local=np.array([0.45, 0.0, 0.45])):
    theta = 2 * np.pi * freq * t
    return center_local + np.array([
         0.0,
        radius * np.cos(theta),
        radius * np.sin(theta)
       
    ])


# ========================= 控制参数 (PD) =========================
Kp = np.array([1200, 1100, 900, 800, 600, 400, 300])
Kd = 0.1 * Kp


# ========================= 记录数据 =========================
num_steps = int(SIM_DURATION / TIMESTEP)
time_axis = np.arange(0, SIM_DURATION, TIMESTEP)[:num_steps]

target_positions = np.zeros((num_steps, 3))
actual_positions = np.zeros((num_steps, 3))
joint_angles = np.zeros((num_steps, 7))
control_signals = np.zeros((num_steps, 7))

# 获取 base body ID
base_body_id = model.body('link_base').id


# ========================= 主仿真循环 =========================
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while step < num_steps and viewer.is_running():

        mujoco.mj_forward(model, data)   # 保证 body 位姿是最新的

        # 当前时间
        t = step * TIMESTEP

        # ===== 1）生成 base 局部坐标下的圆轨迹 =====
        target_local = circular_trajectory_local(t)

        # ===== 2）将局部坐标转换到 world 坐标 =====
        base_pos_world = data.body(base_body_id).xpos.copy()
        base_rot_world = data.body(base_body_id).xmat.reshape(3, 3).copy()
        target_world = base_pos_world + base_rot_world @ target_local

        target_positions[step] = target_world

        # ===== 3）逆运动学求解 =====
        q_target, qd_target = solver.solve(data, target_world)

        # ===== 4）PD 力矩控制 =====
        mujoco.mj_inverse(model, data)
        tau = Kp * (q_target - data.qpos[:7]) - Kd * data.qvel[:7]
        tau = np.clip(tau, -50, 50)
        data.ctrl[:7] = tau

        # ===== 5）记录数据 =====
        actual_positions[step] = data.site("link_tcp").xpos.copy()
        joint_angles[step] = data.qpos[:7].copy()
        control_signals[step] = tau.copy()

        # ===== 6）物理步进 =====
        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1


# ========================= 可视化结果 =========================
plt.figure(figsize=(14, 10))

# ---- 3D 路径 ----
ax1 = plt.subplot(2, 2, 1, projection='3d')
ax1.plot(target_positions[:,0], target_positions[:,1], target_positions[:,2], '--', label='Target')
ax1.plot(actual_positions[:,0], actual_positions[:,1], actual_positions[:,2], label='Actual')
ax1.set_title('End-Effector Trajectory (World Frame)')
ax1.legend()

# ---- 关节角 ----
ax2 = plt.subplot(2, 2, 2)
for j in range(7):
    ax2.plot(time_axis, np.degrees(joint_angles[:, j]), label=f'Joint {j+1}')
ax2.set_title('Joint Angles')
ax2.legend()

# ---- 力矩 ----
ax3 = plt.subplot(2, 2, 3)
for j in range(7):
    ax3.plot(time_axis, control_signals[:, j])
ax3.set_title('Control Torque')

# ---- 跟踪误差 ----
ax4 = plt.subplot(2, 2, 4)
err = np.linalg.norm(actual_positions - target_positions, axis=1)
ax4.plot(time_axis, err * 1000)
ax4.set_title('Tracking Error (mm)')

plt.tight_layout()
plt.show()
