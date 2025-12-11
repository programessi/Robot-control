# 两阶段尝试2 - 修改版：可指定 joint7 的局部轴与 base x 对齐（0:x, 1:y, 2:z）
import sys
import os
import mujoco
import mujoco.viewer
import numpy as np
import select
import termios
import tty
import matplotlib.pyplot as plt

# 仿真参数配置
SIM_DURATION = 10.0      # 总仿真时间（秒）
CTRL_FREQ = 500         # 控制频率（Hz）
TIMESTEP = 0.002        # 物理仿真步长
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from pos_ad_control.ik_solver import InverseKinematicsSolver

# ---------- 工具函数 ----------
def circular_trajectory_local(t, radius=0.20, freq=0.5,
                              center_local=np.array([0.45, 0.0, 0.45])):
    theta = 2 * np.pi * freq * t
    return center_local + np.array([0.0, radius * np.cos(theta), radius * np.sin(theta)])

def key_pressed(key):
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr != []:
        c = sys.stdin.read(1)
        return c == key
    return False

def rotation_matrix_from_two_axes(primary_axis, secondary_hint):
    """
    构造一个右手坐标系：primary_axis -> first column。
    secondary_hint 用于确定第二列（被正交化）。
    返回 3x3 rotation matrix (columns are axes).
    """
    a1 = primary_axis / (np.linalg.norm(primary_axis) + 1e-12)
    s = secondary_hint - np.dot(secondary_hint, a1) * a1
    if np.linalg.norm(s) < 1e-8:
        # pick a fallback orthogonal vector
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(a1[0]) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        s = tmp - np.dot(tmp, a1) * a1
    a2 = s / (np.linalg.norm(s) + 1e-12)
    a3 = np.cross(a1, a2)
    a3 /= np.linalg.norm(a3) + 1e-12
    R = np.column_stack([a1, a2, a3])
    return R

def rotmat_to_rotvec(R):
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz]) / (2.0 * np.sin(angle) + 1e-12)
    return axis * angle

# helper: permute R so that primary becomes column at index 'col_idx'
def place_primary_in_column(R_primary_first, col_idx):
    """R_primary_first has primary in column 0. Return R_permuted with primary at column col_idx."""
    if col_idx == 0:
        return R_primary_first
    R = R_primary_first.copy()
    # swap column 0 and col_idx
    R[:, [0, col_idx]] = R[:, [col_idx, 0]]
    return R

# ---------- 主程序配置 ----------
SIM_DT = 0.002
TOTAL_TIME = 10.0

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "..", "scene.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

solver = InverseKinematicsSolver(model, damping=0.01, max_iter=100)

base_id = model.body("link_base").id
joint7_body_id = model.body("link7").id

Kp = np.array([600, 500, 400, 300, 200, 150, 150])
Kd = 0.1 * Kp
tau_limit = 50

# smoothing params (自动平滑切换时间)
smoothing_time = 0.6   # 秒（你可以调整）
smoothing_steps = max(1, int(smoothing_time / SIM_DT))

# ** 用户配置：选择 joint7 的哪个局部轴 要 对齐到 base x **
# 0 -> joint7 local x
# 1 -> joint7 local y
# 2 -> joint7 local z
joint7_local_axis_idx = 1   # <--- 你想要对齐哪个局部轴？将其改为 0/1/2
# 你说当前 joint7 轴线和 base 的 y 平行，若你希望它和 base x 平行，
# 很可能需要把 local_axis_idx 设置为 1 （即把 joint7 的局部 y 对齐到 base x）。
# 若不确定，先用诊断输出看哪条轴当前最接近 base x。

print("准备阶段1：将机械臂移动到圆起点并调整完整姿态（按 r 平滑切换到画圆）")
print(f"当前 joint7_local_axis_idx = {joint7_local_axis_idx} (0:x,1:y,2:z)")

old_attr = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())

# ---------- 阶段1：静态定位并显示完整姿态误差 ----------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_forward(model, data)

        base_pos = data.body(base_id).xpos.copy()
        base_rot = data.body(base_id).xmat.reshape(3,3).copy()

        # diagnostic: show which joint7 local axis is closest to base x currently
        R7_curr = data.body(joint7_body_id).xmat.reshape(3,3).copy()
        local_axes = [np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0]), np.array([0.0,0.0,1.0])]
        angles = []
        for i, la in enumerate(local_axes):
            world_axis = R7_curr @ la
            world_axis /= np.linalg.norm(world_axis) + 1e-12
            cosang = np.clip(np.dot(world_axis, base_rot[:,0]), -1.0, 1.0)  # base x is base_rot[:,0]
            angdeg = np.degrees(np.arccos(cosang))
            angles.append(angdeg)
        # print diagnostics once per 50 steps to avoid flood
        if np.random.rand() < 0.02:
            print(f"诊断 (deg) local_axes->base_x: x={angles[0]:.1f}°, y={angles[1]:.1f}°, z={angles[2]:.1f}°")

        # target at t=0
        target_local = circular_trajectory_local(0.0)
        target_world = base_pos + base_rot @ target_local

        # 构造完整目标旋转：primary = base x, secondary_hint = base z
        primary = base_rot @ np.array([1.0, 0.0, 0.0])       # desired direction in world
        secondary_hint = base_rot @ np.array([0.0, 0.0, 1.0])
        R_primary_first = rotation_matrix_from_two_axes(primary, secondary_hint)

        # place primary at the column corresponding to selected local axis
        R_des = place_primary_in_column(R_primary_first, joint7_local_axis_idx)

        # 求 IK（完整姿态）
        q_target, _ = solver.solve(data, target_world, target_rot=R_des, weight_ori=1.0, dt=0.5)

        # PD 控制跟踪 q_target
        tau = Kp * (q_target - data.qpos[:7]) - Kd * data.qvel[:7]
        tau = np.clip(tau, -tau_limit, tau_limit)
        data.ctrl[:7] = tau

        # 计算并输出位置误差与完整姿态误差（deg）
        pos_err = np.linalg.norm(target_world - data.site("link_tcp").xpos)
        R7 = data.body(joint7_body_id).xmat.reshape(3,3).copy()
        R_err = R_des @ R7.T
        rotvec_err = rotmat_to_rotvec(R_err)
        ori_err_deg = np.degrees(np.linalg.norm(rotvec_err))

        print(f"位置误差: {pos_err:.4f}   完整姿态误差: {ori_err_deg:.2f}°   (按 r 平滑开始画圆)")

        # 按 r 触发平滑切换到阶段2
        if key_pressed('r'):
            print("\n检测到 r：开始平滑过渡到圆轨迹...")
            # 保存 q_start 和 q_goal（目标 q_target 已求出）
            q_start = data.qpos[:7].copy()
            q_goal = q_target.copy()

            # 平滑跟踪：线性插值 q_ref，PD 去跟踪
            for s_idx in range(1, smoothing_steps + 1):
                s = s_idx / smoothing_steps
                q_ref = (1.0 - s) * q_start + s * q_goal
                mujoco.mj_forward(model, data)
                tau = Kp * (q_ref - data.qpos[:7]) - Kd * data.qvel[:7]
                tau = np.clip(tau, -tau_limit, tau_limit)
                data.ctrl[:7] = tau
                mujoco.mj_step(model, data)
                viewer.sync()
            print("平滑过渡完成，进入阶段2：开始画圆")
            break

        mujoco.mj_step(model, data)
        viewer.sync()

# ---------- 阶段2：开始画圆轨迹（每步使用完整姿态 IK） ----------
# 数据记录配置
num_steps = int(SIM_DURATION / TIMESTEP)
time_axis = np.arange(0, SIM_DURATION, TIMESTEP)[:num_steps]

# 预分配存储空间
target_positions = np.zeros((num_steps, 3))
actual_positions = np.zeros((num_steps, 3))
joint_angles = np.zeros((num_steps, 7))
control_signals = np.zeros((num_steps, 7))

t = 0.0
num_steps = int(TOTAL_TIME / SIM_DT)
step = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and step < num_steps:
        mujoco.mj_forward(model, data)

        base_pos = data.body(base_id).xpos.copy()
        base_rot = data.body(base_id).xmat.reshape(3,3).copy()

        target_local = circular_trajectory_local(t)
        target_world = base_pos + base_rot @ target_local
        target_positions[step]= target_world
        primary = base_rot @ np.array([1.0, 0.0, 0.0])
        secondary_hint = base_rot @ np.array([0.0, 0.0, 1.0])
        R_primary_first = rotation_matrix_from_two_axes(primary, secondary_hint)
        R_des = place_primary_in_column(R_primary_first, joint7_local_axis_idx)

        q_target, _ = solver.solve(data, target_world, target_rot=R_des, weight_ori=1.0, dt=0.25)
        tau = Kp * (q_target - data.qpos[:7]) - Kd * data.qvel[:7]
        tau = np.clip(tau, -tau_limit, tau_limit)
        data.ctrl[:7] = tau
        actual_positions[step] = data.site("link_tcp").xpos
        joint_angles[step] = data.qpos[:7]
        control_signals[step] = tau
        mujoco.mj_step(model, data)
        viewer.sync()

        t += SIM_DT
        step += 1
# Result Visualization
plt.figure(figsize=(14, 10))

# 三维轨迹跟踪结果 3D trajectory tracking results
ax1 = plt.subplot(2, 2, 1, projection='3d')
ax1.plot(target_positions[:,0], target_positions[:,1], target_positions[:,2], 
        label='Target Trajectory', linestyle='--')  # 目标轨迹
ax1.plot(actual_positions[:,0], actual_positions[:,1], actual_positions[:,2], 
        label='Actual Trajectory', alpha=0.7)  # 实际轨迹
ax1.set_xlabel('X (m)')  # X (米)
ax1.set_ylabel('Y (m)')  # Y (米)
ax1.set_zlabel('Z (m)')  # Z (米)
ax1.set_title('End-effector Trajectory Tracking')  # 末端执行器轨迹跟踪
ax1.legend()

# 关节角度变化 Joint angle variations
ax2 = plt.subplot(2, 2, 2)
for j in range(7):
    ax2.plot(time_axis, np.degrees(joint_angles[:,j]), 
            label=f'Joint {j+1}')  # 关节 {j+1}
ax2.set_xlabel('Time (s)')  # 时间 (秒)
ax2.set_ylabel('Joint Angle (°)')  # 关节角度 (度)
ax2.set_title('Joint Motion State')  # 关节运动状态
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 控制信号 Control signals
ax3 = plt.subplot(2, 2, 3)
for j in range(7):
    ax3.plot(time_axis, control_signals[:,j], label=f'Joint {j+1}')  # 关节 {j+1}
ax3.set_xlabel('Time (s)')  # 时间 (秒)
ax3.set_ylabel('Control Torque (N·m)')  # 控制力矩 (牛·米)
ax3.set_title('Control Signal Output')  # 控制信号输出

# 跟踪误差分析 Tracking error analysis
ax4 = plt.subplot(2, 2, 4)
position_error = np.linalg.norm(actual_positions - target_positions, axis=1)
ax4.plot(time_axis, position_error*1000, color='r')
ax4.set_xlabel('Time (s)')  # 时间 (秒)
ax4.set_ylabel('Tracking Error (mm)')  # 跟踪误差 (毫米)
ax4.set_title('Position Tracking Error')  # 位置跟踪误差

plt.tight_layout()
plt.show()
termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)
print("运行结束。")
