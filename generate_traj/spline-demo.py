"""
样条曲线轨迹生成器演示

这个脚本演示了如何使用样条曲线轨迹生成器生成无人机轨迹。
它生成一个轨迹，并运行输入和位置可行性测试，然后生成一些可视化结果。
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from spline_trajectory_generator import SplineTrajectoryGenerator, InputFeasibilityResult, StateFeasibilityResult

# 定义轨迹的起始状态:
pos0 = [0, 0, 2]  # 位置
vel0 = [0, 0, 0]  # 速度
acc0 = [0, 0, 0]  # 加速度

# 定义目标状态:
posf = [1, 0, 1]  # 位置
velf = [0, 0, 1]  # 速度
accf = [0, 9.81, 0]  # 加速度

# 定义持续时间:
Tf = 1

# 定义输入限制:
fmin = 5   # 最小推力 [m/s^2]
fmax = 25  # 最大推力 [m/s^2]
wmax = 20  # 最大角速率 [rad/s]
minTimeSec = 0.02  # 递归检查的最小时间间隔 [s]

# 定义重力方向:
gravity = [0, 0, -9.81]

# 创建轨迹生成器
traj = SplineTrajectoryGenerator(pos0, vel0, acc0, gravity)

# 设置目标状态
traj.set_goal_position(posf)
traj.set_goal_velocity(velf)
traj.set_goal_acceleration(accf)

# 生成轨迹
traj.generate(Tf, num_points=10)

# 测试输入可行性
inputsFeasible = traj.check_input_feasibility(fmin, fmax, wmax, minTimeSec)

# 测试是否会飞入地面
floorPoint = [0, 0, 0]  # 地面上的一点
floorNormal = [0, 0, 1]  # 我们希望在该点的方向上（向上）
positionFeasible = traj.check_position_feasibility(floorPoint, floorNormal)

# 打印轨迹参数和可行性结果
print("总代价 = ", traj.get_cost())
print("输入可行性结果: ", InputFeasibilityResult.to_string(inputsFeasible), "(", inputsFeasible, ")")
print("位置可行性结果: ", StateFeasibilityResult.to_string(positionFeasible), "(", positionFeasible, ")")

###########################################
# 绘制轨迹及其输入 #
###########################################

# 创建评估点
numPlotPoints = 100
time = np.linspace(0, Tf, numPlotPoints)
position = np.zeros([numPlotPoints, 3])
velocity = np.zeros([numPlotPoints, 3])
acceleration = np.zeros([numPlotPoints, 3])
thrust = np.zeros([numPlotPoints, 1])
ratesMagn = np.zeros([numPlotPoints, 1])

# 计算每个时间点的状态和输入
for i in range(numPlotPoints):
    t = time[i]
    position[i, :] = traj.get_position(t)
    velocity[i, :] = traj.get_velocity(t)
    acceleration[i, :] = traj.get_acceleration(t)
    thrust[i] = traj.get_thrust(t)
    ratesMagn[i] = np.linalg.norm(traj.get_body_rates(t))

# 创建图表
figStates, axes = plt.subplots(3, 1, sharex=True)
gs = gridspec.GridSpec(6, 2)
axPos = plt.subplot(gs[0:2, 0])
axVel = plt.subplot(gs[2:4, 0])
axAcc = plt.subplot(gs[4:6, 0])

# 绘制状态
for ax, yvals in zip([axPos, axVel, axAcc], [position, velocity, acceleration]):
    cols = ['r', 'g', 'b']
    labs = ['x', 'y', 'z']
    for i in range(3):
        ax.plot(time, yvals[:, i], cols[i], label=labs[i])

axPos.set_ylabel('Pos [m]')
axVel.set_ylabel('Vel [m/s]')
axAcc.set_ylabel('Acc [m/s^2]')
axAcc.set_xlabel('Time [s]')
axPos.legend()
axPos.set_title('States')

# 绘制输入
infeasibleAreaColour = [1, 0.5, 0.5]
axThrust = plt.subplot(gs[0:3, 1])
axOmega = plt.subplot(gs[3:6, 1])

# 绘制推力
axThrust.plot(time, thrust, 'k', label='command')
axThrust.plot([0, Tf], [fmin, fmin], 'r--', label='fmin')
axThrust.fill_between([0, Tf], [fmin, fmin], -1000, facecolor=infeasibleAreaColour, color=infeasibleAreaColour)
axThrust.fill_between([0, Tf], [fmax, fmax], 1000, facecolor=infeasibleAreaColour, color=infeasibleAreaColour)
axThrust.plot([0, Tf], [fmax, fmax], 'r-.', label='fmax')

axThrust.set_ylabel('Thrust [m/s^2]')
axThrust.legend()

# 绘制角速率
axOmega.plot(time, ratesMagn, 'k', label='command magnitude')
axOmega.plot([0, Tf], [wmax, wmax], 'r--', label='wmax')
axOmega.fill_between([0, Tf], [wmax, wmax], 1000, facecolor=infeasibleAreaColour, color=infeasibleAreaColour)
axOmega.set_xlabel('Time [s]')
axOmega.set_ylabel('Body rates [rad/s]')
axOmega.legend()

axThrust.set_title('Inputs')

# 调整显示范围
axThrust.set_ylim([min(fmin-1, np.min(thrust)), max(fmax+1, np.max(thrust))])
axOmega.set_ylim([0, max(wmax+1, np.max(ratesMagn))])

# 绘制3D轨迹
fig3D = plt.figure()
ax3D = fig3D.add_subplot(111, projection='3d')
ax3D.plot(position[:, 0], position[:, 1], position[:, 2], 'b')
ax3D.scatter(pos0[0], pos0[1], pos0[2], c='g', marker='o', s=100, label='Start')
ax3D.scatter(posf[0], posf[1], posf[2], c='r', marker='o', s=100, label='End')

# 添加地面平面
x_range = np.linspace(min(position[:, 0])-0.5, max(position[:, 0])+0.5, 10)
y_range = np.linspace(min(position[:, 1])-0.5, max(position[:, 1])+0.5, 10)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)  # 地面高度为0
ax3D.plot_surface(X, Y, Z, alpha=0.3, color='gray')

ax3D.set_xlabel('X [m]')
ax3D.set_ylabel('Y [m]')
ax3D.set_zlabel('Z [m]')
ax3D.set_title('3D Trajectory')
ax3D.legend()

# 使坐标轴等比例
max_range = np.array([position[:, 0].max()-position[:, 0].min(), 
                     position[:, 1].max()-position[:, 1].min(), 
                     position[:, 2].max()-position[:, 2].min()]).max() / 2.0
mid_x = (position[:, 0].max()+position[:, 0].min()) * 0.5
mid_y = (position[:, 1].max()+position[:, 1].min()) * 0.5
mid_z = (position[:, 2].max()+position[:, 2].min()) * 0.5
ax3D.set_xlim(mid_x - max_range, mid_x + max_range)
ax3D.set_ylim(mid_y - max_range, mid_y + max_range)
ax3D.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()