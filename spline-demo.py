"""
Spline Trajectory Generator Demo

This script demonstrates how to use the spline trajectory generator to create drone trajectories.
It generates a trajectory, runs input and position feasibility tests, and creates visualizations.
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from spline_trajectory_generator import SplineTrajectoryGenerator, InputFeasibilityResult, StateFeasibilityResult

# Define the trajectory starting state:
pos0 = [0, 0, 2]  # position
vel0 = [0, 0, 0]  # velocity
acc0 = [0, 0, 0]  # acceleration

# Define the goal state:
posf = [1, 0, 1]  # position
velf = [0, 0, 1]  # velocity
accf = [0, 9.81, 0]  # acceleration

# Define the duration:
Tf = 1

# Define the input limits:
fmin = 5   # minimum thrust [m/s^2]
fmax = 25  # maximum thrust [m/s^2]
wmax = 20  # maximum body rates [rad/s]
minTimeSec = 0.02  # minimum time interval for recursive checking [s]

# Define gravity direction:
gravity = [0, 0, -9.81]

# Create trajectory generator
traj = SplineTrajectoryGenerator(pos0, vel0, acc0, gravity)

# Set goal states
traj.set_goal_position(posf)
traj.set_goal_velocity(velf)
traj.set_goal_acceleration(accf)

# Generate trajectory
traj.generate(Tf, num_points=10)

# Test input feasibility
inputsFeasible = traj.check_input_feasibility(fmin, fmax, wmax, minTimeSec)

# Test whether trajectory stays above floor
floorPoint = [0, 0, 0]  # a point on the floor
floorNormal = [0, 0, 1]  # direction pointing up from floor
positionFeasible = traj.check_position_feasibility(floorPoint, floorNormal)

# Print trajectory parameters and feasibility results
print("Total cost = ", traj.get_cost())
print("Input feasibility result: ", InputFeasibilityResult.to_string(inputsFeasible), "(", inputsFeasible, ")")
print("Position feasibility result: ", StateFeasibilityResult.to_string(positionFeasible), "(", positionFeasible, ")")

###########################################
# Plot the trajectory and inputs #
###########################################

# Create evaluation points
numPlotPoints = 100
time = np.linspace(0, Tf, numPlotPoints)
position = np.zeros([numPlotPoints, 3])
velocity = np.zeros([numPlotPoints, 3])
acceleration = np.zeros([numPlotPoints, 3])
thrust = np.zeros([numPlotPoints, 1])
ratesMagn = np.zeros([numPlotPoints, 1])

# Calculate state and inputs at each time point
for i in range(numPlotPoints):
    t = time[i]
    position[i, :] = traj.get_position(t)
    velocity[i, :] = traj.get_velocity(t)
    acceleration[i, :] = traj.get_acceleration(t)
    thrust[i] = traj.get_thrust(t)
    ratesMagn[i] = np.linalg.norm(traj.get_body_rates(t))

# Create plots
figStates, axes = plt.subplots(3, 1, sharex=True)
gs = gridspec.GridSpec(6, 2)
axPos = plt.subplot(gs[0:2, 0])
axVel = plt.subplot(gs[2:4, 0])
axAcc = plt.subplot(gs[4:6, 0])

# Plot states
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

# Plot inputs
infeasibleAreaColour = [1, 0.5, 0.5]
axThrust = plt.subplot(gs[0:3, 1])
axOmega = plt.subplot(gs[3:6, 1])

# Plot thrust
axThrust.plot(time, thrust, 'k', label='command')
axThrust.plot([0, Tf], [fmin, fmin], 'r--', label='fmin')
axThrust.fill_between([0, Tf], [fmin, fmin], -1000, facecolor=infeasibleAreaColour, color=infeasibleAreaColour)
axThrust.fill_between([0, Tf], [fmax, fmax], 1000, facecolor=infeasibleAreaColour, color=infeasibleAreaColour)
axThrust.plot([0, Tf], [fmax, fmax], 'r-.', label='fmax')

axThrust.set_ylabel('Thrust [m/s^2]')
axThrust.legend()

# Plot body rates
axOmega.plot(time, ratesMagn, 'k', label='command magnitude')
axOmega.plot([0, Tf], [wmax, wmax], 'r--', label='wmax')
axOmega.fill_between([0, Tf], [wmax, wmax], 1000, facecolor=infeasibleAreaColour, color=infeasibleAreaColour)
axOmega.set_xlabel('Time [s]')
axOmega.set_ylabel('Body rates [rad/s]')
axOmega.legend()

axThrust.set_title('Inputs')

# Adjust display ranges
axThrust.set_ylim([min(fmin-1, np.min(thrust)), max(fmax+1, np.max(thrust))])
axOmega.set_ylim([0, max(wmax+1, np.max(ratesMagn))])

# Plot 3D trajectory
fig3D = plt.figure()
ax3D = fig3D.add_subplot(111, projection='3d')
ax3D.plot(position[:, 0], position[:, 1], position[:, 2], 'b')
ax3D.scatter(pos0[0], pos0[1], pos0[2], c='g', marker='o', s=100, label='Start')
ax3D.scatter(posf[0], posf[1], posf[2], c='r', marker='o', s=100, label='End')

# Add floor plane
x_range = np.linspace(min(position[:, 0])-0.5, max(position[:, 0])+0.5, 10)
y_range = np.linspace(min(position[:, 1])-0.5, max(position[:, 1])+0.5, 10)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)  # floor at height 0
ax3D.plot_surface(X, Y, Z, alpha=0.3, color='gray')

ax3D.set_xlabel('X [m]')
ax3D.set_ylabel('Y [m]')
ax3D.set_zlabel('Z [m]')
ax3D.set_title('3D Trajectory')
ax3D.legend()

# Make axes equal scale
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