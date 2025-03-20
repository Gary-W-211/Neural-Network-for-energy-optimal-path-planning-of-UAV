"""
Drone Trajectory Planner Comparison

This script compares two trajectory generation methods:
1. Original polynomial method (quadrocoptertrajectory)
2. New spline method (spline_trajectory_generator)

and plots the comparison results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Import two trajectory generators
import quadrocoptertrajectory as quadtraj
from spline_trajectory_generator import SplineTrajectoryGenerator

# Define different test trajectories
def test_trajectories():
    """
    Generate a set of test trajectory definitions
    
    Returns:
        trajectories: List of dictionaries, each defining a trajectory test
    """
    trajectories = [
        {
            'name': 'Simple Forward',
            'pos0': [0, 0, 2],
            'vel0': [0, 0, 0],
            'acc0': [0, 0, 0],
            'posf': [1, 0, 2],
            'velf': [0, 0, 0],
            'accf': [0, 0, 0],
            'Tf': 1.0
        },
        {
            'name': 'Rise and Forward',
            'pos0': [0, 0, 1],
            'vel0': [0, 0, 0],
            'acc0': [0, 0, 0],
            'posf': [1, 0, 2],
            'velf': [1, 0, 0],
            'accf': [0, 0, 0],
            'Tf': 1.5
        },
        {
            'name': 'Spiral Ascent',
            'pos0': [0, 0, 0],
            'vel0': [0, 0, 0],
            'acc0': [0, 0, 0],
            'posf': [0, 0, 3],
            'velf': [1, 1, 0],
            'accf': [0, 0, 0],
            'Tf': 2.0
        },
        {
            'name': 'Complex Maneuver',
            'pos0': [0, 0, 2],
            'vel0': [0, 0, 0],
            'acc0': [0, 0, 0],
            'posf': [2, 2, 1],
            'velf': [0, 0, -1],
            'accf': [0, 0, 9.81],
            'Tf': 2.5
        }
    ]
    
    return trajectories

# Generate trajectories
def generate_trajectories(trajectory_def, gravity=[0, 0, -9.81]):
    """
    Generate trajectories using both methods
    
    Parameters:
        trajectory_def: Trajectory definition dictionary
        gravity: Gravity acceleration vector
        
    Returns:
        poly_traj: Trajectory generated using polynomial method
        spline_traj: Trajectory generated using spline method
        generation_times: Dictionary of generation times
    """
    pos0 = trajectory_def['pos0']
    vel0 = trajectory_def['vel0']
    acc0 = trajectory_def['acc0']
    posf = trajectory_def['posf']
    velf = trajectory_def['velf']
    accf = trajectory_def['accf']
    Tf = trajectory_def['Tf']
    
    # Record generation times
    generation_times = {}
    
    # Generate polynomial trajectory
    start_time = time.time()
    poly_traj = quadtraj.RapidTrajectory(pos0, vel0, acc0, gravity)
    poly_traj.set_goal_position(posf)
    poly_traj.set_goal_velocity(velf)
    poly_traj.set_goal_acceleration(accf)
    poly_traj.generate(Tf)
    generation_times['polynomial'] = time.time() - start_time
    
    # Generate spline trajectory
    start_time = time.time()
    spline_traj = SplineTrajectoryGenerator(pos0, vel0, acc0, gravity)
    spline_traj.set_goal_position(posf)
    spline_traj.set_goal_velocity(velf)
    spline_traj.set_goal_acceleration(accf)
    spline_traj.generate(Tf)
    generation_times['spline'] = time.time() - start_time
    
    return poly_traj, spline_traj, generation_times

# Evaluate trajectories
def evaluate_trajectories(poly_traj, spline_traj, Tf, sample_points=100):
    """
    Evaluate and compare two trajectories
    
    Parameters:
        poly_traj: Polynomial trajectory
        spline_traj: Spline trajectory
        Tf: Total trajectory time
        sample_points: Number of sample points
        
    Returns:
        results: Dictionary containing evaluation results
    """
    # Create time array
    time_points = np.linspace(0, Tf, sample_points)
    
    # Initialize result arrays
    poly_pos = np.zeros((sample_points, 3))
    poly_vel = np.zeros((sample_points, 3))
    poly_acc = np.zeros((sample_points, 3))
    poly_thrust = np.zeros(sample_points)
    poly_rates = np.zeros(sample_points)
    
    spline_pos = np.zeros((sample_points, 3))
    spline_vel = np.zeros((sample_points, 3))
    spline_acc = np.zeros((sample_points, 3))
    spline_thrust = np.zeros(sample_points)
    spline_rates = np.zeros(sample_points)
    
    # Calculate values at each time point
    for i, t in enumerate(time_points):
        # Polynomial trajectory
        poly_pos[i] = poly_traj.get_position(t)
        poly_vel[i] = poly_traj.get_velocity(t)
        poly_acc[i] = poly_traj.get_acceleration(t)
        poly_thrust[i] = poly_traj.get_thrust(t)
        poly_rates[i] = np.linalg.norm(poly_traj.get_body_rates(t))
        
        # Spline trajectory
        spline_pos[i] = spline_traj.get_position(t)
        spline_vel[i] = spline_traj.get_velocity(t)
        spline_acc[i] = spline_traj.get_acceleration(t)
        spline_thrust[i] = spline_traj.get_thrust(t)
        spline_rates[i] = np.linalg.norm(spline_traj.get_body_rates(t))
    
    # Calculate trajectory differences
    pos_diff = np.linalg.norm(poly_pos - spline_pos, axis=1)
    vel_diff = np.linalg.norm(poly_vel - spline_vel, axis=1)
    acc_diff = np.linalg.norm(poly_acc - spline_acc, axis=1)
    thrust_diff = np.abs(poly_thrust - spline_thrust)
    rates_diff = np.abs(poly_rates - spline_rates)
    
    # Calculate smoothness metrics (jerk)
    poly_jerk = np.zeros((sample_points-1, 3))
    spline_jerk = np.zeros((sample_points-1, 3))
    
    for i in range(sample_points-1):
        dt = time_points[i+1] - time_points[i]
        poly_jerk[i] = (poly_acc[i+1] - poly_acc[i]) / dt
        spline_jerk[i] = (spline_acc[i+1] - spline_acc[i]) / dt
    
    poly_jerk_norm = np.linalg.norm(poly_jerk, axis=1)
    spline_jerk_norm = np.linalg.norm(spline_jerk, axis=1)
    
    # Calculate averages and maximums
    results = {
        'time_points': time_points,
        'poly_pos': poly_pos,
        'poly_vel': poly_vel,
        'poly_acc': poly_acc,
        'poly_thrust': poly_thrust,
        'poly_rates': poly_rates,
        'poly_jerk_norm': poly_jerk_norm,
        'spline_pos': spline_pos,
        'spline_vel': spline_vel,
        'spline_acc': spline_acc,
        'spline_thrust': spline_thrust,
        'spline_rates': spline_rates,
        'spline_jerk_norm': spline_jerk_norm,
        'pos_diff': pos_diff,
        'vel_diff': vel_diff,
        'acc_diff': acc_diff,
        'thrust_diff': thrust_diff,
        'rates_diff': rates_diff,
        'avg_pos_diff': np.mean(pos_diff),
        'max_pos_diff': np.max(pos_diff),
        'avg_vel_diff': np.mean(vel_diff),
        'max_vel_diff': np.max(vel_diff),
        'avg_acc_diff': np.mean(acc_diff),
        'max_acc_diff': np.max(acc_diff),
        'avg_thrust_diff': np.mean(thrust_diff),
        'max_thrust_diff': np.max(thrust_diff),
        'avg_rates_diff': np.mean(rates_diff),
        'max_rates_diff': np.max(rates_diff),
        'poly_avg_jerk': np.mean(poly_jerk_norm),
        'poly_max_jerk': np.max(poly_jerk_norm),
        'spline_avg_jerk': np.mean(spline_jerk_norm),
        'spline_max_jerk': np.max(spline_jerk_norm),
        'poly_cost': poly_traj.get_cost(),
        'spline_cost': spline_traj.get_cost()
    }
    
    return results

# Plot trajectory comparison
def plot_trajectory_comparison(results, trajectory_def, generation_times):
    """
    Plot trajectory comparison
    
    Parameters:
        results: Evaluation results dictionary
        trajectory_def: Trajectory definition dictionary
        generation_times: Generation times dictionary
    """
    time_points = results['time_points']
    traj_name = trajectory_def['name']
    
    # Create figure
    plt.figure(figsize=(12, 10))
    plt.suptitle(f'Trajectory Comparison: {traj_name}', fontsize=16)
    
    # 1. Position comparison
    plt.subplot(3, 2, 1)
    plt.plot(time_points, results['poly_pos'][:, 0], 'r-', label='Poly X')
    plt.plot(time_points, results['poly_pos'][:, 1], 'g-', label='Poly Y')
    plt.plot(time_points, results['poly_pos'][:, 2], 'b-', label='Poly Z')
    plt.plot(time_points, results['spline_pos'][:, 0], 'r--', label='Spline X')
    plt.plot(time_points, results['spline_pos'][:, 1], 'g--', label='Spline Y')
    plt.plot(time_points, results['spline_pos'][:, 2], 'b--', label='Spline Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Position Trajectory')
    plt.grid(True)
    plt.legend()
    
    # 2. Position error
    plt.subplot(3, 2, 2)
    plt.plot(time_points, results['pos_diff'], 'k-')
    plt.xlabel('Time [s]')
    plt.ylabel('Position Difference [m]')
    plt.title('Position Error')
    plt.grid(True)
    plt.text(0.5, 0.8, f"Average Error: {results['avg_pos_diff']:.3f} m\nMax Error: {results['max_pos_diff']:.3f} m", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    # 3. Acceleration comparison
    plt.subplot(3, 2, 3)
    plt.plot(time_points, results['poly_acc'][:, 0], 'r-', label='Poly X')
    plt.plot(time_points, results['poly_acc'][:, 1], 'g-', label='Poly Y')
    plt.plot(time_points, results['poly_acc'][:, 2], 'b-', label='Poly Z')
    plt.plot(time_points, results['spline_acc'][:, 0], 'r--', label='Spline X')
    plt.plot(time_points, results['spline_acc'][:, 1], 'g--', label='Spline Y')
    plt.plot(time_points, results['spline_acc'][:, 2], 'b--', label='Spline Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.title('Acceleration Trajectory')
    plt.grid(True)
    plt.legend()
    
    # 4. Thrust comparison
    plt.subplot(3, 2, 4)
    plt.plot(time_points, results['poly_thrust'], 'r-', label='Polynomial')
    plt.plot(time_points, results['spline_thrust'], 'b--', label='Spline')
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust [m/s²]')
    plt.title('Thrust Trajectory')
    plt.grid(True)
    plt.legend()
    
    # 5. Jerk comparison
    plt.subplot(3, 2, 5)
    plt.plot(time_points[:-1], results['poly_jerk_norm'], 'r-', label='Polynomial')
    plt.plot(time_points[:-1], results['spline_jerk_norm'], 'b--', label='Spline')
    plt.xlabel('Time [s]')
    plt.ylabel('Jerk Norm [m/s³]')
    plt.title('Jerk Trajectory')
    plt.grid(True)
    plt.legend()
    plt.text(0.05, 0.8, f"Poly Average: {results['poly_avg_jerk']:.2f}\n"
                       f"Spline Average: {results['spline_avg_jerk']:.2f}", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    # 6. Angular rate comparison
    plt.subplot(3, 2, 6)
    plt.plot(time_points, results['poly_rates'], 'r-', label='Polynomial')
    plt.plot(time_points, results['spline_rates'], 'b--', label='Spline')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Rate Norm [rad/s]')
    plt.title('Angular Rate Trajectory')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Add performance information
    plt.figtext(0.5, 0.01, 
                f"Generation Time: Polynomial {generation_times['polynomial']*1000:.2f} ms, Spline {generation_times['spline']*1000:.2f} ms\n"
                f"Trajectory Cost: Polynomial {results['poly_cost']:.2f}, Spline {results['spline_cost']:.2f}", 
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # 3D trajectory comparison
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot start and end points
    ax.scatter(trajectory_def['pos0'][0], trajectory_def['pos0'][1], trajectory_def['pos0'][2], 
               c='g', marker='o', s=100, label='Start')
    ax.scatter(trajectory_def['posf'][0], trajectory_def['posf'][1], trajectory_def['posf'][2], 
               c='r', marker='o', s=100, label='End')
    
    # Plot both trajectories
    ax.plot(results['poly_pos'][:, 0], results['poly_pos'][:, 1], results['poly_pos'][:, 2], 
            'r-', linewidth=2, label='Polynomial')
    ax.plot(results['spline_pos'][:, 0], results['spline_pos'][:, 1], results['spline_pos'][:, 2], 
            'b--', linewidth=2, label='Spline')
    
    # Add axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'3D Trajectory Comparison: {traj_name}')
    
    # Make axes equal
    max_range = np.array([
        np.max(results['poly_pos'][:, 0]) - np.min(results['poly_pos'][:, 0]),
        np.max(results['poly_pos'][:, 1]) - np.min(results['poly_pos'][:, 1]),
        np.max(results['poly_pos'][:, 2]) - np.min(results['poly_pos'][:, 2])
    ]).max() / 2.0
    
    mid_x = (np.max(results['poly_pos'][:, 0]) + np.min(results['poly_pos'][:, 0])) / 2
    mid_y = (np.max(results['poly_pos'][:, 1]) + np.min(results['poly_pos'][:, 1])) / 2
    mid_z = (np.max(results['poly_pos'][:, 2]) + np.min(results['poly_pos'][:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend()
    
    plt.tight_layout()
    
    # Plot velocity vectors
    fig_vel = plt.figure(figsize=(10, 8))
    ax_vel = fig_vel.add_subplot(111, projection='3d')
    
    # Draw velocity vectors at intervals
    skip = 10
    for i in range(0, len(time_points), skip):
        # Polynomial trajectory velocity vectors
        ax_vel.quiver(results['poly_pos'][i, 0], results['poly_pos'][i, 1], results['poly_pos'][i, 2],
                      results['poly_vel'][i, 0], results['poly_vel'][i, 1], results['poly_vel'][i, 2],
                      color='r', length=0.1, normalize=True)
        
        # Spline trajectory velocity vectors
        ax_vel.quiver(results['spline_pos'][i, 0], results['spline_pos'][i, 1], results['spline_pos'][i, 2],
                      results['spline_vel'][i, 0], results['spline_vel'][i, 1], results['spline_vel'][i, 2],
                      color='b', length=0.1, normalize=True)
    
    # Draw trajectory paths
    ax_vel.plot(results['poly_pos'][:, 0], results['poly_pos'][:, 1], results['poly_pos'][:, 2], 
                'r-', alpha=0.3, label='Polynomial')
    ax_vel.plot(results['spline_pos'][:, 0], results['spline_pos'][:, 1], results['spline_pos'][:, 2], 
                'b--', alpha=0.3, label='Spline')
    
    # Add axis labels
    ax_vel.set_xlabel('X [m]')
    ax_vel.set_ylabel('Y [m]')
    ax_vel.set_zlabel('Z [m]')
    ax_vel.set_title(f'Velocity Vector Comparison: {traj_name}')
    
    # Make axes equal
    ax_vel.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_vel.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_vel.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax_vel.legend()
    
    plt.tight_layout()


# Generate multi-segment trajectory
def generate_multi_segment_trajectory(waypoints, velocities=None, accelerations=None, segment_times=None, gravity=[0, 0, -9.81]):
    """
    Generate trajectory through multiple waypoints
    
    Parameters:
        waypoints: List of waypoints [[x1,y1,z1], [x2,y2,z2], ...]
        velocities: List of velocities at waypoints (optional)
        accelerations: List of accelerations at waypoints (optional)
        segment_times: Time for each segment (optional)
        gravity: Gravity acceleration vector
        
    Returns:
        poly_segments: List of polynomial trajectory segments
        spline_segments: List of spline trajectory segments
        segment_times: List of segment times
    """
    waypoints = np.array(waypoints)
    num_segments = len(waypoints) - 1
    
    # If velocities not provided, set to zero
    if velocities is None:
        velocities = [np.zeros(3) for _ in range(len(waypoints))]
    else:
        velocities = [np.array(v) for v in velocities]
    
    # If accelerations not provided, set to zero
    if accelerations is None:
        accelerations = [np.zeros(3) for _ in range(len(waypoints))]
    else:
        accelerations = [np.array(a) for a in accelerations]
    
    # If times not provided, estimate based on distance
    if segment_times is None:
        segment_times = []
        for i in range(num_segments):
            dist = np.linalg.norm(waypoints[i+1] - waypoints[i])
            # Assume average velocity of 1 m/s, minimum 0.5s
            segment_times.append(max(0.5, dist))
    
    # Generate each segment
    poly_segments = []
    spline_segments = []
    
    for i in range(num_segments):
        pos0 = waypoints[i]
        vel0 = velocities[i]
        acc0 = accelerations[i]
        
        posf = waypoints[i+1]
        velf = velocities[i+1]
        accf = accelerations[i+1]
        
        Tf = segment_times[i]
        
        # Generate polynomial trajectory segment
        poly_traj = quadtraj.RapidTrajectory(pos0, vel0, acc0, gravity)
        poly_traj.set_goal_position(posf)
        poly_traj.set_goal_velocity(velf)
        poly_traj.set_goal_acceleration(accf)
        poly_traj.generate(Tf)
        poly_segments.append(poly_traj)
        
        # Generate spline trajectory segment
        spline_traj = SplineTrajectoryGenerator(pos0, vel0, acc0, gravity)
        spline_traj.set_goal_position(posf)
        spline_traj.set_goal_velocity(velf)
        spline_traj.set_goal_acceleration(accf)
        spline_traj.generate(Tf)
        spline_segments.append(spline_traj)
    
    return poly_segments, spline_segments, segment_times


def evaluate_multi_segment_trajectory(poly_segments, spline_segments, segment_times, sample_points_per_segment=50):
    """
    Evaluate multi-segment trajectory
    
    Parameters:
        poly_segments: List of polynomial trajectory segments
        spline_segments: List of spline trajectory segments
        segment_times: List of segment times
        sample_points_per_segment: Number of sample points per segment
        
    Returns:
        results: Results dictionary
    """
    num_segments = len(poly_segments)
    
    # Calculate time offsets for each segment
    time_offsets = [0]
    for i in range(num_segments-1):
        time_offsets.append(time_offsets[-1] + segment_times[i])
    
    # Calculate total sample points
    total_points = num_segments * sample_points_per_segment
    
    # Initialize result arrays
    time_points = np.zeros(total_points)
    poly_pos = np.zeros((total_points, 3))
    poly_vel = np.zeros((total_points, 3))
    poly_acc = np.zeros((total_points, 3))
    poly_thrust = np.zeros(total_points)
    poly_rates = np.zeros(total_points)
    
    spline_pos = np.zeros((total_points, 3))
    spline_vel = np.zeros((total_points, 3))
    spline_acc = np.zeros((total_points, 3))
    spline_thrust = np.zeros(total_points)
    spline_rates = np.zeros(total_points)
    
    # Calculate values for each segment
    for i in range(num_segments):
        # Time range
        t_start = time_offsets[i]
        t_end = t_start + segment_times[i]
        t_range = np.linspace(t_start, t_end, sample_points_per_segment)
        
        # Index range
        idx_start = i * sample_points_per_segment
        idx_end = (i + 1) * sample_points_per_segment
        
        # Save time points
        time_points[idx_start:idx_end] = t_range
        
        # Current segment trajectories
        poly_traj = poly_segments[i]
        spline_traj = spline_segments[i]
        
        # Calculate values at each time point
        for j, t in enumerate(np.linspace(0, segment_times[i], sample_points_per_segment)):
            idx = idx_start + j
            
            # Polynomial trajectory
            poly_pos[idx] = poly_traj.get_position(t)
            poly_vel[idx] = poly_traj.get_velocity(t)
            poly_acc[idx] = poly_traj.get_acceleration(t)
            poly_thrust[idx] = poly_traj.get_thrust(t)
            poly_rates[idx] = np.linalg.norm(poly_traj.get_body_rates(t))
            
            # Spline trajectory
            spline_pos[idx] = spline_traj.get_position(t)
            spline_vel[idx] = spline_traj.get_velocity(t)
            spline_acc[idx] = spline_traj.get_acceleration(t)
            spline_thrust[idx] = spline_traj.get_thrust(t)
            spline_rates[idx] = np.linalg.norm(spline_traj.get_body_rates(t))
    
    # Calculate trajectory differences
    pos_diff = np.linalg.norm(poly_pos - spline_pos, axis=1)
    vel_diff = np.linalg.norm(poly_vel - spline_vel, axis=1)
    acc_diff = np.linalg.norm(poly_acc - spline_acc, axis=1)
    thrust_diff = np.abs(poly_thrust - spline_thrust)
    rates_diff = np.abs(poly_rates - spline_rates)
    
    # Calculate smoothness metrics (jerk)
    time_diffs = np.diff(time_points)
    poly_jerk = np.zeros((total_points-1, 3))
    spline_jerk = np.zeros((total_points-1, 3))
    
    for i in range(total_points-1):
        dt = time_diffs[i]
        if dt > 0:  # Avoid division by zero
            poly_jerk[i] = (poly_acc[i+1] - poly_acc[i]) / dt
            spline_jerk[i] = (spline_acc[i+1] - spline_acc[i]) / dt
    
    poly_jerk_norm = np.linalg.norm(poly_jerk, axis=1)
    spline_jerk_norm = np.linalg.norm(spline_jerk, axis=1)
    
    # Calculate total cost
    poly_cost = sum(traj.get_cost() for traj in poly_segments)
    spline_cost = sum(traj.get_cost() for traj in spline_segments)
    
    # Collect results
    results = {
        'time_points': time_points,
        'time_offsets': time_offsets,
        'segment_times': segment_times,
        'poly_pos': poly_pos,
        'poly_vel': poly_vel,
        'poly_acc': poly_acc,
        'poly_thrust': poly_thrust,
        'poly_rates': poly_rates,
        'poly_jerk_norm': poly_jerk_norm,
        'spline_pos': spline_pos,
        'spline_vel': spline_vel,
        'spline_acc': spline_acc,
        'spline_thrust': spline_thrust,
        'spline_rates': spline_rates,
        'spline_jerk_norm': spline_jerk_norm,
        'pos_diff': pos_diff,
        'vel_diff': vel_diff,
        'acc_diff': acc_diff,
        'thrust_diff': thrust_diff,
        'rates_diff': rates_diff,
        'avg_pos_diff': np.mean(pos_diff),
        'max_pos_diff': np.max(pos_diff),
        'avg_vel_diff': np.mean(vel_diff),
        'max_vel_diff': np.max(vel_diff),
        'avg_acc_diff': np.mean(acc_diff),
        'max_acc_diff': np.max(acc_diff),
        'avg_thrust_diff': np.mean(thrust_diff),
        'max_thrust_diff': np.max(thrust_diff),
        'avg_rates_diff': np.mean(rates_diff),
        'max_rates_diff': np.max(rates_diff),
        'poly_avg_jerk': np.mean(poly_jerk_norm),
        'poly_max_jerk': np.max(poly_jerk_norm),
        'spline_avg_jerk': np.mean(spline_jerk_norm),
        'spline_max_jerk': np.max(spline_jerk_norm),
        'poly_cost': poly_cost,
        'spline_cost': spline_cost
    }
    
    return results


def plot_multi_segment_trajectory(results, waypoints):
    """
    Plot multi-segment trajectory
    
    Parameters:
        results: Evaluation results dictionary
        waypoints: List of waypoints
    """
    waypoints = np.array(waypoints)
    time_points = results['time_points']
    time_offsets = results['time_offsets']
    segment_times = results['segment_times']
    
    # Plot 3D trajectory
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
               c='g', marker='o', s=100, label='Waypoints')
    
    # Plot trajectories
    ax.plot(results['poly_pos'][:, 0], results['poly_pos'][:, 1], results['poly_pos'][:, 2], 
            'r-', linewidth=2, label='Polynomial')
    ax.plot(results['spline_pos'][:, 0], results['spline_pos'][:, 1], results['spline_pos'][:, 2], 
            'b--', linewidth=2, label='Spline')
    
    # Add segment separators
    for i, t_offset in enumerate(time_offsets[1:]):
        idx = np.argmin(np.abs(time_points - t_offset))
        ax.plot([results['poly_pos'][idx, 0]], [results['poly_pos'][idx, 1]], [results['poly_pos'][idx, 2]], 
                'mo', markersize=8)
    
    # Add axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Multi-Segment 3D Trajectory Comparison')
    
    # Make axes equal
    max_range = np.array([
        np.max(waypoints[:, 0]) - np.min(waypoints[:, 0]),
        np.max(waypoints[:, 1]) - np.min(waypoints[:, 1]),
        np.max(waypoints[:, 2]) - np.min(waypoints[:, 2])
    ]).max() / 2.0
    
    mid_x = (np.max(waypoints[:, 0]) + np.min(waypoints[:, 0])) / 2
    mid_y = (np.max(waypoints[:, 1]) + np.min(waypoints[:, 1])) / 2
    mid_z = (np.max(waypoints[:, 2]) + np.min(waypoints[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range * 1.2, mid_x + max_range * 1.2)
    ax.set_ylim(mid_y - max_range * 1.2, mid_y + max_range * 1.2)
    ax.set_zlim(mid_z - max_range * 1.2, mid_z + max_range * 1.2)
    
    ax.legend()
    
    # Create figure for state comparison
    plt.figure(figsize=(15, 12))
    plt.suptitle('Multi-Segment State Comparison', fontsize=16)
    
    # 1. Position trajectories
    axes = []
    for i in range(3):
        ax = plt.subplot(5, 1, i+1)
        axes.append(ax)
        # Plot positions
        ax.plot(time_points, results['poly_pos'][:, i], 'r-', label='Polynomial')
        ax.plot(time_points, results['spline_pos'][:, i], 'b--', label='Spline')
        
        # Plot waypoints
        for j, t_offset in enumerate(time_offsets):
            ax.plot(t_offset, waypoints[j, i], 'go', markersize=6)
        
        # Add segment separators
        for t_offset in time_offsets[1:]:
            ax.axvline(x=t_offset, color='k', linestyle=':')
        
        labels = ['X', 'Y', 'Z']
        ax.set_ylabel(f'{labels[i]} Position [m]')
        ax.grid(True)
        
        if i == 0:
            ax.legend()
    
    # 4. Thrust trajectory
    ax = plt.subplot(5, 1, 4)
    ax.plot(time_points, results['poly_thrust'], 'r-', label='Polynomial')
    ax.plot(time_points, results['spline_thrust'], 'b--', label='Spline')
    
    # Add segment separators
    for t_offset in time_offsets[1:]:
        ax.axvline(x=t_offset, color='k', linestyle=':')
        
    ax.set_ylabel('Thrust [m/s²]')
    ax.grid(True)
    
    # 5. Jerk trajectory
    ax = plt.subplot(5, 1, 5)
    ax.plot(time_points[:-1], results['poly_jerk_norm'], 'r-', label='Polynomial')
    ax.plot(time_points[:-1], results['spline_jerk_norm'], 'b--', label='Spline')
    
    # Add segment separators
    for t_offset in time_offsets[1:]:
        ax.axvline(x=t_offset, color='k', linestyle=':')
        
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Jerk Norm [m/s³]')
    ax.grid(True)
    
    # Add performance information
    plt.figtext(0.5, 0.01, 
                f"Trajectory Cost: Polynomial {results['poly_cost']:.2f}, Spline {results['spline_cost']:.2f}\n"
                f"Average Position Error: {results['avg_pos_diff']:.3f} m, Max Position Error: {results['max_pos_diff']:.3f} m\n"
                f"Polynomial Average Jerk: {results['poly_avg_jerk']:.2f}, Spline Average Jerk: {results['spline_avg_jerk']:.2f}",
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)
    
    plt.show()


# Main function
def main():
    """Main function"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get test trajectory definitions
    test_trajectories_list = test_trajectories()
    
    # Evaluate each test trajectory
    for traj_def in test_trajectories_list:
        print(f"Evaluating trajectory: {traj_def['name']}")
        
        # Generate trajectories
        poly_traj, spline_traj, generation_times = generate_trajectories(traj_def)
        
        # Evaluate trajectories
        results = evaluate_trajectories(poly_traj, spline_traj, traj_def['Tf'])
        
        # Plot comparison
        plot_trajectory_comparison(results, traj_def, generation_times)
    
    # Test multi-segment trajectory
    print("\nGenerating multi-segment trajectory...")
    
    # Define waypoints
    waypoints = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 0, 2],
        [3, -1, 1],
        [4, 0, 0]
    ]
    
    # Define velocities at each waypoint
    velocities = [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [0.5, 0.5, 0],
        [0, 0, 0]
    ]
    
    # Generate multi-segment trajectory
    poly_segments, spline_segments, segment_times = generate_multi_segment_trajectory(
        waypoints, velocities)
    
    # Evaluate multi-segment trajectory
    results = evaluate_multi_segment_trajectory(poly_segments, spline_segments, segment_times)
    
    # Plot multi-segment trajectory
    plot_multi_segment_trajectory(results, waypoints)
    
    print("\nCompleted all evaluations.")


if __name__ == "__main__":
    main()