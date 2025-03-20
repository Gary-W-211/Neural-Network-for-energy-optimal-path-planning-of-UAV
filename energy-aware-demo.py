"""
Energy-Aware Trajectory Generator Demo

This script demonstrates how to use the energy-aware trajectory generator to evaluate
energy consumption of different trajectories. It generates various trajectories
and uses an energy prediction model to evaluate their energy efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from spline_trajectory_generator import SplineTrajectoryGenerator
from generate_traj import quadrocoptertrajectory as quadtraj
from energy_aware_trajectory import EnergyAwareTrajectory
from sim_data_generator import add_simulation_support_to_energy_trajectory

def main():
    # Set model paths
    # Modify these paths to your model and scaler paths
    model_path = "results/bilstm/bilstm_model.pth"   # or use bilstm/bilstm_model.pth
    feature_scaler_path = "processed_data/feature_scaler.pkl"
    target_scaler_path = "processed_data/target_scaler.pkl"
    
    # If model file doesn't exist, display warning and use simulation
    use_simulation = False
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} does not exist! Using simulated data instead.")
        model_path = None
        feature_scaler_path = None
        target_scaler_path = None
        use_simulation = True
    
    # Create energy-aware trajectory generator
    energy_traj = EnergyAwareTrajectory(
        model_path=model_path,
        model_type='bilstm',  # or 'tcn'
        feature_scaler_path=feature_scaler_path,
        target_scaler_path=target_scaler_path
    )
    
    # Add simulation support if needed
    if use_simulation:
        energy_traj = add_simulation_support_to_energy_trajectory(energy_traj)
    
    # Define gravity direction
    gravity = [0, 0, -9.81]
    
    print("Demo 1: Evaluate Energy Consumption of a Single Trajectory")
    # Define initial state
    pos0 = [0, 0, 0]
    vel0 = [0, 0, 0]
    acc0 = [0, 0, 0]
    
    # Define goal state
    posf = [1, 1, 1]
    velf = [0, 0, 0]
    accf = [0, 0, 0]
    
    # Define duration
    Tf = 2.0
    
    # Create spline trajectory
    print("Generating spline trajectory...")
    spline_traj = SplineTrajectoryGenerator(pos0, vel0, acc0, gravity)
    spline_traj.set_goal_position(posf)
    spline_traj.set_goal_velocity(velf)
    spline_traj.set_goal_acceleration(accf)
    spline_traj.generate(Tf)
    
    # Evaluate trajectory energy consumption
    print("Evaluating trajectory energy consumption...")
    energy_results = energy_traj.evaluate_trajectory_energy(spline_traj)
    
    # Print energy consumption
    print(f"Total energy consumption: {energy_results['total_energy']:.2f} J")
    print(f"Average power: {energy_results['avg_power']:.2f} W")
    print(f"Maximum power: {energy_results['max_power']:.2f} W")
    
    # Plot results
    energy_traj.plot_trajectory_with_energy(spline_traj, energy_results, "Spline Trajectory Energy Analysis")
    
    print("\nDemo 2: Compare Energy Consumption of Different Trajectories")
    # Create different trajectories
    trajectories = []
    labels = []
    durations = []
    
    # Trajectory 1: Straight line
    print("Generating straight trajectory...")
    traj1 = SplineTrajectoryGenerator(pos0, vel0, acc0, gravity)
    traj1.set_goal_position([2, 0, 0])
    traj1.set_goal_velocity([0, 0, 0])
    traj1.generate(2.0)
    trajectories.append(traj1)
    labels.append("Straight Path")
    durations.append(2.0)
    
    # Trajectory 2: Arc path
    print("Generating arc trajectory...")
    traj2 = SplineTrajectoryGenerator(pos0, vel0, acc0, gravity)
    traj2.set_goal_position([2, 0, 0])
    traj2.set_goal_velocity([0, 0, 0])
    
    # Use more control points to make the trajectory an arc
    traj2.generate(2.0, num_points=15)
    trajectories.append(traj2)
    labels.append("Arc Path")
    durations.append(2.0)
    
    # Trajectory 3: Vertical up then horizontal
    print("Generating composite trajectory...")
    traj3 = SplineTrajectoryGenerator(pos0, vel0, acc0, gravity)
    traj3.set_goal_position([2, 0, 1])
    traj3.set_goal_velocity([0, 0, 0])
    traj3.generate(2.5)
    trajectories.append(traj3)
    labels.append("Composite Path")
    durations.append(2.5)
    
    # Compare trajectory energy consumption
    print("Comparing trajectory energy consumption...")
    comparison = energy_traj.compare_trajectories(trajectories, labels, durations)
    
    # Print comparison results
    for i, label in enumerate(labels):
        print(f"{label} - Total Energy: {comparison['total_energy'][i]:.2f} J, Average Power: {comparison['avg_power'][i]:.2f} W")
    
    # Plot comparison results
    energy_traj.plot_trajectory_comparison(comparison)
    
    print("\nDemo 3: Multi-Segment Trajectory Energy Consumption")
    # Define waypoints
    waypoints = [
        [0, 0, 0],    # Start point
        [1, 1, 1],    # First waypoint
        [2, 0, 1.5],  # Second waypoint
        [3, -1, 1],   # Third waypoint
        [4, 0, 0]     # End point
    ]
    
    # Define velocities at each waypoint
    velocities = [
        [0, 0, 0],      # Start velocity
        [0.5, 0.5, 0],  # First waypoint velocity
        [0.5, -0.5, 0], # Second waypoint velocity
        [0.5, 0, 0],    # Third waypoint velocity
        [0, 0, 0]       # End velocity
    ]
    
    # Define segment durations
    segment_times = [1.5, 1.5, 1.5, 1.5]
    
    # Create multi-segment trajectory
    print("Generating multi-segment trajectory...")
    segments = []
    
    for i in range(len(waypoints) - 1):
        # Current segment start and end points
        pos0 = waypoints[i]
        vel0 = velocities[i]
        acc0 = [0, 0, 0]  # Simplified: assume zero acceleration
        
        posf = waypoints[i + 1]
        velf = velocities[i + 1]
        accf = [0, 0, 0]
        
        # Create trajectory segment
        traj = SplineTrajectoryGenerator(pos0, vel0, acc0, gravity)
        traj.set_goal_position(posf)
        traj.set_goal_velocity(velf)
        traj.set_goal_acceleration(accf)
        traj.generate(segment_times[i])
        
        segments.append(traj)
    
    # Evaluate multi-segment trajectory energy consumption
    print("Evaluating multi-segment trajectory energy consumption...")
    multi_results = energy_traj.evaluate_multi_segment_trajectory(segments, segment_times)
    
    # Print total energy consumption
    print(f"Multi-segment trajectory total energy consumption: {multi_results['total_energy']:.2f} J")
    
    # Print each segment's energy consumption
    for i, segment_result in enumerate(multi_results['segments']):
        print(f"Segment {i+1} - Energy consumption: {segment_result['total_energy']:.2f} J")
    
    # Visualize multi-segment trajectory
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='3d')
    
    # Plot waypoints
    waypoints_array = np.array(waypoints)
    ax.scatter(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2], 
              c='g', marker='o', s=100, label='Waypoints')
    
    # Plot each trajectory segment
    time_offset = 0
    colors = plt.cm.jet(np.linspace(0, 1, len(segments)))
    
    for i, (segment, Tf) in enumerate(zip(segments, segment_times)):
        # Sample points
        t_samples = np.linspace(0, Tf, 50)
        positions = np.array([segment.get_position(t) for t in t_samples])
        
        # Plot trajectory segment
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=colors[i], linewidth=2, label=f'Segment {i+1}')
        
        time_offset += Tf
    
    # Set axis labels and title
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Multi-Segment Trajectory 3D Visualization')
    ax.legend()
    
    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\nDemo 4: Energy-Optimized Trajectory Generation by Waypoint Adjustment")
    print("Optimizing energy consumption by adjusting waypoint positions...")

    # Define start and end states
    pos0 = [0, 0, 0]
    vel0 = [0, 0, 0]
    acc0 = [0, 0, 0]

    posf = [3, 0, 0]
    velf = [0, 0, 0]
    accf = [0, 0, 0]

    # Define fixed duration for all trajectories
    fixed_duration = 3.0  # 3 seconds for the entire trajectory

    # Define the number of timesteps (at 0.1s intervals)
    timesteps = int(fixed_duration / 0.1)  # 30 timesteps for 3 seconds

    # Define different intermediate waypoint positions to try
    # We'll vary the y-coordinate of a waypoint at the middle of the trajectory
    y_positions = np.linspace(-1.0, 1.0, 11)  # From -1.0 to 1.0 meters
    z_positions = np.linspace(0.0, 1.0, 6)    # From 0.0 to 1.0 meters

    optimization_trajectories = []
    optimization_labels = []
    waypoint_combinations = []

    # Generate trajectories with different intermediate waypoints
    print("Generating trajectories with different waypoint positions...")
    for y_pos in y_positions:
        for z_pos in z_positions:
            # Create a multi-segment trajectory with an intermediate waypoint
            waypoints = [
                pos0,                # Start point
                [1.5, y_pos, z_pos], # Intermediate waypoint (x=1.5, varying y and z)
                posf                 # End point
            ]
            
            # Define velocities at each waypoint (you can also optimize these)
            velocities = [
                vel0,           # Start velocity
                [0.5, 0, 0],    # Intermediate waypoint velocity
                velf            # End velocity
            ]
            
            # Define segment durations (equal segments)
            segment_times = [fixed_duration/2, fixed_duration/2]
            
            # Create multi-segment trajectory
            segments = []
            
            for i in range(len(waypoints) - 1):
                # Current segment start and end points
                seg_pos0 = waypoints[i]
                seg_vel0 = velocities[i]
                seg_acc0 = [0, 0, 0]  # Assume zero acceleration
                
                seg_posf = waypoints[i + 1]
                seg_velf = velocities[i + 1]
                seg_accf = [0, 0, 0]
                
                # Create trajectory segment
                traj = SplineTrajectoryGenerator(seg_pos0, seg_vel0, seg_acc0, gravity)
                traj.set_goal_position(seg_posf)
                traj.set_goal_velocity(seg_velf)
                traj.set_goal_acceleration(seg_accf)
                traj.generate(segment_times[i])
                
                segments.append(traj)
            
            # Evaluate multi-segment trajectory energy consumption
            multi_results = energy_traj.evaluate_multi_segment_trajectory(segments, segment_times)
            
            # Store trajectory information
            optimization_trajectories.append(segments)
            optimization_labels.append(f"y={y_pos:.1f}, z={z_pos:.1f}")
            waypoint_combinations.append((y_pos, z_pos, multi_results['total_energy']))

    # Convert to numpy array for easier processing
    waypoint_combinations = np.array(waypoint_combinations)

    # Find trajectory with minimum energy consumption
    min_energy_idx = np.argmin(waypoint_combinations[:, 2])
    optimal_y = waypoint_combinations[min_energy_idx, 0]
    optimal_z = waypoint_combinations[min_energy_idx, 1]
    optimal_energy = waypoint_combinations[min_energy_idx, 2]

    print(f"Trajectory with minimum energy consumption has intermediate waypoint at:")
    print(f"Position: [1.5, {optimal_y:.1f}, {optimal_z:.1f}]")
    print(f"Optimal trajectory energy consumption: {optimal_energy:.2f} J")

    # Plot the energy landscape as a function of waypoint position
    plt.figure(figsize=(12, 8))

    # Create meshgrid for plotting
    y_mesh, z_mesh = np.meshgrid(y_positions, z_positions)
    energy_values = np.zeros((len(z_positions), len(y_positions)))

    # Fill in energy values
    count = 0
    for i in range(len(z_positions)):
        for j in range(len(y_positions)):
            energy_values[i, j] = waypoint_combinations[count, 2]
            count += 1

    # Create contour plot
    contour = plt.contourf(y_mesh, z_mesh, energy_values, 20, cmap='viridis')
    plt.colorbar(contour, label='Energy Consumption [J]')

    # Mark optimal point
    plt.scatter([optimal_y], [optimal_z], color='red', s=100, marker='*', 
            label=f'Optimal waypoint (y={optimal_y:.1f}, z={optimal_z:.1f})')

    plt.xlabel('Y Position [m]')
    plt.ylabel('Z Position [m]')
    plt.title('Energy Consumption vs. Intermediate Waypoint Position')
    plt.grid(True)
    plt.legend()

    # Visualize the optimal trajectory in 3D
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111, projection='3d')

    # Get the optimal trajectory segments
    optimal_segments = optimization_trajectories[min_energy_idx]
    optimal_waypoints = [
        [0, 0, 0],                     # Start
        [1.5, optimal_y, optimal_z],   # Optimal intermediate waypoint
        [3, 0, 0]                      # End
    ]

    # Plot waypoints
    waypoints_array = np.array(optimal_waypoints)
    ax.scatter(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2], 
            c='g', marker='o', s=100, label='Waypoints')

    # Plot each segment of the optimal trajectory
    colors = ['b', 'r']
    for i, segment in enumerate(optimal_segments):
        # Sample points along the segment
        t_samples = np.linspace(0, segment_times[i], 50)
        positions = np.array([segment.get_position(t) for t in t_samples])
        
        # Plot trajectory segment
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            color=colors[i], linewidth=2, label=f'Segment {i+1}')

    # Also plot a direct path for comparison
    direct_traj = SplineTrajectoryGenerator(pos0, vel0, acc0, gravity)
    direct_traj.set_goal_position(posf)
    direct_traj.set_goal_velocity(velf)
    direct_traj.set_goal_acceleration(accf)
    direct_traj.generate(fixed_duration)

    # Sample points along the direct trajectory
    t_samples = np.linspace(0, fixed_duration, 100)
    positions = np.array([direct_traj.get_position(t) for t in t_samples])

    # Plot direct trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
        color='gray', linewidth=1, linestyle='--', label='Direct Path')

    # Set axis labels and title
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Energy-Optimized Trajectory 3D Visualization')
    ax.legend()

    # Adjust view angle for better visualization
    ax.view_init(elev=20, azim=30)

    plt.tight_layout()
    plt.show()

    print("Energy-optimized trajectory generation complete!")
    print("\nAll demos completed!")

if __name__ == "__main__":
    main()