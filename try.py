import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import copy
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import matplotlib.collections

from spline_trajectory_generator import SplineTrajectoryGenerator
from energy_aware_trajectory import EnergyAwareTrajectory
from sim_data_generator import add_simulation_support_to_energy_trajectory

class LimitedVisionEnergyPlanner:
    """
    Spline-based energy-optimized path planning system for drones with limited vision.
    The drone can only "see" a short distance ahead and must make energy-efficient
    decisions based on this limited information using spline curves for smooth trajectory.
    """
    
    def __init__(self, energy_model_path=None, vision_horizon=0.8, feature_scaler_path=None, target_scaler_path=None):
        """
        Initialize the planner
        
        Parameters:
            energy_model_path: Path to the energy prediction model
            vision_horizon: Drone's vision range in seconds, default is 0.8
            feature_scaler_path: Path to feature scaler
            target_scaler_path: Path to target scaler
        """
        self.vision_horizon = vision_horizon
        
        # Initialize gravity direction
        self.gravity = [0, 0, -9.81]
        
        # Initialize energy prediction model
        self.setup_energy_model(energy_model_path, feature_scaler_path, target_scaler_path)
        
        # Store complete planned path
        self.planned_trajectory_segments = []
        self.planned_segment_times = []
        self.waypoints = []
        
        # Store energy consumption data
        self.energy_data = []
        self.cumulative_energy = 0
        
        # Trajectory exploration parameters
        self.num_candidates = 7  # Number of candidate trajectories to consider in each planning step
    
    def setup_energy_model(self, model_path, feature_scaler_path=None, target_scaler_path=None):
        """Set up the energy prediction model"""
        # Check if model file exists
        use_simulation = False
        if model_path is None or not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} does not exist! Using simulated data instead.")
            model_path = None
            feature_scaler_path = None
            target_scaler_path = None
            use_simulation = True
        else:
            # If no scaler paths provided, use default values
            if feature_scaler_path is None:
                feature_scaler_path = "processed_data/feature_scaler.pkl"
            if target_scaler_path is None:
                target_scaler_path = "processed_data/target_scaler.pkl"
            
            # Check if scaler files exist
            if not os.path.exists(feature_scaler_path):
                print(f"Warning: Feature scaler file {feature_scaler_path} does not exist!")
            if not os.path.exists(target_scaler_path):
                print(f"Warning: Target scaler file {target_scaler_path} does not exist!")
        
        # Create energy predictor
        self.energy_predictor = EnergyAwareTrajectory(
            model_path=model_path,
            model_type='bilstm',  # Use BiLSTM model
            feature_scaler_path=feature_scaler_path,
            target_scaler_path=target_scaler_path
        )
        
        # Add simulation support if needed
        if use_simulation:
            self.energy_predictor = add_simulation_support_to_energy_trajectory(self.energy_predictor)
    
    def generate_candidate_trajectories(self, current_state, goal_position):
        """
        Generate candidate trajectories using spline curves
        
        Parameters:
            current_state: Current state [position, velocity, acceleration]
            goal_position: Final goal position
        
        Returns:
            List of candidate trajectories and their parameters
        """
        current_pos, current_vel, current_acc = current_state
        
        # Calculate direction vector from current position to goal
        direction_to_goal = np.array(goal_position) - np.array(current_pos)
        distance_to_goal = np.linalg.norm(direction_to_goal)
        
        # If very close to goal, return a direct trajectory to goal
        if distance_to_goal < 0.5:
            # Create direct trajectory to goal with high-order spline for smoothness
            traj = SplineTrajectoryGenerator(current_pos, current_vel, current_acc, self.gravity)
            traj.set_goal_position(goal_position)
            traj.set_goal_velocity([0, 0, 0])
            traj.set_goal_acceleration([0, 0, 0])
            # Increase control points for smoother curve
            traj.generate(self.vision_horizon, num_points=15)
            return [traj], [self.vision_horizon], [goal_position]
        
        # Normalize direction vector
        if distance_to_goal > 0:
            normalized_direction = direction_to_goal / distance_to_goal
        else:
            normalized_direction = np.array([1, 0, 0])  # Default forward direction
        
        # Generate possible intermediate target points
        candidate_trajectories = []
        candidate_durations = []
        candidate_waypoints = []
        
        # Define perturbation magnitude - decrease as approach goal
        deviation_scale = min(2.0, distance_to_goal / 3)
        
        # Generate candidate paths
        for i in range(self.num_candidates):
            # Calculate forward distance to move (within vision range)
            forward_distance = min(self.vision_horizon * 2, distance_to_goal * 0.3)
            
            # Set different control point counts for different candidates to create diversity
            # Higher control point count gives smoother trajectories but more computation
            control_points = 10 + i % 6  # Varies from 10 to 15
            
            # First candidate is always direct path toward goal
            if i == 0:
                deviation = np.array([0, 0, 0])
                # First path uses more control points for smoothness
                control_points = 15
            else:
                # Other candidates have offsets perpendicular to forward direction
                # Create two basis vectors perpendicular to forward direction
                if abs(normalized_direction[0]) < 0.9:
                    perpendicular1 = np.cross(normalized_direction, [1, 0, 0])
                else:
                    perpendicular1 = np.cross(normalized_direction, [0, 1, 0])
                perpendicular1 = perpendicular1 / np.linalg.norm(perpendicular1)
                perpendicular2 = np.cross(normalized_direction, perpendicular1)
                perpendicular2 = perpendicular2 / np.linalg.norm(perpendicular2)
                
                # Use trigonometric functions to generate evenly distributed offsets
                angle = 2 * np.pi * (i-1) / (self.num_candidates-1)
                deviation = (perpendicular1 * np.cos(angle) + perpendicular2 * np.sin(angle)) * deviation_scale
            
            # Calculate intermediate control point for smoother spline curve
            intermediate_waypoint = np.array(current_pos) + normalized_direction * forward_distance + deviation
            
            # Create spline trajectory
            traj = SplineTrajectoryGenerator(current_pos, current_vel, current_acc, self.gravity)
            traj.set_goal_position(intermediate_waypoint.tolist())
            
            # Set goal velocity - maintain toward the final goal
            target_speed = min(3.0, distance_to_goal / 2)  # Speed limit
            goal_vel = normalized_direction * target_speed
            traj.set_goal_velocity(goal_vel.tolist())
            
            # Set goal acceleration - for smoother trajectory
            # Different types of acceleration for different candidates
            if i % 3 == 0:
                # Decelerating trajectory
                goal_acc = -normalized_direction * 0.3
                # Make sure we pass a list not a numpy array
                goal_acc = goal_acc.tolist()
            elif i % 3 == 1:
                # Constant velocity trajectory
                goal_acc = [0, 0, 0]
            else:
                # Accelerating trajectory
                goal_acc = normalized_direction * 0.3
                # Make sure we pass a list not a numpy array
                goal_acc = goal_acc.tolist()

            traj.set_goal_acceleration(goal_acc)
            
            # Generate spline trajectory with specified control points
            traj.generate(self.vision_horizon, num_points=control_points)
            
            candidate_trajectories.append(traj)
            candidate_durations.append(self.vision_horizon)
            candidate_waypoints.append(intermediate_waypoint.tolist())
        
        return candidate_trajectories, candidate_durations, candidate_waypoints
    
    def evaluate_candidates(self, trajectories, durations, waypoints, goal_position):
        """
        Evaluate candidate trajectories and select the best one, considering spline smoothness
        
        Parameters:
            trajectories: List of candidate trajectories
            durations: List of trajectory durations
            waypoints: List of trajectory endpoint positions
            goal_position: Final goal position
        
        Returns:
            Index of best trajectory
        """
        best_score = float('inf')
        best_idx = 0
        
        # Evaluate each candidate trajectory
        for i, (traj, duration) in enumerate(zip(trajectories, durations)):
            # Calculate energy consumption
            energy_results = self.energy_predictor.evaluate_trajectory_energy(traj)
            energy_consumption = energy_results['total_energy']
            
            # Calculate distance from endpoint to goal
            end_waypoint = waypoints[i]
            distance_to_goal = np.linalg.norm(np.array(end_waypoint) - np.array(goal_position))
            
            # Calculate spline smoothness (by evaluating acceleration changes)
            smoothness_score = 0
            num_samples = 10
            t_samples = np.linspace(0, duration, num_samples)
            
            # Get acceleration samples
            acc_samples = np.array([np.linalg.norm(traj.get_acceleration(t)) for t in t_samples])
            
            # Calculate rate of change of acceleration (jerk) - approximate
            if len(acc_samples) > 1:
                acc_changes = np.abs(acc_samples[1:] - acc_samples[:-1])
                max_acc_change = np.max(acc_changes) if len(acc_changes) > 0 else 0
                avg_acc_change = np.mean(acc_changes) if len(acc_changes) > 0 else 0
                
                # Smoothness score (lower is better - less acceleration change)
                smoothness_score = avg_acc_change + 0.5 * max_acc_change
            
            # Comprehensive score (energy consumption + distance weight + smoothness)
            # Gradually increase distance weight as approach goal
            distance_to_goal_normalized = min(1.0, distance_to_goal / 10.0)
            energy_weight = 1.0
            distance_weight = 3.0 * (1.0 - distance_to_goal_normalized)
            smoothness_weight = 0.5  # Smoothness weight
            
            score = (energy_weight * energy_consumption + 
                    distance_weight * distance_to_goal + 
                    smoothness_weight * smoothness_score)
            
            # Print detailed evaluation info (uncomment for debugging)
            # print(f"Trajectory {i}: Energy={energy_consumption:.2f}J, Distance={distance_to_goal:.2f}m, "
            #       f"Smoothness={smoothness_score:.2f}, Total Score={score:.2f}")
            
            if score < best_score:
                best_score = score
                best_idx = i
        
        return best_idx
    
    def plan_trajectory(self, start_state, goal_position, max_iterations=100):
        """
        Plan a complete trajectory from start to goal
        
        Parameters:
            start_state: Initial state [position, velocity, acceleration]
            goal_position: Goal position
            max_iterations: Maximum number of iterations
        
        Returns:
            Planning status (True for success)
        """
        # Initialize planning
        self.planned_trajectory_segments = []
        self.planned_segment_times = []
        self.waypoints = []
        self.energy_data = []
        self.cumulative_energy = 0
        
        # Record start point
        start_pos, start_vel, start_acc = start_state
        self.waypoints.append(start_pos)
        
        # Current state
        current_state = copy.deepcopy(start_state)
        
        # Planning iterations
        for iteration in range(max_iterations):
            print(f"Planning iteration {iteration+1}/{max_iterations}")
            current_pos, current_vel, current_acc = current_state
            
            # Check if goal reached
            distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(goal_position))
            if distance_to_goal < 0.2:  # Threshold for goal reached
                print(f"Goal reached! Total iterations: {iteration+1}")
                self.waypoints.append(goal_position)
                return True
            
            # Generate candidate trajectories
            candidates, durations, candidate_waypoints = self.generate_candidate_trajectories(
                current_state, goal_position
            )
            
            # Visualize candidates (for debugging)
            # self.visualize_candidates(candidates, durations, current_pos, goal_position)
            
            # Evaluate trajectories and select best
            best_idx = self.evaluate_candidates(candidates, durations, candidate_waypoints, goal_position)
            best_trajectory = candidates[best_idx]
            best_duration = durations[best_idx]
            best_waypoint = candidate_waypoints[best_idx]
            
            # Evaluate selected trajectory energy consumption
            energy_results = self.energy_predictor.evaluate_trajectory_energy(best_trajectory)
            segment_energy = energy_results['total_energy']
            self.cumulative_energy += segment_energy
            self.energy_data.append(
                (segment_energy, self.cumulative_energy, best_waypoint)
            )
            
            # Add trajectory segment to plan
            self.planned_trajectory_segments.append(best_trajectory)
            self.planned_segment_times.append(best_duration)
            self.waypoints.append(best_waypoint)
            
            # Update current state for next iteration
            # Use selected trajectory state at time=horizon as next starting point
            next_pos = best_trajectory.get_position(best_duration)
            next_vel = best_trajectory.get_velocity(best_duration)
            next_acc = best_trajectory.get_acceleration(best_duration)
            
            current_state = [next_pos, next_vel, next_acc]
            
            print(f"  Current position: {next_pos}, distance to goal: {distance_to_goal:.2f}m")
            print(f"  Segment energy: {segment_energy:.2f}J, cumulative energy: {self.cumulative_energy:.2f}J")
        
        print("Maximum iterations reached!")
        return False
    
    def visualize_trajectory(self, start_position, goal_position, output_file="trajectory_animation.gif", waypoints_to_display=None):
        """
        Visualize the planned trajectory and save as GIF
        
        Parameters:
            start_position: Starting position
            goal_position: Goal position
            output_file: Output file name
            waypoints_to_display: Additional waypoints to display (e.g., spiral waypoints)
        """
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Create 3D subplot
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)
        
        # Get color map
        cmap = get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=len(self.waypoints))
        
        # Calculate all points in the trajectory to determine plot range
        all_positions = []
        for segment, duration in zip(self.planned_trajectory_segments, self.planned_segment_times):
            t_samples = np.linspace(0, duration, 10)
            for t in t_samples:
                all_positions.append(segment.get_position(t))
        
        # Add start, end, and any provided waypoints
        all_positions.append(start_position)
        all_positions.append(goal_position)
        if waypoints_to_display is not None:
            all_positions.extend(waypoints_to_display)
        
        # Calculate coordinate ranges
        all_positions = np.array(all_positions)
        
        x_min, y_min, z_min = all_positions.min(axis=0) - 0.5
        x_max, y_max, z_max = all_positions.max(axis=0) + 0.5
        
        def update(frame):
            # Clear subplots
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # Set 3D subplot title and labels
            ax1.set_title('Drone Trajectory Planning Visualization')
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.set_zlabel('Z [m]')
            
            # Set fixed axis ranges
            ax1.set_xlim([x_min, x_max])
            ax1.set_ylim([y_min, y_max])
            ax1.set_zlim([z_min, z_max])
            
            # Plot start and end points
            ax1.scatter([start_position[0]], [start_position[1]], [start_position[2]], 
                      color='green', marker='^', s=100, label='Start')
            ax1.scatter([goal_position[0]], [goal_position[1]], [goal_position[2]], 
                      color='red', marker='*', s=100, label='Goal')
            
            # Plot predefined waypoints if provided
            if waypoints_to_display is not None:
                waypoints_array = np.array(waypoints_to_display)
                ax1.scatter(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2], 
                          color='purple', marker='x', s=50, label='Target Waypoints')
                
                # Connect waypoints with a dashed line to show the ideal path
                ax1.plot(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2], 
                       'purple', linestyle='--', linewidth=1, alpha=0.5)
            
            # Determine number of trajectory segments to draw
            segments_to_draw = min(frame + 1, len(self.planned_trajectory_segments))
            
            # Draw the planned trajectory segments
            for i in range(segments_to_draw):
                segment = self.planned_trajectory_segments[i]
                duration = self.planned_segment_times[i]
                
                # Sample points along the segment
                t_samples = np.linspace(0, duration, 50)
                positions = np.array([segment.get_position(t) for t in t_samples])
                
                # Draw trajectory segment
                color = cmap(norm(i))
                ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                       color=color, linewidth=2)
                
                # Draw each waypoint
                if i < len(self.waypoints) - 1:  # Exclude the last point, which is the trajectory end
                    ax1.scatter([self.waypoints[i+1][0]], [self.waypoints[i+1][1]], [self.waypoints[i+1][2]], 
                               color=color, marker='o', s=30)
            
            # Show current drone position (latest planned waypoint)
            if segments_to_draw > 0 and segments_to_draw <= len(self.waypoints) - 1:
                current_pos = self.waypoints[segments_to_draw]
                ax1.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                          color='blue', marker='s', s=100, label='Current Position')
            
            # Draw energy consumption plots
            if len(self.energy_data) > 0:
                # Extract data
                segment_energies = [data[0] for data in self.energy_data[:segments_to_draw]]
                cumulative_energies = [data[1] for data in self.energy_data[:segments_to_draw]]
                iterations = list(range(1, segments_to_draw+1))
                
                # Plot segment energy consumption
                ax2.bar(iterations, segment_energies, color='skyblue')
                ax2.set_title('Segment Energy Consumption')
                ax2.set_xlabel('Trajectory Segment')
                ax2.set_ylabel('Energy [J]')
                ax2.grid(True)
                
                # Plot cumulative energy consumption
                ax3.plot(iterations, cumulative_energies, 'ro-')
                ax3.set_title('Cumulative Energy Consumption')
                ax3.set_xlabel('Trajectory Segment')
                ax3.set_ylabel('Cumulative Energy [J]')
                ax3.grid(True)
            
            ax1.legend()
            plt.tight_layout()
            
            return (ax1, ax2, ax3)
        
        # Create animation
        print("Creating trajectory animation...")
        anim = animation.FuncAnimation(
            fig, update, frames=len(self.planned_trajectory_segments)+5, 
            interval=500, blit=False
        )
        
        # Save as GIF
        print(f"Saving animation to {output_file}...")
        anim.save(output_file, writer='pillow', fps=2, dpi=100)
        
        plt.close(fig)
        print("Animation saved!")
        
        # Create static 3D plot of trajectory
        self.create_static_trajectory_plot(start_position, goal_position, 
                                          output_file.replace('.gif', '_static.png'),
                                          waypoints_to_display)
    
    def create_static_trajectory_plot(self, start_position, goal_position, output_file, waypoints_to_display=None):
        """Create a static 3D plot of the trajectory"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set title and labels
        ax.set_title('Global Energy-Optimized Spiral Trajectory')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        # Plot start and end points
        ax.scatter([start_position[0]], [start_position[1]], [start_position[2]], 
                  color='green', marker='^', s=100, label='Start')
        ax.scatter([goal_position[0]], [goal_position[1]], [goal_position[2]], 
                  color='red', marker='*', s=100, label='Goal')
        
        # Plot predefined waypoints if provided
        if waypoints_to_display is not None:
            waypoints_array = np.array(waypoints_to_display)
            ax.scatter(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2], 
                      color='purple', marker='x', s=50, label='Target Waypoints')
            
            # Connect waypoints with a dashed line to show the ideal path
            ax.plot(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2], 
                   'purple', linestyle='--', linewidth=1, alpha=0.5)
        
        # Plot all trajectory waypoints
        waypoints_array = np.array(self.waypoints)
        ax.scatter(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2], 
                  color='blue', marker='o', s=30, label='Planned Waypoints')
        
        # Plot trajectory lines
        for i, (segment, duration) in enumerate(zip(self.planned_trajectory_segments, self.planned_segment_times)):
            t_samples = np.linspace(0, duration, 50)
            positions = np.array([segment.get_position(t) for t in t_samples])
            
            # Get color based on energy consumption
            if i < len(self.energy_data):
                # Normalize energy value for color
                energy = self.energy_data[i][0]
                max_energy = max([data[0] for data in self.energy_data])
                color_intensity = 0.2 + 0.8 * (energy / max_energy)
                color = plt.cm.jet(color_intensity)
            else:
                color = 'gray'
            
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   color=color, linewidth=2)
        
        # Add energy consumption color bar
        if self.energy_data:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, 
                                      norm=plt.Normalize(vmin=0, 
                                                       vmax=max([data[0] for data in self.energy_data])))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1)
            cbar.set_label('Segment Energy Consumption [J]')
        
        # Set multiple view angles for comprehensive visualization
        views = [
            (30, 135),  # Default view
            (0, 0),     # Front view
            (0, 90),    # Side view
            (90, 0),    # Top view
        ]
        
        # Save all view angles as separate files
        for i, (elev, azim) in enumerate(views):
            ax.view_init(elev=elev, azim=azim)
            view_name = ["default", "front", "side", "top"][i]
            
            # Set legend and save
            ax.legend()
            plt.tight_layout()
            view_output_file = output_file.replace('.png', f'_{view_name}.png')
            plt.savefig(view_output_file, dpi=300, bbox_inches='tight')
            print(f"Static {view_name} view saved to {view_output_file}")
        
        plt.close(fig)
        
        # Create a 2D top-down view with trajectory height encoded as color
        self.create_topdown_height_plot(start_position, goal_position,
                                      output_file.replace('.png', '_topdown_height.png'),
                                      waypoints_to_display)

    def create_topdown_height_plot(self, start_position, goal_position, output_file, waypoints_to_display=None):
        """Create a 2D top-down plot with trajectory height encoded as color"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        # Set title and labels
        ax.set_title('Top-Down View with Height Color Encoding')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        
        # Plot start and end points
        ax.scatter([start_position[0]], [start_position[1]], 
                  color='green', marker='^', s=100, label='Start')
        ax.scatter([goal_position[0]], [goal_position[1]], 
                  color='red', marker='*', s=100, label='Goal')
        
        # Plot predefined waypoints if provided
        if waypoints_to_display is not None:
            waypoints_array = np.array(waypoints_to_display)
            
            # Extract heights for waypoints
            heights = waypoints_array[:, 2]
            
            # Create scatter plot with height-based coloring
            sc = ax.scatter(waypoints_array[:, 0], waypoints_array[:, 1], 
                           c=heights, cmap='viridis', marker='x', s=50, 
                           label='Target Waypoints')
            
            # Connect waypoints with a line
            ax.plot(waypoints_array[:, 0], waypoints_array[:, 1], 
                   'purple', linestyle='--', linewidth=1, alpha=0.5)
            
        # Plot all trajectory segments with height-based coloring
        for i, (segment, duration) in enumerate(zip(self.planned_trajectory_segments, self.planned_segment_times)):
            t_samples = np.linspace(0, duration, 50)
            positions = np.array([segment.get_position(t) for t in t_samples])
            
            # Create line segments for colormapping based on height
            # This uses the Line Collection approach for variable color along a line
            points = np.array([positions[:-1, 0:2], positions[1:, 0:2]]).transpose(1, 0, 2)
            heights = positions[:, 2]  # Z coordinates for coloring
            
            # Create line collection
            lc = matplotlib.collections.LineCollection(
                points, cmap='plasma', 
                norm=plt.Normalize(heights.min(), heights.max())
            )
            lc.set_array(heights[:-1])  # Set color values
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
        
        # Add height color bar
        cbar = plt.colorbar(line, ax=ax)
        cbar.set_label('Height [m]')
        
        # Set equal aspect ratio for better visualization
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Top-down height view saved to {output_file}")


def generate_spiral_waypoints(center=[5, 0, 0], radius_start=3, radius_end=1, 
                             height_start=0, height_end=5, num_turns=2.5, num_points=15,
                             spacing_type="distance"):
    """
    Generate waypoints that form a spiral ascending trajectory.
    
    Parameters:
        center: Center point of the spiral [x, y, z]
        radius_start: Starting radius of the spiral
        radius_end: Ending radius of the spiral
        height_start: Starting height of the spiral
        height_end: Ending height of the spiral
        num_turns: Number of complete turns in the spiral
        num_points: Number of waypoints to generate
        spacing_type: How to space waypoints - "uniform" (equal parameter spacing) 
                     or "distance" (roughly equal 3D distance between points)
        
    Returns:
        List of waypoint coordinates
    """
    if spacing_type == "distance":
        # Use equal arc length parametrization (approximate)
        waypoints = []
        
        # Calculate total spiral length (approximation)
        total_height = height_end - height_start
        avg_radius = (radius_start + radius_end) / 2
        arc_length = np.sqrt((2 * np.pi * avg_radius * num_turns)**2 + total_height**2)
        
        # Calculate how to distribute points along the spiral
        # for more equal spacing between waypoints
        points_per_turn = max(2, int(num_points / num_turns))
        
        # Generate points with gradually decreasing spacing as radius decreases
        actual_points = []
        
        # Make sure we cover from start to end with better spacing
        # Create a non-linear parameterization to ensure better distance distribution
        t_values = np.linspace(0, 1, num_points)**0.75  # Non-linear parameterization
        
        for t in t_values:
            # Calculate angle (0 to 2π × num_turns)
            angle = 2 * np.pi * num_turns * t
            
            # Calculate radius (linearly interpolate from start to end)
            radius = radius_start + (radius_end - radius_start) * t
            
            # Calculate height (linearly interpolate from start to end)
            height = height_start + (height_end - height_start) * t
            
            # Calculate position
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2] + height
            
            actual_points.append([x, y, z])
        
        # Make sure we include the exact start and end points
        start_point = [
            center[0] + radius_start * np.cos(0),
            center[1] + radius_start * np.sin(0),
            center[2] + height_start
        ]
        
        end_point = [
            center[0] + radius_end * np.cos(2 * np.pi * num_turns),
            center[1] + radius_end * np.sin(2 * np.pi * num_turns),
            center[2] + height_end
        ]
        
        # Replace first and last points to ensure exact start/end
        if len(actual_points) > 2:
            actual_points[0] = start_point
            actual_points[-1] = end_point
        
        return actual_points
    else:
        # Original uniform parameter spacing
        waypoints = []
        
        for i in range(num_points):
            # Normalized position (0 to 1)
            t = i / (num_points - 1)
            
            # Calculate angle (0 to 2π × num_turns)
            angle = 2 * np.pi * num_turns * t
            
            # Calculate radius (linearly interpolate from start to end)
            radius = radius_start + (radius_end - radius_start) * t
            
            # Calculate height (linearly interpolate from start to end)
            height = height_start + (height_end - height_start) * t
            
            # Calculate position
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2] + height
            
            waypoints.append([x, y, z])
        
        return waypoints


def plan_through_waypoints(planner, start_state, waypoints, max_iterations_per_segment=15):
    """
    Plan a trajectory that passes through all specified waypoints.
    
    Parameters:
        planner: The LimitedVisionEnergyPlanner instance
        start_state: Initial state [position, velocity, acceleration]
        waypoints: List of waypoints to pass through
        max_iterations_per_segment: Maximum iterations for each waypoint-to-waypoint segment
        
    Returns:
        success: True if planning was successful
    """
    # Initialize planning
    planner.planned_trajectory_segments = []
    planner.planned_segment_times = []
    planner.waypoints = []
    planner.energy_data = []
    planner.cumulative_energy = 0
    
    # Record the start point
    start_pos, start_vel, start_acc = start_state
    planner.waypoints.append(start_pos)
    
    # Current state starts from the initial state
    current_state = copy.deepcopy(start_state)
    
    # Plan through each waypoint
    for waypoint_idx, target_waypoint in enumerate(waypoints):
        print(f"Planning to waypoint {waypoint_idx+1}/{len(waypoints)}: {target_waypoint}")
        
        # Plan a trajectory segment to the current waypoint
        segment_successful = plan_single_segment(
            planner, current_state, target_waypoint, 
            max_iterations_per_segment, f"Waypoint {waypoint_idx+1}"
        )
        
        if not segment_successful:
            print(f"Failed to reach waypoint {waypoint_idx+1}")
            return False
        
        # Update current state to the end of the latest trajectory segment
        latest_segment = planner.planned_trajectory_segments[-1]
        latest_duration = planner.planned_segment_times[-1]
        
        next_pos = latest_segment.get_position(latest_duration)
        next_vel = latest_segment.get_velocity(latest_duration)
        next_acc = latest_segment.get_acceleration(latest_duration)
        
        current_state = [next_pos, next_vel, next_acc]
        
        print(f"  Reached waypoint {waypoint_idx+1}, cumulative energy: {planner.cumulative_energy:.2f}J")
    
    return True


def plan_single_segment(planner, start_state, goal_position, max_iterations, segment_name=""):
    """
    Plan a trajectory segment from start_state to goal_position.
    
    Parameters:
        planner: The LimitedVisionEnergyPlanner instance
        start_state: Starting state [position, velocity, acceleration]
        goal_position: Target position to reach
        max_iterations: Maximum planning iterations
        segment_name: Name of this segment for logging
        
    Returns:
        success: True if planning was successful
    """
    # Current state starts from the provided start state
    current_state = copy.deepcopy(start_state)
    current_pos, current_vel, current_acc = current_state
    
    # Keep track of previous distance to detect if we're making progress
    prev_distance = np.inf
    stall_counter = 0
    
    # Planning iterations for this segment
    for iteration in range(max_iterations):
        print(f"  {segment_name} planning iteration {iteration+1}/{max_iterations}")
        
        # Check if we've reached the goal
        distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(goal_position))
        
        # If we're close enough to the goal, consider it reached
        if distance_to_goal < 0.5:  # Increased threshold for considering goal reached
            print(f"  Reached {segment_name}! Total iterations: {iteration+1}")
            
            # Add a final direct segment to the exact goal position for completeness
            final_traj = SplineTrajectoryGenerator(current_pos, current_vel, current_acc, planner.gravity)
            final_traj.set_goal_position(goal_position)
            final_traj.set_goal_velocity([0, 0, 0])
            final_traj.set_goal_acceleration([0, 0, 0])
            final_traj.generate(0.5, num_points=20)  # Short, precise final segment
            
            # Evaluate energy for final segment
            energy_results = planner.energy_predictor.evaluate_trajectory_energy(final_traj)
            segment_energy = energy_results['total_energy']
            planner.cumulative_energy += segment_energy
            
            # Add final segment
            planner.planned_trajectory_segments.append(final_traj)
            planner.planned_segment_times.append(0.5)
            planner.waypoints.append(goal_position)
            planner.energy_data.append(
                (segment_energy, planner.cumulative_energy, goal_position)
            )
            
            return True
        
        # Check if we're making progress
        if abs(prev_distance - distance_to_goal) < 0.05:
            stall_counter += 1
        else:
            stall_counter = 0
        
        # If we're stalled for too long, adjust planning strategy
        if stall_counter >= 3:
            print(f"  Progress stalled, using direct targeting for {segment_name}")
            
            # Create a more direct trajectory to the goal
            direct_traj = SplineTrajectoryGenerator(current_pos, current_vel, current_acc, planner.gravity)
            direct_traj.set_goal_position(goal_position)
            
            # Set a higher velocity toward goal to make faster progress
            direction_to_goal = np.array(goal_position) - np.array(current_pos)
            if np.linalg.norm(direction_to_goal) > 0:
                direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)
            goal_vel = direction_to_goal * 5.0  # Increased velocity
            direct_traj.set_goal_velocity(goal_vel.tolist())
            direct_traj.set_goal_acceleration([0, 0, 0])
            
            # Longer segment to try to reach goal in one go
            direct_duration = min(2.0, distance_to_goal / 2.0)
            direct_traj.generate(direct_duration, num_points=25)
            
            # Evaluate energy
            energy_results = planner.energy_predictor.evaluate_trajectory_energy(direct_traj)
            segment_energy = energy_results['total_energy']
            planner.cumulative_energy += segment_energy
            
            # Add direct segment
            planner.planned_trajectory_segments.append(direct_traj)
            planner.planned_segment_times.append(direct_duration)
            
            # Get end position
            next_pos = direct_traj.get_position(direct_duration)
            next_vel = direct_traj.get_velocity(direct_duration)
            next_acc = direct_traj.get_acceleration(direct_duration)
            
            planner.waypoints.append(next_pos)
            planner.energy_data.append(
                (segment_energy, planner.cumulative_energy, next_pos)
            )
            
            # Update state
            current_state = [next_pos, next_vel, next_acc]
            current_pos = next_pos
            current_vel = next_vel
            current_acc = next_acc
            
            print(f"    Direct approach: {next_pos}, new distance: {np.linalg.norm(np.array(next_pos) - np.array(goal_position)):.2f}m")
            print(f"    Segment energy: {segment_energy:.2f}J, cumulative energy: {planner.cumulative_energy:.2f}J")
            
            # Reset stall counter
            stall_counter = 0
            prev_distance = np.linalg.norm(np.array(current_pos) - np.array(goal_position))
            continue
        
        # Generate candidate trajectories
        candidates, durations, candidate_waypoints = planner.generate_candidate_trajectories(
            current_state, goal_position
        )
        
        # Evaluate trajectories and select the best
        best_idx = planner.evaluate_candidates(candidates, durations, candidate_waypoints, goal_position)
        best_trajectory = candidates[best_idx]
        best_duration = durations[best_idx]
        best_waypoint = candidate_waypoints[best_idx]
        
        # Evaluate energy consumption of selected trajectory
        energy_results = planner.energy_predictor.evaluate_trajectory_energy(best_trajectory)
        segment_energy = energy_results['total_energy']
        planner.cumulative_energy += segment_energy
        planner.energy_data.append(
            (segment_energy, planner.cumulative_energy, best_waypoint)
        )
        
        # Add trajectory segment to plan
        planner.planned_trajectory_segments.append(best_trajectory)
        planner.planned_segment_times.append(best_duration)
        planner.waypoints.append(best_waypoint)
        
        # Update current state for next iteration
        next_pos = best_trajectory.get_position(best_duration)
        next_vel = best_trajectory.get_velocity(best_duration)
        next_acc = best_trajectory.get_acceleration(best_duration)
        
        current_state = [next_pos, next_vel, next_acc]
        current_pos = next_pos
        current_vel = next_vel
        current_acc = next_acc
        
        print(f"    Current position: {next_pos}, distance to goal: {distance_to_goal:.2f}m")
        print(f"    Segment energy: {segment_energy:.2f}J, cumulative energy: {planner.cumulative_energy:.2f}J")
        
        # Update previous distance
        prev_distance = np.linalg.norm(np.array(current_pos) - np.array(goal_position))
    
    # If we've used all iterations but made good progress (within 1.0m), consider it a success
    final_distance = np.linalg.norm(np.array(current_pos) - np.array(goal_position))
    if final_distance < 1.0:
        print(f"  Close enough to {segment_name} (distance: {final_distance:.2f}m). Proceeding to next waypoint.")
        return True
        
    print(f"  Reached maximum iterations for {segment_name}!")
    return False


def main():
    # Set energy model paths
    model_path = "results/bilstm/bilstm_model.pth"
    feature_scaler_path = "processed_data/feature_scaler.pkl"
    target_scaler_path = "processed_data/target_scaler.pkl"
    
    # Create planner
    planner = LimitedVisionEnergyPlanner(
        energy_model_path=model_path,
        vision_horizon=0.8  # Increased vision horizon for better planning
    )
    
    # Set start state
    start_position = [0, 0, 0]
    start_velocity = [0, 0, 0]
    start_acceleration = [0, 0, 0]
    start_state = [start_position, start_velocity, start_acceleration]
    
    # Define spiral trajectory waypoints (these will guide the planner)
    # Format: [x, y, z]
    spiral_waypoints = generate_spiral_waypoints(
        center=[5, 0, 0],      # Center of spiral (x, y, z)
        radius_start=3,        # Starting radius
        radius_end=1,          # Ending radius
        height_start=0,        # Starting height
        height_end=5,          # Ending height
        num_turns=2.5,         # Number of spiral turns
        num_points=10,         # Reduced number of waypoints for efficiency
        spacing_type="distance" # Equal distance spacing for better planning
    )
    
    # Set final goal position (the last waypoint of the spiral)
    goal_position = spiral_waypoints[-1]
    
    print("=== Spline-Based Energy-Aware Path Planning System with Spiral Trajectory ===")
    print("The drone will generate a smooth spiral ascending trajectory using splines,")
    print("making energy-optimized decisions with limited vision (0.8s horizon).")
    print(f"Start: {start_position}, Goal: {goal_position}")
    print(f"Total waypoints to pass through: {len(spiral_waypoints)}")
    print("Starting planning...")
    
    # Perform multi-waypoint trajectory planning
    planning_successful = plan_through_waypoints(
        planner, start_state, spiral_waypoints, max_iterations_per_segment=15
    )
    
    if planning_successful:
        print("\n=== Trajectory Planning Successful! ===")
        print(f"Total energy consumption: {planner.cumulative_energy:.2f} J")
        print(f"Total waypoints: {len(planner.waypoints)}")
        print(f"Generated trajectory segments: {len(planner.planned_trajectory_segments)}")
        print("\nSpline curve analysis:")
        
        # Analyze the generated spline curve quality
        total_curvature = 0
        max_curvature = 0
        
        for i, segment in enumerate(planner.planned_trajectory_segments):
            duration = planner.planned_segment_times[i]
            # Sampling points
            num_samples = 20
            t_samples = np.linspace(0, duration, num_samples)
            
            # Calculate approximate curvature (using finite differences)
            positions = np.array([segment.get_position(t) for t in t_samples])
            
            if len(positions) > 2:
                # Calculate displacement vectors
                displacements = positions[1:] - positions[:-1]
                
                # Calculate angle changes (rough estimate of curvature)
                dot_products = np.sum(displacements[:-1] * displacements[1:], axis=1)
                magnitudes = np.linalg.norm(displacements[:-1], axis=1) * np.linalg.norm(displacements[1:], axis=1)
                
                # Avoid division by zero
                valid_indices = magnitudes > 1e-6
                
                if np.any(valid_indices):
                    cos_angles = np.clip(dot_products[valid_indices] / magnitudes[valid_indices], -1.0, 1.0)
                    angles = np.arccos(cos_angles)
                    
                    segment_max_curvature = np.max(angles) if len(angles) > 0 else 0
                    segment_avg_curvature = np.mean(angles) if len(angles) > 0 else 0
                    
                    max_curvature = max(max_curvature, segment_max_curvature)
                    total_curvature += segment_avg_curvature
        
        avg_curvature = total_curvature / len(planner.planned_trajectory_segments) if planner.planned_trajectory_segments else 0
        print(f"Average curvature: {avg_curvature:.4f} rad/m (lower values indicate smoother trajectories)")
        print(f"Maximum curvature: {max_curvature:.4f} rad/m")
        print("\nGenerating visualization...")
        
        # Visualize and save as GIF
        # Pass the spiral waypoints for visualization
        planner.visualize_trajectory(
            start_position, goal_position, 
            output_file="spiral_trajectory.gif",
            waypoints_to_display=spiral_waypoints
        )
        print("Visualization complete! Results saved as 'spiral_trajectory.gif'")
    else:
        print("Trajectory planning failed to complete within maximum iterations.")


if __name__ == "__main__":
    main()