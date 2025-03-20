"""
Energy-Aware Trajectory Generator

Combines the spline trajectory generator with energy prediction models
to generate and evaluate energy consumption of drone trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import joblib
from mpl_toolkits.mplot3d import Axes3D

# Import trajectory generator
from spline_trajectory_generator import SplineTrajectoryGenerator

# Import energy prediction models
from neural_network_models import BiLSTMModel, TCNModel, DroneTrajectoryDataset

class EnergyAwareTrajectory:
    """Energy-Aware Trajectory Generator Class
    
    Combines trajectory generator and energy prediction model to create trajectories
    and evaluate their energy consumption.
    """
    
    def __init__(self, model_path, model_type='bilstm', feature_scaler_path=None, 
                target_scaler_path=None, dt=0.02, sequence_length=20):
        """Initialize energy-aware trajectory generator
        
        Args:
            model_path: Path to energy prediction model
            model_type: Model type ('bilstm' or 'tcn')
            feature_scaler_path: Path to feature scaler
            target_scaler_path: Path to target scaler
            dt: Time step
            sequence_length: Sequence length
        """
        self.dt = dt
        self.sequence_length = sequence_length
        self.gravity = np.array([0, 0, -9.81])
        
        # Load model
        self.model_type = model_type
        self.model = self._load_model(model_path, model_type)
        
        # Load scalers
        self.feature_scaler = None
        self.target_scaler = None
        
        if feature_scaler_path and os.path.exists(feature_scaler_path):
            self.feature_scaler = joblib.load(feature_scaler_path)
        
        if target_scaler_path and os.path.exists(target_scaler_path):
            self.target_scaler = joblib.load(target_scaler_path)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        
        # Feature column names, consistent with those used during model training
        self.feature_columns = [
            'rate_roll', 'rate_pitch', 'rate_yaw', 
            'pos_x', 'pos_y', 'pos_z', 
            'vel_x', 'vel_y', 'vel_z', 
            'R11', 'R12', 'R13', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33'
        ]
    
    def _load_model(self, model_path, model_type):
        """Load energy prediction model
        
        Args:
            model_path: Model path
            model_type: Model type
            
        Returns:
            Loaded model or None if path is invalid
        """
        if model_path is None or not os.path.exists(model_path):
            return None
            
        # Determine input and output dimensions
        input_dim = 18  # rate_*, pos_*, vel_*, R*
        output_dim = 2  # current, voltage
        
        if model_type.lower() == 'bilstm':
            model = BiLSTMModel(
                input_dim=input_dim,
                hidden_dim=64,
                num_layers=2,
                output_dim=output_dim
            )
        elif model_type.lower() == 'tcn':
            model = TCNModel(
                input_dim=input_dim,
                num_channels=[32, 64, 128],
                kernel_size=3,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        return model
    
    def extract_features_from_trajectory(self, trajectory, start_time=0, end_time=None, num_points=100):
        """Extract features from trajectory
        
        Args:
            trajectory: Trajectory generator object
            start_time: Start time
            end_time: End time
            num_points: Number of sample points
            
        Returns:
            features: Feature array [num_points, feature_dim]
            time_points: Array of time points
        """
        if end_time is None:
            end_time = trajectory._tf
        
        # Create time points
        time_points = np.linspace(start_time, end_time, num_points)
        
        # Initialize feature array
        features = np.zeros((num_points, len(self.feature_columns)))
        
        # Extract features
        for i, t in enumerate(time_points):
            # Get position, velocity, and acceleration
            pos = trajectory.get_position(t)
            vel = trajectory.get_velocity(t)
            acc = trajectory.get_acceleration(t)
            
            # Calculate attitude (rotation matrix)
            # Note: We approximate attitude from thrust direction
            thrust_dir = acc - self.gravity
            thrust_dir = thrust_dir / (np.linalg.norm(thrust_dir) + 1e-10)  # prevent division by zero
            
            # Calculate rotation matrix (body to world)
            # Align z-axis with thrust direction
            z_body = thrust_dir
            
            # Choose a temporary vector to determine y-axis direction
            temp = np.array([1, 0, 0])
            if np.abs(np.dot(temp, z_body)) > 0.9:
                temp = np.array([0, 1, 0])
            
            # Calculate body coordinate system x and y axes
            y_body = np.cross(z_body, temp)
            y_body = y_body / (np.linalg.norm(y_body) + 1e-10)
            
            x_body = np.cross(y_body, z_body)
            x_body = x_body / (np.linalg.norm(x_body) + 1e-10)
            
            # Rotation matrix (world to body)
            R = np.column_stack((x_body, y_body, z_body))
            
            # Calculate angular velocity (using trajectory's get_body_rates method)
            body_rates = trajectory.get_body_rates(t)
            
            # Collect all features
            features[i, 0:3] = body_rates  # rate_roll, rate_pitch, rate_yaw
            features[i, 3:6] = pos  # pos_x, pos_y, pos_z
            features[i, 6:9] = vel  # vel_x, vel_y, vel_z
            
            # Rotation matrix elements (R11, R12, R13, R21, R22, R23, R31, R32, R33)
            features[i, 9:18] = R.flatten()
        
        return features, time_points
    
    def predict_energy(self, features, time_points):
        """Predict energy consumption of trajectory
        
        Args:
            features: Feature array [num_points, feature_dim]
            time_points: Time points array
            
        Returns:
            energy: Energy array
            current: Current array
            voltage: Voltage array
            power: Power array
            pred_time_points: Time points corresponding to predictions
        """
        # Ensure enough points to form a sequence
        if len(features) < self.sequence_length:
            raise ValueError(f"Number of feature points ({len(features)}) is less than sequence length ({self.sequence_length})")
        
        # Normalize features
        if self.feature_scaler:
            features = self.feature_scaler.transform(features)
        
        # Prepare sequence data
        X_seq = []
        for i in range(len(features) - self.sequence_length):
            X_seq.append(features[i:i+self.sequence_length])
        
        X_seq = np.array(X_seq)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform predictions
        if self.target_scaler:
            predictions = self.target_scaler.inverse_transform(predictions)
        
        # Separate current and voltage
        current = predictions[:, 0]
        voltage = predictions[:, 1]
        
        # Calculate energy (power * dt)
        power = current * voltage
        energy = power * self.dt
        
        # Cumulative energy
        cumulative_energy = np.cumsum(energy)
        
        # Create corresponding time points (dropping first sequence_length points)
        pred_time_points = time_points[self.sequence_length:]
        
        # Ensure result lengths match
        if len(pred_time_points) > len(cumulative_energy):
            pred_time_points = pred_time_points[:len(cumulative_energy)]
        elif len(pred_time_points) < len(cumulative_energy):
            cumulative_energy = cumulative_energy[:len(pred_time_points)]
            current = current[:len(pred_time_points)]
            voltage = voltage[:len(pred_time_points)]
            power = power[:len(pred_time_points)]
        
        return cumulative_energy, current, voltage, power, pred_time_points
    
    def evaluate_trajectory_energy(self, trajectory, start_time=0, end_time=None, num_points=200):
        """Evaluate trajectory energy consumption
        
        Args:
            trajectory: Trajectory generator object
            start_time: Start time
            end_time: End time
            num_points: Number of sample points
            
        Returns:
            results: Dictionary containing evaluation results
        """
        # Extract features
        features, time_points = self.extract_features_from_trajectory(
            trajectory, start_time, end_time, num_points)
        
        # Predict energy
        energy, current, voltage, power, pred_time_points = self.predict_energy(features, time_points)
        
        # Collect results
        results = {
            'time_points': time_points,
            'features': features,
            'pred_time_points': pred_time_points,
            'energy': energy,
            'cumulative_energy': energy,
            'current': current,
            'voltage': voltage,
            'power': power,
            'total_energy': energy[-1] if len(energy) > 0 else 0,
            'avg_power': np.mean(power) if len(power) > 0 else 0,
            'max_power': np.max(power) if len(power) > 0 else 0
        }
        
        return results
    
    def evaluate_multi_segment_trajectory(self, segments, segment_times):
        """Evaluate energy consumption of multi-segment trajectory
        
        Args:
            segments: List of trajectory segments
            segment_times: Time duration of each segment
            
        Returns:
            results: Dictionary containing evaluation results
        """
        all_results = []
        cumulative_energy = 0
        time_offset = 0
        
        for i, (traj, seg_time) in enumerate(zip(segments, segment_times)):
            # Evaluate current segment
            results = self.evaluate_trajectory_energy(traj, 0, seg_time)
            
            # Adjust time offset
            results['pred_time_points'] += time_offset
            
            # Adjust cumulative energy
            if i > 0:
                results['cumulative_energy'] += cumulative_energy
            
            # Update cumulative energy and time offset
            if len(results['energy']) > 0:
                cumulative_energy = results['cumulative_energy'][-1]
            time_offset += seg_time
            
            all_results.append(results)
        
        # Merge results
        merged_results = {
            'segments': all_results,
            'total_energy': cumulative_energy,
            'segment_times': segment_times,
            'num_segments': len(segments)
        }
        
        return merged_results
    
    def compare_trajectories(self, trajectories, labels=None, durations=None):
        """Compare energy consumption of multiple trajectories
        
        Args:
            trajectories: List of trajectory generator objects
            labels: List of trajectory labels
            durations: List of trajectory durations
            
        Returns:
            results: Dictionary containing comparison results
        """
        if labels is None:
            labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]
        
        # Ensure durations match trajectory count
        if durations is None:
            durations = [traj._tf for traj in trajectories]
        elif len(durations) != len(trajectories):
            raise ValueError("Length of durations list does not match trajectory list length")
        
        # Evaluate each trajectory
        all_results = []
        for traj, duration in zip(trajectories, durations):
            results = self.evaluate_trajectory_energy(traj, 0, duration)
            all_results.append(results)
        
        # Compare results
        comparison = {
            'labels': labels,
            'durations': durations,
            'results': all_results,
            'total_energy': [r['total_energy'] for r in all_results],
            'avg_power': [r['avg_power'] for r in all_results],
            'max_power': [r['max_power'] for r in all_results]
        }
        
        return comparison
    
    def plot_energy_results(self, results, title="Trajectory Energy Analysis"):
        """Plot energy analysis results
        
        Args:
            results: Evaluation results dictionary
            title: Chart title
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # 1. Power curve
        axes[0].plot(results['pred_time_points'], results['power'])
        axes[0].set_ylabel('Power [W]')
        axes[0].set_title(f"{title} - Avg Power: {results['avg_power']:.2f} W, Max Power: {results['max_power']:.2f} W")
        axes[0].grid(True)
        
        # 2. Current and voltage curves
        ax2 = axes[1]
        line1 = ax2.plot(results['pred_time_points'], results['current'], 'b-', label='Current [A]')
        ax2.set_ylabel('Current [A]')
        ax2.grid(True)
        
        ax2_twin = ax2.twinx()
        line2 = ax2_twin.plot(results['pred_time_points'], results['voltage'], 'r-', label='Voltage [V]')
        ax2_twin.set_ylabel('Voltage [V]')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
        
        # 3. Cumulative energy curve
        axes[2].plot(results['pred_time_points'], results['cumulative_energy'])
        axes[2].set_xlabel('Time [s]')
        axes[2].set_ylabel('Cumulative Energy [J]')
        axes[2].set_title(f"Total Energy Consumption: {results['total_energy']:.2f} J")
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trajectory_with_energy(self, trajectory, energy_results, title="Trajectory and Energy Analysis"):
        """Plot trajectory and energy consumption
        
        Args:
            trajectory: Trajectory generator object
            energy_results: Energy evaluation results
            title: Chart title
        """
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Extract trajectory points
        time_points = energy_results['time_points']
        positions = np.array([trajectory.get_position(t) for t in time_points])
        
        # Color mapping based on power
        power_normalized = np.zeros_like(time_points)
        if len(energy_results['pred_time_points']) > 0:
            # Find corresponding power values
            for i, t in enumerate(time_points):
                idx = np.argmin(np.abs(energy_results['pred_time_points'] - t)) if t >= energy_results['pred_time_points'][0] else 0
                if idx < len(energy_results['power']):
                    power_normalized[i] = energy_results['power'][idx]
        
        # Normalize power values for coloring
        if np.max(power_normalized) > np.min(power_normalized):
            power_normalized = (power_normalized - np.min(power_normalized)) / (np.max(power_normalized) - np.min(power_normalized))
        
        # Plot colored trajectory
        for i in range(len(time_points)-1):
            ax1.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2], 
                    color=plt.cm.jet(power_normalized[i]), linewidth=2)
        
        # Add start and end points
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', marker='o', s=100, label='Start')
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', marker='o', s=100, label='End')
        
        # Set axis labels
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.set_title('3D Trajectory (color represents power)')
        
        # Add colorbar
        norm = plt.Normalize(np.min(energy_results['power']), np.max(energy_results['power']))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label('Power [W]')
        
        # Trajectory characteristics plot
        ax2 = fig.add_subplot(222)
        
        # Calculate velocity and acceleration magnitudes
        velocities = np.array([np.linalg.norm(trajectory.get_velocity(t)) for t in time_points])
        accelerations = np.array([np.linalg.norm(trajectory.get_acceleration(t)) for t in time_points])
        
        # Plot velocity and acceleration
        ax2.plot(time_points, velocities, 'b-', label='Velocity')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Velocity [m/s]')
        ax2.grid(True)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(time_points, accelerations, 'r-', label='Acceleration')
        ax2_twin.set_ylabel('Acceleration [m/s²]')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2.set_title('Trajectory Characteristics')
        
        # Power and cumulative energy plots
        ax3 = fig.add_subplot(223)
        ax3.plot(energy_results['pred_time_points'], energy_results['power'], 'g-')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Power [W]')
        ax3.grid(True)
        ax3.set_title(f"Power Curve - Avg: {energy_results['avg_power']:.2f} W, Max: {energy_results['max_power']:.2f} W")
        
        ax4 = fig.add_subplot(224)
        ax4.plot(energy_results['pred_time_points'], energy_results['cumulative_energy'], 'b-')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Cumulative Energy [J]')
        ax4.grid(True)
        ax4.set_title(f"Energy Consumption - Total: {energy_results['total_energy']:.2f} J")
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        plt.show()
    
    def plot_trajectory_comparison(self, comparison_results, title="Trajectory Energy Comparison"):
        """Plot energy comparison of multiple trajectories
        
        Args:
            comparison_results: Comparison results dictionary
            title: Chart title
        """
        labels = comparison_results['labels']
        results = comparison_results['results']
        n_trajectories = len(results)
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Cumulative energy comparison
        ax1 = fig.add_subplot(221)
        for i, res in enumerate(results):
            ax1.plot(res['pred_time_points'], res['cumulative_energy'], label=labels[i])
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Cumulative Energy [J]')
        ax1.set_title('Cumulative Energy Comparison')
        ax1.grid(True)
        ax1.legend()
        
        # 2. Power comparison
        ax2 = fig.add_subplot(222)
        for i, res in enumerate(results):
            ax2.plot(res['pred_time_points'], res['power'], label=labels[i])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Power [W]')
        ax2.set_title('Power Curve Comparison')
        ax2.grid(True)
        ax2.legend()
        
        # 3. Total energy bar chart
        ax3 = fig.add_subplot(223)
        x = np.arange(n_trajectories)
        total_energy = [res['total_energy'] for res in results]
        
        bars = ax3.bar(x, total_energy)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.set_ylabel('Total Energy [J]')
        ax3.set_title('Total Energy Consumption Comparison')
        
        # Add value labels
        for bar, energy in zip(bars, total_energy):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{energy:.2f}',
                    ha='center', va='bottom', rotation=0)
        
        # 4. Average and maximum power comparison
        ax4 = fig.add_subplot(224)
        avg_power = [res['avg_power'] for res in results]
        max_power = [res['max_power'] for res in results]
        
        x = np.arange(n_trajectories)
        width = 0.35
        
        ax4.bar(x - width/2, avg_power, width, label='Average Power')
        ax4.bar(x + width/2, max_power, width, label='Maximum Power')
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_ylabel('Power [W]')
        ax4.set_title('Power Comparison')
        ax4.legend()
        
        # Add 3D trajectory comparison plot
        if n_trajectories > 1:
            fig_3d = plt.figure(figsize=(12, 10))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            
            for i, res in enumerate(results):
                # Extract positions from features
                positions = res['features'][:, 3:6]  # pos_x, pos_y, pos_z
                ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=labels[i])
                
                # Add start and end points
                ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], marker='o', s=100)
                ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], marker='x', s=100)
            
            ax_3d.set_xlabel('X [m]')
            ax_3d.set_ylabel('Y [m]')
            ax_3d.set_zlabel('Z [m]')
            ax_3d.set_title('3D Trajectory Comparison')
            ax_3d.legend()
            
            plt.tight_layout()
            plt.suptitle(f"{title} - 3D Trajectories", fontsize=16)
            plt.subplots_adjust(top=0.9)
        
        plt.figure(fig.number)  # Switch back to main figure
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        plt.show()