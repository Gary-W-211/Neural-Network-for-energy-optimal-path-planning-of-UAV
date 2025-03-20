import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import copy
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from spline_trajectory_generator import SplineTrajectoryGenerator
from energy_aware_trajectory import EnergyAwareTrajectory
from sim_data_generator import add_simulation_support_to_energy_trajectory

class LimitedVisionEnergyPlanner:
    """
    基于样条曲线的能量优化路径规划系统，模拟无人机在有限视野下的路径规划。
    无人机只能"看到"未来0.5秒的环境，需要在有限信息下做出能量优化决策。
    使用样条曲线(spline)生成平滑的轨迹，确保无人机运动的连续性。
    """
    
    def __init__(self, energy_model_path=None, vision_horizon=0.1):
        """
        初始化规划器
        
        参数:
            energy_model_path: 能量预测模型的路径
            vision_horizon: 无人机的视野范围(秒)，默认为0.5秒
        """
        self.vision_horizon = vision_horizon
        
        # 初始化重力方向
        self.gravity = [0, 0, -9.81]
        
        # 初始化能量预测模型
        self.setup_energy_model(energy_model_path)
        
        # 存储完整规划路径
        self.planned_trajectory_segments = []
        self.planned_segment_times = []
        self.waypoints = []
        
        # 存储能量消耗数据
        self.energy_data = []
        self.cumulative_energy = 0
        
        # 轨迹探索参数
        self.num_candidates = 20  # 每次规划考虑的候选轨迹数量
        
    def setup_energy_model(self, model_path):
        """设置能量预测模型"""
        # 检查模型文件是否存在
        use_simulation = False
        if model_path is None or not os.path.exists(model_path):
            print("Warning: Model file does not exist! Using simulated data instead.")
            model_path = None
            feature_scaler_path = None
            target_scaler_path = None
            use_simulation = True
        else:
            # 假设缩放器与模型在同一目录
            model_dir = os.path.dirname(model_path)
            feature_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
            target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")
        
        # 创建能量预测器
        self.energy_predictor = EnergyAwareTrajectory(
            model_path=model_path,
            model_type='bilstm',  # 或 'tcn'
            feature_scaler_path=feature_scaler_path,
            target_scaler_path=target_scaler_path
        )
        
        # 如果需要，添加模拟支持
        if use_simulation:
            self.energy_predictor = add_simulation_support_to_energy_trajectory(self.energy_predictor)
    
    def generate_candidate_trajectories(self, current_state, goal_position):
        """
        生成候选轨迹，重点使用样条曲线(spline)方法
        
        参数:
            current_state: 当前状态 [位置, 速度, 加速度]
            goal_position: 最终目标位置
        
        返回:
            候选轨迹列表及其参数
        """
        current_pos, current_vel, current_acc = current_state
        
        # 计算当前位置到目标的方向向量
        direction_to_goal = np.array(goal_position) - np.array(current_pos)
        distance_to_goal = np.linalg.norm(direction_to_goal)
        
        # 如果已经非常接近目标，直接返回到目标的轨迹
        if distance_to_goal < 0.5:
            # 创建直接到目标的样条曲线轨迹，使用更高阶样条保证平滑性
            traj = SplineTrajectoryGenerator(current_pos, current_vel, current_acc, self.gravity)
            traj.set_goal_position(goal_position)
            traj.set_goal_velocity([0, 0, 0])
            traj.set_goal_acceleration([0, 0, 0])
            # 增加控制点数量以获得更平滑的曲线
            traj.generate(self.vision_horizon, num_points=15)
            return [traj], [self.vision_horizon], [goal_position]
        
        # 规格化方向向量
        if distance_to_goal > 0:
            normalized_direction = direction_to_goal / distance_to_goal
        else:
            normalized_direction = np.array([1, 0, 0])  # 默认前进方向
        
        # 生成一系列可能的中间目标点
        candidate_trajectories = []
        candidate_durations = []
        candidate_waypoints = []
        
        # 定义扰动幅度 - 随着接近目标减小扰动
        deviation_scale = min(2.0, distance_to_goal / 3)
        
        # 生成候选路径
        for i in range(self.num_candidates):
            # 计算向目标方向前进的距离（在视野范围内）
            forward_distance = min(self.vision_horizon * 2, distance_to_goal * 0.3)
            
            # 为不同候选轨迹设置不同的控制点数量，创造多样性
            # 越高的控制点数量，轨迹越平滑但计算量更大
            control_points = 10 + i % 6  # 从10到15变化的控制点数量
            
            # 第一个候选始终是直接朝向目标的路径
            if i == 0:
                deviation = np.array([0, 0, 0])
                # 第一条路径使用更多控制点确保平滑
                control_points = 15
            else:
                # 其他候选在垂直于前进方向的平面上有偏移
                # 创建垂直于前进方向的两个基向量
                if abs(normalized_direction[0]) < 0.9:
                    perpendicular1 = np.cross(normalized_direction, [1, 0, 0])
                else:
                    perpendicular1 = np.cross(normalized_direction, [0, 1, 0])
                perpendicular1 = perpendicular1 / np.linalg.norm(perpendicular1)
                perpendicular2 = np.cross(normalized_direction, perpendicular1)
                perpendicular2 = perpendicular2 / np.linalg.norm(perpendicular2)
                
                # 使用三角函数生成均匀分布的偏移
                angle = 2 * np.pi * (i-1) / (self.num_candidates-1)
                deviation = (perpendicular1 * np.cos(angle) + perpendicular2 * np.sin(angle)) * deviation_scale
            
            # 计算中间控制点 - 为了让样条曲线更平滑，我们添加额外的控制点
            intermediate_waypoint = np.array(current_pos) + normalized_direction * forward_distance + deviation
            
            # 创建样条轨迹
            traj = SplineTrajectoryGenerator(current_pos, current_vel, current_acc, self.gravity)
            traj.set_goal_position(intermediate_waypoint.tolist())
            
            # 设置目标速度 - 保持朝着最终目标的方向
            target_speed = min(3.0, distance_to_goal / 2)  # 速度限制
            goal_vel = normalized_direction * target_speed
            traj.set_goal_velocity(goal_vel.tolist())
            
            # 设置目标加速度 - 为了获得更平滑的轨迹
            # 针对不同的候选路径设置不同的加速度
            if i % 3 == 0:
                # 减速型轨迹
                goal_acc = -normalized_direction * 0.3
                # 确保我们传递的是列表而不是numpy数组
                goal_acc = goal_acc.tolist()
            elif i % 3 == 1:
                # 匀速型轨迹
                goal_acc = [0, 0, 0]
            else:
                # 加速型轨迹
                goal_acc = normalized_direction * 0.3
                # 确保我们传递的是列表而不是numpy数组
                goal_acc = goal_acc.tolist()

            traj.set_goal_acceleration(goal_acc)# 设置目标加速度 - 为了获得更平滑的轨迹

            # 生成样条轨迹，指定控制点数量
            traj.generate(self.vision_horizon, num_points=control_points)
            
            candidate_trajectories.append(traj)
            candidate_durations.append(self.vision_horizon)
            candidate_waypoints.append(intermediate_waypoint.tolist())
        
        return candidate_trajectories, candidate_durations, candidate_waypoints
    
    def evaluate_candidates(self, trajectories, durations, waypoints, goal_position):
        """
        评估候选轨迹并选择最佳轨迹，考虑样条曲线的平滑度
        
        参数:
            trajectories: 候选轨迹列表
            durations: 轨迹持续时间列表
            waypoints: 轨迹终点位置列表
            goal_position: 最终目标位置
        
        返回:
            最佳轨迹的索引
        """
        best_score = float('inf')
        best_idx = 0
        
        # 评估每个候选轨迹
        for i, (traj, duration) in enumerate(zip(trajectories, durations)):
            # 计算能量消耗
            energy_results = self.energy_predictor.evaluate_trajectory_energy(traj)
            energy_consumption = energy_results['total_energy']
            
            # 计算终点到目标的距离
            end_waypoint = waypoints[i]
            distance_to_goal = np.linalg.norm(np.array(end_waypoint) - np.array(goal_position))
            
            # 计算样条曲线平滑度 (通过评估加速度变化)
            smoothness_score = 0
            num_samples = 10
            t_samples = np.linspace(0, duration, num_samples)
            
            # 获取加速度样本点
            acc_samples = np.array([np.linalg.norm(traj.get_acceleration(t)) for t in t_samples])
            
            # 计算加速度变化率 (jerk) 的近似值
            if len(acc_samples) > 1:
                acc_changes = np.abs(acc_samples[1:] - acc_samples[:-1])
                max_acc_change = np.max(acc_changes) if len(acc_changes) > 0 else 0
                avg_acc_change = np.mean(acc_changes) if len(acc_changes) > 0 else 0
                
                # 平滑度得分 (加速度变化越小越好)
                smoothness_score = avg_acc_change + 0.5 * max_acc_change
            
            # 构建综合评分 (能量消耗 + 距离权重 + 平滑度)
            # 随着与目标距离的减小，逐渐增加距离权重
            distance_to_goal_normalized = min(1.0, distance_to_goal / 10.0)
            energy_weight = 1.0
            distance_weight = 3.0 * (1.0 - distance_to_goal_normalized)
            smoothness_weight = 0.5  # 平滑度权重
            
            score = (energy_weight * energy_consumption + 
                    distance_weight * distance_to_goal + 
                    smoothness_weight * smoothness_score)
            
            # 打印详细评估信息 (可以在需要调试时取消注释)
            # print(f"轨迹 {i}: 能量={energy_consumption:.2f}J, 距离={distance_to_goal:.2f}m, "
            #       f"平滑度={smoothness_score:.2f}, 总分={score:.2f}")
            
            if score < best_score:
                best_score = score
                best_idx = i
        
        return best_idx
    
    def plan_trajectory(self, start_state, goal_position, max_iterations=100):
        """
        规划从起点到终点的完整轨迹
        
        参数:
            start_state: 起始状态 [位置, 速度, 加速度]
            goal_position: 目标位置
            max_iterations: 最大迭代次数
        
        返回:
            规划状态 (True表示成功)
        """
        # 初始化规划
        self.planned_trajectory_segments = []
        self.planned_segment_times = []
        self.waypoints = []
        self.energy_data = []
        self.cumulative_energy = 0
        
        # 记录起始点
        start_pos, start_vel, start_acc = start_state
        self.waypoints.append(start_pos)
        
        # 当前状态
        current_state = copy.deepcopy(start_state)
        
        # 规划迭代
        for iteration in range(max_iterations):
            print(f"规划迭代 {iteration+1}/{max_iterations}")
            current_pos, current_vel, current_acc = current_state
            
            # 检查是否已到达目标
            distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(goal_position))
            if distance_to_goal < 0.2:  # 认为已达到目标的阈值
                print(f"已达到目标! 总迭代次数: {iteration+1}")
                self.waypoints.append(goal_position)
                return True
            
            # 生成候选轨迹
            candidates, durations, candidate_waypoints = self.generate_candidate_trajectories(
                current_state, goal_position
            )
            
            # 可视化候选轨迹 (如需调试)
            # self.visualize_candidates(candidates, durations, current_pos, goal_position)
            
            # 评估轨迹并选择最佳的
            best_idx = self.evaluate_candidates(candidates, durations, candidate_waypoints, goal_position)
            best_trajectory = candidates[best_idx]
            best_duration = durations[best_idx]
            best_waypoint = candidate_waypoints[best_idx]
            
            # 评估选中轨迹的能量消耗
            energy_results = self.energy_predictor.evaluate_trajectory_energy(best_trajectory)
            segment_energy = energy_results['total_energy']
            self.cumulative_energy += segment_energy
            self.energy_data.append(
                (segment_energy, self.cumulative_energy, best_waypoint)
            )
            
            # 将轨迹段添加到计划中
            self.planned_trajectory_segments.append(best_trajectory)
            self.planned_segment_times.append(best_duration)
            self.waypoints.append(best_waypoint)
            
            # 更新当前状态以进行下一次迭代
            # 这里我们使用选中轨迹在time=horizon时的状态作为下一个起点
            next_pos = best_trajectory.get_position(best_duration)
            next_vel = best_trajectory.get_velocity(best_duration)
            next_acc = best_trajectory.get_acceleration(best_duration)
            
            current_state = [next_pos, next_vel, next_acc]
            
            print(f"  当前位置: {next_pos}, 距离目标: {distance_to_goal:.2f}m")
            print(f"  段能量消耗: {segment_energy:.2f}J, 累计能量: {self.cumulative_energy:.2f}J")
        
        print("达到最大迭代次数!")
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

def main():
    # Set energy model paths
    model_path = "results/bilstm/bilstm_model.pth"
    feature_scaler_path = "processed_data/feature_scaler.pkl"
    target_scaler_path = "processed_data/target_scaler.pkl"
    
    # Create planner
    planner = LimitedVisionEnergyPlanner(
        energy_model_path=model_path,
        vision_horizon=0.5  # Drone can only "see" 0.5 seconds ahead
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
        num_points=15          # Number of waypoints to generate
    )
    
    # Set final goal position (the last waypoint of the spiral)
    goal_position = spiral_waypoints[-1]
    
    print("=== Spline-Based Energy-Aware Path Planning System with Spiral Trajectory ===")
    print("The drone will generate a smooth spiral ascending trajectory using splines,")
    print("making energy-optimized decisions with limited vision (0.5s horizon).")
    print(f"Start: {start_position}, Goal: {goal_position}")
    print(f"Total waypoints to pass through: {len(spiral_waypoints)}")
    print("Starting planning...")
    
    # Perform multi-waypoint trajectory planning
    planning_successful = plan_through_waypoints(
        planner, start_state, spiral_waypoints, max_iterations_per_segment=10
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

def generate_spiral_waypoints(center=[5, 0, 0], radius_start=3, radius_end=1, 
                             height_start=0, height_end=5, num_turns=2.5, num_points=15):
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
        
    Returns:
        List of waypoint coordinates
    """
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

def plan_through_waypoints(planner, start_state, waypoints, max_iterations_per_segment=10):
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
    
    # Planning iterations for this segment
    for iteration in range(max_iterations):
        print(f"  {segment_name} planning iteration {iteration+1}/{max_iterations}")
        
        # Check if we've reached the goal
        distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(goal_position))
        if distance_to_goal < 0.2:  # Threshold for considering goal reached
            print(f"  Reached {segment_name}! Total iterations: {iteration+1}")
            return True
        
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
    
    print(f"  Reached maximum iterations for {segment_name}!")
    return False

if __name__ == "__main__":
    main()
