import os
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入自定义模块
from neural_network_models import calculate_energy, DroneTrajectoryDataset

def evaluate_trajectory_energy(model, X_test, y_test, test_trajectories, target_scaler, dt=0.02, 
                              device=None, output_dir='./evaluation_results'):
    """
    详细评估模型在每个轨迹上的能量预测性能
    
    Args:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集目标
        test_trajectories: 测试集轨迹索引
        target_scaler: 目标归一化器
        dt: 时间步长
        device: 计算设备
        output_dir: 输出目录
        
    Returns:
        trajectory_metrics: 每个轨迹的评估指标
    """
    # 检查设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 将模型设置为评估模式
    model.eval()
    model.to(device)
    
    # 获取唯一的轨迹索引
    unique_trajectories = np.unique(test_trajectories)
    num_trajectories = len(unique_trajectories)
    
    print(f"评估模型在 {num_trajectories} 个轨迹上的性能...")
    
    # 初始化结果字典
    trajectory_metrics = {}
    all_energy_pred = []
    all_energy_true = []
    
    # 对每个轨迹进行评估
    for traj_idx in tqdm(unique_trajectories):
        # 获取当前轨迹的掩码
        mask = test_trajectories == traj_idx
        
        # 提取当前轨迹的数据
        X_traj = X_test[mask]
        y_traj = y_test[mask]
        
        # 创建数据集和数据加载器
        traj_dataset = DroneTrajectoryDataset(X_traj, y_traj)
        traj_loader = DataLoader(traj_dataset, batch_size=64)
        
        # 进行预测
        current_pred = []
        voltage_pred = []
        current_true = []
        voltage_true = []
        
        with torch.no_grad():
            for inputs, targets in traj_loader:
                inputs = inputs.to(device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 收集预测和真实值
                current_pred.append(outputs[:, 0].cpu().numpy())
                voltage_pred.append(outputs[:, 1].cpu().numpy())
                current_true.append(targets[:, 0].numpy())
                voltage_true.append(targets[:, 1].numpy())
        
        # 连接结果
        current_pred = np.concatenate(current_pred)
        voltage_pred = np.concatenate(voltage_pred)
        current_true = np.concatenate(current_true)
        voltage_true = np.concatenate(voltage_true)
        
        # 反标准化预测结果
        if target_scaler:
            # 重塑数组以匹配目标归一化器的形状
            current_pred_2d = current_pred.reshape(-1, 1)
            voltage_pred_2d = voltage_pred.reshape(-1, 1)
            current_true_2d = current_true.reshape(-1, 1)
            voltage_true_2d = voltage_true.reshape(-1, 1)
            
            # 合并电流和电压为一个数组
            pred_combined = np.hstack((current_pred_2d, voltage_pred_2d))
            true_combined = np.hstack((current_true_2d, voltage_true_2d))
            
            # 反标准化
            pred_combined = target_scaler.inverse_transform(pred_combined)
            true_combined = target_scaler.inverse_transform(true_combined)
            
            # 分离电流和电压
            current_pred = pred_combined[:, 0]
            voltage_pred = pred_combined[:, 1]
            current_true = true_combined[:, 0]
            voltage_true = true_combined[:, 1]
        
        # 计算能量
        energy_pred = calculate_energy(current_pred, voltage_pred, dt)
        energy_true = calculate_energy(current_true, voltage_true, dt)
        
        # 累积所有轨迹的能量预测和真实值
        all_energy_pred.extend(energy_pred)
        all_energy_true.extend(energy_true)
        
        # 计算评估指标
        mse = mean_squared_error(energy_true, energy_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(energy_true, energy_pred)
        r2 = r2_score(energy_true, energy_pred) if len(energy_true) > 1 else 0
        
        trajectory_metrics[traj_idx] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'energy_pred': energy_pred,
            'energy_true': energy_true,
            'num_samples': len(energy_true)
        }
        
        # 绘制当前轨迹的能量预测图
        plt.figure(figsize=(12, 6))
        plt.plot(energy_true, label='True Energy')
        plt.plot(energy_pred, label='Predicted Energy')
        plt.title(f'Trajectory {traj_idx} - Energy Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Energy (J)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'trajectory_{traj_idx}_energy.png'))
        plt.close()
    
    # 计算所有轨迹的总体指标
    overall_mse = mean_squared_error(all_energy_true, all_energy_pred)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = mean_absolute_error(all_energy_true, all_energy_pred)
    overall_r2 = r2_score(all_energy_true, all_energy_pred)
    
    overall_metrics = {
        'mse': overall_mse,
        'rmse': overall_rmse,
        'mae': overall_mae,
        'r2': overall_r2
    }
    
    print("\n总体评估指标:")
    print(f"MSE: {overall_mse:.6f}")
    print(f"RMSE: {overall_rmse:.6f}")
    print(f"MAE: {overall_mae:.6f}")
    print(f"R²: {overall_r2:.6f}")
    
    # 创建轨迹RMSE对比图
    plt.figure(figsize=(14, 6))
    traj_indices = list(trajectory_metrics.keys())
    rmse_values = [trajectory_metrics[idx]['rmse'] for idx in traj_indices]
    
    # 按RMSE值排序
    sorted_indices = np.argsort(rmse_values)
    sorted_traj_indices = [traj_indices[i] for i in sorted_indices]
    sorted_rmse_values = [rmse_values[i] for i in sorted_indices]
    
    plt.bar(range(len(sorted_traj_indices)), sorted_rmse_values)
    plt.axhline(y=overall_rmse, color='r', linestyle='-', label=f'Overall RMSE: {overall_rmse:.4f}')
    plt.xlabel('Trajectory Index')
    plt.ylabel('RMSE')
    plt.title('RMSE per Trajectory')
    plt.xticks(range(len(sorted_traj_indices)), sorted_traj_indices, rotation=90)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_rmse_comparison.png'))
    plt.close()
    
    # 保存评估结果
    evaluation_results = {
        'trajectory_metrics': trajectory_metrics,
        'overall_metrics': overall_metrics
    }
    
    joblib.dump(evaluation_results, os.path.join(output_dir, 'trajectory_evaluation_results.pkl'))
    
    return trajectory_metrics, overall_metrics


def analyze_trajectory_characteristics(X_test, y_test, test_trajectories, trajectory_metrics, 
                                      feature_columns, output_dir='./evaluation_results'):
    """
    分析轨迹特征与预测性能的关系
    
    Args:
        X_test: 测试集特征
        y_test: 测试集目标
        test_trajectories: 测试集轨迹索引
        trajectory_metrics: 每个轨迹的评估指标
        feature_columns: 特征列名
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("分析轨迹特征与预测性能的关系...")
    
    # 获取唯一的轨迹索引
    unique_trajectories = np.unique(test_trajectories)
    
    # 初始化轨迹特征字典
    trajectory_features = {}
    
    # 提取每个轨迹的特征统计量
    for traj_idx in unique_trajectories:
        # 获取当前轨迹的掩码
        mask = test_trajectories == traj_idx
        
        # 提取当前轨迹的数据
        X_traj = X_test[mask]
        
        # 计算每个特征的统计量
        feature_stats = {
            'mean': np.mean(X_traj, axis=(0, 1)),
            'std': np.std(X_traj, axis=(0, 1)),
            'min': np.min(X_traj, axis=(0, 1)),
            'max': np.max(X_traj, axis=(0, 1)),
            'range': np.max(X_traj, axis=(0, 1)) - np.min(X_traj, axis=(0, 1)),
        }
        
        # 计算轨迹的复杂度特征
        complexity_features = {
            'length': X_traj.shape[0],
            'velocity_changes': np.mean(np.abs(np.diff(X_traj[:, :, 3:6], axis=0))),  # 速度变化
            'position_changes': np.mean(np.abs(np.diff(X_traj[:, :, 0:3], axis=0))),  # 位置变化
        }
        
        # 结合所有特征
        all_features = {**feature_stats, **complexity_features}
        
        # 存储轨迹特征
        trajectory_features[traj_idx] = all_features
    
    # 分析轨迹特征与RMSE的关系
    feature_correlations = {}
    
    # 提取所有轨迹的RMSE和特征
    traj_rmse = np.array([trajectory_metrics[idx]['rmse'] for idx in unique_trajectories])
    
    # 对每个特征统计量进行分析
    for stat in ['mean', 'std', 'range']:
        # 对每个特征维度进行分析
        for i, feature_name in enumerate(feature_columns):
            # 提取所有轨迹的当前特征统计量
            feature_values = np.array([trajectory_features[idx][stat][i] for idx in unique_trajectories])
            
            # 计算与RMSE的相关性
            corr = np.corrcoef(feature_values, traj_rmse)[0, 1]
            
            feature_correlations[f'{feature_name}_{stat}'] = corr
    
    # 对复杂度特征进行分析
    for feature_name in ['length', 'velocity_changes', 'position_changes']:
        # 提取所有轨迹的当前特征
        feature_values = np.array([trajectory_features[idx][feature_name] for idx in unique_trajectories])
        
        # 计算与RMSE的相关性
        corr = np.corrcoef(feature_values, traj_rmse)[0, 1]
        
        feature_correlations[feature_name] = corr
    
    # 可视化相关性
    plt.figure(figsize=(14, 10))
    
    # 排序相关性
    sorted_correlations = sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    feature_names = [item[0] for item in sorted_correlations]
    correlation_values = [item[1] for item in sorted_correlations]
    
    # 显示前20个最强相关性
    top_n = min(20, len(feature_names))
    feature_names = feature_names[:top_n]
    correlation_values = correlation_values[:top_n]
    
    # 创建颜色映射
    colors = ['red' if c < 0 else 'green' for c in correlation_values]
    
    plt.barh(range(len(feature_names)), [abs(c) for c in correlation_values], color=colors)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Absolute Correlation with RMSE')
    plt.title('Feature Importance: Correlation with Prediction Error')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Positive Correlation'),
        Patch(facecolor='red', label='Negative Correlation')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation_with_rmse.png'))
    plt.close()
    
    # 可视化轨迹复杂度与RMSE的关系
    plt.figure(figsize=(10, 8))
    plt.scatter([trajectory_features[idx]['velocity_changes'] for idx in unique_trajectories],
                [trajectory_metrics[idx]['rmse'] for idx in unique_trajectories])
    plt.xlabel('Velocity Changes')
    plt.ylabel('RMSE')
    plt.title('Trajectory Complexity vs. Prediction Error')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'complexity_vs_rmse.png'))
    plt.close()
    
    # 将分析结果保存为文件
    analysis_results = {
        'trajectory_features': trajectory_features,
        'feature_correlations': feature_correlations
    }
    
    joblib.dump(analysis_results, os.path.join(output_dir, 'trajectory_feature_analysis.pkl'))
    
    print("轨迹特征分析完成")
    
    return feature_correlations


def compare_models_on_trajectories(model_results, trajectory_ids, output_dir='./evaluation_results'):
    """
    对比不同模型在各个轨迹上的性能
    
    Args:
        model_results: 模型结果字典，格式为 {model_name: {trajectory_id: metrics}}
        trajectory_ids: 要比较的轨迹ID列表
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取模型名称
    model_names = list(model_results.keys())
    
    print(f"比较 {len(model_names)} 个模型在 {len(trajectory_ids)} 个轨迹上的性能...")
    
    # 创建轨迹RMSE对比图
    plt.figure(figsize=(14, 8))
    
    for i, model_name in enumerate(model_names):
        # 提取每个轨迹的RMSE
        rmse_values = [model_results[model_name]['trajectory_metrics'][traj_id]['rmse'] 
                      for traj_id in trajectory_ids]
        
        # 绘制柱状图
        x = np.arange(len(trajectory_ids))
        width = 0.8 / len(model_names)
        plt.bar(x + i * width - 0.4 + width/2, rmse_values, width, label=model_name)
    
    plt.xlabel('Trajectory ID')
    plt.ylabel('RMSE')
    plt.title('Model Performance Comparison per Trajectory')
    plt.xticks(np.arange(len(trajectory_ids)), trajectory_ids, rotation=90)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_per_trajectory.png'))
    plt.close()
    
    # 计算每个模型的平均性能
    model_avg_metrics = {}
    
    for model_name in model_names:
        model_avg_metrics[model_name] = {
            'avg_rmse': np.mean([model_results[model_name]['trajectory_metrics'][traj_id]['rmse'] 
                               for traj_id in trajectory_ids]),
            'avg_mae': np.mean([model_results[model_name]['trajectory_metrics'][traj_id]['mae'] 
                              for traj_id in trajectory_ids]),
            'avg_r2': np.mean([model_results[model_name]['trajectory_metrics'][traj_id]['r2'] 
                             for traj_id in trajectory_ids]),
        }
    
    # 创建模型平均性能比较图
    plt.figure(figsize=(10, 6))
    
    metrics = ['avg_rmse', 'avg_mae', 'avg_r2']
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(model_names):
        values = [model_avg_metrics[model_name][metric] for metric in metrics]
        plt.bar(x + i * width - 0.4 + width/2, values, width, label=model_name)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Average Model Performance')
    plt.xticks(x, ['RMSE', 'MAE', 'R²'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'model_average_performance.png'))
    plt.close()
    
    # 统计每个模型在哪些轨迹上表现最好
    best_model_count = {model_name: 0 for model_name in model_names}
    
    for traj_id in trajectory_ids:
        # 比较每个模型在当前轨迹上的RMSE
        rmse_values = {model_name: model_results[model_name]['trajectory_metrics'][traj_id]['rmse'] 
                      for model_name in model_names}
        
        # 找出RMSE最小的模型
        best_model = min(rmse_values, key=rmse_values.get)
        best_model_count[best_model] += 1
    
    # 创建最佳模型计数图
    plt.figure(figsize=(8, 6))
    plt.bar(best_model_count.keys(), best_model_count.values())
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.title('Number of Trajectories Where Each Model Performs Best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'best_model_count.png'))
    plt.close()
    
    # 保存比较结果
    comparison_results = {
        'model_avg_metrics': model_avg_metrics,
        'best_model_count': best_model_count
    }
    
    joblib.dump(comparison_results, os.path.join(output_dir, 'model_comparison_results.pkl'))
    
    print("模型比较完成")
    
    return model_avg_metrics, best_model_count


def evaluate_model_with_real_data(model, data_file, sequence_length, feature_columns, target_columns,
                                 feature_scaler, target_scaler, dt=0.02, device=None, 
                                 output_dir='./evaluation_results'):
    """
    使用真实数据评估模型性能
    
    Args:
        model: 训练好的模型
        data_file: 数据文件路径
        sequence_length: 序列长度
        feature_columns: 特征列名
        target_columns: 目标列名
        feature_scaler: 特征归一化器
        target_scaler: 目标归一化器
        dt: 时间步长
        device: 计算设备
        output_dir: 输出目录
        
    Returns:
        metrics: 评估指标
    """
    # 检查设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"使用文件 {data_file} 评估模型性能...")
    
    # 读取数据
    import pandas as pd
    df = pd.read_csv(data_file)
    
    # 检查必要的列是否存在
    missing_features = [col for col in feature_columns if col not in df.columns]
    missing_targets = [col for col in target_columns if col not in df.columns]
    
    if missing_features:
        raise ValueError(f"数据文件缺少以下特征列: {missing_features}")
    if missing_targets:
        raise ValueError(f"数据文件缺少以下目标列: {missing_targets}")
    
    # 提取特征和目标
    X = df[feature_columns].values
    y = df[target_columns].values
    
    # 标准化特征和目标
    if feature_scaler:
        X = feature_scaler.transform(X)
    if target_scaler:
        y = target_scaler.transform(y)
    
    # 准备序列数据
    X_seq = []
    y_next = []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_next.append(y[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_next = np.array(y_next)
    
    # 创建数据集和数据加载器
    dataset = DroneTrajectoryDataset(X_seq, y_next)
    loader = DataLoader(dataset, batch_size=64)
    
    # 模型评估
    model.eval()
    model.to(device)
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in tqdm(loader, desc='Evaluating'):
            inputs = inputs.to(device)
            
            # 前向传播
            output = model(inputs)
            
            # 收集预测和真实值
            predictions.append(output.cpu().numpy())
            targets.append(target.numpy())
    
    # 连接结果
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # 分离电流和电压预测
    current_pred = predictions[:, 0]
    voltage_pred = predictions[:, 1]
    current_true = targets[:, 0]
    voltage_true = targets[:, 1]
    
    # 反标准化
    if target_scaler:
        # 重塑数组
        current_pred_2d = current_pred.reshape(-1, 1)
        voltage_pred_2d = voltage_pred.reshape(-1, 1)
        current_true_2d = current_true.reshape(-1, 1)
        voltage_true_2d = voltage_true.reshape(-1, 1)
        
        # 合并电流和电压
        pred_combined = np.hstack((current_pred_2d, voltage_pred_2d))
        true_combined = np.hstack((current_true_2d, voltage_true_2d))
        
        # 反标准化
        pred_combined = target_scaler.inverse_transform(pred_combined)
        true_combined = target_scaler.inverse_transform(true_combined)
        
        # 分离电流和电压
        current_pred = pred_combined[:, 0]
        voltage_pred = pred_combined[:, 1]
        current_true = true_combined[:, 0]
        voltage_true = true_combined[:, 1]
    
    # 计算能量
    energy_pred = calculate_energy(current_pred, voltage_pred, dt)
    energy_true = calculate_energy(current_true, voltage_true, dt)
    
    # 计算评估指标
    mse = mean_squared_error(energy_true, energy_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(energy_true, energy_pred)
    r2 = r2_score(energy_true, energy_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print("\n评估指标:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    
    # 绘制预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(energy_true, label='True Energy')
    plt.plot(energy_pred, label='Predicted Energy')
    plt.title('Energy Prediction on Real Data')
    plt.xlabel('Time Step')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'real_data_energy_prediction.png'))
    plt.close()
    
    # 绘制预测vs真实值散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(energy_true, energy_pred, alpha=0.5)
    plt.plot([min(energy_true), max(energy_true)], [min(energy_true), max(energy_true)], 'r--')
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    plt.title('True vs Predicted Energy on Real Data')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'real_data_energy_scatter.png'))
    plt.close()
    
    # 保存评估结果
    evaluation_results = {
        'metrics': metrics,
        'energy_pred': energy_pred,
        'energy_true': energy_true,
        'current_pred': current_pred,
        'current_true': current_true,
        'voltage_pred': voltage_pred,
        'voltage_true': voltage_true
    }
    
    joblib.dump(evaluation_results, os.path.join(output_dir, 'real_data_evaluation_results.pkl'))
    
    print(f"使用真实数据评估完成，结果已保存到 {output_dir}")
    
    return metrics


def calculate_total_trajectory_energy(model, trajectory_data, sequence_length, feature_columns, 
                                     target_columns, feature_scaler, target_scaler, dt=0.02, 
                                     device=None):
    """
    计算整个轨迹的总能量消耗
    
    Args:
        model: 训练好的模型
        trajectory_data: 轨迹数据DataFrame
        sequence_length: 序列长度
        feature_columns: 特征列名
        target_columns: 目标列名
        feature_scaler: 特征归一化器
        target_scaler: 目标归一化器
        dt: 时间步长
        device: 计算设备
        
    Returns:
        total_energy_pred: 预测的总能量消耗
        total_energy_true: 真实的总能量消耗
    """
    # 检查设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 提取特征和目标
    X = trajectory_data[feature_columns].values
    y = trajectory_data[target_columns].values
    
    # 标准化特征和目标
    if feature_scaler:
        X = feature_scaler.transform(X)
    if target_scaler:
        y = target_scaler.transform(y)
    
    # 准备序列数据
    X_seq = []
    y_next = []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_next.append(y[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_next = np.array(y_next)
    
    # 如果没有足够的数据，返回0
    if len(X_seq) == 0:
        return 0.0, 0.0
    
    # 创建数据集和数据加载器
    dataset = DroneTrajectoryDataset(X_seq, y_next)
    loader = DataLoader(dataset, batch_size=64)
    
    # 模型评估
    model.eval()
    model.to(device)
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            
            # 前向传播
            output = model(inputs)
            
            # 收集预测和真实值
            predictions.append(output.cpu().numpy())
            targets.append(target.numpy())
    
    # 连接结果
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # 分离电流和电压预测
    current_pred = predictions[:, 0]
    voltage_pred = predictions[:, 1]
    current_true = targets[:, 0]
    voltage_true = targets[:, 1]
    
    # 反标准化
    if target_scaler:
        # 重塑数组
        current_pred_2d = current_pred.reshape(-1, 1)
        voltage_pred_2d = voltage_pred.reshape(-1, 1)
        current_true_2d = current_true.reshape(-1, 1)
        voltage_true_2d = voltage_true.reshape(-1, 1)
        
        # 合并电流和电压
        pred_combined = np.hstack((current_pred_2d, voltage_pred_2d))
        true_combined = np.hstack((current_true_2d, voltage_true_2d))
        
        # 反标准化
        pred_combined = target_scaler.inverse_transform(pred_combined)
        true_combined = target_scaler.inverse_transform(true_combined)
        
        # 分离电流和电压
        current_pred = pred_combined[:, 0]
        voltage_pred = pred_combined[:, 1]
        current_true = true_combined[:, 0]
        voltage_true = true_combined[:, 1]
    
    # 计算能量
    energy_pred = calculate_energy(current_pred, voltage_pred, dt)
    energy_true = calculate_energy(current_true, voltage_true, dt)
    
    # 计算总能量
    total_energy_pred = np.sum(energy_pred)
    total_energy_true = np.sum(energy_true)
    
    return total_energy_pred, total_energy_true


if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估无人机轨迹能量预测模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_dir', type=str, default='./processed_data', help='处理后的数据目录')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='输出结果目录')
    parser.add_argument('--model_type', type=str, choices=['bilstm', 'tcn'], required=True, help='模型类型')
    args = parser.parse_args()
    
    # 加载处理后的数据
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    test_trajectories = np.load(os.path.join(args.data_dir, 'test_trajectories.npy'))
    feature_scaler = joblib.load(os.path.join(args.data_dir, 'feature_scaler.pkl'))
    target_scaler = joblib.load(os.path.join(args.data_dir, 'target_scaler.pkl'))
    feature_columns = np.load(os.path.join(args.data_dir, 'feature_columns.npy'))
    
    # 加载模型
    if args.model_type == 'bilstm':
        # 导入BiLSTM模型
        from neural_network_models import BiLSTMModel
        
        # 加载模型参数
        model = BiLSTMModel(
            input_dim=X_test.shape[2],
            hidden_dim=64,
            num_layers=2,
            output_dim=y_test.shape[1]
        )
    else:
        # 导入TCN模型
        from neural_network_models import TCNModel
        
        # 加载模型参数
        model = TCNModel(
            input_dim=X_test.shape[2],
            num_channels=[32, 64, 128],
            kernel_size=3,
            output_dim=y_test.shape[1]
        )
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    # 评估模型
    trajectory_metrics, overall_metrics = evaluate_trajectory_energy(
        model, X_test, y_test, test_trajectories, target_scaler,
        output_dir=os.path.join(args.output_dir, args.model_type)
    )
    
    # 分析轨迹特征与性能的关系
    feature_correlations = analyze_trajectory_characteristics(
        X_test, y_test, test_trajectories, trajectory_metrics, feature_columns,
        output_dir=os.path.join(args.output_dir, args.model_type)
    )