import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt

def read_and_merge_csv_files(directory_path='./cleaned_csv'):
    """
    读取指定目录下的所有CSV文件并合并
    
    Args:
        directory_path: CSV文件所在的目录路径
        
    Returns:
        merged_df: 合并后的DataFrame
        trajectory_ids: 每个样本对应的轨迹ID列表
    """
    print(f"正在读取 {directory_path} 目录下的CSV文件...")
    
    # 获取所有CSV文件路径
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"在 {directory_path} 目录下未找到CSV文件")
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 创建一个空列表用于存储每个DataFrame和轨迹ID
    dfs = []
    trajectory_ids = []
    
    # 读取每个CSV文件并添加轨迹ID
    for i, file_path in enumerate(csv_files):
        try:
            # 从文件名中提取轨迹ID
            traj_id = os.path.basename(file_path).split('.')[0]
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查文件是否为空
            if df.empty:
                print(f"警告: 文件 {file_path} 为空，跳过")
                continue
            
            # 添加轨迹ID列
            df['trajectory_id'] = traj_id
            
            # 将DataFrame和轨迹ID添加到列表中
            dfs.append(df)
            
            # 添加轨迹ID到每个样本
            trajectory_ids.extend([traj_id] * len(df))
            
            if i % 10 == 0:
                print(f"已处理 {i}/{len(csv_files)} 个文件")
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    # 合并所有DataFrame
    if not dfs:
        raise ValueError("没有有效的CSV文件可以合并")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"成功合并 {len(dfs)} 个CSV文件，总行数: {len(merged_df)}")
    
    return merged_df, trajectory_ids

def clean_and_normalize_data(df, feature_columns, target_columns=['current', 'voltage']):
    """
    清洗和归一化数据
    
    Args:
        df: 输入DataFrame
        feature_columns: 特征列名列表
        target_columns: 目标列名列表
        
    Returns:
        cleaned_df: 清洗后的DataFrame
        feature_scaler: 特征归一化器
        target_scaler: 目标归一化器
    """
    print("开始清洗和归一化数据...")
    
    # 检查必要的列是否存在
    missing_features = [col for col in feature_columns if col not in df.columns]
    missing_targets = [col for col in target_columns if col not in df.columns]
    
    if missing_features:
        raise ValueError(f"输入数据缺少以下特征列: {missing_features}")
    if missing_targets:
        raise ValueError(f"输入数据缺少以下目标列: {missing_targets}")
    
    # 复制DataFrame以避免修改原始数据
    cleaned_df = df.copy()
    
    # 检查并处理缺失值
    null_counts = cleaned_df[feature_columns + target_columns].isnull().sum()
    print("缺失值统计:")
    print(null_counts)
    
    if null_counts.sum() > 0:
        print(f"发现 {null_counts.sum()} 个缺失值，使用前向填充方法处理")
        # 按轨迹ID分组，使用前向填充处理缺失值
        cleaned_df = cleaned_df.groupby('trajectory_id').apply(
            lambda group: group.fillna(method='ffill').fillna(method='bfill')
        ).reset_index(drop=True)
        
        # 检查是否还有缺失值
        remaining_nulls = cleaned_df[feature_columns + target_columns].isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"警告: 仍有 {remaining_nulls} 个缺失值无法填充，将删除这些行")
            cleaned_df = cleaned_df.dropna(subset=feature_columns + target_columns)
    
    # 检查并处理异常值 (使用IQR方法)
    for col in feature_columns + target_columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"列 {col} 中发现 {outliers} 个异常值")
            # 将异常值替换为边界值
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 初始化特征和目标归一化器
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # 对特征进行归一化
    cleaned_df[feature_columns] = feature_scaler.fit_transform(cleaned_df[feature_columns])
    
    # 对目标进行归一化
    cleaned_df[target_columns] = target_scaler.fit_transform(cleaned_df[target_columns])
    
    print("数据清洗和归一化完成")
    print(f"清洗后数据形状: {cleaned_df.shape}")
    
    return cleaned_df, feature_scaler, target_scaler

def split_trajectories(df, num_splits=5):
    """
    将每个轨迹拆分成多个小轨迹
    
    Args:
        df: 输入DataFrame
        num_splits: 每个轨迹拆分的数量
        
    Returns:
        split_df: 拆分后的DataFrame
    """
    print(f"将每个轨迹拆分为 {num_splits} 个小轨迹...")
    
    # 创建一个新的DataFrame用于存储拆分后的数据
    split_df = pd.DataFrame()
    
    # 获取唯一的轨迹ID
    unique_trajectories = df['trajectory_id'].unique()
    print(f"共有 {len(unique_trajectories)} 个不同的轨迹")
    
    # 为每个轨迹创建新的ID
    new_trajectory_ids = []
    
    # 拆分每个轨迹
    for traj_id in unique_trajectories:
        # 获取当前轨迹的数据
        traj_data = df[df['trajectory_id'] == traj_id].copy()
        
        # 如果轨迹长度太短，则跳过拆分
        if len(traj_data) < num_splits * 10:  # 确保每个小轨迹至少有10个点
            split_df = pd.concat([split_df, traj_data], ignore_index=True)
            new_trajectory_ids.extend([traj_id] * len(traj_data))
            continue
        
        # 计算每个小轨迹的长度
        split_size = len(traj_data) // num_splits
        
        # 拆分轨迹
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i+1) * split_size if i < num_splits-1 else len(traj_data)
            
            # 获取小轨迹数据
            split_traj = traj_data.iloc[start_idx:end_idx].copy()
            
            # 创建新的轨迹ID
            new_traj_id = f"{traj_id}_split_{i+1}"
            split_traj['trajectory_id'] = new_traj_id
            
            # 添加到新的DataFrame
            split_df = pd.concat([split_df, split_traj], ignore_index=True)
            
            # 添加新的轨迹ID
            new_trajectory_ids.extend([new_traj_id] * len(split_traj))
    
    print(f"拆分后共有 {len(split_df['trajectory_id'].unique())} 个轨迹")
    
    return split_df, new_trajectory_ids

def prepare_sequence_data(df, feature_columns, target_columns, sequence_length=10, step=1):
    """
    准备序列数据用于时序预测
    
    Args:
        df: 输入DataFrame
        feature_columns: 特征列名列表
        target_columns: 目标列名列表
        sequence_length: 序列长度
        step: 采样步长
        
    Returns:
        X: 特征序列
        y: 目标值
        trajectory_indices: 每个样本对应的轨迹索引
    """
    print(f"准备序列数据 (序列长度={sequence_length}, 步长={step})...")
    
    X = []
    y = []
    trajectory_indices = []
    
    # 获取所有轨迹ID
    unique_trajectories = df['trajectory_id'].unique()
    
    for traj_idx, traj_id in enumerate(unique_trajectories):
        # 获取当前轨迹的数据
        traj_data = df[df['trajectory_id'] == traj_id]
        
        # 如果轨迹长度小于序列长度+1(因为我们需要预测下一时刻)，则跳过
        if len(traj_data) < sequence_length + 1:
            continue
        
        # 提取特征和目标
        features = traj_data[feature_columns].values
        targets = traj_data[target_columns].values
        
        # 创建序列
        for i in range(0, len(traj_data) - sequence_length, step):
            X.append(features[i:i+sequence_length])
            y.append(targets[i+sequence_length])  # 下一时刻的目标值
            trajectory_indices.append(traj_idx)
    
    X = np.array(X)
    y = np.array(y)
    trajectory_indices = np.array(trajectory_indices)
    
    print(f"序列数据准备完成，形状: X={X.shape}, y={y.shape}")
    
    return X, y, trajectory_indices

def train_test_split_by_trajectory(X, y, trajectory_indices, test_size=0.3, random_state=42):
    """
    按轨迹划分训练集和测试集
    
    Args:
        X: 特征序列
        y: 目标值
        trajectory_indices: 每个样本对应的轨迹索引
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        X_train, X_test: 训练集和测试集特征
        y_train, y_test: 训练集和测试集目标
        train_trajectories, test_trajectories: 训练集和测试集轨迹索引
    """
    print(f"按轨迹划分训练集和测试集 (测试集比例={test_size})...")
    
    # 获取唯一的轨迹索引
    unique_trajectories = np.unique(trajectory_indices)
    
    # 划分轨迹索引
    train_traj, test_traj = train_test_split(
        unique_trajectories, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # 创建训练集和测试集的掩码
    train_mask = np.isin(trajectory_indices, train_traj)
    test_mask = np.isin(trajectory_indices, test_traj)
    
    # 划分数据
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    train_trajectories = trajectory_indices[train_mask]
    test_trajectories = trajectory_indices[test_mask]
    
    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    print(f"训练集轨迹数: {len(np.unique(train_trajectories))}, 测试集轨迹数: {len(np.unique(test_trajectories))}")
    
    return X_train, X_test, y_train, y_test, train_trajectories, test_trajectories

def save_processed_data(output_dir, X_train, X_test, y_train, y_test, 
                       train_trajectories, test_trajectories, 
                       feature_scaler, target_scaler, 
                       feature_columns, target_columns):
    """
    保存处理后的数据
    
    Args:
        output_dir: 输出目录
        X_train, X_test: 训练集和测试集特征
        y_train, y_test: 训练集和测试集目标
        train_trajectories, test_trajectories: 训练集和测试集轨迹索引
        feature_scaler, target_scaler: 特征和目标归一化器
        feature_columns, target_columns: 特征和目标列名
    """
    print(f"保存处理后的数据到 {output_dir} 目录...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(output_dir, 'train_trajectories.npy'), train_trajectories)
    np.save(os.path.join(output_dir, 'test_trajectories.npy'), test_trajectories)
    
    # 保存归一化器
    import joblib
    joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(output_dir, 'target_scaler.pkl'))
    
    # 保存列名
    np.save(os.path.join(output_dir, 'feature_columns.npy'), feature_columns)
    np.save(os.path.join(output_dir, 'target_columns.npy'), target_columns)
    
    print("数据保存完成")

def visualize_data(df, feature_columns, target_columns, output_dir):
    """
    可视化数据分布
    
    Args:
        df: 输入DataFrame
        feature_columns: 特征列名列表
        target_columns: 目标列名列表
        output_dir: 输出目录
    """
    print("生成数据可视化...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化特征分布
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_columns):
        plt.subplot(4, 5, i+1)
        plt.hist(df[col], bins=50)
        plt.title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distribution.png'))
    
    # 可视化目标分布
    plt.figure(figsize=(10, 5))
    for i, col in enumerate(target_columns):
        plt.subplot(1, 2, i+1)
        plt.hist(df[col], bins=50)
        plt.title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    
    # 可视化轨迹样例
    unique_trajectories = df['trajectory_id'].unique()
    sample_trajectories = np.random.choice(unique_trajectories, min(5, len(unique_trajectories)), replace=False)
    
    plt.figure(figsize=(15, 10))
    for i, traj_id in enumerate(sample_trajectories):
        traj_data = df[df['trajectory_id'] == traj_id]
        plt.subplot(2, 3, i+1)
        plt.plot(traj_data['pos_x'], traj_data['pos_y'], 'b-')
        plt.plot(traj_data['pos_x'].iloc[0], traj_data['pos_y'].iloc[0], 'go', markersize=8)  # 起点
        plt.plot(traj_data['pos_x'].iloc[-1], traj_data['pos_y'].iloc[-1], 'ro', markersize=8)  # 终点
        plt.title(f'轨迹 {traj_id}')
        plt.xlabel('X 位置')
        plt.ylabel('Y 位置')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_samples.png'))
    
    print("数据可视化完成")

def process_drone_data(
    input_dir='./cleaned_csv',
    output_dir='./processed_data',
    sequence_length=20,
    test_size=0.3,
    num_splits=5,
    random_state=42
):
    """
    无人机轨迹数据处理的主函数
    
    Args:
        input_dir: 输入目录，包含CSV文件
        output_dir: 输出目录
        sequence_length: 序列长度
        test_size: 测试集比例
        num_splits: 每个轨迹拆分的数量
        random_state: 随机种子
    """
    # 定义特征列和目标列
    feature_columns = [
        'rate_roll', 'rate_pitch', 'rate_yaw', 
        'pos_x', 'pos_y', 'pos_z', 
        'vel_x', 'vel_y', 'vel_z', 
        'R11', 'R12', 'R13', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33'
    ]
    target_columns = ['current', 'voltage']
    
    # 读取并合并CSV文件
    merged_df, trajectory_ids = read_and_merge_csv_files(input_dir)
    
    # 清洗和归一化数据
    cleaned_df, feature_scaler, target_scaler = clean_and_normalize_data(
        merged_df, feature_columns, target_columns
    )
    
    # 将每个轨迹拆分成多个小轨迹
    split_df, new_trajectory_ids = split_trajectories(cleaned_df, num_splits)
    
    # 可视化数据
    visualize_data(split_df, feature_columns, target_columns, output_dir)
    
    # 准备序列数据
    X, y, trajectory_indices = prepare_sequence_data(
        split_df, feature_columns, target_columns, sequence_length
    )
    
    # 按轨迹划分训练集和测试集
    X_train, X_test, y_train, y_test, train_trajectories, test_trajectories = train_test_split_by_trajectory(
        X, y, trajectory_indices, test_size, random_state
    )
    
    # 保存处理后的数据
    save_processed_data(
        output_dir, X_train, X_test, y_train, y_test, 
        train_trajectories, test_trajectories, 
        feature_scaler, target_scaler, 
        feature_columns, target_columns
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_trajectories': train_trajectories,
        'test_trajectories': test_trajectories,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_columns': feature_columns,
        'target_columns': target_columns
    }

if __name__ == "__main__":
    process_drone_data()