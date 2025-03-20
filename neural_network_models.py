import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import joblib
from tqdm import tqdm

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 自定义数据集类
class DroneTrajectoryDataset(Dataset):
    def __init__(self, features, targets):
        """
        初始化数据集
        
        Args:
            features: 特征序列，形状为 [样本数, 序列长度, 特征维度]
            targets: 目标值，形状为 [样本数, 目标维度]
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# BiLSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        初始化BiLSTM模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            output_dim: 输出维度
            dropout: Dropout比例
        """
        super(BiLSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接输出层 (因为是双向LSTM，所以隐藏层维度 * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 2是因为双向
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 只使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        
        return out

# 因果卷积层 (用于TCN)
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        """
        初始化因果卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dilation: 扩张率
        """
        super(CausalConv1d, self).__init__()
        
        # 计算填充大小，使得卷积是因果的（只使用过去的信息）
        self.padding = (kernel_size - 1) * dilation
        
        # 一维卷积
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=self.padding, 
            dilation=dilation
        )
        
    def forward(self, x):
        # x的形状: [batch, channels, seq_len]
        x = self.conv(x)
        # 移除因果填充引入的额外元素
        return x[:, :, :-self.padding]

# 残差块 (用于TCN)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        """
        初始化残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dilation: 扩张率
            dropout: Dropout比例
        """
        super(ResidualBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = CausalConv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            dilation=dilation
        )
        
        # 第二个卷积层
        self.conv2 = CausalConv1d(
            out_channels, 
            out_channels, 
            kernel_size, 
            dilation=dilation
        )
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 如果输入输出通道数不同，添加一个1x1卷积进行调整
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        # 残差连接
        residual = x if self.downsample is None else self.downsample(x)
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 添加残差连接并应用ReLU
        return self.relu(out + residual)

# TCN模型
class TCNModel(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size, output_dim, dropout=0.2):
        """
        初始化TCN模型
        
        Args:
            input_dim: 输入特征维度
            num_channels: 每层的通道数列表
            kernel_size: 卷积核大小
            output_dim: 输出维度
            dropout: Dropout比例
        """
        super(TCNModel, self).__init__()
        
        # 创建TCN模型的层
        layers = []
        num_levels = len(num_channels)
        
        # 输入适配层
        self.input_adapter = nn.Conv1d(input_dim, num_channels[0], 1)
        
        # 添加残差块
        for i in range(num_levels):
            dilation_size = 2 ** i  # 扩张率随层数指数增长
            in_channels = num_channels[i-1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]
            layers.append(
                ResidualBlock(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    dilation=dilation_size, 
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
        # 全连接输出层
        self.fc = nn.Linear(num_channels[-1], output_dim)
        
    def forward(self, x):
        # 输入形状: [batch, seq_len, features]
        # 转换为 [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # 应用输入适配层
        x = self.input_adapter(x)
        
        # 通过TCN网络
        x = self.network(x)
        
        # 只使用最后一个时间步的输出
        x = x[:, :, -1]
        
        # 应用全连接层得到最终输出
        return self.fc(x)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_path, patience=10):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 设备 (CPU/GPU)
        model_path: 模型保存路径
        patience: 早停耐心值
        
    Returns:
        model: 训练好的模型
        train_losses: 训练损失列表
        val_losses: 验证损失列表
    """
    # 将模型移至指定设备
    model.to(device)
    
    # 记录最佳验证损失和没有改进的轮数
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # 保存训练和验证损失
    train_losses = []
    val_losses = []
    
    # 创建目录保存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # 使用tqdm显示进度条
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        # 训练一个epoch
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 累加损失
            train_loss += loss.item() * inputs.size(0)
            
            # 更新进度条
            train_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        # 使用tqdm显示进度条
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 累加损失
                val_loss += loss.item() * inputs.size(0)
                
                # 更新进度条
                val_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 检查是否需要保存模型
        if val_loss < best_val_loss:
            print(f'Validation loss decreased from {best_val_loss:.6f} to {val_loss:.6f}. Saving model...')
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # 保存模型
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not decrease. Epochs without improvement: {epochs_no_improve}')
        
        # 检查早停条件
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_path))
    
    return model, train_losses, val_losses


# 计算能量消耗
def calculate_energy(current, voltage, dt=0.02):
    """
    计算能量消耗
    
    Args:
        current: 电流值
        voltage: 电压值
        dt: 时间步长
        
    Returns:
        energy: 能量消耗
    """
    # 功率 = 电流 * 电压
    power = current * voltage
    
    # 能量 = 功率 * 时间
    energy = power * dt
    
    return energy


# 预测函数
def predict(model, data_loader, device):
    """
    使用模型进行预测
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备 (CPU/GPU)
        
    Returns:
        predictions: 预测值
        targets: 真实值
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in tqdm(data_loader, desc='Predicting'):
            inputs = inputs.to(device)
            
            # 预测
            output = model(inputs)
            
            # 转换为NumPy数组
            predictions.append(output.cpu().numpy())
            targets.append(target.numpy())
    
    # 拼接结果
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    return predictions, targets


# 评估函数
def evaluate_energy_prediction(current_pred, voltage_pred, current_true, voltage_true, trajectory_indices=None, dt=0.02):
    """
    评估能量预测性能
    
    Args:
        current_pred: 预测电流值
        voltage_pred: 预测电压值
        current_true: 真实电流值
        voltage_true: 真实电压值
        trajectory_indices: 轨迹索引
        dt: 时间步长
        
    Returns:
        metrics: 评估指标字典
    """
    # 计算预测能量和真实能量
    energy_pred = calculate_energy(current_pred, voltage_pred, dt)
    energy_true = calculate_energy(current_true, voltage_true, dt)
    
    # 计算整体评估指标
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
    
    # 如果提供了轨迹索引，计算每个轨迹的评估指标
    if trajectory_indices is not None:
        unique_trajectories = np.unique(trajectory_indices)
        trajectory_metrics = {}
        
        for traj_idx in unique_trajectories:
            # 获取当前轨迹的掩码
            mask = trajectory_indices == traj_idx
            
            # 计算当前轨迹的评估指标
            traj_energy_pred = energy_pred[mask]
            traj_energy_true = energy_true[mask]
            
            traj_mse = mean_squared_error(traj_energy_true, traj_energy_pred)
            traj_rmse = np.sqrt(traj_mse)
            traj_mae = mean_absolute_error(traj_energy_true, traj_energy_pred)
            traj_r2 = r2_score(traj_energy_true, traj_energy_pred) if len(traj_energy_true) > 1 else 0
            
            trajectory_metrics[traj_idx] = {
                'mse': traj_mse,
                'rmse': traj_rmse,
                'mae': traj_mae,
                'r2': traj_r2
            }
        
        metrics['trajectory_metrics'] = trajectory_metrics
    
    return metrics


# 可视化结果
def visualize_results(train_losses, val_losses, energy_pred, energy_true, current_pred=None, 
                     current_true=None, voltage_pred=None, voltage_true=None, 
                     trajectory_indices=None, output_dir='./results'):
    """
    可视化训练和预测结果
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        energy_pred: 预测能量
        energy_true: 真实能量
        current_pred: 预测电流
        current_true: 真实电流
        voltage_pred: 预测电压
        voltage_true: 真实电压
        trajectory_indices: 轨迹索引
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()
    
    # 绘制预测vs真实值散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(energy_true, energy_pred, alpha=0.5)
    plt.plot([min(energy_true), max(energy_true)], [min(energy_true), max(energy_true)], 'r--')
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    plt.title('True vs Predicted Energy')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'energy_scatter.png'))
    plt.close()
    
    # 如果提供了电流和电压，绘制它们的预测vs真实值散点图
    if current_pred is not None and current_true is not None:
        plt.figure(figsize=(10, 6))
        plt.scatter(current_true, current_pred, alpha=0.5)
        plt.plot([min(current_true), max(current_true)], [min(current_true), max(current_true)], 'r--')
        plt.xlabel('True Current')
        plt.ylabel('Predicted Current')
        plt.title('True vs Predicted Current')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'current_scatter.png'))
        plt.close()
    
    if voltage_pred is not None and voltage_true is not None:
        plt.figure(figsize=(10, 6))
        plt.scatter(voltage_true, voltage_pred, alpha=0.5)
        plt.plot([min(voltage_true), max(voltage_true)], [min(voltage_true), max(voltage_true)], 'r--')
        plt.xlabel('True Voltage')
        plt.ylabel('Predicted Voltage')
        plt.title('True vs Predicted Voltage')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'voltage_scatter.png'))
        plt.close()
    
    # 如果提供了轨迹索引，绘制一些轨迹的预测与真实能量对比图
    if trajectory_indices is not None:
        unique_trajectories = np.unique(trajectory_indices)
        
        # 最多绘制5个轨迹
        sample_trajectories = unique_trajectories[:min(5, len(unique_trajectories))]
        
        for traj_idx in sample_trajectories:
            # 获取当前轨迹的掩码
            mask = trajectory_indices == traj_idx
            
            # 提取当前轨迹的能量值
            traj_energy_pred = energy_pred[mask]
            traj_energy_true = energy_true[mask]
            
            # 绘制当前轨迹的能量预测与真实值对比图
            plt.figure(figsize=(12, 6))
            plt.plot(np.arange(len(traj_energy_true)), traj_energy_true, label='True Energy')
            plt.plot(np.arange(len(traj_energy_pred)), traj_energy_pred, label='Predicted Energy')
            plt.xlabel('Time Step')
            plt.ylabel('Energy')
            plt.title(f'Trajectory {traj_idx} - Energy Prediction')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'trajectory_{traj_idx}_energy.png'))
            plt.close()


# 训练模型的主函数
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, 
                       test_trajectories=None, model_type='bilstm', 
                       input_dim=18, hidden_dim=64, num_layers=2, 
                       output_dim=2, num_channels=[32, 64, 128], kernel_size=3,
                       batch_size=32, learning_rate=0.001, num_epochs=100, 
                       patience=10, device=None, output_dir='./results'):
    """
    训练和评估模型
    
    Args:
        X_train, y_train: 训练集特征和目标
        X_val, y_val: 验证集特征和目标
        X_test, y_test: 测试集特征和目标
        test_trajectories: 测试集轨迹索引
        model_type: 模型类型 ('bilstm' 或 'tcn')
        input_dim: 输入特征维度
        hidden_dim: LSTM隐藏层维度
        num_layers: LSTM层数
        output_dim: 输出维度
        num_channels: TCN通道数列表
        kernel_size: TCN卷积核大小
        batch_size: 批大小
        learning_rate: 学习率
        num_epochs: 训练轮数
        patience: 早停耐心值
        device: 设备 (CPU/GPU)
        output_dir: 输出目录
        
    Returns:
        model: 训练好的模型
        metrics: 评估指标
    """
    # 检查设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 创建输出目录
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建数据集和数据加载器
    train_dataset = DroneTrajectoryDataset(X_train, y_train)
    val_dataset = DroneTrajectoryDataset(X_val, y_val)
    test_dataset = DroneTrajectoryDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 创建模型
    if model_type.lower() == 'bilstm':
        model = BiLSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim
        )
        model_path = os.path.join(model_dir, 'bilstm_model.pth')
    elif model_type.lower() == 'tcn':
        model = TCNModel(
            input_dim=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            output_dim=output_dim
        )
        model_path = os.path.join(model_dir, 'tcn_model.pth')
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 打印模型结构
    print(f"\n{model_type.upper()} 模型结构:")
    print(model)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print(f"\n开始训练 {model_type.upper()} 模型...")
    start_time = time.time()
    
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs, device, model_path, patience
    )
    
    training_time = time.time() - start_time
    print(f"{model_type.upper()} 模型训练完成，耗时: {training_time:.2f} 秒")
    
    # 在测试集上进行预测
    print("\n在测试集上进行预测...")
    predictions, targets = predict(model, test_loader, device)
    
    # 将预测结果分离为电流和电压
    current_pred = predictions[:, 0]
    voltage_pred = predictions[:, 1]
    current_true = targets[:, 0]
    voltage_true = targets[:, 1]
    
    # 评估能量预测性能
    print("\n评估能量预测性能...")
    metrics = evaluate_energy_prediction(
        current_pred, voltage_pred, current_true, voltage_true, 
        trajectory_indices=test_trajectories
    )
    
    # 打印评估结果
    print("\n整体评估指标:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    
    # 计算预测能量和真实能量
    energy_pred = calculate_energy(current_pred, voltage_pred)
    energy_true = calculate_energy(current_true, voltage_true)
    
    # 可视化结果
    print("\n生成可视化结果...")
    visualize_results(
        train_losses, val_losses, energy_pred, energy_true,
        current_pred, current_true, voltage_pred, voltage_true,
        trajectory_indices=test_trajectories, output_dir=model_dir
    )
    
    # 保存评估结果
    results = {
        'model_type': model_type,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time
    }
    
    joblib.dump(results, os.path.join(model_dir, f'{model_type}_results.pkl'))
    
    print(f"\n{model_type.upper()} 模型评估完成，结果已保存到 {model_dir}")
    
    return model, metrics