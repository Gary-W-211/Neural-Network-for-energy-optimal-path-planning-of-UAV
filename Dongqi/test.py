import torch
import torchvision.models as models

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
# 读取数据
df = pd.read_csv("fully_cleaned_uav_data.csv") 

# 划分训练集和测试集（80% 训练, 20% 测试）
train_data, test_data = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

# 保存划分后的数据
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
class UAVTimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq_data = self.data.iloc[idx:idx+self.seq_length, :-1].values.astype(np.float32)  # 特征
        target = self.data.iloc[idx+self.seq_length, -1]  # 目标变量（能量消耗）
        return torch.tensor(seq_data), torch.tensor(target, dtype=torch.float32)

# 设定时间步长度
seq_length = 50
batch_size = 16

# 创建训练集和测试集数据加载器
train_dataset = UAVTimeSeriesDataset(csv_file="train_data.csv", seq_length=seq_length)
test_dataset = UAVTimeSeriesDataset(csv_file="test_data.csv", seq_length=seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练数据样本数: {len(train_dataset)}, 测试数据样本数: {len(test_dataset)}")
# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置参数
input_dim = train_data.shape[1] - 1  # 特征数
hidden_dim = 128
num_layers = 2
output_dim = 1

# 初始化模型
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]  # 取最后时间步
        output = self.fc(gru_out)
        return output
    
class TCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TCN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整形状适应 Conv1d
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x
gru_model = GRURegressor(input_dim, hidden_dim, num_layers, output_dim).to(device)
tcn_model = TCN(input_dim, output_dim).to(device)

# 损失函数 & 优化器
criterion = nn.MSELoss()
optimizer_gru = torch.optim.Adam(gru_model.parameters(), lr=0.001)
optimizer_tcn = torch.optim.Adam(tcn_model.parameters(), lr=0.001)

# 训练循环
num_epochs = 50
gru_losses = []
tcn_losses = []

for epoch in range(num_epochs):
    running_loss_gru = 0.0
    running_loss_tcn = 0.0
    
    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        
        # 训练 GRU
        optimizer_gru.zero_grad()
        outputs_gru = gru_model(sequences)
        loss_gru = criterion(outputs_gru.squeeze(), targets)
        loss_gru.backward()
        optimizer_gru.step()
        running_loss_gru += loss_gru.item()
        
        # 训练 TCN
        optimizer_tcn.zero_grad()
        outputs_tcn = tcn_model(sequences)
        loss_tcn = criterion(outputs_tcn.squeeze(), targets)
        loss_tcn.backward()
        optimizer_tcn.step()
        running_loss_tcn += loss_tcn.item()

    # 计算当前 epoch 平均损失
    epoch_loss_gru = running_loss_gru / len(train_loader)
    epoch_loss_tcn = running_loss_tcn / len(train_loader)
    gru_losses.append(epoch_loss_gru)
    tcn_losses.append(epoch_loss_tcn)

    print(f"Epoch [{epoch+1}/{num_epochs}], GRU Loss: {epoch_loss_gru:.4f}, TCN Loss: {epoch_loss_tcn:.4f}")

print("训练完成！")
gru_model.eval()
tcn_model.eval()

test_loss_gru = 0.0
test_loss_tcn = 0.0

with torch.no_grad():
    for sequences, targets in test_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        
        # GRU 预测
        outputs_gru = gru_model(sequences)
        loss_gru = criterion(outputs_gru.squeeze(), targets)
        test_loss_gru += loss_gru.item()
        
        # TCN 预测
        outputs_tcn = tcn_model(sequences)
        loss_tcn = criterion(outputs_tcn.squeeze(), targets)
        test_loss_tcn += loss_tcn.item()

# 计算测试集平均损失
test_loss_gru /= len(test_loader)
test_loss_tcn /= len(test_loader)

print(f"GRU 测试集损失 (MSE): {test_loss_gru:.4f}")
print(f"TCN 测试集损失 (MSE): {test_loss_tcn:.4f}")
gru_model.eval()
tcn_model.eval()

actual_values = []
gru_predictions = []
tcn_predictions = []

with torch.no_grad():
    for sequences, targets in test_loader:
        sequences, targets = sequences.to(device), targets.to(device)

        # 真实值
        actual_values.extend(targets.cpu().numpy())

        # GRU 预测
        outputs_gru = gru_model(sequences)
        gru_predictions.extend(outputs_gru.cpu().numpy())

        # TCN 预测
        outputs_tcn = tcn_model(sequences)
        tcn_predictions.extend(outputs_tcn.cpu().numpy())

# 转换为 NumPy 数组
actual_values = np.array(actual_values)
gru_predictions = np.array(gru_predictions).flatten()
tcn_predictions = np.array(tcn_predictions).flatten()

# 计算误差
gru_errors = np.abs(actual_values - gru_predictions)
tcn_errors = np.abs(actual_values - tcn_predictions)

# 计算成功率
gru_success_rate = 1 - (gru_errors / (actual_values + 1e-6))  # 避免除零
tcn_success_rate = 1 - (tcn_errors / (actual_values + 1e-6))

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), gru_losses, label="GRU Loss", marker="o")
plt.plot(range(1, num_epochs + 1), tcn_losses, label="TCN Loss", marker="s")

plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("GRU vs TCN Training Loss")
plt.legend()
plt.grid(True)
plt.show()
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 5))
# plt.plot(range(len(actual_values)), gru_success_rate, label="GRU Success Rate", marker="o", linestyle="dashed", alpha=0.7)
# plt.plot(range(len(actual_values)), tcn_success_rate, label="TCN Success Rate", marker="s", linestyle="dashed", alpha=0.7)

# plt.xlabel("Test Samples")
# plt.ylabel("Success Rate")
# plt.title("GRU vs TCN Success Rate in Energy Consumption Prediction")
# plt.legend()
# plt.grid(True)
# plt.show()
import seaborn as sns

gru_errors = gru_predictions - actual_values
tcn_errors = tcn_predictions - actual_values

plt.figure(figsize=(8, 5))
sns.histplot(gru_errors, color="blue", label="GRU Errors", kde=True, alpha=0.5)
sns.histplot(tcn_errors, color="red", label="TCN Errors", kde=True, alpha=0.5)

plt.xlabel("Error (Predicted - Actual)")
plt.ylabel("Count")
plt.title("Error Distribution")
plt.legend()
plt.show()
time_axis = np.arange(len(actual_values))  # 或者真实的时间戳

# 只取前 500 个数据
num_samples = 500
time_axis_subset = time_axis[:num_samples]
actual_subset = actual_values[:num_samples]
gru_subset = gru_predictions[:num_samples]
tcn_subset = tcn_predictions[:num_samples]

plt.figure(figsize=(10, 5))
plt.plot(time_axis_subset, actual_subset, label="Actual", color="black", marker="o", markersize=4)
plt.plot(time_axis_subset, gru_subset, label="GRU", color="blue", marker="s", markersize=4)
plt.plot(time_axis_subset, tcn_subset, label="TCN", color="red", marker="d", markersize=4)

plt.xlabel("Time Index")
plt.ylabel("Energy Consumption")
plt.title("Time-series View (First 500 samples)")
plt.legend()
plt.grid(True)
plt.show()
# # 保存模型
# torch.save(gru_model.state_dict(), "gru_model.pth")
# torch.save(tcn_model.state_dict(), "tcn_model.pth")
# print("模型已保存！")
# # 加载模型
# gru_model_loaded = GRURegressor(input_dim, hidden_dim, num_layers, output_dim).to(device)
# tcn_model_loaded = TCN(input_dim, output_dim).to(device)
