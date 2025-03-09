import torch
import torchvision.models as models

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

#below, processed_uav_data.csv has some data missing, so it won't work

# # 读取 CSV 文件
# df = pd.read_csv("17.csv")

# # 提取所需列
# columns_needed = list(range(4, 7)) + [10,11]+ list(range(46, 50)) + list(range(60, 64)) + list(range(66, 69))
# df = df.iloc[:, columns_needed]
# print(df.head())
# # 计算目标变量（能量消耗 = 电流 × 电压）
# df["energy"] = df.iloc[:, 3] * df.iloc[:, 4]  # 电流 × 电压
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])  # 归一化特征数据  already 2-D
# df["energy"] = scaler.fit_transform(df["energy"].values.reshape(-1, 1))  # 归一化目标变量 original 1-D, so need reshape to transfer into 2-D

# # 删除电流和电压列（因为能量已计算）
# df.drop(columns=[df.columns[3], df.columns[4]], inplace=True)
# # 保存处理后数据
# df.to_csv("processed_uav_data.csv", index=False)

# print(df.isnull().sum())
# print(np.isinf(df.values).sum())

# print("数据处理完成，已保存到 processed_uav_data.csv")
# print(f"imput dim :{df.shape[1]-1}")
class UAVTimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq_data = self.data.iloc[idx:idx+self.seq_length, :-1].values.astype(np.float32)  # 输入特征
        target = self.data.iloc[idx+self.seq_length, -1]  # 目标变量（能量消耗）
        return torch.tensor(seq_data), torch.tensor(target, dtype=torch.float32)

# 设定时间步长度
seq_length = 50
batch_size = 16

train_dataset = UAVTimeSeriesDataset(csv_file="fully_cleaned_uav_data.csv", seq_length=seq_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("数据加载完成！")
import torch.nn as nn

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
# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置参数
input_dim = 14  # 除去电流和电压后的特征数
hidden_dim = 64
num_layers = 2
output_dim = 1

# 初始化模型
gru_model = GRURegressor(input_dim, hidden_dim, num_layers, output_dim).to(device)
tcn_model = TCN(input_dim, output_dim).to(device)

# 损失函数 & 优化器
criterion = nn.MSELoss()
optimizer_gru = torch.optim.Adam(gru_model.parameters(), lr=0.001)
optimizer_tcn = torch.optim.Adam(tcn_model.parameters(), lr=0.001)

# 训练循环
num_epochs = 1
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

    # 计算当前 epoch 平均损失并存储
    epoch_loss_gru = running_loss_gru / len(train_loader)
    epoch_loss_tcn = running_loss_tcn / len(train_loader)
    gru_losses.append(epoch_loss_gru)
    tcn_losses.append(epoch_loss_tcn)

    print(f"Epoch [{epoch+1}/{num_epochs}], GRU Loss: {epoch_loss_gru:.4f}, TCN Loss: {epoch_loss_tcn:.4f}")

print("训练完成！")

# 读取测试数据
test_data = pd.read_csv("fully_cleaned_uav_data.csv").iloc[-50:, :-1].values.astype(np.float32)
test_tensor = torch.tensor(test_data).unsqueeze(0).to(device)  # 添加 batch 维度

# 预测
gru_model.eval()
tcn_model.eval()

with torch.no_grad():
    gru_prediction = gru_model(test_tensor).item()
    tcn_prediction = tcn_model(test_tensor).item()

print(f"GRU 预测的能量消耗: {gru_prediction:.4f}")
print(f"TCN 预测的能量消耗: {tcn_prediction:.4f}")
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), gru_losses, label="GRU Loss", marker="o")
plt.plot(range(1, num_epochs + 1), tcn_losses, label="TCN Loss", marker="s")

plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("GRU vs TCN Loss during Training")
plt.legend()
plt.grid(True)
plt.show()
gru_model.eval()
tcn_model.eval()

actual_values = []
gru_predictions = []
tcn_predictions = []

with torch.no_grad():
    for sequences, targets in train_loader:
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

# 绘制实际值 vs 预测值
plt.figure(figsize=(8, 5))
plt.plot(actual_values, label="Actual Energy", color="black", linestyle="dashed", marker="o", alpha=0.7)
plt.plot(gru_predictions, label="GRU Predicted Energy", color="blue", linestyle="solid", marker="s", alpha=0.7)
plt.plot(tcn_predictions, label="TCN Predicted Energy", color="red", linestyle="solid", marker="d", alpha=0.7)

plt.xlabel("Test Samples")
plt.ylabel("Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption (GRU & TCN)")
plt.legend()
plt.grid(True)
plt.show()