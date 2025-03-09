import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ================================================
# STEP 1: READ THE CSV FILE
# ================================================
# 读取文件 "./Flight_csv/Feb4_f01r.csv"
df = pd.read_csv("./Flight_csv/Feb4_f01r.csv")
print("Initial Data Sample:")
print(df.head())

# ================================================
# STEP 2: DUPLICATE REMOVAL AND SORTING
# ================================================
df = df.drop_duplicates()
# 将 time 转为数值，并按时间排序
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df = df.sort_values("time").reset_index(drop=True)
print(f"Data shape after duplicate removal and sorting: {df.shape}")

# ================================================
# STEP 3: MISSING VALUE HANDLING
# ================================================
# 对时序数据采用线性插值，然后用前向填充和后向填充
df = df.interpolate(method="linear", limit_direction="both")
df = df.fillna(method="ffill").fillna(method="bfill")
print("Missing values handled using interpolation and ffill/bfill.")
print(df.head())

# ================================================
# STEP 4: COMPUTE dt, INSTANTANEOUS ENERGY LOSS, AND KINETIC ENERGY
# ================================================
# 计算 dt：相邻记录时间差（秒）
df["dt"] = df["time"].diff().fillna(0)

# 计算每一步的能量损耗（单位：Joule）
# 使用绝对值计算（处理 voltage 或 current 为负的情况）
df["energy_step"] = abs(df["voltage"] * df["current"]) * df["dt"]

# 计算速度（假设 estimator 中给出的速度列：vel_x, vel_y, vel_z）
df["speed"] = np.sqrt(df["vel_x"]**2 + df["vel_y"]**2 + df["vel_z"]**2)

# 计算动能（KE），假设质量 m=1 kg： KE = 0.5 * m * speed^2
df["kinetic_energy"] = 0.5 * (df["speed"]**2)

# 计算残余能量：当前步能量损耗 - 动能
df["residual_energy"] = df["energy_step"] - df["kinetic_energy"]

# 对负的 residual_energy 值进行剪切（clip），赋值为 0
df["residual_energy"] = df["residual_energy"].clip(lower=0)

print("Energy and kinetic energy computed.")
print(df[["time", "energy_step", "kinetic_energy", "residual_energy"]].head())

# ================================================
# STEP 5: OUTLIER DETECTION AND REMOVAL (OPTIONAL)
# ================================================
def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# 对部分关键数值列剔除异常值（如能量、动能、残余能量和速度）
for col in ["energy_step", "kinetic_energy", "residual_energy", "speed"]:
    df = remove_outliers_iqr(df, col)
print(f"Data shape after outlier removal: {df.shape}")

# # ================================================
# # STEP 6: NORMALIZATION
# # ================================================
# # 设定目标变量为 residual_energy，其它为特征
# features = df.drop("residual_energy", axis=1)
# target = df[["residual_energy"]]

# scaler_features = StandardScaler()
# features_scaled = scaler_features.fit_transform(features)
# features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

# scaler_target = StandardScaler()
# target_scaled = scaler_target.fit_transform(target)
# target_scaled = pd.DataFrame(target_scaled, columns=["residual_energy"])

# df_normalized = pd.concat([features_scaled, target_scaled], axis=1)
# print("Normalization complete.")
# print(df_normalized.head())

# ================================================
# STEP 7: SAVE THE CLEANED DATA
# ================================================
output_file = "./Flight_csv/Fully_cleaned_uav_data.csv"
df.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}.")

# ================================================
# STEP 8: OPTIONAL VISUALIZATION
# ================================================
plt.figure(figsize=(12,8))
df.hist(bins=30, figsize=(12,8))
plt.tight_layout()
plt.show()
