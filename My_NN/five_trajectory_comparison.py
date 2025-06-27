#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
# Paths
results_dir = os.path.expanduser(
    "~/Capstone/catkin_ws_79/src/hiperlab_rostools/src/MEng79/Results"
)
model_path = os.path.expanduser(
    "~/Capstone/Neural/NN/My_NN/models/gru_model.pth"
)
features_scaler_path = os.path.expanduser(
    "~/Capstone/Neural/NN/My_NN/scalers/features_scaler.pkl"
)
target_scaler_path = os.path.expanduser(
    "~/Capstone/Neural/NN/My_NN/scalers/target_scaler.pkl"
)
norm_out_dir = "./My_NN/Normalized"
pred_out_dir = "./Results"
os.makedirs(norm_out_dir, exist_ok=True)
os.makedirs(pred_out_dir, exist_ok=True)

# Trajectory CSVs to process
trajectory_files = [
    "direct.csv",
    "speed.csv",
    "gru.csv",
    "lstm.csv",
    "tcn.csv",
]

# Neural network parameters
seq_length = 50
batch_size = 32
dt = 0.01  # time interval per step

# -----------------------------------------------------------------------------
# 2. Define model and dataset
# -----------------------------------------------------------------------------
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]     # take the output at the final time step
        output = self.fc(last_step)
        return output

class UAVTimeSeriesTestDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data.iloc[idx : idx + self.seq_length, :].values.astype(np.float32)
        return torch.tensor(seq)

# -----------------------------------------------------------------------------
# 3. Load scalers and model
# -----------------------------------------------------------------------------
features_scaler = joblib.load(features_scaler_path)
target_scaler = joblib.load(target_scaler_path)

# Determine input dimension from training CSV (number of features)
train_norm_csv = "./My_NN/data_for_train/train_data.csv"
input_dim = pd.read_csv(train_norm_csv).shape[1] - 1

hidden_dim = 128
num_layers = 2
output_dim = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GRURegressor(input_dim, hidden_dim, num_layers, output_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------------------------------------------------------------
# 4. Inference loop over trajectories
# -----------------------------------------------------------------------------
results = {}
all_lengths = {}
all_cum_energies = {}
all_preds    = {} 

for fname in trajectory_files:
    file_path = os.path.join(results_dir, fname)
    print(f"Processing {file_path}…")
    raw_df = pd.read_csv(file_path)
    num_steps = len(raw_df)
    traj_name = os.path.splitext(fname)[0]
    all_lengths[traj_name] = num_steps

    # Drop the first column if it's time or timestamp
    if raw_df.columns[0].lower() in ("time", "timestamp"):
        raw_df = raw_df.drop(raw_df.columns[0], axis=1)

    # Normalize features
    feats_norm = features_scaler.transform(raw_df.values)
    norm_df = pd.DataFrame(feats_norm, columns=raw_df.columns)
    norm_csv = os.path.join(norm_out_dir, f"normalized_{fname}")
    norm_df.to_csv(norm_csv, index=False)

    # Prepare DataLoader
    dataset = UAVTimeSeriesTestDataset(norm_csv, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Run inference
    preds_norm = []
    with torch.no_grad():
        for seq in loader:
            seq = seq.to(device)
            out = model(seq)
            preds_norm.extend(out.cpu().numpy())
    preds_norm = np.array(preds_norm).reshape(-1, 1)

    # Inverse transform predictions
    preds = target_scaler.inverse_transform(preds_norm).flatten()
    # Pad initial seq_length steps by repeating the first prediction
    padded = np.pad(preds, (seq_length, 0), 'edge')[:num_steps]
    padded = np.abs(padded) * dt
    all_preds[traj_name] = padded

    # Compute cumulative and total energy
    cum_energy = np.cumsum(padded)
    total_energy = padded.sum()

    results[traj_name] = total_energy
    all_cum_energies[traj_name] = cum_energy

    # Save per-trajectory CSV
    out_df = pd.DataFrame({
        "step": np.arange(num_steps),
        "predicted_energy": padded,
        "cumulative_energy": cum_energy
    })
    out_df.to_csv(os.path.join(pred_out_dir, f"predictions_{traj_name}.csv"), index=False)
    print(f" → {traj_name}: total_energy = {total_energy:.4f}")

# -----------------------------------------------------------------------------
# 5. Summary and plotting
# -----------------------------------------------------------------------------
print("\nSummary of predicted total energies:")
for name, energy in results.items():
    print(f"  {name}: {energy:.4f}")
best = min(results, key=results.get)
print(f"\nLowest-energy trajectory: {best} ({results[best]:.4f})")

trajectory_names = list(results.keys())
energy_values = [results[n] for n in trajectory_names]
length_values = [all_lengths[n] for n in trajectory_names]
energy_per_step = [e / l for e, l in zip(energy_values, length_values)]
energy_percentages = [(e / results[best] - 1) * 100 for e in energy_values]

plt.style.use('seaborn-v0_8-darkgrid')
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, axs = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Comparison of Energy Consumption for Five Trajectories', fontsize=20)

# Total energy comparison bar chart
axs[0, 0].bar(trajectory_names, energy_values, color=custom_colors)
axs[0, 0].set_title('Total Energy Consumption Comparison')
axs[0, 0].set_xlabel('Trajectory')
axs[0, 0].set_ylabel('Total Energy')
axs[0, 0].tick_params(axis='x', rotation=45)

max_energy = max(energy_values)
axs[0, 0].set_ylim(0, max_energy * 1.1) 

for i, v in enumerate(energy_values):
    axs[0, 0].text(i, v, f"{v:.3f}", ha='center', va='bottom')
best_idx = trajectory_names.index(best)

axs[0, 0].text(best_idx, energy_values[best_idx]+max(energy_values) * 0.1, 'Best', ha='center', va='bottom', color='red', fontweight='bold')

# Cumulative energy curves
for i, name in enumerate(trajectory_names):
    cum_e = all_cum_energies[name]
    norm_time = np.linspace(0, 1, len(cum_e))
    lw = 3 if name == best else 2
    ls = '-' if name == best else '--'
    axs[0, 1].plot(norm_time, cum_e,
                   label=f"{name}{' (lowest)' if name == best else ''} ({len(cum_e)} steps)",
                   color=custom_colors[i], linewidth=lw, linestyle=ls)
axs[0, 1].set_title('Cumulative Energy over Normalized Time')
axs[0, 1].set_xlabel('Normalized Time (0 → 1)')
axs[0, 1].set_ylabel('Cumulative Energy')
axs[0, 1].legend(loc='upper left')
axs[0, 1].grid(True)

# Average energy per step bar chart
axs[1, 0].bar(trajectory_names, energy_per_step, color=custom_colors)
axs[1, 0].set_title('Average Energy per Step')
axs[1, 0].set_xlabel('Trajectory')
axs[1, 0].set_ylabel('Energy per Step')
axs[1, 0].tick_params(axis='x', rotation=45)

max_step_energy = max(energy_per_step)
axs[1, 0].set_ylim(0, max_step_energy * 1.2)

for i, v in enumerate(energy_per_step):
    axs[1, 0].text(i, v, f"{v:.3f}", ha='center', va='bottom')
best_step_idx = energy_per_step.index(min(energy_per_step))
# pick an offset that's a few percent of the tallest bar
y0 = energy_per_step[best_step_idx]
y_offset = max(energy_per_step) * 0.1

axs[1, 0].text(best_step_idx,
               y0 + y_offset,
               'Most Efficient',
               ha='center',
               va='bottom',
               color='red',
               fontweight='bold',
               fontsize=12)
# Percentage decrease relative to  most regular way
direct_energy = results['direct']             # baseline
percent_reduction = []
for name, e in zip(trajectory_names, energy_values):
    if name == 'direct':
        percent_reduction.append(0.0)
    else:
        percent_reduction.append((direct_energy - e) / direct_energy * 100.0)

axs[1, 1].bar(trajectory_names, percent_reduction, color=custom_colors)
axs[1, 1].set_title('Energy Reduction vs Direct Method')
axs[1, 1].set_xlabel('Trajectory')
axs[1, 1].set_ylabel('Reduction (%)')
axs[1, 1].tick_params(axis='x', rotation=45)

# annotate bars
for i, reduction in enumerate(percent_reduction):
    if trajectory_names[i] == 'direct':
        label = 'Direct (baseline)'
        va = 'bottom'
    else:
        label = f"{reduction:.1f}% less"
        va = 'bottom'
    axs[1, 1].text(
        i,
        reduction + max(percent_reduction)*0.02,  # small vertical offset
        label,
        ha='center',
        va=va,
        color='green' if trajectory_names[i] != 'direct' else 'black',
        fontweight='bold'
    )

# adjust layout and save as before
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(pred_out_dir, "five_trajectory_comparison.png"), dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# 6. Instantaneous energy per step comparison on normalized time
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for i, name in enumerate(trajectory_names):
    preds = all_preds[name]
    # normalized time [0..1]
    norm_time = np.linspace(0, 1, len(preds))
    plt.plot(norm_time, preds,
             label=name,
             linewidth=2,
             linestyle='-' if name == best else '--',
             color=custom_colors[i])
plt.title('Instantaneous Energy Consumption vs Normalized Time')
plt.xlabel('Normalized Time (0 → 1)')
plt.ylabel('Energy per Step')
plt.legend(title='Trajectory', loc='upper right', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pred_out_dir, "instantaneous_energy_comparison.png"), dpi=300)
plt.show()

print("\nAll plots and results have been saved in the Results directory.")
