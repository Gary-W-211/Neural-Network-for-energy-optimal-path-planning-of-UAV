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

# Model configuration (统一参数)
model_config = {
    'gru': {
        'path': os.path.expanduser("~/Capstone/Neural/NN/My_NN/models/gru_model.pth"),
        'class': 'GRU',
        'params': {'input_dim': 18, 'hidden_dim': 128, 'num_layers': 2, 'output_dim': 1}
    },
    'lstm': {
        'path': os.path.expanduser("~/Capstone/Neural/NN/My_NN/models/lstm_model.pth"),
        'class': 'LSTM', 
        'params': {'input_dim': 18, 'hidden_dim': 128, 'num_layers': 2, 'output_dim': 1}
    },
    'tcn': {
        'path': os.path.expanduser("~/Capstone/Neural/NN/My_NN/models/tcn_model.pth"),
        'class': 'TCN',
        'params': {'input_dim': 18, 'output_dim': 1, 'num_channels': 128, 
                  'kernel_size': 3, 'dropout': 0.1}
    }
}

# Trajectory CSVs to process
trajectory_files = [
    "direct.csv",
    "speed.csv",
    "gru.csv",
    "lstm.csv",
    "tcn.csv",
]

# Network parameters
seq_length = 50
batch_size = 32
dt = 0.01  # time interval per step

# -----------------------------------------------------------------------------
# 2. Model Definitions (统一输入输出维度)
# -----------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TCNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        # 保持与GRU/LSTM相同的隐藏层维度(128)
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, num_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(num_channels, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq_length)
        tcn_out = self.tcn(x)
        return self.fc(tcn_out[:, :, -1])

# -----------------------------------------------------------------------------
# 3. Data Loading and Model Initialization
# -----------------------------------------------------------------------------
class UAVTimeSeriesTestDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data.iloc[idx : idx + self.seq_length, :].values.astype(np.float32)
        return torch.tensor(seq)

def load_models(device):
    models = {}
    for model_name, config in model_config.items():
        if config['class'] == 'GRU':
            model = GRUModel(**config['params']).to(device)
        elif config['class'] == 'LSTM':
            model = LSTMModel(**config['params']).to(device)
        elif config['class'] == 'TCN':
            model = TCNModel(**config['params']).to(device)
        
        model.load_state_dict(torch.load(config['path'], map_location=device))
        model.eval()
        models[model_name] = model
    return models

# Initialize
features_scaler = joblib.load(features_scaler_path)
target_scaler = joblib.load(target_scaler_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_models(device)

# -----------------------------------------------------------------------------
# 4. Enhanced Multi-Model Prediction
# -----------------------------------------------------------------------------
results = {}
all_preds = {}
model_breakdown = {}

for fname in trajectory_files:
    file_path = os.path.join(results_dir, fname)
    print(f"\nProcessing {file_path}...")
    
    # Load and preprocess
    raw_df = pd.read_csv(file_path)
    if raw_df.columns[0].lower() in ("time", "timestamp"):
        raw_df = raw_df.drop(raw_df.columns[0], axis=1)
    
    # Verify input dimension matches model expectation (18)
    assert raw_df.shape[1] == 18, f"Input dimension mismatch. Expected 18, got {raw_df.shape[1]}"
    
    num_steps = len(raw_df)
    traj_name = os.path.splitext(fname)[0]
    
    # Normalize
    feats_norm = features_scaler.transform(raw_df.values)
    norm_df = pd.DataFrame(feats_norm, columns=raw_df.columns)
    norm_csv = os.path.join(norm_out_dir, f"normalized_{fname}")
    norm_df.to_csv(norm_csv, index=False)
    
    # Prepare data loader
    dataset = UAVTimeSeriesTestDataset(norm_csv, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Run predictions with all models
    model_preds = {}
    for model_name, model in models.items():
        print(f"  {model_name.upper()} prediction...")
        preds_norm = []
        
        with torch.no_grad():
            for seq in loader:
                seq = seq.to(device)
                if isinstance(model, TCNModel):
                    seq = seq.permute(0, 2, 1)  # Adjust for TCN
                output = model(seq)
                preds_norm.extend(output.cpu().numpy().flatten())
        
        # Post-process
        preds = target_scaler.inverse_transform(
            np.array(preds_norm).reshape(-1, 1)).flatten()
        padded = np.pad(preds, (seq_length, 0), 'edge')[:num_steps]
        model_preds[model_name] = np.abs(padded) * dt
    
    # Calculate average prediction
    avg_pred = np.mean(list(model_preds.values()), axis=0)
    
    # Store results
    model_breakdown[traj_name] = model_preds
    all_preds[traj_name] = avg_pred
    total_energy = avg_pred.sum()
    results[traj_name] = total_energy
    
    # Save detailed predictions
    output_data = {
        "step": np.arange(num_steps),
        "average_energy": avg_pred,
        "cumulative_energy": np.cumsum(avg_pred)
    }
    for mname, pred in model_preds.items():
        output_data[f"{mname}_energy"] = pred
        output_data[f"{mname}_cumulative"] = np.cumsum(pred)
    
    pd.DataFrame(output_data).to_csv(
        os.path.join(pred_out_dir, f"detailed_{traj_name}.csv"), index=False)
    
    print(f"  {traj_name} total energy (average): {total_energy:.4f}")

# -----------------------------------------------------------------------------
# 5. Enhanced Visualization
# -----------------------------------------------------------------------------
def plot_results():
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'gru': '#1f77b4', 'lstm': '#ff7f0e', 'tcn': '#2ca02c'}
    
    # 1. Model Consistency Plot
    plt.figure(figsize=(12, 6))
    for traj_name in trajectory_files[:3]:  # Show first 3 for clarity
        tname = os.path.splitext(traj_name)[0]
        time_axis = np.linspace(0, 1, len(all_preds[tname]))
        
        # Plot individual models
        for mname in models:
            plt.plot(time_axis, model_breakdown[tname][mname], 
                    '--', alpha=0.6, color=colors[mname],
                    label=f'{mname.upper()}' if traj_name==trajectory_files[0] else "")
        
        # Plot average
        plt.plot(time_axis, all_preds[tname], 
                '-', linewidth=2, color='k',
                label='Average' if traj_name==trajectory_files[0] else "")
    
    plt.title('Model Predictions Comparison (First 3 Trajectories)')
    plt.xlabel('Normalized Time')
    plt.ylabel('Instantaneous Power (W)')
    plt.legend()
    plt.savefig(os.path.join(pred_out_dir, "model_consistency.png"), dpi=300)
    
    # 2. Energy Comparison Bar Plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results))
    width = 0.25
    
    # Plot individual model results
    for i, mname in enumerate(models):
        model_energies = [
            model_breakdown[os.path.splitext(f)[0]][mname].sum() 
            for f in trajectory_files
        ]
        plt.bar(x + i*width, model_energies, width, 
               color=colors[mname], label=mname.upper())
    
    # Plot average results
    plt.bar(x + len(models)*width, list(results.values()), width,
           color='#d62728', label='Average')
    
    plt.xlabel('Trajectory')
    plt.ylabel('Total Energy (J)')
    plt.title('Energy Consumption by Model')
    plt.xticks(x + width*1.5, [os.path.splitext(f)[0] for f in trajectory_files])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pred_out_dir, "energy_comparison.png"), dpi=300)
    
    # 3. Cumulative Energy Plot
    plt.figure(figsize=(12, 6))
    for traj_name in trajectory_files:
        tname = os.path.splitext(traj_name)[0]
        time_axis = np.linspace(0, 1, len(all_preds[tname]))
        plt.plot(time_axis, np.cumsum(all_preds[tname]), 
                label=tname, linewidth=2)
    
    plt.title('Cumulative Energy Consumption (Model Average)')
    plt.xlabel('Normalized Time')
    plt.ylabel('Cumulative Energy (J)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(pred_out_dir, "cumulative_energy.png"), dpi=300)

plot_results()
print("\nAnalysis complete. Results saved to:", pred_out_dir)