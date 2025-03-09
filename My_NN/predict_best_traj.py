import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Define the GRU model architecture (must match training)
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]  # Use the output from the last time step
        output = self.fc(gru_out)
        return output

# Define a test dataset class that returns only feature sequences 
class UAVTimeSeriesTestDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        
        seq_data = self.data.iloc[idx:idx+self.seq_length, :].values.astype(np.float32)
        return torch.tensor(seq_data)

# Settings (must match training settings)
seq_length = 50
batch_size = 32

# Determine input dimension from normalized training CSV so it matches the model.
train_norm_csv = "./My_NN/data_for_train/train_data.csv"
input_dim = pd.read_csv(train_norm_csv).shape[1] - 1

print(f"Determined input dimension: {input_dim}")

hidden_dim = 128
num_layers = 2
output_dim = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the GRU model
model_path = "./My_NN/models/gru_model.pth"
gru_model = GRURegressor(input_dim, hidden_dim, num_layers, output_dim).to(device)
gru_model.load_state_dict(torch.load(model_path, map_location=device))
gru_model.eval()
print(f"Loaded model from {model_path}")

# Directory containing test trajectory CSV files
test_folder = "./Trajectories/Candidates"
trajectory_files = [f for f in os.listdir(test_folder) if f.endswith(".csv")]

# Load saved scalers (assumed to have been saved during training)
features_scaler = joblib.load("./My_NN/scalers/features_scaler.pkl")
target_scaler = joblib.load("./My_NN/scalers/target_scaler.pkl")

# Loop through each trajectory file, process and predict total energy consumption.
results = {}

for file in trajectory_files:
    file_path = os.path.join(test_folder, file)
    print(f"\nProcessing trajectory file: {file_path}")
    
    # Load raw trajectory CSV (un-normalized)
    raw_df = pd.read_csv(file_path)
    
    # Drop unwanted columns. 
    # raw_df = raw_df.drop(columns=["t", "trajID"], errors='ignore')
    
    # Normalize raw data using the saved features scaler.
    features = raw_df  # All remaining columns are features 
    features_norm = features_scaler.transform(features)
    
    # Reconstruct normalized DataFrame.
    norm_df = pd.DataFrame(features_norm, columns=features.columns, index=raw_df.index)
    
    # Save the normalized file.
    norm_file = os.path.join("./My_NN/Normalized", f"normalized_{file}")
    norm_df.to_csv(norm_file, index=False)
    print(f"Normalized trajectory saved to: {norm_file}")
    
    # Create a test dataset and DataLoader from the normalized file.
    test_dataset = UAVTimeSeriesTestDataset(csv_file=norm_file, seq_length=seq_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print(f"Test samples for {file}: {len(test_dataset)}")
    
    # Run the model on the trajectory to get predictions.
    all_predictions_norm = []
    with torch.no_grad():
        for sequences in test_loader:
            sequences = sequences.to(device)
            outputs = gru_model(sequences)
            all_predictions_norm.extend(outputs.cpu().numpy())
    
    # Convert predictions list to numpy array and reshape.
    all_predictions_norm = np.array(all_predictions_norm).reshape(-1, 1)
    
    # Inverse transform predictions to original scale.
    predicted_energy = target_scaler.inverse_transform(all_predictions_norm).flatten()
    
    # Compute total predicted energy consumption for this trajectory.
    total_energy = np.sum(predicted_energy)
    results[os.path.splitext(file)[0]] = total_energy
    print(f"Total Predicted Energy Consumption for {file}: {total_energy:.4f}")

# Print summary of results for all trajectories.
print("\nSummary of Predicted Energy Consumption for each trajectory:")
for traj_file, energy in results.items():
    print(f"{traj_file}: {energy:.4f}")
    
# Find and print the trajectory with the minimum predicted energy consumption.
min_traj = min(results, key=results.get)
min_energy = results[min_traj]
print(f"\nTrajectory with minimum predicted energy consumption: {min_traj} with {min_energy:.4f}")
    
# Plot a bar chart for total predicted energy consumption per trajectory.
import matplotlib.pyplot as plt

# Create lists from the results dictionary.
trajectory_names = list(results.keys())
energy_values = list(results.values())

# plt.figure(figsize=(18, 6))
# plt.bar(trajectory_names, energy_values, color="lightgreen")
# plt.xlabel("Trajectories")
# plt.ylabel("Total Predicted Energy Consumption")
# plt.title("Predicted Energy Consumption per Trajectory")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

# min_energy = min(energy_values)


colors = ['green' if energy == min_energy else 'lightgreen' for energy in energy_values]

plt.figure(figsize=(18, 6))
plt.bar(trajectory_names, energy_values, color=colors)
#
bars = plt.bar(trajectory_names, energy_values, color=colors)

plt.xlabel("Trajectories")
plt.ylabel("Total Predicted Energy Consumption")
plt.title("Predicted Energy Consumption per Trajectory")
plt.xticks(rotation=45, ha="right")

#add the words on the top
min_idx = energy_values.index(min_energy)
bar = bars[min_idx]
plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "Lowest",
         ha="center", va="bottom", color="red", fontsize=12)

plt.tight_layout()
plt.show()

# Plot cumulative predicted energy for one example
if trajectory_files:
    example_file = trajectory_files[-1]
    print(f"\nPlotting cumulative predicted energy for {example_file}")
    cumulative_energy = np.cumsum(predicted_energy)
    time_axis = np.arange(len(cumulative_energy))
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, cumulative_energy, label="Cumulative Predicted Energy", color="blue", marker="o", markersize=3)
    plt.xlabel("Time Index")
    plt.ylabel("Cumulative Energy Consumption")
    plt.title("Cumulative Predicted Energy Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()
