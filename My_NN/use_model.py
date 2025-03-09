import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # used for loading the saved scalers

###############################
# Define the GRU model architecture (must match training)
###############################
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

###############################
# Define the dataset class (same as training)
###############################
class UAVTimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Use all columns except the last ("energy") as features.
        seq_data = self.data.iloc[idx:idx+self.seq_length, :-1].values.astype(np.float32)
        # Target: last column ("energy")
        target = self.data.iloc[idx+self.seq_length, -1]
        return torch.tensor(seq_data), torch.tensor(target, dtype=torch.float32)

###############################
# Settings (must match training settings)
###############################
seq_length = 50
batch_size = 32
input_dim = 10  # (This will be automatically adjusted later if needed)
hidden_dim = 128
num_layers = 2
output_dim = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################
# Load the saved GRU model
###############################
model_path = "./My_NN/gru_model.pth"
gru_model = GRURegressor(input_dim, hidden_dim, num_layers, output_dim).to(device)
gru_model.load_state_dict(torch.load(model_path, map_location=device))
gru_model.eval()
print(f"Loaded model from {model_path}")

###############################
# Load raw test data (un-normalized)
###############################
# Specify your raw test CSV file here:
raw_test_csv = "./Test_data/Jan29_f03r.csv"  # Update this path as needed.
if not os.path.exists(raw_test_csv):
    raise ValueError(f"Raw test CSV file {raw_test_csv} does not exist.")
raw_test_df = pd.read_csv(raw_test_csv)
# Drop unwanted columns: "time", "current", "voltage"
raw_test_df = raw_test_df.drop(columns=["time", "current", "voltage"])
# print("Raw Test Data Sample:")
# print(raw_test_df.head())

###############################
# Normalize the raw test data using saved scalers
###############################
features_scaler = joblib.load("./My_NN/features_scaler.pkl")
target_scaler = joblib.load("./My_NN/target_scaler.pkl")

# Separate features and target from the raw test data.
features_test = raw_test_df.drop("energy", axis=1)
target_test = raw_test_df[["energy"]]

# Normalize using the loaded scalers.
features_test_norm = features_scaler.transform(features_test)
target_test_norm = target_scaler.transform(target_test)

# Reconstruct a normalized test DataFrame.
test_norm_df = pd.DataFrame(features_test_norm, columns=features_test.columns, index=raw_test_df.index)
test_norm_df["energy"] = target_test_norm

# Save the normalized test file for future reference.
normalized_test_csv = "./My_NN/test_data.csv"
test_norm_df.to_csv(normalized_test_csv, index=False)
print(f"Normalized test CSV saved to {normalized_test_csv}")

###############################
# Create the test DataLoader from normalized test file
###############################
test_dataset = UAVTimeSeriesDataset(csv_file=normalized_test_csv, seq_length=seq_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Test samples: {len(test_dataset)}")

###############################
# Evaluate the model on the test set
###############################
all_targets_norm = []
all_predictions_norm = []

with torch.no_grad():
    for sequences, targets in test_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        outputs = gru_model(sequences)
        all_targets_norm.extend(targets.cpu().numpy())
        all_predictions_norm.extend(outputs.cpu().numpy())

# Convert lists to numpy arrays and reshape for inverse transformation.
all_targets_norm = np.array(all_targets_norm).reshape(-1, 1)
all_predictions_norm = np.array(all_predictions_norm).reshape(-1, 1)

# Inverse transform to original scale.
actual_energy = target_scaler.inverse_transform(all_targets_norm).flatten()
predicted_energy = target_scaler.inverse_transform(all_predictions_norm).flatten()

# Compute evaluation metrics.
mae = np.mean(np.abs(actual_energy - predicted_energy))
print(f"Mean Absolute Error on original scale: {mae:.4f}")

total_actual_energy = np.sum(actual_energy)
total_predicted_energy = np.sum(predicted_energy)
print(f"Total Actual Energy Consumption (Test Set): {total_actual_energy:.4f}")
print(f"Total Predicted Energy Consumption (Test Set): {total_predicted_energy:.4f}")

###############################
# Plot a sample time-series view (first 500 samples)
###############################
time_axis = np.arange(len(actual_energy))
num_samples = 500
plt.figure(figsize=(10, 5))
plt.plot(time_axis[:num_samples], actual_energy[:num_samples], label="Actual", color="black", marker="o", markersize=4)
plt.plot(time_axis[:num_samples], predicted_energy[:num_samples], label="GRU Predictions", color="blue", marker="s", markersize=4)
plt.xlabel("Time Index")
plt.ylabel("Energy Consumption")
plt.title("Time-series View (First 500 Samples)")
plt.legend()
plt.grid(True)
plt.show()

###############################
# Plot error distribution (optional)
###############################
gru_errors = predicted_energy - actual_energy
plt.figure(figsize=(8, 5))
sns.histplot(gru_errors, color="blue", label="GRU Errors", kde=True, alpha=0.5)
plt.xlabel("Error (Predicted - Actual)")
plt.ylabel("Count")
plt.title("Error Distribution")
plt.legend()
plt.show()
