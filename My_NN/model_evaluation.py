import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Define the GRU model architecture (must match training)
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        # Use the output from the last time step
        gru_out = gru_out[:, -1, :]
        output = self.fc(gru_out)
        return output

# Define the test dataset class (returns feature sequences and corresponding target)
class UAVTimeSeriesTestDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        #Features
        seq_data = self.data.iloc[idx:idx+self.seq_length, :-1].values.astype(np.float32)
        # Target
        target = self.data.iloc[idx+self.seq_length, -1]
        return torch.tensor(seq_data), torch.tensor(target, dtype=torch.float32)

# Settings (must match training settings)
seq_length = 50
batch_size = 32

# Determine input dimension from the normalized training CSV so it matches the model.
train_norm_csv = "./My_NN/data_for_train/train_data.csv"
input_dim = pd.read_csv(train_norm_csv).shape[1] - 1  # Subtract target column if present

hidden_dim = 128
num_layers = 2
output_dim = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the saved GRU model
model_path = "./My_NN/models/gru_model.pth"
gru_model = GRURegressor(input_dim, hidden_dim, num_layers, output_dim).to(device)
gru_model.load_state_dict(torch.load(model_path, map_location=device))
gru_model.eval()

# Load normalized test CSV file (test data must be preprocessed as in training)
test_csv = "./My_NN/data_for_train/test_data.csv"
test_dataset = UAVTimeSeriesTestDataset(csv_file=test_csv, seq_length=seq_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Number of test samples: {len(test_dataset)}")

# Load the target scaler saved during training (for inverse normalization)
target_scaler = joblib.load("./My_NN/scalers/target_scaler.pkl")

# Evaluate the model on the test set and collect normalized predictions and targets
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

# Compute evaluation metrics
mae = mean_absolute_error(all_targets_norm, all_predictions_norm)
mse = mean_squared_error(all_targets_norm, all_predictions_norm)
rmse = np.sqrt(mean_squared_error(all_targets_norm, all_predictions_norm))
r2 = r2_score(all_targets_norm, all_predictions_norm)

# Inverse transform to original scale using the saved target scaler.
actual_energy = target_scaler.inverse_transform(all_targets_norm).flatten()
predicted_energy = target_scaler.inverse_transform(all_predictions_norm).flatten()

print(f"Mean Absolute Error (MAE) on original scale: {mae:.4f}")
print(f"Mean Squared Error (MSE) on original scale: {mse:.4f}")
print(f"Root Mean Squared Error (RMSE) on original scale: {rmse:.4f}")
print(f"R² Score on original scale: {r2:.4f}")

# Compute and print total energy consumption on the test set (original scale)
#assume that dt is 0.01
dt = 0.01
total_actual_energy = np.sum(actual_energy*dt)
total_predicted_energy = np.sum(predicted_energy*dt)
print(f"Total Actual Energy Consumption (Test Set): {total_actual_energy:.4f}")
print(f"Total Predicted Energy Consumption (Test Set): {total_predicted_energy:.4f}")

# Plot time series comparison between actual and predicted energy
time_axis = np.arange(len(actual_energy))
plt.figure(figsize=(10, 5))
plt.plot(time_axis, actual_energy, label="Actual Energy", color="black", marker="o", markersize=4)
plt.plot(time_axis, predicted_energy, label="Predicted Energy", color="blue", marker="s", markersize=4)
plt.xlabel("Time Index")
plt.ylabel("Energy Consumption")
plt.title("Comparison of Actual and Predicted Energy (Test Set)")
plt.legend()
plt.grid(True)
plt.show()

# Plot error distribution
gru_errors = predicted_energy - actual_energy
plt.figure(figsize=(8, 5))
sns.histplot(gru_errors, color="blue", label="Prediction Error", kde=True, alpha=0.5)
plt.xlabel("Error (Predicted - Actual)")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors")
plt.legend()
plt.show()
