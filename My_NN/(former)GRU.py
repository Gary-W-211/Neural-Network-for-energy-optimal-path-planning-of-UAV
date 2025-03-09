import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# STEP 1: Read the Cleaned CSV File
# -------------------------------
df = pd.read_csv("./Cleaned_csv/Feb4_f01r.csv")
print("Initial Cleaned Data Sample:")
print(df.head())

# -------------------------------
# STEP 2: Split Data into Train (80%) and Test (20%) Sets
# -------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
print(f"Training set size: {len(train_df)}, Testing set size: {len(test_df)}")

# -------------------------------
# STEP 3: Normalize the Data and Save Normalized CSV Files
# -------------------------------
# Assume that the target variable is "energy" and all other columns are features.
features_train = train_df.drop("energy", axis=1)
target_train = train_df[["energy"]]
features_test = test_df.drop("energy", axis=1)
target_test = test_df[["energy"]]

# Create separate scalers for features and target.
features_scaler = StandardScaler()
target_scaler = StandardScaler()

# Fit the scalers on the training data only.
features_scaler.fit(features_train)
target_scaler.fit(target_train)

# Transform both training and test data.
features_train_norm = features_scaler.transform(features_train)
target_train_norm = target_scaler.transform(target_train)
features_test_norm = features_scaler.transform(features_test)
target_test_norm = target_scaler.transform(target_test)

# Reconstruct DataFrames with normalized values.
train_norm_df = pd.DataFrame(features_train_norm, columns=features_train.columns, index=train_df.index)
train_norm_df["energy"] = target_train_norm
test_norm_df = pd.DataFrame(features_test_norm, columns=features_test.columns, index=test_df.index)
test_norm_df["energy"] = target_test_norm

# Save normalized CSV files for NN training/testing.
os.makedirs("./My_NN", exist_ok=True)
train_norm_df.to_csv("./My_NN/train_data.csv", index=False)
test_norm_df.to_csv("./My_NN/test_data.csv", index=False)
print("Normalized train and test CSV files saved.")

# -------------------------------
# STEP 4: Define the Dataset Class for Time Series
# -------------------------------
class UAVTimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length):
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Features: all columns except the last (target "energy")
        seq_data = self.data.iloc[idx:idx+self.seq_length, :-1].values.astype(np.float32)
        # Target: last column (energy)
        target = self.data.iloc[idx+self.seq_length, -1]
        return torch.tensor(seq_data), torch.tensor(target, dtype=torch.float32)

# -------------------------------
# STEP 5: Create Data Loaders
# -------------------------------
seq_length = 50
batch_size = 32

train_dataset = UAVTimeSeriesDataset(csv_file="./My_NN/train_data.csv", seq_length=seq_length)
test_dataset = UAVTimeSeriesDataset(csv_file="./My_NN/test_data.csv", seq_length=seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

# -------------------------------
# STEP 6: Define the GRU Model
# -------------------------------
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_length, input_dim)
        gru_out, _ = self.gru(x)
        # Use the output from the last time step.
        gru_out = gru_out[:, -1, :]
        output = self.fc(gru_out)
        return output

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set parameters.
# input_dim: number of features (all columns except "energy")
input_dim = train_norm_df.shape[1] - 1
hidden_dim = 128
num_layers = 2
output_dim = 1

gru_model = GRURegressor(input_dim, hidden_dim, num_layers, output_dim).to(device)

# -------------------------------
# STEP 7: Define Loss Function and Optimizer
# -------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)

# -------------------------------
# STEP 8: Training Loop
# -------------------------------
num_epochs = 50
gru_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = gru_model(sequences)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    gru_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], GRU Loss: {epoch_loss:.4f}")

print("Training completed!")
gru_model.eval()

# -------------------------------
# STEP 9: Evaluation on Test Set (with Inverse Normalization)
# -------------------------------
test_loss = 0.0
all_targets_norm = []
all_predictions_norm = []

with torch.no_grad():
    for sequences, targets in test_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        outputs = gru_model(sequences)
        loss = criterion(outputs.squeeze(), targets)
        test_loss += loss.item()
        all_targets_norm.extend(targets.cpu().numpy())
        all_predictions_norm.extend(outputs.cpu().numpy())

test_loss /= len(test_loader)
print(f"Normalized GRU Test Loss (MSE): {test_loss:.4f}")

# Inverse-transform the normalized energy predictions and targets to original scale.
all_targets_norm = np.array(all_targets_norm).reshape(-1, 1)
all_predictions_norm = np.array(all_predictions_norm).reshape(-1, 1)

actual_energy = target_scaler.inverse_transform(all_targets_norm).flatten()
predicted_energy = target_scaler.inverse_transform(all_predictions_norm).flatten()

# Compute error metrics on original scale (e.g., Mean Absolute Error)
mae = np.mean(np.abs(actual_energy - predicted_energy))
print(f"Mean Absolute Error on original scale: {mae:.4f}")

# Print the total energy consumption for the test set (actual and predicted)
total_actual_energy = np.sum(actual_energy)
total_predicted_energy = np.sum(predicted_energy)
print(f"Total Actual Energy Consumption (Test Set): {total_actual_energy:.4f}")
print(f"Total Predicted Energy Consumption (Test Set): {total_predicted_energy:.4f}")

# -------------------------------
# STEP 10: Plot Training Loss (Optional)
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), gru_losses, label="GRU Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("GRU Training Loss")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# STEP 11: Additional Plots
# -------------------------------
# Plot error distribution using seaborn
gru_errors = predicted_energy - actual_energy  # Compute errors

plt.figure(figsize=(8, 5))
sns.histplot(gru_errors, color="blue", label="GRU Errors", kde=True, alpha=0.5)
plt.xlabel("Error (Predicted - Actual)")
plt.ylabel("Count")
plt.title("Error Distribution")
plt.legend()
plt.show()

# Plot time-series view for the first 500 samples
time_axis = np.arange(len(actual_energy))  # Create a time index
num_samples = 500  # Only take the first 500 samples
time_axis_subset = time_axis[:num_samples]
actual_subset = actual_energy[:num_samples]
gru_subset = predicted_energy[:num_samples]

plt.figure(figsize=(10, 5))
plt.plot(time_axis_subset, actual_subset, label="Actual", color="black", marker="o", markersize=4)
plt.plot(time_axis_subset, gru_subset, label="GRU", color="blue", marker="s", markersize=4)
plt.xlabel("Time Index")
plt.ylabel("Energy Consumption")
plt.title("Time-series View (First 500 Samples)")
plt.legend()
plt.grid(True)
plt.show()
