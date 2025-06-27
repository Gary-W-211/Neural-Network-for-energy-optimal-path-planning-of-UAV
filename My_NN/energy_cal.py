#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler

class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_length, input_dim)
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]      # last time-step
        return self.fc(out)

def main():
    parser = argparse.ArgumentParser(
        description="Compute total energy consumption using a trained GRU model"
    )
    parser.add_argument(
        "--csv", type=str,
        default="~/Capstone/catkin_ws_79/src/hiperlab_rostools/src/MEng79/Results/lstm_vehicle_110_trajectory.csv",
        help="Path to the CSV file containing the trajectory data"
    )
    parser.add_argument(
        "--model", type=str,
        default="~/Capstone/Neural/NN/My_NN/models/gru_model.pth",
        help="Path to the trained GRU .pth model"
    )
    parser.add_argument(
        "--feat_scaler", type=str,
        default="~/Capstone/Neural/NN/My_NN/scalers/features_scaler.pkl",
        help="Path to the saved feature scaler (.pkl)"
    )
    parser.add_argument(
        "--tgt_scaler", type=str,
        default="~/Capstone/Neural/NN/My_NN/scalers/target_scaler.pkl",
        help="Path to the saved target scaler (.pkl)"
    )
    parser.add_argument(
        "--seq_length", type=int, default=50,
        help="Sequence length (window size) for the GRU input"
    )
    parser.add_argument(
        "--dt", type=float, default=0.01,
        help="Time step delta for energy integration"
    )
    args = parser.parse_args()

    # Expand and load
    csv_path         = os.path.expanduser(args.csv)
    model_path       = os.path.expanduser(args.model)
    feat_scaler_path = os.path.expanduser(args.feat_scaler)
    tgt_scaler_path  = os.path.expanduser(args.tgt_scaler)

    features_scaler = joblib.load(feat_scaler_path)
    target_scaler   = joblib.load(tgt_scaler_path)

    # Read CSV and drop the first (time) column
    df = pd.read_csv(csv_path)
    df = df.drop(df.columns[0], axis=1)
    features = df.values.astype(np.float32)
    features_norm = features_scaler.transform(features)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU.")

    # Build and load model
    input_dim  = features.shape[1]
    hidden_dim = 128
    num_layers = 2
    output_dim = 1

    model = GRURegressor(input_dim, hidden_dim, num_layers, output_dim).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Sliding-window predictions
    preds_norm = []
    with torch.no_grad():
        for i in range(len(features_norm) - args.seq_length):
            window = features_norm[i : i + args.seq_length]
            tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            preds_norm.append(model(tensor).item())

    # Inverse-transform & integrate
    preds_norm = np.array(preds_norm).reshape(-1, 1)
    preds = target_scaler.inverse_transform(preds_norm).flatten()
    total_energy = np.sum(preds * args.dt)

    print(f"Predicted total energy consumption: {total_energy:.4f}")

if __name__ == "__main__":
    main()
