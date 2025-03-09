import pandas as pd
import os

input_csv = "./Trajectories/trajectories(15).csv"
output_dir = "./Trajectories/Candidates"

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the CSV file
df = pd.read_csv(input_csv)
df["dt"] = 0.01

# Define the desired column order
desired_order = [
    "trajID",    # trajectory id
    "t",         # time
    "acc_x", "acc_y", "acc_z",   # acceleration
    "pos_x", "pos_y", "pos_z",   # position
    "vel_x", "vel_y", "vel_z",   # velocity
    "dt"         # delta time
]

# Check for missing required columns
missing_cols = [col for col in desired_order if col not in df.columns]
if missing_cols:
    raise ValueError(f"The following required columns are missing: {missing_cols}")

# Get all unique trajectory IDs
unique_ids = df["trajID"].unique()

# Loop over each trajectory id, filter rows, reorder columns, and save to a separate CSV file.
for traj_id in unique_ids:
    df_traj = df[df["trajID"] == traj_id]
    df_traj = df_traj[desired_order]  # reorder columns
    output_csv = os.path.join(output_dir, f"Candidate_{traj_id}.csv")
    df_traj = df_traj.drop(columns= ["t", "trajID"], errors= "ignore")
    df_traj.to_csv(output_csv, index=False)
    print(f"Filtered data for trajID={traj_id} saved to: {output_csv}")
