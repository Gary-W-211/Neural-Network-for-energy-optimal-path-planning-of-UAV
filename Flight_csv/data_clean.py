import os
import pandas as pd
import numpy as np

# Define input and output directories
input_dir = "./Flight_csv"
output_dir = "./Cleaned_csv"

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define field groups used for merging
group_A = ["rate_roll", "rate_pitch", "rate_yaw", "current", "voltage"]
group_B = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "roll", "pitch", "yaw"]

# Function to determine the group for each row based on non-null values
def determine_group(row):
    # If any of the Group B (position/velocity) fields is not null, label as "B"
    if row[group_B].notna().sum() > 0:
        return "B"
    # If any of the Group A (acceleration, current, voltage) fields is not null, label as "A"
    elif row[group_A].notna().sum() > 0:
        return "A"
    else:
        return "unknown"
    
#euler to rotation matrix
def euler_to_rotmat(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Multiply in ZYX order: R = R_z * R_y * R_x
    R = R_z.dot(R_y).dot(R_x)
    return R.flatten()

# Function to process and merge a single CSV file
def process_csv_file(file_path, output_folder):

    #Read CSV file
    df = pd.read_csv(file_path)
    
    # Separate into 2 groups based on fields
    df["group"] = df.apply(determine_group, axis=1)

    # Merge adjacent rows
    merged_rows = []
    i = 0
    n = len(df)
    while i < n:
        row1 = df.iloc[i] #including titles
        # If there is a next row, merge current row with the next one
        if i < n - 1:
            row2 = df.iloc[i+1]
            merged = {}
            # Compute the mean time of the two rows
            try:
                t1 = float(row1["time"])
                t2 = float(row2["time"])
                merged["time"] = (t1 + t2) / 2.0
            except Exception as e:
                # print("Time conversion error:", e)
                merged["time"] = row1["time"]
                
            # Merge Group A fields: if both rows have a value, take the average; otherwise take the non-null value.
            for col in group_A:
                val1 = row1.get(col, np.nan) #if no falue, just return nan
                val2 = row2.get(col, np.nan)
                if pd.notna(val1) and pd.notna(val2):
                    merged[col] = (val1 + val2) / 2.0
                elif pd.notna(val1):
                    merged[col] = val1
                elif pd.notna(val2):
                    merged[col] = val2
                else:
                    merged[col] = np.nan

            # Merge Group B fields
            for col in group_B:
                val1 = row1.get(col, np.nan)
                val2 = row2.get(col, np.nan)
                if pd.notna(val1) and pd.notna(val2):
                    merged[col] = (val1 + val2) / 2.0
                elif pd.notna(val1):
                    merged[col] = val1
                elif pd.notna(val2):
                    merged[col] = val2
                else:
                    merged[col] = np.nan

            # # For other columns (excluding time and group), take row1's value if available, else row2's value
            # other_cols = [col for col in df.columns if col not in (group_A + group_B + ["time", "group"])]
            # for col in other_cols:
            #     merged[col] = row1[col] if pd.notna(row1[col]) else row2[col]
            
            merged_rows.append(merged)
            i += 2 
        else:
            # If only one row is left, add it as is.
            merged_rows.append(df.iloc[i].to_dict())
            i += 1

    merged_df = pd.DataFrame(merged_rows)
    # print(merged_df.head())

    # Drop rows with missing core fields
    essential_columns = ["time"] + group_A + group_B
    merged_df = merged_df.dropna(subset=essential_columns)

    # Compute energy for each row
    
    # dt = merged_df["time"].diff().mean()
    # Fill NaN for the first row (if any) with dt, so that each row has a valid dt.
    # merged_df["dt"] = merged_df["time"].diff().fillna(dt)
    
    #assume that dt = 0.01
    # dt = 0.01
    # Compute energy for each row
    merged_df["energy"] = (merged_df["voltage"] * merged_df["current"]).abs()

    # Drop the 'group' column since it's no longer needed.
    if "group" in merged_df.columns:
        merged_df = merged_df.drop(columns=["group"])
        
    # Convert Euler angles (roll, pitch, yaw) to rotation matrix.
    if all(col in merged_df.columns for col in ["roll", "pitch", "yaw"]):
        rot_cols = ["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"]
        
        rot_mat = merged_df.apply(lambda row: pd.Series(euler_to_rotmat(row["roll"], row["pitch"], row["yaw"]), index=rot_cols), axis=1)
        # Drop the original Euler angle columns.
        merged_df = merged_df.drop(columns=["roll", "pitch", "yaw"])
        # Concatenate the rotation matrix columns.
        merged_df = pd.concat([merged_df, rot_mat], axis=1)
    
    # Reorder
    # Get all columns except 'current', 'voltage', and 'energy'
    cols = [col for col in merged_df.columns if col not in ['current', 'voltage', 'energy']]
    # New order: other columns + current + voltage + energy
    new_order = cols + ['current', 'voltage', 'energy']
    merged_df = merged_df[new_order]
    
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, file_name)
    merged_df.to_csv(output_file, index=False)
    # print(f"Cleaned data saved to {output_file}.\n")

# Iterate over all CSV files in the input directory and process them one by one.
n = 0
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)
        process_csv_file(file_path, output_dir)
        n += 1
        print(f"Finish {n} files")
