import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ================================================
# STEP 1: READ THE CSV FILE AND SELECT COLUMNS
# ================================================
# Read the raw data
df = pd.read_csv("12.csv")

# Select the required columns:
# - Columns 4-6: attitude (index 4,5,6)
# - Columns 10-11: current and voltage (index 10,11)
# - Columns 46-49: motor RPM
# - Columns 60-63: additional features
# - Columns 66-68: position
columns_needed = list(range(4, 7)) + list(range(10, 12)) + list(range(46, 50)) + list(range(60, 64)) + list(range(66, 69))
df = df.iloc[:, columns_needed]
print("Initial Data Sample:")
print(df.head())

# ================================================
# STEP 2: DUPLICATE REMOVAL
# ================================================
df = df.drop_duplicates()
print(f"Data shape after duplicate removal: {df.shape}")

# ================================================
# STEP 3: MISSING VALUE HANDLING
# ================================================
# Check for missing values and fill them with the median of the column (or you could drop them)
if df.isnull().values.any():
    df = df.fillna(df.median())
print("Missing values handled (if any were present).")

# ================================================
# STEP 4: FEATURE ENGINEERING: COMPUTE ENERGY
# ================================================
# Compute energy consumption (energy = current * voltage)
df["energy"] = df.iloc[:, 3] * df.iloc[:, 4]  # Note: after selection, index 3 and 4 correspond to current and voltage

# Remove the current and voltage columns since energy has been computed
df.drop(columns=[df.columns[3], df.columns[4]], inplace=True)  
# (Dropping twice because after dropping one, the columns shift)
print("Energy column computed and current/voltage columns removed.")
print(df.head())

# ================================================
# STEP 5: OUTLIER DETECTION AND REMOVAL
# ================================================
# Define a function to remove outliers using the IQR method for a given column
def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Remove outliers from every numeric column
for col in df.columns:
    df = remove_outliers_iqr(df, col)
print(f"Data shape after outlier removal: {df.shape}")

# ================================================
# STEP 6: NORMALIZATION
# ================================================
# Separate features and target
features = df.drop("energy", axis=1)
target = df[["energy"]]

# Normalize features using StandardScaler
scaler_features = StandardScaler()
features_scaled = scaler_features.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns) #features.columns (column name)

# Normalize target variable using StandardScaler
scaler_target = StandardScaler()
target_scaled = scaler_target.fit_transform(target)
target_scaled = pd.DataFrame(target_scaled, columns=["energy"])

# Combine the normalized features and target
df_normalized = pd.concat([features_scaled, target_scaled], axis=1)
print("Normalization complete.")
print(df_normalized.head())

# ================================================
# STEP 7: SAVE THE CLEANED DATA
# ================================================
df_normalized.to_csv("fully_cleaned_uav_data.csv", index=False)
print("Cleaned data saved to fully_cleaned_uav_data.csv.")

# ================================================
# STEP 8: OPTIONAL VISUALIZATION OF DATA DISTRIBUTION
# ================================================
plt.figure(figsize=(12, 8))
df_normalized.hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()
