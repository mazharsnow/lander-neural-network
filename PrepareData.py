# PrepareData.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json

RAW_FILE = "ce889_dataCollection.csv"
SCALED_ALL_FILE = "lander_data_scaled.csv"
TRAIN_FILE = "lander_train.csv"
VAL_FILE = "lander_val.csv"
NORM_FILE = "normalization.json"


def main():
    # --- 1) Load raw data ---
    # With ALL_DATA = FALSE your columns are:
    # [x_target, y_target, vel_y, vel_x]
    df = pd.read_csv(RAW_FILE, header=None)

    if df.shape[1] < 4:
        print("Expected 4 columns in ce889_dataCollection.csv, found", df.shape[1])
        return

    df = df.iloc[:, :4]
    df.columns = ["x_target", "y_target", "vel_y", "vel_x"]

    print("Loaded", len(df), "rows from", RAW_FILE)

    # --- 2) Cleaning: remove NaN / inf and obvious nonsense ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print("Removed", before - len(df), "rows with NaN/inf")

    # (Optional) Clip extreme outliers (1% / 99%)
    df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)

    # --- 3) Normalisation (0–1) ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_array = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_array, columns=df.columns)

    # Save full scaled dataset
    df_scaled.to_csv(SCALED_ALL_FILE, index=False)
    print("Saved scaled data to", SCALED_ALL_FILE)

    # --- 4) Train/validation split ---
    train_df, val_df = train_test_split(
        df_scaled, test_size=0.2, random_state=42, shuffle=True
    )
    print("Train rows:", len(train_df), " | Val rows:", len(val_df))

    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    print("Saved train data to", TRAIN_FILE)
    print("Saved validation data to", VAL_FILE)

    # --- 5) Save min/max for use in the game (for scaling at runtime) ---
    mins = scaler.data_min_.tolist()
    maxs = scaler.data_max_.tolist()

    # Order is [x_target, y_target, vel_y, vel_x]
    norm_info = {
        "columns": ["x_target", "y_target", "vel_y", "vel_x"],
        "min": mins,
        "max": maxs,
    }

    with open(NORM_FILE, "w") as f:
        json.dump(norm_info, f)

    print("Saved normalisation parameters to", NORM_FILE)


if __name__ == "__main__":
    main()
