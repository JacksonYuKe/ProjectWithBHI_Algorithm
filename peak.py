
import pandas as pd
import numpy as np
import glob
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ✅ Define directory containing the weekly CSV files
DATA_DIR = "/Users/jacson/Desktop/BHI/Decrypted_Files/filtered_EV_weekly_csv_files"

# ✅ Get all CSV files that match the pattern "week_2023-*.csv"
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "week_2023-*.csv")))

# ✅ Ensure there are files to process
if len(file_paths) == 0:
    print("❌ No CSV files found in the directory!")
    exit()

# ✅ Load all weekly data
def load_all_data(file_paths):
    all_dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path, engine="openpyxl")
            df.columns = df.columns.str.strip()
            all_dfs.append(df)
            print(f"✅ Loaded: {file_path}")
        except Exception as e:
            print(f"❌ Error loading file {file_path}: {e}")

    # ✅ Combine all weeks into a single DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

# ✅ Transform data & reduce data size
def transform_data(df):
    usage_columns = [f"R{i}" for i in range(1, 25)]
    df_long = df.melt(id_vars=["YYYYMMDD", "LOCATION"], value_vars=usage_columns,
                      var_name="Hour", value_name="Usage")
    df_long["Hour"] = df_long["Hour"].str.extract("(\d+)").astype(int)

    # ✅ 转换 `Usage` 列为数值类型，并删除 `NaN`
    df_long["Usage"] = pd.to_numeric(df_long["Usage"], errors="coerce")
    df_long = df_long.dropna(subset=["Usage"])

    # ✅ 按 `Hour` 计算均值，减少数据点
    df_grouped = df_long.groupby(["Hour"])["Usage"].mean().reset_index()
    return df_grouped


# ✅ Apply MiniBatchKMeans Clustering
def classify_peak_mini_batch(df, n_clusters=3):
    scaler = MinMaxScaler()
    df["Usage_Norm"] = scaler.fit_transform(df[["Usage"]])

    X = df[["Usage_Norm"]].values.reshape(-1, 1)
    clustering = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
    df["Cluster"] = clustering.fit_predict(X)

    # ✅ Debug: Print cluster counts
    print("Cluster Distribution:", df["Cluster"].value_counts())

    # ✅ Assign categories based on average consumption per cluster
    cluster_means = df.groupby("Cluster")["Usage"].mean().sort_values()
    labels = {cluster_means.index[0]: "Off-Peak",
              cluster_means.index[1]: "Mid-Peak",
              cluster_means.index[2]: "On-Peak"}
    df["Peak_Period"] = df["Cluster"].map(labels)

    return df

# ✅ Print peak hours
def print_peak_hours(df):
    off_peak_hours = sorted(df[df["Peak_Period"] == "Off-Peak"]["Hour"].tolist())
    mid_peak_hours = sorted(df[df["Peak_Period"] == "Mid-Peak"]["Hour"].tolist())
    on_peak_hours = sorted(df[df["Peak_Period"] == "On-Peak"]["Hour"].tolist())

    print(f"🌙 Off-Peak Hours (低谷时段): {off_peak_hours}")
    print(f"⏳ Mid-Peak Hours (中间时段): {mid_peak_hours}")
    print(f"🔥 On-Peak Hours (高峰时段): {on_peak_hours}")

# ✅ Process data and save results
def main():
    df = load_all_data(file_paths)

    if df is None or df.empty:
        print("❌ No valid data loaded.")
        return

    required_cols = ["YYYYMMDD", "LOCATION"] + [f"R{i}" for i in range(1, 25)]
    if not all(col in df.columns for col in required_cols):
        print("❌ Missing required columns in the data files.")
        return

    df_grouped = transform_data(df)
    df_clustered = classify_peak_mini_batch(df_grouped)

    output_file = "processed_peak_periods_all_weeks.xlsx"
    df_clustered.to_excel(output_file, index=False)

    print(f"✅ Processed data saved to {output_file}")
    print_peak_hours(df_clustered)

    # ✅ Visualization
    plt.figure(figsize=(10, 5))
    plt.scatter(df_clustered["Hour"], df_clustered["Usage"], c=df_clustered["Cluster"], cmap="coolwarm", edgecolors="k")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Electricity Usage (kWh)")
    plt.title("Electricity Usage Clustering (All Weeks Combined)")
    plt.colorbar(label="Cluster")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
