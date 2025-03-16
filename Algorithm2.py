import pandas as pd
import glob
import os

# ✅ Define the directory containing filtered weekly CSV files
INPUT_DIR = "/Users/jacson/Desktop/BHI/Decrypted_Files/weekly_csv_files_test"

# ✅ Get all CSV files matching the pattern "week_2023-*"
file_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "week_2023-*.csv")))

# ✅ Ensure there are files to process
if len(file_paths) == 0:
    print("❌ No CSV files found in the directory!")
    exit()

total_weeks = len(file_paths)  # Total weeks processed
# window_sizes = [2, 3]
# thresholds = [1, 1.5]
# change
window_sizes = [2, 3, 4, 5,6,7,8,9,10]  # Rolling window sizes
thresholds = [1, 1.5, 2, 2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]  # Consumption thresholds

# ✅ Iterate over 16 (4x4) combinations of window_size and threshold
for window_size in window_sizes:
    for threshold in thresholds:
        location_data = {}

        # ✅ Process each weekly CSV file
        for file in file_paths:
            try:
                week_number = os.path.basename(file).split("_")[1].split(".")[0]  # Extract week number
                df = pd.read_csv(file, low_memory=False)

                # ✅ Normalize column names (remove spaces)
                df.columns = df.columns.str.strip()

                # ✅ Ensure "LOCATION" column exists
                if "LOCATION" not in df.columns:
                    continue

                # ✅ Process each location
                for location, group in df.groupby("LOCATION"):
                    usage_matrix = group.iloc[:, 4:28].astype(float).values  # Extract hourly usage data

                    # ✅ Compute rolling sum over the window size
                    rolling_sums = pd.DataFrame(usage_matrix).rolling(window=window_size, axis=1).sum()

                    # ✅ Check if any row meets the condition
                    condition_met = (rolling_sums > threshold).any(axis=1).any()

                    if location not in location_data:
                        location_data[location] = {"Exceed_Count": 0}

                    if condition_met:
                        location_data[location]["Exceed_Count"] += 1

            except Exception as e:
                print(f"❌ Error processing file {file}: {e}")

        # ✅ Compute high-probability users
        prob_results = [
            data["Exceed_Count"] / total_weeks if total_weeks > 0 else 0
            for data in location_data.values()
        ]
        high_prob_users = sum(prob > 0.5 for prob in prob_results)
        total_users = len(prob_results)
        high_prob_ratio = f"{round((high_prob_users / total_users) * 100, 2)}%" if total_users > 0 else "0%"

        # ✅ Print results
        print(f"🛠 WINDOW_SIZE = {window_size}, THRESHOLD = {threshold} ➝ 📊 High Probability Users: {high_prob_ratio}")
