import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time


def process_weekly_csv(window_size, initial_threshold=1.0):
    """
    Enhanced algorithm for detecting EV charging stations from electricity usage data.
    Implements seasonal baselines and ML-based thresholds.

    Args:
        window_size: Number of consecutive hours required to exceed threshold
        initial_threshold: Starting threshold value (will be optimized by ML)

    Returns:
        DataFrame with prediction results and performance metrics
    """
    start_time = time.time()
    input_dir = "/Users/jackson/Project/Project_WIth_BHI/Data/Data_By_Week_CSV_Sample"
    file_paths = sorted(glob.glob(os.path.join(input_dir, "week_2023-*.csv")))
    print(f"🧐 Found {len(file_paths)} files")

    if len(file_paths) == 0:
        print("❌ No CSV files found in the directory!")
        return pd.DataFrame(columns=["LOCATION", "Probability", "Weeks"]), 0, 0, 0, 0

    # First collect features and data for all locations
    all_features = []
    unique_locations = set()
    location_raw_data = {}  # Store raw data for later judgment
    total_weeks = len(file_paths)

    # Create season mapping: 1-3 months for winter, 4-6 for spring, 7-9 for summer, 10-12 for fall
    def get_season(date_str):
        month = int(date_str.split('-')[1])
        if 1 <= month <= 3:
            return 'winter'
        elif 4 <= month <= 6:
            return 'spring'
        elif 7 <= month <= 9:
            return 'summer'
        else:
            return 'fall'

    # Step 1: Collect features and group data by season
    for file in file_paths:
        try:
            date_str = os.path.basename(file).split("_")[1].split(".")[0]
            season = get_season(date_str)
            df = pd.read_csv(file, low_memory=False)
            df.columns = df.columns.str.strip()

            if "LOCATION" not in df.columns:
                print(f"⚠️ Skipping {file}: 'LOCATION' column not found!")
                continue

            # Process each location
            for location, group in df.groupby("LOCATION"):
                usage_matrix = group.iloc[:, 4:28].astype(float).values  # Extract 24-hour electricity usage data for this location

                # Handle NaN values
                if np.isnan(usage_matrix).any():
                    usage_matrix = np.nan_to_num(usage_matrix, nan=0.0)

                # Create location data structure (only for new locations)
                if location not in unique_locations:
                    unique_locations.add(location)
                    # Initialize storage structure
                    location_raw_data[location] = {
                        "weeks": {},
                        "seasons": {"winter": {"hours": [[] for _ in range(24)]},
                                    "spring": {"hours": [[] for _ in range(24)]},
                                    "summer": {"hours": [[] for _ in range(24)]},
                                    "fall": {"hours": [[] for _ in range(24)]}},
                        "features": {}
                    }

                    # Extract features
                    mean_usage = np.mean(usage_matrix)
                    std_dev = np.std(usage_matrix)
                    max_usage = np.max(usage_matrix)

                    # Time period features
                    morning_avg = np.mean(usage_matrix[:, 6:12])  # 6AM-12PM
                    afternoon_avg = np.mean(usage_matrix[:, 12:18])  # 12PM-6PM
                    evening_avg = np.mean(usage_matrix[:, 18:24])  # 6PM-12AM
                    night_avg = np.mean(usage_matrix[:, 0:6])  # 12AM-6AM

                    # Ratio features
                    peak_to_avg = max_usage / mean_usage if mean_usage > 0 else 0
                    evening_to_day = evening_avg / (morning_avg + afternoon_avg) if (
                                                                                                morning_avg + afternoon_avg) > 0 else 0

                    # Store features
                    location_raw_data[location]["features"] = {
                        "overall_mean": mean_usage,
                        "std_dev": std_dev,
                        "max_usage": max_usage,
                        "morning_avg": morning_avg,
                        "afternoon_avg": afternoon_avg,
                        "evening_avg": evening_avg,
                        "night_avg": night_avg
                    }

                    # Add to feature list
                    all_features.append([
                        mean_usage, std_dev, max_usage,
                        morning_avg, afternoon_avg, evening_avg, night_avg,
                        peak_to_avg, evening_to_day
                    ])

                # Store raw data for each week
                location_raw_data[location]["weeks"][date_str] = {
                    "usage_matrix": usage_matrix,
                    "season": season
                }

                # Collect data by season and hour
                for day_idx in range(usage_matrix.shape[0]):
                    for hour_idx in range(24):
                        hour_value = usage_matrix[day_idx, hour_idx]
                        location_raw_data[location]["seasons"][season]["hours"][hour_idx].append(hour_value)

        except Exception as e:
            print(f"❌ Error processing file {file}: {e}")

    # Calculate seasonal baselines for each location
    all_locations = list(unique_locations)
    for location in all_locations:
        # Initialize seasonal baselines
        season_baselines = {}

        # Calculate 24-hour average electricity usage for each season
        for season in location_raw_data[location]["seasons"]:
            hourly_baseline = np.zeros(24)

            for hour in range(24):
                hour_data = location_raw_data[location]["seasons"][season]["hours"][hour]
                if len(hour_data) > 0:
                    hourly_baseline[hour] = np.mean(hour_data)
                else:
                    # If no data for this season/hour, use overall average
                    hourly_baseline[hour] = location_raw_data[location]["features"]["overall_mean"]

            season_baselines[season] = hourly_baseline

        # Store seasonal baselines
        location_raw_data[location]["baselines"] = season_baselines

    # Create feature dataframe
    features_df = pd.DataFrame(all_features, columns=[
        "Mean_Usage", "Std_Dev", "Max_Usage",
        "Morning_Avg", "Afternoon_Avg", "Evening_Avg", "Night_Avg",
        "Peak_To_Avg_Ratio", "Evening_To_Day_Ratio"
    ])

    # Step 2: Prepare data for training machine learning models
    try:
        # Get true labels
        real_data = pd.read_csv(file_paths[-1], usecols=["LOCATION", "# of Chargers"])
        real_data["Has_Charger"] = real_data["# of Chargers"].notna().astype(int)

        # Merge features and labels
        ml_data = pd.DataFrame({"LOCATION": all_locations})
        ml_data = ml_data.merge(real_data[["LOCATION", "Has_Charger"]], on="LOCATION", how="left").fillna(0)

        # Add features
        for i, col in enumerate(features_df.columns):
            ml_data[col] = features_df[col].values

        # Separate features and labels
        X = ml_data.drop(["LOCATION", "Has_Charger"], axis=1)
        y = ml_data["Has_Charger"]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Optimize Random Forest classifier parameters for faster computation
        model_start_time = time.time()
        classifier = RandomForestClassifier(
            n_estimators=50,  # Reduce number of trees (from 100 to 50)
            max_depth=8,      # Limit maximum tree depth
            min_samples_split=5,  # Increase minimum samples required for splitting
            min_samples_leaf=2,   # Set minimum samples for leaf nodes
            max_features='sqrt',  # Use square root of features
            bootstrap=True,       # Use bootstrap sampling
            n_jobs=-1,            # Use all CPU cores for parallel computation
            random_state=42
        )
        classifier.fit(X_train, y_train)
        model_train_time = time.time() - model_start_time
        print(f"Classifier training time: {model_train_time:.2f} seconds")

        # Calculate test set performance
        y_pred = classifier.predict(X_test)
        print(f"\nModel accuracy on test set: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        # Extract feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': classifier.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nFeature importance:")
        for i, row in feature_importance.head(5).iterrows():  # Only show top 5 important features
            print(f"{row['Feature']:20}: {row['Importance']:.4f}")

        # Optimize threshold prediction model
        threshold_model = RandomForestRegressor(
            n_estimators=30,      # Fewer trees
            max_depth=6,          # Shallower trees
            min_samples_split=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )

        # Create threshold targets: locations with chargers get higher thresholds, those without get lower thresholds
        threshold_target = np.array([initial_threshold * (1.5 if val == 1 else 0.8) for val in y])

        # Train model
        threshold_model_start_time = time.time()
        threshold_model.fit(X_scaled, threshold_target)
        threshold_model_train_time = time.time() - threshold_model_start_time
        print(f"Threshold model training time: {threshold_model_train_time:.2f} seconds")

        # Predict optimal threshold for each location
        predicted_thresholds = threshold_model.predict(X_scaled)
        location_thresholds = dict(zip(all_locations, predicted_thresholds))

        print("\nPredicted threshold range:", np.min(predicted_thresholds), "to", np.max(predicted_thresholds))

    except Exception as e:
        print(f"❌ Error training ML model: {e}")
        # If ML model fails, fall back to default threshold
        location_thresholds = {loc: initial_threshold for loc in all_locations}

    # Step 3: Use seasonal baselines and ML-predicted thresholds for judgment
    location_data = {}
    for location in all_locations:
        # Get ML-predicted threshold for this location
        ml_threshold = location_thresholds.get(location, initial_threshold)

        # Initialize location data
        location_data[location] = {
            "Exceed_Count": 0,
            "Weeks": [],
            "Features": location_raw_data[location]["features"],
            "ML_Threshold": ml_threshold
        }

        # Apply seasonal baselines and ML thresholds to weekly data
        for week_date, week_data in location_raw_data[location]["weeks"].items():
            usage_matrix = week_data["usage_matrix"]
            season = week_data["season"]

            # Get 24-hour baseline for this season
            seasonal_baseline = location_raw_data[location]["baselines"][season]

            # Check if there are window_size consecutive hours exceeding seasonal baseline + ML threshold
            condition_met = False

            for day_idx, row in enumerate(usage_matrix):
                for start_hour in range(len(row) - window_size + 1):
                    # Check if window_size consecutive hours all exceed their corresponding hourly seasonal baseline + ML threshold
                    hours_exceeded = True
                    for i in range(window_size):
                        hour_idx = start_hour + i
                        if row[hour_idx] <= (seasonal_baseline[hour_idx] + ml_threshold):
                            hours_exceeded = False
                            break

                    if hours_exceeded:
                        condition_met = True
                        break

                if condition_met:
                    break

            if condition_met:
                location_data[location]["Exceed_Count"] += 1
                location_data[location]["Weeks"].append(week_date)

    # Calculate probability
    prob_data = []
    for loc, data in location_data.items():
        # Calculate average baseline (for display)
        avg_baselines = {}
        for season in location_raw_data[loc]["baselines"]:
            avg_baselines[f"{season}_baseline"] = np.mean(location_raw_data[loc]["baselines"][season])

        prob_data.append({
            "LOCATION": loc,
            "Probability": round(data["Exceed_Count"] / total_weeks, 3) if total_weeks > 0 else 0,
            "Weeks": ", ".join(data["Weeks"]),
            "Mean_Usage": data["Features"]["overall_mean"],
            "ML_Threshold": data["ML_Threshold"],
            **avg_baselines  # Add average baselines for all seasons
        })

    prob_df = pd.DataFrame(prob_data)

    # Read actual charger data for evaluation
    try:
        real_data = pd.read_csv(file_paths[-1], usecols=["LOCATION", "# of Chargers"])
        real_data["Has_Charger"] = real_data["# of Chargers"].notna().astype(int)
    except Exception as e:
        print(f"❌ Error reading charger data: {e}")
        real_data = pd.DataFrame(columns=["LOCATION", "# of Chargers", "Has_Charger"])

    # Merge predicted data with actual data
    merged_df = prob_df.merge(real_data, on="LOCATION", how="left").fillna(0)
    merged_df["Prediction"] = (merged_df["Probability"] > 0.5).astype(int)

    # Evaluation section
    eval_df = merged_df.copy()
    tp_fn = eval_df[eval_df["Has_Charger"] == 1]
    fp_tn = eval_df[eval_df["Has_Charger"] == 0]

    tp_fn_count = len(tp_fn)
    fp_tn_count = len(fp_tn)

    if tp_fn_count == 0 or fp_tn_count == 0:
        print("⚠️ Warning: One of the classes has zero samples. Metrics may be invalid.")
        accuracy, precision, recall, f1 = 0, 0, 0, 0
    else:
        if tp_fn_count > fp_tn_count:
            tp_fn = tp_fn.sample(n=fp_tn_count, random_state=42)
        elif fp_tn_count > tp_fn_count:
            fp_tn = fp_tn.sample(n=tp_fn_count, replace=True, random_state=42)

        balanced_df = pd.concat([tp_fn, fp_tn])
        y_true = balanced_df["Has_Charger"]
        y_pred = balanced_df["Prediction"]

        accuracy = accuracy_score(y_true, y_pred) * 100

        try:
            precision = precision_score(y_true, y_pred) * 100
        except:
            precision = 0

        try:
            recall = recall_score(y_true, y_pred) * 100
        except:
            recall = 0

        try:
            f1 = f1_score(y_true, y_pred) * 100
        except:
            f1 = 0

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            print(f"True Positive (TP): {tp}")
            print(f"False Positive (FP): {fp}")
            print(f"False Negative (FN): {fn}")
            print(f"True Negative (TN): {tn}")
        except:
            print("Could not compute confusion matrix")

    print(f"✅ Accuracy: {accuracy:.3f}")
    print(f"🎯 Precision: {precision:.3f}")
    print(f"📢 Recall: {recall:.3f}")
    print(f"📊 F1 Score: {f1:.3f}")
    
    total_time = time.time() - start_time
    print(f"⏱️ Total runtime: {total_time:.2f} seconds")

    merged_df = merged_df.drop_duplicates().drop(columns=["Has_Charger"])
    return merged_df, accuracy, precision, recall, f1


if __name__ == "__main__":
    # Test the algorithm with different parameters
    window_sizes = [3]
    initial_thresholds = [1.0]

    results = []
    for window_size in window_sizes:
        for threshold in initial_thresholds:
            print(f"\n===== Testing window_size={window_size}, initial_threshold={threshold} =====")
            _, accuracy, precision, recall, f1 = process_weekly_csv(window_size, threshold)
            results.append({
                "Window Size": window_size,
                "Initial Threshold": threshold,
                "Accuracy": f"{accuracy:.2f}%",
                "Precision": f"{precision:.2f}%",
                "Recall": f"{recall:.2f}%",
                "F1 Score": f"{f1:.2f}%"
            })

    # Print summary
    results_df = pd.DataFrame(results)
    print("\n===== Results Summary =====")
    print(results_df)
