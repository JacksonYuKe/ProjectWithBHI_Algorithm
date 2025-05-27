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
    input_dir = "/Users/jackson/BHI/Decrypted_Files/weekly_csv_files_test"
    file_paths = sorted(glob.glob(os.path.join(input_dir, "week_2023-*.csv")))
    print(f"🧐 Found {len(file_paths)} files")

    if len(file_paths) == 0:
        print("❌ No CSV files found in the directory!")
        return pd.DataFrame(columns=["LOCATION", "Probability", "Weeks"]), 0, 0, 0, 0

    # 首先收集所有位置的特征和数据
    all_features = []
    all_locations = []
    location_raw_data = {}  # 存储原始数据，稍后用于判断
    total_weeks = len(file_paths)

    # 创建季节映射: 1-3月为冬季, 4-6月为春季, 7-9月为夏季, 10-12月为秋季
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

    # 第一步：收集特征和按季节分组数据
    for file in file_paths:
        try:
            date_str = os.path.basename(file).split("_")[1].split(".")[0]
            season = get_season(date_str)
            df = pd.read_csv(file, low_memory=False)
            df.columns = df.columns.str.strip()

            if "LOCATION" not in df.columns:
                print(f"⚠️ Skipping {file}: 'LOCATION' column not found!")
                continue

            # 对每个 location 进行处理
            for location, group in df.groupby("LOCATION"):
                usage_matrix = group.iloc[:, 4:28].astype(float).values  # 提取该地点的24小时用电数据

                # 处理NaN值
                if np.isnan(usage_matrix).any():
                    usage_matrix = np.nan_to_num(usage_matrix, nan=0.0)

                # 创建位置数据结构
                if location not in all_locations:
                    all_locations.append(location)
                    # 初始化存储结构
                    location_raw_data[location] = {
                        "weeks": {},
                        "seasons": {"winter": {"hours": [[] for _ in range(24)]},
                                    "spring": {"hours": [[] for _ in range(24)]},
                                    "summer": {"hours": [[] for _ in range(24)]},
                                    "fall": {"hours": [[] for _ in range(24)]}},
                        "features": {}
                    }

                    # 提取特征
                    mean_usage = np.mean(usage_matrix)
                    std_dev = np.std(usage_matrix)
                    max_usage = np.max(usage_matrix)

                    # 时段特征
                    morning_avg = np.mean(usage_matrix[:, 6:12])  # 6AM-12PM
                    afternoon_avg = np.mean(usage_matrix[:, 12:18])  # 12PM-6PM
                    evening_avg = np.mean(usage_matrix[:, 18:24])  # 6PM-12AM
                    night_avg = np.mean(usage_matrix[:, 0:6])  # 12AM-6AM

                    # 比率特征
                    peak_to_avg = max_usage / mean_usage if mean_usage > 0 else 0
                    evening_to_day = evening_avg / (morning_avg + afternoon_avg) if (
                                                                                                morning_avg + afternoon_avg) > 0 else 0

                    # 存储特征
                    location_raw_data[location]["features"] = {
                        "overall_mean": mean_usage,
                        "std_dev": std_dev,
                        "max_usage": max_usage,
                        "morning_avg": morning_avg,
                        "afternoon_avg": afternoon_avg,
                        "evening_avg": evening_avg,
                        "night_avg": night_avg
                    }

                    # 加入特征列表
                    all_features.append([
                        mean_usage, std_dev, max_usage,
                        morning_avg, afternoon_avg, evening_avg, night_avg,
                        peak_to_avg, evening_to_day
                    ])

                # 存储每周的原始数据
                location_raw_data[location]["weeks"][date_str] = {
                    "usage_matrix": usage_matrix,
                    "season": season
                }

                # 按季节和小时收集数据
                for day_idx in range(usage_matrix.shape[0]):
                    for hour_idx in range(24):
                        hour_value = usage_matrix[day_idx, hour_idx]
                        location_raw_data[location]["seasons"][season]["hours"][hour_idx].append(hour_value)

        except Exception as e:
            print(f"❌ Error processing file {file}: {e}")

    # 计算每个位置的季节性基线
    for location in all_locations:
        # 初始化季节基线
        season_baselines = {}

        # 对每个季节计算24小时的平均用电量
        for season in location_raw_data[location]["seasons"]:
            hourly_baseline = np.zeros(24)

            for hour in range(24):
                hour_data = location_raw_data[location]["seasons"][season]["hours"][hour]
                if len(hour_data) > 0:
                    hourly_baseline[hour] = np.mean(hour_data)
                else:
                    # 如果该季节该小时没有数据，使用总体平均值
                    hourly_baseline[hour] = location_raw_data[location]["features"]["overall_mean"]

            season_baselines[season] = hourly_baseline

        # 存储季节性基线
        location_raw_data[location]["baselines"] = season_baselines

    # 创建特征数据框
    features_df = pd.DataFrame(all_features, columns=[
        "Mean_Usage", "Std_Dev", "Max_Usage",
        "Morning_Avg", "Afternoon_Avg", "Evening_Avg", "Night_Avg",
        "Peak_To_Avg_Ratio", "Evening_To_Day_Ratio"
    ])

    # 第二步：准备训练机器学习模型的数据
    try:
        # 获取真实标签
        real_data = pd.read_csv(file_paths[-1], usecols=["LOCATION", "# of Chargers"])
        real_data["Has_Charger"] = real_data["# of Chargers"].notna().astype(int)

        # 合并特征和标签
        ml_data = pd.DataFrame({"LOCATION": all_locations})
        ml_data = ml_data.merge(real_data[["LOCATION", "Has_Charger"]], on="LOCATION", how="left").fillna(0)

        # 添加特征
        for i, col in enumerate(features_df.columns):
            ml_data[col] = features_df[col].values


        # 分离特征和标签
        X = ml_data.drop(["LOCATION", "Has_Charger"], axis=1)
        y = ml_data["Has_Charger"]

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # 优化随机森林分类器参数以加快计算速度
        model_start_time = time.time()
        classifier = RandomForestClassifier(
            n_estimators=50,  # 减少树的数量（从100减少到50）
            max_depth=8,      # 限制树的最大深度
            min_samples_split=5,  # 增加分裂所需的最小样本数
            min_samples_leaf=2,   # 设置叶节点最小样本数
            max_features='sqrt',  # 使用特征的平方根数量
            bootstrap=True,       # 使用自助采样
            n_jobs=-1,            # 使用所有CPU核心并行计算
            random_state=42
        )
        classifier.fit(X_train, y_train)
        model_train_time = time.time() - model_start_time
        print(f"分类器训练时间: {model_train_time:.2f} 秒")

        # 计算测试集性能
        y_pred = classifier.predict(X_test)
        print(f"\n模型在测试集上的准确率: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        # 提取特征重要性
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': classifier.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\n特征重要性:")
        for i, row in feature_importance.head(5).iterrows():  # 只显示前5个重要特征
            print(f"{row['Feature']:20}: {row['Importance']:.4f}")

        # 优化阈值预测模型
        threshold_model = RandomForestRegressor(
            n_estimators=30,      # 更少的树
            max_depth=6,          # 较浅的树
            min_samples_split=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )

        # 创建阈值目标：有充电桩的位置设置较高阈值，没有的设置较低阈值
        threshold_target = np.array([initial_threshold * (1.5 if val == 1 else 0.8) for val in y])

        # 训练模型
        threshold_model_start_time = time.time()
        threshold_model.fit(X_scaled, threshold_target)
        threshold_model_train_time = time.time() - threshold_model_start_time
        print(f"阈值模型训练时间: {threshold_model_train_time:.2f} 秒")

        # 预测每个位置的最佳阈值
        predicted_thresholds = threshold_model.predict(X_scaled)
        location_thresholds = dict(zip(all_locations, predicted_thresholds))

        print("\n预测的阈值范围:", np.min(predicted_thresholds), "到", np.max(predicted_thresholds))

    except Exception as e:
        print(f"❌ Error training ML model: {e}")
        # 如果机器学习模型失败，回退到默认阈值
        location_thresholds = {loc: initial_threshold for loc in all_locations}

    # 第三步：使用季节性基线和ML预测的阈值进行判断
    location_data = {}
    for location in all_locations:
        # 获取该位置的ML预测阈值
        ml_threshold = location_thresholds.get(location, initial_threshold)

        # 初始化位置数据
        location_data[location] = {
            "Exceed_Count": 0,
            "Weeks": [],
            "Features": location_raw_data[location]["features"],
            "ML_Threshold": ml_threshold
        }

        # 对每周数据应用季节性基线和ML阈值
        for week_date, week_data in location_raw_data[location]["weeks"].items():
            usage_matrix = week_data["usage_matrix"]
            season = week_data["season"]

            # 获取该季节的24小时基线
            seasonal_baseline = location_raw_data[location]["baselines"][season]

            # 判断是否有连续window_size个小时超过季节基线+ML阈值
            condition_met = False

            for day_idx, row in enumerate(usage_matrix):
                for start_hour in range(len(row) - window_size + 1):
                    # 检查连续window_size个小时是否都超过了对应小时的季节基线+ML阈值
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

    # 计算概率
    prob_data = []
    for loc, data in location_data.items():
        # 计算基线的平均值（用于展示）
        avg_baselines = {}
        for season in location_raw_data[loc]["baselines"]:
            avg_baselines[f"{season}_baseline"] = np.mean(location_raw_data[loc]["baselines"][season])

        prob_data.append({
            "LOCATION": loc,
            "Probability": round(data["Exceed_Count"] / total_weeks, 3) if total_weeks > 0 else 0,
            "Weeks": ", ".join(data["Weeks"]),
            "Mean_Usage": data["Features"]["overall_mean"],
            "ML_Threshold": data["ML_Threshold"],
            **avg_baselines  # 添加所有季节的平均基线
        })

    prob_df = pd.DataFrame(prob_data)

    # 读取真实充电桩数据进行评估
    try:
        real_data = pd.read_csv(file_paths[-1], usecols=["LOCATION", "# of Chargers"])
        real_data["Has_Charger"] = real_data["# of Chargers"].notna().astype(int)
    except Exception as e:
        print(f"❌ Error reading charger data: {e}")
        real_data = pd.DataFrame(columns=["LOCATION", "# of Chargers", "Has_Charger"])

    # 合并预测数据和真实数据
    merged_df = prob_df.merge(real_data, on="LOCATION", how="left").fillna(0)
    merged_df["Prediction"] = (merged_df["Probability"] > 0.5).astype(int)

    # 评估部分
    eval_df = merged_df.copy()
    tp_fn = eval_df[eval_df["Has_Charger"] >= 1]
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
    print(f"⏱️ 总运行时间: {total_time:.2f} 秒")

    merged_df = merged_df.drop_duplicates().drop(columns=["Has_Charger"])
    return merged_df, accuracy, precision, recall, f1


def evaluate_forest_classifier_direct(window_size, initial_threshold=1.0):
    """
    Directly use Random Forest Classifier to predict charger presence and evaluate.
    Returns: DataFrame with predictions and metrics (accuracy, precision, recall, F1).
    """
    input_dir = "/Users/jackson/BHI/Decrypted_Files/weekly_csv_files_test"
    file_paths = sorted(glob.glob(os.path.join(input_dir, "week_2023-*.csv")))
    if len(file_paths) == 0:
        print("❌ No CSV files found in the directory!")
        return pd.DataFrame(), 0, 0, 0, 0

    # Collect features as in process_weekly_csv
    all_features = []
    all_locations = []
    for file in file_paths:
        try:
            df = pd.read_csv(file, low_memory=False)
            df.columns = df.columns.str.strip()
            if "LOCATION" not in df.columns:
                continue
            for location, group in df.groupby("LOCATION"):
                if location not in all_locations:
                    usage_matrix = group.iloc[:, 4:28].astype(float).values
                    if np.isnan(usage_matrix).any():
                        usage_matrix = np.nan_to_num(usage_matrix, nan=0.0)
                    all_locations.append(location)
                    mean_usage = np.mean(usage_matrix)
                    std_dev = np.std(usage_matrix)
                    max_usage = np.max(usage_matrix)
                    morning_avg = np.mean(usage_matrix[:, 6:12])
                    afternoon_avg = np.mean(usage_matrix[:, 12:18])
                    evening_avg = np.mean(usage_matrix[:, 18:24])
                    night_avg = np.mean(usage_matrix[:, 0:6])
                    peak_to_avg = max_usage / mean_usage if mean_usage > 0 else 0
                    evening_to_day = evening_avg / (morning_avg + afternoon_avg) if (morning_avg + afternoon_avg) > 0 else 0
                    all_features.append([
                        mean_usage, std_dev, max_usage,
                        morning_avg, afternoon_avg, evening_avg, night_avg,
                        peak_to_avg, evening_to_day
                    ])
        except Exception as e:
            continue

    features_df = pd.DataFrame(all_features, columns=[
        "Mean_Usage", "Std_Dev", "Max_Usage",
        "Morning_Avg", "Afternoon_Avg", "Evening_Avg", "Night_Avg",
        "Peak_To_Avg_Ratio", "Evening_To_Day_Ratio"
    ])

    # Debug print to check dataset size
    print(f"Found {len(all_locations)} unique locations for analysis")

    # Get true labels
    try:
        real_data = pd.read_csv(file_paths[-1], usecols=["LOCATION", "# of Chargers"])
        real_data = real_data.drop_duplicates("LOCATION")
        real_data["Has_Charger"] = real_data["# of Chargers"].notna().astype(int)
    except Exception as e:
        print(f"❌ Error reading charger data: {e}")
        return pd.DataFrame(), 0, 0, 0, 0

    ml_data = pd.DataFrame({"LOCATION": all_locations})
    ml_data = ml_data.merge(real_data[["LOCATION", "Has_Charger"]], on="LOCATION", how="left").fillna(0)
    for i, col in enumerate(features_df.columns):
        ml_data[col] = features_df[col].values

    # Print class distribution
    pos_count = ml_data["Has_Charger"].sum()
    neg_count = len(ml_data) - pos_count
    print(f"Original class distribution - Positive: {pos_count}, Negative: {neg_count}")

    # Balance the dataset before splitting
    pos_samples = ml_data[ml_data["Has_Charger"] >= 1]
    neg_samples = ml_data[ml_data["Has_Charger"] == 0]

    if len(pos_samples) == 0 or len(neg_samples) == 0:
        print("⚠️ Warning: One of the classes has zero samples. Cannot balance dataset.")
        balanced_data = ml_data
    else:
        # Balance by upsampling minority class to match majority class
        if len(pos_samples) < len(neg_samples):
            # Upsample positive class
            pos_upsampled = pos_samples.sample(n=len(neg_samples), replace=True, random_state=42)
            balanced_data = pd.concat([pos_upsampled, neg_samples])
        elif len(neg_samples) < len(pos_samples):
            # Upsample negative class
            neg_upsampled = neg_samples.sample(n=len(pos_samples), replace=True, random_state=42)
            balanced_data = pd.concat([pos_samples, neg_upsampled])
        else:
            # Already balanced
            balanced_data = ml_data

    print(f"Balanced class distribution - Total: {len(balanced_data)}, Positive: {balanced_data['Has_Charger'].sum()}, Negative: {len(balanced_data) - balanced_data['Has_Charger'].sum()}")

    X = balanced_data.drop(["LOCATION", "Has_Charger"], axis=1)
    y = balanced_data["Has_Charger"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Check train/test split distribution
    print(f"Training set - Positive: {y_train.sum()}, Negative: {len(y_train) - y_train.sum()}")
    print(f"Test set - Positive: {y_test.sum()}, Negative: {len(y_test) - y_test.sum()}")

    classifier = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    try:
        precision = precision_score(y_test, y_pred) * 100
    except:
        precision = 0
    try:
        recall = recall_score(y_test, y_pred) * 100
    except:
        recall = 0
    try:
        f1 = f1_score(y_test, y_pred) * 100
    except:
        f1 = 0

    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"Forest Classifier - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    except:
        print("Could not compute confusion matrix for Forest Classifier")

    print(f"Forest Classifier Accuracy: {accuracy:.3f}")
    print(f"Forest Classifier Precision: {precision:.3f}")
    print(f"Forest Classifier Recall: {recall:.3f}")
    print(f"Forest Classifier F1 Score: {f1:.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': classifier.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nForest Classifier Feature Importance:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"{row['Feature']:20}: {row['Importance']:.4f}")

    # For output, add predictions to original data
    indices_to_loc = {i: loc for i, loc in enumerate(balanced_data["LOCATION"])}
    test_indices = y_test.index
    test_locs = [indices_to_loc[i] for i in test_indices]
    
    test_results = pd.DataFrame({
        "LOCATION": test_locs,
        "Has_Charger": y_test.values,
        "Prediction": y_pred
    })

    return test_results, accuracy, precision, recall, f1


if __name__ == "__main__":
    # Test the algorithm with different parameters
    # window_sizes = [4, 3, 2]
    # initial_thresholds = [1.0]
    #
    # results = []
    # for window_size in window_sizes:
    #     for threshold in initial_thresholds:
    #         print(f"\n===== Testing window_size={window_size}, initial_threshold={threshold} =====")
    #         _, accuracy, precision, recall, f1 = process_weekly_csv(window_size, threshold)
    #         results.append({
    #             "Window Size": window_size,
    #             "Initial Threshold": threshold,
    #             "Accuracy": f"{accuracy:.2f}%",
    #             "Precision": f"{precision:.2f}%",
    #             "Recall": f"{recall:.2f}%",
    #             "F1 Score": f"{f1:.2f}%"
    #         })
    #
    # # Print summary
    # results_df = pd.DataFrame(results)
    # print("\n===== Results Summary =====")
    # print(results_df)

    # Add direct forest classifier evaluation
    print("\n===== Direct Forest Classifier Evaluation =====")
    _, acc, prec, rec, f1 = evaluate_forest_classifier_direct(window_size=3)
    print(f"Forest Classifier - Accuracy: {acc:.2f}%, Precision: {prec:.2f}%, Recall: {rec:.2f}%, F1: {f1:.2f}%")





