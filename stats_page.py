from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import glob
import os

import os
import glob
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def process_weekly_csv(window_size, threshold):
    # input_dir = "/Users/jacson/BHI/Decrypted_Files/weekly"  # Updated directory
    input_dir = "/Users/jackson/BHI/Decrypted_Files/weekly_csv_files_test"  # Updated directory
    file_paths = sorted(glob.glob(os.path.join(input_dir, "week_2023-*.csv")))
    print(f"🧐 Found {len(file_paths)} files")

    if len(file_paths) == 0:
        print("❌ No CSV files found in the directory!")
        return pd.DataFrame(columns=["LOCATION", "Probability", "Weeks"]), 0, 0, 0, 0

    location_data = {}
    total_weeks = len(file_paths)

    # 处理每个文件
    for file in file_paths:
        try:
            week_number = os.path.basename(file).split("_")[1].split(".")[0]
            df = pd.read_csv(file, low_memory=False)
            df.columns = df.columns.str.strip()
            if "LOCATION" not in df.columns:
                print(f"⚠️ Skipping {file}: 'LOCATION' column not found!")
                continue

            # 对每个 location 进行处理
            for location, group in df.groupby("LOCATION"):
                usage_matrix = group.iloc[:, 4:28].astype(float).values  # 提取该地点的24小时用电数据

                # 计算该 location 的 baseline：即平均每小时用电量
                baseline = usage_matrix.mean()  # 计算每周24小时的平均值
                baseline = baseline.mean()  # 计算该地点的整体 baseline（平均每小时）

                # 判断是否有连续 window_size 个时间点都超过阈值
                condition_met = False
                for row in usage_matrix:
                    # 对每一行(每天)检查是否有连续window_size个时间点都超过阈值
                    for i in range(len(row) - window_size + 1):
                        if all(point > (threshold + baseline) for point in row[i:i + window_size]):
                            condition_met = True
                            break
                    if condition_met:
                        break

                if location not in location_data:
                    location_data[location] = {"Exceed_Count": 0, "Weeks": [], "Baseline": baseline}

                if condition_met:
                    location_data[location]["Exceed_Count"] += 1
                    location_data[location]["Weeks"].append(week_number)

        except Exception as e:
            print(f"❌ Error processing file {file}: {e}")

    # 计算概率
    prob_df = pd.DataFrame([{
        "LOCATION": loc,
        "Probability": round(data["Exceed_Count"] / total_weeks, 3) if total_weeks > 0 else 0,
        "Weeks": ", ".join(data["Weeks"]),
        "Baseline": data["Baseline"]
    } for loc, data in location_data.items()])

    # 读取真实充电桩数据
    real_data = pd.read_csv(file_paths[-1], usecols=["LOCATION", "# of Chargers"])
    real_data["Has_Charger"] = real_data["# of Chargers"].notna().astype(int)

    # 合并预测数据和真实数据
    merged_df = prob_df.merge(real_data, on="LOCATION", how="left").fillna(0)
    merged_df["Prediction"] = (merged_df["Probability"] > 0.5).astype(int)

    # 创建评估数据的副本，避免影响原始数据
    eval_df = merged_df.copy()

    # 计算样本比例差异
    tp_fn = eval_df[eval_df["Has_Charger"] == 1]  # 有充电桩的数据
    fp_tn = eval_df[eval_df["Has_Charger"] == 0]  # 没有充电桩的数据

    tp_fn_count = len(tp_fn)
    fp_tn_count = len(fp_tn)

    if tp_fn_count > fp_tn_count:
        tp_fn = tp_fn.sample(n=fp_tn_count, random_state=42)
    elif fp_tn_count > tp_fn_count:
        fp_tn = fp_tn.sample(n=tp_fn_count, replace=True, random_state=42)

    # 合并回缩放后的数据（仅用于评估）
    balanced_df = pd.concat([tp_fn, fp_tn])

    # 重新计算评估指标（使用平衡后的数据）
    y_true = balanced_df["Has_Charger"]
    y_pred = balanced_df["Prediction"]

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"True Positive (TP): {tp}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Negative (TN): {tn}")

    print(f"✅ Accuracy: {accuracy:.3f}")
    print(f"🎯 Precision: {precision:.3f}")
    print(f"📢 Recall: {recall:.3f}")
    print(f"📊 F1 Score: {f1:.3f}")

    # 返回原始合并的数据框（不含重复），而不是平衡后的数据框
    merged_df = merged_df.drop_duplicates().drop(columns=["Has_Charger"])
    return merged_df, accuracy, precision, recall, f1


# **📌 统计页面布局**
def create_stats_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Rolling Window Size (Hours)"),
                dcc.Slider(
                    id="window-size-slider",
                    min=1, max=12, step=1, value=4,
                    marks={i: str(i) for i in range(1, 13)}
                )
            ], width=6),

            dbc.Col([
                html.Label("Threshold Value"),
                dcc.Input(
                    id="threshold-input",
                    type="number",
                    value=1.5,
                    step=0.1
                )
            ], width=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Button("Calculate", id="calculate-btn", color="primary", className="mb-3")
            ], width=12, className="text-center")
        ]),

        dbc.Row([
            dbc.Col([
                html.H4("Model Performance Metrics"),
                html.H5("Accuracy: ", id="accuracy-output", className="text-primary"),
                html.H5("Precision: ", id="precision-output", className="text-primary"),
                html.H5("Recall: ", id="recall-output", className="text-primary"),
                html.H5("F1 Score: ", id="f1-score-output", className="text-primary")
            ], width=12, className="text-center mt-4")
        ]),

        dbc.Row([
            dbc.Col(dash_table.DataTable(
                id='location-prob-table',
                columns=[
                    {"name": "LOCATION", "id": "LOCATION"},
                    {"name": "Probability", "id": "Probability"},
                    {"name": "Weeks", "id": "Weeks"},
                    {"name": "# of Chargers", "id": "# of Chargers"},
                    {"name": "Prediction", "id": "Prediction"},
                    # {"name": "Has_Charger", "id": "Has_Charger"}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                page_size=10
            ), width=12)
        ])
    ], fluid=True)