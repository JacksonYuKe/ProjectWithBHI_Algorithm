from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import glob
import os


# **📌 计算 `LOCATION` 过去 52 周的用电概率**
def process_weekly_csv(window_size, threshold):
    input_dir = "/Users/jacson/Desktop/BHI/Decrypted_Files/weekly_csv_files_test"

    # 获取所有符合条件的文件
    file_paths = sorted(glob.glob(os.path.join(input_dir, "week_2023-*.csv")))
    print(f"🧐 Found {len(file_paths)} files")  # ✅ 调试信息

    if len(file_paths) == 0:
        print("❌ No CSV files found in the directory!")
        return pd.DataFrame(columns=["LOCATION", "Probability", "Weeks"]), "0%"  # 返回空 DataFrame

    location_data = {}
    total_weeks = len(file_paths)  # 计算总周数

    for file in file_paths:
        try:
            week_number = os.path.basename(file).split("_")[1].split(".")[0]  # 提取周号
            df = pd.read_csv(file, low_memory=False)

            # 规范列名
            df.columns = df.columns.str.strip()
            if "LOCATION" not in df.columns:
                print(f"⚠️ Skipping {file}: 'LOCATION' column not found!")
                continue

            for location, group in df.groupby("LOCATION"):
                usage_matrix = group.iloc[:, 4:28].astype(float).values  # 取时间数据

                # **✅ 让 `window_size` 和 `threshold` 可调**
                rolling_sums = pd.DataFrame(usage_matrix).rolling(window=window_size, axis=1).sum()
                condition_met = (rolling_sums > threshold).any().any()

                if location not in location_data:
                    location_data[location] = {"Exceed_Count": 0, "Weeks": []}

                if condition_met:
                    location_data[location]["Exceed_Count"] += 1
                    location_data[location]["Weeks"].append(week_number)

        except Exception as e:
            print(f"❌ Error processing file {file}: {e}")

    # 计算概率 = 超过阈值的次数 / 总周数
    prob_df = pd.DataFrame([{
        "LOCATION": loc,
        "Probability": round(data["Exceed_Count"] / total_weeks, 3) if total_weeks > 0 else 0,
        "Weeks": ", ".join(data["Weeks"])
    } for loc, data in location_data.items()])

    # 计算高概率用户的占比
    if not prob_df.empty:
        high_prob_users = (prob_df["Probability"] > 0.5).sum()
        total_users = len(prob_df)
        high_prob_ratio = f"{round((high_prob_users / total_users) * 100, 2)}%" if total_users > 0 else "0%"
    else:
        high_prob_ratio = "0%"

    print(f"✅ Successfully processed {len(prob_df)} locations!")
    print(f"📊 High Probability Users (>0.5): {high_prob_users}/{total_users} = {high_prob_ratio}")

    return prob_df, high_prob_ratio


# **📌 统计页面布局**
def create_stats_layout():
    return dbc.Container([
        # **📌 变量输入区域**
        dbc.Row([
            dbc.Col([
                html.Label("Rolling Window Size (Hours)"),
                dcc.Slider(
                    id="window-size-slider",
                    min=1, max=12, step=1, value=4,  # 默认值 4
                    marks={i: str(i) for i in range(1, 13)}
                )
            ], width=6),

            dbc.Col([
                html.Label("Threshold Value"),
                dcc.Input(
                    id="threshold-input",
                    type="number",
                    value=1.5,  # 默认值 1.5
                    step=0.1
                )
            ], width=6),
        ], className="mb-4"),

        # **📌 计算按钮**
        dbc.Row([
            dbc.Col([
                dbc.Button("Calculate", id="calculate-btn", color="primary", className="mb-3")
            ], width=12, className="text-center")
        ]),

        # **📌 高概率用户占比**
        dbc.Row([
            dbc.Col([
                html.H4("High Probability Users (>0.5)"),
                html.H2(id="high-prob-ratio", children="Waiting...", className="text-primary")
            ], width=12, className="text-center mt-4")
        ]),

        # **📌 数据表**
        dbc.Row([
            dbc.Col(dash_table.DataTable(
                id='location-prob-table',
                columns=[
                    {"name": "LOCATION", "id": "LOCATION"},
                    {"name": "Probability", "id": "Probability"},
                    {"name": "Weeks", "id": "Weeks"}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                page_size=10
            ), width=12)
        ])
    ], fluid=True)
