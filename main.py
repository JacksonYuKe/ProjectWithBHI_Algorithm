import os
import dash
from dash import html, dcc, Input, Output, State
import dash_uploader as du
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate
import datetime
from Ev_Detection import detect
from stats_page import  create_stats_layout, process_weekly_csv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
UPLOAD_FOLDER = "uploads"
CHUNK_SIZE = 100000
MAX_POINTS_PER_TRACE = 5000

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                prevent_initial_callbacks=True
                )
server = app.server
du.configure_upload(app, UPLOAD_FOLDER)

# Layout - Remove sheet selection elements


main_layout = dbc.Container([
    dbc.Spinner(
        html.Div(id="loading-spinner"),
        color="primary",
        type="grow",
        fullscreen=True,
        fullscreen_style={"display": "none"},
    ),

    dbc.Row([
        dbc.Col(
            html.H1("Electricity Consumption Dashboard",
                    className="text-primary text-center my-4",
                    style={'font-weight': 'bold'}),
            width=12
        )
    ]),

    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4("Data Upload", className="text-info mb-3"),
                    html.Div([
                        du.Upload(
                            id='dash-uploader',
                            text='Drag and drop or click to upload',
                            text_completed='Upload complete!',
                            filetypes=['xlsx', 'xls', 'csv'],
                            max_file_size=1024,
                            cancel_button=True,
                            pause_button=True,
                        ),
                    ], style={'border': '2px dashed #cccccc', 'padding': '20px', 'border-radius': '10px'})
                ], width=12)
            ]),
        ])
    ], className="mb-4 shadow"),

    dbc.Card([
        dbc.CardBody([
            html.Div(id='output-data-upload')
        ])
    ], className="mb-4 shadow"),

    dbc.Row([
        dbc.Col([
            html.Label("Charge Threshold Ratio (0.1 - 1.0)"),
            dcc.Input(
                id="charge-threshold-ratio",
                type="number",
                value=0.6,
                step=0.1,
                min=0.1,
                max=1.0
            )
        ], width=6),
        dbc.Col([
            html.Label("Minimum Consecutive Hours (1 - 12)"),
            dcc.Slider(
                id="min-consecutive-hours",
                min=1, max=12, step=1, value=2,
                marks={i: str(i) for i in range(1, 12)}
            )
        ], width=6)
    ], className="mb-4"),

    dcc.Store(id="stored-data"),
    dcc.Store(id="data-summary"),

    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Filters", className="text-info mb-3"),
                    dbc.Label("Select User(s):", className="font-weight-bold"),
                    dcc.Dropdown(
                        id="location-dropdown",
                        placeholder="Select one or more users",
                        multi=True,
                        className="mb-3"
                    ),
                    dbc.Label("Select Date Range:", className="font-weight-bold"),
                    dcc.DatePickerRange(
                        id='date-range-picker',
                        className="mb-3"
                    ),
                ], width=12),
            ]),
        ])
    ], className="mb-4 shadow"),

    dbc.Card([
        dbc.CardBody([
            dcc.Graph(
                id="line-chart",
                config={'displayModeBar': True, 'scrollZoom': True}
            )
        ])
    ], className="shadow"),

    dbc.Card([
        dbc.CardBody([
            dbc.Button("Download Charging Periods CSV", id="download-btn", color="primary", className="mb-2"),
            dcc.Download(id="download-dataframe-csv")
        ])
    ], className="mb-4 shadow"),

], fluid=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # 监听 URL 变化
    dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Button("Go to Statistics", href="/stats", color="primary", className="mb-3"), width=3),
            dbc.Col(dbc.Button("Go to Home", href="/", color="secondary", className="mb-3"), width=3)
        ]),
        html.Div(id='page-content')  # 动态渲染页面
    ], fluid=True)
])


def process_large_file(filename):
    """Process large files in chunks"""
    try:
        summary = {
            'total_rows': 0,
            'columns': [],
            'min_date': None,
            'max_date': None,
            'locations': set()
        }

        ext = os.path.splitext(filename)[1].lower()

        if ext in ['.xls', '.xlsx']:
            # Always read the first sheet
            df = pd.read_excel(filename)

            # Update summary information
            summary['total_rows'] = len(df)
            summary['columns'] = list(df.columns)

            if 'YYYYMMDD' in df.columns:
                df['YYYYMMDD'] = pd.to_numeric(df['YYYYMMDD'], errors='coerce')
                df = df.dropna(subset=['YYYYMMDD'])
                if not df.empty:
                    summary['min_date'] = int(df['YYYYMMDD'].min())
                    summary['max_date'] = int(df['YYYYMMDD'].max())

            if 'LOCATION' in df.columns:
                summary['locations'].update(df['LOCATION'].astype(str).unique())

        elif ext == '.csv':
            try:
                # First pass: get column names and check data structure
                first_chunk = next(pd.read_csv(filename, chunksize=CHUNK_SIZE))
                summary['columns'] = list(first_chunk.columns)

                if 'YYYYMMDD' not in first_chunk.columns:
                    raise ValueError("CSV file must contain 'YYYYMMDD' column")
                if 'LOCATION' not in first_chunk.columns:
                    raise ValueError("CSV file must contain 'LOCATION' column")

                # Reset file pointer for second pass
                chunks = pd.read_csv(filename, chunksize=CHUNK_SIZE)

                for chunk in chunks:
                    # Count total rows
                    summary['total_rows'] += len(chunk)

                    if 'YYYYMMDD' in chunk.columns:
                        # Convert to numeric and handle errors
                        chunk['YYYYMMDD'] = pd.to_numeric(chunk['YYYYMMDD'], errors='coerce')
                        chunk = chunk.dropna(subset=['YYYYMMDD'])

                        if not chunk.empty:
                            chunk_min = int(chunk['YYYYMMDD'].min())
                            chunk_max = int(chunk['YYYYMMDD'].max())

                            if summary['min_date'] is None or chunk_min < summary['min_date']:
                                summary['min_date'] = chunk_min
                            if summary['max_date'] is None or chunk_max > summary['max_date']:
                                summary['max_date'] = chunk_max

                    if 'LOCATION' in chunk.columns:
                        chunk_locations = chunk['LOCATION'].astype(str).unique()
                        summary['locations'].update(chunk_locations)

            except StopIteration:
                raise ValueError("CSV file appears to be empty")

        # Validate summary data
        if summary['min_date'] is None or summary['max_date'] is None:
            raise ValueError("No valid dates found in the YYYYMMDD column")

        if not summary['locations']:
            raise ValueError("No valid locations found in the LOCATION column")

        summary['locations'] = sorted(list(summary['locations']))
        return summary

    except pd.errors.EmptyDataError:
        raise Exception("The file appears to be empty")
    except pd.errors.ParserError:
        raise Exception("Error parsing the file. Please check the file format")
    except ValueError as ve:
        raise Exception(str(ve))
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")



# 添加一个字典来缓存上传文件的完整路径
file_path_cache = {}


# 更新callback以存储完整的文件路径
@du.callback(
    output=[
        Output('output-data-upload', 'children'),
        Output('data-summary', 'data')
    ],
    id='dash-uploader'
)
def update_output(filenames):
    if not filenames:
        raise PreventUpdate

    filename = filenames[0] if isinstance(filenames, list) else filenames

    # 获取完整的文件路径 - dash_uploader的上传回调直接提供完整路径
    file_path = filename

    # 存储完整路径到缓存中，以文件名为键
    base_filename = os.path.basename(file_path)
    file_path_cache[base_filename] = file_path

    print(f"File uploaded: {base_filename} -> {file_path}")

    try:
        summary = process_large_file(file_path)

        # Display data summary
        try:
            min_date_str = str(summary['min_date'])
            max_date_str = str(summary['max_date'])
            min_date_formatted = f"{min_date_str[:4]}-{min_date_str[4:6]}-{min_date_str[6:]}"
            max_date_formatted = f"{max_date_str[:4]}-{max_date_str[4:6]}-{max_date_str[6:]}"

            preview_content = [
                html.H4("Data Summary", className="text-info mb-3"),
                html.Ul([
                    html.Li(f"Total rows: {summary['total_rows']:,}"),
                    html.Li(f"Date range: {min_date_formatted} to {max_date_formatted}"),
                    html.Li(f"Number of unique locations: {len(summary['locations'])}")
                ], className="list-unstyled")
            ]

            return dbc.Card(dbc.CardBody(preview_content)), summary
        except Exception as e:
            error_card = dbc.Card(
                dbc.CardBody([
                    html.H4("Error Processing Data", className="text-danger"),
                    html.P(
                        f"Error formatting data summary: {str(e)}. Please check if the file contains valid date and location information.")
                ])
            )
            return error_card, None

    except Exception as e:
        error_card = dbc.Card(
            dbc.CardBody([
                html.H4("Error", className="text-danger"),
                html.P(str(e))
            ])
        )
        return error_card, None


# 获取完整的文件路径的辅助函数
def get_full_file_path(filename):
    """
    从缓存中获取文件的完整路径，如果没有找到则尝试搜索目录
    """
    base_filename = os.path.basename(filename)
    if base_filename in file_path_cache:
        return file_path_cache[base_filename]
    for root, dirs, files in os.walk(UPLOAD_FOLDER):
        for file in files:
            if file == base_filename:
                full_path = os.path.join(root, file)
                file_path_cache[base_filename] = full_path  # 添加到缓存
                return full_path
    default_path = os.path.join(UPLOAD_FOLDER, base_filename)
    return default_path


# 修改process_large_file函数使用完整路径
def process_large_file(filename):
    """Process large files in chunks"""
    try:
        full_path = filename
        print(f"Processing file: {full_path}")

        summary = {
            'total_rows': 0,
            'columns': [],
            'min_date': None,
            'max_date': None,
            'locations': set()
        }

        ext = os.path.splitext(full_path)[1].lower()

        if ext in ['.xls', '.xlsx']:
            df = pd.read_excel(full_path)
            summary['total_rows'] = len(df)
            summary['columns'] = list(df.columns)
            if 'YYYYMMDD' in df.columns:
                df['YYYYMMDD'] = pd.to_numeric(df['YYYYMMDD'], errors='coerce')
                df = df.dropna(subset=['YYYYMMDD'])
                if not df.empty:
                    summary['min_date'] = int(df['YYYYMMDD'].min())
                    summary['max_date'] = int(df['YYYYMMDD'].max())
            if 'LOCATION' in df.columns:
                summary['locations'].update(df['LOCATION'].astype(str).unique())

        elif ext == '.csv':
            try:
                first_chunk = next(pd.read_csv(full_path, chunksize=CHUNK_SIZE))
                summary['columns'] = list(first_chunk.columns)
                if 'YYYYMMDD' not in first_chunk.columns:
                    raise ValueError("CSV file must contain 'YYYYMMDD' column")
                if 'LOCATION' not in first_chunk.columns:
                    raise ValueError("CSV file must contain 'LOCATION' column")
                chunks = pd.read_csv(full_path, chunksize=CHUNK_SIZE)
                for chunk in chunks:
                    summary['total_rows'] += len(chunk)
                    if 'YYYYMMDD' in chunk.columns:
                        chunk['YYYYMMDD'] = pd.to_numeric(chunk['YYYYMMDD'], errors='coerce')
                        chunk = chunk.dropna(subset=['YYYYMMDD'])
                        if not chunk.empty:
                            chunk_min = int(chunk['YYYYMMDD'].min())
                            chunk_max = int(chunk['YYYYMMDD'].max())
                            if summary['min_date'] is None or chunk_min < summary['min_date']:
                                summary['min_date'] = chunk_min
                            if summary['max_date'] is None or chunk_max > summary['max_date']:
                                summary['max_date'] = chunk_max
                    if 'LOCATION' in chunk.columns:
                        chunk_locations = chunk['LOCATION'].astype(str).unique()
                        summary['locations'].update(chunk_locations)
            except StopIteration:
                raise ValueError("CSV file appears to be empty")

        if summary['min_date'] is None or summary['max_date'] is None:
            raise ValueError("No valid dates found in the YYYYMMDD column")
        if not summary['locations']:
            raise ValueError("No valid locations found in the LOCATION column")

        summary['locations'] = sorted(list(summary['locations']))
        return summary

    except pd.errors.EmptyDataError:
        raise Exception("The file appears to be empty")
    except pd.errors.ParserError:
        raise Exception("Error parsing the file. Please check the file format")
    except ValueError as ve:
        raise Exception(str(ve))
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")


# 修改get_charging_periods函数
charging_cache = {}

def get_charging_periods(filename, charge_threshold_ratio, min_consecutive_hours):
    """
    Get or calculate charging periods from cache
    Returns a dictionary with (location, date) as key and charging hours set as value
    """
    base_filename = os.path.basename(filename)
    if base_filename not in charging_cache:
        full_path = get_full_file_path(filename)
        print(f'Detecting charging periods from: {full_path}')
        charging_df = detect(full_path, charge_threshold_ratio, min_consecutive_hours)
        periods = {}
        for _, row in charging_df.iterrows():
            loc = str(row['LOCATION'])
            date = str(row['YYYYMMDD'])
            period = row['charging_period']
            start_hour = int(period.split('-')[0].replace('R', ''))
            end_hour = int(period.split('-')[1].replace('R', ''))
            hours_range = list(range(start_hour, end_hour + 1))
            key = (loc, date)
            if key not in periods:
                periods[key] = set()
            periods[key].update(hours_range)
        charging_cache[base_filename] = periods
    return charging_cache[base_filename]


# Callback: 更新过滤下拉框的选项
@app.callback(
    Output("location-dropdown", "options"),
    Input("data-summary", "data")
)
def update_location_dropdown(summary):
    if not summary or "locations" not in summary:
        raise PreventUpdate
    options = [{"label": loc, "value": loc} for loc in summary["locations"]]
    return options


# 修改download_charging_periods函数
@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-btn", "n_clicks")],
    [State('dash-uploader', 'fileNames'),
     State("charge-threshold-ratio", "value"),
     State("min-consecutive-hours", "value")
     ]
)
def export_charging_periods(n_clicks, filenames, charge_threshold_ratio, min_consecutive_hours):
    if not n_clicks or not filenames:
        raise PreventUpdate

    filename = filenames[0] if isinstance(filenames, list) else filenames
    base_filename = os.path.basename(filename)
    charging_periods = get_charging_periods(filename, charge_threshold_ratio, min_consecutive_hours)

    data = []
    for (location, date), hours in charging_periods.items():
        data.append([location, date, hours])

    df = pd.DataFrame(data, columns=["LOCATION", "YYYYMMDD", "CHARGING_HOUR"])
    return dcc.send_data_frame(df.to_csv, "charging_periods.csv", index=False)

# 新增回调：根据 data-summary 更新日期选择器的默认值和允许范围
@app.callback(
    [Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date'),
     Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed')],
    Input('data-summary', 'data')
)
def update_date_picker(summary):
    if not summary or "min_date" not in summary or "max_date" not in summary:
        raise PreventUpdate

    # summary 中存储的日期格式为整数形式，例如 20220101
    min_date_int = summary['min_date']
    max_date_int = summary['max_date']

    # 转换为 "YYYY-MM-DD" 格式的字符串
    min_date_str = f"{str(min_date_int)[:4]}-{str(min_date_int)[4:6]}-{str(min_date_int)[6:]}"
    max_date_str = f"{str(max_date_int)[:4]}-{str(max_date_int)[4:6]}-{str(max_date_int)[6:]}"

    # 如果希望默认只选择数据的第一天，将 start_date 和 end_date 都设为第一天
    return min_date_str, min_date_str, min_date_str, max_date_str


@app.callback(
    Output("line-chart", "figure"),
    [Input("location-dropdown", "value"),
     Input("date-range-picker", "start_date"),
     Input("date-range-picker", "end_date")],
    [State('dash-uploader', 'fileNames'),
     State("charge-threshold-ratio", "value"),
     State("min-consecutive-hours", "value")
     ]
)
def update_line_chart(selected_locations, start_date, end_date, filenames, charge_threshold_ratio,
                      min_consecutive_hours):
    if not all([selected_locations, start_date, end_date, filenames]):
        raise PreventUpdate

    filename = filenames[0] if isinstance(filenames, list) else filenames
    full_path = get_full_file_path(filename)
    print(f"📌 Reading data for chart from: {full_path}")

    start_date_int = int(datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d'))
    end_date_int = int(datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d'))
    charging_periods = get_charging_periods(filename, charge_threshold_ratio, min_consecutive_hours)

    print(f"🔍 Total detected charging events: {len(charging_periods)}")

    fig = go.Figure()
    ext = os.path.splitext(full_path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        chunks = [pd.read_excel(full_path)]
    else:
        chunks = pd.read_csv(full_path, chunksize=CHUNK_SIZE)

    r_columns = [f"R{i}" for i in range(1, 25)]
    color_sequence = px.colors.qualitative.Plotly
    hours = list(range(1, 25))

    for chunk in chunks:
        if 'YYYYMMDD' in chunk.columns:
            chunk['YYYYMMDD'] = pd.to_numeric(chunk['YYYYMMDD'], errors='coerce')
        if 'LOCATION' in chunk.columns:
            chunk['LOCATION'] = chunk['LOCATION'].astype(str)

        mask = (
                chunk["LOCATION"].isin([str(loc) for loc in selected_locations]) &
                (chunk["YYYYMMDD"] >= start_date_int) &
                (chunk["YYYYMMDD"] <= end_date_int)
        )
        filtered = chunk[mask]

        if filtered.empty:
            print("⚠️ No data after filtering! Check filters.")
            continue

        if len(filtered) > MAX_POINTS_PER_TRACE:
            filtered = filtered.sample(n=MAX_POINTS_PER_TRACE)

        for _, row in filtered.iterrows():
            loc = row["LOCATION"]
            date = str(row["YYYYMMDD"])
            date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:]}"

            y_values = row[r_columns].tolist()
            base_color = color_sequence[hash(str(loc)) % len(color_sequence)]
            charging_hours = charging_periods.get((loc, date), set())

            print(f"📌 User: {loc}, Date: {date}, Charging Hours: {charging_hours}")

            fig.add_trace(go.Scatter(
                x=hours,
                y=y_values,
                mode="lines+markers",
                name=f"User {loc}, Date {date_formatted}",
                marker=dict(
                    color=[base_color if h not in charging_hours else 'rgba(0,0,0,0)' for h in hours]
                ),
                line=dict(
                    color=base_color
                ),
                showlegend=True
            ))

            # 画红色点标记充电时段
            charging_data = [(h, v) for h, v in zip(hours, y_values) if h in charging_hours]
            if charging_data:
                charging_x, charging_y = zip(*sorted(charging_data))
                print(f"🔴 Marking Charging Points: {charging_x}")

                fig.add_trace(go.Scatter(
                    x=charging_x,
                    y=charging_y,
                    mode="markers",
                    name="Charging Periods",
                    marker=dict(color='red', size=8),
                    showlegend=False
                ))

    fig.update_layout(
        title={
            'text': "Hourly Electricity Consumption",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Hour (1-24)",
        yaxis_title="Electricity Consumption (KW)",
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=60, r=30, t=60, b=60)
    )
    return fig


@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    print(f"📢 Page changed: {pathname}")  # ✅ 调试信息

    if pathname == "/stats":
        print("✅ Rendering Stats Page!")  # ✅ 确保被触发
        return create_stats_layout()  # 返回统计页面

    print("✅ Rendering Home Page!")  # ✅ 确保主页正常加载
    return main_layout  # 返回主页


# @app.callback(
#     [Output("location-prob-table", "data"),
#      Output("high-prob-ratio", "children")],
#     [Input("calculate-btn", "n_clicks")],  # **📌 只有点击按钮才计算**
#     [State("window-size-slider", "value"),
#      State("threshold-input", "value")]
# )

@app.callback(
    [Output("location-prob-table", "data"),
     Output("accuracy-output", "children"),
     Output("precision-output", "children"),
     Output("recall-output", "children"),
     Output("f1-score-output", "children")],
    [Input("calculate-btn", "n_clicks")],
    [State("window-size-slider", "value"),
     State("threshold-input", "value")]
)

# def update_location_prob(n_clicks, window_size, threshold):
#     if not n_clicks:
#         return [], "Waiting..."  # **初始状态**
#
#     print(f"📢 Button clicked! Window Size: {window_size}, Threshold: {threshold}")  # ✅ 调试信息
#
#     try:
#         prob_df, high_prob_ratio = process_weekly_csv(window_size, threshold)
#         if prob_df is None or prob_df.empty:
#             print("⚠️ No data loaded from process_weekly_csv()!")
#             return [], "0%"
#
#         print(f"📊 Loaded {len(prob_df)} records!")  # ✅ 确保数据加载成功
#         return prob_df.to_dict("records"), high_prob_ratio  # **返回表格数据 & 占比**
#
#     except Exception as e:
#         print(f"❌ Error in update_location_prob: {e}")
#         return [], "0%"

def update_location_prob(n_clicks, window_size, threshold):
    if not n_clicks:
        return [], "Waiting...", "Waiting...", "Waiting...", "Waiting..."

    print(f"📢 Button clicked! Window Size: {window_size}, Threshold: {threshold}")

    try:
        merged_df, accuracy, precision, recall, f1 = process_weekly_csv(window_size, threshold)
        if merged_df is None or merged_df.empty:
            print("⚠️ No data loaded from process_weekly_csv()!")
            return [], "0", "0", "0", "0"

        print(f"📊 Loaded {len(merged_df)} records!")
        return merged_df.to_dict(
            "records"), f"Accuracy: {accuracy:.2f}%", f"Precision: {precision:.2f}%", f"Recall: {recall:.2f}%", f"F1 Score: {f1:.2f}%"

    except Exception as e:
        print(f"❌ Error in update_location_prob: {e}")
        return [], "0", "0", "0", "0"


if __name__ == '__main__':
    app.run(debug=False)