import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ✅ 读取 Excel 文件
FILE_PATH = "/Users/jacson/PycharmProjects/BHI/processed_peak_periods.xlsx"  # 你的文件路径
df = pd.read_excel(FILE_PATH, engine="openpyxl")

# ✅ 计算每个小时的聚类结果
hourly_clusters = df.groupby("Hour")["Cluster"].agg(lambda x: x.value_counts().idxmax())  # 统计每小时的主要类别

# ✅ 获取每个类别的小时范围
off_peak_hours = sorted(hourly_clusters[hourly_clusters == 0].index.tolist())  # 低谷时段
mid_peak_hours = sorted(hourly_clusters[hourly_clusters == 1].index.tolist())  # 中间时段
on_peak_hours = sorted(hourly_clusters[hourly_clusters == 2].index.tolist())  # 高峰时段

# ✅ 打印结果
print(f"🌙 Off-Peak Hours (低谷时段): {off_peak_hours}")
print(f"⏳ Mid-Peak Hours (中间时段): {mid_peak_hours}")
print(f"🔥 On-Peak Hours (高峰时段): {on_peak_hours}")
