import pandas as pd
from stats_page import process_weekly_csv
import traceback

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def batch_process_weekly_csv():
    # window_sizes = [4]
    # thresholds = [0.5, 1]
    # window_sizes = [2,3]
    # thresholds = [0.5,1,1.5,2]
    window_sizes = [2]
    thresholds = [2.5]

    results = []

    for window_size in window_sizes:
        for threshold in thresholds:
            try:
                print(f"\nProcessing with window_size={window_size}, threshold={threshold}")
                _, accuracy, precision, recall, f1 = process_weekly_csv(window_size, threshold)
                results.append({
                    "Window Size": window_size,
                    "Threshold": threshold,
                    "Accuracy": f"{accuracy:.2f}%",
                    "Precision": f"{precision:.2f}%",
                    "Recall": f"{recall:.2f}%",
                    "F1 Score": f"{f1:.2f}%"
                })
            except Exception as e:
                print(f"❌ Error processing window_size={window_size}, threshold={threshold}:")
                print(traceback.format_exc())
                results.append({
                    "Window Size": window_size,
                    "Threshold": threshold,
                    "Accuracy": "Error",
                    "Precision": "Error",
                    "Recall": "Error",
                    "F1 Score": "Error"
                })

    results_df = pd.DataFrame(results)
    print("\nSummary of results:")
    print(results_df)
    return results_df


if __name__ == "__main__":
    batch_process_weekly_csv()
