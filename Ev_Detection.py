import pandas as pd

def detect(filename, charge_threshold_ratio, min_consecutive_hours):
    # Parameter settings
    threshold_multiplier = 1.5  # Multiplier for detecting statistical anomalies (adjustable)
    ev_charger_kw = 7.0  # Typical EV charger power (kilowatts)
    min_extra_kw = 2.0  # Minimum extra load (kilowatts), i.e., extra load must exceed 2 kW
    full_charge_threshold_kw = charge_threshold_ratio * ev_charger_kw  # Must observe at least 60% of charger power during charging (e.g., 4.2 kW)

    df = pd.read_csv(filename)
    # Assumes filtered_df has loaded CSV data containing the following fields:
    # - 'location_id': household identifier
    # - 'YYYYMMDD': date (format: YYYYMMDD)
    # - 'R1' to 'R24': hourly consumption for each day

    # Define list of hour columns
    hours = [f'R{i}' for i in range(1, 25)]

    results = []

    # Group by 'LOCATION' (each household) and analyze data for each household
    for loc, group in df.groupby('LOCATION'):
        # Calculate baseline consumption for each hour using household data (median and standard deviation)
        baseline_mean = group[hours].median()
        baseline_std = group[hours].std()

        # Iterate through each day's data for this household
        for idx, row in group.iterrows():
            day = row['YYYYMMDD']
            abnormal_hours = []  # Record hours that exceed anomaly threshold for the day (as numbers, e.g., R5 becomes 5)

            # Check each hour
            for hour in hours:
                # Calculate statistical threshold for this hour
                threshold_stat = baseline_mean[hour] + threshold_multiplier * baseline_std[hour]
                # Also require extra load to be at least min_extra_kw above baseline
                threshold_kw = baseline_mean[hour] + min_extra_kw
                # Use the larger of the two thresholds
                threshold_value = max(threshold_stat, threshold_kw)

                if row[hour] > threshold_value:
                    abnormal_hours.append(int(hour[1:]))  # Convert 'R5' to number 5

            # Check if there's at least one consecutive period of 2 hours or more with anomalies
            if abnormal_hours:
                abnormal_hours = sorted(abnormal_hours)
                consecutive_blocks = []
                current_block = [abnormal_hours[0]]

                for hour_val in abnormal_hours[1:]:
                    if hour_val == current_block[-1] + 1:
                        current_block.append(hour_val)
                    else:
                        if len(current_block) >= min_consecutive_hours:
                            consecutive_blocks.append(current_block.copy())
                        current_block = [hour_val]
                # Check the last period
                if len(current_block) >= min_consecutive_hours:
                    consecutive_blocks.append(current_block.copy())

                # Further evaluate each consecutive anomaly period
                for block in consecutive_blocks:
                    # Calculate extra load for each hour in the period (increase relative to baseline)
                    block_extra = []
                    for h in block:
                        col = f'R{h}'
                        extra = row[col] - baseline_mean[col]
                        block_extra.append(extra)
                    max_extra = max(block_extra)

                    # Only consider this period as a potential EV charging event
                    # if at least one hour's extra load reaches or exceeds 60% of typical EV charger power (e.g., 4.2 kW)
                    if max_extra >= full_charge_threshold_kw:
                        time_interval = f"R{block[0]}-R{block[-1]}"
                        results.append({
                            'LOCATION': loc,
                            'YYYYMMDD': day,
                            'charging_period': time_interval,
                            'max_extra_kw': round(max_extra, 2)
                        })
    return pd.DataFrame(results)
