import pandas as pd

def detect(filename, charge_threshold_ratio, min_consecutive_hours):
    # Parameter settings
    threshold_multiplier = 1.5  # Multiplier for detecting statistical anomalies (adjustable)
    ev_charger_kw = 7.0  # Typical EV charger power in kilowatts
    min_extra_kw = 2.0  # Minimum additional load in kilowatts, i.e., the extra load must exceed 2 kW
    full_charge_threshold_kw = charge_threshold_ratio * ev_charger_kw  # At least 60% of the charger power must be observed during charging (e.g., 4.2 kW)

    df = pd.read_csv(filename)
    # Assume that filtered_df has loaded the CSV data, containing the fields:
    # - 'location_id': household identifier
    # - 'YYYYMMDD': date (format: YYYYMMDD)
    # - 'R1' to 'R24': hourly consumption for each day

    # Define the list of hour columns
    hours = [f'R{i}' for i in range(1, 25)]

    results = []

    # Group by 'LOCATION' (each household) and analyze each household's data
    for loc, group in df.groupby('LOCATION'):
        # Calculate the baseline consumption for each hour (median and standard deviation) using data for the household
        baseline_mean = group[hours].median()
        baseline_std = group[hours].std()

        # Iterate over each day's data for the household
        for idx, row in group.iterrows():
            day = row['YYYYMMDD']
            abnormal_hours = []  # Record hours (as numbers, e.g., 5 for R5) that exceed the anomaly threshold on this day

            # Check each hour
            for hour in hours:
                # Calculate the statistical threshold for the hour
                threshold_stat = baseline_mean[hour] + threshold_multiplier * baseline_std[hour]
                # Also require that the extra load is at least min_extra_kw higher than the baseline
                threshold_kw = baseline_mean[hour] + min_extra_kw
                # Use the larger of the two thresholds
                threshold_value = max(threshold_stat, threshold_kw)

                if row[hour] > threshold_value:
                    abnormal_hours.append(int(hour[1:]))  # Convert 'R5' to the number 5

            # Check if there is at least one continuous block of 2 or more abnormal hours
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
                # Check the final block
                if len(current_block) >= min_consecutive_hours:
                    consecutive_blocks.append(current_block.copy())

                # Further evaluate each consecutive abnormal block
                for block in consecutive_blocks:
                    # Compute the additional load for each hour in the block (the increase compared to the baseline)
                    block_extra = []
                    for h in block:
                        col = f'R{h}'
                        extra = row[col] - baseline_mean[col]
                        block_extra.append(extra)
                    max_extra = max(block_extra)

                    # Only consider this block as a potential EV charging event if at least one hour has an extra load
                    # reaching or exceeding 60% of the typical EV charger power (e.g., 4.2 kW)
                    if max_extra >= full_charge_threshold_kw:
                        time_interval = f"R{block[0]}-R{block[-1]}"
                        results.append({
                            'LOCATION': loc,
                            'YYYYMMDD': day,
                            'charging_period': time_interval,
                            'max_extra_kw': round(max_extra, 2)
                        })
    return pd.DataFrame(results)
