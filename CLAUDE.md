# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an EV (Electric Vehicle) Charging Detection Dashboard that analyzes electricity consumption data to identify potential EV charging patterns and locations. The application uses both statistical and machine learning approaches to detect anomalous electricity usage patterns consistent with EV charging.

## Architecture

### Core Components

1. **main.py** - Dash web application entry point
   - Multi-page dashboard using Dash Bootstrap Components
   - Handles file uploads, data visualization, and result exports

2. **Ev_Detection.py** - Primary detection algorithm
   - Statistical anomaly detection based on baseline consumption
   - Parameters: charge_threshold_ratio, min_consecutive_hours, ev_charger_kw

3. **Random_forest.py** - ML-enhanced detection
   - Implements seasonal baselines and Random Forest models
   - Features engineering for time-of-day patterns
   - Dynamic threshold prediction

4. **stats_page.py** - Performance analysis module
   - Evaluates detection algorithms against ground truth data
   - Calculates accuracy, precision, recall, and F1 scores

5. **sliding_window.py** - Batch processing for parameter optimization
   - Tests multiple parameter combinations
   - Generates performance comparison tables

## Development Commands

### Running the Application
```bash
python main.py
```
The application will start on http://127.0.0.1:8050/

### Running Individual Modules
```bash
# Run the Random Forest algorithm directly
python Random_forest.py

# Run sliding window batch processing
python sliding_window.py
```

## Data Format

The application expects electricity consumption data in CSV format with:
- Column 0: LOCATION (identifier)
- Column 1: YYYYMMDD (date)
- Columns 4-27: R1-R24 (hourly consumption readings)
- Optional: "# of Chargers" column for ground truth validation

## Key Dependencies

- dash and related packages (dash-uploader, dash-bootstrap-components)
- pandas, numpy for data processing
- plotly for visualizations
- scikit-learn for machine learning components

## Important Notes

- The project currently has hardcoded paths in Random_forest.py and stats_page.py pointing to `/Users/jackson/BHI/Decrypted_Files/weekly_csv_files_test`
- No requirements.txt exists - dependencies must be inferred from imports
- The .idea folder indicates PyCharm usage with Python 3.9/.venv configuration