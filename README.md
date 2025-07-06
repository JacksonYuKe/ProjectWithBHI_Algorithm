# EV Charging Detection Dashboard

A comprehensive dashboard application for analyzing electricity consumption data to identify potential Electric Vehicle (EV) charging patterns and locations. The system combines statistical anomaly detection with machine learning approaches to detect consumption patterns consistent with EV charging behavior.

## Features

- **Statistical Anomaly Detection**: Baseline consumption analysis to identify unusual electricity usage patterns
- **Machine Learning Detection**: Random Forest model with seasonal baselines and feature engineering
- **Interactive Dashboard**: Web-based interface for data upload, visualization, and analysis
- **Performance Analytics**: Evaluation tools with accuracy, precision, recall, and F1 score metrics
- **Batch Processing**: Parameter optimization through sliding window analysis
- **Export Capabilities**: Results export functionality for further analysis

## Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| `main.py` | Dash web application entry point with multi-page dashboard |
| `Ev_Detection.py` | Primary statistical anomaly detection algorithm |
| `Random_forest.py` | ML-enhanced detection with seasonal baselines |
| `stats_page.py` | Performance analysis and evaluation module |
| `sliding_window.py` | Batch processing for parameter optimization |

## Installation

### Prerequisites

- Python 3.9+
- Required packages (install via pip):

```bash
pip install dash dash-uploader dash-bootstrap-components
pip install pandas numpy plotly scikit-learn
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ProjectWithBHI_Algorithm
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt  # If requirements.txt exists
```

## Usage

### Running the Dashboard

Start the main application:
```bash
python main.py
```

The dashboard will be available at `http://127.0.0.1:8050/`

### Running Individual Modules

Execute specific algorithms directly:

```bash
# Run Random Forest algorithm
python Random_forest.py

# Run sliding window batch processing
python sliding_window.py
```

## Data Format

The application expects electricity consumption data in CSV format with the following structure:

- **Column 0**: `LOCATION` - Unique identifier for each location/household
- **Column 1**: `YYYYMMDD` - Date in YYYYMMDD format
- **Columns 4-27**: `R1-R24` - Hourly consumption readings (24 hours)
- **Optional**: `# of Chargers` - Ground truth data for validation

### Example Data Structure
```csv
LOCATION,YYYYMMDD,R1,R2,R3,...,R24
LOC001,20240101,2.5,2.3,2.1,...,3.2
LOC001,20240102,2.4,2.2,2.0,...,3.1
```

## Algorithm Parameters

### Statistical Detection (`Ev_Detection.py`)

- `charge_threshold_ratio`: Minimum percentage of charger power to detect (default: 0.6)
- `min_consecutive_hours`: Minimum consecutive hours for valid EV charging session
- `ev_charger_kw`: Typical EV charger power in kilowatts (default: 7.0)

### Machine Learning Detection (`Random_forest.py`)

- Seasonal baseline calculation
- Time-of-day pattern features
- Dynamic threshold prediction
- Cross-validation for model optimization

## Performance Metrics

The system evaluates detection algorithms using:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1 Score**: Harmonic mean of precision and recall

## File Structure

```
ProjectWithBHI_Algorithm/
├── main.py                 # Main dashboard application
├── Ev_Detection.py         # Statistical anomaly detection
├── Random_forest.py        # ML-enhanced detection
├── stats_page.py          # Performance analysis
├── sliding_window.py      # Batch processing
├── CLAUDE.md              # Development guidelines
└── README.md              # This file
```

## Configuration

### Data Paths

**Note**: The current implementation has hardcoded paths in `Random_forest.py` and `stats_page.py` pointing to:
```
/Users/jackson/BHI/Decrypted_Files/weekly_csv_files_test
```

These paths may need to be updated based on your data location.

### Dashboard Settings

- Upload folder: `uploads/`
- Chunk size: 100,000 rows
- Maximum points per trace: 5,000

## Development

### IDE Configuration

The project includes PyCharm configuration (`.idea/` folder) with Python 3.9 and virtual environment setup.

### Key Dependencies

- **Dash Framework**: Web application framework
- **Pandas/NumPy**: Data processing and analysis
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning components
- **Dash Bootstrap Components**: UI components

## Contributing

1. Follow the existing code style and patterns
2. Test changes with sample data
3. Update documentation as needed
4. Ensure all algorithms maintain performance metrics

## License

[Add your license information here]

## Support

For issues or questions, please refer to the project documentation or contact the development team.

---

*This dashboard provides a comprehensive solution for EV charging pattern detection, combining statistical and machine learning approaches to identify potential charging locations from electricity consumption data.*