import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the folder where the CSV files are located
folder_path = '/Users/jacson/BHI/Decrypted_Files/weekly_csv_files_test'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Create an empty list to store the dataframes
dfs_with_chargers = []
dfs_without_chargers = []

# Load each CSV file into a pandas dataframe and categorize based on "# of Chargers"
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    # Split into two categories based on "# of Chargers"
    with_chargers = df[df['# of Chargers'].notna()]
    without_chargers = df[df['# of Chargers'].isna()]

    # Append to respective lists
    dfs_with_chargers.append(with_chargers)
    dfs_without_chargers.append(without_chargers)

# Combine data from all CSVs for both categories
df_with_chargers = pd.concat(dfs_with_chargers, ignore_index=True)
df_without_chargers = pd.concat(dfs_without_chargers, ignore_index=True)

# Display the results to the user
try:
    import ace_tools as tools
    
    tools.display_dataframe_to_user(name="Locations with Chargers", dataframe=df_with_chargers)
    tools.display_dataframe_to_user(name="Locations without Chargers", dataframe=df_without_chargers)
except ImportError:
    # Alternative display if ace_tools is not available
    print("\n--- Locations with Chargers ---")
    print(f"Shape: {df_with_chargers.shape}")
    print(df_with_chargers.head())
    
    print("\n--- Locations without Chargers ---")
    print(f"Shape: {df_without_chargers.shape}")
    print(df_without_chargers.head())

# Feature Engineering Function
def extract_electricity_features(df):
    """
    Extract useful features from hourly electricity consumption data.
    
    Args:
        df: DataFrame containing electricity consumption data
    
    Returns:
        DataFrame with engineered features
    """
    features = pd.DataFrame()
    
    # Assuming 'Power_kW' or similar column exists for electricity consumption
    # If the column name is different, replace it with the actual column name
    power_col = [col for col in df.columns if 'power' in col.lower() or 'kw' in col.lower() or 'consumption' in col.lower()]
    if not power_col:
        print("Warning: Could not detect power/consumption column. Please specify the column manually.")
        return None
        
    power_col = power_col[0]  # Use the first matching column
    
    # Basic statistical features
    features['mean_consumption'] = df.groupby('Location ID')[power_col].mean()
    features['max_consumption'] = df.groupby('Location ID')[power_col].max()
    features['min_consumption'] = df.groupby('Location ID')[power_col].min()
    features['std_consumption'] = df.groupby('Location ID')[power_col].std()
    
    # Time-based patterns
    # Assuming there's a timestamp column
    time_col = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if time_col:
        time_col = time_col[0]
        df['timestamp'] = pd.to_datetime(df[time_col])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Average consumption by hour (to capture daily patterns)
        pivot_hour = df.groupby(['Location ID', 'hour'])[power_col].mean().unstack()
        for hour in pivot_hour.columns:
            features[f'hour_{hour}_avg'] = pivot_hour[hour]
            
        # Average consumption by day of week
        pivot_day = df.groupby(['Location ID', 'day_of_week'])[power_col].mean().unstack()
        for day in pivot_day.columns:
            features[f'day_{day}_avg'] = pivot_day[day]
            
        # Workday vs weekend difference
        weekday_mask = df['day_of_week'] < 5  # 0-4 are weekdays (Mon-Fri)
        weekend_mask = df['day_of_week'] >= 5  # 5-6 are weekend (Sat-Sun)
        
        weekday_avg = df[weekday_mask].groupby('Location ID')[power_col].mean()
        weekend_avg = df[weekend_mask].groupby('Location ID')[power_col].mean()
        features['weekday_weekend_diff'] = weekday_avg - weekend_avg
        
        # Evening peaks (typical EV charging time)
        evening_mask = (df['hour'] >= 17) & (df['hour'] <= 23)
        features['evening_avg'] = df[evening_mask].groupby('Location ID')[power_col].mean()
        features['evening_max'] = df[evening_mask].groupby('Location ID')[power_col].max()
        
    # Calculate peak-to-average ratio
    features['peak_to_avg_ratio'] = features['max_consumption'] / features['mean_consumption']
    
    # Fill NaN values
    features = features.fillna(0)
    
    return features

# Prepare data for machine learning
def prepare_data_for_ml():
    # Create target variable: 1 if location has chargers, 0 if not
    locations_with_chargers = set(df_with_chargers['Location ID'].unique())
    
    # Extract features from both datasets
    print("Extracting features from locations with chargers...")
    features_with_chargers = extract_electricity_features(df_with_chargers)
    
    print("Extracting features from locations without chargers...")
    features_without_chargers = extract_electricity_features(df_without_chargers)
    
    if features_with_chargers is None or features_without_chargers is None:
        print("Error: Could not extract features. Check your data format.")
        return None, None
    
    # Add target label
    features_with_chargers['has_chargers'] = 1
    features_without_chargers['has_chargers'] = 0
    
    # Combine datasets
    all_features = pd.concat([features_with_chargers, features_without_chargers])
    
    # Separate features and target
    X = all_features.drop('has_chargers', axis=1)
    y = all_features['has_chargers']
    
    return X, y

# Train and evaluate model
def train_and_evaluate_model(X, y):
    if X is None or y is None:
        print("Error: No valid data for training.")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Parameter grid for optimization
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Grid search for best parameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Chargers', 'Has Chargers'],
                yticklabels=['No Chargers', 'Has Chargers'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('/Users/jacson/PycharmProjects/BHI/confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Feature importance
    if hasattr(best_model['classifier'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model['classifier'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('/Users/jacson/PycharmProjects/BHI/feature_importance.png')
        print("Feature importance plot saved as 'feature_importance.png'")
    
    return best_model

# Function to make predictions for new locations
def predict_charger_presence(model, new_location_data):
    """
    Predict whether a new location has EV chargers.
    
    Args:
        model: Trained machine learning model
        new_location_data: DataFrame with features for new location(s)
    
    Returns:
        Predictions (1 = has chargers, 0 = no chargers)
    """
    features = extract_electricity_features(new_location_data)
    predictions = model.predict(features)
    
    results = pd.DataFrame({
        'Location ID': features.index,
        'Has Chargers': predictions
    })
    
    return results

# Run the machine learning pipeline if this file is run directly
if __name__ == "__main__":
    # Display the initial data first
    try:
        import ace_tools as tools

        tools.display_dataframe_to_user(name="Locations with Chargers", dataframe=df_with_chargers)
        tools.display_dataframe_to_user(name="Locations without Chargers", dataframe=df_without_chargers)
    except ImportError:
        # Alternative display if ace_tools is not available
        print("\n--- Locations with Chargers ---")
        print(f"Shape: {df_with_chargers.shape}")
        print(df_with_chargers.head())

        print("\n--- Locations without Chargers ---")
        print(f"Shape: {df_without_chargers.shape}")
        print(df_without_chargers.head())

    print("\n=== Starting EV Charger Prediction Model ===")

    # Prepare data
    print("\nPreparing data for machine learning...")
    X, y = prepare_data_for_ml()

    # Train and evaluate model
    print("\nTraining model...")
    model = train_and_evaluate_model(X, y)

    if model:
        print("\nModel training complete. You can now use this model to predict whether new locations have EV chargers.")
        print("Use the predict_charger_presence() function with new location data to make predictions.")
