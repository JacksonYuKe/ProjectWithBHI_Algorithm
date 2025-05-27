import os
import pandas as pd

# Define the directory containing CSV files
input_dir = "/Users/jacson/Desktop/BHI/Decrypted_Files/weekly_csv_files"

# Iterate through all CSV files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(input_dir, filename)

        # Read the CSV file
        df = pd.read_csv(file_path, header=None)  # Read without setting header

        # Modify D1 (column index 3, row index 0)
        df.iloc[0, 3] = "# of Chargers"

        # Save the modified file (overwrite)
        df.to_csv(file_path, index=False, header=False)

        print(f"Updated D1 in: {filename}")

print("✅ All CSV files updated successfully!")
