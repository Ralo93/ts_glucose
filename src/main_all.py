import requests
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from datetime import timedelta
import glob
import os


def load_and_concatenate_files(file_path, file_pattern, start_date):
    # Combine the path and file pattern to create the full search pattern
    full_pattern = os.path.join(file_path, file_pattern)

    # Get all files matching the pattern (e.g., p12_*.csv)
    files = sorted(glob.glob(full_pattern))

    # Initialize an empty list to store dataframes
    df_list = []

    # Convert the start_date to a datetime object
    current_date = pd.to_datetime(start_date)

    # Loop through each file, load it into a dataframe, and adjust the 'time' column
    for i, file in enumerate(files):
        # Load the CSV into a dataframe
        df = pd.read_csv(file)

        # Ensure 'time' is a datetime object
        df['time'] = pd.to_datetime(df['time'])

        # Calculate the day adjustment based on the current file index
        day_delta = timedelta(days=i)

        # Add the day delta to the 'time' column
        df['time'] = df['time'] + day_delta

        # Append the updated dataframe to the list
        df_list.append(df)

    # Concatenate all dataframes in the list
    concatenated_df = pd.concat(df_list, ignore_index=True)

    return concatenated_df


# Example usage:
file_path = r'C:\Users\Administrator\Desktop\raphi_other\repositories\ts\data\processed\all_days_p12'
# Assuming you want to start from '2024-10-02' and files match the pattern 'p12_*.csv'
concatenated_df = load_and_concatenate_files(file_path, 'p12_*_day.csv', '2024-10-03')

# Save the concatenated dataframe to a new CSV if needed
concatenated_df.to_csv(r"C:\Users\Administrator\Desktop\raphi_other\repositories\ts\data/processed/concat/concatenated.csv", index=False)
