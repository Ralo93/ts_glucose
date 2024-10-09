import pandas as pd
import numpy as np

class BloodGlucosePreprocessing:

    def __init__(self, file_path):
        # Load the dataset
        self.df = pd.read_csv(file_path, low_memory=False)
        print("Initial Data Info:")
        print(self.df.info())

    def downcast_floats(self):
        # Downcast float64 columns to float32 to save memory
        float64_cols = self.df.select_dtypes(include=['float64']).columns
        self.df[float64_cols] = self.df[float64_cols].astype(np.float32)
        print(f"Downcasted {len(float64_cols)} float64 columns to float32.")

    def convert_object_to_category(self):
        # Convert object columns to category to save memory
        object_cols = self.df.select_dtypes(include=['object']).columns
        self.df[object_cols] = self.df[object_cols].astype('category')
        print(f"Converted {len(object_cols)} object columns to category.")

    def filter_patient_data(self, patient, bg_only=False, bg_or_plus_one=True):
        # Filter the data for the specified patient
        self.df = self.df[self.df['p_num'] == patient].copy()

        if bg_only:
            self.df = self.df[['time', 'bg-0:00']]
        
        if bg_or_plus_one:
            self.df.rename(columns={'bg-0:00': 'bg'}, inplace=True)
            print("Renamed 'bg-0:00' to 'bg'.")
        print(f"Filtered patient {patient}, shape: {self.df.shape}")

    def load_patient_csv(self, file_path):
        # Load the preprocessed patient data
        self.df = pd.read_csv(file_path)
        print(f"Loaded patient data from {file_path}")


    def interpolate_bg(self):

        if 'time' in self.df.columns:
            self.df['time'] = pd.to_datetime(self.df['time'])  # Convert to datetime if not already
            self.df = self.df.set_index('time')  # Set 'time' as index
        # Create a continuous range of time (hourly frequency)
        full_time_range = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='h')
        # Reindex the DataFrame to this full time range (inserting missing times)
        self.df= self.df.reindex(full_time_range)

        self.df = self.df.reset_index().rename(columns={'index': 'time'})

        # Display the result

        # Interpolate the 'bg-0:00' column for missing values
        if 'bg' in self.df.columns:
            self.df['bg'] = self.df['bg'].interpolate(method='linear')
            print(f"Interpolated missing 'bg' values.")
        else:
            print("'bg-0:00' column not found for interpolation.")

        self.df = self.df.set_index('time')
        
         
        # assert self.df[bg_columns].isnull().sum().sum() == 0, "There are still missing values after interpolation."
        

    def filter_hourly_data(self):
        # Ensure 'time' column is in datetime format and filter hourly data
        self.df['time'] = pd.to_datetime(self.df['time'], format='%H:%M:%S')
        self.df = self.df[self.df['time'].dt.minute == 0].reset_index(drop=True)
        print(f"Filtered hourly data, shape: {self.df.shape}")

    def adjust_time_with_date(self, start_date):
        # Ensure the 'time' column is in datetime format
        if 'time' in self.df.columns:
            self.df['time'] = pd.to_datetime(self.df['time'], errors='coerce')  # Convert to datetime, handle errors
        else:
            raise ValueError("'time' column not found in the DataFrame.")
        
        # Set the starting date
        current_date = pd.Timestamp(start_date)
        updated_times = []

        # Track the previous hour to determine when the time rolls over to the next day
        previous_hour = None

        # Iterate through the rows and update the time with the correct date
        for idx, row in self.df.iterrows():
            current_hour = row['time'].hour  # Now works since 'time' is datetime

            # If the current hour is less than the previous one, it means we've moved to the next day
            if previous_hour is not None and current_hour < previous_hour:
                current_date += pd.Timedelta(days=1)

            # Combine current_date with the time from the row
            new_time = pd.Timestamp.combine(current_date, row['time'].time())
            updated_times.append(new_time)

            # Update the previous hour
            previous_hour = current_hour

        # Ensure no duplicates in the time column before setting it as index
        if len(updated_times) != len(set(updated_times)):
            raise ValueError("Duplicate times detected after adjusting dates")

        # Update the 'time' column with the new datetime values and set as index
        self.df['time'] = updated_times
        self.df.set_index('time', inplace=True)
        print("Time column adjusted with date and set as index.")


    def save_patient_data(self, patient, save_path):
        # Save the processed data to a CSV file
        self.df.sort_index(inplace=True)
        self.df.to_csv(f'{save_path}/{patient}.csv', index=True)
        print(f"Saved cleaned data for patient {patient} to {save_path}/{patient}.csv")
