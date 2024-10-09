import pandas as pd
from datetime import timedelta


def remove_redundancy(df):
    """
    Function to remove redundancy in a DataFrame by ensuring unique 'time' values 
    and preserving as much information in the 'bg' column as possible.

    Parameters:
    df (DataFrame): The input DataFrame containing 'time' and 'bg' columns.

    Returns:
    DataFrame: The processed DataFrame with unique 'time' values and cleaned 'bg' values.
    """
    # Sort the DataFrame by time to ensure chronological order
    df = df.sort_values(by='time')

    # Drop duplicate 'time' entries, prioritizing non-NaN values in 'bg'
    df = df.drop_duplicates(subset='time', keep='first')

    # Forward-fill NaN values in 'bg' to preserve as much data as possible
    df['bg'] = df['bg'].fillna(method='ffill')

    # Alternatively, you can use backward fill if desired (use 'bfill' instead of 'ffill')
    # df['bg'] = df['bg'].fillna(method='bfill')

    # Reset the index after removing duplicates
    df.reset_index(drop=True, inplace=True)

    return df

# Example usage:
# final_df = remove_redundancy(df)
# print(final_df)


def transform_into_ts(df):
    # Initialize an empty DataFrame that will hold all the concatenated results
    final_df = pd.DataFrame(columns=['time', 'bg'])

    # Loop through each row in the DataFrame
    for _, row in df.iterrows():  # Use iterrows() to iterate through DataFrame rows
        # Ensure 'time' is a datetime object
        start_time = pd.to_datetime(row['time'])

        # Extract the first value (assuming the first value is associated with 'bg-0:00')
        first_value = row['bg-0:00']

        # Create a DataFrame for the first entry
        new_row = pd.DataFrame({'time': [start_time], 'bg': [first_value]})
        
        # Concatenate the first row into the final DataFrame
        final_df = pd.concat([final_df, new_row], ignore_index=True)

        # Iterate over the remaining columns (assuming bg columns start at index 2)
        for i, column in enumerate(reversed(row.index[2:-2])):  # Skip the last two columns if needed
            value = row[column]  # Get the value for the current column

            # Create the time for each 'bg-*' column by adding a time delta (adjust as needed)
            time_delta = timedelta(minutes=-5 * (i + 1))  # Example: 5-minute intervals
            new_time = start_time + time_delta  # Adjust time based on column

            # Create a new row for the time and value
            new_row = pd.DataFrame({'time': [new_time], 'bg': [value]})
            
            # Concatenate the new row into the final DataFrame
            final_df = pd.concat([final_df, new_row], ignore_index=True)

    #Set 'time' as the index and sort by time
    #final_df.set_index('time', inplace=True)
    #final_df.sort_index(inplace=True)

    return final_df

# Example usage with a DataFrame:
# Assuming you have a DataFrame 'df' with rows of data
# final_df = transform_into_ts(df)
# print(final_df)



def transform_into_ts_preserved(df):
    # Initialize an empty DataFrame that will hold all the concatenated results
    final_df = pd.DataFrame(columns=['time', 'bg'])

    # Loop through each row in the DataFrame
    for _, row in df.iterrows():  # Use iterrows() to iterate through DataFrame rows
        # Ensure 'time' is a datetime object
        start_time = pd.to_datetime(row['time'])
        original_day = start_time.date()  # Save the original date

        # Extract the first value (assuming the first value is associated with 'bg-0:00')
        first_value = row['bg-0:00']

        # Create a DataFrame for the first entry
        new_row = pd.DataFrame({'time': [start_time], 'bg': [first_value]})
        
        # Concatenate the first row into the final DataFrame
        final_df = pd.concat([final_df, new_row], ignore_index=True)

        # Iterate over the remaining columns (assuming bg columns start at index 2)
        for i, column in enumerate(reversed(row.index[2:-2])):  # Skip the last two columns if needed
            value = row[column]  # Get the value for the current column

            # Create the time for each 'bg-*' column by adding a time delta (adjust as needed)
            time_delta = timedelta(minutes=-5 * (i + 1))  # Example: 5-minute intervals
            new_time = start_time + time_delta  # Adjust time based on column

            # Reset the date part of the new_time to the original day to ensure the day remains unchanged
            new_time = new_time.replace(year=original_day.year, month=original_day.month, day=original_day.day)

            # Create a new row for the time and value
            new_row = pd.DataFrame({'time': [new_time], 'bg': [value]})
            
            # Concatenate the new row into the final DataFrame
            final_df = pd.concat([final_df, new_row], ignore_index=True)

    #Set 'time' as the index and sort by time
    #final_df.set_index('time', inplace=True)
    #final_df.sort_index(inplace=True)

    return final_df



def transform_into_ts_preserved_with_date(df, start_date):
    # Initialize an empty DataFrame that will hold all the concatenated results
    final_df = pd.DataFrame(columns=['time', 'bg'])

    # Ensure 'start_date' is a datetime object
    fixed_date = pd.to_datetime(start_date)

    # Loop through each row in the DataFrame
    for _, row in df.iterrows():  # Use iterrows() to iterate through DataFrame rows
        # Set the time to the start_date and reset time to match row['time']
        start_time = pd.to_datetime(row['time']).replace(year=fixed_date.year, 
                                                         month=fixed_date.month, 
                                                         day=fixed_date.day)
        
        # Extract the first value (assuming the first value is associated with 'bg-0:00')
        first_value = row['bg-0:00']

        # Create a DataFrame for the first entry
        new_row = pd.DataFrame({'time': [start_time], 'bg': [first_value]})
        
        # Concatenate the first row into the final DataFrame
        final_df = pd.concat([final_df, new_row], ignore_index=True)

        # Iterate over the remaining columns (assuming bg columns start at index 2)
        for i, column in enumerate(reversed(row.index[2:-2])):  # Skip the last two columns if needed
            value = row[column]  # Get the value for the current column

            # Create the time for each 'bg-*' column by adding a time delta (adjust as needed)
            time_delta = timedelta(minutes=-5 * (i + 1))  # Example: 5-minute intervals
            new_time = start_time + time_delta  # Adjust time based on column

            # Ensure that only the time part changes while the day remains the same
            new_row = pd.DataFrame({'time': [new_time], 'bg': [value]})
            
            # Concatenate the new row into the final DataFrame
            final_df = pd.concat([final_df, new_row], ignore_index=True)

    return final_df
















