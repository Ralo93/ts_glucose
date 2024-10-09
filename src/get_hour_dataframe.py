import pandas as pd


df = pd.read_csv(r"../data/interim/p12_filled_linear.csv", low_memory=False)

# Load the CSV file
#df = pd.read_csv('your_file.csv')

# Convert 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Set 'time' as the index and retain only the 'bg' column
df.set_index('time', inplace=True)
df = df[['bg']]

# Display the DataFrame
print(df)

hourly_df = df.resample('h').first()

# Save to CSV
file_path = r"../data/interim/hourly_filled_lin.csv"
hourly_df.to_csv(file_path, index=True)

print(f"CSV file saved to {file_path}")
