import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the data
data = pd.read_csv(r"C:\Users\Administrator\Desktop\raphi_other\repositories\ts\data/processed/concat/concat_clean.csv")

# Ensure the 'time' column is parsed as datetime and set as index
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Create new features
df['hour'] = df.index.hour
#df['day_of_week'] = df.index.dayofweek
df['rolling_mean'] = df['bg'].rolling(window=12).mean()  # Rolling mean over 1 hour

# Drop missing values that rolling mean might create
df.dropna(inplace=True)
print(df)
# Define the 'bg' series (target variable) and exogenous variables
bg = df['bg']
exog = df[['hour', 'rolling_mean']]

# Train/test split
train_size = len(bg) - 24  # Adjust the size accordingly
train_bg = bg.iloc[:train_size]
test_bg = bg.iloc[train_size:]

train_exog = exog.iloc[:train_size]
test_exog = exog.iloc[train_size:]

# Fit a SARIMAX model with exogenous variables
model = SARIMAX(train_bg, exog=train_exog, order=(1, 0, 1), seasonal_order=(1, 0, 1, 288))
sarimax_model = model.fit()

# Forecast the test period
pred = sarimax_model.forecast(steps=len(test_bg), exog=test_exog)

# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(train_bg.index[-288:], train_bg[-288:], label='Train (last 24)')
plt.plot(test_bg.index, test_bg, label='Test')
plt.plot(test_bg.index, pred, label='SARIMAX Forecast', linestyle='--')
plt.legend(loc='best')
plt.title('SARIMAX Model with Exogenous Variables')
plt.show()

# You can also print the summary of the SARIMAX model
print(sarimax_model.summary())
