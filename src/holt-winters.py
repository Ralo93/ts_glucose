import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the data
data = pd.read_csv(r"C:\Users\Administrator\Desktop\raphi_other\repositories\ts\data/interim/hourly_filled_lin_again.csv")

# Ensure the 'time' column is parsed as datetime and set as index
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

#df['hour'] = df.index.hour
#df['day_of_week'] = df.index.dayofweek
#df['rolling_mean'] = df['bg'].rolling(window=12).mean()  # Rolling mean over 1 hour

print(df)

# Define the 'bg' series
bg = df['bg']

from statsmodels.tsa.stattools import adfuller

# Perform ADF test
result = adfuller(bg)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")


# Manually split the data, preserving the index
train_size = len(bg) - 60  # Adjust the size accordingly
train = bg.iloc[:train_size]
test = bg.iloc[train_size:]

# Holt-Winters model with multiplicative seasonality
hw_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=24, trend=None).fit()

# Make predictions
pred = hw_model.forecast(steps=len(test))

# Set proper index for predictions
pred.index = test.index

# Only plot the last 24 values from the training data
plt.plot(train.index[-60:], train[-60:], label='Train (last 24)')

# Plot the test data
plt.plot(test.index, test, label='Test')

# Plot the prediction
plt.plot(test.index, pred, label='Holt-Winters Forecast', linestyle='--')

# Add legend and title
plt.legend(loc='best')
plt.title('Holt-Winters Model Prediction (with last 24 values from train)')
plt.show()

exit()
# Simulate future values for confidence intervals
simulated = hw_model.simulate(anchor="end", nsimulations=len(test), repetitions=100)

# Manually set the proper index for the simulations
simulated.index = test.index

# Calculate the 95% confidence interval
ci_lower = simulated.quantile(0.025, axis=1)
ci_upper = simulated.quantile(0.975, axis=1)

# Plot the actual and predicted values with confidence intervals
plt.figure(figsize=(10, 6))

# Plot the test data
plt.plot(test.index, test, label='Test Data')

# Plot the prediction
plt.plot(test.index, pred, label='Holt-Winters Forecast', linestyle='--', color='r')

# Plot the simulated intervals
plt.fill_between(test.index, ci_lower, ci_upper, color='gray', alpha=0.3, label="95% Prediction Interval")

# Add more repetitions for simulation visualization
for i in range(simulated.shape[1]):
    simulated.iloc[:, i].plot(label="_", color="gray", alpha=0.05)

plt.legend()
plt.title('Holt-Winters Model with Prediction Intervals')
plt.show()
