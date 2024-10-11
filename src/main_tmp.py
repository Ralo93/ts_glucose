
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima import model_selection
import matplotlib.dates as mdates
import numpy as np
#from c_models import HWModel, ARIMAModel, STLModel, GRUResidualModel

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from competition.c_models import *
from competition.c_helpers import adf_test, create_sequences
from competition.c_metrics import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df = pd.read_csv("data/processed/cleaned_up_patients/p10.csv")
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df = df[['bg']]
#print(df.head())

bg = df['bg']
test_size = 72
train, test = model_selection.train_test_split(bg, train_size=len(bg)-test_size)
#print(train.shape)


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# Example data (replace this with your actual time series data)
#np.random.seed(42)  # For reproducibility
#data = np.random.randn(2000)  # Simulated time series data (replace with your actual data)

# Function to compute RMS
# Parameters

# Parameters
train_size = 1000
test_size = 72
n_splits = (len(bg) - train_size) // test_size  # Number of windows

# Placeholder for RMSE values
rmse_scores = []

# Rolling window approach
for i in range(n_splits):
    # Define training and test data for current split
    train_data = bg[i * test_size : train_size + i * test_size]
    test_data = bg[train_size + i * test_size : train_size + (i + 1) * test_size]
    
    # Simple model (you can replace this with any other model)
    X_train = np.arange(len(train_data)).reshape(-1, 1)  # Using indices as features
    y_train = train_data
    X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)

    # Support Vector Regressor (SVR)
    svr_model = SVR(kernel='rbf')  # You can adjust the kernel or other parameters
    svr_model.fit(X_train, y_train)
    
    # Make predictions
    predictions = svr_model.predict(X_test)
    
    # Calculate RMSE for the current split
    score = rmse(test_data, predictions)
    rmse_scores.append(score)

    print(f"Window {i+1} - RMSE: {score}")

# Calculate and display overall RMSE across all splits
mean_rmse = np.mean(rmse_scores)
print(f"Mean RMSE over all windows: {mean_rmse}")

# Final Forecasting using the last available data in the training window
X_forecast = np.arange(len(bg), len(bg) + test_size).reshape(-1, 1)  # Only the future points
svr_forecast = svr_model.predict(X_forecast)

# Plot actual data, training data, test data, and forecast
plt.figure(figsize=(10, 6))

# Plot the actual data
plt.plot(np.arange(len(bg)), bg, label='Actual Data', color='blue')

# Plot the training data
plt.plot(np.arange(train_size), bg[:train_size], label='Training Data', color='green')

# Plot the test data
plt.plot(np.arange(train_size, train_size + test_size), bg[train_size:train_size + test_size], label='Test Data', color='orange')

# Plot the predictions for the test set
plt.plot(np.arange(train_size, train_size + test_size), predictions, label='Predictions', color='red')

# Plot the forecast beyond the test set
plt.plot(np.arange(len(bg), len(bg) + test_size), svr_forecast, label='SVR Forecast', color='purple', linestyle='dashed')

plt.legend()
plt.title('SVR Forecasting')
plt.show()



exit()


arima_model = ARIMAModel(seasonal=True, m=24)
arima_model.fit(train)
arima_forecast = arima_model.forecast(steps=len(test))
print(arima_forecast)
arima_model.plot(train, test, arima_forecast)
plt.show()

## Holt-Winters Model
hw_model = HWModel(seasonal_periods=24, trend=None, seasonal='add')
hw_model.fit(train)
hw_forecast = hw_model.forecast(steps=len(test))#
print(hw_forecast)
hw_model.plot(train, test, hw_forecast)
plt.show()

# STL Model with Forecasting            MUST BE ODD
stl_model = STLModel(period=24, seasonal_window=5, trend=False)
stl_model.decompose(train)
stl_model.fit(train, 'h')
stl_forecast, test_residual_df = stl_model.forecast(train, test, steps=test_size)
print(stl_forecast)
print(test_residual_df)
stl_model.plot(train, test, stl_forecast)
plt.show()

# Combine forecasts from ARIMA, Holt-Winters, and STL into a dataframe
forecast_df = pd.DataFrame({
    'arima_forecast': arima_forecast,
    'hw_forecast': hw_forecast,
    'stl_forecast': stl_forecast,
    'true_values': test  # Actual values for comparison
})

# Print the first few rows of the forecast dataframe to check
print(forecast_df.head())

# Prepare training data for the meta-model (Linear Regression)
X_train = forecast_df[['arima_forecast', 'hw_forecast', 'stl_forecast']].values
y_train = forecast_df['true_values'].values

# Initialize and fit the linear regression model
meta_model = LinearRegression()
meta_model.fit(X_train, y_train)

# Make predictions using the meta-model
meta_forecast = meta_model.predict(X_train)

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    # Train meta-model on each fold
    meta_model.fit(X_train_fold, y_train_fold)
    predictions = meta_model.predict(X_test_fold)
    mse_fold = rmse(y_test_fold, predictions)
    print(f'Root Mean Squared Error for fold: {mse_fold}')

# Evaluate performance using Mean Squared Error (or another suitable metric)
rmse = rmse(y_train, meta_forecast)
print(f'Root Mean Squared Error for meta-model: {rmse}')

# Optional: Print the coefficients (weights) assigned to each model by the meta-model
print(f'Linear Regression Coefficients: {meta_model.coef_}')






### LSTM MODEL:
test_size = 96
n_input = 72
n_features = 1
epochs = 5
neurons = 15

train = df[:-test_size]
test = df[-test_size:]

print(train.shape)
print(test.shape)

scaler = MinMaxScaler()

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Create and fit the model
lstm_model = LSTMModel(n_input=n_input, n_features=n_features, neurons=neurons, epochs=epochs)
lstm_model.fit(scaled_train)

test_predictions = lstm_model.predict(scaled_train, test=test)

true_predictions = scaler.inverse_transform(test_predictions)
final_test = test.copy()
final_test['Predictions'] = true_predictions
final_test.plot(figsize=(12, 6))
plt.show()
# print(rmse(final_test['bg'], final_test['Predictions']))