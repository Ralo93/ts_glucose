# Example usage:
# data = {'time': ..., 'bg': ...}  # Provide your dataset here
# model = TimeSeriesModel(data)
# model.adf_test()
# model.decompose_seasonality()
# model.train_test_split()
# model.holt_winters_model()
# model.forecast()
# model.plot_forecast()
# model.simulate_future_values()
# model.plot_with_confidence_intervals()
# model.run_arima()
# model.plot_arima_results()

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima import model_selection
import matplotlib.dates as mdates
import numpy as np
from c_models import HWModel, ARIMAModel, STLModel, GRUResidualModel
from c_helpers import adf_test, create_sequences
from c_metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = pd.read_csv(r"C:\Users\Administrator\Desktop\raphi_other\repositories\ts\data/processed/concat/hourly_avg_data_stop.csv")
#model = TimeSeriesModel(df)

df = pd.DataFrame(df)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
bg = df['bg']
test_size = 150
train, test = model_selection.train_test_split(bg, train_size=len(bg)-test_size)

print(train.shape)
print(test)

#adf_test(train)
##ARIMA Model
#arima_model = ARIMAModel(seasonal=True, m=24)
#arima_model.fit(train)
#arima_forecast = arima_model.forecast(steps=len(test))
#arima_model.plot(train, test, arima_forecast)
##arima_model.plot_train(train)
#print("SMAPE ARIMA: ", rmse(test, arima_forecast))

## Holt-Winters Model
#hw_model = HWModel(seasonal_periods=24, trend=None, seasonal='add')
#hw_model.fit(train)
#hw_forecast = hw_model.forecast(steps=len(test))#
#hw_model.plot(train, test, hw_forecast)
##hw_model.plot_train(train)
#print("SMAPE HW: ", rmse(test, hw_forecast))

### Add here the predicted forecast for the residuals!!!


# STL Model with Forecasting            MUST BE ODD
stl_model = STLModel(period=24, seasonal_window=5, trend=False)
stl_model.decompose(train)
stl_model.fit(train, 'h')
stl_forecast, test_residual_df = stl_model.forecast(train, test, steps=test_size)
stl_model.plot(train, test, stl_forecast)
#stl_model.plot_train(train)
print("SMAPE STL: ", rmse(test, stl_forecast))
# Assuming res_df comes from STL decomposition (residuals DataFrame)

timesteps = 96

# Extract residuals from STL model (TRAINING data residuals)
res_df = stl_model.df  # Residual DataFrame from training
test_residual_df = pd.DataFrame(test_residual_df.resid)  # Test residuals

# Reshape and scale the residuals from the training data
train_residuals = res_df['resid'].values.reshape(-1, 1)  # Training residuals for GRU input
scaler = StandardScaler()
train_residuals_scaled = scaler.fit_transform(train_residuals)

# Create training sequences for GRU from TRAINING residuals only
X_train, y_train = create_sequences(train_residuals_scaled, timesteps)

# Initialize and train the GRU model using TRAINING data
model = GRUResidualModel(timesteps=timesteps, features=1, gru_units=32, dropout_rate=0.1, learning_rate=0.01)
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Prepare test residuals for scaling (but we won't use it directly in forecasting)
test_residuals_scaled = scaler.transform(test_residual_df.values.reshape(-1, 1))  # Use the trained scaler

# Start the forecasting process iteratively
predicted_residuals = []

# Get the initial sequence from the first part of the test residuals
last_sequence = test_residuals_scaled[:timesteps].reshape(1, timesteps, 1)  # Shape for GRU input

for i in range(test_size):
    # Predict the next residual
    next_residual_scaled = model.predict(last_sequence)
    
    # Inverse scale the predicted residual
    next_residual = scaler.inverse_transform(next_residual_scaled)
    
    # Append to the list of predicted residuals
    predicted_residuals.append(next_residual[0][0])  # Append the scalar value
    
    # Update the sequence: remove the first element, add the new prediction
    new_sequence = np.append(last_sequence[0][1:], next_residual_scaled)  # Remove first, add new
    last_sequence = new_sequence.reshape(1, timesteps, 1)  # Reshape back to (1, timesteps, 1)

# Convert the list of predicted residuals to a NumPy array for evaluation
predicted_residuals = np.array(predicted_residuals)

# Calculate RMSE between actual and predicted residuals (make sure lengths match)
actual_residuals = test_residual_df.values[:test_size]
print("RMSE RES: ", rmse(actual_residuals, predicted_residuals))
print(predicted_residuals)
print(len(predicted_residuals))

# Adjust the STL forecast using the GRU-predicted residuals
adjusted_forecast = stl_forecast + predicted_residuals

# Compute RMSE for the adjusted forecast (STL + GRU)
rmse_adjusted = rmse(test[:test_size], adjusted_forecast)
print("RMSE Adjusted Forecast: ", rmse_adjusted)

# Plot the actual test data vs the adjusted forecast
plt.figure(figsize=(10, 6))
plt.plot(test[:test_size], label='Actual Test Data', color='blue')
plt.plot(stl_forecast, label='STL Forecast', color='green')
plt.plot(adjusted_forecast, label='Adjusted Forecast (STL + GRU Residuals)', color='red')
plt.legend()
plt.title('STL vs Adjusted Forecast (STL + GRU Residuals)')
plt.show()