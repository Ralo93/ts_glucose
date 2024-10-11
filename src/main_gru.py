import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima import model_selection
import numpy as np
from c_models import HWModel, ARIMAModel, STLModel, GRUResidualModel
from c_helpers import adf_test, create_sequences
from c_metrics import rmse
from sklearn.preprocessing import StandardScaler

# Load and process the dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\raphi_other\repositories\ts\data/processed/concat/hourly_avg_data_stop.csv")
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
bg = df['bg']
test_size = 150

# Split train and test sets
train, test = model_selection.train_test_split(bg, train_size=len(bg) - test_size)

# STL Model with Forecasting
stl_model = STLModel(period=24, seasonal_window=5, trend=False)
stl_model.decompose(train)
stl_model.fit(train, 'h')
stl_forecast, test_residual_df = stl_model.forecast(train, test, steps=test_size)

# Plot STL forecast results
stl_model.plot(train, test, stl_forecast)
print("SMAPE STL: ", rmse(test, stl_forecast))

timesteps = 72
# Forecast using STL
stl_forecast, test_residual_df = stl_model.forecast(stl_model, train, test, test, steps=test_size)

# Extract the residuals from the STL decomposition result
res_df = stl_model.df
residuals = res_df['resid'].values.reshape(-1, 1)  # Ensure it's a 2D array (n_samples, 1)

# Standardize the residuals using StandardScaler
scaler = StandardScaler()
residuals_scaled = scaler.fit_transform(residuals)  # Fit and scale the residuals


# Create X_train and y_train from the scaled residuals
X_train, y_train = create_sequences(residuals_scaled, timesteps)

# Initialize the GRUResidualModel
gru_model = GRUResidualModel(timesteps=timesteps, features=1, gru_units=64, dropout_rate=0.2, learning_rate=0.01)

# Train the GRU model
gru_model.fit(X_train, y_train, epochs=10, batch_size=32)

# Prepare test residuals for prediction
test_residuals = test_residual_df['resid'].values.reshape(-1, 1)  # Ensure 2D shape (n_samples, 1)

# Scale the test residuals using the same scaler
test_residuals_scaled = scaler.transform(test_residuals)

# Create sequences from the scaled test residuals for GRU input
X_test, _ = create_sequences(test_residuals_scaled, timesteps)

# Predict residuals using the trained GRU model
residual_forecast_scaled = gru_model.predict(X_test)

# Inverse transform the predicted residuals back to the original scale
residual_forecast = scaler.inverse_transform(residual_forecast_scaled)

# Plot the predicted residuals vs actual test data
gru_model.plot_residuals(test, residual_forecast)
