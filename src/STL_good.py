import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data
data = pd.read_csv(r"C:\Users\Administrator\Desktop\raphi_other\repositories\ts\data/interim/hourly_filled_lin_again.csv")

# Ensure the 'time' column is parsed as datetime and set as index
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

from pmdarima import model_selection

bg = df['bg']

test_size = 24

train, test = model_selection.train_test_split(bg, train_size=len(bg)-test_size)

print(len(train))

# Define the 'bg' series

from statsmodels.tsa.seasonal import STL

stl = STL(train, period=24, seasonal=3)
res = stl.fit()
res.plot()

plt.show()

df = pd.concat([res.trend, res.seasonal, res.resid], axis=1)

# Trend
strength_trend = max(0, 1-df['resid'].var()/(df['resid']+df['trend']).var())

# Season
strength_season = max(0, 1-df['resid'].var()/(df['resid']+df['season']).var())

print(strength_trend, strength_season)

from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.statespace import exponential_smoothing

index = pd.date_range(train.index[0], periods=len(train), freq='D')
train.index = index

ES = exponential_smoothing.ExponentialSmoothing
config = {"trend": False, "initialization_method": "estimated"}
period = 24
s_window = 3

stlf = STLForecast(train, ES, model_kwargs=config, period=period, seasonal=s_window)
resf = stlf.fit()
forecasts = resf.forecast(test_size)

#plt.plot(train.index[-24:], train[-24:], label='Train (last 24)')
plt.plot(test.index, test, label='bg')
plt.plot(test.index, forecasts, label='Forecast', c='r', linestyle='--')
plt.xticks(rotation=70)
plt.legend()

plt.show()

import numpy as np
from keras import Sequential
from keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler

# Scale the residuals
scaler = MinMaxScaler(feature_range=(-1, 1))
residuals_scaled = scaler.fit_transform(df['resid'].values.reshape(-1, 1))

print(train)
# Prepare the data for GRU
timesteps = test_size  # based on the seasonal period

X, y = [], []
for i in range(len(residuals_scaled) - timesteps):
    X.append(residuals_scaled[i:i+timesteps])
    y.append(residuals_scaled[i+timesteps])

X, y = np.array(X), np.array(y)
import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Define model parameters
timesteps = 24  # Assuming 24 hours of data
features = 1  # Blood sugar level
gru_units = 64
dropout_rate = 0.2
learning_rate = 0.1

# Define the GRU model
model = Sequential([
    # First GRU layer with return sequences
    GRU(units=gru_units, return_sequences=True, input_shape=(timesteps, features),
        activation='tanh', recurrent_activation='sigmoid',
        kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(dropout_rate),

    # Second GRU layer
    GRU(units=gru_units, return_sequences=False,
        activation='tanh', recurrent_activation='sigmoid',
        kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(dropout_rate),

    # Dense layers for prediction
    Dense(32, activation='linear', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(dropout_rate),
    Dense(1)  # Output layer for blood sugar level prediction
])

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit model with early stopping
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Forecast residuals using the GRU model
residual_forecast_scaled = model.predict(X[-test_size:])
residual_forecast = scaler.inverse_transform(residual_forecast_scaled)
print("residuals")
print(residual_forecast.flatten())
# Combine the forecast
final_forecast = forecasts.values + residual_forecast.flatten()

# Plot the forecasted residuals (GRU forecast alone)
plt.figure(figsize=(14, 6))
plt.plot(test.index, residual_forecast.flatten(), label='GRU Forecasted Residuals', color='orange', linestyle='--')
plt.plot(test.index, test, label='Original bg', color='blue')
plt.title('GRU Forecasted Residuals Alone')
plt.xticks(rotation=70)
plt.legend()
plt.show()

# Plot the combined forecast (STL + GRU) together with the test data
plt.figure(figsize=(14, 6))
plt.plot(test.index, test, label='Original bg', color='blue')
plt.plot(test.index, final_forecast, label='STL + GRU Forecast', color='red', linestyle='--')
plt.title('Combined STL + GRU Forecast')
plt.xticks(rotation=70)
plt.legend()
plt.show()

