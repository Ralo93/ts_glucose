
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
#from c_models import HWModel, ARIMAModel, STLModel, GRUResidualModel
from competition.c_helpers import adf_test, create_sequences
from competition.c_metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from competition.c_models import *
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



df = pd.read_csv("data/processed/cleaned_up_patients/p10.csv")
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df = df[['bg']]
print(df.head())

bg = df['bg']
test_size = 96
train, test = model_selection.train_test_split(bg, train_size=len(bg)-test_size)
print(train.shape)
plt.plot(df)
plt.show()

plot_acf(df.bg, lags=100);
adf_test(df)
plt.show()

#df = pd.DataFrame(df)
#df['time'] = pd.to_datetime(df['time'])
#df.set_index('time', inplace=True)


#adf_test(train)
#ARIMA Model
arima_model = ARIMAModel(seasonal=True, m=24)
arima_model.fit(train)
arima_forecast = arima_model.forecast(steps=len(test))
arima_model.plot(train, test, arima_forecast)
arima_model.plot_train(train)
print("RMSE ARIMA: ", rmse(test, arima_forecast))
print("SMAPE ARIMA: ", smape(test, arima_forecast))

#exit()
## Holt-Winters Model
hw_model = HWModel(seasonal_periods=24, trend=None, seasonal='add')
hw_model.fit(train)
hw_forecast = hw_model.forecast(steps=len(test))#
hw_model.plot(train, test, hw_forecast)
hw_model.plot_train(train)
print("RMSE HW: ", rmse(test, hw_forecast))
print("SMAPE HW: ", smape(test, hw_forecast))


# STL Model with Forecasting            MUST BE ODD
stl_model = STLModel(period=24, seasonal_window=3, trend=False)
stl_model.decompose(train)
stl_model.fit(train, 'h')
stl_forecast, test_residual_df = stl_model.forecast(train, test, steps=test_size)
stl_model.plot(train, test, stl_forecast)
stl_model.plot_train(train)
print("RMSE STL: ", rmse(test, stl_forecast))
print("SMAPE STL: ", smape(test, stl_forecast))
# Assuming res_df comes from STL decomposition (residuals DataFrame)

exit()

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