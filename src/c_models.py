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
from statsmodels.tsa.api import ExponentialSmoothing, SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace import exponential_smoothing
from pmdarima import auto_arima
#from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import STLForecast


# Holt-Winters (Exponential Smoothing) Model Class
class HWModel:
    def __init__(self, seasonal_periods=24, trend=None, seasonal='add', initialization_method='estimated'):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.initialization_method = initialization_method
        self.model = None

    def fit(self, train):
        self.model = ExponentialSmoothing(train, seasonal=self.seasonal, trend=self.trend,
                                          seasonal_periods=self.seasonal_periods, 
                                          initialization_method=self.initialization_method).fit()

    def forecast(self, steps):
        return self.model.forecast(steps=steps)

    def plot_train(self, train):
        """Plot fitted values on the train set"""
        fitted_values = self.model.fittedvalues
        plt.figure(figsize=(10, 6))
        plt.plot(train.index, train, label='Actual Train Data', color='blue')
        plt.plot(train.index, fitted_values, label='Fitted Values', color='orange', linestyle='--')
        plt.title('Holt-Winters Model - Fitted vs Actual on Training Data')
        plt.legend()
        plt.xticks(rotation=70)
        plt.show()

    def plot(self, train, test, forecast, test_size=24):
        plt.plot(train.index[-test_size:], train[-test_size:], label='Train (last 24)')
        plt.plot(test.index, test, label='Test')
        plt.plot(test.index, forecast, label='HW Forecast', linestyle='--', color='red')
        plt.xticks(rotation=70)
        plt.legend()
        plt.show()


# ARIMA (SARIMAX) Model Class
class ARIMAModel:
    def __init__(self, seasonal=True, m=24):
        self.seasonal = seasonal
        self.m = m
        self.model = None

    def fit(self, train):
        self.model = auto_arima(train, start_p=0, start_q=0, max_p=2, max_q=2, 
                                seasonal=self.seasonal, m=self.m, 
                                start_P=0, start_Q=0, max_P=1, max_Q=1,
                                d=0, D=0, trace=True, error_action='ignore', 
                                suppress_warnings=True, maxiter=1)

    def forecast(self, steps):
        return self.model.predict(n_periods=steps)

    def plot_train(self, train):
        """Plot fitted values on the train set"""
        fitted_values = self.model.predict_in_sample()
        plt.figure(figsize=(10, 6))
        plt.plot(train.index, train, label='Actual Train Data', color='blue')
        plt.plot(train.index, fitted_values, label='Fitted Values', color='orange', linestyle='--')
        plt.title('ARIMA Model - Fitted vs Actual on Training Data')
        plt.legend()
        plt.xticks(rotation=70)
        plt.show()

    def plot(self, train, test, forecast, test_size=24):
        plt.plot(train.index[-test_size:], train[-test_size:], label='Train (last 24)')
        plt.plot(test.index, test, label='Test')
        plt.plot(test.index, forecast, label='ARIMA Forecast', linestyle='--', color='red')
        plt.xticks(rotation=70)
        plt.legend()
        plt.show()



# STL with Forecasting Class
class STLModel:
    def __init__(self, period=24, seasonal_window=3, trend=False, initialization_method='estimated'):
        self.period = period
        self.seasonal_window = seasonal_window
        self.trend = trend
        self.initialization_method = initialization_method
        self.df = None
        self.df_test = None

    def decompose(self, train):
        stl = STL(train, period=self.period, seasonal=self.seasonal_window)
        self.res = stl.fit()
        self.res.plot()
        plt.show()

        self.df = pd.concat([self.res.trend, self.res.seasonal, self.res.resid], axis=1)

        # Trend strength
        self.strength_trend = max(0, 1 - self.df['resid'].var() / (self.df['resid'] + self.df['trend']).var())

        # Season strength
        self.strength_season = max(0, 1 - self.df['resid'].var() / (self.df['resid'] + self.df['season']).var())

        print(f"Strength of Trend: {self.strength_trend}")
        print(f"Strength of Seasonality: {self.strength_season}")

    def fit(self, train, frequency):
        # Check if the DatetimeIndex has a frequency
        if train.index.freq is None:
            print(f"No frequency found in DatetimeIndex. Setting frequency Parameter specified: {frequency}.")
            train = train.asfreq(frequency)  # You can change 'D' to any appropriate frequency, e.g., 'H' for hourly, 'W' for weekly.

        # Now perform STLForecast with Exponential Smoothing
        ES = exponential_smoothing.ExponentialSmoothing
        config = {"trend": self.trend, "initialization_method": self.initialization_method}
        
        stlf = STLForecast(train, ES, model_kwargs=config, period=self.period, seasonal=self.seasonal_window)
        self.stlf_result = stlf.fit()
        
        print("STL forecast fitted successfully.")

    def plot_train(self, train):
        """Plot fitted values on the train set"""
        fitted_values = self.stlf_result.fittedvalues
        plt.figure(figsize=(10, 6))
        plt.plot(train.index, train, label='Actual Train Data', color='blue')
        plt.plot(train.index, fitted_values, label='Fitted Values', color='orange', linestyle='--')
        plt.title('STL + Forecasting - Fitted vs Actual on Training Data')
        plt.legend()
        plt.xticks(rotation=70)
        plt.show()


    def plot(self, train, test, forecast, test_size=24):
        plt.plot(train.index[-test_size:], train[-test_size:], label='Train (last 24)')
        plt.plot(test.index, test, label='Test')
        plt.plot(test.index, forecast, label='STL + Forecast', linestyle='--', color='red')
        plt.xticks(rotation=70)
        plt.legend()
        plt.show()

    def forecast(self, train, y_test, steps):
        """
        Forecast and plot the last 'test_size' points from the train set, the test set, and the forecast.
        
        Parameters:
        - train: Training dataset (pandas Series/DataFrame)
        - test: Test dataset (pandas Series/DataFrame)
        - y_test: The residuals or target values for the test set (pandas Series/DataFrame)
        - steps: Number of steps to forecast
        - test_size: Number of points from the training data to plot (default=24)
        
        Returns:
        - forecast_values: The forecasted values for the given number of steps
        - residuals_df: DataFrame containing the residuals for the test set
        """
        # Generate forecast for the number of steps
        forecast_values = self.stlf_result.forecast(steps)

        # Create a DataFrame to store residuals for the test set (difference between forecast and actual values)
        residuals_df = pd.DataFrame({
            'forecast': forecast_values,
            'actual': y_test.values[:steps]  # Take the first 'steps' number of actual test values
        })
        residuals_df['resid'] = residuals_df['actual'] - residuals_df['forecast']



        # Return the forecast and residual DataFrame
        return forecast_values, residuals_df


    
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

class GRUResidualModel:
    def __init__(self, timesteps=24, features=1, gru_units=64, dropout_rate=0.2, learning_rate=0.1, l2_reg=0.01, patience=3):
        """
        Initialize the GRU model with configurable hyperparameters.
        """
        self.timesteps = timesteps
        self.features = features
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.patience = patience

        # Define the GRU model
        self.model = Sequential([
            GRU(units=self.gru_units, return_sequences=True, input_shape=(self.timesteps, self.features),
                activation='tanh', recurrent_activation='sigmoid',
                kernel_regularizer=l2(self.l2_reg), recurrent_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(self.dropout_rate),

            GRU(units=self.gru_units, return_sequences=False,
                activation='tanh', recurrent_activation='sigmoid',
                kernel_regularizer=l2(self.l2_reg), recurrent_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(self.dropout_rate),

            Dense(32, activation='linear', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Dense(1)  # Output layer for blood sugar level prediction
        ])

        # Compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])

    def fit(self, X_train, y_train, validation_split=0.2, epochs=10, batch_size=32):
        """
        Fit the GRU model to the training data with early stopping.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                        validation_split=validation_split, callbacks=[early_stopping])

    def predict(self, X_test):
        """
        Make predictions using the trained GRU model.
        """
        return self.model.predict(X_test)

    def plot_residuals(self, test, residual_forecast, title='GRU Forecasted Residuals Alone'):
        """
        Plot the forecasted residuals (GRU forecast alone).
        """
        plt.figure(figsize=(14, 6))
        plt.plot(residual_forecast.flatten(), label='GRU Forecasted Residuals', color='orange', linestyle='--')
        plt.plot(test, label='Original Residuals', color='blue')
        plt.title(title)
        plt.xticks(rotation=70)
        plt.legend()
        plt.show()

    def plot_combined_forecast(self, test, final_forecast, title='Combined STL + GRU Forecast'):
        """
        Plot the combined forecast (STL + GRU) together with the test data.
        """
        plt.figure(figsize=(14, 6))
        plt.plot(test.index, test, label='Original bg', color='blue')
        plt.plot(test.index, final_forecast, label='STL + GRU Forecast', color='red', linestyle='--')
        plt.title(title)
        plt.xticks(rotation=70)
        plt.legend()
        plt.show()



