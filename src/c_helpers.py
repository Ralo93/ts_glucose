
from statsmodels.tsa.stattools import adfuller

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import numpy as np

def create_sequences(data, timesteps):
    """
    Create sequences of length 'timesteps' from the data for GRU/LSTM models.
    
    Parameters:
    - data: The time series data (e.g., residuals from the residual column)
    - timesteps: Number of time steps to look back (sequence length)
    
    Returns:
    - X: Input data as sequences (shape: [samples, timesteps, features])
    - y: Target data (shape: [samples, 1])
    """
    X, y = [], []
    
    # Loop through the data to create sequences
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        y.append(data[i + timesteps])  # The target is the next value after the sequence

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to [samples, timesteps, features] (required by GRU/LSTM)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # 1 feature: the residual itself
    
    return X, y



def adf_test(train):
    # Perform ADF test
    result = adfuller(train.dropna())  # Drop NaN values to avoid errors
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]

    # Print results
    print(f"ADF Statistic: {adf_stat}")
    print(f"p-value: {p_value}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"\t{key}: {value}")

    # Check for trend based on p-value
    if p_value < 0.05:
        print("The data is stationary. (No significant trend)")
    else:
        print("The data is non-stationary. (Potential trend present)")




def train_test_ols_forecast(train, test, test_size):
    # Ensure 'train' is a DataFrame
    if isinstance(train, pd.Series):
        train = train.to_frame()  # Convert Series to DataFrame
    # Ensure 'test' is a DataFrame
    if isinstance(test, pd.Series):
        test = test.to_frame()  # Convert Series to DataFrame

    # Set up the index for the 'train' DataFrame (960 hours)
    index = pd.date_range('2024-10-02', periods=960, freq='H')
    if len(train) < len(index):
        # Append rows with NaN for missing periods
        additional_rows = pd.DataFrame(np.nan, index=index[len(train):], columns=train.columns)
        train = pd.concat([train, additional_rows])

    # Assign the new index to 'train'
    train.index = index

    # Ensure train index is a DatetimeIndex
    if not pd.api.types.is_datetime64_any_dtype(train.index):
        train.index = pd.to_datetime(train.index)

    # Add the weekday and time index columns to the train DataFrame
    train['wday'] = train.index.weekday
    train['sin_hour'] = np.sin(2 * np.pi * train.index.hour / 24)
    train['cos_hour'] = np.cos(2 * np.pi * train.index.hour / 24)
    train['t'] = pd.RangeIndex(len(train))

    # Running the OLS regression using the statsmodels formula API
    mod_dummy = smf.ols("bg ~ t + sin_hour + cos_hour + C(wday)", data=train).fit()

    # Print the summary of the regression results
    print(mod_dummy.summary())

    # Prepare the index for the 'test' DataFrame (test_size hours)
    index = pd.date_range('2024-11-04', periods=test_size, freq='H')
    if len(test) < len(index):
        # Append rows with NaN for missing periods
        additional_rows = pd.DataFrame(np.nan, index=index[len(test):], columns=test.columns)
        test = pd.concat([test, additional_rows])

    # Assign the new index to 'test'
    test.index = index

    # Ensure test index is a DatetimeIndex
    if not pd.api.types.is_datetime64_any_dtype(test.index):
        test.index = pd.to_datetime(test.index)

    # Add the weekday and time index columns to the test DataFrame
    test['wday'] = test.index.weekday
    test['sin_hour'] = np.sin(2 * np.pi * test.index.hour / 24)
    test['cos_hour'] = np.cos(2 * np.pi * test.index.hour / 24)
    test['t'] = pd.RangeIndex(len(test)) + len(train)

    # Forecast using the OLS model
    fcast = mod_dummy.predict(test[['t', 'sin_hour', 'cos_hour', 'wday']])

    # Plotting the forecast and actual data
    plt.figure(figsize=(10, 6))
    test['bg'].plot(label='Actual bg', color='blue')
    fcast.plot(label='Forecast', linestyle='--', color='red')
    plt.title('OLS Forecast vs Actual')
    plt.legend()
    plt.show()

    return fcast
