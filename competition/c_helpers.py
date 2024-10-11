
from statsmodels.tsa.stattools import adfuller

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import numpy as np

import pandas as pd

def fill_missing_datetime_rows(df, expected_freq='h'):
    """
    Fills missing datetime rows in a DataFrame and inserts NaN values in the rest of the columns.
    
    Parameters:
    df (pd.DataFrame): The DataFrame with a datetime index that may have missing rows.
    expected_freq (str): The expected frequency of the datetime index. Default is 'H' (hourly).
    
    Returns:
    pd.DataFrame: A DataFrame with continuous datetime index and NaN values for the missing rows.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Generate the expected full range of time based on the min and max of the index
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)

    # Reindex the DataFrame to fill in missing datetime rows
    df_filled = df.reindex(expected_index)

    # Ensure that the index name is set correctly
    df_filled.index.name = df.index.name

    print("Missing datetime rows have been inserted with NaN values in other columns.")
    return df_filled



def check_continuous_datetime_index(df, expected_freq='h'):
    """
    Check if the DataFrame has continuous datetime information as the index.

    Parameters:
    df (pd.DataFrame): DataFrame to check.
    expected_freq (str): Expected frequency of the datetime index. Default is 'H' (hourly).

    Returns:
    bool: True if the datetime index is continuous, False otherwise.
    """
    # Check if the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Index is not a DatetimeIndex.")
        return False

    # Generate the expected full range of time based on the min and max of the index
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)

    # Compare the length of the actual index to the expected index
    if len(df.index) != len(expected_index):
        print("The datetime index is not continuous. Missing time steps found.")
        return False

    # Check if all values in the actual index match the expected index
    if not df.index.equals(expected_index):
        print("The datetime index is not in order or has missing entries.")
        return False

    print("The datetime index is continuous.")
    return True


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


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def forecast_with_gru(stl_model, test_residual_df, test_size, timesteps):
    res_df = stl_model.df  # Residual DataFrame from training
    
    # Reshape and scale the residuals from the training data
    train_residuals = res_df['resid'].values.reshape(-1, 1)
    scaler = StandardScaler()
    train_residuals_scaled = scaler.fit_transform(train_residuals)

    # Create training sequences for GRU from TRAINING residuals only
    X_train, y_train = create_sequences(train_residuals_scaled, timesteps)

    # Initialize and train the GRU model using TRAINING data
    model = GRUResidualModel(timesteps=timesteps, features=1, gru_units=32, dropout_rate=0.1, learning_rate=0.01)
    model.fit(X_train, y_train, epochs=10, batch_size=16)

    # Scale test residuals
    test_residuals_scaled = scaler.transform(test_residual_df.values.reshape(-1, 1))

    # Start the forecasting process iteratively
    predicted_residuals = []
    last_sequence = test_residuals_scaled[:timesteps].reshape(1, timesteps, 1)

    for _ in range(test_size):
        # Predict the next residual
        next_residual_scaled = model.predict(last_sequence)

        # Inverse scale the predicted residual
        next_residual = scaler.inverse_transform(next_residual_scaled)

        # Append to the list of predicted residuals
        predicted_residuals.append(next_residual[0][0])

        # Update the sequence
        new_sequence = np.append(last_sequence[0][1:], next_residual_scaled)
        last_sequence = new_sequence.reshape(1, timesteps, 1)

    # Convert predicted residuals to a NumPy array
    predicted_residuals = np.array(predicted_residuals)

    # Calculate RMSE between actual and predicted residuals
    actual_residuals = test_residual_df.values[:test_size]
    rmse_res = rmse(actual_residuals, predicted_residuals)
    print("RMSE RES: ", rmse_res)

    # Adjust the STL forecast using the GRU-predicted residuals
    adjusted_forecast = stl_model.forecast + predicted_residuals

    # Compute RMSE for the adjusted forecast (STL + GRU)
    rmse_adjusted = rmse(test[:test_size], adjusted_forecast)
    print("RMSE Adjusted Forecast: ", rmse_adjusted)

    # Plot the actual test data vs the adjusted forecast
    plt.figure(figsize=(10, 6))
    plt.plot(test[:test_size], label='Actual Test Data', color='blue')
    plt.plot(stl_model.forecast, label='STL Forecast', color='green')
    plt.plot(adjusted_forecast, label='Adjusted Forecast (STL + GRU Residuals)', color='red')
    plt.legend()
    plt.title('STL vs Adjusted Forecast (STL + GRU Residuals)')
    plt.show()

# Example usage:
# forecast_with_gru(stl_model, test_residual_df, test_size, timesteps)
