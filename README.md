# Blood Glucose Level Prediction Using Time Series and Ensemble Models

This repository contains code and notebooks for predicting blood glucose levels on an hourly basis using various time series models, including ARIMA, Holt-Winters Exponential Smoothing, and Seasonal-Trend Decomposition using Loess (STL).
In the second half of the repository, a solution for the Bloodglucose Competition on Kaggle is provided and explained.

## Project Overview

This is an ongoing project, check again anytime!

## Repository Structure

```text
├── data/                   # Contains sample data for blood glucose levels (not pushed to the repo but you can get it at: https://www.kaggle.com/competitions/brist1d/data
├── notebooks/              # Jupyter notebooks demonstrating EDA, and preprocessing
├── src/                    # Source code for the models, the main.py, the preprocessing_main.py etc.
├── README.md               # Project overview and instructions
├── requirements.txt        # Python dependencies
├── competition             # Competition files
└── LICENSE                 # License information

```
### Global Prevalence:
Type 1 diabetes accounts for about 5-10% of all diabetes cases worldwide. Most other cases are Type 2 diabetes.
Type 1 diabetes is more commonly diagnosed in children, teenagers, and young adults, though it can occur at any age.

Blood glucose level prediction is a crucial task in managing diabetes. The ability to forecast glucose levels helps individuals make informed decisions about their diet, insulin intake, and lifestyle choices. In this project, we leverage time series analysis to predict blood glucose levels based on historical data.


<p align="center">
  <img src="https://github.com/user-attachments/assets/4dce3c84-32de-478a-8dfa-237be80dec0d" alt="Diabetes Illustration" width="400"/>
</p>

<p align="center"><em>Source: Getty Images</em></p>

### Data

The data consists of several patients' data, where each row is basically a collection of 5min intervalls with corresponding information. It was collected using Continous Glucose Monitoring.

- id - row id consisting of participant number and a count for that participant
- p_num - participant number
- time - time of day in the format HH:MM:SS
- bg-X:XX - blood glucose reading in mmol/L, X:XX(H:MM) time in the past (e.g. bg-2:35, would be the blood glucose reading from 2 hours and 35 minutes before the time value for that row), recorded by the continuous glucose monitor
- insulin-X:XX - total insulin dose received in units in the last 5 minutes, X:XX(H:MM) time in the past (e.g. insulin-2:35, would be the total insulin dose received between 2 hours and 40 minutes and 2 hours and 35 minutes before the time value for that row), recorded by the insulin pump
- carbs-X:XX - total carbohydrate value consumed in grammes in the last 5 minutes, X:XX(H:MM) time in the past (e.g. carbs-2:35, would be the total carbohydrate value consumed between 2 hours and 40 minutes and 2 hours and 35 minutes before the time value for that row), recorded by the participant
- hr-X:XX - mean heart rate in beats per minute in the last 5 minutes, X:XX(H:MM) time in the past (e.g. hr-2:35, would be the mean heart rate between 2 hours and 40 minutes and 2 hours and 35 minutes before the time value for that row), recorded by the smartwatch
- steps-X:XX - total steps walked in the last 5 minutes, X:XX(H:MM) time in the past (e.g. steps-2:35, would be the total steps walked between 2 hours and 40 minutes and 2 hours and 35 minutes before the time value for that row), recorded by the smartwatch
- cals-X:XX - total calories burnt in the last 5 minutes, X:XX(H:MM) time in the past (e.g. cals-2:35, would be the total calories burned between 2 hours and 40 minutes and 2 hours and 35 minutes before the time value for that row), calculated by the smartwatch
- activity-X:XX - self-declared activity performed in the last 5 minutes, X:XX(H:MM) time in the past (e.g. activity-2:35, would show a string name of the activity performed between 2 hours and 40 minutes and 2 hours and 35 minutes before the time value for that row), set on the smartwatch
- bg+1:00 - blood glucose reading in mmol/L an hour in the future, this is the value you will be predicting (not provided in test.csv)

*data_train.shape: *
*data_test.shape: *

  


### Models Used in Time Series Forecasting:

- **ARIMA (AutoRegressive Integrated Moving Average):** A popular statistical model that captures temporal dependencies in the data.
- **Holt-Winters Exponential Smoothing:** A method that accounts for seasonality in the data, which is especially useful when glucose levels follow a cyclical pattern.
- **STL Decomposition (Seasonal-Trend decomposition using Loess):** This method breaks down the time series data into trend, seasonal, and residual components. The residuals can be further modeled using advanced methods like GRU/LSTM for improved forecasting accuracy.
- **LSTM**
- **GRU**


# Competition Section
This repository also includes the solution to a blood glucose prediction competition. The solution leverages a combination XGBoost, Lightgbm, NN and SVR to get to a reasonably good score (currently placed 42th).

# Approaches:

My first approach was using a simple RegressionForrest to be used on the whole dataset, using AutoEncoders for each batch of feature columns, e.g. 'activities', which consists of 72 columns.
Having 6 different autoencoders proved to be a lot of training time for each of them, so I shallowed them without having any hidden layers first. Using this approach did only yield a bad prediction power of ca. **3.1686 RMSE** against the competition testset.

As a rather poor performing baseline, I decided to just leave everything as it is and run a RandomForrest Regressor having 200 estimators, simply baseN Encoding all categorical features and leaving numericals as they were - which gave me a **3.0100 RSME** on the test set.
Switching to a more complex model, XGBoost which uses sequemtially trained, shallow trees and completing 20 different models using MLFlow on a considerable large search space yielded these hyperparameters:

```python
n_estimators = 608,
max_depth = 5,
learning_rate = 0.04381358730114617,
min_child_weight = 5,
subsample = 0.8, 
colsample_bytree = 0.8,
reg_alpha = 0.00013679746641535526,  # L1 regularization term
reg_lambda = 0.0005,  # L2 regularization term
random_state = 42
```

This actually performed quite well with **RSME  2.5821**. Repeating the same steps with an LGBMClassifier yielded a **RSME 2.5374**.

The next iteration of modeling will consist of an ensemble model, which shall utilize a SVR and a NN, in combination with a Forrest Regressor to form an ensemble.


## How to Replicate the Solution:
Data Preprocessing: The data is preprocessed to fill missing values, handle outliers, and adjust the time intervals.

Feature Engineering: The time-series data is enhanced by adding features such as moving averages, differences, and lag features.

Modeling: We use a combination of ARIMA, Holt-Winters, and STL with GRU/LSTM for final predictions.


Ensembling: The predictions from the different models are ensembled to improve accuracy.
The notebooks used for the competition are available in the competition/ directory:

competition/data_preprocessing.ipynb
competition/feature_engineering.ipynb
competition/modeling.ipynb
competition/ensemble.ipynb
Follow the notebooks in sequence to replicate the solution.

# Contributing
Feel free to open issues or submit pull requests if you'd like to contribute to this project.

License
This project is licensed under the MIT License - see the LICENSE file for details.
