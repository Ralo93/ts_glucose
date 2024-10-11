# Blood Glucose Level Prediction Using Time Series and Ensemble Models

## Project Overview

This repository contains code and notebooks for predicting blood glucose levels on an hourly basis using various time series models, including ARIMA, Holt-Winters Exponential Smoothing, and Seasonal-Trend Decomposition using Loess (STL).
In the second half of the repository, a solution for the Bloodglucose Competition on Kaggle is provided and explained.

This is an ongoing project, check again anytime!

## Repository Structure

```text
├── data/                   # Contains sample data for blood glucose levels 
├── notebooks/              # Jupyter notebooks demonstrating EDA, and preprocessing
├── images                  # Project images
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
  <img src="https://github.com/user-attachments/assets/4dce3c84-32de-478a-8dfa-237be80dec0d" alt="Diabetes Illustration" width="450"/>
</p>

<p align="center"><em>Source: Getty Images</em></p>

### Data

The data consists of several patients' data, where each row is basically a collection of 5min intervalls with corresponding information. It was collected using Continous Glucose Monitoring.

It can be found here: https://www.kaggle.com/competitions/brist1d/data

### Models Used in Time Series Forecasting:

- **ARIMA (AutoRegressive Integrated Moving Average):** A popular statistical model that captures temporal dependencies in the data.
- **Holt-Winters Exponential Smoothing:** A method that accounts for seasonality in the data, which is especially useful when glucose levels follow a cyclical pattern.
- **STL Decomposition (Seasonal-Trend decomposition using Loess):** This method breaks down the time series data into trend, seasonal, and residual components. The residuals can be further modeled using advanced methods like GRU/LSTM for improved forecasting accuracy.
- **LSTM**
- **GRU**

### Preprocessing

I first preprocessed each patient into an hourly sequence of datapoints. Since some patients have 15min intervalls instead of 5min intervalls, and I linearly interpolate missing values, I will only consider patients with 5min intervalls going onward, as the 15min intervall patients are having too many interpolations as can be seen here:


<p align="center">
  <img src="https://github.com/user-attachments/assets/1c135685-54f9-4fc5-bf30-0c100551b703" alt="p10 Illustration" width="700"/>
</p>

<p align="center"><em>Patient 04</em></p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b9ad805c-5992-47bc-b3a7-163923366e68" alt="p04 Illustration" width="700"/>
</p>

<p align="center"><em>Patient 10</em></p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/3c36e11a-8959-42c6-b334-3eabdd29a7fd" alt="p10 Illustration" width="700"/>
</p>

<p align="center"><em>Patient 01 (discarded)</em></p>


Before modeling, I checked wether the data provided does have a trend and a seasonality.

The adf-test result shows a clear picture:

<p align="center">
  <img src="https://github.com/user-attachments/assets/8c6235a5-e608-4353-9fce-cd89a1d2f96b" alt="Diabetes Illustration" width="350"/>
</p>

<p align="center"><em>ADF-Test on Patient 04</em></p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/394ec3dd-e831-4c4c-8f63-596f6e43bb2e" alt="Trend P10" width="350"/>
</p>

<p align="center"><em>ADF-Test on Patient 10</em></p>

Regarding the seasonality, there seems to be a 24 hour seasonality in the hourly data, which makes sense. Even if it is not very strong (at least for patient 04, it certainly plays a role)

<p align="center">
  <img src="https://github.com/user-attachments/assets/aa7ff6c1-551c-4552-ac05-cc7d74bc3112" alt="Diabetes Illustration" width="800"/>
</p>

<p align="center"><em>Seasonality Patient 04</em></p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e4dbffc7-9b91-4d2b-9439-15e12d6e4187" alt="Diabetes Illustration" width="800"/>
</p>

<p align="center"><em>Seasonality Patient 10</em></p>



I will use a test-set of 96 hours into the future (ot the last 96 hours of the dataset). Since there was no date information given, I added date information myself which from this point in writing actually shows dates from the future, but it should not be a problem.

## ARIMA Model

I did not expect the ARIMA model to be very useful in this scenario, still as a base model it is probably interesting to look at first.
AutoArima doing its thing, while given a seasonality of 24:

<p align="center">
  <img src="https://github.com/user-attachments/assets/59fea7f3-1704-45de-8e4a-7e9c9f136c44" alt="Diabetes Illustration" width="350"/>
</p>

<p align="center"><em>AutoArima Patient 04</em></p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/ecef5a0f-87b5-4ea9-9f53-b153a77954e0" alt="Diabetes Illustration" width="350"/>
</p>

<p align="center"><em>AutoArima Patient 10</em></p>

The algorithm came up with the best fitting model to be ARIMA(0,0,2)(0,0,1)[24] intercept.
This is actually surprising to me, as no autoregressive terms were used. It includes two lagged errors in the Moving Averages term, as well as one in the seasonality term. The intercept makes sense as the data does not osciallte around 0.

For Patient 10 it used ARIMA(1,0,0)(0,0,1)[24] intercept.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9bcd68dc-ea37-44d9-99ee-3c23c3175353" alt="Diabetes Illustration" width="700"/>
</p>

<p align="center"><em>96-hour forecast Patient 04</em></p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/28c6dfac-15b3-41c4-b094-ab9c5b5a17c3" alt="Diabetes Illustration" width="700"/>
</p>

<p align="center"><em>96-hour forecast Patient 10</em></p>

As expected, the ARIMA model quickly loses predictive power, as its predictions will continously be dependent on the last values it predicted itself.
The metrics are therefore taken not quite serious, but its a starting point.

- **RMSE ARIMA:**  2.255 (p04), 2.409 (p10)
- **SMAPE ARIMA:**  21.874 (p04), 20.980 (p10)


## Holt-Winters Model

<p align="center">
  <img src="https://github.com/user-attachments/assets/edc9472d-737a-43f7-99b7-33a33308fbbb" alt="Diabetes Illustration" width="700"/>
</p>

<p align="center"><em>96-hour forecast HW Patient 04</em></p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2e51c6a5-56d4-48ef-a94a-5c0e459a2350" alt="Diabetes Illustration" width="700"/>
</p>

<p align="center"><em>96-hour forecast HW Patient 10</em></p>

- **RMSE HW:**  2.2574 (p04), 2.217 (p10)
- **SMAPE HW:**  21.793 (p04), 19.368(p10)


Even though the metrics are just marginally better, the HW forecast at least found some seasonal pattern (here daily).
Also from the fitting on the train data, the HW model seems to be less noisy compared to the ARIMA model:



<p align="center">
  <img src="https://github.com/user-attachments/assets/8543c7fb-6468-49f9-b358-7310a449cdcc" alt="Diabetes Illustration" width="900"/>
</p>
<p align="center"><em>Train-Fit ARIMA</em></p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/0c2198fc-683c-42f4-b237-4fe85f185114" alt="Diabetes Illustration" width="900"/>
</p>
<p align="center"><em>Train-Fit HW</em></p>


## Seasonal-Trend-Loess (STL):

STL provides a nicely formatted decomposition into trend, seasonality and residuals of the model.

- **Strength of Trend: 0.177** (p10)
- **Strength of Seasonality: 0.566** (p10)

This shows (as expected) there is no trend in the data, and a medium to strong seasonality. For the STL forecast, I tried different seasonal_windows, here the window is 5. This hyperparameter tells me how many "seasons" in the past it should take into consideration for its forecast.

Decomposed it looks like this:

<p align="center">
  <img src="https://github.com/user-attachments/assets/d947204b-0a67-4ed8-9d8b-23dee2f2c713" alt="Diabetes Illustration" width="900"/>
</p>
<p align="center"><em>Seasonal Decomposition p04 HW</em></p>


Forecasts using STL:

<p align="center">
  <img src="https://github.com/user-attachments/assets/d4f564e9-cc61-4c98-b0aa-c9ad21a828c4" alt="Diabetes Illustration" width="900"/>
</p>
<p align="center"><em>Forecast STL p04</em></p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e6ec38ca-b66d-4870-badb-12ed702108dd" alt="Diabetes Illustration" width="900"/>
</p>
<p align="center"><em>Forecast STL p10</em></p>


- **RMSE STL:  2.783 (p04), 2.463(p10)**
- **SMAPE STL:  26.955(p04), 22.414 (p10)**


The code in the repository can be used to change the patient, the test-size and which model to run. Use the main.py file to make any changes and just run it. :)


# Competition Section
This repository also includes the solution to a blood glucose prediction competition. The solution leverages a combination XGBoost, Lightgbm, NN and SVR to get to a reasonably good score (currently placed 42th).

# Approaches:

My first approach was using a simple RandomForrest Regressor to be used on the whole dataset, using AutoEncoders for each batch of feature columns, e.g. 'activities', which consists of 72 columns.
Having 6 different autoencoders proved to be a lot of training time for each of them, so I shallowed them without having any hidden layers first. Using this approach did only yield a bad prediction power of ca. **3.1686 RMSE** against the competition testset.

As a rather poor performing baseline, I decided to just leave everything as it is and run a RandomForrest Regressor having 200 estimators, simply baseN-Encoding all categorical features and leaving numericals as they were - which gave me a **3.0100 RSME** on the test set.
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
