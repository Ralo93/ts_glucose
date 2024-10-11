# Blood Glucose Level Prediction Using Time Series Models

This repository contains code and notebooks for predicting blood glucose levels using various time series models, including ARIMA, Holt-Winters Exponential Smoothing, and Seasonal-Trend Decomposition using Loess (STL).

## Project Overview

Blood glucose level prediction is a crucial task in managing diabetes. The ability to forecast glucose levels helps individuals make informed decisions about their diet, insulin intake, and lifestyle choices. In this project, we leverage time series analysis to predict blood glucose levels based on historical data.

### Models Used:

- **ARIMA (AutoRegressive Integrated Moving Average):** A popular statistical model that captures temporal dependencies in the data.
- **Holt-Winters Exponential Smoothing:** A method that accounts for seasonality in the data, which is especially useful when glucose levels follow a cyclical pattern.
- **STL Decomposition (Seasonal-Trend decomposition using Loess):** This method breaks down the time series data into trend, seasonal, and residual components. The residuals can be further modeled using advanced methods like GRU/LSTM for improved forecasting accuracy.

## Repository Structure

```text
├── data/                   # Contains sample data for blood glucose levels
├── notebooks/              # Jupyter notebooks demonstrating model implementation
├── src/                    # Source code for the models
├── README.md               # Project overview and instructions
├── requirements.txt        # Python dependencies
└── LICENSE                 # License information
```


# Competition Section
This repository also includes the solution to a blood glucose prediction competition. The solution leverages a combination of the ARIMA, Holt-Winters, and STL models, along with advanced techniques like GRU/LSTM for residual modeling.

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