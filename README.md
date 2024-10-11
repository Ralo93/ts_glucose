# Blood Glucose Level Prediction Using Time Series Models

This repository contains code and notebooks for predicting blood glucose levels using various time series models, including ARIMA, Holt-Winters Exponential Smoothing, and Seasonal-Trend Decomposition using Loess (STL).
In the second half of the repository, a solution for the Bloodglucose Competition on Kaggle is provided and explained.

## Project Overview

This is an ongoing project, check back later!

### Global Prevalence:
Type 1 diabetes accounts for about 5-10% of all diabetes cases worldwide. Most other cases are Type 2 diabetes.
Type 1 diabetes is more commonly diagnosed in children, teenagers, and young adults, though it can occur at any age.




### Incidence and Prevalence by Region:
The global incidence of T1D is estimated to be 15 per 100,000 people per year.
The highest incidence rates are observed in countries like Finland and Sweden, with over 60 cases per 100,000 people per year.
In the United States, roughly 1.6 million people have Type 1 diabetes, which includes both children and adults.
![diabetes-776999_640](https://github.com/user-attachments/assets/509b0dd9-2be9-451a-be6d-6ff6482d734a)

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
├── competition             # Competition files
└── LICENSE                 # License information

```


# Competition Section
This repository also includes the solution to a blood glucose prediction competition. The solution leverages a combination XGBoost, Lightgbm, NN and SVR to get to a reasonably good score (currently placed 42th).

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
