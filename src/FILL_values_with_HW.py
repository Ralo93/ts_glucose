

import requests
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download the dataset
df = pd.read_csv(r"../data/interim/hourly_filled_lin.csv", low_memory=False)
df_with_nan = df[df.isna().any(axis=1)]
print(df_with_nan)


from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(df['bg'], trend=None, seasonal='add', seasonal_periods=288)
fit = model.fit()
df['bg'] = df['bg'].fillna(fit.fittedvalues)

df.info()
# Show rows with any NaN values
df_with_nan = df[df.isna().any(axis=1)]
print(df_with_nan)

df_filled = df

df_filled['bg'] = df['bg'].interpolate(method='linear')

print("at the end:")
df_with_nan = df_filled[df_filled.isna().any(axis=1)]
print(df_with_nan)

df_filled.to_csv('../data/interim/hourly_filled_lin_again.csv', index=True)  # Save to a CSV file

