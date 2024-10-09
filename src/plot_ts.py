
import pandas as pd
from datetime import timedelta
import glob
import os
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv(r"C:\Users\Administrator\Desktop\raphi_other\repositories\ts\data/interim/hourly_filled_lin_again.csv")


df.info()
# Show rows with any NaN values
df_with_nan = df[df.isna().any(axis=1)]
print(df_with_nan)