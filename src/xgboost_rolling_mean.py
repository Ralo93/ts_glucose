import pandas as pd
from matplotlib import pyplot as plt
from src.xgboost_rolling_mean import XGBRegressor
from sklearn.model_selection import train_test_split


data = pd.read_csv(r"C:\Users\Administrator\Desktop\raphi_other\repositories\ts\data/processed/concat/concat_clean.csv")
df = pd.DataFrame(data)
# Create lag features
df['lag_1'] = df['bg'].shift(12)
df['lag_2'] = df['bg'].shift(24)
df['lag_3'] = df['bg'].shift(288)

# Drop missing values due to lagging
df.dropna(inplace=True)

# Define features (including exogenous and lag features) and target
X = df[['hour', 'day_of_week', 'rolling_mean', 'lag_1', 'lag_2', 'lag_3']]
y = df['bg']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train an XGBoost model
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

# Predict the test set
y_pred = xgb_model.predict(X_test)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='XGBoost Forecast', linestyle='--')
plt.legend(loc='best')
plt.title('XGBoost Model with Exogenous and Lag Features')
plt.show()
