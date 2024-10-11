
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from category_encoders import BaseNEncoder
import numpy as np
from c_preprocessing import *
from c_helpers import *

patient = "p02"
# Example usage:
preprocessing = BloodGlucosePreprocessing(r"C:\Users\rapha\repositories\ts_glucose\data\raw/blood_glucose/train.csv")
preprocessing.filter_patient_data(patient)
print(preprocessing.df)

df = preprocessing.df.copy()

# Assuming df is your DataFrame, with the last column as the target
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The target column


# Split into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Select categorical and numerical columns
cat_cols = x_train.select_dtypes(include=['object', 'category']).columns
num_cols = x_train.select_dtypes(exclude=['object', 'category']).columns


rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)


# Create a ColumnTransformer: BaseN for categorical, 'passthrough' for numerical
preprocessor = ColumnTransformer(
    transformers=[
        ('baseN', BaseNEncoder(base=3), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', rf)
])

# Perform 5-fold cross-validation using RSME
rmse_scorer = make_scorer(mean_squared_error, squared=False)
scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring=rmse_scorer)

# Print the average RMSE across the folds
print("Average RMSE:", np.mean(scores))