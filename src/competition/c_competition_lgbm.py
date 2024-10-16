import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, root_mean_squared_error
import numpy as np
from c_preprocessing import *
from c_helpers import *
from category_encoders import BaseNEncoder
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import gc
#

patient = "p01"
# Example usage:
preprocessing = BloodGlucosePreprocessing(r"C:\Users\rapha\repositories\ts_glucose\data\raw/blood_glucose/train.csv")
#preprocessing.filter_patient_data(patient, bg_only=False, bg_or_plus_one=False)
preprocessing.downcast_floats()
# bg_only=False, bg_or_plus_one=True
#print(preprocessing.df)

df = preprocessing.df.copy()
#gc.collect()

print("DataFrame after preprocessing:")
print(df.head())
print(df.info())

# Root mean squared error function
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Assuming df is your DataFrame, with the last column as the target
print("Shape of the dataframe:", df.shape)

X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The target column

print("Shape of X (features):", X.shape)
print("Shape of y (target):", y.shape)

print("X columns:", X.columns)
print("y name:", y.name)

# Split into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)

x_train = x_train.drop(columns=['id'])  # Drop the 'id' column from test data
y_train = y_train.drop(columns=['id']) 

print("x_train shape after dropping 'id':", x_train.shape)
print("y_train shape after dropping 'id':", y_train.shape)

# Select categorical and numerical columns
cat_cols = x_train.select_dtypes(include=['object', 'category']).columns
num_cols = x_train.select_dtypes(exclude=['object', 'category']).columns

print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# Base model pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('baseN', BaseNEncoder(base=3), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

# Base learners with regularization
# RandomForest with regularization (using min_samples_split and min_samples_leaf)
rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,  # Add regularization
        min_samples_leaf=5,    # Add regularization
        random_state=42
    ))
])

# XGBoost with regularization (using reg_alpha and reg_lambda)
xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        tree_method='hist',
        n_estimators=608,
        max_depth=5,
        learning_rate=0.04381358730114617,
        min_child_weight=5,
        subsample=0.8, 
        colsample_bytree=0.8,
        reg_alpha=0.00013679746641535526,  # L1 regularization term
        reg_lambda=0.0005,  # L2 regularization term
        random_state=42
    ))
])

# LightGBM with regularization (using reg_alpha and reg_lambda)
lgbm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(
        n_estimators=608,
        max_depth=5,
        learning_rate=0.04381358730114617,
        subsample=0.8, 
        colsample_bytree=0.8516886888374169,
        reg_alpha=0.1,  # L1 regularization term
        reg_lambda=0.2,  # L2 regularization term
        random_state=42,
        device='gpu'  # Using GPU
    ))
])

print("Base models created")

from sklearn.linear_model import Ridge

# Meta-model using stacking (final model)
meta_model = StackingRegressor(
    estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
    final_estimator=Ridge(alpha=1.0)
)

print("Meta-model created")

# Cross-validation with RMSE
rmse_scorer = make_scorer(root_mean_squared_error)
##scores = cross_val_score(meta_model, x_train, y_train, cv=5, scoring=rmse_scorer)#

# Print RMSE scores for each fold and average RMSE
#print("RMSE scores for each fold:", scores)
#print("Average RMSE:", np.mean(scores))

model_var = xgb


# Fit the meta-model on the training data
print(f"Fitting {model_var.__repr__}-model...")
model_var.fit(X, y)
print(f"{model_var.__repr__}-model fitted")

# Fit the meta-model on the training data
#print("Fitting meta-model...")
#meta_model.fit(x_train, y_train)
#print("Meta-model fitted")



# RMSE for the training set
y_train_pred = model_var.predict(x_train)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
print("Train set RMSE:", train_rmse)

# RMSE for the validation set
y_val_pred = model_var.predict(x_val)
val_rmse = root_mean_squared_error(y_val, y_val_pred)
print("Validation set RMSE:", val_rmse)

# Load test.csv and make predictions
test_df = pd.read_csv(r"C:\Users\rapha\repositories\ts_glucose\data\raw/blood_glucose/test.csv")
print("Test data shape:", test_df.shape)
test_ids = test_df['id']
X_test = test_df.drop(columns=['id'])
print("X_test shape:", X_test.shape)

# Make predictions on the test data
print("Making predictions on test data...")
y_test_pred = model_var.predict(X_test)
print("Predictions made")

# Create and save the submission file
submission = pd.DataFrame({
    'id': test_ids,
    'bg+1:00': y_test_pred
})
print("Submission DataFrame created")
print("Submission DataFrame shape:", submission.shape)
submission.to_csv('submissiontest_full_data.csv', index=False)
print("Submission file 'submission.csv' created successfully.")

exit()


## CROSS-VALIDATION

#print("Starting 5-fold cross-validation with GPU...")
#scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring=rmse_scorer)

#print("RMSE scores for each fold:", scores)

# Print the average RMSE across the folds
#print("Average RMSE:", np.mean(scores))


# Fit the model on the training set and calculate the RMSE for the training set
pipeline.fit(x_train, y_train)
y_train_pred = pipeline.predict(x_train)
train_rmse = root_mean_squared_error(y_train, y_train_pred)

# Print the RMSE for the training set
print("Train set RMSE:", train_rmse)

# Now, calculate RMSE on the validation set
y_val_pred = pipeline.predict(x_val)
val_rmse = root_mean_squared_error(y_val, y_val_pred)




test_df = pd.read_csv(r"C:\Users\rapha\repositories\ts_glucose\data\raw/blood_glucose/test.csv")

# Ensure the test set has the same preprocessing as the train set
test_ids = test_df['id']  # Save the 'id' column for the submission
X_test = test_df.drop(columns=['id'])  # Drop the 'id' column from test data

# Make predictions on the test data using the fitted pipeline
y_test_pred = pipeline.predict(X_test)

# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'bg+1:00': y_test_pred
})


# Save the submission DataFrame to a CSV file
submission.to_csv(r"C:\Users\rapha\repositories\ts_glucose\data\processed/submission.csv", index=False)

print("Submission file 'submission.csv' created successfully.")
