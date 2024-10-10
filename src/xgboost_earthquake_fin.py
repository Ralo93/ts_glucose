from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
import mlflow
import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score, make_scorer, mean_squared_error, precision_score, recall_score
from lightgbm import LGBMClassifier, LGBMRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import BaseNEncoder
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from c_preprocessing import BloodGlucosePreprocessing
pd.set_option('future.no_silent_downcasting', True)


# Define the search space for hyperparameter optimization
space = {
    "n_estimators": hp.uniformint("n_estimators", 300, 1000),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
    "max_depth": hp.uniformint("max_depth", 3, 7),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1),
    "subsample": hp.uniform("subsample", 0.6, 0.95),
    "min_child_weight": hp.uniformint("min_child_weight", 1, 10),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-8), np.log(10)),  # L2 regularization
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-8), np.log(10)),    # L1 regularization
    "min_split_gain": hp.loguniform("min_split_gain", np.log(1e-8), np.log(10))  # Equivalent to gamma
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading Data...")
preprocessing = BloodGlucosePreprocessing(r"C:\Users\rapha\repositories\ts_glucose\data\raw/blood_glucose/train.csv")
#preprocessing.filter_patient_data(patient, bg_only=False, bg_or_plus_one=False)
preprocessing.downcast_floats()
# bg_only=False, bg_or_plus_one=True
#print(preprocessing.df)

data = preprocessing.df.copy()
logging.info("Done")


# Root mean squared error function
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Assuming df is your DataFrame, with the last column as the target
print("Shape of the dataframe:", data.shape)

X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The target column

X = X.drop(columns=['id'])

print("Shape of X (features):", X.shape)
print("Shape of y (target):", y.shape)

print("X columns:", X.columns)
print("y name:", y.name)



# Select categorical and numerical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns
num_cols = X.select_dtypes(exclude=['object', 'category']).columns

print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# Base model pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('baseN', BaseNEncoder(base=3), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)



# XGBoost with regularization (using reg_alpha and reg_lambda)
xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor())
])

# Cross-validation with RMSE
rmse_scorer = make_scorer(root_mean_squared_error)
##scores = cross_val_score(meta_model, x_train, y_train, cv=5, scoring=rmse_scorer)#


def create_model(params):
    return XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        colsample_bytree=params['colsample_bytree'],
        subsample=params['subsample'],
        min_child_weight=int(params['min_child_weight']),
        reg_lambda=params['reg_lambda'],  # L2 regularization
        reg_alpha=params['reg_alpha'],  # L1 regularization
        gamma=params['min_split_gain'],  # Split gain
        random_state=42,
        objective='reg:squarederror'  # Objective function for regression
    )



pipe = xgb

def cross_validate_with_downsampling(params, X, y): 
    mlflow.xgboost.autolog()

    logging.info("Starting Cross-Validation with Downsampling...")

    pipe.set_params(
        xgb__n_estimators=params['n_estimators'],
        xgb__learning_rate=params['learning_rate'],
        xgb__max_depth=params['max_depth'],
        xgb__subsample=params['subsample'],
        xgb__reg_lambda=params['reg_lambda'],  # L2 regularization
        xgb__colsample_bytree=params['colsample_bytree'],
        xgb__min_child_weight=int(params['min_child_weight']),
        xgb__reg_alpha=params['reg_alpha']  # L1 regularization
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    train_rmse_scores = []

    use_cross_validation = True

    with mlflow.start_run(nested=True):
        if use_cross_validation:
            for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
                logging.info(f"Starting fold {fold_idx + 1}...")

                # Split the data
                X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
                y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

                # Train the model on the training data for this fold
                pipe.fit(X_train_fold, y_train_fold)

                # Calculate training metrics (RMSE)
                train_preds = pipe.predict(X_train_fold)
                train_rmse = np.sqrt(mean_squared_error(y_train_fold, train_preds))

                # Calculate validation metrics (RMSE)
                valid_preds = pipe.predict(X_valid_fold)
                fold_rmse = np.sqrt(mean_squared_error(y_valid_fold, valid_preds))

                # Append metrics
                train_rmse_scores.append(train_rmse)
                rmse_scores.append(fold_rmse)

            # Log cross-validation metrics
            mlflow.log_metric("train_cv_mean_rmse", np.mean(train_rmse_scores))
            mlflow.log_metric("train_cv_std_rmse", np.std(train_rmse_scores))

            mlflow.log_metric("cv_mean_rmse", np.mean(rmse_scores))
            mlflow.log_metric("cv_std_rmse", np.std(rmse_scores))

            result_loss = np.mean(rmse_scores)

        mlflow.set_tag("tag", "rmse now")

        logging.info("Training process completed")

        return {"loss": result_loss, "status": STATUS_OK, "model": pipe}



def objective(params):
    return cross_validate_with_downsampling(params, X, y)


# Set the MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("/c_xgb_full")

# Start a new MLflow run for hyperparameter optimization
with mlflow.start_run():
    logging.info("Starting MLflow with Cross-Validation...")

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
    mlflow.log_params(best)
    mlflow.log_metric("micro_f1_score", -best_run["loss"])

    lightgbm = best_run["model"].named_steps['xgb']
    mlflow.sklearn.log_model(pipe, "model")

    print(f"Best parameters: {best}")
    print(f"Best eval macro F1 score: {-best_run['loss']}")
