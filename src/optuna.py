import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error
import gc  # Import garbage collection for manual memory management


def remove_columns(df, columns_to_remove):
    """
    Remove specified columns from the DataFrame.

    :param df: Input DataFrame.
    :param columns_to_remove: List of column names to remove.
    :return: DataFrame with specified columns removed.
    """
    df = df.drop(columns=columns_to_remove, errors='ignore')
    return df


# Function to reduce memory usage by downcasting numerical data types
def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            # Convert object types to category type for memory efficiency
            df[col] = df[col].astype('category')

    return df


def preprocess_and_train_with_optuna(dataset_path, target_variable, columns_to_remove=None, n_trials=3,
                                     use_knn_imputer=True):
    """
    Preprocesses the dataset and trains a model with hyperparameter tuning using Optuna.

    Args:
    - dataset_path: Path to the dataset (CSV file).
    - target_variable: The name of the target column in the dataset.
    - columns_to_remove: List of columns to remove from the dataset (default is None).
    - n_trials: Number of trials for Optuna hyperparameter tuning (default is 3).
    - use_knn_imputer: Whether to use KNN imputer for missing data (default is True).

    Returns:
    - study: Optuna study object after hyperparameter tuning.
    """

    # Load the dataset entirely into memory
    data = pd.read_csv(dataset_path)

    # Reduce memory usage if needed (implement reduce_memory_usage function as per your requirements)
    data = reduce_memory_usage(data)

    # Drop specified columns if provided
    if columns_to_remove is not None:
        data = remove_columns(data, columns_to_remove)

    # Check if the target variable exists in the dataset
    if target_variable not in data.columns:
        raise ValueError(f"{target_variable} not found in the dataset")

    # Separate features and target
    X = data.drop(columns=[target_variable])
    y = data[target_variable]

    # Optimize target column by converting to float32
    y = y.astype(np.float32)

    # Identify numerical and categorical columns
    numerical_features = X.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Intelligent imputation for numerical features
    if use_knn_imputer:
        # Use KNN Imputer for numerical columns (more advanced, but can be heavy on memory)
        numerical_imputer = KNNImputer(n_neighbors=3)
    else:
        # Use Median Imputer for numerical columns
        numerical_imputer = SimpleImputer(strategy='median')

    # Preprocessing for numerical features (KNN Imputation or Median Imputation and Scaling)
    numerical_transformer = Pipeline(steps=[
        ('imputer', numerical_imputer),
        ('scaler', StandardScaler())
    ])

    # Intelligent imputation for categorical features (add 'Missing' as a new category)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine numerical and categorical transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the objective function for Optuna
    def objective(trial):
        # Define hyperparameter search space
        xgboost_param = {
            'device': 'gpu',
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist',  # Enable GPU for XGBoost
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        }

        # Create LightGBM model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xg.XGBRegressor(**xgboost_param, random_state=42))
        ])
        print("Training")
        # Cross-validation with 5-fold to evaluate the performance using RMSE
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        rmse = -np.mean(cv_scores)  # RMSE (scikit-learn uses negative RMSE by default)
        return rmse

    # Create an Optuna study and optimize it
    study = optuna.create_study(direction='minimize')

    study.optimize(objective, n_trials=n_trials)

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    # Train the final model using the best hyperparameters
    best_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(**best_params, random_state=42))
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    best_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Calculate and print final performance using RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Test RMSE: {rmse}')

    # Explicit garbage collection
    gc.collect()

    return best_model, study


dataset_path = pd.read_csv(r'../data/raw/blood_glucose/train.csv')
target_variable = 'bg+1:00'

best_model, study = preprocess_and_train_with_optuna(dataset_path,
                                                     target_variable)  # , columns_to_remove=None, n_trials=3, use_knn_imputer=True, chunk_size=5000):



