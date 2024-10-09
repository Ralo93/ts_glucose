import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd


def encode_baseN(df, cols, base=3):
    print("base N encoding")
    # Instantiate the encoder
    encoder = ce.BaseNEncoder(cols=cols, return_df=True, base=base)
    # Fit and transform the data
    df_encoded = encoder.fit_transform(df[cols])
    return df_encoded, encoder


    # Custom BaseNEncoder class
class BaseNEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, base=3):
        self.columns = columns
        self.base = base
        self.encoder = None
        self.new_column_names = None

    def fit(self, X, y=None):
        # Fit BaseN encoder and get new column names
        self.encoder, new_columns = encode_baseN(X, self.columns, base=self.base)
        self.new_column_names = new_columns  # Save the new column names
        return self

    def transform(self, X):
        # Transform and return a DataFrame with the correct column names
        X_encoded = self.encoder.transform(X[self.columns])
        return pd.DataFrame(X_encoded, columns=self.new_column_names, index=X.index)
    
    def get_feature_names_out(self, input_features=None):
        return self.new_column_names

    def save_encoder(self, path):
        joblib.dump(self.encoder, path)
        print(f"BaseNEncoder saved to {path}")

    # Custom BaseNEncoder class
class BaseNEncoderA(BaseEstimator, TransformerMixin):
    def __init__(self, columns, base=3):
        self.columns = columns
        self.base = base
        self.encoder = None
        self.new_column_names = None

    def fit(self, X, y=None):
        # Fit BaseN encoder and get new column names
        self.encoder, new_columns = encode_baseN(X, self.columns, base=self.base)
        self.new_column_names = new_columns  # Save the new column names
        return self

    def transform(self, X):
        # Transform and return a DataFrame with the correct column names
        X_encoded = self.encoder.transform(X[self.columns])
        return pd.DataFrame(X_encoded, columns=self.new_column_names, index=X.index)
    
    def get_feature_names_out(self, input_features=None):
        return self.new_column_names

    def save_encoder(self, path):
        joblib.dump(self.encoder, path)
        print(f"BaseNEncoder saved to {path}")

class ActivityAutoEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_names, encoding_dim, autoencoder_id):
        self.column_names = column_names  # Column names for input
        self.encoding_dim = encoding_dim
        self.autoencoder_id = autoencoder_id
        self.autoencoder = None

    def fit(self, X, y=None):
        # Train the autoencoder to reduce the dimensionality of the specified columns
        X_selected = X[self.column_names]
        self.autoencoder = autoencode_columns_999(X_selected, column_indices=None, encoding_dim=self.encoding_dim, autoencoder_id=self.autoencoder_id)
        return self

    def transform(self, X, train=True):
        # Only return the encoded (reduced) data
        X_selected = X[self.column_names]
        if train:
            X_autoencoded = autoencode_columns_999(X_selected, column_indices=None, encoding_dim=self.encoding_dim, autoencoder_id=self.autoencoder_id)
        else:
            X_autoencoded = autoencode_columns_999(X_selected, column_indices=None, encoding_dim=self.encoding_dim, autoencoder_id=self.autoencoder_id, train=False)
        # Return only the forward-pass (encoded) data, not reconstruction
        return pd.DataFrame(X_autoencoded, index=X.index)

