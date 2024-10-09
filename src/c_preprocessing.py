import os

import pandas as pd
import numpy as np




class Preprocessor:  
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def preprocess(self):
        pass


    def _impute_missing_values(self):
        pass



    def _check_for_missing_values(self):
        pass


    def _reduce_memory_usage(self):


            # Function to downcast float64 to float32
        # Downcast float64 columns to float32
        float64_cols = df.select_dtypes(include=['float64']).columns
        df[float64_cols] = df[float64_cols].astype(np.float32)
        
        # Convert object columns to category
        object_cols = df.select_dtypes(include=['object']).columns
        df[object_cols] = df[object_cols].astype('category')
        
        
        return df



df = pd.read_csv(r"C:\Users\Administrator\Desktop\raphi_other\repositories\ts\data/processed/concat/hourly_avg_data_stop.csv")


pp = Preprocessor(df)
df = Preprocessor._reduce_memory_usage(df)
print(df)















