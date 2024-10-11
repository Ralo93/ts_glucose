from competition.c_preprocessing import *
from competition.c_helpers import *

patient = "p02"
# Example usage:
preprocessing = BloodGlucosePreprocessing("data/raw/blood_glucose/train.csv")
preprocessing.downcast_floats()
print(preprocessing.df.shape)
print(preprocessing.df)
#preprocessing.convert_object_to_category()

preprocessing.filter_patient_data(patient=patient, bg_only=True)
print(preprocessing.df.shape)
print(preprocessing.df)

preprocessing.adjust_time_with_date(start_date="2024-10-02")
print(preprocessing.df.shape)
print(preprocessing.df)

#preprocessing.filter_hourly_data()
preprocessing.df = fill_missing_datetime_rows(preprocessing.df)
print(preprocessing.df.shape)
print(preprocessing.df)

print(preprocessing.df.isna().any())
preprocessing.interpolate_bg()
print(preprocessing.df.shape)
print(preprocessing.df)

print("Any missing values in the dataframe:")
print(preprocessing.df.isna().any())
cont = check_continuous_datetime_index(preprocessing.df)
print(cont)


preprocessing.save_patient_data(patient=patient, save_path="data/processed/cleaned_up_patients")