import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians = None):
    # Creating a copy of the raw data file
    clean_df = raw.copy()

    # Removing exact duplicate records
    clean_df = clean_df.drop_duplicates()

    # Here, we are standardizing the service type names
    clean_df['service_type'] = clean_df['service_type'].str.strip().str.title()

    # Here, we are replacing missing payment methods with 'Unknown'
    clean_df['payment_mode'] = clean_df['payment_mode'].fillna('Unknown')

    # Here, we are removing the rupee sign and the thousands commas, converting to a number, store it in a new column
    clean_df['declared_value_inr'] = pd.to_numeric(clean_df['declared_value'].str.replace('₹','',regex=False).str.replace(',','',regex=False), errors = 'coerce')

    # Since we created a new column above, we are removing the original "declared_value" column
    clean_df = clean_df.drop(columns = ['declared_value'])

    # Here, we are converting the dispatch_time column to a datetime
    clean_df['dispatch_time'] = pd.to_datetime(clean_df['dispatch_time'])

    # Here, we are checking for a parcel weight of more than 100kgs since they are scanner-faults
    # We will treat these as missing values
    clean_df.loc[clean_df['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan

    # Here, we are checking the numerical columns that requires calculating the median
    numeric_cols = ['distance_km','parcel_weight_kg','declared_value_inr']

    #Calculating the median for the numerical columns
    if medians is None:
        medians = {}

        for col in numeric_cols:
            medians[col] = float(clean_df[col].median())

    # Here, we are filling the missing values with the median with calculated above
    for col in numeric_cols:
        clean_df[col] = clean_df[col].fillna(medians[col])

    return clean_df, medians

def build_features(clean, hubs):
    features_df = clean.copy()

    # Here, we are attaching the hub attributes to each shipment
    hub_cols = ['hub_id','hub_city','region','hub_tier','hub_daily_capacity']

    features_df = features_df.merge(hubs[hub_cols], on = 'hub_id', how = 'left', validate = 'many_to_one')

    # Here, we are assigning promised hours for every shipments service type
    features_df['promised_hours'] = features_df['service_type'].map(PROMISED_HOURS)

    # Here, we are assigning 1 to COD and 0 to everything else
    features_df['is_cod'] = np.where(features_df['payment_mode'] == 'COD',1,0)

    # Here, we are assigning 1 for weekend dispatches and 0 for weekday dispatches
    features_df['is_weekend'] = np.where(features_df['dispatch_time'].dt.dayofweek >= 5,1,0)

    # Here, we are understanding the time a shipment leaves
    features_df['dispatch_hour'] = features_df['dispatch_time'].dt.hour

    # Here, we are applying a natural log transformation to reduce skewness in parcel values
    features_df['log_declared_value'] = np.log(features_df['declared_value_inr'])

    return features_df
