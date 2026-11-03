import pandas as pd

from src.shipment_prep import clean_shipments, build_features, FEATURES

def main():
    # Here, we are loading the raw_shipments and hub data
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # Here, we are applying the cleaning and feature-engineering functions
    clean,medians = clean_shipments(raw)
    modelling_df = build_features(clean,hubs)

    modelling_df['is_delayed'] = (
        modelling_df['delivery_hours'] > modelling_df['promised_hours']
    ).astype(int)

    modelling_df.to_csv("data/modelling_table.csv",index=False)
    # Here, we are calculating the delivery hour quartiles and IQR outlier boundaries
    hours_q1 = modelling_df['delivery_hours'].quantile(0.25)
    hours_q3 = modelling_df['delivery_hours'].quantile(0.75)

    hours_iqr = hours_q3 - hours_q1
    hours_lower_bound = hours_q1 - 1.5 * hours_iqr
    hours_upper_bound = hours_q3 + 1.5 * hours_iqr

    # Here, we are identifying shipments that are outside the IQR boundaries
    hours_outlier_count = ((modelling_df['delivery_hours'] < hours_lower_bound) | (modelling_df['delivery_hours'] > hours_upper_bound)).sum()

    # Here, we are removing the duplicates and saving before making calculations
    duplicate_raw = raw.drop_duplicates()

    # Here, we are calculating the metrics and its value to cross check the same in the reference table
    checks = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": len(raw) - len(clean),
        "rows_after_dedup": len(clean),
        "delayed_count": int(modelling_df['is_delayed'].sum()),
        "delay_rate": round(modelling_df['is_delayed'].mean(),4),
        "missing_distance_km_before": int(duplicate_raw['distance_km'].isna().sum()),
        "missing_parcel_weight_before": int(duplicate_raw['parcel_weight_kg'].isna().sum()),
        "missing_declared_value_before": int(duplicate_raw['declared_value'].isna().sum()),
        "missing_payment_mode_before": int(duplicate_raw['payment_mode'].isna().sum()),
        "sentinel_weight_rows": int((duplicate_raw['parcel_weight_kg'] > 100).sum()),
        "service_type_spellings_before": int(raw["service_type"].nunique()),
        "service_type_spellings_after":int(clean["service_type"].nunique()),
        "median_distance_km": round(medians['distance_km'], 4),
        "median_parcel_weight_kg": round(medians['parcel_weight_kg'], 4),
        "median_declared_value_inr": round(medians['declared_value_inr'],4),
        "rows_after_hub_merge": len(modelling_df),
        "nulls_in_features_and_target": int(modelling_df[FEATURES + ['delivery_hours','is_delayed']].isna().sum().sum()),
        "hours_q1": round(hours_q1, 4),
        "hours_q3": round(hours_q3,4),
        "hours_lower_bound": round(hours_lower_bound,4),
        "hours_upper_bound": round(hours_upper_bound,4),
        "hours_outlier_count": int(hours_outlier_count),
        "declared_value_mean_inr": round(modelling_df['declared_value_inr'].mean(),4),
        "declared_value_median_inr": round(modelling_df['declared_value_inr'].median(),4),
        "log_declared_value_mean": round(modelling_df['log_declared_value'].mean(),4),
        "log_declared_value_median": round(modelling_df['log_declared_value'].median(),4)
    }

    # Here, we are saving them into a csv file
    checks_df = pd.DataFrame(list(checks.items()), columns = ['metric','value'])
    checks_df.to_csv('outputs/part_a_checks.csv', index=False)

    # For each delivery service, here we are summarizing there parameters
    delays_in_service = (modelling_df.groupby("service_type").agg(shipments = ('is_delayed','size'), delayed = ('is_delayed','sum'), delay_rate = ('is_delayed','mean'), mean_hours = ('delivery_hours','mean')).reset_index())
    delays_in_service['delay_rate'] = delays_in_service['delay_rate'].round(4)
    delays_in_service['mean_hours'] = delays_in_service['mean_hours'].round(4)

    delays_in_tier = (modelling_df.groupby('hub_tier').agg(shipments = ("is_delayed","size"), delayed = ("is_delayed","sum"), delay_rate = ("is_delayed","mean"), mean_hours = ('delivery_hours','mean')).reset_index())
    delays_in_tier['delay_rate'] = delays_in_tier['delay_rate'].round(4)
    delays_in_tier['mean_hours'] = delays_in_tier['mean_hours'].round(4)

    # Here, we are dividing the distance_band into 4 bands mentioned in the question
    modelling_df['distance_band'] = pd.cut(modelling_df['distance_km'],
    bins = [0,200,400,600,2000],
    labels = ["0-200","201-400","401-600","601+"],
    right = True,
    include_lowest = True)

    # Here, we are calculating the shipment count and delay rates for each distance_band
    delays_in_distance = (modelling_df.groupby('distance_band',observed=True).agg(shipments = ('is_delayed','size'), delayed = ('is_delayed','sum'), delay_rate = ('is_delayed','mean'), mean_hours = ('delivery_hours','mean')).reset_index())
    delays_in_distance['delay_rate'] = delays_in_distance['delay_rate'].round(4)
    delays_in_distance['mean_hours'] = delays_in_distance['mean_hours'].round(4)

    # Here, we are giving common column for each grouping
    delays_in_service = delays_in_service.rename(columns = {"service_type": "group"})
    delays_in_service.insert(0, "dimension", "service_type")
    delays_in_tier = delays_in_tier.rename(columns = {"hub_tier": "group"})
    delays_in_tier.insert(0,"dimension","hub_tier")
    delays_in_distance = delays_in_distance.rename(columns = {"distance_band":"group"})
    delays_in_distance.insert(0, "dimension", "distance_band")

    # Here, we are combining the analysis to generate the csv
    delays_by_group = pd.concat([delays_in_service, delays_in_tier, delays_in_distance], ignore_index = True)
    delays_by_group = delays_by_group[["dimension","group","shipments","delayed","delay_rate","mean_hours"]]
    delays_by_group.to_csv("outputs/part_a_delay_by_group.csv", index = False)

    # Here, we are calculating the correlation matrix of the numeric columns of the modelling table
    numeric_df = modelling_df.select_dtypes(include="number")
    correlations = numeric_df.corr()
    correlations.round(4).to_csv("outputs/part_a_correlations.csv", index = True)

if __name__ == "__main__":
    main()