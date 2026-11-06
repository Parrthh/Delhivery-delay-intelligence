import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, clean_shipments

def main():

    # Here, we are loading the modelling dataset created in part A
    df = pd.read_csv("data/modelling_table.csv")

    # Here, we are assigning the input features to X and delay labels to y
    X = df[FEATURES]
    y = df['is_delayed']

    # Here, we are reproducing the exact training split used in part C
    X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size = 0.3, random_state = 42, stratify = y)

    # Here, we are reproducing the validation and test split
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42, stratify = y_temp)

    # Here, we are verifying that the dataset split matches part C
    print("The training rows: ", len(X_train))
    print("The validation rows: ", len(X_val))
    print("The testing rows: ", len(X_test))

    # Here, we are loading the final model configurations we saved in part C
    config_df = pd.read_csv("outputs/part_c_final_config.csv")

    # Here, we are retrieving the selected model configuration
    config = config_df.iloc[0]

    # Here, we are extracting the model parameters from the saved configuration
    selector_k = int(config['selector_k'])
    n_estimators = int(config["n_estimators"])
    min_samples_leaf = int(config["min_samples_leaf"])

    # Here, we are converting unlimited depth to None as required by Random Forest
    if str(config["max_depth"]) == "unlimited":
        max_depth = None
    else:
        max_depth = int(config["max_depth"])
    
    # Here, we are retrieving the classification threshold selected in part C
    threshold = float(config["threshold"])

    # Here, we are rebuilding the final classification pipeline
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k = selector_k)),
        ("model", RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf=min_samples_leaf, random_state = 42))
    ])

    # Here, we are fitting the pipeline using only the training shipments
    final_pipeline.fit(X_train, y_train)

    # Here, we are displaying the selected configuration
    print("The selected configuration is: ", config["chosen_from"])
    print("The number of selected features: ", selector_k)
    print("The number of trees: ", n_estimators)
    print("The maximum tree depth: ", max_depth)
    print("The minimum samples per leaf: ", min_samples_leaf)
    print("The selected threshold: ", threshold)

    # Here, we are calculating delay probabilities for the 180 test shipments
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]

    # Here, we are verifying the number of test probabilities generated
    print("The number of test probabilities: ", len(test_probabilities))
    print("The first five test probabilities: ", test_probabilities[:5])

    # Here, we are creating the model directory if it does not already exist
    Path("models").mkdir(parents = True, exist_ok = True)

    # Here, we are saving the trained classification pipeline using joblib
    joblib.dump(final_pipeline, "models/delay_model.pkl")

    # Here, we are confirming that the trained model has been saved
    print("The trained model has been saved to models/delay_model.pkl")

    # Here, we are saving the classification threshold selected in part C
    threshold_df = pd.DataFrame({"threshold" : [threshold]})

    threshold_df.to_csv("models/delay_threshold.csv", index = False)

    # Here, we are saving the 17 feature names in their original order
    feature_order_df = pd.DataFrame({"feature":FEATURES})
    feature_order_df.to_csv("models/feature_order.csv", index = False)

    # Here, we are loading the historical raw shipments used in part A
    raw_df = pd.read_csv("data/raw_shipments.csv")

    # Here, we are using our cleaning function to retrieve the original medians
    clean_df, medians = clean_shipments(raw_df)

    # Here, we are storing the three median values and their column names
    impute_values_df = pd.DataFrame({
        "column": ["distance_km","parcel_weight_kg","declared_value_inr"],
        "median":[medians["distance_km"],medians["parcel_weight_kg"],medians["declared_value_inr"]]
    })

    # Here, we are saving the medias for cleaning future shipments
    impute_values_df.to_csv("models/impute_values.csv", index = False)

    # Here, we are displaying the saved configuration details
    print("The classification threshold has been saved.")
    print("The feature order has been saved.")
    print("The imputation medians have been saved.")

    print("Saved threshold:")
    print(threshold_df.to_string(index = False))
    print("Saved feature count: ", len(feature_order_df))
    print("Saved imputation medians:")
    print(impute_values_df.to_string(index = False))

    # Here, we are loading the saved classification pipeline
    reloaded_pipeline = joblib.load("models/delay_model.pkl")

    # Here, we are calculating the delay probabilities using the reloaded model
    reloaded_probabilities = reloaded_pipeline.predict_proba(X_test)[:, 1]

    # Here, we are calculating the maximum absolute difference between the original model probabilities and the reloaded model probabilities
    max_abs_probability_diff = abs(test_probabilities - reloaded_probabilities).max()

    # Here, we are verifying that the saved model produces identical probabilities
    assert max_abs_probability_diff <= 1e-9, "Reloaded model probabilities do not match."

    # Here, we are storing the required reloaded verification metrics
    reload_check_df = pd.DataFrame({
        "metric": [
            "test_rows_scored",
            "reload_max_abs_probability_diff",
            "threshold"
        ],
        "value": [
            len(X_test),
            max_abs_probability_diff,
            threshold
        ]
    })

    # Here, we are saving the reload verification results
    reload_check_df.to_csv("outputs/part_d_reload_check.csv", index = False)

    # Here, we are displaying the reload verification results
    print("Part D Model Reload Verification: ")
    print(reload_check_df.to_string(index = False))


if __name__ == "__main__":
    main()
