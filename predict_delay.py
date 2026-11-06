import pandas as pd
import joblib 
from pathlib import Path
from src.shipment_prep import clean_shipments, build_features

def main():

    # Here, we are loading the new shipments and hub information
    new_shipments = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # Here, we are loading the trained classification pipeline
    model = joblib.load("models/delay_model.pkl")

    # Here, we are loading the classification threshold
    threshold_df = pd.read_csv("models/delay_threshold.csv")
    threshold = float(threshold_df["threshold"].iloc[0])

    # Here, we are loading the original feature order
    feature_order_df = pd.read_csv("models/feature_order.csv")
    feature_order = feature_order_df["feature"].tolist()

    # Here, we are loading the historical imputation values
    impute_values_df = pd.read_csv("models/impute_values.csv")

    # Here, we are loading the hub segments cerated earlier in Part D
    hub_segments = pd.read_csv("outputs/hub_segments.csv")

    # Here, we are verifying that all required inputs have been loaded
    print("Total New shipments: ", len(new_shipments))
    print("Total hubs: ", len(hubs))
    print("Classification threshold is: ", threshold)
    print("Number of model features: ", len(feature_order))
    print("Number of Imputation values: ", len(impute_values_df))
    print("Total hub segments recorded: ", len(hub_segments))

    # Here, we are converting the saved imputation values into a dictionary
    medians = dict(zip(impute_values_df["column"],impute_values_df["median"]))

    # Here, we are cleaning new shipments using the historical medians
    clean_new, _ = clean_shipments(new_shipments, medians = medians)

    # Here, we are verifying that all new shipments are retained
    print("Cleaned new shipments: ", len(clean_new))

    # Here, we are verifying that the numeric columns have no missing values
    numeric_cols = ["distance_km","parcel_weight_kg","declared_value_inr"]
    print("Missing numeric values after cleaning: ")
    print(clean_new[numeric_cols].isna().sum())

    # Here, we are building the features using our existing function
    features_df = build_features(clean_new, hubs)

    # Here, we are arranging the features in the saved training order
    X_new = features_df[feature_order]

    # Here, we are verifying the feature dimensions
    print("Feature rows: ", len(X_new))
    print("Feature columns: ", len(X_new.columns))

    # Here, we are predicting the probability of delay for each shipment
    probabilities = model.predict_proba(X_new)[:,1]

    # Here, we are flagging shipments using the original unrounded probabilities
    flagged = (probabilities >= threshold).astype(int)

    # Here, we are storing the rounded probabilities and flags
    features_df["delay_probability"] = probabilities.round(4)
    features_df["flagged"] = flagged

    # Here, we are displaying the number of flagged shipments
    print("Total flagged shipments: ", int(features_df["flagged"].sum()))

    # Here, we are attaching the hub segments to each shipment
    actions_df = features_df.merge(hub_segments[["hub_id","segment"]],
    on = "hub_id",
    how = "left",
    sort = False,
    validate = "many_to_one")

    # Here, we are verifying that every shipment has a hub segment
    assert actions_df["segment"].notna().all(), "Some hub segments are missing"

    # Here, we are assigning "No action" to all shipments initially
    actions_df["action"] = "No action"

    # Here, we are identifying flagged shipments
    flagged_mask = actions_df["flagged"] == 1

    # Here, we are assigning proactive customer calls to flagged shipments
    actions_df.loc[flagged_mask, "action"] = "Proactive customer call"

    # Here, we are identifying flagged shipments in Escalate hubs
    escalate_mask = flagged_mask & (actions_df["segment"] == "Escalate")

    # Here, we are assigning hub manager escalations
    actions_df.loc[escalate_mask, "action"] = "Escalate to hub manager"

    # Here, we are selecting the required columns in the correct order
    output_columns = ["shipment_id","hub_id","segment","delay_probability","flagged","action"]

    final_df = actions_df[output_columns]

    # Here, we are verifying that the saved probabilitis and flags agree
    assert (final_df["flagged"] == (final_df["delay_probability"] >= threshold).astype(int)).all(), "Flagged values do not match the saved probabilities."

    # Here, we are verifying the number of output shipments
    assert len(final_df) == len(new_shipments), "Shipment count does not match"

    # Here, we are verifying that shipments IDs are unique
    assert final_df["shipment_id"].is_unique, "Duplicate shipment IDs found."

    # Here, we are verifying that the original shipment order is preserved
    assert final_df["shipment_id"].tolist() == new_shipments["shipment_id"].tolist(), ("Original shipment order was not preserved")

    # Here, we are verifying that the final output contains no missing values
    assert final_df.notna().all().all(), "Missing values found in the output"

    # Here, we are creating the outputs directory if it does not already exist
    Path("outputs").mkdir(parents = True, exist_ok = True)

    # Here, we are saving the final shipment actions
    final_df.to_csv("outputs/new_shipment_actions.csv", index = False, float_format = "%.4f")

    # Here, we are displaying the final output summary
    print("Part D - New Shipment Scoring Results")
    print("Total shipments scored: ", len(final_df))
    print("Total flagged shipments:", int(final_df["flagged"].sum()))
    print("Hub manager escalations:", int(escalate_mask.sum()))

    # Here, we are displaying the distribution of operational actions
    print("Following is the distribution of operational actions:")
    print(final_df["action"].value_counts())

    # Here, we are displaying the first five score shipments
    print("Following is the first five scored shipments: ")
    print(final_df.head().to_string(index = False))
    print("Saved output: outputs/new_shipment_actions.csv")

if __name__ == "__main__":
    main()
