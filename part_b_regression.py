import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from src.shipment_prep import FEATURES

def main():
    # Here, we are loading the modelling dataset we created in PART A
    modelling_df = pd.read_csv("data/modelling_table.csv")

    # Following are feature variables and target variables
    X = modelling_df[FEATURES]
    y = modelling_df['delivery_hours']

    # Here, we are understanding the shape of the dataset, feature matrix and target variable
    print("Shape of the dataset: ", modelling_df.shape)
    print("Shape of the feature matrix: ", X.shape)
    print("Shape of the target variable: ", y.shape)

    # Here, we are splitting the dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size = 0.3,random_state = 42)
    X_val,X_test, y_val, y_test = train_test_split(X_temp,y_temp, test_size = 0.50, random_state = 42)

    print("The number of rows in the training set after splitting are: ", len(X_train))
    print("The number of rows in the validation set after splitting are: ", len(X_val))
    print("The number of rows in the test set after splitting are: ", len(X_test))

    # Here, we are using the training data to fit the scaler
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)

    # Here, in the similar manner we are implementing the training-fitted scaler to our validation and test data
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)

    # Here, we are calculating the largest absolute column mean of the scaled training row and scaled test row
    train_scaled_max_abs_mean = np.abs(X_train_scaled.mean(axis = 0)).max()
    test_scaled_max_abs_mean = np.abs(X_test_scaled.mean(axis = 0)).max()

    # Print the above calculation results to check
    print("The max absolute mean for the scaled train set is: ", train_scaled_max_abs_mean)
    print("The max absolute mean for the scaled test set is: ", test_scaled_max_abs_mean)

    # Here, we are storing the required results mentioned in the question in part_b_checks.csv
    part_b_checks = {
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "train_scaled_max_abs_mean": train_scaled_max_abs_mean,
        "test_scaled_max_abs_mean": test_scaled_max_abs_mean
    }

    checks_df = pd.DataFrame({
        "metric":part_b_checks.keys(),
        "value":part_b_checks.values()
    })  

    # Following is the path wherein we will be storing the file
    checks_df.to_csv("outputs/part_b_checks.csv", index = False)

    # Here, we are calculating the mean delivery duration from our training data
    baseline_mean_y_train = y_train.mean()

    # Here, we are trying to predict the same value for every shipment in the validation set
    baseline_val_prediction = [baseline_mean_y_train] * len(y_val)

    # Here, we are evaluating the performance on the validation set
    baseline_mae = mean_absolute_error(y_val, baseline_val_prediction)
    baseline_mse = mean_squared_error(y_val, baseline_val_prediction)
    baseline_rmse = np.sqrt(baseline_mse)

    print("The y-train baseline mean delivery hours are: ", round(baseline_mean_y_train,4))
    print("Baseline MAE: ",round(baseline_mae,4))
    print("Baseline MSE: ",round(baseline_mse,4))
    print("Baseline RMSE: ", round(baseline_rmse,4))

    # Here, we are defining the tree depths required for the experiment
    tree_depths = [1,2,3,4,6,8,12,None]

    # Here, we are storing the results for each tree depth
    tree_results = []

    # Here, we are implementing the tree-depth diagnostics
    for d in tree_depths:
        # Here, we are creating and training a Decision Tree with the current depth
        tree_model = DecisionTreeRegressor(
            max_depth = d,
            random_state = 42
        )
        tree_model.fit(X_train, y_train)

        # Here, we are predicting the delivery hours for our training and validation shipments
        train_predictions = tree_model.predict(X_train)
        val_predictions = tree_model.predict(X_val)

        # Here, we are calculating RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

        tree_results.append({
            "max_depth": "unlimited" if d is None else d,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse
        })
    
    # Here, we are creating the CSV as instructed
    tree_depth_df = pd.DataFrame(tree_results)
    tree_depth_df.to_csv("outputs/part_b_tree_depth.csv", index = False)
    print(tree_depth_df)
    
    # Here, we are finding out the depth with the lowest val_rmse, as it will be selected as the decision-tree depth used in the next task
    lowest_val_rmse_depth = tree_depth_df.sort_values("val_rmse").iloc[0]

    print("The ideal Decision Tree depth: ", lowest_val_rmse_depth['max_depth'])
    print("The ideal validation RMSE: ", round(lowest_val_rmse_depth['val_rmse'],4))

    # Here, we are using the ideal decision tree depth we identified in point 4
    ideal_depth = int(lowest_val_rmse_depth["max_depth"])

    # Here, we are defining our four candidate model
    models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "decision_tree":DecisionTreeRegressor(
            max_depth = ideal_depth,
            random_state = 42
        ),
        "random_forest": RandomForestRegressor(
            n_estimators = 100,
            random_state = 42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators = 100,
            learning_rate = 0.1,
            random_state = 42
        )
    }

    baseline_r2 = r2_score(y_val, baseline_val_prediction)

    model_results = []

    model_results.append({
        "model":"mean_baseline",
        "split":"val",
        "MAE":baseline_mae,
        "MSE":baseline_mse,
        "RMSE":baseline_rmse,
        "R2":baseline_r2
    })

    for model_name, model in models.items():
        
        # Here, we are training the model using the training set
        model.fit (X_train, y_train)

        # Here, we are predicting delivery hours using the validation set
        val_predictions = model.predict(X_val)

        # Here, we are calculating the metrics
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_mse = mean_squared_error(y_val, val_predictions)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(y_val, val_predictions)

        # Here, we are storing the results in CSV
        model_results.append({
            "model": model_name,
            "split": "val",
            "MAE": val_mae,
            "MSE": val_mse,
            "RMSE": val_rmse,
            "R2": val_r2
        })

    # Here, we are converting the model results into a DataFrame
    metrics_df = pd.DataFrame(model_results)

    # Here, we are exporting the validation metrics
    metrics_df.to_csv("outputs/part_b_metrics.csv", index = False)
    print(metrics_df)

    # Following are the learning_rates
    learning_rates = [0.01,0.1,1.0]
    gradientboost_results = []

    for lr in learning_rates:
        gradientboost_model = GradientBoostingRegressor(
            n_estimators = 100,
            learning_rate = lr,
            random_state = 42
        )

        gradientboost_model.fit(X_train, y_train)
        gradientboost_val_predictions = gradientboost_model.predict(X_val)

        gradientboost_val_mse = mean_squared_error(y_val, gradientboost_val_predictions)
        gradientboost_val_rmse = np.sqrt(gradientboost_val_mse)
        gradientboost_val_r2 = r2_score(y_val, gradientboost_val_predictions)

        gradientboost_results.append({
            "learning_rate": lr,
            "val_rmse": gradientboost_val_rmse,
            "val_r2":gradientboost_val_r2
        })

    gradientboost_results_df = pd.DataFrame(gradientboost_results)
    gradientboost_results_df.to_csv("outputs/part_b_gb_learning_rate.csv", index = False)
    print(gradientboost_results_df)

    # Here, we are selecting the top two candidates models using validation RMSE
    candidate_model_metrics = metrics_df[metrics_df["model"] != "mean_baseline"]
    top_two_models = candidate_model_metrics.sort_values(by="RMSE").head(2)

    # Printing top two models
    print(top_two_models[["model", "RMSE"]])

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    # Here, we are adding the training and validation rows
    print("The total number of rows after combining training and validation rows are: ", len(X_train_val))

    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    cv_results = []

    # Here, we are running the two best, run 5-fold cross-validation
    for model_name in top_two_models["model"]:
        model = models[model_name]

        cv_scores = cross_val_score(model,X_train_val, y_train_val, cv = cv, scoring = "r2")
        cv_results.append({
            "model" : model_name,
            "mean_r2" : cv_scores.mean(),
            "std_r2" : cv_scores.std()
        })

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv("outputs/part_b_cv.csv", index = False)
    print(cv_df)

    # Here, we are finding out the best model out of the two previously selected models
    best_model_name = top_two_models.iloc[0]["model"]
    final_model = models[best_model_name]
    print("The selected best model is: ", best_model_name)

    # Here, we are evaluating the model on the test set
    test_predictions = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test,test_predictions)

    model_results.append({
        "model" : best_model_name,
        "split" : "test",
        "MAE": test_mae,
        "MSE": test_mse,
        "RMSE": test_rmse,
        "R2": test_r2 
    })

    metrics_df = pd.DataFrame(model_results)
    metrics_df.to_csv("outputs/part_b_metrics.csv", index = False)
    print(metrics_df)

if __name__ == "__main__":
    main()