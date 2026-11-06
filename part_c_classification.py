import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import FEATURES, NOISE_COLUMNS

def main():
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df['is_delayed']

    X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size = 0.3, random_state = 42, stratify = y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42, stratify = y_temp)

    # Here, we are checking if the split is according to the expected size
    print("The training rows are: ", len(X_train))
    print("The validation rows are: ", len(X_val))
    print("The testing rows are: ", len(X_test))
    print("The training delayed parcel are: ", y_train.sum())
    print("The validation delayed parcel are: ",y_val.sum())
    print("The test delayed parcel are: ", y_test.sum())

    # Here, we are assuming that every shipments reaches on time
    blind_all_on_time = [0] * len(y_val)

    # Here, we are comparing baseline predictions with actual delays to identify missed shipments
    cm = confusion_matrix(y_val, blind_all_on_time, labels = [0,1])
    tn, fp = cm[0]
    fn, tp = cm[1]

    # Here, we are calculating accuracy to understand how many predictions were correct
    baseline_accuracy = accuracy_score(y_val, blind_all_on_time)

    # Here, we are calculating precision to check when we predict a delay, how often we are correct
    baseline_precision = precision_score(y_val, blind_all_on_time, zero_division = 0)
    
    # Here, we are calculating recall to check how many actual delays did we identify
    baseline_recall = recall_score(y_val, blind_all_on_time)

    # Here, we are calculating f1_score 
    baseline_f1 = f1_score(y_val, blind_all_on_time)

    # Here, we are checking the values:
    print("Following are our Blind Baseline Results: ")
    print("TN: ", tn)
    print("FP: ", fp)
    print("FN: ", fn)
    print("TP: ", tp)
    print("Accuracy: ", baseline_accuracy)
    print("Precision: ", baseline_precision)
    print("Recall: ", baseline_recall)
    print("F1: ", baseline_f1)

    # Here, we are implementing a pipeline of StandardScaler and then Logistic Regression
    logistic_regression = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter = 1000, random_state = 42))
    ])

    # Here, we are fitting the scaler and classifier using training data only.
    logistic_regression.fit(X_train, y_train)

    # Here, we are evaluating the validation data without refitting
    logistic_predictions = logistic_regression.predict(X_val)

    # Here, we are checking how many delays the classifier is correctly identifying or missing
    cm = confusion_matrix(y_val, logistic_predictions, labels = [0,1])
    tn_lr, fp_lr = cm[0]
    fn_lr, tp_lr = cm[1]

    # Here, we are calculating accuracy, precision, recall and f1_score
    logistic_accuracy = accuracy_score(y_val, logistic_predictions)
    logistic_precision = precision_score(y_val, logistic_predictions, zero_division = 0)
    logistic_recall = recall_score(y_val, logistic_predictions)
    logistic_f1 = f1_score(y_val, logistic_predictions)

    # Here, we are checking the results recorded for Logistic Regression
    print("Following are the results for Logistic Regression:")
    print("TN: ", tn_lr)
    print("FP: ", fp_lr)
    print("FN: ", fn_lr)
    print("TP: ", tp_lr)
    print("Accuracy: ", logistic_accuracy)
    print("Precision: ", logistic_precision)
    print("Recall: ", logistic_recall)
    print("F1: ", logistic_f1)

    # Here, we are comparing tree depths to find the highest validation F1
    depths = [2,3,4,5,6,8,10]

    best_depth = None

    # The reason for -1 is: F1 score always falls between 0 and 1. Since -1 is lower than the worst F1 score, the first model evaluated will be our best model
    best_f1 = -1

    # Here, the for loop trains seven different Decision Trees one for each depth, and for every decision tree we calculate its f1_score using the validation data
    for d in depths:
        decision_tree = DecisionTreeClassifier(criterion='gini',max_depth = d, random_state = 42)

        decision_tree.fit(X_train, y_train)
        tree_predictions = decision_tree.predict(X_val)
        tree_f1 = f1_score(y_val, tree_predictions)
        
        # Here, we are printing the depth and the f1_score recorded
        print("Depth: ",d)
        print("F1: ", tree_f1)

        if tree_f1 > best_f1:
            best_f1 = tree_f1
            best_depth = d
    
    print("The Best Decision Tree depth is: ", best_depth)
    print("The Best Validation F1: ", best_f1)

    # Here, we are training the selected Decision Tree using the best validation depth we recorded in our previous run
    best_decision_tree = DecisionTreeClassifier(criterion = "gini", max_depth = best_depth, random_state = 42)
    best_decision_tree.fit(X_train,y_train)
    best_tree_predictions = best_decision_tree.predict(X_val)

    # Here, we are calculating the confusion matrix values for the selected decision tree
    cm = confusion_matrix(y_val, best_tree_predictions, labels = [0,1])
    tn_dt, fp_dt = cm[0]
    fn_dt, tp_dt = cm[1]

    # Here, we are calculating accuracy, precision, recall, and f1_score for the selected Decision Tree
    tree_accuracy = accuracy_score(y_val, best_tree_predictions)
    tree_precision = precision_score(y_val, best_tree_predictions, zero_division = 0)
    tree_recall = recall_score(y_val, best_tree_predictions)
    tree_f1_best = f1_score(y_val, best_tree_predictions)

    # Here, are the results recorded for the chosen Decision Tree
    print("Following are the results for the Decision Tree: ")
    print("Best Depth: ", best_depth)
    print("TN: ", tn_dt)
    print("FP: ", fp_dt)
    print("FN: ", fn_dt)
    print("TP: ",tp_dt)
    print("Accuracy: ", tree_accuracy)
    print("Precision: ", tree_precision)
    print("Recall: ", tree_recall)
    print("F1: ", tree_f1_best)

    # Here, we are training a Random Forest using 100 trees
    random_forest_default = RandomForestClassifier(n_estimators = 100, random_state = 42)
    random_forest_default.fit(X_train, y_train)

    # Here, we are predicting the shipments delays on the validation data
    random_forest_predictions = random_forest_default.predict(X_val)

    # Here, we are comparing the predicted delays with actual validation labels
    cm = confusion_matrix(y_val, random_forest_predictions, labels = [0,1])
    tn_rf, fp_rf = cm[0]
    fn_rf, tp_rf = cm[1]

    # Here, we are calculating the classification metrics for the Random Forest
    random_forest_accuracy = accuracy_score(y_val, random_forest_predictions)
    random_forest_precision = precision_score(y_val, random_forest_predictions, zero_division = 0)
    random_forest_recall = recall_score(y_val, random_forest_predictions)
    random_forest_f1 = f1_score(y_val, random_forest_predictions)

    # Here, are the results records for Random Forest
    print("Following are the results from Random Forest: ")
    print("TN: ", tn_rf)
    print("FP: ", fp_rf)
    print("FN: ", fn_rf)
    print("TP: ", tp_rf)
    print("Accuracy: ", random_forest_accuracy)
    print("Precision: ", random_forest_precision)
    print("Recall: ", random_forest_recall)
    print("F1: ", random_forest_f1)

    # Here, we are storing the validation metrics for all the classifiers
    model_results = [
        ["blind_all_on_time", tn, fp, fn, tp, baseline_accuracy, baseline_precision, baseline_recall, baseline_f1],
        ["logistic_regression", tn_lr, fp_lr, fn_lr, tp_lr, logistic_accuracy, logistic_precision, logistic_recall, logistic_f1],
        [f"decision_tree_depth_{best_depth}", tn_dt, fp_dt, fn_dt, tp_dt, tree_accuracy, tree_precision, tree_recall, tree_f1_best],
        [f"random_forest_default", tn_rf, fp_rf, fn_rf, tp_rf, random_forest_accuracy, random_forest_precision, random_forest_recall, random_forest_f1]
    ]

    # Here, we are converting the results into a DataFrame
    models_df = pd.DataFrame(model_results,columns = ["model","TN","FP","FN","TP","accuracy","precision","recall","f1"])
    models_df.to_csv("outputs/part_c_models.csv", index = False)
    print("Part C Model Comparison: ")
    print(models_df)

    # Here, we are reusing the default Random Forest to obtain feature importances
    rf_importance = random_forest_default

    # Here, we are calculating an ANOVA F-score for every training feature
    feature_selector = SelectKBest(f_classif, k = "all")
    feature_selector.fit(X_train, y_train)

    # Here, we are storing each features F-score and RF importance
    feature_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": feature_selector.scores_,
        "rf_importance": rf_importance.feature_importances_
    })

    # Here, we rank features by F-score
    feature_scores_df["f_rank"] = (
        feature_scores_df["f_score"].rank(ascending = False, method = "min").astype(int)
    )

    # Here, we are identifying all the five noise features
    feature_scores_df['is_noise'] = (
        feature_scores_df['feature'].isin(NOISE_COLUMNS)
    )

    # Here, can you match the exact column order that was given in the PART C Point 3
    feature_scores_df = feature_scores_df[["feature","f_score","f_rank","rf_importance","is_noise"]]
    feature_scores_df = feature_scores_df.sort_values("f_rank")
    feature_scores_df.to_csv("outputs/part_c_feature_scores.csv", index = False)

    print("Part C Feature Scores: ")
    print(feature_scores_df)

    # Here, we are measuring how much belongs to noise features
    noise_importance_share = feature_scores_df.loc[feature_scores_df['is_noise'], "rf_importance"].sum()
    print("The noise importance share are: ", noise_importance_share)

    # Here, we are comparing how Random Forests performs with different number of selected features
    k_values = [5,7,10,12,17]
    selectk_results = []

    for k in k_values:
        selectk_pipeline = Pipeline([
            ("selectk", SelectKBest(f_classif, k=k)),
            ("random_forest", RandomForestClassifier(
                n_estimators = 100,
                random_state = 42
            ))
        ])

        # Here, we are fitting feature selection and Random Forest using training data only
        selectk_pipeline.fit(X_train, y_train)

        # Here, we are evaluating Random Forest using the top k selected features
        selectk_predictions = selectk_pipeline.predict(X_val)
        val_f1 = f1_score(y_val, selectk_predictions)
        selectk_results.append([k, val_f1])
        print("k: ",k)
        print("Validation F1: ", val_f1)

    # Here, we are saving the validation F1 scores
    selectk_df = pd.DataFrame(selectk_results,columns = ['k','val_f1'])
    selectk_df.to_csv("outputs/part_c_selectk.csv", index = False)
    print("Part C SelectKBest Results: ")
    print(selectk_df)

    # Here, we are selecting the top 7 features using training data
    selector_7 = SelectKBest(f_classif, k = 7)
    selector_7.fit(X_train, y_train)

    # Here, we are transforming the training and validation data using the same selected features
    X_train_7 = selector_7.transform(X_train)
    X_val_7 = selector_7.transform(X_val)

    # Here, we are verifying the number of rows and selected features
    selected_features = []

    for feature,selected in zip(X_train.columns, selector_7.get_support()):
        if selected:
            selected_features.append(feature)
    
    print("The selected features: ", selected_features)
    print("The training shape is: ", X_train_7.shape)
    print("The Validation shape is: ",X_val_7.shape)

    # Here, we are defining the hyperparameter combinations for Random Forest Tuning
    param_grid = {
        "n_estimators": [50,100,150],
        "max_depth": [4,6,8],
        "min_samples_leaf": [1,5,10]
    }

    # Here, we are evaluating all parameter combinations using 5fold cross-validation
    grid_search = GridSearchCV(
        estimator = RandomForestClassifier(random_state = 42),
        param_grid = param_grid,
        cv=5
    )

    # Here, we fit the search using only the seven selected training features
    grid_search.fit(X_train_7, y_train)

    # Here, we are displaying the following configurations
    print("The Best GridSearchCV parameters: ", grid_search.best_params_)
    print("The Best GridSearchCV accuracy: ", grid_search.best_score_)

    # Here, we are evaluating the best GridSearchCV model on the validation rows
    grid_predictions = grid_search.best_estimator_.predict(X_val_7)
    grid_val_f1 = f1_score(y_val, grid_predictions)
    print("The GridSearchCV Validation F1: ", grid_val_f1)

    # Here, we are evaluating the combination using 5 fold cross validation
    randomized_search = RandomizedSearchCV(
        estimator = RandomForestClassifier(random_state = 42),
        param_distributions = param_grid,
        n_iter = 9,
        cv = 5,
        random_state = 42
    )
    randomized_search.fit(X_train_7, y_train)

    # Here, we are displaying the best parameters and cross-validation accuracy
    print("The Best RandomizedSearchCV parameters: ", randomized_search.best_params_)
    print("The Best RandomizedSearchCV accuracy: ", randomized_search.best_score_)

    # Here, we are evaluating the best RandomizedSearchCV model on validation rows
    randomized_predictions = randomized_search.best_estimator_.predict(X_val_7)
    randomized_val_f1 = f1_score(y_val, randomized_predictions)
    print("The RandomizedSearchCV Validation F1: ", randomized_val_f1)

    # Here, we are training the default Random Forest using the same 7 selected features
    default_forest_same_7_features = RandomForestClassifier(
        n_estimators = 100,
        random_state = 42
    )

    default_forest_same_7_features.fit(X_train_7, y_train)

    # Here, we are evaluating the default forest on validation data
    default_7_predictions = default_forest_same_7_features.predict(X_val_7)
    default_7_val_f1 = f1_score(y_val, default_7_predictions)
    print("The Default Random Forest with 7 feature validation f1 is: ", default_7_val_f1)

    # Storing the results and validation performance:
    tuning_results = [
        [
            "grid_search",
            len(grid_search.cv_results_["params"]),
            5,
            len(grid_search.cv_results_["params"]) * 5,
            str(grid_search.best_params_),
            grid_search.best_score_,
            grid_val_f1
        ],
        [
            "randomized_search",
            len(randomized_search.cv_results_["params"]),
            5,
            len(randomized_search.cv_results_["params"]) * 5,
            str(randomized_search.best_params_),
            randomized_search.best_score_,
            randomized_val_f1
        ],
        [
            "default_forest_same_7_features",
            1,
            0,
            1,
            "defaults",
            None,
            default_7_val_f1
        ]
    ]

    # Here, we are presenting the comparison table with the requierd column names
    tuning_df = pd.DataFrame(tuning_results, columns = ["search","candidates","cv_folds","total_fits","best_params","best_cv_accuracy","val_f1"])

    # Here, we are saving the results
    tuning_df.to_csv("outputs/part_c_tuning.csv", index = False)
    print("Part C Tuning Results: ")
    print(tuning_df.to_string(index = False))

    # Here, we are selecting the configuration with the highest validation F1
    # According to the instruction, if there is a tie, prefer default forest, then grid search, and then randomized search
    tie = {
        "default_forest_same_7_features": 0,
        "grid_search": 1,
        "randomized_search": 2
    }

    tuning_df["tie"] = tuning_df["search"].map(tie)
    chosen_config = tuning_df.sort_values(by = ["val_f1","tie"], ascending = [False, True]).iloc[0]
    chosen = chosen_config["search"]
    print("The chosen configuration is: ", chosen)
    print("The F1 validation is: ", chosen_config["val_f1"])

    # Here, we are building the pipeline using our selected configuration
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k = 7)),
        ("model", RandomForestClassifier(
            n_estimators = 100,
            random_state = 42
        ))
    ])

    # Here, we are fitting both the feature selectors and Random Forest using training rows
    final_pipeline.fit(X_train, y_train)

    final_predictions = final_pipeline.predict(X_val)
    final_val_f1 = f1_score(y_val, final_predictions)
    print("The Final Pipeline Validation F1: ", final_val_f1)

    # Here, we are obtaining the probability that each validation shipment is delayed
    validation_probabilities = final_pipeline.predict_proba(X_val)[:, 1]

    # Here, we are defining the thresholds mentioned in the project document
    threshold = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    print("The Number of validation probabilities: ", len(validation_probabilities))
    print("The first five validation probabilities are: ", validation_probabilities[:5])
    print("Threshold: ", threshold)

    # Here, we are storing the classification metrics for each threshold
    threshold_result = []

    # Here, we are converting the validation probabilities into predicted classes for each threshold
    for t in threshold:
        # Here, we predict delay (i.e. 1) when probability is at least the threshold. Otherwise, we the predict the delivery is on-time(0)
        threshold_predictions = (validation_probabilities >= t).astype(int)

        # Here, we are counting the shipments that are predicted as delayed
        predicted_delays = threshold_predictions.sum()
        print("Threshold: ", t)
        print("The predicted delayed shipments are: ", predicted_delays)
    
        # Here, we are comparing predicted delays with actual validation labels
        cm = confusion_matrix(y_val, threshold_predictions, labels = [0,1])
        tn_t, fp_t = cm[0]
        fn_t, tp_t = cm[1]

        # Here, we are calculating classification metrics for this threshold
        accuracy_t = accuracy_score(y_val, threshold_predictions)
        precision_t = precision_score(y_val, threshold_predictions, zero_division = 0)
        recall_t = recall_score(y_val, threshold_predictions)
        f1_t = f1_score(y_val, threshold_predictions)

        # Here, we are storing the threshold and its evaluation results
        threshold_result.append([t, tn_t, fp_t, fn_t, tp_t, accuracy_t, precision_t, recall_t, f1_t])
        print("The Recall is: ", recall_t)
        print("The F1 is: ", f1_t)
    
    threshold_df = pd.DataFrame(
        threshold_result,
        columns = ["threshold","TN","FP","FN","TP","accuracy","precision","recall","f1"]
    )

    # Here, we are saving the results for all nine thresholds
    threshold_df.to_csv("outputs/part_c_thresholds.csv", index = False)
    print("Part C Threshold Evaluation Results: ")
    print(threshold_df.to_string(index = False))

    # Here, we are filtering thresholds that identify atleast 80% of actual delayed shipments
    eligible_thresholds = threshold_df[threshold_df["recall"] >= 0.80]

    # Here, we are selecting the highest threshold that satisfies the recall requirement
    if not eligible_thresholds.empty:
        chosen_threshold = eligible_thresholds["threshold"].max()
    else:
        chosen_threshold = 0.1
    print("The chosen threshold is: ", chosen_threshold)

    # Here, we calculate the delay probabilities for the unseen test shipments
    test_probabilities = final_pipeline.predict_proba(X_test)[:,1]

    # Here, we are applying the threshold selected using our validation set
    test_predictions = (test_probabilities >= chosen_threshold).astype(int)

    # Here, we are verifying the number of test predictions
    print("The number of test predictions is: ", len(test_predictions))
    print("The predicted delayed test shipments are: ", test_predictions.sum())

    # Here, we compare the test predictions with the actual shipment delays
    test_cm = confusion_matrix(y_test, test_predictions, labels = [0,1])

    # Here, we extract the four confusion matrix values
    tn_test, fp_test = test_cm[0]
    fn_test, tp_test = test_cm[1]

    # Here, we display how many test shipments were correctly or incorrectly classified
    print("The final test confusion matrix: ")
    print("TN: ", tn_test)
    print("FP: ", fp_test)
    print("FN: ", fn_test)
    print("TP: ", tp_test)

    # Here, we are calculating the final classification metrics on the test data
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions, zero_division = 0)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)

    # Here, we are displaying the final test performance
    print("The final test evaluation results are: ")
    print("Accuracy: ", test_accuracy)
    print("Precision: ", test_precision)
    print("Recall: ", test_recall)
    print("F1: ", test_f1)

    # Here, we are storing the final test results in a single row
    final_test_results = [[chosen_threshold,tn_test,fp_test,fn_test,tp_test,test_accuracy,test_precision,test_recall,test_f1]]

    # Here, we are converting the test results into a DataFrame
    final_test_df = pd.DataFrame(
        final_test_results,
        columns = ["threshold","TN","FP","FN","TP","accuracy","precision","recall","f1"]
    )

    # Here, we are saving the final test evaluation
    final_test_df.to_csv("outputs/part_c_final_test.csv", index = False)
    print("Part C Final Test Results: ")
    print(final_test_df.to_string(index = False))

    # Here, we are retrieving the selected models parameters
    final_model = final_pipeline.named_steps["model"]

    # Here, we are storing the final model configurations
    final_config_df = pd.DataFrame([{
        "selector_k": final_pipeline.named_steps["selector"].k,
        "n_estimators": final_model.n_estimators,
        "max_depth": "unlimited" if final_model.max_depth is None else final_model.max_depth,
        "min_samples_leaf": final_model.min_samples_leaf,
        "threshold": chosen_threshold,
        "chosen_from": chosen
    }])

    # Here, we are saving the selected configuration for future reference
    final_config_df.to_csv("outputs/part_c_final_config.csv", index = False)
    print("Part C Final Configurations: ")
    print(final_config_df.to_string(index = False))


if __name__ == "__main__":
    main()

