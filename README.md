# DELHIVERY LAST-MILE DELAY INTELLIGENCE PIPELINE

This is the final capstone project for the Artificial Intelligence and Machine Learning Certification Program.

## Scenario:

You have joined the regional operations analytics team of Delhivery, one of India's largest logistics networks. Every parcel is booked with a promised delivery time (Express 36 hours, Standard 60 hours, Economy 96 hours). The regional operations head has three open problems and no pipeline that answers any of them:

1. How long will a parcel really take? Planners need a delivery-hours estimate for each parcel at dispatch time.
2. Which parcels will miss their promise? Ops can place a proactive customer call or escalate to a hub manager, but only for parcels flagged before they are late. A missed delayed parcel costs more than an unnecessary call.
3. Which hubs need which playbook? Each of the 24 hubs is labelled Tier 1, 2 or 3, but nobody has checked whether the label matches how the hub really behaves.

You will answer all three in one connected repository. The cleaning and feature code you write in Part A feeds the regression in Part B and the classifier in Part C; the classifier from Part C and the hub segments from Part D are combined by a final scoring script that turns a batch of new parcels into an action list. No later Part may use a number it did not compute itself or receive from an earlier Part.

The data is synthetic. Delhivery's real shipment records are never used. Every row in this brief's dataset is produced by the generator script below; the brand is only the setting for the decisions.

## How to Run:

1. CREATE A VIRTUAL ENVIRONMENT:

Commands to use:
python -m venv .venv
source .venv/bin/activate

2. INSTALL THE REQUIRED LIBRARIES:

Command to use: pip install -r requirements.txt

3. GENERATE THE DATASETS

Command to use: python generate_data.py

This command will create the historical shipment data, new shipment data, and hub information.

4. RUN PART A

Commands to use: python part_a_clean_eda.py

5. RUN PART B

Command to use: python part_b_regression.py

This command will train and evaluate the regression models to predict the delivery duration of shipments. It will also generate the model comparison, decision tree depth, learning rate, and cross-validation results.

6. RUN PART C

Command to use: python part_c_classification.py

This command will train and evaluate the classification models to identify delayed shipments. It will also perform feature selection, hyperparameter tuning, threshold selection, and final model evaluation.

7. RUN PART D

Command to use: python part_d_segments.py

This command will generate the hub profiles and evaluate K-Means clustering for K values from 2 to 5. It will also assign the hubs to Standard, Watch, and Escalate segments.

8. PACKAGE THE CLASSIFICATION MODEL

Command to use: python package_model.py

This command will regenerate models/delay_model.pkl using the selected Random Forest model. It will also save the classification threshold, feature order, and historical imputation values required for scoring new shipments.

9. SCORE THE NEW SHIPMENTS

Command to use: python predict_delay.py

This command will load the saved classification model and hub segments, clean the new shipment data using the historical imputation values, and generate the operational action list in outputs/new_shipment_actions.csv

## DATA FINDINGS (PART A):

1. Which service type is delayed most often and at what rate?

Express shipments have the highest delay rate. From 1200 total historical shipments, 278 total shipments were delayed, out of which Express shipments recorded the highest delay rate at 48.97%. The Standard shipment recorded 20.56% delay rate, and Economy shipment recorded 8.57% delay rate.

Out of the 1200 historical shipments, 278 shipments were delayed, resulting in an overall delay rate of 23.17%.

2. How delay rate changes from Tier 1 to Tier 3?

Tier 1 had a delay rate of 4%, Tier 2 had a delay rate of 21.5%, and Tier 3 had a delay rate of 44%. All 3 tiers had 400 shipments each, but Tier 3 recorded 176 delayed shipments compared with 16 delayed shipments for Tier 1.

3. How it changes across distance bands?

In the results, it shows that delay rates increased across the four distance bands. The following is the distribution of distance with its respective delay rate:
Distance (0 - 200 km): 2.19% of delay rate
Distance (201 - 400 km): 13.68% of delay rate
Distance (401 - 600 km): 43.64% of delay rate
Distance (601+ km): 85.23% of delay rate

4. The three columns most strong correlated with delivery_hours - rank the columns of part_a_correlations.csv

The three columns with most strong correlations with delivery_hours:

promised_hours : +0.8372
distance_km : +0.4013
n_stops : +0.3247

5. One sentence on why declared_value_inr was log-transformed (compare its mean and median before and after, using the filled columns):

The mean for declared_value_inr is INR 1533.0542 and the median was INR 1121.5. After applying the log transformation, the mean and median were 7.0277 and 7.0224. The log transformation helped compress large numerical values and help reduce the highly skewed inputs during modelling.

# Model results (regression) - PART B

## BASELINE PERFORMANCE:

My results showed that the mean baseline predicted a constant delivery duration of 57.7493 hours which was calculated from the training dataset. On our validation dataset, the baseline achieved a MAE of 15.7952 hours, RMSE of 18.8552 hours, and R2 of -0.021221. The negative R2 here indicates that the baseline performed slightly worse than predicting the mean delivery duration of the validation set. 

## MODEL COMPARISON:

We compared four regression models that were trained using 840 training shipments and evaluated on 180 validation shipments. Following are the metrics that were recorded for each model:
1. Linear Regression: Validation RMSE of 4.891169 hours; Validation R2 of 0.9313
2. Decision Tree: Validation RMSE of 6.279637 hours; Validation R2 of 0.886728
3. Random Forest: Validation RMSE of 5.509538 hours; Validation R2 of 0.912806
4. Gradient Boosting: Validation RMSE of 4.9165; Validation R2 of 0.9306

In this comparison, we found that Linear Regression achieved the lowest validation RMSE of 4.891169 hours. Therefore, it was selected as the final model.

## HIGH BIAS vs HIGH VARIANCE:

We evaluated the Decision Tree models at depth 1, 2, 3, 4, 6, 8, 12, and unlimited. 

High bias (depth 1): At max_depth 1, the tree produced a training RMSE of 11.633436 hours and validation RMSE of 12.132363 hours. The gap between training RMSE and validation RMSE is 0.498927 hours. Even if the gap between the training RMSE and validation RMSE was small, both errors were high, which indicated that the model was underfitting.

Best performing depth (depth 6): At max_depth 6, the tree achieved its lowest validation RMSE of 6.279637 hours, and a training RMSE of 4.711358 hours. The gap between training RMSE and validation RMSE is 1.568279 hours. This shows that model achieved a better balance between fitting the training data and performing well on the validation data.

High variance (unlimited depth): At max_depth "unlimited", the tree achieved a training RMSE of 0.000000 hours but a validation RMSE of 7.836781 hours. The gap of 7.836781 hours indicates that the model was overfitting, as it performed perfectly on the training data but had higher errors on the validation dataset.

## LEARNING RATE:

The gradient boosting was evaluated with 100 estimators and learning rates of 0.01, 0.1, and 1.0. Following are the metrics that were recorded:
1. Learning Rate (0.01): Validation RMSE of 10.001759 hours; Validation R2 of 0.712652
2. Learning Rate (0.1): Validation RMSE of 4.916525 hours; Validation R2 of 0.930566
3. Learning Rate (1.0): Validation RMSE of 7.370066 hours; Validation R2 of 0.843974

The best validation performance was achieved with a learning rate of 0.1 with an RMSE of 4.916525 hours and an R2 of 0.930566.

At a learning rate of 0.01 (Validation RMSE of 10.001759 hours; Validation R2 of 0.712652), the model learned too slowly because each boosting stage made only a small adjustment. With 100 boosting stages, it could not capture the underlying patterns therefore resulting into underfitting.

At a learning rate of 1.0 (Validation RMSE of 7.370066 hours; Validation R2 of 0.843974), the model made large adjustments while learning, which resulted in missing the best predictions and increased the validation error.

## CROSS-VALIDATION

The two best models with the lowest validation RMSE are: Linear Regression and Gradient Boosting, were evaluated across 5-fold cross-validation on the combined dataset (Training Set and Validation Set) of 1020 shipments. Following are the metrics recorded:

1. Linear Regression: Mean CV R2 of 0.926133; Standard Deviation CV R2 of 0.004754
2. Gradient Boosting: Mean CV R2 of 0.935266; Standard Deviation CV R2 of 0.004477

Both models achieved a mean cross-validation R2 score above 0.90 and low standard deviation, which indicates consistent performance across folds. The cross-validation ranking differs from the validation RMSE ranking. Gradient Boosting achieved a higher mean CV R2, whereas Linear Regression achieved a lower validation RMSE. In this scenario, I selected Linear Regression because it achieved the lowest validation RMSE (4.891169 hours) and offered a simpler model with competitive cross-validation performance. Although Gradient Boosting achieved a slightly higher mean cross-validation R2, Linear Regression was selected based on validation performance, without using the test dataset for model selection.

After selecting Linear Regression based on validation RMSE, the final model was evaluated once on the held-out test dataset on 180 shipments. The following test metrics were recorded:
1. MAE: 3.567078 hours
2. MSE: 20.462444
3. RMSE: 4.523543 hours
4. R2: 0.949250

Therefore, the final model chosen was the Linear Regression model which achieved an R2 of 0.949250 on the test dataset, with an RMSE of 4.523543 hours. Linear Regression provided the lowest validation RMSE among all the four models and performed very well on the test dataset

# Model results (classification) - PART C

## BLIND BASELINE PERFORMANCE:

The blind baseline predicted that every shipment would be delivered on time. On the validation dataset of 180 shipments, it achieved an accuracy of 0.7666666666666667. The precision, recall, and F1 score were all 0 because the model failed to identify any of the 42 delayed shipments. Therefore, even though the accuracy was high, the baseline was not useful for identifying delayed shipment.

## FEATURE SELECTION

The five noise features contributed a combined Random Forest importance of 0.22437846660639194. All five noise features ranked outside the top 7 features based on their ANOVA F-scores. We evaluated SelectKBest with the k values of 5,7,10,12, and 17. The highest validation F1 score of 0.8333333333333334 was achieved with k = 7. Therefore, we selected the top 7 features for further model evaluation.

## HYPERPARAMETER TUNING:

We evaluated GridSearchCV and RandomizedSearchCV using 5-fold cross-validation. GridSearchCV achieved a validation F1 score of 0.775, while RandomizedSearchCV achieved 0.7848101265822784. The default Random Forest using the same 7 features achieved a higher validation F1 score of 0.8333333333333334. Therefore, hyperparameter tuning did not improve the model's validation performance. Since the validation dataset contained only 180 shipments, the results may also be sensitive to small changes in predictions.

## THRESHOLD SELECTION:

We have evaluated threshold from 0.1 to 0.9 and selected the highest threshold that maintained a validation recall of at least 0.80. The selected threshold was 0.5, which achieved a validation precision of 0.795455 and recall of 0.833333. Increasing the threshold to 0.6 improved precision to 0.888889 but reduced recall to 0.761905. Therefore, we selected 0.5 to identify more delayed shipments while maintaining the precision.

## FINAL TEST PERFORMANCE:

The final Random Forest model was evaluated on 180 test shipments using the selected threshold of 0.5. The following are the metrics recorded:
1. Accuracy: 0.8666666666666667
2. Precision: 0.6888888888888889
3. Recall: 0.7560975609756098
4. F1: 0.7209302325581395

The model correctly identified 31 out of 41 delayed shipments. The test recall of 0.7560975609756098 was lower than the validation recall of 0.8333333333333334. Since the test set has contained only 41 delayed shipments, a small change in the number of correctly identified delays can affect the recall significantly.

Therefore, the final model chosen was the Random Forest with 7 selected features and a threshold of 0.5, which achieved a test F1 score of 0.7209302325581395

# Operations recommendation - Part D

## HUB SEGMENTATION:

We evaluated K-Means clustering using the 24 historical hub profiles, with K values of 2,3,4, and 5. The following are the separation ratios and cluster sizes recorded:

1. K = 2: Separation ratio of 2.893964; Cluster sizes of 15 and 9 hubs.
2. K = 3: Separation ratio of 5.392845; Cluster sizes of 9,8, and 7 hubs.
3. K = 4: Separation ratio of 6.304705; Cluster sizes of 8,7,5, and 4 hubs.
4. K = 5: Separation ratio of 6.886628; Cluster sizes of 8,7,4,3, and 2 hubs.

Although the separation ratio increased for K = 4 and  K = 5, I selected K = 3 because the other two solutions created smaller clusters containing fewer than 6 hubs. These smaller clusters would be difficult to manage with separate operational plans. K = 3 provided three meaningful groups with at least 7 hubs in each segment.

## SEGMENT RESULTS:

After selecting K = 3, we assigned the segment names based on their average delay rates. Following are the results:
1. Standard: 7 hubs, with an average delay rate of 2.57% and mean delivery duration of 47.239629 hours.
2. Watch: 8 hubs, with an average delay rate of 15.50% and mean delivery duration of 56.813675 hours.
3. Escalate: 9 hubs, with an average delay rate of 46.00% and mean delivery duration of 64.650444 hours.

The Escalate segment recorded the highest delay rate and delivery duration. Therefore, these hubs should receive the highest operational attention

## HUBS REQUIRING ESCALATION:

The following nine hubs were assigned to the Escalate segment based on their historical performance:
1. HUB13 - Nagpur: 46% delay rate
2. HUB15 - Kochi: 36% delay rate
3. HUB17 - Patna: 46% delay rate
4. HUB18 - Ranchi: 38% delay rate
5. HUB19 - Guwahati: 62% delay rate
6. HUB20 - Raipur: 48% delay rate
7. HUB21 - Bhubaneswar: 40% delay rate
8. HUB23 - Siliguri: 44% delay rate
9. HUB24 - Agartala: 54% delay rate

Among these hubs, Guwahati recorded the highest delay rate at 62%, followed by Agartala at 54% and Raipur at 48%. These hubs should be prioritized for reviewing the operational delays, shipment handling, and hub capacity.

## TIER VS SEGMENT COMPARISON:

We have compared the existing hub tiers with the newly assigned operational segments. Four hubs showed a mismatch between their existing tier and the assigned segment:
1. HUB07 - Kolkata: Tier 1, assigned to Watch
2. HUB13 - Nagpur: Tier 2, assigned to Escalate
3. HUB15 - Kochi: Tier 2, assigned to Escalate
4. HUB22 - Dehradun: Tier 3, assigned to Watch

These results show that the existing hub tier does not always represent the historical delay performance. For Example: Nagpur and Kochi are Tier 2 hubs but were assigned to the Escalate segment. Therefore, the operational teams should consider the actual delay performance along with the existing hub tiers when prioritizing hubs.

## NEW SHIPMENT SCORING AND ACTIONS:

We used the saved Random Forest classification model with a threshold of 0.5 to score 48 new shipments. The following are the results:
1. Total shipments scored: 48
2. Total shipments flagged as delayed: 8
3. Escalate to hub manager: 6 shipments
4. Proactive customer call: 2 shipments
5. No action: 40 shipments

The operational action depends on the predicted delay and the assigned hub segment. If a shipment is flagged and belongs to the Escalate segment, it is assigned to the hub manager. If a shipment is flagged but belongs to another segment, it is assigned for a proactive customer call. Shipments that are not flagged receive no action. The final results are saved in outputs/new_shipment_actions.csv

# PRODUCTION PLAN:

The current pipeline was developed and evaluated using synthetic shipment data. Before using the pipeline in a real logistics environment, the following areas would need to be considered.

## PRODUCTION READINESS:

1. Accuracy: The final regression model achieved a test RMSE of 4.523543 hours, while the classification model achieved a test recall of 0.756098. These results provide a starting point for evaluation, but the models would need to be tested using real shipment data before deployment.
2. Reliability: The scoring pipeline uses the saved model, threshold, feature order, and historical imputation values. The pipeline also checks for missing output values and duplicate shipment IDs to reduce the chances of generating incomplete operational results.
3. Reproducibility: The project uses fixed random states and pinned library versions. The saved model, classification threshold, feature order, and historical imputation values are reused when scoring new shipments.
4. Performance: The pipeline currently processes a batch of 48 new shipments. Before deployment, we would need to measure the processing time for larger shipment batches and evaluate whether the pipeline can handle the expected daily shipment volume.
5. Maintainability: The cleaning and feature engineering functions are maintained in src/shipment_prep.py and reused across the project. This helps avoid implementing different preprocessing logic for historical and new shipments.
6. Consistency: The saved feature order and historical imputation values are used when scoring new shipments. This helps ensure that the new data follows the same preprocessing and feature structure used during model training.
7. Monitorability: The pipeline should record prediction results, flagged shipments, actual delivery outcomes, and operational actions. These records would help monitor model performance and identify changes in shipment patterns.

## MONITORING PLAN:

We would monitor the following metrics after deployment. The alert thresholds below are proposed starting points and would need to be adjusted using actual production data.

1. Weekly delay rate: Monitor the percentage of shipments that were delivered after their promised delivery time. If the weekly delay rate exceeds 30%, an alert should be generated for the operations team.
2. Flagged shipment rate: Monitor the percentage of shipments flagged by the classification model. If more than 25% of shipments are flagged in a week, the operations team should review whether shipment conditions or model predictions have changed.
3. Classification recall: Once actual delivery outcomes are available, compare the predicted delays with the actual delayed shipments. If recall falls below 0.65, the model should be reviewed because it may be missing too many delayed shipments.
4. Missing input values: Monitor the percentage of shipments requiring missing-value imputation. If more than 10% of shipments have a missing value in any required numeric input column, the data source should be reviewed.

## RETRAINING PLAN:

I would recommend evaluating and retraining the model every month using newly collected shipment data. The model should also be reviewed earlier  if there is a consistent decline in prediction performance or a significant change in shipment patterns.

During retraining, the training data should include recent shipment outcomes, and the updated model should be evaluated on a separate validation and test dataset before replacing the existing model. 

Special attention should also be given to sale events because shipment volumes and delays may behave differently during these periods. For example, the dataset includes a sale-event period from Sept 15 to Sept 28. Performance during these periods should be compared with similar sale-event periods rather than only with normal operating weeks.

The updated model should be packaged again using package_model.py, with the feature order, threshold, and imputation values saved together with the model.

# LIMITATIONS:

1. Synthetic dataset: The entire project uses synthetic shipment data generated through generate_data.py. Although the results are useful for understanding the machine learning pipeline, they do not represent actual Delhivery operations. The model would need to be evaluated on real shipment data before being used for operational decisions.
2. Limited dataset size: The historical dataset contains 1200 shipments across 24 hubs. The classification test dataset contained only 41 delayed shipments. Therefore, a small change in the number of correctly identified delays can significantly affect the recall and other classification metrics.
3. Threshold selection: We evaluated classification thresholds from 0.1 to 0.9 and selected 0.5 because it was the highest threshold that maintained a validation recall of at least 0.80. At this threshold, the model achieved a validation precision of 0.795455 and recall of 0.833333. However, on the final test dataset, recall decreased to 0.756098, with 31 out of 41 delayed shipments correctly identified. This shows that the selected threshold may not maintain the same recall across different datasets and would need to be monitored and evaluated using new shipment outcomes before production deployment.
4. Limited evaluation period: The model was evaluated using the generated historical shipment dataset and a single held-out test split. The current results do not establish how the model would perform over longer periods or during changing operational conditions.
5. Production deployment: The current pipeline generates predictions and operational actions as CSV files. It has not been integrated  with a real logistics system, customer communication platform or hub management workflow. Additional testing, monitoring, and system integration would be required before production deployment.