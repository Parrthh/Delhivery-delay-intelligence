# DELHIVERY LAST-MILE DELAY INTELLIGENCE PIPELINE

This is the final capstone project for the Artificial Intelligence and Machine Learning Certification Program.

## Scenario:

You have joined the regional operations analytics team of Delhivery, one of India's largest logistics networks. Every parcel is booked with a promised delivery time (Express 36 hours, Standard 60 hours Economy 96 hours). The regional operations head has three open problems and no pipeline that answers any of them:

1. How long will a parcel really take? Planners need a delivery-hours estimate for each parcel at dispatch time.
2. Which parcels will miss their promise? Ops can place a proactive customer call or escalate to a hub manager, but only for parcels flagged before they are late. A missed delayed parcel costs more than an unnecessary call.
3. Which hubs need which playbook? Each of the 24 hubs is labelled Tier 1, 2 or 3, but nobody has checked whether the label matches how the hub really behaves.

You will answer all three in one connected repository. The cleaning and feature code you write in Part A feeds the regression in Part B and the classifier in Part C; the classifier from Part C and the hub segments from Part D are combined by a final scoring script that turns a batch of new parcels into an action list. No later Part may use a number it did not compute itself or receive from an earlier Part.

The data is synthetic. Delhivery's real shipment records are never used. Every row in this brief's dataset is produced by the generator script below; the brand is only the setting for the decisions.

### How to Run:

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

### DATA FINDINGS (PART A):

1. Which service type is delayed most often and at what rate?

Express shipments have the highest delay rate. From 1200 total historical shipments, 278 total shipments were delayed, out of which Express shipments recorded the highest delay rate at 48.97%. The Standard shipment recorded 20.56% delay rate, and Economy shipment recorded 8.57% delay rate.

Out of the 1200 historical shipments, 278 shipments were delayed, resulting to overall delay rate of 23.17%.

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

The mean for declared_value_inr is INR 1533.0542 and the median was INR 1121.5. After applying the log transformation, the mean and median were 7.0277 and 7.0224. The log transformation helped compressed large numerical values and help reduce the highly skewed inputs during modelling.
