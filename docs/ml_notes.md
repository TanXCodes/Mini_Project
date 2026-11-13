\# ML Notes — AI-Based Emergency Corridor System



\## 1. ML Objective



The ML module predicts road-segment traffic delay in minutes.



The predicted delay is used by the routing module to calculate the additional time required on road segments. The routing system can then use these predicted delays when selecting an emergency corridor for an ambulance.



The ML target is:



\- \*\*Target:\*\* `delay\_min`



The prediction is used as a recommendation for routing and is not a permanent label.



\---



\## 2. Data Used



The dataset is based on synthetic traffic and road-network data.



The ML data preparation combines:



\- `traffic\_raw.csv`

\- `edges.csv`

\- `nodes.csv`



The traffic data is merged with road information using:



```text

segment\_id

Node information is then added for both the starting and ending junctions.



The final dataset contains traffic, road, node, signal, and engineered feature information.



Missing values and duplicate records were checked after dataset preparation.



The resulting train, validation, and test datasets contain no missing values in the checked features and no duplicate rows.

3\. Features

The ML model uses traffic, road, node, signal, and engineered features.



Important features include:

\- day

\- time\_slot

\- free\_flow\_speed\_kmph

\- volume

\- capacity

\- incident\_flag

\- from\_node

\- to\_node

\- road\_class

\- length\_km

\- maxspeed

\- oneway

\- from\_lat

\- from\_lon

\- from\_has\_signal

\- to\_lat

\- to\_lon

\- to\_has\_signal

\- signal\_near\_segment

\- vc\_ratio

Engineered feature

The volume-to-capacity ratio was created as:

vc\_ratio = volume / capacity



It represents the traffic volume relative to the road capacity.

4\. Data Leakage Prevention

The target variable is:

delay\_min



The following columns were deliberately excluded from the ML features:

speed\_kmph

travel\_time\_min



These variables were not used because they are directly related to the observed travel conditions and could provide information that would not be appropriate for predicting delay in the intended setup.



Using them could cause data leakage and make the model performance look better than it would be in a real prediction scenario.



A leakage check was performed during model training:

speed\_kmph: False

travel\_time\_min: False



Therefore, neither leakage feature was included in the training feature set.

5\. Day-Wise Train/Test Split

The data was split by day instead of randomly splitting individual rows.



Training data:

Days 1, 2, 3, 4



Test data:

Day 5



The validation data was kept separate during dataset preparation.



The day-wise split is important because traffic records from the same day can be related. Testing on a separate day provides a better indication of how the model performs on unseen daily traffic conditions.

6\. Models

Two models were trained and compared.

Linear Regression

Linear Regression was used as the baseline model.



Results:

Metric	Linear Regression

MAE	0.009169

RMSE	0.026290

R²	0.001337





Random Forest

A Random Forest model was trained with the configured tree-based approach.



Results:

Metric	Random Forest

MAE	0.000850

RMSE	0.007896

R²	0.909910





Random Forest performed substantially better than the Linear Regression baseline.

7\. Model Selection

The Random Forest model was selected as the best model based on the evaluation results.



It achieved:

MAE = 0.000850

RMSE = 0.007896

R² = 0.909910



The trained best model is saved as:

models/delay\_model.joblib



8\. Evaluation Outputs

The following outputs were generated:

outputs/model\_metrics.csv

outputs/actual\_vs\_predicted.png

outputs/feature\_importance.csv

outputs/feature\_importance.png



These outputs contain the model comparison metrics, actual-vs-predicted visualization, and feature importance information.

9\. Delay Prediction

The ML prediction module provides:

predict\_delay(features)



This returns predicted delay in:

minutes



The routing module uses:

delay\_fn(segment\_id, scenario)



which returns predicted delay in:

seconds



This conversion allows the routing system to combine free-flow travel time with the ML-predicted additional delay.

10\. Accident Scenario

The ML routing integration supports an accident scenario using:

affected\_segments



as a list of affected segment IDs.



For an accident scenario, affected segments receive an additional higher predicted delay while unaffected segments retain their normal predicted delay.



Example validation:

Affected segment delay: 17.1206 seconds

Unaffected segment delay: 0.5964 seconds



This confirms that the accident scenario increases delay on the affected segment.

11\. Scenario Validation

Four scenarios were tested using delay\_fn().

Scenario	Average Delay

Normal / midday	0.0303 sec

Morning peak	0.3658 sec

Evening peak	0.7683 sec

Accident / evening peak	8.8585 sec





The results show increasing predicted delay under heavier traffic conditions and a substantially higher delay for the accident scenario.

12\. Testing

The project tests were executed using:

python -m pytest -q tests



Result:

8 passed



The ML prediction and routing integration tests passed successfully.

13\. Summary

The ML module prepares the traffic dataset, creates useful traffic features such as vc\_ratio, prevents leakage from speed\_kmph and travel\_time\_min, trains and compares Linear Regression and Random Forest models, and provides predicted segment delay for the routing system.



Random Forest was selected as the best-performing model with an R² of approximately 0.91 on the day-wise test set.



The delay\_fn() interface provides the routing module with predicted delay in seconds and supports accident scenarios for affected road segments.

