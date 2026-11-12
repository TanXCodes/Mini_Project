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


10. Controlled Accident Scenario

The routing integration supports localized accident simulation using:

affected_segments

This is a list of selected road segment IDs affected by an accident.

For the controlled accident experiment, the existing evening-peak ML prediction is used as the reference.

Accident handling:

- The model's incident_flag feature is kept False for all segments when selected affected_segments are supplied.
- Only selected affected segments receive a 2x multiplier.
- Unaffected segments retain exactly their normal evening-peak predictions.
- The normal evening-peak prediction behaviour remains unchanged.
- The existing backward-compatible network-wide accident behaviour is retained when accident=True and affected_segments is empty.

Formula:

Affected segment:
Accident delay = 2 x Normal evening-peak delay

Unaffected segment:
Accident delay = Normal evening-peak delay

The 2x multiplier is a controlled simulation assumption and does not represent a separately calibrated real-world accident model.

Validation requirements:

1. Unaffected segment predictions must remain identical.
2. Affected segment predictions must be exactly twice their normal predictions.
3. Route-level travel times must reflect the localized accident delays.

## 11. Controlled Accident Scenario Validation

The localized accident scenario was tested using three demo trips and three selected affected road segments.

The experiment compared normal evening-peak predictions with localized accident predictions.

### Affected segments

| Trip | Affected segment ID |
|---|---|
| Trip 1 | `12451306645_3399067454_0` |
| Trip 2 | `4761188837_11964724285_0` |
| Trip 3 | `1620344798_254920166_0` |

### Route comparison results

| Trip | Normal evening (s) | Localized accident (s) | Additional delay (s) |
|---|---:|---:|---:|
| Trip 1 | 275.499 | 279.236 | 3.737 |
| Trip 2 | 214.821 | 218.713 | 3.892 |
| Trip 3 | 272.675 | 274.920 | 2.245 |

The route remained unchanged for all three trips because the alternative routes were not faster than the original routes after applying the localized accident delays.

### Segment-level validation

All 3,151 road segments were checked.

- Affected segments received exactly twice their normal evening-peak predictions.
- Unaffected segments retained exactly the same predictions.
- Both validation checks passed successfully.

The route comparison results are saved in:

`outputs/localized_accident_comparison.csv`

## 12. Testing

The complete project test suite was executed using:

```bash
python -m pytest -v
```

**Result: 23 passed in 2.50 seconds.**

The tests covered:

- Dijkstra shortest-path selection.
- ML delay prediction.
- Affected-segment 2x accident delay.
- Unaffected-segment prediction consistency.
- Three real demo routes.
- Invalid ML delay handling.
- Empty graph handling.
- Signal simulation and emergency priority.

All 23 tests passed successfully.

## 13. Summary

The ML module prepares traffic data, creates engineered features such as `vc_ratio`, prevents data leakage, trains and evaluates Linear Regression and Random Forest models, and provides predicted road-segment delays to the routing module.

Random Forest was selected as the best-performing model, achieving an R² of approximately 0.91 on the day-wise test set.

The `delay_fn()` interface supplies predicted delay in seconds for route calculation. The localized accident experiment applies a controlled 2x multiplier only to selected affected segments while keeping unaffected predictions unchanged.

The final implementation was validated across 3,151 road segments and three demo trips. The complete project test suite passed with 23 tests.