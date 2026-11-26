# Centrifugal Pump Predictive Maintenance — Project Report

## Abstract
Unplanned failure of rotating equipment such as centrifugal pumps causes
costly downtime in process industries. This project builds a machine
learning pipeline that predicts near-term pump failure from sensor and
operating-condition readings — vibration, bearing/motor temperature,
suction/discharge pressure, flow, seal leak rate, duty cycle and age — and
surfaces a maintenance recommendation through a Streamlit web application.
Because public rotating-equipment datasets (AI4I-2020, NASA C-MAPSS) are
already heavily reused, a 12,000-row synthetic dataset was generated from
pump-affinity-law physics, with realistic noise, missing values and sensor
glitches added afterwards. Three candidate models were compared by 5-fold
cross-validation; a Random Forest classifier was selected and achieves
**ROC-AUC 0.907** and **F1 0.698** on a held-out test set at a
validation-tuned decision threshold, with model-agnostic permutation
importance used for explainability.

## Table of Contents
1. Introduction
2. Problem Statement and Objectives
3. Proposed Solution and Architecture
4. Dataset Description
5. Data Preparation and Exploratory Analysis
6. Feature Engineering
7. Modelling Methodology
8. Model Evaluation and Results
9. Risk Categories, Recommendations and Potential Benefits
10. Model Explainability
11. Web Application
12. Technology Stack
13. Testing and Validation
14. Limitations and Future Scope
15. Conclusion
Appendix A. Model Configuration
Appendix B. Review Demonstration Checklist

## 1. Introduction
Centrifugal pumps are among the most common rotating machines in process
plants, water treatment, oil & gas and HVAC systems. Their failure modes —
bearing wear, seal leakage, impeller wear and cavitation — develop
gradually and leave measurable signatures in vibration and temperature
before an unplanned stoppage occurs. This project frames pump health
monitoring as a binary classification problem: given a snapshot of sensor
and operating-condition readings, predict whether the pump is at risk of
failure, and translate that prediction into a concrete maintenance action.

## 2. Problem Statement and Objectives

### 2.1 Problem statement
Maintenance teams typically operate on fixed time-based schedules
(calendar- or runtime-based servicing) rather than the pump's actual
condition, which leads to either costly unplanned failures or unnecessary
preventive maintenance on healthy equipment. A condition-based model that
flags at-risk pumps from live sensor data can shift maintenance from
reactive/time-based to predictive.

### 2.2 Objectives
- Build a representative, reproducible dataset of pump sensor readings and
  failure outcomes.
- Clean and explore the data, understanding which signals separate healthy
  from failing pumps.
- Engineer features that do not leak failure-mode information recorded
  after the fact.
- Train and fairly compare multiple classification models.
- Select and tune a decision threshold appropriate for the class imbalance.
- Explain the model's predictions in a way a maintenance engineer, not
  just a data scientist, can act on.
- Package the model behind an interactive web application.

### 2.3 Proposed solution
A supervised classification pipeline: data validation → exploratory
analysis → leakage-safe feature engineering → model comparison → threshold
tuning → persisted scikit-learn pipeline → Streamlit front end that
returns a failure probability, a risk category (Low/Moderate/High/
Critical) and a maintenance recommendation.

## 3. Proposed Solution and Architecture
```
 data/pump_sensor_data.csv
          │
          ▼
 src/data_validation.py  (schema, missingness, duplicates, range checks)
          │
          ▼
 src/eda.py              (class balance, correlations, distributions)
          │
          ▼
 src/features.py          (engineered ratios, leakage-column exclusion)
          │
          ▼
 src/train.py   ── 5-fold CV across 3 models ── threshold tuning on
          │          validation split ── final fit on train+val
          ▼
 models/pump_failure_pipeline.joblib  +  model_metadata.json
          │
          ├── src/explain_importance.py  (permutation importance)
          ├── tests/test_pipeline.py     (11 unit tests)
          └── app.py                     (Streamlit UI) ──► maintenance
                                                              recommendation
```

## 4. Dataset Description

### 4.1 Source and size
The dataset (`data/pump_sensor_data.csv`, 12,000 rows, 22 columns) is
**synthetically generated** by `src/generate_dataset.py` rather than taken
from an existing public source. Real plant SCADA/historian exports for
pumps are proprietary and rarely released, and the two public datasets
that are commonly reused for this kind of project — AI4I-2020 ("machine
failure.csv") and the NASA C-MAPSS turbofan degradation set — were
deliberately avoided so that the whole data pipeline, not just the model,
is original to this project.

The generator is physics-informed rather than purely random: pump
affinity laws relate rotational speed to flow and head; a latent "wear"
clock (driven by age, duty cycle and a random maintenance-quality factor)
degrades achievable flow/head and drives vibration and heat upward
non-linearly; an NPSH-margin calculation flags cavitation risk when
suction conditions are poor. The script is seeded (`numpy` default_rng
seed 42) so the dataset is exactly reproducible.

### 4.2 Columns and their roles

| Column | Role | Description |
|---|---|---|
| `Pump_ID` | identifier | not used as a feature |
| `Pump_Type` | categorical feature | pump class A / B / C (different design duty points) |
| `Age_Years`, `Operating_Hours`, `Duty_Cycle_Pct` | feature | equipment usage history |
| `Rotational_Speed_RPM`, `Flow_Rate_m3h` | feature | operating point |
| `Suction_Pressure_bar`, `Discharge_Pressure_bar` | feature | hydraulic condition |
| `Vibration_X_mm_s`, `Vibration_Y_mm_s` | feature | radial vibration, two axes |
| `Motor_Current_A`, `Motor_Temperature_C`, `Bearing_Temperature_C` | feature | electrical / thermal condition |
| `Seal_Leak_Rate_ml_min` | feature | mechanical seal condition |
| `Ambient_Temperature_C` | feature | environment |
| `Maintenance_Quality_Score` | feature | proxy for upkeep history (0–1) |
| `Cavitation_Event`, `Bearing_Failure`, `Seal_Leakage`, `Impeller_Wear` | **excluded from model** | failure-mode outcome flags, logged after inspection |
| `Pump_Failure` | **target** | binary — did the pump fail at/near this reading |

### 4.3 Class balance and pump types
669 of 12,000 readings (5.6%) are labelled `Pump_Failure = 1`. Failure rate
is similar across the three pump classes (A 5.51%, B 5.87%, C 5.29% —
`reports/tables/failure_rate_by_pump_type.csv`), so `Pump_Type` is kept as
a feature for the small residual signal rather than a dominant one.

### 4.4 Why the failure-mode flags are excluded
`Cavitation_Event`, `Bearing_Failure`, `Seal_Leakage` and `Impeller_Wear`
are recorded by a technician *after* investigating a failure — they are
outcomes, not predictors available at inference time. This mirrors
AI4I-2020's own modelling guidance, where `TWF`/`HDF`/`PWF`/`OSF`/`RNF` are
excluded for the same reason. `src/features.py` hard-codes them in
`LEAKAGE_COLUMNS` and `tests/test_pipeline.py::test_split_X_y_excludes_leakage_columns`
checks this holds.

## 5. Data Preparation and Exploratory Analysis

### 5.1 Data quality checks
`src/data_validation.py` runs before any modelling:
- **Schema check** — all 18 required columns present, else the pipeline
  raises immediately.
- **Duplicates** — 15 exact duplicate rows found (simulating a
  double-logged reading) and dropped.
- **Missingness** — `Suction_Pressure_bar` (144 rows, 1.2%),
  `Seal_Leak_Rate_ml_min` (240 rows, 2.0%) and `Ambient_Temperature_C` (96
  rows, 0.8%) contain missing values, handled downstream by median
  imputation inside the model pipeline (not by dropping rows).

### 5.2 Outlier policy
13 rows contain an implausible `Vibration_X_mm_s` spike (18–25 mm/s,
consistent with a stuck sensor rather than a real mechanical event) and 12
rows contain an implausible `Motor_Temperature_C` reading (140–160 °C).
These are **flagged** by `data_validation.py`'s range checks but not
silently dropped — a tree-based model is robust to a small number of such
outliers, and removing them would also remove any genuinely extreme (but
real) failing-pump readings.

### 5.3 Failure patterns
- **Class balance** (`reports/figures/01_class_balance.png`): 94.4% normal
  vs 5.6% failure — a realistic rate for rotating equipment, comparable to
  AI4I-2020's ~3.4%.
- **Failure rate by pump type** (`02_failure_rate_by_type.png`): broadly
  flat across A/B/C, confirming failure is driven by condition, not pump
  class alone.
- **Correlation with target** (`03_correlation_heatmap.png`,
  `reports/tables/correlation_with_target.csv`): the strongest linear
  correlations with `Pump_Failure` are `Temp_Differential_C` (0.367),
  `Vibration_Y_mm_s` (0.361), `Seal_Leak_Rate_ml_min` (0.356),
  `Bearing_Temperature_C` (0.350) and `Vibration_Resultant_mm_s` (0.326).
  `Suction_Pressure_bar` is negatively correlated (-0.177) — lower suction
  pressure means less NPSH margin and more cavitation risk.
- **Sensor distributions split by outcome** (`04_sensor_distributions.png`):
  visibly separated distributions for vibration, bearing temperature,
  seal-leak rate and temperature differential between normal and failing
  pumps — a strong early signal that these will be useful features.
- **Age effect** (`05_failure_rate_by_age.png`): failure rate rises with
  pump age bracket, consistent with the wear-clock model used to generate
  the data.

## 6. Feature Engineering
`src/features.py` adds four engineered ratios on top of the 15 raw numeric
sensors, designed to capture physically meaningful combinations rather
than relying on the model to discover them from raw values alone:

| Engineered feature | Formula | Rationale |
|---|---|---|
| `Vibration_Resultant_mm_s` | √(VibX² + VibY²) | overall radial vibration magnitude, independent of sensor orientation |
| `Temp_Differential_C` | Bearing temp − Ambient temp | normalises bearing heating for ambient conditions |
| `Pressure_Ratio` | Discharge pressure / Suction pressure | a single number capturing head development relative to design |
| `Specific_Energy_Index` | (Motor current × Discharge pressure) / Flow rate | a rough energy-per-unit-output proxy; rises when a pump works harder for less output |

All 15 raw numeric sensors plus these 4 engineered features (19 numeric
features total) and the one-hot encoded `Pump_Type` (3 categories) form
the final feature set. Missing numeric values are median-imputed and
scaled (`StandardScaler`) inside the pipeline; the categorical feature is
most-frequent-imputed and one-hot encoded — all inside a single
`ColumnTransformer` so the same transformation is guaranteed at both
training and inference time.

## 7. Modelling Methodology

### 7.1 Train/test split
The 12,000-row cleaned dataset is split 80/20 (stratified on
`Pump_Failure`) into a train+validation pool (9,600 rows) and a held-out
test set (2,400 rows), untouched until final evaluation. The pool is
further split 80/20 (stratified) into training (7,680 rows) and validation
(1,920 rows) for threshold tuning.

### 7.2 Models compared
| Model | Key hyperparameters |
|---|---|
| Logistic Regression | `class_weight="balanced"`, `max_iter=2000` |
| Random Forest | 400 trees, `max_depth=10`, `min_samples_leaf=3`, `class_weight="balanced"` |
| Gradient Boosting | 300 estimators, `max_depth=3`, `learning_rate=0.05` |

### 7.3 Model selection
All three were scored with identical 5-fold stratified cross-validation on
the train+validation pool (`reports/tables/model_comparison_cv.csv`):

| Model | CV F1 (mean ± std) | CV ROC-AUC | CV Avg. Precision | Fit time (s) |
|---|---|---|---|---|
| **Random Forest** | **0.722 ± 0.020** | **0.926 ± 0.016** | **0.702** | 25.8 |
| Gradient Boosting | 0.674 ± 0.018 | 0.923 ± 0.018 | 0.685 | 56.0 |
| Logistic Regression | 0.392 ± 0.017 | 0.898 ± 0.020 | 0.529 | 0.7 |

Random Forest was selected: highest F1 and ROC-AUC, tight cross-fold
variance, and roughly half the fit time of Gradient Boosting. Logistic
Regression's linear decision boundary clearly underfits the non-linear
wear/vibration/temperature interactions that drive failure.

### 7.4 Cross-validation
5-fold `StratifiedKFold` (shuffled, seed 42) preserves the ~5.6% failure
rate in every fold, which matters at this level of class imbalance — an
unstratified split could leave a fold with very few (or zero) failure
examples.

### 7.5 Threshold selection
The default 0.5 probability threshold is not necessarily optimal under
class imbalance. The precision-recall curve on the **validation split
only** was scanned to find the threshold maximising F1 on the failure
class: **0.611**. This threshold is frozen before the model ever sees the
test set, avoiding any test-set leakage into the decision rule itself.

## 8. Model Evaluation and Results

### 8.1 Model comparison on the test set
The selected Random Forest, refit on the full train+validation pool, was
evaluated once on the held-out test set:

| Setting | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Default | 0.500 | 0.962 | 0.617 | 0.828 | 0.707 | 0.907 |
| **Tuned (selected)** | **0.611** | **0.963** | **0.640** | **0.769** | **0.698** | **0.907** |

The tuned threshold trades a little recall for better precision (fewer
false alarms) — a reasonable choice when every "high risk" flag triggers a
real inspection, since 0.611 raises precision from 61.7% to 64.0% while
recall only drops from 82.8% to 76.9%. A maintenance team that prefers to
catch more failures at the cost of more false positives can simply use the
default-threshold row instead; both are reported in
`reports/tables/test_set_results.csv` and the choice is a single
configuration constant (`metadata["selected_threshold"]`), not a retrain.

### 8.2 Cross-validation stability
Random Forest's 5-fold CV F1 standard deviation was 0.020 (on a mean of
0.722) and ROC-AUC standard deviation was 0.016 (on a mean of 0.926) —
tight enough to trust the single train/val/test split's results as
representative rather than a lucky draw.

### 8.3 Overfitting check
`reports/tables/overfitting_check.csv` compares in-sample (train+val) vs.
held-out test performance at the same tuned threshold:

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Train+val (in-sample) | 0.979 | 0.771 | 0.894 | 0.828 | 0.996 |
| Held-out test | 0.963 | 0.640 | 0.769 | 0.698 | 0.907 |

There is a gap (expected for a 400-tree Random Forest that has partially
memorised its training folds), but the test-set ROC-AUC of 0.907 and F1 of
0.698 are still strong in absolute terms, and the gap is consistent with
normal Random Forest variance rather than a sign the model has learned
something spurious — the top features (§10) align with physically
sensible fault signatures, not an accidental artefact.

### 8.4 Threshold analysis
`reports/figures/09_threshold_sweep.png` plots precision, recall and F1
against threshold on the test set: F1 peaks near the selected 0.611,
precision rises steadily as the threshold increases, and recall falls off
— the classic trade-off curve, confirming 0.611 sits at a sensible point
rather than an edge case of the validation split.

### 8.5 Held-out test results at the selected threshold
At threshold 0.611, on 2,400 test readings (134 true failures): 103 true
positives, 31 false negatives, 58 false positives, 2,208 true negatives
(`reports/figures/08_confusion_matrix.png`). In practical terms: roughly
3 in 4 real failures are caught, and when the model raises a flag it is
correct about 2 times in 3.

## 9. Risk Categories, Recommendations and Potential Benefits

### 9.1 Risk categories
The raw probability is bucketed into four maintenance-facing categories
(`src/predict.py::risk_category`):

| Probability | Category | Recommendation |
|---|---|---|
| < 0.20 | Low | No action; continue routine monitoring |
| 0.20 – 0.50 | Moderate | Condition check at next planned maintenance window |
| 0.50 – 0.75 | High | Plan an intervention within days; inspect seals/bearings/suction |
| ≥ 0.75 | Critical | Recommend immediate inspection / controlled shutdown |

### 9.2 Potential benefits and how they could be measured
- **Reduced unplanned downtime**: if a High/Critical flag reliably
  precedes failure by enough lead time to schedule an intervention,
  downtime hours per failure event could be tracked before/after
  deployment.
- **Reduced unnecessary preventive maintenance**: pumps consistently
  scored Low/Moderate could have time-based service intervals safely
  extended; the saving is measurable as technician-hours per pump-year.
- **Prioritised maintenance backlog**: with many pumps monitored
  simultaneously, the probability score gives a natural ranking for a
  maintenance queue with limited crew capacity.
No real cost data exists for this synthetic dataset, so these are
presented as a measurement framework for a real deployment rather than a
dollar estimate.

## 10. Model Explainability

### 10.1 Global importance
Two independent methods agree on the dominant signals:
- **Random Forest native importances**
  (`reports/figures/10_feature_importance.png`): `Vibration_X_mm_s`
  (0.175), `Suction_Pressure_bar` (0.158), `Vibration_Resultant_mm_s`
  (0.139), `Seal_Leak_Rate_ml_min` (0.106), `Temp_Differential_C` (0.081),
  `Bearing_Temperature_C` (0.081).
- **Permutation importance** (`reports/figures/11_permutation_importance.png`,
  `reports/tables/permutation_importance.csv`, F1-drop when a feature is
  shuffled): `Suction_Pressure_bar` (0.171), `Vibration_X_mm_s` (0.140),
  `Seal_Leak_Rate_ml_min` (0.086), `Temp_Differential_C` (0.012).

Both methods point to vibration, suction pressure (cavitation margin) and
seal-leak rate as the primary drivers — consistent with the physical fault
model used to generate the data, which is a useful sanity check that the
model has learned genuine signal rather than an artefact of the simulation.

SHAP was intentionally not made a hard dependency (it is listed as
optional in `requirements.txt`); permutation importance needs no extra
packages and is model-agnostic, so the same explainability code would work
unchanged if the underlying model were swapped later.

### 10.2 Local explanations
For a single prediction, `src/predict.py` and the Streamlit app report how
many standard deviations each sensor reading sits from the healthy-fleet
average (`reports/tables/background_feature_stats.csv`) — a lightweight,
dependency-free stand-in for a per-prediction SHAP waterfall, showing at a
glance *which* readings are driving an individual pump's score up or down.

## 11. Web Application

### 11.1 Equipment parameter input
`app.py` (Streamlit) presents a form with all 15 raw sensor/operating
inputs grouped into three columns (usage history, operating point /
pressures, vibration / electrical / thermal), plus pump type and
maintenance quality.

### 11.2 Prediction result and maintenance recommendation
On submission the app shows the failure probability, the binary
prediction at the tuned threshold, a colour-coded risk category badge, and
the matching maintenance recommendation text from §9.1.

### 11.3 Explanation of the individual prediction
A bar chart of the top 6 sensor readings by standard-deviation distance
from the healthy-fleet mean is shown alongside the prediction, giving an
immediate, model-agnostic explanation of which readings pushed the score.

### 11.4 Model, dataset and limitations panel
A second tab exposes the model name, selected threshold and test metrics
as raw JSON, a plain-language description of the dataset and its
provenance, and the limitations listed in §14.1 — so a reviewer does not
have to read the source code to understand what they are looking at.

## 12. Technology Stack
| Layer | Choice |
|---|---|
| Data generation | Python, NumPy (physics-informed simulation) |
| Data handling | pandas |
| Modelling | scikit-learn 1.8.0 (Pipeline, ColumnTransformer, RandomForest/GradientBoosting/LogisticRegression) |
| Explainability | scikit-learn `permutation_importance` (SHAP optional) |
| Visualisation | matplotlib, seaborn |
| Persistence | joblib (pipeline), JSON (metadata) |
| Testing | pytest |
| Web application | Streamlit |
| Version control | Git (to be initialised by the user on their own machine) |

## 13. Testing and Validation
`tests/test_pipeline.py` contains 11 unit tests, all passing:
1. Required columns are present after validation.
2. No exact duplicates remain after validation.
3. Feature engineering adds the four expected engineered columns.
4. `split_X_y` excludes all four leakage columns and the target.
5. The persisted model and metadata load successfully.
6. `predict_one`'s output has the expected schema and valid probability range.
7. A hand-crafted healthy pump scores a lower failure probability than a
   hand-crafted degraded pump, and lands in a lower risk category.
8. A borderline case runs without error and returns a valid category.
9. `risk_category` boundaries map correctly to Low/Moderate/High/Critical.
10. `predict_batch` output matches row-by-row `predict_one` calls exactly.
11. A missing (`NaN`) sensor value is handled gracefully by the pipeline's
    median imputer rather than raising an error.

`tests/sample_cases.py` defines the three hand-crafted pumps (healthy,
degraded, borderline) reused across the test suite and available for
manual demonstration in the Streamlit app.

## 14. Limitations and Future Scope

### 14.1 Limitations
- The dataset is simulated; fault signatures are physics-informed but not
  calibrated against a real plant's sensors.
- The minority (failure) class is only 5.6% of rows; the tuned threshold
  and reported precision/recall would need re-tuning if deployed against a
  different real-world failure base rate.
- Each reading is scored independently — there is no time-series/sequence
  modelling, so the model cannot see a developing trend across readings
  from the same pump.
- The four excluded failure-mode flags, while correctly excluded from the
  model, mean the model predicts *that* a pump will fail, not *which*
  specific failure mode is most likely — a useful follow-on classification
  task, not currently in scope.

### 14.2 Future scope
- Replace single-reading scoring with a rolling-window/time-series model
  (e.g. gradient boosting on lagged/rate-of-change features, or an
  LSTM/temporal model) to capture degradation trends.
- Retrain and recalibrate on real sensor data once available.
- Extend to a multi-class failure-mode prediction (cavitation vs bearing
  vs seal vs impeller) to make the maintenance recommendation more
  specific.
- Add a cost-sensitive threshold (using real repair/downtime cost data) in
  place of the F1-maximising threshold used here.

## 15. Conclusion
This project demonstrates an end-to-end predictive-maintenance pipeline —
from a self-generated, leakage-aware dataset through model comparison,
threshold tuning, explainability and a usable web front end — achieving a
held-out ROC-AUC of 0.907 and F1 of 0.698 on a realistically imbalanced
failure-prediction task. The two independent explainability methods
agreeing with the physical fault model used to generate the data is strong
evidence the Random Forest has learned genuine condition-monitoring
signal rather than a dataset artefact, and the modular `src/` structure
(validation → features → EDA → train → explain → predict) makes it
straightforward to swap in real sensor data later without changing the
rest of the pipeline.

---

## Appendix A. Model Configuration
```json
{
  "model_name": "random_forest",
  "hyperparameters": {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": 42
  },
  "selected_threshold": 0.6109,
  "sklearn_version": "1.8.0",
  "trained_on_rows": 9600,
  "test_rows": 2400
}
```
Full metadata, including the complete cross-validation comparison table
and both threshold settings' test metrics, is persisted at
`models/model_metadata.json`.

## Appendix B. Review Demonstration Checklist
- [ ] Run `python src/generate_dataset.py` and show the resulting row
      count / failure rate.
- [ ] Run `python src/eda.py` and walk through 2–3 figures in
      `reports/figures/`.
- [ ] Run `python src/train.py` and show the model-comparison table and
      final test metrics printed to console.
- [ ] Run `pytest tests/ -v` and show all 11 tests passing.
- [ ] Launch `streamlit run app.py`, submit the hand-crafted "degraded
      pump" values from `tests/sample_cases.py`, and show the Critical risk
      flag and recommendation.
- [ ] Open the "Model, dataset & limitations" tab and discuss §14.1 with
      the reviewer.
