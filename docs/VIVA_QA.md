# Viva / Review Q&A

### 1. Why didn't you use a public dataset like AI4I-2020 or NASA C-MAPSS?
Those are the two most common public predictive-maintenance datasets and
are already heavily reused in student projects. To make this project's data
pipeline genuinely mine, I generated a synthetic centrifugal-pump dataset
from first-principles pump physics (affinity laws relating speed, flow and
head, plus an Arrhenius-style bearing-heating relationship) rather than
downloading an existing CSV. It is seeded and reproducible
(`src/generate_dataset.py`), and I deliberately injected the same kinds of
data-quality issues (missing values, sensor glitches, duplicate rows) that
a real plant extract would have, so the cleaning and EDA stages are not
trivial.

### 2. What is the target variable and how imbalanced is it?
`Pump_Failure` (binary). 669 of 12,000 rows (≈5.6%) are failures — a
realistic rate for rotating equipment, comparable to AI4I-2020's ~3.4%.

### 3. How did you avoid target leakage?
Four columns — `Cavitation_Event`, `Bearing_Failure`, `Seal_Leakage`,
`Impeller_Wear` — are *outcomes* a technician would only know after
inspecting a failed pump. They are excluded from the model's feature set in
`src/features.py` (`LEAKAGE_COLUMNS`) and used only descriptively in the EDA
to explain *why* pumps fail, not to predict *whether* they will.

### 4. Why Random Forest over Logistic Regression or Gradient Boosting?
All three were compared with identical 5-fold stratified cross-validation
on the same train+validation pool (`reports/tables/model_comparison_cv.csv`).
Random Forest had the highest mean CV F1 (0.722) and ROC-AUC (0.926) with
a lower fit time than Gradient Boosting (0.674 F1) and much higher recall
than Logistic Regression (0.392 F1) — logistic regression's linear decision
boundary underfits the non-linear wear/vibration interactions that drive
failures.

### 5. How was the decision threshold chosen, and why not just use 0.5?
Because the classes are imbalanced, the default 0.5 threshold is not
necessarily optimal for the business objective (catching failures). The
threshold was swept against the **validation split only** (never the test
set) to maximise F1 on the failure class, giving 0.611. The held-out test
set is then evaluated once, at that frozen threshold, to get an unbiased
estimate (`reports/tables/test_set_results.csv`).

### 6. How do you know the model isn't overfitting?
`reports/tables/overfitting_check.csv` compares the same metrics on the
training+validation pool vs. the untouched test set at the same threshold;
the gap is small (a few points of F1/recall), and 5-fold CV standard
deviations (`cv_f1_std` = 0.020) are tight, indicating stable performance
across folds rather than a lucky single split.

### 7. What does the model actually look at to make a decision?
Permutation importance (`src/explain_importance.py`,
`reports/figures/11_permutation_importance.png`) and the Random Forest's
native feature importances (`reports/figures/10_feature_importance.png`)
agree: radial vibration, suction pressure, resultant vibration, seal-leak
rate and the bearing-to-ambient temperature differential dominate — which
matches the physical fault model used to generate the data (vibration and
heat are the classic symptoms of bearing wear and cavitation).

### 8. Why permutation importance and not SHAP?
SHAP is listed as an optional dependency. Permutation importance is
model-agnostic, needs no extra packages, and directly answers "how much
does the model's F1 degrade if this feature is shuffled?" — a defensible
global-explainability method on its own. `explain_importance.py` is written
so that installing `shap` locally is a drop-in upgrade for richer
per-prediction attributions; it isn't required to run the project.

### 9. How would this generalise to a real plant?
It wouldn't, directly — the fault signatures are physics-informed but not
calibrated against real sensor data. Before deployment you would: (a)
retrain on real historian data once available, (b) recalibrate the
probability outputs (e.g. Platt scaling) against the true failure base
rate, and (c) move from single-reading scoring to a windowed/time-series
model that can see trends, not just snapshots.

### 10. What would you do with more time?
Add a time-series/sequence view (rolling means, rate-of-change features)
instead of scoring each reading independently; try SMOTE/class-weighted
boosting variants; and build a proper per-pump maintenance-cost simulation
to turn the precision/recall trade-off into a dollar figure (see
PROJECT_REPORT.md §9.2 for the framework used instead, given no real cost
data is available).
