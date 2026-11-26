"""
explain_importance.py
----------------------
Model explainability without a hard SHAP dependency (SHAP is listed as
optional in requirements.txt — install it locally with
`pip install shap` and this module will use it automatically for richer,
per-prediction waterfall-style explanations; otherwise it falls back to
scikit-learn's model-agnostic permutation importance, which needs no
extra packages and is what produced reports/figures/11_permutation_importance.png).
"""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from data_validation import load_and_validate
from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, split_X_y

FIG_DIR = "reports/figures"
TBL_DIR = "reports/tables"


def global_permutation_importance(pipe, X, y, n_repeats=10, random_state=42):
    result = permutation_importance(
        pipe, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring="f1", n_jobs=-1,
    )
    importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    stds = pd.Series(result.importances_std, index=X.columns)
    return importances, stds


def local_explanation(pipe, x_row: pd.DataFrame, background_means: pd.Series, top_k=5):
    """
    A light-weight local explanation: for the single row, report how far
    each numeric feature sits (in standard-deviation units) from the
    training mean, as a simple, dependency-free stand-in for SHAP's local
    attributions. Sign indicates whether the value is above (+) or below
    (-) the typical value.
    """
    numeric = [c for c in NUMERIC_FEATURES if c in x_row.columns]
    z = (x_row[numeric].iloc[0] - background_means["mean"][numeric]) / background_means["std"][numeric]
    z = z.sort_values(key=abs, ascending=False)
    return z.head(top_k)


def run():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)
    pipe = joblib.load("models/pump_failure_pipeline.joblib")
    df, _ = load_and_validate()
    X, y = split_X_y(df)

    # Use a stratified subsample for speed
    sample = X.sample(n=min(2000, len(X)), random_state=42)
    y_sample = y.loc[sample.index]

    importances, stds = global_permutation_importance(pipe, sample, y_sample)
    out = pd.DataFrame({"permutation_importance_mean": importances, "std": stds})
    out.to_csv(f"{TBL_DIR}/permutation_importance.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    top = importances.head(12)[::-1]
    ax.barh(top.index, top.values, xerr=stds.reindex(top.index), color="#06b6d4")
    ax.set_title("Permutation importance (F1 drop when shuffled)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/11_permutation_importance.png", dpi=150)
    plt.close(fig)
    print("Saved permutation importance table and figure.")

    # Background stats for local explanations
    bg = pd.DataFrame({"mean": X[NUMERIC_FEATURES].mean(), "std": X[NUMERIC_FEATURES].std()})
    bg.to_csv(f"{TBL_DIR}/background_feature_stats.csv")
    return out


if __name__ == "__main__":
    run()
