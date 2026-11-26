"""
train.py
--------
Trains and compares candidate models for Pump_Failure prediction, selects
the best one by cross-validated F1 on the minority (failure) class, tunes
the decision threshold on a validation split, and persists the final
pipeline + metadata to models/.

Usage:
    python src/train.py
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                              confusion_matrix, f1_score, precision_score,
                              precision_recall_curve, recall_score,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_validation import load_and_validate
from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, split_X_y

RANDOM_STATE = 42
MODEL_DIR = "models"
TBL_DIR = "reports/tables"
FIG_DIR = "reports/figures"


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


CANDIDATES = {
    "logistic_regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
    ),
}


def compare_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"f1": "f1", "roc_auc": "roc_auc", "average_precision": "average_precision",
               "precision": "precision", "recall": "recall"}
    rows = []
    fitted = {}
    for name, clf in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        t0 = time.time()
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - t0
        rows.append({
            "model": name,
            "cv_f1_mean": cvres["test_f1"].mean(), "cv_f1_std": cvres["test_f1"].std(),
            "cv_roc_auc_mean": cvres["test_roc_auc"].mean(), "cv_roc_auc_std": cvres["test_roc_auc"].std(),
            "cv_avg_precision_mean": cvres["test_average_precision"].mean(),
            "cv_precision_mean": cvres["test_precision"].mean(),
            "cv_recall_mean": cvres["test_recall"].mean(),
            "fit_time_s": round(elapsed, 2),
        })
        fitted[name] = pipe
    results = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)
    return results, fitted


def tune_threshold(y_true, y_prob):
    """Scan thresholds, pick the one maximising F1 on the minority class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.nanargmax(f1s[:-1]) if len(thresholds) else 0
    best_threshold = thresholds[best_idx] if len(thresholds) else 0.5
    return float(best_threshold), float(f1s[best_idx])


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df, val_report = load_and_validate()
    X, y = split_X_y(df)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=RANDOM_STATE
    )

    print(f"Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")
    print(f"Failure rate -> train={y_train.mean():.3%} val={y_val.mean():.3%} test={y_test.mean():.3%}")

    # 1) Compare candidate models with 5-fold CV on train+val pool
    cv_results, fitted_pipes = compare_models(X_trainval, y_trainval)
    cv_results.to_csv(f"{TBL_DIR}/model_comparison_cv.csv", index=False)
    print("\nModel comparison (5-fold CV):\n", cv_results.to_string(index=False))

    best_name = cv_results.iloc[0]["model"]
    print(f"\nSelected model: {best_name}")

    # 2) Fit the selected model on train, tune threshold on validation
    final_pipe = Pipeline([("prep", build_preprocessor()), ("clf", CANDIDATES[best_name])])
    final_pipe.fit(X_train, y_train)

    val_prob = final_pipe.predict_proba(X_val)[:, 1]
    best_threshold, best_val_f1 = tune_threshold(y_val, val_prob)
    print(f"Tuned threshold on validation set: {best_threshold:.3f} (val F1={best_val_f1:.3f})")

    # 3) Refit on train+val (train+val pool) with the tuned threshold frozen, evaluate on held-out test
    final_pipe.fit(X_trainval, y_trainval)
    test_prob = final_pipe.predict_proba(X_test)[:, 1]
    test_metrics_tuned = evaluate_at_threshold(y_test, test_prob, best_threshold)
    test_metrics_default = evaluate_at_threshold(y_test, test_prob, 0.5)

    print("\nHeld-out test metrics @ default threshold 0.50:", test_metrics_default)
    print("Held-out test metrics @ tuned threshold {:.3f}:".format(best_threshold), test_metrics_tuned)

    pd.DataFrame([{"setting": "default_0.50", **test_metrics_default},
                  {"setting": f"tuned_{best_threshold:.3f}", **test_metrics_tuned}]
                 ).to_csv(f"{TBL_DIR}/test_set_results.csv", index=False)

    # 4) Overfitting check: train vs test metrics at tuned threshold
    train_prob = final_pipe.predict_proba(X_trainval)[:, 1]
    train_metrics = evaluate_at_threshold(y_trainval, train_prob, best_threshold)
    pd.DataFrame([{"split": "train+val (in-sample)", **train_metrics},
                  {"split": "held-out test", **test_metrics_tuned}]
                 ).to_csv(f"{TBL_DIR}/overfitting_check.csv", index=False)

    # 5) ROC and PR curve figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#ef4444", lw=2, label=f"ROC AUC = {test_metrics_tuned['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve — held-out test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_roc_curve.png", dpi=150)
    plt.close(fig)

    prec, rec, thr = precision_recall_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#3b82f6", lw=2, label=f"AP = {test_metrics_tuned['avg_precision']:.3f}")
    ax.axvline(test_metrics_tuned["recall"], color="gray", ls=":", lw=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — held-out test set")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/07_pr_curve.png", dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y_test, (test_prob >= best_threshold).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.3))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred Normal", "Pred Failure"],
                yticklabels=["Actual Normal", "Actual Failure"])
    ax.set_title(f"Confusion matrix (threshold={best_threshold:.2f})")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/08_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # threshold sweep plot
    threshs = np.linspace(0.05, 0.95, 37)
    precs, recs, f1s = [], [], []
    for t in threshs:
        m = evaluate_at_threshold(y_test, test_prob, t)
        precs.append(m["precision"]); recs.append(m["recall"]); f1s.append(m["f1"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(threshs, precs, label="Precision", color="#3b82f6")
    ax.plot(threshs, recs, label="Recall", color="#ef4444")
    ax.plot(threshs, f1s, label="F1", color="#10b981", lw=2.2)
    ax.axvline(best_threshold, color="gray", ls="--", label=f"Selected threshold={best_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sensitivity — held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/09_threshold_sweep.png", dpi=150)
    plt.close(fig)

    # 6) Feature importance (model-native; permutation importance done separately)
    try:
        ohe = final_pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        feat_names = NUMERIC_FEATURES + cat_names
        importances = final_pipe.named_steps["clf"].feature_importances_
        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        fi.to_csv(f"{TBL_DIR}/feature_importance_native.csv", header=["importance"])
        fig, ax = plt.subplots(figsize=(7, 6))
        fi.head(12)[::-1].plot(kind="barh", ax=ax, color="#8b5cf6")
        ax.set_title(f"Top feature importances ({best_name})")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/10_feature_importance.png", dpi=150)
        plt.close(fig)
    except (AttributeError, KeyError):
        print("Selected model has no native feature_importances_; skipping plot (see explain_importance.py).")

    # 7) Persist pipeline + metadata
    model_path = f"{MODEL_DIR}/pump_failure_pipeline.joblib"
    joblib.dump(final_pipe, model_path)

    metadata = {
        "model_name": best_name,
        "trained_on_rows": int(len(X_trainval)),
        "test_rows": int(len(X_test)),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "selected_threshold": round(best_threshold, 4),
        "test_metrics_at_selected_threshold": test_metrics_tuned,
        "test_metrics_at_default_0.5": test_metrics_default,
        "cv_model_comparison": cv_results.to_dict(orient="records"),
        "data_validation_report": {
            k: (v if not isinstance(v, dict) else v) for k, v in val_report.items()
        },
        "sklearn_version": __import__("sklearn").__version__,
        "random_state": RANDOM_STATE,
    }
    with open(f"{MODEL_DIR}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved pipeline to {model_path}")
    print(f"Saved metadata to {MODEL_DIR}/model_metadata.json")


if __name__ == "__main__":
    main()
