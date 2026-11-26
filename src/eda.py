"""
eda.py
------
Generates the exploratory-analysis figures and summary tables used in the
project report. Run after data_validation has produced a clean frame.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_validation import load_and_validate
from features import engineer_features, NUMERIC_FEATURES, TARGET

sns.set_theme(style="whitegrid", font_scale=1.0)
FIG_DIR = "reports/figures"
TBL_DIR = "reports/tables"


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)


def class_balance_plot(df):
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Normal (0)", "Failure (1)"], counts.values, color=["#3b82f6", "#ef4444"])
    ax.set_ylim(0, counts.max() * 1.18)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + counts.max() * 0.02, f"{v}\n({v/len(df):.1%})",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of records")
    ax.set_title("Class balance: Pump_Failure", pad=14)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_class_balance.png", dpi=150)
    plt.close(fig)


def pump_type_breakdown(df):
    tab = pd.crosstab(df["Pump_Type"], df[TARGET], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(5.5, 4))
    tab[1].sort_index().plot(kind="bar", color="#ef4444", ax=ax)
    ax.set_ylabel("Failure rate (%)")
    ax.set_xlabel("Pump type")
    ax.set_title("Failure rate by pump class")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_failure_rate_by_type.png", dpi=150)
    plt.close(fig)
    tab.round(2).to_csv(f"{TBL_DIR}/failure_rate_by_pump_type.csv")


def correlation_heatmap(df):
    feats = engineer_features(df)
    cols = NUMERIC_FEATURES + [TARGET]
    corr = feats[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix: engineered numeric features")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3)\
        .to_csv(f"{TBL_DIR}/correlation_with_target.csv", header=["corr_with_failure"])


def sensor_distributions(df):
    feats = engineer_features(df)
    sensors = ["Vibration_Resultant_mm_s", "Bearing_Temperature_C", "Motor_Current_A",
               "Seal_Leak_Rate_ml_min", "Temp_Differential_C", "Pressure_Ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, sensors):
        sns.kdeplot(data=feats, x=col, hue=TARGET, fill=True, common_norm=False,
                    palette={0: "#3b82f6", 1: "#ef4444"}, ax=ax, legend=(col == sensors[0]))
        ax.set_title(col)
    fig.suptitle("Sensor distributions: Normal vs Failure", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_sensor_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def age_vs_failure(df):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bins = pd.cut(df["Age_Years"], bins=[0, 2, 4, 6, 8, 10, 13])
    rate = df.groupby(bins, observed=True)[TARGET].mean() * 100
    rate.plot(kind="bar", color="#f59e0b", ax=ax)
    ax.set_ylabel("Failure rate (%)")
    ax.set_xlabel("Pump age bracket (years)")
    ax.set_title("Failure rate increases with pump age")
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/05_failure_rate_by_age.png", dpi=150)
    plt.close(fig)


def missingness_table(df):
    na = df.isna().sum()
    na = na[na > 0]
    tbl = pd.DataFrame({"missing_count": na, "missing_pct": (na / len(df) * 100).round(2)})
    tbl.to_csv(f"{TBL_DIR}/missingness.csv")
    return tbl


def summary_stats_table(df):
    feats = engineer_features(df)
    tbl = feats[NUMERIC_FEATURES].describe().T.round(3)
    tbl.to_csv(f"{TBL_DIR}/summary_statistics.csv")
    return tbl


def run_all():
    ensure_dirs()
    df, report = load_and_validate()
    class_balance_plot(df)
    pump_type_breakdown(df)
    correlation_heatmap(df)
    sensor_distributions(df)
    age_vs_failure(df)
    missingness_table(df)
    summary_stats_table(df)
    print("EDA figures written to", FIG_DIR)
    print("EDA tables written to", TBL_DIR)
    return report


if __name__ == "__main__":
    run_all()
